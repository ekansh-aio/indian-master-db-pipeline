"""
Process State Acts: PDF → Surya OCR → IndicTrans2 translation → semantic chunks → embeddings → ADLS.

OCR and translation are the expensive steps — that is the priority here.
ADLS upload is included but can be skipped with --no-upload.

Run:
  .venv/bin/python pipeline/process_state_acts.py --states karnataka,delhi,maharashtra,haryana
  .venv/bin/python pipeline/process_state_acts.py --states maharashtra --no-upload
  .venv/bin/python pipeline/process_state_acts.py --all-states

Environment:
  PROCESS_WORKERS   — parallel worker threads (default 4; GPU calls are serialized via a lock)
  CHUNKING_DEVICE   — torch device for OCR / translation / embedding (default 'cuda')
  ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY, ADLS_CONTAINER_NAME
"""
import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional heavyweight deps — imported lazily so the script runs even if
# a single library is missing (with degraded functionality and a warning).
# ---------------------------------------------------------------------------

try:
    from pdf2image import convert_from_path as _pdf2images
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False

try:
    from surya.recognition import RecognitionPredictor as _SuryaRecognitionPredictor
    _SURYA_AVAILABLE = True
except ImportError:
    _SURYA_AVAILABLE = False

try:
    from langdetect import detect_langs as _langdetect_langs, DetectorFactory as _DetectorFactory
    _DetectorFactory.seed = 0  # reproducible detection
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

try:
    from IndicTransToolkit.processor import IndicProcessor as _IndicProcessor
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM
    _INDICTRANS_AVAILABLE = True
except ImportError:
    _INDICTRANS_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process_state_acts.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Quality gates for PyMuPDF extraction (same thresholds as process_central_acts.py)
MIN_TEXT_LENGTH = 500
MIN_PAGE_LENGTH = 50
MIN_TEXT_RATIO  = 0.5

# Worker threads — CPU/IO run in parallel; all GPU calls are serialized via _gpu_lock
PROCESS_WORKERS = int(os.getenv("PROCESS_WORKERS", "4"))

# langdetect ISO code → IndicTrans2 Flores-200 source language code
FLORES_CODE_MAP: dict = {
    "hi":  "hin_Deva",   # Hindi
    "kn":  "kan_Knda",   # Kannada
    "mr":  "mar_Deva",   # Marathi
    "ta":  "tam_Taml",   # Tamil
    "te":  "tel_Telu",   # Telugu
    "ml":  "mal_Mlym",   # Malayalam
    "gu":  "guj_Gujr",   # Gujarati
    "pa":  "pan_Guru",   # Punjabi
    "bn":  "ben_Beng",   # Bengali
    "or":  "ory_Orya",   # Odia
    "as":  "asm_Beng",   # Assamese
    "ur":  "urd_Arab",   # Urdu
    "ne":  "npi_Deva",   # Nepali
    "mai": "mai_Deva",   # Maithili
    "mni": "mni_Mtei",   # Meitei / Manipuri
    "kok": "kok_Deva",   # Konkani
    "doi": "doi_Deva",   # Dogri
    "sd":  "snd_Arab",   # Sindhi
    "ks":  "kas_Arab",   # Kashmiri
    "sat": "sat_Olck",   # Santali
}

_HYPHEN_NEWLINE_RE = re.compile(r"-\n(\w)")
_MULTI_NEWLINE_RE  = re.compile(r"\n{3,}")
_MULTI_SPACE_RE    = re.compile(r"[ \t]{2,}")


# ---------------------------------------------------------------------------
# Surya OCR wrapper
# ---------------------------------------------------------------------------

class SuryaOCR:
    """Load Surya RecognitionPredictor once; expose ocr_pdf(). Caller must hold gpu_lock."""

    def __init__(self, device: str = "cuda"):
        if not _SURYA_AVAILABLE or not _PDF2IMAGE_AVAILABLE:
            raise ImportError("surya and pdf2image must be installed for OCR")
        log.info("Loading Surya RecognitionPredictor on %s …", device)
        self.predictor = _SuryaRecognitionPredictor()
        self.device = device
        log.info("Surya loaded.")

    def ocr_pdf(self, pdf_path: Path) -> Optional[str]:
        try:
            images = _pdf2images(str(pdf_path), dpi=200, fmt="png", thread_count=2)
        except Exception as exc:
            log.warning("pdf2image failed for %s: %s", pdf_path.name, exc)
            return None

        try:
            predictions = self.predictor(images, full_page=True)
        except Exception as exc:
            log.warning("Surya OCR failed for %s: %s", pdf_path.name, exc)
            return None

        from bs4 import BeautifulSoup as _BS4
        page_texts = []
        for page_pred in predictions:
            blocks_text = []
            for blk in sorted(page_pred.blocks, key=lambda b: b.reading_order):
                if blk.skipped or not blk.html:
                    continue
                plain = _BS4(blk.html, "html.parser").get_text(separator=" ", strip=True)
                if plain:
                    blocks_text.append(plain)
            page_texts.append("\n".join(blocks_text))

        full_text = "\n\n".join(p for p in page_texts if p)
        full_text = _HYPHEN_NEWLINE_RE.sub(r"\1", full_text)
        full_text = _MULTI_SPACE_RE.sub(" ", full_text)
        full_text = _MULTI_NEWLINE_RE.sub("\n\n", full_text).strip()

        if len(full_text) < MIN_TEXT_LENGTH:
            log.warning("Surya OCR yielded too little text for %s (%d chars)", pdf_path.name, len(full_text))
            return None

        log.info("Surya OCR: %d chars from %s (%d pages)", len(full_text), pdf_path.name, len(images))
        return full_text


# ---------------------------------------------------------------------------
# IndicTrans2 translation wrapper
# ---------------------------------------------------------------------------

class IndicTranslator:
    """Load IndicTrans2 indic→en model once; expose translate(). Caller must hold gpu_lock."""

    _BATCH_PARAGRAPHS    = 8
    _MAX_WORDS_PER_CHUNK = 120

    def __init__(self, device: str = "cuda"):
        if not _INDICTRANS_AVAILABLE:
            raise ImportError("IndicTransToolkit and transformers must be installed")
        model_name = "ai4bharat/indictrans2-indic-en-1B"
        log.info("Loading IndicTrans2 model %s on %s …", model_name, device)
        self.ip        = _IndicProcessor(inference=True)
        self.tokenizer = _AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model     = _AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)
        self.device = device
        log.info("IndicTrans2 loaded.")

    def _split_text(self, text: str) -> list:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= self._MAX_WORDS_PER_CHUNK:
                chunks.append(para)
            else:
                sentences = re.split(r'(?<=[।.!?])\s+', para)
                current: list = []
                current_words = 0
                for sent in sentences:
                    sw = len(sent.split())
                    if current_words + sw > self._MAX_WORDS_PER_CHUNK and current:
                        chunks.append(" ".join(current))
                        current = [sent]
                        current_words = sw
                    else:
                        current.append(sent)
                        current_words += sw
                if current:
                    chunks.append(" ".join(current))
        return chunks if chunks else [text[:500]]

    def _translate_batch(self, texts: list, flores_src: str) -> list:
        batch = self.ip.preprocess_batch(texts, src_lang=flores_src, tgt_lang="eng_Latn", show_progress_bar=False)
        tokenized = self.tokenizer(batch, return_tensors="pt", padding="longest", truncation=True, max_length=256)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                num_beams=4,
                num_return_sequences=1,
                max_length=512,
            )
        decoded = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(decoded, lang="eng_Latn")

    def translate(self, text: str, src_lang: str) -> str:
        flores_code = FLORES_CODE_MAP.get(src_lang)
        if not flores_code:
            log.warning("No Flores code for lang '%s' — skipping translation", src_lang)
            return text
        chunks = self._split_text(text)
        translated_chunks: list = []
        for i in range(0, len(chunks), self._BATCH_PARAGRAPHS):
            batch = chunks[i: i + self._BATCH_PARAGRAPHS]
            try:
                results = self._translate_batch(batch, flores_code)
                translated_chunks.extend(results)
            except Exception as exc:
                log.warning("Translation batch failed: %s — keeping original for chunk", exc)
                translated_chunks.extend(batch)
        return "\n\n".join(translated_chunks)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _slug(handle_id: str, state_key: str) -> str:
    """Convert handle '123456789/14523' + state key → 'SA_karnataka_14523'."""
    parts = handle_id.split("/")
    if len(parts) == 2:
        return f"SA_{state_key}_{parts[1]}"
    return f"SA_{state_key}_" + handle_id.replace("/", "_")


def _pymupdf_extract(pdf_path: Path) -> Tuple[Optional[str], int]:
    """Extract text via PyMuPDF. Returns (text, total_pages); text is None if quality gate fails."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.warning("Cannot open %s: %s", pdf_path.name, exc)
        return None, 0

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return None, 0

    pages_text = []
    pages_with_text = 0
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
        if len(text.strip()) >= MIN_PAGE_LENGTH:
            pages_with_text += 1
    doc.close()

    full_text  = "\n".join(pages_text)
    text_ratio = pages_with_text / total_pages

    if len(full_text.strip()) >= MIN_TEXT_LENGTH and text_ratio >= MIN_TEXT_RATIO:
        return full_text, total_pages

    log.info("Low text quality (ratio=%.2f, chars=%d) for %s — needs OCR (%d pages)",
             text_ratio, len(full_text.strip()), pdf_path.name, total_pages)
    return None, total_pages


def _detect_language(text: str) -> Tuple[str, float]:
    """Returns (iso_lang_code, confidence). Falls back to ('unknown', 0.0)."""
    if not _LANGDETECT_AVAILABLE:
        return "unknown", 0.0
    try:
        langs = _langdetect_langs(text[:2000])
        if langs:
            return langs[0].lang, round(langs[0].prob, 4)
    except Exception as exc:
        log.warning("langdetect failed: %s", exc)
    return "unknown", 0.0


# ---------------------------------------------------------------------------
# Shared processing context — one instance, passed to all worker threads
# ---------------------------------------------------------------------------

class _ProcessContext:
    __slots__ = (
        "gpu_lock", "progress_lock", "progress_path", "done_ids",
        "chunker", "cleaner", "uploader", "surya_ocr", "translator",
    )

    def __init__(self, gpu_lock, progress_lock, progress_path, done_ids,
                 chunker, cleaner, uploader, surya_ocr, translator):
        self.gpu_lock      = gpu_lock
        self.progress_lock = progress_lock
        self.progress_path = progress_path
        self.done_ids      = done_ids
        self.chunker       = chunker
        self.cleaner       = cleaner
        self.uploader      = uploader
        self.surya_ocr     = surya_ocr
        self.translator    = translator


def _save_done_id(progress_path: Path, doc_id: str) -> None:
    """Append doc_id to JSONL progress file. Caller must hold progress_lock."""
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc_id) + "\n")


def process_act(meta: dict, pdf_dir: Path, state_key: str, ctx: _ProcessContext) -> Optional[bool]:
    """
    Process one act end-to-end: PDF → OCR → translate → chunk → embed → ADLS.

    Returns:
      True  — newly processed and saved
      False — skipped (already in done_ids)
      None  — failed (upload error)
    Raises on unexpected/unrecoverable errors.
    """
    handle_id = meta.get("handle_id", "")
    doc_id    = _slug(handle_id, state_key)
    year      = meta.get("year") or 0
    act_name  = meta.get("act_name", "Unknown")

    with ctx.progress_lock:
        if doc_id in ctx.done_ids:
            return False

    # ------------------------------------------------------------------
    # Phase 1: Locate PDF + measure size  (I/O — no GPU lock)
    # ------------------------------------------------------------------
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", act_name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()[:200]
    pdf_path  = pdf_dir / (sanitized + ".pdf")
    if not pdf_path.exists():
        alt = pdf_dir / (doc_id + ".pdf")
        if alt.exists():
            pdf_path = alt

    pdf_exists    = pdf_path.exists()
    pdf_file_size = int(pdf_path.stat().st_size) if pdf_exists else 0
    raw_text      = None
    ocr_applied   = False
    page_count    = 0

    # ------------------------------------------------------------------
    # Phase 2: PyMuPDF text extraction  (CPU — no GPU lock)
    # ------------------------------------------------------------------
    if pdf_exists:
        raw_text, page_count = _pymupdf_extract(pdf_path)

    # ------------------------------------------------------------------
    # Phase 3: Surya OCR fallback  (GPU — locked)
    # ------------------------------------------------------------------
    if pdf_exists and raw_text is None and ctx.surya_ocr is not None:
        with ctx.gpu_lock:
            raw_text = ctx.surya_ocr.ocr_pdf(pdf_path)
        ocr_applied = raw_text is not None
        if not ocr_applied:
            log.warning("Both PyMuPDF and Surya failed for %s — metadata-only doc", act_name[:60])

    # ------------------------------------------------------------------
    # Phase 4: Language detection  (CPU — no GPU lock)
    # ------------------------------------------------------------------
    language_original   = "en"
    language_confidence = 1.0
    is_translated       = False
    final_text          = raw_text

    if raw_text:
        language_original, language_confidence = _detect_language(raw_text)

    # ------------------------------------------------------------------
    # Phase 5: IndicTrans2 translation if non-English  (GPU — locked)
    # ------------------------------------------------------------------
    if (raw_text
            and language_original not in ("en", "unknown")
            and language_original in FLORES_CODE_MAP):
        if ctx.translator is not None:
            try:
                with ctx.gpu_lock:
                    translated = ctx.translator.translate(raw_text, language_original)
                final_text    = translated
                is_translated = True
                log.info("Translated '%s'→en (%d→%d chars) for %s",
                         language_original, len(raw_text), len(translated), doc_id)
            except Exception as exc:
                log.error("Translation failed for %s: %s — keeping original", doc_id, exc)
                final_text = raw_text
        else:
            log.warning("Translation needed ('%s') but IndicTrans2 not loaded — keeping original",
                        language_original)
            final_text = raw_text

    # ------------------------------------------------------------------
    # Phase 6: Chunk + embed  (GPU — locked)
    # ------------------------------------------------------------------
    chunks: list = []
    if final_text:
        cleaned = ctx.cleaner.clean(final_text)
        chunk_dicts, _ = ctx.chunker.split(cleaned)
        if len(chunk_dicts) < 2:
            log.warning("Too few chunks (%d) for %s — skipping embedding", len(chunk_dicts), doc_id)
            chunk_dicts = []
        if chunk_dicts:
            with ctx.gpu_lock:
                embeddings = ctx.chunker.encode_batch([c["text"] for c in chunk_dicts])
            for c, emb in zip(chunk_dicts, embeddings):
                assert emb.shape[0] == 384, f"Bad embedding dim {emb.shape} for {doc_id}"
                chunks.append({
                    "chunk_id":  f"{doc_id}_{c['chunk_id']}",
                    "text":      c["text"],
                    "embedding": emb.tolist(),
                })

    log.info(
        "%s | lang=%s(conf=%.2f) translated=%s ocr=%s pages=%d chunks=%d chars=%d",
        doc_id, language_original, language_confidence,
        is_translated, ocr_applied, page_count, len(chunks),
        len(final_text) if final_text else 0,
    )

    # ------------------------------------------------------------------
    # Phase 7: Build doc  (CPU)
    # ------------------------------------------------------------------
    year_str = str(year) if year else "unknown"

    base_doc = {
        "doc_id":               doc_id,
        "act_name":             act_name,
        "act_number":           meta.get("act_number", ""),
        "year":                 year,
        "state":                state_key,
        "state_display":        meta.get("state_display", ""),
        "jurisdiction":         "India",
        "source":               "indiacode.nic.in",
        "source_url":           meta.get("source_url", ""),
        "handle_id":            handle_id,
        "pdf_url":              meta.get("pdf_url", ""),
        "pdf_exists":           pdf_exists,
        "pdf_file_size_bytes":  pdf_file_size,
        "language_original":    language_original,
        "language_confidence":  language_confidence,
        "ocr_applied":          ocr_applied,
        "ocr_page_count":       page_count if ocr_applied else 0,
        "total_page_count":     page_count,
        "_is_translated":       is_translated,
        "total_chars":          len(final_text) if final_text else 0,
        "total_chunks":         len(chunks),
        "scrape_timestamp":     meta.get("scraped_at", ""),
        "processed_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text":                 final_text or "",
    }

    # ------------------------------------------------------------------
    # Phase 8: ADLS upload  (I/O — no GPU lock)
    # ------------------------------------------------------------------
    if ctx.uploader is None:
        # --no-upload mode: save progress, return success
        with ctx.progress_lock:
            ctx.done_ids.add(doc_id)
            _save_done_id(ctx.progress_path, doc_id)
        return True

    app_path = f"state_acts/app/state={state_key}/year={year_str}/{doc_id}.json"
    ok_app   = ctx.uploader.upload_json_file(base_doc, app_path, overwrite=True)
    if not ok_app:
        log.error("app/ upload failed for %s", doc_id)
        return None

    processed_doc = {**base_doc, "chunks": chunks}
    processed_doc.pop("text", None)
    proc_path = f"state_acts/processed/state={state_key}/year={year_str}/{doc_id}.json"
    ok_proc   = ctx.uploader.upload_json_file(processed_doc, proc_path, overwrite=True)
    if not ok_proc:
        log.error("processed/ upload failed for %s", doc_id)
        return None

    with ctx.progress_lock:
        ctx.done_ids.add(doc_id)
        _save_done_id(ctx.progress_path, doc_id)

    log.info("Uploaded %s (%d chunks) → app/ + processed/", doc_id, len(chunks))
    return True


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_done_ids(progress_path: Path) -> set:
    ids: set = set()
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line))
                except json.JSONDecodeError:
                    ids.add(line.strip('"'))
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Process State Acts: PDF→OCR→translate→chunk→ADLS")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--states", help="Comma-separated state keys, e.g. karnataka,delhi")
    grp.add_argument("--all-states", action="store_true", help="Process all states found in pdf-dir")
    ap.add_argument("--pdf-dir", default="state_acts_pdfs",
                    help="Base directory with state subdirs (default: state_acts_pdfs)")
    ap.add_argument("--similarity-threshold", type=float, default=0.65)
    ap.add_argument("--workers", type=int, default=PROCESS_WORKERS,
                    help=f"Parallel worker threads (default {PROCESS_WORKERS}; GPU calls are serialized)")
    ap.add_argument("--no-resume",         action="store_true", help="Reprocess already-done acts")
    ap.add_argument("--no-upload",         action="store_true", help="Skip ADLS upload (OCR+translate+chunk only)")
    ap.add_argument("--no-ocr",            action="store_true", help="Disable Surya OCR fallback")
    ap.add_argument("--no-translate",      action="store_true", help="Disable IndicTrans2 translation")
    ap.add_argument("--stream",            action="store_true",
                    help="Poll metadata.jsonl for new lines; start processing while scraping is still running")
    ap.add_argument("--stream-idle-secs",  type=int, default=300,
                    help="--stream: stop after this many seconds with no new acts (default 300)")
    args = ap.parse_args()

    pdf_base = Path(args.pdf_dir)

    if args.all_states:
        states = sorted(d.name for d in pdf_base.iterdir() if d.is_dir())
    else:
        states = [s.strip() for s in args.states.split(",") if s.strip()]

    if not states:
        log.error("No states to process.")
        sys.exit(1)

    for state_key in states:
        meta_path = pdf_base / state_key / "metadata.jsonl"
        if not args.stream and not meta_path.exists():
            log.error("No metadata.jsonl for '%s' in %s — run scrape_state_acts.py first",
                      state_key, pdf_base / state_key)
            sys.exit(1)

    progress_path = Path("pipeline_progress") / "done_ids_state_acts.jsonl"
    progress_path.parent.mkdir(exist_ok=True)
    done_ids: set = set() if args.no_resume else load_done_ids(progress_path)
    log.info("Loaded %d done IDs from progress file", len(done_ids))

    if not torch.cuda.is_available():
        log.error("CUDA not available — set CUDA_VISIBLE_DEVICES correctly")
        sys.exit(1)
    device = os.getenv("CHUNKING_DEVICE", "cuda")
    log.info("GPU ready: %d visible  device=%s  workers=%d",
             torch.cuda.device_count(), device, args.workers)

    # Load Surya OCR (once, shared across all workers — gpu_lock serializes actual calls)
    surya_ocr: Optional[SuryaOCR] = None
    if not args.no_ocr:
        if _SURYA_AVAILABLE and _PDF2IMAGE_AVAILABLE:
            try:
                surya_ocr = SuryaOCR(device=device)
            except Exception as exc:
                log.warning("Failed to load Surya: %s — OCR disabled", exc)
        else:
            missing = [p for p, ok in [("surya", _SURYA_AVAILABLE), ("pdf2image", _PDF2IMAGE_AVAILABLE)] if not ok]
            log.warning("OCR disabled — missing packages: %s", ", ".join(missing))

    # Load IndicTrans2 (once, shared across all workers)
    translator: Optional[IndicTranslator] = None
    if not args.no_translate:
        if _INDICTRANS_AVAILABLE and _LANGDETECT_AVAILABLE:
            try:
                translator = IndicTranslator(device=device)
            except Exception as exc:
                log.warning("Failed to load IndicTrans2: %s — translation disabled", exc)
        else:
            missing = [p for p, ok in [
                ("IndicTransTokenizer", _INDICTRANS_AVAILABLE),
                ("langdetect", _LANGDETECT_AVAILABLE),
            ] if not ok]
            log.warning("Translation disabled — missing packages: %s", ", ".join(missing))

    cleaner = LegalTextCleaner()
    chunker = SemanticChunker(
        similarity_threshold=args.similarity_threshold,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=12,
        min_chunk_size=150,
        role_file_path=None,
    )

    uploader: Optional[ADLSUploader] = None
    if not args.no_upload:
        uploader = ADLSUploader(
            account_name=os.environ["ADLS_ACCOUNT_NAME"],
            account_key=os.environ["ADLS_ACCOUNT_KEY"],
            container_name=os.environ["ADLS_CONTAINER_NAME"],
        )

    ctx = _ProcessContext(
        gpu_lock      = threading.Lock(),
        progress_lock = threading.Lock(),
        progress_path = progress_path,
        done_ids      = done_ids,
        chunker       = chunker,
        cleaner       = cleaner,
        uploader      = uploader,
        surya_ocr     = surya_ocr,
        translator    = translator,
    )

    total_processed = total_skipped = total_failed = 0
    t_start = time.time()

    for state_key in states:
        meta_path = pdf_base / state_key / "metadata.jsonl"
        pdf_dir   = pdf_base / state_key

        log.info("=" * 60)
        log.info("State: %s  mode=%s  workers=%d", state_key,
                 "stream" if args.stream else "batch", args.workers)

        processed = skipped = failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures: dict = {}

            def _harvest(block: bool = False) -> None:
                nonlocal processed, skipped, failed
                completed = []
                for fut, m in list(futures.items()):
                    if fut.done() or block:
                        completed.append((fut, m))
                if block:
                    import concurrent.futures
                    for fut, m in completed:
                        try:
                            result = fut.result()
                        except Exception as exc:
                            log.error("Unhandled error for '%s': %s", m.get("act_name", "?"), exc)
                            failed += 1
                            continue
                        if result is True:
                            processed += 1
                        elif result is False:
                            skipped += 1
                        else:
                            failed += 1
                        del futures[fut]
                else:
                    for fut, m in completed:
                        if not fut.done():
                            continue
                        try:
                            result = fut.result()
                        except Exception as exc:
                            log.error("Unhandled error for '%s': %s", m.get("act_name", "?"), exc)
                            failed += 1
                            continue
                        if result is True:
                            processed += 1
                        elif result is False:
                            skipped += 1
                        else:
                            failed += 1
                        del futures[fut]

            def _log_progress(submitted: int) -> None:
                done_count = processed + skipped + failed
                if done_count > 0 and done_count % 25 == 0:
                    elapsed = time.time() - t_start
                    rate    = done_count / elapsed if elapsed > 0 else 0
                    log.info(
                        "  [%s] done=%d submitted=%d  processed=%d  skipped=%d  "
                        "failed=%d  rate=%.1f/min",
                        state_key, done_count, submitted,
                        processed, skipped, failed, rate * 60,
                    )

            seen_handles: set = set()
            submitted = 0

            if args.stream:
                # Poll metadata.jsonl, submit each new act immediately.
                # Exit after stream_idle_secs with no new lines.
                idle_deadline = time.time() + args.stream_idle_secs
                file_offset = 0

                while True:
                    if meta_path.exists():
                        with meta_path.open(encoding="utf-8") as f:
                            f.seek(file_offset)
                            new_lines = f.readlines()
                            file_offset = f.tell()

                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                meta = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            h = meta.get("handle_id", "")
                            if h in seen_handles:
                                continue
                            seen_handles.add(h)
                            fut = pool.submit(process_act, meta, pdf_dir, state_key, ctx)
                            futures[fut] = meta
                            submitted += 1
                            idle_deadline = time.time() + args.stream_idle_secs

                    _harvest(block=False)
                    _log_progress(submitted)

                    if time.time() >= idle_deadline:
                        log.info("[%s] No new acts for %ds — scraping done, waiting for in-flight work",
                                 state_key, args.stream_idle_secs)
                        break

                    time.sleep(5)

                # Drain remaining futures
                from concurrent.futures import as_completed as _as_completed
                for fut in _as_completed(list(futures.keys())):
                    m = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.error("Unhandled error for '%s': %s", m.get("act_name", "?"), exc)
                        failed += 1
                        continue
                    if result is True:
                        processed += 1
                    elif result is False:
                        skipped += 1
                    else:
                        failed += 1
                    del futures[fut]

            else:
                # Batch mode: load all metadata upfront (scraping must be complete)
                metas = []
                with meta_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                metas.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass

                log.info("Loaded %d acts for %s", len(metas), state_key)
                total_acts = len(metas)

                for meta in metas:
                    fut = pool.submit(process_act, meta, pdf_dir, state_key, ctx)
                    futures[fut] = meta
                    submitted += 1
                    _harvest(block=False)
                    _log_progress(submitted)

                from concurrent.futures import as_completed as _as_completed
                for fut in _as_completed(list(futures.keys())):
                    m = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.error("Unhandled error for '%s': %s", m.get("act_name", "?"), exc)
                        failed += 1
                        continue
                    if result is True:
                        processed += 1
                    elif result is False:
                        skipped += 1
                    else:
                        failed += 1
                    del futures[fut]
                    done_count = processed + skipped + failed
                    if done_count % 25 == 0:
                        elapsed = time.time() - t_start
                        rate    = done_count / elapsed if elapsed > 0 else 0
                        eta_s   = (total_acts - done_count) / rate if rate > 0 else 0
                        log.info(
                            "  [%s] %d/%d  processed=%d  skipped=%d  failed=%d  "
                            "rate=%.1f/min  ETA=%.0fmin",
                            state_key, done_count, total_acts,
                            processed, skipped, failed,
                            rate * 60, eta_s / 60,
                        )

        log.info("State %s done — processed=%d  skipped=%d  failed=%d",
                 state_key, processed, skipped, failed)
        total_processed += processed
        total_skipped   += skipped
        total_failed    += failed

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("ALL DONE — processed=%d  skipped=%d  failed=%d  elapsed=%.0fmin",
             total_processed, total_skipped, total_failed, elapsed / 60)


if __name__ == "__main__":
    main()

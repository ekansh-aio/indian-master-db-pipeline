"""
Process Central Acts: PDF → text → semantic chunks → GPU embeddings → ADLS upload.

Run: .venv/bin/python pipeline/process_central_acts.py [--pdf-dir central_acts_pdfs] [--gpu 0]
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv

# OCR deps — imported lazily so the script still runs if not installed
try:
    from pdf2image import convert_from_path as _pdf2images
    import pytesseract as _pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Quality gates (same as txt_to_pdf.py in state acts)
MIN_TEXT_LENGTH = 500
MIN_PAGE_LENGTH = 50
MIN_TEXT_RATIO  = 0.5   # fraction of pages that must yield usable text

ADLS_APP_PATH       = "app/central_acts"        # unchunked: PDF→text only
ADLS_PROCESSED_PATH = "processed/central_acts"  # chunked + embeddings


def _slug(handle_id: str) -> str:
    """Convert handle '123456789/2345' to 'CA_1362_2345' style slug."""
    parts = handle_id.split("/")
    if len(parts) == 2:
        return f"CA_{parts[0].lstrip('0')}_{parts[1]}"
    return "CA_" + handle_id.replace("/", "_")


_OCR_NOISE_RE = re.compile(
    r"[^\x20-\x7E -ɏ‐-‧‰-⁞⁠-⁯\n\t]"
)
_HYPHEN_NEWLINE_RE = re.compile(r"-\n(\w)")
_MULTI_NEWLINE_RE  = re.compile(r"\n{3,}")
_MULTI_SPACE_RE    = re.compile(r"[ \t]{2,}")


def _clean_ocr_page(text: str) -> str:
    """Remove OCR noise characters and normalise whitespace."""
    text = _OCR_NOISE_RE.sub(" ", text)
    text = _HYPHEN_NEWLINE_RE.sub(r"\1", text)   # rejoin hyphenated line-breaks
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _ocr_pdf(pdf_path: Path) -> Optional[str]:
    """Rasterise each page and run Tesseract OCR. Returns cleaned text or None."""
    if not _OCR_AVAILABLE:
        log.warning("OCR skipped (pdf2image/pytesseract not installed): %s", pdf_path.name)
        return None
    try:
        images = _pdf2images(str(pdf_path), dpi=300, fmt="png", thread_count=2)
    except Exception as exc:
        log.warning("pdf2image failed for %s: %s", pdf_path.name, exc)
        return None

    pages_text = []
    for i, img in enumerate(images):
        try:
            raw = _pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
        except Exception as exc:
            log.warning("Tesseract failed on page %d of %s: %s", i + 1, pdf_path.name, exc)
            raw = ""
        pages_text.append(_clean_ocr_page(raw))

    full_text = "\n\n".join(p for p in pages_text if p)
    if len(full_text.strip()) < MIN_TEXT_LENGTH:
        log.warning("OCR yielded too little text for %s (%d chars)", pdf_path.name, len(full_text))
        return None

    log.info("OCR extracted %d chars from %s (%d pages)", len(full_text), pdf_path.name, len(images))
    return full_text


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Return full text of a PDF, falling back to OCR for scanned pages."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.warning("Cannot open %s: %s", pdf_path.name, exc)
        return None

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return None

    pages_text = []
    pages_with_text = 0
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
        if len(text.strip()) >= MIN_PAGE_LENGTH:
            pages_with_text += 1

    doc.close()

    full_text = "\n".join(pages_text)
    text_ratio = pages_with_text / total_pages

    if len(full_text.strip()) >= MIN_TEXT_LENGTH and text_ratio >= MIN_TEXT_RATIO:
        return full_text

    if text_ratio < MIN_TEXT_RATIO:
        log.info("Low text ratio (%.2f) for %s — attempting OCR (%d pages)",
                 text_ratio, pdf_path.name, total_pages)
        ocr_text = _ocr_pdf(pdf_path)
        if ocr_text:
            return ocr_text
        log.warning("OCR fallback failed for %s — no usable text", pdf_path.name)
        return None

    log.warning("Skipping %s: text too short (%d chars)", pdf_path.name, len(full_text.strip()))
    return None


def build_doc_id(handle_id: str) -> str:
    return _slug(handle_id)


def process_act(
    meta: dict,
    pdf_dir: Path,
    chunker: SemanticChunker,
    cleaner: LegalTextCleaner,
    uploader: ADLSUploader,
    done_ids: set,
) -> bool:
    """Process one act: PDF→text→chunks→embed→ADLS. Returns True if uploaded."""
    doc_id = build_doc_id(meta["handle_id"])
    year = meta.get("year") or 0
    act_name = meta.get("act_name", "Unknown")

    if doc_id in done_ids:
        log.debug("Skip (already done): %s", doc_id)
        return False

    # Locate PDF
    pdf_filename = re.sub(r'[<>:"/\\|?*]', "_", act_name)
    pdf_filename = re.sub(r"\s+", " ", pdf_filename).strip()[:200] + ".pdf"
    pdf_path = pdf_dir / pdf_filename

    pdf_exists = pdf_path.exists()
    if not pdf_exists:
        # Try by handle slug as fallback name
        alt = pdf_dir / (doc_id + ".pdf")
        if alt.exists():
            pdf_path = alt
            pdf_exists = True

    raw_text = extract_text_from_pdf(pdf_path) if pdf_exists else None

    if raw_text is None and pdf_exists:
        log.warning("Text extraction failed for %s — uploading metadata-only doc", act_name[:60])

    chunks = []
    if raw_text:
        cleaned = cleaner.clean(raw_text)
        chunk_dicts, _ = chunker.split(cleaned)
        if len(chunk_dicts) < 2:
            log.warning("Too few chunks (%d) for %s — uploading metadata-only doc", len(chunk_dicts), act_name[:60])
            chunk_dicts = []
        if chunk_dicts:
            embeddings = chunker.encode_batch([c["text"] for c in chunk_dicts])
            for c, emb in zip(chunk_dicts, embeddings):
                assert emb.shape[0] == 384, f"Bad embedding dim: {emb.shape}"
                chunks.append({
                    "chunk_id": f"{doc_id}_{c['chunk_id']}",
                    "text": c["text"],
                    "embedding": emb.tolist(),
                })

    base_doc = {
        "doc_id":      doc_id,
        "act_name":    act_name,
        "act_number":  meta.get("act_number", ""),
        "year":        year,
        "jurisdiction":"India",
        "pdf_url":     meta.get("pdf_url", ""),
        "pdf_exists":  pdf_exists,
        "handle_id":   meta["handle_id"],
        "source":      "indiacode.nic.in",
        "text":        raw_text or "",
    }

    # Upload unchunked doc to app/central_acts/
    app_path = f"{ADLS_APP_PATH}/{year}/{doc_id}.json"
    ok_app = uploader.upload_json_file(base_doc, app_path, overwrite=True)
    if not ok_app:
        log.error("app/ upload failed for %s", doc_id)
        return False

    # Upload chunked+embedded doc to processed/central_acts/
    processed_doc = {**base_doc, "chunks": chunks}
    processed_doc.pop("text", None)  # text lives in app/, processed/ has chunks
    proc_path = f"{ADLS_PROCESSED_PATH}/{year}/{doc_id}.json"
    ok_proc = uploader.upload_json_file(processed_doc, proc_path, overwrite=True)
    if not ok_proc:
        log.error("processed/ upload failed for %s", doc_id)
        return False

    done_ids.add(doc_id)
    log.info("Uploaded %s  (%d chunks)  → app/ + processed/", doc_id, len(chunks))
    return True


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


def save_done_id(progress_path: Path, doc_id: str) -> None:
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc_id) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Process Central Acts: PDF→chunks→ADLS")
    ap.add_argument("--pdf-dir", default="central_acts_pdfs",
                    help="Directory with PDFs and metadata.jsonl (default: central_acts_pdfs)")
    ap.add_argument("--similarity-threshold", type=float, default=0.65,
                    help="SemanticChunker similarity threshold (default: 0.65)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore local progress and reprocess everything")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    metadata_path = pdf_dir / "metadata.jsonl"
    if not metadata_path.exists():
        log.error("metadata.jsonl not found in %s — run scrape_central_acts.py first", pdf_dir)
        sys.exit(1)

    progress_path = Path("pipeline_progress") / "done_ids_central_acts.jsonl"
    progress_path.parent.mkdir(exist_ok=True)

    done_ids: set = set() if args.no_resume else load_done_ids(progress_path)
    log.info("Loaded %d done IDs from progress file", len(done_ids))

    chunking_device = os.getenv("CHUNKING_DEVICE", "cpu")
    if not torch.cuda.is_available():
        log.error("CUDA not available — set CUDA_VISIBLE_DEVICES and CHUNKING_DEVICE correctly")
        sys.exit(1)
    log.info("GPU ready: CUDA_VISIBLE_DEVICES=%s  CHUNKING_DEVICE=%s  (%d device(s) visible)",
             os.getenv("CUDA_VISIBLE_DEVICES", "unset"), chunking_device, torch.cuda.device_count())

    cleaner = LegalTextCleaner()
    chunker = SemanticChunker(
        similarity_threshold=args.similarity_threshold,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=12,
        min_chunk_size=150,
        role_file_path=None,
    )

    uploader = ADLSUploader(
        account_name=os.environ["ADLS_ACCOUNT_NAME"],
        account_key=os.environ["ADLS_ACCOUNT_KEY"],
        container_name=os.environ["ADLS_CONTAINER_NAME"],
    )

    metas = []
    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metas.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    log.info("Processing %d acts from %s", len(metas), metadata_path)

    uploaded = skipped = failed = 0
    for meta in metas:
        doc_id = build_doc_id(meta.get("handle_id", "unknown"))
        if doc_id in done_ids:
            skipped += 1
            continue

        ok = process_act(meta, pdf_dir, chunker, cleaner, uploader, done_ids)
        if ok:
            uploaded += 1
            save_done_id(progress_path, doc_id)
        else:
            failed += 1

    log.info("Done. uploaded=%d  skipped=%d  failed=%d", uploaded, skipped, failed)


if __name__ == "__main__":
    main()

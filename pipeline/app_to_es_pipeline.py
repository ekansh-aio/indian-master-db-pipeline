"""
Combined pipeline: ADLS app/ → SemanticChunk+RoleClassify → ADLS processed/ → GPU embed → ES

Combines adls_pipeline.py (app/→processed/) and adls_to_es_pipeline.py (processed/→ES)
into a single pass: each document is chunked, uploaded to ADLS processed/, then
immediately embedded and uploaded to Elasticsearch — no intermediate scan step.

Architecture (per-year, 7 stages):
  1. Background lister/reader   IO        Scan app/ paths, filter by delta1, read raw JSON
  2. Stage2: split sentences     CPU       LegalTextCleaner → sentence split
  3. Stage3: encode sentences    GPU       Batch sentence embeddings
  4. Stage4: assemble chunks     CPU       SemanticChunk assembly
  5. Stage5: role classify       GPU       Batch role classification
  6. Stage6: ADLS upload         IO        Upload *_all_chunks.json to processed/
  7. Stage7: ES embed+upload     GPU+IO    Top-K select → embed → bulk upload to ES

Resume (inventory-delta):
  - Delta1 = app/_inventory.json − processed/_inventory.json  → goes through all 7 stages
  - Delta2 = processed/_inventory.json − processed/_inventory_es.json  → fed directly to stage7
  - --no-resume: ignore inventories, reprocess all app/ docs through all stages

Usage:
  python pipeline/app_to_es_pipeline.py --doc-type hc [--years 2020 2021] [--recreate-index]
  python pipeline/app_to_es_pipeline.py --doc-type hc --year-range 2020 2025
  python pipeline/app_to_es_pipeline.py --doc-type hc --years 2023 --no-resume

Environment:
  ES_URL, ES_API_KEY (or ES_USER/ES_PASS), ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY,
  ADLS_CONTAINER_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, IO_WORKERS,
  ES_BULK_BATCH_SIZE, PIPELINE_BUFFER_SIZE, ES_MAX_RETRIES, ES_RETRY_DELAY
"""

import argparse
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import ADLS_CONFIG, CHUNKING_CONFIG, DOC_TYPE_CONFIG, EMBEDDING_CONFIG, ROLE_CLASSIFICATION_CONFIG
from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from core.role_classifier import create_classifier_from_config
from utils.weighted_selector import weighted_topk_selection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_to_es_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("app_to_es")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOP = object()

IO_WORKERS       = int(os.getenv("IO_WORKERS",             "32"))
BULK_BATCH_SIZE  = int(os.getenv("ES_BULK_BATCH_SIZE",     "100"))
BUFFER_SIZE      = int(os.getenv("PIPELINE_BUFFER_SIZE",   "500"))
ES_MAX_RETRIES   = int(os.getenv("ES_MAX_RETRIES",         "3"))
ES_RETRY_DELAY   = float(os.getenv("ES_RETRY_DELAY",       "2.0"))
EMBED_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE",   "2048"))
SENT_BATCH       = int(os.getenv("SENT_BATCH",             "40000"))
CHUNK_BATCH      = int(os.getenv("CHUNK_BATCH",            "20000"))
CHECK_WORKERS    = int(os.getenv("CHECK_WORKERS",          "400"))
UPLOAD_WORKERS   = int(os.getenv("UPLOAD_WORKERS",         "64"))

TOP_K = 12

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sample" / "es_schema_prod.json"

# Resolved in main() from --doc-type
DOC_TYPE: str = "hc"
_ADLS_APP_PATH: str = ""       # e.g. "app/High_Court_Judgements"
_ADLS_PROCESSED_PATH: str = "" # e.g. "processed/High_Court_Judgements"
INDEX_NAME: str = ""           # e.g. "hc_judgements"

_DOC_TYPE_APP_ROOTS = {
    "hc": DOC_TYPE_CONFIG[0]["adls_input_path"].rstrip("/"),
    "sc": DOC_TYPE_CONFIG[1]["adls_input_path"].rstrip("/"),
}
_DOC_TYPE_PROCESSED_ROOTS = {
    "hc": "processed/High_Court_Judgements",
    "sc": "processed/Supreme_Court_Judgements",
}
_DOC_TYPE_COLLECTION_DIRS = {
    "hc": "High_Court_Judgements",
    "sc": "Supreme_Court_Judgements",
}
_DOC_TYPE_INDEX_NAMES = {
    "hc": "hc_judgements",
    "sc": "sc_judgements",
}

METADATA_FIELDS = [
    "doc_id", "date", "jurisdiction", "doc_name", "year", "bench",
    "court_code", "title", "judge", "pdf_link", "cnr",
    "date_of_registration", "decision_date", "disposal_nature",
    "court", "pdf_exists", "original_source_path", "all_chunks_path",
    "source_file",
]
CHUNK_FIELDS = ["chunk_id", "text", "role", "same_role_chunk_ids"]

ROLE_WEIGHTS: Dict[str, float] = {
    "Decision":   3.0,
    "Precedents": 3.0,
    "Issues":     2.5,
    "Preamble":   2.5,
    "Facts":      2.5,
    "Statute":    1.5,
    "Reasoning":  0.4,
    "Arguments":  0.3,
    "Others":     0.2,
}

# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def generate_doc_id(source_file_path: str) -> str:
    path_without_ext = source_file_path.replace(".json", "")
    parts = Path(path_without_ext).parts
    skip = {"raw", "newapp", "input", "data", "app", "high_court_judgements", "supreme_court_judgements"}
    relevant = [p for p in parts if p.lower() not in skip]
    doc_id = "_".join(relevant).replace("-", "_").replace(" ", "_").lower()
    if len(doc_id) > 200:
        base = relevant[-1] if relevant else "doc"
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    return doc_id


def get_all_chunks_path(source_file_path: str) -> str:
    path_without_ext = source_file_path.replace(".json", "")
    parts = Path(path_without_ext).parts
    skip = {"raw", "newapp", "input", "data", "app", "high_court_judgements", "supreme_court_judgements"}
    relevant = [p for p in parts if p.lower() not in skip]
    if relevant:
        dir_parts = relevant[:-1]
        fname = relevant[-1]
        if dir_parts:
            return f"{_ADLS_PROCESSED_PATH}/{'/'.join(dir_parts)}/{fname}_all_chunks.json"
        return f"{_ADLS_PROCESSED_PATH}/{fname}_all_chunks.json"
    return f"{_ADLS_PROCESSED_PATH}/unknown_all_chunks.json"


def _doc_id_from_chunks_path(path: str) -> str:
    name = Path(path).stem
    if name.endswith("_all_chunks"):
        return name[: -len("_all_chunks")]
    return name


def attach_same_role_chunk_ids(chunks: List[Dict]) -> None:
    role_to_ids: Dict = defaultdict(list)
    for c in chunks:
        role_to_ids[c.get("role", "Others")].append(c["chunk_id"])
    for c in chunks:
        role = c.get("role", "Others")
        c["same_role_chunk_ids"] = [cid for cid in role_to_ids[role] if cid != c["chunk_id"]]


# ---------------------------------------------------------------------------
# ADLS helpers
# ---------------------------------------------------------------------------

def build_adls_clients() -> Tuple[ADLSFetcher, ADLSUploader]:
    account_name = ADLS_CONFIG["account_name"]
    account_key  = ADLS_CONFIG["account_key"]
    container    = ADLS_CONFIG["container_name"]
    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing in .env")
        sys.exit(1)
    return (
        ADLSFetcher(account_name, account_key, container),
        ADLSUploader(account_name, account_key, container),
    )


def iter_year_app_paths(fetcher: ADLSFetcher, year: int) -> Generator[str, None, None]:
    folder = f"year={year}" if DOC_TYPE == "hc" else str(year)
    base = f"{_ADLS_APP_PATH}/{folder}"
    return fetcher.list_files_iter(path=base, pattern="*.json", recursive=True)


def iter_year_proc_paths(fetcher: ADLSFetcher, year: int) -> Generator[str, None, None]:
    folder = f"year={year}" if DOC_TYPE == "hc" else str(year)
    base = f"{_ADLS_PROCESSED_PATH}/{folder}"
    return fetcher.list_files_iter(path=base, pattern="*_all_chunks.json", recursive=True)


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

def build_es_client() -> Elasticsearch:
    es_url = os.getenv("ES_URL")
    if not es_url:
        log.error("ES_URL not set in .env")
        sys.exit(1)
    api_key = os.getenv("ES_API_KEY")
    if api_key:
        es = Elasticsearch(es_url, api_key=api_key, request_timeout=120)
    else:
        user, password = os.getenv("ES_USER"), os.getenv("ES_PASS")
        if user and password:
            es = Elasticsearch(es_url, http_auth=(user, password), request_timeout=120)
        else:
            es = Elasticsearch(es_url, request_timeout=120)
    if not es.ping():
        log.error("Cannot connect to Elasticsearch — check ES_URL / credentials")
        sys.exit(1)
    log.info("Elasticsearch connected: %s", es_url)
    return es


def ensure_index(es: Elasticsearch, recreate: bool = False) -> None:
    exists = es.indices.exists(index=INDEX_NAME)
    if exists and recreate:
        es.indices.delete(index=INDEX_NAME)
        log.info("Dropped index: %s", INDEX_NAME)
        exists = False
    if not exists:
        if not SCHEMA_PATH.exists():
            log.error("Schema file not found: %s", SCHEMA_PATH)
            sys.exit(1)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        schema.setdefault("settings", {})
        schema["settings"].setdefault("number_of_shards", 8)
        schema["settings"].setdefault("number_of_replicas", 1)
        es.indices.create(index=INDEX_NAME, body=schema)
        log.info("Created index: %s", INDEX_NAME)
    else:
        log.info("Index already exists: %s", INDEX_NAME)


def bulk_mget_exists(es: Elasticsearch, doc_ids: List[str]) -> set:
    if not doc_ids:
        return set()
    resp = es.mget(index=INDEX_NAME, body={"ids": doc_ids}, _source=False)
    return {d["_id"] for d in resp["docs"] if d.get("found")}


def bulk_upload_with_retry(es: Elasticsearch, parent_docs: List[Dict]) -> Tuple[int, int]:
    remaining = list(parent_docs)
    total_ok = 0
    last_errors: list = []
    for attempt in range(ES_MAX_RETRIES):
        def _actions(docs):
            for doc in docs:
                yield {"_index": INDEX_NAME, "_id": doc.get("doc_id"), "_source": doc}
        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=BULK_BATCH_SIZE)
        total_ok += ok
        last_errors = errors
        if not errors:
            break
        failed_ids = set()
        for e in errors:
            op = e.get("index") or e.get("create") or {}
            if "_id" in op:
                failed_ids.add(op["_id"])
        remaining = [d for d in remaining if d.get("doc_id") in failed_ids]
        if not remaining:
            break
        log.warning("  Retrying %d failed docs (attempt %d/%d)", len(remaining), attempt + 2, ES_MAX_RETRIES)
        time.sleep(ES_RETRY_DELAY * (2 ** attempt))
    if last_errors and remaining:
        log.error("  %d docs permanently failed after %d retries", len(remaining), ES_MAX_RETRIES)
    return total_ok, len(remaining)


# ---------------------------------------------------------------------------
# GPU model
# ---------------------------------------------------------------------------

def load_model_multi_gpu() -> Tuple[SentenceTransformer, Optional[object]]:
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_CONFIG["model_name"])
    log.info("Loading embedding model: %s", model_name)
    n_gpu = torch.cuda.device_count()
    device = "cuda" if n_gpu > 0 else "cpu"
    if n_gpu == 0:
        log.warning("No CUDA devices found — running on CPU")
    model = SentenceTransformer(model_name, device=device)
    pool = None
    if n_gpu > 1:
        target_devices = [f"cuda:{i}" for i in range(n_gpu)]
        log.info("Starting multi-GPU pool on: %s", target_devices)
        pool = model.start_multi_process_pool(target_devices=target_devices)
    return model, pool


def embed_texts(model: SentenceTransformer, texts: List[str], pool: Optional[object]):
    import numpy as np
    if pool is not None:
        embeddings = model.encode_multi_process(texts, pool, batch_size=EMBED_BATCH_SIZE,
                                                show_progress_bar=len(texts) > 1000)
    else:
        embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE,
                                  show_progress_bar=len(texts) > 1000, convert_to_numpy=True)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------

def load_inventory(fetcher: ADLSFetcher, inv_path: str) -> set:
    """Read _inventory.json from ADLS, return set of filenames. Empty set if missing."""
    try:
        inv = fetcher.read_json_file(inv_path)
        if inv and "files" in inv:
            return set(inv["files"])
    except Exception:
        pass
    return set()


def save_inventory(
    uploader: ADLSUploader,
    inv_path: str,
    files_set: set,
    collection: str,
    year: int,
    year_dir: str,
) -> None:
    payload = json.dumps({
        "collection": collection, "year": year, "path": year_dir,
        "files": sorted(files_set), "file_count": len(files_set),
    }, ensure_ascii=False).encode("utf-8")
    try:
        fc = uploader.file_system_client.get_file_client(inv_path)
        fc.upload_data(payload, overwrite=True)
        log.info("  Saved _inventory.json for year %d (%d files)", year, len(files_set))
    except Exception as e:
        log.error("  Failed to save _inventory.json for year %d: %s", year, e)


def upload_inventory_es(fetcher: ADLSFetcher, year: int, done_ids: set) -> None:
    if not done_ids:
        return
    folder = f"year={year}" if DOC_TYPE == "hc" else str(year)
    adls_path = f"{_ADLS_PROCESSED_PATH}/{folder}/_inventory_es.json"
    collection = "HC_Judgements" if DOC_TYPE == "hc" else "SC_Judgements"
    payload = json.dumps({
        "collection": collection, "year": year, "path": adls_path,
        "files": sorted(done_ids), "file_count": len(done_ids),
    }, ensure_ascii=False).encode("utf-8")
    try:
        fc = fetcher.file_system_client.get_file_client(adls_path)
        fc.upload_data(payload, overwrite=True)
        log.info("  Uploaded _inventory_es.json for year %d (%d docs)", year, len(done_ids))
    except Exception as e:
        log.error("  Failed to upload _inventory_es.json for year %d: %s", year, e)


# ---------------------------------------------------------------------------
# Per-year combined pipeline
# ---------------------------------------------------------------------------

def process_year(
    year: int,
    fetcher: ADLSFetcher,
    uploader: ADLSUploader,
    es: Elasticsearch,
    text_cleaner: LegalTextCleaner,
    chunker: SemanticChunker,
    role_classifier,
    model: SentenceTransformer,
    pool: Optional[object],
    no_resume: bool = False,
) -> Dict:
    log.info("=" * 60)
    log.info("Year %d — starting", year)
    t_year_start = time.time()

    # ------------------------------------------------------------------
    # Inventory delta computation
    # ------------------------------------------------------------------
    folder = f"year={year}" if DOC_TYPE == "hc" else str(year)
    collection_dir = _DOC_TYPE_COLLECTION_DIRS[DOC_TYPE]
    collection_name = "HC_Judgements" if DOC_TYPE == "hc" else "SC_Judgements"
    proc_year_dir = f"{_ADLS_PROCESSED_PATH}/{folder}"

    app_inv_path  = f"{_ADLS_APP_PATH}/{folder}/_inventory.json"
    proc_inv_path = f"{_ADLS_PROCESSED_PATH}/{folder}/_inventory.json"
    es_inv_path   = f"{_ADLS_PROCESSED_PATH}/{folder}/_inventory_es.json"

    app_files  = load_inventory(fetcher, app_inv_path)   # basenames: "abc.json"
    proc_files = load_inventory(fetcher, proc_inv_path)  # basenames: "abc_all_chunks.json"
    es_files   = load_inventory(fetcher, es_inv_path)    # doc_ids

    if no_resume:
        delta1_basenames = set(app_files)
        delta2_basenames: set = set()
    else:
        delta1_basenames = {
            f for f in app_files
            if f.replace(".json", "_all_chunks.json") not in proc_files
        }
        # _inventory_es.json stores bare stems (e.g. "APHC010000032023_1_2023-06-14").
        # Strip the "_all_chunks.json" suffix to get the matching doc_id.
        def _proc_basename_to_doc_id(basename: str) -> str:
            return basename.replace("_all_chunks.json", "")
        delta2_basenames = {
            f for f in proc_files
            if _proc_basename_to_doc_id(f) not in es_files
        }

    log.info(
        "  Year %d: app=%d  proc=%d  es=%d  delta1=%d  delta2=%d",
        year, len(app_files), len(proc_files), len(es_files),
        len(delta1_basenames), len(delta2_basenames),
    )

    stats = {
        "year": year,
        "scanned": 0,
        "skipped_adls": len(app_files) - len(delta1_basenames),
        "skipped_es": len(proc_files) - len(delta2_basenames),
        "read_fail": 0,
        "processed": 0,
        "adls_uploaded": 0,
        "es_uploaded": 0,
        "es_errors": 0,
    }
    stats_lock = threading.Lock()

    # Accumulators written by stage6 and stage7 closures
    proc_uploaded_basenames: set = set()
    es_uploaded_ids: set = set()

    # ------------------------------------------------------------------
    # Stage queues
    # ------------------------------------------------------------------
    q_raw:       queue.Queue = queue.Queue()
    q_sentences: queue.Queue = queue.Queue()
    q_embedded:  queue.Queue = queue.Queue()
    q_chunks:    queue.Queue = queue.Queue()
    q_roles:     queue.Queue = queue.Queue()
    q_es:        queue.Queue = queue.Queue()  # fed by stage6 AND delta2_feeder

    errors: List[Exception] = []

    # ------------------------------------------------------------------
    # Stage 1: background lister + reader (delta1 only)
    # ------------------------------------------------------------------

    def _read_one(fp: str) -> Optional[Dict]:
        try:
            doc = fetcher.read_json_file(fp)
            if doc is None:
                return None
            doc["_source_file"] = fp
            return doc
        except Exception as e:
            log.warning("Read error %s: %s", fp, e)
            return None

    def stage1_reader():
        path_buf: List[str] = []

        def _flush_buf(buf: List[str]):
            threads = []
            sem = threading.Semaphore(IO_WORKERS)

            def _worker(fp):
                try:
                    doc = _read_one(fp)
                    if doc is None:
                        with stats_lock:
                            stats["read_fail"] += 1
                    else:
                        q_raw.put(doc)
                except Exception:
                    with stats_lock:
                        stats["read_fail"] += 1
                finally:
                    sem.release()

            for fp in buf:
                sem.acquire()
                t = threading.Thread(target=_worker, args=(fp,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        try:
            for fp in iter_year_app_paths(fetcher, year):
                if fp.endswith("_inventory.json") or "_done_" in fp:
                    continue
                basename = Path(fp).name
                if basename not in delta1_basenames:
                    continue
                with stats_lock:
                    stats["scanned"] += 1
                path_buf.append(fp)
                if len(path_buf) >= BUFFER_SIZE:
                    _flush_buf(path_buf)
                    path_buf = []
            if path_buf:
                _flush_buf(path_buf)
        except Exception as e:
            log.error("Stage 1 error: %s", e)
            errors.append(e)
        finally:
            q_raw.put(_STOP)

    # ------------------------------------------------------------------
    # Delta2 feeder: load already-processed docs directly into q_es
    # ------------------------------------------------------------------

    def delta2_feeder():
        try:
            # Build basename→full_path index by recursive listing (handles court/bench nesting).
            proc_path_map: Dict[str, str] = {}
            for fp in iter_year_proc_paths(fetcher, year):
                bn = Path(fp).name
                if bn not in proc_path_map:
                    proc_path_map[bn] = fp

            # Build app basename→full_path index for fallback requeue (same nesting).
            app_path_map: Dict[str, str] = {}
            for fp in iter_year_app_paths(fetcher, year):
                if fp.endswith("_inventory.json") or "_done_" in fp:
                    continue
                bn = Path(fp).name
                if bn not in app_path_map:
                    app_path_map[bn] = fp

            for basename in sorted(delta2_basenames):
                full_path = proc_path_map.get(basename)
                if full_path is None:
                    # Processed blob not found — fall back to full reprocess from app source.
                    app_basename = basename.replace("_all_chunks.json", ".json")
                    app_path = app_path_map.get(app_basename)
                    if app_path:
                        log.warning("  delta2 missing %s — requeueing via delta1", basename)
                        doc = _read_one(app_path)
                        if doc:
                            q_raw.put(doc)
                    else:
                        log.warning("  delta2 missing %s — no app blob found either", basename)
                    continue
                try:
                    chunks = fetcher.read_json_file(full_path)
                    if not isinstance(chunks, list) or not chunks:
                        app_basename = basename.replace("_all_chunks.json", ".json")
                        app_path = app_path_map.get(app_basename)
                        if app_path:
                            log.warning("  delta2 empty %s — requeueing via delta1", basename)
                            doc = _read_one(app_path)
                            if doc:
                                q_raw.put(doc)
                        continue
                    doc_id = chunks[0].get("doc_id") or _doc_id_from_chunks_path(full_path)
                    q_es.put((doc_id, full_path, chunks))
                except Exception as e:
                    app_basename = basename.replace("_all_chunks.json", ".json")
                    app_path = app_path_map.get(app_basename)
                    if app_path:
                        log.warning("  delta2 error %s — requeueing via delta1: %s", basename, e)
                        doc = _read_one(app_path)
                        if doc:
                            q_raw.put(doc)
        except Exception as e:
            log.error("Delta2 feeder error for year %d: %s", year, e)
            errors.append(e)
        finally:
            q_es.put(_STOP)

    # ------------------------------------------------------------------
    # Stage 2: clean + split sentences
    # ------------------------------------------------------------------
    def stage2_split(doc: Dict):
        source_file = doc.get("_source_file", "unknown.json")
        doc_id = generate_doc_id(source_file)
        all_chunks_path = get_all_chunks_path(source_file)
        try:
            text = doc.get("full_text") or doc.get("judgment_text") or doc.get("text", "")
            if not text:
                return
            cleaned = text_cleaner.clean(text)
            if not cleaned:
                return
            sentences = chunker._split_sentences(cleaned)
            if not sentences:
                return
            q_sentences.put((doc_id, doc, sentences, cleaned, all_chunks_path))
        except Exception as e:
            log.error("Stage 2 error for %s: %s", doc_id, e)
            errors.append(e)

    def stage2_producer():
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool_:
            futs = []
            while True:
                doc = q_raw.get()
                if doc is _STOP:
                    break
                futs.append(pool_.submit(stage2_split, doc))
            for f in as_completed(futs):
                exc = f.exception()
                if exc:
                    errors.append(exc)
        q_sentences.put(_STOP)

    # ------------------------------------------------------------------
    # Stage 3: GPU batch sentence encoding
    # ------------------------------------------------------------------
    def stage3_encode():
        pending_items = []
        pending_counts = []

        def flush():
            if not pending_items:
                return
            all_sents = []
            for _, _, sents, _, _ in pending_items:
                all_sents.extend(sents)
            embs = chunker.encode_batch(all_sents)
            offset = 0
            for (did, d, sents, ct, acp), cnt in zip(pending_items, pending_counts):
                doc_embs = embs[offset: offset + cnt]
                offset += cnt
                q_embedded.put((did, d, sents, doc_embs, ct, acp))
            pending_items.clear()
            pending_counts.clear()

        while True:
            item = q_sentences.get()
            if item is _STOP:
                flush()
                q_embedded.put(_STOP)
                return
            doc_id, doc, sentences, cleaned_text, all_chunks_path = item
            pending_items.append((doc_id, doc, sentences, cleaned_text, all_chunks_path))
            pending_counts.append(len(sentences))
            if sum(pending_counts) >= SENT_BATCH:
                flush()

    # ------------------------------------------------------------------
    # Stage 4: CPU chunk assembly
    # ------------------------------------------------------------------
    def stage4_assemble():
        assemble_workers = int(os.getenv("ASSEMBLE_WORKERS", "256"))

        def _assemble_one(item):
            doc_id, doc, sentences, sent_embs, cleaned_text, all_chunks_path = item
            chunks, chunk_texts = chunker.assemble_chunks_cpu(cleaned_text, sentences, sent_embs)
            if not chunks:
                return None
            return (doc_id, doc, all_chunks_path, chunks, chunk_texts)

        fut_q: queue.Queue = queue.Queue()

        def _submitter():
            with ThreadPoolExecutor(max_workers=assemble_workers) as p:
                while True:
                    item = q_embedded.get()
                    if item is _STOP:
                        break
                    fut_q.put(p.submit(_assemble_one, item))
            fut_q.put(_STOP)

        def _drainer():
            while True:
                f = fut_q.get()
                if f is _STOP:
                    q_chunks.put(_STOP)
                    return
                try:
                    result = f.result()
                except Exception as e:
                    errors.append(e)
                    continue
                if result is not None:
                    q_chunks.put(result)

        t_sub   = threading.Thread(target=_submitter, daemon=True)
        t_drain = threading.Thread(target=_drainer,   daemon=True)
        t_sub.start(); t_drain.start()
        t_sub.join(); t_drain.join()

    # ------------------------------------------------------------------
    # Stage 5: GPU batch role classification
    # ------------------------------------------------------------------
    def stage5_classify():
        pending = []
        pending_counts = []

        def flush():
            if not pending:
                return
            if role_classifier:
                all_texts = []
                for _, _, _, _, ctexts in pending:
                    all_texts.extend(ctexts)
                preds = role_classifier.predict(
                    all_texts,
                    batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
                    return_probabilities=True,
                )
                offset = 0
                for (did, d, acp, chunks, ctexts), cnt in zip(pending, pending_counts):
                    doc_preds = preds[offset: offset + cnt]
                    offset += cnt
                    enriched = []
                    for ch, pred in zip(chunks, doc_preds):
                        cd = ch.to_dict() if hasattr(ch, "to_dict") else dict(ch)
                        cd["role"] = pred["role"]
                        cd["confidence"] = float(pred["confidence"])
                        enriched.append(cd)
                    q_roles.put((did, d, acp, enriched))
            else:
                for (did, d, acp, chunks, _) in pending:
                    plain = [(ch.to_dict() if hasattr(ch, "to_dict") else dict(ch)) for ch in chunks]
                    q_roles.put((did, d, acp, plain))
            pending.clear()
            pending_counts.clear()

        while True:
            item = q_chunks.get()
            if item is _STOP:
                flush()
                q_roles.put(_STOP)
                return
            doc_id, doc, acp, chunks, chunk_texts = item
            pending.append((doc_id, doc, acp, chunks, chunk_texts))
            pending_counts.append(len(chunk_texts))
            if sum(pending_counts) >= CHUNK_BATCH:
                flush()

    # ------------------------------------------------------------------
    # Stage 6: ADLS upload → assemble all_chunks → pass to ES queue
    # ------------------------------------------------------------------
    def stage6_adls_upload():
        upload_batch_size = int(os.getenv("UPLOAD_BATCH_SIZE", "512"))
        pending: List[Tuple] = []
        upload_futs = []

        def _build_all_chunks(doc_id, doc, acp, chunk_dicts):
            source_file = doc.get("_source_file", "unknown.json")
            excluded = {"text", "full_text", "judgment_text", "embedding", "_source_file", "raw_html_text"}
            metadata = {k: v for k, v in doc.items() if k not in excluded}
            if isinstance(metadata.get("metadata"), dict):
                nested = metadata.pop("metadata")
                for k, v in nested.items():
                    if k not in {"raw_html", "description"}:
                        metadata[k] = v

            resolved_date = (
                metadata.get("date") or metadata.get("decision_date", "") or metadata.get("year", "")
            )
            config_jurisdiction = DOC_TYPE_CONFIG[0 if DOC_TYPE == "hc" else 1].get("jurisdiction")
            resolved_jurisdiction = config_jurisdiction or (
                metadata.get("jurisdiction") or metadata.get("court", "") or metadata.get("bench", "")
            )

            all_chunks_out = []
            for idx, cd in enumerate(chunk_dicts):
                all_chunks_out.append({
                    "doc_id":               doc_id,
                    "date":                 resolved_date,
                    "jurisdiction":         resolved_jurisdiction,
                    **metadata,
                    "id":                   f"{doc_id}_{idx}",
                    "chunk_id":             f"{doc_id}_{idx}",
                    "original_source_path": source_file,
                    "text":                 cd["text"],
                    "start_char":           cd.get("start_char", 0),
                    "end_char":             cd.get("end_char", 0),
                    "num_sentences":        cd.get("num_sentences", 0),
                    "doc_similarity":       float(cd.get("doc_similarity", 0.0)),
                    "avg_similarity":       float(cd.get("avg_similarity", 0.0)),
                    "role":                 cd.get("role", "Others"),
                    "confidence":           float(cd.get("confidence", 0.0)),
                })
            attach_same_role_chunk_ids(all_chunks_out)
            return all_chunks_out

        def _flush_upload(batch: List[Tuple]):
            if not batch:
                return
            adls_ok: Dict[str, bool] = {}

            def _upload_one(item):
                doc_id, acp, chunks = item
                ok = uploader.upload_json_file(data=chunks, adls_path=acp, overwrite=True)
                return doc_id, ok

            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as adls_pool:
                for doc_id, ok in adls_pool.map(_upload_one, batch):
                    adls_ok[doc_id] = ok

            with stats_lock:
                for doc_id, acp, chunks in batch:
                    if adls_ok.get(doc_id):
                        stats["adls_uploaded"] += 1
                        proc_uploaded_basenames.add(Path(acp).name)
                    q_es.put((doc_id, acp, chunks))

        pbar = tqdm(desc=f"Year {year} (chunk+ADLS)", unit="doc")
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as upload_pool:
            while True:
                item = q_roles.get()
                if item is _STOP:
                    if pending:
                        upload_futs.append(upload_pool.submit(_flush_upload, list(pending)))
                    for f in upload_futs:
                        try:
                            f.result()
                        except Exception as e:
                            log.error("ADLS upload error: %s", e)
                    # Update processed inventory with newly uploaded files
                    if proc_uploaded_basenames:
                        save_inventory(
                            uploader, proc_inv_path,
                            proc_files | proc_uploaded_basenames,
                            collection_name, year, proc_year_dir,
                        )
                    pbar.close()
                    q_es.put(_STOP)
                    return
                doc_id, doc, acp, chunk_dicts = item
                all_chunks = _build_all_chunks(doc_id, doc, acp, chunk_dicts)
                pending.append((doc_id, acp, all_chunks))
                with stats_lock:
                    stats["processed"] += 1
                pbar.update(1)
                if len(pending) >= upload_batch_size:
                    upload_futs.append(upload_pool.submit(_flush_upload, list(pending)))
                    pending.clear()

    # ------------------------------------------------------------------
    # Stage 7: ES top-K select → embed → bulk upload
    # Stage6 sends one _STOP; delta2_feeder sends one _STOP.
    # Stage7 waits for both before exiting.
    # ------------------------------------------------------------------
    def stage7_es_upload():
        es_buf: List[Tuple[str, List[Dict]]] = []
        pending_stops = 2  # one from stage6, one from delta2_feeder

        def _flush_es(buf: List[Tuple[str, List[Dict]]]):
            # mget pre-filter: skip docs already in ES (handles stale inventories)
            found = bulk_mget_exists(es, [doc_id for doc_id, _ in buf])
            if found:
                with stats_lock:
                    stats["skipped_es"] += len(found)
                buf = [(did, chunks) for did, chunks in buf if did not in found]
            if not buf:
                return

            prepared = []
            all_texts = []
            offsets = []
            for doc_id, all_chunks in buf:
                try:
                    indices = weighted_topk_selection(
                        all_chunks, top_k=TOP_K,
                        similarity_key="doc_similarity", role_weights=ROLE_WEIGHTS,
                    )
                    selected = [all_chunks[i] for i in indices]
                    metadata = {k: all_chunks[0][k] for k in METADATA_FIELDS if k in all_chunks[0]}
                    subservice = all_chunks[0].get("subservice", {})
                    if isinstance(subservice, dict) and "service_type" in subservice:
                        metadata["service_type"] = subservice["service_type"]
                    metadata.setdefault("all_chunks_path", all_chunks[0].get("original_source_path", ""))
                    slim_chunks = [{k: c[k] for k in CHUNK_FIELDS if k in c} for c in selected]
                    texts = [c.get("text", "") for c in selected]
                    offsets.append(len(all_texts))
                    all_texts.extend(texts)
                    prepared.append((doc_id, metadata, slim_chunks))
                except Exception as e:
                    log.warning("  Skipping %s during ES prep: %s", doc_id, e)
                    offsets.append(None)
                    prepared.append(None)

            if not all_texts:
                return

            embeddings = embed_texts(model, all_texts, pool)
            with stats_lock:
                stats["embedded_texts"] = stats.get("embedded_texts", 0) + len(all_texts)

            parent_docs = []
            for i, item in enumerate(prepared):
                if item is None or offsets[i] is None:
                    continue
                doc_id, metadata, slim_chunks = item
                offset = offsets[i]
                for j, sc in enumerate(slim_chunks):
                    sc["embedding"] = embeddings[offset + j].tolist()
                doc = dict(metadata)
                for bool_field in ("pdf_exists",):
                    if bool_field in doc and isinstance(doc[bool_field], str):
                        doc[bool_field] = doc[bool_field].strip().lower() == "true"
                doc["doc_id"] = doc_id
                doc["chunks"] = slim_chunks
                parent_docs.append(doc)

            if not parent_docs:
                return

            ok, errs = bulk_upload_with_retry(es, parent_docs)
            with stats_lock:
                stats["es_uploaded"] += ok
                stats["es_errors"] += errs

            for doc in parent_docs:
                did = doc.get("doc_id", "")
                if did:
                    es_uploaded_ids.add(did)

            log.info(
                "  Buffer done — processed=%d  adls=%d  es=%d  errors=%d",
                stats["processed"], stats["adls_uploaded"],
                stats["es_uploaded"], stats["es_errors"],
            )

        nonlocal_stops = [pending_stops]
        while nonlocal_stops[0] > 0:
            item = q_es.get()
            if item is _STOP:
                nonlocal_stops[0] -= 1
                if nonlocal_stops[0] == 0 and es_buf:
                    _flush_es(es_buf)
                    es_buf.clear()
                continue
            doc_id, acp, all_chunks = item
            es_buf.append((doc_id, all_chunks))
            if len(es_buf) >= BUFFER_SIZE:
                _flush_es(es_buf)
                es_buf.clear()

        # Merge with any existing ES inventory and write
        all_es_ids = es_files | es_uploaded_ids
        upload_inventory_es(fetcher, year, all_es_ids)

    # ------------------------------------------------------------------
    # Launch all stages
    # ------------------------------------------------------------------
    threads = [
        threading.Thread(target=stage1_reader,      name="s1-reader",   daemon=True),
        threading.Thread(target=delta2_feeder,       name="s1-delta2",   daemon=True),
        threading.Thread(target=stage2_producer,     name="s2-split",    daemon=True),
        threading.Thread(target=stage3_encode,       name="s3-encode",   daemon=True),
        threading.Thread(target=stage4_assemble,     name="s4-assemble", daemon=True),
        threading.Thread(target=stage5_classify,     name="s5-classify", daemon=True),
        threading.Thread(target=stage6_adls_upload,  name="s6-adls",     daemon=True),
        threading.Thread(target=stage7_es_upload,    name="s7-es",       daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - t_year_start
    stats["elapsed_s"] = round(elapsed, 1)
    stats["throughput_docs_s"] = round(stats["es_uploaded"] / max(elapsed, 1), 2)

    log.info(
        "Year %d done in %.0fs | scanned=%d  skip_adls=%d  skip_es=%d  "
        "processed=%d  adls_uploaded=%d  es_uploaded=%d  errors=%d  (%.1f docs/s)",
        year, elapsed,
        stats["scanned"], stats["skipped_adls"], stats["skipped_es"],
        stats["processed"], stats["adls_uploaded"],
        stats["es_uploaded"], stats["es_errors"],
        stats["throughput_docs_s"],
    )
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="app/ → chunk → ADLS processed/ → embed → ES  (combined pipeline)"
    )
    p.add_argument("--doc-type", choices=["hc", "sc"], default="hc", dest="doc_type")
    p.add_argument("--years", nargs="+", type=int, default=None, metavar="YEAR")
    p.add_argument("--year-range", nargs=2, type=int, default=None, metavar=("FROM", "TO"))
    p.add_argument("--no-resume", action="store_true", dest="no_resume",
                   help="Disable resume — reprocess all app/ docs ignoring inventories")
    p.add_argument("--recreate-index", action="store_true", dest="recreate_index",
                   help="Drop and recreate ES index (destructive)")
    p.add_argument("--top-k", type=int, default=None, dest="top_k",
                   help=f"Chunks per doc for ES (default: {TOP_K})")
    p.add_argument("--reverse", action="store_true",
                   help="Process years in descending order")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global DOC_TYPE, _ADLS_APP_PATH, _ADLS_PROCESSED_PATH, INDEX_NAME, TOP_K
    DOC_TYPE = args.doc_type
    _ADLS_APP_PATH       = _DOC_TYPE_APP_ROOTS[DOC_TYPE]
    _ADLS_PROCESSED_PATH = _DOC_TYPE_PROCESSED_ROOTS[DOC_TYPE]
    INDEX_NAME           = _DOC_TYPE_INDEX_NAMES[DOC_TYPE]
    if args.top_k:
        TOP_K = args.top_k

    log.info("Doc type : %s  |  app/ : %s  |  processed/ : %s  |  index : %s",
             DOC_TYPE.upper(), _ADLS_APP_PATH, _ADLS_PROCESSED_PATH, INDEX_NAME)

    if args.year_range and args.years:
        log.error("Use --years or --year-range, not both")
        sys.exit(1)
    if args.year_range:
        a, b = args.year_range
        target_years = list(range(min(a, b), max(a, b) + 1))
    elif args.years:
        target_years = args.years
    else:
        target_years = list(range(2000, 2027))

    fetcher, uploader = build_adls_clients()
    es = build_es_client()
    ensure_index(es, recreate=args.recreate_index)

    log.info("Loading chunking models ...")
    text_cleaner = LegalTextCleaner()
    chunker = SemanticChunker(
        model_name=EMBEDDING_CONFIG["model_name"],
        similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
        min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
        max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
        min_chunk_size=CHUNKING_CONFIG["min_chunk_size"],
    )
    role_classifier = None
    if ROLE_CLASSIFICATION_CONFIG.get("enabled"):
        log.info("Loading role classifier ...")
        role_classifier = create_classifier_from_config()

    model, gpu_pool = load_model_multi_gpu()

    all_stats = []
    total_start = time.time()

    for i, year in enumerate(sorted(target_years, reverse=args.reverse), 1):
        log.info("[%d/%d] Processing year %d", i, len(target_years), year)
        year_stats = process_year(
            year=year,
            fetcher=fetcher,
            uploader=uploader,
            es=es,
            text_cleaner=text_cleaner,
            chunker=chunker,
            role_classifier=role_classifier,
            model=model,
            pool=gpu_pool,
            no_resume=args.no_resume,
        )
        all_stats.append(year_stats)

    if gpu_pool is not None:
        model.stop_multi_process_pool(gpu_pool)
        log.info("Multi-GPU pool stopped")

    total_docs = sum(s["es_uploaded"] for s in all_stats)
    total_errors = sum(s["es_errors"] for s in all_stats)
    total_elapsed = time.time() - total_start

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("Years : %s", sorted(target_years))
    log.info("ES uploaded  : %s", f"{total_docs:,}")
    log.info("ES errors    : %s", f"{total_errors:,}")
    log.info("Wall time    : %.0fs (%.1fmin)", total_elapsed, total_elapsed / 60)
    for s in all_stats:
        log.info("  year=%s  processed=%s  adls=%s  es=%s  errors=%s  %.0fs",
                 s["year"], f"{s['processed']:,}", f"{s['adls_uploaded']:,}",
                 f"{s['es_uploaded']:,}", f"{s['es_errors']:,}", s["elapsed_s"])


if __name__ == "__main__":
    main()

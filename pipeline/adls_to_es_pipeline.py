"""
Production pipeline: ADLS → Elasticsearch, year-wise.

Supports High Court (hc) and Supreme Court (sc) judgements via --doc-type flag.
For each year:
  1. Scan ADLS processed/{HC|SC}/year={Y}/ for *_all_chunks.json
  2. Background reader thread pre-fetches batches into a bounded queue
  3. Main thread: ES mget pre-filter → select top-K chunks → GPU embed → bulk upload
  4. Write per-year JSONL progress file (append-only) for resumability

Resume modes (default ON — use --no-resume to disable):
  local resume    Skip doc_ids in local JSONL progress files (fastest)
  ES resume       Skip doc_ids already in ES via single mget per buffer
  --verify-resume Sanity-check local progress against ES, report discrepancies (read-only)

Usage:
  python pipeline/adls_to_es_pipeline.py --doc-type hc [--years 2020 2021 ...] [--recreate-index]
  python pipeline/adls_to_es_pipeline.py --doc-type sc --year-range 2020 2025
  python pipeline/adls_to_es_pipeline.py --verify-resume --doc-type sc --years 2023

Environment:
  ES_URL                  Elasticsearch endpoint
  ES_API_KEY              Elasticsearch API key (optional, basic-auth fallback via ES_USER/ES_PASS)
  ES_USER / ES_PASS       Elasticsearch credentials (if no API key)
  ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
  EMBEDDING_MODEL         (default: sentence-transformers/all-MiniLM-L6-v2)
  EMBEDDING_BATCH_SIZE    per-GPU batch size (default: 2048 — tuned for A100 40GB)
  IO_WORKERS              parallel ADLS read threads (default: 32)
  ES_BULK_BATCH_SIZE      docs per ES bulk request (default: 100)
  PIPELINE_BUFFER_SIZE    paths per processing window (default: 500)
  ES_MAX_RETRIES          retry attempts on transient failures (default: 3)
  ES_RETRY_DELAY          base retry delay seconds, exponential backoff (default: 2.0)
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from sentence_transformers import SentenceTransformer

from config import ADLS_CONFIG, DOC_TYPE_CONFIG, EMBEDDING_CONFIG
from core.adls_fetcher import ADLSFetcher
from utils.weighted_selector import weighted_topk_selection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("adls_to_es_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("adls_to_es")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants — all tunable via env vars without code changes
# ---------------------------------------------------------------------------
YEARS: List[int] = list(range(2020, 2026))

# These are resolved at startup from --doc-type; placeholders replaced in main()
_ADLS_PROCESSED_PATH: str = ""   # e.g. "processed/High_Court_Judgements"
INDEX_NAME: str = ""              # e.g. "hc_judgements"
DOC_TYPE: str = "hc"             # "hc" or "sc" — set in main() from --doc-type

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sample" / "es_schema_prod.json"

# ADLS processed-path roots (no trailing slash)
_DOC_TYPE_ADLS_ROOTS = {
    "hc": "processed/" + DOC_TYPE_CONFIG[0]["adls_input_path"].lstrip("app/").rstrip("/"),
    "sc": "processed/" + DOC_TYPE_CONFIG[1]["adls_input_path"].lstrip("app/").rstrip("/"),
}
# ES index names per doc-type (distinct from Azure AI Search names in config)
_DOC_TYPE_INDEX_NAMES = {
    "hc": "hc_judgements",
    "sc": "sc_judgements",
}

TOP_K           = 12
IO_WORKERS      = int(os.getenv("IO_WORKERS",              "32"))
BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE",      "100"))
BUFFER_SIZE     = int(os.getenv("PIPELINE_BUFFER_SIZE",    "500"))
ES_MAX_RETRIES  = int(os.getenv("ES_MAX_RETRIES",          "3"))
ES_RETRY_DELAY  = float(os.getenv("ES_RETRY_DELAY",        "2.0"))
EMBED_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE",   "2048"))
PROGRESS_DIR    = Path("pipeline_progress")

METADATA_FIELDS = [
    "doc_id", "date", "jurisdiction", "doc_name", "year", "bench",
    "court_code", "title", "judge", "pdf_link", "cnr",
    "date_of_registration", "decision_date", "disposal_nature",
    "court", "pdf_exists", "original_source_path", "all_chunks_path",
    "source_file",
]
CHUNK_FIELDS = ["chunk_id", "text", "role", "same_role_chunk_ids"]

# subservice.service_type — the service-type field path per user instruction
SERVICE_TYPE_FIELD = "subservice.service_type"

# Role weights tuned for retrieval utility.
# High: roles that directly answer legal queries or identify a case.
# Low: Reasoning/Arguments are verbose and retrievable from ADLS on demand.
PIPELINE_ROLE_WEIGHTS: Dict[str, float] = {
    "Decision":   3.0,   # The ruling — top retrieval target for legal queries
    "Precedents": 3.0,   # Prior case citations — core of precedent search
    "Issues":     2.5,   # Legal questions framed by court — primary case association signal
    "Preamble":   2.5,   # Judges, parties, court profile — metadata-dense identifier
    "Facts":      2.5,   # Case scenario — primary anchor for fact-based queries
    "Statute":    1.5,   # Act/section refs — useful for statute-based search
    "Reasoning":  0.4,   # Court analysis — verbose, fetchable from ADLS later
    "Arguments":  0.3,   # Party submissions — verbose, fetchable from ADLS later
    "Others":     0.2,   # Noise
}


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
        user = os.getenv("ES_USER")
        password = os.getenv("ES_PASS")
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
        schema["settings"]["number_of_shards"] = schema["settings"].get("number_of_shards", 8)
        schema["settings"]["number_of_replicas"] = schema["settings"].get("number_of_replicas", 1)
        es.indices.create(index=INDEX_NAME, body=schema)
        log.info("Created index: %s  (shards=%s, replicas=%s)",
                 INDEX_NAME,
                 schema["settings"]["number_of_shards"],
                 schema["settings"]["number_of_replicas"])
    else:
        log.info("Index already exists: %s", INDEX_NAME)


def bulk_mget_exists(es: Elasticsearch, doc_ids: List[str]) -> set:
    """Single mget call to check which doc_ids exist in the index. Returns set of found IDs."""
    if not doc_ids:
        return set()
    resp = es.mget(index=INDEX_NAME, body={"ids": doc_ids}, _source=False)
    return {d["_id"] for d in resp["docs"] if d.get("found")}


def bulk_upload_with_retry(es: Elasticsearch, parent_docs: List[Dict]) -> Tuple[int, int]:
    """Bulk upload with exponential-backoff retry. Retries only the failed subset."""
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

        failed_ids: set = set()
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
# GPU embedding
# ---------------------------------------------------------------------------

def load_model_multi_gpu() -> Tuple[SentenceTransformer, Optional[object]]:
    """
    Load SentenceTransformer and start a multi-process pool across all available GPUs.
    Falls back to single-GPU or CPU if no CUDA devices are detected.
    """
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_CONFIG["model_name"])
    log.info("Loading embedding model: %s", model_name)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        log.info("CUDA devices available: %d", n_gpu)
        device = "cuda"
    else:
        log.warning("No CUDA devices found — running on CPU (expect slow embedding)")
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)

    pool = None
    if n_gpu > 1:
        target_devices = [f"cuda:{i}" for i in range(n_gpu)]
        log.info("Starting multi-GPU pool on: %s", target_devices)
        pool = model.start_multi_process_pool(target_devices=target_devices)

    return model, pool


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    pool: Optional[object],
) -> "np.ndarray":
    """
    Embed texts. Uses multi-process pool when available (one process per GPU),
    otherwise falls back to single-device encode.
    """
    import numpy as np

    if pool is not None:
        embeddings = SentenceTransformer.encode_multi_process(
            texts,
            pool,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(texts) > 1000,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
        )

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    return embeddings


# ---------------------------------------------------------------------------
# ADLS reading
# ---------------------------------------------------------------------------

def iter_year_paths(fetcher: ADLSFetcher, year: int) -> Generator[str, None, None]:
    base = f"{_ADLS_PROCESSED_PATH}/year={year}"
    return fetcher.list_files_iter(path=base, pattern="*_all_chunks.json", recursive=True)


def read_chunks_file_with_retry(fetcher: ADLSFetcher, path: str) -> Optional[List[Dict]]:
    """Read a chunks file from ADLS with exponential-backoff retry on failure."""
    for attempt in range(ES_MAX_RETRIES):
        try:
            data = fetcher.read_json_file(path)
            if isinstance(data, list) and data:
                return data
        except Exception as e:
            if attempt < ES_MAX_RETRIES - 1:
                time.sleep(ES_RETRY_DELAY * (2 ** attempt))
            else:
                log.warning("Failed to read %s after %d attempts: %s", path, ES_MAX_RETRIES, e)
    return None


def read_batch_parallel(
    fetcher: ADLSFetcher,
    paths: List[str],
    workers: int = IO_WORKERS,
) -> List[Tuple[str, List[Dict]]]:
    # Use raw threads instead of ThreadPoolExecutor to avoid
    # "cannot schedule new futures after interpreter shutdown" when the
    # sentence-transformers multi-process pool is also running.
    results: List[Tuple[str, List[Dict]]] = []
    lock = threading.Lock()
    sem = threading.Semaphore(workers)

    def _worker(path: str) -> None:
        try:
            chunks = read_chunks_file_with_retry(fetcher, path)
            if chunks:
                with lock:
                    results.append((path, chunks))
        except Exception as e:
            log.warning("Error reading %s: %s", path, e)
        finally:
            sem.release()

    threads = []
    for p in paths:
        sem.acquire()
        t = threading.Thread(target=_worker, args=(p,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

def extract_metadata(chunk: Dict) -> Dict:
    meta = {k: chunk[k] for k in METADATA_FIELDS if k in chunk}
    # Flatten subservice.service_type into the document if present
    subservice = chunk.get("subservice", {})
    if isinstance(subservice, dict) and "service_type" in subservice:
        meta["service_type"] = subservice["service_type"]
    return meta


def select_and_prepare(
    path: str,
    chunks: List[Dict],
    role_weights: Dict[str, float],
) -> Tuple[Dict, List[Dict], List[str]]:
    """
    Returns (metadata_dict, slim_chunks_without_embeddings, text_list).
    slim_chunks each contain CHUNK_FIELDS only; embeddings added later.
    """
    indices = weighted_topk_selection(
        chunks, top_k=TOP_K, similarity_key="doc_similarity", role_weights=role_weights
    )
    selected = [chunks[i] for i in indices]

    metadata = extract_metadata(chunks[0])
    if "all_chunks_path" not in metadata:
        metadata["all_chunks_path"] = path

    slim_chunks = [{k: c[k] for k in CHUNK_FIELDS if k in c} for c in selected]
    texts = [c.get("text", "") for c in selected]

    return metadata, slim_chunks, texts


# ---------------------------------------------------------------------------
# Local progress — JSONL append (O(new) per flush, not O(total))
# ---------------------------------------------------------------------------

def _doc_id_from_path(path: str) -> str:
    """Best-effort doc_id extraction from ADLS path without reading the file."""
    name = Path(path).stem           # e.g. "HC_123456_all_chunks"
    if name.endswith("_all_chunks"):
        return name[: -len("_all_chunks")]
    return name


def progress_path(year: int) -> Path:
    PROGRESS_DIR.mkdir(exist_ok=True)
    return PROGRESS_DIR / f"done_ids_{DOC_TYPE}_year={year}.jsonl"


def load_done_ids(year: int) -> set:
    ids: set = set()
    # Backward compat: load legacy .json (un-namespaced) if present
    old_json = PROGRESS_DIR / f"done_ids_year={year}.json"
    old_jsonl = PROGRESS_DIR / f"done_ids_year={year}.jsonl"
    for old in (old_json, old_jsonl):
        if old.exists():
            try:
                if old.suffix == ".json":
                    ids.update(json.loads(old.read_text(encoding="utf-8")))
                else:
                    for line in old.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            ids.add(line.strip('"'))
                log.info("  Loaded %d IDs from legacy progress file: %s", len(ids), old)
            except Exception as e:
                log.warning("  Could not load legacy progress file %s: %s", old, e)
    # Load current namespaced JSONL
    p = progress_path(year)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.add(line.strip('"'))
    if ids:
        log.info("  Resuming %s year %d: %d done IDs loaded", DOC_TYPE.upper(), year, len(ids))
    return ids


def append_done_ids(year: int, new_ids: set) -> None:
    if not new_ids:
        return
    with progress_path(year).open("a", encoding="utf-8") as f:
        for id_ in new_ids:
            f.write(json.dumps(id_) + "\n")


# ---------------------------------------------------------------------------
# Background reader thread — overlaps ADLS I/O with GPU embedding
# ---------------------------------------------------------------------------

def _background_reader(
    fetcher: ADLSFetcher,
    year: int,
    done_ids: set,
    local_resume: bool,
    out_queue: queue.Queue,
    stats: Dict,
) -> None:
    """Scan ADLS paths, apply local filter, read batches in parallel, put into out_queue.

    out_queue items: ("batch", path_buf, read_results) or ("done", None, None).
    Queue maxsize=2 caps prefetch at one batch ahead, preventing runaway RAM usage.
    """
    path_buf: List[str] = []
    try:
        for path in iter_year_paths(fetcher, year):
            stats["scanned"] += 1
            if local_resume and done_ids and _doc_id_from_path(path) in done_ids:
                stats["skipped_local"] += 1
                continue
            path_buf.append(path)
            if len(path_buf) >= BUFFER_SIZE:
                batch = read_batch_parallel(fetcher, path_buf)
                out_queue.put(("batch", path_buf, batch))
                path_buf = []
        if path_buf:
            batch = read_batch_parallel(fetcher, path_buf)
            out_queue.put(("batch", path_buf, batch))
    except Exception as e:
        log.error("Background reader crashed for year %d: %s", year, e)
    finally:
        out_queue.put(("done", None, None))


# ---------------------------------------------------------------------------
# Per-year pipeline
# ---------------------------------------------------------------------------

def process_year(
    year: int,
    fetcher: ADLSFetcher,
    es: Elasticsearch,
    model: SentenceTransformer,
    pool: Optional[object],
    resume: bool,
    local_resume: bool,
    role_weights: Dict[str, float],
) -> Dict:
    log.info("=" * 60)
    log.info("Year %d — starting", year)
    t_year_start = time.time()

    done_ids: set = load_done_ids(year) if local_resume else set()
    stats = {
        "year": year,
        "scanned": 0,
        "skipped_local": 0,
        "skipped_es": 0,
        "read_ok": 0,
        "read_fail": 0,
        "embedded": 0,
        "uploaded_ok": 0,
        "upload_errors": 0,
    }

    # Background reader fills the queue while the main thread runs GPU embedding
    read_queue: queue.Queue = queue.Queue(maxsize=2)
    reader = threading.Thread(
        target=_background_reader,
        args=(fetcher, year, done_ids, local_resume, read_queue, stats),
        daemon=True,
    )
    reader.start()

    while True:
        tag, paths, raw = read_queue.get()
        if tag == "done":
            break

        if not raw:
            continue

        stats["read_ok"] += len(raw)
        stats["read_fail"] += len(paths or []) - len(raw)

        # --- ES mget pre-filter: single HTTP call for the whole buffer ---
        if resume:
            path_ids = {p: _doc_id_from_path(p) for p, _ in raw}
            found_in_es = bulk_mget_exists(es, list(path_ids.values()))
            if found_in_es:
                new_es_ids = found_in_es - done_ids
                done_ids.update(new_es_ids)
                append_done_ids(year, new_es_ids)
                stats["skipped_es"] += len(found_in_es)
                raw = [(p, c) for p, c in raw if path_ids[p] not in found_in_es]
            if not raw:
                continue

        # --- Select chunks & gather texts ---
        prepared: List[Optional[Tuple[Dict, List[Dict]]]] = []
        all_texts: List[str] = []
        chunk_offsets: List[Optional[int]] = []

        for path, chunks in raw:
            try:
                metadata, slim_chunks, texts = select_and_prepare(path, chunks, role_weights)
                chunk_offsets.append(len(all_texts))
                all_texts.extend(texts)
                prepared.append((metadata, slim_chunks))
            except Exception as e:
                log.warning("  Skipping %s during preparation: %s", path, e)
                chunk_offsets.append(None)
                prepared.append(None)

        if not all_texts:
            continue

        # --- Embed (GPU) — background reader pre-fetches next batch concurrently ---
        log.debug("  Embedding %d texts for %d docs ...", len(all_texts), len(prepared))
        embeddings = embed_texts(model, all_texts, pool)
        stats["embedded"] += len(all_texts)

        # --- Attach embeddings & build parent docs ---
        parent_docs: List[Dict] = []
        for i, item in enumerate(prepared):
            if item is None:
                continue
            metadata, slim_chunks = item
            offset = chunk_offsets[i]
            if offset is None:
                continue
            for j, sc in enumerate(slim_chunks):
                sc["embedding"] = embeddings[offset + j].tolist()
            doc = dict(metadata)
            doc["chunks"] = slim_chunks
            parent_docs.append(doc)

        if not parent_docs:
            continue

        # --- Bulk upload with retry ---
        ok, errs = bulk_upload_with_retry(es, parent_docs)
        stats["uploaded_ok"] += ok
        stats["upload_errors"] += errs

        # --- Append only newly processed IDs (O(new), not O(total)) ---
        new_ids: set = set()
        for doc in parent_docs:
            doc_id = doc.get("doc_id", "")
            if doc_id and doc_id not in done_ids:
                new_ids.add(doc_id)
                done_ids.add(doc_id)
        append_done_ids(year, new_ids)

        log.info(
            "  Batch done — scanned=%d  read_ok=%d  embedded=%d  uploaded=%d  errors=%d",
            stats["scanned"], stats["read_ok"], stats["embedded"],
            stats["uploaded_ok"], stats["upload_errors"],
        )

    reader.join(timeout=60)

    elapsed = time.time() - t_year_start
    stats["elapsed_s"] = round(elapsed, 1)
    stats["throughput_docs_s"] = round(stats["uploaded_ok"] / max(elapsed, 1), 2)

    log.info(
        "Year %d done in %.0fs | scanned=%d  skip_local=%d  skip_es=%d  "
        "read_ok=%d  uploaded=%d  errors=%d  (%.1f docs/s)",
        year, elapsed,
        stats["scanned"], stats["skipped_local"], stats["skipped_es"],
        stats["read_ok"], stats["uploaded_ok"], stats["upload_errors"],
        stats["throughput_docs_s"],
    )
    return stats


# ---------------------------------------------------------------------------
# Index stats
# ---------------------------------------------------------------------------

def log_index_stats(es: Elasticsearch) -> None:
    try:
        es.indices.refresh(index=INDEX_NAME)
        stats = es.indices.stats(index=INDEX_NAME, metric="store")
        count = es.count(index=INDEX_NAME)["count"]
        size_bytes = stats["indices"][INDEX_NAME]["primaries"]["store"]["size_in_bytes"]
        size_gb = size_bytes / (1024 ** 3)
        log.info("=" * 60)
        log.info("Index : %s", INDEX_NAME)
        log.info("Docs  : %s", f"{count:,}")
        log.info("Size  : %.2f GB  (%s bytes)", size_gb, f"{size_bytes:,}")
        if count:
            log.info("Avg   : %.1f KB/doc", size_bytes / count / 1024)
        log.info("=" * 60)
    except Exception as e:
        log.warning("Could not fetch index stats: %s", e)


# ---------------------------------------------------------------------------
# --verify-resume: sanity-check local progress against ES (read-only)
# ---------------------------------------------------------------------------

def verify_resume(es: Elasticsearch, years: List[int]) -> None:
    log.info("Verifying local progress against Elasticsearch ...")
    for year in sorted(years):
        done = load_done_ids(year)
        if not done:
            log.info("  Year %d: no local progress found", year)
            continue
        ids = list(done)
        missing = []
        for i in range(0, len(ids), 1000):
            batch = ids[i : i + 1000]
            found = bulk_mget_exists(es, batch)
            missing.extend(id_ for id_ in batch if id_ not in found)
        if missing:
            log.warning(
                "  Year %d: %d local done / %d in ES → %d MISSING (re-run --years %d to fix)",
                year, len(done), len(done) - len(missing), len(missing), year,
            )
        else:
            log.info("  Year %d: %d local done / all present in ES", year, len(done))
    log.info("Verification complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload HC or SC judgements from ADLS to Elasticsearch, year by year."
    )
    p.add_argument(
        "--doc-type", choices=["hc", "sc"], default="hc", dest="doc_type",
        help="Document type: hc = High Court, sc = Supreme Court (default: hc)",
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=None,
        metavar="YEAR",
        help="Specific years to process (e.g. --years 2022 2023)",
    )
    p.add_argument(
        "--year-range", nargs=2, type=int, default=None,
        metavar=("FROM", "TO"),
        help="Inclusive year range (e.g. --year-range 2020 2025)",
    )
    p.add_argument(
        "--no-resume", action="store_true", dest="no_resume",
        help="Disable all resume checks — process every doc regardless of prior runs (use with --recreate-index)",
    )
    p.add_argument(
        "--recreate-index", action="store_true", dest="recreate_index",
        help="Drop and recreate the ES index before starting (destructive — all existing data lost)",
    )
    p.add_argument(
        "--top-k", type=int, default=TOP_K, dest="top_k",
        help=f"Chunks to select per document (default: {TOP_K})",
    )
    p.add_argument(
        "--verify-resume", action="store_true", dest="verify_resume",
        help="Check local done IDs against ES and report discrepancies (read-only, no upload)",
    )
    p.add_argument(
        "--reverse", action="store_true",
        help="Process years in descending order (most recent first)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Resolve doc-type globals before anything else ---
    global DOC_TYPE, _ADLS_PROCESSED_PATH, INDEX_NAME, TOP_K
    DOC_TYPE = args.doc_type
    _ADLS_PROCESSED_PATH = _DOC_TYPE_ADLS_ROOTS[DOC_TYPE]
    INDEX_NAME = _DOC_TYPE_INDEX_NAMES[DOC_TYPE]
    log.info("Doc type : %s  |  ADLS path : %s  |  Index : %s",
             DOC_TYPE.upper(), _ADLS_PROCESSED_PATH, INDEX_NAME)

    if args.year_range and args.years:
        log.error("Use --years or --year-range, not both")
        sys.exit(1)
    if args.year_range:
        a, b = args.year_range
        target_years = list(range(min(a, b), max(a, b) + 1))
    elif args.years:
        target_years = args.years
    else:
        target_years = YEARS

    # --- Validate ADLS credentials ---
    account_name = ADLS_CONFIG["account_name"]
    account_key = ADLS_CONFIG["account_key"]
    container = ADLS_CONFIG["container_name"]
    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing — set ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME in .env")
        sys.exit(1)

    es = build_es_client()

    if args.verify_resume:
        verify_resume(es, target_years)
        return

    ensure_index(es, recreate=args.recreate_index)

    fetcher = ADLSFetcher(account_name, account_key, container)

    TOP_K = args.top_k

    model, pool = load_model_multi_gpu()

    role_weights = PIPELINE_ROLE_WEIGHTS

    all_stats = []
    total_start = time.time()

    for i, year in enumerate(sorted(target_years, reverse=args.reverse), 1):
        log.info("[%d/%d] Processing year %d", i, len(target_years), year)
        year_stats = process_year(
            year=year,
            fetcher=fetcher,
            es=es,
            model=model,
            pool=pool,
            resume=not args.no_resume,
            local_resume=not args.no_resume,
            role_weights=role_weights,
        )
        all_stats.append(year_stats)

    if pool is not None:
        model.stop_multi_process_pool(pool)
        log.info("Multi-GPU pool stopped")

    # --- Final summary ---
    total_docs = sum(s["uploaded_ok"] for s in all_stats)
    total_errors = sum(s["upload_errors"] for s in all_stats)
    total_elapsed = time.time() - total_start

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("Years processed : %s", sorted(target_years))
    log.info("Total uploaded  : %s", f"{total_docs:,}")
    log.info("Total errors    : %s", f"{total_errors:,}")
    log.info("Wall time       : %.0f s  (%.1f min)", total_elapsed, total_elapsed / 60)

    for s in all_stats:
        log.info(
            "  year=%s  scanned=%s  uploaded=%s  errors=%s  %.1fs",
            s["year"], f"{s['scanned']:,}", f"{s['uploaded_ok']:,}",
            f"{s['upload_errors']:,}", s["elapsed_s"],
        )

    log_index_stats(es)


if __name__ == "__main__":
    main()

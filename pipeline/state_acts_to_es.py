"""
Pipeline: ADLS state_acts/processed → Elasticsearch index 'state_acts'.

Embeddings are pre-computed by process_state_acts.py and stored in ADLS,
so no GPU is needed — this is a pure I/O → ES upload pipeline.

For each state:
  1. Scan ADLS state_acts/processed/state={state}/ for *.json
  2. Background reader thread pre-fetches batches into a bounded queue
  3. Main thread: ES mget pre-filter → bulk upload (embeddings already in file)
  4. Write per-state JSONL progress file for resumability

Usage:
  # Upload 21 states already processed (run now):
  python pipeline/state_acts_to_es.py --states karnataka delhi maharashtra ...

  # Upload all states that exist in ADLS (auto-discover):
  python pipeline/state_acts_to_es.py --all

  # Upload specific states, recreating the index:
  python pipeline/state_acts_to_es.py --all --recreate-index

  # Disable resume (re-upload everything):
  python pipeline/state_acts_to_es.py --all --no-resume

Environment:
  ES_URL / ES_API_KEY / ES_USER / ES_PASS
  ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
  IO_WORKERS              (default: 32)
  ES_BULK_BATCH_SIZE      (default: 100)
  PIPELINE_BUFFER_SIZE    (default: 200)
  ES_MAX_RETRIES          (default: 3)
  ES_RETRY_DELAY          (default: 2.0)
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

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk

from config import ADLS_CONFIG
from core.adls_fetcher import ADLSFetcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("state_acts_to_es.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("state_acts_to_es")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_NAME    = "state_acts"
SCHEMA_PATH   = Path(__file__).resolve().parent.parent / "sample" / "es_schema_state_acts.json"
ADLS_BASE     = "state_acts/processed"
PROGRESS_DIR  = Path("pipeline_progress")

IO_WORKERS      = int(os.getenv("IO_WORKERS",           "32"))
BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE",   "100"))
BUFFER_SIZE     = int(os.getenv("PIPELINE_BUFFER_SIZE", "200"))
ES_MAX_RETRIES  = int(os.getenv("ES_MAX_RETRIES",       "3"))
ES_RETRY_DELAY  = float(os.getenv("ES_RETRY_DELAY",     "2.0"))

# Fields to keep from the processed doc (excludes chunks, which are handled separately)
METADATA_FIELDS = [
    "doc_id", "act_name", "act_number", "year", "state", "state_display",
    "jurisdiction", "source", "source_url", "handle_id", "pdf_url",
    "pdf_exists", "pdf_file_size_bytes", "language_original",
    "language_confidence", "ocr_applied", "ocr_page_count", "total_page_count",
    "_is_translated", "total_chars", "total_chunks", "scrape_timestamp",
    "processed_at",
]


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
        schema["settings"].setdefault("number_of_shards", 4)
        schema["settings"].setdefault("number_of_replicas", 0)
        es.indices.create(index=INDEX_NAME, body=schema)
        log.info("Created index: %s  (shards=%s, replicas=%s)",
                 INDEX_NAME,
                 schema["settings"]["number_of_shards"],
                 schema["settings"]["number_of_replicas"])
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
    total_failed = 0

    for attempt in range(ES_MAX_RETRIES):
        def _actions(docs):
            for doc in docs:
                yield {"_index": INDEX_NAME, "_id": doc.get("doc_id"), "_source": doc}

        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=BULK_BATCH_SIZE)
        total_ok += ok

        if not errors:
            remaining = []
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

    total_failed = len(remaining)
    if total_failed:
        log.error("  %d docs permanently failed after %d retries", total_failed, ES_MAX_RETRIES)

    return total_ok, total_failed


# ---------------------------------------------------------------------------
# Progress tracking — JSONL append per state
# ---------------------------------------------------------------------------

def progress_path(state: str) -> Path:
    PROGRESS_DIR.mkdir(exist_ok=True)
    return PROGRESS_DIR / f"done_ids_state_acts_{state}.jsonl"


def load_done_ids(state: str) -> set:
    ids: set = set()
    p = progress_path(state)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.add(line.strip('"'))
    if ids:
        log.info("  Resuming state %s: %d done IDs loaded", state, len(ids))
    return ids


def append_done_ids(state: str, new_ids: set) -> None:
    if not new_ids:
        return
    with progress_path(state).open("a", encoding="utf-8") as f:
        for id_ in new_ids:
            f.write(json.dumps(id_) + "\n")


# ---------------------------------------------------------------------------
# ADLS reading
# ---------------------------------------------------------------------------

def iter_state_paths(fetcher: ADLSFetcher, state: str) -> Generator[str, None, None]:
    base = f"{ADLS_BASE}/state={state}"
    return fetcher.list_files_iter(path=base, pattern="*.json", recursive=True)


def read_doc_with_retry(fetcher: ADLSFetcher, path: str) -> Optional[Dict]:
    for attempt in range(ES_MAX_RETRIES):
        try:
            data = fetcher.read_json_file(path)
            if isinstance(data, dict) and data.get("doc_id"):
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
) -> List[Tuple[str, Dict]]:
    results: List[Tuple[str, Dict]] = []
    lock = threading.Lock()
    sem = threading.Semaphore(workers)

    def _worker(path: str) -> None:
        try:
            doc = read_doc_with_retry(fetcher, path)
            if doc:
                with lock:
                    results.append((path, doc))
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
# Background reader — overlaps ADLS I/O with ES upload
# ---------------------------------------------------------------------------

def _background_reader(
    fetcher: ADLSFetcher,
    state: str,
    done_ids: set,
    local_resume: bool,
    out_queue: queue.Queue,
    stats: Dict,
) -> None:
    path_buf: List[str] = []
    try:
        for path in iter_state_paths(fetcher, state):
            stats["scanned"] += 1
            if local_resume and done_ids:
                doc_id = Path(path).stem  # SA_state_handle — stem is doc_id
                if doc_id in done_ids:
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
        log.error("Background reader crashed for state %s: %s", state, e)
    finally:
        out_queue.put(("done", None, None))


# ---------------------------------------------------------------------------
# Per-state pipeline
# ---------------------------------------------------------------------------

def process_state(
    state: str,
    fetcher: ADLSFetcher,
    es: Elasticsearch,
    resume: bool,
    local_resume: bool,
) -> Dict:
    log.info("=" * 60)
    log.info("State: %s — starting", state)
    t_start = time.time()

    done_ids: set = load_done_ids(state) if local_resume else set()
    stats = {
        "state": state,
        "scanned": 0,
        "skipped_local": 0,
        "skipped_es": 0,
        "read_ok": 0,
        "read_fail": 0,
        "uploaded_ok": 0,
        "upload_errors": 0,
    }

    read_queue: queue.Queue = queue.Queue(maxsize=2)
    reader = threading.Thread(
        target=_background_reader,
        args=(fetcher, state, done_ids, local_resume, read_queue, stats),
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

        # ES mget pre-filter: single HTTP call for the whole buffer
        if resume:
            path_ids = {p: doc.get("doc_id", Path(p).stem) for p, doc in raw}
            found_in_es = bulk_mget_exists(es, list(path_ids.values()))
            if found_in_es:
                new_es_ids = found_in_es - done_ids
                done_ids.update(new_es_ids)
                append_done_ids(state, new_es_ids)
                stats["skipped_es"] += len(found_in_es)
                raw = [(p, doc) for p, doc in raw if path_ids[p] not in found_in_es]
            if not raw:
                continue

        # Build parent docs — keep metadata fields + chunks (embeddings already present)
        parent_docs: List[Dict] = []
        for path, doc in raw:
            try:
                parent = {k: doc[k] for k in METADATA_FIELDS if k in doc}
                # Coerce string booleans to real booleans
                for bool_field in ("pdf_exists", "ocr_applied", "_is_translated"):
                    if bool_field in parent and isinstance(parent[bool_field], str):
                        parent[bool_field] = parent[bool_field].strip().lower() == "true"
                # Keep chunks as-is (embeddings already in ADLS)
                parent["chunks"] = doc.get("chunks", [])
                if not parent.get("doc_id"):
                    parent["doc_id"] = Path(path).stem
                parent_docs.append(parent)
            except Exception as e:
                log.warning("  Skipping %s: %s", path, e)

        if not parent_docs:
            continue

        ok, errs = bulk_upload_with_retry(es, parent_docs)
        stats["uploaded_ok"] += ok
        stats["upload_errors"] += errs

        new_ids: set = set()
        for doc in parent_docs:
            doc_id = doc.get("doc_id", "")
            if doc_id and doc_id not in done_ids:
                new_ids.add(doc_id)
                done_ids.add(doc_id)
        append_done_ids(state, new_ids)

        log.info(
            "  Batch — scanned=%d  read_ok=%d  uploaded=%d  errors=%d",
            stats["scanned"], stats["read_ok"], stats["uploaded_ok"], stats["upload_errors"],
        )

    reader.join(timeout=60)

    elapsed = time.time() - t_start
    stats["elapsed_s"] = round(elapsed, 1)
    log.info(
        "State %s done in %.0fs | scanned=%d  skip_local=%d  skip_es=%d  "
        "read_ok=%d  uploaded=%d  errors=%d",
        state, elapsed,
        stats["scanned"], stats["skipped_local"], stats["skipped_es"],
        stats["read_ok"], stats["uploaded_ok"], stats["upload_errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# ADLS discovery
# ---------------------------------------------------------------------------

def discover_states(fetcher: ADLSFetcher) -> List[str]:
    states = []
    for path in fetcher.list_subdirs_iter(ADLS_BASE):
        # path like "state_acts/processed/state=karnataka"
        part = path.rstrip("/").split("/")[-1]
        if part.startswith("state="):
            states.append(part[len("state="):])
    return sorted(states)


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
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload state acts from ADLS to the 'state_acts' Elasticsearch index."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--states", nargs="+", metavar="STATE",
        help="Specific states to upload (e.g. --states karnataka delhi maharashtra)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Auto-discover all states present in ADLS and upload all of them",
    )
    p.add_argument(
        "--recreate-index", action="store_true", dest="recreate_index",
        help="Drop and recreate the ES index before starting (destructive)",
    )
    p.add_argument(
        "--no-resume", action="store_true", dest="no_resume",
        help="Disable resume — re-upload every doc regardless of prior runs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    account_name = ADLS_CONFIG["account_name"]
    account_key = ADLS_CONFIG["account_key"]
    container = ADLS_CONFIG["container_name"]
    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing — set ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME in .env")
        sys.exit(1)

    es = build_es_client()
    ensure_index(es, recreate=args.recreate_index)
    fetcher = ADLSFetcher(account_name, account_key, container)

    if args.all:
        states = discover_states(fetcher)
        log.info("Discovered %d states in ADLS: %s", len(states), states)
    else:
        states = args.states

    resume = not args.no_resume
    all_stats = []
    total_start = time.time()

    for i, state in enumerate(states, 1):
        log.info("[%d/%d] Processing state: %s", i, len(states), state)
        s = process_state(
            state=state,
            fetcher=fetcher,
            es=es,
            resume=resume,
            local_resume=resume,
        )
        all_stats.append(s)

    total_docs = sum(s["uploaded_ok"] for s in all_stats)
    total_errors = sum(s["upload_errors"] for s in all_stats)
    total_elapsed = time.time() - total_start

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("States processed : %s", states)
    log.info("Total uploaded   : %s", f"{total_docs:,}")
    log.info("Total errors     : %s", f"{total_errors:,}")
    log.info("Wall time        : %.0f s  (%.1f min)", total_elapsed, total_elapsed / 60)

    for s in all_stats:
        log.info(
            "  state=%-35s  scanned=%s  uploaded=%s  errors=%s  %.1fs",
            s["state"], f"{s['scanned']:,}", f"{s['uploaded_ok']:,}",
            f"{s['upload_errors']:,}", s["elapsed_s"],
        )

    log_index_stats(es)


if __name__ == "__main__":
    main()

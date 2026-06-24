"""
Upload Central Acts from ADLS to Elasticsearch.

Embeddings are pre-computed by process_central_acts.py and already stored
in ADLS — this script reads them and uploads directly (no GPU needed).

Reads from: processed/central_acts/{year}/{doc_id}.json
Writes to:  ES index "central_acts"

Run:
  .venv/bin/python pipeline/central_acts_to_es.py [--recreate-index] [--no-resume]

Environment (same .env as adls_to_es_pipeline.py):
  ES_URL, ES_API_KEY, ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY, ADLS_CONTAINER_NAME
  IO_WORKERS, ES_BULK_BATCH_SIZE, PIPELINE_BUFFER_SIZE, ES_MAX_RETRIES, ES_RETRY_DELAY
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
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk

from core.adls_fetcher import ADLSFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("central_acts_to_es.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("central_acts_to_es")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_NAME      = "central_acts"
SCHEMA_PATH     = Path(__file__).resolve().parent.parent / "sample" / "es_schema_central_acts.json"
ADLS_BASE_PATH  = "processed/central_acts"

IO_WORKERS      = int(os.getenv("IO_WORKERS",           "32"))
BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE",   "100"))
BUFFER_SIZE     = int(os.getenv("PIPELINE_BUFFER_SIZE", "500"))
ES_MAX_RETRIES  = int(os.getenv("ES_MAX_RETRIES",       "3"))
ES_RETRY_DELAY  = float(os.getenv("ES_RETRY_DELAY",     "2.0"))
PROGRESS_DIR    = Path("pipeline_progress")


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
        es = Elasticsearch(es_url, http_auth=(user, password), request_timeout=120) \
            if (user and password) else Elasticsearch(es_url, request_timeout=120)
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
        es.indices.create(index=INDEX_NAME, body=schema)
        log.info("Created index: %s", INDEX_NAME)
    else:
        log.info("Index already exists: %s", INDEX_NAME)


def bulk_mget_exists(es: Elasticsearch, doc_ids: List[str]) -> set:
    if not doc_ids:
        return set()
    resp = es.mget(index=INDEX_NAME, body={"ids": doc_ids}, _source=False)
    return {d["_id"] for d in resp["docs"] if d.get("found")}


def bulk_upload_with_retry(es: Elasticsearch, docs: List[Dict]) -> Tuple[int, int]:
    remaining = list(docs)
    total_ok = 0
    last_errors: list = []

    for attempt in range(ES_MAX_RETRIES):
        def _actions(d):
            for doc in d:
                yield {"_index": INDEX_NAME, "_id": doc.get("doc_id"), "_source": doc}

        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=BULK_BATCH_SIZE)
        total_ok += ok
        last_errors = errors

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

    if last_errors and remaining:
        log.error("  %d docs permanently failed after %d retries", len(remaining), ES_MAX_RETRIES)

    return total_ok, len(remaining)


# ---------------------------------------------------------------------------
# ADLS helpers
# ---------------------------------------------------------------------------

def iter_adls_paths(fetcher: ADLSFetcher):
    return fetcher.list_files_iter(path=ADLS_BASE_PATH, pattern="*.json", recursive=True)


def _doc_id_from_path(path: str) -> str:
    return Path(path).stem  # e.g. "CA_1362_2345"


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


def read_batch_parallel(fetcher: ADLSFetcher, paths: List[str]) -> List[Tuple[str, Dict]]:
    results: List[Tuple[str, Dict]] = []
    lock = threading.Lock()
    sem = threading.Semaphore(IO_WORKERS)

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
# Progress (JSONL append)
# ---------------------------------------------------------------------------

def _progress_path() -> Path:
    PROGRESS_DIR.mkdir(exist_ok=True)
    return PROGRESS_DIR / "done_ids_central_acts_es.jsonl"


def load_done_ids() -> set:
    ids: set = set()
    p = _progress_path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line))
                except json.JSONDecodeError:
                    ids.add(line.strip('"'))
    if ids:
        log.info("Resuming: %d done IDs loaded", len(ids))
    return ids


def append_done_ids(new_ids: set) -> None:
    if not new_ids:
        return
    with _progress_path().open("a", encoding="utf-8") as f:
        for id_ in new_ids:
            f.write(json.dumps(id_) + "\n")


# ---------------------------------------------------------------------------
# Background reader
# ---------------------------------------------------------------------------

def _background_reader(
    fetcher: ADLSFetcher,
    done_ids: set,
    local_resume: bool,
    out_queue: queue.Queue,
    stats: Dict,
) -> None:
    path_buf: List[str] = []
    try:
        for path in iter_adls_paths(fetcher):
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
        log.error("Background reader crashed: %s", e)
    finally:
        out_queue.put(("done", None, None))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(resume: bool, local_resume: bool, recreate_index: bool) -> None:
    es = build_es_client()
    ensure_index(es, recreate=recreate_index)

    fetcher = ADLSFetcher(
        account_name=os.environ["ADLS_ACCOUNT_NAME"],
        account_key=os.environ["ADLS_ACCOUNT_KEY"],
        container_name=os.environ["ADLS_CONTAINER_NAME"],
    )

    done_ids: set = load_done_ids() if local_resume else set()

    stats = {
        "scanned": 0, "skipped_local": 0, "skipped_es": 0,
        "read_ok": 0, "read_fail": 0,
        "uploaded_ok": 0, "upload_errors": 0,
    }
    t_start = time.time()

    read_queue: queue.Queue = queue.Queue(maxsize=2)
    reader = threading.Thread(
        target=_background_reader,
        args=(fetcher, done_ids, local_resume, read_queue, stats),
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

        # ES mget pre-filter (skip docs already in ES)
        if resume:
            path_ids = {p: _doc_id_from_path(p) for p, _ in raw}
            found_in_es = bulk_mget_exists(es, list(path_ids.values()))
            if found_in_es:
                new_es_ids = found_in_es - done_ids
                done_ids.update(new_es_ids)
                append_done_ids(new_es_ids)
                stats["skipped_es"] += len(found_in_es)
                raw = [(p, d) for p, d in raw if path_ids[p] not in found_in_es]
            if not raw:
                continue

        # Docs from ADLS already contain pre-computed embeddings in chunks[].embedding
        # Coerce types and build parent doc list
        parent_docs: List[Dict] = []
        newly_done: set = set()

        for _, doc in raw:
            try:
                if isinstance(doc.get("pdf_exists"), str):
                    doc["pdf_exists"] = doc["pdf_exists"].strip().lower() == "true"
                if isinstance(doc.get("year"), str):
                    try:
                        doc["year"] = int(doc["year"])
                    except (ValueError, TypeError):
                        pass
                parent_docs.append(doc)
                newly_done.add(doc["doc_id"])
            except Exception as e:
                log.warning("Skipping doc during preparation: %s", e)

        if not parent_docs:
            continue

        ok, errs = bulk_upload_with_retry(es, parent_docs)
        stats["uploaded_ok"] += ok
        stats["upload_errors"] += errs
        done_ids.update(newly_done)
        append_done_ids(newly_done)

        elapsed = time.time() - t_start
        log.info(
            "Progress: scanned=%d  skipped_local=%d  skipped_es=%d  "
            "read_ok=%d  uploaded=%d  errors=%d  elapsed=%.0fs",
            stats["scanned"], stats["skipped_local"], stats["skipped_es"],
            stats["read_ok"], stats["uploaded_ok"], stats["upload_errors"],
            elapsed,
        )

    reader.join()
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("DONE — %s", INDEX_NAME)
    log.info("  scanned=%d  skipped_local=%d  skipped_es=%d",
             stats["scanned"], stats["skipped_local"], stats["skipped_es"])
    log.info("  read_ok=%d  read_fail=%d",
             stats["read_ok"], stats["read_fail"])
    log.info("  uploaded=%d  errors=%d  elapsed=%.0fs",
             stats["uploaded_ok"], stats["upload_errors"], elapsed)


def main():
    ap = argparse.ArgumentParser(description="Upload Central Acts from ADLS to Elasticsearch")
    ap.add_argument("--recreate-index", action="store_true",
                    help="Drop and recreate the ES index before uploading")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore local progress and re-upload everything")
    args = ap.parse_args()

    local_resume = not args.no_resume
    run(resume=local_resume, local_resume=local_resume, recreate_index=args.recreate_index)


if __name__ == "__main__":
    main()

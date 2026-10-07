"""
Central Acts Pipeline — scrape → process (PDF→chunks→embed→ADLS) → ES upload.

Single command, end-to-end. Can also run individual stages.

Usage:
    # Full pipeline (scrape if metadata missing, then process, then ES):
    python pipelines/central_acts/run.py

    # Specific stage:
    python pipelines/central_acts/run.py --stage scrape
    python pipelines/central_acts/run.py --stage process
    python pipelines/central_acts/run.py --stage es

    # Force reprocess (ignore progress):
    python pipelines/central_acts/run.py --stage process --no-resume

    # Recreate ES index:
    python pipelines/central_acts/run.py --stage es --recreate-index

Environment:
    ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
    ES_URL / ES_API_KEY (or ES_USER/ES_PASS)
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, ...
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

from config import LOGGING_CONFIG, ADLS_CONFIG, EMBEDDING_CONFIG, validate_config
from core.adls_fetcher import ADLSFetcher
from core.elasticsearch_client import (
    build_es_client, ensure_index, bulk_mget_exists,
    bulk_upload_with_retry, log_index_stats,
)
from core.adls_paths import (
    COLLECTION_CONFIG, get_processed_root, get_es_index, get_schema_path,
    get_processed_path, get_es_inventory_path,
)
from core.inventory import load_inventory, upload_inventory_es
from core.progress_tracker import ProgressTracker

COLLECTION = "central_acts"
log = logging.getLogger(__name__)

ADLS_BASE_PATH = get_processed_root(COLLECTION)
INDEX_NAME = get_es_index(COLLECTION)
SCHEMA_PATH = get_schema_path(COLLECTION)

METADATA_FIELDS = [
    "doc_id", "act_name", "act_number", "year", "jurisdiction",
    "pdf_url", "pdf_exists", "handle_id", "source",
]

BOOL_FIELDS = ["pdf_exists"]


# ---------------------------------------------------------------------------
# ES upload (reads from ADLS processed/, uploads to ES)
# ---------------------------------------------------------------------------

def run_es_stage(
    fetcher: ADLSFetcher,
    recreate_index: bool = False,
    no_resume: bool = False,
) -> Dict:
    log.info("=" * 60)
    log.info("CENTRAL ACTS — ES upload stage")
    log.info("=" * 60)

    es = build_es_client()
    ensure_index(es, INDEX_NAME, SCHEMA_PATH, recreate=recreate_index)

    IO_WORKERS = int(os.getenv("IO_WORKERS", "32"))
    BUFFER_SIZE = int(os.getenv("PIPELINE_BUFFER_SIZE", "500"))

    years = set()
    for fp in fetcher.list_files_iter(path=ADLS_BASE_PATH, pattern="*.json", recursive=True):
        if not fp.endswith("_inventory.json") and not fp.endswith("_inventory_es.json"):
            p = Path(fp)
            # Extract year from path: processed/Central_Acts/{year}/{doc_id}.json
            parts = p.parts
            if len(parts) >= 3:
                try:
                    year_val = int(parts[-2])
                    years.add(year_val)
                except (ValueError, IndexError):
                    pass

    if not years:
        log.info("No processed docs found in ADLS at %s", ADLS_BASE_PATH)
        return {"uploaded": 0, "errors": 0}

    log.info("Found years in ADLS processed/: %s", sorted(years))
    total_uploaded = 0
    total_errors = 0

    for year in sorted(years):
        year_proc_path = get_processed_path(COLLECTION, year)
        es_inv_path = get_es_inventory_path(COLLECTION, year)
        es_files = load_inventory(fetcher, es_inv_path) if not no_resume else set()

        tracker = ProgressTracker(COLLECTION, f"year={year}", no_resume=no_resume)

        # Build file index
        file_idx: Dict[str, str] = {}
        for fp in fetcher.list_files_iter(path=year_proc_path, pattern="*.json", recursive=True):
            name = Path(fp).name
            if not name.endswith("_inventory.json") and not name.endswith("_inventory_es.json"):
                file_idx[name] = fp

        # Filter to docs not yet in ES
        pending_paths = []
        for basename, full_path in file_idx.items():
            doc_id = Path(basename).stem
            if not no_resume and (doc_id in es_files or tracker.is_done(doc_id)):
                continue
            pending_paths.append(full_path)

        if not pending_paths:
            log.info("Year %d: nothing to upload", year)
            continue

        log.info("Year %d: %d docs to upload to ES", year, len(pending_paths))

        # Process in batches
        path_buf: List[str] = []
        all_new_ids: set = set()

        for p in pending_paths:
            path_buf.append(p)
            if len(path_buf) >= BUFFER_SIZE:
                ok, errs = _flush_es_batch(path_buf, es, fetcher, tracker, no_resume,
                                          all_new_ids, IO_WORKERS)
                total_uploaded += ok
                total_errors += errs
                path_buf = []

        if path_buf:
            ok, errs = _flush_es_batch(path_buf, es, fetcher, tracker, no_resume,
                                      all_new_ids, IO_WORKERS)
            total_uploaded += ok
            total_errors += errs

        # Save ES inventory
        merged = es_files | all_new_ids
        upload_inventory_es(fetcher, COLLECTION, year, merged)

    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    log.info("Total docs in ES index '%s': %d", INDEX_NAME, count)

    return {"uploaded": total_uploaded, "errors": total_errors}


def _flush_es_batch(
    paths: List[str],
    es: "Elasticsearch",
    fetcher: ADLSFetcher,
    tracker: ProgressTracker,
    no_resume: bool,
    all_new_ids: set,
    workers: int,
) -> tuple:
    """Upload a batch of docs to ES. Returns (uploaded_ok, upload_errors)."""
    raw = _read_batch_parallel(fetcher, paths, workers)

    # ES mget pre-filter
    if not no_resume:
        path_ids = {p: Path(p).stem for p, _ in raw}
        found = bulk_mget_exists(es, INDEX_NAME, list(path_ids.values()))
        raw = [(p, d) for p, d in raw if path_ids[p] not in found]
        if not raw:
            return (0, 0)

    parent_docs = []
    for path, doc in raw:
        try:
            parent = {k: doc[k] for k in METADATA_FIELDS if k in doc}
            for bf in BOOL_FIELDS:
                if bf in parent and isinstance(parent[bf], str):
                    parent[bf] = parent[bf].strip().lower() == "true"
            if isinstance(doc.get("year"), str):
                try:
                    parent["year"] = int(doc["year"])
                except (ValueError, TypeError):
                    pass
            parent["chunks"] = doc.get("chunks", [])
            if not parent.get("doc_id"):
                parent["doc_id"] = Path(path).stem
            parent_docs.append(parent)
        except Exception as e:
            log.warning("Skipping %s: %s", path, e)

    if not parent_docs:
        return (0, 0)

    ok, errs = bulk_upload_with_retry(es, INDEX_NAME, parent_docs)

    new_ids = {d.get("doc_id", "") for d in parent_docs if d.get("doc_id")}
    tracker.mark_batch_done(new_ids)
    all_new_ids.update(new_ids)

    log.info("  Batch uploaded=%d errors=%d", ok, errs)
    return (ok, errs)


def _read_batch_parallel(fetcher, paths, workers):
    results: List[Tuple[str, Dict]] = []
    lock = threading.Lock()
    sem = threading.Semaphore(workers)

    def _worker(path):
        try:
            doc = fetcher.read_json_file(path)
            if isinstance(doc, dict) and doc.get("doc_id"):
                with lock:
                    results.append((path, doc))
        except Exception:
            pass
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
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Central Acts: scrape → process (PDF→chunks→ADLS) → ES upload"
    )
    ap.add_argument("--stage", choices=["all", "scrape", "process", "es"], default="all",
                    help="Pipeline stage to run (default: all)")
    ap.add_argument("--pdf-dir", default="central_acts_pdfs",
                    help="Local PDF/metadata directory (default: central_acts_pdfs)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore progress and reprocess everything")
    ap.add_argument("--recreate-index", action="store_true",
                    help="Drop and recreate ES index (ES stage only)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Scrape stage: only metadata, skip PDFs")
    args = ap.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    level = getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt,
                        handlers=[logging.StreamHandler()], force=True)

    validate_config()

    pdf_dir = Path(args.pdf_dir)

    # Stage: scrape
    if args.stage in ("all", "scrape"):
        meta_path = pdf_dir / "metadata.jsonl"
        has_pdfs = pdf_dir.exists() and any(pdf_dir.iterdir()) if pdf_dir.exists() else False
        if not meta_path.exists() or not has_pdfs:
            log.info("Metadata/PDFs not found — running scrape")
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "scrape.py"),
                 "--output-dir", str(pdf_dir)] +
                (["--metadata-only"] if args.metadata_only else []),
                cwd=Path(__file__).resolve().parent.parent.parent,
            )
            if result.returncode != 0:
                log.error("Scrape failed (exit %d)", result.returncode)
                sys.exit(1)
        else:
            log.info("Metadata found at %s — skipping scrape", meta_path)

    # Stage: process
    if args.stage in ("all", "process"):
        meta_path = pdf_dir / "metadata.jsonl"
        if not meta_path.exists():
            log.error("metadata.jsonl not found at %s — run scrape stage first", meta_path)
            sys.exit(1)

        log.info("Running process stage")
        process_args = [sys.executable, str(Path(__file__).parent / "process.py"),
                        "--pdf-dir", str(pdf_dir)]
        if args.no_resume:
            process_args.append("--no-resume")
        result = subprocess.run(process_args, cwd=Path(__file__).resolve().parent.parent.parent)
        if result.returncode != 0:
            log.error("Process failed (exit %d)", result.returncode)
            sys.exit(1)

    # Stage: ES
    if args.stage in ("all", "es"):
        fetcher = ADLSFetcher(
            ADLS_CONFIG["account_name"], ADLS_CONFIG["account_key"],
            ADLS_CONFIG["container_name"],
        )
        run_es_stage(fetcher, recreate_index=args.recreate_index, no_resume=args.no_resume)

    log.info("=== Central Acts pipeline complete ===")


if __name__ == "__main__":
    main()

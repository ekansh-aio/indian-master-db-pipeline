"""
State Acts Pipeline — scrape → process (OCR→translate→chunks→embed→ADLS) → ES upload.

Single command, end-to-end. GPU queue scheduler distributes states across GPUs.

Usage:
    # Full pipeline for specific states:
    python pipelines/state_acts/run.py --states karnataka,delhi,maharashtra,haryana

    # Full pipeline for all 36 states/UTs:
    python pipelines/state_acts/run.py --all-states

    # Specific stage:
    python pipelines/state_acts/run.py --stage scrape --all-states
    python pipelines/state_acts/run.py --stage process --states karnataka,delhi
    python pipelines/state_acts/run.py --stage process --all-states --gpu-workers 8
    python pipelines/state_acts/run.py --stage es --all-states

    # GPU queue scheduler with custom worker count:
    python pipelines/state_acts/run.py --stage process --all-states --gpu-workers 4

    # Force reprocess:
    python pipelines/state_acts/run.py --stage process --states karnataka --no-resume

    # Recreate ES index:
    python pipelines/state_acts/run.py --stage es --all-states --recreate-index

Environment:
    ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
    ES_URL / ES_API_KEY (or ES_USER/ES_PASS)
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, PROCESS_WORKERS, ...
"""

import argparse
import json
import logging
import os
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

from config import LOGGING_CONFIG, ADLS_CONFIG, validate_config
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

COLLECTION = "state_acts"
log = logging.getLogger(__name__)

ADLS_BASE_PATH = get_processed_root(COLLECTION)
INDEX_NAME = get_es_index(COLLECTION)
SCHEMA_PATH = get_schema_path(COLLECTION)

# All 36 states/UTs sorted alphabetically
ALL_STATES = [
    "andaman_nicobar", "andhra_pradesh", "arunachal_pradesh", "assam",
    "bihar", "chandigarh", "chhattisgarh",
    "dadra_nagar_haveli_daman_diu", "delhi",
    "goa", "gujarat",
    "haryana", "himachal_pradesh",
    "jammu_kashmir", "jharkhand",
    "karnataka", "kerala",
    "ladakh", "lakshadweep",
    "madhya_pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland",
    "odisha",
    "puducherry", "punjab",
    "rajasthan",
    "sikkim",
    "tamil_nadu", "telangana", "tripura",
    "uttarakhand", "uttar_pradesh",
    "west_bengal",
]

METADATA_FIELDS = [
    "doc_id", "act_name", "act_number", "year", "state", "state_display",
    "jurisdiction", "source", "source_url", "handle_id", "pdf_url",
    "pdf_exists", "pdf_file_size_bytes", "language_original",
    "language_confidence", "ocr_applied", "ocr_page_count", "total_page_count",
    "_is_translated", "total_chars", "total_chunks", "scrape_timestamp",
    "processed_at",
]

BOOL_FIELDS = ["pdf_exists", "ocr_applied", "_is_translated"]


# ---------------------------------------------------------------------------
# ES upload (reads from ADLS processed/, uploads to ES)
# ---------------------------------------------------------------------------

def run_es_stage(
    fetcher: ADLSFetcher,
    states: List[str],
    recreate_index: bool = False,
    no_resume: bool = False,
) -> Dict:
    log.info("=" * 60)
    log.info("STATE ACTS — ES upload stage (%d states)", len(states))

    es = build_es_client()
    ensure_index(es, INDEX_NAME, SCHEMA_PATH, recreate=recreate_index)

    IO_WORKERS = int(os.getenv("IO_WORKERS", "32"))
    BUFFER_SIZE = int(os.getenv("PIPELINE_BUFFER_SIZE", "200"))

    total_uploaded = 0
    total_errors = 0
    all_stats = []

    for state in states:
        log.info("-" * 50)
        log.info("State: %s — ES upload", state)
        t_start = time.time()

        state_proc_path = f"{ADLS_BASE_PATH}/state={state}"
        es_inv_path = get_es_inventory_path(COLLECTION, None, state)
        es_files = load_inventory(fetcher, es_inv_path) if not no_resume else set()

        tracker = ProgressTracker(COLLECTION, f"state={state}", no_resume=no_resume)

        # Build file index for this state
        file_idx: Dict[str, str] = {}
        try:
            for fp in fetcher.list_files_iter(path=state_proc_path, pattern="*.json", recursive=True):
                name = Path(fp).name
                if not name.endswith("_inventory.json") and not name.endswith("_inventory_es.json"):
                    file_idx[name] = fp
        except Exception:
            log.info("  No processed docs in ADLS for state %s", state)
            continue

        pending_paths = []
        for basename, full_path in file_idx.items():
            doc_id = Path(basename).stem
            if not no_resume and (doc_id in es_files or tracker.is_done(doc_id)):
                continue
            pending_paths.append(full_path)

        if not pending_paths:
            log.info("  State %s: nothing to upload to ES", state)
            continue

        log.info("  State %s: %d docs to upload to ES", state, len(pending_paths))

        path_buf: List[str] = []
        all_new_ids: set = set()
        state_uploaded = 0
        state_errors = 0

        for p in pending_paths:
            path_buf.append(p)
            if len(path_buf) >= BUFFER_SIZE:
                ok, errs = _flush_es_batch(path_buf, es, fetcher, tracker, no_resume,
                                          all_new_ids, IO_WORKERS)
                total_uploaded += ok
                total_errors += errs
                state_uploaded += ok
                state_errors += errs
                path_buf = []

        if path_buf:
            ok, errs = _flush_es_batch(path_buf, es, fetcher, tracker, no_resume,
                                      all_new_ids, IO_WORKERS)
            total_uploaded += ok
            total_errors += errs
            state_uploaded += ok
            state_errors += errs

        # Save ES inventory for this state
        if all_new_ids:
            merged = es_files | all_new_ids
            upload_inventory_es(fetcher, COLLECTION, None, merged, state=state)

        elapsed = time.time() - t_start
        log.info("  State %s ES done — uploaded=%d errors=%d %.0fs",
                 state, state_uploaded, state_errors, elapsed)

    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    log.info("Total docs in ES index '%s': %d", INDEX_NAME, count)

    return {"uploaded": total_uploaded, "errors": total_errors}


def _flush_es_batch(paths, es, fetcher, tracker, no_resume,
                    all_new_ids, workers):
    """Upload a batch of docs to ES. Returns (uploaded_ok, upload_errors)."""
    raw = _read_batch_parallel(fetcher, paths, workers)

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
            parent["chunks"] = doc.get("chunks", [])
            if not parent.get("doc_id"):
                parent["doc_id"] = Path(path).stem
            parent_docs.append(parent)
        except Exception as e:
            log.warning("  Skipping %s: %s", path, e)

    if not parent_docs:
        return (0, 0)

    ok, errs = bulk_upload_with_retry(es, INDEX_NAME, parent_docs)

    new_ids = {d.get("doc_id", "") for d in parent_docs if d.get("doc_id")}
    tracker.mark_batch_done(new_ids)
    all_new_ids.update(new_ids)

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
# GPU queue scheduler
# ---------------------------------------------------------------------------

def run_process_stage(
    states: List[str],
    pdf_dir: str,
    gpu_workers: int = 8,
    no_resume: bool = False,
    no_upload: bool = False,
) -> None:
    """Run process_state_acts.py for each state, distributing across GPUs.

    States are assigned round-robin to GPU workers. Each GPU processes
    one state at a time; when it finishes, the next queued state starts.
    """
    import subprocess

    log.info("=" * 60)
    log.info("STATE ACTS — Process stage (%d states, %d GPU workers)",
             len(states), gpu_workers)
    log.info("=" * 60)

    if gpu_workers < 1:
        log.error("gpu-workers must be >= 1")
        sys.exit(1)

    time_start = time.time()

    # Build round-robin queue assignments
    gpu_queues: Dict[int, List[str]] = {g: [] for g in range(gpu_workers)}
    for i, state in enumerate(states):
        gpu_queues[i % gpu_workers].append(state)

    log.info("GPU queue assignments:")
    for gpu, q in gpu_queues.items():
        log.info("  GPU %d: %s", gpu, ", ".join(q) if q else "(none)")

    process_script = str(Path(__file__).parent / "process.py")
    cwd = Path(__file__).resolve().parent.parent.parent

    results: Dict[int, List[Dict]] = {g: [] for g in range(gpu_workers)}
    errors: List[str] = []
    lock = threading.Lock()

    def _worker_fn(gpu: int):
        queue = gpu_queues[gpu]
        for state in queue:
            log.info("[GPU %d] Starting state: %s", gpu, state)
            t0 = time.time()
            cmd = [
                sys.executable, process_script,
                "--states", state,
                "--pdf-dir", pdf_dir,
                "--workers", "4",
                "--stream",
                "--stream-idle-secs", "900",
            ]
            if no_resume:
                cmd.append("--no-resume")
            if no_upload:
                cmd.append("--no-upload")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            try:
                result = subprocess.run(
                    cmd, cwd=cwd, env=env,
                    capture_output=True, text=True, timeout=86400,
                )
                elapsed = time.time() - t0
                if result.returncode == 0:
                    log.info("[GPU %d] State %s done in %.0fs", gpu, state, elapsed)
                    with lock:
                        results[gpu].append({
                            "state": state, "status": "ok",
                            "elapsed_s": round(elapsed, 1),
                        })
                else:
                    log.error("[GPU %d] State %s FAILED (exit %d) in %.0fs",
                              gpu, state, result.returncode, elapsed)
                    log.error("  stdout: %s", result.stdout[-500:] if result.stdout else "")
                    log.error("  stderr: %s", result.stderr[-500:] if result.stderr else "")
                    with lock:
                        results[gpu].append({
                            "state": state, "status": "failed",
                            "exit_code": result.returncode,
                            "elapsed_s": round(elapsed, 1),
                        })
                        errors.append(f"GPU {gpu}, state {state}: exit {result.returncode}")
            except subprocess.TimeoutExpired:
                log.error("[GPU %d] State %s TIMEOUT after 24h", gpu, state)
                with lock:
                    results[gpu].append({
                        "state": state, "status": "timeout",
                    })
                    errors.append(f"GPU {gpu}, state {state}: timeout")
            except Exception as e:
                log.error("[GPU %d] State %s error: %s", gpu, state, e)
                with lock:
                    results[gpu].append({
                        "state": state, "status": "error",
                        "error": str(e),
                    })
                    errors.append(f"GPU {gpu}, state {state}: {e}")

    threads = []
    for gpu in range(gpu_workers):
        if gpu_queues[gpu]:
            t = threading.Thread(target=_worker_fn, args=(gpu,), daemon=True)
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    log.info("=" * 60)
    log.info("STATE ACTS — Process stage complete")
    ok_count = sum(1 for rlist in results.values() for r in rlist if r["status"] == "ok")
    fail_count = sum(1 for rlist in results.values() for r in rlist if r["status"] != "ok")
    total_elapsed = time.time() - time_start
    log.info("Results: ok=%d failed=%d", ok_count, fail_count)
    for gpu in range(gpu_workers):
        for r in results[gpu]:
            log.info("  GPU %d | %s | %.0fs | %s",
                     gpu, r["state"], r.get("elapsed_s", 0), r["status"])
    if errors:
        log.warning("Errors:")
        for e in errors:
            log.warning("  %s", e)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def resolve_states(states_arg: str, all_states: bool) -> List[str]:
    if all_states:
        return list(ALL_STATES)
    if states_arg:
        return [s.strip() for s in states_arg.split(",") if s.strip()]
    return []


def main():
    ap = argparse.ArgumentParser(
        description="State Acts: scrape → process (OCR→translate→chunks→ADLS) → ES upload"
    )
    ap.add_argument("--stage", choices=["all", "scrape", "process", "es"], default="all",
                    help="Pipeline stage to run (default: all)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--states", help="Comma-separated state keys, e.g. karnataka,delhi")
    group.add_argument("--all-states", action="store_true", help="Process all 36 states/UTs")
    ap.add_argument("--pdf-dir", default="state_acts_pdfs",
                    help="Local PDF/metadata directory (default: state_acts_pdfs)")
    ap.add_argument("--gpu-workers", type=int, default=8,
                    help="Number of GPU workers for parallel processing (default: 8)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore progress and reprocess everything")
    ap.add_argument("--no-upload", action="store_true",
                    help="Process stage: skip ADLS upload (OCR+translate+chunks only)")
    ap.add_argument("--recreate-index", action="store_true",
                    help="ES stage: drop and recreate index")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Scrape stage: only metadata, skip PDFs")
    args = ap.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    level = getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt,
                        handlers=[logging.StreamHandler()], force=True)

    validate_config()

    states = resolve_states(args.states, args.all_states)
    if not states:
        log.error("Specify --states or --all-states")
        sys.exit(1)

    pdf_dir = Path(args.pdf_dir)

    # Stage: scrape
    if args.stage in ("all", "scrape"):
        # Check which states need scraping
        states_to_scrape = []
        for state in states:
            meta_path = pdf_dir / state / "metadata.jsonl"
            if not meta_path.exists():
                states_to_scrape.append(state)

        if states_to_scrape:
            log.info("Scraping %d states without metadata: %s",
                     len(states_to_scrape), states_to_scrape)
            chunk_size = 7
            for i in range(0, len(states_to_scrape), chunk_size):
                batch = states_to_scrape[i:i + chunk_size]
                batch_str = ",".join(batch)
                log.info("Scraping batch %d/%d: %s",
                         i // chunk_size + 1,
                         (len(states_to_scrape) + chunk_size - 1) // chunk_size,
                         batch_str)
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "scrape.py"),
                     "--states", batch_str,
                     "--pdf-dir", str(pdf_dir)] +
                    (["--metadata-only"] if args.metadata_only else []),
                    cwd=Path(__file__).resolve().parent.parent.parent,
                )
                if result.returncode != 0:
                    log.warning("Scrape batch failed (exit %d) — continuing", result.returncode)
        else:
            log.info("All states have metadata — skipping scrape")

    # Stage: process
    if args.stage in ("all", "process"):
        run_process_stage(
            states=states,
            pdf_dir=str(pdf_dir),
            gpu_workers=args.gpu_workers,
            no_resume=args.no_resume,
            no_upload=args.no_upload,
        )

    # Stage: ES
    if args.stage in ("all", "es"):
        fetcher = ADLSFetcher(
            ADLS_CONFIG["account_name"], ADLS_CONFIG["account_key"],
            ADLS_CONFIG["container_name"],
        )
        run_es_stage(
            fetcher=fetcher,
            states=states,
            recreate_index=args.recreate_index,
            no_resume=args.no_resume,
        )

    log.info("=== State Acts pipeline complete ===")


if __name__ == "__main__":
    main()

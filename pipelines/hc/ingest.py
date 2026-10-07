#!/usr/bin/env python3
"""
HC India — Stage 1: S3 → ADLS ingest.

Downloads High Court judgment PDFs from the public S3 bucket, extracts text
(pypdf primary, optional GPU OCR fallback for scanned documents), cleans the
JSON, and uploads to ADLS under app/High_Court_Judgements/.

Run this before pipelines/hc/run.py (which picks up from ADLS and pushes to ES).

Usage:
  python pipelines/hc/ingest.py --year 2024
  python pipelines/hc/ingest.py --year 2024 --use-gpu-ocr --num-gpus 8
  python pipelines/hc/ingest.py --year 2024 --court 10_8 --dry-run
  python pipelines/hc/ingest.py --year 2024 --skip-inventory --skip-fetch
  python pipelines/hc/ingest.py --year 2024 --batch-size 8 --pdf-workers 96

Full HC pipeline:
  python pipelines/hc/ingest.py --year 2024          # S3 → ADLS app/
  python pipelines/hc/run.py --year-range 2024 2024   # ADLS app/ → ES
"""

import os
import sys
import json
import time
import shutil
import signal
import threading
import subprocess
import argparse
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Ensure project root is on the path so `core/` imports work when called from
# any working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import pandas as pd

from pipelines.hc.upload_tracker import (
    get_conn, exists as tracker_exists, insert_one as tracker_insert,
    count as tracker_count, datasets as tracker_datasets, batch_inserter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("hc.ingest")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# ── Config ─────────────────────────────────────────────────────────────────────
ACCOUNT_NAME   = os.getenv("ADLS_ACCOUNT_NAME", "")
ACCOUNT_KEY    = os.getenv("ADLS_ACCOUNT_KEY", "")
SAS_TOKEN      = os.getenv("ADLS_SAS_TOKEN", "")
CONTAINER      = os.getenv("ADLS_CONTAINER", "raw")
REMOTE_PREFIX  = "app/High_Court_Judgements"
DATASET_NAME   = "High_Court_Judgements"

S3_BASE        = "s3://indian-high-court-judgments"
MIN_TEXT_CHARS = 100   # pypdf result shorter than this triggers GPU OCR fallback

DB_PATH        = Path(__file__).parent / "upload_tracker.db"

# Fields duplicated at root level that should only live inside metadata{}
_DUPLICATE_ROOT_FIELDS = {
    "title", "judge", "pdf_link", "cnr", "date_of_registration",
    "decision_date", "disposal_nature", "raw_html_text", "court",
}

FIELD_MAPPINGS = {
    "court_name":           "court_name",
    "title":                "title",
    "description":          "raw_html_text",
    "judge":                "judge",
    "pdf_link":             "pdf_link",
    "cnr":                  "cnr",
    "date_of_registration": "date_of_registration",
    "decision_date":        "decision_date",
    "disposal_nature":      "disposal_nature",
}

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown    = threading.Event()
_active_procs: list = []
_procs_lock  = threading.Lock()


def _signal_handler(sig, frame):
    if _shutdown.is_set():
        print("\n[hc.ingest] Forced exit.", file=sys.stderr, flush=True)
        os._exit(1)
    _shutdown.set()
    with _procs_lock:
        for p in _active_procs:
            try:
                p.kill()
            except Exception:
                pass
    print(
        "\n[hc.ingest] Interrupt — stopping after current work... (Ctrl+C again to force kill)",
        file=sys.stderr, flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Worker functions — module-level so ProcessPoolExecutor can pickle them
# ══════════════════════════════════════════════════════════════════════════════

_OCR_ENGINE  = None
_OCR_BACKEND = None
_USE_GPU_OCR = False


def _worker_ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _init_cpu_worker():
    _worker_ignore_sigint()


def _init_gpu_worker(gpu_id: int, ocr_backend: str):
    _worker_ignore_sigint()
    global _OCR_ENGINE, _OCR_BACKEND, _USE_GPU_OCR
    _OCR_BACKEND = ocr_backend
    _USE_GPU_OCR = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if ocr_backend == "rapidocr":
        try:
            from rapidocr_paddle import RapidOCR
            _OCR_ENGINE = RapidOCR(use_gpu=True)
            print(f"[GPU {gpu_id}] RapidOCR (Paddle) ready", flush=True)
        except ImportError:
            print(f"[GPU {gpu_id}] rapidocr-paddle not installed — OCR disabled", flush=True)
    elif ocr_backend == "easyocr":
        try:
            import easyocr
            _OCR_ENGINE = easyocr.Reader(["en"], gpu=True, verbose=False)
            print(f"[GPU {gpu_id}] EasyOCR ready", flush=True)
        except ImportError:
            print(f"[GPU {gpu_id}] easyocr not installed — OCR disabled", flush=True)


def _pypdf_extract(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
        if pdf_path.stat().st_size < 4000:
            return ""
        reader = PdfReader(pdf_path)
        parts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(parts)
    except Exception:
        return ""


def _ocr_extract(pdf_path: Path) -> str:
    if _OCR_ENGINE is None:
        return ""
    try:
        import numpy as np
        try:
            import fitz
        except ImportError:
            import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        texts = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]

            if _OCR_BACKEND == "rapidocr":
                result, _ = _OCR_ENGINE(img)
                if result:
                    texts.append(" ".join(line[1] for line in result))
            elif _OCR_BACKEND == "easyocr":
                result = _OCR_ENGINE.readtext(img, detail=0)
                if result:
                    texts.append(" ".join(result))
        return "\n\n".join(texts)
    except Exception:
        return ""


def _fix_escape_sequences(data):
    """Recursively replace escaped \\n \\r \\t sequences with real characters."""
    if isinstance(data, dict):
        return {k: _fix_escape_sequences(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_fix_escape_sequences(item) for item in data]
    if isinstance(data, str):
        return data.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    return data


def _clean_judgment(data: dict) -> dict:
    """
    Remove duplicate root-level fields already present in metadata{},
    and fix escaped newline sequences throughout.
    """
    if isinstance(data.get("metadata"), dict):
        for field in _DUPLICATE_ROOT_FIELDS:
            if field in data and field in data["metadata"]:
                del data[field]
    return _fix_escape_sequences(data)


def _pdf_worker(task: dict) -> dict:
    """
    Process one PDF → JSON.  Runs inside a worker process.
    Returns {"success", "skipped", "path", "error"}.
    """
    pdf_path   = Path(task["pdf_path"])
    output_dir = Path(task["output_dir"])
    out_file   = output_dir / f"{pdf_path.stem}.json"

    if out_file.exists():
        return {"success": True, "skipped": True, "path": str(out_file), "error": ""}

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        text = _pypdf_extract(pdf_path)
        if _USE_GPU_OCR and len(text.strip()) < MIN_TEXT_CHARS:
            text = _ocr_extract(pdf_path)
    except Exception as e:
        return {"success": False, "skipped": False, "path": "", "error": str(e)}

    meta = task["metadata"]
    judgment = {
        "doc_id":        pdf_path.stem,
        "doc_name":      pdf_path.name,
        "year":          task["year"],
        "court":         task["court"],
        "bench":         task["bench"],
        "judgment_text": text,
        "metadata":      meta,
    }
    for meta_key, json_key in FIELD_MAPPINGS.items():
        val = meta.get(meta_key)
        if val is not None:
            judgment[json_key] = val

    judgment = _clean_judgment(judgment)

    try:
        out_file.write_text(json.dumps(judgment, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"success": True, "skipped": False, "path": str(out_file), "error": ""}
    except Exception as e:
        return {"success": False, "skipped": False, "path": "", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(msg: str):
    log.info("─" * 60)
    log.info(f"  {msg}")
    log.info("─" * 60)


def _make_adls_client(pool_size: int = 32):
    try:
        from azure.storage.filedatalake import DataLakeServiceClient
        from azure.core.pipeline.transport import RequestsTransport
    except ImportError:
        log.error("Run: pip install azure-storage-file-datalake")
        sys.exit(1)
    if not ACCOUNT_NAME:
        log.error("ADLS_ACCOUNT_NAME not set in .env")
        sys.exit(1)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
    transport = RequestsTransport(connection_pool_size=pool_size)
    url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    if ACCOUNT_KEY:
        return DataLakeServiceClient(account_url=url, credential=ACCOUNT_KEY, transport=transport)
    if SAS_TOKEN:
        tok = SAS_TOKEN if SAS_TOKEN.startswith("?") else "?" + SAS_TOKEN
        return DataLakeServiceClient(account_url=f"{url}{tok}", transport=transport)
    log.error("Set ADLS_ACCOUNT_KEY or ADLS_SAS_TOKEN in .env")
    sys.exit(1)


def _upload_with_retry(fs, remote_path: str, data: bytes, retries: int = 5) -> str:
    delay = 1.0
    for attempt in range(retries):
        try:
            fc = fs.get_file_client(file_path=remote_path)
            fc.upload_data(data, overwrite=True)
            return ""
        except Exception as e:
            if attempt == retries - 1:
                return str(e)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return "max retries exceeded"


def _get_courts(year: str, court_filter) -> list:
    cmd = f"aws s3 ls {S3_BASE}/metadata/parquet/year={year}/ --no-sign-request"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"S3 court listing failed: {r.stderr.strip()}")
        return []
    courts = []
    for line in r.stdout.strip().splitlines():
        if "PRE" in line and "court=" in line:
            courts.append(line.split("court=")[1].rstrip("/").strip())
    courts = sorted(courts)
    if court_filter:
        courts = [c for c in courts if c == court_filter]
    return courts


def _chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _count_s3_files(year: str, court: str) -> int:
    cmd = (f"aws s3 ls {S3_BASE}/data/pdf/year={year}/court={court}/"
           f" --recursive --no-sign-request")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        return 0
    return sum(1 for line in r.stdout.strip().splitlines() if line.strip())


def _make_file_count_batches(courts: list, year: str, target_size: int,
                              workers: int = 32) -> list:
    log.info(f"  Counting S3 files per court (target ~{target_size:,} per batch)...")
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_count_s3_files, year, c): c for c in courts}
        for fut in as_completed(futs):
            counts[futs[fut]] = fut.result()
    total = sum(counts.values())
    log.info(f"  Total files for year={year}: {total:,} across {len(courts)} courts")

    batches, cur_batch, cur_count = [], [], 0
    for court in courts:
        n = counts[court]
        if cur_batch and cur_count + n > target_size * 1.5:
            batches.append(cur_batch)
            cur_batch, cur_count = [], 0
        cur_batch.append(court)
        cur_count += n
        if cur_count >= target_size:
            batches.append(cur_batch)
            cur_batch, cur_count = [], 0
    if cur_batch:
        batches.append(cur_batch)

    for i, b in enumerate(batches):
        n = sum(counts[c] for c in b)
        log.info(f"  Batch {i+1}/{len(batches)}: {len(b)} courts, ~{n:,} files")
    return batches


def _read_parquet_metadata(parquet_path: Path) -> list:
    try:
        df = pd.read_parquet(parquet_path)
        rows = []
        for _, row in df.iterrows():
            d = {}
            for col in row.index:
                v = row[col]
                try:
                    is_na = pd.isna(v)
                except (TypeError, ValueError):
                    is_na = False
                if is_na:
                    d[col] = None
                elif isinstance(v, pd.Timestamp):
                    d[col] = v.isoformat()
                elif isinstance(v, (int, float, bool)):
                    d[col] = v
                else:
                    d[col] = str(v)
            rows.append(d)
        return rows
    except Exception as e:
        log.warning(f"Failed to read parquet {parquet_path}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 0: Refresh inventory from ADLS
# ══════════════════════════════════════════════════════════════════════════════

def stage_inventory(conn, workers: int, year: str = None):
    _banner("Stage 0: Refresh Inventory (scan ADLS)")
    client = _make_adls_client()
    fs = client.get_file_system_client(CONTAINER)
    try:
        fs.get_file_system_properties()
    except Exception as e:
        log.error(f"ADLS connect failed: {e}")
        sys.exit(1)

    scan_path = (f"{REMOTE_PREFIX}/year={year}" if year
                 else REMOTE_PREFIX)
    prefix = scan_path.rstrip("/") + "/"
    already = tracker_count(conn, DATASET_NAME)
    log.info(f"  Already in tracker: {already:,}")
    log.info(f"  Scanning: {scan_path}")

    # Fan out per subdir (year=YYYY subdirs, or bench subdirs when year given)
    subdirs = [
        item.name for item in fs.get_paths(path=scan_path, recursive=False)
        if item.is_directory
    ]
    if not subdirs:
        subdirs = [scan_path]

    total_inserted = 0
    completed = 0
    lock = threading.Lock()

    def _scan_subdir(subdir_path: str):
        # Each thread uses its own ADLS client — not thread-safe to share
        c = _make_adls_client(pool_size=4)
        fs2 = c.get_file_system_client(CONTAINER)
        rows = []
        try:
            for item in fs2.get_paths(path=subdir_path, recursive=True):
                if not item.is_directory:
                    full = item.name
                    rel = full[len(prefix):] if full.startswith(prefix) else full
                    rows.append((rel, DATASET_NAME, item.content_length or 0, ""))
        except Exception as e:
            return subdir_path, rows, str(e)
        return subdir_path, rows, ""

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(_scan_subdir, sd): sd for sd in subdirs}
    with batch_inserter(conn) as insert_row:
        for fut in as_completed(futures):
            subdir_path, rows, error = fut.result()
            name = subdir_path.split("/")[-1]
            with lock:
                completed += 1
            if error:
                log.warning(f"  scan error {name}: {error}")
                continue
            for row in rows:
                insert_row(*row)
            with lock:
                total_inserted += len(rows)
                log.info(f"  [{completed}/{len(subdirs)}] {name}: {len(rows):,} files")
    executor.shutdown(wait=False)
    final = tracker_count(conn, DATASET_NAME)
    log.info(f"  Done. {DATASET_NAME}: {final:,} total (+{final - already:,} new)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Download batch from S3
# ══════════════════════════════════════════════════════════════════════════════

def stage_download_batch(year: str, courts: list, data_path: Path, metadata_path: Path):
    _banner(f"Stage 1: Download  [{', '.join(courts)}]")
    t0 = time.time()

    S3_FLAGS = "--no-sign-request --no-progress --cli-read-timeout 60 --cli-connect-timeout 10"

    def _run_sync(cmd: str) -> int:
        proc = subprocess.Popen(cmd, shell=True)
        with _procs_lock:
            _active_procs.append(proc)
        proc.wait()
        with _procs_lock:
            _active_procs.remove(proc)
        return proc.returncode

    def _sync_court(court: str):
        if _shutdown.is_set():
            return court, False, "interrupted"
        pdf_dst  = str(data_path     / f"year={year}" / f"court={court}") + "/"
        meta_dst = str(metadata_path / f"year={year}" / f"court={court}") + "/"
        Path(pdf_dst).mkdir(parents=True, exist_ok=True)
        Path(meta_dst).mkdir(parents=True, exist_ok=True)

        cmd_pdf  = (f"aws s3 sync {S3_BASE}/data/pdf/year={year}/court={court}/ {pdf_dst}"
                    f" {S3_FLAGS} --request-payer requester --cli-read-timeout 60")
        cmd_meta = (f"aws s3 sync {S3_BASE}/metadata/parquet/year={year}/court={court}/ {meta_dst}"
                    f" {S3_FLAGS} --quiet")

        with ThreadPoolExecutor(max_workers=2) as inner:
            rc_pdf  = inner.submit(_run_sync, cmd_pdf).result()
            rc_meta = inner.submit(_run_sync, cmd_meta).result()

        if _shutdown.is_set():
            return court, False, "interrupted"
        if rc_pdf != 0:
            return court, False, cmd_pdf
        if rc_meta != 0:
            return court, False, cmd_meta
        return court, True, ""

    with ThreadPoolExecutor(max_workers=len(courts)) as ex:
        for fut in as_completed({ex.submit(_sync_court, c): c for c in courts}):
            if _shutdown.is_set():
                break
            court, ok, err = fut.result()
            if ok:
                log.info(f"  Downloaded court={court}")
            elif err == "interrupted":
                log.warning(f"  Skipped court={court} (interrupted)")
            else:
                log.warning(f"  FAILED court={court}: {err}")

    log.info(f"  Download done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2-4: Extract → Clean → Upload  (streaming, all run concurrently)
# ══════════════════════════════════════════════════════════════════════════════

def stage_extract_clean_upload(
    year: str,
    courts: list,
    data_path: Path,
    metadata_path: Path,
    json_path: Path,
    conn,
    pdf_workers: int,
    num_gpus: int,
    use_gpu_ocr: bool,
    ocr_backend: str,
    workers_per_gpu: int,
    upload_workers: int,
    dry_run: bool,
):
    """
    Streaming pipeline:
      PDF workers write raw JSON → cleaner threads receive it immediately →
      uploader threads push to ADLS and record in the SQLite tracker.

    All three stages run concurrently — extract/clean/upload overlap.
    """
    import queue as _q

    _banner(f"Stage 2-4: Extract → Clean → Upload  [{', '.join(courts)}]")
    t0 = time.time()

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = []
    for court in courts:
        court_meta = metadata_path / f"year={year}" / f"court={court}"
        if not court_meta.exists():
            log.warning(f"  No metadata dir for court={court} — skipping")
            continue
        for bench_dir in sorted(court_meta.glob("bench=*")):
            bench      = bench_dir.name.split("=")[1]
            parquet    = bench_dir / "metadata.parquet"
            if not parquet.exists():
                continue
            rows = _read_parquet_metadata(parquet)
            if not rows:
                continue
            # Index rows by pdf_link stem so each PDF gets its own metadata row.
            # pdf_link values look like ".../BRHC01234.pdf" — we match on stem.
            # Falls back to rows[0] for any PDF not found in the index.
            meta_by_stem = {}
            for row in rows:
                link = row.get("pdf_link") or ""
                stem = Path(link).stem
                if stem:
                    meta_by_stem[stem] = row
            fallback_meta = rows[0]
            pdf_dir  = data_path / f"year={year}" / f"court={court}" / f"bench={bench}"
            out_dir  = json_path / f"year={year}" / f"court={court}" / f"bench={bench}"
            for pdf in sorted(pdf_dir.glob("*.pdf")):
                tasks.append({
                    "pdf_path":   str(pdf),
                    "metadata":   meta_by_stem.get(pdf.stem, fallback_meta),
                    "year":       year,
                    "court":      court,
                    "bench":      bench,
                    "output_dir": str(out_dir),
                })

    if not tasks:
        log.info("  No PDFs in this batch — nothing to process")
        return

    log.info(f"  PDFs to process: {len(tasks):,}")

    _DONE    = object()
    upload_q = _q.Queue(maxsize=400)

    ex_done = ex_skip = ex_fail = 0
    up_done = up_skip = up_fail = 0
    _lock    = threading.Lock()   # guards counters AND all SQLite calls

    # ── Uploader threads ───────────────────────────────────────────────────────
    def _uploader_worker():
        nonlocal up_done, up_skip, up_fail
        _client = _make_adls_client(pool_size=4)
        _fs     = _client.get_file_system_client(CONTAINER)
        while True:
            item = upload_q.get()
            if item is _DONE:
                upload_q.task_done()
                break
            out_path = Path(item)
            rel = str(out_path.relative_to(json_path)).replace("\\", "/")
            with _lock:
                already = tracker_exists(conn, rel)
            if already:
                with _lock:
                    up_skip += 1
                upload_q.task_done()
                continue
            if dry_run:
                with _lock:
                    up_done += 1
                upload_q.task_done()
                continue
            try:
                data = out_path.read_bytes()
                err = _upload_with_retry(_fs, f"{REMOTE_PREFIX}/{rel}", data)
                if not err:
                    with _lock:
                        tracker_insert(conn, rel, DATASET_NAME, len(data),
                                       datetime.now(timezone.utc).isoformat())
                        up_done += 1
                    out_path.unlink(missing_ok=True)
                else:
                    log.warning(f"  upload failed {rel}: {err}")
                    with _lock:
                        up_fail += 1
            except Exception as e:
                log.warning(f"  upload error {rel}: {e}")
                with _lock:
                    up_fail += 1
            upload_q.task_done()

    uploader_threads = [threading.Thread(target=_uploader_worker, daemon=True)
                        for _ in range(upload_workers)]
    for t in uploader_threads:
        t.start()

    # ── PDF extraction ─────────────────────────────────────────────────────────
    def _on_result(res: dict):
        nonlocal ex_done, ex_skip, ex_fail
        if res["skipped"]:
            with _lock:
                ex_skip += 1
            return
        if not res["success"]:
            with _lock:
                ex_fail += 1
            if res.get("error"):
                log.warning(f"  extract error: {res['error']}")
            return
        with _lock:
            ex_done += 1
        out_file = res.get("path")
        if out_file:
            upload_q.put(out_file)

    total = len(tasks)

    def _progress():
        processed = ex_done + ex_skip + ex_fail
        if processed % 500 == 0 or processed == total:
            log.info(
                f"  [{processed}/{total}]  "
                f"extract={ex_done}  upload={up_done}  skip={up_skip}  "
                f"fail(ex/up)={ex_fail}/{up_fail}"
            )

    if use_gpu_ocr and num_gpus > 0:
        pools = [
            ProcessPoolExecutor(
                max_workers=workers_per_gpu,
                initializer=_init_gpu_worker,
                initargs=(gpu_id, ocr_backend),
            )
            for gpu_id in range(num_gpus)
        ]
        try:
            futures = {}
            for idx, task in enumerate(tasks):
                futures[pools[idx % num_gpus].submit(_pdf_worker, task)] = idx
            for fut in as_completed(futures):
                if _shutdown.is_set():
                    break
                _on_result(fut.result())
                _progress()
        finally:
            for p in pools:
                p.shutdown(wait=False, cancel_futures=True)
    else:
        ex = ProcessPoolExecutor(max_workers=pdf_workers, initializer=_init_cpu_worker)
        futures = {ex.submit(_pdf_worker, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            if _shutdown.is_set():
                ex.shutdown(wait=False, cancel_futures=True)
                break
            _on_result(fut.result())
            _progress()
        if not _shutdown.is_set():
            ex.shutdown(wait=False)

    for _ in uploader_threads:
        upload_q.put(_DONE)
    for t in uploader_threads:
        t.join()

    elapsed = time.time() - t0
    log.info(
        f"  Batch done: extract={ex_done}  upload={up_done}  skip={up_skip}  "
        f"fail(ex/up)={ex_fail}/{up_fail}  elapsed={elapsed/60:.1f}m"
    )


def _cleanup_batch(year: str, courts: list, data_path: Path, metadata_path: Path):
    for court in courts:
        for d in [
            data_path     / f"year={year}" / f"court={court}",
            metadata_path / f"year={year}" / f"court={court}",
        ]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    log.info(f"  Disk cleanup: removed PDFs + metadata for {len(courts)} courts")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(
        description="HC India ingest: S3 → ADLS (run before pipelines/hc/run.py)"
    )
    parser.add_argument("--year",    required=True, help="Year to process, e.g. 2024")
    parser.add_argument("--court",   default=None,  help="Process only this court code")
    # Batching
    parser.add_argument("--batch-size",  type=int, default=None,  help="Courts per batch")
    parser.add_argument("--batch-files", type=int, default=25000, help="Target files per batch (default 25000)")
    # Workers
    parser.add_argument("--pdf-workers",     type=int, default=cpu_count)
    parser.add_argument("--upload-workers",  type=int,
                        default=int(os.getenv("ADLS_WORKERS", "32")))
    # GPU OCR
    parser.add_argument("--use-gpu-ocr",     action="store_true")
    parser.add_argument("--ocr-backend",     choices=["rapidocr", "easyocr"], default="rapidocr")
    parser.add_argument("--num-gpus",        type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    # Local paths (temp storage for PDFs and extracted JSON)
    parser.add_argument("--data-path",     type=Path, default=_ROOT / "tmp" / "pdf")
    parser.add_argument("--metadata-path", type=Path, default=_ROOT / "tmp" / "parquet")
    parser.add_argument("--json-path",     type=Path, default=_ROOT / "tmp" / "json")
    # DB path override
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help="Upload tracker SQLite DB (default: pipelines/hc/upload_tracker.db)")
    # Control flags
    parser.add_argument("--skip-inventory",  action="store_true", help="Skip ADLS scan (Stage 0)")
    parser.add_argument("--skip-fetch",      action="store_true", help="Skip S3 download (Stage 1)")
    parser.add_argument("--no-disk-cleanup", action="store_true", help="Keep local PDFs after each batch")
    parser.add_argument("--dry-run",         action="store_true", help="Skip ADLS writes")
    args = parser.parse_args()

    # Tune S3 concurrency once at startup
    subprocess.run("aws configure set default.s3.max_concurrent_requests 50", shell=True)
    subprocess.run("aws configure set default.s3.multipart_threshold 64MB",   shell=True)
    subprocess.run("aws configure set default.s3.multipart_chunksize 32MB",   shell=True)

    t_start = time.time()
    log.info("=" * 60)
    log.info("  HC INDIA INGEST PIPELINE  (S3 → ADLS)")
    log.info("=" * 60)
    log.info(f"  Year:           {args.year}")
    log.info(f"  Court filter:   {args.court or 'all'}")
    log.info(f"  PDF workers:    {args.pdf_workers}  (cpu_count={cpu_count})")
    log.info(f"  Upload workers: {args.upload_workers}")
    if args.use_gpu_ocr:
        log.info(f"  GPU OCR:        {args.ocr_backend}  "
                 f"({args.num_gpus} GPUs × {args.workers_per_gpu} workers)")
    else:
        log.info("  GPU OCR:        disabled")
    if args.dry_run:
        log.info("  Mode:           DRY RUN (no ADLS writes)")

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    conn = get_conn(args.db_path)
    log.info(f"  Upload tracker: {args.db_path}")
    for ds, cnt in tracker_datasets(conn):
        log.info(f"    {ds}: {cnt:,} tracked files")

    if not args.skip_inventory:
        stage_inventory(conn, args.upload_workers, year=args.year)

    courts = _get_courts(args.year, args.court)
    if not courts:
        log.error(f"No courts found for year={args.year}. Check AWS CLI and S3 access.")
        conn.close()
        sys.exit(1)

    if args.batch_size:
        batches = list(_chunked(courts, args.batch_size))
        log.info(f"\n  Courts: {len(courts)}  Batches: {len(batches)}  ({args.batch_size} courts each)")
    else:
        log.info(f"\n  Courts: {len(courts)}")
        batches = _make_file_count_batches(courts, args.year, args.batch_files)
        log.info(f"  Batches: {len(batches)}  (~{args.batch_files:,} files each)")

    # Producer-consumer: prefetch next batch while processing current
    import queue as _queue
    dl_queue = _queue.Queue(maxsize=1)
    SENTINEL = object()

    def _downloader():
        for batch in batches:
            if _shutdown.is_set():
                break
            stage_download_batch(args.year, batch, args.data_path, args.metadata_path)
            dl_queue.put(batch)
        dl_queue.put(SENTINEL)

    if not args.skip_fetch:
        threading.Thread(target=_downloader, daemon=True).start()
    else:
        for batch in batches:
            dl_queue.put(batch)
        dl_queue.put(SENTINEL)

    b_idx = 0
    while True:
        if _shutdown.is_set():
            log.warning("Shutdown requested — exiting batch loop")
            break
        batch = dl_queue.get()
        if batch is SENTINEL:
            break

        b_idx += 1
        log.info(f"\n{'═'*60}")
        log.info(f"  BATCH {b_idx}/{len(batches)}:  courts={batch}")
        log.info(f"{'═'*60}")
        t_batch = time.time()

        stage_extract_clean_upload(
            args.year, batch,
            args.data_path, args.metadata_path, args.json_path,
            conn,
            pdf_workers=args.pdf_workers,
            num_gpus=args.num_gpus,
            use_gpu_ocr=args.use_gpu_ocr,
            ocr_backend=args.ocr_backend,
            workers_per_gpu=args.workers_per_gpu,
            upload_workers=args.upload_workers,
            dry_run=args.dry_run,
        )

        if not args.no_disk_cleanup and not args.skip_fetch:
            _cleanup_batch(args.year, batch, args.data_path, args.metadata_path)

        if _shutdown.is_set():
            log.warning(f"  Batch {b_idx} interrupted — stopping")
            break

        log.info(f"  Batch {b_idx}/{len(batches)} complete in {(time.time()-t_batch)/60:.1f}m")

    conn.close()
    log.info(f"\n{'='*60}")
    log.info(f"  INGEST COMPLETE  year={args.year}  total={(time.time()-t_start)/60:.1f}m")
    log.info(f"{'='*60}")
    log.info(f"  Next step: python pipelines/hc/run.py --year-range {args.year} {args.year}")


if __name__ == "__main__":
    main()

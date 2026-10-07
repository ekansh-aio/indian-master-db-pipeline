#!/usr/bin/env python3
"""
SC India - PDF mirror: S3 -> ADLS pdf/ storage (cloud-to-cloud, no local disk).

Streams each PDF directly from the public S3 HTTPS endpoint into ADLS.
No bytes written to local disk.

Flow per file:
  1. GET https://indian-supreme-court-judgments.s3.amazonaws.com/<key>
  2. Read response body into memory
  3. PUT to ADLS under pdf/Supreme_Court_Judgements/...
  4. Record in SQLite tracker

Stages:
  0. Refresh tracker  - scan ADLS pdf/ and register already-present files
  1. Discover         - list all S3 keys for the requested year(s)/language(s)
  2. Stream-upload    - parallel workers stream S3 -> ADLS
  3. Inventory        - write _inventory.json per year to ADLS

Usage:
  python pipelines/sc/sync_pdfs.py --year 2024
  python pipelines/sc/sync_pdfs.py --year 2024 --language english
  python pipelines/sc/sync_pdfs.py --all-years --workers 16
  python pipelines/sc/sync_pdfs.py --year 2024 --workers 16 --dry-run

S3 source layout:
  s3://indian-supreme-court-judgments/data/pdf/year=Y/english/{file}.pdf
  s3://indian-supreme-court-judgments/data/pdf/year=Y/regional/{file}.pdf
  (public bucket - no credentials required)

ADLS target layout:
  pdf/Supreme_Court_Judgements/year=Y/english/{file}.pdf
  pdf/Supreme_Court_Judgements/year=Y/regional/{file}.pdf
"""

import os
import sys
import io as _io
import json
import time
import signal
import threading
import argparse
import logging
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from azure.storage.filedatalake import DataLakeServiceClient, ContentSettings

from pipelines.sc.upload_tracker import (
    get_conn, exists as tracker_exists, insert_one as tracker_insert,
    count as tracker_count, datasets as tracker_datasets, batch_inserter,
)

# ── Logging ────────────────────────────────────────────────────────────────────

if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("sc.sync_pdfs")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────

ACCOUNT_NAME = os.getenv("ADLS_ACCOUNT_NAME", "")
ACCOUNT_KEY  = os.getenv("ADLS_ACCOUNT_KEY", "")
SAS_TOKEN    = os.getenv("ADLS_SAS_TOKEN", "")
CONTAINER    = os.getenv("ADLS_CONTAINER", "raw")

S3_BUCKET       = "indian-supreme-court-judgments"
S3_HTTP_BASE    = f"https://{S3_BUCKET}.s3.amazonaws.com"
S3_LIST_BASE    = f"https://{S3_BUCKET}.s3.amazonaws.com"
ADLS_PDF_ROOT   = "pdf/Supreme_Court_Judgements"
DATASET_NAME    = "SC_PDFs"
ALL_LANGUAGES   = ["english"]

DB_PATH = Path(__file__).parent / "upload_tracker.db"

S3_CONNECT_TIMEOUT = 15
S3_READ_TIMEOUT    = 120
UPLOAD_RETRIES     = 4
UPLOAD_BACKOFF     = [2, 5, 15, 30]
S3_LIST_PAGE       = 1000

# ADLS client timeouts. The SAS-token path previously passed NO timeouts, so the
# SDK fell back to its weak defaults (connect=20s, read=60s) and any slow socket
# write on a throttled link aborted with "The write operation timed out". These
# values are generous because uploads run over a slow/flaky link with 32 workers.
ADLS_CONNECT_TIMEOUT = 120
ADLS_READ_TIMEOUT    = 600
# upload_data() defaults to a single 100MB append request (max_block_size in the
# client constructor does NOT bound this path). A whole PDF in one request body is
# exactly what times out mid-write; chunking keeps every request small and lets
# the SDK retry just the failed chunk instead of the whole file.
ADLS_CHUNK_SIZE      = 8 * 1024 * 1024

# ── Thread-local connection pools ─────────────────────────────────────────────

_tls = threading.local()


def _get_s3_session() -> requests.Session:
    if not getattr(_tls, "s3_session", None):
        s = requests.Session()
        # FIX: previously `Retry(total=3, ...)` with connect/read left as None
        # meant urllib3 ALSO retried read timeouts internally (with its own
        # exponential backoff), stacked on top of _stream_from_s3's own
        # 3-attempt retry loop below. A single slow response could trigger
        # up to 3x3=9 effective attempts with compounding sleeps across two
        # layers, tanking real throughput even though each individual layer
        # looked reasonable in isolation. Now: urllib3 only retries fast,
        # cheap connection-level failures (connect=2) and 5xx statuses;
        # read timeouts are retried exactly once, by the application-level
        # loop in _stream_from_s3, which already logs and backs off.
        retry = Retry(
            total=2, connect=2, read=0, redirect=0, status=2,
            backoff_factor=1, status_forcelist=[500, 502, 503, 504],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=32))
        _tls.s3_session = s
    return _tls.s3_session


def _get_adls_fs():
    if not getattr(_tls, "adls_fs", None):
        _tls.adls_fs = _make_adls_client().get_file_system_client(CONTAINER)
    return _tls.adls_fs


# ── Graceful shutdown ──────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _signal_handler(*_):
    if _shutdown.is_set():
        log.warning("Forced exit requested - terminating immediately.")
        os._exit(1)
    _shutdown.set()
    log.warning(
        "Interrupt received - finishing current uploads then stopping. "
        "(Ctrl+C again to force-kill)"
    )


# ── ADLS helpers ───────────────────────────────────────────────────────────────

def _make_adls_client() -> DataLakeServiceClient:
    account_url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    # FIX: previously timeouts/block sizes were only set on the account-key path;
    # the SAS-token path used azure-core's defaults (connect=20s, read=60s), which
    # is exactly what produced "The write operation timed out" on slow uploads.
    # Apply the same generous settings to both auth paths, and cap SDK-internal
    # retries so they don't stack 3x deep on top of _upload_to_adls_with_retry.
    kwargs = dict(
        connection_timeout=ADLS_CONNECT_TIMEOUT,
        read_timeout=ADLS_READ_TIMEOUT,
        retry_total=2,
        max_block_size=ADLS_CHUNK_SIZE,
        connection_data_block_size=ADLS_CHUNK_SIZE,
    )
    if SAS_TOKEN:
        return DataLakeServiceClient(account_url=account_url, credential=SAS_TOKEN, **kwargs)
    return DataLakeServiceClient(
        account_url=account_url,
        credential=ACCOUNT_KEY,
        **kwargs,
    )


def _upload_to_adls_with_retry(fs_client, adls_path: str, data: bytes) -> None:
    """Upload bytes to ADLS. Raises on permanent failure."""
    last_exc = None
    for attempt in range(UPLOAD_RETRIES):
        try:
            fc = fs_client.get_file_client(adls_path)
            fc.upload_data(
                data,
                overwrite=True,
                chunk_size=ADLS_CHUNK_SIZE,
                content_settings=ContentSettings(content_type="application/pdf"),
            )
            return
        except Exception as exc:
            last_exc = exc
            wait = UPLOAD_BACKOFF[min(attempt, len(UPLOAD_BACKOFF) - 1)]
            log.warning(
                f"  ADLS upload attempt {attempt+1}/{UPLOAD_RETRIES} failed for "
                f"{adls_path!r}: {exc}  (retry in {wait}s)"
            )
            if attempt < UPLOAD_RETRIES - 1 and not _shutdown.is_set():
                time.sleep(wait)
    raise RuntimeError(
        f"ADLS upload permanently failed after {UPLOAD_RETRIES} attempts: {last_exc}"
    )


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _stream_from_s3(key: str) -> bytes:
    """Download one S3 object via HTTPS using a thread-local persistent session."""
    url      = f"{S3_HTTP_BASE}/{key}"
    session  = _get_s3_session()
    last_exc = None
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=(S3_CONNECT_TIMEOUT, S3_READ_TIMEOUT),
                               headers={"User-Agent": "sc-sync/1.0"})
            if resp.status_code == 404:
                raise FileNotFoundError(f"S3 key not found: {key}")
            resp.raise_for_status()
            return resp.content
        except FileNotFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning(f"  S3 fetch error for {key!r} (attempt {attempt+1}/3): {exc}")
        if attempt < 2 and not _shutdown.is_set():
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"S3 fetch permanently failed for {key!r}: {last_exc}")


def _list_s3_keys_for_prefix(prefix: str) -> List[str]:
    """List all PDF keys under an S3 prefix using ListObjectsV2."""
    keys: List[str] = []
    continuation_token = None
    page = 0

    while True:
        if _shutdown.is_set():
            log.info("  Shutdown - stopping S3 key listing early.")
            break

        params = f"list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys={S3_LIST_PAGE}"
        if continuation_token:
            params += f"&continuation-token={urllib.parse.quote(continuation_token)}"

        try:
            resp = _get_s3_session().get(
                f"{S3_LIST_BASE}/?{params}", timeout=30,
                headers={"User-Agent": "sc-sync/1.0"}
            )
            resp.raise_for_status()
            body = resp.text
        except Exception as exc:
            raise RuntimeError(f"S3 ListObjectsV2 failed for prefix {prefix!r}: {exc}") from exc

        page += 1
        page_keys = _parse_s3_list_xml(body)
        keys.extend(k for k in page_keys if k.endswith(".pdf"))

        if page % 10 == 0:
            log.info(f"    ... still listing S3: page {page}, {len(keys):,} keys so far")

        if "<IsTruncated>true</IsTruncated>" not in body:
            break

        token_start = body.find("<NextContinuationToken>")
        token_end   = body.find("</NextContinuationToken>")
        if token_start == -1 or token_end == -1:
            log.error("  S3 list XML truncated but no NextContinuationToken - stopping.")
            break
        continuation_token = body[token_start + len("<NextContinuationToken>"):token_end]

    return keys


def _parse_s3_list_xml(xml: str) -> List[str]:
    keys, pos = [], 0
    while True:
        start = xml.find("<Key>", pos)
        if start == -1:
            break
        end = xml.find("</Key>", start)
        if end == -1:
            break
        keys.append(xml[start + 5:end])
        pos = end + 6
    return keys


def _list_s3_years() -> List[str]:
    """List all year= prefixes available under data/pdf/ on S3."""
    params = (
        f"list-type=2"
        f"&prefix={urllib.parse.quote('data/pdf/')}"
        f"&delimiter={urllib.parse.quote('/')}"
        f"&max-keys={S3_LIST_PAGE}"
    )
    try:
        resp = _get_s3_session().get(
            f"{S3_LIST_BASE}/?{params}", timeout=30,
            headers={"User-Agent": "sc-sync/1.0"}
        )
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        raise RuntimeError(f"S3 year listing failed: {exc}") from exc

    years, pos = [], 0
    while True:
        start = body.find("<Prefix>", pos)
        if start == -1:
            break
        end = body.find("</Prefix>", start)
        if end == -1:
            break
        val = body[start + 8:end]
        pos = end + 9
        if "year=" in val:
            part = val.rstrip("/").split("/")[-1]
            if part.startswith("year="):
                years.append(part[len("year="):])

    return sorted(set(years))


# ── Stage 0: Refresh tracker from existing ADLS content ───────────────────────

def stage_refresh_tracker(conn, year: str, workers: int):
    log.info("=" * 60)
    log.info("Stage 0: Refresh tracker - scan ADLS for already-present PDFs")
    t0     = time.time()
    client = _make_adls_client()
    fs     = client.get_file_system_client(CONTAINER)

    scan_path = f"{ADLS_PDF_ROOT}/year={year}"
    prefix    = scan_path.rstrip("/") + "/"
    before    = tracker_count(conn, DATASET_NAME)
    log.info(f"  Tracker before scan : {before:,}")
    log.info(f"  Scanning ADLS path  : {scan_path}")

    try:
        subdirs = [
            item.name
            for item in fs.get_paths(path=scan_path, recursive=False)
            if item.is_directory
        ]
    except Exception as exc:
        log.warning(f"  ADLS scan failed (path may not exist yet): {exc}")
        subdirs = []

    if not subdirs:
        log.info("  No subdirs found in ADLS - nothing to pre-populate.")
        return

    log.info(f"  Found {len(subdirs)} language subdirs to scan.")
    lock, completed, errors = threading.Lock(), 0, []

    def _scan_subdir(subdir_path: str) -> Tuple[str, list, str]:
        c, rows = _make_adls_client(), []
        fs2     = c.get_file_system_client(CONTAINER)
        try:
            for item in fs2.get_paths(path=subdir_path, recursive=True):
                if not item.is_directory and item.name.endswith(".pdf"):
                    full = item.name
                    rel  = full[len(prefix):] if full.startswith(prefix) else full
                    rows.append((rel, DATASET_NAME, item.content_length or 0, ""))
        except Exception as exc:
            return subdir_path, rows, str(exc)
        return subdir_path, rows, ""

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_subdir, sd): sd for sd in subdirs}
        with batch_inserter(conn) as insert_row:
            for fut in as_completed(futures):
                subdir_path, rows, error = fut.result()
                name = subdir_path.split("/")[-1]
                with lock:
                    completed += 1
                if error:
                    log.warning(f"  [{completed}/{len(subdirs)}] scan error {name}: {error}")
                    errors.append(name)
                    continue
                for row in rows:
                    insert_row(*row)
                log.info(f"  [{completed}/{len(subdirs)}] {name}: {len(rows):,} PDFs found")

    after = tracker_count(conn, DATASET_NAME)
    log.info(
        f"  Stage 0 done in {time.time()-t0:.0f}s - "
        f"tracker: {after:,} (+{after-before:,} new)  errors: {len(errors)}"
    )
    if errors:
        log.warning(f"  Subdirs with scan errors: {errors}")


# ── Stage 1: Discover S3 keys ─────────────────────────────────────────────────

def stage_discover_keys(
    year: str, languages: List[str], conn, list_workers: int = 16,
) -> List[Tuple[str, str]]:
    """
    Returns list of (s3_key, adls_rel_path) not yet tracked.
    adls_rel_path is relative to ADLS_PDF_ROOT, e.g.
      'year=2024/english/foo.pdf'
    """
    log.info("=" * 60)
    log.info(f"Stage 1: Discover S3 keys - year={year}  languages={languages}")
    t0       = time.time()
    all_work: List[Tuple[str, str]] = []
    skipped  = 0
    errors   = 0

    # ── Phase A: fetch all S3 key lists in parallel ───────────────────────────
    lang_keys: dict = {}  # lang -> List[str] | Exception

    def _list_one_lang(lang: str):
        prefix = f"data/pdf/year={year}/{lang}/"
        return lang, _list_s3_keys_for_prefix(prefix)

    active_langs = [l for l in languages if not _shutdown.is_set()]
    eff_workers  = min(list_workers, max(1, len(active_langs)))
    with ThreadPoolExecutor(max_workers=eff_workers) as lex:
        futs = {lex.submit(_list_one_lang, l): l for l in active_langs}
        for fut in as_completed(futs):
            lang = futs[fut]
            try:
                _, keys = fut.result()
                lang_keys[lang] = keys
            except Exception as exc:
                lang_keys[lang] = exc

    # ── Phase B: check tracker (sequential — one SQLite connection) ───────────
    for i, lang in enumerate(sorted(lang_keys), 1):
        val = lang_keys[lang]
        if isinstance(val, Exception):
            log.error(f"  [{i}/{len(languages)}] Failed to list S3 keys for {lang}: {val}")
            errors += 1
            continue
        keys = val
        log.info(f"  [{i}/{len(languages)}] language={lang}: {len(keys):,} S3 keys found")

        lang_prefix = f"year={year}/{lang}/"

        # Fast path: count shortcut for fully-tracked languages
        tracked_count = conn.execute(
            "SELECT COUNT(*) FROM files WHERE dataset=? AND path LIKE ?",
            (DATASET_NAME, lang_prefix + "%"),
        ).fetchone()[0]
        if tracked_count >= len(keys):
            skipped += len(keys)
            log.info(
                f"  [{i}/{len(languages)}] language={lang}: "
                f"0 to upload, {len(keys):,} already tracked"
            )
            continue

        # Bulk set diff
        tracked = set(
            r[0] for r in conn.execute(
                "SELECT path FROM files WHERE dataset=? AND path LIKE ?",
                (DATASET_NAME, lang_prefix + "%"),
            ).fetchall()
        )
        lang_work = 0
        for key in keys:
            rel = key[len("data/pdf/"):]
            if rel in tracked:
                skipped += 1
            else:
                all_work.append((key, rel))
                lang_work += 1

        log.info(
            f"  [{i}/{len(languages)}] language={lang}: "
            f"{lang_work:,} to upload, {len(keys)-lang_work:,} already tracked"
        )

    # Interleave by language so workers spread across english/regional simultaneously
    all_work = _interleave_by_language(all_work)

    log.info(
        f"  Stage 1 done in {time.time()-t0:.0f}s - "
        f"total to upload: {len(all_work):,}  already tracked: {skipped:,}  "
        f"list errors: {errors}"
    )
    return all_work


def _interleave_by_language(work: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Round-robin across language buckets (english / regional)."""
    buckets: dict = defaultdict(list)
    for item in work:
        _, rel = item
        # rel: year=2024/english/foo.pdf -> lang key = english
        parts = rel.split("/")
        lang_key = parts[1] if len(parts) >= 2 else "unknown"
        buckets[lang_key].append(item)

    bucket_lists = list(buckets.values())
    if not bucket_lists:
        return []

    log.info(
        f"  Interleaving {len(work):,} files across "
        f"{len(bucket_lists)} language(s): {list(buckets.keys())}"
    )
    interleaved: List[Tuple[str, str]] = []
    max_len = max(len(b) for b in bucket_lists)
    for i in range(max_len):
        for b in bucket_lists:
            if i < len(b):
                interleaved.append(b[i])
    return interleaved


# ── Stage 2: Stream S3 -> ADLS ─────────────────────────────────────────────────

def stage_stream_upload(
    work: List[Tuple[str, str]],
    conn,
    workers: int,
    dry_run: bool,
) -> Tuple[int, int, int]:
    """Upload all (s3_key, rel_path) pairs. Returns (done, skipped, failed)."""
    log.info("=" * 60)
    log.info(
        f"Stage 2: Stream upload - {len(work):,} files  "
        f"workers={workers}  dry_run={dry_run}"
    )
    t0   = time.time()
    done = skipped = failed = 0
    TICK        = max(1, min(500, len(work) // 20)) if work else 1
    COMMIT_EACH = 500

    pending_lock = threading.Lock()
    pending_rows: list = []

    # Each worker thread gets its own read-only connection to the tracker DB so
    # tracker_exists() calls never contend on the single shared `conn`. Writes
    # still only happen on the main thread via _flush_pending().
    _db_path_row = conn.execute("PRAGMA database_list").fetchone()
    _tracker_db_path = _db_path_row[2] if _db_path_row else None

    def _get_thread_tracker_conn():
        if not getattr(_tls, "tracker_conn", None):
            if _tracker_db_path:
                import sqlite3
                _tls.tracker_conn = sqlite3.connect(_tracker_db_path, check_same_thread=False)
            else:
                _tls.tracker_conn = conn
        return _tls.tracker_conn

    def _upload_one(s3_key: str, rel: str) -> str:
        if _shutdown.is_set():
            return "shutdown"
        if tracker_exists(_get_thread_tracker_conn(), rel):
            return "skipped"
        if dry_run:
            return "dry_run"

        try:
            data = _stream_from_s3(s3_key)
        except FileNotFoundError as exc:
            return f"s3_404: {exc}"
        except Exception as exc:
            return f"s3_fetch_error: {exc}"

        adls_path = f"{ADLS_PDF_ROOT}/{rel}"
        try:
            _upload_to_adls_with_retry(_get_adls_fs(), adls_path, data)
        except Exception as exc:
            return f"adls_upload_error: {exc}"

        with pending_lock:
            pending_rows.append((rel, DATASET_NAME, len(data), datetime.now(timezone.utc).isoformat()))
        return ""

    def _flush_pending():
        with pending_lock:
            rows = pending_rows.copy()
            pending_rows.clear()
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO files (path, dataset, size_bytes, uploaded_at) VALUES (?,?,?,?)",
                rows,
            )
            conn.commit()

    WINDOW    = workers * 4
    completed = 0
    work_iter = iter(work)
    in_flight = {}  # future -> rel

    def _fill():
        while len(in_flight) < WINDOW:
            try:
                key, rel = next(work_iter)
            except StopIteration:
                break
            in_flight[executor.submit(_upload_one, key, rel)] = rel

    with ThreadPoolExecutor(max_workers=workers) as executor:
        _fill()
        while in_flight:
            fut = next(as_completed(in_flight))
            rel    = in_flight.pop(fut)
            # FIX: fut.result() can raise if _upload_one ever throws instead of
            # returning a string. Previously this would propagate and crash the
            # whole pipeline mid-run. Now treated as a per-file failure instead.
            try:
                result = fut.result()
            except Exception as exc:
                result = f"worker_crashed: {exc}"
            completed += 1

            if result == "":
                done += 1
            elif result in ("skipped", "dry_run", "shutdown"):
                skipped += 1
            else:
                failed += 1
                log.warning(f"  FAILED [{rel}]: {result}")

            if done % COMMIT_EACH == 0 and done > 0:
                _flush_pending()

            if completed % TICK == 0 or completed == len(work):
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta_s   = (len(work) - completed) / rate if rate > 0 else 0
                log.info(
                    f"  Progress: {completed:,}/{len(work):,}  "
                    f"done={done:,}  skip={skipped:,}  fail={failed:,}  "
                    f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}m"
                )

            _fill()

    _flush_pending()

    elapsed = time.time() - t0
    rate    = done / elapsed if elapsed > 0 else 0
    log.info(
        f"  Stage 2 done in {elapsed:.0f}s - "
        f"done={done:,}  skip={skipped:,}  fail={failed:,}  avg={rate:.1f} files/s"
    )
    return done, skipped, failed


# ── Stage 3: Write ADLS _inventory.json ───────────────────────────────────────

def stage_write_inventory(conn, year: str):
    """Write _inventory.json for the year to ADLS. Always overwrites."""
    log.info("=" * 60)
    log.info("Stage 3: Write _inventory.json to ADLS")

    try:
        cur  = conn.execute(
            "SELECT path FROM files WHERE dataset=? AND path LIKE ?",
            (DATASET_NAME, f"year={year}/%"),
        )
        rows = sorted(r[0] for r in cur.fetchall())
    except Exception as exc:
        log.error(f"  Could not read tracker DB: {exc}")
        return

    if not rows:
        log.warning(f"  No PDFs tracked for year={year} - skipping _inventory.json")
        return

    inv_path = f"{ADLS_PDF_ROOT}/year={year}/_inventory.json"
    payload  = json.dumps(
        {
            "collection":    "sc_pdfs",
            "year":          int(year),
            "path":          f"{ADLS_PDF_ROOT}/year={year}",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "file_count":    len(rows),
            "files":         rows,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    try:
        _get_adls_fs().get_file_client(inv_path).upload_data(
            payload,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        log.info(f"  Written _inventory.json: {len(rows):,} files -> {inv_path}")
    except Exception as exc:
        log.error(f"  Failed to write _inventory.json: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _banner(msg: str):
    log.info("")
    log.info("=" * 64)
    log.info(f"  {msg}")
    log.info("=" * 64)


def main():
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Stream SC judgment PDFs from public S3 -> ADLS (no local disk).",
    )
    parser.add_argument("--year",           default=None,
                        help="Decision year (e.g. 2024). Omit when using --all-years.")
    parser.add_argument("--all-years",      action="store_true",
                        help="Discover all available years on S3 and run each in sequence.")
    parser.add_argument("--from-year",      type=int, default=None,
                        help="Skip all years before this one (e.g. --from-year 2019)")
    parser.add_argument("--language",       default=None,
                        choices=ALL_LANGUAGES,
                        help="Single language to sync (english or regional). Default: both.")
    parser.add_argument("--workers",        type=int, default=8,
                        help="Parallel upload workers (default 8)")
    parser.add_argument("--list-workers",   type=int, default=16,
                        help="Parallel S3 language-listing workers (default 16)")
    parser.add_argument("--year-workers",   type=int, default=1,
                        help="Process N years in parallel (default 1; set >1 only if DB supports concurrent writers)")
    parser.add_argument("--db-path",        default=None,
                        help="SQLite tracker DB path (default: pipelines/sc/upload_tracker.db)")
    parser.add_argument("--skip-refresh",   action="store_true",
                        help="Skip Stage 0 ADLS scan (tracker is already up to date)")
    parser.add_argument("--skip-inventory", action="store_true",
                        help="Skip Stage 3 _inventory.json write")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Cap total files uploaded (for smoke-tests, e.g. --limit 100)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Discover and count work without uploading anything")
    args = parser.parse_args()

    if not ACCOUNT_NAME:
        log.error("ADLS_ACCOUNT_NAME not set in environment / .env - cannot continue.")
        sys.exit(1)
    if not ACCOUNT_KEY and not SAS_TOKEN:
        log.error("Neither ADLS_ACCOUNT_KEY nor ADLS_SAS_TOKEN set - cannot continue.")
        sys.exit(1)
    if not args.year and not args.all_years:
        log.error("Provide --year <YYYY> or --all-years.")
        sys.exit(1)
    if args.year and args.all_years:
        log.error("--year and --all-years are mutually exclusive.")
        sys.exit(1)

    db_path   = Path(args.db_path) if args.db_path else DB_PATH
    conn      = get_conn(db_path)
    languages = [args.language] if args.language else ALL_LANGUAGES

    # Resolve year list
    if args.all_years:
        log.info("Discovering available years on S3 ...")
        # FIX: a single transient DNS/network blip previously aborted the whole
        # run before it started. Retry a few times with backoff instead.
        years = None
        _last_exc = None
        for _attempt in range(4):
            try:
                years = _list_s3_years()
                break
            except Exception as exc:
                _last_exc = exc
                _wait = [5, 15, 30][min(_attempt, 2)]
                log.warning(f"Year listing failed (attempt {_attempt+1}/4): {exc}  (retry in {_wait}s)")
                if _attempt < 3:
                    time.sleep(_wait)
        if years is None:
            log.error(f"Failed to list S3 years after retries: {_last_exc}")
            sys.exit(1)
        if not years:
            log.error("No years found on S3 - check network / bucket access.")
            sys.exit(1)
        if args.from_year:
            before = len(years)
            years = [y for y in years if int(y) >= args.from_year]
            log.info(f"Found {before} years on S3; skipping before {args.from_year}: {len(years)} remaining: {years}")
        else:
            log.info(f"Found {len(years)} years on S3: {years}")
    else:
        years = [args.year]

    _banner(
        f"SC PDF Sync (cloud-to-cloud) - "
        f"{'all years: ' + str(years) if args.all_years else 'year=' + years[0]}"
        f"{'  [DRY RUN]' if args.dry_run else ''}"
    )
    log.info(f"  ADLS account  : {ACCOUNT_NAME}")
    log.info(f"  Container     : {CONTAINER}")
    log.info(f"  ADLS prefix   : {ADLS_PDF_ROOT}")
    log.info(f"  S3 bucket     : s3://{S3_BUCKET}/data/pdf/")
    log.info(f"  Transport     : HTTPS (no local staging, no AWS CLI)")
    log.info(f"  Languages     : {languages}")
    log.info(f"  Workers       : {args.workers}")
    log.info(f"  Limit         : {args.limit if args.limit else 'unlimited'}")
    log.info(f"  Tracker DB    : {db_path}")
    for ds, cnt in tracker_datasets(conn):
        log.info(f"  Tracker [{ds}]: {cnt:,} files")

    total_done = total_failed = 0
    lock = threading.Lock()

    def _process_year(year_idx: int, year: str) -> Tuple[int, int]:
        """Returns (done, failed) for one year. Uses its own DB conn when year_workers>1."""
        year_conn = get_conn(db_path) if args.year_workers > 1 else conn
        if len(years) > 1:
            _banner(f"Year {year_idx}/{len(years)}: {year}")

        # Stage 0
        if not args.skip_refresh:
            stage_refresh_tracker(year_conn, year, workers=args.workers)
        else:
            log.info("Stage 0: Skipped (--skip-refresh)")
        if _shutdown.is_set():
            return 0, 0

        # Stage 1 - discover keys
        work = stage_discover_keys(
            year, languages, year_conn, list_workers=args.list_workers,
        )
        if args.limit and len(work) > args.limit:
            log.info(f"  --limit {args.limit}: capping work list from {len(work):,} to {args.limit:,}")
            work = work[: args.limit]

        y_done = y_failed = 0
        if not work:
            log.info(f"  year={year}: nothing to upload - all files already tracked.")
        elif not _shutdown.is_set():
            y_done, _, y_failed = stage_stream_upload(
                work, year_conn, workers=args.workers, dry_run=args.dry_run,
            )
            if y_failed > 0:
                log.warning(
                    f"  {y_failed:,} file(s) failed for year={year}. "
                    f"Re-run to retry only the failed files."
                )

        # Stage 3
        if not args.skip_inventory and not args.dry_run and not _shutdown.is_set():
            stage_write_inventory(year_conn, year)
        return y_done, y_failed

    if args.year_workers <= 1:
        for year_idx, year in enumerate(years, 1):
            if _shutdown.is_set():
                break
            y_done, y_failed = _process_year(year_idx, year)
            total_done   += y_done
            total_failed += y_failed
    else:
        log.info(f"Processing {len(years)} years with {args.year_workers} parallel year workers")
        with ThreadPoolExecutor(max_workers=args.year_workers) as yex:
            futs = {yex.submit(_process_year, i, y): y for i, y in enumerate(years, 1)}
            for fut in as_completed(futs):
                if _shutdown.is_set():
                    break
                try:
                    y_done, y_failed = fut.result()
                    with lock:
                        total_done   += y_done
                        total_failed += y_failed
                # FIX: previously only `Exception` was caught here, so a
                # BaseException (e.g. KeyboardInterrupt inside a worker thread)
                # could propagate up and kill year-worker aggregation silently.
                except BaseException as exc:
                    log.error(f"Year worker crashed: {exc}")

    _banner("Done")
    for ds, cnt in tracker_datasets(conn):
        log.info(f"  Tracker [{ds}]: {cnt:,} files")
    if len(years) > 1:
        log.info(f"  Total uploaded: {total_done:,}  Total failed: {total_failed:,}")
    if _shutdown.is_set():
        log.warning("  Pipeline was interrupted - re-run to resume from where it stopped.")


if __name__ == "__main__":
    main()

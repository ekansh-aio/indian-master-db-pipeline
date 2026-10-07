#!/usr/bin/env python3
"""
HC India — eCourts scraper: portal → ADLS pdf/

Scrapes judgments from the eCourts portal (judgments.ecourts.gov.in) and
uploads PDFs + raw metadata JSON directly to ADLS, bypassing S3 entirely.

ADLS output layout:
  pdf/High_Court_Judgements/year=Y/court=C/bench=B/{file}.pdf
  pdf/High_Court_Judgements/year=Y/court=C/bench=B/{file}.json  (raw HTML metadata)

Resume: SQLite scrape_progress table tracks the last scraped_through date per
court/bench. Re-running a completed date range is safe (files already in the
tracker are skipped without re-downloading or re-uploading).

Prerequisites:
  1. Clone https://github.com/vanga/indian-high-court-judgments somewhere.
  2. Set SCRAPER_REPO env var to that path (or pass --scraper-repo).
     The CAPTCHA ONNX model lives at src/captcha_solver/captcha.onnx inside it.
  3. pip install onnx onnxruntime pillow lxml bs4 requests (see requirements.txt).

Usage:
  python pipelines/hc/scrape/run.py --court 33~10 --start-date 2024-01-01 --end-date 2024-01-31
  python pipelines/hc/scrape/run.py --all-courts --start-date 2024-01-01 --end-date 2024-03-31
  python pipelines/hc/scrape/run.py --court 33~10 --dry-run
  python pipelines/hc/scrape/run.py --status   # show scrape_progress table
  python pipelines/hc/scrape/run.py --get-url cnrorders/taphc/orders/2017/x.pdf [--court 33~10]
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from azure.storage.filedatalake import DataLakeServiceClient, ContentSettings

from pipelines.hc.scrape.tracker import (
    get_conn, file_count, paths_for_year, distinct_years,
    get_scraped_through, set_scraped_through, all_progress,
)
from pipelines.hc.scrape.downloader import CourtScraper, _parse_pdf_link_payload
from pipelines.hc.scrape.session import ECourtSession, ROOT_URL, PDF_LINK_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("hc.scrape")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
ACCOUNT_NAME = os.getenv("ADLS_ACCOUNT_NAME", "")
ACCOUNT_KEY  = os.getenv("ADLS_ACCOUNT_KEY", "")
SAS_TOKEN    = os.getenv("ADLS_SAS_TOKEN", "")
CONTAINER    = os.getenv("ADLS_CONTAINER", "raw")
ADLS_PREFIX  = "pdf/High_Court_Judgements"
APP_PREFIX   = "app/High_Court_Judgements"
DATASET      = "HC_Scraped"
APP_DATASET  = "High_Court_Judgements"

DB_PATH     = Path(__file__).parent / "scrape.db"
APP_DB_PATH = Path(__file__).parents[1] / "upload_tracker.db"

# Madras has two bench filters ("1", "2") — add more here as needed
DIST_CODES_BY_COURT = {
    "33~10": ("1", "2"),
}

DEFAULT_BOOTSTRAP = "2008-01-01"
OVERLAP_DAYS      = 14

# ── Shutdown ───────────────────────────────────────────────────────────────────
_shutdown = threading.Event()


def _handle_signal(*_):
    if _shutdown.is_set():
        os._exit(1)
    _shutdown.set()
    print("\n[hc.scrape] Interrupted — stopping after current tasks...", flush=True)


# ── ADLS helpers ───────────────────────────────────────────────────────────────

def _make_adls_client() -> DataLakeServiceClient:
    url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    return DataLakeServiceClient(
        account_url=url,
        credential=SAS_TOKEN if SAS_TOKEN else ACCOUNT_KEY,
        connection_timeout=60,
        read_timeout=120,
    )


def _make_uploader(fs):
    """Return a callable(adls_path, data_bytes) → error_str | '' for one ADLS fs client."""
    def upload(adls_path: str, data: bytes) -> str:
        content_type = (
            "application/pdf"
            if adls_path.lower().endswith(".pdf")
            else "application/json"
        )
        try:
            fc = fs.get_file_client(adls_path)
            fc.upload_data(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            return ""
        except Exception as e:
            return str(e)
    return upload


# ── Court codes ────────────────────────────────────────────────────────────────

def _load_court_codes(scraper_repo: Path) -> dict:
    path = scraper_repo / "court-codes.json"
    if not path.exists():
        raise FileNotFoundError(f"court-codes.json not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Single-URL resolver ────────────────────────────────────────────────────────

def resolve_pdf_url(fragment: str, court_code: str, scraper_repo: Path) -> str:
    """
    Resolve the complete eCourts PDF URL for one fragment
    (e.g. "cnrorders/taphc/orders/2017/x.pdf").

    The returned URL is session-bound — it only works with the cookies from
    this session (JSESSION + JUDGEMENTSSEARCH_SESSID), so use it immediately.
    """
    session = ECourtSession(court_code, scraper_repo)
    session.init()
    session.refresh_token()

    payload              = _parse_pdf_link_payload()
    payload["path"]      = fragment
    payload["val"]       = "0"
    payload["app_token"] = session.app_token

    resp = session.request_api("POST", PDF_LINK_URL, payload)
    try:
        rd = resp.json()
    except ValueError:
        raise RuntimeError(f"non-JSON response resolving {fragment}: {resp.text[:200]}")

    if "outputfile" not in rd:
        raise RuntimeError(f"no outputfile for {fragment}: {rd}")
    return ROOT_URL + rd["outputfile"]


# ── Per-court date ranges ──────────────────────────────────────────────────────

def _build_tasks(court_code: str, start_date: str, end_date: str) -> list[tuple]:
    """
    Return list of (from_date, to_date, dist_code) tuples for this court.

    If DIST_CODES_BY_COURT has an entry, one task is generated per dist_code.
    Otherwise a single task with dist_code=None.
    """
    dist_codes = DIST_CODES_BY_COURT.get(court_code, (None,))
    return [(start_date, end_date, dc) for dc in dist_codes]


def _resolve_start_date(court_code: str, end_date: str, conn,
                        explicit_start: Optional[str]) -> str:
    """Pick the start date for this court's scrape run.

    Uses the maximum (most recent) scraped_through date as the resume point,
    then subtracts OVERLAP_DAYS to catch late-appearing judgments.
    Using max (not min) means we only re-scrape the latest window, not the
    entire history if one bench happens to lag.
    """
    if explicit_start:
        return explicit_start

    progress = all_progress(conn)
    court_dates = [row[2] for row in progress if row[0] == court_code]
    if not court_dates:
        return DEFAULT_BOOTSTRAP

    # max: most recent cursor across all benches for this court
    max_cursor = max(court_dates)
    overlap_start = (
        datetime.strptime(max_cursor, "%Y-%m-%d") - timedelta(days=OVERLAP_DAYS)
    )
    return overlap_start.strftime("%Y-%m-%d")


# ── Validation ─────────────────────────────────────────────────────────────────

def _parse_date(s: str, flag: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        log.error("%s must be YYYY-MM-DD, got: %r", flag, s)
        sys.exit(1)


# ── Inventory ─────────────────────────────────────────────────────────────────

def write_inventory(conn, fs, year: int) -> None:
    """Write _inventory.json for one year= partition to ADLS."""
    paths = paths_for_year(conn, DATASET, year)
    if not paths:
        log.info("  inventory year=%d: no files tracked, skipping", year)
        return

    inv_adls = f"{ADLS_PREFIX}/year={year}/_inventory.json"
    payload  = json.dumps({
        "collection": "hc_pdfs",
        "year":       year,
        "path":       f"{ADLS_PREFIX}/year={year}",
        "files":      paths,
        "file_count": len(paths),
    }, ensure_ascii=False).encode("utf-8")

    try:
        fs.get_file_client(inv_adls).upload_data(
            payload,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        log.info("  inventory year=%d: %d files -> %s", year, len(paths), inv_adls)
    except Exception as e:
        log.warning("  inventory year=%d write failed: %s", year, e)


# ── Main ───────────────────────────────────────────────────────────────────────

def _banner(msg: str):
    log.info("")
    log.info("-" * 60)
    log.info("  %s", msg)
    log.info("-" * 60)


def run_scrape(
    courts: list[str],
    start_date: Optional[str],
    end_date: str,
    conn,
    app_conn,
    scraper_repo: Path,
    max_workers: int,
    dry_run: bool,
    limit: Optional[int] = None,
) -> dict:
    court_codes = _load_court_codes(scraper_repo)

    total_stats: dict = {
        "courts_ok": 0, "courts_err": 0, "courts_interrupted": 0,
        "downloaded": 0, "skipped": 0,
        "uploaded": 0, "upload_fail": 0,
        "app_uploaded": 0, "app_upload_fail": 0,
        "download_fail": 0, "parse_fail": 0,
    }
    lock = threading.Lock()

    def _scrape_one(court_code: str):
        if _shutdown.is_set():
            return court_code, None, "interrupted"
        if court_code not in court_codes:
            return court_code, None, f"unknown court code {court_code}"

        court_name = court_codes[court_code]
        start      = _resolve_start_date(court_code, end_date, conn, start_date)
        task_list  = _build_tasks(court_code, start, end_date)

        # Each thread gets its own ADLS fs client (DataLakeServiceClient is not thread-safe)
        fs       = _make_adls_client().get_file_system_client(CONTAINER)
        uploader = _make_uploader(fs)

        court_stats: dict = {}
        for from_dt, to_dt, dist_code in task_list:
            if _shutdown.is_set():
                break
            scraper = CourtScraper(
                court_code=court_code,
                court_name=court_name,
                from_date=from_dt,
                to_date=to_dt,
                dist_code=dist_code,
                conn=conn,
                uploader=uploader,
                adls_prefix=ADLS_PREFIX,
                app_prefix=APP_PREFIX,
                app_conn=app_conn,
                app_dataset=APP_DATASET,
                scraper_repo=scraper_repo,
                dataset=DATASET,
                dry_run=dry_run,
                stop_event=_shutdown,
                limit=limit,
            )
            try:
                s = scraper.run()
                for k, v in s.items():
                    court_stats[k] = court_stats.get(k, 0) + v
            except Exception as e:
                log.error("  court=%s task failed: %s", court_code, e, exc_info=True)
                return court_code, court_stats or None, str(e)

        if not dry_run and not _shutdown.is_set():
            set_scraped_through(conn, court_code, "__all__", end_date)

        return court_code, court_stats, ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_scrape_one, c): c for c in courts}
        for fut in as_completed(futures):
            court = futures[fut]
            try:
                court_code, stats, err = fut.result()
            except Exception as e:
                err   = str(e)
                stats = None
                court_code = court

            with lock:
                if err == "interrupted":
                    total_stats["courts_interrupted"] += 1
                elif err:
                    total_stats["courts_err"] += 1
                    log.error("  court=%s error: %s", court_code, err)
                else:
                    total_stats["courts_ok"] += 1

                if stats:
                    for k in ("downloaded", "skipped", "uploaded", "upload_fail",
                              "app_uploaded", "app_upload_fail",
                              "download_fail", "parse_fail"):
                        total_stats[k] += stats.get(k, 0)

            if stats and not err:
                log.info(
                    "  court=%s  dl=%d skip=%d dl_fail=%d"
                    "  pdf_up=%d app_up=%d up_fail=%d",
                    court_code,
                    stats.get("downloaded", 0),    stats.get("skipped", 0),
                    stats.get("download_fail", 0), stats.get("uploaded", 0),
                    stats.get("app_uploaded", 0),  stats.get("upload_fail", 0),
                )

    # Write inventory for every year that has files in the tracker
    if not dry_run and total_stats["uploaded"] > 0:
        log.info("Writing inventory files...")
        inv_fs = _make_adls_client().get_file_system_client(CONTAINER)
        for yr in distinct_years(conn, DATASET):
            write_inventory(conn, inv_fs, yr)

    return total_stats


def main():
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = argparse.ArgumentParser(description="HC eCourts scraper → ADLS pdf/")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--court",      help="Single court in tilde format (e.g. 33~10)")
    mode.add_argument("--all-courts", action="store_true", help="Scrape all courts")
    mode.add_argument("--status",     action="store_true",
                      help="Print scrape progress and exit")

    parser.add_argument("--start-date", default=None,
                        help="Start date YYYY-MM-DD (default: auto from progress DB)")
    parser.add_argument("--end-date",   default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel courts (default 4)")
    parser.add_argument("--scraper-repo", default=None,
                        help="Path to indian-high-court-judgments checkout "
                             "(default: SCRAPER_REPO env var)")
    parser.add_argument("--db-path",    default=None,
                        help="SQLite DB path (default: pipelines/hc/scrape/scrape.db)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Discover and log without uploading")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Stop after downloading this many PDFs (for testing)")
    args = parser.parse_args()

    # Resolve scraper repo
    repo_str = args.scraper_repo or os.getenv("SCRAPER_REPO")
    if not repo_str:
        log.error(
            "Set SCRAPER_REPO env var or pass --scraper-repo to the "
            "indian-high-court-judgments checkout directory"
        )
        sys.exit(1)
    scraper_repo = Path(repo_str)
    if not (scraper_repo / "src" / "captcha_solver" / "captcha.onnx").exists():
        log.error("CAPTCHA model not found at %s/src/captcha_solver/captcha.onnx",
                  scraper_repo)
        sys.exit(1)

    db_path  = Path(args.db_path) if args.db_path else DB_PATH
    conn     = get_conn(db_path)
    app_conn = get_conn(APP_DB_PATH)

    if args.status:
        rows = all_progress(conn)
        if rows:
            log.info("Scrape progress (%d entries):", len(rows))
            for court, bench, scraped_through in rows:
                log.info("  court=%-8s bench=%-20s scraped_through=%s",
                         court, bench, scraped_through)
        else:
            log.info("No scrape progress recorded yet.")
        log.info("Total tracked files: %d", file_count(conn, DATASET))
        sys.exit(0)

    if not ACCOUNT_NAME:
        log.error("ADLS_ACCOUNT_NAME not set")
        sys.exit(1)

    # Validate dates before doing any work
    end_dt = _parse_date(args.end_date, "--end-date")
    if args.start_date:
        start_dt = _parse_date(args.start_date, "--start-date")
        if start_dt > end_dt:
            log.error("--start-date %s is after --end-date %s",
                      args.start_date, args.end_date)
            sys.exit(1)

    court_codes = _load_court_codes(scraper_repo)

    if args.court:
        if args.court not in court_codes:
            log.error("Unknown court code %s. Valid codes: %s",
                      args.court, list(court_codes.keys()))
            sys.exit(1)
        courts = [args.court]
    else:
        courts = list(court_codes.keys())

    _banner(
        f"HC Scraper  courts={len(courts)}"
        f"  {args.start_date or 'auto'} to {args.end_date}"
        f"{'  [DRY RUN]' if args.dry_run else ''}"
    )
    log.info("  ADLS account : %s", ACCOUNT_NAME)
    log.info("  Container    : %s", CONTAINER)
    log.info("  ADLS prefix  : %s", ADLS_PREFIX)
    log.info("  Scraper repo : %s", scraper_repo)
    log.info("  DB           : %s", db_path)
    log.info("  Workers      : %d", args.workers)
    log.info("  Tracked files (pdf/) : %d", file_count(conn, DATASET))
    log.info("  Tracked files (app/) : %d", file_count(app_conn, APP_DATASET))

    t0 = time.time()
    stats = run_scrape(
        courts=courts,
        start_date=args.start_date,
        end_date=args.end_date,
        conn=conn,
        app_conn=app_conn,
        scraper_repo=scraper_repo,
        max_workers=args.workers,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    _banner("Done")
    elapsed = time.time() - t0
    log.info(
        "  courts ok=%d err=%d interrupted=%d",
        stats["courts_ok"], stats["courts_err"], stats["courts_interrupted"],
    )
    log.info(
        "  downloaded=%d skipped=%d dl_fail=%d parse_fail=%d  %.0fs",
        stats["downloaded"], stats["skipped"],
        stats["download_fail"], stats["parse_fail"], elapsed,
    )
    log.info(
        "  pdf/ uploaded=%d fail=%d   app/ uploaded=%d fail=%d",
        stats["uploaded"],     stats["upload_fail"],
        stats["app_uploaded"], stats["app_upload_fail"],
    )
    log.info("  Tracked files (pdf/) : %d", file_count(conn, DATASET))
    log.info("  Tracked files (app/) : %d", file_count(app_conn, APP_DATASET))


if __name__ == "__main__":
    main()

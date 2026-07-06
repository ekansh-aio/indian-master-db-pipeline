"""
Re-download the 3 central acts whose PDFs were silently replaced by HTML error pages.

Handles: 123456789/15689, 123456789/12030, 123456789/19795

Usage: python pipeline/redownload_corrupt_pdfs.py [--pdf-dir central_acts_pdfs]
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_HANDLES = {"123456789/15689", "123456789/12030", "123456789/19795"}

DELAY_ON_FAILURE = 45
PDF_MAX_RETRIES  = 5


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=10,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    return session


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200]


def download_pdf_validated(session: requests.Session, pdf_url: str, dest_path: Path) -> bool:
    """Download PDF and reject if the response body is HTML (not a real PDF)."""
    for attempt in range(1, PDF_MAX_RETRIES + 1):
        try:
            resp = session.get(pdf_url, timeout=120)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", DELAY_ON_FAILURE))
                log.warning("429 Too Many Requests — sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except Exception as e:
            log.warning("Attempt %d/%d failed (%s): %s", attempt, PDF_MAX_RETRIES, pdf_url, e)
            time.sleep(DELAY_ON_FAILURE)
            continue

        content = resp.content

        # Validate: real PDFs start with "%PDF-"
        if not content.startswith(b"%PDF-"):
            # Log the first 200 bytes for diagnosis
            preview = content[:200].decode("utf-8", errors="replace")
            log.error(
                "Attempt %d/%d: server returned non-PDF content for %s\n  Preview: %s",
                attempt, PDF_MAX_RETRIES, pdf_url, preview[:150]
            )
            # Don't retry immediately — try alternate bitstream numbers
            break

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        log.info("  Saved valid PDF: %s (%d KB)", dest_path.name, len(content) // 1024)
        return True

    return False


def try_alternate_bitstreams(session: requests.Session, handle_id: str, dest_path: Path) -> bool:
    """
    indiacode.nic.in PDF URLs look like:
      /bitstream/123456789/15689/1/A2017-12.pdf
    Try bitstream slots 1–5 looking for a real PDF.
    """
    from urllib.parse import urljoin
    BASE_URL = "https://www.indiacode.nic.in"

    for slot in range(1, 6):
        # DSpace REST API to list bitstreams for an item
        rest_url = f"{BASE_URL}/rest/handle/{handle_id}"
        try:
            resp = session.get(rest_url, timeout=30)
            if resp.ok:
                # Not always available — parse item page instead
                pass
        except Exception:
            pass

        url = f"{BASE_URL}/bitstream/{handle_id}/{slot}/"
        log.info("  Trying bitstream slot %d: %s", slot, url)
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
            if not resp.ok:
                time.sleep(3)
                continue
            if resp.content.startswith(b"%PDF-"):
                # Detect English (skip Hindi) - check URL
                if "hindi" in resp.url.lower() or re.search(r"/[Hh]\d*\.", resp.url):
                    log.info("  Slot %d is Hindi — skipping", slot)
                    time.sleep(3)
                    continue
                dest_path.write_bytes(resp.content)
                log.info("  Saved via slot %d: %s (%d KB)", slot, resp.url, len(resp.content) // 1024)
                return True
        except Exception as e:
            log.warning("  Slot %d error: %s", slot, e)
        time.sleep(5)

    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="central_acts_pdfs")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    metadata_path = pdf_dir / "metadata.jsonl"
    if not metadata_path.exists():
        log.error("metadata.jsonl not found in %s", pdf_dir)
        sys.exit(1)

    targets = {}
    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            if meta.get("handle_id") in TARGET_HANDLES:
                targets[meta["handle_id"]] = meta

    if not targets:
        log.error("None of the target handles found in metadata.jsonl")
        sys.exit(1)

    log.info("Found %d/%d target acts in metadata", len(targets), len(TARGET_HANDLES))
    session = _make_session()

    for handle_id, meta in targets.items():
        act_name = meta["act_name"]
        pdf_url  = meta.get("pdf_url", "")
        filename = _sanitize_filename(act_name) + ".pdf"
        dest     = pdf_dir / filename

        log.info("--- %s (%s) ---", act_name[:70], handle_id)
        log.info("  Stored URL: %s", pdf_url)

        success = False

        # 1. Try original URL with PDF validation
        if pdf_url:
            log.info("  Attempt: original PDF URL")
            success = download_pdf_validated(session, pdf_url, dest)
            if success:
                log.info("  OK via original URL")

        # 2. Try alternate bitstream slots
        if not success:
            log.info("  Original URL failed — trying alternate bitstream slots")
            success = try_alternate_bitstreams(session, handle_id, dest)

        if not success:
            log.error(
                "  PERMANENTLY FAILED: %s\n"
                "  Manual check needed at: https://www.indiacode.nic.in/handle/%s",
                act_name, handle_id
            )
        else:
            log.info("  Successfully re-downloaded: %s", dest.name)

        time.sleep(8)

    log.info("Done.")


if __name__ == "__main__":
    main()

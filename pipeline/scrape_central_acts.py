"""
Scrape Central Acts from indiacode.nic.in.

Design principles:
- rpp=200 (max DSpace supports) — minimise round-trips to the server
- 15s between list pages, 8s between PDF downloads, 45s on any failure
- Respect Retry-After header when server asks us to back off
- No parallel requests: strictly sequential, one connection at a time

Source: https://www.indiacode.nic.in/handle/123456789/1362
Run:    python pipeline/scrape_central_acts.py [--output-dir central_acts_pdfs]
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://www.indiacode.nic.in"
CENTRAL_ACTS_HANDLE = "123456789/1362"

# rpp=200 — maximum items per DSpace browse page, halves total list requests
BROWSE_URL = (
    f"{BASE_URL}/handle/{CENTRAL_ACTS_HANDLE}"
    f"/browse?type=shorttitle&rpp=200&offset={{offset}}"
)

# Rate limiting — be a polite citizen
DELAY_BETWEEN_PAGES = 12   # seconds between list-page fetches
DELAY_BETWEEN_ACTS  = 2    # seconds between individual act-page fetches
DELAY_BETWEEN_PDFS  = 6    # seconds between PDF downloads
DELAY_ON_FAILURE    = 45   # seconds after any request failure
PDF_MAX_RETRIES     = 3    # retries per PDF (each backed off by DELAY_ON_FAILURE)


def _make_session() -> requests.Session:
    session = requests.Session()
    # Retry only on network-level errors, NOT on 4xx/5xx — we handle those manually
    # so we can respect Retry-After and log properly.
    retry = Retry(
        total=4,
        backoff_factor=5,           # 5, 10, 20, 40 seconds
        status_forcelist=[502, 503, 504],   # NOT 429 — handled below
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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return session


def _get(session: requests.Session, url: str, timeout: int = 45) -> Optional[requests.Response]:
    """Single GET with Retry-After handling and a hard failure sleep."""
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", DELAY_ON_FAILURE))
            log.warning("429 Too Many Requests — sleeping %ds as instructed", wait)
            time.sleep(wait)
            return None
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        log.warning("HTTP error for %s: %s — sleeping %ds", url, e, DELAY_ON_FAILURE)
        time.sleep(DELAY_ON_FAILURE)
        return None
    except Exception as e:
        log.warning("Request failed for %s: %s — sleeping %ds", url, e, DELAY_ON_FAILURE)
        time.sleep(DELAY_ON_FAILURE)
        return None


def scrape_list_page(session: requests.Session, offset: int) -> tuple[list[str], bool]:
    """Fetch one browse page; return (list_of_act_urls, has_next)."""
    url = BROWSE_URL.format(offset=offset)
    log.info("List page offset=%d …", offset)

    resp = _get(session, url)
    if resp is None:
        log.warning("Skipping offset=%d (request failed)", offset)
        return [], True  # signal caller to retry or advance

    soup = BeautifulSoup(resp.text, "html.parser")

    # Act links use ?view_type=browse suffix on individual handle pages
    seen: set[str] = set()
    act_urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "view_type=browse" in href and "/handle/" in href:
            full = urljoin(BASE_URL, href)
            if full not in seen:
                seen.add(full)
                act_urls.append(full)

    # "next" link has offset= pointing to offset+rpp
    has_next = bool(
        soup.find("a", href=re.compile(r"offset=\d+"), string=None)
        and soup.find("a", href=re.compile(rf"offset={offset + 200}"))
    )
    # simpler fallback: any link with offset greater than current
    if not has_next:
        for a in soup.find_all("a", href=True):
            m = re.search(r"offset=(\d+)", a["href"])
            if m and int(m.group(1)) > offset:
                has_next = True
                break

    log.info("  → %d acts  has_next=%s", len(act_urls), has_next)
    return act_urls, has_next


def _extract_table_value(soup: BeautifulSoup, label: str) -> Optional[str]:
    for row in soup.select("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2 and label.lower() in cells[0].get_text(strip=True).lower():
            return cells[1].get_text(strip=True)
    return None


def _extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", text or "")
    return int(m.group(1)) if m else None


def scrape_act_page(session: requests.Session, act_url: str) -> Optional[dict]:
    """Fetch act detail page and return metadata dict, or None on failure."""
    resp = _get(session, act_url, timeout=30)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title — 3-method fallback
    title = None
    el = soup.find(id="short_title")
    if el:
        title = el.get_text(strip=True)
    if not title:
        title = _extract_table_value(soup, "Short Title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None
    if not title:
        log.warning("No title found for %s — skipping", act_url)
        return None

    act_number = _extract_table_value(soup, "Act Number") or ""
    year_raw   = (
        _extract_table_value(soup, "Act Year")
        or _extract_table_value(soup, "Enacted Year")
        or _extract_table_value(soup, "Year")
        or ""
    )
    year = _extract_year(year_raw) or _extract_year(act_number) or _extract_year(title)

    parsed = urlparse(act_url)
    handle_id = parsed.path.lstrip("/").removeprefix("handle/")

    # PDF link — English only.
    # indiacode stores as /bitstream/{handle}/1/engXXX.pdf  (English)
    #                  and /bitstream/{handle}/2/hindiXXX.pdf (Hindi)
    # Skip anything with "hindi" in the filename or path.
    pdf_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        # Skip help PDF and Hindi PDFs
        if "/help/" in href:
            continue
        lower = href.lower()
        if "hindi" in lower or re.search(r"/[Hh]\d*\.", href):
            continue
        pdf_url = urljoin(BASE_URL, href)
        break

    return {
        "act_name":   title,
        "handle_id":  handle_id,
        "act_number": act_number,
        "year":       year,
        "pdf_url":    pdf_url or "",
        "pdf_exists": bool(pdf_url),
        "source_url": act_url,
    }


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200]


def download_pdf(session: requests.Session, pdf_url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        log.debug("Already downloaded: %s", dest_path.name)
        return True

    for attempt in range(1, PDF_MAX_RETRIES + 1):
        resp = _get(session, pdf_url, timeout=120)
        if resp is None:
            log.warning("PDF attempt %d/%d failed: %s", attempt, PDF_MAX_RETRIES, pdf_url)
            continue
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with dest_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = dest_path.stat().st_size // 1024
            log.info("  PDF saved: %s (%d KB)", dest_path.name, size_kb)
            return True
        except Exception as e:
            log.warning("Write failed for %s: %s", dest_path.name, e)
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)

    log.error("PDF permanently failed after %d attempts: %s", PDF_MAX_RETRIES, pdf_url)
    return False


def main():
    ap = argparse.ArgumentParser(description="Scrape Central Acts from indiacode.nic.in")
    ap.add_argument("--output-dir", default="central_acts_pdfs",
                    help="Directory for PDFs and metadata.jsonl (default: central_acts_pdfs)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Only scrape metadata, skip PDF downloads (fast first pass)")
    ap.add_argument("--pdfs-only", action="store_true",
                    help="Only download PDFs for acts already in metadata.jsonl")
    ap.add_argument("--start-offset", type=int, default=0,
                    help="Start pagination at this offset (for manual resume)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    # Load already-scraped handle IDs
    seen_handles: set[str] = set()
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen_handles.add(json.loads(line)["handle_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        log.info("Resuming: %d acts already in metadata.jsonl", len(seen_handles))

    # --pdfs-only: download PDFs for records already scraped
    if args.pdfs_only:
        if not metadata_path.exists():
            log.error("No metadata.jsonl found — run without --pdfs-only first")
            sys.exit(1)
        session = _make_session()
        metas = []
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metas.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        log.info("Downloading PDFs for %d acts …", sum(1 for m in metas if m.get("pdf_exists")))
        downloaded = skipped = failed = 0
        for meta in metas:
            if not meta.get("pdf_url"):
                continue
            filename = _sanitize_filename(meta["act_name"]) + ".pdf"
            dest = out_dir / filename
            if dest.exists() and dest.stat().st_size > 1024:
                skipped += 1
                continue
            ok = download_pdf(session, meta["pdf_url"], dest)
            if ok:
                downloaded += 1
            else:
                failed += 1
            time.sleep(DELAY_BETWEEN_PDFS)
        log.info("PDFs — downloaded=%d  skipped=%d  failed=%d", downloaded, skipped, failed)
        return

    session = _make_session()
    offset = args.start_offset
    total_scraped = 0
    total_downloaded = 0
    consecutive_empty = 0

    while True:
        act_urls, has_next = scrape_list_page(session, offset)

        if not act_urls:
            consecutive_empty += 1
            if consecutive_empty >= 3 or not has_next:
                log.info("Stopping — no acts found for %d consecutive pages", consecutive_empty)
                break
            log.info("Empty page at offset=%d — advancing anyway", offset)
            offset += 200
            time.sleep(DELAY_BETWEEN_PAGES)
            continue
        consecutive_empty = 0

        for i, act_url in enumerate(act_urls):
            meta = scrape_act_page(session, act_url)
            if meta is None:
                continue

            if meta["handle_id"] in seen_handles:
                log.debug("Already scraped: %s", meta["handle_id"])
                continue

            seen_handles.add(meta["handle_id"])
            total_scraped += 1

            with metadata_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            log.info(
                "[%d] %s  year=%s  pdf=%s",
                total_scraped,
                meta["act_name"][:70],
                meta["year"] or "?",
                "yes" if meta["pdf_exists"] else "no",
            )

            if not args.metadata_only and meta["pdf_exists"]:
                filename = _sanitize_filename(meta["act_name"]) + ".pdf"
                dest = out_dir / filename
                ok = download_pdf(session, meta["pdf_url"], dest)
                if ok:
                    total_downloaded += 1
                time.sleep(DELAY_BETWEEN_PDFS)

            # Polite gap between individual act pages (not for the last item)
            if i < len(act_urls) - 1:
                time.sleep(DELAY_BETWEEN_ACTS)

        if not has_next:
            log.info("Last page reached at offset=%d", offset)
            break

        offset += 200
        log.info("Sleeping %ds before next list page …", DELAY_BETWEEN_PAGES)
        time.sleep(DELAY_BETWEEN_PAGES)

    log.info("=" * 55)
    log.info("Done. Scraped=%d  PDFs downloaded=%d", total_scraped, total_downloaded)
    log.info("Metadata: %s", metadata_path)


if __name__ == "__main__":
    main()

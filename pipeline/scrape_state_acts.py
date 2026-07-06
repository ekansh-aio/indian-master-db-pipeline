"""
Scrape State Acts from indiacode.nic.in for one or more Indian states/UTs.

Run:
  python pipeline/scrape_state_acts.py --states karnataka,delhi,maharashtra
  python pipeline/scrape_state_acts.py --all-states
  python pipeline/scrape_state_acts.py --states karnataka --metadata-only
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

STATE_IDS = {
    "andaman_nicobar":              "2454",
    "andhra_pradesh":               "2486",
    "arunachal_pradesh":            "2487",
    "assam":                        "2513",
    "bihar":                        "2488",
    "chandigarh":                   "2489",
    "chhattisgarh":                 "2490",
    "dadra_nagar_haveli_daman_diu": "2492",
    "delhi":                        "2493",
    "goa":                          "2514",
    "gujarat":                      "2455",
    "haryana":                      "2193",
    "himachal_pradesh":             "2494",
    "jammu_kashmir":                "2495",
    "jharkhand":                    "2515",
    "karnataka":                    "2485",
    "kerala":                       "2516",
    "ladakh":                       "14011",
    "lakshadweep":                  "2496",
    "madhya_pradesh":               "2497",
    "maharashtra":                  "2517",
    "manipur":                      "2498",
    "meghalaya":                    "2499",
    "mizoram":                      "2500",
    "nagaland":                     "2501",
    "odisha":                       "2502",
    "puducherry":                   "2503",
    "punjab":                       "2504",
    "rajasthan":                    "2505",
    "sikkim":                       "2506",
    "tamil_nadu":                   "2507",
    "telangana":                    "2508",
    "tripura":                      "2509",
    "uttarakhand":                  "2511",
    "uttar_pradesh":                "2510",
    "west_bengal":                  "2512",
}

STATE_DISPLAY_NAMES = {
    "andaman_nicobar":              "Andaman and Nicobar Islands",
    "andhra_pradesh":               "Andhra Pradesh",
    "arunachal_pradesh":            "Arunachal Pradesh",
    "assam":                        "Assam",
    "bihar":                        "Bihar",
    "chandigarh":                   "Chandigarh",
    "chhattisgarh":                 "Chhattisgarh",
    "dadra_nagar_haveli_daman_diu": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi":                        "Delhi",
    "goa":                          "Goa",
    "gujarat":                      "Gujarat",
    "haryana":                      "Haryana",
    "himachal_pradesh":             "Himachal Pradesh",
    "jammu_kashmir":                "Jammu and Kashmir",
    "jharkhand":                    "Jharkhand",
    "karnataka":                    "Karnataka",
    "kerala":                       "Kerala",
    "ladakh":                       "Ladakh",
    "lakshadweep":                  "Lakshadweep",
    "madhya_pradesh":               "Madhya Pradesh",
    "maharashtra":                  "Maharashtra",
    "manipur":                      "Manipur",
    "meghalaya":                    "Meghalaya",
    "mizoram":                      "Mizoram",
    "nagaland":                     "Nagaland",
    "odisha":                       "Odisha",
    "puducherry":                   "Puducherry",
    "punjab":                       "Punjab",
    "rajasthan":                    "Rajasthan",
    "sikkim":                       "Sikkim",
    "tamil_nadu":                   "Tamil Nadu",
    "telangana":                    "Telangana",
    "tripura":                      "Tripura",
    "uttarakhand":                  "Uttarakhand",
    "uttar_pradesh":                "Uttar Pradesh",
    "west_bengal":                  "West Bengal",
}

DELAY_BETWEEN_PAGES = 12
DELAY_BETWEEN_ACTS  = 2
DELAY_BETWEEN_PDFS  = 6
DELAY_ON_FAILURE    = 45
PDF_MAX_RETRIES     = 3


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=5,
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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


def _get(session: requests.Session, url: str, timeout: int = 45) -> Optional[requests.Response]:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", DELAY_ON_FAILURE))
            log.warning("429 Too Many Requests — sleeping %ds", wait)
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


def scrape_list_page(session: requests.Session, state_id: str, offset: int) -> tuple:
    browse_url = (
        f"{BASE_URL}/handle/123456789/{state_id}"
        f"/browse?type=shorttitle&rpp=200&offset={offset}"
    )
    log.info("List page state_id=%s offset=%d …", state_id, offset)

    resp = _get(session, browse_url)
    if resp is None:
        return [], True

    soup = BeautifulSoup(resp.text, "html.parser")

    seen: set = set()
    act_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "view_type=browse" in href and "/handle/" in href:
            full = urljoin(BASE_URL, href)
            if full not in seen:
                seen.add(full)
                act_urls.append(full)

    has_next = False
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


def scrape_act_page(
    session: requests.Session,
    act_url: str,
    state_key: str,
) -> Optional[dict]:
    resp = _get(session, act_url, timeout=30)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

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
    year_raw = (
        _extract_table_value(soup, "Act Year")
        or _extract_table_value(soup, "Enacted Year")
        or _extract_table_value(soup, "Year")
        or ""
    )
    year = _extract_year(year_raw) or _extract_year(act_number) or _extract_year(title)

    parsed = urlparse(act_url)
    handle_id = parsed.path.lstrip("/").removeprefix("handle/")

    pdf_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        if "/help/" in href:
            continue
        lower = href.lower()
        if "hindi" in lower or re.search(r"/[Hh]\d*\.", href):
            continue
        pdf_url = urljoin(BASE_URL, href)
        break

    return {
        "act_name":    title,
        "handle_id":   handle_id,
        "act_number":  act_number,
        "year":        year,
        "pdf_url":     pdf_url or "",
        "pdf_exists":  bool(pdf_url),
        "source_url":  act_url,
        "state":       state_key,
        "state_display": STATE_DISPLAY_NAMES[state_key],
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
        resp = _get(session, pdf_url, timeout=45)
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
            dest_path.unlink(missing_ok=True)

    log.error("PDF permanently failed after %d attempts: %s", PDF_MAX_RETRIES, pdf_url)
    return False


def scrape_state(
    session: requests.Session,
    state_key: str,
    out_dir: Path,
    metadata_only: bool,
    pdfs_only: bool,
    start_offset: int,
) -> None:
    state_id = STATE_IDS[state_key]
    state_dir = out_dir / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = state_dir / "metadata.jsonl"

    log.info("=" * 60)
    log.info("State: %s  (id=%s)  out=%s", STATE_DISPLAY_NAMES[state_key], state_id, state_dir)

    seen_handles: set = set()
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

    if pdfs_only:
        if not metadata_path.exists():
            log.error("No metadata.jsonl for %s — run without --pdfs-only first", state_key)
            return
        metas = []
        with metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metas.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        downloaded = skipped = failed = 0
        for meta in metas:
            if not meta.get("pdf_url"):
                continue
            dest = state_dir / (_sanitize_filename(meta["act_name"]) + ".pdf")
            if dest.exists() and dest.stat().st_size > 1024:
                skipped += 1
                continue
            ok = download_pdf(session, meta["pdf_url"], dest)
            (downloaded if ok else failed).__class__  # no-op
            if ok:
                downloaded += 1
            else:
                failed += 1
            time.sleep(DELAY_BETWEEN_PDFS)
        log.info("PDFs for %s — downloaded=%d  skipped=%d  failed=%d",
                 state_key, downloaded, skipped, failed)
        return

    offset = start_offset
    total_scraped = 0
    total_downloaded = 0
    consecutive_empty = 0

    while True:
        act_urls, has_next = scrape_list_page(session, state_id, offset)

        if not act_urls:
            consecutive_empty += 1
            if consecutive_empty >= 3 or not has_next:
                log.info("No more acts at offset=%d for %s", offset, state_key)
                break
            offset += 200
            time.sleep(DELAY_BETWEEN_PAGES)
            continue
        consecutive_empty = 0

        for i, act_url in enumerate(act_urls):
            meta = scrape_act_page(session, act_url, state_key)
            if meta is None:
                continue
            if meta["handle_id"] in seen_handles:
                log.debug("Already scraped: %s", meta["handle_id"])
                continue

            seen_handles.add(meta["handle_id"])
            total_scraped += 1

            with metadata_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            log.info("[%d] %s  year=%s  pdf=%s",
                     total_scraped, meta["act_name"][:70],
                     meta["year"] or "?", "yes" if meta["pdf_exists"] else "no")

            if not metadata_only and meta["pdf_exists"]:
                dest = state_dir / (_sanitize_filename(meta["act_name"]) + ".pdf")
                ok = download_pdf(session, meta["pdf_url"], dest)
                if ok:
                    total_downloaded += 1
                time.sleep(DELAY_BETWEEN_PDFS)

            if i < len(act_urls) - 1:
                time.sleep(DELAY_BETWEEN_ACTS)

        if not has_next:
            log.info("Last page reached at offset=%d for %s", offset, state_key)
            break

        offset += 200
        log.info("Sleeping %ds before next list page …", DELAY_BETWEEN_PAGES)
        time.sleep(DELAY_BETWEEN_PAGES)

    log.info("Done %s — scraped=%d  pdfs=%d", state_key, total_scraped, total_downloaded)


def main():
    ap = argparse.ArgumentParser(description="Scrape State Acts from indiacode.nic.in")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--states", help="Comma-separated state keys, e.g. karnataka,delhi")
    group.add_argument("--all-states", action="store_true", help="Scrape all 36 states/UTs")
    ap.add_argument("--pdf-dir", default="state_acts_pdfs",
                    help="Base output directory (default: state_acts_pdfs)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Only scrape metadata, skip PDF downloads")
    ap.add_argument("--pdfs-only", action="store_true",
                    help="Only download PDFs for acts already in metadata.jsonl")
    ap.add_argument("--start-offset", type=int, default=0,
                    help="Start pagination at this offset (single-state resume)")
    args = ap.parse_args()

    if args.all_states:
        states = list(STATE_IDS.keys())
    else:
        states = [s.strip() for s in args.states.split(",") if s.strip()]

    unknown = [s for s in states if s not in STATE_IDS]
    if unknown:
        log.error("Unknown state key(s): %s", ", ".join(unknown))
        log.error("Valid keys: %s", ", ".join(sorted(STATE_IDS.keys())))
        sys.exit(1)

    out_dir = Path(args.pdf_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session()

    for i, state_key in enumerate(states):
        scrape_state(
            session, state_key, out_dir,
            metadata_only=args.metadata_only,
            pdfs_only=args.pdfs_only,
            start_offset=args.start_offset if len(states) == 1 else 0,
        )
        if i < len(states) - 1:
            log.info("Pausing 20s before next state …")
            time.sleep(20)

    log.info("=" * 60)
    log.info("All done. States processed: %d", len(states))


if __name__ == "__main__":
    main()

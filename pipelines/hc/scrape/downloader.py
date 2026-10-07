"""
eCourts judgment downloader — ported from indian-high-court-judgments/download.py.

Key differences from the original:
  - No S3 integration — PDFs and JSON metadata go directly to ADLS
  - Dedup via SQLite tracker instead of S3 index files
  - Bench is extracted from the PDF fragment path (cnrorders/<bench>/...)
  - Decision year is parsed from raw_html so files land in the right year= partition
  - ADLS upload retried 3 times before marking upload_fail
  - PDF download retried 3 times before counting as download_fail (not parse_fail)
  - Failures counted in separate named buckets (parse_fail vs download_fail vs upload_fail)
  - Periodic progress log every PROGRESS_INTERVAL results so long runs are observable
"""

import io
import json
import logging
import re
import time
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from lxml import html as LH

from pipelines.hc.scrape.session import (
    ECourtSession, ROOT_URL, SEARCH_URL, PDF_LINK_URL,
)
from pipelines.hc.scrape.tracker import file_exists, file_insert

log = logging.getLogger("hc.scrape.downloader")

PAGE_SIZE         = 1000
NO_CAPTCHA_BATCH  = 25
PDF_DL_RETRIES    = 3
UPLOAD_RETRIES    = 3
PROGRESS_INTERVAL = 500   # log a progress line every N results processed

# ── search payload template ────────────────────────────────────────────────────
_SEARCH_PAYLOAD_TEMPLATE = (
    "sEcho=1&iColumns=2&sColumns=,&iDisplayStart=0&iDisplayLength=100"
    "&mDataProp_0=0&sSearch_0=&bRegex_0=false&bSearchable_0=true&bSortable_0=true"
    "&mDataProp_1=1&sSearch_1=&bRegex_1=false&bSearchable_1=true&bSortable_1=true"
    "&sSearch=&bRegex=false&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1"
    "&search_txt1=&search_txt2=&search_txt3=&search_txt4=&search_txt5="
    "&pet_res=&state_code=27~1&state_code_li=&dist_code=null&case_no=&case_year="
    "&from_date=&to_date=&judge_name=&reg_year=&fulltext_case_type="
    "&int_fin_party_val=undefined&int_fin_case_val=undefined"
    "&int_fin_court_val=undefined&int_fin_decision_val=undefined"
    "&act=&sel_search_by=undefined&sections=undefined&judge_txt=&act_txt="
    "&section_txt=&judge_val=&act_val=&year_val=&judge_arr=&flag="
    "&disp_nature=&search_opt=PHRASE&date_val=ALL&fcourt_type=2"
    "&citation_yr=&citation_vol=&citation_supl=&citation_page="
    "&case_no1=&case_year1=&pet_res1=&fulltext_case_type1="
    "&citation_keyword=&sel_lang=&proximity=&neu_cit_year=&neu_no="
    "&ajax_req=true&app_token=1fbc7fbb840eb95975c684565909fe6b3b82b8119472020ff10f40c0b1c901fe"
)

_PDF_LINK_PAYLOAD_TEMPLATE = (
    "val=0&lang_flg=undefined&path=cnrorders/taphc/orders/2017/dummy.pdf"
    "&search=+&citation_year=&fcourt_type=2&file_type=undefined"
    "&nc_display=undefined&ajax_req=true"
    "&app_token=c64944b84c687f501f9692e239e2a0ab007eabab497697f359a2f62e4fcd3d10"
)


def _parse_search_payload() -> dict:
    qs = urllib.parse.parse_qs(_SEARCH_PAYLOAD_TEMPLATE)
    return {k: v[0] for k, v in qs.items()}


def _parse_pdf_link_payload() -> dict:
    qs = urllib.parse.parse_qs(_PDF_LINK_PAYLOAD_TEMPLATE)
    return {k: v[0] for k, v in qs.items()}


def _extract_pdf_fragment(onclick: str) -> Optional[str]:
    m = re.search(r"javascript:open_pdf\('.*?','.*?','(.*?)'\)", onclick)
    if m:
        return m.group(1).split("#")[0]
    return None


def _extract_bench(pdf_fragment: str) -> Optional[str]:
    """Extract bench name from fragment like cnrorders/sikkimhc_pg/orders/..."""
    parts = Path(pdf_fragment).parts
    if "cnrorders" in parts:
        idx = parts.index("cnrorders")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _extract_case_details(raw_html: str) -> dict:
    """Parse all structured fields from the portal HTML card."""
    result = {
        "cnr": "", "date_of_registration": "", "decision_date": None,
        "disposal_nature": "", "court": "", "title": "", "judge": "",
    }
    if not raw_html:
        return result
    try:
        tree = LH.fromstring(raw_html)
        def _first(xpath):
            vals = tree.xpath(xpath)
            return vals[0].strip() if vals else ""

        result["cnr"]  = _first(
            './/span[contains(text(),"CNR")]/following-sibling::font/text()')
        result["date_of_registration"] = _first(
            './/span[contains(text(),"Date of registration")]/following-sibling::font/text()')
        result["decision_date"] = _first(
            './/span[contains(text(),"Decision Date")]/following-sibling::font/text()') or None
        result["disposal_nature"] = _first(
            './/span[contains(text(),"Disposal Nature")]/following-sibling::font/text()')
        court_raw = _first('.//span[contains(text(),"Court")]/text()')
        result["court"] = court_raw.split(":", 1)[1].strip() if ":" in court_raw else court_raw
        result["title"] = _first('.//button//text()')
        judge_raw = _first('.//strong/text()')
        result["judge"] = judge_raw.split(":", 1)[1].strip() if ":" in judge_raw else judge_raw
    except Exception:
        pass
    return result


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf. Returns empty string on failure."""
    try:
        from pypdf import PdfReader
        if len(pdf_bytes) < 4000:
            return ""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(parts)
    except Exception:
        return ""


def _build_app_doc(pdf_stem: str, year: int, court_s3: str, bench: str,
                   court_code: str, court_name: str,
                   raw_html: str, pdf_bytes: bytes) -> dict:
    """
    Build the structured JSON document that ingest.py would produce,
    using portal HTML metadata + pypdf text extraction.
    Matches the schema consumed by run.py (chunk + embed stage).
    """
    details = _extract_case_details(raw_html)
    text    = _extract_text_from_pdf(pdf_bytes)
    return {
        "doc_id":        pdf_stem,
        "doc_name":      f"{pdf_stem}.pdf",
        "year":          str(year),
        "court":         court_s3,
        "bench":         bench,
        "judgment_text": text,
        "court_name":    court_name,
        "title":         details["title"],
        "judge":         details["judge"],
        "cnr":           details["cnr"],
        "date_of_registration": details["date_of_registration"],
        "decision_date": details["decision_date"],
        "disposal_nature": details["disposal_nature"],
        "metadata": {
            "court_code":           court_code,
            "court_name":           court_name,
            "cnr":                  details["cnr"],
            "judge":                details["judge"],
            "date_of_registration": details["date_of_registration"],
            "decision_date":        details["decision_date"],
            "disposal_nature":      details["disposal_nature"],
            "court":                details["court"],
            "title":                details["title"],
        },
    }


def _parse_decision_year(raw_html: str) -> Optional[int]:
    """Parse decision year from the judgment HTML snippet."""
    if not raw_html:
        return None
    try:
        tree = LH.fromstring(raw_html)
        dates = tree.xpath(
            './/strong[@class="caseDetailsTD"]'
            '//span[contains(text(),"Decision Date")]/following-sibling::font/text()'
        )
        if dates:
            return datetime.strptime(dates[0].strip(), "%d-%m-%Y").year
    except Exception:
        pass
    return None


class CourtScraper:
    """
    Scrapes eCourts for one court over a date range, uploading each
    PDF + metadata JSON directly to ADLS as it downloads.

    Args:
        court_code:    eCourts tilde format, e.g. "33~10"
        court_name:    Human-readable name for logging
        from_date:     "YYYY-MM-DD"
        to_date:       "YYYY-MM-DD"
        dist_code:     Optional bench filter (Madras uses "1"/"2")
        conn:          SQLite tracker connection
        uploader:      Callable(adls_path, data_bytes) → error_str | ""
        adls_prefix:   ADLS root, e.g. "pdf/High_Court_Judgements"
        scraper_repo:  Path to indian-high-court-judgments checkout (CAPTCHA model)
        dataset:       Dataset name for the tracker
        dry_run:       Log only, no uploads
        stop_event:    threading.Event — set to request early exit
    """

    def __init__(
        self,
        court_code: str,
        court_name: str,
        from_date: str,
        to_date: str,
        conn,
        uploader,
        adls_prefix: str,
        scraper_repo: Path,
        dataset: str = "HC_Scraped",
        dist_code: Optional[str] = None,
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
        limit: Optional[int] = None,
        app_prefix: Optional[str] = None,
        app_conn=None,
        app_dataset: str = "High_Court_Judgements",
    ):
        self.court_code   = court_code
        self.court_s3     = court_code.replace("~", "_")
        self.court_name   = court_name
        self.from_date    = from_date
        self.to_date      = to_date
        self.dist_code    = dist_code
        self.conn         = conn
        self.uploader     = uploader
        self.adls_prefix  = adls_prefix
        self.app_prefix   = app_prefix          # e.g. "app/High_Court_Judgements"
        self.app_conn     = app_conn            # tracker conn for app/ uploads
        self.app_dataset  = app_dataset
        self.scraper_repo = scraper_repo
        self.dataset      = dataset
        self.dry_run      = dry_run
        self.limit        = limit
        self._stop        = stop_event or threading.Event()

        self.session = ECourtSession(court_code, scraper_repo, self._stop)

        # Fixed stat buckets — extra outcomes go in "other" so we never silently
        # accumulate keys that callers don't know about.
        self._total_downloaded = 0  # tracks downloads across pages for --limit
        self.stats = {
            "pages": 0, "results": 0,
            "downloaded": 0, "skipped": 0,
            "uploaded": 0, "upload_fail": 0,
            "app_uploaded": 0, "app_upload_fail": 0,
            "download_fail": 0, "parse_fail": 0,
            "session_refreshes": 0,
        }

    # ── ADLS upload ────────────────────────────────────────────────────────────

    def _upload(self, rel_path: str, data: bytes) -> bool:
        """Upload bytes to ADLS at adls_prefix/rel_path. Returns True on success."""
        if self.dry_run:
            log.debug("  [dry-run] would upload %s (%d bytes)", rel_path, len(data))
            return True
        if file_exists(self.conn, rel_path):
            return True

        adls_path = f"{self.adls_prefix}/{rel_path}"
        last_err  = ""
        for attempt in range(UPLOAD_RETRIES):
            err = self.uploader(adls_path, data)
            if not err:
                file_insert(
                    self.conn, rel_path, self.dataset, len(data),
                    datetime.now(timezone.utc).isoformat(),
                )
                self.stats["uploaded"] += 1
                return True
            last_err = err
            if attempt < UPLOAD_RETRIES - 1:
                delay = 2 ** attempt
                log.debug("  upload retry %d for %s (%.0fs): %s",
                          attempt + 1, rel_path, delay, err)
                time.sleep(delay)

        log.warning("  upload failed after %d attempts %s: %s",
                    UPLOAD_RETRIES, rel_path, last_err)
        self.stats["upload_fail"] += 1
        return False

    # ── PDF download ───────────────────────────────────────────────────────────

    def _download_pdf(self, pdf_fragment: str, row_pos: int) -> Optional[bytes]:
        """
        Fetch a PDF from eCourts. Returns bytes or None on failure.

        Retried PDF_DL_RETRIES times on network errors; non-retried on
        hard failures (empty body, 404-size, non-PDF bytes).
        """
        payload = _parse_pdf_link_payload()
        payload["path"]      = pdf_fragment
        payload["val"]       = str(row_pos)
        payload["app_token"] = self.session.app_token

        last_err = ""
        for attempt in range(PDF_DL_RETRIES):
            try:
                resp = self.session.request_api("POST", PDF_LINK_URL, payload)
            except Exception as e:
                last_err = str(e)
                log.debug("  PDF link request error (attempt %d): %s", attempt + 1, e)
                if attempt < PDF_DL_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue

            try:
                rd = resp.json()
            except Exception:
                log.warning("  non-JSON PDF link response for %s", pdf_fragment)
                return None

            if "outputfile" not in rd:
                log.warning("  no outputfile in PDF response for %s: %s", pdf_fragment, rd)
                return None

            dl_url = ROOT_URL + rd["outputfile"]
            try:
                pdf_resp = requests.get(
                    dl_url, verify=False,
                    headers=self.session.get_headers(), timeout=30,
                )
            except requests.RequestException as e:
                last_err = str(e)
                log.debug("  PDF download error (attempt %d): %s", attempt + 1, e)
                if attempt < PDF_DL_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue

            content = pdf_resp.content
            if not content:
                log.warning("  empty PDF for %s", pdf_fragment)
                return None
            if len(content) == 315:
                log.warning("  404 PDF response for %s", pdf_fragment)
                return None
            if pdf_resp.status_code != 200:
                log.warning("  HTTP %d for %s", pdf_resp.status_code, pdf_fragment)
                return None
            if not content.startswith(b"%PDF"):
                log.warning("  non-PDF bytes for %s: %r", pdf_fragment, content[:16])
                return None
            return content

        log.warning("  PDF download failed after %d attempts for %s: %s",
                    PDF_DL_RETRIES, pdf_fragment, last_err)
        return None

    # ── result row ─────────────────────────────────────────────────────────────

    def _upload_app_doc(self, rel_path: str, data: bytes) -> bool:
        """Upload a processed app/ JSON doc using the app tracker connection."""
        if not self.app_prefix or self.app_conn is None:
            return True  # app upload not configured — skip silently
        if self.dry_run:
            log.debug("  [dry-run] would upload app %s", rel_path)
            return True
        if file_exists(self.app_conn, rel_path):
            return True
        adls_path = f"{self.app_prefix}/{rel_path}"
        last_err  = ""
        for attempt in range(UPLOAD_RETRIES):
            err = self.uploader(adls_path, data)
            if not err:
                file_insert(
                    self.app_conn, rel_path, self.app_dataset, len(data),
                    datetime.now(timezone.utc).isoformat(),
                )
                self.stats["app_uploaded"] += 1
                return True
            last_err = err
            if attempt < UPLOAD_RETRIES - 1:
                time.sleep(2 ** attempt)
        log.warning("  app upload failed after %d attempts %s: %s",
                    UPLOAD_RETRIES, rel_path, last_err)
        self.stats["app_upload_fail"] += 1
        return False

    def _process_row(self, row: list, row_pos: int) -> str:
        """
        Returns one of: "skipped", "downloaded", "parse_fail", "download_fail".
        For each judgment:
          - PDF uploaded to pdf/ (permanent raw copy)
          - Processed app doc (text + metadata) uploaded to app/
          - No intermediate JSON file written anywhere
        """
        html_str = row[1]
        soup     = BeautifulSoup(html_str, "html.parser")

        if not (soup.button and "onclick" in soup.button.attrs):
            return "parse_fail"

        pdf_fragment = _extract_pdf_fragment(soup.button["onclick"])
        if not pdf_fragment:
            return "parse_fail"

        bench = _extract_bench(pdf_fragment)
        if not bench:
            log.debug("  no bench in fragment %s", pdf_fragment)
            return "parse_fail"

        decision_year = _parse_decision_year(html_str)
        if decision_year is None:
            decision_year = datetime.strptime(self.to_date, "%Y-%m-%d").year

        fname    = Path(pdf_fragment).name
        stem     = Path(fname).stem
        part     = f"year={decision_year}/court={self.court_s3}/bench={bench}"
        pdf_rel  = f"{part}/{fname}"
        app_rel  = f"{part}/{stem}.json"

        pdf_already = file_exists(self.conn, pdf_rel)
        app_already = self.app_conn is not None and file_exists(self.app_conn, app_rel)

        if pdf_already and (app_already or self.app_conn is None):
            return "skipped"

        # Download PDF (needed for both pdf/ upload and text extraction)
        pdf_bytes: Optional[bytes] = None
        if not pdf_already or (not app_already and self.app_conn is not None):
            pdf_bytes = self._download_pdf(pdf_fragment, row_pos)
            if pdf_bytes is None:
                return "download_fail"

        # Upload raw PDF to pdf/
        if not pdf_already and pdf_bytes is not None:
            self._upload(pdf_rel, pdf_bytes)

        # Build and upload processed app doc to app/
        if not app_already and pdf_bytes is not None:
            app_doc = _build_app_doc(
                pdf_stem=stem,
                year=decision_year,
                court_s3=self.court_s3,
                bench=bench,
                court_code=self.court_code,
                court_name=self.court_name,
                raw_html=html_str,
                pdf_bytes=pdf_bytes,
            )
            self._upload_app_doc(
                app_rel,
                json.dumps(app_doc, ensure_ascii=False).encode("utf-8"),
            )

        return "downloaded"

    # ── main search loop ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the search + download loop for this court/date range."""
        log.info("  court=%s  %s to %s  dist_code=%s",
                 self.court_code, self.from_date, self.to_date,
                 self.dist_code or "all")
        self.session.init()
        self.session.refresh_token()

        payload = _parse_search_payload()
        payload["from_date"]      = self.from_date
        payload["to_date"]        = self.to_date
        payload["state_code"]     = self.court_code
        payload["app_token"]      = self.session.app_token
        payload["iDisplayLength"] = str(PAGE_SIZE)
        payload["iDisplayStart"]  = "0"
        if self.dist_code is not None:
            payload["dist_code"] = self.dist_code

        pdfs_since_refresh = 0
        results_since_log  = 0

        while not self._stop.is_set():
            try:
                resp = self.session.request_api("POST", SEARCH_URL, payload)
            except Exception as e:
                log.warning("  search request error, retrying: %s", e)
                time.sleep(5)
                continue

            try:
                rd = resp.json()
            except Exception:
                log.warning("  non-JSON search response for %s — retrying", self.court_code)
                time.sleep(5)
                continue

            if rd.get("session_expire") == "Y":
                raise RuntimeError(f"session expired for {self.court_code} after retries")
            if "errormsg" in rd and "reportrow" not in rd:
                raise RuntimeError(f"search API error for {self.court_code}: {rd['errormsg']}")

            rows = rd.get("reportrow", {}).get("aaData", [])
            if not rows:
                break

            self.stats["pages"]   += 1
            self.stats["results"] += len(rows)
            results_since_log     += len(rows)

            for idx, row in enumerate(rows):
                if self._stop.is_set():
                    break
                try:
                    outcome = self._process_row(row, row_pos=idx)
                except Exception as e:
                    log.error("  row error [%s row %d]: %s", self.court_code, idx, e,
                              exc_info=True)
                    outcome = "parse_fail"

                if outcome in self.stats:
                    self.stats[outcome] += 1
                else:
                    self.stats["parse_fail"] += 1

                if outcome == "downloaded":
                    pdfs_since_refresh     += 1
                    self._total_downloaded += 1
                    if self.limit and self._total_downloaded >= self.limit:
                        log.info("  [%s] --limit %d reached, stopping",
                                 self.court_code, self.limit)
                        self._stop.set()
                        break

                if pdfs_since_refresh >= NO_CAPTCHA_BATCH:
                    pdfs_since_refresh = 0
                    self.stats["session_refreshes"] += 1
                    log.debug("  session refresh after %d downloads", NO_CAPTCHA_BATCH)
                    self.session.init()
                    payload["sEcho"]     = str(int(payload.get("sEcho", 1)))
                    payload["app_token"] = self.session.app_token
                    break  # re-fetch same page with fresh session
            else:
                # Normal page advance (no break from refresh above)
                payload["sEcho"]         = str(int(payload.get("sEcho", 1)) + 1)
                payload["iDisplayStart"] = str(
                    int(payload.get("iDisplayStart", 0)) + PAGE_SIZE
                )

            if results_since_log >= PROGRESS_INTERVAL:
                results_since_log = 0
                log.info(
                    "  [%s] results=%d dl=%d skip=%d dl_fail=%d up=%d up_fail=%d",
                    self.court_code,
                    self.stats["results"],    self.stats["downloaded"],
                    self.stats["skipped"],    self.stats["download_fail"],
                    self.stats["uploaded"],   self.stats["upload_fail"],
                )

        log.info(
            "  done [%s]: pages=%d results=%d dl=%d skip=%d "
            "dl_fail=%d parse_fail=%d up=%d up_fail=%d refreshes=%d",
            self.court_code,
            self.stats["pages"],     self.stats["results"],
            self.stats["downloaded"], self.stats["skipped"],
            self.stats["download_fail"], self.stats["parse_fail"],
            self.stats["uploaded"],  self.stats["upload_fail"],
            self.stats["session_refreshes"],
        )
        return self.stats

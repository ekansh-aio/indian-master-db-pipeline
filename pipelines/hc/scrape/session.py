"""
eCourts session management — ported from indian-high-court-judgments/download.py.

Handles:
  - Session init (cookie + JSESSION token) with retry
  - CAPTCHA solving via the ONNX model in the scraper repo (thread-safe model load)
  - app_token refresh cycle (handles malformed JSON responses)
  - request_api with session-expire / errormsg retry + interruptible exponential backoff
"""

import logging
import random
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

import requests
import urllib3

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("hc.scrape.session")

ROOT_URL          = "https://judgments.ecourts.gov.in"
SEARCH_URL        = f"{ROOT_URL}/pdfsearch/?p=pdf_search/home/"
CAPTCHA_URL       = f"{ROOT_URL}/pdfsearch/vendor/securimage/securimage_show.php"
CAPTCHA_TOKEN_URL = f"{ROOT_URL}/pdfsearch/?p=pdf_search/checkCaptcha"
PDF_LINK_URL      = f"{ROOT_URL}/pdfsearch/?p=pdf_search/openpdfcaptcha"
PDF_LINK_URL_WO   = f"{ROOT_URL}/pdfsearch/?p=pdf_search/openpdf"

SESSION_COOKIE       = "JUDGEMENTSSEARCH_SESSID"
ECOURTS_TOKEN_COOKIE = "JSESSION"

_DEFAULT_APP_TOKEN = "490a7e9b99e4553980213a8b86b3235abc51612b038dbdb1f9aa706b633bbd6c"

_CAPTCHA_TMP  = Path("./captcha-tmp")
_CAPTCHA_FAIL = Path("./captcha-failures")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# ── CAPTCHA model ──────────────────────────────────────────────────────────────

_ort_session = None
_model_lock  = threading.Lock()

_CAPTCHA_CHARSET = r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_ITOS  = ["[E]"] + list(_CAPTCHA_CHARSET) + ["[UNK]", "[B]", "[P]"]
_EOS_ID = 0


def _load_captcha_model(scraper_repo: Path) -> None:
    """Thread-safe lazy load of the ONNX CAPTCHA model (double-checked locking)."""
    global _ort_session
    if _ort_session is not None:
        return
    with _model_lock:
        if _ort_session is not None:
            return
        import onnx
        import onnxruntime as rt
        model_path = str(scraper_repo / "src" / "captcha_solver" / "captcha.onnx")
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        _ort_session = rt.InferenceSession(model_path)
        log.info("CAPTCHA ONNX model loaded from %s", model_path)


def _softmax(logits):
    import numpy as np
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _decode_probs(probs):
    ids = probs.argmax(axis=-1).tolist()
    chars = []
    for token_id in ids:
        if token_id == _EOS_ID:
            break
        if 0 <= token_id < len(_ITOS):
            token = _ITOS[token_id]
            if not token.startswith("["):
                chars.append(token)
    return "".join(chars)


def _solve_captcha_image(image_bytes: bytes) -> str:
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - 0.5) / 0.5
    arr = np.expand_dims(arr, axis=0)

    inputs = {_ort_session.get_inputs()[0].name: arr}
    logits = _ort_session.run(None, inputs)[0]
    probs  = _softmax(logits)
    return _decode_probs(probs[0])


# ── Session ────────────────────────────────────────────────────────────────────

class ECourtSession:
    """
    Manages an eCourts HTTP session: cookie, app_token, CAPTCHA solving.

    One instance per download task — do not share across threads.

    stop_event: if set, backoff sleeps and CAPTCHA loops exit early.
    """

    MAX_RETRIES = 5

    def __init__(self, court_code: str, scraper_repo: Path,
                 stop_event: Optional[threading.Event] = None):
        self.court_code    = court_code
        self.scraper_repo  = scraper_repo
        self.session_id    = None
        self.ecourts_token = None
        self.app_token     = _DEFAULT_APP_TOKEN
        self._stop         = stop_event or threading.Event()

        # Create captcha dirs lazily here, not at module import
        _CAPTCHA_TMP.mkdir(parents=True, exist_ok=True)
        _CAPTCHA_FAIL.mkdir(parents=True, exist_ok=True)

    def init(self) -> None:
        """Fetch the search page to obtain session cookies. Retries 4 times."""
        for attempt in range(4):
            if self._stop.is_set():
                raise RuntimeError("shutting down")
            try:
                resp = requests.get(
                    f"{ROOT_URL}/pdfsearch/",
                    verify=False,
                    headers={"User-Agent": _USER_AGENT},
                    timeout=30,
                )
                self.session_id    = resp.cookies.get(SESSION_COOKIE)
                self.ecourts_token = resp.cookies.get(ECOURTS_TOKEN_COOKIE)
                if not self.ecourts_token:
                    raise RuntimeError("JSESSION token missing — IP may be flagged")
                return
            except requests.RequestException as e:
                log.warning("session init attempt %d failed: %s", attempt + 1, e)
                if attempt < 3:
                    self._sleep_backoff(attempt)
        raise RuntimeError(f"session init failed for {self.court_code} after 4 attempts")

    def get_cookie(self) -> str:
        return f"{ECOURTS_TOKEN_COOKIE}={self.ecourts_token}; {SESSION_COOKIE}={self.session_id}"

    def get_headers(self) -> dict:
        return {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cookie": self.get_cookie(),
            "DNT": "1",
            "Origin": ROOT_URL,
            "Referer": ROOT_URL + "/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": _USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        }

    def _update_session_id(self, response: requests.Response) -> None:
        new = response.cookies.get(SESSION_COOKIE)
        if new:
            self.session_id = new

    def solve_captcha(self, captcha_url: str = CAPTCHA_URL) -> str:
        """Iterative CAPTCHA solver — tries up to 10 times before raising."""
        _load_captcha_model(self.scraper_repo)

        for attempt in range(10):
            if self._stop.is_set():
                raise RuntimeError("shutting down")

            try:
                resp = requests.get(
                    captcha_url,
                    headers={"Cookie": self.get_cookie()},
                    verify=False,
                    timeout=30,
                )
            except requests.RequestException as e:
                log.warning("CAPTCHA fetch error (attempt %d): %s", attempt + 1, e)
                continue

            uid          = uuid.uuid4().hex[:8]
            captcha_file = _CAPTCHA_TMP / f"captcha_{self.court_code}_{uid}.png"
            captcha_file.write_bytes(resp.content)
            try:
                text = _solve_captcha_image(resp.content).strip()
            except Exception as e:
                log.warning("CAPTCHA solve error (attempt %d): %s", attempt + 1, e)
                text = ""
            finally:
                try:
                    captcha_file.unlink()
                except OSError:
                    pass

            if len(text) == 6:
                return text
            log.debug("CAPTCHA bad length %d (attempt %d): %r", len(text), attempt + 1, text)

        raise RuntimeError(f"Could not solve CAPTCHA after 10 attempts ({self.court_code})")

    def refresh_token(self) -> None:
        """Solve CAPTCHA and exchange it for a new app_token. Retries 3 times."""
        answer = self.solve_captcha()
        last_err: str = ""
        for attempt in range(3):
            if self._stop.is_set():
                raise RuntimeError("shutting down")
            try:
                resp = requests.post(
                    CAPTCHA_TOKEN_URL,
                    headers=self.get_headers(),
                    data={"captcha": answer, "search_opt": "PHRASE", "ajax_req": "true"},
                    verify=False,
                    timeout=30,
                )
                res = resp.json()
                self.app_token = res["app_token"]
                self._update_session_id(resp)
                return
            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = str(e)
                log.warning("refresh_token attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    self._sleep_backoff(attempt)
        raise RuntimeError(f"refresh_token failed after 3 attempts: {last_err}")

    def _sleep_backoff(self, retry: int, base: float = 1.0, cap: float = 30.0) -> None:
        """Exponential backoff with full jitter. Interruptible via stop_event."""
        delay = random.uniform(0, min(cap, base * (2 ** retry)))
        log.debug("Backoff %.1fs (retry %d)", delay, retry + 1)
        self._stop.wait(delay)  # returns immediately if stop_event is set

    def _solve_pdf_captcha(self, response_dict: dict, payload: dict) -> requests.Response:
        """Handle CAPTCHA challenge in PDF download response. Up to 3 attempts."""
        from lxml import html as LH
        html_str = response_dict["filename"]
        tree     = LH.fromstring(html_str)
        img_src  = ROOT_URL + tree.xpath("//img[@id='captcha_image_pdf']/@src")[0]

        for attempt in range(3):
            text = self.solve_captcha(captcha_url=img_src)
            payload["captcha1"]  = text
            payload["app_token"] = response_dict["app_token"]
            resp     = requests.post(
                PDF_LINK_URL_WO,
                headers=self.get_headers(),
                data=payload,
                verify=False,
                timeout=60,
            )
            try:
                res_json = resp.json()
            except ValueError:
                return resp
            if res_json.get("message") != "Captcha not solved":
                return resp
            log.debug("PDF CAPTCHA not solved (attempt %d)", attempt + 1)
        return resp

    def request_api(self, method: str, url: str, payload: dict,
                    _retry: int = 0) -> requests.Response:
        headers = self.get_headers()
        resp = requests.request(
            method, url, headers=headers, data=payload,
            verify=False, timeout=60,
        )
        try:
            rd = resp.json()
        except Exception:
            rd = {}

        if "app_token" in rd:
            self.app_token = rd["app_token"]
        self._update_session_id(resp)

        if url == CAPTCHA_TOKEN_URL:
            return resp

        # PDF download CAPTCHA challenge
        if "filename" in rd and "securimage_show" in rd.get("filename", ""):
            self.app_token = rd["app_token"]
            return self._solve_pdf_captcha(rd, payload)

        if rd.get("session_expire") == "Y":
            if _retry >= self.MAX_RETRIES:
                return resp
            self._sleep_backoff(_retry)
            self.init()
            self.refresh_token()
            if payload:
                payload["app_token"] = self.app_token
            return self.request_api(method, url, payload, _retry + 1)

        if "errormsg" in rd and "reportrow" not in rd:
            if _retry >= self.MAX_RETRIES:
                return resp
            self._sleep_backoff(_retry)
            self.refresh_token()
            if payload:
                payload["app_token"] = self.app_token
            return self.request_api(method, url, payload, _retry + 1)

        return resp

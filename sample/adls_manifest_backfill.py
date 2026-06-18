"""
Consolidate existing done_0 marker files into per-year manifest files.
done_1 markers are discarded (deletion deferred to a separate cleanup pass).

For each year 2020–2025:
  1. List all *_done_0.json under processed/High_Court_Judgements/year={Y}/
  2. Extract canonical doc keys from paths
  3. Upload processed/manifest/done_0_year={Y}.json (list of doc keys)
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from config import ADLS_CONFIG, DOC_TYPE_CONFIG
from utils.json_helper import safe_json_dumps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

YEARS = list(range(2020, 2026))

HC_NAME = DOC_TYPE_CONFIG[0]["adls_input_path"].lstrip("app/").rstrip("/")  # High_Court_Judgements
PROCESSED_BASE = "processed"
MANIFEST_DIR = f"{PROCESSED_BASE}/manifest"


def _doc_key(marker_path: str, index_type: int) -> str:
    """Strip processed/ prefix and _done_{n}.json suffix → canonical doc key."""
    suffix = f"_done_{index_type}.json"
    rel = marker_path[len(PROCESSED_BASE) + 1:]  # remove "processed/"
    if rel.endswith(suffix):
        rel = rel[: -len(suffix)]
    return rel


def process_year(fetcher: ADLSFetcher, uploader: ADLSUploader, year: int) -> None:
    base_path = f"{PROCESSED_BASE}/{HC_NAME}/year={year}"
    log.info(f"=== Year {year} ===")

    # Collect done_0 markers → manifest
    suffix_0 = "_done_0.json"
    log.info(f"  Scanning for *{suffix_0} under {base_path} ...")
    done0_paths = []
    for p in fetcher.list_files_iter(path=base_path, pattern=f"*{suffix_0}", recursive=True):
        done0_paths.append(p)
        if len(done0_paths) % 50_000 == 0:
            log.info(f"    {len(done0_paths):,} found so far ...")
    log.info(f"  done_0: {len(done0_paths):,} markers found")

    if done0_paths:
        doc_keys = [_doc_key(p, 0) for p in done0_paths]
        manifest_path = f"{MANIFEST_DIR}/done_0_year={year}.json"
        log.info(f"  Uploading manifest → {manifest_path} ...")
        json_bytes = json.dumps(doc_keys, ensure_ascii=False).encode("utf-8")
        file_client = uploader.file_system_client.get_file_client(manifest_path)
        file_client.upload_data(json_bytes, overwrite=True, connection_timeout=600)
        log.info(f"  Manifest uploaded: {len(doc_keys):,} entries")
    else:
        log.info(f"  No done_0 markers for {year}, skipping.")

    # done_1 markers are discarded — deletion deferred to a separate cleanup pass


def main():
    account_name = ADLS_CONFIG["account_name"]
    account_key = ADLS_CONFIG["account_key"]
    container = ADLS_CONFIG["container_name"]

    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing — check .env")
        sys.exit(1)

    fetcher = ADLSFetcher(account_name, account_key, container)
    uploader = ADLSUploader(account_name, account_key, container)

    total_years = len(YEARS)
    for i, year in enumerate(YEARS, 1):
        log.info(f"[{i}/{total_years}] Processing year {year}")
        process_year(fetcher, uploader, year)

    log.info("=== Backfill complete ===")


if __name__ == "__main__":
    main()

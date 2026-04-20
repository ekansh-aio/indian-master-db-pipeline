"""
Cleanup script: deletes all 4 AI Search indexes and all processed/ ADLS files.
Run this before re-indexing from scratch.

Usage:
    python cleanup_for_reindex.py
"""
import logging
from dotenv import load_dotenv

load_dotenv()

from config import SEARCH_CONFIG, ADLS_CONFIG, DOC_TYPE_CONFIG
from core.search_uploader import SearchIndexManager
from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_BASE = "processed"


def delete_all_indexes():
    manager = SearchIndexManager(
        endpoint=SEARCH_CONFIG["endpoint"],
        key=SEARCH_CONFIG["key"]
    )
    all_index_names = [
        name
        for doc_cfg in DOC_TYPE_CONFIG.values()
        for name in doc_cfg["index_names"].values()
    ]
    for index_name in all_index_names:
        if manager.index_exists(index_name):
            ok = manager.delete_index(index_name)
            logger.info(f"  {'Deleted' if ok else 'FAILED to delete'}: {index_name}")
        else:
            logger.info(f"  Already gone: {index_name}")


def delete_processed_adls_files():
    fetcher = ADLSFetcher(
        account_name=ADLS_CONFIG["account_name"],
        account_key=ADLS_CONFIG["account_key"],
        container_name=ADLS_CONFIG["container_name"]
    )
    uploader = ADLSUploader(
        account_name=ADLS_CONFIG["account_name"],
        account_key=ADLS_CONFIG["account_key"],
        container_name=ADLS_CONFIG["container_name"]
    )

    logger.info(f"Listing files under '{PROCESSED_BASE}/'...")
    files = fetcher.list_files(path=PROCESSED_BASE, pattern="*.json", recursive=True)
    logger.info(f"Found {len(files)} files to delete")

    deleted = 0
    failed = 0
    for f in files:
        if uploader.delete_file(f):
            deleted += 1
        else:
            failed += 1

    logger.info(f"ADLS cleanup done: {deleted} deleted, {failed} failed")


if __name__ == "__main__":
    logger.info("=== Step 1: Deleting AI Search indexes ===")
    delete_all_indexes()

    logger.info("=== Step 2: Deleting ADLS processed/ files ===")
    delete_processed_adls_files()

    logger.info("=== Cleanup complete. Safe to re-run the pipeline. ===")

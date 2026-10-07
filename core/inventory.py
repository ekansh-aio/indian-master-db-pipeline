"""
ADLS inventory management for all collections.

Each partition (year or state/year) can have:
  - _inventory.json     : list of files in that partition (basenames)
  - _inventory_es.json  : list of doc_ids already uploaded to ES

Delta computation:
  delta1 = files in app/ but not yet in processed/ (need chunking)
  delta2 = files in processed/ but not yet in ES (need ES upload)

Usage:
    app_files  = load_inventory(fetcher, app_inv_path)
    proc_files = load_inventory(fetcher, proc_inv_path)
    es_files   = load_inventory(fetcher, es_inv_path)

    delta1, delta2 = compute_deltas(app_files, proc_files, es_files, collection)
"""

import json
import logging
from typing import Optional, Set, Tuple

from core.adls_paths import get_es_inventory_path, get_inventory_path, get_processed_path

log = logging.getLogger(__name__)

INVENTORY_FILENAME = "_inventory.json"
ES_INVENTORY_FILENAME = "_inventory_es.json"


# ---------------------------------------------------------------------------
# Load / save inventory
# ---------------------------------------------------------------------------


def load_inventory(fetcher, inv_path: str) -> set:
    """Read an _inventory.json or _inventory_es.json from ADLS.

    Returns set of filename basenames (for _inventory.json) or doc_ids
    (for _inventory_es.json). Empty set if missing or corrupt.
    """
    try:
        inv = fetcher.read_json_file(inv_path)
        if inv and "files" in inv:
            return set(inv["files"])
    except Exception:
        pass
    return set()


def save_inventory(
    uploader,
    inv_path: str,
    files_set: set,
    collection: str,
    year: int,
    year_dir: str,
) -> None:
    """Write _inventory.json to ADLS for a partition."""
    if not files_set:
        log.warning("  Empty files set — skipping _inventory.json write for %s", year_dir)
        return
    payload = json.dumps({
        "collection": collection,
        "year": year,
        "path": year_dir,
        "files": sorted(files_set),
        "file_count": len(files_set),
    }, ensure_ascii=False).encode("utf-8")
    try:
        fc = uploader.file_system_client.get_file_client(inv_path)
        fc.upload_data(payload, overwrite=True)
        log.info("  Saved _inventory.json for %s year %d (%d files)", collection, year, len(files_set))
    except Exception as e:
        log.error("  Failed to save _inventory.json for %s year %d: %s", collection, year, e)


# ---------------------------------------------------------------------------
# ES inventory
# ---------------------------------------------------------------------------


def upload_inventory_es(
    fetcher,
    collection: str,
    year: Optional[int],
    done_ids: set,
    state: Optional[str] = None,
) -> None:
    """Write _inventory_es.json to ADLS — tracks which doc_ids are in ES."""
    if not done_ids:
        year_label = year if year is not None else "N/A"
        log.info("  %s year %s: no uploaded docs — skipping _inventory_es.json", collection, year_label)
        return
    es_inv_path = get_es_inventory_path(collection, year, state)
    payload = json.dumps({
        "collection": collection,
        "year": year,
        "path": es_inv_path,
        "files": sorted(done_ids),
        "file_count": len(done_ids),
    }, ensure_ascii=False).encode("utf-8")
    try:
        fc = fetcher.file_system_client.get_file_client(es_inv_path)
        fc.upload_data(payload, overwrite=True)
        year_label = year if year is not None else "N/A"
        log.info("  Uploaded _inventory_es.json for %s year %s (%d docs) → %s",
                 collection, year_label, len(done_ids), es_inv_path)
    except Exception as e:
        year_label = year if year is not None else "N/A"
        log.error("  Failed to upload _inventory_es.json for %s year %s: %s", collection, year_label, e)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_deltas(
    app_files: set,
    proc_files: set,
    es_files: set,
    collection: str,
) -> Tuple[set, set]:
    """Compute the two deltas for incremental processing.

    delta1: files in app/ that are NOT yet in processed/ (need full pipeline).
    delta2: files in processed/ that are NOT yet in ES (need ES upload only).

    The mapping from app basename to processed basename:
      - HC/SC: "abc.json" → "abc_all_chunks.json"
      - Acts:  "CA_123.json" → "CA_123.json" (same basename, processed has chunks field)
    """
    if collection in ("hc", "sc"):
        def app_to_proc(bn):
            return bn.replace(".json", "_all_chunks.json")

        def proc_to_doc_id(bn):
            return bn.replace("_all_chunks.json", "")
    else:
        def app_to_proc(bn):
            return bn  # acts use the same filename

        def proc_to_doc_id(bn):
            return bn.replace(".json", "")

    delta1 = {f for f in app_files if app_to_proc(f) not in proc_files}
    delta2 = {f for f in proc_files if proc_to_doc_id(f) not in es_files}

    return delta1, delta2

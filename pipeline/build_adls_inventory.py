"""
Build per-year inventory files across all ADLS collections.

Walks:
  processed/High_Court_Judgements/year={Y}/   → writes processed/High_Court_Judgements/year={Y}/_inventory.json
  processed/Supreme_Court_Judgements/year={Y}/ → writes processed/Supreme_Court_Judgements/year={Y}/_inventory.json
  processed/central_acts/{year}/               → writes processed/central_acts/{year}/_inventory.json
  app/High_Court_Judgements/year={Y}/          → writes app/High_Court_Judgements/year={Y}/_inventory.json
  app/Supreme_Court_Judgements/year={Y}/       → writes app/Supreme_Court_Judgements/year={Y}/_inventory.json

Each _inventory.json:
  {
    "collection": "hc_processed" | "sc_processed" | "ca_processed" | "hc_app" | "sc_app",
    "year": 2020,
    "path": "processed/High_Court_Judgements/year=2020",
    "files": ["file1.json", ...],        # basenames only
    "file_count": 12345,
    "es_uploaded": true | false | null,  # null = not checked
    "built_at": "2026-06-20T..."
  }

ES upload tracking:
  If --check-es is passed, for each processed collection year, checks whether the doc_ids
  derived from filenames exist in ES using bulk_mget. Sets es_uploaded = true/false.
  Equivalent to --verify-resume in adls_to_es_pipeline.py but writes the result into the
  inventory file so it persists across sessions.

Usage:
  # Build inventory for all collections (no ES check, fast)
  python pipeline/build_adls_inventory.py

  # Build only HC processed
  python pipeline/build_adls_inventory.py --collections hc_processed

  # Build + check ES upload status
  python pipeline/build_adls_inventory.py --check-es

  # Force rebuild even if _inventory.json already exists
  python pipeline/build_adls_inventory.py --force

  # Read existing inventories and print a summary table (no ADLS walk)
  python pipeline/build_adls_inventory.py --summary

Environment (same .env as adls_to_es_pipeline.py):
  ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY, ADLS_CONTAINER_NAME
  ES_URL, ES_API_KEY  (only if --check-es)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import AzureError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("adls_inventory")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Collection definitions
# ---------------------------------------------------------------------------

COLLECTIONS = {
    "hc_processed": {
        "root": "processed/High_Court_Judgements",
        "year_fmt": "year={Y}",          # subdirs are year=2020, year=2021, ...
        "pattern": "_all_chunks.json",
        "es_index": "hc_judgements",
        "doc_id_from_stem": lambda stem: stem[:-len("_all_chunks")] if stem.endswith("_all_chunks") else stem,
    },
    "sc_processed": {
        "root": "processed/Supreme_Court_Judgements",
        "year_fmt": "year={Y}",
        "pattern": "_all_chunks.json",
        "es_index": "sc_judgements",
        "doc_id_from_stem": lambda stem: stem[:-len("_all_chunks")] if stem.endswith("_all_chunks") else stem,
    },
    "ca_processed": {
        "root": "processed/central_acts",
        "year_fmt": "{Y}",               # subdirs are plain integers: 2020, 2021, ...
        "pattern": ".json",
        "es_index": "central_acts",
        "doc_id_from_stem": lambda stem: stem,
    },
    "hc_app": {
        "root": "app/High_Court_Judgements",
        "year_fmt": "year={Y}",
        "pattern": ".json",
        "es_index": None,
        "doc_id_from_stem": None,
    },
    "sc_app": {
        "root": "app/Supreme_Court_Judgements",
        "year_fmt": "year={Y}",
        "pattern": ".json",
        "es_index": None,
        "doc_id_from_stem": None,
    },
}

INVENTORY_FILENAME = "_inventory.json"


# ---------------------------------------------------------------------------
# ADLS helpers
# ---------------------------------------------------------------------------

def _build_fs_client() -> object:
    account_name = os.environ.get("ADLS_ACCOUNT_NAME")
    account_key = os.environ.get("ADLS_ACCOUNT_KEY")
    container = os.environ.get("ADLS_CONTAINER_NAME")
    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing in .env (ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME)")
        sys.exit(1)
    url = f"https://{account_name}.dfs.core.windows.net"
    svc = DataLakeServiceClient(account_url=url, credential=account_key)
    return svc.get_file_system_client(container)


def _list_subdirs(fs_client, path: str) -> List[str]:
    """Return immediate child directory names under path."""
    try:
        items = fs_client.get_paths(path=path, recursive=False)
        return [Path(p.name).name for p in items if p.is_directory]
    except AzureError as e:
        log.warning("Could not list %s: %s", path, e)
        return []


def _list_files_under(fs_client, path: str, suffix: str) -> List[str]:
    """Return basenames of all files with given suffix under path (recursive)."""
    try:
        items = fs_client.get_paths(path=path, recursive=True)
        return [
            Path(p.name).name
            for p in items
            if not p.is_directory
            and p.name.endswith(suffix)
            and Path(p.name).name != INVENTORY_FILENAME
        ]
    except AzureError as e:
        log.warning("Could not list files under %s: %s", path, e)
        return []


def _upload_json(fs_client, adls_path: str, obj: dict) -> bool:
    try:
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        fc = fs_client.get_file_client(adls_path)
        fc.upload_data(data, overwrite=True)
        return True
    except AzureError as e:
        log.error("Failed to write %s: %s", adls_path, e)
        return False


def _read_json(fs_client, adls_path: str) -> Optional[dict]:
    try:
        fc = fs_client.get_file_client(adls_path)
        content = fc.download_file().readall()
        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ES helpers (only used with --check-es)
# ---------------------------------------------------------------------------

def _build_es_client():
    from elasticsearch import Elasticsearch
    es_url = os.environ.get("ES_URL")
    if not es_url:
        log.error("ES_URL not set")
        sys.exit(1)
    api_key = os.environ.get("ES_API_KEY")
    if api_key:
        es = Elasticsearch(es_url, api_key=api_key, request_timeout=120)
    else:
        user, pw = os.environ.get("ES_USER"), os.environ.get("ES_PASS")
        es = Elasticsearch(es_url, http_auth=(user, pw), request_timeout=120) \
            if (user and pw) else Elasticsearch(es_url, request_timeout=120)
    if not es.ping():
        log.error("Cannot connect to Elasticsearch — check ES_URL / credentials")
        sys.exit(1)
    return es


def _check_es_coverage(es, index: str, doc_ids: List[str]) -> float:
    """Return fraction of doc_ids found in ES index (0.0–1.0). Returns -1.0 on error."""
    if not doc_ids:
        return 1.0
    try:
        found = 0
        batch_size = 500
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i + batch_size]
            resp = es.mget(index=index, body={"ids": batch}, _source=False)
            found += sum(1 for d in resp["docs"] if d.get("found"))
        return found / len(doc_ids)
    except Exception as e:
        log.warning("ES mget failed for index %s: %s", index, e)
        return -1.0


# ---------------------------------------------------------------------------
# Core inventory builder
# ---------------------------------------------------------------------------

def build_year_inventory(
    fs_client,
    collection_key: str,
    year: int,
    year_dir: str,       # full ADLS path to the year directory
    col: dict,
    es=None,
    force: bool = False,
) -> Optional[dict]:
    inv_path = f"{year_dir}/{INVENTORY_FILENAME}"

    if not force:
        existing = _read_json(fs_client, inv_path)
        if existing and existing.get("file_count") is not None:
            # Re-check ES if requested and not already checked
            if es and col["es_index"] and existing.get("es_uploaded") is None:
                pass  # fall through to rebuild
            else:
                log.info("  %s year=%s: inventory exists (%d files), skipping",
                         collection_key, year, existing.get("file_count", 0))
                return existing

    log.info("  %s year=%s: listing files ...", collection_key, year)
    t0 = time.time()
    files = _list_files_under(fs_client, year_dir, col["pattern"])
    elapsed = time.time() - t0
    log.info("  %s year=%s: %d files in %.1fs", collection_key, year, len(files), elapsed)

    es_uploaded = None
    if es and col["es_index"] and col["doc_id_from_stem"] and files:
        doc_ids = [col["doc_id_from_stem"](Path(f).stem) for f in files]
        log.info("  %s year=%s: checking ES coverage for %d docs ...", collection_key, year, len(doc_ids))
        coverage = _check_es_coverage(es, col["es_index"], doc_ids)
        if coverage < 0:
            es_uploaded = None
        else:
            es_uploaded = coverage >= 0.99   # treat ≥99% as "uploaded"
            log.info("  %s year=%s: ES coverage=%.1f%%  uploaded=%s",
                     collection_key, year, coverage * 100, es_uploaded)

    inv = {
        "collection": collection_key,
        "year": year,
        "path": year_dir,
        "files": sorted(files),
        "file_count": len(files),
        "es_uploaded": es_uploaded,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    ok = _upload_json(fs_client, inv_path, inv)
    if ok:
        log.info("  Wrote %s (%d files)", inv_path, len(files))
    return inv


def build_collection(
    fs_client,
    collection_key: str,
    col: dict,
    years_filter: Optional[Set[int]],
    es=None,
    force: bool = False,
) -> List[dict]:
    root = col["root"]
    year_fmt = col["year_fmt"]
    log.info("Collection: %s  root=%s", collection_key, root)

    subdirs = _list_subdirs(fs_client, root)
    if not subdirs:
        log.warning("  No subdirectories found under %s", root)
        return []

    results = []
    for subdir in sorted(subdirs):
        # Parse year from subdir name
        try:
            if year_fmt == "year={Y}":
                if not subdir.startswith("year="):
                    continue
                year = int(subdir[len("year="):])
            else:
                year = int(subdir)
        except ValueError:
            continue

        if years_filter and year not in years_filter:
            continue

        year_dir = f"{root}/{subdir}"
        inv = build_year_inventory(
            fs_client, collection_key, year, year_dir, col, es=es, force=force
        )
        if inv:
            results.append(inv)

    return results


# ---------------------------------------------------------------------------
# Summary printer (reads existing inventories, no ADLS walk)
# ---------------------------------------------------------------------------

def print_summary(fs_client, collections_filter: Optional[Set[str]] = None) -> None:
    print("\n{:<15} {:>6} {:>10} {:>12}  {}".format(
        "COLLECTION", "YEAR", "FILES", "ES_UPLOADED", "BUILT_AT"
    ))
    print("-" * 72)

    for ckey, col in sorted(COLLECTIONS.items()):
        if collections_filter and ckey not in collections_filter:
            continue
        root = col["root"]
        subdirs = _list_subdirs(fs_client, root)
        for subdir in sorted(subdirs):
            try:
                if col["year_fmt"] == "year={Y}":
                    if not subdir.startswith("year="):
                        continue
                    year = int(subdir[len("year="):])
                else:
                    year = int(subdir)
            except ValueError:
                continue

            inv_path = f"{root}/{subdir}/{INVENTORY_FILENAME}"
            inv = _read_json(fs_client, inv_path)
            if inv:
                es_str = {True: "yes", False: "no", None: "?"}[inv.get("es_uploaded")]
                built = inv.get("built_at", "")[:19]
                print("{:<15} {:>6} {:>10,} {:>12}  {}".format(
                    ckey, year, inv.get("file_count", 0), es_str, built
                ))
            else:
                print("{:<15} {:>6} {:>10}  {}".format(ckey, year, "NO_INV", ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-year ADLS inventory files")
    ap.add_argument(
        "--collections", nargs="+",
        choices=list(COLLECTIONS.keys()),
        default=list(COLLECTIONS.keys()),
        help="Which collections to process (default: all)",
    )
    ap.add_argument(
        "--years", nargs="+", type=int, default=None,
        metavar="YEAR",
        help="Limit to specific years (default: all years found in ADLS)",
    )
    ap.add_argument(
        "--check-es", action="store_true",
        help="Check ES upload coverage for processed collections (requires ES_URL / ES_API_KEY)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild inventory even if _inventory.json already exists",
    )
    ap.add_argument(
        "--summary", action="store_true",
        help="Print summary of existing inventories (no ADLS walk, fast)",
    )
    args = ap.parse_args()

    fs_client = _build_fs_client()

    if args.summary:
        col_filter = set(args.collections) if args.collections != list(COLLECTIONS.keys()) else None
        print_summary(fs_client, col_filter)
        return

    es = _build_es_client() if args.check_es else None
    years_filter = set(args.years) if args.years else None

    all_results = {}
    for ckey in args.collections:
        col = COLLECTIONS[ckey]
        results = build_collection(
            fs_client, ckey, col,
            years_filter=years_filter,
            es=es,
            force=args.force,
        )
        all_results[ckey] = results

    # Final summary
    print("\n=== Inventory Build Summary ===")
    for ckey, results in all_results.items():
        total_files = sum(r.get("file_count", 0) for r in results)
        years_done = sorted(r["year"] for r in results)
        print(f"  {ckey}: {len(results)} years  {total_files:,} files  years={years_done}")



if __name__ == "__main__":
    main()

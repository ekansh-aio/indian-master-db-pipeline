"""
Unified ADLS path builder for all 4 collections: hc, sc, central_acts, state_acts.

Single source of truth for ADLS directory structure and ES index names.
Every pipeline module imports from here instead of hardcoding paths.
"""

import hashlib
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Collection definitions — each entry defines:
#   app / processed : ADLS root paths
#   es_index        : Elasticsearch index name
#   schema          : ES mapping schema file (relative to project root)
#   year_fmt        : format string for year subdirectory (e.g. "year={year}" or "{year}")
#   doc_id_source   : how doc_id is derived — "filename_stem" (HC/SC) or "from_data" (acts)
# ---------------------------------------------------------------------------

COLLECTION_CONFIG = {
    "hc": {
        "app": "app/High_Court_Judgements",
        "processed": "processed/High_Court_Judgements",
        "es_index": "hc_judgements",
        "schema": "schemas/hc.json",
        "year_fmt": "year={year}",
        "doc_id_source": "filename_stem",
    },
    "sc": {
        "app": "app/Supreme_Court_Judgements",
        "processed": "processed/Supreme_Court_Judgements",
        "es_index": "sc_judgements",
        "schema": "schemas/sc.json",
        "year_fmt": "{year}",        # SC uses bare year dirs (1950/), not year=1950/
        "doc_id_source": "filename_stem",
    },
    "central_acts": {
        "app": "app/central_acts",           # lowercase — actual ADLS casing
        "processed": "processed/central_acts",
        "es_index": "central_acts",
        "schema": "schemas/central_acts.json",
        "year_fmt": "{year}",
        "doc_id_source": "from_data",
    },
    "state_acts": {
        "app": "state_acts/app",             # separate root prefix on ADLS
        "processed": "state_acts/processed",
        "es_index": "state_acts",
        "schema": "schemas/state_acts.json",
        "year_fmt": "year={year}",
        "doc_id_source": "from_data",
    },
}

VALID_COLLECTIONS = set(COLLECTION_CONFIG.keys())
VALID_ACT_COLLECTIONS = {"central_acts", "state_acts"}
VALID_JUDGMENT_COLLECTIONS = {"hc", "sc"}


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def get_app_root(collection: str) -> str:
    return COLLECTION_CONFIG[collection]["app"]


def get_processed_root(collection: str) -> str:
    return COLLECTION_CONFIG[collection]["processed"]


def get_es_index(collection: str) -> str:
    return COLLECTION_CONFIG[collection]["es_index"]


def get_schema_path(collection: str) -> str:
    return COLLECTION_CONFIG[collection]["schema"]


def _year_subdir(collection: str, year: Optional[int] = None) -> str:
    fmt = COLLECTION_CONFIG[collection]["year_fmt"]
    return fmt.format(year=year) if year is not None else ""


def get_app_path(collection: str, year: int, state: Optional[str] = None) -> str:
    base = get_app_root(collection)
    if collection == "state_acts" and state:
        return f"{base}/state={state}/{_year_subdir(collection, year)}"
    return f"{base}/{_year_subdir(collection, year)}"


def get_processed_path(collection: str, year: Optional[int] = None, state: Optional[str] = None) -> str:
    base = get_processed_root(collection)
    if collection == "state_acts" and state:
        return f"{base}/state={state}/{_year_subdir(collection, year)}" if year is not None else f"{base}/state={state}"
    return f"{base}/{_year_subdir(collection, year)}"


def get_inventory_path(collection: str, year: int, state: Optional[str] = None) -> str:
    return f"{get_processed_path(collection, year, state)}/_inventory.json"


def get_es_inventory_path(collection: str, year: Optional[int] = None, state: Optional[str] = None) -> str:
    if collection == "state_acts" and year is None and state:
        base = get_processed_root(collection)
        return f"{base}/state={state}/_inventory_es.json"
    return f"{get_processed_path(collection, year, state)}/_inventory_es.json"


# ---------------------------------------------------------------------------
# doc_id derivation
# ---------------------------------------------------------------------------

def generate_doc_id(source_file_path: str, collection: str) -> str:
    """Generate a stable doc_id from an ADLS source file path.

    For HC/SC: strip path prefixes, join remaining parts with underscores.
    For acts: doc_id is embedded in the data (process.py generates it).
    This function is used by the judgment pipelines.
    """
    path_without_ext = source_file_path.replace(".json", "")
    parts = Path(path_without_ext).parts
    skip = {"raw", "newapp", "input", "data", "app",
            "high_court_judgements", "supreme_court_judgements",
            "central_acts", "state_acts"}
    relevant = [p for p in parts if p.lower() not in skip]
    doc_id = "_".join(relevant).replace("-", "_").replace(" ", "_").lower()
    if len(doc_id) > 200:
        base = relevant[-1] if relevant else "doc"
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    return doc_id


def doc_id_from_path(adls_path: str) -> str:
    """Extract doc_id from an ADLS path by parsing the filename stem."""
    name = Path(adls_path).stem
    if name.endswith("_all_chunks"):
        return name[: -len("_all_chunks")]
    if name.endswith("_done_0"):
        return name[: -len("_done_0")]
    return name


# ---------------------------------------------------------------------------
# Chunks / marker path builders (used by judgment pipeline)
# ---------------------------------------------------------------------------

def get_all_chunks_path(source_file_path: str, collection: str) -> str:
    """Build the processed/ path for the *_all_chunks.json file.

    Derives the relative path structure by stripping known prefix directories,
    then prepends the collection's processed root.
    """
    base = get_processed_root(collection)
    path_without_ext = source_file_path.replace(".json", "")
    parts = Path(path_without_ext).parts
    skip = {"raw", "newapp", "input", "data", "app",
            "high_court_judgements", "supreme_court_judgements",
            "central_acts", "state_acts"}
    relevant = [p for p in parts if p.lower() not in skip]
    if relevant:
        dir_parts = relevant[:-1]
        fname = relevant[-1]
        if dir_parts:
            return f"{base}/{'/'.join(dir_parts)}/{fname}_all_chunks.json"
        return f"{base}/{fname}_all_chunks.json"
    return f"{base}/unknown_all_chunks.json"


def get_done_marker_path(source_file_path: str, collection: str) -> str:
    """Build the done_0 marker path (marks successful ADLS processed/ upload)."""
    base = get_processed_root(collection)
    path_without_ext = source_file_path.replace(".json", "")
    parts = Path(path_without_ext).parts
    skip = {"raw", "newapp", "input", "data", "app",
            "high_court_judgements", "supreme_court_judgements",
            "central_acts", "state_acts"}
    relevant = [p for p in parts if p.lower() not in skip]
    if relevant:
        dir_parts = relevant[:-1]
        fname = relevant[-1]
        if dir_parts:
            return f"{base}/{'/'.join(dir_parts)}/{fname}_done_0.json"
        return f"{base}/{fname}_done_0.json"
    return f"{base}/unknown_done_0.json"

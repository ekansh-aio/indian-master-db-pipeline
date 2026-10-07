"""
Legal Dictionary — migrate Azure AI Search index → Elasticsearch.

Fetches all docs from AI Search (legal-dictionary-index-india), re-embeds
definition text with sentence-transformers, and bulk-uploads to ES.

Usage:
    python pipelines/legal_dictionary/run.py
    python pipelines/legal_dictionary/run.py --recreate-index
    python pipelines/legal_dictionary/run.py --no-resume

Environment:
    ES_URL / ES_API_KEY (or ES_USER/ES_PASS)
    SEARCH_ENDPOINT / SEARCH_KEY
    EMBEDDING_MODEL  (default: sentence-transformers/all-MiniLM-L6-v2)
    EMBEDDING_BATCH_SIZE, ES_BULK_BATCH_SIZE, ES_MAX_RETRIES, ES_RETRY_DELAY
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from sentence_transformers import SentenceTransformer

from config import LOGGING_CONFIG
from core.elasticsearch_client import build_es_client, ensure_index

log = logging.getLogger(__name__)

INDEX_NAME = "legal_dictionary"
SCHEMA_PATH = str(Path(__file__).resolve().parent.parent.parent / "schemas" / "legal_dictionary.json")

PROGRESS_DIR = Path("pipeline_progress")
PROGRESS_FILE = PROGRESS_DIR / "done_ids_legal_dictionary.jsonl"

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE", "200"))
ES_MAX_RETRIES = int(os.getenv("ES_MAX_RETRIES", "3"))
ES_RETRY_DELAY = float(os.getenv("ES_RETRY_DELAY", "2.0"))
AI_SEARCH_PAGE = 1000


def _build_es_client():
    return build_es_client()


def _ensure_index(es: Elasticsearch, recreate: bool) -> None:
    ensure_index(es, INDEX_NAME, SCHEMA_PATH, recreate=recreate)


def bulk_upload_with_retry(es: Elasticsearch, docs: List[Dict]) -> Tuple[int, int, List[str]]:
    """Upload docs to ES with retries. Returns (uploaded_ok, upload_errs, succeeded_ids)."""
    remaining = list(docs)
    total_ok = 0

    for attempt in range(ES_MAX_RETRIES):
        def _actions(d):
            for doc in d:
                yield {"_index": INDEX_NAME, "_id": doc["id"], "_source": doc}

        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=BULK_BATCH_SIZE)
        total_ok += ok

        if not errors:
            remaining = []
            break

        failed_ids = {(e.get("index") or e.get("create") or {}).get("_id") for e in errors} - {None}
        remaining = [d for d in remaining if d["id"] in failed_ids]
        if not remaining:
            break

        log.warning("Retrying %d failed docs (attempt %d/%d)", len(remaining), attempt + 2, ES_MAX_RETRIES)
        time.sleep(ES_RETRY_DELAY * (2 ** attempt))

    if remaining:
        log.error("%d docs permanently failed after %d retries", len(remaining), ES_MAX_RETRIES)

    failed_ids_set = {r["id"] for r in remaining}
    succeeded_ids = [d["id"] for d in docs if d["id"] not in failed_ids_set]
    return total_ok, len(remaining), succeeded_ids


def fetch_all_docs_from_search() -> List[Dict]:
    endpoint = os.getenv("SEARCH_ENDPOINT", "").rstrip("/")
    key = os.getenv("SEARCH_KEY", "")
    if not endpoint or not key:
        log.error("SEARCH_ENDPOINT or SEARCH_KEY not set in .env")
        sys.exit(1)

    url = f"{endpoint}/indexes/legal-dictionary-index-india/docs/search?api-version=2024-07-01"
    headers = {"api-key": key, "Content-Type": "application/json"}

    all_docs: List[Dict] = []
    skip = 0

    while True:
        payload = {
            "search": "*",
            "top": AI_SEARCH_PAGE,
            "skip": skip,
            "select": "id,term,part_of_speech,definition",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("value", [])
        if not batch:
            break
        all_docs.extend(batch)
        log.info("Fetched %d docs from AI Search (total: %d)", len(batch), len(all_docs))
        if len(batch) < AI_SEARCH_PAGE:
            break
        skip += AI_SEARCH_PAGE

    log.info("Total docs fetched from AI Search: %d", len(all_docs))
    return all_docs


def load_done_ids() -> set:
    PROGRESS_DIR.mkdir(exist_ok=True)
    if not PROGRESS_FILE.exists():
        return set()
    ids = set()
    for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line))
            except json.JSONDecodeError:
                ids.add(line.strip('"'))
    log.info("Resuming: %d done IDs loaded", len(ids))
    return ids


def append_done_ids(new_ids: List[str]) -> None:
    if not new_ids:
        return
    PROGRESS_DIR.mkdir(exist_ok=True)
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        for id_ in new_ids:
            f.write(json.dumps(id_) + "\n")


def run(recreate_index: bool, no_resume: bool) -> None:
    es = _build_es_client()
    _ensure_index(es, recreate=recreate_index)

    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    log.info("Model loaded")

    all_docs = fetch_all_docs_from_search()

    done_ids = set() if no_resume else load_done_ids()
    pending = [d for d in all_docs if d["id"] not in done_ids]
    log.info("Docs to upload: %d (already done: %d)", len(pending), len(all_docs) - len(pending))

    if not pending:
        log.info("Nothing to do.")
        return

    t_start = time.time()
    uploaded_ok = 0
    upload_errors = 0

    for batch_start in range(0, len(pending), EMBED_BATCH):
        batch = pending[batch_start: batch_start + EMBED_BATCH]
        texts = [d.get("definition") or "" for d in batch]

        embeddings = model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=False, normalize_embeddings=True)

        es_docs = []
        for doc, emb in zip(batch, embeddings):
            es_docs.append({
                "id": doc["id"],
                "term": doc.get("term", ""),
                "part_of_speech": doc.get("part_of_speech", ""),
                "definition": doc.get("definition", ""),
                "embedding": emb.tolist(),
            })

        ok, errs, succeeded_ids = bulk_upload_with_retry(es, es_docs)
        uploaded_ok += ok
        upload_errors += errs
        append_done_ids(succeeded_ids)

        elapsed = time.time() - t_start
        log.info("Progress: %d/%d  uploaded=%d  errors=%d  elapsed=%.0fs",
                 min(batch_start + EMBED_BATCH, len(pending)), len(pending),
                 uploaded_ok, upload_errors, elapsed)

    log.info("=" * 60)
    log.info("DONE — %s  uploaded=%d  errors=%d  elapsed=%.0fs",
             INDEX_NAME, uploaded_ok, upload_errors, time.time() - t_start)


def main():
    ap = argparse.ArgumentParser(
        description="Migrate legal dictionary from AI Search to Elasticsearch"
    )
    ap.add_argument("--recreate-index", action="store_true",
                    help="Drop and recreate the ES index before uploading")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-upload everything ignoring local progress")
    args = ap.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    level = getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt,
                        handlers=[logging.StreamHandler()], force=True)

    run(recreate_index=args.recreate_index, no_resume=args.no_resume)


if __name__ == "__main__":
    main()

"""
Migrate legal-dictionary-index-india (Azure AI Search) → Elasticsearch.

Fetches all 5,014 docs from AI Search (embedding not retrievable), re-embeds
definition text with all-MiniLM-L6-v2 (384-dim cosine), and bulk-uploads to
a new ES index "legal_dictionary".

Run:
  python pipeline/legal_dict_to_es.py [--recreate-index] [--no-resume]

Environment (.env):
  ES_URL, ES_API_KEY
  SEARCH_ENDPOINT, SEARCH_KEY
  EMBEDDING_MODEL  (default: sentence-transformers/all-MiniLM-L6-v2)
  ES_BULK_BATCH_SIZE, ES_MAX_RETRIES, ES_RETRY_DELAY
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("legal_dict_to_es.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("legal_dict_to_es")
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

INDEX_NAME      = "legal_dictionary"
SCHEMA_PATH     = Path(__file__).resolve().parent.parent / "sample" / "es_schema_legal_dictionary.json"
PROGRESS_DIR    = Path("pipeline_progress")
PROGRESS_FILE   = PROGRESS_DIR / "done_ids_legal_dictionary.jsonl"

EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH     = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE", "200"))
ES_MAX_RETRIES  = int(os.getenv("ES_MAX_RETRIES", "3"))
ES_RETRY_DELAY  = float(os.getenv("ES_RETRY_DELAY", "2.0"))
AI_SEARCH_PAGE  = 1000  # AI Search max $top per request


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------

def build_es_client() -> Elasticsearch:
    es_url = os.getenv("ES_URL")
    if not es_url:
        log.error("ES_URL not set in .env")
        sys.exit(1)
    api_key = os.getenv("ES_API_KEY")
    if api_key:
        es = Elasticsearch(es_url, api_key=api_key, request_timeout=120)
    else:
        user, password = os.getenv("ES_USER"), os.getenv("ES_PASS")
        es = Elasticsearch(es_url, http_auth=(user, password), request_timeout=120) \
            if (user and password) else Elasticsearch(es_url, request_timeout=120)
    if not es.ping():
        log.error("Cannot connect to Elasticsearch — check ES_URL / credentials")
        sys.exit(1)
    log.info("Elasticsearch connected: %s", es_url)
    return es


def ensure_index(es: Elasticsearch, recreate: bool) -> None:
    exists = es.indices.exists(index=INDEX_NAME)
    if exists and recreate:
        es.indices.delete(index=INDEX_NAME)
        log.info("Dropped index: %s", INDEX_NAME)
        exists = False
    if not exists:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        es.indices.create(index=INDEX_NAME, body=schema)
        log.info("Created index: %s", INDEX_NAME)
    else:
        log.info("Index already exists: %s (skipping create)", INDEX_NAME)


def bulk_upload_with_retry(es: Elasticsearch, docs: List[Dict]) -> Tuple[int, int]:
    remaining = list(docs)
    total_ok = 0
    last_errors: list = []

    for attempt in range(ES_MAX_RETRIES):
        def _actions(d):
            for doc in d:
                yield {"_index": INDEX_NAME, "_id": doc["id"], "_source": doc}

        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=BULK_BATCH_SIZE)
        total_ok += ok
        last_errors = errors

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

    return total_ok, len(remaining)


# ---------------------------------------------------------------------------
# Azure AI Search fetcher
# ---------------------------------------------------------------------------

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
        log.info("Fetched %d docs from AI Search (total so far: %d)", len(batch), len(all_docs))
        if len(batch) < AI_SEARCH_PAGE:
            break
        skip += AI_SEARCH_PAGE

    log.info("Total docs fetched from AI Search: %d", len(all_docs))
    return all_docs


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(recreate_index: bool, no_resume: bool) -> None:
    es = build_es_client()
    ensure_index(es, recreate=recreate_index)

    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    log.info("Model loaded")

    all_docs = fetch_all_docs_from_search()

    done_ids = set() if no_resume else load_done_ids()
    pending = [d for d in all_docs if d["id"] not in done_ids]
    log.info("Docs to upload: %d  (skipped already done: %d)", len(pending), len(all_docs) - len(pending))

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
                "id":            doc["id"],
                "term":          doc.get("term", ""),
                "part_of_speech": doc.get("part_of_speech", ""),
                "definition":    doc.get("definition", ""),
                "embedding":     emb.tolist(),
            })

        ok, errs = bulk_upload_with_retry(es, es_docs)
        uploaded_ok += ok
        upload_errors += errs
        append_done_ids([d["id"] for d in es_docs[:ok]])

        elapsed = time.time() - t_start
        log.info(
            "Progress: %d/%d  uploaded=%d  errors=%d  elapsed=%.0fs",
            min(batch_start + EMBED_BATCH, len(pending)), len(pending),
            uploaded_ok, upload_errors, elapsed,
        )

    log.info("=" * 60)
    log.info("DONE — %s  uploaded=%d  errors=%d  elapsed=%.0fs",
             INDEX_NAME, uploaded_ok, upload_errors, time.time() - t_start)


def main():
    ap = argparse.ArgumentParser(description="Migrate legal dictionary from AI Search to Elasticsearch")
    ap.add_argument("--recreate-index", action="store_true",
                    help="Drop and recreate the ES index before uploading")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-upload everything, ignoring local progress")
    args = ap.parse_args()
    run(recreate_index=args.recreate_index, no_resume=args.no_resume)


if __name__ == "__main__":
    main()

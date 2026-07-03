"""
Upload 1000 randomly sampled HC all_chunks files from ADLS to Elasticsearch
as parent-document JSON. Drops and recreates hc_judgements on each run.

Output: logs doc count + index store size on completion.
"""
import os
import sys
import json
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk

from core.adls_fetcher import ADLSFetcher
from utils.weighted_selector import weighted_topk_selection
from config import ADLS_CONFIG, EMBEDDING_CONFIG, DOC_TYPE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_SIZE = 1000
TOP_K = 15
SCAN_LIMIT = 15_000
BULK_BATCH_SIZE = 50
READ_WORKERS = 16
INDEX_NAME = "hc_judgements"
SCHEMA_PATH = Path(__file__).parent / "es_schema_int8.json"

HC_PROCESSED_PATH = "processed/" + DOC_TYPE_CONFIG[0]["adls_input_path"].lstrip("app/").rstrip("/")

METADATA_FIELDS = [
    "doc_id", "date", "jurisdiction", "doc_name", "year", "bench",
    "court_code", "title", "judge", "pdf_link", "cnr",
    "date_of_registration", "decision_date", "disposal_nature",
    "court", "pdf_exists", "original_source_path", "all_chunks_path",
    "source_file",
]

CHUNK_FIELDS = ["chunk_id", "text", "role", "same_role_chunk_ids"]


def _reservoir_sample(iterator, k, limit=None):
    reservoir = []
    for i, item in enumerate(iterator):
        if limit and i >= limit:
            break
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def extract_metadata(chunk):
    return {k: chunk[k] for k in METADATA_FIELDS if k in chunk}


def build_parent_doc(metadata, slim_chunks):
    doc = dict(metadata)
    doc["chunks"] = slim_chunks
    return doc


def recreate_index(es, index_name, schema_path):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        log.info(f"Dropped existing index: {index_name}")
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    es.indices.create(index=index_name, body=schema)
    log.info(f"Created index: {index_name}")


def _actions(parent_docs, index_name):
    for doc in parent_docs:
        yield {"_index": index_name, "_id": doc.get("doc_id"), "_source": doc}


def main():
    es_url = os.getenv("ES_URL")
    es_api_key = os.getenv("ES_API_KEY")

    if not es_url:
        log.error("ES_URL not set in .env")
        sys.exit(1)

    es = Elasticsearch(es_url, api_key=es_api_key) if es_api_key else Elasticsearch(es_url)
    if not es.ping():
        log.error("Cannot connect to Elasticsearch — check ES_URL and ES_API_KEY")
        sys.exit(1)
    log.info("Elasticsearch connected")

    account_name = ADLS_CONFIG["account_name"]
    account_key = ADLS_CONFIG["account_key"]
    container = ADLS_CONFIG["container_name"]
    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing — check .env")
        sys.exit(1)

    fetcher = ADLSFetcher(account_name, account_key, container)

    log.info(f"Reservoir-sampling {SAMPLE_SIZE} files from first {SCAN_LIMIT} under {HC_PROCESSED_PATH} ...")
    paths = _reservoir_sample(
        fetcher.list_files_iter(path=HC_PROCESSED_PATH, pattern="*_all_chunks.json", recursive=True),
        SAMPLE_SIZE,
        limit=SCAN_LIMIT,
    )
    if not paths:
        log.error(f"No files found under {HC_PROCESSED_PATH}")
        sys.exit(1)
    log.info(f"Sampled {len(paths)} files")

    def _read_file(p):
        chunks = fetcher.read_json_file(p)
        if isinstance(chunks, list) and chunks:
            return (p, chunks)
        return None

    raw_docs = []
    done = 0
    log.info(f"Reading {len(paths)} files with {READ_WORKERS} threads ...")
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        futures = {pool.submit(_read_file, p): p for p in paths}
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
                if result:
                    raw_docs.append(result)
                else:
                    log.warning(f"Skipping {futures[fut]}: empty or not a list")
            except Exception as e:
                log.warning(f"Skipping {futures[fut]}: {e}")
            if done % 100 == 0:
                log.info(f"  {done}/{len(paths)} read, {len(raw_docs)} ok ...")
    log.info(f"Successfully read {len(raw_docs)}/{len(paths)} docs")

    selected_per_doc = []
    for path, chunks in raw_docs:
        indices = weighted_topk_selection(chunks, top_k=TOP_K, similarity_key="doc_similarity", role_weights={})
        selected = [chunks[i] for i in indices]
        selected_per_doc.append((path, chunks[0], selected))

    log.info("Loading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(EMBEDDING_CONFIG["model_name"])

    all_texts = []
    chunk_map = []
    for doc_idx, (_, _, selected) in enumerate(selected_per_doc):
        for chunk_idx, chunk in enumerate(selected):
            all_texts.append(chunk["text"])
            chunk_map.append((doc_idx, chunk_idx))

    cpu_batch_size = 64
    log.info(f"Encoding {len(all_texts)} chunks (model: {EMBEDDING_CONFIG['model_name']}, batch={cpu_batch_size}) ...")
    embeddings = model.encode(
        all_texts,
        batch_size=cpu_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    for flat_idx, (doc_idx, chunk_idx) in enumerate(chunk_map):
        selected_per_doc[doc_idx][2][chunk_idx]["embedding"] = embeddings[flat_idx].tolist()

    parent_docs = []
    for adls_path, first_chunk, selected in selected_per_doc:
        metadata = extract_metadata(first_chunk)
        if "all_chunks_path" not in metadata:
            metadata["all_chunks_path"] = adls_path
        slim_chunks = []
        for chunk in selected:
            slim = {k: chunk[k] for k in CHUNK_FIELDS if k in chunk}
            slim["embedding"] = chunk["embedding"]
            slim_chunks.append(slim)
        parent_docs.append(build_parent_doc(metadata, slim_chunks))

    recreate_index(es, INDEX_NAME, SCHEMA_PATH)

    log.info(f"Uploading {len(parent_docs)} docs in batches of {BULK_BATCH_SIZE} ...")
    total_ok = 0
    for i in range(0, len(parent_docs), BULK_BATCH_SIZE):
        batch = parent_docs[i : i + BULK_BATCH_SIZE]
        ok, errors = es_bulk(es, _actions(batch, INDEX_NAME), raise_on_error=False)
        total_ok += ok
        if errors:
            log.warning(f"  Batch {i // BULK_BATCH_SIZE + 1}: {len(errors)} errors — {errors[0]}")
        log.info(f"  {total_ok}/{len(parent_docs)} uploaded")

    es.indices.refresh(index=INDEX_NAME)
    stats = es.indices.stats(index=INDEX_NAME, metric="store")
    size_bytes = stats["indices"][INDEX_NAME]["primaries"]["store"]["size_in_bytes"]
    size_mb = size_bytes / (1024 * 1024)
    count = es.count(index=INDEX_NAME)["count"]

    log.info("=" * 50)
    log.info(f"Index: {INDEX_NAME}")
    log.info(f"Docs:  {count}")
    log.info(f"Size:  {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    if count:
        log.info(f"Avg:   {size_mb / count * 1024:.1f} KB/doc")
    log.info("=" * 50)


if __name__ == "__main__":
    main()

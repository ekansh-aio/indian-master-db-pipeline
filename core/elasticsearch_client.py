"""
Elasticsearch and embedding helpers shared across all pipelines.

Consolidates duplicate ES client builders, bulk upload with retry,
mget pre-filter, index management, and GPU embedding logic from:
  - adls_to_es_pipeline.py
  - app_to_es_pipeline.py
  - state_acts_to_es.py
  - central_acts_to_es.py
  - legal_dict_to_es.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ES client
# ---------------------------------------------------------------------------


def build_es_client(timeout: int = 120) -> Elasticsearch:
    """Build an Elasticsearch client from .env settings.

    Prefers ES_API_KEY; falls back to ES_USER/ES_PASS basic auth.
    Exits with error if ES_URL is missing or unreachable.
    """
    es_url = os.getenv("ES_URL")
    if not es_url:
        log.error("ES_URL not set in .env")
        sys.exit(1)

    api_key = os.getenv("ES_API_KEY")
    if api_key:
        es = Elasticsearch(es_url, api_key=api_key, request_timeout=timeout)
    else:
        user = os.getenv("ES_USER")
        password = os.getenv("ES_PASS")
        if user and password:
            es = Elasticsearch(es_url, http_auth=(user, password), request_timeout=timeout)
        else:
            es = Elasticsearch(es_url, request_timeout=timeout)

    if not es.ping():
        log.error("Cannot connect to Elasticsearch — check ES_URL / credentials")
        sys.exit(1)
    log.info("Elasticsearch connected: %s", es_url)
    return es


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def ensure_index(
    es: Elasticsearch,
    index_name: str,
    schema_path: str,
    recreate: bool = False,
    shards: int = 8,
    replicas: int = 1,
) -> None:
    """Create or verify an ES index with the given schema.

    Args:
        recreate: Drop and recreate if index exists (destructive).
        shards, replicas: Override schema settings if index is created.
    """
    exists = es.indices.exists(index=index_name)
    if exists and recreate:
        es.indices.delete(index=index_name)
        log.info("Dropped index: %s", index_name)
        exists = False

    if not exists:
        schema_file = Path(schema_path)
        if not schema_file.exists():
            log.error("Schema file not found: %s", schema_path)
            sys.exit(1)
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        schema.setdefault("settings", {})
        schema["settings"]["number_of_shards"] = schema["settings"].get("number_of_shards", shards)
        schema["settings"]["number_of_replicas"] = schema["settings"].get("number_of_replicas", replicas)
        es.indices.create(index=index_name, body=schema)
        log.info("Created index: %s  (shards=%s, replicas=%s)",
                 index_name,
                 schema["settings"]["number_of_shards"],
                 schema["settings"]["number_of_replicas"])
    else:
        log.info("Index already exists: %s", index_name)


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------


def bulk_mget_exists(es: Elasticsearch, index_name: str, doc_ids: List[str]) -> set:
    """Single mget call to check which doc_ids exist in the index. Returns set of found IDs."""
    if not doc_ids:
        return set()
    resp = es.mget(index=index_name, body={"ids": doc_ids}, _source=False)
    return {d["_id"] for d in resp["docs"] if d.get("found")}


def bulk_upload_with_retry(
    es: Elasticsearch,
    index_name: str,
    parent_docs: List[Dict],
    bulk_batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    id_field: str = "doc_id",
) -> Tuple[int, int]:
    """Bulk upload with exponential-backoff retry. Retries only the failed subset.

    Args:
        parent_docs: List of document dicts. Each must have the id_field key.
        id_field: Field name to use as ES _id (default: "doc_id").

    Returns:
        (total_ok, total_failed)
    """
    remaining = list(parent_docs)
    total_ok = 0
    last_errors: list = []

    for attempt in range(max_retries):
        def _actions(docs):
            for doc in docs:
                yield {"_index": index_name, "_id": doc.get(id_field), "_source": doc}

        ok, errors = es_bulk(es, _actions(remaining), raise_on_error=False, chunk_size=bulk_batch_size)
        total_ok += ok
        last_errors = errors

        if not errors:
            remaining = []
            break

        failed_ids: set = set()
        for e in errors:
            op = e.get("index") or e.get("create") or {}
            if "_id" in op:
                failed_ids.add(op["_id"])

        if not failed_ids:
            # Cluster-level error: no per-doc _id extractable — treat all remaining as failed
            log.error(
                "  ES bulk: %d errors but no per-doc _id found — "
                "%d docs may not be indexed",
                len(errors), len(remaining),
            )
            break

        remaining = [d for d in remaining if d.get(id_field) in failed_ids]
        if not remaining:
            break

        log.warning("  Retrying %d failed docs (attempt %d/%d)", len(remaining), attempt + 2, max_retries)
        time.sleep(retry_delay * (2 ** attempt))

    total_failed = len(remaining)
    if last_errors and remaining:
        log.error("  %d docs permanently failed after %d retries", total_failed, max_retries)

    return total_ok, total_failed


# ---------------------------------------------------------------------------
# Index stats
# ---------------------------------------------------------------------------


def log_index_stats(es: Elasticsearch, index_name: str) -> None:
    """Log doc count and storage size for an index."""
    try:
        es.indices.refresh(index=index_name)
        stats = es.indices.stats(index=index_name, metric="store")
        count = es.count(index=index_name)["count"]
        size_bytes = stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"]
        size_gb = size_bytes / (1024 ** 3)
        log.info("=" * 60)
        log.info("Index : %s", index_name)
        log.info("Docs  : %s", f"{count:,}")
        log.info("Size  : %.2f GB  (%s bytes)", size_gb, f"{size_bytes:,}")
        if count:
            log.info("Avg   : %.1f KB/doc", size_bytes / count / 1024)
        log.info("=" * 60)
    except Exception as e:
        log.warning("Could not fetch index stats: %s", e)


# ---------------------------------------------------------------------------
# GPU embedding
# ---------------------------------------------------------------------------


def load_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer model on the best available device.

    Falls back: cuda → cpu. Multi-GPU requires explicit start_multi_process_pool.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if device is None:
        n_gpu = torch.cuda.device_count()
        device = "cuda" if n_gpu > 0 else "cpu"

    log.info("Loading embedding model: %s  (device=%s)", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    log.info("Model loaded")
    return model


def load_model_multi_gpu(
    model_name: Optional[str] = None,
) -> Tuple[SentenceTransformer, Optional[object]]:
    """Load model and optionally start a multi-process pool across all GPUs.

    Returns:
        (model, pool) where pool is None if single-GPU or CPU.
    """
    model = load_embedding_model(model_name)

    n_gpu = torch.cuda.device_count()
    pool = None
    if n_gpu > 1:
        target_devices = [f"cuda:{i}" for i in range(n_gpu)]
        log.info("Starting multi-GPU pool on: %s", target_devices)
        pool = model.start_multi_process_pool(target_devices=target_devices)
    elif n_gpu == 0:
        log.warning("No CUDA devices found — running on CPU (expect slow embedding)")

    return model, pool


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    pool: Optional[object] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
):
    """Embed a list of texts using the given model.

    Uses multi-process pool when available, otherwise single-device encode.
    """
    import numpy as np

    if batch_size is None:
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "2048"))

    if pool is not None:
        embeddings = model.encode_multi_process(
            texts, pool,
            batch_size=batch_size,
            show_progress_bar=show_progress and len(texts) > 1000,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress and len(texts) > 1000,
            convert_to_numpy=True,
        )

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    return embeddings

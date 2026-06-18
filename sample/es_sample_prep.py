"""
Fetch 20 randomly sampled HC all_chunks files from ADLS, select top-15
per doc, embed, and write as parent-document JSON ready for Elasticsearch
evaluation.

Output: sample/output/es_sample_20docs.json
"""
import os
import sys
import json
import random
import logging
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.adls_fetcher import ADLSFetcher
from utils.weighted_selector import weighted_topk_selection
from config import ADLS_CONFIG, EMBEDDING_CONFIG, DOC_TYPE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SAMPLE_SIZE = 20
TOP_K = 15
RANDOM_SEED = None  # None = different sample each run; set an int for reproducibility
SCAN_LIMIT = 5_000  # scan only this many files before sampling — fast, good enough randomness
# HC processed path — mirrors how the pipeline writes output
HC_PROCESSED_PATH = "processed/" + DOC_TYPE_CONFIG[0]["adls_input_path"].lstrip("app/").rstrip("/")
# Falls back to scanning all of processed/ if the HC sub-path doesn't exist
HC_FALLBACK_PATH = "processed"
OUTPUT_PATH = Path(__file__).parent / "output" / "es_sample_20docs.json"

# Fields that are identical across all chunks of a doc — hoisted to top level
METADATA_FIELDS = [
    "doc_id", "date", "jurisdiction", "doc_name", "year", "bench",
    "court_code", "title", "judge", "pdf_link", "cnr",
    "date_of_registration", "decision_date", "disposal_nature",
    "court", "pdf_exists", "original_source_path", "all_chunks_path",
    # SC-specific (may be absent in HC docs)
    "source_file",
]

# Fields kept per chunk (search-relevant only, noise stripped)
CHUNK_FIELDS = ["chunk_id", "text", "role", "same_role_chunk_ids"]
# embedding added after encoding


def _reservoir_sample(iterator, k: int, limit: int = None) -> list:
    """Random sample of k items from an iterator, stopping after `limit` items scanned."""
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


def extract_metadata(chunk: dict) -> dict:
    return {k: chunk[k] for k in METADATA_FIELDS if k in chunk}


def build_parent_doc(metadata: dict, selected_chunks: list) -> dict:
    doc = dict(metadata)
    doc["chunks"] = selected_chunks
    return doc


def main():
    account_name = ADLS_CONFIG["account_name"]
    account_key = ADLS_CONFIG["account_key"]
    container = ADLS_CONFIG["container_name"]

    if not all([account_name, account_key, container]):
        log.error("ADLS credentials missing — set ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY, ADLS_CONTAINER_NAME")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: List all HC all_chunks files, then random-sample 20
    # ------------------------------------------------------------------
    log.info("Connecting to ADLS...")
    fetcher = ADLSFetcher(account_name, account_key, container)

    log.info(f"Sampling {SAMPLE_SIZE} HC *_all_chunks.json from first {SCAN_LIMIT} files under {HC_PROCESSED_PATH} ...")
    paths = _reservoir_sample(
        fetcher.list_files_iter(path=HC_PROCESSED_PATH, pattern="*_all_chunks.json", recursive=True),
        SAMPLE_SIZE,
        limit=SCAN_LIMIT,
    )

    if not paths:
        log.warning(f"No files found under {HC_PROCESSED_PATH}, falling back to {HC_FALLBACK_PATH}")
        paths = _reservoir_sample(
            fetcher.list_files_iter(path=HC_FALLBACK_PATH, pattern="*_all_chunks.json", recursive=True),
            SAMPLE_SIZE,
            limit=SCAN_LIMIT,
        )

    if not paths:
        log.error("No all_chunks files found. Check ADLS path and credentials.")
        sys.exit(1)

    n = len(paths)
    log.info(f"Reservoir-sampled {n} files (seed={RANDOM_SEED})")
    for p in paths:
        log.info(f"  {p}")

    log.info("Reading sampled files...")
    raw_docs = []
    for p in paths:
        try:
            chunks = fetcher.read_json_file(p)
            if isinstance(chunks, list) and chunks:
                raw_docs.append((p, chunks))
            else:
                log.warning(f"Skipping {p}: empty or not a list")
        except Exception as e:
            log.warning(f"Skipping {p}: {e}")

    log.info(f"Successfully read {len(raw_docs)}/{n} docs")

    # ------------------------------------------------------------------
    # Step 2: Select top-15 per doc
    # ------------------------------------------------------------------
    log.info(f"Selecting top-{TOP_K} chunks per doc (uniform weights)...")
    selected_per_doc = []
    for path, chunks in raw_docs:
        indices = weighted_topk_selection(chunks, top_k=TOP_K, similarity_key="doc_similarity", role_weights={})
        selected = [chunks[i] for i in indices]
        selected_per_doc.append((path, chunks[0], selected))  # (adls_path, first_chunk for metadata, selected)
        log.debug(f"  {Path(path).name}: {len(chunks)} chunks → {len(selected)} selected")

    # ------------------------------------------------------------------
    # Step 3: Embed all selected chunk texts in one batch
    # ------------------------------------------------------------------
    log.info("Loading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model_name = EMBEDDING_CONFIG["model_name"]
    batch_size = EMBEDDING_CONFIG["batch_size"]
    log.info(f"Model: {model_name}")

    model = SentenceTransformer(model_name)

    # Flatten all texts with a doc index so we can reassign embeddings
    all_texts = []
    chunk_map = []  # (doc_idx, chunk_idx_within_doc)
    for doc_idx, (_, _, selected) in enumerate(selected_per_doc):
        for chunk_idx, chunk in enumerate(selected):
            all_texts.append(chunk["text"])
            chunk_map.append((doc_idx, chunk_idx))

    log.info(f"Encoding {len(all_texts)} chunks in batches of {batch_size}...")
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Attach embeddings back
    for flat_idx, (doc_idx, chunk_idx) in enumerate(chunk_map):
        selected_per_doc[doc_idx][2][chunk_idx]["embedding"] = embeddings[flat_idx].tolist()

    # ------------------------------------------------------------------
    # Step 4: Build parent documents
    # ------------------------------------------------------------------
    log.info("Building parent-document structures...")
    parent_docs = []
    for adls_path, first_chunk, selected in selected_per_doc:
        metadata = extract_metadata(first_chunk)
        # all_chunks_path may not be in the chunk itself for older files — fall back to adls_path
        if "all_chunks_path" not in metadata:
            metadata["all_chunks_path"] = adls_path

        slim_chunks = []
        for chunk in selected:
            slim = {k: chunk[k] for k in CHUNK_FIELDS if k in chunk}
            slim["embedding"] = chunk["embedding"]
            slim_chunks.append(slim)

        parent_docs.append(build_parent_doc(metadata, slim_chunks))

    # ------------------------------------------------------------------
    # Step 5: Write output
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(parent_docs, f, ensure_ascii=False, indent=2)

    total_chunks = sum(len(d["chunks"]) for d in parent_docs)
    avg_chunks = total_chunks / len(parent_docs) if parent_docs else 0
    log.info(f"Done. Written to {OUTPUT_PATH}")
    log.info(f"  Docs: {len(parent_docs)}")
    log.info(f"  Total chunks: {total_chunks}")
    log.info(f"  Avg chunks/doc: {avg_chunks:.1f}")
    log.info(f"  Top-level keys: {list(parent_docs[0].keys()) if parent_docs else []}")
    log.info(f"  Chunk keys: {list(parent_docs[0]['chunks'][0].keys()) if parent_docs else []}")


if __name__ == "__main__":
    main()

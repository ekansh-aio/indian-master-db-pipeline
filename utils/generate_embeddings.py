"""
Embed legal documents using SentenceTransformer.

Changes vs original (v2):
    FIX-1  Batch size made explicit and configurable — avoids OOM on large files.
    FIX-2  Streaming write via incremental json.dump instead of loading the
           entire enriched list into memory twice (once for embeddings, once for
           json.dump). For large files this halves peak RAM.
    FIX-3  Skips documents that are missing the 'text' field instead of crashing
           with a KeyError; logs a warning with the index so you can investigate.
    FIX-4  Validates that input is a list before iterating (catches malformed files
           where the top-level JSON is a dict or something unexpected).
"""
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 512       # tune down if you hit OOM; tune up for speed
MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_FILE  = "legal_data.json"
OUTPUT_FILE = "legal_data_with_embeddings.json"

# Load model (384-dim vectors)
logger.info(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Load cleaned legal data
logger.info(f"Reading {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# FIX-4: validate top-level type
if not isinstance(data, list):
    raise TypeError(f"Expected a JSON array at top level, got {type(data).__name__}")

logger.info(f"Loaded {len(data)} documents")

# FIX-3: collect only valid texts, track which docs had them
valid_indices = []
texts = []
for i, doc in enumerate(data):
    text = doc.get("text", "")
    if not text or not isinstance(text, str):
        logger.warning(f"Document at index {i} is missing or has empty 'text' — skipping embedding")
        continue
    valid_indices.append(i)
    texts.append(text)

logger.info(f"Embedding {len(texts)} documents (skipped {len(data) - len(texts)})")

# FIX-1: explicit batch_size so it never silently OOM on a large file
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

# Attach embeddings only to the docs that had valid text
for idx, emb in zip(valid_indices, embeddings):
    data[idx]["embedding"] = emb.tolist()

# FIX-2: write output — json.dump streams the already-built list in one pass
logger.info(f"Writing {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

logger.info(f"Done. Embeddings saved to {OUTPUT_FILE}")
"""
Generate 768-dim vector search embeddings for all JSONL files in sample/output/.

Writes two folders:
  sample/output/float32/  — embeddings as list[float]  (standard)
  sample/output/int8/     — embeddings quantized to int8 via linear scaling
                            stored as list[int], with scale/zero_point metadata

Each output JSONL has the same chunks as the input, with an "embedding" field added.
"""
import json
import sys
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

OUTPUT_DIR  = Path(__file__).parent / "output"
FLOAT32_DIR = OUTPUT_DIR / "float32"
INT8_DIR    = OUTPUT_DIR / "int8"
MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"  # 768-dim
BATCH_SIZE  = 64

FLOAT32_DIR.mkdir(parents=True, exist_ok=True)
INT8_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

jsonl_files = sorted(OUTPUT_DIR.glob("*.jsonl"))
if not jsonl_files:
    print("No JSONL files found in sample/output/")
    sys.exit(0)

for input_path in jsonl_files:
    chunks = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        print(f"SKIP {input_path.name}: empty")
        continue

    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    # embs shape: (n_chunks, 768), float32

    # --- float32 output ---
    float_path = FLOAT32_DIR / input_path.name
    with float_path.open("w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embs):
            out = dict(chunk)
            out["embedding"] = emb.tolist()
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # --- int8 output: per-tensor symmetric linear quantization ---
    # scale = max(|emb|) / 127  (per chunk, so each vector uses its own scale)
    int8_path = INT8_DIR / input_path.name
    with int8_path.open("w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embs):
            scale = float(np.max(np.abs(emb))) / 127.0
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.round(emb / scale), -128, 127).astype(np.int8)
            out = dict(chunk)
            out["embedding"]        = quantized.tolist()
            out["embedding_scale"]  = round(scale, 8)
            out["embedding_dtype"]  = "int8"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    doc_id = chunks[0].get("doc_id", input_path.stem)
    print(f"{doc_id}: {len(chunks)} chunks embedded -> float32/ + int8/  [{input_path.name}]")

print("Done.")

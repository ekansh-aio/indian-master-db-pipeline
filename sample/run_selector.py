"""
Run weighted_topk_selection on all chunked JSONs in sample/input/
and write one JSONL per document (8 selected chunks) to sample/output/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.weighted_selector import weighted_topk_selection

INPUT_DIR  = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
TOP_K      = 8

# Default uniform weights — override if you want role-biased selection.
ROLE_WEIGHTS = {}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for input_path in sorted(INPUT_DIR.glob("*.json")):
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        print(f"SKIP {input_path.name}: empty or not a list")
        continue

    indices = weighted_topk_selection(
        chunks=chunks,
        top_k=TOP_K,
        similarity_key="doc_similarity",
        role_weights=ROLE_WEIGHTS,
    )
    selected = [chunks[i] for i in indices]

    stem = input_path.stem  # e.g. ODHC010000092022_1_2022-10-14_all_chunks
    output_path = OUTPUT_DIR / f"{stem}_top{TOP_K}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in selected:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    doc_id = chunks[0].get("doc_id", input_path.stem)
    roles  = [chunks[i].get("role", "?") for i in indices]
    print(f"{doc_id}: {len(indices)} chunks selected -> {output_path.name}")
    print(f"  roles: {roles}")

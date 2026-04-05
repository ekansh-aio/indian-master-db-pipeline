"""
Weighted Proportional Top-K Chunk Selector

Selects top-k chunks from a document using a role-aware proportional algorithm.

Algorithm:
  - Let x = number of chunks to select (top_k)
  - Let n = total chunks in the document
  - Let r(i) = number of chunks with role i
  - Quota for role i = floor(x * r(i) / n)
  - Within each role, select by descending doc_similarity
  - If a role has fewer chunks than its quota, take only what's available
    (unused slots are left unfilled — NOT redistributed)
  - Rounding remainder (x - sum of quotas) is also left unfilled

Key design decision: strict floor, no redistribution, no remainder filling.
top_k is a ceiling, not a guarantee.
"""
import math
import logging
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__)


def weighted_topk_selection(
    chunks: List[Dict],
    top_k: int,
    similarity_key: str = "doc_similarity"
) -> List[int]:
    """
    Select chunk indices using the weighted proportional algorithm.

    Args:
        chunks: List of chunk dicts. Each must have:
                  - 'role': the assigned role label (str)
                  - similarity_key: relevance score to the document (float)
        top_k: Target number of chunks to select. Actual count may be
               lower due to strict floor quotas and role underflow.
        similarity_key: Field name holding the chunk-to-document
                        cosine similarity score.

    Returns:
        List of selected chunk indices into the original chunks list.
    """
    n = len(chunks)

    # If top_k exceeds total chunks, return everything
    if top_k >= n:
        logger.debug(f"top_k ({top_k}) >= n ({n}), returning all chunks")
        return list(range(n))

    # --- Group chunk indices by role, sorted by similarity descending ---
    role_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        role = chunk.get('role', 'Others')
        role_to_indices[role].append(idx)

    for role in role_to_indices:
        role_to_indices[role].sort(
            key=lambda i: chunks[i].get(similarity_key, 0.0),
            reverse=True
        )

    # --- Compute strict floor quota per role, take up to what's available ---
    selected_indices: List[int] = []

    for role, indices in role_to_indices.items():
        role_count = len(indices)
        quota = math.floor(top_k * role_count / n)
        take = min(quota, role_count)   # never exceed actual chunk count

        selected_indices.extend(indices[:take])

        logger.debug(
            f"Role '{role}': {role_count} chunks, quota={quota}, selected={take}"
        )

    logger.info(
        f"Weighted selection: top_k={top_k}, n={n}, "
        f"selected={len(selected_indices)} "
        f"(remainder {top_k - len(selected_indices)} slots unfilled)"
    )

    return selected_indices
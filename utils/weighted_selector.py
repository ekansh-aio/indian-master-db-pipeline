"""
Weighted Proportional Top-K Chunk Selector

Selects top-k chunks using a role-aware proportional algorithm.

Algorithm:
  - effective_count(role) = chunk_count(role) * role_weight(role)
  - total_effective       = sum of all effective_counts
  - quota(role)           = floor(top_k * effective_count(role) / total_effective)
  - Within each role, select by descending doc_similarity
  - If a role has fewer chunks than its quota, leftover slots are redistributed
    to the highest-similarity unchosen chunks across all roles (FIX-1).

With uniform weights (all 1.0) this reduces to the original proportional algorithm.
With biased weights (e.g. Precedent Finder) high-weight roles claim more slots.

Changes vs original (v2):
    FIX-1  Unfilled floor-quota slots are now redistributed to the globally
           highest-similarity unchosen chunks. Previously they were silently
           discarded, so callers consistently got fewer than top_k results
           when any role had fewer chunks than its quota.
    FIX-2  Roles with weight=0 are excluded from selection entirely (rather
           than getting a floor-quota of 0 and cluttering the debug log).
    FIX-3  Guard against total_effective=0 when all weights are 0.0.
    FIX-4  Return list is sorted by original chunk order (preserves document
           flow) instead of the arbitrary role-iteration order.
"""
import math
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


def weighted_topk_selection(
    chunks: List[Dict],
    top_k: int,
    similarity_key: str = "doc_similarity",
    role_weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    """
    Select chunk indices using the weighted proportional algorithm.

    Args:
        chunks:         List of chunk dicts, each with 'role' and similarity_key fields.
        top_k:          Target number of chunks to return.
        similarity_key: Field used to rank chunks within a role.
        role_weights:   Dict mapping role name -> weight multiplier.
                        Roles with weight=0 are excluded entirely.
                        Defaults to uniform (1.0) for missing/unknown roles.

    Returns:
        List of selected indices into the original chunks list, sorted by
        their original position (preserves document reading order).
    """
    n = len(chunks)

    if n == 0:
        return []

    if top_k >= n:
        logger.debug(f"top_k ({top_k}) >= n ({n}), returning all chunks")
        return list(range(n))

    if role_weights is None:
        role_weights = {}

    # --- Group chunk indices by role, sorted by similarity descending ---
    role_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        role = chunk.get("role", "Others")
        role_to_indices[role].append(idx)

    for role in role_to_indices:
        role_to_indices[role].sort(
            key=lambda i: chunks[i].get(similarity_key, 0.0),
            reverse=True,
        )

    # FIX-2: exclude zero-weight roles from quota calculation entirely
    active_roles = {
        role: indices
        for role, indices in role_to_indices.items()
        if role_weights.get(role, 1.0) > 0.0
    }
    excluded_roles = set(role_to_indices) - set(active_roles)
    if excluded_roles:
        logger.debug(f"Excluding zero-weight roles: {excluded_roles}")

    # --- Compute effective counts using weights ---
    total_effective = sum(
        len(indices) * role_weights.get(role, 1.0)
        for role, indices in active_roles.items()
    )

    # FIX-3: guard against all-zero weights
    if total_effective == 0:
        logger.warning("total_effective=0 (all weights zero or no active roles), returning top-k by similarity")
        all_sorted = sorted(range(n), key=lambda i: chunks[i].get(similarity_key, 0.0), reverse=True)
        return sorted(all_sorted[:top_k])

    # --- Compute floor quota per role, take up to what's available ---
    selected_set: Set[int] = set()

    for role, indices in active_roles.items():
        weight          = role_weights.get(role, 1.0)
        effective_count = len(indices) * weight
        quota           = math.floor(top_k * effective_count / total_effective)
        take            = min(quota, len(indices))
        selected_set.update(indices[:take])

        logger.debug(
            f"Role '{role}': {len(indices)} chunks, weight={weight:.2f}, "
            f"quota={quota}, selected={take}"
        )

    # FIX-1: redistribute unfilled slots to globally best unchosen chunks
    deficit = top_k - len(selected_set)
    if deficit > 0:
        # Candidates: all active-role chunks not yet selected, sorted by similarity
        candidates = [
            i for role, indices in active_roles.items()
            for i in indices
            if i not in selected_set
        ]
        candidates.sort(key=lambda i: chunks[i].get(similarity_key, 0.0), reverse=True)
        selected_set.update(candidates[:deficit])
        logger.debug(f"Redistributed {min(deficit, len(candidates))} slot(s) from global top-similarity pool")

    # FIX-4: sort by original position to preserve document reading order
    result = sorted(selected_set)

    logger.info(
        f"Weighted selection: top_k={top_k}, n={n}, "
        f"selected={len(result)} "
        f"({'exact' if len(result) == top_k else f'{top_k - len(result)} short'})"
    )

    return result
"""Retrieval quality metrics.

All metrics operate on (query, ranked_results, ground_truth) triples.
"""
from __future__ import annotations

from typing import Iterable


def hit_at_k(matched_ranks: list[int | None], k: int) -> float:
    """Fraction of queries where the correct answer appeared in top-k."""
    if not matched_ranks:
        return 0.0
    hits = sum(1 for r in matched_ranks if r is not None and r < k)
    return hits / len(matched_ranks)


def mrr(matched_ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank. Rank is 0-indexed; MRR uses 1-indexed."""
    if not matched_ranks:
        return 0.0
    total = 0.0
    for r in matched_ranks:
        if r is not None:
            total += 1.0 / (r + 1)
    return total / len(matched_ranks)


def ndcg_at_k(matched_ranks: list[int | None], k: int) -> float:
    """nDCG@k for binary relevance, single relevant doc per query."""
    import math
    if not matched_ranks:
        return 0.0
    total = 0.0
    for r in matched_ranks:
        if r is not None and r < k:
            # Binary relevance, ideal DCG = 1/log2(1+1) = 1
            total += 1.0 / math.log2(r + 2)
    return total / len(matched_ranks)


def find_match_rank(
    results: Iterable,
    expected_substrings: list[str],
) -> int | None:
    """Return the 0-indexed rank of the first result whose content contains
    ALL expected_substrings (case-insensitive). None if not found."""
    needles = [s.lower() for s in expected_substrings if s]
    if not needles:
        return None
    for i, r in enumerate(results):
        # Support both ScoredMemory(.memory.content) and Memory(.content)
        content = getattr(getattr(r, "memory", r), "content", "").lower()
        if all(n in content for n in needles):
            return i
    return None

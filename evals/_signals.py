"""Pure helpers for non-leaky confidence signals captured during eval sweeps.

Kept in a separate module so they're cheap to import and unit-test (no engram
runtime / sqlite cost). All functions are pure and side-effect free.
"""
from __future__ import annotations

from typing import Iterable


def compute_bm25_top_gap(
    bm25_scores: Iterable[float],
) -> tuple[float | None, float | None, float | None]:
    """Return (top1, top2, gap) where gap = top1 - top2.

    Sorts the input descending. Returns Nones when there aren't enough scores.
    Invariants (enforced by tests in tests/property/test_bm25_gap_signal.py):
      - top1 is the maximum, top2 is the second-largest (so top1 >= top2)
      - gap == top1 - top2 when both are present, else None
      - gap >= 0 (never negative)
      - empty -> (None, None, None); singleton -> (x, None, None)
    """
    scores = sorted((float(s) for s in bm25_scores), reverse=True)
    if not scores:
        return None, None, None
    if len(scores) == 1:
        return scores[0], None, None
    top1 = scores[0]
    top2 = scores[1]
    return top1, top2, top1 - top2


def normalized_gap(top1: float | None, top2: float | None) -> float | None:
    """Gap normalized by top1: (top1 - top2) / top1.

    Why: in regimes where BM25 ties on entity tokens (strict-paraphrase),
    raw gap is degenerate near zero. Normalized gap is scale-invariant and
    measures relative separation. Range: [0, 1] when both scores >= 0
    (1 = top2 is zero, 0 = perfect tie).

    Returns None when top1/top2 missing or top1 == 0 (undefined).
    Invariants:
      - 0 <= result <= 1 when top1 > 0 and top2 >= 0 (tolerant of fp noise)
      - normalized_gap(t, t) == 0
      - normalized_gap(t, 0) == 1 for t > 0
    """
    if top1 is None or top2 is None:
        return None
    if top1 == 0:
        return None
    g = (float(top1) - float(top2)) / float(top1)
    # Clamp tiny FP underflow (top2 slightly > top1 should not happen by contract)
    if g < 0:
        g = 0.0
    return g


def crowdedness(
    bm25_scores: Iterable[float],
    frac: float = 0.95,
) -> int | None:
    """Count of candidates with score >= frac * top1.

    Robust to ties at zero: when top1 == 0, all-zero scores all qualify
    (crowdedness == n). Higher crowdedness == more competitive top region
    == BM25 less confident. Inverse signal vs gap.

    Returns None on empty input.
    Invariants (enforced by property tests):
      - 1 <= result <= len(scores) when scores non-empty
      - frac=1.0 returns count of items tied with top1
      - frac=0.0 returns len(scores)
      - permutation invariant
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0,1], got {frac}")
    scores = [float(s) for s in bm25_scores]
    if not scores:
        return None
    top1 = max(scores)
    if top1 == 0:
        # Threshold collapses to 0; everything qualifies. Define 1 (just top1)
        # when frac == 1.0 to keep the "tied with top1" reading intact.
        if frac == 1.0:
            return sum(1 for s in scores if s == 0.0)
        return len(scores)
    threshold = frac * top1
    return sum(1 for s in scores if s >= threshold)

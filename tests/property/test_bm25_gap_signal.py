"""Property-based tests for the non-leaky bm25_gap signal used by adaptive-vw.

This is the cheap regression guard called out in NEXT.md priority #3:
fuzz the helper with arbitrary BM25 score arrays and assert the algebraic
invariants the offline analyzer relies on.
"""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from evals._signals import compute_bm25_top_gap, crowdedness, normalized_gap


# Realistic BM25 scores are non-negative finite floats. Allow zero and small
# values; cap at 1e6 so we don't burn cycles on edge-case float math.
_score = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(scores=st.lists(_score, min_size=0, max_size=64))
@settings(max_examples=300, deadline=None)
def test_top_gap_invariants(scores: list[float]) -> None:
    top1, top2, gap = compute_bm25_top_gap(scores)

    if not scores:
        assert top1 is None and top2 is None and gap is None
        return

    if len(scores) == 1:
        assert top1 == scores[0]
        assert top2 is None
        assert gap is None
        return

    # Both present
    assert top1 is not None and top2 is not None and gap is not None
    # top1 is the max, top2 is the second-largest
    assert top1 == max(scores)
    assert top1 >= top2
    # gap algebra
    assert math.isclose(gap, top1 - top2, rel_tol=0, abs_tol=1e-9)
    assert gap >= 0


@given(scores=st.lists(_score, min_size=2, max_size=64))
@settings(max_examples=100, deadline=None)
def test_permutation_invariance(scores: list[float]) -> None:
    """Result is invariant under input permutation (it's a sort)."""
    a = compute_bm25_top_gap(scores)
    b = compute_bm25_top_gap(list(reversed(scores)))
    assert a == b


def test_known_values() -> None:
    assert compute_bm25_top_gap([]) == (None, None, None)
    assert compute_bm25_top_gap([3.0]) == (3.0, None, None)
    top1, top2, gap = compute_bm25_top_gap([1.0, 5.0, 3.0])
    assert (top1, top2) == (5.0, 3.0)
    assert math.isclose(gap, 2.0)


# ---------- normalized_gap ----------

@given(top1=_score, top2=_score)
@settings(max_examples=300, deadline=None)
def test_normalized_gap_invariants(top1: float, top2: float) -> None:
    # Enforce contract: top2 <= top1
    if top2 > top1:
        top1, top2 = top2, top1
    g = normalized_gap(top1, top2)
    if top1 == 0:
        assert g is None
    else:
        assert g is not None
        assert 0.0 <= g <= 1.0 + 1e-9
        # algebra (within fp tolerance)
        assert math.isclose(g, (top1 - top2) / top1, rel_tol=1e-9, abs_tol=1e-9)


def test_normalized_gap_known() -> None:
    assert normalized_gap(None, 1.0) is None
    assert normalized_gap(1.0, None) is None
    assert normalized_gap(0.0, 0.0) is None
    assert normalized_gap(2.0, 2.0) == 0.0
    assert normalized_gap(2.0, 0.0) == 1.0
    assert math.isclose(normalized_gap(4.0, 1.0), 0.75)


# ---------- crowdedness ----------

@given(scores=st.lists(_score, min_size=1, max_size=64),
       frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=300, deadline=None)
def test_crowdedness_invariants(scores: list[float], frac: float) -> None:
    c = crowdedness(scores, frac=frac)
    assert c is not None
    assert 1 <= c <= len(scores)
    # permutation invariance
    assert c == crowdedness(list(reversed(scores)), frac=frac)


def test_crowdedness_known() -> None:
    assert crowdedness([], frac=0.5) is None
    # frac=1.0 counts ties with top1
    assert crowdedness([5.0, 5.0, 3.0, 1.0], frac=1.0) == 2
    # frac=0.0 returns all (any non-negative score >= 0)
    assert crowdedness([5.0, 5.0, 3.0, 1.0], frac=0.0) == 4
    # All-zero with frac<1 -> all qualify (degenerate-tie regime)
    assert crowdedness([0.0, 0.0, 0.0], frac=0.95) == 3
    assert crowdedness([0.0, 0.0, 0.0], frac=1.0) == 3
    # Mixed
    assert crowdedness([10.0, 9.6, 5.0, 1.0], frac=0.95) == 2  # 10, 9.6


# --- Targeted edge-case fuzz: the "all top1 == 0" / frac=1.0 branches in
# crowdedness(). These are degenerate-tie regimes the policy actually hits in
# strict-paraphrase BM25 (entity-stripped queries produce ties at zero), so
# it's worth fuzzing rather than only spot-checking.

@given(n=st.integers(min_value=1, max_value=64),
       frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_crowdedness_all_zero(n: int, frac: float) -> None:
    """All-zero scores: both frac<1 and frac==1 branches collapse to n."""
    scores = [0.0] * n
    assert crowdedness(scores, frac=frac) == n


@given(scores=st.lists(_score, min_size=1, max_size=64))
@settings(max_examples=200, deadline=None)
def test_crowdedness_frac_one_counts_top_ties(scores: list[float]) -> None:
    """frac=1.0 returns count of candidates tied with top1.

    This is what the routing policy reads as "BM25 confidence" — fewer ties
    at the top = more confident.
    """
    c = crowdedness(scores, frac=1.0)
    top1 = max(scores)
    expected = sum(1 for s in scores if s == top1)
    assert c == expected


@given(scores=st.lists(_score, min_size=1, max_size=64))
@settings(max_examples=200, deadline=None)
def test_crowdedness_frac_zero_returns_n(scores: list[float]) -> None:
    """frac=0.0: threshold is 0, every non-negative score qualifies."""
    assert crowdedness(scores, frac=0.0) == len(scores)


@given(scores=st.lists(_score, min_size=2, max_size=32),
       f1=st.floats(min_value=0.0, max_value=1.0),
       f2=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200, deadline=None)
def test_crowdedness_monotone_in_frac(
    scores: list[float], f1: float, f2: float
) -> None:
    """Lower frac -> lower threshold -> more qualify. Monotone non-increasing
    in frac (when top1 > 0; the all-zero degenerate branch is covered above)."""
    if max(scores) == 0:
        return
    if f1 > f2:
        f1, f2 = f2, f1
    c1 = crowdedness(scores, frac=f1)
    c2 = crowdedness(scores, frac=f2)
    assert c1 is not None and c2 is not None
    assert c1 >= c2


def test_crowdedness_invalid_frac() -> None:
    import pytest
    with pytest.raises(ValueError):
        crowdedness([1.0], frac=1.5)
    with pytest.raises(ValueError):
        crowdedness([1.0], frac=-0.1)

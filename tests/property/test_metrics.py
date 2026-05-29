"""Property-based tests for evals.metrics aggregate metrics: hit_at_k, mrr, ndcg_at_k.

Locks the contracts of the three numbers we report in every eval table.

Invariants enforced:
  hit_at_k:
    - 0.0 <= h <= 1.0
    - empty input -> 0.0
    - k=0 -> 0.0 (no rank can be < 0)
    - monotone non-decreasing in k for fixed input
    - k >= len(input) AND all matched -> 1.0
    - all-None input -> 0.0
    - permutation invariant (set semantics over query outcomes)

  mrr:
    - 0.0 <= m <= 1.0
    - empty input -> 0.0
    - all-None -> 0.0
    - all rank-0 hits -> 1.0
    - permutation invariant
    - moving any hit earlier (smaller rank) never decreases MRR

  ndcg_at_k:
    - 0.0 <= n <= 1.0 (binary relevance, single relevant doc)
    - empty -> 0.0
    - k=0 -> 0.0
    - monotone non-decreasing in k
    - all rank-0 within k -> 1.0
    - moving a hit earlier never decreases nDCG@k
    - permutation invariant
    - matches closed-form: sum(1/log2(r+2)) / N over hits with r<k

  Cross-metric:
    - hit_at_k(ranks, k) >= ndcg_at_k(ranks, k) is FALSE in general (ndcg can
      exceed hit when... actually no, with single-relevant binary at k>=1,
      hit_at_k >= ndcg_at_k since each contributing term <= 1). Lock that.
    - mrr <= hit_at_inf == hit_at_k for k > max_rank.
"""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from evals.metrics import hit_at_k, mrr, ndcg_at_k


# Ranks are 0-indexed positions, capped to keep test fast.
_rank = st.one_of(st.none(), st.integers(min_value=0, max_value=50))
_ranks = st.lists(_rank, min_size=0, max_size=30)
_ranks_nonempty = st.lists(_rank, min_size=1, max_size=30)
_k = st.integers(min_value=0, max_value=60)


# ---------- hit_at_k ----------

def test_hit_empty() -> None:
    assert hit_at_k([], 5) == 0.0


def test_hit_k_zero() -> None:
    assert hit_at_k([0, 1, 2], 0) == 0.0


def test_hit_all_none() -> None:
    assert hit_at_k([None, None, None], 10) == 0.0


@given(ranks=_ranks, k=_k)
@settings(max_examples=300, deadline=None)
def test_hit_bounded(ranks: list[int | None], k: int) -> None:
    h = hit_at_k(ranks, k)
    assert 0.0 <= h <= 1.0


@given(ranks=_ranks_nonempty, k=_k)
@settings(max_examples=200, deadline=None)
def test_hit_monotone_in_k(ranks: list[int | None], k: int) -> None:
    assert hit_at_k(ranks, k) <= hit_at_k(ranks, k + 1)


@given(ranks=_ranks_nonempty)
@settings(max_examples=100, deadline=None)
def test_hit_large_k_eq_match_rate(ranks: list[int | None]) -> None:
    big_k = 10_000
    matched = sum(1 for r in ranks if r is not None)
    assert hit_at_k(ranks, big_k) == matched / len(ranks)


@given(ranks=_ranks_nonempty, k=_k, perm_seed=st.integers(0, 10_000))
@settings(max_examples=100, deadline=None)
def test_hit_permutation_invariant(
    ranks: list[int | None], k: int, perm_seed: int
) -> None:
    import random
    rng = random.Random(perm_seed)
    shuffled = list(ranks)
    rng.shuffle(shuffled)
    assert hit_at_k(ranks, k) == hit_at_k(shuffled, k)


# ---------- mrr ----------

def test_mrr_empty() -> None:
    assert mrr([]) == 0.0


def test_mrr_all_none() -> None:
    assert mrr([None] * 5) == 0.0


def test_mrr_all_rank0() -> None:
    assert mrr([0, 0, 0]) == 1.0


@given(ranks=_ranks)
@settings(max_examples=300, deadline=None)
def test_mrr_bounded(ranks: list[int | None]) -> None:
    m = mrr(ranks)
    assert 0.0 <= m <= 1.0


@given(ranks=_ranks_nonempty, perm_seed=st.integers(0, 10_000))
@settings(max_examples=100, deadline=None)
def test_mrr_permutation_invariant(
    ranks: list[int | None], perm_seed: int
) -> None:
    import random
    rng = random.Random(perm_seed)
    shuffled = list(ranks)
    rng.shuffle(shuffled)
    assert math.isclose(mrr(ranks), mrr(shuffled), rel_tol=1e-9, abs_tol=1e-12)


@given(ranks=_ranks_nonempty, idx=st.integers(0, 100))
@settings(max_examples=200, deadline=None)
def test_mrr_moving_hit_earlier_does_not_decrease(
    ranks: list[int | None], idx: int
) -> None:
    # Find a position with a hit; decrement its rank by 1 (if >0). MRR must
    # not decrease.
    if not ranks:
        return
    pos = idx % len(ranks)
    r = ranks[pos]
    if r is None or r == 0:
        return
    moved = list(ranks)
    moved[pos] = r - 1
    assert mrr(moved) >= mrr(ranks) - 1e-12


# ---------- ndcg_at_k ----------

def test_ndcg_empty() -> None:
    assert ndcg_at_k([], 5) == 0.0


def test_ndcg_k_zero() -> None:
    assert ndcg_at_k([0, 1, 2], 0) == 0.0


def test_ndcg_all_rank0_within_k() -> None:
    assert ndcg_at_k([0, 0, 0], 5) == 1.0


@given(ranks=_ranks, k=_k)
@settings(max_examples=300, deadline=None)
def test_ndcg_bounded(ranks: list[int | None], k: int) -> None:
    n = ndcg_at_k(ranks, k)
    assert 0.0 <= n <= 1.0 + 1e-12


@given(ranks=_ranks_nonempty, k=_k)
@settings(max_examples=200, deadline=None)
def test_ndcg_monotone_in_k(ranks: list[int | None], k: int) -> None:
    assert ndcg_at_k(ranks, k) <= ndcg_at_k(ranks, k + 1) + 1e-12


@given(ranks=_ranks_nonempty, k=_k, perm_seed=st.integers(0, 10_000))
@settings(max_examples=100, deadline=None)
def test_ndcg_permutation_invariant(
    ranks: list[int | None], k: int, perm_seed: int
) -> None:
    import random
    rng = random.Random(perm_seed)
    shuffled = list(ranks)
    rng.shuffle(shuffled)
    assert math.isclose(
        ndcg_at_k(ranks, k), ndcg_at_k(shuffled, k), rel_tol=1e-9, abs_tol=1e-12
    )


@given(ranks=_ranks_nonempty, idx=st.integers(0, 100), k=_k)
@settings(max_examples=200, deadline=None)
def test_ndcg_moving_hit_earlier_does_not_decrease(
    ranks: list[int | None], idx: int, k: int
) -> None:
    if not ranks:
        return
    pos = idx % len(ranks)
    r = ranks[pos]
    if r is None or r == 0:
        return
    moved = list(ranks)
    moved[pos] = r - 1
    assert ndcg_at_k(moved, k) >= ndcg_at_k(ranks, k) - 1e-12


@given(ranks=_ranks_nonempty, k=_k)
@settings(max_examples=200, deadline=None)
def test_ndcg_closed_form(ranks: list[int | None], k: int) -> None:
    expected = 0.0
    for r in ranks:
        if r is not None and r < k:
            expected += 1.0 / math.log2(r + 2)
    expected /= len(ranks)
    assert math.isclose(
        ndcg_at_k(ranks, k), expected, rel_tol=1e-9, abs_tol=1e-12
    )


# ---------- cross-metric ----------

@given(ranks=_ranks_nonempty, k=st.integers(1, 60))
@settings(max_examples=200, deadline=None)
def test_hit_dominates_ndcg(ranks: list[int | None], k: int) -> None:
    # With binary relevance + single relevant doc, each ndcg term is
    # 1/log2(r+2) <= 1, with equality only at r=0. Per-query the term
    # is <= the indicator [r<k], so averaged ndcg <= hit.
    assert ndcg_at_k(ranks, k) <= hit_at_k(ranks, k) + 1e-12


@given(ranks=_ranks_nonempty)
@settings(max_examples=200, deadline=None)
def test_mrr_le_hit_at_inf(ranks: list[int | None]) -> None:
    # Each MRR term 1/(r+1) <= 1 = indicator at infinite k.
    assert mrr(ranks) <= hit_at_k(ranks, 10_000) + 1e-12

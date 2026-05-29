"""Unit tests for evals.bootstrap_ci._per_category_paired_diff."""
from __future__ import annotations

from evals.bootstrap_ci import _per_category_paired_diff


def _make(sample_id: str, cat: str, h1: int, hk: int, rr: float) -> dict:
    return {
        "sample_id": sample_id,
        "category": cat,
        "rank": 0,
        "hit_at_1": h1,
        "hit_at_k": hk,
        "reciprocal_rank": rr,
    }


def test_per_category_slices_independent() -> None:
    # cat 1: a is uniformly worse by 1 (h@1 always 0 vs 1)
    # cat 2: a is uniformly the same as b
    a, b = [], []
    for i in range(40):
        a.append(_make(f"s{i}", "1", 0, 0, 0.0))
        b.append(_make(f"s{i}", "1", 1, 1, 1.0))
    for i in range(40, 80):
        a.append(_make(f"s{i}", "2", 1, 1, 0.5))
        b.append(_make(f"s{i}", "2", 1, 1, 0.5))

    out = _per_category_paired_diff(a, b, resamples=400, seed=7)
    assert set(out.keys()) == {"1", "2"}
    c1 = out["1"]
    c2 = out["2"]
    assert c1["n"] == 40
    assert c2["n"] == 40
    # cat 1: Δ should be -1 deterministically
    assert c1["hit_at_1"]["mean"] == -1.0
    assert c1["hit_at_1"]["ci_lo"] == -1.0
    assert c1["hit_at_1"]["ci_hi"] == -1.0
    # cat 2: Δ should be 0 deterministically
    assert c2["hit_at_1"]["mean"] == 0.0
    assert c2["hit_at_1"]["ci_lo"] == 0.0
    assert c2["hit_at_1"]["ci_hi"] == 0.0


def test_unaligned_pairs_are_skipped() -> None:
    a = [_make("s0", "1", 1, 1, 1.0), _make("s1", "1", 0, 0, 0.0)]
    # second pair disagrees on sample_id -> must be skipped
    b = [_make("s0", "1", 1, 1, 1.0), _make("sX", "1", 1, 1, 1.0)]
    out = _per_category_paired_diff(a, b, resamples=200, seed=1)
    assert out["_skipped_unaligned"] == 1
    assert out["1"]["n"] == 1

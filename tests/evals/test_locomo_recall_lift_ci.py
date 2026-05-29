"""Tests for evals.locomo_recall_lift_ci — the §94c paired bootstrap CI.

The pairing is intrinsic to per_query_pairs (each row already holds a
treatment - baseline delta). We test:
  - mean of bootstrap == sample mean
  - CI brackets the mean
  - constant input is degenerate
  - p_two_sided == ~1.0 when mean ≈ 0
  - p_two_sided is small when all deltas are positive
  - per-category sums to overall n_pairs
"""
from __future__ import annotations

import statistics

from evals.locomo_recall_lift_ci import (
    _bootstrap_mean_ci,
    _frac_pos,
    _per_category,
    _summarize,
)


def _mk(delta_h1, delta_hk=None, delta_rr=None, category="2", sid=None):
    return {
        "sample_id": sid or f"s{id(delta_h1)}",
        "category": category,
        "delta_h1": delta_h1,
        "delta_hk": delta_h1 if delta_hk is None else delta_hk,
        "delta_rr": float(delta_h1) if delta_rr is None else delta_rr,
    }


def test_bootstrap_mean_matches_sample_mean():
    vals = [1, 0, -1, 1, 0, 0, 1, 1, -1, 1]
    m, lo, hi = _bootstrap_mean_ci(vals, resamples=2000, seed=42)
    assert abs(m - statistics.fmean(vals)) < 1e-9
    assert lo <= m <= hi


def test_summarize_constant_zero_deltas_has_p_one():
    pairs = [_mk(0) for _ in range(50)]
    out = _summarize(pairs, resamples=500, seed=1)
    for k in ("delta_h1", "delta_hk", "delta_rr"):
        c = out[k]
        assert c["mean"] == 0.0
        assert c["ci_lo"] == 0.0 == c["ci_hi"]
        # symmetric two-sided p around zero mean -> ~1.0
        assert c["p_bootstrap_two_sided"] >= 0.99
        assert c["frac_pairs_treatment_better"] == 0.0


def test_summarize_all_positive_has_small_p():
    pairs = [_mk(1) for _ in range(40)]
    out = _summarize(pairs, resamples=500, seed=1)
    c = out["delta_h1"]
    assert c["mean"] == 1.0
    assert c["p_bootstrap_two_sided"] == 0.0
    assert c["frac_pairs_treatment_better"] == 1.0


def test_frac_pos():
    assert _frac_pos([1, 0, -1, 1]) == 0.5
    assert _frac_pos([]) == 0.0
    assert _frac_pos([0, 0, 0]) == 0.0


def test_per_category_partitions_pairs():
    pairs = ([_mk(1, category="A") for _ in range(10)] +
             [_mk(0, category="B") for _ in range(5)])
    pc = _per_category(pairs, resamples=200, seed=3)
    assert set(pc.keys()) == {"A", "B"}
    assert pc["A"]["n_pairs"] == 10
    assert pc["B"]["n_pairs"] == 5
    assert pc["A"]["delta_h1"]["mean"] == 1.0
    assert pc["B"]["delta_h1"]["mean"] == 0.0

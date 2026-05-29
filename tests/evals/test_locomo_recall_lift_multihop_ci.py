"""Tests for evals.locomo_recall_lift_multihop_ci — the §95-CI driver.

Verifies:
  - constant input has p≈1, ci_lo==ci_hi==0
  - all-positive input has p==0, mean=1
  - multi-hop / single-hop slices partition the input on n_gold
  - filter cutoff respected (min_n_gold tunable)
  - frac_better + frac_worse <= 1 and consistent
  - empty slice degrades gracefully
  - bootstrap mean ≈ sample mean
"""
from __future__ import annotations

import statistics

from evals.locomo_recall_lift_multihop_ci import (
    _bootstrap_paired,
    _slice,
    _summarize,
)


def _mk(delta=0.0, n_gold=1, sid=None, category="2", **extra):
    row = {
        "sample_id": sid or f"s{id(extra)}",
        "category": category,
        "n_gold": n_gold,
        "delta_prk": delta,
        "delta_grk": delta,
        "delta_h1": delta,
        "delta_hk": delta,
        "delta_rr": delta,
    }
    row.update(extra)
    return row


def test_bootstrap_mean_matches_sample_mean():
    vals = [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0]
    out = _bootstrap_paired(vals, resamples=2000, seed=42)
    assert abs(out["mean"] - statistics.fmean(vals)) < 1e-9
    assert out["ci_lo"] <= out["mean"] <= out["ci_hi"]
    assert out["n"] == len(vals)


def test_constant_zero_is_degenerate():
    out = _bootstrap_paired([0.0] * 50, resamples=500, seed=1)
    assert out["mean"] == 0.0
    assert out["ci_lo"] == 0.0 == out["ci_hi"]
    assert out["p_bootstrap_two_sided"] >= 0.99
    assert out["frac_pairs_treatment_better"] == 0.0
    assert out["frac_pairs_treatment_worse"] == 0.0


def test_all_positive_has_zero_p():
    out = _bootstrap_paired([1.0] * 40, resamples=500, seed=1)
    assert out["mean"] == 1.0
    assert out["p_bootstrap_two_sided"] == 0.0
    assert out["frac_pairs_treatment_better"] == 1.0
    assert out["frac_pairs_treatment_worse"] == 0.0


def test_empty_slice_returns_zero_n():
    out = _bootstrap_paired([], resamples=100, seed=1)
    assert out["n"] == 0
    assert out["mean"] == 0.0
    assert out["p_bootstrap_two_sided"] == 1.0


def test_slice_predicate_partitions():
    pairs = (
        [_mk(1.0, n_gold=1) for _ in range(7)]
        + [_mk(-1.0, n_gold=2) for _ in range(3)]
        + [_mk(-2.0, n_gold=5) for _ in range(2)]
    )
    multi = _slice(pairs, lambda p: p["n_gold"] >= 2)
    single = _slice(pairs, lambda p: p["n_gold"] < 2)
    assert len(multi) == 5
    assert len(single) == 7
    assert all(p["n_gold"] >= 2 for p in multi)
    assert all(p["n_gold"] < 2 for p in single)


def test_summarize_includes_all_delta_keys():
    pairs = [_mk(0.5, n_gold=2) for _ in range(20)]
    out = _summarize(pairs, resamples=300, seed=7)
    assert out["n_pairs"] == 20
    for k in ("delta_prk", "delta_grk", "delta_h1", "delta_hk", "delta_rr"):
        assert out[k]["mean"] == 0.5
        assert out[k]["ci_lo"] <= 0.5 <= out[k]["ci_hi"]


def test_better_plus_worse_le_one():
    """frac_better + frac_worse must equal frac of nonzero deltas."""
    out = _bootstrap_paired([1.0, -1.0, 0.0, 1.0, 0.0], resamples=200, seed=1)
    assert out["frac_pairs_treatment_better"] == 0.4
    assert out["frac_pairs_treatment_worse"] == 0.2
    assert (
        out["frac_pairs_treatment_better"] + out["frac_pairs_treatment_worse"]
        <= 1.0
    )


def test_summarize_skips_missing_keys():
    pairs = [{"n_gold": 2, "delta_prk": 0.0, "category": "x"}] * 10
    out = _summarize(pairs, resamples=100, seed=1)
    assert "delta_prk" in out
    # other delta keys missing → not present
    assert "delta_h1" not in out
    assert "delta_hk" not in out

"""Property-based + scenario tests for evals.prf_x_shareprior_stack_ci.

These guard the headline §5.4 paper number: paired bootstrap CIs over
the PRF × share_prior 2×2 stack. The interaction term is what tells the
reader whether the +0.07 Δ_BOTH is a true super-additive effect or
sample noise — so the stats helpers themselves need their own tests.
"""
from __future__ import annotations

import statistics

import pytest
from hypothesis import given, settings, strategies as st

from evals.prf_x_shareprior_stack_ci import (
    _bootstrap_paired, _percentile, _two_sided_p,
)


# ------------------- _percentile -------------------

def test_percentile_endpoints():
    xs = sorted([0.1, 0.2, 0.3, 0.5, 0.9])
    assert _percentile(xs, 0.0) == 0.1
    assert _percentile(xs, 1.0) == 0.9


def test_percentile_empty():
    assert _percentile([], 0.5) == 0.0


@given(st.lists(st.floats(min_value=-1.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False),
                min_size=1, max_size=50))
def test_percentile_bounded_by_extremes(xs):
    s = sorted(xs)
    for q in (0.0, 0.025, 0.5, 0.975, 1.0):
        v = _percentile(s, q)
        assert s[0] - 1e-9 <= v <= s[-1] + 1e-9


# ------------------- _two_sided_p -------------------

def test_two_sided_p_centered_at_zero():
    # symmetric around 0 → p ≈ 1
    samples = [-0.1, -0.05, 0.0, 0.05, 0.1]
    p = _two_sided_p(samples, 0.0)
    assert 0.4 <= p <= 1.0  # not significant


def test_two_sided_p_far_from_zero():
    # all positive → p as small as possible
    samples = [0.1, 0.2, 0.3, 0.4]
    assert _two_sided_p(samples, 0.0) == 0.0


def test_two_sided_p_empty():
    assert _two_sided_p([], 0.0) == 1.0


@given(st.lists(st.floats(min_value=-1.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False),
                min_size=1, max_size=200))
def test_two_sided_p_bounds(samples):
    p = _two_sided_p(samples, 0.0)
    assert 0.0 <= p <= 1.0


# ------------------- _bootstrap_paired -------------------

def _mk_per_seed(values_per_cell: dict[str, list[float]]) -> dict[str, list[dict]]:
    """Helper: turn {cell: [v_seed0, v_seed1, ...]} into the per-seed
    list-of-dicts shape that _bootstrap_paired consumes."""
    return {
        cell: [{"pair_recall@10": v} for v in vs]
        for cell, vs in values_per_cell.items()
    }


def test_bootstrap_paired_zero_effect():
    # All four cells identical → every Δ should bootstrap to ~0 with
    # CI containing 0 and a non-significant p.
    same = [0.3, 0.4, 0.5, 0.4, 0.3]
    per_seed = _mk_per_seed({
        "C0_baseline": same,
        "CP_prf_only": same,
        "CR_share_prior_only": same,
        "CB_both": same,
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=2000, seed=17)
    for k in ("delta_prf", "delta_sp", "delta_both", "interaction"):
        assert out[k]["point"] == pytest.approx(0.0, abs=1e-9)
        lo, hi = out[k]["ci95"]
        assert lo <= 0.0 <= hi
        assert out[k]["p_two_sided_vs_0"] >= 0.5


def test_bootstrap_paired_pure_super_additive_signal():
    # Construct a clean super-additive case: PRF=+0.05, SP=+0.05,
    # BOTH=+0.20 (interaction = +0.10) with low seed variance.
    base = [0.30, 0.31, 0.29, 0.30, 0.31, 0.29, 0.30, 0.31]
    per_seed = _mk_per_seed({
        "C0_baseline": base,
        "CP_prf_only": [b + 0.05 for b in base],
        "CR_share_prior_only": [b + 0.05 for b in base],
        "CB_both": [b + 0.20 for b in base],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=3000, seed=42)
    assert out["delta_prf"]["point"] == pytest.approx(0.05, abs=1e-3)
    assert out["delta_sp"]["point"] == pytest.approx(0.05, abs=1e-3)
    assert out["delta_both"]["point"] == pytest.approx(0.20, abs=1e-3)
    assert out["interaction"]["point"] == pytest.approx(0.10, abs=1e-3)
    # CI on the interaction excludes 0 (signal is much bigger than noise)
    lo, hi = out["interaction"]["ci95"]
    assert lo > 0.0
    assert out["interaction"]["p_two_sided_vs_0"] < 0.05


def test_bootstrap_paired_decomposition_identity():
    # interaction must algebraically equal Δ_BOTH − (Δ_PRF + Δ_SP) at the
    # point estimate, regardless of noise.
    per_seed = _mk_per_seed({
        "C0_baseline": [0.20, 0.25, 0.18, 0.22],
        "CP_prf_only": [0.27, 0.31, 0.20, 0.29],
        "CR_share_prior_only": [0.21, 0.30, 0.22, 0.24],
        "CB_both": [0.40, 0.45, 0.39, 0.41],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=500, seed=1)
    expected = (out["delta_both"]["point"]
                - (out["delta_prf"]["point"] + out["delta_sp"]["point"]))
    assert out["interaction"]["point"] == pytest.approx(expected, abs=1e-9)


def test_bootstrap_paired_ci_brackets_point():
    per_seed = _mk_per_seed({
        "C0_baseline": [0.2, 0.3, 0.25, 0.22, 0.27],
        "CP_prf_only": [0.25, 0.33, 0.28, 0.24, 0.30],
        "CR_share_prior_only": [0.21, 0.31, 0.26, 0.23, 0.28],
        "CB_both": [0.30, 0.38, 0.33, 0.29, 0.35],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=2000, seed=7)
    for key in ("delta_prf", "delta_sp", "delta_both", "interaction"):
        lo, hi = out[key]["ci95"]
        # CI should bracket the point estimate (or at worst touch it).
        assert lo - 1e-3 <= out[key]["point"] <= hi + 1e-3


@settings(max_examples=15, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0,
                       allow_nan=False, allow_infinity=False),
             min_size=4, max_size=10),
    st.floats(min_value=-0.2, max_value=0.2),
    st.floats(min_value=-0.2, max_value=0.2),
    st.floats(min_value=-0.2, max_value=0.2),
)
def test_bootstrap_paired_point_matches_seed_means(base, dp, dr, db):
    per_seed = _mk_per_seed({
        "C0_baseline": base,
        "CP_prf_only": [max(0.0, min(1.0, b + dp)) for b in base],
        "CR_share_prior_only": [max(0.0, min(1.0, b + dr)) for b in base],
        "CB_both": [max(0.0, min(1.0, b + db)) for b in base],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=200, seed=11)
    # Point estimates equal differences of seed means
    m0 = statistics.fmean(per_seed["C0_baseline"][i]["pair_recall@10"]
                          for i in range(len(base)))
    mp = statistics.fmean(per_seed["CP_prf_only"][i]["pair_recall@10"]
                          for i in range(len(base)))
    mr = statistics.fmean(per_seed["CR_share_prior_only"][i]["pair_recall@10"]
                          for i in range(len(base)))
    mb = statistics.fmean(per_seed["CB_both"][i]["pair_recall@10"]
                          for i in range(len(base)))
    assert out["delta_prf"]["point"] == pytest.approx(mp - m0, abs=1e-3)
    assert out["delta_sp"]["point"] == pytest.approx(mr - m0, abs=1e-3)
    assert out["delta_both"]["point"] == pytest.approx(mb - m0, abs=1e-3)
    assert out["interaction"]["point"] == pytest.approx(
        (mb - m0) - ((mp - m0) + (mr - m0)), abs=1e-3,
    )

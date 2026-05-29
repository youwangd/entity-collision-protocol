"""Tests for evals.prf_x_shareprior_axis_ci — bootstrap CI generalized
across structural-axis sweeps (gate, breadth, noise, scale, alpha).

Functionally identical to the headline-stack bootstrap, so the property
guarantees should match: decomposition identity, point ≡ seed-mean Δs,
CIs brackets points.
"""
from __future__ import annotations

import statistics

import pytest

from evals.prf_x_shareprior_axis_ci import (
    _bootstrap_paired, _percentile, _two_sided_p,
)


def _mk(values_per_cell: dict[str, list[float]]) -> dict[str, list[dict]]:
    return {
        cell: [{"pair_recall@10": v} for v in vs]
        for cell, vs in values_per_cell.items()
    }


def test_axis_ci_zero_effect_brackets_zero():
    same = [0.30, 0.40, 0.32, 0.35, 0.31]
    per_seed = _mk({
        "C0_baseline": same, "CP_prf_only": same,
        "CR_share_prior_only": same, "CB_both": same,
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=2000, seed=17)
    for k in ("delta_prf", "delta_sp", "delta_both", "interaction"):
        assert out[k]["point"] == pytest.approx(0.0, abs=1e-9)
        lo, hi = out[k]["ci95"]
        assert lo <= 0.0 <= hi
        assert out[k]["p_two_sided_vs_0"] >= 0.5


def test_axis_ci_decomposition_identity():
    per_seed = _mk({
        "C0_baseline": [0.20, 0.25, 0.18, 0.22],
        "CP_prf_only": [0.27, 0.31, 0.20, 0.29],
        "CR_share_prior_only": [0.21, 0.30, 0.22, 0.24],
        "CB_both": [0.40, 0.45, 0.39, 0.41],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=500, seed=1)
    expected = (out["delta_both"]["point"]
                - (out["delta_prf"]["point"] + out["delta_sp"]["point"]))
    assert out["interaction"]["point"] == pytest.approx(expected, abs=1e-9)


def test_axis_ci_super_additive_excludes_zero():
    base = [0.30, 0.31, 0.29, 0.30, 0.31, 0.29, 0.30, 0.31]
    per_seed = _mk({
        "C0_baseline": base,
        "CP_prf_only": [b + 0.05 for b in base],
        "CR_share_prior_only": [b + 0.05 for b in base],
        "CB_both": [b + 0.20 for b in base],
    })
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=3000, seed=42)
    assert out["interaction"]["point"] == pytest.approx(0.10, abs=1e-3)
    lo, _ = out["interaction"]["ci95"]
    assert lo > 0.0
    assert out["interaction"]["p_two_sided_vs_0"] < 0.05


def test_axis_ci_cells_match_seed_means():
    base = [0.20, 0.30, 0.25]
    cp = [0.25, 0.33, 0.28]
    cr = [0.21, 0.31, 0.26]
    cb = [0.30, 0.38, 0.33]
    per_seed = _mk({"C0_baseline": base, "CP_prf_only": cp,
                    "CR_share_prior_only": cr, "CB_both": cb})
    out = _bootstrap_paired(per_seed, "pair_recall@10", resamples=300, seed=7)
    assert out["cells"]["C0_baseline"] == pytest.approx(
        round(statistics.fmean(base), 4), abs=1e-9
    )
    assert out["cells"]["CB_both"] == pytest.approx(
        round(statistics.fmean(cb), 4), abs=1e-9
    )


def test_axis_ci_helpers_reuse_same_logic():
    # Smoke: percentile + two-sided p in this module behave like the
    # stack module's helpers (they share semantics).
    s = sorted([0.1, 0.2, 0.3, 0.4, 0.5])
    assert _percentile(s, 0.0) == 0.1
    assert _percentile(s, 1.0) == 0.5
    assert _two_sided_p([0.1, 0.2, 0.3], 0.0) == 0.0
    assert _two_sided_p([], 0.0) == 1.0

"""§85 — Bootstrap CI on Δ for the §82 SYNTHETIC per-tau curve.

Smoke-level regression tests; the heavy real-recipe B=200 run lives
in ``bench/results/synthetic_fragmentation_per_tau_bootstrap.json``.
"""
from __future__ import annotations

from evals.schema_fragmentation_per_tau_calibration import Cell
from evals.synthetic_fragmentation_per_tau_bootstrap import (
    _delta_for_seed,
    _percentile,
    evaluate_tau,
    run,
)


def _small_cell(tau: float = 0.15) -> Cell:
    return Cell(
        n_clusters=20,
        cluster_size=4,
        vocab_size=200,
        core_size=8,
        schema_size=6,
        tau=tau,
    )


def test_percentile_basic():
    assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_delta_for_seed_paired_returns_floats():
    cell = _small_cell(tau=0.15)
    f0, f10, d = _delta_for_seed(cell, seed=42)
    assert 0.0 <= f0 <= 1.0
    assert 0.0 <= f10 <= 1.0
    assert abs((f10 - f0) - d) < 1e-9


def test_evaluate_tau_pure_determinism():
    cell = _small_cell(tau=0.15)
    a = evaluate_tau(cell, tau=0.15, n_boot=4, seed=123)
    b = evaluate_tau(cell, tau=0.15, n_boot=4, seed=123)
    assert a == b


def test_evaluate_tau_seed_changes_distribution():
    cell = _small_cell(tau=0.15)
    a = evaluate_tau(cell, tau=0.15, n_boot=4, seed=123)
    b = evaluate_tau(cell, tau=0.15, n_boot=4, seed=456)
    assert a["deltas_head"] != b["deltas_head"]


def test_evaluate_tau_ci_brackets_mean():
    cell = _small_cell(tau=0.15)
    r = evaluate_tau(cell, tau=0.15, n_boot=6, seed=7)
    assert r["ci95_lo"] <= r["mean"] <= r["ci95_hi"]
    assert r["sd"] >= 0.0
    assert isinstance(r["ci_positive"], bool)
    assert isinstance(r["ci_above_lift"], bool)
    assert len(r["f0_all"]) == r["n_boot"]
    assert len(r["f10_all"]) == r["n_boot"]


def test_run_smoke_small_taus():
    res = run(taus=(0.15, 0.20), n_boot=3, seed=11, cell=_small_cell())
    assert res["corpus"] == "synthetic_disjoint_core"
    assert len(res["by_tau"]) == 2
    for r in res["by_tau"]:
        assert {"tau", "ci95_lo", "ci95_hi", "f0_all", "f10_all"} <= set(r)

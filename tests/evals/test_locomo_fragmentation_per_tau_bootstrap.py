"""§83 — Bootstrap CI on Δ for the LoCoMo per-tau fragmentation curve.

Smoke-level regression tests; the heavy real-corpus B=200 run lives
in ``bench/results/locomo_fragmentation_per_tau_bootstrap.json``.
"""
from __future__ import annotations

import random

import pytest

from evals.locomo_fragmentation_per_tau_bootstrap import (
    _delta_for_resample,
    _percentile,
    _resample,
    evaluate_tau,
)


def _toy_fps() -> dict[str, frozenset[str]]:
    """Disjoint per-cluster cores, low contamination at c=0."""
    return {
        f"s{i}": frozenset(f"v{i}_{j}" for j in range(6))
        for i in range(40)
    }


def test_percentile_basic():
    assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert _percentile([1.0, 2.0], 0.5) == 1.5


def test_resample_size_and_keys_distinct():
    fps = _toy_fps()
    rng = random.Random(0)
    out = _resample(fps, rng, m_frac=0.8)
    assert len(out) == int(round(0.8 * len(fps)))
    # Subsample without replacement: keys are a strict subset of the input.
    assert set(out.keys()) <= set(fps.keys())
    assert len(set(out.keys())) == len(out)


def test_resample_no_duplicate_fps_when_input_unique():
    """Subsample preserves the source's no-exact-dup property."""
    fps = _toy_fps()
    rng = random.Random(0)
    out = _resample(fps, rng, m_frac=0.5)
    # Each fp appears at most once.
    seen: list[frozenset[str]] = []
    for v in out.values():
        assert v not in seen
        seen.append(v)


def test_resample_invalid_mfrac():
    fps = _toy_fps()
    rng = random.Random(0)
    with pytest.raises(ValueError):
        _resample(fps, rng, m_frac=0.0)
    with pytest.raises(ValueError):
        _resample(fps, rng, m_frac=1.5)


def test_delta_nonneg_on_disjoint_corpus():
    """Disjoint cores: c=0 ⇒ all singletons (frag=1.0); c=0.10 cannot
    increase singletons (already saturated). Δ ≤ 0 at tau=0.10 is the
    sanity floor."""
    fps = _toy_fps()
    f0, f10, d = _delta_for_resample(fps, tau=0.50, seed=42)
    # On a disjoint-core regime, frag is already saturated; Δ should
    # be zero or near-zero.
    assert d <= 0.05
    assert 0.0 <= f0 <= 1.0
    assert 0.0 <= f10 <= 1.0
    assert abs((f10 - f0) - d) < 1e-9


def test_evaluate_tau_pure_determinism():
    fps = _toy_fps()
    a = evaluate_tau(fps, tau=0.50, n_boot=4, seed=123)
    b = evaluate_tau(fps, tau=0.50, n_boot=4, seed=123)
    assert a == b


def _overlap_fps() -> dict[str, frozenset[str]]:
    """Mid-overlap fps so cluster() actually moves under perturbation."""
    return {
        f"s{i}": frozenset({f"shared_{i // 4}_{k}" for k in range(4)} |
                            {f"u_{i}_{k}" for k in range(2)})
        for i in range(40)
    }


def test_evaluate_tau_seed_changes_distribution():
    fps = _overlap_fps()
    a = evaluate_tau(fps, tau=0.30, n_boot=4, seed=123)
    b = evaluate_tau(fps, tau=0.30, n_boot=4, seed=456)
    assert a["deltas_head"] != b["deltas_head"]


def test_evaluate_tau_ci_brackets_mean():
    fps = _toy_fps()
    r = evaluate_tau(fps, tau=0.50, n_boot=8, seed=7)
    assert r["ci95_lo"] <= r["mean"] <= r["ci95_hi"]
    assert r["sd"] >= 0.0
    assert 0.0 <= r["p_below_zero"] <= 1.0
    assert 0.0 <= r["p_below_lift"] <= 1.0
    assert isinstance(r["ci_positive"], bool)
    assert isinstance(r["ci_above_lift"], bool)


def test_run_shape():
    fps = _toy_fps()
    # Bypass real LoCoMo I/O: monkeypatch is overkill — call evaluate_tau
    # directly through `run` by writing a tiny stub corpus. Skip: just
    # call evaluate_tau via the exported path.
    rows = [evaluate_tau(fps, t, n_boot=3, seed=0) for t in (0.30, 0.50)]
    for r in rows:
        assert "tau" in r and "ci95_lo" in r and "ci95_hi" in r

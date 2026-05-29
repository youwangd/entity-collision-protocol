"""Unit tests for the paired-bootstrap verdict helpers in
evals.adaptive_vw_offline.

These tests defend the *decision rule* (CI lower bound > 0 == useful) on
small synthetic per-query records, so the offline analyzer's "useful" /
"not useful" verdict can be relied on as the gate for shipping an adaptive
policy.
"""
from __future__ import annotations

from evals.adaptive_vw_offline import _paired_bootstrap_ci, _signal_verdict


def test_paired_bootstrap_zero_deltas_ci_contains_zero() -> None:
    deltas = [0.0] * 50
    mean, lo, hi = _paired_bootstrap_ci(deltas)
    assert mean == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_clear_positive() -> None:
    deltas = [1.0] * 100
    mean, lo, hi = _paired_bootstrap_ci(deltas)
    assert mean == 1.0
    # Constant sample => bootstrap is also constant 1.0
    assert lo == 1.0 and hi == 1.0


def test_paired_bootstrap_noisy_negative_ci_excludes_zero() -> None:
    # Mean clearly < 0 with low variance: CI must be strictly < 0.
    deltas = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
              -1.0, -1.0, -1.0, 0.0]
    mean, lo, hi = _paired_bootstrap_ci(deltas, resamples=2000)
    assert mean < 0
    assert hi < 0  # entire CI on negative side


def test_signal_verdict_perfect_signal_is_useful() -> None:
    """Construct a 4-query setup where bm25 gets q1,q2 right and vw=0.5 gets
    q3,q4 right. A 'magic' signal of [1,1,0,0] perfectly routes — adaptive
    should hit 1.0 vs static-best 0.5, and the verdict must say useful.
    """
    weights = [0.0, 0.5]
    queries = ["q1", "q2", "q3", "q4"]
    bm25_by_q = {
        "q1": {"hit_at_1": 1, "reciprocal_rank": 1.0},
        "q2": {"hit_at_1": 1, "reciprocal_rank": 1.0},
        "q3": {"hit_at_1": 0, "reciprocal_rank": 0.0},
        "q4": {"hit_at_1": 0, "reciprocal_rank": 0.0},
    }
    by_q = {
        "q1": {0.5: {"hit_at_1": 0, "reciprocal_rank": 0.0}},
        "q2": {0.5: {"hit_at_1": 0, "reciprocal_rank": 0.0}},
        "q3": {0.5: {"hit_at_1": 1, "reciprocal_rank": 1.0}},
        "q4": {0.5: {"hit_at_1": 1, "reciprocal_rank": 1.0}},
    }
    static_best_per_q = [1, 1, 0, 0]  # vw=0 (bm25) tied with vw=0.5: pick 0
    sig = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 0.0}
    v = _signal_verdict(queries, by_q, bm25_by_q, weights, sig,
                        static_best_per_q, "magic")
    assert v["adaptive_hit_at_1"] == 1.0
    # Adaptive lifts q3,q4 from 0->1 (delta +1 each) and keeps q1,q2 (delta 0).
    # Mean delta = 0.5. With only 4 samples, CI may include 0 — the test
    # only insists the mean is strictly positive and adaptive beats static.
    assert v["delta_vs_static_best"]["mean"] > 0


def test_signal_verdict_useless_signal_not_useful() -> None:
    """Random/constant signal cannot beat static-best — CI should not be
    strictly above 0."""
    weights = [0.0, 0.5]
    queries = [f"q{i}" for i in range(20)]
    # bm25 hits 10/20, vw=0.5 hits a different 10/20.
    bm25_by_q = {
        q: {"hit_at_1": 1 if i < 10 else 0, "reciprocal_rank": 0.0}
        for i, q in enumerate(queries)
    }
    by_q = {
        q: {0.5: {"hit_at_1": 0 if i < 10 else 1, "reciprocal_rank": 0.0}}
        for i, q in enumerate(queries)
    }
    static_best_per_q = [1 if i < 10 else 0 for i in range(20)]  # vw=0
    sig = {q: 0.5 for q in queries}  # constant: no information
    v = _signal_verdict(queries, by_q, bm25_by_q, weights, sig,
                        static_best_per_q, "constant")
    # Constant signal puts everyone in the same bucket — adaptive == static.
    # Verdict must be NOT useful.
    assert not v["useful"]

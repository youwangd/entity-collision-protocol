"""Tests for evals.bootstrap_ci — the math, not the CLI.

Properties we want:
  - mean of the bootstrap distribution ≈ sample mean (within tolerance)
  - CI brackets the true mean for known distributions with high probability
  - CI on a constant input is degenerate [c, c]
  - paired-diff CI on identical samples is degenerate at 0
"""
from __future__ import annotations

import statistics

from evals.bootstrap_ci import _bootstrap_mean_ci, _paired_diff_ci, _summarize_pq


def test_bootstrap_mean_matches_sample_mean():
    vals = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    m, lo, hi = _bootstrap_mean_ci(vals, resamples=2000, seed=42)
    assert abs(m - statistics.fmean(vals)) < 1e-9
    assert lo <= m <= hi
    # 0/1 hit data: CI is bounded in [0, 1]
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_constant_is_degenerate():
    m, lo, hi = _bootstrap_mean_ci([0.7] * 50, resamples=500, seed=1)
    assert abs(m - 0.7) < 1e-9
    assert abs(lo - 0.7) < 1e-9
    assert abs(hi - 0.7) < 1e-9


def test_bootstrap_empty_safe():
    m, lo, hi = _bootstrap_mean_ci([], resamples=100, seed=1)
    assert (m, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_diff_zero_for_identical_paired_samples():
    a = [0, 1, 1, 0, 1, 0, 1, 1]
    b = list(a)
    m, lo, hi = _paired_diff_ci(a, b, resamples=500, seed=7)
    assert m == 0
    assert lo == 0 and hi == 0


def test_paired_diff_detects_clear_separation():
    # a is reliably better than b; CI of diff should exclude 0 with comfortable margin.
    a = [1] * 80 + [0] * 20
    b = [0] * 80 + [1] * 20
    m, lo, hi = _paired_diff_ci(a, b, resamples=2000, seed=11)
    assert m == 0.6
    assert lo > 0  # significantly positive
    assert hi <= 1.0


def test_summarize_pq_shape():
    pq = [
        {"hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0},
        {"hit_at_1": 0, "hit_at_k": 1, "reciprocal_rank": 0.5},
        {"hit_at_1": 0, "hit_at_k": 0, "reciprocal_rank": 0.0},
    ]
    s = _summarize_pq(pq, resamples=500, seed=3)
    assert set(s) == {"hit_at_1", "hit_at_k", "mrr", "n"}
    assert s["n"] == 3
    for k in ("hit_at_1", "hit_at_k", "mrr"):
        assert {"mean", "ci_lo", "ci_hi"} <= set(s[k])
        assert s[k]["ci_lo"] <= s[k]["mean"] <= s[k]["ci_hi"]

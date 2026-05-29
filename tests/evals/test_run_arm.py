"""Unit tests for evals/run_arm.py — pairing + bootstrap CI bundle."""
from __future__ import annotations


import pytest

from evals.run_arm import _extract_pairs, _paired_summary


def _mk_lme(per_instance):
    return {"per_instance": per_instance}


def _mk_locomo(per_query):
    return {"per_query": per_query}


def test_extract_pairs_lme_aligned():
    b = _mk_lme([
        {"question_id": "q1", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0},
        {"question_id": "q2", "hit_at_1": 0, "hit_at_k": 1, "reciprocal_rank": 0.5},
    ])
    t = _mk_lme([
        {"question_id": "q1", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0},
        {"question_id": "q2", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0},
    ])
    bh1, th1, *_rest, n = _extract_pairs("lme", b, t)
    assert n == 2
    assert bh1 == [1.0, 0.0]
    assert th1 == [1.0, 1.0]


def test_extract_pairs_locomo_aligned():
    b = _mk_locomo([
        {"sample_id": "s1", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0},
    ])
    t = _mk_locomo([
        {"sample_id": "s1", "hit_at_1": 0, "hit_at_k": 1, "reciprocal_rank": 0.5},
    ])
    bh1, th1, _, _, br, tr, n = _extract_pairs("locomo", b, t)
    assert n == 1
    assert bh1 == [1.0]
    assert th1 == [0.0]
    assert br == [1.0]
    assert tr == [0.5]


def test_extract_pairs_length_mismatch_raises():
    b = _mk_lme([{"question_id": "q1", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0}])
    t = _mk_lme([])
    with pytest.raises(SystemExit, match="length mismatch"):
        _extract_pairs("lme", b, t)


def test_extract_pairs_id_divergence_raises():
    b = _mk_lme([{"question_id": "q1", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0}])
    t = _mk_lme([{"question_id": "qX", "hit_at_1": 1, "hit_at_k": 1, "reciprocal_rank": 1.0}])
    with pytest.raises(SystemExit, match="ID divergence"):
        _extract_pairs("lme", b, t)


def test_paired_summary_significance_positive():
    # Treatment dominates baseline on every paired instance → Δ CI must be > 0.
    b_h1 = [0.0] * 30
    t_h1 = [1.0] * 30
    b_hk = [0.0] * 30
    t_hk = [1.0] * 30
    b_rr = [0.0] * 30
    t_rr = [1.0] * 30
    out = _paired_summary(b_h1, t_h1, b_hk, t_hk, b_rr, t_rr, resamples=200, seed=1)
    assert out["n"] == 30
    assert out["hit_at_1"]["delta"]["mean"] == 1.0
    assert out["hit_at_1"]["significant_at_05"] is True


def test_paired_summary_no_effect():
    # Identical paired vectors → Δ mean = 0, not significant.
    vals = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    out = _paired_summary(vals, vals, vals, vals, vals, vals, resamples=200, seed=2)
    assert out["hit_at_1"]["delta"]["mean"] == 0.0
    assert out["hit_at_1"]["significant_at_05"] is False

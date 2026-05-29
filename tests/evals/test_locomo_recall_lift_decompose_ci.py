"""Tests for §94c-decompose-CI bootstrap on (S_a − S_b) paired delta."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_ci import (
    SUBSET_PRESETS,
    _bootstrap_mean_ci,
    run_decompose_ci,
)


def _toy_dataset(tmp_path: Path) -> Path:
    data = [{
        "sample_id": "S1",
        "sessions": [
            {"id": "sess_a",
             "turns": [
                 {"speaker": "user", "content": "Alice loves apricots."},
                 {"speaker": "user", "content": "Apricots ripen in July."},
             ]},
            {"id": "sess_b",
             "turns": [
                 {"speaker": "user", "content": "Bob enjoys badminton."},
                 {"speaker": "user", "content": "Badminton is a racquet sport."},
             ]},
        ],
        "qa": [
            {"question": "What does Alice love?", "answer": "apricots",
             "category": "single_session_user", "evidence": ["sess_a"]},
            {"question": "What does Bob enjoy?", "answer": "badminton",
             "category": "single_session_user", "evidence": ["sess_b"]},
        ],
    }]
    p = tmp_path / "locomo_toy.json"
    p.write_text(json.dumps(data))
    return p


def test_subset_presets_consistent_with_decompose():
    """SUBSET_PRESETS must mirror locomo_recall_lift_decompose.DEFAULT_SUBSETS."""
    from evals.locomo_recall_lift_decompose import DEFAULT_SUBSETS
    expected = {name: stages for name, stages in DEFAULT_SUBSETS}
    assert SUBSET_PRESETS == expected


def test_bootstrap_mean_ci_zero_is_p_one():
    """Constant-zero diffs must yield p=1.0 (CI brackets zero, mean 0)."""
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 50, resamples=200, seed=7)
    assert m == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_bootstrap_mean_ci_positive_signal():
    """A clearly positive sample should yield p < 0.05 and CI > 0."""
    vals = [0.10] * 30 + [0.05] * 20  # mean = 0.08, all > 0
    m, lo, hi, p = _bootstrap_mean_ci(vals, resamples=2000, seed=42)
    assert m > 0
    assert lo > 0  # CI strictly above zero
    assert p < 0.05


def test_run_decompose_ci_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_decompose_ci(
        ds,
        stage_a="S2_+fact",
        stage_b="S7_full_default",
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        resamples=200,
        seed=42,
    )
    assert rep["stage_a"] == "S2_+fact"
    assert rep["stage_b"] == "S7_full_default"
    assert rep["n_paired"] >= 0  # may be 0 on toy if pipelines diverge
    assert "summary" in rep
    for key in ("delta_h1", "delta_hk", "delta_rr"):
        c = rep["summary"][key]
        assert "mean_diff_a_minus_b" in c
        assert "ci_lo" in c and "ci_hi" in c
        assert "p_bootstrap_two_sided" in c
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0


def test_run_decompose_ci_unknown_stage(tmp_path):
    ds = _toy_dataset(tmp_path)
    import pytest
    with pytest.raises(ValueError):
        run_decompose_ci(ds, stage_a="S99_bogus", stage_b="S7_full_default",
                         max_instances=1, k=5, resamples=10)

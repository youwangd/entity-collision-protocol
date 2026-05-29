"""Tests for §94c-appraisal-bound-CI paired bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_appraisal_bound_ci import (
    _bootstrap_mean_ci,
    _parse_cap,
    render_markdown,
    run_appraisal_bound_ci,
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


def test_parse_cap_handles_none_and_floats():
    assert _parse_cap("none") is None
    assert _parse_cap("None") is None
    assert _parse_cap("") is None
    assert _parse_cap("0.30") == 0.30
    assert _parse_cap("  0.5  ") == 0.5


def test_bootstrap_mean_ci_zero_is_p_one():
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 50, resamples=200, seed=7)
    assert m == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_bootstrap_mean_ci_positive_signal():
    vals = [0.10] * 30 + [0.05] * 20
    m, lo, hi, p = _bootstrap_mean_ci(vals, resamples=2000, seed=42)
    assert m > 0
    assert lo > 0
    assert p < 0.05


def test_run_appraisal_bound_ci_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_ci(
        ds,
        cap_a=0.30,
        cap_b=None,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        resamples=200,
        seed=42,
    )
    assert rep["cap_a"] == 0.30
    assert rep["cap_b"] is None
    assert rep["n_paired"] >= 0
    assert "summary" in rep
    for key in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][key]
        assert "mean_diff_a_minus_b" in c
        assert "ci_lo" in c and "ci_hi" in c
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0


def test_render_markdown_includes_caps_and_table(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_ci(
        ds, cap_a=0.30, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=50, seed=42,
    )
    md = render_markdown(rep)
    assert "appraisal-bound-CI" in md
    assert "cap=0.30" in md and "cap=None" in md
    assert "| metric | mean | 95% CI | p (two-sided) |" in md
    for k_ in ("delta_h1", "delta_hk", "delta_prk", "delta_grk", "delta_rr"):
        assert k_ in md


def test_pairing_is_symmetric_in_summary_keys(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_ci(
        ds, cap_a=None, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=50, seed=42,
    )
    # Same cap on both arms => per-pair diffs should all be exactly zero.
    for k_ in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        assert c["mean_diff_a_minus_b"] == 0.0
        assert c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0
        assert c["p_bootstrap_two_sided"] == 1.0

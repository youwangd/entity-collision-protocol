"""Tests for §94c-appraisal-bound-multihop-CI paired bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_appraisal_bound_multihop_ci import (
    _bootstrap_mean_ci,
    _parse_cap,
    render_markdown,
    run_appraisal_bound_multihop_ci,
)


def _toy_dataset(tmp_path: Path) -> Path:
    """Toy dataset with a single multi-hop question (n_gold=2)."""
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
                 {"speaker": "user", "content": "Alice also loves badminton."},
                 {"speaker": "user", "content": "Badminton is a racquet sport."},
             ]},
        ],
        "qa": [
            {"question": "What does Alice love?", "answer": "apricots",
             "category": "single_session_user", "evidence": ["sess_a"]},
            # Multi-hop: gold = both sessions (n_gold=2)
            {"question": "What are Alice's two interests?",
             "answer": "apricots and badminton",
             "category": "multi_session", "evidence": ["sess_a", "sess_b"]},
        ],
    }]
    p = tmp_path / "locomo_toy_mh.json"
    p.write_text(json.dumps(data))
    return p


def test_parse_cap_handles_none_and_floats():
    assert _parse_cap("none") is None
    assert _parse_cap("None") is None
    assert _parse_cap("") is None
    assert _parse_cap("0.30") == 0.30


def test_bootstrap_mean_ci_zero_is_p_one():
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 50, resamples=200, seed=7)
    assert m == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_run_multihop_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_multihop_ci(
        ds, cap_a=0.30, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=200, seed=42, n_gold_min=2,
    )
    assert rep["cap_a"] == 0.30
    assert rep["cap_b"] is None
    assert rep["n_gold_min"] == 2
    assert rep["n_paired_total"] >= rep["n_paired_multihop"]
    for key in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][key]
        assert "mean_diff_a_minus_b" in c
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
        assert c["n_paired"] == rep["n_paired_multihop"]


def test_same_cap_diffs_are_zero(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_multihop_ci(
        ds, cap_a=None, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=50, seed=42, n_gold_min=2,
    )
    for k_ in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        assert c["mean_diff_a_minus_b"] == 0.0
        assert c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0


def test_n_gold_min_filters(tmp_path):
    ds = _toy_dataset(tmp_path)
    # Setting n_gold_min huge should yield zero paired multihop rows.
    rep = run_appraisal_bound_multihop_ci(
        ds, cap_a=0.30, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=50, seed=42, n_gold_min=999,
    )
    assert rep["n_paired_multihop"] == 0
    for k_ in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        assert rep["summary"][k_]["n_paired"] == 0


def test_render_markdown_has_multihop_section(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_appraisal_bound_multihop_ci(
        ds, cap_a=0.30, cap_b=None,
        max_instances=1, k=5, embedder_name="hashtrigram",
        resamples=50, seed=42, n_gold_min=2,
    )
    md = render_markdown(rep)
    assert "appraisal-bound-multihop-CI" in md
    assert "n_gold≥2" in md
    assert "Multi-hop headline" in md
    assert "| metric | mean | 95% CI | p (two-sided) |" in md

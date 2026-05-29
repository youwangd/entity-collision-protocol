"""Tests for §94d-tau-CI paired bootstrap on schema_synthesis_tau."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_tau_ci import (
    _bootstrap_mean_ci,
    render_markdown,
    run_tau_ci,
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


def test_bootstrap_mean_ci_zero_is_p_one():
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 50, resamples=200, seed=7)
    assert m == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_bootstrap_mean_ci_negative_signal():
    vals = [-0.10] * 30 + [-0.05] * 20
    m, lo, hi, p = _bootstrap_mean_ci(vals, resamples=2000, seed=42)
    assert m < 0
    assert hi < 0
    assert p < 0.05


def test_run_tau_ci_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_tau_ci(
        ds,
        tau_a=0.30, tau_b=0.05,
        min_supports=2,
        max_instances=1, k=5,
        embedder_name="hashtrigram",
        synthesis=True,
        resamples=200,
        seed=42,
    )
    assert rep["tau_a"] == 0.30
    assert rep["tau_b"] == 0.05
    assert rep["min_supports"] == 2
    assert rep["synthesis"] is True
    assert rep["n_paired"] >= 0
    assert "summary" in rep
    for key in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][key]
        assert "mean_diff_a_minus_b" in c
        assert "ci_lo" in c and "ci_hi" in c
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
        assert c["n_paired"] == rep["n_paired"]


def test_same_tau_yields_zero_diffs(tmp_path):
    """Both arms at the same tau ⇒ per-pair diffs identically zero."""
    ds = _toy_dataset(tmp_path)
    rep = run_tau_ci(
        ds, tau_a=0.20, tau_b=0.20,
        min_supports=2,
        max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=True, resamples=50, seed=42,
    )
    for k_ in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        assert c["mean_diff_a_minus_b"] == 0.0
        assert c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0
        assert c["p_bootstrap_two_sided"] == 1.0


def test_synthesis_off_makes_tau_inert(tmp_path):
    """With synthesis=False, schema_synthesis_tau is structurally inert:
    the synthesizer never runs, so any two taus must produce identical
    per-pair diffs."""
    ds = _toy_dataset(tmp_path)
    rep = run_tau_ci(
        ds, tau_a=0.30, tau_b=0.05,
        min_supports=2,
        max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=False, resamples=50, seed=42,
    )
    for k_ in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        assert c["mean_diff_a_minus_b"] == 0.0


def test_render_markdown_shape(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_tau_ci(
        ds, tau_a=0.30, tau_b=0.05, min_supports=2,
        max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=True, resamples=50, seed=42,
    )
    md = render_markdown(rep)
    assert "§94d-tau-CI" in md
    assert "tau=0.30" in md and "tau=0.05" in md
    assert "| metric | mean | 95% CI | p (two-sided) |" in md
    for k_ in ("delta_h1", "delta_hk", "delta_prk", "delta_grk", "delta_rr"):
        assert k_ in md

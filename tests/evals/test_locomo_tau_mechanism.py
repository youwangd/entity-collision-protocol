"""Tests for §94d-mechanism — schema-presence-in-top-k diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_tau_mechanism import (
    _verdict,
    render_markdown,
    run_tau_mechanism,
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


def test_verdict_no_schemas_means_vacuous():
    arms = {
        "tau=0.30": {"frac_topk_with_schema": 0.0, "n_schemas_total": 0,
                     "frac_rank1_schema": 0.0},
        "tau=0.05": {"frac_topk_with_schema": 0.0, "n_schemas_total": 0,
                     "frac_rank1_schema": 0.0},
    }
    msg = _verdict(arms)
    assert "never enter top-k" in msg or "vacuous" in msg


def test_verdict_schemas_in_topk_flags_non_structural():
    arms = {
        "tau=0.30": {"frac_topk_with_schema": 0.13, "n_schemas_total": 7,
                     "frac_rank1_schema": 0.02},
        "tau=0.05": {"frac_topk_with_schema": 0.07, "n_schemas_total": 2,
                     "frac_rank1_schema": 0.01},
    }
    msg = _verdict(arms)
    assert "NOT structural" in msg


def test_run_tau_mechanism_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_tau_mechanism(
        ds, max_instances=1, k=5,
        taus=(0.30, 0.05), min_supports=2,
    )
    assert rep["n_samples"] == 1
    assert "tau=0.3" in rep["arms"] and "tau=0.05" in rep["arms"]
    for a in rep["arms"].values():
        # Both arms see the same questions on the same fixture.
        assert a["n_questions_total"] >= 0
        assert a["n_topk_with_schema_total"] <= a["n_questions_total"]
        assert a["n_rank1_schema_total"] <= a["n_topk_with_schema_total"]
        # Every recorded rank position is within k.
        for r in a["schema_topk_ranks"]:
            assert 1 <= r <= 5


def test_run_tau_mechanism_determinism(tmp_path):
    """Running the same fixture twice with the same tau must produce
    identical aggregates (deterministic harness, no clocks in scoring)."""
    ds = _toy_dataset(tmp_path)
    rep1 = run_tau_mechanism(ds, max_instances=1, k=5,
                             taus=(0.30,), min_supports=2)
    rep2 = run_tau_mechanism(ds, max_instances=1, k=5,
                             taus=(0.30,), min_supports=2)
    a1 = rep1["arms"]["tau=0.3"]
    a2 = rep2["arms"]["tau=0.3"]
    assert a1["n_schemas_total"] == a2["n_schemas_total"]
    assert a1["n_questions_total"] == a2["n_questions_total"]
    assert a1["n_topk_with_schema_total"] == a2["n_topk_with_schema_total"]
    assert a1["n_rank1_schema_total"] == a2["n_rank1_schema_total"]


def test_render_markdown_shape():
    rep = {
        "config": {"dataset_path": "x.json", "max_instances": 2, "k": 10,
                   "embedder": "HashTrigram-256", "taus": [0.30, 0.05],
                   "min_supports": 2},
        "n_samples": 2,
        "wall_seconds": 12.3,
        "arms": {
            "tau=0.3": {"tau": 0.3, "n_schemas_total": 7,
                        "n_questions_total": 301, "n_topk_with_schema_total": 38,
                        "n_rank1_schema_total": 7,
                        "frac_topk_with_schema": 0.1262,
                        "frac_rank1_schema": 0.0233},
            "tau=0.05": {"tau": 0.05, "n_schemas_total": 2,
                         "n_questions_total": 301, "n_topk_with_schema_total": 21,
                         "n_rank1_schema_total": 3,
                         "frac_topk_with_schema": 0.0698,
                         "frac_rank1_schema": 0.01},
        },
        "verdict": "test verdict",
    }
    md = render_markdown(rep)
    assert "§94d-mechanism" in md
    assert "| tau |" in md
    assert "0.1262" in md
    assert "test verdict" in md


def test_no_samples_returns_error(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]")
    rep = run_tau_mechanism(p, max_instances=2)
    assert rep.get("error") == "no samples"

"""LoCoMo adapter sanity tests.

We don't ship the LoCoMo dataset — fixture below mimics the public
snap-research/locomo schema (flat `session_N` + `session_N_date_time`
keys plus a `qa` list with `evidence` items like `D1:5`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.locomo_adapter import (
    _evidence_to_sessions,
    load_locomo,
    run_locomo,
)


def _make_fixture(tmp_path: Path) -> Path:
    data = [
        {
            "sample_id": "loco_1",
            "conversation": {
                "session_1_date_time": "2024-01-01 10:00",
                "session_1": [
                    {"speaker": "Alice", "text": "I bought a red bicycle yesterday", "dia_id": "D1:1"},
                    {"speaker": "Bob", "text": "Cool, what brand?", "dia_id": "D1:2"},
                ],
                "session_2_date_time": "2024-01-08 14:00",
                "session_2": [
                    {"speaker": "Alice", "text": "weather forecast looks rainy this week", "dia_id": "D2:1"},
                ],
                "session_3_date_time": "2024-01-15 09:00",
                "session_3": [
                    {"speaker": "Bob", "text": "I started learning Spanish on Duolingo", "dia_id": "D3:1"},
                    {"speaker": "Alice", "text": "nice, how is it going", "dia_id": "D3:2"},
                ],
            },
            "qa": [
                {
                    "question": "what color was the bicycle Alice bought?",
                    "answer": "red",
                    "category": "single_hop",
                    "evidence": ["D1:1"],
                },
                {
                    "question": "what language is Bob learning?",
                    "answer": "Spanish",
                    "category": "single_hop",
                    "evidence": ["D3:1"],
                },
                # No-evidence question: should be skipped from recall scoring
                {
                    "question": "philosophical musings?",
                    "answer": "n/a",
                    "category": "open_domain",
                    "evidence": [],
                },
            ],
        },
    ]
    p = tmp_path / "locomo_tiny.json"
    p.write_text(json.dumps(data))
    return p


def test_evidence_parser():
    assert _evidence_to_sessions(["D1:1", "D3:5", "D1:7"]) == ["D1", "D3"]
    assert _evidence_to_sessions("D2:4") == ["D2"]
    assert _evidence_to_sessions(["D5"]) == ["D5"]
    assert _evidence_to_sessions(None) == []
    assert _evidence_to_sessions(["nonsense"]) == []


def test_loader_parses_schema(tmp_path):
    p = _make_fixture(tmp_path)
    samples = load_locomo(p)
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "loco_1"
    assert len(s.sessions) == 3
    assert s.sessions[0]["id"] == "D1"
    assert s.sessions[0]["turns"][0]["content"].startswith("I bought")
    assert s.sessions[0]["turns"][0]["speaker"] == "Alice"
    assert len(s.qa) == 3
    assert s.qa[0].evidence_sessions == ["D1"]
    assert s.qa[2].evidence_sessions == []  # no-evidence Q


def test_loader_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_locomo(tmp_path / "nope.json")


def test_loader_max_instances(tmp_path):
    # build 3-sample fixture by copying
    base = json.loads(_make_fixture(tmp_path).read_text())
    big = base + base + base
    p = tmp_path / "locomo_big.json"
    p.write_text(json.dumps(big))
    assert len(load_locomo(p, max_instances=2)) == 2


@pytest.mark.evals
def test_run_locomo_end_to_end_session_recall(tmp_path):
    """Tiny LoCoMo run: gold session should appear in top-k for evidence-bearing QAs.
    The no-evidence QA should be counted in n_questions but not n_questions_scored."""
    p = _make_fixture(tmp_path)
    metrics = run_locomo(p, max_instances=1, k=5)
    assert metrics["n_samples"] == 1
    assert metrics["n_questions"] == 3
    assert metrics["n_questions_scored"] == 2
    # Tiny haystack with strong lexical overlap — expect both scored Qs to hit
    assert metrics["session_hit_at_k"] == 1.0
    assert "single_hop" in metrics["per_category_session_hit_at_k"]
    assert metrics["per_category_n"]["single_hop"] == 2
    # open_domain category had no evidence — should not appear in scored cats
    assert "open_domain" not in metrics["per_category_session_hit_at_k"]
    assert "p50" in metrics["ingest_ms"]
    assert "p50" in metrics["recall_ms"]

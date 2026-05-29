"""LongMemEval adapter sanity tests.

We don't ship the real dataset, so the test builds a tiny LME-shaped JSON
fixture and runs the full ingest+retrieve loop. Asserts:
  - loader handles the public schema
  - session_id metadata is preserved through ingest -> recall
  - per-type aggregation works
  - the harness skips cleanly when the dataset path is missing
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.longmemeval_adapter import load_longmemeval, run_lme


def _make_fixture(tmp_path: Path) -> Path:
    """Construct a 2-instance LME-shaped JSON file."""
    data = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "what is alice's favorite editor?",
            "answer": "vim",
            "haystack_session_ids": ["sA", "sB", "sC"],
            "haystack_dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "haystack_sessions": [
                # sA: noise
                [{"role": "user", "content": "discussed quarterly planning"},
                 {"role": "assistant", "content": "ok noted"}],
                # sB: gold — contains the answer
                [{"role": "user", "content": "alice told me her favorite editor is vim"},
                 {"role": "assistant", "content": "got it"}],
                # sC: noise
                [{"role": "user", "content": "weather was nice yesterday"}],
            ],
            "answer_session_ids": ["sB"],
        },
        {
            "question_id": "q2",
            "question_type": "temporal-reasoning",
            "question": "when does bob's stripe key expire?",
            "answer": "2026-07-15",
            "haystack_session_ids": ["t1", "t2"],
            "haystack_dates": ["2024-02-01", "2024-02-02"],
            "haystack_sessions": [
                [{"role": "user", "content": "bob's stripe credential lapses on 2026-07-15"}],
                [{"role": "user", "content": "unrelated meeting notes about coffee"}],
            ],
            "answer_session_ids": ["t1"],
        },
    ]
    p = tmp_path / "lme_tiny.json"
    p.write_text(json.dumps(data))
    return p


def test_loader_parses_schema(tmp_path):
    p = _make_fixture(tmp_path)
    insts = load_longmemeval(p)
    assert len(insts) == 2
    assert insts[0].question_id == "q1"
    assert insts[0].answer_session_ids == ["sB"]
    assert len(insts[0].sessions) == 3
    assert insts[0].sessions[1]["id"] == "sB"
    assert insts[0].sessions[1]["turns"][0]["content"].startswith("alice")


def test_loader_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_longmemeval(tmp_path / "does_not_exist.json")


def test_loader_max_instances(tmp_path):
    p = _make_fixture(tmp_path)
    insts = load_longmemeval(p, max_instances=1)
    assert len(insts) == 1


@pytest.mark.evals
def test_run_lme_end_to_end_session_recall(tmp_path):
    """Tiny LME run: gold session should be retrieved at top-k for both Qs."""
    p = _make_fixture(tmp_path)
    metrics = run_lme(p, max_instances=2, k=5)
    assert metrics["n_instances"] == 2
    # Tiny haystack — strong lexical overlap means we expect hit@k on both
    assert metrics["session_hit_at_k"] == 1.0
    # Per-type aggregation
    assert "single-session-user" in metrics["per_type_session_hit_at_k"]
    assert "temporal-reasoning" in metrics["per_type_session_hit_at_k"]
    assert metrics["per_type_n"] == {"single-session-user": 1, "temporal-reasoning": 1}
    # Ingest/recall latency reported
    assert "p50" in metrics["ingest_ms"]
    assert "p50" in metrics["recall_ms"]

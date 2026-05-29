"""Unit tests for evals.locomo_c5_adversarial.

Covers:
  - _content_tokens stoplist + length filter behavior.
  - run() produces per-vw aligned per-query rows with c5-only filter.
  - aggregate keys present.
  - top1 records carry adv_overlap fields.

We use a tiny synthetic LoCoMo sample (single conv, two c5 questions plus
a non-c5 distractor) to keep the test fast (<2s).
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_c5_adversarial import _content_tokens, run


def test_content_tokens_filters_stop_and_short_words() -> None:
    toks = _content_tokens("The quick brown fox is just a test")
    # short words and stoplist items should be removed
    assert "the" not in toks and "is" not in toks and "just" not in toks
    assert "quick" in toks
    assert "brown" in toks


def test_content_tokens_lowercases() -> None:
    assert "lgbtq" in _content_tokens("LGBTQ individuals")


def _fake_locomo(tmp: Path) -> Path:
    sample = {
        "sample_id": "tiny",
        "conversation": {
            "session_1_date_time": "2024-01-01",
            "session_1": [
                {"speaker": "Alice", "text": "I went horseback riding with my father.",
                 "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "Cool, I prefer cycling.", "dia_id": "D1:2"},
            ],
            "session_2_date_time": "2024-01-02",
            "session_2": [
                {"speaker": "Alice", "text": "My grandma gave me a necklace as a gift.",
                 "dia_id": "D2:1"},
            ],
        },
        "qa": [
            {"question": "What activity did Alice do with her dad?",
             "answer": None, "category": 5, "evidence": ["D1:1"],
             "adversarial_answer": "Horseback riding"},
            {"question": "What was grandma's gift?",
             "answer": None, "category": 5, "evidence": ["D2:1"],
             "adversarial_answer": "necklace"},
            # non-c5 should be filtered out
            {"question": "ignored", "answer": "x", "category": 1,
             "evidence": ["D1:1"], "adversarial_answer": None},
        ],
    }
    p = tmp / "tiny.json"
    p.write_text(json.dumps([sample]))
    return p


def test_run_produces_aligned_per_vw_rows(tmp_path: Path) -> None:
    ds = _fake_locomo(tmp_path)
    res = run(str(ds), vws=[0.0, 0.5], max_instances=1, k=5,
              embedder="hashtrigram")
    summ = res["per_vw_summary"]
    assert "0.0" in summ and "0.5" in summ
    # only c5 rows
    assert summ["0.0"]["n"] == 2
    assert summ["0.5"]["n"] == 2

    pq0 = res["per_query"][0.0]
    pq5 = res["per_query"][0.5]
    assert len(pq0) == len(pq5) == 2
    # row alignment: same question at same index
    for r0, r5 in zip(pq0, pq5):
        assert r0["question"] == r5["question"]
        assert "adv_overlap_tokens" in r0
        assert "adv_overlap_frac" in r0
        assert "top1_session" in r0
        assert "bm25_score" in r0


def test_run_aggregates_have_required_keys(tmp_path: Path) -> None:
    ds = _fake_locomo(tmp_path)
    res = run(str(ds), vws=[0.0], max_instances=1, k=5, embedder="hashtrigram")
    cell = res["per_vw_summary"]["0.0"]
    for k in ("n", "hit_at_1", "n_misses",
              "miss_with_adv_overlap_ge_0.5",
              "miss_adv_overlap_rate",
              "miss_top1_in_gold_session"):
        assert k in cell

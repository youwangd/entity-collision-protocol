"""Tests for §94c-appraisal-rerank-inspect driver."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_appraisal_rerank_inspect import (
    PROBE_STAGE,
    S6_NAME,
    _bin,
    _displacing,
    _stages_with,
    render_markdown,
    run_appraisal_rerank_inspect,
)
from evals.locomo_recall_lift_decompose_ci import SUBSET_PRESETS


def _toy(tmp_path: Path) -> Path:
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


def test_stages_with_probe_appends_appraisal():
    base = list(SUBSET_PRESETS[S6_NAME])
    s = _stages_with(PROBE_STAGE)
    assert s[: len(base)] == base
    assert s[-1] == PROBE_STAGE
    assert s.count(PROBE_STAGE) == 1


def test_stages_with_none_returns_s6():
    assert _stages_with(None) == list(SUBSET_PRESETS[S6_NAME])


def test_stages_with_idempotent_when_already_present():
    sample_stage = SUBSET_PRESETS[S6_NAME][0]
    assert _stages_with(sample_stage) == list(SUBSET_PRESETS[S6_NAME])


def test_bin_classifies_canonical_movements():
    k = 10
    assert _bin(1, 1, k) == "stable_rank1"
    assert _bin(1, 3, k) == "lost_rank1"
    assert _bin(3, 1, k) == "gained_rank1"
    assert _bin(0, 5, k) == "entered_topk"
    assert _bin(5, 0, k) == "left_topk"
    assert _bin(5, 3, k) == "improved_within_topk"
    assert _bin(3, 5, k) == "worsened_within_topk"
    assert _bin(0, 0, k) == "absent_both"
    # Out-of-topk pair (>k both) -> absent_both
    assert _bin(15, 20, k) == "absent_both"


def test_displacing_returns_none_when_b_top1_is_gold():
    a = [{"session_id": "g", "salience": 0.5, "rel": 1.0, "nov": 1.0, "gc": 1.0,
          "memory_id": "1", "score": 0.9, "content_head": ""}]
    b = [{"session_id": "g", "salience": 0.5, "rel": 1.0, "nov": 1.0, "gc": 1.0,
          "memory_id": "1", "score": 0.9, "content_head": ""}]
    assert _displacing(a, b, gold={"g"}) is None


def test_displacing_returns_record_when_nongold_at_b_rank1():
    a = [{"session_id": "g", "salience": 0.4, "rel": 1.0, "nov": 1.0, "gc": 1.0,
          "memory_id": "1", "score": 0.9, "content_head": ""}]
    b = [{"session_id": "x", "salience": 0.7, "rel": 1.5, "nov": 1.0, "gc": 1.0,
          "memory_id": "2", "score": 0.95, "content_head": ""},
         {"session_id": "g", "salience": 0.4, "rel": 1.0, "nov": 1.0, "gc": 1.0,
          "memory_id": "1", "score": 0.9, "content_head": ""}]
    rec = _displacing(a, b, gold={"g"})
    assert rec is not None
    assert rec["displacing"]["session_id"] == "x"
    assert rec["displaced_gold_in_a"]["session_id"] == "g"
    assert abs(rec["salience_gap"] - 0.3) < 1e-6


def test_run_inspect_smoke(tmp_path):
    ds = _toy(tmp_path)
    rep = run_appraisal_rerank_inspect(
        str(ds), max_instances=1, k=5, embedder_name="hashtrigram",
    )
    assert rep["anchor"] == S6_NAME
    assert rep["probe_stage"] == PROBE_STAGE
    assert "aggregate" in rep
    assert "movement_bins_overall" in rep["aggregate"]
    assert isinstance(rep["per_question"], list)
    assert len(rep["per_question"]) == rep["n_questions"]
    # Each per-question record carries the topk arrays.
    for q in rep["per_question"]:
        assert "topk_a" in q and "topk_b" in q
        assert "movement_bin" in q


def test_render_markdown_smoke(tmp_path):
    ds = _toy(tmp_path)
    rep = run_appraisal_rerank_inspect(
        str(ds), max_instances=1, k=5, embedder_name="hashtrigram",
    )
    md = render_markdown(rep)
    assert "appraisal" in md
    assert "movement_bin" in md


def _synthetic_report() -> dict:
    """Hand-built report for deterministic md rendering checks."""
    return {
        "dataset_path": "x",
        "max_instances": 0,
        "k": 10,
        "embedder": "test",
        "anchor": S6_NAME,
        "probe_stage": PROBE_STAGE,
        "n_questions": 0,
        "aggregate": {
            "movement_bins_overall": {
                "stable_rank1": 4, "lost_rank1": 3, "gained_rank1": 1,
                "absent_both": 2,
            },
            "movement_bins_by_category": {
                "single_session_user": {
                    "stable_rank1": 2, "lost_rank1": 3, "gained_rank1": 0,
                    "absent_both": 1,
                },
                "multi_session": {
                    "stable_rank1": 2, "lost_rank1": 0, "gained_rank1": 1,
                    "absent_both": 1,
                },
            },
            "salience_gap_displacing_minus_gold": {"n": 0},
            "rel_gap_displacing_minus_gold": {"n": 0},
            "displacing_item_salience": {"n": 0},
            "displaced_gold_salience": {"n": 0},
        },
        "per_question": [],
        "wall_seconds": 0.0,
    }


def test_render_markdown_includes_category_breakdown_table():
    md = render_markdown(_synthetic_report())
    # Category-breakdown header rendered.
    assert "Movement bins by category" in md
    # Both categories appear.
    assert "single_session_user" in md
    assert "multi_session" in md
    # Stable_rank1 header column present (most-frequent overall bin).
    assert "`stable_rank1`" in md


def test_render_markdown_lost_vs_gained_table_shows_asymmetry():
    md = render_markdown(_synthetic_report())
    assert "Lost vs gained rank-1 by category" in md
    # single_session_user: lost=3, gained=0 -> net -3
    # multi_session:        lost=0, gained=1 -> net +1
    # Net columns are formatted with sign.
    assert "-3" in md
    assert "+1" in md


def test_render_markdown_skips_lost_gained_table_when_no_movements():
    rep = _synthetic_report()
    # Zero out lost/gained across categories.
    for cat in rep["aggregate"]["movement_bins_by_category"].values():
        cat["lost_rank1"] = 0
        cat["gained_rank1"] = 0
    md = render_markdown(rep)
    # Category breakdown table still rendered (other bins present).
    assert "Movement bins by category" in md
    # But lost-vs-gained section is suppressed when both columns are zero.
    assert "Lost vs gained rank-1 by category" not in md

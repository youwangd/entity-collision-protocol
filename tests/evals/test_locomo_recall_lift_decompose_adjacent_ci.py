"""Tests for §94c-decompose-adjacent-CI per-transition bootstrap CI."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_adjacent_ci import (
    METRIC_KEYS,
    SUBSET_ORDER,
    _added_stage,
    _bootstrap_mean_ci,
    _pair_diffs,
    render_markdown,
    run_adjacent_ci,
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


def test_subset_order_matches_decompose_default():
    """Adjacent driver's stage list must mirror the cumulative driver's."""
    from evals.locomo_recall_lift_decompose import DEFAULT_SUBSETS
    expected = [name for name, _ in DEFAULT_SUBSETS]
    assert SUBSET_ORDER == expected


def test_added_stage_cumulative_one_per_step():
    """Each adjacent transition before S7 adds exactly one stage."""
    for i in range(len(SUBSET_ORDER) - 2):  # exclude S6->S7 (None target)
        a, b = SUBSET_ORDER[i], SUBSET_ORDER[i + 1]
        added = _added_stage(a, b)
        assert added is not None
        # Cumulative subsets: target = source + exactly one new stage.
        assert "," not in added, f"{a} -> {b} added more than one stage: {added}"


def test_added_stage_full_default_is_marked():
    """S6 -> S7 transitions to None (full default) and is marked specially."""
    added = _added_stage("S6_+merge_persist", "S7_full_default")
    assert added == "(implicit_full_default)"


def test_pair_diffs_aligns_on_question_key():
    """_pair_diffs must align on (sample_id, question, category) and skip
    rows with no match in the other arm."""
    pairs_a = [
        {"sample_id": "S1", "question": "Q1", "category": "c",
         "delta_h1": 1.0, "delta_hk": 0.5, "delta_rr": 0.3,
         "delta_prk": 0.2, "delta_grk": 0.1},
        {"sample_id": "S1", "question": "Q2", "category": "c",
         "delta_h1": 0.0, "delta_hk": 0.0, "delta_rr": 0.0,
         "delta_prk": 0.0, "delta_grk": 0.0},
    ]
    pairs_b = [
        {"sample_id": "S1", "question": "Q1", "category": "c",
         "delta_h1": 0.5, "delta_hk": 0.4, "delta_rr": 0.2,
         "delta_prk": 0.1, "delta_grk": 0.05},
        # Q2 missing -> dropped from pairing.
    ]
    diffs, paired = _pair_diffs(pairs_a, pairs_b)
    assert paired == 1
    assert diffs["delta_h1"] == [0.5]   # 1.0 - 0.5
    assert diffs["delta_grk"] == [0.05]  # 0.1 - 0.05
    for k in METRIC_KEYS:
        assert len(diffs[k]) == 1


def test_bootstrap_mean_ci_constant_zero_is_p_one():
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 30, resamples=200, seed=1)
    assert m == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_run_adjacent_ci_smoke_and_render(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_adjacent_ci(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        resamples=200,
        seed=42,
    )
    # 7 subsets -> 6 adjacent transitions.
    assert len(rep["transitions"]) == len(SUBSET_ORDER) - 1
    for t in rep["transitions"]:
        assert "summary" in t
        for mk in METRIC_KEYS:
            c = t["summary"][mk]
            assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
            assert c["ci_lo"] <= c["ci_hi"]
            assert c["ci_excludes_zero"] in (True, False)
    # Markdown table renders.
    md = render_markdown(rep)
    assert "decompose-adjacent-CI" in md
    for name in SUBSET_ORDER[:-1]:
        assert name in md

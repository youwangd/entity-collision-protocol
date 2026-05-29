"""Tests for §94c-decompose-suffix-CI driver."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_ci import SUBSET_PRESETS
from evals.locomo_recall_lift_decompose_suffix_ci import (
    METRIC_KEYS,
    S6_NAME,
    S7_NAME,
    SUFFIX_STAGES,
    _s6_plus,
    render_markdown,
    run_suffix_ci,
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


def test_suffix_stages_match_s7_minus_s6():
    """The 7 suffix stages == default S7 stage list minus S6's."""
    # S6 explicit list.
    s6 = set(SUBSET_PRESETS[S6_NAME])
    # Default S7 pipeline (manually mirrored from pipeline.py for the
    # invariant — if pipeline.py changes, this guards the suffix list).
    full_default = {
        "replay", "deduplication", "extraction", "fact_extraction",
        "appraisal", "emotion_tagging", "interference", "schema_update",
        "somatic_marking", "decay", "suppression", "temperament_drift",
        "mood_update", "mechanical_merge", "persistence",
    }
    # 'replay' and 'persistence' are mandatory and added implicitly;
    # exclude them from the user-visible suffix.
    expected = (full_default - s6) - {"replay", "persistence"}
    assert set(SUFFIX_STAGES) == expected, (
        f"SUFFIX_STAGES drifted; expected {expected}, got {set(SUFFIX_STAGES)}"
    )
    # Exactly 7 stages bundled into S6→S7 (matches NEXT.md attribution).
    assert len(SUFFIX_STAGES) == 7


def test_s6_plus_appends_one_stage():
    """_s6_plus(stage) returns S6's list with `stage` appended exactly once."""
    base = list(SUBSET_PRESETS[S6_NAME])
    for stage in SUFFIX_STAGES:
        plus = _s6_plus(stage)
        assert plus[: len(base)] == base
        assert plus[-1] == stage
        assert plus.count(stage) == 1
        assert len(plus) == len(base) + 1


def test_s6_plus_idempotent_when_already_present():
    """If the stage is already in S6, _s6_plus is a no-op."""
    sample = SUBSET_PRESETS[S6_NAME][0]
    plus = _s6_plus(sample)
    # No duplicate appended.
    assert plus == list(SUBSET_PRESETS[S6_NAME])


def test_run_suffix_ci_smoke_on_toy_dataset(tmp_path):
    """Driver runs end-to-end on a toy dataset with cheap CI settings."""
    ds = _toy_dataset(tmp_path)
    rep = run_suffix_ci(
        str(ds),
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        resamples=64,
        seed=0,
    )
    # Top-level shape.
    assert rep["anchor"] == S6_NAME
    assert rep["suffix_stages"] == SUFFIX_STAGES
    assert len(rep["transitions"]) == 7
    # Bundle row present and well-formed.
    assert rep["bundle"]["transition"] == f"{S6_NAME} -> {S7_NAME}"
    for mk in METRIC_KEYS:
        assert mk in rep["bundle"]["summary"]
        s = rep["bundle"]["summary"][mk]
        assert "mean_diff_a_minus_b" in s
        assert "ci_lo" in s and "ci_hi" in s
        assert "ci_excludes_zero" in s
    # Each per-stage probe carries summary on every metric.
    seen = set()
    for t in rep["transitions"]:
        seen.add(t["added_stage"])
        for mk in METRIC_KEYS:
            assert mk in t["summary"]
    assert seen == set(SUFFIX_STAGES)


def test_render_markdown_includes_every_probe_and_bundle(tmp_path):
    """Markdown render covers all 7 probes plus the bundle reference row."""
    ds = _toy_dataset(tmp_path)
    rep = run_suffix_ci(
        str(ds), max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=False, resamples=32, seed=0,
    )
    md = render_markdown(rep)
    for stage in SUFFIX_STAGES:
        assert stage in md, f"{stage} missing from md"
    assert "(bundle)" in md
    assert "★ = 95% CI excludes zero" in md

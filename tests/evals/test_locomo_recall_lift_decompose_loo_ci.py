"""Tests for §94c-decompose-LOO-CI driver."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_loo_ci import (
    METRIC_KEYS,
    S7_NAME,
    S7_STAGES_DROPPABLE,
    _s7_minus,
    render_markdown,
    run_loo_ci,
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


def test_droppable_stages_excludes_mandatory():
    """`replay` and `persistence` are forced back in by the manager and
    must not appear in the LOO list — dropping them is operationally
    meaningless.
    """
    assert "replay" not in S7_STAGES_DROPPABLE
    assert "persistence" not in S7_STAGES_DROPPABLE
    # Should be 13 droppable stages (15 default minus 2 mandatory).
    assert len(S7_STAGES_DROPPABLE) == 13


def test_droppable_stages_match_default_pipeline():
    """Mirror the default S7 pipeline as a structural invariant — if
    `pipeline.py` adds a new stage, this test fails loud.
    """
    full_default = {
        "replay", "deduplication", "extraction", "fact_extraction",
        "appraisal", "emotion_tagging", "interference", "schema_update",
        "somatic_marking", "decay", "suppression", "temperament_drift",
        "mood_update", "mechanical_merge", "persistence",
    }
    expected = full_default - {"replay", "persistence"}
    assert set(S7_STAGES_DROPPABLE) == expected


def test_s7_minus_drops_one_stage():
    """_s7_minus(stage) removes exactly the named stage."""
    for stage in S7_STAGES_DROPPABLE:
        out = _s7_minus(stage)
        assert stage not in out
        assert len(out) == len(S7_STAGES_DROPPABLE) - 1
        # All other droppable stages preserved.
        assert set(out) == set(S7_STAGES_DROPPABLE) - {stage}


def test_s7_minus_unknown_stage_is_noop():
    """Asking to drop a stage not in S7 leaves the list unchanged."""
    out = _s7_minus("nonexistent_stage")
    assert out == list(S7_STAGES_DROPPABLE)


def test_run_loo_ci_smoke_on_toy_dataset(tmp_path):
    """Driver runs end-to-end on a toy dataset with cheap CI settings.

    Cost-shaped: 13 LOO probes + 1 anchor = 14 full recall_lift runs.
    Toy dataset (1 sample, 4 turns, 2 QAs) keeps each run sub-second.
    """
    ds = _toy_dataset(tmp_path)
    rep = run_loo_ci(
        str(ds),
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        resamples=32,
        seed=0,
    )
    # Top-level shape.
    assert rep["anchor"] == S7_NAME
    assert rep["droppable_stages"] == S7_STAGES_DROPPABLE
    assert len(rep["transitions"]) == len(S7_STAGES_DROPPABLE)
    # Anchor headline echoed.
    assert "delta_h1" in rep["headline_anchor"]
    # Each probe row covers every metric, and the dropped stage is
    # exactly one of S7_STAGES_DROPPABLE.
    seen: set[str] = set()
    for t in rep["transitions"]:
        seen.add(t["dropped_stage"])
        assert t["transition"].startswith(f"S7-{t['dropped_stage']}")
        for mk in METRIC_KEYS:
            assert mk in t["summary"]
            s = t["summary"][mk]
            assert "mean_diff_a_minus_b" in s
            assert "ci_lo" in s and "ci_hi" in s
            assert "ci_excludes_zero" in s
    assert seen == set(S7_STAGES_DROPPABLE)


def test_render_markdown_includes_every_dropped_stage(tmp_path):
    """Markdown render covers all LOO probes and the legend."""
    ds = _toy_dataset(tmp_path)
    rep = run_loo_ci(
        str(ds), max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=False, resamples=16, seed=0,
    )
    md = render_markdown(rep)
    for stage in S7_STAGES_DROPPABLE:
        assert f"`{stage}`" in md, f"{stage} missing from md"
    assert "★ = 95% CI excludes zero" in md
    assert "leave-one-out necessity" in md


def test_loo_no_lift_drop_is_a_negative_signal(tmp_path):
    """Sanity: on the toy fixture (where consolidation does very
    little), the LOO CIs should mostly bracket zero — i.e. no stage is
    individually load-bearing on a 4-turn corpus. We don't assert
    *every* CI brackets zero (FTS quirks can produce small movements);
    we just confirm that a random LOO drop does not always exclude
    zero.
    """
    ds = _toy_dataset(tmp_path)
    rep = run_loo_ci(
        str(ds), max_instances=1, k=5, embedder_name="hashtrigram",
        synthesis=False, resamples=64, seed=0,
    )
    excludes = []
    for t in rep["transitions"]:
        for mk in METRIC_KEYS:
            excludes.append(t["summary"][mk]["ci_excludes_zero"])
    # If *every* CI excluded zero, something is wrong with the
    # bootstrap. On a tiny dataset most should bracket zero.
    assert not all(excludes), (
        "every LOO CI excluded zero on a 2-QA toy dataset — "
        "bootstrap sanity check failed"
    )

"""Tests for §94c-decompose stage bisection driver."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose import (
    DEFAULT_SUBSETS,
    render_markdown,
    run_decompose,
)


def _toy_dataset(tmp_path: Path) -> Path:
    """A 1-sample LoCoMo-shaped fixture, sufficient to exercise wiring."""
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
            {"question": "What does Alice love?",
             "answer": "apricots",
             "category": "single_session_user",
             "evidence": ["sess_a"]},
            {"question": "What does Bob enjoy?",
             "answer": "badminton",
             "category": "single_session_user",
             "evidence": ["sess_b"]},
        ],
    }]
    p = tmp_path / "locomo_toy.json"
    p.write_text(json.dumps(data))
    return p


def test_default_subsets_form_a_chain():
    """S1..S6 should be cumulative (each adds one stage); S7 = full."""
    prev: list[str] | None = []
    for name, stages in DEFAULT_SUBSETS[:-1]:
        assert stages is not None
        # cumulative growth
        assert prev is None or set(prev).issubset(set(stages)), (
            f"{name} broke the chain: prev={prev} stages={stages}"
        )
        prev = stages
    assert DEFAULT_SUBSETS[-1][1] is None  # full default at the end


def test_run_decompose_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    # Just two subsets to keep this fast; full sweep is exercised at runtime
    rep = run_decompose(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=False,
        subsets=[
            ("S2_+fact", ["extraction", "fact_extraction"]),
            ("S7_full",  None),
        ],
    )
    assert "rows" in rep
    assert len(rep["rows"]) == 2
    for r in rep["rows"]:
        assert "error" not in r, r
        # Either we have pairs, or n_pairs is zero — both shapes are valid
        # for a tiny fixture, but the driver must not crash.
        assert "delta_h1" in r
        assert "delta_hk" in r
        assert "delta_mrr" in r
        assert isinstance(r["wall_seconds"], (int, float))


def test_render_markdown_shape(tmp_path):
    rep = {
        "dataset_path": "x",
        "max_instances": 2,
        "k": 10,
        "embedder": "hashtrigram",
        "synthesis": False,
        "wall_seconds": 1.0,
        "rows": [
            {"name": "S1", "stages": ["extraction"], "n_pairs": 100,
             "delta_h1": 0.05, "delta_hk": 0.10, "delta_mrr": 0.07,
             "delta_prk_overall": 0.06, "delta_grk_overall": 0.08,
             "delta_prk_multihop": -0.02, "delta_grk_multihop": -0.01,
             "n_multihop": 20, "wall_seconds": 12.3,
             "n_consolidation_errors": 0},
        ],
    }
    md = render_markdown(rep)
    assert "§94c-decompose" in md
    assert "S1" in md
    # well-formed markdown table: header + sep + 1 row
    lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(lines) == 3


def test_run_decompose_propagates_synthesis_flag(tmp_path):
    """`synthesis=True` must round-trip through report metadata."""
    ds = _toy_dataset(tmp_path)
    rep = run_decompose(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        synthesis=True,
        subsets=[("S2_+fact", ["extraction", "fact_extraction"])],
    )
    assert rep["synthesis"] is True

"""Tests for §94c-decompose-positive-control driver.

The driver re-runs the cumulative ``DEFAULT_SUBSETS`` with synthesis=on
and a sweep of ``schema_synthesis_tau`` to test whether forcing
SCHEMA writes makes the schema-family gate stages move retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_positive_control import (
    DEFAULT_TAUS,
    render_markdown,
    run_positive_control,
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
                 {"speaker": "user", "content": "Badminton is a sport."},
             ]},
        ],
        "qa": [
            {"question": "What does Alice love?", "answer": "apricots",
             "category": "single_session_user", "evidence": ["sess_a"]},
        ],
    }]
    p = tmp_path / "locomo_toy.json"
    p.write_text(json.dumps(data))
    return p


def test_default_taus_strictly_decreasing():
    assert all(
        DEFAULT_TAUS[i] > DEFAULT_TAUS[i + 1]
        for i in range(len(DEFAULT_TAUS) - 1)
    )
    # min_supports lower bound: synthesis cannot fire below 2 supports
    # by definition of "cluster of like things".
    assert DEFAULT_TAUS[0] <= 0.5
    assert DEFAULT_TAUS[-1] > 0.0


def test_run_positive_control_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_positive_control(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        taus=(0.30, 0.05),
        min_supports=2,
        subsets=[
            ("S1_extraction_only", ["extraction"]),
            ("S7_full",            None),
        ],
    )
    assert "rows" in rep
    # 2 taus × 2 subsets = 4 rows
    assert len(rep["rows"]) == 4
    for r in rep["rows"]:
        assert "error" not in r, r
        assert "delta_h1" in r
        assert "delta_hk" in r
        assert "delta_grk" in r
        assert isinstance(r["tau"], float)
        assert isinstance(r["wall_seconds"], (int, float))
    assert rep["taus"] == [0.30, 0.05]
    assert rep["min_supports"] == 2


def test_render_markdown_shape():
    rep = {
        "dataset_path": "x",
        "max_instances": 2,
        "k": 10,
        "embedder": "hashtrigram",
        "min_supports": 2,
        "taus": [0.30, 0.05],
        "subsets": ["S1_extraction_only", "S7_full_default"],
        "wall_seconds": 1.0,
        "rows": [
            {"tau": 0.30, "subset": "S1_extraction_only",
             "stages": ["extraction"], "n_pairs": 100,
             "delta_h1": 0.05, "delta_hk": 0.10, "delta_mrr": 0.07,
             "delta_prk": 0.06, "delta_grk": 0.08,
             "n_consolidation_errors": 0, "wall_seconds": 12.3},
            {"tau": 0.30, "subset": "S7_full_default",
             "stages": None, "n_pairs": 100,
             "delta_h1": 0.04, "delta_hk": 0.10, "delta_mrr": 0.06,
             "delta_prk": 0.05, "delta_grk": 0.07,
             "n_consolidation_errors": 0, "wall_seconds": 12.0},
        ],
    }
    md = render_markdown(rep)
    assert "§94c-decompose-positive-control" in md
    assert "0.30" in md
    assert "S1_extraction_only" in md
    assert "S7_full_default" in md
    # header + sep + 2 rows
    lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(lines) == 4


def test_run_positive_control_threads_synthesis(tmp_path):
    """The driver must hard-wire synthesis=True; tau must round-trip."""
    ds = _toy_dataset(tmp_path)
    rep = run_positive_control(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        taus=(0.10,),
        min_supports=2,
        subsets=[("S2_+fact", ["extraction", "fact_extraction"])],
    )
    assert rep["taus"] == [0.10]
    assert rep["rows"][0]["tau"] == 0.10
    assert rep["rows"][0]["subset"] == "S2_+fact"


def test_render_markdown_includes_error_rows():
    rep = {
        "dataset_path": "x", "max_instances": 1, "k": 5,
        "embedder": "hashtrigram", "min_supports": 2,
        "taus": [0.05], "subsets": ["S1"], "wall_seconds": 0.1,
        "rows": [{"tau": 0.05, "subset": "S1", "stages": ["extraction"],
                  "error": "boom"}],
    }
    md = render_markdown(rep)
    assert "error: boom" in md

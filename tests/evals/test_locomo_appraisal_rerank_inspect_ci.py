"""Tests for §94c-appraisal-inspect-CI driver."""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_appraisal_rerank_inspect_ci import (
    _bootstrap_p,
    _permutation_lost_minus_gained,
    _two_sided_p_from_diffs,
    render_markdown,
    run_ci,
)


def _fake_per_q() -> list[dict]:
    """Synthetic per_question rows that exercise every code path."""
    rows: list[dict] = []
    # 3 stable_rank1
    for _ in range(3):
        rows.append({"movement_bin": "stable_rank1"})
    # 2 lost_rank1 with displacing items (positive salience gap)
    for sg, rg in [(0.5, 0.2), (0.3, 0.1)]:
        rows.append({
            "movement_bin": "lost_rank1",
            "displacing": {
                "displacing": {"salience": sg, "rel": rg},
                "displaced_gold_in_a": {"salience": 0.0, "rel": 0.0},
                "salience_gap": sg,
            },
        })
    # 1 gained_rank1 with no displacing (gold is rank1)
    rows.append({"movement_bin": "gained_rank1", "displacing": None})
    # 1 worsened, 1 improved
    rows.append({"movement_bin": "worsened_within_topk"})
    rows.append({"movement_bin": "improved_within_topk"})
    return rows


def test_bootstrap_p_zero_mean_returns_one():
    # All zeros → mean=0 → returns 1.0.
    assert _bootstrap_p([0.0] * 10, resamples=200, seed=1) == 1.0


def test_bootstrap_p_strong_signal_low_p():
    # Strong positive signal — p should be small.
    p = _bootstrap_p([0.5] * 50, resamples=500, seed=1)
    assert p < 0.05


def test_two_sided_p_symmetric():
    # Half positive, half negative — two-sided p ≈ 1.
    p = _two_sided_p_from_diffs([+1, +1, -1, -1])
    assert p == 1.0


def test_two_sided_p_all_positive():
    p = _two_sided_p_from_diffs([+1, +1, +1, +1])
    assert p == 0.0


def test_permutation_returns_smoothed_pvalue_in_range():
    rows = _fake_per_q()
    perm = _permutation_lost_minus_gained(rows, permutations=200, seed=42)
    assert perm["observed_lost_minus_gained"] == 2 - 1  # 2 lost, 1 gained
    assert 0.0 < perm["p_value_two_sided"] <= 1.0
    assert perm["permutations"] == 200


def test_run_ci_end_to_end(tmp_path: Path):
    # Build a minimal inspect-format artifact and feed it to run_ci.
    artifact = {
        "anchor": "S6_+merge_persist",
        "probe_stage": "appraisal",
        "embedder": "HashTrigram-256",
        "max_instances": 2,
        "n_questions": 8,
        "k": 10,
        "per_question": _fake_per_q(),
    }
    p = tmp_path / "inspect.json"
    p.write_text(json.dumps(artifact))
    rep = run_ci(str(p), resamples=200, permutations=200, seed=42)
    assert rep["n_questions"] == 8
    sg = rep["salience_gap_ci"]
    assert sg["n"] == 2
    assert sg["mean"] > 0  # both displacing items are positive
    perm = rep["permutation_lost_minus_gained"]
    assert "p_value_two_sided" in perm
    md = render_markdown(rep)
    assert "§94c-appraisal-inspect-CI" in md
    assert "Salience gap" in md
    assert "Permutation" in md


def test_run_ci_handles_no_displacing(tmp_path: Path):
    artifact = {
        "anchor": "x",
        "probe_stage": "y",
        "embedder": "z",
        "max_instances": 1,
        "n_questions": 2,
        "k": 5,
        "per_question": [
            {"movement_bin": "stable_rank1"},
            {"movement_bin": "absent_both"},
        ],
    }
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(artifact))
    rep = run_ci(str(p), resamples=50, permutations=50, seed=1)
    assert rep["salience_gap_ci"] == {"n": 0}
    assert rep["rel_gap_ci"] == {"n": 0}
    assert rep["permutation_lost_minus_gained"]["observed_lost_minus_gained"] == 0

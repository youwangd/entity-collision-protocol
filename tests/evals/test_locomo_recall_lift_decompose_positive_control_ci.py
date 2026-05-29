"""Tests for §94c-decompose-positive-control-CI driver.

The driver bootstraps the per-pair (Δ_S4 − Δ_S3) signal at fixed
``schema_synthesis_tau`` / ``min_supports``, with synthesis hard-wired
on. We test:

  * ``_bootstrap_mean_ci`` shape on degenerate inputs
  * ``_pair_diffs`` pairing key correctness
  * ``run_positive_control_ci`` smoke run on a toy fixture
  * ``render_markdown`` shape + sign convention is documented
  * stages_a / stages_b are exactly S4 / S3 with `schema_update` as the
    sole differing element (this is the entire point of the driver)
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_recall_lift_decompose_positive_control_ci import (
    METRIC_KEYS,
    STAGES_S3,
    STAGES_S4,
    _bootstrap_mean_ci,
    _pair_diffs,
    render_markdown,
    run_positive_control_ci,
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


def test_stage_definitions_isolate_schema_update():
    # The ENTIRE point of this driver: S4 \ S3 == ['schema_update'].
    assert set(STAGES_S4) - set(STAGES_S3) == {"schema_update"}
    assert set(STAGES_S3) - set(STAGES_S4) == set()
    # And in cumulative order, schema_update is the *added* stage.
    assert STAGES_S4[: len(STAGES_S3)] == STAGES_S3
    assert STAGES_S4[-1] == "schema_update"


def test_bootstrap_mean_ci_zero_variance():
    # All zeros: mean=0, CI=[0,0], p=1.0 (every resample mean is 0).
    m, lo, hi, p = _bootstrap_mean_ci([0.0] * 50, resamples=200, seed=0)
    assert m == 0.0
    assert lo == 0.0 and hi == 0.0
    assert p == 1.0


def test_bootstrap_mean_ci_strictly_positive_excludes_zero():
    # All +1 -> mean=1, CI=[1,1], two-sided p ~= 0 (no resample <= 0).
    m, lo, hi, p = _bootstrap_mean_ci([1.0] * 50, resamples=500, seed=1)
    assert m == 1.0
    assert lo > 0 and hi > 0
    assert p < 0.05


def test_bootstrap_mean_ci_empty_input():
    m, lo, hi, p = _bootstrap_mean_ci([], resamples=10, seed=0)
    assert (m, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)


def test_pair_diffs_key_is_sample_question_category():
    pairs_a = [
        {"sample_id": "X", "question": "q1", "category": "c", **{k: 1.0 for k in METRIC_KEYS}},
        {"sample_id": "X", "question": "q2", "category": "c", **{k: 2.0 for k in METRIC_KEYS}},
        {"sample_id": "Y", "question": "q1", "category": "c", **{k: 3.0 for k in METRIC_KEYS}},
    ]
    pairs_b = [
        {"sample_id": "X", "question": "q1", "category": "c", **{k: 0.5 for k in METRIC_KEYS}},
        {"sample_id": "X", "question": "q2", "category": "c", **{k: 1.0 for k in METRIC_KEYS}},
        {"sample_id": "Z", "question": "q9", "category": "c", **{k: 9.0 for k in METRIC_KEYS}},
    ]
    diffs, paired = _pair_diffs(pairs_a, pairs_b)
    assert paired == 2  # (X,q1,c) and (X,q2,c); Y/Z drop
    for k in METRIC_KEYS:
        # diffs are a-minus-b, in pair_a order: (1.0-0.5)=0.5, (2.0-1.0)=1.0
        assert diffs[k] == [0.5, 1.0]


def test_run_positive_control_ci_smoke(tmp_path):
    ds = _toy_dataset(tmp_path)
    rep = run_positive_control_ci(
        ds,
        max_instances=1,
        k=5,
        embedder_name="hashtrigram",
        tau=0.30,
        min_supports=2,
        resamples=200,  # keep tests fast
        seed=42,
    )
    assert rep["stages_a"] == STAGES_S4
    assert rep["stages_b"] == STAGES_S3
    assert rep["tau"] == 0.30
    assert rep["min_supports"] == 2
    assert rep["synthesis"] is True
    assert rep["ci_config"]["method"] == "percentile_paired_diff"
    assert rep["ci_config"]["resamples"] == 200
    for mk in METRIC_KEYS:
        c = rep["summary"][mk]
        assert "mean_diff_s4_minus_s3" in c
        assert "ci_lo" in c and "ci_hi" in c
        assert c["ci_lo"] <= c["ci_hi"]
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
        assert isinstance(c["ci_excludes_zero"], bool)
        assert c["n_paired"] == rep["n_paired"]


def test_render_markdown_shape_and_sign_convention():
    rep = {
        "dataset_path": "x",
        "max_instances": 2,
        "k": 10,
        "embedder": "hashtrigram",
        "tau": 0.30,
        "min_supports": 2,
        "synthesis": True,
        "stages_a": STAGES_S4,
        "stages_b": STAGES_S3,
        "n_paired": 301,
        "ci_config": {"resamples": 10000, "seed": 42, "alpha": 0.05,
                      "method": "percentile_paired_diff"},
        "summary": {
            "delta_h1": {"mean_diff_s4_minus_s3": -0.0033, "ci_lo": -0.01,
                         "ci_hi": 0.0, "p_bootstrap_two_sided": 0.74,
                         "n_paired": 301, "ci_excludes_zero": False},
            "delta_hk": {"mean_diff_s4_minus_s3": -0.0100, "ci_lo": -0.0233,
                         "ci_hi": 0.0, "p_bootstrap_two_sided": 0.0956,
                         "n_paired": 301, "ci_excludes_zero": False},
            "delta_rr": {"mean_diff_s4_minus_s3": -0.0031, "ci_lo": -0.0078,
                         "ci_hi": 0.0006, "p_bootstrap_two_sided": 0.1118,
                         "n_paired": 301, "ci_excludes_zero": False},
            "delta_prk": {"mean_diff_s4_minus_s3": -0.0100, "ci_lo": -0.0233,
                          "ci_hi": 0.0, "p_bootstrap_two_sided": 0.0956,
                          "n_paired": 301, "ci_excludes_zero": False},
            "delta_grk": {"mean_diff_s4_minus_s3": -0.0100, "ci_lo": -0.0233,
                          "ci_hi": 0.0, "p_bootstrap_two_sided": 0.0956,
                          "n_paired": 301, "ci_excludes_zero": False},
        },
    }
    md = render_markdown(rep)
    assert "§94c-decompose-positive-control-CI" in md
    assert "S4 − S3" in md  # sign convention spelled out
    assert "schema_update" in md  # the differing stage shows up via stages list
    # 5 metric rows + header + separator
    table_lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(table_lines) == 5 + 2

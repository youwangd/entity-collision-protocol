"""Tests for evals.locomo_recall_lift (§90)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.locomo_recall_lift import RECIPE, run_recall_lift


DATA = Path(__file__).resolve().parents[2] / "bench" / "data" / "locomo10.json"


@pytest.mark.skipif(not DATA.exists(), reason="locomo10.json not present")
def test_recipe_shape():
    assert set(RECIPE) == {
        "schema_family_share",
        "schema_family_tau",
        "schema_family_fragmentation_max",
        "schema_family_contamination_max",
    }
    assert RECIPE["schema_family_share"] == 0.75
    assert RECIPE["schema_family_tau"] == 0.20


@pytest.mark.slow
@pytest.mark.skipif(not DATA.exists(), reason="locomo10.json not present")
def test_smoke_one_sample_returns_paired():
    """1-sample smoke: structure + pairing + zero consolidation errors.

    The §87 hypothesis is that the gate at this recipe is operationally
    inert on LoCoMo recall. We don't enforce zero delta here (that's
    a paper claim, not a test invariant) — we just enforce that the
    driver runs end-to-end, pairs every QA, and never throws inside
    consolidate(). The §90 *full-corpus* artifact carries the lift
    number for the paper.
    """
    out = run_recall_lift(
        DATA, max_instances=1, k=10, embedder_name="hashtrigram",
    )
    assert out["recipe"] == RECIPE
    assert out["n_samples"] == 1
    assert out["n_pairs"] >= 1
    assert out["consolidation_errors"] == []
    # baseline / treatment + delta keys present
    for arm in ("baseline", "treatment", "delta"):
        for key in ("session_hit_at_1", "session_hit_at_k",
                    "pair_recall_at_k", "gold_recall_at_k",
                    "mean_reciprocal_rank"):
            assert key in out[arm]
    # paired delta is internally consistent: |Δ| ≤ max(arm) value.
    for key in ("session_hit_at_1", "session_hit_at_k",
                "pair_recall_at_k", "gold_recall_at_k"):
        assert abs(out["delta"][key]) <= 1.0
    # §95 — multi_hop block is present and self-consistent.
    mh = out["multi_hop"]
    assert "n_pairs" in mh
    assert mh["n_pairs"] <= out["n_pairs"]
    if mh["n_pairs"] > 0:
        assert 0.0 <= mh["baseline_pair_recall_at_k"] <= 1.0
        assert 0.0 <= mh["treatment_pair_recall_at_k"] <= 1.0
        assert abs(mh["delta_pair_recall_at_k"]) <= 1.0
    # §95 — every per-query pair carries n_gold + pair/gold recall.
    for p in out["per_query_pairs"]:
        assert p["n_gold"] >= 1
        assert p["baseline_prk"] in (0, 1)
        assert p["treatment_prk"] in (0, 1)
        assert 0.0 <= p["baseline_grk"] <= 1.0
        assert 0.0 <= p["treatment_grk"] <= 1.0
        # If pair_recall=1 then gold_recall must be 1.0 (all golds present).
        if p["baseline_prk"] == 1:
            assert p["baseline_grk"] == 1.0
        if p["treatment_prk"] == 1:
            assert p["treatment_grk"] == 1.0
        # h@k >= prk (pair_recall implies first-gold hit in top-k).
        assert p["baseline_hk"] >= p["baseline_prk"]
        assert p["treatment_hk"] >= p["treatment_prk"]


@pytest.mark.skipif(not DATA.exists(), reason="locomo10.json not present")
def test_full_corpus_artifact_present_and_consistent():
    """If the §90 full-corpus artifact has been generated, sanity-check
    its shape. Skipped silently if the file is absent (CI doesn't run
    the 3-min driver)."""
    art = (
        Path(__file__).resolve().parents[2]
        / "bench" / "results" / "locomo_recall_lift_full.json"
    )
    if not art.exists():
        pytest.skip("locomo_recall_lift_full.json not present")
    d = json.loads(art.read_text())
    assert d["recipe"] == RECIPE
    # Pairing must be total — every treatment row must have a baseline.
    n = d["n_pairs"]
    assert n > 0
    # delta_h1 / delta_hk are bounded in [-1,1] by construction.
    for p in d["per_query_pairs"]:
        assert p["delta_h1"] in (-1, 0, 1)
        assert p["delta_hk"] in (-1, 0, 1)

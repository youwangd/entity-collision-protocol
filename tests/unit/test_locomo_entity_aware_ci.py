"""§D3-collateral-(c) — entity-aware-default ablation CI driver smoke test.

Pins the public interface of `evals.locomo_recall_lift_entity_aware_ci`:
the run_entity_aware_ci function returns a dict with the expected
schema (summary keyed by 5 deltas, headline_off/on, n_paired). Avoids
running the full LoCoMo10 corpus (too slow for unit tests) by using
max_instances=1 against the existing fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evals.locomo_recall_lift_entity_aware_ci import run_entity_aware_ci


REPO = Path(__file__).resolve().parents[2]
LOCOMO = REPO / "bench" / "data" / "locomo10.json"


@pytest.mark.slow
@pytest.mark.skipif(not LOCOMO.exists(), reason="locomo10 fixture not present")
def test_entity_aware_ci_smoke_schema():
    rep = run_entity_aware_ci(
        str(LOCOMO),
        max_instances=1,
        k=10,
        embedder_name="hashtrigram",
        entity_min=0.7,
        resamples=200,
        seed=7,
    )
    assert rep["max_instances"] == 1
    assert rep["entity_min"] == 0.7
    assert rep["n_paired"] >= 1
    for k in ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk"):
        c = rep["summary"][k]
        # CI is well-formed: lo <= mean <= hi.
        assert c["ci_lo"] <= c["mean_diff_off_minus_on"] <= c["ci_hi"]
        assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
    assert "headline_off" in rep and "headline_on" in rep


@pytest.mark.slow
def test_run_recall_lift_threads_entity_aware_flags():
    """Mechanical: result dict surfaces the new config knobs."""
    from evals.locomo_recall_lift import run_recall_lift

    if not LOCOMO.exists():
        pytest.skip("locomo10 fixture not present")
    rep = run_recall_lift(
        str(LOCOMO),
        max_instances=1,
        k=10,
        embedder_name="hashtrigram",
        interference_entity_aware=True,
        interference_entity_overlap_min=0.6,
    )
    assert rep["interference_entity_aware"] is True
    assert rep["interference_entity_overlap_min"] == 0.6

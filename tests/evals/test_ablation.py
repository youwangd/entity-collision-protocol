"""Smoke test for the ablation harness — runs a tiny ablation on 2 variants."""
from __future__ import annotations

import pytest

from evals.ablation import run_ablation


@pytest.mark.evals
def test_ablation_smoke():
    summary = run_ablation(
        n_sessions=2,
        facts_per_session=2,
        distractors_per_session=3,
        seed=11,
        k=5,
        only=["baseline", "no_extraction_confidence"],
    )
    names = [r["variant"] for r in summary["results"]]
    assert "baseline" in names
    assert "no_extraction_confidence" in names
    for r in summary["results"]:
        for m in ("hit_at_1", "hit_at_k", "mrr", "ndcg_at_k"):
            assert 0.0 <= r[m] <= 1.0
    # Non-baseline variants must have a delta block
    nonbase = [r for r in summary["results"] if r["variant"] != "baseline"]
    for r in nonbase:
        assert "delta_vs_baseline" in r

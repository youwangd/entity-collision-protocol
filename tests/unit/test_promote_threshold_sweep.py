"""Smoke test: schema_promote_threshold flows through evals.locomo_recall_lift."""
from __future__ import annotations

import pytest

from evals.locomo_recall_lift import _build_config


def test_build_config_threshold_default():
    cfg = _build_config("/tmp/test", treatment=True)
    assert cfg.consolidation.schema_promote_threshold == 3


def test_build_config_threshold_override():
    cfg = _build_config("/tmp/test", treatment=True, schema_promote_threshold=7)
    assert cfg.consolidation.schema_promote_threshold == 7


def test_build_config_threshold_baseline_arm_unaffected():
    # Baseline arm has no ConsolidationConfig at all; threshold override
    # is silently ignored on the baseline path.
    cfg = _build_config("/tmp/test", treatment=False, schema_promote_threshold=7)
    assert cfg.consolidation is None


@pytest.mark.slow
def test_run_sweep_smoke_returns_rows():
    """Tiny end-to-end smoke: 1 sample, 2 thresholds. Just exercises plumbing."""
    from evals.locomo_promote_threshold_sweep import run_sweep
    out = run_sweep(
        "bench/data/locomo10.json",
        thresholds=[3, 5],
        max_instances=1,
        resamples=50,
        synthesis=False,
    )
    assert len(out["rows"]) == 2
    for row in out["rows"]:
        assert "delta_h1" in row
        assert "churn" in row
        assert row["n_pairs"] > 0

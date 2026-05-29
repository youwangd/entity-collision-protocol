"""§87 — Tests for the end-to-end gate replication on LoCoMo."""
from __future__ import annotations

from pathlib import Path

import pytest

from evals.locomo_gate_replication import run as gate_run


LOCOMO = Path(__file__).resolve().parents[2] / "bench" / "data" / "locomo10.json"


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo bench data missing")
def test_gate_replication_default_taus_pin_2026_05_22():
    """Pin the §87 default-cell artifact so future schema changes
    surface as a test failure, not a silent paper-figure drift."""
    res = gate_run(LOCOMO, taus=(0.10, 0.15, 0.20))
    assert res["n_samples"] == 10
    assert res["n_schemas"] == 543
    assert res["calibration_table"] == "LOCOMO"
    by_tau = {row["tau"]: row for row in res["by_tau"]}
    # tau=0.10: gate passes by frag but cluster topology is unsafe
    # (one mega-cluster). The fragmentation gate and contamination
    # tell different stories — the §87 lesson.
    r10 = by_tau[0.10]
    assert r10["calibrated_fmax_max_margin"] == pytest.approx(0.1486)
    assert r10["gate_passes"] is True
    assert r10["max_cluster_size"] >= 400  # mega-cluster
    assert r10["contamination_rate"] > 0.95  # screams "unsafe"
    # tau=0.15: gate passes, small meaningful clusters
    r15 = by_tau[0.15]
    assert r15["gate_passes"] is True
    assert r15["share_active_and_useful"] is True
    assert 1 < r15["mean_non_singleton_size"] < 10
    # tau=0.20: gate passes, very small clusters
    r20 = by_tau[0.20]
    assert r20["gate_passes"] is True
    assert r20["max_cluster_size"] <= 8
    # default-flip is defensible — at least one tau has share active+useful
    assert res["summary"]["default_flip_defensible"] is True


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo bench data missing")
def test_gate_replication_singleton_only_yields_no_lift():
    """Higher tau eventually shatters everything to singletons —
    in that regime the gate may pass but share is a no-op."""
    res = gate_run(LOCOMO, taus=(0.30,))
    row = res["by_tau"][0]
    # Not in the calibration table → fmax is None → gate cannot pass.
    assert row["calibrated_fmax_max_margin"] is None
    assert row["gate_passes"] is False
    assert row["share_active_and_useful"] is False

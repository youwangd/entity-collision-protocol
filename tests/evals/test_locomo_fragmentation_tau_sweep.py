"""Smoke + regression tests for §78 LoCoMo low-tau regime sweep."""
from __future__ import annotations

import json
from pathlib import Path

from evals.locomo_fragmentation_tau_sweep import run

DATA = Path(__file__).resolve().parents[2] / "bench" / "data" / "locomo10.json"


def test_run_full_tau_sweep_serializable():
    res = run(DATA, taus=(0.1, 0.5))
    json.dumps(res)
    assert res["n_schemas"] == 543
    assert len(res["by_tau"]) == 2


def test_low_tau_collapse_regime():
    """At tau ≤ 0.10 the metric collapses everything into mega-clusters.

    Locks the §78 headline: fragmentation < 0.10 *and* contamination
    > 0.85 simultaneously at tau=0.05. The §73 contamination gate
    (cmax=0.10) therefore trips trivially in this regime — confirming
    that the gate's defensibility is tau-conditional.
    """
    res = run(DATA, taus=(0.05, 0.10))
    rows = {row["tau"]: row for row in res["by_tau"]}
    # tau=0.05: single mega-cluster
    assert rows[0.05]["n_clusters"] <= 5
    assert rows[0.05]["fragmentation_rate"] < 0.05
    assert rows[0.05]["contamination_rate"] > 0.85
    # tau=0.10: still in the collapse regime, contamination near saturation
    assert rows[0.10]["contamination_rate"] > 0.95


def test_high_tau_singleton_regime():
    """At tau ≥ 0.30, fragmentation > 0.99 and contamination = 0.0.

    Locks the §78 finding that the §74 "contamination meter reads
    identically 0" conclusion holds *only* in this regime.
    """
    res = run(DATA, taus=(0.30, 0.50))
    rows = {row["tau"]: row for row in res["by_tau"]}
    for tau in (0.30, 0.50):
        assert rows[tau]["fragmentation_rate"] > 0.99
        assert rows[tau]["contamination_rate"] == 0.0

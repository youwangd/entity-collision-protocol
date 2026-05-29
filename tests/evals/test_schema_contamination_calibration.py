"""Regression tests for the §74 contamination-meter calibration."""
from __future__ import annotations

from evals.schema_contamination_calibration import CalibConfig, run_grid


def test_realistic_regime_meter_reads_zero():
    """In the disjoint-core / large-vocab regime, single-link clustering
    expels outsiders as singletons; the meter never registers contamination
    even at extreme true-c. This is the §74 headline (the gate's
    ``cmax=0.10`` default is a no-op in this regime — fragmentation is
    the real signal).
    """
    cfg = CalibConfig(
        n_clusters=50, cluster_size=4, vocab_size=2000,
        core_size=8, schema_size=6, tau=0.5, seed=0xCA11B,
    )
    res = run_grid(cfg, [0.0, 0.1, 0.25, 0.5, 1.0])
    assert all(c["meter_rate"] == 0.0 for c in res["cells"])
    # c=0.0: no outsiders -> no fragmentation
    assert res["cells"][0]["fragmentation"] == 0.0
    # c=1.0: all outsiders -> near-total fragmentation
    assert res["cells"][-1]["fragmentation"] >= 0.9


def test_fragmentation_monotone_in_true_c():
    """Fragmentation is the calibrated, monotone-in-true-c signal."""
    cfg = CalibConfig(
        n_clusters=50, cluster_size=4, vocab_size=2000,
        core_size=8, schema_size=6, tau=0.5, seed=0xCA11B,
    )
    res = run_grid(cfg, [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    frags = [c["fragmentation"] for c in res["cells"]]
    # weakly monotone non-decreasing
    for a, b in zip(frags, frags[1:]):
        assert b + 1e-9 >= a, f"fragmentation regressed: {frags}"
    # endpoints sane
    assert frags[0] == 0.0
    assert frags[-1] >= 0.9


def test_small_vocab_regime_meter_saturated_even_at_zero_c():
    """When vocab is small enough that random pairs already exceed tau,
    single-link transitivity glues most schemas into giant clusters
    whose pairwise floors collapse below tau — the meter saturates at
    c=0.0 already. This documents the meter's domain of applicability:
    it requires sparse-feature schemas, not dense ones.
    """
    cfg = CalibConfig(
        n_clusters=50, cluster_size=4, vocab_size=80,
        core_size=8, schema_size=6, tau=0.3, seed=0xDEAD,
    )
    res = run_grid(cfg, [0.0, 0.5, 1.0])
    # at c=0, the meter already reads >>0.10 (the gate threshold)
    assert res["cells"][0]["meter_rate"] > 0.5

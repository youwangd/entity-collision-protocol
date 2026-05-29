"""Regression tests for the §76 fragmentation-meter calibration."""
from __future__ import annotations

from evals.schema_fragmentation_calibration import (
    RegimeCell,
    evaluate_cell,
    run_calibration,
    default_regime_grid,
)


def test_realistic_cell_fragmentation_tracks_true_c_at_010():
    """In the realistic regime (sparse schemas / disjoint cores / mid-tau),
    fragmentation reads ≈0.10 at true_c=0.10 — the §69 deployment frontier.

    This is the §76 headline: the fragmentation gate's calibrated default
    is ``fragmentation_max ≈ 0.10``, lifted directly from this curve.
    """
    cell = RegimeCell(
        n_clusters=200, cluster_size=4, vocab_size=2000,
        core_size=8, schema_size=6, tau=0.5, seed=0xCA11B,
    )
    res = evaluate_cell(cell, [0.0, 0.05, 0.10, 0.25, 1.0])
    assert res["frag_at_c0"] == 0.0
    # Tight band around 0.10: no other cell in the §76 grid drifted
    # past 0.105 at c=0.10. Lock at 0.105 to allow tiny seed sensitivity.
    assert 0.085 <= res["frag_at_c10"] <= 0.105
    assert res["monotone_in_true_c"] is True


def test_calibration_summary_recommends_010_default():
    """The driver's summary across the realistic regime grid should
    recommend a fragmentation_max default in [0.09, 0.11]."""
    cells = default_regime_grid()
    res = run_calibration(cells, [0.0, 0.05, 0.10, 0.25, 1.0])
    s = res["summary"]
    assert s["n_realistic_cells"] >= 4, (
        "expected at least 4 realistic cells in the default grid"
    )
    # The §76 recommendation: fragmentation_max ≈ 0.10
    assert 0.09 <= s["median_frag_at_c10"] <= 0.11
    # Cross-cell variance should be tight in the realistic envelope.
    if "stdev_frag_at_c10" in s:
        assert s["stdev_frag_at_c10"] < 0.005


def test_dense_schema_regime_is_out_of_domain():
    """schema_size=4 from core_size=8 fragments naturally at c=0
    (low within-core Jaccard ⇒ many singletons). The driver flags
    these as out-of-domain via ``frag_at_c0 > 0.05``."""
    cell = RegimeCell(
        n_clusters=200, cluster_size=4, vocab_size=2000,
        core_size=8, schema_size=4, tau=0.5, seed=0xCA11B,
    )
    res = evaluate_cell(cell, [0.0, 0.10])
    # Baseline fragmentation already huge — gate would mis-fire here.
    assert res["frag_at_c0"] > 0.10


def test_zero_outsider_zero_fragmentation_in_realistic_cell():
    """Realistic-regime sanity: with no outsiders, single-link clusters
    every schema with its cluster-mates, so fragmentation = 0 exactly.
    """
    cell = RegimeCell(
        n_clusters=100, cluster_size=4, vocab_size=2000,
        core_size=8, schema_size=6, tau=0.5, seed=0xCA11B,
    )
    res = evaluate_cell(cell, [0.0])
    assert res["points"][0]["fragmentation"] == 0.0

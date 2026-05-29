"""Regression tests for `evals.schema_fragmentation_per_tau_calibration`.

Locks the per-tau headline numbers from SCALE_REPORT §79: under the
disjoint-core synthetic regime (vocab=2000, schema_size=6, core_size=8,
n_clusters=200), the recommended ``fragmentation_max`` at the §69
``c=0.10`` safety frontier is bimodal — 0.0925 for tau ∈ {0.10, 0.15,
0.20}, 0.1013 for tau ∈ {0.25, 0.30, 0.40, 0.50}. tau=0.05 collapses
(meter dead).

The §76 default of ``fragmentation_max=0.10`` lives within ±0.01 of
every gateable tau — small but real per-tau re-calibration room.
"""
from __future__ import annotations

import pytest

from evals.schema_fragmentation_per_tau_calibration import (
    Cell,
    evaluate_tau_row,
    run_per_tau,
)


@pytest.fixture(scope="module")
def grid() -> dict:
    return run_per_tau(
        taus=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        contaminations=[0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0],
    )


def test_gateable_count_seven_of_eight(grid: dict) -> None:
    """Only tau=0.05 (the §78 collapse cliff) is non-gateable."""
    s = grid["summary"]
    assert s["n_taus"] == 8
    assert s["n_gateable"] == 7
    assert s["n_collapsed"] == 1
    assert 0.05 not in s["gateable_taus"]


def test_recommended_fmax_bimodal(grid: dict) -> None:
    """Headline §79 numbers locked: low-tau 0.0925, mid-high tau 0.1013."""
    by_tau = {r["regime"]["tau"]: r["recommended_fmax"] for r in grid["rows"]}
    assert by_tau[0.05] is None
    for t in (0.10, 0.15, 0.20):
        assert by_tau[t] == pytest.approx(0.0925, abs=1e-4)
    for t in (0.25, 0.30, 0.40, 0.50):
        assert by_tau[t] == pytest.approx(0.1013, abs=1e-4)


def test_section76_default_within_one_pp_of_every_gateable_tau(grid: dict) -> None:
    """The §76 default `fragmentation_max=0.10` is at most 0.01 off
    every gateable tau — the per-tau re-calibration is real but small."""
    for r in grid["rows"]:
        if not r["gateable"]:
            continue
        rec = r["recommended_fmax"]
        assert abs(rec - 0.10) <= 0.01, (
            f"tau={r['regime']['tau']} recommended {rec} too far from 0.10"
        )


def test_collapse_at_tau_005(grid: dict) -> None:
    """§78 collapse cliff: at tau=0.05 fragmentation is dead across c."""
    row_005 = next(r for r in grid["rows"] if r["regime"]["tau"] == 0.05)
    assert row_005["collapsed_at_c0"] is True
    assert row_005["frag_at_c0"] == pytest.approx(0.0, abs=1e-9)
    assert row_005["frag_at_c10"] == pytest.approx(0.0, abs=1e-9)
    assert row_005["recommended_fmax"] is None


def test_monotone_on_gateable_taus(grid: dict) -> None:
    """Fragmentation is weakly non-decreasing in true-c on every
    gateable tau row — sanity for using fmax as a one-sided gate."""
    for r in grid["rows"]:
        if r["gateable"]:
            assert r["monotone_in_true_c"] is True


def test_byte_identity_with_section76_at_tau_05() -> None:
    """At tau=0.5 this driver must reproduce §76's headline reading
    (median frag_at_c10 ~= 0.1013) byte-identically — same generator
    seed, same corpus."""
    row = evaluate_tau_row(
        Cell(
            n_clusters=200,
            cluster_size=4,
            vocab_size=2000,
            core_size=8,
            schema_size=6,
            tau=0.5,
        ),
        contaminations=[0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0],
    )
    assert row["frag_at_c10"] == pytest.approx(0.1013, abs=1e-4)
    assert row["frag_at_c0"] == pytest.approx(0.0, abs=1e-9)


def test_singleton_cliff_visible_at_c25(grid: dict) -> None:
    """At c=0.25, §69's 'no setting is safe' cliff, fragmentation should
    rise meaningfully on every gateable tau (singleton cliff visible)."""
    for r in grid["rows"]:
        if not r["gateable"]:
            continue
        assert r["frag_at_c25"] >= 0.20, r["regime"]

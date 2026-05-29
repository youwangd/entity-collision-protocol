"""Regression tests for §80 LoCoMo per-tau fragmentation_max calibration.

Locks the headline finding: on the real LoCoMo 543-schema corpus, the
§79 gateability test fails for every tau in the swept range — the
operational ``fragmentation_max`` gate has no real-corpus calibration
domain on this corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.locomo_fragmentation_per_tau_calibration import (
    _inject_outsiders,
    evaluate_tau_row,
    run,
)
from evals.locomo_fragmentation_replication import _extract_schemas

LOCOMO_PATH = Path(__file__).resolve().parents[2] / "bench" / "data" / "locomo10.json"


@pytest.fixture(scope="module")
def fps():
    data = json.loads(LOCOMO_PATH.read_text())
    return _extract_schemas(data)[0]


def test_inject_zero_is_identity(fps):
    """``p=0`` is byte-identical to the input (lazy-RNG contract)."""
    out = _inject_outsiders(fps, 0.0, seed=0)
    assert out == dict(fps)


def test_inject_one_replaces_all(fps):
    """``p=1.0`` replaces every fingerprint."""
    out = _inject_outsiders(fps, 1.0, seed=0)
    diff = sum(1 for sid in fps if out[sid] != fps[sid])
    # All replaced fingerprints differ (probability of exact match is
    # ~10^-large given vocab=3716 and median fp size 35).
    assert diff >= int(0.99 * len(fps))


def test_inject_seed_determinism(fps):
    """Same seed ⇒ same perturbation."""
    a = _inject_outsiders(fps, 0.10, seed=42)
    b = _inject_outsiders(fps, 0.10, seed=42)
    assert a == b
    c = _inject_outsiders(fps, 0.10, seed=43)
    assert c != a  # different seeds diverge


def test_no_locomo_tau_is_gateable(fps):
    """§80 headline: every tau in the operational range is non-gateable.

    Real LoCoMo at tau≥0.25 saturates (singletons at c=0); at tau≤0.20
    the natural baseline ``frag_at_c0`` already exceeds 0.05 so the
    §79 gateability test (frag_at_c0 ≤ 0.05 AND ≥0.05 dynamic range
    across c=0→c=0.10) fails by construction.
    """
    cs = [0.0, 0.10]
    for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        row = evaluate_tau_row(fps, tau, cs, seed=0xC0FFEE)
        assert row["gateable"] is False, (tau, row)
        assert row["recommended_fmax"] is None


def test_high_tau_saturated_at_c0(fps):
    """tau ≥ 0.25 is in the singleton cliff: frag_at_c0 ≥ 0.9."""
    cs = [0.0, 0.10]
    for tau in (0.25, 0.30, 0.40, 0.50):
        row = evaluate_tau_row(fps, tau, cs, seed=0xC0FFEE)
        assert row["saturated_at_c0"] is True, (tau, row)


def test_low_tau_above_baseline(fps):
    """tau ≤ 0.20 has nontrivial natural fragmentation > 0.05."""
    cs = [0.0, 0.10]
    for tau in (0.10, 0.15, 0.20):
        row = evaluate_tau_row(fps, tau, cs, seed=0xC0FFEE)
        assert row["frag_at_c0"] > 0.05, (tau, row)


def test_run_summary_shape(fps, tmp_path):
    """``run()`` returns the documented JSON shape."""
    res = run(
        LOCOMO_PATH,
        taus=(0.20, 0.50),
        contaminations=(0.0, 0.10),
        seed=0xC0FFEE,
    )
    assert res["n_schemas"] == len(fps)
    assert res["seed"] == 0xC0FFEE
    assert len(res["by_tau"]) == 2
    for r in res["by_tau"]:
        assert {"frag_at_c0", "frag_at_c10", "gateable", "points"} <= set(r)
    assert res["summary"]["n_gateable"] == 0
    assert res["summary"]["median_recommended_fmax"] is None

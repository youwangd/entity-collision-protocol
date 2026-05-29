"""§86 — Static per-tau calibration table tests.

Covers ``engram.consolidation.calibration`` (the static lookup) and
the ``recommended_fmax(..., max_margin=True)`` runtime path on
``sharing_regime`` that consumes it.

Numbers are pinned to the on-disk artifacts:
- ``bench/results/fmax_max_margin.json`` (LoCoMo, 543 schemas, B=100)
- ``bench/results/synthetic_fmax_max_margin.json`` (synthetic, 800 schemas, B=100)

If a future change shifts those artifacts, both this test and the
calibration table need to be regenerated together. That's intentional:
a silent table drift would invalidate the §84 paper figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.consolidation import calibration
from engram.consolidation.sharing_regime import (
    GATEABLE,
    classify_curve,
    recommended_fmax,
)

REPO = Path(__file__).resolve().parents[2]
LOCOMO_ARTIFACT = REPO / "bench" / "results" / "fmax_max_margin.json"
SYNTHETIC_ARTIFACT = REPO / "bench" / "results" / "synthetic_fmax_max_margin.json"


def test_locomo_table_pins_match_artifact():
    """Static LoCoMo table must agree exactly with the bootstrap artifact."""
    assert LOCOMO_ARTIFACT.exists(), f"missing artifact: {LOCOMO_ARTIFACT}"
    data = json.loads(LOCOMO_ARTIFACT.read_text())
    by_tau = {round(row["tau"], 4): row["fmax_max_margin"] for row in data["by_tau"]}
    for tau, expected in by_tau.items():
        got = calibration.lookup_max_margin_fmax(tau, table="LOCOMO")
        assert got is not None, f"missing tau={tau}"
        assert got == pytest.approx(expected, abs=1e-4), (
            f"tau={tau}: table={got}, artifact={expected}"
        )


def test_synthetic_table_pins_match_artifact():
    assert SYNTHETIC_ARTIFACT.exists()
    data = json.loads(SYNTHETIC_ARTIFACT.read_text())
    by_tau = {round(row["tau"], 4): row["fmax_max_margin"] for row in data["by_tau"]}
    for tau, expected in by_tau.items():
        got = calibration.lookup_max_margin_fmax(tau, table="SYNTHETIC")
        assert got is not None, f"missing tau={tau}"
        assert got == pytest.approx(expected, abs=1e-4)


def test_lookup_returns_none_for_uncalibrated_tau():
    # 0.42 is nowhere near any calibrated key.
    assert calibration.lookup_max_margin_fmax(0.42) is None
    assert calibration.lookup_max_margin_fmax(None) is None


def test_lookup_tolerant_of_float_jitter():
    # tau=0.10 calibrated; pass 0.1000001 — should still hit.
    got = calibration.lookup_max_margin_fmax(0.10 + 1e-7)
    assert got == pytest.approx(0.1486, abs=1e-4)


def test_select_table_unknown_raises():
    with pytest.raises(KeyError):
        calibration.select_table("BOGUS")


def test_select_table_default_is_locomo():
    assert calibration.select_table() is calibration.LOCOMO_FMAX_MAX_MARGIN
    assert calibration.select_table("locomo") is calibration.LOCOMO_FMAX_MAX_MARGIN
    assert calibration.select_table("SYNTHETIC") is calibration.SYNTHETIC_FMAX_MAX_MARGIN


# ---------------------------------------------------------------------------
# recommended_fmax(..., max_margin=True) integration
# ---------------------------------------------------------------------------


def _gateable_row(tau: float, f0: float, f10: float) -> dict:
    return {"tau": tau, "frag_at_c0": f0, "frag_at_c10": f10}


def test_max_margin_requires_debiased():
    row = _gateable_row(0.15, 0.10, 0.30)
    with pytest.raises(ValueError):
        recommended_fmax(row, debiased=False, max_margin=True)


def test_max_margin_uses_calibration_when_tau_matches():
    # Synthesise a debiased-GATEABLE row at tau=0.15 (LoCoMo calibrated key).
    # f0=0.10, f10=0.30 ⇒ delta=0.20 ≥ contam_lift, not saturated, gateable.
    row = _gateable_row(0.15, 0.10, 0.30)
    midpoint = recommended_fmax(row, debiased=True, max_margin=False)
    mm = recommended_fmax(row, debiased=True, max_margin=True)
    assert midpoint == pytest.approx(0.20, abs=1e-4)
    # max-margin pulls from LOCOMO_FMAX_MAX_MARGIN[0.15] = 0.6740
    assert mm == pytest.approx(0.6740, abs=1e-4)


def test_max_margin_falls_back_to_midpoint_when_tau_uncalibrated():
    # tau=0.42 is not in any table; should silently fall back to midpoint.
    row = _gateable_row(0.42, 0.10, 0.30)
    midpoint = recommended_fmax(row, debiased=True, max_margin=False)
    mm = recommended_fmax(row, debiased=True, max_margin=True)
    assert mm == midpoint


def test_max_margin_threads_through_classify_curve():
    rows = [
        _gateable_row(0.10, 0.10, 0.30),
        _gateable_row(0.15, 0.10, 0.30),
    ]
    out = classify_curve(rows, debiased=True, max_margin=True)
    fmaxes = {r["tau"]: r["recommended_fmax"] for r in out["rows"]}
    # Both tau bins are GATEABLE; values come from LOCOMO calibration.
    assert all(out_r["regime"] == GATEABLE for out_r in out["rows"])
    assert fmaxes[0.10] == pytest.approx(0.1486, abs=1e-4)
    assert fmaxes[0.15] == pytest.approx(0.6740, abs=1e-4)


def test_max_margin_synthetic_table_routing():
    row = _gateable_row(0.20, 0.0, 0.10)
    mm_locomo = recommended_fmax(row, debiased=True, max_margin=True, calibration_table="LOCOMO")
    mm_syn = recommended_fmax(row, debiased=True, max_margin=True, calibration_table="SYNTHETIC")
    assert mm_locomo == pytest.approx(0.9021, abs=1e-4)
    assert mm_syn == pytest.approx(0.0369, abs=1e-4)

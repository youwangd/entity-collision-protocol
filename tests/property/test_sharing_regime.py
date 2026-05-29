"""Tests for §81 sharing-operability regime classifier."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from engram.consolidation.sharing_regime import (
    COLLAPSED,
    GATEABLE,
    NATURALLY_FRAGMENTED,
    REGIME_LABELS,
    SINGLETON_CLIFF,
    classify_curve,
    classify_row,
    recommended_fmax,
)


# ---------------------------------------------------------------------------
# R1-R5 invariants (Hypothesis)
# ---------------------------------------------------------------------------


frag = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=400, deadline=None)
@given(frag, frag)
def test_R1_classification_is_one_of_four_labels(c0, c10):
    """R1: every (frag_c0, frag_c10) ∈ [0,1]² lands in exactly one label."""
    label = classify_row({"frag_at_c0": c0, "frag_at_c10": c10})
    assert label in REGIME_LABELS


@settings(max_examples=200, deadline=None)
@given(frag, frag)
def test_R2_label_is_deterministic(c0, c10):
    """R2: classify_row is a pure function of its inputs (no hidden state)."""
    row = {"frag_at_c0": c0, "frag_at_c10": c10, "tau": 0.5}
    a = classify_row(row)
    b = classify_row(row)
    c = classify_row(dict(row))  # different dict object, same content
    assert a == b == c


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.91, max_value=1.0),
    frag,
)
def test_R3_saturated_baseline_implies_singleton_cliff(c0, c10):
    """R3: c0 ≥ saturate_eps (default 0.9) ⇒ SINGLETON_CLIFF, regardless of c10."""
    assert classify_row({"frag_at_c0": c0, "frag_at_c10": c10}) == SINGLETON_CLIFF


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=0.05),
    st.floats(min_value=0.0, max_value=0.05),
)
def test_R4_clean_and_collapsed_implies_COLLAPSED(c0, c10):
    """R4: both ≤ clean_eps ⇒ COLLAPSED (meter dead)."""
    assert classify_row({"frag_at_c0": c0, "frag_at_c10": c10}) == COLLAPSED


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=0.05),
    st.floats(min_value=0.10, max_value=0.89),
)
def test_R5_clean_with_lift_implies_GATEABLE(c0, c10):
    """R5: c0 ≤ 0.05, c10 - c0 ≥ 0.05 (here ≥ 0.05 since c10 ≥ 0.10) ⇒ GATEABLE."""
    assert classify_row({"frag_at_c0": c0, "frag_at_c10": c10}) == GATEABLE


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.06, max_value=0.89),
    frag,
)
def test_R6_dirty_baseline_below_saturation_is_NATURALLY_FRAGMENTED(c0, c10):
    """R6: c0 ∈ (clean_eps, saturate_eps) ⇒ NATURALLY_FRAGMENTED.

    No matter what c10 is — the gate is unconditionally hot on a clean
    corpus, so the regime is naturally fragmented (not contamination-driven).
    """
    assert classify_row({"frag_at_c0": c0, "frag_at_c10": c10}) == NATURALLY_FRAGMENTED


# ---------------------------------------------------------------------------
# Concrete smoke tests
# ---------------------------------------------------------------------------


def test_none_inputs_collapse():
    """Missing meter values route to COLLAPSED — defensive default."""
    assert classify_row({"frag_at_c0": None, "frag_at_c10": None}) == COLLAPSED
    assert classify_row({"frag_at_c0": 0.5, "frag_at_c10": None}) == COLLAPSED
    assert classify_row({}) == COLLAPSED


def test_recommended_fmax_only_for_GATEABLE():
    assert recommended_fmax({"frag_at_c0": 0.02, "frag_at_c10": 0.10}) == 0.10
    # Saturated: SINGLETON_CLIFF, no fmax.
    assert recommended_fmax({"frag_at_c0": 0.97, "frag_at_c10": 0.99}) is None
    # Both clean: COLLAPSED, no fmax.
    assert recommended_fmax({"frag_at_c0": 0.01, "frag_at_c10": 0.02}) is None
    # Naturally fragmented, no fmax.
    assert recommended_fmax({"frag_at_c0": 0.30, "frag_at_c10": 0.40}) is None


def test_recommended_fmax_rounds_to_four_places():
    fmax = recommended_fmax({"frag_at_c0": 0.04, "frag_at_c10": 0.10131234})
    assert fmax == 0.1013


def test_classify_row_invalid_eps_rejected():
    with pytest.raises(ValueError):
        classify_row(
            {"frag_at_c0": 0.0, "frag_at_c10": 0.0},
            clean_eps=0.0,
        )
    with pytest.raises(ValueError):
        classify_row(
            {"frag_at_c0": 0.0, "frag_at_c10": 0.0},
            clean_eps=0.5,
            saturate_eps=0.5,
        )
    with pytest.raises(ValueError):
        classify_row(
            {"frag_at_c0": 0.0, "frag_at_c10": 0.0},
            contam_lift=-0.01,
        )


# ---------------------------------------------------------------------------
# Curve-level tests
# ---------------------------------------------------------------------------


def test_classify_curve_locomo_s80_replication():
    """§80 LoCoMo per-tau result: every tau is non-gateable.

    tau=0.10 row: NATURALLY_FRAGMENTED (frag_c0=0.077 > 0.05).
    tau=0.15-0.20 rows: NATURALLY_FRAGMENTED.
    tau>=0.25 rows: SINGLETON_CLIFF (frag_c0 >= 0.97).
    """
    rows = [
        {"tau": 0.10, "frag_at_c0": 0.0773, "frag_at_c10": 0.1823},
        {"tau": 0.15, "frag_at_c0": 0.5838, "frag_at_c10": 0.6575},
        {"tau": 0.20, "frag_at_c0": 0.8729, "frag_at_c10": 0.8932},
        {"tau": 0.25, "frag_at_c0": 0.9742, "frag_at_c10": 0.978},
        {"tau": 0.30, "frag_at_c0": 0.9963, "frag_at_c10": 0.997},
        {"tau": 0.40, "frag_at_c0": 1.0, "frag_at_c10": 1.0},
        {"tau": 0.50, "frag_at_c0": 1.0, "frag_at_c10": 1.0},
    ]
    rep = classify_curve(rows)
    summary = rep["summary"]
    assert summary["operable"] is False
    assert summary["by_regime"][GATEABLE] == 0
    labels = [r["regime"] for r in rep["rows"]]
    assert labels[0] == NATURALLY_FRAGMENTED
    assert labels[1] == NATURALLY_FRAGMENTED
    assert labels[2] == NATURALLY_FRAGMENTED
    assert all(L == SINGLETON_CLIFF for L in labels[3:])


def test_classify_curve_s79_synthetic_replication():
    """§79 synthetic: tau ∈ {0.10, 0.15, 0.20} → fmax≈0.0925;
    tau ∈ {0.25..0.50} → fmax≈0.1013; tau=0.05 collapsed."""
    rows = [
        {"tau": 0.05, "frag_at_c0": 0.01, "frag_at_c10": 0.02},  # COLLAPSED
        {"tau": 0.10, "frag_at_c0": 0.02, "frag_at_c10": 0.0925},
        {"tau": 0.15, "frag_at_c0": 0.02, "frag_at_c10": 0.0925},
        {"tau": 0.20, "frag_at_c0": 0.02, "frag_at_c10": 0.0925},
        {"tau": 0.25, "frag_at_c0": 0.02, "frag_at_c10": 0.1013},
        {"tau": 0.30, "frag_at_c0": 0.02, "frag_at_c10": 0.1013},
        {"tau": 0.40, "frag_at_c0": 0.02, "frag_at_c10": 0.1013},
        {"tau": 0.50, "frag_at_c0": 0.02, "frag_at_c10": 0.1013},
    ]
    rep = classify_curve(rows)
    summary = rep["summary"]
    assert summary["by_regime"][GATEABLE] == 7
    assert summary["by_regime"][COLLAPSED] == 1
    assert summary["operable"] is True
    assert summary["gateable_taus"] == [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    # Median of {0.0925×3, 0.1013×4} = 0.1013.
    assert math.isclose(summary["median_recommended_fmax"], 0.1013, abs_tol=1e-9)


def test_classify_curve_preserves_row_order():
    rows = [
        {"tau": 0.5, "frag_at_c0": 1.0, "frag_at_c10": 1.0},
        {"tau": 0.1, "frag_at_c0": 0.02, "frag_at_c10": 0.10},
    ]
    rep = classify_curve(rows)
    assert [r["tau"] for r in rep["rows"]] == [0.5, 0.1]
    # Gateable taus come out in input order, not sorted.
    assert rep["summary"]["gateable_taus"] == [0.1]


def test_classify_curve_all_collapsed_has_no_median():
    rows = [
        {"tau": 0.05, "frag_at_c0": 0.0, "frag_at_c10": 0.0},
        {"tau": 0.10, "frag_at_c0": 0.01, "frag_at_c10": 0.02},
    ]
    rep = classify_curve(rows)
    assert rep["summary"]["operable"] is False
    assert rep["summary"]["median_recommended_fmax"] is None
    assert rep["summary"]["gateable_taus"] == []


def test_threshold_eps_arguments_are_honored():
    """Tighter clean_eps demotes a previously-GATEABLE row to NATURALLY_FRAGMENTED."""
    row = {"frag_at_c0": 0.04, "frag_at_c10": 0.12}
    assert classify_row(row) == GATEABLE
    # Tighter clean_eps: 0.04 > 0.03 ⇒ baseline no longer "clean".
    assert classify_row(row, clean_eps=0.03) == NATURALLY_FRAGMENTED


def test_contam_lift_threshold_honored():
    """Lower the lift requirement to flip a borderline row.

    Row: c0=0.02 (clean), c10=0.07 (not clean, not saturated), lift=0.05.
    Default contam_lift=0.05 ⇒ GATEABLE.
    Tighten contam_lift=0.10 ⇒ NATURALLY_FRAGMENTED (lift insufficient).
    """
    row = {"frag_at_c0": 0.02, "frag_at_c10": 0.07}
    assert classify_row(row) == GATEABLE
    assert classify_row(row, contam_lift=0.10) == NATURALLY_FRAGMENTED


def test_partition_invariant_on_curve():
    """Sum of by_regime counts equals n_taus for any curve."""
    rows = [
        {"tau": t, "frag_at_c0": v, "frag_at_c10": v + 0.05}
        for t, v in [(0.1, 0.0), (0.2, 0.5), (0.3, 0.95), (0.4, 0.02)]
    ]
    rep = classify_curve(rows)
    assert sum(rep["summary"]["by_regime"].values()) == rep["summary"]["n_taus"]
    assert rep["summary"]["n_taus"] == len(rows)


# ---------------------------------------------------------------------------
# §82 — baseline-debiased mode invariants
# ---------------------------------------------------------------------------


@settings(max_examples=400, deadline=None)
@given(frag, frag)
def test_R7_debiased_label_in_set(c0, c10):
    """R7: debiased mode also lands in exactly one label."""
    label = classify_row({"frag_at_c0": c0, "frag_at_c10": c10}, debiased=True)
    assert label in REGIME_LABELS


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=0.06, max_value=0.84), st.floats(min_value=0.01, max_value=0.05))
def test_R8_debiased_promotes_naturally_fragmented(c0, lift):
    """R8: a NATURALLY_FRAGMENTED row with sufficient lift becomes GATEABLE under
    debiased semantics. Default mode requires `f0 ≤ clean_eps`; debiased does not.
    """
    c10 = c0 + 0.05 + lift  # Δ = 0.05 + lift > contam_lift
    row = {"frag_at_c0": c0, "frag_at_c10": c10}
    assert classify_row(row) == NATURALLY_FRAGMENTED
    assert classify_row(row, debiased=True) == GATEABLE


@settings(max_examples=200, deadline=None)
@given(frag)
def test_R9_debiased_preserves_singleton_cliff(c0):
    """R9: SINGLETON_CLIFF (f0 ≥ saturate_eps) is preserved across modes."""
    c0_sat = max(c0, 0.91)
    row = {"frag_at_c0": c0_sat, "frag_at_c10": c0_sat}
    assert classify_row(row, debiased=True) == SINGLETON_CLIFF


def test_R10_debiased_recommended_fmax_midpoint():
    """R10: debiased recommended_fmax is the midpoint of (f0, f10)."""
    row = {"frag_at_c0": 0.10, "frag_at_c10": 0.20}
    assert recommended_fmax(row, debiased=True) == 0.15


def test_R11_locomo_82_replication():
    """§82 falsifiable prediction: under debiased semantics the LoCoMo
    per-tau curve promotes tau ∈ {0.10, 0.15} from NATURALLY_FRAGMENTED
    to GATEABLE; tau ≥ 0.20 stays SINGLETON_CLIFF / NATURALLY_FRAGMENTED.

    Numbers below are byte-identical to bench/results/locomo_fragmentation_per_tau_calibration.json.
    """
    locomo_rows = [
        {"tau": 0.10, "frag_at_c0": 0.0773480662, "frag_at_c10": 0.1823204420},
        {"tau": 0.15, "frag_at_c0": 0.5837937385, "frag_at_c10": 0.6575221239},
        {"tau": 0.20, "frag_at_c0": 0.8729281768, "frag_at_c10": 0.8932000000},
        {"tau": 0.25, "frag_at_c0": 0.9742173112, "frag_at_c10": 0.9779000000},
        {"tau": 0.30, "frag_at_c0": 0.9963099631, "frag_at_c10": 0.9963099631},
        {"tau": 0.40, "frag_at_c0": 1.0, "frag_at_c10": 1.0},
        {"tau": 0.50, "frag_at_c0": 1.0, "frag_at_c10": 1.0},
    ]
    # default mode: 0/7 GATEABLE (the §80/§81 baseline finding).
    rep_default = classify_curve(locomo_rows)
    assert rep_default["summary"]["by_regime"][GATEABLE] == 0

    # debiased mode: tau ∈ {0.10, 0.15} should promote to GATEABLE.
    rep_debias = classify_curve(locomo_rows, debiased=True)
    by_tau = {r["tau"]: r["regime"] for r in rep_debias["rows"]}
    assert by_tau[0.10] == GATEABLE
    assert by_tau[0.15] == GATEABLE
    assert by_tau[0.20] != GATEABLE  # lift only 0.020 < 0.05
    # SINGLETON_CLIFF preserved on saturated rows
    assert by_tau[0.30] == SINGLETON_CLIFF
    assert by_tau[0.40] == SINGLETON_CLIFF
    assert by_tau[0.50] == SINGLETON_CLIFF
    # operable
    assert rep_debias["summary"]["operable"] is True
    assert rep_debias["summary"]["by_regime"][GATEABLE] == 2

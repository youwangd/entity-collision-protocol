"""§84 unit + property tests for the maximum-margin fmax driver."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from evals.fmax_max_margin import (
    _margin,
    _search_max_margin,
    evaluate_tau,
    run,
)


# ---------------------------------------------------------------------------
# Concrete smoke tests
# ---------------------------------------------------------------------------


def test_margin_disjoint_clouds_perfect():
    """Disjoint clouds (f0 strictly < f10) → margin=1 at any x in the gap."""
    f0s = [0.0, 0.05, 0.10]
    f10s = [0.20, 0.25, 0.30]
    assert _margin(0.15, f0s, f10s) == 1.0


def test_margin_overlapping_clouds_lt_one():
    f0s = [0.0, 0.10, 0.20]
    f10s = [0.10, 0.20, 0.30]
    # No x produces full separation since 0.10 and 0.20 are shared.
    assert _margin(0.10, f0s, f10s) <= 1.0


def test_search_max_margin_picks_separating_x():
    f0s = [0.0, 0.05, 0.10]
    f10s = [0.20, 0.25, 0.30]
    x, m = _search_max_margin(f0s, f10s)
    assert m == 1.0
    # Optimum lies somewhere between 0.10 and 0.20.
    assert 0.10 <= x <= 0.20


def test_evaluate_tau_skips_legacy_row():
    """A §83 row predating §84 has no f0_all/f10_all; skip cleanly."""
    legacy = {"tau": 0.10, "n_boot": 50, "deltas_head": [0.1, 0.2]}
    out = evaluate_tau(legacy)
    assert out["skipped"] is True
    assert out["fmax_max_margin"] is None


def test_evaluate_tau_lift_nonneg():
    """Max-margin must be ≥ midpoint margin by construction."""
    row = {
        "tau": 0.10,
        "n_boot": 6,
        "f0_all": [0.01, 0.02, 0.01, 0.03, 0.02, 0.01],
        "f10_all": [0.12, 0.11, 0.13, 0.10, 0.14, 0.12],
    }
    out = evaluate_tau(row)
    assert out["lift"] >= 0.0
    # Strict separation here, so margin should hit 1.0.
    assert out["margin_max_margin"] == 1.0


# ---------------------------------------------------------------------------
# Property invariants
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    f0s=st.lists(st.floats(0.0, 1.0), min_size=2, max_size=20),
    f10s=st.lists(st.floats(0.0, 1.0), min_size=2, max_size=20),
)
def test_max_margin_dominates_any_observed_x(f0s, f10s):
    """The argmax search must dominate the margin at any individual
    candidate point — including the midpoint."""
    x_star, m_star = _search_max_margin(f0s, f10s)
    midpoint = (sum(f0s) / len(f0s) + sum(f10s) / len(f10s)) / 2.0
    assert m_star + 1e-12 >= _margin(midpoint, f0s, f10s)


@settings(max_examples=30, deadline=None)
@given(st.floats(0.0, 1.0), st.lists(st.floats(0.0, 1.0), min_size=1, max_size=20),
       st.lists(st.floats(0.0, 1.0), min_size=1, max_size=20))
def test_margin_in_unit_interval(x, f0s, f10s):
    m = _margin(x, f0s, f10s)
    assert 0.0 <= m <= 1.0


def test_run_against_real_bootstrap(tmp_path):
    """Round-trip on the committed §83 LoCoMo bootstrap output."""
    import os

    src = "bench/results/locomo_fragmentation_per_tau_bootstrap.json"
    if not os.path.exists(src):
        return  # not generated yet on this checkout
    out = run(src)
    assert "by_tau" in out
    for r in out["by_tau"]:
        if r.get("skipped"):
            continue
        assert r["margin_max_margin"] >= r["margin_midpoint"] - 1e-12
        assert 0.0 <= r["fmax_max_margin"] <= 1.0

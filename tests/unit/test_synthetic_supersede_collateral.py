"""Smoke test for §D3-collateral sweep driver."""
from __future__ import annotations

from evals.synthetic_supersede_d3_collateral import run_sweep


def test_sweep_smoke_two_points():
    """Two-point sweep produces well-formed report; Δstale@1 stays negative."""
    rep = run_sweep(
        n_slots_list=[10, 20],
        updates_per_slot=2,
        distractors=20,
        seed=42,
        k=10,
        resamples=200,
        boot_seed=42,
    )
    assert rep["config"]["n_slots_list"] == [10, 20]
    assert len(rep["points"]) == 2
    for p in rep["points"]:
        assert p["n_queries"] == p["n_slots"]
        # Default supersede should reduce stale@1 vs add-only.
        assert p["delta"]["d_stale_at_1"]["mean_diff_default_minus_addonly"] < 0
        # Default supersede should improve hit@1 vs add-only.
        assert p["delta"]["d_hit_at_1"]["mean_diff_default_minus_addonly"] > 0
        # CI fields present and consistent.
        for key in ("d_hit_at_1", "d_hit_at_k", "d_stale_at_1", "d_stale_at_k"):
            ci = p["delta"][key]
            assert ci["ci_lo"] <= ci["mean_diff_default_minus_addonly"] <= ci["ci_hi"]


def test_sweep_collateral_monotone_in_hit_at_k():
    """Δhit@k regression should be monotone non-increasing in n_slots.

    Cross-slot template-overlap density grows with n_slots, so the
    Jaccard-only detector's false-positive FADE rate should grow too.
    Test the qualitative monotone pattern at small scale (kept tight to
    keep wall <2s).
    """
    rep = run_sweep(
        n_slots_list=[10, 30, 60],
        updates_per_slot=2,
        distractors=30,
        seed=42,
        k=10,
        resamples=200,
        boot_seed=42,
    )
    means = [p["delta"]["d_hit_at_k"]["mean_diff_default_minus_addonly"]
             for p in rep["points"]]
    # Monotone non-increasing (with small slack for bootstrap noise).
    for a, b in zip(means, means[1:]):
        assert b <= a + 0.05, f"non-monotone Δhit@k: {means}"

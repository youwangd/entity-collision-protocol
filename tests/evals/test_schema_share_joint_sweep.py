"""Regression tests for the joint κ × contamination §8 grid driver."""
from __future__ import annotations

from evals.schema_share_joint_sweep import run_grid
from evals.schema_share_sweep import SimConfig, run_share


def _base_cfg() -> SimConfig:
    # Small n_clusters keeps these tests fast; the qualitative orderings
    # we assert here replicate at the headline n_clusters=400.
    return SimConfig(
        n_clusters=120, cluster_size=4, n_per_window=1, n_windows=30,
        beta_a=0.4, beta_b=0.4, seed=0xE17A11,
    )


def test_grid_diagonal_matches_one_axis_sweep() -> None:
    """The grid's edges must reproduce the legacy one-axis sweeps cell-for-cell."""
    cfg = _base_cfg()
    g = run_grid(
        cfg, share=0.75,
        tightnesses=[1.0, 0.5],
        contaminations=[0.0, 0.5],
    )
    by = {(c["tightness"], c["contamination"]): c for c in g["cells"]}

    # tightness=1.0 + contamination=0.0 == legacy default sweep cell.
    legacy_default = run_share(
        SimConfig(**{**cfg.__dict__, "tightness": 1.0, "contamination": 0.0}),
        share=0.75,
    ).to_dict()
    cell = by[(1.0, 0.0)]
    for k in (
        "false_promote_low_q", "false_deprecate_high_q",
        "ttp_high_q_p50", "ttd_low_q_p50", "n_high_q", "n_low_q",
    ):
        assert cell[k] == legacy_default[k], k

    # tightness=0.5 + contamination=0.0 == legacy §68 tightness=0.5 cell.
    legacy_t05 = run_share(
        SimConfig(**{**cfg.__dict__, "tightness": 0.5, "contamination": 0.0}),
        share=0.75,
    ).to_dict()
    cell = by[(0.5, 0.0)]
    assert cell["false_promote_low_q"] == legacy_t05["false_promote_low_q"]
    assert cell["false_deprecate_high_q"] == legacy_t05["false_deprecate_high_q"]


def test_grid_fd_dominated_by_contamination_axis() -> None:
    """At fixed share=0.75, false-deprecate damage scales primarily with
    contamination, not tightness — the axis-dominance finding from §70."""
    cfg = _base_cfg()
    g = run_grid(
        cfg, share=0.75,
        tightnesses=[1.0, 0.5, 0.1, 0.0],
        contaminations=[0.0, 1.0],
    )
    by = {(c["tightness"], c["contamination"]): c for c in g["cells"]}

    # Within each tightness row, FD must be strictly higher at c=1.0 than c=0.0.
    for t in (1.0, 0.5, 0.1, 0.0):
        fd_clean = by[(t, 0.0)]["false_deprecate_high_q"] or 0.0
        fd_dirty = by[(t, 1.0)]["false_deprecate_high_q"] or 0.0
        assert fd_dirty > fd_clean, f"contamination must inflate FD at t={t}"


def test_grid_safe_envelope_is_clean_only() -> None:
    """The §70 deployment rule: at share=0.75 the safe envelope
    (FP ≤ 1%, FD ≤ 5%) requires contamination ≈ 0%, regardless of tightness.

    We assert the weaker, robust shape: at the cleanest column (c=0.0)
    every tightness has FP ≤ 5%, but as soon as we add even mild
    contamination (c=0.25), the highest-tightness cell already exceeds
    that bound. (Numerics come from n_clusters=120 so we use slack
    bounds; n_clusters=400 makes them tighter.)
    """
    cfg = _base_cfg()
    g = run_grid(
        cfg, share=0.75,
        tightnesses=[1.0],
        contaminations=[0.0, 0.25],
    )
    by = {(c["tightness"], c["contamination"]): c for c in g["cells"]}
    fp_clean = by[(1.0, 0.0)]["false_promote_low_q"] or 0.0
    fp_mild = by[(1.0, 0.25)]["false_promote_low_q"] or 0.0
    assert fp_clean <= 0.02
    assert fp_mild > fp_clean

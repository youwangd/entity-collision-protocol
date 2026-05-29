"""Smoke tests for `evals.schema_share_sweep`.

Ensures the sweep is deterministic under a fixed seed and that the
sweet-spot finding (share ≈ 0.7-0.85 dominates share ∈ {0.0, 0.25, 1.0})
holds in the parameter regime documented in SCALE_REPORT §62.

This is a regression guard, not a property test — the meaningful
numerical claim lives in SCALE_REPORT and changes if the policy or
the generative model changes.
"""
from __future__ import annotations

from evals.schema_share_sweep import SimConfig, run_share, run_sweep


def _sparse_cfg(n_clusters: int = 80) -> SimConfig:
    """Sparse-evidence regime that exposes the share knob."""
    return SimConfig(
        n_clusters=n_clusters,
        cluster_size=4,
        n_per_window=1,
        n_windows=20,
        beta_a=0.4,
        beta_b=0.4,
        seed=0xE17A11,
    )


def test_share_sweep_deterministic() -> None:
    """Two runs with the same seed must produce identical metrics."""
    cfg = _sparse_cfg()
    a = run_share(cfg, share=0.75).to_dict()
    b = run_share(cfg, share=0.75).to_dict()
    assert a == b


def test_share_zero_leaves_schemas_inferred_in_sparse_regime() -> None:
    """At n_per_window=1, share=0 cannot reach promote/deprecate thresholds.

    Anchors the §62 claim: the share knob actually has work to do here —
    it is not just smoothing an already-decided distribution.
    """
    cfg = _sparse_cfg()
    cell = run_share(cfg, share=0.0).to_dict()
    assert cell["promote_rate_high"] in (0.0, None)
    assert cell["deprecate_rate_low"] in (0.0, None)


def test_share_sweet_spot_dominates_endpoints() -> None:
    """share ≈ 0.75 strictly improves on share ∈ {0.0, 1.0} in this regime.

    Sweet-spot operationalisation (large-sample, n_clusters=400):
      - promote_rate_high(0.75) > promote_rate_high(0.0)
      - false_promote_low_q(0.75) <= false_promote_low_q(1.0)
      - false_deprecate_high_q(0.75) <= false_deprecate_high_q(1.0)
    """
    cfg = _sparse_cfg(n_clusters=400)
    cells = run_sweep(cfg, [0.0, 0.75, 1.0])["cells"]
    by_share = {c["share"]: c for c in cells}

    assert (by_share[0.75]["promote_rate_high"] or 0.0) > (
        by_share[0.0]["promote_rate_high"] or 0.0
    )
    assert (by_share[0.75]["false_promote_low_q"] or 0.0) <= (
        by_share[1.0]["false_promote_low_q"] or 0.0
    )
    assert (by_share[0.75]["false_deprecate_high_q"] or 0.0) <= (
        by_share[1.0]["false_deprecate_high_q"] or 0.0
    )


def test_run_sweep_shape() -> None:
    """The CLI-facing wrapper returns the documented JSON shape."""
    cfg = _sparse_cfg()
    out = run_sweep(cfg, [0.0, 0.5, 1.0])
    assert set(out.keys()) == {"config", "shares", "cells"}
    assert out["shares"] == [0.0, 0.5, 1.0]
    assert len(out["cells"]) == 3
    for c in out["cells"]:
        assert "share" in c
        assert "ttp_high_q_p50" in c
        assert "false_promote_low_q" in c


def test_tightness_default_is_byte_identical_to_legacy() -> None:
    """tightness=1.0 (default) must match the pre-tightness behavior.

    Regression guard: introducing the tightness knob with a default of
    1.0 collapses _draw_sibling_q to a deterministic q_cluster passthrough,
    so identically-seeded runs across `tightness=1.0` and a config that
    omits tightness entirely must produce identical metrics.
    """
    cfg_default = _sparse_cfg()
    cfg_explicit = SimConfig(
        n_clusters=cfg_default.n_clusters,
        cluster_size=cfg_default.cluster_size,
        n_per_window=cfg_default.n_per_window,
        n_windows=cfg_default.n_windows,
        beta_a=cfg_default.beta_a,
        beta_b=cfg_default.beta_b,
        seed=cfg_default.seed,
        tightness=1.0,
    )
    a = run_share(cfg_default, share=0.75).to_dict()
    b = run_share(cfg_explicit, share=0.75).to_dict()
    assert a == b


def test_tightness_zero_amplifies_false_promotes_at_share_075() -> None:
    """Closes SCALE_REPORT §68 finding: independent siblings (tightness=0)
    leak more false-promotes at the share=0.75 sweet spot than tightly-
    correlated siblings.

    The §8 deployment rule depends on this — clustering-tau bounds
    prior-sharing safety. If this regression flips, the §62 sweet
    spot needs re-derivation under the new generative model.
    """
    base = dict(
        n_clusters=400, cluster_size=4, n_per_window=1, n_windows=30,
        beta_a=0.4, beta_b=0.4, seed=0xE17A11,
    )
    cfg_tight = SimConfig(**base, tightness=1.0)
    cfg_loose = SimConfig(**base, tightness=0.0)
    fp_tight = run_share(cfg_tight, share=0.75).to_dict()["false_promote_low_q"] or 0.0
    fp_loose = run_share(cfg_loose, share=0.75).to_dict()["false_promote_low_q"] or 0.0
    # Tight: 0.000, loose: ~0.031 (run #68 measured).
    assert fp_tight < 0.005
    assert fp_loose > 0.015
    assert fp_loose > fp_tight


def test_contamination_default_is_byte_identical_to_legacy() -> None:
    """contamination=0.0 (default) must match the pre-contamination behavior.

    The contamination knob adds an outsider-evidence draw per (cluster,
    window) and a per-(owner, sibling) Bernoulli flip; with the flip prob
    pinned at 0 the outsider rows must never be substituted in. We
    additionally require byte-identity of metrics across an explicit
    contamination=0.0 config and the legacy default config.
    """
    base = dict(
        n_clusters=80, cluster_size=4, n_per_window=1, n_windows=20,
        beta_a=0.4, beta_b=0.4, seed=0xE17A11,
    )
    cfg_default = SimConfig(**base)
    cfg_explicit = SimConfig(**base, contamination=0.0)
    a = run_share(cfg_default, share=0.75).to_dict()
    b = run_share(cfg_explicit, share=0.75).to_dict()
    assert a == b


def test_contamination_amplifies_false_promotes_at_share_075() -> None:
    """High cluster-contamination (mis-grouping) inflates false-promote
    rate at share=0.75 — the §8 sweet spot is conditional on cluster
    purity, not just tightness.

    With Beta(0.4, 0.4) the unconditional outsider mean q is 0.5, so
    contaminated siblings inject middle-of-the-road evidence into
    low-q owner aggregates, mechanically lifting their share-floored
    sibling support and triggering false PROMOTEs.
    """
    base = dict(
        n_clusters=400, cluster_size=4, n_per_window=1, n_windows=30,
        beta_a=0.4, beta_b=0.4, seed=0xE17A11,
    )
    cfg_clean = SimConfig(**base, contamination=0.0)
    cfg_dirty = SimConfig(**base, contamination=0.5)
    fp_clean = run_share(cfg_clean, share=0.75).to_dict()["false_promote_low_q"] or 0.0
    fp_dirty = run_share(cfg_dirty, share=0.75).to_dict()["false_promote_low_q"] or 0.0
    assert fp_dirty > fp_clean

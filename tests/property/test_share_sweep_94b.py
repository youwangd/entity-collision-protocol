"""§94b — share-sweep driver smoke tests.

Validates:

  * The driver runs end-to-end on a tiny corpus.
  * Metrics are deterministic given seed.
  * The §94b finding (share is operationally inert on this corpus)
    is reproducible: all five share values produce identical metrics
    at fixed seed/embedder. If this regression breaks, either the
    decide_with_family math gained bite, the synthesizer changed,
    or the corpus generator drifted — any of which we want to know
    about loudly.
"""
from __future__ import annotations

import pytest

from evals.xs_share_sweep import run_share_sweep


@pytest.mark.evals
def test_share_sweep_runs_e2e_tiny():
    out = run_share_sweep(
        shares=[0.0, 0.75],
        n_facts=20,
        n_sessions=6,
        distractors_per_session=4,
        seed=42,
        k=5,
        embedder_name="hashtrigram",
    )
    assert out["k"] == 5
    assert len(out["points"]) == 2
    for p in out["points"]:
        assert p["treatment"]["n_pairs"] > 0
        # All four metric keys present.
        for key in ("session_hit_at_1", "session_hit_at_k",
                    "pair_recall_at_k", "mean_reciprocal_rank"):
            assert key in p["treatment"]
            assert key in p["delta"]


@pytest.mark.evals
def test_share_sweep_deterministic():
    kw = dict(
        shares=[0.0, 0.5, 1.0],
        n_facts=20,
        n_sessions=6,
        distractors_per_session=4,
        seed=42,
        k=5,
        embedder_name="hashtrigram",
    )
    a = run_share_sweep(**kw)
    b = run_share_sweep(**kw)
    # Strip wall-time noise.
    for d in (a, b):
        d.pop("wall_seconds", None)
        for p in d["points"]:
            p.pop("wall_seconds", None)
    assert a == b


@pytest.mark.evals
def test_share_is_operationally_inert_at_n60():
    """§94b lock: across share ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on the
    §94 reference corpus (n_facts=60, hashtrigram-256, k=10), all
    five treatment arms produce *bit-identical* end-to-end metrics.

    This is the regression. If a future change to `decide_with_family`,
    `schema_family_share` propagation, or the synthesizer makes any
    share value diverge from the others, this test fails — and we
    want to read about it before merging.
    """
    out = run_share_sweep(
        shares=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_facts=60,
        n_sessions=10,
        distractors_per_session=10,
        seed=42,
        k=10,
        embedder_name="hashtrigram",
    )
    treats = [p["treatment"] for p in out["points"]]
    # Drop n_pairs (already structurally constant) — compare metric tuple.
    keys = ("session_hit_at_1", "session_hit_at_k",
            "pair_recall_at_k", "mean_reciprocal_rank")
    tuples = {tuple(t[k] for k in keys) for t in treats}
    assert len(tuples) == 1, (
        f"§94b expected all share values to produce identical metrics, "
        f"got {len(tuples)} distinct tuples: {tuples}"
    )

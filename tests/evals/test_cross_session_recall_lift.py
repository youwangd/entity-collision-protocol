"""Smoke test for the §91 cross-session recall-lift driver."""
from __future__ import annotations

from evals.cross_session_recall_lift import run_cross_session_recall_lift


def test_xs_recall_lift_smoke():
    out = run_cross_session_recall_lift(
        n_facts=12,
        n_sessions=4,
        distractors_per_session=2,
        seed=11,
        k=10,
        embedder_name="hashtrigram",
    )
    assert out["n_pairs"] > 0
    for arm in ("baseline", "treatment"):
        for m in ("session_hit_at_1", "session_hit_at_k",
                  "pair_recall_at_k", "mean_reciprocal_rank"):
            v = out[arm][m]
            assert 0.0 <= v <= 1.0, f"{arm}.{m}={v} out of range"
    # Delta values bounded by [-1, 1]
    for m in ("session_hit_at_1", "session_hit_at_k",
              "pair_recall_at_k", "mean_reciprocal_rank"):
        v = out["delta"][m]
        assert -1.0 <= v <= 1.0
    # Recipe locked from §87.
    assert out["recipe"]["schema_family_share"] == 0.75
    assert out["recipe"]["schema_family_tau"] == 0.20
    assert out["consolidation_errors"] == []


def test_xs_recall_lift_baseline_correctness():
    """Sanity: the cross-session corpus is solvable. Both halves of each
    pair contain the answer anchor, so a working retriever should hit
    *some* gold session for most queries even at the baseline."""
    out = run_cross_session_recall_lift(
        n_facts=24,
        n_sessions=6,
        distractors_per_session=3,
        seed=7,
        k=10,
        embedder_name="hashtrigram",
    )
    # We don't claim a specific number; we claim recall@k is "useful".
    assert out["baseline"]["session_hit_at_k"] >= 0.5

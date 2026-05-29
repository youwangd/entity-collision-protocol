"""Regression guard: v0.3 default operating point.

Pins the *defaults* the v0.3 paper draft and SCALE_REPORT.md depend on.
Any future change to these values must:

  1. Be deliberate (this test must be edited in the same commit).
  2. Be backed by a re-bench against LongMemEval / LoCoMo so the
     headline numbers stay defensible.

The defaults pinned here come from:

  * commit `e428687` (PRF + anchor-share gate ON by default)
  * commit flipping `vector_weight` 0.5 → 0.3 (Pareto-defensible
    operating point, §4.5 in `paper/40_results.md`)
  * commit `c97cd0a` (§4.15-profile-skip-rerank lever stays default OFF
    because measured gain didn't clear the pre-registered ≥10% p95
    threshold)
  * commit `b0` setting `respect_schema_lifecycle` default True

If you're flipping a default, edit this test in the same patch and cite
the bench artifact in the commit message.
"""
from __future__ import annotations

from engram.core.config import RetrievalConfig


def test_v0_3_retrieval_defaults_locked() -> None:
    cfg = RetrievalConfig()

    # §4.5 — fusion weight Pareto-optimal at 0.3 (BM25-leaning).
    assert cfg.vector_weight == 0.3, (
        f"vector_weight default drifted: {cfg.vector_weight}; "
        "see paper/40_results.md §4.5"
    )

    # §4.8.2.4 — n=500 LongMemEval-S paired re-bench falsified ON-by-
    # default; flipped back to None (OFF). Knob remains for opt-in.
    assert cfg.query_expansion_min_dominance is None, (
        f"query_expansion_min_dominance default drifted: "
        f"{cfg.query_expansion_min_dominance}; "
        "see paper/40_results.md §4.8.2.4 — Δhit@1 [−0.042, −0.002]"
    )
    assert cfg.query_expansion_anchor_share_max == 0.5, (
        f"query_expansion_anchor_share_max default drifted: "
        f"{cfg.query_expansion_anchor_share_max}; see commit e428687"
    )

    # §4.15-profile — measured p95 cut did NOT clear pre-registered
    # ≥10% bar; lever stays default OFF.
    assert cfg.query_expansion_skip_rerank_first_pass is False, (
        "skip-rerank-first-pass default flipped without clearing the "
        "pre-registered ≥10% p95 threshold; see commit c97cd0a"
    )

    # Lifecycle filter on by default — deprecated SCHEMA candidates
    # must not surface.
    assert cfg.respect_schema_lifecycle is True, (
        "respect_schema_lifecycle default drifted from True; "
        "deprecated schemas would leak into recall"
    )

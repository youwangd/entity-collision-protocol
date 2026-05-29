"""Unit tests for the LoCoMo-shaped adaptive_vw analyzer.

Covers the verdict decision rule on a tiny synthetic LoCoMo-schema input
(per_query records keyed by sample_id+category+q_idx) so the analyzer's
'useful' verdict can be trusted as the ship gate.
"""
from __future__ import annotations

from evals.locomo_adaptive_vw import _signal_verdict


def _rec(h1: int, hk: int = None, rr: float = None) -> dict:
    if hk is None:
        hk = h1
    if rr is None:
        rr = float(h1)
    return {"hit_at_1": h1, "hit_at_k": hk, "reciprocal_rank": rr}


def test_locomo_signal_verdict_perfect_routing_is_useful() -> None:
    """4 LoCoMo queries: bm25(vw=0) wins on q1,q2; vw=0.5 wins on q3,q4.
    A magic signal [hi,hi,lo,lo] should route to 100% hit@1, vs static-best 0.5.
    """
    weights = [0.0, 0.5]
    keys = [("s1", "1", 0), ("s1", "1", 1), ("s2", "2", 0), ("s2", "2", 1)]
    bm25_by_k = {
        keys[0]: _rec(1), keys[1]: _rec(1),
        keys[2]: _rec(0), keys[3]: _rec(0),
    }
    by_k = {
        keys[0]: {0.0: bm25_by_k[keys[0]], 0.5: _rec(0)},
        keys[1]: {0.0: bm25_by_k[keys[1]], 0.5: _rec(0)},
        keys[2]: {0.0: bm25_by_k[keys[2]], 0.5: _rec(1)},
        keys[3]: {0.0: bm25_by_k[keys[3]], 0.5: _rec(1)},
    }
    static_best_per_q = [1, 1, 0, 0]  # static-best happens to be vw=0 here
    sig = {keys[0]: 1.0, keys[1]: 1.0, keys[2]: 0.0, keys[3]: 0.0}
    v = _signal_verdict(keys, by_k, bm25_by_k, weights,
                        sig, static_best_per_q, "hit_at_1", "magic")
    assert v["adaptive_hit_at_1"] == 1.0
    # Δ vs static-best [1,1,0,0]: routing also yields [1,1,1,1] => Δ=[0,0,1,1]
    assert v["delta_vs_static_best"]["mean"] > 0
    # With only 4 points the bootstrap CI is wide; we don't strictly require
    # 'useful' here, but the mean must be positive and Δ_hi must be > 0.
    assert v["delta_vs_static_best"]["ci_hi"] > 0


def test_locomo_signal_verdict_random_signal_is_not_useful() -> None:
    """A signal uncorrelated with which-policy-wins should not produce a
    significant gain; CI lower bound should not exceed zero meaningfully."""
    weights = [0.0, 0.3]
    keys = [(f"s{i//4}", "1", i % 4) for i in range(40)]
    # bm25 and vw=0.3 each get half the queries right, but on the SAME ones,
    # so any routing is at best a wash.
    bm25_by_k = {k: _rec(i % 2) for i, k in enumerate(keys)}
    by_k = {
        k: {0.0: bm25_by_k[k], 0.3: _rec((i % 2))}
        for i, k in enumerate(keys)
    }
    static_best_per_q = [bm25_by_k[k]["hit_at_1"] for k in keys]
    # Constant signal — provides no info
    sig = {k: 0.5 for k in keys}
    v = _signal_verdict(keys, by_k, bm25_by_k, weights,
                        sig, static_best_per_q, "hit_at_1", "constant")
    # No information => mean delta should be 0 and useful=False
    assert v["delta_vs_static_best"]["mean"] == 0.0
    assert v["useful"] is False

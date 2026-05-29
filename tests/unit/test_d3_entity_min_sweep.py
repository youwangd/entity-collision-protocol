"""§D3-collateral-(d) — entity_min sweep driver smoke test.

Pins the schema of `evals.synthetic_supersede_d3_entity_min_sweep.run_sweep`:
list of points, each with delta dict keyed by d_hit_at_1/d_hit_at_k/
d_stale_at_1/d_stale_at_k. Uses a tiny corpus (n_slots=4) and few
bootstrap resamples so this stays under a few hundred ms.
"""
from __future__ import annotations

from evals.synthetic_supersede_d3_entity_min_sweep import run_sweep


def test_entity_min_sweep_schema():
    rep = run_sweep(
        entity_min_list=[0.3, 0.7],
        n_slots=4,
        updates_per_slot=2,
        distractors=4,
        seed=42,
        k=5,
        resamples=100,
        boot_seed=42,
    )
    assert rep["config"]["entity_min_list"] == [0.3, 0.7]
    assert len(rep["points"]) == 2
    seen_em = [p["entity_min"] for p in rep["points"]]
    assert seen_em == [0.3, 0.7]
    for p in rep["points"]:
        d = p["delta"]
        for k in ("d_hit_at_1", "d_hit_at_k", "d_stale_at_1", "d_stale_at_k"):
            assert k in d
            c = d[k]
            assert c["ci_lo"] <= c["mean_diff_default_minus_addonly"] <= c["ci_hi"]
            assert 0.0 <= c["p_bootstrap_two_sided"] <= 1.0
        assert p["n_queries"] >= 1
        assert p["default_interference_actions"] >= 0

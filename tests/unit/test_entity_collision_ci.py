"""Smoke test for entity_collision_ci driver."""
from __future__ import annotations

import json
from pathlib import Path

from evals.entity_collision_ci import _paired, _summarize


def _mk_pq(hits, ranks_for_mrr=None):
    return [{"hit_at_1": h, "hit_at_k": h,
             "reciprocal_rank": (1.0 if h else 0.0)} for h in hits]


def test_summarize_basic():
    pq = _mk_pq([1, 1, 0, 0])
    out = _summarize(pq, resamples=200, seed=1)
    assert out["n"] == 4
    for m in ("hit_at_1", "hit_at_k", "mrr"):
        cell = out[m]
        assert 0.0 <= cell["ci_lo"] <= cell["mean"] <= cell["ci_hi"] <= 1.0


def test_paired_signed_diff():
    a = _mk_pq([1, 1, 1, 0])  # vec
    b = _mk_pq([0, 0, 1, 0])  # bm25
    out = _paired(a, b, resamples=200, seed=1)
    assert out["n"] == 4
    # Δhit@1 mean should be ~0.5
    assert out["hit_at_1"]["mean"] == 0.5
    assert out["hit_at_1"]["ci_lo"] >= 0.0


def test_cli_roundtrip(tmp_path: Path, monkeypatch):
    # Build a tiny sweep JSON in the schema entity_collision_sweep emits.
    pq_bm = _mk_pq([1, 1, 0, 0])
    pq_vec = _mk_pq([1, 1, 1, 0])
    sweep = {
        "config": {"tag": "service"},
        "sweep": [{
            "collision_degree": 4,
            "bm25_floor": 0.25,
            "delta_hit_at_1": 0.25,
            "delta_mrr": 0.25,
            "bm25_only": {"per_query": pq_bm},
            "vector_fusion": {"per_query": pq_vec},
        }],
    }
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    inp.write_text(json.dumps(sweep))

    import sys
    from evals import entity_collision_ci as mod

    argv = ["entity_collision_ci", "--in", str(inp), "--out", str(out),
            "--resamples", "100", "--seed", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()

    obj = json.loads(out.read_text())
    assert obj["rows"][0]["delta_ci"]["hit_at_1"]["mean"] == 0.25
    assert obj["rows"][0]["bm25_only_summary"]["n"] == 4

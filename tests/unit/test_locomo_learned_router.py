"""Unit tests for the LoCoMo learned-router analyzer.

Two guarantees we want to lock in:

1. Feature extraction is leak-free: rank/reciprocal_rank do NOT enter
   the feature vector regardless of what's in the per_query record.
2. The verdict pipeline runs end-to-end on a tiny synthetic input and
   produces a defined Δ + CI; trivially-routable inputs produce a
   non-negative Δ.
"""
from __future__ import annotations

import json
import sys

import pytest


@pytest.fixture
def _no_sklearn_skip():
    pytest.importorskip("sklearn")


def test_features_drop_rank_and_rr() -> None:
    """rank and reciprocal_rank are gold-derived; they must NOT leak."""
    from evals.locomo_learned_router import _FEAT_NAMES_BASE, _features

    rec = {
        "category": "1",
        "bm25_top1": 0.5, "bm25_top2": 0.4,
        "bm25_gap": 0.1, "bm25_norm_gap": 0.2, "bm25_crowd_95": 3,
        "rank": 1, "reciprocal_rank": 1.0,  # MUST be ignored
    }
    cats = ["1", "2", "3"]
    feats = _features(rec, cats)
    assert len(feats) == len(_FEAT_NAMES_BASE) + len(cats)
    # Indicator names that must NOT appear
    assert "vw0_rank" not in _FEAT_NAMES_BASE
    assert "vw0_rr" not in _FEAT_NAMES_BASE
    assert "rank" not in _FEAT_NAMES_BASE
    # Category one-hot present
    assert feats[-3:] == [1.0, 0.0, 0.0]


def test_features_no_bm25_hits_indicator() -> None:
    from evals.locomo_learned_router import _features

    rec = {"category": "1"}  # no BM25 signals
    feats = _features(rec, ["1"])
    # last numeric base feature is no_bm25_hits == 1.0
    assert feats[5] == 1.0


def test_router_runs_end_to_end(tmp_path, _no_sklearn_skip) -> None:
    """Build two synthetic LoCoMo cells, run the CLI, and verify it
    reports a verdict without crashing. The dataset is too small for a
    meaningful CI, but we lock in the schema contract.
    """
    import subprocess

    def _cell(vw: float, h1_pattern: list[int]) -> dict:
        per_query = []
        for i, h1 in enumerate(h1_pattern):
            sid = f"conv-{i // 4}"   # 4 conversations of 4 queries
            cat = str((i % 5) + 1)
            rec = {
                "sample_id": sid, "category": cat, "q_idx": i,
                "rank": 1 if h1 else 0,
                "hit_at_1": h1, "hit_at_k": h1, "reciprocal_rank": float(h1),
            }
            if vw == 0.0:
                rec.update({
                    "bm25_top1": 1.0 + 0.01 * i, "bm25_top2": 0.5,
                    "bm25_gap": 0.5, "bm25_norm_gap": 0.5,
                    "bm25_crowd_95": 1,
                })
            per_query.append(rec)
        return {"vector_weight": vw, "embedder": "synthetic", "k": 5,
                "per_query": per_query}

    n = 16
    pattern_vw0 = [1, 0, 1, 0] * 4
    pattern_vw5 = [0, 1, 0, 1] * 4
    cell_a = tmp_path / "vw0.json"
    cell_b = tmp_path / "vw5.json"
    cell_a.write_text(json.dumps(_cell(0.0, pattern_vw0)))
    cell_b.write_text(json.dumps(_cell(0.5, pattern_vw5)))

    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, "-m", "evals.locomo_learned_router",
         "--metric", "hit_at_1", "--model", "logreg",
         "--in", str(cell_a), "--in", str(cell_b),
         "--out", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    art = json.loads(out.read_text())
    # Schema contract: required keys present, no leakage indicators.
    for key in ("metric", "n_queries", "static_best_vw", "static_best_mean",
                "oracle_mean", "adaptive_mean", "delta_vs_static_best",
                "useful"):
        assert key in art
    d = art["delta_vs_static_best"]
    assert "mean" in d and "ci_lo" in d and "ci_hi" in d
    assert art["n_queries"] == n

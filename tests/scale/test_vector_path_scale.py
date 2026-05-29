"""Vector-path scale tests — opt-in via pytest -m scale.

Mirrors test_ingest_scale.py but with the *full* retrieval stack wired up:
HashTrigram embeddings + SQLiteVecStore + hybrid RRF (BM25 + vector). This is
the configuration that exercises every code path the v0.2 paper plans to
defend, and the one whose latency curve has been a question mark in
SCALE_REPORT.md.

Why HashTrigram and not SentenceTransformer? Two reasons:
  1. ST cold-load is ~3-5s and a single embed is ~3-10ms — at 10k that adds
     30-100s of pure model time and makes the latency histogram unreadable.
  2. We already have an ST sweep (n=100) banked in SCALE_REPORT §"ST sweep".
     The interesting unknown is *throughput at scale of the vector pipe
     itself*, which HashTrigram isolates: ~50µs per embed, dominated by
     I/O and the sqlite-vec INSERT.

Recall is measured on a held-out *strict-paraphrase* slice from
evals.synthetic so the numbers are comparable to the strict-paraphrase ST
sweep (vw=0.5 cell). Expected: HashTrigram + vw=0.5 should beat pure BM25
on hit@1 (trigrams overlap on morphological variants like
"prefers"/"favors") but lose to ST.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram import Engram, Config
from engram.providers.embeddings import HashTrigramEmbeddingProvider


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _record(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"{name}_{sha}_{ts}.json"
    payload["meta"] = {"sha": sha, "timestamp": ts, "name": name}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[scale] wrote {out}")


@pytest.mark.scale
@pytest.mark.slow
def test_vector_path_ingest_10k(tmp_path: Path):
    """10k ingest with HashTrigram + sqlite-vec wired up. p50/p95/p99 + throughput."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.vector_weight = 0.5  # hybrid; matches ST sweep midpoint
    eng = Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=256))
    try:
        templates = [
            "User {} prefers {} for {} workflows.",
            "Deploy of service {} failed at {} with error code {}.",
            "Meeting note: {} aligned on {} by EOQ {}.",
            "Bug report: {} returned {} when {} was set.",
            "Insight: {} correlates with {} in cohort {}.",
        ]
        n = 10_000
        contents = [
            templates[i % len(templates)].format(f"user{i}", f"x{i % 17}", f"y{i % 31}")
            for i in range(n)
        ]
        latencies_ms: list[float] = []
        embed_latencies_us: list[float] = []
        wall_start = time.monotonic()
        for c in contents:
            t0 = time.monotonic()
            eng.remember(c)
            latencies_ms.append((time.monotonic() - t0) * 1000)
        wall_total = time.monotonic() - wall_start

        # Standalone embed cost (informational; remember() already embeds inline).
        for c in contents[:200]:
            t0 = time.monotonic()
            eng._embeddings.embed(c)
            embed_latencies_us.append((time.monotonic() - t0) * 1_000_000)

        status = eng.status()
        assert status["total_memories"] == n
        assert eng._vector.count() == n, f"vector count {eng._vector.count()} != {n}"

        result = {
            "n": n,
            "embed_provider": "HashTrigram",
            "embed_dimension": 256,
            "vector_weight": cfg.retrieval.vector_weight,
            "wall_seconds": round(wall_total, 3),
            "throughput_per_sec": round(n / wall_total, 1),
            "latency_ms": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
                "p99": round(_percentile(latencies_ms, 99), 3),
                "max": round(max(latencies_ms), 3),
                "mean": round(statistics.mean(latencies_ms), 3),
            },
            "embed_latency_us_p50": round(_percentile(embed_latencies_us, 50), 1),
            "embed_latency_us_p95": round(_percentile(embed_latencies_us, 95), 1),
            "vector_count": eng._vector.count(),
        }
        _record("vector_ingest_10k_hashtrigram", result)

        assert result["latency_ms"]["p99"] < 50, (
            f"p99 vector-write latency too high: {result['latency_ms']['p99']}ms"
        )
    finally:
        eng.close()


@pytest.mark.mega_scale
@pytest.mark.slow
def test_vector_path_ingest_100k(tmp_path: Path):
    """100k ingest with HashTrigram + sqlite-vec. Stratified latency curve.

    Records p50/p95/p99 per 10k bucket so we can see if sqlite-vec INSERT
    degrades super-linearly as the index grows. ~16 min wall expected.
    """
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.vector_weight = 0.5
    eng = Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=256))
    try:
        templates = [
            "User {} prefers {} for {} workflows.",
            "Deploy of service {} failed at {} with error code {}.",
            "Meeting note: {} aligned on {} by EOQ {}.",
            "Bug report: {} returned {} when {} was set.",
            "Insight: {} correlates with {} in cohort {}.",
        ]
        n = 100_000
        bucket = 10_000
        latencies_ms: list[float] = []
        bucket_curves: list[dict] = []
        wall_start = time.monotonic()
        for i in range(n):
            c = templates[i % len(templates)].format(
                f"user{i}", f"x{i % 17}", f"y{i % 31}"
            )
            t0 = time.monotonic()
            eng.remember(c)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            if (i + 1) % bucket == 0:
                window = latencies_ms[-bucket:]
                bucket_curves.append({
                    "n_so_far": i + 1,
                    "p50": round(_percentile(window, 50), 3),
                    "p95": round(_percentile(window, 95), 3),
                    "p99": round(_percentile(window, 99), 3),
                    "mean": round(statistics.mean(window), 3),
                })
        wall_total = time.monotonic() - wall_start

        status = eng.status()
        assert status["total_memories"] == n
        assert eng._vector.count() == n

        result = {
            "n": n,
            "embed_provider": "HashTrigram",
            "embed_dimension": 256,
            "vector_weight": cfg.retrieval.vector_weight,
            "wall_seconds": round(wall_total, 3),
            "throughput_per_sec": round(n / wall_total, 1),
            "latency_ms_overall": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
                "p99": round(_percentile(latencies_ms, 99), 3),
                "max": round(max(latencies_ms), 3),
                "mean": round(statistics.mean(latencies_ms), 3),
            },
            "latency_curve_per_10k": bucket_curves,
            "vector_count": eng._vector.count(),
        }
        _record("vector_ingest_100k_hashtrigram", result)

        # Sanity: p99 should still be under 100ms even at 100k.
        assert result["latency_ms_overall"]["p99"] < 100, (
            f"p99 too high at 100k: {result['latency_ms_overall']['p99']}ms"
        )
        # Super-linear check: last bucket p50 not more than 5x first bucket p50.
        first_p50 = bucket_curves[0]["p50"]
        last_p50 = bucket_curves[-1]["p50"]
        assert last_p50 < first_p50 * 5 + 5, (
            f"super-linear degradation: p50 went {first_p50} -> {last_p50}"
        )
    finally:
        eng.close()


@pytest.mark.scale
@pytest.mark.slow
def test_vector_path_recall_strict_paraphrase(tmp_path: Path):
    """Strict-paraphrase recall@k with HashTrigram + sqlite-vec across vw cells."""
    from evals.synthetic import generate_dataset
    from evals.metrics import find_match_rank, hit_at_k, mrr

    ds = generate_dataset(
        n_sessions=20,
        facts_per_session=5,
        distractors_per_session=10,
        seed=42,
        strict_paraphrase=True,
    )

    cells = [("bm25_only", 0.0), ("hybrid_05", 0.5), ("vector_only", 1.0)]
    cell_results: dict[str, dict] = {}
    for label, vw in cells:
        cfg = Config(path=str(tmp_path / f"engram_{label}"))
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.vector_weight = vw
        eng = Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=256))
        try:
            for content, _meta in ds.memories:
                eng.remember(content)
            ranks: list[int | None] = []
            lats: list[float] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=10)
                lats.append((time.monotonic() - t0) * 1000)
                ranks.append(find_match_rank(results, q.expected_substrings))
            cell_results[label] = {
                "vector_weight": vw,
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, 5), 4),
                "hit_at_10": round(hit_at_k(ranks, 10), 4),
                "mrr": round(mrr(ranks), 4),
                "recall_lat_ms_p50": round(_percentile(lats, 50), 3),
                "recall_lat_ms_p95": round(_percentile(lats, 95), 3),
                "recall_lat_ms_p99": round(_percentile(lats, 99), 3),
            }
        finally:
            eng.close()

    _record("vector_recall_strict_hashtrigram", {
        "embed_provider": "HashTrigram",
        "embed_dimension": 256,
        "dataset": "strict_paraphrase n_sessions=20 facts=5 distractors=10",
        "cells": cell_results,
    })

    best_h1 = max(c["hit_at_1"] for c in cell_results.values())
    assert best_h1 > 0.10, f"all cells suspiciously bad: {cell_results}"


@pytest.mark.mega_scale
@pytest.mark.slow
def test_vector_path_recall_st_1k(tmp_path: Path):
    """ST (all-MiniLM-L6-v2) recall@k at ~1k corpus, strict-paraphrase.

    Closes the last open scale item from NEXT.md. The HashTrigram strict
    sweep (n=232 queries) plateaued at hit@1 ≈ 0.38 with vw=0.7. The open
    question is whether a real semantic embedder (ST/MiniLM-384) clears
    that bar at 1k corpus and whether the vw curve looks the same.

    Budget: ST cold-load ~7s + ~10ms/embed → ingest ~10-15s/cell, recall
    ~5-10s/cell across 3 cells. Total ~60-90s. Marked mega_scale so it
    runs in the cron's mega slot, not the default scale tier.
    """
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("sqlite_vec")
    from engram.providers.embeddings import SentenceTransformerProvider
    from evals.synthetic import generate_dataset
    from evals.metrics import find_match_rank, hit_at_k, mrr

    ds = generate_dataset(
        n_sessions=50,         # 50 * (5 facts + 15 distractors) = 1000 memories
        facts_per_session=5,
        distractors_per_session=15,
        seed=42,
        strict_paraphrase=True,
    )
    assert len(ds.memories) == 1000, f"expected 1000 memories, got {len(ds.memories)}"
    n_queries = len(ds.queries)

    # Single shared ST provider — cold-load once, reuse across cells.
    embedder = SentenceTransformerProvider()

    cells = [
        ("bm25_only", 0.0),
        ("hybrid_03", 0.3),
        ("hybrid_05", 0.5),
        ("hybrid_07", 0.7),
        ("vector_only", 1.0),
    ]
    cell_results: dict[str, dict] = {}
    cell_per_query: dict[str, list[dict]] = {}
    ingest_walls: dict[str, float] = {}

    for label, vw in cells:
        cfg = Config(path=str(tmp_path / f"engram_st_{label}"))
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.vector_weight = vw
        eng = Engram(config=cfg, embeddings=embedder)
        try:
            t_ingest = time.monotonic()
            ingest_lats: list[float] = []
            for content, _meta in ds.memories:
                t0 = time.monotonic()
                eng.remember(content)
                ingest_lats.append((time.monotonic() - t0) * 1000)
            ingest_walls[label] = round(time.monotonic() - t_ingest, 3)

            ranks: list[int | None] = []
            recall_lats: list[float] = []
            pq: list[dict] = []
            for q_idx, q in enumerate(ds.queries):
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=10)
                recall_lats.append((time.monotonic() - t0) * 1000)
                r = find_match_rank(results, q.expected_substrings)
                ranks.append(r)
                # find_match_rank is 0-indexed; convert to 1-indexed for the dump.
                pq.append({
                    "query_idx": q_idx,
                    "rank": (r + 1) if r is not None else None,
                    "hit_at_1": 1.0 if r == 0 else 0.0,
                    "hit_at_k": 1.0 if (r is not None and r < 10) else 0.0,
                    "reciprocal_rank": (1.0 / (r + 1)) if r is not None else 0.0,
                })
            cell_per_query[label] = pq

            cell_results[label] = {
                "vector_weight": vw,
                "n_memories": len(ds.memories),
                "n_queries": n_queries,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, 5), 4),
                "hit_at_10": round(hit_at_k(ranks, 10), 4),
                "mrr": round(mrr(ranks), 4),
                "ingest_lat_ms_p50": round(_percentile(ingest_lats, 50), 3),
                "ingest_lat_ms_p99": round(_percentile(ingest_lats, 99), 3),
                "recall_lat_ms_p50": round(_percentile(recall_lats, 50), 3),
                "recall_lat_ms_p95": round(_percentile(recall_lats, 95), 3),
                "recall_lat_ms_p99": round(_percentile(recall_lats, 99), 3),
            }
        finally:
            eng.close()

    # Build bootstrap_ci-compatible "sweep rows": pair each non-bm25 cell against
    # bm25_only as the baseline-vs-bm25 contrast, and dump per_query for paired CIs.
    bm25_pq = cell_per_query.get("bm25_only", [])
    sweep_rows = []
    for label, vw in cells:
        sweep_rows.append({
            "label": label,
            "vector_weight": vw,
            "n_queries": n_queries,
            "baseline_per_query": cell_per_query[label],
            "bm25_only_per_query": bm25_pq if label != "bm25_only" else None,
            **cell_results[label],
        })

    _record("vector_recall_st_1k_strict", {
        "embed_provider": "SentenceTransformer",
        "embed_model": "all-MiniLM-L6-v2",
        "embed_dimension": 384,
        "dataset": "strict_paraphrase n_sessions=50 facts=5 distractors=15",
        "n_memories": len(ds.memories),
        "n_queries": n_queries,
        "ingest_walls_seconds": ingest_walls,
        "cells": cell_results,
        "sweep": sweep_rows,
    })

    # Sanity: at least one cell beats HashTrigram strict baseline (~0.38 hit@1).
    best_h1 = max(c["hit_at_1"] for c in cell_results.values())
    assert best_h1 > 0.30, f"ST cells suspiciously bad: {cell_results}"

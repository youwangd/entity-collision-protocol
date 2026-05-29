"""Run the synthetic benchmark against Engram and print recall metrics.

Usage:
    python -m evals.run --n-sessions 10 --distractors 20 --k 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time

from engram import Engram, Config

from .metrics import find_match_rank, hit_at_k, mrr, ndcg_at_k
from .synthetic import generate_dataset
from evals.io_utils import atomic_write_json


def run_eval(
    n_sessions: int = 10,
    facts_per_session: int = 5,
    distractors_per_session: int = 20,
    seed: int = 42,
    k: int = 5,
) -> dict:
    ds = generate_dataset(
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        distractors_per_session=distractors_per_session,
        seed=seed,
    )
    print(f"[evals] dataset: {len(ds.memories)} memories, {len(ds.queries)} queries")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0
            print(f"[evals] ingested in {ingest_s:.2f}s ({len(ds.memories)/ingest_s:.0f}/s)")

            ranks: list[int | None] = []
            recall_lat: list[float] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=k)
                recall_lat.append((time.monotonic() - t0) * 1000)
                ranks.append(find_match_rank(results, q.expected_substrings))

            metrics = {
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "k": k,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, min(5, k)), 4),
                "hit_at_k": round(hit_at_k(ranks, k), 4),
                "mrr": round(mrr(ranks), 4),
                "ndcg_at_k": round(ndcg_at_k(ranks, k), 4),
                "recall_latency_ms": {
                    "p50": round(statistics.median(recall_lat), 3),
                    "mean": round(statistics.mean(recall_lat), 3),
                    "max": round(max(recall_lat), 3),
                },
                "ingest_seconds": round(ingest_s, 3),
            }
            return metrics
        finally:
            eng.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, default=10)
    p.add_argument("--facts", type=int, default=5)
    p.add_argument("--distractors", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    metrics = run_eval(
        n_sessions=args.n_sessions,
        facts_per_session=args.facts,
        distractors_per_session=args.distractors,
        seed=args.seed,
        k=args.k,
    )
    print(json.dumps(metrics, indent=2))
    if args.out:
        atomic_write_json(args.out, metrics)
        print(f"[evals] wrote {args.out}")


if __name__ == "__main__":
    main()

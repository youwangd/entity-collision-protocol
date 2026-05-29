"""§4.15-profile follow-up — measure `query_expansion_skip_rerank_first_pass`
latency lift on the `both` arm.

Mirrors scripts/profile_recall_prf_sp.py (n=30k, 200 paired queries, seed=42,
HashTrigram-256, vw=0.3, OMP=MKL=1) but compares only:

  both          — PRF on, share_prior on, skip_first_rerank=False (status quo)
  both_skip     — PRF on, share_prior on, skip_first_rerank=True  (v0.3 lever)

Prints a paired latency table and dumps JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from engram import Engram
from engram.core.config import Config
from engram.providers.embeddings import HashTrigramEmbeddingProvider
from evals.io_utils import atomic_write_json


TEMPLATES = [
    "User {} prefers {} for {} workflows.",
    "Deploy of service {} failed at {} with error code {}.",
    "Meeting note: {} aligned on {} by EOQ {}.",
    "Bug report: {} returned {} when {} was set.",
    "Insight: {} correlates with {} in cohort {}.",
]
QUERY_TEMPLATES = [
    "what does user{} prefer",
    "why did service{} fail",
    "what was decided about {}",
    "what bug was reported in {}",
    "what correlates with {}",
]


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _build(eng, n):
    for i in range(n):
        c = TEMPLATES[i % len(TEMPLATES)].format(
            f"user{i}", f"x{i % 17}", f"y{i % 31}"
        )
        eng.remember(c)


def _engine(tmp, label, *, skip):
    cfg = Config(path=str(tmp / f"engram_{label}"))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.vector_weight = 0.3
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_top_k = 10
    cfg.retrieval.query_expansion_anchor_share_max = 0.5  # v0.3 default
    cfg.retrieval.reranker = "share_prior"
    cfg.retrieval.rerank_pool_size = 20
    cfg.retrieval.query_expansion_skip_rerank_first_pass = skip
    return Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=256))


def _measure(eng, queries):
    lats = []
    for q in queries:
        t0 = time.monotonic()
        eng.recall(q, limit=10)
        lats.append((time.monotonic() - t0) * 1000)
    return lats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-memories", type=int, default=30_000)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--tmpdir", type=str, default="/tmp/engram_profile_skip")
    args = p.parse_args()

    import random
    rng = random.Random(args.seed)
    queries = [
        QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)].format(
            rng.choice([f"user{rng.randint(0, args.n_memories - 1)}",
                        f"x{rng.randint(0, 16)}",
                        f"y{rng.randint(0, 30)}"])
        )
        for i in range(args.n_queries)
    ]

    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)

    out = {"config": {
        "n_memories": args.n_memories, "n_queries": args.n_queries,
        "seed": args.seed, "embed": "HashTrigram-256", "vector_weight": 0.3,
        "anchor_share_max": 0.5, "min_dominance": 0.3,
    }, "arms": {}}

    for label, skip in [("both", False), ("both_skip", True)]:
        print(f"[{label}] building n={args.n_memories} ...", flush=True)
        eng = _engine(tmp, label, skip=skip)
        try:
            t0 = time.monotonic()
            _build(eng, args.n_memories)
            ingest_s = time.monotonic() - t0
            print(f"[{label}] ingest {ingest_s:.1f}s, recall x{args.n_queries} ...",
                  flush=True)
            lats = _measure(eng, queries)
            out["arms"][label] = {
                "ingest_seconds": round(ingest_s, 2),
                "lat_ms": {
                    "p50": round(_percentile(lats, 50), 3),
                    "p95": round(_percentile(lats, 95), 3),
                    "p99": round(_percentile(lats, 99), 3),
                    "mean": round(statistics.mean(lats), 3),
                    "max": round(max(lats), 3),
                },
            }
        finally:
            eng.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out, indent=2)
    print(json.dumps(out["arms"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

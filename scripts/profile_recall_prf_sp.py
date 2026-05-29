"""§4.15 latency follow-up — cProfile stage breakdown for PRF × share_prior.

Builds a 30k-memory HashTrigram corpus (deterministic templates), runs N
recall queries under four arms, and dumps per-arm cumulative cProfile
stats plus p50/p95/p99 wall-clock per query.

Arms:
  baseline       — vw=0.3, no PRF, no reranker
  prf            — vw=0.3, PRF on (min_dominance=0.3, top_k=10), no reranker
  sp             — vw=0.3, no PRF, reranker=share_prior (α=0.05)
  both           — vw=0.3, PRF on, reranker=share_prior (the §4.11 op point)

Usage:
  python scripts/profile_recall_prf_sp.py \
      --n-memories 30000 --n-queries 200 --seed 42 \
      --out bench/results/profile_recall_prf_sp.json

Notes:
  - Uses HashTrigram (no model load), so deltas are pure pipeline cost.
  - Single-threaded (OMP/MKL/etc set to 1) — caller's responsibility if
    running alongside another job; this script does NOT clamp env.
  - Saves the top 40 cumulative-time entries per arm in JSON.

Run locally only; this is a profiling helper, not part of pytest."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path

from engram import Engram
from evals.io_utils import atomic_write_json
from engram.core.config import Config
from engram.providers.embeddings import HashTrigramEmbeddingProvider


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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _build_corpus(eng: Engram, n: int) -> None:
    for i in range(n):
        c = TEMPLATES[i % len(TEMPLATES)].format(
            f"user{i}", f"x{i % 17}", f"y{i % 31}"
        )
        eng.remember(c)


def _make_engine(tmp: Path, label: str, *, prf: bool, sp: bool) -> Engram:
    cfg = Config(path=str(tmp / f"engram_{label}"))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.vector_weight = 0.3
    if prf:
        cfg.retrieval.query_expansion_min_dominance = 0.3
        cfg.retrieval.query_expansion_top_k = 10
    if sp:
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.rerank_pool_size = 20
    return Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=256))


def _profile_arm(
    eng: Engram, queries: list[str], *, top_n: int = 40
) -> tuple[list[float], list[dict]]:
    pr = cProfile.Profile()
    lats_ms: list[float] = []
    pr.enable()
    for q in queries:
        t0 = time.monotonic()
        eng.recall(q, limit=10)
        lats_ms.append((time.monotonic() - t0) * 1000)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(top_n)

    rows: list[dict] = []
    # Walk pstats internal table; topN is the order we just printed.
    stats_items = sorted(
        ps.stats.items(), key=lambda kv: kv[1][3], reverse=True
    )[:top_n]
    for (file, line, func), (cc, nc, tt, ct, _callers) in stats_items:
        rows.append(
            {
                "func": f"{Path(file).name}:{line}({func})",
                "ncalls": nc,
                "tottime_s": round(tt, 4),
                "cumtime_s": round(ct, 4),
            }
        )
    return lats_ms, rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-memories", type=int, default=30_000)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n-funcs", type=int, default=40)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--tmpdir", type=str, default="/tmp/engram_profile")
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

    arms = [
        ("baseline", False, False),
        ("prf", True, False),
        ("sp", False, True),
        ("both", True, True),
    ]

    out: dict = {
        "config": {
            "n_memories": args.n_memories,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "embed_provider": "HashTrigram-256",
            "vector_weight": 0.3,
        },
        "arms": {},
    }

    for label, prf, sp in arms:
        print(f"[{label}] building corpus n={args.n_memories} ...", flush=True)
        eng = _make_engine(tmp, label, prf=prf, sp=sp)
        try:
            t0 = time.monotonic()
            _build_corpus(eng, args.n_memories)
            ingest_s = time.monotonic() - t0
            print(f"[{label}] ingest done in {ingest_s:.1f}s, profiling recall ...",
                  flush=True)
            lats, rows = _profile_arm(eng, queries, top_n=args.top_n_funcs)
            out["arms"][label] = {
                "ingest_seconds": round(ingest_s, 2),
                "lat_ms": {
                    "p50": round(_percentile(lats, 50), 3),
                    "p95": round(_percentile(lats, 95), 3),
                    "p99": round(_percentile(lats, 99), 3),
                    "mean": round(statistics.mean(lats), 3),
                    "max": round(max(lats), 3),
                },
                "top_funcs_cum": rows,
            }
        finally:
            eng.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

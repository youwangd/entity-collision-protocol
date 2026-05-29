"""Paraphrase-density sweep driver.

Runs the BM25-only and vector-fusion baselines across a sweep of
``overlap_target`` T values to test the phase-transition hypothesis:

    Δhit@1(vector − bm25)  is small (near 0) at high T (verbatim),
                            and grows as T drops (paraphrased).

If the hypothesis holds, we'll see a monotone-ish curve where vector
retrieval *only* pays once lexical overlap drops below some threshold T*.
That threshold becomes the headline number for the v0.2 paper framing.

Usage
-----
    python -m evals.paraphrase_density_sweep \\
        --embed st --n-facts 120 --distractors 5 \\
        --overlaps 0.0,0.25,0.5,0.75,1.0 \\
        --out bench/results/pd_sweep_st.json

Notes
-----
- We compare exactly two configs per T cell: ``bm25_only`` (vw=0) and
  ``vector_fusion`` (vw=0.5). Other signals (recency/salience/context)
  match the bm25_only ablation so the only varying factor is vw.
- Same seed across cells → same fact bindings, only query phrasing varies.
- Per-query ranks captured so a paired bootstrap CI can run later via
  ``evals.bootstrap_ci``.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from engram import Config, Engram

from .ablation import _make_embedder
from ._embed_cache import CachingEmbeddingProvider
from .metrics import find_match_rank, hit_at_k, mrr
from .paraphrase_density import generate_dataset
from evals.io_utils import atomic_write_json


def _run_cell(ds, k: int, vector_weight: float, embedder) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        # Hold all non-vw signals at zero so the only varying factor is
        # bm25 vs vector fusion. This mirrors the ablation's bm25_only
        # variant but with vector_weight as a knob.
        cfg.retrieval.bm25_weight = 1.0
        cfg.retrieval.vector_weight = vector_weight
        cfg.retrieval.salience_weight = 0.0
        cfg.retrieval.recency_weight = 0.0
        cfg.retrieval.context_weight = 0.0
        cfg.retrieval.use_extraction_confidence = False
        eng = (Engram(config=cfg, embeddings=embedder)
               if embedder and vector_weight > 0 else Engram(config=cfg))
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0

            ranks: list[int | None] = []
            lats: list[float] = []
            per_query: list[dict] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=k)
                lat = (time.monotonic() - t0) * 1000
                lats.append(lat)
                rank = find_match_rank(results, q.expected_substrings)
                ranks.append(rank)
                per_query.append({
                    "query": q.text,
                    "tag": q.tags[0] if q.tags else None,
                    "realized_overlap": round(q.realized_overlap, 4),
                    "rank": rank,
                    "hit_at_1": 1 if (rank is not None and rank == 0) else 0,
                    "hit_at_k": 1 if (rank is not None and rank < k) else 0,
                    "reciprocal_rank": (1.0 / (rank + 1)) if rank is not None else 0.0,
                    "latency_ms": round(lat, 3),
                })
            return {
                "vector_weight": vector_weight,
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "k": k,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_k": round(hit_at_k(ranks, k), 4),
                "mrr": round(mrr(ranks), 4),
                "recall_p50_ms": round(statistics.median(lats), 3),
                "ingest_seconds": round(ingest_s, 3),
                "per_query": per_query,
            }
        finally:
            eng.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=120)
    p.add_argument("--distractors", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--overlaps", type=str, default="0.0,0.25,0.5,0.75,1.0",
                   help="comma-separated overlap_target values to sweep")
    p.add_argument("--vector-weight", type=float, default=0.5,
                   help="vw to use for the vector_fusion arm (bm25_only stays 0)")
    p.add_argument("--embed", type=str, default="hash",
                   choices=["none", "hash", "hash512", "st"])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    targets = [float(x) for x in args.overlaps.split(",")]
    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)

    rows = []
    for t in targets:
        ds = generate_dataset(
            n_facts=args.n_facts,
            distractors_per_fact=args.distractors,
            overlap_target=t,
            seed=args.seed,
        )
        print(f"\n=== T={t} (realized={ds.realized_overlap:.3f}) "
              f"n_mem={len(ds.memories)} n_q={len(ds.queries)} ===")
        bm25 = _run_cell(ds, args.k, vector_weight=0.0, embedder=None)
        vec = _run_cell(ds, args.k, vector_weight=args.vector_weight,
                        embedder=embedder)
        delta = round(vec["hit_at_1"] - bm25["hit_at_1"], 4)
        delta_mrr = round(vec["mrr"] - bm25["mrr"], 4)
        print(f"  bm25 hit@1={bm25['hit_at_1']}  vec hit@1={vec['hit_at_1']}  "
              f"Δhit@1={delta:+.4f}  Δmrr={delta_mrr:+.4f}")
        rows.append({
            "overlap_target": t,
            "realized_overlap": round(ds.realized_overlap, 4),
            "bm25_only": bm25,
            "vector_fusion": vec,
            "delta_hit_at_1": delta,
            "delta_mrr": delta_mrr,
        })

    out = {
        "config": {
            "n_facts": args.n_facts,
            "distractors": args.distractors,
            "k": args.k,
            "seed": args.seed,
            "embed": args.embed,
            "vector_weight": args.vector_weight,
            "overlaps": targets,
        },
        "sweep": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"\n[pd_sweep] wrote {args.out}")
    else:
        # Compact summary
        print("\n=== SUMMARY ===")
        print(f"{'T':>6}  {'realized':>9}  {'bm25_h1':>8}  {'vec_h1':>8}  "
              f"{'Δh1':>8}  {'Δmrr':>8}")
        for r in rows:
            print(f"{r['overlap_target']:>6.2f}  {r['realized_overlap']:>9.3f}  "
                  f"{r['bm25_only']['hit_at_1']:>8.4f}  "
                  f"{r['vector_fusion']['hit_at_1']:>8.4f}  "
                  f"{r['delta_hit_at_1']:>+8.4f}  {r['delta_mrr']:>+8.4f}")


if __name__ == "__main__":
    main()

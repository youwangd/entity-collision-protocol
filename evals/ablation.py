"""Ablation harness: run the synthetic eval under different config flags
and report the metric delta. Useful for defending each retrieval mechanism
with empirical numbers (paper ammunition).

Usage:
    python -m evals.ablation --n-sessions 50 --facts 5 --distractors 30 --k 10
    python -m evals.ablation --out bench/results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from engram import Config, Engram
from engram.providers.embeddings import (
    EmbeddingProvider,
    HashTrigramEmbeddingProvider,
)

from ._signals import compute_bm25_top_gap, crowdedness, normalized_gap
from .metrics import find_match_rank, hit_at_k, mrr, ndcg_at_k
from .synthetic import generate_dataset
from evals.io_utils import atomic_write_json


def _make_embedder(name: str | None) -> EmbeddingProvider | None:
    """Factory for the --embed CLI flag. Returns None for 'none' / unset."""
    if not name or name == "none":
        return None
    if name == "hash" or name == "hash256":
        return HashTrigramEmbeddingProvider(dimension=256)
    if name == "hash128":
        return HashTrigramEmbeddingProvider(dimension=128)
    if name == "hash512":
        return HashTrigramEmbeddingProvider(dimension=512)
    if name == "hash1024":
        return HashTrigramEmbeddingProvider(dimension=1024)
    if name == "st":
        from engram.providers.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider()
    if name == "bge_large":
        from engram.providers.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider("BAAI/bge-large-en-v1.5")
    raise ValueError(f"unknown embedder: {name}")


@dataclass
class Variant:
    name: str
    apply: Callable[[Config], None]
    description: str = ""


def _variants() -> list[Variant]:
    def baseline(cfg: Config) -> None:
        pass

    def no_extraction_conf(cfg: Config) -> None:
        cfg.retrieval.use_extraction_confidence = False

    def no_recency(cfg: Config) -> None:
        cfg.retrieval.recency_weight = 0.0

    def no_salience(cfg: Config) -> None:
        cfg.retrieval.salience_weight = 0.0

    def no_context(cfg: Config) -> None:
        cfg.retrieval.context_weight = 0.0

    def bm25_only(cfg: Config) -> None:
        cfg.retrieval.bm25_weight = 1.0
        cfg.retrieval.vector_weight = 0.0
        cfg.retrieval.salience_weight = 0.0
        cfg.retrieval.recency_weight = 0.0
        cfg.retrieval.context_weight = 0.0
        cfg.retrieval.use_extraction_confidence = False

    return [
        Variant("baseline", baseline, "default config (all signals on)"),
        Variant("no_extraction_confidence", no_extraction_conf,
                "ablate per-fact extraction_confidence multiplier"),
        Variant("no_recency", no_recency, "recency_weight = 0"),
        Variant("no_salience", no_salience, "salience_weight = 0"),
        Variant("no_context", no_context, "context_weight = 0"),
        Variant("bm25_only", bm25_only, "naive RAG baseline: pure FTS5"),
    ]


def _run_variant(variant: Variant, ds, k: int,
                 embedder: EmbeddingProvider | None = None,
                 vector_weight: float | None = None,
                 save_per_query: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        variant.apply(cfg)
        # Override vector_weight AFTER variant.apply so the sweep wins,
        # except bm25_only which must stay 0.
        if vector_weight is not None and variant.name != "bm25_only":
            cfg.retrieval.vector_weight = vector_weight
        # bm25_only must stay pure FTS even if a global embedder is passed
        if variant.name == "bm25_only":
            eng = Engram(config=cfg)
        else:
            eng = Engram(config=cfg, embeddings=embedder) if embedder else Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0

            ranks: list[int | None] = []
            recall_lat: list[float] = []
            per_query: list[dict] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=k)
                lat = (time.monotonic() - t0) * 1000
                recall_lat.append(lat)
                rank = find_match_rank(results, q.expected_substrings)
                ranks.append(rank)
                if save_per_query:
                    # Capture FTS top-1/top-2 raw bm25 score gap as a non-leaky
                    # confidence signal for adaptive-vw experiments.
                    bm25_scores = [
                        float(r.sources["bm25"])
                        for r in results
                        if isinstance(r.sources, dict) and "bm25" in r.sources
                    ]
                    bm25_top1, bm25_top2, bm25_gap = compute_bm25_top_gap(bm25_scores)
                    bm25_norm_gap = normalized_gap(bm25_top1, bm25_top2)
                    bm25_crowd_95 = crowdedness(bm25_scores, frac=0.95) if bm25_scores else None
                    bm25_crowd_99 = crowdedness(bm25_scores, frac=0.99) if bm25_scores else None
                    per_query.append({
                        "query": q.text,
                        "rank": rank,
                        "hit_at_1": 1 if (rank is not None and rank == 0) else 0,
                        "hit_at_k": 1 if (rank is not None and rank < k) else 0,
                        "reciprocal_rank": (1.0 / (rank + 1)) if rank is not None else 0.0,
                        "latency_ms": round(lat, 3),
                        "bm25_top1": round(bm25_top1, 4) if bm25_top1 is not None else None,
                        "bm25_top2": round(bm25_top2, 4) if bm25_top2 is not None else None,
                        "bm25_gap": round(bm25_gap, 4) if bm25_gap is not None else None,
                        "bm25_norm_gap": round(bm25_norm_gap, 4) if bm25_norm_gap is not None else None,
                        "bm25_crowd_95": bm25_crowd_95,
                        "bm25_crowd_99": bm25_crowd_99,
                    })

            out = {
                "variant": variant.name,
                "description": variant.description,
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "k": k,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, min(5, k)), 4),
                "hit_at_k": round(hit_at_k(ranks, k), 4),
                "mrr": round(mrr(ranks), 4),
                "ndcg_at_k": round(ndcg_at_k(ranks, k), 4),
                "recall_p50_ms": round(statistics.median(recall_lat), 3),
                "recall_p95_ms": round(
                    statistics.quantiles(recall_lat, n=20)[18]
                    if len(recall_lat) >= 20 else max(recall_lat), 3),
                "ingest_seconds": round(ingest_s, 3),
            }
            if save_per_query:
                out["per_query"] = per_query
            return out
        finally:
            eng.close()


def run_ablation(
    n_sessions: int = 50,
    facts_per_session: int = 5,
    distractors_per_session: int = 30,
    seed: int = 42,
    k: int = 10,
    only: list[str] | None = None,
    paraphrase: bool = False,
    strict_paraphrase: bool = False,
    hard_distractors_per_fact: int = 0,
    embedder: EmbeddingProvider | None = None,
    vector_weight: float | None = None,
    save_per_query: bool = False,
) -> dict:
    ds = generate_dataset(
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        distractors_per_session=distractors_per_session,
        seed=seed,
        paraphrase=paraphrase,
        strict_paraphrase=strict_paraphrase,
        hard_distractors_per_fact=hard_distractors_per_fact,
    )
    print(f"[ablation] dataset: {len(ds.memories)} memories, {len(ds.queries)} queries, k={k}")

    results = []
    for v in _variants():
        if only and v.name not in only:
            continue
        print(f"[ablation] running variant: {v.name}")
        r = _run_variant(v, ds, k, embedder=embedder, vector_weight=vector_weight,
                         save_per_query=save_per_query)
        print(f"  hit@1={r['hit_at_1']}  hit@{k}={r['hit_at_k']}  MRR={r['mrr']}  nDCG={r['ndcg_at_k']}  p50={r['recall_p50_ms']}ms")
        results.append(r)

    # Compute deltas vs. baseline
    baseline = next((r for r in results if r["variant"] == "baseline"), None)
    if baseline:
        for r in results:
            if r["variant"] == "baseline":
                continue
            r["delta_vs_baseline"] = {
                "hit_at_1": round(r["hit_at_1"] - baseline["hit_at_1"], 4),
                "hit_at_k": round(r["hit_at_k"] - baseline["hit_at_k"], 4),
                "mrr": round(r["mrr"] - baseline["mrr"], 4),
                "ndcg_at_k": round(r["ndcg_at_k"] - baseline["ndcg_at_k"], 4),
            }

    return {
        "config": {
            "n_sessions": n_sessions,
            "facts_per_session": facts_per_session,
            "distractors_per_session": distractors_per_session,
            "seed": seed,
            "k": k,
            "paraphrase": paraphrase,
            "strict_paraphrase": strict_paraphrase,
            "hard_distractors_per_fact": hard_distractors_per_fact,
            "vector_weight_override": vector_weight,
        },
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, default=50)
    p.add_argument("--facts", type=int, default=5)
    p.add_argument("--distractors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated variant names to run")
    p.add_argument("--paraphrase", action="store_true",
                   help="use paraphrased queries with low lexical overlap (harder)")
    p.add_argument("--strict-paraphrase", action="store_true",
                   help="use strict-paraphrase queries — zero non-entity token overlap (hardest, lexical fails)")
    p.add_argument("--hard-distractors", type=int, default=0,
                   help="N adversarial distractors per fact (share entity tokens, no answer)")
    p.add_argument("--embed", type=str, default="none",
                   choices=["none", "hash", "hash512", "st"],
                   help="embedding provider: none|hash|hash512|st (sentence-transformers)")
    p.add_argument("--vector-weight", type=float, default=None,
                   help="override cfg.retrieval.vector_weight (skips bm25_only)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    only = args.only.split(",") if args.only else None
    embedder = _make_embedder(args.embed)
    summary = run_ablation(
        n_sessions=args.n_sessions,
        facts_per_session=args.facts,
        distractors_per_session=args.distractors,
        seed=args.seed,
        k=args.k,
        only=only,
        paraphrase=args.paraphrase,
        strict_paraphrase=args.strict_paraphrase,
        hard_distractors_per_fact=args.hard_distractors,
        embedder=embedder,
        vector_weight=args.vector_weight,
    )
    summary["config"]["embed"] = args.embed
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, summary)
        print(f"[ablation] wrote {args.out}")


if __name__ == "__main__":
    main()

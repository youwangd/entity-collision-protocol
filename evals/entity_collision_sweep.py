"""Entity-collision sweep driver.

Sweeps collision degree K against bm25_only vs vector_fusion. The
expected curve under our hypothesis:

    K=1: BM25 hit@1 ≈ 1.0  (entity alone disambiguates), Δ ≈ 0
    K=2: BM25 hit@1 ≈ 0.5  (coin flip within the colliding shell)
    K=4: BM25 hit@1 ≈ 0.25
    K=8: BM25 hit@1 ≈ 0.125
    Vector hit@1: ideally holds up because the discriminator's synonym
    in the query is *semantically* close to the discriminator in the
    memory, even though their surface tokens disagree.

A clear monotonically growing Δhit@1 is the v0.2 phase-transition
result we're looking for.

Usage
-----
    python -m evals.entity_collision_sweep \\
        --embed st --n-entities 8 --distractors 3 \\
        --degrees 1,2,4,8 \\
        --out bench/results/ec_sweep_st.json
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
from .entity_collision import _SPECS, generate_dataset
from .metrics import find_match_rank, hit_at_k, mrr
from evals.io_utils import atomic_write_json


def _run_cell(ds, k: int, vector_weight: float, embedder, *,
              rm3: bool = False, rm3_top_k: int = 10,
              rm3_num_terms: int = 10, rm3_lambda: float = 0.5) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.bm25_weight = 1.0
        cfg.retrieval.vector_weight = vector_weight
        cfg.retrieval.salience_weight = 0.0
        cfg.retrieval.recency_weight = 0.0
        cfg.retrieval.context_weight = 0.0
        cfg.retrieval.use_extraction_confidence = False
        eng = (Engram(config=cfg, embeddings=embedder)
               if embedder and vector_weight > 0 else Engram(config=cfg))

        # Build a content lookup once if RM3 is in use. Memory IDs come
        # from the recall result; map them back to ds.memories text via
        # the inserted-content set.
        _content_pool: dict[str, str] = {}
        if rm3:
            from evals.rm3 import (
                RM3Config,
                build_expanded_query_string,
                expand_query,
            )
            _rm3_cfg = RM3Config(
                top_k=rm3_top_k,
                num_terms=rm3_num_terms,
                lambda_orig=rm3_lambda,
            )

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
                if rm3:
                    first = eng.recall(q.text, limit=_rm3_cfg.top_k)
                    _id_to_text: dict[str, str] = {}
                    first_ids: list[str] = []
                    for r in first:
                        mem = getattr(r, "memory", r)
                        mid = str(getattr(mem, "id", id(mem)))
                        first_ids.append(mid)
                        _id_to_text[mid] = getattr(mem, "content", "") or ""
                    expanded = expand_query(
                        q.text, first_ids, _id_to_text.get, _rm3_cfg
                    )
                    expanded_q = build_expanded_query_string(
                        q.text, expanded, _rm3_cfg
                    )
                    results = eng.recall(expanded_q, limit=k)
                else:
                    results = eng.recall(q.text, limit=k)
                lat = (time.monotonic() - t0) * 1000
                lats.append(lat)
                rank = find_match_rank(results, q.expected_substrings)
                ranks.append(rank)
                per_query.append({
                    "query": q.text,
                    "tag": q.tags[0] if q.tags else None,
                    "collision_degree": q.collision_degree,
                    "rank": rank,
                    "hit_at_1": 1 if (rank is not None and rank == 0) else 0,
                    "hit_at_k": 1 if (rank is not None and rank < k) else 0,
                    "reciprocal_rank": (1.0 / (rank + 1)) if rank is not None else 0.0,
                    "latency_ms": round(lat, 3),
                })
            return {
                "vector_weight": vector_weight,
                "rm3": rm3,
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
    p.add_argument("--n-entities", type=int, default=8)
    p.add_argument("--distractors", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--degrees", type=str, default="1,2,4,8",
                   help="comma-separated collision degrees to sweep")
    p.add_argument("--tag", type=str, default="preference",
                   choices=list(_SPECS.keys()))
    p.add_argument("--vector-weight", type=float, default=0.5)
    p.add_argument("--paraphrase-memory", action="store_true",
                   help="sample memory templates from spec.memory_variants "
                        "(addresses §6.1 fixed-template threat)")
    p.add_argument("--embed", type=str, default="hash",
                   choices=["none", "hash", "hash128", "hash256", "hash512", "hash1024", "st", "bge_large"])
    p.add_argument("--rm3", action="store_true",
                   help="AUDIT-D: also run an RM3 PRF cell alongside bm25_only and vector_fusion.")
    p.add_argument("--rm3-top-k", type=int, default=10)
    p.add_argument("--rm3-num-terms", type=int, default=10)
    p.add_argument("--rm3-lambda", type=float, default=0.5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    degrees = [int(x) for x in args.degrees.split(",")]
    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)

    rows = []
    for K in degrees:
        ds = generate_dataset(
            n_entities=args.n_entities,
            collision_degree=K,
            distractors_per_entity=args.distractors,
            seed=args.seed,
            tag=args.tag,
            paraphrase_memory=args.paraphrase_memory,
        )
        print(f"\n=== K={K}  n_mem={len(ds.memories)} n_q={len(ds.queries)} ===")
        bm25 = _run_cell(ds, args.k, vector_weight=0.0, embedder=None)
        vec = _run_cell(ds, args.k, vector_weight=args.vector_weight,
                        embedder=embedder)
        delta = round(vec["hit_at_1"] - bm25["hit_at_1"], 4)
        delta_mrr = round(vec["mrr"] - bm25["mrr"], 4)
        # Theoretical BM25 floor inside the colliding shell: 1/K
        floor = round(1.0 / K, 4)
        rm3_cell = None
        delta_rm3 = None
        delta_rm3_mrr = None
        if args.rm3:
            rm3_cell = _run_cell(
                ds, args.k, vector_weight=0.0, embedder=None,
                rm3=True, rm3_top_k=args.rm3_top_k,
                rm3_num_terms=args.rm3_num_terms, rm3_lambda=args.rm3_lambda,
            )
            delta_rm3 = round(rm3_cell["hit_at_1"] - bm25["hit_at_1"], 4)
            delta_rm3_mrr = round(rm3_cell["mrr"] - bm25["mrr"], 4)
        print(f"  bm25 hit@1={bm25['hit_at_1']} (1/K floor={floor})  "
              f"vec hit@1={vec['hit_at_1']}  Δhit@1={delta:+.4f}  "
              f"Δmrr={delta_mrr:+.4f}"
              + (f"  rm3 hit@1={rm3_cell['hit_at_1']}  Δrm3={delta_rm3:+.4f}"
                 if rm3_cell else ""))
        row = {
            "collision_degree": K,
            "bm25_floor": floor,
            "bm25_only": bm25,
            "vector_fusion": vec,
            "delta_hit_at_1": delta,
            "delta_mrr": delta_mrr,
        }
        if rm3_cell is not None:
            row["rm3_fusion"] = rm3_cell
            row["delta_hit_at_1_rm3"] = delta_rm3
            row["delta_mrr_rm3"] = delta_rm3_mrr
        rows.append(row)

    out = {
        "config": {
            "n_entities": args.n_entities,
            "distractors": args.distractors,
            "k": args.k,
            "seed": args.seed,
            "embed": args.embed,
            "tag": args.tag,
            "vector_weight": args.vector_weight,
            "degrees": degrees,
            "paraphrase_memory": args.paraphrase_memory,
            "rm3": args.rm3,
            **({"rm3_config": {
                "top_k": args.rm3_top_k,
                "num_terms": args.rm3_num_terms,
                "lambda": args.rm3_lambda,
            }} if args.rm3 else {}),
        },
        "sweep": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"\n[ec_sweep] wrote {args.out}")
    else:
        print("\n=== SUMMARY ===")
        print(f"{'K':>3}  {'1/K':>6}  {'bm25_h1':>8}  {'vec_h1':>8}  "
              f"{'Δh1':>8}  {'Δmrr':>8}")
        for r in rows:
            print(f"{r['collision_degree']:>3}  {r['bm25_floor']:>6.3f}  "
                  f"{r['bm25_only']['hit_at_1']:>8.4f}  "
                  f"{r['vector_fusion']['hit_at_1']:>8.4f}  "
                  f"{r['delta_hit_at_1']:>+8.4f}  {r['delta_mrr']:>+8.4f}")


if __name__ == "__main__":
    main()

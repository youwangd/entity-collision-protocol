"""§4.13d — synth_entity × typed-PRF × share_prior stack interaction.

NEXT.md priority #1. Defends decision-#2 (wire PRF + share_prior into
RetrievalEngine.search) on the type-paired collider fixture, and tests
whether the type-aware PRF gate (`query_expansion_type_purity_min`) bites
on the discriminator-paraphrase channel where untyped PRF does not.

Arms (all at pivot ``vector_weight=0.3``, ``bm25_weight=1.0``):

    C0_baseline    : no PRF, no reranker
    CP_prf_heur    : untyped PRF (heuristic NER) — anchor 18 default
    CP_prf_typed   : typed PRF (spaCy + purity gate)
    CR_share_prior : share_prior reranker only (alpha=0.05, pool=20)
    CB_both_heur   : untyped PRF + share_prior
    CB_both_typed  : typed PRF + share_prior

Per-query bookkeeping: hit@1, hit@5, MRR, latency. Paired-bootstrap CIs
vs C0_baseline (5000 resamples, α=0.05).

Output: ``evals/results/synth_entity_typed_prf_stack.json`` + console MD.
Wall budget: ~3 min for the default 6-arm grid on a 320-query corpus.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from engram import Config, Engram

from evals.ablation import _make_embedder
from evals._embed_cache import CachingEmbeddingProvider
from evals.bootstrap_ci import _paired_diff_ci
from evals.entity_channel_sweep import _build_synth_entity_dataset
from evals.metrics import find_match_rank, hit_at_k, mrr
from evals.io_utils import atomic_write_json, atomic_write_text


ARMS: list[dict] = [
    {"name": "C0_baseline",    "prf": None,      "reranker": None},
    {"name": "CP_prf_heur",    "prf": "heur",    "reranker": None},
    {"name": "CP_prf_typed",   "prf": "typed",   "reranker": None},
    {"name": "CR_share_prior", "prf": None,      "reranker": "share_prior"},
    {"name": "CB_both_heur",   "prf": "heur",    "reranker": "share_prior"},
    {"name": "CB_both_typed",  "prf": "typed",   "reranker": "share_prior"},
]


def _build_cfg(arm: dict, *, vector_weight: float, qe_dominance: float,
               type_purity_min: float, share_prior_alpha: float,
               path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.bm25_weight = 1.0
    cfg.retrieval.vector_weight = float(vector_weight)
    cfg.retrieval.salience_weight = 0.0
    cfg.retrieval.recency_weight = 0.0
    cfg.retrieval.context_weight = 0.0
    cfg.retrieval.use_extraction_confidence = False
    if arm["prf"] is not None:
        cfg.retrieval.query_expansion_min_dominance = float(qe_dominance)
        if arm["prf"] == "typed":
            cfg.retrieval.entity_ner = "spacy_sm"
            cfg.retrieval.query_expansion_type_purity_min = float(type_purity_min)
        else:
            cfg.retrieval.entity_ner = "heuristic"
    if arm["reranker"] == "share_prior":
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.rerank_pool_size = 20
        cfg.retrieval.share_prior_alpha = float(share_prior_alpha)
    return cfg


def _run_arm(arm: dict, ds, *, k: int, embedder, **knobs) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(arm, path=tmp, **knobs)
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if (embedder is not None and cfg.retrieval.vector_weight > 0)
            else Engram(config=cfg)
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
                results = eng.recall(q.text, limit=k)
                lat = (time.monotonic() - t0) * 1000
                lats.append(lat)
                rank = find_match_rank(results, q.expected_substrings)
                ranks.append(rank)
                hit1 = 1 if (rank is not None and rank == 0) else 0
                hit5 = 1 if (rank is not None and rank < 5) else 0
                rr = (1.0 / (rank + 1)) if rank is not None else 0.0
                per_query.append({
                    "query": q.text,
                    "tag": q.tags[0] if q.tags else None,
                    "rank": rank,
                    "hit_at_1": hit1,
                    "hit_at_5": hit5,
                    "reciprocal_rank": rr,
                    "latency_ms": round(lat, 3),
                })
            return {
                "arm": arm["name"],
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "k": k,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, 5), 4),
                "mrr": round(mrr(ranks), 4),
                "p50_ms": round(statistics.median(lats), 3),
                "p95_ms": round(sorted(lats)[int(0.95 * (len(lats) - 1))], 3),
                "ingest_s": round(ingest_s, 2),
                "per_query": per_query,
            }
        finally:
            eng.close()


def _paired_block(cells: list[dict], pivot_arm: str, *,
                  resamples: int, seed: int) -> dict:
    pivot = next((c for c in cells if c["arm"] == pivot_arm), None)
    if pivot is None:
        return {}
    pv = pivot["per_query"]
    block: dict = {"pivot_arm": pivot_arm, "n_queries": len(pv), "deltas": {}}
    for c in cells:
        if c["arm"] == pivot_arm:
            continue
        cv = c["per_query"]
        if len(cv) != len(pv):
            continue
        entry: dict = {}
        for metric in ("hit_at_1", "hit_at_5", "mrr"):
            key = "reciprocal_rank" if metric == "mrr" else metric
            a = [r[key] for r in cv]
            b = [r[key] for r in pv]
            m, lo, hi = _paired_diff_ci(a, b, resamples, seed)
            entry[metric] = {
                "mean_delta": round(m, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "arm_mean": round(sum(a) / max(len(a), 1), 4),
                "pivot_mean": round(sum(b) / max(len(b), 1), 4),
                "significant": (lo > 0) or (hi < 0),
            }
        block["deltas"][c["arm"]] = entry
    return block


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--distractors-per-entity", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--vector-weight", type=float, default=0.3)
    p.add_argument("--qe-dominance", type=float, default=0.3)
    p.add_argument("--type-purity-min", type=float, default=0.7)
    p.add_argument("--share-prior-alpha", type=float, default=0.05)
    p.add_argument("--embed", type=str, default="hash",
                   choices=["hash", "hash128", "hash256", "hash512", "hash1024", "st"])
    p.add_argument("--pivot-arm", type=str, default="C0_baseline")
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--out", type=str,
                   default="evals/results/synth_entity_typed_prf_stack.json")
    p.add_argument("--update-report", action="store_true")
    p.add_argument("--arms", type=str, default=None,
                   help="Comma-sep arm-name filter (default: all 6).")
    args = p.parse_args()

    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)

    ds, fixture_meta = _build_synth_entity_dataset(
        n_entities=args.n_entities,
        collision_degree=args.K,
        distractors_per_entity=args.distractors_per_entity,
        seed=args.seed,
    )
    print(f"[se-tprf] fixture: {fixture_meta['n_memories']} mems, "
          f"{fixture_meta['n_queries']} queries "
          f"(n_entities={args.n_entities}, K={args.K}, "
          f"tags={','.join(fixture_meta['tags'])})")

    selected = ARMS
    if args.arms:
        names = {n.strip() for n in args.arms.split(",") if n.strip()}
        selected = [a for a in ARMS if a["name"] in names]

    knobs = dict(
        vector_weight=args.vector_weight,
        qe_dominance=args.qe_dominance,
        type_purity_min=args.type_purity_min,
        share_prior_alpha=args.share_prior_alpha,
    )

    t_all = time.monotonic()
    cells: list[dict] = []
    for arm in selected:
        cell = _run_arm(arm, ds, k=args.k, embedder=embedder, **knobs)
        cells.append(cell)
        print(f"  {arm['name']:<18}  hit@1={cell['hit_at_1']:.3f} "
              f"hit@5={cell['hit_at_5']:.3f} MRR={cell['mrr']:.3f} "
              f"p50={cell['p50_ms']}ms p95={cell['p95_ms']}ms "
              f"(ingest={cell['ingest_s']}s)")
    wall = round(time.monotonic() - t_all, 2)

    paired = _paired_block(cells, args.pivot_arm, resamples=args.resamples,
                           seed=args.seed)

    out = {
        "config": {
            "fixture": fixture_meta,
            "vector_weight": args.vector_weight,
            "qe_dominance": args.qe_dominance,
            "type_purity_min": args.type_purity_min,
            "share_prior_alpha": args.share_prior_alpha,
            "embed": args.embed,
            "k": args.k,
            "pivot_arm": args.pivot_arm,
            "resamples": args.resamples,
            "alpha": 0.05,
            "seed": args.seed,
        },
        "cells": [{k: v for k, v in c.items() if k != "per_query"} for c in cells],
        "per_query_by_arm": {c["arm"]: c["per_query"] for c in cells},
        "paired_bootstrap_ci": paired,
        "wall_seconds": wall,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(f"[se-tprf] wrote {args.out}  (wall={wall}s)")

    md = [
        f"\n## §4.13d — synth_entity typed-PRF × share_prior stack "
        f"(n_entities={args.n_entities}, K={args.K}, vw={args.vector_weight}, "
        f"embed={args.embed}, pivot={args.pivot_arm})\n",
        "| arm | hit@1 | hit@5 | MRR | p50 ms | p95 ms |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        md.append(
            f"| {c['arm']} | {c['hit_at_1']:.3f} | {c['hit_at_5']:.3f} | "
            f"{c['mrr']:.3f} | {c['p50_ms']:.2f} | {c['p95_ms']:.2f} |"
        )
    if paired and paired.get("deltas"):
        md.append(f"\nPaired bootstrap Δ vs **{args.pivot_arm}** "
                  f"({args.resamples} resamples, α=0.05):\n")
        md.append("| arm | metric | Δ mean | 95% CI | sig? |")
        md.append("|:---|:---|---:|:---:|:---:|")
        for arm_name, e in paired["deltas"].items():
            for metric in ("hit_at_1", "hit_at_5", "mrr"):
                d = e[metric]
                md.append(
                    f"| {arm_name} | {metric} | {d['mean_delta']:+.4f} | "
                    f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | "
                    f"{'**yes**' if d['significant'] else 'no'} |"
                )
    md_text = "\n".join(md) + "\n"
    print(md_text)

    if args.update_report:
        report = Path("ENTITY_CHANNEL_REPORT.md")
        if report.exists():
            atomic_write_text(report, report.read_text() + md_text)
        else:
            atomic_write_text(report,
                "# Entity-Link Channel — Δrecall@k Report\n"
                "\nDriver: `evals/synth_entity_typed_prf_stack.py`\n" + md_text
            )
        print(f"[se-tprf] appended to {report}")


if __name__ == "__main__":
    main()

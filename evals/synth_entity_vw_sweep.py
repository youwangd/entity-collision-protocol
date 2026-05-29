"""§4.13c — synth_entity vector_weight Pareto sweep.

Defends decision-#3 (default `RetrievalConfig.vector_weight = 0.3`) on the
type-paired collider fixture. The legacy D1 corpus is hit@5-saturated and
cannot move under entity- or vector-channel knobs, so the Pareto basis for
the default flip needs the redesigned synth_entity fixture.

Design
------
* Fixture: `_build_synth_entity_dataset(n_entities=16, K=4,
  distractors_per_entity=3)` from `evals.entity_channel_sweep` — 5 tag
  families, 320 colliding queries, ~560 mems by default.
* Embedder: `HashTrigramEmbeddingProvider(dim=256)` (no torch on the box,
  reproducible char-trigram features that exercise vector fusion without
  a ST checkpoint pull).
* Arms: `vector_weight ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7}` — vw=0.0 is
  the BM25-only baseline, vw=0.3 is the new default we are defending,
  vw=0.5 is the v0.1 default.
* Per-query bookkeeping: hit@1, hit@5, MRR, latency_ms.
* CI: paired bootstrap on Δ-vs-vw=0.3 (5000 resamples, α=0.05).

Outputs
-------
* `evals/results/synth_entity_vw_sweep.json` — full grid + per_query
  vectors + paired-CI block.
* Markdown summary on stdout (and ENTITY_CHANNEL_REPORT.md if requested).

Wall budget: ~30 s for the default grid on a 320-query corpus.
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


def _run_cell(ds, *, k: int, vector_weight: float, embedder) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.bm25_weight = 1.0
        cfg.retrieval.vector_weight = float(vector_weight)
        cfg.retrieval.salience_weight = 0.0
        cfg.retrieval.recency_weight = 0.0
        cfg.retrieval.context_weight = 0.0
        cfg.retrieval.use_extraction_confidence = False
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if (embedder is not None and vector_weight > 0)
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
                hitk = 1 if (rank is not None and rank < k) else 0
                rr = (1.0 / (rank + 1)) if rank is not None else 0.0
                per_query.append({
                    "query": q.text,
                    "tag": q.tags[0] if q.tags else None,
                    "rank": rank,
                    "hit_at_1": hit1,
                    "hit_at_5": hit5,
                    "hit_at_k": hitk,
                    "reciprocal_rank": rr,
                    "latency_ms": round(lat, 3),
                })
            return {
                "vector_weight": float(vector_weight),
                "n_memories": len(ds.memories),
                "n_queries": len(ds.queries),
                "k": k,
                "hit_at_1": round(hit_at_k(ranks, 1), 4),
                "hit_at_5": round(hit_at_k(ranks, 5), 4),
                "hit_at_k": round(hit_at_k(ranks, k), 4),
                "mrr": round(mrr(ranks), 4),
                "p50_ms": round(statistics.median(lats), 3),
                "p95_ms": round(sorted(lats)[int(0.95 * (len(lats) - 1))], 3),
                "ingest_s": round(ingest_s, 2),
                "per_query": per_query,
            }
        finally:
            eng.close()


def _paired_block(cells: list[dict], pivot_vw: float, *, resamples: int,
                  seed: int) -> dict:
    """Paired-bootstrap Δ vs the pivot cell (e.g. vw=0.3 default)."""
    pivot = next((c for c in cells if abs(c["vector_weight"] - pivot_vw) < 1e-9), None)
    if pivot is None:
        return {}
    pv = pivot["per_query"]
    block: dict = {"pivot_vw": pivot_vw, "n_queries": len(pv), "deltas": {}}
    for c in cells:
        if c is pivot:
            continue
        cv = c["per_query"]
        if len(cv) != len(pv):
            continue
        entry: dict = {}
        for metric in ("hit_at_1", "hit_at_5", "mrr"):
            a = [r[metric] if metric != "mrr" else r["reciprocal_rank"] for r in cv]
            b = [r[metric] if metric != "mrr" else r["reciprocal_rank"] for r in pv]
            m, lo, hi = _paired_diff_ci(a, b, resamples, seed)
            entry[metric] = {
                "mean_delta": round(m, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "arm_mean": round(sum(a) / max(len(a), 1), 4),
                "pivot_mean": round(sum(b) / max(len(b), 1), 4),
                "significant": (lo > 0) or (hi < 0),
            }
        block["deltas"][f"{c['vector_weight']:.2f}"] = entry
    return block


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str,
                   default="0.0,0.1,0.2,0.3,0.4,0.5,0.7",
                   help="Comma-separated vector_weight values to sweep.")
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--K", type=int, default=4,
                   help="collision_degree per entity (synth_entity)")
    p.add_argument("--distractors-per-entity", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed", type=str, default="hash",
                   choices=["hash", "hash128", "hash256", "hash512", "hash1024", "st"])
    p.add_argument("--pivot-vw", type=float, default=0.3,
                   help="Paired-CI pivot (default = 0.3, the new default).")
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--out", type=str,
                   default="evals/results/synth_entity_vw_sweep.json")
    p.add_argument("--update-report", action="store_true")
    args = p.parse_args()

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)

    ds, fixture_meta = _build_synth_entity_dataset(
        n_entities=args.n_entities,
        collision_degree=args.K,
        distractors_per_entity=args.distractors_per_entity,
        seed=args.seed,
    )
    print(f"[se-vw] fixture: {fixture_meta['n_memories']} mems, "
          f"{fixture_meta['n_queries']} queries "
          f"(n_entities={args.n_entities}, K={args.K}, "
          f"tags={','.join(fixture_meta['tags'])})")

    t_all = time.monotonic()
    cells: list[dict] = []
    for vw in weights:
        cell = _run_cell(ds, k=args.k, vector_weight=vw, embedder=embedder)
        cells.append(cell)
        print(f"  vw={vw:.2f}  hit@1={cell['hit_at_1']:.3f} "
              f"hit@5={cell['hit_at_5']:.3f} MRR={cell['mrr']:.3f} "
              f"p50={cell['p50_ms']}ms p95={cell['p95_ms']}ms "
              f"(ingest={cell['ingest_s']}s)")
    wall = round(time.monotonic() - t_all, 2)

    paired = _paired_block(cells, args.pivot_vw, resamples=args.resamples,
                           seed=args.seed)

    out = {
        "config": {
            "fixture": fixture_meta,
            "weights": weights,
            "embed": args.embed,
            "k": args.k,
            "pivot_vw": args.pivot_vw,
            "resamples": args.resamples,
            "alpha": 0.05,
            "seed": args.seed,
        },
        "cells": [{k: v for k, v in c.items() if k != "per_query"} for c in cells],
        "per_query_by_vw": {f"{c['vector_weight']:.2f}": c["per_query"] for c in cells},
        "paired_bootstrap_ci": paired,
        "wall_seconds": wall,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(f"[se-vw] wrote {args.out}  (wall={wall}s)")

    # Markdown summary
    md_lines = [
        f"\n## §4.13c — synth_entity vector_weight sweep "
        f"(n_entities={args.n_entities}, K={args.K}, embed={args.embed}, "
        f"pivot vw={args.pivot_vw})\n",
        "| vw | hit@1 | hit@5 | MRR | p50 ms | p95 ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        md_lines.append(
            f"| {c['vector_weight']:.2f} | {c['hit_at_1']:.3f} | "
            f"{c['hit_at_5']:.3f} | {c['mrr']:.3f} | "
            f"{c['p50_ms']:.2f} | {c['p95_ms']:.2f} |"
        )
    if paired and paired.get("deltas"):
        md_lines.append("\nPaired bootstrap Δ vs pivot "
                        f"(vw={args.pivot_vw}), {args.resamples} resamples, "
                        "α=0.05:\n")
        md_lines.append("| arm_vw | metric | Δ mean | 95% CI | sig? |")
        md_lines.append("|---:|:---|---:|:---:|:---:|")
        for arm_vw, e in sorted(paired["deltas"].items(), key=lambda kv: float(kv[0])):
            for metric in ("hit_at_1", "hit_at_5", "mrr"):
                d = e[metric]
                md_lines.append(
                    f"| {arm_vw} | {metric} | {d['mean_delta']:+.4f} | "
                    f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | "
                    f"{'**yes**' if d['significant'] else 'no'} |"
                )
    md = "\n".join(md_lines) + "\n"
    print(md)

    if args.update_report:
        report = Path("ENTITY_CHANNEL_REPORT.md")
        if report.exists():
            atomic_write_text(report, report.read_text() + md)
        else:
            atomic_write_text(report,
                "# Entity-Link Channel — Δrecall@k Report\n"
                "\nDriver: `evals/synth_entity_vw_sweep.py`\n" + md
            )
        print(f"[se-vw] appended to {report}")


if __name__ == "__main__":
    main()

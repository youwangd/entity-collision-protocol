"""Sweep cfg.retrieval.vector_weight ∈ {0.1, 0.3, 0.5, 0.7, 1.0} on a fixed
dataset configuration with a chosen embedder, and report the baseline-variant
hit@1 / MRR / nDCG curve.

Goal: find the Pareto-optimal vector fusion weight for hash-trigram embeddings
on (a) the easy synthetic bench and (b) the paraphrase+hard-distractors bench.

Usage:
    python -m evals.sweep_vector_weight --embed hash \\
        --n-sessions 50 --facts 5 --distractors 30 --k 10 \\
        --out bench/results/sweep_vw_hash_easy.json

    python -m evals.sweep_vector_weight --embed hash --paraphrase \\
        --hard-distractors 5 --out bench/results/sweep_vw_hash_hard.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import _make_embedder, run_ablation
from ._embed_cache import CachingEmbeddingProvider
from evals.io_utils import atomic_write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, default=50)
    p.add_argument("--facts", type=int, default=5)
    p.add_argument("--distractors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--paraphrase", action="store_true")
    p.add_argument("--strict-paraphrase", action="store_true",
                   help="use strict-paraphrase queries (zero non-entity token overlap)")
    p.add_argument("--hard-distractors", type=int, default=0)
    p.add_argument("--embed", type=str, default="hash",
                   choices=["none", "hash", "hash512", "st"])
    p.add_argument("--weights", type=str, default="0.0,0.1,0.3,0.5,0.7,1.0",
                   help="comma-separated vector_weight values to sweep")
    # Only run baseline + bm25_only — the sweep is about how baseline moves
    # vs. the pure-FTS reference as vw varies.
    p.add_argument("--only", type=str, default="baseline,bm25_only")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-per-query", action="store_true",
                   help="capture per-query rank/hit/RR for bootstrap CI computation")
    p.add_argument("--no-embed-cache", action="store_true",
                   help="disable embedding memoization across sweep cells (debug)")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="disable per-cell partial checkpoints (debug)")
    p.add_argument("--resume", action="store_true",
                   help="reuse existing per-cell partial files instead of re-running them")
    args = p.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    only = args.only.split(",") if args.only else None
    embedder = _make_embedder(args.embed)
    # Wrap in a caching layer so repeated cells don't re-encode the same
    # dataset. ST CPU encode is the dominant cost at n>=100; this gives a
    # roughly N-fold speedup for an N-weight sweep (one cold pass + N-1 hits).
    if embedder is not None and not args.no_embed_cache:
        embedder = CachingEmbeddingProvider(embedder)

    # Per-cell partial checkpoint plumbing. If --out is given and checkpointing
    # is enabled, each cell is written to <out>.cell-<i>.partial.json before
    # we move on. A run that gets killed mid-sweep can therefore be resumed
    # with --resume (or its data harvested manually).
    checkpoint_dir = None
    checkpoint_stem = None
    if args.out and not args.no_checkpoint:
        out_path = Path(args.out)
        checkpoint_dir = out_path.parent
        checkpoint_stem = out_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _cell_path(i: int) -> Path | None:
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / f"{checkpoint_stem}.cell-{i}.partial.json"

    sweep = []
    for i, vw in enumerate(weights):
        print(f"\n=== vector_weight = {vw} (cell {i+1}/{len(weights)}) ===")
        cp = _cell_path(i)
        if args.resume and cp is not None and cp.exists():
            try:
                cached = json.loads(cp.read_text())
                if cached.get("vector_weight") == vw:
                    print(f"  [resume] using {cp.name}")
                    sweep.append(cached)
                    continue
                else:
                    print(f"  [resume] {cp.name} vw mismatch ({cached.get('vector_weight')} != {vw}); re-running")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [resume] {cp.name} unreadable ({e}); re-running")
        # Re-instantiate embedder per run? No — provider is stateless for hash.
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
            vector_weight=vw,
            save_per_query=args.save_per_query,
        )
        baseline = next(r for r in summary["results"] if r["variant"] == "baseline")
        bm25 = next((r for r in summary["results"] if r["variant"] == "bm25_only"), None)
        row = {
            "vector_weight": vw,
            "baseline_hit_at_1": baseline["hit_at_1"],
            "baseline_hit_at_k": baseline["hit_at_k"],
            "baseline_mrr": baseline["mrr"],
            "baseline_ndcg_at_k": baseline["ndcg_at_k"],
            "bm25_only_hit_at_1": bm25["hit_at_1"] if bm25 else None,
            "bm25_only_mrr": bm25["mrr"] if bm25 else None,
            "delta_hit_at_1_vs_bm25": (
                round(baseline["hit_at_1"] - bm25["hit_at_1"], 4) if bm25 else None
            ),
            "delta_mrr_vs_bm25": (
                round(baseline["mrr"] - bm25["mrr"], 4) if bm25 else None
            ),
        }
        if args.save_per_query:
            row["baseline_per_query"] = baseline.get("per_query")
            if bm25:
                row["bm25_only_per_query"] = bm25.get("per_query")
        sweep.append(row)
        # Write per-cell partial AFTER computing the row so a kill any time
        # before this point loses only the in-flight cell, not finished ones.
        if cp is not None:
            tmp = cp.with_suffix(cp.suffix + ".tmp")
            atomic_write_json(tmp, row)
            tmp.replace(cp)  # atomic on POSIX
            print(f"  [checkpoint] wrote {cp.name}")
        print(f"  baseline: hit@1={row['baseline_hit_at_1']} MRR={row['baseline_mrr']} "
              f"Δhit@1_vs_bm25={row['delta_hit_at_1_vs_bm25']}")

    out = {
        "config": {
            "embed": args.embed,
            "n_sessions": args.n_sessions,
            "facts": args.facts,
            "distractors": args.distractors,
            "paraphrase": args.paraphrase,
            "strict_paraphrase": args.strict_paraphrase,
            "hard_distractors": args.hard_distractors,
            "k": args.k,
            "seed": args.seed,
        },
        "sweep": sweep,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"[sweep] wrote {args.out}")
    # Print cache stats so we can verify the speedup is real.
    if isinstance(embedder, CachingEmbeddingProvider):
        s = embedder.stats
        print(f"[sweep] embed cache: hits={s['hits']} misses={s['misses']} "
              f"size={s['size']} hit_rate={s['hit_rate']:.3f}")


if __name__ == "__main__":
    main()

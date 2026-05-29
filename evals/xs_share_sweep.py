"""§94b — schema_family_share sweep with synthesis on.

§94 measured a single recipe point (share=0.75) and found:

  * session_hit@1     +1.7pp
  * pair_recall@k    -25.0pp   ← winner-take-all displacement
  * MRR               +1.1pp

This driver sweeps ``schema_family_share`` across {0.0, 0.25, 0.5, 0.75, 1.0}
holding everything else at the §87 recipe + synthesis on, and asks the
Pareto question:

  Is there *any* share value that retains the +h@1 gain without the
  pair_recall collapse?

If yes  → that's the operational recommendation, paper-defensibly.
If no   → §87's family gate is structurally winner-take-all on
          evidence-spanning corpora; that is itself a strong claim.

We run a single shared baseline (synthesis on, family gate off) and
each share point as its own treatment arm, monkey-patching the §94
driver's ``RECIPE["schema_family_share"]`` for the duration of the
arm. This keeps the gate / contamination / fragmentation / tau values
identical across arms; only ``share`` varies.

Pure: deterministic given seed. No clocks, no I/O beyond optional --out.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager

from evals import cross_session_recall_lift as xs
from evals.synthetic import generate_cross_session_dataset
from evals.io_utils import atomic_write_json


@contextmanager
def _patched_share(share: float):
    prev = xs.RECIPE["schema_family_share"]
    xs.RECIPE["schema_family_share"] = share
    try:
        yield
    finally:
        xs.RECIPE["schema_family_share"] = prev


def _mean(xs_):
    return round(sum(xs_) / len(xs_), 4) if xs_ else 0.0


def _summarise(rows: list[dict]) -> dict:
    rows = [r for r in rows if "_consolidation_error" not in r]
    return {
        "session_hit_at_1": _mean([r["hit_at_1"] for r in rows]),
        "session_hit_at_k": _mean([r["hit_at_k"] for r in rows]),
        "pair_recall_at_k": _mean([r["pair_recall_at_k"] for r in rows]),
        "mean_reciprocal_rank": _mean([r["reciprocal_rank"] for r in rows]),
        "n_pairs": len(rows),
    }


def run_share_sweep(
    *,
    shares: list[float],
    n_facts: int = 60,
    n_sessions: int = 10,
    distractors_per_session: int = 10,
    seed: int = 42,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
) -> dict:
    if embedder_name == "hashtrigram":
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        embedder = HashTrigramEmbeddingProvider(dimension=256)
        emb_label = "HashTrigram-256"
    elif embedder_name in ("st", "minilm", "sentence_transformer"):
        from engram.providers.embeddings import SentenceTransformerProvider
        embedder = SentenceTransformerProvider()
        emb_label = "SentenceTransformer-MiniLM-384"
    elif embedder_name is None:
        embedder = None
        emb_label = "none(BM25-only)"
    else:
        raise ValueError(f"unknown embedder: {embedder_name!r}")

    ds = generate_cross_session_dataset(
        n_facts=n_facts,
        n_sessions=n_sessions,
        distractors_per_session=distractors_per_session,
        seed=seed,
    )

    t0 = time.monotonic()
    # Baseline: synthesis on, family gate disabled (treatment=False).
    baseline_rows = xs._run_arm(
        ds, treatment=False, embedder=embedder, k=k, synthesis=True
    )
    baseline = _summarise(baseline_rows)

    points = []
    for share in shares:
        t_arm = time.monotonic()
        with _patched_share(share):
            rows = xs._run_arm(
                ds, treatment=True, embedder=embedder, k=k, synthesis=True
            )
        treat = _summarise(rows)
        delta = {
            key: round(treat[key] - baseline[key], 4)
            for key in baseline if key != "n_pairs"
        }
        points.append({
            "share": share,
            "treatment": treat,
            "delta": delta,
            "wall_seconds": round(time.monotonic() - t_arm, 2),
        })
    wall = time.monotonic() - t0

    pareto = [
        p for p in points
        if p["delta"]["session_hit_at_1"] > 0
        and p["delta"]["pair_recall_at_k"] >= 0
    ]

    return {
        "synthesis": True,
        "embedder": emb_label,
        "corpus": {
            "n_facts": n_facts,
            "n_sessions": n_sessions,
            "distractors_per_session": distractors_per_session,
            "seed": seed,
            "n_memories": len(ds.memories),
            "n_queries": len(ds.queries),
        },
        "k": k,
        "shares": shares,
        "baseline": baseline,
        "points": points,
        "pareto_winners": pareto,
        "wall_seconds": round(wall, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shares", type=str, default="0.0,0.25,0.5,0.75,1.0",
        help="comma-separated list of schema_family_share values",
    )
    p.add_argument("--n-facts", type=int, default=60)
    p.add_argument("--n-sessions", type=int, default=10)
    p.add_argument("--distractors", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--embedder", default="hashtrigram",
        choices=["hashtrigram", "st", "minilm", "sentence_transformer"],
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    shares = [float(s) for s in args.shares.split(",")]
    out = run_share_sweep(
        shares=shares,
        n_facts=args.n_facts,
        n_sessions=args.n_sessions,
        distractors_per_session=args.distractors,
        seed=args.seed,
        k=args.k,
        embedder_name=args.embedder,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, out, default=str)
        print(f"[xs_share_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()

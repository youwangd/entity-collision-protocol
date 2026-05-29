"""§5.4 open-angle empirical: constant-alpha vs max_deg-adaptive alpha.

Reuses the bridge multi-hop corpus from `share_prior_sweep` and adds a
parallel arm that flips `cfg.share_prior_adaptive_alpha = True`. Reports
Δpair@10 of adaptive-alpha vs constant-alpha across alpha ∈ {0.05, 0.10,
0.20, 0.40} so we can see whether tapering by max_deg costs us anything
on the bridge recipe (where we already have a positive signal) and
whether it helps when alpha is pushed past its sweet spot.

Hypothesis (paper §3.5 / §A.4.7.6, formerly PAPER_NOTES §5.4 open-angle 2):
    Adaptive alpha should match constant alpha when alpha is small (the
    rank-0 cap dominates), and should *protect* recall at large alpha by
    shrinking the boost when the entity-sharing graph is dense.

Run:
    python -m evals.share_prior_adaptive_alpha
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import tempfile
from pathlib import Path
from evals.io_utils import atomic_write_json

# Reuse the corpus + helpers from the existing sweep.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.share_prior_sweep import (  # noqa: E402
    _eval_bridge,
    generate_bridge_corpus as _make_bridge_corpus,
)
from engram import Engram  # noqa: E402
from engram.core.config import Config, RetrievalConfig  # noqa: E402


def _eval_bridge_adaptive(ds, *, alpha, pool_size, adaptive: bool, k_max=10):
    """Adaptive variant of `_eval_bridge` that flips the cfg flag."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            reranker="share_prior",
            rerank_pool_size=pool_size,
            share_prior_alpha=alpha,
            share_prior_adaptive_alpha=adaptive,
        )
        eng = Engram(config=cfg)
        try:
            for content, meta in ds.memories:
                clean = {
                    k: v
                    for k, v in meta.items()
                    if isinstance(v, (str, int, float, bool))
                }
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                a, b = q.expected_substrings[0], q.expected_substrings[1]
                top10 = texts[:10]
                pair10 = int(
                    any(a.lower() in t.lower() for t in top10)
                    and any(b.lower() in t.lower() for t in top10)
                )
                top5 = texts[:5]
                pair5 = int(
                    any(a.lower() in t.lower() for t in top5)
                    and any(b.lower() in t.lower() for t in top5)
                )
                rows.append({"pair@5": pair5, "pair@10": pair10})
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "alpha": alpha,
        "adaptive": adaptive,
        "n": len(rows),
        "pair_recall@5": sum(r["pair@5"] for r in rows) / n,
        "pair_recall@10": sum(r["pair@10"] for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20,0.40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-distractors", type=int, default=40)
    ap.add_argument("--out", default="evals/results/share_prior_adaptive_alpha.json")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = list(range(args.seeds))

    # Compute baseline pair@10 (no reranker) for context.
    baseline_seed_vals = []
    for s in seeds:
        ds = _make_bridge_corpus(
            n_pairs=args.n_pairs,
            plain_distractors=args.n_distractors,
            seed=s,
        )
        b = _eval_bridge(ds, reranker=None, alpha=0.0, pool_size=20)
        baseline_seed_vals.append(b["pair_recall@10"])
    baseline_mean = stats.mean(baseline_seed_vals)

    rows = []
    for alpha in alphas:
        for adaptive in (False, True):
            seed_vals = []
            for s in seeds:
                ds = _make_bridge_corpus(
                    n_pairs=args.n_pairs,
                    plain_distractors=args.n_distractors,
                    seed=s,
                )
                r = _eval_bridge_adaptive(
                    ds, alpha=alpha, pool_size=20, adaptive=adaptive
                )
                seed_vals.append(r["pair_recall@10"])
            mean = stats.mean(seed_vals)
            sd = stats.pstdev(seed_vals) if len(seed_vals) > 1 else 0.0
            rows.append(
                {
                    "alpha": alpha,
                    "adaptive": adaptive,
                    "pair_recall@10": round(mean, 4),
                    "sd": round(sd, 4),
                    "delta_vs_baseline": round(mean - baseline_mean, 4),
                }
            )

    # Δ adaptive vs constant per alpha
    deltas = []
    for alpha in alphas:
        const = next(r for r in rows if r["alpha"] == alpha and not r["adaptive"])
        adapt = next(r for r in rows if r["alpha"] == alpha and r["adaptive"])
        deltas.append(
            {
                "alpha": alpha,
                "constant_pair@10": const["pair_recall@10"],
                "adaptive_pair@10": adapt["pair_recall@10"],
                "delta_adapt_minus_const": round(
                    adapt["pair_recall@10"] - const["pair_recall@10"], 4
                ),
            }
        )

    report = {
        "config": {
            "alphas": alphas,
            "seeds": seeds,
            "n_pairs": args.n_pairs,
            "n_distractors": args.n_distractors,
        },
        "baseline_pair@10_mean": round(baseline_mean, 4),
        "rows": rows,
        "deltas": deltas,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

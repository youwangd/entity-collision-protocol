"""§5.4 stack interaction × scale — does PRF × share_prior super-additivity
hold as the corpus grows, or is it a small-corpus artifact?

The 2×2 stack (`prf_x_shareprior_stack.py`) showed +0.067 super-additive
gain on Δpair_recall@10 at n_pairs=60. Reviewers will ask: does this
interaction scale? Specifically, three plausible regimes:

  A) Super-additivity grows  — PRF expands the candidate pool more
     aggressively as the corpus grows, giving the share_prior reranker
     more room to operate on. Interaction trends up.
  B) Super-additivity decays — PRF saturates (already finds the bridge
     pair via expansion alone) so SP has nothing to add. Interaction
     trends down.
  C) Super-additivity is stable — the orthogonality is structural
     (PRF expands pool, SP reorders within) and corpus-size
     independent.

JSON shape: top-level `by_scale` array (one entry per n_pairs) with
`per_seed.bridge` / `per_seed.unique` payloads keyed by full cell
names — directly consumable by `prf_x_shareprior_axis_ci`.

Driver: python -m evals.prf_x_shareprior_scale --seeds 17,42,101 \\
            --n-pairs-list 30,60,120,200

Output: evals/results/prf_x_shareprior_scale.json + markdown stdout.

Wall budget: ~3-5 min for default 4 scales × 4 cells × 3 seeds.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from evals.io_utils import atomic_write_json
from evals.prf_x_shareprior_stack import (
    CELLS, _eval_bridge, _eval_unique,
)
from evals.share_prior_sweep import generate_bridge_corpus
from evals.entity_channel_sweep import generate_entity_corpus


def run(*, seeds, n_pairs_list, plain_distractors, n_facts,
        top_k_for_prf, max_entities, min_dominance) -> dict:
    t0 = time.monotonic()
    bridge_keys = ["any_hit@10", "any_hit@20",
                   "pair_recall@10", "pair_recall@20"]
    unique_keys = ["hit@1", "hit@5"]

    out_levels = []

    for n_pairs in n_pairs_list:
        bridge_per_cell = {name: [] for name, *_ in CELLS}
        unique_per_cell = {name: [] for name, *_ in CELLS}
        for s in seeds:
            ds_b = generate_bridge_corpus(
                n_pairs=n_pairs, plain_distractors=plain_distractors, seed=s,
            )
            ds_u = generate_entity_corpus(
                n_facts=n_facts, hard_distractors_per_fact=2,
                plain_distractors=plain_distractors, seed=s + 1000,
            )
            for name, expand, reranker, alpha in CELLS:
                bridge_per_cell[name].append(_eval_bridge(
                    ds_b, expand=expand, reranker=reranker, alpha=alpha,
                    top_k_for_prf=top_k_for_prf, max_entities=max_entities,
                    min_dominance=min_dominance,
                ))
                unique_per_cell[name].append(_eval_unique(
                    ds_u, expand=expand, reranker=reranker, alpha=alpha,
                    top_k_for_prf=top_k_for_prf, max_entities=max_entities,
                    min_dominance=min_dominance,
                ))

        def agg(rows, keys):
            return {k: round(statistics.mean(r[k] for r in rows), 4)
                    for k in keys}

        c0 = agg(bridge_per_cell["C0_baseline"], bridge_keys)
        cp = agg(bridge_per_cell["CP_prf_only"], bridge_keys)
        cr = agg(bridge_per_cell["CR_share_prior_only"], bridge_keys)
        cb = agg(bridge_per_cell["CB_both"], bridge_keys)
        interactions = {}
        for k in ["pair_recall@10", "pair_recall@20"]:
            d_p = cp[k] - c0[k]
            d_r = cr[k] - c0[k]
            d_b = cb[k] - c0[k]
            interactions[k] = {
                "delta_PRF": round(d_p, 4),
                "delta_SP": round(d_r, 4),
                "delta_BOTH": round(d_b, 4),
                "interaction": round(d_b - (d_p + d_r), 4),
            }

        out_levels.append({
            "n_pairs": n_pairs,
            "bridge": {name: agg(bridge_per_cell[name], bridge_keys)
                       for name in bridge_per_cell},
            "unique_donoharm": {name: agg(unique_per_cell[name], unique_keys)
                                for name in unique_per_cell},
            "interactions": interactions,
            "per_seed": {
                "seeds": list(seeds),
                "bridge": {n: bridge_per_cell[n] for n in bridge_per_cell},
                "unique": {n: unique_per_cell[n] for n in unique_per_cell},
            },
        })

    return {
        "config": {
            "seeds": list(seeds),
            "n_pairs_list": list(n_pairs_list),
            "plain_distractors": plain_distractors,
            "n_facts": n_facts,
            "top_k_for_prf": top_k_for_prf,
            "max_entities": max_entities,
            "min_dominance": min_dominance,
        },
        "by_scale": out_levels,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"d={cfg['min_dominance']} α=0.05 pool=20",
        "",
        "### Stack interaction × n_pairs",
        "",
        "| n_pairs | Δ_PRF@10 | Δ_SP@10 | Δ_BOTH@10 | inter@10 | inter@20 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in rep["by_scale"]:
        i10 = lv["interactions"]["pair_recall@10"]
        i20 = lv["interactions"]["pair_recall@20"]
        tag = ("↑" if i10["interaction"] > 0.01
               else "↓" if i10["interaction"] < -0.01 else "≈")
        lines.append(
            f"| {lv['n_pairs']} | {i10['delta_PRF']:+.3f} | "
            f"{i10['delta_SP']:+.3f} | {i10['delta_BOTH']:+.3f} | "
            f"{i10['interaction']:+.3f} {tag} | "
            f"{i20['interaction']:+.3f} |"
        )
    lines += ["", "### Unique do-no-harm hit@1 (per scale)", ""]
    lines.append("| n_pairs | C0 | CP | CR | CB |")
    lines.append("|---:|---:|---:|---:|---:|")
    for lv in rep["by_scale"]:
        u = lv["unique_donoharm"]
        lines.append(
            f"| {lv['n_pairs']} | {u['C0_baseline']['hit@1']:.3f} "
            f"| {u['CP_prf_only']['hit@1']:.3f} "
            f"| {u['CR_share_prior_only']['hit@1']:.3f} "
            f"| {u['CB_both']['hit@1']:.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--n-pairs-list", type=str, default="30,60,120,200")
    p.add_argument("--plain-distractors", type=int, default=80)
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--top-k-for-prf", type=int, default=10)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.3)
    p.add_argument(
        "--out", default="evals/results/prf_x_shareprior_scale.json"
    )
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_pairs_list = [int(x) for x in args.n_pairs_list.split(",") if x.strip()]
    rep = run(
        seeds=seeds, n_pairs_list=n_pairs_list,
        plain_distractors=args.plain_distractors, n_facts=args.n_facts,
        top_k_for_prf=args.top_k_for_prf, max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 scale — PRF × share_prior interaction across n_pairs")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[prf-x-shareprior-scale] wrote {args.out}")


if __name__ == "__main__":
    main()

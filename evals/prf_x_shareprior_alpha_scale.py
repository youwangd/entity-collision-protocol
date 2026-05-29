"""§5.4 anchor 27 — PRF × share_prior joint α × n_pairs structural map.

Background. Anchor 25 (10-seed alpha replication at headline n=60)
confirmed α=0.05 as the only CI-clean super-additive operating point.
Anchor 26 (10-seed scale replication at α=0.05) showed the interaction
sign is non-monotone in n_pairs while Δ_BOTH stays strictly positive:

  n_pairs   |  30    60    120   200
  inter@10  | -0.20 +0.075 -0.053 +0.059   (all p<0.001 vs 0)

The natural follow-up reviewers will ask: at the larger scales where
α=0.05 produces a weaker (or sign-flipped) interaction, does relaxing
α to 0.10 / 0.20 recover super-additivity, or does it still collapse
the way it does at n=60? In other words: is α=0.05 the only safe
operating point at every scale, or only at headline?

This anchor is a 3-seed structural map of the joint α × n_pairs grid.
3 αs × 4 n_pairs × 4 cells × 3 seeds (~150-300s wall, fits in cron).
We do NOT run paired bootstrap CIs here — that's anchor-28 work if
the structural map shows anything interesting. The point of this run
is to *find* the interesting cells.

JSON shape: top-level `grid` array with one entry per (alpha, n_pairs)
cell, each carrying full `interactions` payload + 4-cell aggregates.
Markdown emits a compact 2-D table of interaction@10.

Driver:  python -m evals.prf_x_shareprior_alpha_scale \\
            --seeds 17,42,101 \\
            --alphas 0.05,0.10,0.20 \\
            --n-pairs-list 30,60,120,200
Output:  evals/results/prf_x_shareprior_alpha_scale.json
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from evals.prf_x_shareprior_stack import _eval_bridge, _eval_unique
from evals.share_prior_sweep import generate_bridge_corpus
from evals.entity_channel_sweep import generate_entity_corpus
from evals.io_utils import atomic_write_json


def _cells_for_alpha(alpha: float):
    return [
        ("C0_baseline", False, None, 0.0),
        ("CP_prf_only", True, None, 0.0),
        ("CR_share_prior_only", False, "share_prior", alpha),
        ("CB_both", True, "share_prior", alpha),
    ]


def run(*, seeds, alphas, n_pairs_list, plain_distractors, n_facts,
        top_k_for_prf, max_entities, min_dominance) -> dict:
    t0 = time.monotonic()
    bridge_keys = ["pair_recall@10", "pair_recall@20", "any_hit@20"]
    unique_keys = ["hit@1", "hit@5"]

    grid = []
    for alpha in alphas:
        for n_pairs in n_pairs_list:
            cells = _cells_for_alpha(alpha)
            bridge_per_cell = {name: [] for name, *_ in cells}
            unique_per_cell = {name: [] for name, *_ in cells}
            for s in seeds:
                ds_b = generate_bridge_corpus(
                    n_pairs=n_pairs,
                    plain_distractors=plain_distractors,
                    seed=s,
                )
                ds_u = generate_entity_corpus(
                    n_facts=n_facts, hard_distractors_per_fact=2,
                    plain_distractors=plain_distractors, seed=s + 1000,
                )
                for name, expand, reranker, a in cells:
                    bridge_per_cell[name].append(_eval_bridge(
                        ds_b, expand=expand, reranker=reranker, alpha=a,
                        top_k_for_prf=top_k_for_prf,
                        max_entities=max_entities,
                        min_dominance=min_dominance,
                    ))
                    unique_per_cell[name].append(_eval_unique(
                        ds_u, expand=expand, reranker=reranker, alpha=a,
                        top_k_for_prf=top_k_for_prf,
                        max_entities=max_entities,
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

            grid.append({
                "alpha": alpha,
                "n_pairs": n_pairs,
                "bridge": {n: agg(bridge_per_cell[n], bridge_keys)
                           for n in bridge_per_cell},
                "unique_donoharm": {n: agg(unique_per_cell[n], unique_keys)
                                    for n in unique_per_cell},
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
            "alphas": list(alphas),
            "n_pairs_list": list(n_pairs_list),
            "plain_distractors": plain_distractors,
            "n_facts": n_facts,
            "top_k_for_prf": top_k_for_prf,
            "max_entities": max_entities,
            "min_dominance": min_dominance,
        },
        "grid": grid,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    alphas = cfg["alphas"]
    n_list = cfg["n_pairs_list"]
    cell = {(g["alpha"], g["n_pairs"]): g for g in rep["grid"]}

    def fmt(v):
        return f"{v:+.3f}"

    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"d={cfg['min_dominance']} me={cfg['max_entities']}",
        "",
        "### α × n_pairs interaction@10 (Δ_BOTH − (Δ_PRF + Δ_SP))",
        "",
        "| α \\ n_pairs | " + " | ".join(str(n) for n in n_list) + " |",
        "|---:|" + "|".join("---:" for _ in n_list) + "|",
    ]
    for a in alphas:
        row = [f"{a}"]
        for n in n_list:
            g = cell[(a, n)]
            row.append(fmt(g["interactions"]["pair_recall@10"]["interaction"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### α × n_pairs Δ_BOTH@10 (CB − C0 absolute lift)",
        "",
        "| α \\ n_pairs | " + " | ".join(str(n) for n in n_list) + " |",
        "|---:|" + "|".join("---:" for _ in n_list) + "|",
    ]
    for a in alphas:
        row = [f"{a}"]
        for n in n_list:
            g = cell[(a, n)]
            row.append(fmt(g["interactions"]["pair_recall@10"]["delta_BOTH"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### α × n_pairs unique do-no-harm hit@1 (CB)",
        "",
        "| α \\ n_pairs | " + " | ".join(str(n) for n in n_list) + " |",
        "|---:|" + "|".join("---:" for _ in n_list) + "|",
    ]
    for a in alphas:
        row = [f"{a}"]
        for n in n_list:
            g = cell[(a, n)]
            row.append(f"{g['unique_donoharm']['CB_both']['hit@1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--alphas", type=str, default="0.05,0.10,0.20")
    p.add_argument("--n-pairs-list", type=str, default="30,60,120,200")
    p.add_argument("--plain-distractors", type=int, default=80)
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--top-k-for-prf", type=int, default=10)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.3)
    p.add_argument(
        "--out",
        default="evals/results/prf_x_shareprior_alpha_scale.json",
    )
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    n_list = [int(n) for n in args.n_pairs_list.split(",") if n.strip()]

    rep = run(
        seeds=seeds, alphas=alphas, n_pairs_list=n_list,
        plain_distractors=args.plain_distractors,
        n_facts=args.n_facts,
        top_k_for_prf=args.top_k_for_prf,
        max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 anchor 27 — PRF × share_prior joint α × n_pairs map")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[prf-x-sp-alpha-scale] wrote {args.out}")


if __name__ == "__main__":
    main()

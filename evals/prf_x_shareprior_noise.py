"""§5.4 anchor 13 — PRF × share_prior under varying distractor density.

Anchor 10 established super-additivity for PRF+SP at the canonical bridge
config. Anchor 11 swept corpus size (n_pairs); anchor 12 swept α. The
remaining structural axis is signal-to-noise: how much plain-distractor
mass surrounds the bridge pairs?

Hypothesis: the super-additive cell (CB > CP+CR) is a low-SNR phenomenon.
At low distractor count, PRF alone already saturates the pool (no
headroom for SP); at high distractor count, neither alone reaches @10
reliably, so stacking compounds.

Sweep: plain_distractors ∈ {40, 80, 160, 320}, α=0.05, n_pairs=60,
d=0.3, 3 seeds. Reuses the C0/CP/CR/CB cells from prf_x_shareprior_stack.

Driver: python -m evals.prf_x_shareprior_noise --seeds 17,42,101
Output: evals/results/prf_x_shareprior_noise.json + markdown stdout.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from evals.prf_x_shareprior_stack import (
    CELLS,
    _eval_bridge,
    _eval_unique,
)
from evals.share_prior_sweep import generate_bridge_corpus
from evals.entity_channel_sweep import generate_entity_corpus
from evals.io_utils import atomic_write_json


def run(*, seeds, n_pairs, distractor_levels, n_facts,
        top_k_for_prf, max_entities, min_dominance) -> dict:
    t0 = time.monotonic()
    out_levels = []

    for pd in distractor_levels:
        bridge_per_cell = {name: [] for name, *_ in CELLS}
        unique_per_cell = {name: [] for name, *_ in CELLS}
        for s in seeds:
            ds_b = generate_bridge_corpus(
                n_pairs=n_pairs, plain_distractors=pd, seed=s,
            )
            ds_u = generate_entity_corpus(
                n_facts=n_facts, hard_distractors_per_fact=2,
                plain_distractors=pd, seed=s + 1000,
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
            return {
                k: round(statistics.mean(r[k] for r in rows), 4)
                for k in keys
            }

        bridge_keys = ["any_hit@10", "any_hit@20",
                       "pair_recall@10", "pair_recall@20"]
        unique_keys = ["hit@1", "hit@5"]

        # Interaction term @10 and @20.
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
            "plain_distractors": pd,
            "bridge": {
                name: agg(bridge_per_cell[name], bridge_keys)
                for name in bridge_per_cell
            },
            "unique_donoharm": {
                name: agg(unique_per_cell[name], unique_keys)
                for name in unique_per_cell
            },
            "interactions": interactions,
            "per_seed": {
                "seeds": list(seeds),
                "bridge": {n: bridge_per_cell[n] for n in bridge_per_cell},
                "unique": {n: unique_per_cell[n] for n in unique_per_cell},
            },
        })

    return {
        "config": {
            "seeds": list(seeds), "n_pairs": n_pairs,
            "distractor_levels": list(distractor_levels),
            "n_facts": n_facts,
            "top_k_for_prf": top_k_for_prf, "max_entities": max_entities,
            "min_dominance": min_dominance,
            "share_prior_alpha": 0.05,
        },
        "by_distractors": out_levels,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"n_pairs={cfg['n_pairs']} α=0.05 d={cfg['min_dominance']} "
        f"levels={cfg['distractor_levels']}",
        "",
        "### Bridge — pair_recall@10 by distractor density",
        "",
        "| plain_distractors | C0 | CP (PRF) | CR (SP) | CB (both) | Δ_PRF | Δ_SP | Δ_BOTH | interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in rep["by_distractors"]:
        b = lv["bridge"]
        i = lv["interactions"]["pair_recall@10"]
        lines.append(
            f"| {lv['plain_distractors']} | "
            f"{b['C0_baseline']['pair_recall@10']:.3f} | "
            f"{b['CP_prf_only']['pair_recall@10']:.3f} | "
            f"{b['CR_share_prior_only']['pair_recall@10']:.3f} | "
            f"{b['CB_both']['pair_recall@10']:.3f} | "
            f"{i['delta_PRF']:+.3f} | {i['delta_SP']:+.3f} | "
            f"{i['delta_BOTH']:+.3f} | {i['interaction']:+.3f} |"
        )
    lines += [
        "",
        "### Bridge — pair_recall@20 by distractor density",
        "",
        "| plain_distractors | C0 | CP | CR | CB | Δ_PRF | Δ_SP | Δ_BOTH | interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in rep["by_distractors"]:
        b = lv["bridge"]
        i = lv["interactions"]["pair_recall@20"]
        lines.append(
            f"| {lv['plain_distractors']} | "
            f"{b['C0_baseline']['pair_recall@20']:.3f} | "
            f"{b['CP_prf_only']['pair_recall@20']:.3f} | "
            f"{b['CR_share_prior_only']['pair_recall@20']:.3f} | "
            f"{b['CB_both']['pair_recall@20']:.3f} | "
            f"{i['delta_PRF']:+.3f} | {i['delta_SP']:+.3f} | "
            f"{i['delta_BOTH']:+.3f} | {i['interaction']:+.3f} |"
        )
    lines += [
        "",
        "### Unique — do-no-harm hit@1 by distractor density",
        "",
        "| plain_distractors | C0 | CP | CR | CB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lv in rep["by_distractors"]:
        u = lv["unique_donoharm"]
        lines.append(
            f"| {lv['plain_distractors']} | "
            f"{u['C0_baseline']['hit@1']:.3f} | "
            f"{u['CP_prf_only']['hit@1']:.3f} | "
            f"{u['CR_share_prior_only']['hit@1']:.3f} | "
            f"{u['CB_both']['hit@1']:.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--distractor-levels", type=str, default="40,80,160,320")
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--top-k-for-prf", type=int, default=10)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.3)
    p.add_argument("--out",
                   default="evals/results/prf_x_shareprior_noise.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    levels = [int(s) for s in args.distractor_levels.split(",") if s.strip()]
    rep = run(
        seeds=seeds, n_pairs=args.n_pairs,
        distractor_levels=levels, n_facts=args.n_facts,
        top_k_for_prf=args.top_k_for_prf, max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 anchor 13 — PRF × share_prior × distractor density")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[prf-x-shareprior-noise] wrote {args.out}")


if __name__ == "__main__":
    main()

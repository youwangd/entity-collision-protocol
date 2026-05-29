"""§5.4 anchor 12 — PRF × share_prior interaction across α.

Anchor 10 fixed α=0.05 and found super-additivity (+0.067 Δpair@10).
Anchor 11 swept n_pairs at fixed α and showed the interaction is
regime-dependent (non-monotone). This anchor sweeps α at fixed
n_pairs to characterize how the SP-side knob shapes the surface.

Cells (4 per α): C0_baseline, CP_prf_only, CR_share_prior_only,
CB_both. α ∈ {0.05, 0.10, 0.20}. 3 seeds. ~60-90s wall.

JSON shape: top-level `by_alpha` array (one entry per α) with
`per_seed.bridge` / `per_seed.unique` payloads keyed by full cell
names — directly consumable by `prf_x_shareprior_axis_ci`.

Driver: python -m evals.prf_x_shareprior_alpha --seeds 17,42,101
Output: evals/results/prf_x_shareprior_alpha.json + markdown stdout.
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
    # Same names as prf_x_shareprior_stack.CELLS so axis_ci consumes
    # our JSON without translation.
    return [
        ("C0_baseline", False, None, 0.0),
        ("CP_prf_only", True, None, 0.0),
        ("CR_share_prior_only", False, "share_prior", alpha),
        ("CB_both", True, "share_prior", alpha),
    ]


def run(*, seeds, n_pairs, plain_distractors, n_facts,
        alphas, top_k_for_prf, max_entities, min_dominance) -> dict:
    t0 = time.monotonic()
    bridge_keys = ["pair_recall@10", "pair_recall@20", "any_hit@20"]
    unique_keys = ["hit@1", "hit@5"]

    by_alpha = []

    for alpha in alphas:
        cells = _cells_for_alpha(alpha)
        bridge_per_cell = {name: [] for name, *_ in cells}
        unique_per_cell = {name: [] for name, *_ in cells}

        for s in seeds:
            ds_b = generate_bridge_corpus(
                n_pairs=n_pairs, plain_distractors=plain_distractors, seed=s,
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
            return {
                k: round(statistics.mean(r[k] for r in rows), 4)
                for k in keys
            }

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

        by_alpha.append({
            "alpha": alpha,
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
            "seeds": list(seeds), "n_pairs": n_pairs,
            "plain_distractors": plain_distractors, "n_facts": n_facts,
            "alphas": list(alphas),
            "top_k_for_prf": top_k_for_prf,
            "max_entities": max_entities,
            "min_dominance": min_dominance,
        },
        "by_alpha": by_alpha,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"n_pairs={cfg['n_pairs']} d={cfg['min_dominance']}",
        "",
        "### Bridge — PRF × SP interaction Δs vs C0 by α (pair_recall@10)",
        "",
        "| α | Δ_PRF | Δ_SP | Δ_BOTH | interaction | regime |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for lv in rep["by_alpha"]:
        i = lv["interactions"]["pair_recall@10"]
        inter = i["interaction"]
        regime = ("super-additive" if inter > 0.005
                  else "sub-additive" if inter < -0.005
                  else "additive")
        lines.append(
            f"| {lv['alpha']} | {i['delta_PRF']:+.3f} | {i['delta_SP']:+.3f} "
            f"| {i['delta_BOTH']:+.3f} | {inter:+.3f} | {regime} |"
        )

    lines += ["", "### Unique do-no-harm hit@1 by α (CB cell)", ""]
    lines.append("| α | C0 | CB |")
    lines.append("|---:|---:|---:|")
    for lv in rep["by_alpha"]:
        u = lv["unique_donoharm"]
        lines.append(
            f"| {lv['alpha']} | {u['C0_baseline']['hit@1']:.3f} "
            f"| {u['CB_both']['hit@1']:.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=80)
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--alphas", type=str, default="0.05,0.10,0.20")
    p.add_argument("--top-k-for-prf", type=int, default=10)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.3)
    p.add_argument("--out",
                   default="evals/results/prf_x_shareprior_alpha.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    rep = run(
        seeds=seeds, n_pairs=args.n_pairs,
        plain_distractors=args.plain_distractors, n_facts=args.n_facts,
        alphas=alphas,
        top_k_for_prf=args.top_k_for_prf,
        max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 anchor 12 — PRF × share_prior across α")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[prf-x-sp-alpha] wrote {args.out}")


if __name__ == "__main__":
    main()

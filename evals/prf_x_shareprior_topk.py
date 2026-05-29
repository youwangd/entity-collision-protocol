"""§5.4 anchor 29 — PRF × share_prior across top_k_for_prf.

The 5-axis n=10 CI sweep (anchors 22-26) varied dominance gate, breadth
(max_entities), noise (plain_distractors), α, and corpus scale. The PRF
expansion budget itself — `top_k_for_prf`, the number of top-ranked hits
PRF mines for entity terms — has been held at 10 throughout. Reviewers
will ask: is the +0.075 super-additive interaction at the headline cell
robust to a tighter (k=5) or looser (k=20, k=40) PRF pool, or is it an
artifact of one specific budget?

Mechanism: at small k, PRF gets fewer terms but they're high-precision;
at large k, PRF gets more terms with more noise. SP is unchanged — it
only re-scores the union pool. So the prediction under H_super-additive
is that the interaction sign is stable across k while magnitude tracks
the bias-variance curve of PRF alone.

This is a 3-seed structural map (4 cells × 4 ks × 3 seeds = 48 evals
on bridge + 48 on unique, ~150-300s wall, fits in cron).

JSON shape: top-level `by_topk` array (one entry per k) with full
`per_seed.bridge` / `per_seed.unique` payloads — directly consumable by
`prf_x_shareprior_axis_ci` if anchor-30 paired-bootstrap is warranted.

Driver:
    python -m evals.prf_x_shareprior_topk \\
        --seeds 17,42,101 --topks 5,10,20,40
Output: evals/results/prf_x_shareprior_topk.json
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


CELLS = [
    ("C0_baseline", False, None, 0.0),
    ("CP_prf_only", True, None, 0.0),
    ("CR_share_prior_only", False, "share_prior", None),  # alpha filled in
    ("CB_both", True, "share_prior", None),
]


def _cells(alpha: float):
    return [
        ("C0_baseline", False, None, 0.0),
        ("CP_prf_only", True, None, 0.0),
        ("CR_share_prior_only", False, "share_prior", alpha),
        ("CB_both", True, "share_prior", alpha),
    ]


def run(*, seeds, topks, alpha, n_pairs, plain_distractors, n_facts,
        max_entities, min_dominance) -> dict:
    t0 = time.monotonic()
    bridge_keys = ["pair_recall@10", "pair_recall@20", "any_hit@20"]
    unique_keys = ["hit@1", "hit@5"]

    by_topk = []
    for k_prf in topks:
        cells = _cells(alpha)
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
                    top_k_for_prf=k_prf,
                    max_entities=max_entities,
                    min_dominance=min_dominance,
                ))
                unique_per_cell[name].append(_eval_unique(
                    ds_u, expand=expand, reranker=reranker, alpha=a,
                    top_k_for_prf=k_prf,
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

        by_topk.append({
            "top_k_for_prf": k_prf,
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
            "topks": list(topks),
            "alpha": alpha,
            "n_pairs": n_pairs,
            "plain_distractors": plain_distractors,
            "n_facts": n_facts,
            "max_entities": max_entities,
            "min_dominance": min_dominance,
        },
        "by_topk": by_topk,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"α={cfg['alpha']} n_pairs={cfg['n_pairs']} "
        f"d={cfg['min_dominance']} me={cfg['max_entities']}",
        "",
        "### Bridge — PRF × SP interaction Δs vs C0 by top_k_for_prf "
        "(pair_recall@10)",
        "",
        "| k_prf | Δ_PRF | Δ_SP | Δ_BOTH | interaction | regime |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for lv in rep["by_topk"]:
        i = lv["interactions"]["pair_recall@10"]
        inter = i["interaction"]
        regime = ("super-additive" if inter > 0.005
                  else "sub-additive" if inter < -0.005
                  else "additive")
        lines.append(
            f"| {lv['top_k_for_prf']} | {i['delta_PRF']:+.3f} "
            f"| {i['delta_SP']:+.3f} | {i['delta_BOTH']:+.3f} "
            f"| {inter:+.3f} | {regime} |"
        )

    lines += ["", "### Unique do-no-harm hit@1 by k_prf (CB cell)", ""]
    lines.append("| k_prf | C0 | CB |")
    lines.append("|---:|---:|---:|")
    for lv in rep["by_topk"]:
        u = lv["unique_donoharm"]
        lines.append(
            f"| {lv['top_k_for_prf']} | {u['C0_baseline']['hit@1']:.3f} "
            f"| {u['CB_both']['hit@1']:.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--topks", type=str, default="5,10,20,40")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=80)
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.3)
    p.add_argument("--out",
                   default="evals/results/prf_x_shareprior_topk.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    topks = [int(k) for k in args.topks.split(",") if k.strip()]
    rep = run(
        seeds=seeds, topks=topks, alpha=args.alpha,
        n_pairs=args.n_pairs,
        plain_distractors=args.plain_distractors, n_facts=args.n_facts,
        max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 anchor 29 — PRF × share_prior across top_k_for_prf")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[prf-x-sp-topk] wrote {args.out}")


if __name__ == "__main__":
    main()

"""§95-CI: paired bootstrap CI on the multi-hop pair_recall@k / gold_recall@k
regression observed in §95.

The §95 driver (`evals.locomo_recall_lift`) emits `per_query_pairs[]` with
per-pair `delta_prk` (pair_recall@k Δ) and `delta_grk` (gold_recall@k Δ),
plus `n_gold`. Multi-hop = `n_gold >= 2`. We filter to that slice and
percentile-bootstrap the paired mean.

Usage:
    python -m evals.locomo_recall_lift_multihop_ci \\
        --in bench/results/locomo_recall_lift_§95_synth_off.json \\
        --resamples 10000 --seed 42 \\
        --out bench/results/locomo_recall_lift_§95_multihop_ci.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from evals.io_utils import atomic_write_json


def _bootstrap_paired(values, resamples, seed, alpha=0.05):
    """Percentile bootstrap CI on paired-mean. Also returns two-sided p."""
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "p_bootstrap_two_sided": 1.0,
            "frac_pairs_treatment_better": 0.0,
            "frac_pairs_treatment_worse": 0.0,
        }
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * resamples))
    hi_idx = min(int(math.ceil((1 - alpha / 2) * resamples)) - 1, resamples - 1)
    leq = sum(1 for m in means if m <= 0)
    geq = sum(1 for m in means if m >= 0)
    p = min(1.0, 2 * min(leq, geq) / resamples)
    return {
        "n": n,
        "mean": round(statistics.fmean(values), 6),
        "ci_lo": round(means[lo_idx], 6),
        "ci_hi": round(means[hi_idx], 6),
        "p_bootstrap_two_sided": round(p, 6),
        "frac_pairs_treatment_better": round(
            sum(1 for v in values if v > 0) / n, 4
        ),
        "frac_pairs_treatment_worse": round(
            sum(1 for v in values if v < 0) / n, 4
        ),
    }


def _slice(pairs, predicate):
    return [p for p in pairs if predicate(p)]


def _summarize(pairs, resamples, seed):
    out = {"n_pairs": len(pairs)}
    for key in ("delta_prk", "delta_grk", "delta_h1", "delta_hk", "delta_rr"):
        vals = [float(p[key]) for p in pairs if key in p]
        if not vals:
            continue
        out[key] = _bootstrap_paired(vals, resamples, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-n-gold",
        type=int,
        default=2,
        help="Filter to pairs with n_gold >= this (default 2 = multi-hop)",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    pairs = data.get("per_query_pairs") or []
    if not pairs:
        raise SystemExit("no per_query_pairs in input")

    multi_hop = _slice(pairs, lambda p: int(p.get("n_gold", 0)) >= args.min_n_gold)
    single_hop = _slice(pairs, lambda p: int(p.get("n_gold", 0)) < args.min_n_gold)

    out_obj = {
        "source": str(args.inp),
        "recipe": data.get("recipe"),
        "synthesis": data.get("synthesis"),
        "n_pairs_total": len(pairs),
        "min_n_gold": args.min_n_gold,
        "ci_config": {
            "resamples": args.resamples,
            "seed": args.seed,
            "alpha": 0.05,
            "method": "percentile_paired",
        },
        "multi_hop": _summarize(multi_hop, args.resamples, args.seed),
        "single_hop": _summarize(single_hop, args.resamples, args.seed),
        "all": _summarize(pairs, args.resamples, args.seed),
    }

    print(f"§95-CI on {args.inp}")
    print(
        f"  synthesis={data.get('synthesis')}  n_total={len(pairs)}  "
        f"n_multi_hop={len(multi_hop)}  n_single_hop={len(single_hop)}  "
        f"resamples={args.resamples}"
    )
    for slice_name in ("multi_hop", "single_hop", "all"):
        block = out_obj[slice_name]
        print(f"  --- {slice_name} (n={block['n_pairs']}) ---")
        for k in ("delta_prk", "delta_grk", "delta_h1", "delta_hk", "delta_rr"):
            c = block.get(k)
            if not c:
                continue
            print(
                f"    {k:>10}: mean={c['mean']:+.4f}  "
                f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
                f"p={c['p_bootstrap_two_sided']:.4f}  "
                f"better={c['frac_pairs_treatment_better']:.3f}  "
                f"worse={c['frac_pairs_treatment_worse']:.3f}"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out_obj)
        print(f"\n[locomo_recall_lift_multihop_ci] wrote {args.out}")


if __name__ == "__main__":
    main()

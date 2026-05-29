"""Bootstrap CI on the §94c paired LoCoMo recall-lift deltas.

Reads a `locomo_recall_lift.py` output JSON (which has
`per_query_pairs[].delta_h1`, `delta_hk`, `delta_rr`) and emits a percentile
bootstrap CI on the *paired* per-pair deltas. The pairing is intrinsic — each
`per_query_pairs` row already holds `treatment_x - baseline_x` for the same
QA pair under both arms — so we just resample rows with replacement.

Usage:
    python -m evals.locomo_recall_lift_ci \\
        --in bench/results/locomo_recall_lift_§94c_synth_off.json \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§94c_ci.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from evals.io_utils import atomic_write_json


def _bootstrap_mean_ci(values, resamples, seed, alpha=0.05):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
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
    return statistics.fmean(values), means[lo_idx], means[hi_idx]


def _frac_pos(values):
    if not values:
        return 0.0
    return sum(1 for v in values if v > 0) / len(values)


def _summarize(pairs, resamples, seed):
    out = {"n_pairs": len(pairs)}
    for key in ("delta_h1", "delta_hk", "delta_rr"):
        vals = [float(p[key]) for p in pairs]
        m, lo, hi = _bootstrap_mean_ci(vals, resamples, seed)
        # Two-sided bootstrap p-value: 2 * min(P(mean<=0), P(mean>=0)).
        rng = random.Random(seed + 1)
        n = len(vals)
        leq, geq = 0, 0
        for _ in range(resamples):
            s = 0.0
            for _ in range(n):
                s += vals[rng.randrange(n)]
            mu = s / n
            if mu <= 0:
                leq += 1
            if mu >= 0:
                geq += 1
        # Percentile-bootstrap two-sided p for H0: mean=0.
        # When the observed mean is 0, leq ≈ geq ≈ resamples → p ≈ 2.0,
        # clamp to 1.0 so the constant-zero case returns p=1.
        p_two_sided = min(1.0, 2 * min(leq, geq) / resamples)
        out[key] = {
            "mean": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_bootstrap_two_sided": round(p_two_sided, 6),
            "frac_pairs_treatment_better": round(_frac_pos(vals), 4),
        }
    return out


def _per_category(pairs, resamples, seed):
    by_cat = {}
    for p in pairs:
        by_cat.setdefault(str(p.get("category", "?")), []).append(p)
    return {cat: _summarize(ps, resamples, seed) for cat, ps in sorted(by_cat.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    pairs = data.get("per_query_pairs") or []
    if not pairs:
        raise SystemExit("no per_query_pairs in input")

    overall = _summarize(pairs, args.resamples, args.seed)

    out_obj = {
        "source": str(args.inp),
        "recipe": data.get("recipe"),
        "synthesis": data.get("synthesis"),
        "n_pairs": len(pairs),
        "ci_config": {"resamples": args.resamples, "seed": args.seed,
                      "alpha": 0.05, "method": "percentile_paired"},
        "overall": overall,
    }
    if args.per_category:
        out_obj["per_category"] = _per_category(pairs, args.resamples, args.seed)

    print(f"§94c-CI on {args.inp}")
    print(f"  synthesis={data.get('synthesis')}  n_pairs={len(pairs)}  "
          f"resamples={args.resamples}")
    for k in ("delta_h1", "delta_hk", "delta_rr"):
        c = overall[k]
        print(f"  {k:>9}: mean={c['mean']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}  "
              f"frac_better={c['frac_pairs_treatment_better']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out_obj)
        print(f"\n[locomo_recall_lift_ci] wrote {args.out}")


if __name__ == "__main__":
    main()

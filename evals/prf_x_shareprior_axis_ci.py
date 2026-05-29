"""§5.4 anchor 18 — paired bootstrap CIs across a structural-axis sweep.

Generalizes the headline-stack bootstrap (anchor 17) to the swept-axis JSONs
produced by `prf_x_shareprior_{gate,breadth,noise,scale,alpha}`. Each input
must include a top-level swept array (`by_gate` / `by_breadth` / etc.); each
level must have a `per_seed.bridge` / `per_seed.unique` payload of
`dict[cell_name, list[per-seed metric dict]]`.

For every level and every metric we report the four cell means + paired
bootstrap-CI Δs (Δ_PRF, Δ_SP, Δ_BOTH) + interaction term, with empirical
two-sided p-value at H0=0. The same seed-resample index is shared across all
four cells per draw so paired Δs and the interaction share rng — that's the
property that makes interaction CIs honest.

Why this matters for the paper: anchors 11-15 currently report point estimates
across 3 seeds. Reviewers will ask "is the +0.100 interaction at d=0.3
distinguishable from the +0.000 at d=0.5?" — this tool answers that with
overlapping/non-overlapping CIs and per-cell p-values.

Usage:
    python -m evals.prf_x_shareprior_axis_ci \\
        --in evals/results/prf_x_shareprior_gate.json \\
        --axis-key by_gate --level-key min_dominance \\
        --resamples 5000 --seed 17 \\
        --out evals/results/prf_x_shareprior_gate_ci.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from evals.io_utils import atomic_write_json


CELLS = ["C0_baseline", "CP_prf_only", "CR_share_prior_only", "CB_both"]
BRIDGE_METRICS = ["pair_recall@10", "pair_recall@20"]
UNIQUE_METRICS = ["hit@1", "hit@5"]


def _percentile(sorted_xs: list[float], q: float) -> float:
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def _two_sided_p(samples: list[float], h0: float = 0.0) -> float:
    n = len(samples)
    if n == 0:
        return 1.0
    le = sum(1 for x in samples if x <= h0)
    ge = sum(1 for x in samples if x >= h0)
    return min(1.0, 2.0 * min(le, ge) / n)


def _bootstrap_paired(per_seed: dict[str, list[dict]], metric: str,
                      resamples: int, seed: int) -> dict:
    seeds_n = len(next(iter(per_seed.values())))
    rng = random.Random(seed)

    delta_prf: list[float] = []
    delta_sp: list[float] = []
    delta_both: list[float] = []
    interaction: list[float] = []

    for _ in range(resamples):
        idx = [rng.randrange(seeds_n) for _ in range(seeds_n)]
        means: dict[str, float] = {}
        for c in per_seed:
            vals = [per_seed[c][i][metric] for i in idx]
            means[c] = statistics.fmean(vals)
        d_p = means["CP_prf_only"] - means["C0_baseline"]
        d_r = means["CR_share_prior_only"] - means["C0_baseline"]
        d_b = means["CB_both"] - means["C0_baseline"]
        delta_prf.append(d_p)
        delta_sp.append(d_r)
        delta_both.append(d_b)
        interaction.append(d_b - (d_p + d_r))

    def _summary(point: float, samples: list[float]) -> dict:
        s = sorted(samples)
        return {
            "point": round(point, 4),
            "ci95": [round(_percentile(s, 0.025), 4),
                     round(_percentile(s, 0.975), 4)],
            "p_two_sided_vs_0": round(_two_sided_p(samples, 0.0), 4),
        }

    cells_pt = {
        c: round(statistics.fmean(per_seed[c][i][metric]
                                  for i in range(seeds_n)), 4)
        for c in per_seed
    }
    pt_dp = cells_pt["CP_prf_only"] - cells_pt["C0_baseline"]
    pt_dr = cells_pt["CR_share_prior_only"] - cells_pt["C0_baseline"]
    pt_db = cells_pt["CB_both"] - cells_pt["C0_baseline"]
    pt_int = pt_db - (pt_dp + pt_dr)

    return {
        "cells": cells_pt,
        "delta_prf": _summary(pt_dp, delta_prf),
        "delta_sp": _summary(pt_dr, delta_sp),
        "delta_both": _summary(pt_db, delta_both),
        "interaction": _summary(pt_int, interaction),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--axis-key", required=True,
                   help="top-level array key (by_gate, by_breadth, ...)")
    p.add_argument("--level-key", required=True,
                   help="per-level identifier key (min_dominance, ...)")
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rep = json.loads(Path(args.inp).read_text())
    if args.axis_key not in rep:
        raise SystemExit(f"input JSON missing '{args.axis_key}'")
    levels = rep[args.axis_key]
    if not levels or "per_seed" not in levels[0]:
        raise SystemExit(
            "swept levels missing 'per_seed' — re-run the upstream eval "
            "after adding per-seed payloads"
        )

    out = {
        "source": str(args.inp),
        "axis_key": args.axis_key,
        "level_key": args.level_key,
        "resamples": args.resamples,
        "seed": args.seed,
        "n_seeds": len(levels[0]["per_seed"]["seeds"]),
        "levels": [],
    }
    for lv in levels:
        lv_out = {args.level_key: lv[args.level_key],
                  "bridge": {}, "unique": {}}
        for m in BRIDGE_METRICS:
            lv_out["bridge"][m] = _bootstrap_paired(
                lv["per_seed"]["bridge"], m, args.resamples, args.seed,
            )
        for m in UNIQUE_METRICS:
            lv_out["unique"][m] = _bootstrap_paired(
                lv["per_seed"]["unique"], m, args.resamples, args.seed,
            )
        out["levels"].append(lv_out)

    print(f"§5.4 axis bootstrap CI ({args.axis_key}, n_seeds={out['n_seeds']}, "
          f"resamples={args.resamples})")
    print()
    print(f"### Bridge pair_recall@10 — interaction by {args.level_key}")
    print()
    print(f"| {args.level_key} | Δ_PRF | Δ_SP | Δ_BOTH | interaction |")
    print("|---:|---:|---:|---:|---:|")
    for lv in out["levels"]:
        b = lv["bridge"]["pair_recall@10"]
        def cell(d):
            lo, hi = d["ci95"]
            return (f"{d['point']:+.3f} [{lo:+.3f},{hi:+.3f}] "
                    f"p={d['p_two_sided_vs_0']:.3f}")
        print(f"| {lv[args.level_key]} | {cell(b['delta_prf'])} "
              f"| {cell(b['delta_sp'])} | {cell(b['delta_both'])} "
              f"| {cell(b['interaction'])} |")

    print()
    print(f"### Unique do-no-harm hit@1 — by {args.level_key}")
    print()
    print(f"| {args.level_key} | Δ_PRF | Δ_SP | Δ_BOTH | interaction |")
    print("|---:|---:|---:|---:|---:|")
    for lv in out["levels"]:
        u = lv["unique"]["hit@1"]
        def cell(d):
            lo, hi = d["ci95"]
            return (f"{d['point']:+.3f} [{lo:+.3f},{hi:+.3f}] "
                    f"p={d['p_two_sided_vs_0']:.3f}")
        print(f"| {lv[args.level_key]} | {cell(u['delta_prf'])} "
              f"| {cell(u['delta_sp'])} | {cell(u['delta_both'])} "
              f"| {cell(u['interaction'])} |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"\n[prf-x-sp-axis-ci] wrote {args.out}")


if __name__ == "__main__":
    main()

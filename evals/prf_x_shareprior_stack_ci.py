"""Paired bootstrap CIs over per-seed cell metrics from the PRF × share_prior
2×2 stack run.

Reads the JSON produced by `evals.prf_x_shareprior_stack` (which now stores
`per_seed.bridge` / `per_seed.unique`: dict[cell_name, list[per-seed dict]]).
For each metric we compute:

  - per-cell mean with paired-percentile 95% CI (resamples seeds with
    replacement; same resample index applied to all four cells so paired
    differences and the interaction term share rng).
  - Δ_PRF = CP − C0
  - Δ_SP  = CR − C0
  - Δ_BOTH = CB − C0
  - interaction = Δ_BOTH − (Δ_PRF + Δ_SP)
    each with 95% bootstrap CI and an empirical two-sided p-value at 0.

Usage:
    python -m evals.prf_x_shareprior_stack_ci \\
        --in evals/results/prf_x_shareprior_stack_10seed.json \\
        --resamples 5000 --seed 17

This is the headline-defensibility number for §5.4: it tells the reader
whether the +0.073 interaction at 10 seeds is distinguishable from 0 once
seed variance is bootstrapped.
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
BRIDGE_METRICS = ["pair_recall@5", "pair_recall@10", "pair_recall@20",
                  "any_hit@5", "any_hit@10", "any_hit@20"]
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
    """Empirical two-sided p-value: 2 * min(P(X≤h0), P(X≥h0))."""
    n = len(samples)
    if n == 0:
        return 1.0
    le = sum(1 for x in samples if x <= h0)
    ge = sum(1 for x in samples if x >= h0)
    return min(1.0, 2.0 * min(le, ge) / n)


def _bootstrap_paired(per_seed: dict[str, list[dict]], metric: str,
                      resamples: int, seed: int) -> dict:
    """Paired bootstrap over seeds. Returns means + CIs for each cell and
    for the three Δs and the interaction term."""
    seeds_n = len(next(iter(per_seed.values())))
    rng = random.Random(seed)

    cell_samples: dict[str, list[float]] = {c: [] for c in per_seed}
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
            cell_samples[c].append(means[c])
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

    cells_out = {}
    for c in per_seed:
        point = statistics.fmean(per_seed[c][i][metric]
                                 for i in range(seeds_n))
        cells_out[c] = {
            "point": round(point, 4),
            "ci95": [round(_percentile(sorted(cell_samples[c]), 0.025), 4),
                     round(_percentile(sorted(cell_samples[c]), 0.975), 4)],
        }
    pt_dp = (cells_out["CP_prf_only"]["point"]
             - cells_out["C0_baseline"]["point"])
    pt_dr = (cells_out["CR_share_prior_only"]["point"]
             - cells_out["C0_baseline"]["point"])
    pt_db = cells_out["CB_both"]["point"] - cells_out["C0_baseline"]["point"]
    pt_int = pt_db - (pt_dp + pt_dr)

    return {
        "cells": cells_out,
        "delta_prf": _summary(pt_dp, delta_prf),
        "delta_sp": _summary(pt_dr, delta_sp),
        "delta_both": _summary(pt_db, delta_both),
        "interaction": _summary(pt_int, interaction),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rep = json.loads(Path(args.inp).read_text())
    if "per_seed" not in rep:
        raise SystemExit(
            "input JSON missing 'per_seed' — re-run "
            "evals.prf_x_shareprior_stack to populate per-seed rows"
        )

    out = {
        "source": str(args.inp),
        "resamples": args.resamples,
        "seed": args.seed,
        "n_seeds": len(rep["per_seed"]["seeds"]),
        "bridge": {},
        "unique": {},
    }
    for m in BRIDGE_METRICS:
        out["bridge"][m] = _bootstrap_paired(
            rep["per_seed"]["bridge"], m, args.resamples, args.seed,
        )
    for m in UNIQUE_METRICS:
        out["unique"][m] = _bootstrap_paired(
            rep["per_seed"]["unique"], m, args.resamples, args.seed,
        )

    print(f"§5.4 stack bootstrap CI (n_seeds={out['n_seeds']}, "
          f"resamples={args.resamples})")
    print()
    print("### Bridge — paired Δs vs C0 (95% bootstrap CI; p vs 0)")
    print()
    print("| metric | Δ_PRF | Δ_SP | Δ_BOTH | interaction |")
    print("|:---|---:|---:|---:|---:|")
    for m in BRIDGE_METRICS:
        b = out["bridge"][m]
        def cell(d):
            lo, hi = d["ci95"]
            return f"{d['point']:+.3f} [{lo:+.3f},{hi:+.3f}] p={d['p_two_sided_vs_0']:.3f}"
        print(f"| {m} | {cell(b['delta_prf'])} | {cell(b['delta_sp'])} "
              f"| {cell(b['delta_both'])} | {cell(b['interaction'])} |")

    print()
    print("### Unique do-no-harm — paired Δs vs C0")
    print()
    print("| metric | Δ_PRF | Δ_SP | Δ_BOTH | interaction |")
    print("|:---|---:|---:|---:|---:|")
    for m in UNIQUE_METRICS:
        u = out["unique"][m]
        def cell(d):
            lo, hi = d["ci95"]
            return f"{d['point']:+.3f} [{lo:+.3f},{hi:+.3f}] p={d['p_two_sided_vs_0']:.3f}"
        print(f"| {m} | {cell(u['delta_prf'])} | {cell(u['delta_sp'])} "
              f"| {cell(u['delta_both'])} | {cell(u['interaction'])} |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"\n[prf-x-sp-ci] wrote {args.out}")


if __name__ == "__main__":
    main()

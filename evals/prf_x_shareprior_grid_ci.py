"""§5.4 anchor 31 — paired bootstrap CIs on a 2-D (α × n_pairs) cell grid.

Companion to `prf_x_shareprior_axis_ci` for joint structural sweeps. Consumes
the JSON produced by `prf_x_shareprior_alpha_scale` (top-level `grid` array,
each entry carrying `per_seed.bridge` / `per_seed.unique` payloads keyed by
the four CELLS) and emits paired bootstrap CIs + p-values per cell.

Why a joint driver: anchor 27 was 3-seed structural. With n=10 reruns the
question becomes whether the headline α=0.05 super-additive regime survives
at n_pairs=200 (where anchor 26 saw it flip back to +0.059). A joint n=10
CI panel either confirms α=0.05's monopoly across all 12 cells or pinpoints
where it breaks.

Usage:
    python -m evals.prf_x_shareprior_grid_ci \\
        --in evals/results/prf_x_shareprior_alpha_scale_n10.json \\
        --resamples 10000 --seed 17 \\
        --out evals/results/prf_x_shareprior_alpha_scale_n10_ci.json
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
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rep = json.loads(Path(args.inp).read_text())
    if "grid" not in rep:
        raise SystemExit("input JSON missing 'grid' (expected alpha_scale shape)")
    grid = rep["grid"]
    if not grid or "per_seed" not in grid[0]:
        raise SystemExit(
            "grid cells missing 'per_seed' — re-run upstream eval with "
            "per-seed payloads"
        )

    n_seeds = len(grid[0]["per_seed"]["seeds"])
    out = {
        "source": str(args.inp),
        "resamples": args.resamples,
        "seed": args.seed,
        "n_seeds": n_seeds,
        "cells": [],
    }
    for g in grid:
        c_out = {
            "alpha": g["alpha"],
            "n_pairs": g["n_pairs"],
            "bridge": {},
            "unique": {},
        }
        for m in BRIDGE_METRICS:
            c_out["bridge"][m] = _bootstrap_paired(
                g["per_seed"]["bridge"], m, args.resamples, args.seed,
            )
        for m in UNIQUE_METRICS:
            c_out["unique"][m] = _bootstrap_paired(
                g["per_seed"]["unique"], m, args.resamples, args.seed,
            )
        out["cells"].append(c_out)

    print(f"§5.4 anchor 31 — α × n_pairs joint paired bootstrap CIs "
          f"(n_seeds={n_seeds}, resamples={args.resamples})")
    print()
    # Build pivoted interaction@10 table
    alphas = sorted({c["alpha"] for c in out["cells"]})
    npairs = sorted({c["n_pairs"] for c in out["cells"]})
    by_cell = {(c["alpha"], c["n_pairs"]): c for c in out["cells"]}

    def fmt(d):
        lo, hi = d["ci95"]
        return (f"{d['point']:+.3f} [{lo:+.3f},{hi:+.3f}] "
                f"p={d['p_two_sided_vs_0']:.3f}")

    print("### Bridge pair_recall@10 — interaction CI by (α, n_pairs)")
    print()
    print("| α \\ n_pairs | " + " | ".join(str(n) for n in npairs) + " |")
    print("|---:|" + "|".join("---:" for _ in npairs) + "|")
    for a in alphas:
        row = [f"{a}"]
        for n in npairs:
            row.append(fmt(by_cell[(a, n)]["bridge"]["pair_recall@10"]
                           ["interaction"]))
        print("| " + " | ".join(row) + " |")

    print()
    print("### Bridge pair_recall@10 — Δ_BOTH CI by (α, n_pairs)")
    print()
    print("| α \\ n_pairs | " + " | ".join(str(n) for n in npairs) + " |")
    print("|---:|" + "|".join("---:" for _ in npairs) + "|")
    for a in alphas:
        row = [f"{a}"]
        for n in npairs:
            row.append(fmt(by_cell[(a, n)]["bridge"]["pair_recall@10"]
                           ["delta_both"]))
        print("| " + " | ".join(row) + " |")

    print()
    print("### Unique do-no-harm hit@1 — Δ_BOTH CI by (α, n_pairs)")
    print()
    print("| α \\ n_pairs | " + " | ".join(str(n) for n in npairs) + " |")
    print("|---:|" + "|".join("---:" for _ in npairs) + "|")
    for a in alphas:
        row = [f"{a}"]
        for n in npairs:
            row.append(fmt(by_cell[(a, n)]["unique"]["hit@1"]["delta_both"]))
        print("| " + " | ".join(row) + " |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"\n[prf-x-sp-grid-ci] wrote {args.out}")


if __name__ == "__main__":
    main()

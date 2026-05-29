"""§4.6 — L0→schema promotion churn-budget sweep.

Sweeps `ConsolidationConfig.schema_promote_threshold` over a grid and
measures, on the LoCoMo recall-lift harness, both the recall delta vs.
the §87 baseline and a churn proxy (schemas_created per consolidation
tick). Output drives the §4.6 recall × churn frontier figure in
`paper/40_results.md`.

Driver design
-------------
For each threshold `t` in the grid, we re-run the §94c paired
recall-lift harness with `schema_promote_threshold=t` injected into
the *treatment* arm's `ConsolidationConfig`. The baseline arm is
unchanged across the sweep — so per-row `delta_h1`, `delta_hk`,
`delta_grk` measure (treatment@t − baseline_no_schemafamily). We
then bootstrap-CI those paired deltas (10k resamples, α=0.05)
against the published §94c headline at default `t=3`.

Pure: deterministic given dataset + embedder. Output is a JSON
list of (threshold, delta_h1_mean+CI, delta_hk_mean+CI, delta_grk,
schemas_created_mean_per_sample, n_pairs).

CLI
---
    python -m evals.locomo_promote_threshold_sweep \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --thresholds 1,2,3,5,7,10 \\
        --resamples 2000 \\
        --out bench/results/locomo_promote_threshold_sweep.json

Notes
-----
Default `schema_promote_threshold` is 3 (matches `ConsolidationConfig`
default). Lowering it makes promotion easier → higher schema churn,
potentially more recall lift but more table bloat. Raising it makes
promotion stricter → fewer schemas, sparser SCHEMA table, possibly
weaker session-pivot signal at recall.

This is the §4.6 *operational* sweep — it complements §4.6's
information-theoretic derivation by probing real LoCoMo recall.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

from evals.locomo_recall_lift import run_recall_lift
from evals.io_utils import atomic_write_json


def _bootstrap_mean_ci(values, resamples: int, seed: int, alpha: float = 0.05):
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


def run_sweep(
    dataset_path: str | os.PathLike,
    *,
    thresholds: list[int],
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    resamples: int = 2000,
    seed: int = 42,
    synthesis: bool = True,
) -> dict:
    rows: list[dict] = []
    t0 = time.monotonic()
    for t in thresholds:
        sub_t = time.monotonic()
        result = run_recall_lift(
            dataset_path,
            max_instances=max_instances,
            k=k,
            embedder_name=embedder_name,
            schema_promote_threshold=t,
            synthesis=synthesis,
        )
        pairs = result.get("per_query_pairs") or []
        d_h1 = [float(p["delta_h1"]) for p in pairs]
        d_hk = [float(p["delta_hk"]) for p in pairs]
        d_grk = [float(p["delta_grk"]) for p in pairs]
        m_h1, lo_h1, hi_h1 = _bootstrap_mean_ci(d_h1, resamples, seed)
        m_hk, lo_hk, hi_hk = _bootstrap_mean_ci(d_hk, resamples, seed)
        m_grk, lo_grk, hi_grk = _bootstrap_mean_ci(d_grk, resamples, seed)
        rows.append({
            "schema_promote_threshold": t,
            "n_pairs": len(pairs),
            "delta_h1": {"mean": round(m_h1, 6),
                         "ci_lo": round(lo_h1, 6),
                         "ci_hi": round(hi_h1, 6)},
            "delta_hk": {"mean": round(m_hk, 6),
                         "ci_lo": round(lo_hk, 6),
                         "ci_hi": round(hi_hk, 6)},
            "delta_grk": {"mean": round(m_grk, 6),
                          "ci_lo": round(lo_grk, 6),
                          "ci_hi": round(hi_grk, 6)},
            "churn": result.get("churn", {}),
            "wall_seconds": round(time.monotonic() - sub_t, 2),
        })
    return {
        "dataset": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "resamples": resamples,
        "seed": seed,
        "thresholds": list(thresholds),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    ap.add_argument("--max-instances", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embedder", default="hashtrigram")
    ap.add_argument("--thresholds", default="1,2,3,5,7,10")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-synthesis", action="store_true",
                    help="disable §93 schema synthesis (default: on)")
    args = ap.parse_args()
    thr = [int(x) for x in args.thresholds.split(",") if x.strip()]
    out = run_sweep(
        args.dataset,
        thresholds=thr,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        resamples=args.resamples,
        seed=args.seed,
        synthesis=not args.no_synthesis,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out, default=str)
        print(f"[promote_threshold_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()

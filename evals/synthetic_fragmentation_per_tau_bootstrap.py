"""§85 — Bootstrap CI on Δ for the §82 SYNTHETIC per-tau fragmentation curve.

Sister driver to §83 (`evals.locomo_fragmentation_per_tau_bootstrap`).
§82 / §79 published a point-estimate per-tau Δ on the synthetic
disjoint-core corpus (n_clusters=200, cluster_size=4, vocab=2000,
core=8, schema=6) and reported 7/8 GATEABLE taus under debiased
semantics with a clean Δ = 0.0925 at every gateable tau ≤ 0.20.

That uniform Δ is suspicious: it could be a true property of the
generative recipe (one outsider rate, one fragmentation lift) or
an artifact of a single (seed-pinned) draw. This driver answers
that with an interval estimate.

Method — *parametric* bootstrap
-------------------------------
Unlike LoCoMo (real fingerprints, m-out-of-n subsample), the synthetic
corpus IS its population: ``_build_corpus(cell, p)`` is a sampler
from a generative recipe with deterministic ``cell.seed``. Drawing
fresh independent corpora by varying the seed across B draws is the
correct bootstrap of the *generator*, not the realized data.

For each tau and B draws:

* draw a seed ``s_b`` (deterministic from ``(seed, b, tau)``);
* recompute fragmentation at c=0 with seed ``s_b``;
* recompute fragmentation at c=0.10 with seed ``s_b``
  (both calls share the same seed → paired);
* record (f0, f10, Δ).

Reports the same columns as §83: per-tau (mean, sd, ci95_lo,
ci95_hi, p_below_zero, p_below_lift, ci_positive, ci_above_lift)
plus ``f0_all`` / ``f10_all`` so §84 (max-margin fmax) can replay.

Compute budget
--------------
~0.25s per (c=0, c=0.10) pair on the §82 cell (800 fps). B=200
× 8 taus ≈ 6 min, well inside cron budget. Default narrowed to
the GATEABLE-debiased interior (taus 0.10..0.30).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from evals.io_utils import atomic_write_json
from evals.schema_fragmentation_per_tau_calibration import Cell, _build_corpus
from engram.consolidation.schema_family import cluster as cluster_fn
from engram.consolidation.schema_family_contamination import fragmentation_rate


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interp percentile (q in [0, 1]); pure, no numpy dep."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _delta_for_seed(cell: Cell, seed: int) -> tuple[float, float, float]:
    """One paired (f0, f10, Δ) draw: same seed for both contamination
    levels so the only varying factor is true-c."""
    cell_b = Cell(
        n_clusters=cell.n_clusters,
        cluster_size=cell.cluster_size,
        vocab_size=cell.vocab_size,
        core_size=cell.core_size,
        schema_size=cell.schema_size,
        tau=cell.tau,
        seed=seed,
    )
    feats0 = _build_corpus(cell_b, 0.0)
    cls0 = cluster_fn(feats0, tau=cell.tau)
    f0 = fragmentation_rate(feats0, cls0)
    feats10 = _build_corpus(cell_b, 0.10)
    cls10 = cluster_fn(feats10, tau=cell.tau)
    f10 = fragmentation_rate(feats10, cls10)
    return f0, f10, f10 - f0


def evaluate_tau(
    base_cell: Cell,
    tau: float,
    n_boot: int,
    seed: int,
) -> dict:
    cell_t = Cell(
        n_clusters=base_cell.n_clusters,
        cluster_size=base_cell.cluster_size,
        vocab_size=base_cell.vocab_size,
        core_size=base_cell.core_size,
        schema_size=base_cell.schema_size,
        tau=tau,
        seed=base_cell.seed,
    )
    rng = random.Random((seed * 1_000_003) ^ int(round(tau * 1e9)))
    deltas: list[float] = []
    f0s: list[float] = []
    f10s: list[float] = []
    for _ in range(n_boot):
        s_b = rng.randint(1, 2**31 - 1)
        f0, f10, d = _delta_for_seed(cell_t, s_b)
        f0s.append(f0)
        f10s.append(f10)
        deltas.append(d)
    mean = statistics.fmean(deltas)
    sd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    p025 = _percentile(deltas, 0.025)
    p975 = _percentile(deltas, 0.975)
    p_below_zero = sum(1 for d in deltas if d <= 0) / len(deltas)
    p_below_lift = sum(1 for d in deltas if d < 0.05) / len(deltas)
    return {
        "tau": tau,
        "n_boot": n_boot,
        "mean": mean,
        "sd": sd,
        "ci95_lo": p025,
        "ci95_hi": p975,
        "p_below_zero": p_below_zero,
        "p_below_lift": p_below_lift,
        "ci_positive": p025 > 0.0,
        "ci_above_lift": p025 > 0.05,
        "deltas_head": deltas[:50],
        "f0_head": f0s[:50],
        "f10_head": f10s[:50],
        "f0_all": f0s,
        "f10_all": f10s,
    }


def run(
    taus: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30),
    n_boot: int = 200,
    seed: int = 0xCA11B,
    cell: Cell | None = None,
) -> dict:
    """Pure given (taus, n_boot, seed, cell)."""
    if cell is None:
        cell = Cell(
            n_clusters=200,
            cluster_size=4,
            vocab_size=2000,
            core_size=8,
            schema_size=6,
            tau=0.15,  # placeholder — overridden per row
        )
    rows = [evaluate_tau(cell, tau, n_boot, seed) for tau in taus]
    return {
        "corpus": "synthetic_disjoint_core",
        "cell": {
            "n_clusters": cell.n_clusters,
            "cluster_size": cell.cluster_size,
            "vocab_size": cell.vocab_size,
            "core_size": cell.core_size,
            "schema_size": cell.schema_size,
        },
        "n_schemas": cell.n_clusters * cell.cluster_size,
        "n_boot": n_boot,
        "seed": seed,
        "by_tau": rows,
        "summary": {
            "ci_positive_taus": [r["tau"] for r in rows if r["ci_positive"]],
            "ci_above_lift_taus": [r["tau"] for r in rows if r["ci_above_lift"]],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/synthetic_fragmentation_per_tau_bootstrap.json",
    )
    p.add_argument("--taus", default="0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0xCA11B)
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(","))
    res = run(taus=taus, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    print(json.dumps(res["summary"], indent=2))
    print()
    print(
        f"{'tau':>6} {'mean':>8} {'sd':>8} {'ci95_lo':>9} {'ci95_hi':>9} "
        f"{'p≤0':>6} {'p<.05':>6} {'>0?':>5} {'>.05?':>6}"
    )
    for r in res["by_tau"]:
        print(
            f"{r['tau']:>6.2f} "
            f"{r['mean']:>8.4f} "
            f"{r['sd']:>8.4f} "
            f"{r['ci95_lo']:>9.4f} "
            f"{r['ci95_hi']:>9.4f} "
            f"{r['p_below_zero']:>6.3f} "
            f"{r['p_below_lift']:>6.3f} "
            f"{str(r['ci_positive']):>5} "
            f"{str(r['ci_above_lift']):>6}"
        )


if __name__ == "__main__":
    main()

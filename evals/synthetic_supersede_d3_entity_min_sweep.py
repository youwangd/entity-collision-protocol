"""§D3-collateral-(d) — entity_min threshold sweep on supersede corpus.

Now that ``interference_entity_aware`` is the default (§D3-collateral-(c)),
the remaining knob is ``interference_entity_overlap_min`` — the Jaccard
threshold on entity-token overlap that gates a candidate (older,newer)
pair from firing the supersede transition.

Question
--------
At what value of ``entity_min`` does the entity-aware detector best
balance true-positive supersede capture (Δhit@1>0, Δstale@1<0) against
cross-slot collateral (keep Δhit@k≥0)?

  * entity_min=0.0 ⇒ degenerate, equivalent to entity_aware OFF
    (Jaccard-only, susceptible to template overlap collateral).
  * entity_min=1.0 ⇒ degenerate, requires identical entity sets,
    likely too strict to fire on any natural rephrase.
  * §D3-collateral-(b) probed 0.5 (default) on n_slots=200 with the
    full +9pp Δhit@1, 0 Δhit@k result.
  * §D3-collateral-(c) probed 0.7 on full LoCoMo10 — null, but corpus
    is inert.

This driver sweeps {0.3, 0.5, 0.7, 0.9} on the planted supersede
corpus (n_slots=200, updates_per_slot=2, distractors=100) and reports
all four headline deltas with 95% paired-bootstrap CI per point.

Wall budget: ~7s × 4 = ~30s on 1 CPU.

Usage
-----
    python -m evals.synthetic_supersede_d3_entity_min_sweep \\
        --entity-min-list 0.3,0.5,0.7,0.9 \\
        --n-slots 200 --distractors 100 \\
        --resamples 10000 \\
        --out bench/results/d3_entity_min_sweep.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from evals.synthetic_supersede_d3_real import run_d3_real
from evals.io_utils import atomic_write_json


def _fmt_ci(s: dict) -> str:
    return (f"{s['mean_diff_default_minus_addonly']:+.3f} "
            f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] "
            f"p={s['p_bootstrap_two_sided']:.3g}")


def run_sweep(
    *,
    entity_min_list: list[float],
    n_slots: int = 200,
    updates_per_slot: int = 2,
    distractors: int = 100,
    seed: int = 42,
    k: int = 10,
    resamples: int = 10000,
    boot_seed: int = 42,
) -> dict:
    t0 = time.monotonic()
    points: list[dict] = []
    for em in entity_min_list:
        rep = run_d3_real(
            n_slots=n_slots,
            updates_per_slot=updates_per_slot,
            distractors=distractors,
            seed=seed,
            k=k,
            resamples=resamples,
            boot_seed=boot_seed,
            entity_aware=True,
            entity_min=em,
        )
        a = rep["arms"]["default"]
        b = rep["arms"]["addonly"]
        points.append({
            "entity_min": em,
            "n_queries": rep["corpus"]["n_queries"],
            "n_memories": rep["corpus"]["n_memories"],
            "wall_seconds": rep["wall_seconds"],
            "default_interference_actions": a["interference_actions"],
            "default_hit_at_1": a["hit_at_1"],
            "default_hit_at_k": a["hit_at_k"],
            "default_stale_at_1": a["stale_at_1"],
            "addonly_hit_at_1": b["hit_at_1"],
            "addonly_hit_at_k": b["hit_at_k"],
            "addonly_stale_at_1": b["stale_at_1"],
            "delta": rep["summary"],
        })

    return {
        "config": {
            "entity_min_list": entity_min_list,
            "n_slots": n_slots,
            "updates_per_slot": updates_per_slot,
            "distractors": distractors,
            "seed": seed,
            "k": k,
            "resamples": resamples,
            "boot_seed": boot_seed,
        },
        "points": points,
        "wall_seconds_total": round(time.monotonic() - t0, 2),
    }


def _print_table(rep: dict) -> None:
    cfg = rep["config"]
    print(f"§D3-collateral-(d) entity_min sweep — n_slots={cfg['n_slots']} "
          f"updates={cfg['updates_per_slot']} distractors={cfg['distractors']} "
          f"k={cfg['k']} resamples={cfg['resamples']}")
    print(f"{'entity_min':>10} {'IFAct':>6} | {'Δhit@1':>22} | "
          f"{'Δhit@k':>22} | {'Δstale@1':>22}")
    for p in rep["points"]:
        d = p["delta"]
        print(f"{p['entity_min']:>10.2f} "
              f"{p['default_interference_actions']:>6d} | "
              f"{_fmt_ci(d['d_hit_at_1']):>22} | "
              f"{_fmt_ci(d['d_hit_at_k']):>22} | "
              f"{_fmt_ci(d['d_stale_at_1']):>22}")
    print(f"  total_wall={rep['wall_seconds_total']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity-min-list", type=str, default="0.3,0.5,0.7,0.9",
                    help="comma-separated list of entity_min thresholds")
    ap.add_argument("--n-slots", type=int, default=200)
    ap.add_argument("--updates-per-slot", type=int, default=2)
    ap.add_argument("--distractors", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    em_list = [float(x) for x in args.entity_min_list.split(",") if x]
    rep = run_sweep(
        entity_min_list=em_list,
        n_slots=args.n_slots,
        updates_per_slot=args.updates_per_slot,
        distractors=args.distractors,
        seed=args.seed,
        k=args.k,
        resamples=args.resamples,
        boot_seed=args.boot_seed,
    )
    _print_table(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[D3-collateral-(d)] wrote {args.out}")


if __name__ == "__main__":
    main()

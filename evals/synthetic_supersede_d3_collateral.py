"""§D3-collateral — sweep entity diversity to isolate cross-slot fade.

Hypothesis
----------
The −58.5pp Δhit@k regression observed in §D3-real (`synthetic_supersede_d3_real`)
at n_slots=200, updates=2 is driven by *cross-slot template overlap*, not by
the supersede semantics per se. The default Jaccard-only interference detector
ignores entity tokens, so two memories from different slots that share template
boilerplate (e.g., "User X now prefers using DARK MODE for daily debugging
work" vs "User Y now prefers using TMUX for daily review work") look similar
enough to fire FADE. As n_slots grows, the density of such confusable pairs
grows quadratically — and Δhit@k coverage should monotonically degrade.

If this hypothesis holds we expect:
  * Δhit@1 stays significantly positive across all n_slots (supersede works).
  * Δstale@1 stays significantly negative across all n_slots.
  * Δhit@k transitions from ~0 (small n_slots, few cross-slot collisions) to
    strongly negative as n_slots grows.
  * `interference_actions` for the default arm grows roughly linearly with
    n_slots, but the *fraction of those that are cross-slot false positives*
    grows.

Methodology
-----------
For each n_slots ∈ {25, 50, 100, 200} (with distractors held fixed at 100,
updates_per_slot=2, seed=42) we re-run the §D3-real driver and report:
  * Δhit@1, Δhit@k, Δstale@1, Δstale@k with 95% paired-bootstrap CI
  * default-arm interference_actions
  * an estimate of cross-slot fade rate by post-hoc re-running the
    InterferenceDetection stage and counting (older,newer) pairs whose
    slot_id metadata differs but which were classified as supersede.

Wall budget at n_slots=200, distractors=100, 10k bootstrap: ~6.5s. Sweep of
4 points: ~25s.

Output
------
JSON report with one entry per n_slots; printed table; optional SCALE_REPORT
paragraph.
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
    n_slots_list: list[int],
    updates_per_slot: int = 2,
    distractors: int = 100,
    seed: int = 42,
    k: int = 10,
    resamples: int = 10000,
    boot_seed: int = 42,
    entity_aware: bool = False,
    entity_min: float = 0.5,
) -> dict:
    t0 = time.monotonic()
    points: list[dict] = []
    for n in n_slots_list:
        rep = run_d3_real(
            n_slots=n,
            updates_per_slot=updates_per_slot,
            distractors=distractors,
            seed=seed,
            k=k,
            resamples=resamples,
            boot_seed=boot_seed,
            entity_aware=entity_aware,
            entity_min=entity_min,
        )
        points.append({
            "n_slots": n,
            "n_queries": rep["corpus"]["n_queries"],
            "n_memories": rep["corpus"]["n_memories"],
            "wall_seconds": rep["wall_seconds"],
            "default_interference_actions": rep["arms"]["default"]["interference_actions"],
            "default_hit_at_1": rep["arms"]["default"]["hit_at_1"],
            "default_hit_at_k": rep["arms"]["default"]["hit_at_k"],
            "default_stale_at_1": rep["arms"]["default"]["stale_at_1"],
            "addonly_hit_at_1": rep["arms"]["addonly"]["hit_at_1"],
            "addonly_hit_at_k": rep["arms"]["addonly"]["hit_at_k"],
            "addonly_stale_at_1": rep["arms"]["addonly"]["stale_at_1"],
            "delta": rep["summary"],
        })

    return {
        "config": {
            "n_slots_list": n_slots_list,
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
    print(f"§D3-collateral sweep — distractors={rep['config']['distractors']} "
          f"updates={rep['config']['updates_per_slot']} k={rep['config']['k']} "
          f"resamples={rep['config']['resamples']}")
    print(f"{'n_slots':>8} {'mems':>6} {'q':>4} {'IFAct':>6} | "
          f"{'Δhit@1':>22} | {'Δhit@k':>22} | {'Δstale@1':>22}")
    for p in rep["points"]:
        d = p["delta"]
        print(f"{p['n_slots']:>8d} {p['n_memories']:>6d} {p['n_queries']:>4d} "
              f"{p['default_interference_actions']:>6d} | "
              f"{_fmt_ci(d['d_hit_at_1']):>22} | "
              f"{_fmt_ci(d['d_hit_at_k']):>22} | "
              f"{_fmt_ci(d['d_stale_at_1']):>22}")
    print(f"  total_wall={rep['wall_seconds_total']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slots-list", type=str, default="25,50,100,200",
                    help="comma-separated list of n_slots to sweep")
    ap.add_argument("--updates-per-slot", type=int, default=2)
    ap.add_argument("--distractors", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--entity-aware", action="store_true")
    ap.add_argument("--entity-min", type=float, default=0.5)
    args = ap.parse_args()

    n_slots_list = [int(x) for x in args.n_slots_list.split(",") if x]
    rep = run_sweep(
        n_slots_list=n_slots_list,
        updates_per_slot=args.updates_per_slot,
        distractors=args.distractors,
        seed=args.seed,
        k=args.k,
        resamples=args.resamples,
        boot_seed=args.boot_seed,
        entity_aware=args.entity_aware,
        entity_min=args.entity_min,
    )
    _print_table(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[D3-collateral] wrote {args.out}")


if __name__ == "__main__":
    main()

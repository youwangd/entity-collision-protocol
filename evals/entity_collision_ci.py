"""Paired bootstrap CIs for an entity_collision_sweep JSON.

The sweep stores per-query records under ``bm25_only.per_query`` and
``vector_fusion.per_query``. This driver consumes those records and emits
paired Δ-CIs (vector_fusion − bm25_only) per collision degree on
``hit_at_1``, ``hit_at_k``, and ``mrr``, plus marginal CIs on each arm.

Usage::

    python -m evals.entity_collision_ci \\
        --in bench/results/ec_sweep_st_service_n16.json \\
        --out bench/results/ec_sweep_st_service_n16_ci.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap_ci import _bootstrap_mean_ci, _paired_diff_ci
from evals.io_utils import atomic_write_json


_METRIC_KEYS = {"hit_at_1": "hit_at_1", "hit_at_k": "hit_at_k", "mrr": "reciprocal_rank"}


def _summarize(pq: list[dict], resamples: int, seed: int) -> dict:
    out: dict[str, dict] = {"n": len(pq)}
    for metric, key in _METRIC_KEYS.items():
        vals = [r[key] for r in pq]
        m, lo, hi = _bootstrap_mean_ci(vals, resamples, seed)
        out[metric] = {"mean": round(m, 4), "ci_lo": round(lo, 4),
                       "ci_hi": round(hi, 4)}
    return out


def _paired(a_pq: list[dict], b_pq: list[dict], resamples: int,
            seed: int) -> dict:
    assert len(a_pq) == len(b_pq), f"length mismatch {len(a_pq)} vs {len(b_pq)}"
    out: dict[str, dict] = {"n": len(a_pq)}
    for metric, key in _METRIC_KEYS.items():
        a = [r[key] for r in a_pq]
        b = [r[key] for r in b_pq]
        m, lo, hi = _paired_diff_ci(a, b, resamples, seed)
        out[metric] = {"mean": round(m, 4), "ci_lo": round(lo, 4),
                       "ci_hi": round(hi, 4)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    rows = data.get("sweep") or []
    if not rows:
        raise SystemExit("no sweep rows in input")

    augmented = []
    print(f"{'K':>3} | {'n':>4} | {'bm25 hit@1 [CI]':>26} | "
          f"{'vec hit@1 [CI]':>26} | {'Δhit@1 [CI]':>26}")
    print("-" * 100)
    for row in rows:
        K = row["collision_degree"]
        bm_pq = row["bm25_only"]["per_query"]
        vec_pq = row["vector_fusion"]["per_query"]
        bm_sum = _summarize(bm_pq, args.resamples, args.seed)
        vec_sum = _summarize(vec_pq, args.resamples, args.seed)
        diff = _paired(vec_pq, bm_pq, args.resamples, args.seed)

        def fmt(c):
            return f"{c['mean']:.3f} [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]"

        print(f"{K:>3} | {bm_sum['n']:>4} | {fmt(bm_sum['hit_at_1']):>26} | "
              f"{fmt(vec_sum['hit_at_1']):>26} | "
              f"{fmt(diff['hit_at_1']):>26}")

        out_row = {k: v for k, v in row.items()
                   if k not in ("bm25_only", "vector_fusion")}
        out_row["bm25_only_summary"] = bm_sum
        out_row["vector_fusion_summary"] = vec_sum
        out_row["delta_ci"] = diff
        augmented.append(out_row)

    out_obj = {
        "config": data.get("config"),
        "ci_config": {"resamples": args.resamples, "seed": args.seed},
        "rows": augmented,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out_obj)
        print(f"\n[entity_collision_ci] wrote {args.out}")


if __name__ == "__main__":
    main()

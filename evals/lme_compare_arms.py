"""Pair-bootstrap LongMemEval treatment arms vs the baseline.

Reads adapter JSON outputs (each must carry the per_instance vector
emitted by `evals.longmemeval_adapter` after the §4.8.1 treatment
landing) and computes paired Δhit@1 / Δhit@k 95% bootstrap CIs against
a chosen baseline run, both overall and sliced by question_type.

Usage:
    python -m evals.lme_compare_arms \
        --baseline bench/results/lme_full500_k10.json \
        --arms     bench/results/lme_full500_k10_prf.json \
                   bench/results/lme_full500_k10_sp.json \
                   bench/results/lme_full500_k10_both.json \
        --out      bench/results/lme_full500_arms_delta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.bootstrap_ci import _paired_diff_ci
from evals.io_utils import atomic_write_text


def _index(rows: list[dict]) -> dict[str, dict]:
    return {r["question_id"]: r for r in rows if r.get("question_id")}


def _paired_arrays(base: list[dict], arm: list[dict]) -> tuple[list[dict], list[dict]]:
    bidx = _index(base)
    aidx = _index(arm)
    keys = [k for k in bidx if k in aidx]
    keys.sort()
    return [bidx[k] for k in keys], [aidx[k] for k in keys]


def _delta_block(base_rows: list[dict], arm_rows: list[dict], resamples: int, seed: int) -> dict:
    if not base_rows:
        return {"n": 0}
    h1_b = [r["hit_at_1"] for r in base_rows]
    h1_a = [r["hit_at_1"] for r in arm_rows]
    hk_b = [r["hit_at_k"] for r in base_rows]
    hk_a = [r["hit_at_k"] for r in arm_rows]
    d_h1 = _paired_diff_ci(h1_a, h1_b, resamples, seed)
    d_hk = _paired_diff_ci(hk_a, hk_b, resamples, seed)
    return {
        "n": len(base_rows),
        "baseline_hit_at_1": round(sum(h1_b) / len(h1_b), 4),
        "arm_hit_at_1": round(sum(h1_a) / len(h1_a), 4),
        "delta_hit_at_1": {
            "mean": round(d_h1[0], 4),
            "ci_lo": round(d_h1[1], 4),
            "ci_hi": round(d_h1[2], 4),
        },
        "baseline_hit_at_k": round(sum(hk_b) / len(hk_b), 4),
        "arm_hit_at_k": round(sum(hk_a) / len(hk_a), 4),
        "delta_hit_at_k": {
            "mean": round(d_hk[0], 4),
            "ci_lo": round(d_hk[1], 4),
            "ci_hi": round(d_hk[2], 4),
        },
    }


def compare(base_path: Path, arm_paths: list[Path], resamples: int, seed: int) -> dict:
    base = json.loads(Path(base_path).read_text())
    base_rows = base["per_instance"]
    out = {"baseline": str(base_path), "n_baseline": len(base_rows), "arms": {}}
    for ap in arm_paths:
        arm = json.loads(Path(ap).read_text())
        arm_rows_full = arm["per_instance"]
        b_paired, a_paired = _paired_arrays(base_rows, arm_rows_full)
        block = {
            "path": str(ap),
            "arm_label": arm.get("arm"),
            "arm_config": arm.get("arm_config"),
            "n_paired": len(b_paired),
            "overall": _delta_block(b_paired, a_paired, resamples, seed),
            "per_type": {},
        }
        types = sorted({r["question_type"] for r in b_paired})
        for qt in types:
            br = [r for r in b_paired if r["question_type"] == qt]
            ar = [r for r in a_paired if r["question_type"] == qt]
            block["per_type"][qt] = _delta_block(br, ar, resamples, seed)
        out["arms"][Path(ap).stem] = block
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--arms", required=True, nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    result = compare(args.baseline, args.arms, args.resamples, args.seed)
    txt = json.dumps(result, indent=2)
    print(txt)
    if args.out:
        atomic_write_text(args.out, txt)


if __name__ == "__main__":
    main()

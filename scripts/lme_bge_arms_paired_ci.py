#!/usr/bin/env python3
"""§A.4.8.1 third-column — Paired CI within BGE-large for treatment arms
(PRF, both) vs the BGE-large baseline arm on LongMemEval-S n=500.

Companion to scripts/lme_bge_vs_minilm_n500_paired_ci.py. Different
question: not "BGE vs MiniLM", but "PRF/both vs baseline, all under BGE
encoder". Mirrors the schema of bench/results/lme_full500_arms_delta.json
(the existing MiniLM within-embedder treatment-effect file the cron
diffs against).

Inputs (default — override with --baseline / --arms):
  bench/results/lme_n500_bge_large_baseline.json   (off-PRF, off-SP)
  bench/results/lme_n500_bge_large_prf.json        (PRF on)
  bench/results/lme_n500_bge_large_both.json       (PRF + share_prior)

Output: bench/results/lme_n500_bge_large_arms_delta.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def boot_ci(diffs: list[float], n: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means: list[float] = []
    L = len(diffs)
    if L == 0:
        return (0.0, 0.0, 0.0)
    for _ in range(n):
        sample = [diffs[rng.randrange(L)] for _ in range(L)]
        means.append(sum(sample) / L)
    means.sort()
    return (
        sum(diffs) / L,
        means[int(0.025 * n)],
        means[int(0.975 * n)],
    )


def _delta_block(arm_inst: list[dict], base_inst: list[dict],
                 boot_n: int, seed: int) -> dict:
    base_idx = {x["question_id"]: x for x in base_inst}
    paired = [
        (a, base_idx[a["question_id"]])
        for a in arm_inst if a["question_id"] in base_idx
    ]
    block: dict = {"n_paired": len(paired), "overall": {}, "per_type": {}}
    for metric in ("hit_at_1", "hit_at_k"):
        diffs = [a[metric] - b[metric] for a, b in paired]
        mean, lo, hi = boot_ci(diffs, n=boot_n, seed=seed)
        block["overall"][metric] = {
            "arm_rate": sum(a[metric] for a, _ in paired) / max(len(paired), 1),
            "baseline_rate": sum(b[metric] for _, b in paired) / max(len(paired), 1),
            "delta": mean,
            "ci95": [lo, hi],
            "sig": (lo > 0) or (hi < 0),
        }
    by_type: dict[str, list] = defaultdict(list)
    for a, b in paired:
        by_type[a["question_type"]].append((a, b))
    for qtype, pairs in by_type.items():
        cell: dict = {"n": len(pairs)}
        for metric in ("hit_at_1", "hit_at_k"):
            diffs = [a[metric] - b[metric] for a, b in pairs]
            mean, lo, hi = boot_ci(diffs, n=boot_n, seed=seed)
            cell[metric] = {
                "arm_rate": sum(a[metric] for a, _ in pairs) / max(len(pairs), 1),
                "baseline_rate": sum(b[metric] for _, b in pairs) / max(len(pairs), 1),
                "delta": mean,
                "ci95": [lo, hi],
                "sig": (lo > 0) or (hi < 0),
            }
        block["per_type"][qtype] = cell
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="BGE-large baseline arm result JSON")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="One or more BGE-large treatment-arm result JSONs")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text())
    out: dict = {
        "baseline_artifact": args.baseline,
        "arm_artifacts": list(args.arms),
        "boot_resamples": args.boot,
        "boot_seed": args.seed,
        "delta": {},
    }
    for arm_path in args.arms:
        arm = json.loads(Path(arm_path).read_text())
        # Use the arm field if present, else fall back to filename stem suffix.
        arm_name = arm.get("arm")
        if not arm_name or arm_name == "baseline":
            stem = Path(arm_path).stem
            arm_name = stem.split("_")[-1]
        out["delta"][arm_name] = _delta_block(
            arm["per_instance"], base["per_instance"], args.boot, args.seed
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[lme-bge-arms] wrote {args.out}")
    # Echo overall deltas
    for arm_name, block in out["delta"].items():
        for metric, m in block["overall"].items():
            sig = "SIG" if m["sig"] else "null"
            ci = m["ci95"]
            print(f"  {arm_name:>5}  {metric}: arm={m['arm_rate']:.3f} base={m['baseline_rate']:.3f} "
                  f"Δ={m['delta']:+.3f} 95%=[{ci[0]:+.3f},{ci[1]:+.3f}] {sig}")


if __name__ == "__main__":
    main()

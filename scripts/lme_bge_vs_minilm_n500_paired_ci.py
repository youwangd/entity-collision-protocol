#!/usr/bin/env python3
"""§A.4.16.3 v0.3 follow-up — Paired BGE-large vs default-MiniLM CI on the
full LongMemEval-S n=500.

Companion to scripts/lme_bge_vs_minilm_n100_paired_ci.py — same logic,
different inputs:

  bench/results/lme_n500_bge_large_baseline.json  (BGE-large, n=500)
  bench/results/lme_full500_k10_baseline.json     (MiniLM, n=500)

Output: bench/results/lme_n500_bge_vs_default_ci.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def boot_ci(diffs: list[float], n: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means: list[float] = []
    L = len(diffs)
    for _ in range(n):
        sample = [diffs[rng.randrange(L)] for _ in range(L)]
        means.append(sum(sample) / L)
    means.sort()
    return (
        sum(diffs) / L,
        means[int(0.025 * n)],
        means[int(0.975 * n)],
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bge = json.loads((root / "bench/results/lme_n500_bge_large_baseline.json").read_text())
    ml = json.loads((root / "bench/results/lme_full500_k10_baseline.json").read_text())

    ml_idx = {x["question_id"]: x for x in ml["per_instance"]}
    paired: list[tuple[dict, dict]] = []
    for b in bge["per_instance"]:
        m = ml_idx.get(b["question_id"])
        if m is not None:
            paired.append((b, m))

    out: dict = {
        "n_paired": len(paired),
        "bge_artifact": "bench/results/lme_n500_bge_large_baseline.json",
        "default_artifact": "bench/results/lme_full500_k10_baseline.json",
        "boot_resamples": 10000,
        "boot_seed": 42,
        "overall": {},
        "per_type": {},
    }
    for metric in ("hit_at_1", "hit_at_k"):
        diffs = [bp[metric] - mp[metric] for bp, mp in paired]
        mean, lo, hi = boot_ci(diffs)
        out["overall"][metric] = {
            "bge_rate": sum(bp[metric] for bp, _ in paired) / len(paired),
            "default_rate": sum(mp[metric] for _, mp in paired) / len(paired),
            "delta": mean,
            "ci95": [lo, hi],
            "sig": (lo > 0) or (hi < 0),
        }

    by_type: dict[str, list] = defaultdict(list)
    for bp, mp in paired:
        by_type[bp["question_type"]].append((bp, mp))
    for qtype, pairs in by_type.items():
        cell: dict = {"n": len(pairs)}
        for metric in ("hit_at_1", "hit_at_k"):
            diffs = [bp[metric] - mp[metric] for bp, mp in pairs]
            mean, lo, hi = boot_ci(diffs)
            cell[metric] = {
                "bge_rate": sum(bp[metric] for bp, _ in pairs) / len(pairs),
                "default_rate": sum(mp[metric] for _, mp in pairs) / len(pairs),
                "delta": mean,
                "ci95": [lo, hi],
                "sig": (lo > 0) or (hi < 0),
            }
        out["per_type"][qtype] = cell

    out_path = root / "bench/results/lme_n500_bge_vs_default_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[lme-bge-ci] wrote {out_path.relative_to(root)}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

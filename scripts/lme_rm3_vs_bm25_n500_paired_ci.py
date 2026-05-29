#!/usr/bin/env python3
"""§A.4.16 AUDIT-D — Paired RM3 vs BM25 baseline CI on the full LongMemEval-S n=500.

Companion to scripts/lme_bge_vs_minilm_n500_paired_ci.py — same logic,
different inputs:

  bench/results/lme_n500_rm3.json              (RM3 PRF over BM25, n=500)
  bench/results/lme_full500_k10_baseline.json  (BM25 baseline, n=500)

Output: bench/results/lme_n500_rm3_vs_bm25_ci.json

Both artifacts share the BM25-only retrieval channel (embed=None);
the only treatment difference is the two-pass RM3 query-expansion dance.
Paired by question_id; non-overlapping question_ids are dropped.
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


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    rm3 = json.loads((root / "bench/results/lme_n500_rm3.json").read_text())
    bm25 = json.loads((root / "bench/results/lme_full500_k10_baseline.json").read_text())

    bm25_idx = {x["question_id"]: x for x in bm25["per_instance"]}
    paired: list[tuple[dict, dict]] = []
    for r in rm3["per_instance"]:
        b = bm25_idx.get(r["question_id"])
        if b is not None:
            paired.append((r, b))

    out: dict = {
        "n_paired": len(paired),
        "n_rm3": len(rm3["per_instance"]),
        "n_bm25": len(bm25["per_instance"]),
        "rm3_artifact": "bench/results/lme_n500_rm3.json",
        "bm25_artifact": "bench/results/lme_full500_k10_baseline.json",
        "boot_resamples": 10000,
        "boot_seed": 42,
        "rm3_config": rm3.get("rm3_config", {}),
        "overall": {},
        "per_type": {},
    }
    for metric in ("hit_at_1", "hit_at_k"):
        diffs = [rp[metric] - bp[metric] for rp, bp in paired]
        mean, lo, hi = boot_ci(diffs)
        out["overall"][metric] = {
            "rm3_rate": sum(rp[metric] for rp, _ in paired) / len(paired),
            "bm25_rate": sum(bp[metric] for _, bp in paired) / len(paired),
            "delta": mean,
            "ci95": [lo, hi],
            "sig": (lo > 0) or (hi < 0),
        }

    by_type: dict[str, list] = defaultdict(list)
    for rp, bp in paired:
        by_type[rp["question_type"]].append((rp, bp))
    for qtype, pairs in by_type.items():
        cell: dict = {"n": len(pairs)}
        for metric in ("hit_at_1", "hit_at_k"):
            diffs = [rp[metric] - bp[metric] for rp, bp in pairs]
            mean, lo, hi = boot_ci(diffs)
            cell[metric] = {
                "rm3_rate": sum(rp[metric] for rp, _ in pairs) / len(pairs),
                "bm25_rate": sum(bp[metric] for _, bp in pairs) / len(pairs),
                "delta": mean,
                "ci95": [lo, hi],
                "sig": (lo > 0) or (hi < 0),
            }
        out["per_type"][qtype] = cell

    out_path = root / "bench/results/lme_n500_rm3_vs_bm25_ci.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"\nn_paired = {out['n_paired']}")
    print(f"\nOverall:")
    for metric, cell in out["overall"].items():
        sig = "SIG" if cell["sig"] else "n.s."
        print(f"  {metric}: rm3={cell['rm3_rate']:.4f} bm25={cell['bm25_rate']:.4f} "
              f"delta={cell['delta']:+.4f} CI=[{cell['ci95'][0]:+.4f}, {cell['ci95'][1]:+.4f}] {sig}")
    print(f"\nPer-type:")
    for qtype, cell in sorted(out["per_type"].items()):
        h1 = cell["hit_at_1"]
        hk = cell["hit_at_k"]
        sig1 = "SIG" if h1["sig"] else "n.s."
        sigk = "SIG" if hk["sig"] else "n.s."
        print(f"  {qtype:30s} n={cell['n']:>3}  "
              f"h1: {h1['delta']:+.4f} [{h1['ci95'][0]:+.4f}, {h1['ci95'][1]:+.4f}] {sig1}  "
              f"hk: {hk['delta']:+.4f} [{hk['ci95'][0]:+.4f}, {hk['ci95'][1]:+.4f}] {sigk}")


if __name__ == "__main__":
    main()

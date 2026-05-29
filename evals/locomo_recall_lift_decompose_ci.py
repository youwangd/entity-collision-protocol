"""§94c-decompose-CI — bootstrap CI on the (S_a − S_b) paired delta.

§94c-decompose showed S1=+0.0831 vs S7=+0.0764 on Δh@1 — a 0.7pp slip
that is well within the §94c-CI ±4pp half-width. This driver answers
the natural question: *is the slip a real regression, or noise?*

Method. Re-run `evals.locomo_recall_lift` under stage_a (default
S1=extraction-only) and stage_b (default S7=full pipeline) on the same
fixture, pair on (sample_id, question), compute per-pair
(Δ_a − Δ_b), and percentile-bootstrap the mean. If the 95% CI on the
mean of (Δ_a − Δ_b) brackets zero, the downstream stages neither help
nor hurt at the §94c-decompose configuration.

Usage:
    python -m evals.locomo_recall_lift_decompose_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§94c_decompose_ci.json
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
from pathlib import Path

from evals.locomo_recall_lift import run_recall_lift
from evals.io_utils import atomic_write_json


# Same stage definitions as locomo_recall_lift_decompose.DEFAULT_SUBSETS.
SUBSET_PRESETS: dict[str, list[str] | None] = {
    "S1_extraction_only": ["extraction"],
    "S2_+fact":           ["extraction", "fact_extraction"],
    "S3_+interference":   ["extraction", "fact_extraction", "interference"],
    "S4_+schema_update":  ["extraction", "fact_extraction", "interference",
                           "schema_update"],
    "S5_+somatic":        ["extraction", "fact_extraction", "interference",
                           "schema_update", "somatic_marking"],
    "S6_+merge_persist":  ["extraction", "fact_extraction", "interference",
                           "schema_update", "somatic_marking",
                           "mechanical_merge"],
    "S7_full_default":    None,
}


def _bootstrap_mean_ci(values, resamples, seed, alpha=0.05):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * resamples))
    hi_idx = min(int(math.ceil((1 - alpha / 2) * resamples)) - 1,
                 resamples - 1)
    # Two-sided percentile bootstrap p-value for H0: mean=0.
    leq = sum(1 for m in means if m <= 0)
    geq = sum(1 for m in means if m >= 0)
    p = min(1.0, 2 * min(leq, geq) / resamples)
    return statistics.fmean(values), means[lo_idx], means[hi_idx], p


def run_decompose_ci(
    dataset_path: str,
    *,
    stage_a: str = "S1_extraction_only",
    stage_b: str = "S7_full_default",
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = False,
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    if stage_a not in SUBSET_PRESETS or stage_b not in SUBSET_PRESETS:
        raise ValueError(f"unknown subset name (use {list(SUBSET_PRESETS)})")

    t0 = time.monotonic()
    res_a = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        stages=SUBSET_PRESETS[stage_a],
    )
    res_b = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        stages=SUBSET_PRESETS[stage_b],
    )

    pairs_a = res_a.get("per_query_pairs") or []
    pairs_b = res_b.get("per_query_pairs") or []
    bkey = lambda r: (r["sample_id"], r.get("question"), r["category"])
    # Pair on (sample_id, question, category) — both arms read the same
    # dataset deterministically, so this aligns 1:1.
    bmap = {}
    for r in pairs_b:
        bmap.setdefault(bkey(r), []).append(r)
    diffs = {"delta_h1": [], "delta_hk": [], "delta_rr": [],
             "delta_prk": [], "delta_grk": []}
    paired = 0
    for ra in pairs_a:
        bucket = bmap.get(bkey(ra))
        if not bucket:
            continue
        rb = bucket.pop(0)
        paired += 1
        for key in diffs:
            diffs[key].append(float(ra[key]) - float(rb[key]))

    summary = {}
    for key, vals in diffs.items():
        m, lo, hi, p = _bootstrap_mean_ci(vals, resamples, seed)
        summary[key] = {
            "mean_diff_a_minus_b": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_bootstrap_two_sided": round(p, 6),
            "n_paired": len(vals),
        }

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "synthesis": synthesis,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stages_a": SUBSET_PRESETS[stage_a],
        "stages_b": SUBSET_PRESETS[stage_b],
        "n_pairs_a": len(pairs_a),
        "n_pairs_b": len(pairs_b),
        "n_paired": paired,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "summary": summary,
        "headline_a": {
            "delta_h1": res_a["delta"]["session_hit_at_1"],
            "delta_hk": res_a["delta"]["session_hit_at_k"],
            "delta_mrr": res_a["delta"]["mean_reciprocal_rank"],
        },
        "headline_b": {
            "delta_h1": res_b["delta"]["session_hit_at_1"],
            "delta_hk": res_b["delta"]["session_hit_at_k"],
            "delta_mrr": res_b["delta"]["mean_reciprocal_rank"],
        },
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--stage-a", default="S1_extraction_only")
    p.add_argument("--stage-b", default="S7_full_default")
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--synthesis", action="store_true")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rep = run_decompose_ci(
        args.dataset,
        stage_a=args.stage_a,
        stage_b=args.stage_b,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94c-decompose-CI  {args.stage_a} − {args.stage_b}")
    print(f"  n_paired={rep['n_paired']}  resamples={args.resamples}  "
          f"wall={rep['wall_seconds']}s")
    for k_ in ("delta_h1", "delta_hk", "delta_rr",
               "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        print(f"  Δ({k_:>9}): mean={c['mean_diff_a_minus_b']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[decompose-CI] wrote {args.out}")


if __name__ == "__main__":
    main()

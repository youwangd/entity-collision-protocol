"""§D3 — ADD-only ablation, paired-bootstrap CI on (Δ_default − Δ_addonly).

Runs `evals.locomo_recall_lift` twice on the same fixture:
  * arm A: full default consolidation pipeline (supersede ON).
  * arm B: full default pipeline with ``consolidation.add_only=True``
    (interference stage no-ops; mechanical-merge dedup unaffected).

For each (sample_id, question, category) we pair the per-query
deltas-vs-baseline-no-consolidation and bootstrap the mean of
(Δ_a − Δ_b). If the 95% CI brackets zero across all five retrieval
metrics, that is a publishable null result: ADD-only is not worse than
supersede on this corpus.

Usage:
    python -m evals.locomo_recall_lift_add_only_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§D3_add_only_ci.json
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from evals.locomo_recall_lift import run_recall_lift
from evals.locomo_recall_lift_decompose_ci import _bootstrap_mean_ci
from evals.io_utils import atomic_write_json


def run_add_only_ci(
    dataset_path: str,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = False,
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    t0 = time.monotonic()
    res_a = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        add_only=False,
    )
    res_b = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        add_only=True,
    )

    pairs_a = res_a.get("per_query_pairs") or []
    pairs_b = res_b.get("per_query_pairs") or []
    bkey = lambda r: (r["sample_id"], r.get("question"), r["category"])
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
            "mean_diff_default_minus_addonly": round(m, 6),
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
        "n_pairs_default": len(pairs_a),
        "n_pairs_addonly": len(pairs_b),
        "n_paired": paired,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "summary": summary,
        "headline_default": res_a["delta"],
        "headline_addonly": res_b["delta"],
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
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

    rep = run_add_only_ci(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print("§D3-add-only-CI  default − add_only")
    print(f"  n_paired={rep['n_paired']}  resamples={args.resamples}  "
          f"wall={rep['wall_seconds']}s")
    for k_ in ("delta_h1", "delta_hk", "delta_rr",
               "delta_prk", "delta_grk"):
        c = rep["summary"][k_]
        print(f"  Δ({k_:>9}): mean={c['mean_diff_default_minus_addonly']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[add-only-CI] wrote {args.out}")


if __name__ == "__main__":
    main()

"""§94c-appraisal-bound-CI — paired bootstrap CI on (Δ_cap_a − Δ_cap_b).

Motivation
----------
§94c-appraisal-bound (sweep) showed cap=0.30 dominates cap=None on every
primary metric on the LoCoMo10 max_instances=2 fixture (n_pairs=301):

    cap=None : Δh@1 +0.0764, ΔMRR +0.0899, Δgrk −0.0075, Δprk +0.1395
    cap=0.30 : Δh@1 +0.0864, ΔMRR +0.0970, Δgrk +0.0067, Δprk +0.1462

The point estimate gap on Δh@1 is ~+1.0pp. §94c-decompose-CI half-width
on Δh@1 was ±2.0pp at the same n. So we need the *paired* bootstrap on
the per-pair (Δ_cap_a − Δ_cap_b) diff to claim the cap-induced
improvement is real and not within-noise.

Method. Re-run `evals.locomo_recall_lift` under cap_a (default 0.30) and
cap_b (default None) on the same fixture, pair on (sample_id, question,
category), compute per-pair (Δ_cap_a − Δ_cap_b), and percentile-
bootstrap the mean for each of {Δh@1, Δh@k, ΔRR, Δprk, Δgrk}. If the
95% CI on Δh@1 excludes zero on the *positive* side, ship cap=0.30 as
the no-LLM default.

Pattern is a direct copy of `evals.locomo_recall_lift_decompose_ci`
swapping `stages` for `appraisal_salience_cap`.

Usage
-----
    python -m evals.locomo_appraisal_bound_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_appraisal_bound_ci.json \\
        --md-out bench/results/locomo_appraisal_bound_ci.md
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
from evals.io_utils import atomic_write_json, atomic_write_text


_METRIC_KEYS = ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk")


def _parse_cap(tok: str) -> float | None:
    tok = tok.strip()
    if not tok or tok.lower() == "none":
        return None
    return float(tok)


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
    leq = sum(1 for m in means if m <= 0)
    geq = sum(1 for m in means if m >= 0)
    p = min(1.0, 2 * min(leq, geq) / resamples)
    return statistics.fmean(values), means[lo_idx], means[hi_idx], p


def run_appraisal_bound_ci(
    dataset_path: str,
    *,
    cap_a: float | None = 0.30,
    cap_b: float | None = None,
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
        appraisal_salience_cap=cap_a,
    )
    res_b = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        appraisal_salience_cap=cap_b,
    )

    pairs_a = res_a.get("per_query_pairs") or []
    pairs_b = res_b.get("per_query_pairs") or []
    bkey = lambda r: (r["sample_id"], r.get("question"), r["category"])
    bmap: dict = {}
    for r in pairs_b:
        bmap.setdefault(bkey(r), []).append(r)
    diffs: dict[str, list[float]] = {k_: [] for k_ in _METRIC_KEYS}
    paired = 0
    for ra in pairs_a:
        bucket = bmap.get(bkey(ra))
        if not bucket:
            continue
        rb = bucket.pop(0)
        paired += 1
        for key in diffs:
            diffs[key].append(float(ra[key]) - float(rb[key]))

    summary: dict[str, dict] = {}
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
        "cap_a": cap_a,
        "cap_b": cap_b,
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
            "delta_grk": res_a["delta"].get("gold_recall_at_k"),
            "delta_prk": res_a["delta"].get("pair_recall_at_k"),
        },
        "headline_b": {
            "delta_h1": res_b["delta"]["session_hit_at_1"],
            "delta_hk": res_b["delta"]["session_hit_at_k"],
            "delta_mrr": res_b["delta"]["mean_reciprocal_rank"],
            "delta_grk": res_b["delta"].get("gold_recall_at_k"),
            "delta_prk": res_b["delta"].get("pair_recall_at_k"),
        },
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(rep: dict) -> str:
    cap_a = "None" if rep["cap_a"] is None else f"{rep['cap_a']:.2f}"
    cap_b = "None" if rep["cap_b"] is None else f"{rep['cap_b']:.2f}"
    lines = [
        f"# §94c-appraisal-bound-CI — paired bootstrap (cap={cap_a} − cap={cap_b})",
        "",
        f"Dataset: {rep['dataset_path']} "
        f"(max_instances={rep['max_instances']}, k={rep['k']}, "
        f"embedder={rep['embedder']}).",
        f"n_paired={rep['n_paired']} | "
        f"resamples={rep['ci_config']['resamples']} | "
        f"seed={rep['ci_config']['seed']} | "
        f"wall={rep['wall_seconds']}s.",
        "",
        "## Headline (point estimates)",
        "",
        "| arm | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |",
        "|---|---|---|---|---|---|",
    ]
    for label, h in (("a (cap={})".format(cap_a), rep["headline_a"]),
                     ("b (cap={})".format(cap_b), rep["headline_b"])):
        lines.append(
            f"| {label} | {h['delta_h1']:+.4f} | {h['delta_hk']:+.4f} | "
            f"{h.get('delta_prk', 0.0):+.4f} | "
            f"{h.get('delta_grk', 0.0):+.4f} | {h['delta_mrr']:+.4f} |"
        )
    lines += [
        "",
        "## Paired bootstrap CI on per-pair (Δ_a − Δ_b)",
        "",
        "| metric | mean | 95% CI | p (two-sided) |",
        "|---|---|---|---|",
    ]
    for k_ in _METRIC_KEYS:
        c = rep["summary"][k_]
        lines.append(
            f"| {k_} | {c['mean_diff_a_minus_b']:+.4f} | "
            f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
            f"{c['p_bootstrap_two_sided']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--cap-a", default="0.30",
                   help="treatment cap; 'none' for no-cap")
    p.add_argument("--cap-b", default="none",
                   help="control cap; 'none' for no-cap")
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--synthesis", action="store_true")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    cap_a = _parse_cap(args.cap_a)
    cap_b = _parse_cap(args.cap_b)

    rep = run_appraisal_bound_ci(
        args.dataset,
        cap_a=cap_a,
        cap_b=cap_b,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    cap_a_s = "None" if cap_a is None else f"{cap_a:.2f}"
    cap_b_s = "None" if cap_b is None else f"{cap_b:.2f}"
    print(f"§94c-appraisal-bound-CI  cap={cap_a_s} − cap={cap_b_s}")
    print(f"  n_paired={rep['n_paired']}  resamples={args.resamples}  "
          f"wall={rep['wall_seconds']}s")
    for k_ in _METRIC_KEYS:
        c = rep["summary"][k_]
        print(f"  Δ({k_:>9}): mean={c['mean_diff_a_minus_b']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[appraisal_bound_ci] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep))
        print(f"[appraisal_bound_ci] wrote {args.md_out}")


if __name__ == "__main__":
    main()

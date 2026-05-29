"""§94d-tau-CI — paired bootstrap CI on (Δ_tau_a − Δ_tau_b) at the full pipeline.

Motivation
----------
§94c-decompose-positive-control swept ``schema_synthesis_tau`` ∈
{0.30, 0.20, 0.10, 0.05} on cumulative subsets S1..S7 with synthesis on
and ``min_supports=2``. The point-estimate verdict was that the
schema-family stages produce indistinguishable per-pair retrieval
diffs at any tau; §94c-decompose-positive-control-CI then locked the
formal verdict on the S3→S4 transition only (Δ_S4 − Δ_S3 brackets
zero on every metric).

This driver is the *retrieval-end* complement: instead of bisecting
stages, it holds the full pipeline (S7) fixed and compares two tau
arms head-to-head with paired bootstrap. The hypothesis is that tau
is inert at the **headline retrieval level** when synthesis is on —
so flipping the production default away from 0.30 should be a no-op
on §94 metrics. If any metric's CI on (Δ_a − Δ_b) excludes zero we
have a tau dependence at the retrieval end and have to pick a
defensible default.

Method
------
Run ``evals.locomo_recall_lift`` twice with synthesis=True under
``tau_a`` (default 0.30) and ``tau_b`` (default 0.05) on the same
fixture. Pair on ``(sample_id, question, category)``. For each of the
five primary metrics (Δh@1, Δh@k, ΔRR, Δprk, Δgrk) compute per-pair
(Δ_a − Δ_b) and percentile-bootstrap the mean (10k resamples,
seed=42). Print + emit JSON + Markdown.

Pattern is a straight copy of ``evals.locomo_appraisal_bound_ci``
swapping ``appraisal_salience_cap`` for ``schema_synthesis_tau`` /
``schema_synthesis_min_supports``.

Usage
-----
    python -m evals.locomo_tau_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --tau-a 0.30 --tau-b 0.05 --min-supports 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_tau_ci.json \\
        --md-out bench/results/locomo_tau_ci.md
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


def run_tau_ci(
    dataset_path: str,
    *,
    tau_a: float = 0.30,
    tau_b: float = 0.05,
    min_supports: int = 2,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = True,
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Run two full-pipeline arms at different ``schema_synthesis_tau`` and
    paired-bootstrap the per-pair (Δ_a − Δ_b) diff on every primary metric.

    Synthesis must be on for tau to have any chance of moving — with
    synthesis=False the synthesizer never runs and tau is structurally
    inert. We default ``synthesis=True``; setting ``--no-synthesis`` is
    available as a sanity-check arm (should produce zero diffs).
    """
    t0 = time.monotonic()
    res_a = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        schema_synthesis_tau=float(tau_a),
        schema_synthesis_min_supports=int(min_supports),
    )
    res_b = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        schema_synthesis_tau=float(tau_b),
        schema_synthesis_min_supports=int(min_supports),
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
        "tau_a": float(tau_a),
        "tau_b": float(tau_b),
        "min_supports": int(min_supports),
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
    lines = [
        f"# §94d-tau-CI — paired bootstrap "
        f"(tau={rep['tau_a']:.2f} − tau={rep['tau_b']:.2f}) "
        f"@ full pipeline, synthesis={rep['synthesis']}",
        "",
        f"Dataset: {rep['dataset_path']} "
        f"(max_instances={rep['max_instances']}, k={rep['k']}, "
        f"embedder={rep['embedder']}, "
        f"min_supports={rep['min_supports']}).",
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
    for label, h in ((f"a (tau={rep['tau_a']:.2f})", rep["headline_a"]),
                     (f"b (tau={rep['tau_b']:.2f})", rep["headline_b"])):
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
    p.add_argument("--tau-a", type=float, default=0.30)
    p.add_argument("--tau-b", type=float, default=0.05)
    p.add_argument("--min-supports", type=int, default=2)
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--no-synthesis", dest="synthesis", action="store_false",
                   help="sanity-check arm (should yield identically-zero diffs)")
    p.set_defaults(synthesis=True)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_tau_ci(
        args.dataset,
        tau_a=args.tau_a,
        tau_b=args.tau_b,
        min_supports=args.min_supports,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94d-tau-CI  tau={args.tau_a:.2f} − tau={args.tau_b:.2f}  "
          f"(synthesis={args.synthesis}, min_supports={args.min_supports})")
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
        print(f"[tau_ci] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep))
        print(f"[tau_ci] wrote {args.md_out}")


if __name__ == "__main__":
    main()

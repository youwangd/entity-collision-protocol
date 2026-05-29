"""§94c-decompose-positive-control-CI — bootstrap CI on the schema_update signal.

Background. §94c-decompose-positive-control swept ``schema_synthesis_tau``
∈ {0.30, 0.20, 0.10, 0.05} with ``--synthesis`` on and
``min_supports=2`` to force SCHEMA writes. The lone non-trivial gate
stage was ``schema_update`` (S3 → S4): point-estimate Δh@1 −0.0034pp at
*every* tau (n=301), and Δh@k/Δgrk −1.00pp at tau=0.30. The signal is
small but suspiciously consistent across taus.

Question. Does the per-pair (Δ_S4 − Δ_S3) bootstrap CI exclude zero on
*any* primary metric at tau=0.30 (the strongest of the four taus on Δh@k
and Δgrk)?

Method. Re-run S3 (`extraction,fact_extraction,interference`) and S4
(`extraction,fact_extraction,interference,schema_update`) at
``schema_synthesis_tau=0.30``, ``schema_synthesis_min_supports=2``,
``synthesis=True``, ``embedder='hashtrigram'``,
``appraisal_salience_cap=None`` (the §94c-positive-control config).
Pair on (sample_id, question, category). Percentile-bootstrap the mean
of per-pair (Δ_S4 − Δ_S3) at 10k resamples, seed=42, on five primary
metrics (h@1, h@k, MRR, prk, grk).

Decision rule.
  * If every metric CI brackets zero → file ``schema_update`` as
    formally inert; recommend deleting or default-disabling the stage.
  * If any metric CI excludes zero → write up the negative-stage claim
    with CI evidence and propose either keeping the stage as a tunable
    knob or default-disabling.

Usage:
    python -m evals.locomo_recall_lift_decompose_positive_control_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --tau 0.30 \\
        --min-supports 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§94c_decompose_positive_control_ci.json \\
        --md-out bench/results/locomo_recall_lift_§94c_decompose_positive_control_ci.md
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


STAGES_S3 = ["extraction", "fact_extraction", "interference"]
STAGES_S4 = ["extraction", "fact_extraction", "interference", "schema_update"]

METRIC_KEYS = ("delta_h1", "delta_hk", "delta_rr", "delta_prk", "delta_grk")


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


def _pair_diffs(pairs_a, pairs_b):
    """Pair on (sample_id, question, category); per-metric (a-b) diff lists."""
    bkey = lambda r: (r["sample_id"], r.get("question"), r["category"])
    bmap: dict = {}
    for r in pairs_b:
        bmap.setdefault(bkey(r), []).append(r)
    diffs = {k: [] for k in METRIC_KEYS}
    paired = 0
    for ra in pairs_a:
        bucket = bmap.get(bkey(ra))
        if not bucket:
            continue
        rb = bucket.pop(0)
        paired += 1
        for k in METRIC_KEYS:
            diffs[k].append(float(ra[k]) - float(rb[k]))
    return diffs, paired


def run_positive_control_ci(
    dataset_path: str,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    tau: float = 0.30,
    min_supports: int = 2,
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap CI on per-pair (Δ_S4 − Δ_S3) at fixed tau / min_supports.

    The "a minus b" sign convention here is **S4 − S3**: a positive mean
    means schema_update *helps* on that metric, a negative mean means it
    hurts. Matches the convention used in the positive-control table.
    """
    t0 = time.monotonic()
    res_s3 = run_recall_lift(
        dataset_path,
        max_instances=max_instances,
        k=k,
        embedder_name=embedder_name,
        synthesis=True,
        stages=STAGES_S3,
        schema_synthesis_tau=float(tau),
        schema_synthesis_min_supports=int(min_supports),
    )
    res_s4 = run_recall_lift(
        dataset_path,
        max_instances=max_instances,
        k=k,
        embedder_name=embedder_name,
        synthesis=True,
        stages=STAGES_S4,
        schema_synthesis_tau=float(tau),
        schema_synthesis_min_supports=int(min_supports),
    )

    pairs_s3 = res_s3.get("per_query_pairs") or []
    pairs_s4 = res_s4.get("per_query_pairs") or []
    diffs, paired = _pair_diffs(pairs_s4, pairs_s3)  # S4 − S3

    summary = {}
    for mk in METRIC_KEYS:
        m, lo, hi, p = _bootstrap_mean_ci(diffs[mk], resamples, seed)
        summary[mk] = {
            "mean_diff_s4_minus_s3": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_bootstrap_two_sided": round(p, 6),
            "n_paired": len(diffs[mk]),
            "ci_excludes_zero": (lo > 0) or (hi < 0),
        }

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "tau": float(tau),
        "min_supports": int(min_supports),
        "synthesis": True,
        "stage_a": "S4_+schema_update",
        "stage_b": "S3_+interference",
        "stages_a": STAGES_S4,
        "stages_b": STAGES_S3,
        "n_pairs_a": len(pairs_s4),
        "n_pairs_b": len(pairs_s3),
        "n_paired": paired,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "summary": summary,
        "headline_s3": {
            "delta_h1": res_s3["delta"]["session_hit_at_1"],
            "delta_hk": res_s3["delta"]["session_hit_at_k"],
            "delta_mrr": res_s3["delta"]["mean_reciprocal_rank"],
            "delta_prk": res_s3["delta"]["pair_recall_at_k"],
            "delta_grk": res_s3["delta"]["gold_recall_at_k"],
        },
        "headline_s4": {
            "delta_h1": res_s4["delta"]["session_hit_at_1"],
            "delta_hk": res_s4["delta"]["session_hit_at_k"],
            "delta_mrr": res_s4["delta"]["mean_reciprocal_rank"],
            "delta_prk": res_s4["delta"]["pair_recall_at_k"],
            "delta_grk": res_s4["delta"]["gold_recall_at_k"],
        },
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(
        f"### §94c-decompose-positive-control-CI — paired bootstrap on "
        f"(S4 − S3), tau={report['tau']:.2f}, "
        f"min_supports={report['min_supports']}, "
        f"max_instances={report['max_instances']}, "
        f"k={report['k']}, embedder={report['embedder']}, "
        f"synthesis=True, resamples={report['ci_config']['resamples']}"
    )
    lines.append("")
    lines.append(
        f"S3 stages = `{','.join(report['stages_b'])}`  "
        f"S4 stages = `{','.join(report['stages_a'])}`  "
        f"n_paired = {report['n_paired']}"
    )
    lines.append("")
    lines.append(
        "| metric | mean (S4−S3) | 95% CI | p_two_sided | n | flag |"
    )
    lines.append("| --- | ---:| --- | ---:| ---:| ---:|")
    for mk in METRIC_KEYS:
        c = report["summary"][mk]
        star = "★" if c["ci_excludes_zero"] else ""
        lines.append(
            f"| `{mk}` | {c['mean_diff_s4_minus_s3']:+.4f} | "
            f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
            f"{c['p_bootstrap_two_sided']:.3f}{star} | "
            f"{c['n_paired']} | {'★' if c['ci_excludes_zero'] else ''} |"
        )
    lines.append("")
    lines.append(
        "**Reading.** Sign convention is S4 − S3, so positive means "
        "`schema_update` *helps* and negative means it *hurts*. ★ = 95% "
        "CI excludes zero. If every metric brackets zero, the lone "
        "non-trivial signal from the positive-control sweep is within "
        "noise — file `schema_update` as formally inert and recommend "
        "default-disabling the stage."
    )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--tau", type=float, default=0.30)
    p.add_argument("--min-supports", type=int, default=2)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_positive_control_ci(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        tau=args.tau,
        min_supports=args.min_supports,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94c-decompose-positive-control-CI  tau={rep['tau']:.2f}  "
          f"min_supports={rep['min_supports']}  "
          f"n_paired={rep['n_paired']}  wall={rep['wall_seconds']}s")
    for mk in METRIC_KEYS:
        c = rep["summary"][mk]
        tag = "★" if c["ci_excludes_zero"] else ""
        print(f"  Δ({mk:>9}): mean={c['mean_diff_s4_minus_s3']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}{tag}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[positive-control-CI] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[positive-control-CI] wrote {args.md_out}")


if __name__ == "__main__":
    main()

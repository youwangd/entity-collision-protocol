"""§94c-decompose-adjacent-CI — per-adjacent-stage paired bootstrap CI.

§94c-decompose-CI (cumulative S1 vs S7) flagged Δgold_recall@k with
p=0.038 — a borderline bite that does not survive Bonferroni across 5
metrics. Question for this driver: *which* adjacent stage transition
(S_i → S_{i+1}) owns that ~0.75pp gold_recall@k movement?

Method. Reuse `evals.locomo_recall_lift.run_recall_lift` to compute
each cumulative subset (S1..S7) exactly once, capture the
`per_query_pairs` for each, then for every adjacent pair (S_i, S_{i+1})
percentile-bootstrap the per-pair (Δ_{S_i} − Δ_{S_{i+1}}) mean for all
five primary metrics. Localization rule: a transition "owns" a metric
movement iff its 95% CI excludes zero.

Output. JSON artifact + Markdown table; rendered into SCALE_REPORT.md.

Usage:
    python -m evals.locomo_recall_lift_decompose_adjacent_ci \
        --dataset bench/data/locomo10.json \
        --max-instances 2 \
        --resamples 10000 \
        --out bench/results/locomo_recall_lift_§94c_decompose_adjacent_ci.json \
        --md-out bench/results/locomo_recall_lift_§94c_decompose_adjacent_ci.md
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
from evals.locomo_recall_lift_decompose_ci import SUBSET_PRESETS
from evals.io_utils import atomic_write_json, atomic_write_text


SUBSET_ORDER: list[str] = [
    "S1_extraction_only",
    "S2_+fact",
    "S3_+interference",
    "S4_+schema_update",
    "S5_+somatic",
    "S6_+merge_persist",
    "S7_full_default",
]

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
    """Pair on (sample_id, question, category); return per-metric diff lists."""
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


def _added_stage(a_name: str, b_name: str) -> str | None:
    sa = SUBSET_PRESETS[a_name] or []
    sb = SUBSET_PRESETS[b_name]
    if sb is None:
        added = "(implicit_full_default)"
    else:
        diff = [s for s in sb if s not in set(sa)]
        added = ",".join(diff) if diff else None
    return added


def run_adjacent_ci(
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
    subset_results: dict[str, dict] = {}
    subset_walls: dict[str, float] = {}
    for name in SUBSET_ORDER:
        sub_t = time.monotonic()
        subset_results[name] = run_recall_lift(
            dataset_path,
            max_instances=max_instances,
            k=k,
            embedder_name=embedder_name,
            synthesis=synthesis,
            stages=SUBSET_PRESETS[name],
        )
        subset_walls[name] = round(time.monotonic() - sub_t, 2)

    transitions: list[dict] = []
    for i in range(len(SUBSET_ORDER) - 1):
        a_name = SUBSET_ORDER[i]
        b_name = SUBSET_ORDER[i + 1]
        pairs_a = subset_results[a_name].get("per_query_pairs") or []
        pairs_b = subset_results[b_name].get("per_query_pairs") or []
        diffs, paired = _pair_diffs(pairs_a, pairs_b)
        summary = {}
        for mk in METRIC_KEYS:
            m, lo, hi, p = _bootstrap_mean_ci(diffs[mk], resamples, seed)
            summary[mk] = {
                "mean_diff_a_minus_b": round(m, 6),
                "ci_lo": round(lo, 6),
                "ci_hi": round(hi, 6),
                "p_bootstrap_two_sided": round(p, 6),
                "n_paired": len(diffs[mk]),
                "ci_excludes_zero": (lo > 0) or (hi < 0),
            }
        transitions.append({
            "transition": f"{a_name} -> {b_name}",
            "stage_a": a_name,
            "stage_b": b_name,
            "added_stage": _added_stage(a_name, b_name),
            "n_paired": paired,
            "summary": summary,
            "headline_a": {
                "delta_h1": subset_results[a_name]["delta"]["session_hit_at_1"],
                "delta_hk": subset_results[a_name]["delta"]["session_hit_at_k"],
                "delta_grk": subset_results[a_name]["delta"]["gold_recall_at_k"],
            },
            "headline_b": {
                "delta_h1": subset_results[b_name]["delta"]["session_hit_at_1"],
                "delta_hk": subset_results[b_name]["delta"]["session_hit_at_k"],
                "delta_grk": subset_results[b_name]["delta"]["gold_recall_at_k"],
            },
        })

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "synthesis": synthesis,
        "subset_order": SUBSET_ORDER,
        "subset_walls_s": subset_walls,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "transitions": transitions,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(f"### §94c-decompose-adjacent-CI — per-transition paired "
                 f"bootstrap CI (max_instances={report['max_instances']}, "
                 f"k={report['k']}, embedder={report['embedder']}, "
                 f"synthesis={report['synthesis']}, "
                 f"resamples={report['ci_config']['resamples']})")
    lines.append("")
    lines.append("| transition | added | n_paired | Δh@1 mean (CI) p | "
                 "Δh@k mean (CI) p | ΔMRR mean (CI) p | "
                 "Δprk mean (CI) p | Δgrk mean (CI) p |")
    lines.append("| --- | --- | ---:| --- | --- | --- | --- | --- |")
    for t in report["transitions"]:
        cells = [f"`{t['transition']}`",
                 f"`{t['added_stage']}`",
                 str(t["n_paired"])]
        for mk in METRIC_KEYS:
            c = t["summary"][mk]
            star = "★" if c["ci_excludes_zero"] else ""
            cells.append(
                f"{c['mean_diff_a_minus_b']:+.4f} "
                f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
                f"p={c['p_bootstrap_two_sided']:.3f}{star}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("★ = 95% CI excludes zero. Pairing key = "
                 "(sample_id, question, category). Method = percentile "
                 "bootstrap on per-pair (Δ_{S_a} − Δ_{S_b}).")
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
    p.add_argument("--synthesis", action="store_true")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_adjacent_ci(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94c-decompose-adjacent-CI  wall={rep['wall_seconds']}s")
    for t in rep["transitions"]:
        line = (f"  {t['transition']}  added={t['added_stage']}  "
                f"n={t['n_paired']}  ")
        bites = []
        for mk in METRIC_KEYS:
            c = t["summary"][mk]
            tag = "★" if c["ci_excludes_zero"] else ""
            bites.append(f"{mk}={c['mean_diff_a_minus_b']:+.4f}{tag}")
        print(line + "  ".join(bites))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[adjacent-CI] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[adjacent-CI] wrote {args.md_out}")


if __name__ == "__main__":
    main()

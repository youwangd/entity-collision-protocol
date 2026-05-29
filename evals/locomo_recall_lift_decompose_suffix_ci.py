"""§94c-decompose-suffix-CI — localize the S6→S7 Δgrk bite.

§94c-decompose-adjacent-CI showed that of the 6 adjacent transitions
S1→S2 ... S6→S7, only S6→S7 emitted a CI excluding zero on any of the
five primary metrics: a single ★ on Δgrk = −0.0075pp [−0.0166, −0.0008]
p=0.038. That transition bundles 7 stages in one jump — the entire
"S7 default minus S6" suffix:

    suffix = {appraisal, emotion_tagging, deduplication, decay,
              suppression, temperament_drift, mood_update}

This driver runs S6 + each one of those 7 stages individually and
percentile-bootstraps the per-pair (Δ_S6 − Δ_{S6+x}) mean for all five
metrics. The stage that owns the Δgrk bite (or any other metric
movement) is the one whose CI excludes zero.

Pairing key = (sample_id, question, category). Method = percentile
bootstrap on per-pair (Δ_a − Δ_b).

Usage:
    python -m evals.locomo_recall_lift_decompose_suffix_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§94c_decompose_suffix_ci.json \\
        --md-out bench/results/locomo_recall_lift_§94c_decompose_suffix_ci.md
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from evals.locomo_recall_lift import run_recall_lift
from evals.locomo_recall_lift_decompose_adjacent_ci import (
    METRIC_KEYS,
    _bootstrap_mean_ci,
    _pair_diffs,
)
from evals.locomo_recall_lift_decompose_ci import SUBSET_PRESETS
from evals.io_utils import atomic_write_json, atomic_write_text


# S7 default minus S6 — the 7 stages bundled into the S6→S7 jump.
SUFFIX_STAGES: list[str] = [
    "deduplication",      # Stage 2 (pre-extraction)
    "appraisal",          # Stage 5
    "emotion_tagging",    # Stage 6
    "decay",              # Stage 10
    "suppression",        # Stage 11
    "temperament_drift",  # Stage 12a
    "mood_update",        # Stage 12b
]

S6_NAME = "S6_+merge_persist"
S7_NAME = "S7_full_default"


def _s6_plus(stage: str) -> list[str]:
    """S6's stage list with one suffix stage appended."""
    base = list(SUBSET_PRESETS[S6_NAME] or [])
    if stage not in base:
        base.append(stage)
    return base


def run_suffix_ci(
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

    # Anchor: S6 alone (one full run, reused as 'a' for every comparison).
    sub_t = time.monotonic()
    res_s6 = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        stages=SUBSET_PRESETS[S6_NAME],
    )
    s6_wall = round(time.monotonic() - sub_t, 2)
    pairs_s6 = res_s6.get("per_query_pairs") or []

    # Reference: S7 full default (sanity-check the bundled bite reproduces).
    sub_t = time.monotonic()
    res_s7 = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        stages=SUBSET_PRESETS[S7_NAME],
    )
    s7_wall = round(time.monotonic() - sub_t, 2)
    pairs_s7 = res_s7.get("per_query_pairs") or []

    transitions: list[dict] = []
    walls: dict[str, float] = {S6_NAME: s6_wall, S7_NAME: s7_wall}

    for stage in SUFFIX_STAGES:
        sub_t = time.monotonic()
        res_x = run_recall_lift(
            dataset_path, max_instances=max_instances, k=k,
            embedder_name=embedder_name, synthesis=synthesis,
            stages=_s6_plus(stage),
        )
        walls[f"S6+{stage}"] = round(time.monotonic() - sub_t, 2)
        pairs_x = res_x.get("per_query_pairs") or []
        diffs, paired = _pair_diffs(pairs_s6, pairs_x)
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
            "transition": f"{S6_NAME} -> S6+{stage}",
            "added_stage": stage,
            "n_paired": paired,
            "summary": summary,
            "headline_b": {
                "delta_h1": res_x["delta"]["session_hit_at_1"],
                "delta_hk": res_x["delta"]["session_hit_at_k"],
                "delta_grk": res_x["delta"]["gold_recall_at_k"],
            },
        })

    # Sanity: also report S6 vs S7 bundled CI for direct comparison.
    diffs_bundle, paired_bundle = _pair_diffs(pairs_s6, pairs_s7)
    bundle_summary = {}
    for mk in METRIC_KEYS:
        m, lo, hi, p = _bootstrap_mean_ci(diffs_bundle[mk], resamples, seed)
        bundle_summary[mk] = {
            "mean_diff_a_minus_b": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_bootstrap_two_sided": round(p, 6),
            "n_paired": len(diffs_bundle[mk]),
            "ci_excludes_zero": (lo > 0) or (hi < 0),
        }

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "synthesis": synthesis,
        "anchor": S6_NAME,
        "suffix_stages": SUFFIX_STAGES,
        "subset_walls_s": walls,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "bundle": {
            "transition": f"{S6_NAME} -> {S7_NAME}",
            "n_paired": paired_bundle,
            "summary": bundle_summary,
        },
        "transitions": transitions,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(
        f"### §94c-decompose-suffix-CI — localize S6→S7 Δgrk bite "
        f"(max_instances={report['max_instances']}, k={report['k']}, "
        f"embedder={report['embedder']}, synthesis={report['synthesis']}, "
        f"resamples={report['ci_config']['resamples']})"
    )
    lines.append("")
    lines.append(
        "Anchor: **S6** = [extraction, fact_extraction, interference, "
        "schema_update, somatic_marking, mechanical_merge]. "
        "Each row probes S6 + one of the 7 stages bundled into S6→S7. "
        "The bottom 'bundle' row reports the full S6→S7 CI for reference."
    )
    lines.append("")
    lines.append(
        "| probe | added | n | Δh@1 mean (CI) p | Δh@k mean (CI) p | "
        "ΔMRR mean (CI) p | Δprk mean (CI) p | Δgrk mean (CI) p |"
    )
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
    b = report["bundle"]
    cells = [f"`{b['transition']}`", "`(bundle)`", str(b["n_paired"])]
    for mk in METRIC_KEYS:
        c = b["summary"][mk]
        star = "★" if c["ci_excludes_zero"] else ""
        cells.append(
            f"{c['mean_diff_a_minus_b']:+.4f} "
            f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
            f"p={c['p_bootstrap_two_sided']:.3f}{star}"
        )
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "★ = 95% CI excludes zero. Pairing key = "
        "(sample_id, question, category). Method = percentile "
        "bootstrap on per-pair (Δ_S6 − Δ_{S6+x})."
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
    p.add_argument("--synthesis", action="store_true")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_suffix_ci(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94c-decompose-suffix-CI  wall={rep['wall_seconds']}s")
    for t in rep["transitions"]:
        bites = []
        for mk in METRIC_KEYS:
            c = t["summary"][mk]
            tag = "★" if c["ci_excludes_zero"] else ""
            bites.append(f"{mk}={c['mean_diff_a_minus_b']:+.4f}{tag}")
        print(f"  added={t['added_stage']:<20s}  n={t['n_paired']}  "
              + "  ".join(bites))
    print(f"  [bundle S6->S7]  n={rep['bundle']['n_paired']}  " + "  ".join(
        f"{mk}={rep['bundle']['summary'][mk]['mean_diff_a_minus_b']:+.4f}"
        f"{'★' if rep['bundle']['summary'][mk]['ci_excludes_zero'] else ''}"
        for mk in METRIC_KEYS))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[suffix-CI] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[suffix-CI] wrote {args.md_out}")


if __name__ == "__main__":
    main()

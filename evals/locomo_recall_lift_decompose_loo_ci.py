"""§94c-decompose-LOO-CI — leave-one-out necessity test on S7.

Adjacent-CI + suffix-CI together established (a) that the only adjacent
transition with a CI excluding zero is `S6→S7`, and (b) that of the 7
stages bundled into that jump, only `appraisal` reproduces the bundled
Δgrk bite. Both probes traverse the pipeline *cumulatively*, so they
identify *sufficiency* of an added stage on top of S6. They cannot
identify *necessity* — whether the full default S7 still moves a
metric without each individual stage.

This driver runs the LOO complement: anchor = S7 (full default), then
for each non-mandatory stage `x`, run S7 \\ {x} and percentile-bootstrap
the per-pair (Δ_{S7\\x} − Δ_S7) mean. A stage `x` is *necessary* on
metric m iff S7\\x materially differs from S7 — i.e. removing x bites
metric m. The mandatory stages `replay` and `persistence` are filtered
in `pipeline.ConsolidationManager.__init__` regardless and so cannot
be dropped.

Usage:
    python -m evals.locomo_recall_lift_decompose_loo_ci \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --resamples 10000 \\
        --out bench/results/locomo_recall_lift_§94c_decompose_loo_ci.json \\
        --md-out bench/results/locomo_recall_lift_§94c_decompose_loo_ci.md
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
from evals.io_utils import atomic_write_json, atomic_write_text


# All stages in the default S7 pipeline that *can* be dropped via
# `ConsolidationConfig.stages`. `replay` and `persistence` are forced
# back in by the manager and cannot be left out, so they don't appear
# here. Order matches `pipeline.ConsolidationManager.__init__`.
S7_STAGES_DROPPABLE: list[str] = [
    "deduplication",       # Stage 2
    "extraction",          # Stage 3
    "fact_extraction",     # Stage 4
    "appraisal",           # Stage 5
    "emotion_tagging",     # Stage 6
    "interference",        # Stage 7
    "schema_update",       # Stage 8
    "somatic_marking",     # Stage 9
    "decay",               # Stage 10
    "suppression",         # Stage 11
    "temperament_drift",   # Stage 12a
    "mood_update",         # Stage 12b
    "mechanical_merge",    # Stage 12c
]

S7_NAME = "S7_full_default"


def _s7_minus(stage: str) -> list[str]:
    """All droppable S7 stages except `stage`."""
    return [s for s in S7_STAGES_DROPPABLE if s != stage]


def run_loo_ci(
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

    # Anchor: S7 full default (passed as None → all stages on).
    sub_t = time.monotonic()
    res_s7 = run_recall_lift(
        dataset_path, max_instances=max_instances, k=k,
        embedder_name=embedder_name, synthesis=synthesis,
        stages=None,
    )
    s7_wall = round(time.monotonic() - sub_t, 2)
    pairs_s7 = res_s7.get("per_query_pairs") or []

    transitions: list[dict] = []
    walls: dict[str, float] = {S7_NAME: s7_wall}

    for stage in S7_STAGES_DROPPABLE:
        sub_t = time.monotonic()
        stages_minus = _s7_minus(stage)
        res_x = run_recall_lift(
            dataset_path, max_instances=max_instances, k=k,
            embedder_name=embedder_name, synthesis=synthesis,
            stages=stages_minus,
        )
        walls[f"S7-{stage}"] = round(time.monotonic() - sub_t, 2)
        pairs_x = res_x.get("per_query_pairs") or []
        diffs, paired = _pair_diffs(pairs_x, pairs_s7)  # a=S7\x, b=S7
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
            "transition": f"S7-{stage} -> {S7_NAME}",
            "dropped_stage": stage,
            "n_paired": paired,
            "summary": summary,
            "headline_a": {
                "delta_h1": res_x["delta"]["session_hit_at_1"],
                "delta_hk": res_x["delta"]["session_hit_at_k"],
                "delta_grk": res_x["delta"]["gold_recall_at_k"],
            },
        })

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "synthesis": synthesis,
        "anchor": S7_NAME,
        "droppable_stages": S7_STAGES_DROPPABLE,
        "subset_walls_s": walls,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "headline_anchor": {
            "delta_h1": res_s7["delta"]["session_hit_at_1"],
            "delta_hk": res_s7["delta"]["session_hit_at_k"],
            "delta_grk": res_s7["delta"]["gold_recall_at_k"],
        },
        "transitions": transitions,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(
        f"### §94c-decompose-LOO-CI — leave-one-out necessity on S7 "
        f"(max_instances={report['max_instances']}, k={report['k']}, "
        f"embedder={report['embedder']}, synthesis={report['synthesis']}, "
        f"resamples={report['ci_config']['resamples']})"
    )
    lines.append("")
    h = report["headline_anchor"]
    lines.append(
        f"Anchor: **S7** = full default pipeline. "
        f"Headline (S7 vs no-consolidation baseline): "
        f"Δh@1={h['delta_h1']:+.4f}, Δh@k={h['delta_hk']:+.4f}, "
        f"Δgrk={h['delta_grk']:+.4f}. "
        f"Each row drops one stage from S7 and bootstraps "
        f"per-pair (Δ_{{S7\\x}} − Δ_S7). A stage is *necessary* on "
        f"metric m iff its CI excludes zero on m."
    )
    lines.append("")
    lines.append(
        "| dropped | n | Δh@1 mean (CI) p | Δh@k mean (CI) p | "
        "ΔMRR mean (CI) p | Δprk mean (CI) p | Δgrk mean (CI) p |"
    )
    lines.append("| --- | ---:| --- | --- | --- | --- | --- |")
    for t in report["transitions"]:
        cells = [f"`{t['dropped_stage']}`", str(t["n_paired"])]
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
    lines.append(
        "★ = 95% CI excludes zero (stage is *necessary* on that metric "
        "— removing it materially shifts the per-pair Δ). Pairing key = "
        "(sample_id, question, category). Method = percentile bootstrap "
        "on per-pair (Δ_{S7\\x} − Δ_S7)."
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

    rep = run_loo_ci(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(f"§94c-decompose-LOO-CI  wall={rep['wall_seconds']}s")
    for t in rep["transitions"]:
        bites = []
        for mk in METRIC_KEYS:
            c = t["summary"][mk]
            tag = "★" if c["ci_excludes_zero"] else ""
            bites.append(f"{mk}={c['mean_diff_a_minus_b']:+.4f}{tag}")
        print(f"  dropped={t['dropped_stage']:<20s}  n={t['n_paired']}  "
              + "  ".join(bites))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[LOO-CI] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[LOO-CI] wrote {args.md_out}")


if __name__ == "__main__":
    main()

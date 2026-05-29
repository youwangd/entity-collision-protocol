"""§94c-appraisal-bound — sweep `appraisal_salience_cap` on LoCoMo recall-lift.

Motivation
----------
§94c-decompose-suffix-CI localized the lone Δgrk bite (gold_recall@k
−0.0075pp [−0.0166, −0.0008] p=0.038 ★) to the `appraisal` stage:
appraisal pushes a non-gold item to rank 1 in 5/301 questions and
recovers one gold per swap of 6 — net negative. §94c-appraisal-inspect-CI
(n=106 clean rows) showed the displacing items carry +0.0325 [+0.0174,
+0.0500] more salience than the gold they displace (p=0.001), which is
real but small.

This driver tests whether **bounding** appraisal's salience output flips
Δgrk back toward zero (or positive) without sacrificing the +0.0764pp
Δh@1 from §94c-decompose. We sweep `appraisal_salience_cap ∈
{None, 0.9, 0.7, 0.5, 0.3}` — None reproduces §94c-decompose-CI; lower
caps shrink appraisal's contribution to the retrieval score
(`salience * salience_weight=0.2`).

Both arms still run the full default pipeline; the cap is the only knob
that varies between cap=X runs. Baseline is always cap=None,
treatment=False (i.e. the §94c-decompose `S6+appraisal` stand-in baked
into `run_recall_lift`).

Pure: deterministic given the input json + embedder + cap.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from evals.locomo_recall_lift import run_recall_lift
from evals.io_utils import atomic_write_json, atomic_write_text


DEFAULT_CAPS: list[float | None] = [None, 0.9, 0.7, 0.5, 0.3]


def run_sweep(
    dataset_path: str | os.PathLike,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    caps: list[float | None] | None = None,
) -> dict:
    if caps is None:
        caps = list(DEFAULT_CAPS)
    rows: list[dict] = []
    t0 = time.monotonic()
    for cap in caps:
        out = run_recall_lift(
            dataset_path,
            max_instances=max_instances,
            k=k,
            embedder_name=embedder_name,
            appraisal_salience_cap=cap,
        )
        rows.append({
            "appraisal_salience_cap": cap,
            "n_pairs": out.get("n_pairs"),
            "delta_h1": out.get("delta", {}).get("session_hit_at_1"),
            "delta_hk": out.get("delta", {}).get("session_hit_at_k"),
            "delta_prk": out.get("delta", {}).get("pair_recall_at_k"),
            "delta_grk": out.get("delta", {}).get("gold_recall_at_k"),
            "delta_mrr": out.get("delta", {}).get("mean_reciprocal_rank"),
            "treatment_h1": out.get("treatment", {}).get("session_hit_at_1"),
            "treatment_grk": out.get("treatment", {}).get("gold_recall_at_k"),
            "wall_seconds": out.get("wall_seconds"),
        })
    return {
        "dataset": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "caps": [c for c in caps],
        "rows": rows,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# §94c-appraisal-bound — cap sweep")
    lines.append("")
    lines.append(
        f"Dataset: {report['dataset']} (max_instances={report['max_instances']}, "
        f"k={report['k']}, embedder={report['embedder']})."
    )
    lines.append(f"Wall: {report['wall_seconds']}s.")
    lines.append("")
    lines.append("| cap | n | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["rows"]:
        cap = r["appraisal_salience_cap"]
        cap_s = "None" if cap is None else f"{cap:.2f}"
        lines.append(
            f"| {cap_s} | {r['n_pairs']} | {r['delta_h1']:+.4f} | "
            f"{r['delta_hk']:+.4f} | {r['delta_prk']:+.4f} | "
            f"{r['delta_grk']:+.4f} | {r['delta_mrr']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram")
    p.add_argument("--caps", default=None,
                   help="comma-separated floats; 'none' for the no-cap arm")
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    caps: list[float | None] | None = None
    if args.caps:
        caps = []
        for tok in args.caps.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "none":
                caps.append(None)
            else:
                caps.append(float(tok))

    out = run_sweep(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        caps=caps,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, out, default=str)
        print(f"[appraisal_bound_sweep] wrote {args.out}")
    if args.md_out:
        atomic_write_text(args.md_out, render_markdown(out))
        print(f"[appraisal_bound_sweep] wrote {args.md_out}")


if __name__ == "__main__":
    main()

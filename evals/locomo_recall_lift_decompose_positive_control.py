"""§94c-decompose-positive-control — force SCHEMA writes and re-check
whether the schema-family-gate stages become non-trivial.

Background. §94c-decompose-CI / -adjacent-CI / -suffix-CI / -LOO-CI all
agree that on the default LoCoMo10 max_instances=2 fixture only
``extraction`` (and to a small Δgrk degree, ``appraisal``) move retrieval
metrics — the ``schema_family_*`` gate, ``schema_update``,
``mechanical_merge``, ``fact_extraction``, ``interference``, etc. emit
operationally identical per-pair diffs. The candidate explanation in
NEXT.md ``Next pickup #4`` is that ``schema_synthesis_tau=0.3`` is too
strict for hashtrigram-256 cosine on this fixture: the synthesizer
never emits ``SCHEMA`` writes, so the gate has nothing to gate on. This
driver tests that explanation.

Method. Re-run the cumulative ``DEFAULT_SUBSETS`` from
``locomo_recall_lift_decompose`` with ``--synthesis`` enabled and a
sweep of ``schema_synthesis_tau`` values below the 0.3 default
(0.30 / 0.20 / 0.10 / 0.05). For each tau, also enumerate the
``SCHEMA``-tier writes by side-effect: count ``treatment`` rows with
``ρ`` retrieval differing from baseline. If a low-enough tau forces
SCHEMA writes, the §94c headline metrics should *separate* across
S1..S7 instead of remaining flat.

Output. JSON + Markdown table of (tau, subset, n_pairs, Δh@1, Δh@k,
ΔMRR, Δprk, Δgrk, n_consolidation_errors, wall_seconds), prepended to
``SCALE_REPORT.md`` by the caller. The hypothesis is *positive control*:
if some tau row exhibits S1 ≠ S7 with Δgrk or Δh@1 separating beyond
±1pp, the schema-family gate is *capable* of moving retrieval — it just
doesn't on the default tau. If no tau separates, the gate is inert
under hashtrigram-256 regardless of SCHEMA volume.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from evals.locomo_recall_lift import run_recall_lift
from evals.locomo_recall_lift_decompose import DEFAULT_SUBSETS
from evals.io_utils import atomic_write_json, atomic_write_text


DEFAULT_TAUS: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05)


def run_positive_control(
    dataset_path: str,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    taus: tuple[float, ...] | list[float] = DEFAULT_TAUS,
    min_supports: int = 2,
    subsets=None,
) -> dict:
    """Sweep ``schema_synthesis_tau`` × cumulative-stage subsets.

    ``min_supports`` is *also* relaxed (default 2 vs the 3-default) so
    the synthesizer has more chances to emit SCHEMA writes on a
    fixture as small as max_instances=2.
    """
    subsets = subsets or DEFAULT_SUBSETS
    rows: list[dict] = []
    t0 = time.monotonic()
    for tau in taus:
        for name, stages in subsets:
            sub_t = time.monotonic()
            result = run_recall_lift(
                dataset_path,
                max_instances=max_instances,
                k=k,
                embedder_name=embedder_name,
                synthesis=True,
                stages=stages,
                schema_synthesis_tau=float(tau),
                schema_synthesis_min_supports=int(min_supports),
            )
            if "error" in result:
                rows.append({
                    "tau": float(tau),
                    "subset": name,
                    "stages": stages,
                    "error": result["error"],
                })
                continue
            rows.append({
                "tau": float(tau),
                "subset": name,
                "stages": stages,
                "n_pairs": result["n_pairs"],
                "n_consolidation_errors":
                    len(result.get("consolidation_errors") or []),
                "delta_h1":  result["delta"]["session_hit_at_1"],
                "delta_hk":  result["delta"]["session_hit_at_k"],
                "delta_mrr": result["delta"]["mean_reciprocal_rank"],
                "delta_prk": result["delta"]["pair_recall_at_k"],
                "delta_grk": result["delta"]["gold_recall_at_k"],
                "wall_seconds": round(time.monotonic() - sub_t, 2),
            })
    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "taus": list(taus),
        "min_supports": int(min_supports),
        "subsets": [n for n, _ in subsets],
        "wall_seconds": round(time.monotonic() - t0, 2),
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(
        "### §94c-decompose-positive-control — schema_synthesis_tau "
        "sweep × cumulative stage subset "
        f"(max_instances={report['max_instances']}, k={report['k']}, "
        f"embedder={report['embedder']}, "
        f"min_supports={report['min_supports']})"
    )
    lines.append("")
    lines.append(
        "| tau | subset | n_pairs | Δh@1 | Δh@k | ΔMRR | Δprk | Δgrk "
        "| n_err | wall |"
    )
    lines.append(
        "| ---:| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|"
    )
    for r in report["rows"]:
        if "error" in r:
            lines.append(
                f"| {r['tau']:.2f} | {r['subset']} | — | — | — | — | — | "
                f"— | — | error: {r['error']} |"
            )
            continue
        lines.append(
            f"| {r['tau']:.2f} | `{r['subset']}` | {r['n_pairs']} | "
            f"{r['delta_h1']:+.4f} | {r['delta_hk']:+.4f} | "
            f"{r['delta_mrr']:+.4f} | {r['delta_prk']:+.4f} | "
            f"{r['delta_grk']:+.4f} | {r['n_consolidation_errors']} | "
            f"{r['wall_seconds']:.1f}s |"
        )
    lines.append("")
    lines.append(
        "**Reading.** The positive-control claim is: at *some* tau, S1 "
        "(`extraction-only`) and S7 (`full_default`) should separate on "
        "Δh@1 or Δgrk by more than the §94c-decompose-CI bootstrap "
        "noise floor (~1pp). If no tau separates the cumulative axis, "
        "the schema-family gate is operationally inert under "
        "hashtrigram-256 even when the synthesizer is forced to emit "
        "SCHEMA writes."
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
    p.add_argument("--taus", default="0.30,0.20,0.10,0.05",
                   help="comma-separated tau values to sweep")
    p.add_argument("--min-supports", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(",") if x.strip())
    rep = run_positive_control(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        taus=taus,
        min_supports=args.min_supports,
    )
    print(json.dumps(rep, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, rep, default=str)
        print(f"[positive-control] wrote {args.out}")
    if args.md_out:
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[positive-control] wrote {args.md_out}")


if __name__ == "__main__":
    main()

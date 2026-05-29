"""§94c-decompose — bisect which consolidation stage drives the LoCoMo
recall lift.

Background. §94c showed that under the live pipeline (synth-off, all
stages on), the schema-family treatment delivers Δh@1 +7.6pp, Δh@k
+15.3pp, ΔMRR +9.0pp on LoCoMo10 max_instances=2 (n_pairs=301), all
significant at p<1e-3 by 10k bootstrap (§94c-CI). §94c also showed the
Δ-of-Δ between synth-on and synth-off is ~−0.3pp, so §93 is *not* the
mechanism.

Question. With §93 ruled out, which stage(s) carry the lift? Candidates:
extraction (episode→L0), fact_extraction (events→facts), interference
(rewrite + suppress on collisions), schema_update (auto-schemas), and
the family gate stages (which only fire when SCHEMA writes happen).

Method. Run `evals.locomo_recall_lift` repeatedly with `--stages`
limiting the active set, on the *same* baseline-vs-treatment harness.
The treatment arm always includes the schema-family knobs from RECIPE,
but only the stages we name will actually execute. `replay` and
`persistence` are added implicitly by `ConsolidationManager` as
mandatory.

Output. JSON + Markdown table of (stages, n_pairs, Δh@1, Δh@k, ΔMRR,
Δpair_recall multi_hop). Identifies the *minimal* stage subset that
preserves the §94c headline.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from evals.locomo_recall_lift import run_recall_lift
from evals.io_utils import atomic_write_json, atomic_write_text


# Cumulative subsets — each row adds one stage to the previous. The
# §94c headline lives at the bottom (all stages on). If the lift
# appears at row i and is preserved on rows >i, then the *new* stage
# at row i is sufficient (necessity needs leave-one-out).
DEFAULT_SUBSETS: list[tuple[str, list[str]]] = [
    # name → stages (replay + persistence implicit)
    ("S1_extraction_only",        ["extraction"]),
    ("S2_+fact",                  ["extraction", "fact_extraction"]),
    ("S3_+interference",          ["extraction", "fact_extraction", "interference"]),
    ("S4_+schema_update",         ["extraction", "fact_extraction", "interference",
                                   "schema_update"]),
    ("S5_+somatic",               ["extraction", "fact_extraction", "interference",
                                   "schema_update", "somatic_marking"]),
    ("S6_+merge_persist",         ["extraction", "fact_extraction", "interference",
                                   "schema_update", "somatic_marking",
                                   "mechanical_merge"]),
    ("S7_full_default",           None),  # None = all stages, the §94c headline
]


def run_decompose(
    dataset_path: str,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = False,
    subsets: list[tuple[str, list[str] | None]] | None = None,
) -> dict:
    subsets = subsets or DEFAULT_SUBSETS
    rows: list[dict] = []
    t0 = time.monotonic()
    for name, stages in subsets:
        sub_t = time.monotonic()
        result = run_recall_lift(
            dataset_path,
            max_instances=max_instances,
            k=k,
            embedder_name=embedder_name,
            synthesis=synthesis,
            stages=stages,
        )
        if "error" in result:
            rows.append({"name": name, "stages": stages, "error": result["error"]})
            continue
        rows.append({
            "name": name,
            "stages": stages,
            "n_pairs": result["n_pairs"],
            "n_consolidation_errors": len(result.get("consolidation_errors") or []),
            "delta_h1":  result["delta"]["session_hit_at_1"],
            "delta_hk":  result["delta"]["session_hit_at_k"],
            "delta_mrr": result["delta"]["mean_reciprocal_rank"],
            "delta_prk_overall":   result["delta"]["pair_recall_at_k"],
            "delta_grk_overall":   result["delta"]["gold_recall_at_k"],
            "delta_prk_multihop":  result["multi_hop"]["delta_pair_recall_at_k"],
            "delta_grk_multihop":  result["multi_hop"]["delta_gold_recall_at_k"],
            "n_multihop":          result["multi_hop"]["n_pairs"],
            "wall_seconds":        round(time.monotonic() - sub_t, 2),
        })
    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": embedder_name,
        "synthesis": synthesis,
        "wall_seconds": round(time.monotonic() - t0, 2),
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(f"### §94c-decompose — stage bisection (max_instances="
                 f"{report['max_instances']}, k={report['k']}, "
                 f"embedder={report['embedder']}, "
                 f"synthesis={report['synthesis']})")
    lines.append("")
    lines.append("| subset | n_pairs | Δh@1 | Δh@k | ΔMRR | Δprk_all | Δprk_mh "
                 "(n_mh) | wall |")
    lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"| {r['name']} | — | — | — | — | — | error: "
                         f"{r['error']} | — |")
            continue
        lines.append(
            f"| `{r['name']}` | {r['n_pairs']} | "
            f"{r['delta_h1']:+.4f} | {r['delta_hk']:+.4f} | "
            f"{r['delta_mrr']:+.4f} | {r['delta_prk_overall']:+.4f} | "
            f"{r['delta_prk_multihop']:+.4f} ({r['n_multihop']}) | "
            f"{r['wall_seconds']:.1f}s |"
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
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None,
                   help="optional path for Markdown table")
    args = p.parse_args()
    rep = run_decompose(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
    )
    print(json.dumps(rep, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, rep, default=str)
        print(f"[decompose] wrote {args.out}")
    if args.md_out:
        atomic_write_text(args.md_out, render_markdown(rep) + "\n")
        print(f"[decompose] wrote {args.md_out}")


if __name__ == "__main__":
    main()

"""§94c-appraisal-bound-multihop-CI — paired bootstrap on the multi-hop slice.

Motivation
----------
§94c-appraisal-bound-CI (n_paired=301) had cap=0.30 vs cap=None brackets
zero on every primary metric (Δh@1 p=0.399). But §95-CI showed the
multi-hop slice (n_gold≥2) is where appraisal hurts most: the regression
in pair_recall on the multi-hop subset, and §94c-appraisal-inspect-cat
showed `lost_rank1` clusters in category 5 (multi-hop / open-ended).

This driver filters the §94c-appraisal-bound-CI machinery to the
multi-hop slice (questions with n_gold ≥ 2) and re-runs the paired
percentile bootstrap on the per-pair (Δ_cap=A − Δ_cap=B) diff. If
cap=0.30 wins on Δprk or Δgrk on multi-hop with CI excluding zero on
the positive side, that's evidence cap should be category- or
n_gold-conditional.

Design
------
* Re-uses ``run_recall_lift`` with ``appraisal_salience_cap`` set
  for both arms — same fixture, same embedder, same stages. Pure.
* Pairs on (sample_id, question, category), then *filters* to
  ``n_gold >= n_gold_min`` (default 2).
* Percentile bootstrap (10k resamples, seed=42) on the mean of the
  per-pair difference for each of {Δh@1, Δh@k, ΔRR, Δprk, Δgrk}.

Pure: deterministic given the input json + embedder + caps + seed.
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


def run_appraisal_bound_multihop_ci(
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
    n_gold_min: int = 2,
) -> dict:
    """Pair on (sample_id, question, category), filter to n_gold>=n_gold_min,
    then percentile-bootstrap per-pair (Δ_a − Δ_b) on each metric."""
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
    paired_multihop = 0
    for ra in pairs_a:
        bucket = bmap.get(bkey(ra))
        if not bucket:
            continue
        rb = bucket.pop(0)
        paired += 1
        n_gold = int(ra.get("n_gold") or 0)
        if n_gold < n_gold_min:
            continue
        paired_multihop += 1
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
        "n_gold_min": n_gold_min,
        "n_pairs_a": len(pairs_a),
        "n_pairs_b": len(pairs_b),
        "n_paired_total": paired,
        "n_paired_multihop": paired_multihop,
        "ci_config": {"resamples": resamples, "seed": seed,
                      "alpha": 0.05, "method": "percentile_paired_diff",
                      "slice": f"n_gold>={n_gold_min}"},
        "summary": summary,
        "headline_a_multihop": (res_a.get("multi_hop") or {}),
        "headline_b_multihop": (res_b.get("multi_hop") or {}),
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(rep: dict) -> str:
    cap_a = "None" if rep["cap_a"] is None else f"{rep['cap_a']:.2f}"
    cap_b = "None" if rep["cap_b"] is None else f"{rep['cap_b']:.2f}"
    lines = [
        f"# §94c-appraisal-bound-multihop-CI — paired bootstrap "
        f"(cap={cap_a} − cap={cap_b}) on n_gold≥{rep['n_gold_min']}",
        "",
        f"Dataset: {rep['dataset_path']} "
        f"(max_instances={rep['max_instances']}, k={rep['k']}, "
        f"embedder={rep['embedder']}).",
        f"n_paired_total={rep['n_paired_total']} | "
        f"n_paired_multihop={rep['n_paired_multihop']} | "
        f"resamples={rep['ci_config']['resamples']} | "
        f"seed={rep['ci_config']['seed']} | "
        f"wall={rep['wall_seconds']}s.",
        "",
        "## Multi-hop headline (point estimates per arm)",
        "",
        "| arm | n | Δprk | Δgrk |",
        "|---|---|---|---|",
    ]
    for label, h in (("a (cap={})".format(cap_a), rep["headline_a_multihop"]),
                     ("b (cap={})".format(cap_b), rep["headline_b_multihop"])):
        n = h.get("n_pairs", 0)
        dp = h.get("delta_pair_recall_at_k", 0.0) or 0.0
        dg = h.get("delta_gold_recall_at_k", 0.0) or 0.0
        lines.append(
            f"| {label} | {n} | {dp:+.4f} | {dg:+.4f} |"
        )
    lines += [
        "",
        f"## Paired bootstrap CI on per-pair (Δ_a − Δ_b), n_gold≥{rep['n_gold_min']}",
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
    p.add_argument("--embedder", default="hashtrigram")
    p.add_argument("--synthesis", action="store_true")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-gold-min", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    cap_a = _parse_cap(args.cap_a)
    cap_b = _parse_cap(args.cap_b)

    rep = run_appraisal_bound_multihop_ci(
        args.dataset,
        cap_a=cap_a,
        cap_b=cap_b,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        resamples=args.resamples,
        seed=args.seed,
        n_gold_min=args.n_gold_min,
    )

    cap_a_s = "None" if cap_a is None else f"{cap_a:.2f}"
    cap_b_s = "None" if cap_b is None else f"{cap_b:.2f}"
    print(f"§94c-appraisal-bound-multihop-CI  cap={cap_a_s} − cap={cap_b_s}  "
          f"n_gold>={args.n_gold_min}")
    print(f"  n_paired_total={rep['n_paired_total']}  "
          f"n_paired_multihop={rep['n_paired_multihop']}  "
          f"resamples={args.resamples}  wall={rep['wall_seconds']}s")
    for k_ in _METRIC_KEYS:
        c = rep["summary"][k_]
        print(f"  Δ({k_:>9}): mean={c['mean_diff_a_minus_b']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[appraisal_bound_multihop_ci] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep))
        print(f"[appraisal_bound_multihop_ci] wrote {args.md_out}")


if __name__ == "__main__":
    main()

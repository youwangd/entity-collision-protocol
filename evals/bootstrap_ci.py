"""Bootstrap confidence intervals over per-query results from a sweep run.

Reads a sweep_vector_weight JSON that was produced with --save-per-query, and
emits hit@1 / hit@k / MRR with bootstrapped 95% CIs for each (variant,
vector_weight) cell. Also computes paired CIs on Δ(baseline − bm25_only) so we
can ask whether the difference is significant.

Usage:
    python -m evals.bootstrap_ci --in bench/results/sweep_vw_X.json --resamples 5000

Output: prints a table to stdout and (if --out given) writes augmented JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from evals.io_utils import atomic_write_json


def _bootstrap_mean_ci(values: list[float], resamples: int, seed: int,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for the bootstrap percentile CI on the mean."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * resamples)) - 1
    hi_idx = min(hi_idx, resamples - 1)
    return statistics.fmean(values), means[lo_idx], means[hi_idx]


def _paired_diff_ci(a: list[float], b: list[float], resamples: int, seed: int,
                    alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap CI of mean(a_i - b_i) over paired samples."""
    assert len(a) == len(b), f"paired length mismatch: {len(a)} vs {len(b)}"
    diffs = [a[i] - b[i] for i in range(len(a))]
    return _bootstrap_mean_ci(diffs, resamples, seed, alpha)


def _summarize_pq(pq: list[dict], resamples: int, seed: int) -> dict:
    h1 = [r["hit_at_1"] for r in pq]
    hk = [r["hit_at_k"] for r in pq]
    rr = [r["reciprocal_rank"] for r in pq]
    out = {}
    for name, vals in (("hit_at_1", h1), ("hit_at_k", hk), ("mrr", rr)):
        m, lo, hi = _bootstrap_mean_ci(vals, resamples, seed)
        out[name] = {"mean": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    out["n"] = len(pq)
    return out


def _per_category_paired_diff(baseline_pq: list[dict], bm25_pq: list[dict],
                              resamples: int, seed: int) -> dict:
    """Slice paired per-query lists by `category` and bootstrap Δ-CIs per slice.

    Pairing is positional (same convention as the top-level paired CI). Each
    paired record must share `sample_id` and `category` for the slice to be
    well-defined; positions where they disagree are dropped with a warning.
    """
    assert len(baseline_pq) == len(bm25_pq), \
        f"paired length mismatch: {len(baseline_pq)} vs {len(bm25_pq)}"
    by_cat: dict[str, dict[str, list[float]]] = {}
    skipped = 0
    for a, b in zip(baseline_pq, bm25_pq):
        if a.get("category") != b.get("category") or a.get("sample_id") != b.get("sample_id"):
            skipped += 1
            continue
        cat = str(a.get("category"))
        slot = by_cat.setdefault(cat, {"a_h1": [], "b_h1": [], "a_hk": [], "b_hk": [],
                                       "a_rr": [], "b_rr": []})
        slot["a_h1"].append(a["hit_at_1"]); slot["b_h1"].append(b["hit_at_1"])
        slot["a_hk"].append(a["hit_at_k"]); slot["b_hk"].append(b["hit_at_k"])
        slot["a_rr"].append(a["reciprocal_rank"]); slot["b_rr"].append(b["reciprocal_rank"])
    out: dict[str, dict] = {}
    for cat, slot in sorted(by_cat.items()):
        cell = {"n": len(slot["a_h1"])}
        for metric, akey, bkey in (("hit_at_1", "a_h1", "b_h1"),
                                    ("hit_at_k", "a_hk", "b_hk"),
                                    ("mrr", "a_rr", "b_rr")):
            m, lo, hi = _paired_diff_ci(slot[akey], slot[bkey], resamples, seed)
            cell[metric] = {"mean": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
            # baseline mean for context
            bm, blo, bhi = _bootstrap_mean_ci(slot[akey], resamples, seed)
            cell[metric]["baseline_mean"] = round(bm, 4)
            bmm, _, _ = _bootstrap_mean_ci(slot[bkey], resamples, seed)
            cell[metric]["bm25_mean"] = round(bmm, 4)
        out[cat] = cell
    if skipped:
        out["_skipped_unaligned"] = skipped
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category-ci", action="store_true",
                   help="emit a per-category paired Δ-CI block per row")
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    rows = data.get("sweep") or data.get("rows") or []
    if not rows:
        raise SystemExit("no sweep rows in input")

    augmented = []
    print(f"{'vw':>5} | {'n':>4} | "
          f"{'baseline hit@1 [CI]':>28} | "
          f"{'bm25 hit@1 [CI]':>28} | "
          f"{'Δhit@1 [CI]':>28}")
    print("-" * 110)
    for row in rows:
        baseline_pq = row.get("baseline_per_query")
        bm25_pq = row.get("bm25_only_per_query")
        if not baseline_pq:
            print(f"vw={row.get('vector_weight')}: no per_query; skip")
            continue
        b_summary = _summarize_pq(baseline_pq, args.resamples, args.seed)
        bm_summary = _summarize_pq(bm25_pq, args.resamples, args.seed) if bm25_pq else None
        diff = None
        if bm25_pq and len(baseline_pq) == len(bm25_pq):
            for metric in ("hit_at_1", "hit_at_k", "mrr"):
                key = {"hit_at_1": "hit_at_1", "hit_at_k": "hit_at_k", "mrr": "reciprocal_rank"}[metric]
                a = [r[key] for r in baseline_pq]
                b = [r[key] for r in bm25_pq]
                m, lo, hi = _paired_diff_ci(a, b, args.resamples, args.seed)
                if diff is None:
                    diff = {}
                diff[metric] = {"mean": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        out_row = dict(row)
        out_row["ci"] = {"baseline": b_summary, "bm25_only": bm_summary, "delta": diff}
        if args.per_category_ci and bm25_pq and len(baseline_pq) == len(bm25_pq):
            out_row["ci"]["per_category_delta"] = _per_category_paired_diff(
                baseline_pq, bm25_pq, args.resamples, args.seed)
        # Strip per_query arrays from the output (large) unless user wants them
        out_row.pop("baseline_per_query", None)
        out_row.pop("bm25_only_per_query", None)
        augmented.append(out_row)

        def fmt(s):
            return f"{s['mean']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]"
        bm_str = fmt(bm_summary["hit_at_1"]) if bm_summary else "-"
        d_str = fmt(diff["hit_at_1"]) if diff else "-"
        print(f"{row.get('vector_weight'):>5} | {b_summary['n']:>4} | "
              f"{fmt(b_summary['hit_at_1']):>28} | {bm_str:>28} | {d_str:>28}")

    out_obj = {"config": data.get("config"), "rows": augmented,
               "ci_config": {"resamples": args.resamples, "seed": args.seed}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out_obj)
        print(f"\n[bootstrap_ci] wrote {args.out}")


if __name__ == "__main__":
    main()

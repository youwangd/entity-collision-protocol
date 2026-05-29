"""Offline adaptive vector_weight analysis.

Hypothesis: per-query, choose vw based on a cheap signal. Oracle = best vw per
query (upper bound). A clean threshold policy needs a non-leaky signal — e.g.
the FTS top-1/top-2 raw score gap — which is not in current per_query records.
We surface (a) the oracle gap as motivation, and (b) a *leaky* tau policy
using bm25's reciprocal-rank-vs-gold as a placeholder, clearly flagged.

Reads a sweep_vector_weight JSON saved with --save-per-query and joins per_query
arrays across vector_weight cells by `query` string.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def _paired_bootstrap_ci(
    deltas: list[float], resamples: int = 5000, seed: int = 13, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap percentile CI on the mean of per-query deltas.

    Returns (mean_delta, lo, hi). Mirrors evals.bootstrap_ci._bootstrap_mean_ci
    but kept local to avoid coupling the analyzer to the CI module's I/O shape.
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * resamples))]
    hi_idx = min(int(math.ceil((1 - alpha / 2) * resamples)) - 1, resamples - 1)
    return sum(deltas) / n, lo, means[hi_idx]


def _signal_verdict(
    queries: list[str],
    by_q: dict,
    bm25_by_q: dict,
    weights: list[float],
    signal_per_q: dict[str, float],
    static_best_h1_per_q: list[float],
    label: str,
) -> dict:
    """For a given non-leaky signal, sweep tau over signal quantiles, pick the
    best (tau, vw_fallback) by hit@1, and return paired-bootstrap CI on
    (adaptive_h1 - static_best_h1) per-query.

    "Useful" iff CI lower bound > 0 (strict win, 95%).
    """
    n = len(queries)
    sorted_sig = sorted(signal_per_q.values()) or [0.0]
    m = len(sorted_sig)
    best = None  # (tau, vw_fb, mean_delta, lo, hi, h1)
    for tau_q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        tau = sorted_sig[min(int(tau_q * m), m - 1)]
        for vw_fb in [w for w in weights if w > 0]:
            adaptive_h1_per_q = []
            for q in queries:
                sig = signal_per_q.get(q, 0.0)
                if sig >= tau:
                    adaptive_h1_per_q.append(bm25_by_q[q]["hit_at_1"])
                else:
                    adaptive_h1_per_q.append(by_q[q][vw_fb]["hit_at_1"])
            deltas = [a - s for a, s in zip(adaptive_h1_per_q, static_best_h1_per_q)]
            mean_d, lo, hi = _paired_bootstrap_ci(deltas)
            h1 = sum(adaptive_h1_per_q) / n
            cell = (tau_q, tau, vw_fb, mean_d, lo, hi, h1)
            if best is None or mean_d > best[3]:
                best = cell
    if best is None:
        return {"signal": label, "verdict": "no-data"}
    tau_q, tau, vw_fb, mean_d, lo, hi, h1 = best
    useful = lo > 0
    return {
        "signal": label,
        "best_cell": {"tau_q": tau_q, "tau": tau, "vw_fb": vw_fb},
        "adaptive_hit_at_1": round(h1, 4),
        "delta_vs_static_best": {
            "mean": round(mean_d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        },
        "useful": useful,
        "n_queries": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.inp).read_text())
    sweep = data["sweep"]
    weights = [s["vector_weight"] for s in sweep]

    # Build: query -> {vw -> per_query_record}
    by_q: dict[str, dict[float, dict]] = defaultdict(dict)
    bm25_by_q: dict[str, dict] = {}
    for s in sweep:
        vw = s["vector_weight"]
        for r in s["baseline_per_query"]:
            by_q[r["query"]][vw] = r
        for r in s["bm25_only_per_query"]:
            bm25_by_q[r["query"]] = r

    queries = list(by_q.keys())
    n = len(queries)
    print(f"queries: {n}, weights: {weights}")

    # Static baseline: each fixed vw
    print("\n=== Static fixed-vw policies ===")
    for vw in weights:
        h1 = sum(by_q[q][vw]["hit_at_1"] for q in queries) / n
        mrr = sum(by_q[q][vw]["reciprocal_rank"] for q in queries) / n
        print(f"  vw={vw}: hit@1={h1:.3f} MRR={mrr:.3f}")

    bm25_h1 = sum(bm25_by_q[q]["hit_at_1"] for q in queries) / n
    bm25_mrr = sum(bm25_by_q[q]["reciprocal_rank"] for q in queries) / n
    print(f"  bm25_only: hit@1={bm25_h1:.3f} MRR={bm25_mrr:.3f}")

    # Oracle: pick best vw per query (upper bound on adaptive)
    print("\n=== Oracle (per-query best, upper bound) ===")
    oracle_h1 = 0
    oracle_mrr = 0.0
    for q in queries:
        best_h1 = max(by_q[q][vw]["hit_at_1"] for vw in weights)
        best_mrr = max(by_q[q][vw]["reciprocal_rank"] for vw in weights)
        # also include bm25_only
        best_h1 = max(best_h1, bm25_by_q[q]["hit_at_1"])
        best_mrr = max(best_mrr, bm25_by_q[q]["reciprocal_rank"])
        oracle_h1 += best_h1
        oracle_mrr += best_mrr
    print(f"  oracle: hit@1={oracle_h1/n:.3f} MRR={oracle_mrr/n:.3f}")

    # Threshold policy: trust BM25 when its rank-1 hit; else fall back to vw=0.5
    # We don't have raw FTS scores here, so use bm25 reciprocal_rank as a proxy:
    # if bm25 rr >= tau (i.e. bm25 ranked correct candidate well), use bm25_only;
    # else use a vw fallback.
    print("\n=== Threshold policies (proxy = bm25 reciprocal_rank) ===")
    for tau in [0.5, 0.34, 0.25, 0.2, 0.0]:
        for vw_fallback in weights:
            if vw_fallback == 0.0:
                continue
            h1 = 0
            mrr = 0.0
            n_fb = 0
            for q in queries:
                if bm25_by_q[q]["reciprocal_rank"] >= tau:
                    h1 += bm25_by_q[q]["hit_at_1"]
                    mrr += bm25_by_q[q]["reciprocal_rank"]
                else:
                    h1 += by_q[q][vw_fallback]["hit_at_1"]
                    mrr += by_q[q][vw_fallback]["reciprocal_rank"]
                    n_fb += 1
            print(
                f"  tau={tau} vw_fb={vw_fallback}: hit@1={h1/n:.3f} MRR={mrr/n:.3f} "
                f"(fallback used {n_fb}/{n} = {n_fb/n:.0%})"
            )

    # Non-leaky threshold policy: switch on bm25_top1/top2 gap (must be present).
    has_gap = all(bm25_by_q[q].get("bm25_gap") is not None for q in queries)
    print("\n=== Non-leaky threshold policies (signal = bm25_top1 - bm25_top2 gap) ===")
    if not has_gap:
        print("  (skipped — bm25_gap missing; re-run sweep with updated ablation.py)")
        return
    gaps = sorted(bm25_by_q[q]["bm25_gap"] for q in queries)
    print(f"  gap distribution: min={gaps[0]:.3f} q25={gaps[n//4]:.3f} "
          f"med={gaps[n//2]:.3f} q75={gaps[3*n//4]:.3f} max={gaps[-1]:.3f}")
    # Sweep tau over gap quantiles; rule: gap >= tau -> trust bm25_only,
    # else fall back to vw_fallback.
    for tau_q in [0.0, 0.25, 0.5, 0.75]:
        tau = gaps[min(int(tau_q * n), n - 1)]
        for vw_fallback in [w for w in weights if w > 0]:
            h1 = 0
            mrr = 0.0
            n_fb = 0
            for q in queries:
                if bm25_by_q[q]["bm25_gap"] >= tau:
                    h1 += bm25_by_q[q]["hit_at_1"]
                    mrr += bm25_by_q[q]["reciprocal_rank"]
                else:
                    h1 += by_q[q][vw_fallback]["hit_at_1"]
                    mrr += by_q[q][vw_fallback]["reciprocal_rank"]
                    n_fb += 1
            print(
                f"  tau_q={tau_q} (tau={tau:.3f}) vw_fb={vw_fallback}: "
                f"hit@1={h1/n:.3f} MRR={mrr/n:.3f} "
                f"(fallback used {n_fb}/{n} = {n_fb/n:.0%})"
            )

    # Try alternate signals derived from existing per_query data:
    # (1) normalized_gap = (top1 - top2) / top1  (computed post-hoc here so we
    #     don't need to re-run sweeps on data that predates the analyzer change).
    # (2) crowdedness signal: count of bm25 candidates with score >= 0.95*top1.
    #     Requires --save-per-query data with bm25_crowd_95 captured (newer
    #     sweeps only). Skipped silently when missing.
    from evals._signals import normalized_gap as _norm

    print("\n=== Non-leaky threshold policies (signal = normalized gap = (top1-top2)/top1, computed post-hoc) ===")
    norm_gaps_by_q: dict[str, float] = {}
    n_norm_valid = 0
    for q in queries:
        ng = _norm(bm25_by_q[q].get("bm25_top1"), bm25_by_q[q].get("bm25_top2"))
        if ng is not None:
            norm_gaps_by_q[q] = ng
            n_norm_valid += 1
    if n_norm_valid < n:
        print(f"  ({n - n_norm_valid}/{n} queries have undefined normalized gap (top1==0); treated as ng=0 for routing)")
    sorted_ng = sorted(norm_gaps_by_q.values()) or [0.0]
    m = len(sorted_ng)
    print(f"  norm_gap distribution: min={sorted_ng[0]:.3f} q25={sorted_ng[m//4]:.3f} "
          f"med={sorted_ng[m//2]:.3f} q75={sorted_ng[3*m//4]:.3f} max={sorted_ng[-1]:.3f}")
    for tau_q in [0.0, 0.25, 0.5, 0.75]:
        tau = sorted_ng[min(int(tau_q * m), m - 1)]
        for vw_fallback in [w for w in weights if w > 0]:
            h1 = 0
            mrr = 0.0
            n_fb = 0
            for q in queries:
                ng = norm_gaps_by_q.get(q, 0.0)
                if ng >= tau:
                    h1 += bm25_by_q[q]["hit_at_1"]
                    mrr += bm25_by_q[q]["reciprocal_rank"]
                else:
                    h1 += by_q[q][vw_fallback]["hit_at_1"]
                    mrr += by_q[q][vw_fallback]["reciprocal_rank"]
                    n_fb += 1
            print(
                f"  tau_q={tau_q} (tau={tau:.3f}) vw_fb={vw_fallback}: "
                f"hit@1={h1/n:.3f} MRR={mrr/n:.3f} "
                f"(fallback used {n_fb}/{n} = {n_fb/n:.0%})"
            )

    # Crowdedness: only available on sweeps captured after this run.
    has_crowd = all(bm25_by_q[q].get("bm25_crowd_95") is not None for q in queries)
    if has_crowd:
        print("\n=== Non-leaky threshold policies (signal = crowdedness@0.95, lower = more confident BM25) ===")
        crowds = sorted(bm25_by_q[q]["bm25_crowd_95"] for q in queries)
        print(f"  crowd@0.95 distribution: min={crowds[0]} q25={crowds[n//4]} "
              f"med={crowds[n//2]} q75={crowds[3*n//4]} max={crowds[-1]}")
        # Lower crowdedness = trust bm25; rule: crowd <= tau -> bm25_only
        for tau_q in [0.25, 0.5, 0.75, 1.0]:
            tau = crowds[min(int(tau_q * n), n - 1)]
            for vw_fallback in [w for w in weights if w > 0]:
                h1 = 0
                mrr = 0.0
                n_fb = 0
                for q in queries:
                    if bm25_by_q[q]["bm25_crowd_95"] <= tau:
                        h1 += bm25_by_q[q]["hit_at_1"]
                        mrr += bm25_by_q[q]["reciprocal_rank"]
                    else:
                        h1 += by_q[q][vw_fallback]["hit_at_1"]
                        mrr += by_q[q][vw_fallback]["reciprocal_rank"]
                        n_fb += 1
                print(
                    f"  tau_q={tau_q} (tau={tau}) vw_fb={vw_fallback}: "
                    f"hit@1={h1/n:.3f} MRR={mrr/n:.3f} "
                    f"(fallback used {n_fb}/{n} = {n_fb/n:.0%})"
                )
    else:
        print("\n=== Crowdedness signal: skipped (re-run sweep with updated ablation.py to capture bm25_crowd_95) ===")

    # === Verdict block: paired bootstrap CIs on (adaptive - static_best) ===
    print("\n=== Signal verdict (paired bootstrap, 95% CI on Δhit@1 vs static-best) ===")
    # Determine static-best vw by mean hit@1 (including bm25_only as vw=0).
    static_means = {0.0: bm25_h1}
    for vw in weights:
        if vw == 0.0:
            continue
        static_means[vw] = sum(by_q[q][vw]["hit_at_1"] for q in queries) / n
    best_vw = max(static_means, key=static_means.get)
    print(f"  static-best: vw={best_vw} hit@1={static_means[best_vw]:.4f}")
    if best_vw == 0.0:
        static_best_per_q = [bm25_by_q[q]["hit_at_1"] for q in queries]
    else:
        static_best_per_q = [by_q[q][best_vw]["hit_at_1"] for q in queries]

    verdicts = []
    # Signal 1: raw bm25 gap
    if all(bm25_by_q[q].get("bm25_gap") is not None for q in queries):
        sig = {q: bm25_by_q[q]["bm25_gap"] for q in queries}
        verdicts.append(_signal_verdict(queries, by_q, bm25_by_q, weights,
                                        sig, static_best_per_q, "raw_gap"))
    # Signal 2: normalized gap (post-hoc)
    sig_ng: dict[str, float] = {}
    from evals._signals import normalized_gap as _norm
    for q in queries:
        ng = _norm(bm25_by_q[q].get("bm25_top1"), bm25_by_q[q].get("bm25_top2"))
        sig_ng[q] = ng if ng is not None else 0.0
    verdicts.append(_signal_verdict(queries, by_q, bm25_by_q, weights,
                                    sig_ng, static_best_per_q, "normalized_gap"))
    # Signal 3: crowdedness — invert to "lower=trust", so use -crowd as signal.
    if all(bm25_by_q[q].get("bm25_crowd_95") is not None for q in queries):
        sig_c = {q: -float(bm25_by_q[q]["bm25_crowd_95"]) for q in queries}
        verdicts.append(_signal_verdict(queries, by_q, bm25_by_q, weights,
                                        sig_c, static_best_per_q, "neg_crowdedness_95"))
    for v in verdicts:
        if v.get("verdict") == "no-data":
            print(f"  {v['signal']}: no-data")
            continue
        d = v["delta_vs_static_best"]
        flag = "✅ USEFUL" if v["useful"] else "❌ not useful"
        print(f"  {v['signal']:>22s}: best_vw_fb={v['best_cell']['vw_fb']} "
              f"tau_q={v['best_cell']['tau_q']:.2f}  "
              f"Δh1={d['mean']:+.4f} CI=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  {flag}")


if __name__ == "__main__":
    main()

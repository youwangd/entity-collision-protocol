"""Offline adaptive vector_weight analysis on LoCoMo per-query data.

Mirrors evals.adaptive_vw_offline (synthetic) but reads the LoCoMo schema:
per_query entries are keyed by (sample_id, category, q_idx) and join
across vw cells by that triple. The "bm25-only" stream is the vw=0.0 cell.

Inputs: a list of LoCoMo sweep result JSONs (one per vw cell), each
written by `evals.locomo_sweep_ci --save-bm25-signals`. The bm25 signals
must be present on at least the vw=0.0 cell (the routing signal).

Tests if a non-leaky signal — bm25 raw gap, normalized gap, or
crowdedness — can route per-query between BM25-only (vw=0) and a
fixed vw_fb to beat the static-best vw, with a paired bootstrap CI.

Usage:
    python -m evals.locomo_adaptive_vw \\
        --in bench/results/locomo10_st_ci_vw0.0.json \\
        --in bench/results/locomo10_st_ci_vw0.3.json \\
        --in bench/results/locomo10_st_ci_vw0.5.json \\
        --in bench/results/locomo10_st_ci_vw0.7.json \\
        --in bench/results/locomo10_st_ci_vw1.0.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _paired_bootstrap_ci(
    deltas, resamples: int = 2000, seed: int = 13, alpha: float = 0.05
) -> tuple[float, float, float]:
    # Vectorized when numpy is available (much faster for n>200).
    try:
        import numpy as np  # type: ignore
        d = np.asarray(deltas, dtype=float)
        n = len(d)
        if n == 0:
            return 0.0, 0.0, 0.0
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(resamples, n))
        means = d[idx].mean(axis=1)
        means.sort()
        lo = float(means[int(math.floor((alpha / 2) * resamples))])
        hi_idx = min(int(math.ceil((1 - alpha / 2) * resamples)) - 1, resamples - 1)
        return float(d.mean()), lo, float(means[hi_idx])
    except Exception:
        pass
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


def _key(r: dict) -> tuple:
    return (r["sample_id"], str(r["category"]), int(r.get("q_idx", -1)))


def _signal_verdict(
    keys: list[tuple],
    by_k: dict,           # key -> {vw -> per_query_record}
    bm25_by_k: dict,      # key -> per_query_record (vw=0)
    weights: list[float],
    sig: dict,            # key -> float
    static_best_per_q: list[int],  # hit@1
    metric: str,          # "hit_at_1", "hit_at_k", "reciprocal_rank"
    label: str,
) -> dict:
    n = len(keys)
    sorted_sig = sorted(sig.values()) or [0.0]
    m = len(sorted_sig)
    best = None
    for tau_q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        tau = sorted_sig[min(int(tau_q * m), m - 1)]
        for vw_fb in [w for w in weights if w > 0]:
            adaptive_per_q = []
            for k in keys:
                v = sig.get(k, 0.0)
                if v >= tau:
                    adaptive_per_q.append(bm25_by_k[k][metric])
                else:
                    adaptive_per_q.append(by_k[k][vw_fb][metric])
            deltas = [a - s for a, s in zip(adaptive_per_q, static_best_per_q)]
            mean_d, lo, hi = _paired_bootstrap_ci(deltas)
            score = sum(adaptive_per_q) / n
            cell = (tau_q, tau, vw_fb, mean_d, lo, hi, score)
            if best is None or mean_d > best[3]:
                best = cell
    if best is None:
        return {"signal": label, "verdict": "no-data"}
    tau_q, tau, vw_fb, mean_d, lo, hi, score = best
    return {
        "signal": label,
        "metric": metric,
        "best_cell": {"tau_q": tau_q, "tau": tau, "vw_fb": vw_fb},
        f"adaptive_{metric}": round(score, 4),
        "delta_vs_static_best": {
            "mean": round(mean_d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        },
        "useful": lo > 0,
        "n_queries": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", action="append", required=True,
                   help="LoCoMo sweep result JSON (one per vw); pass --in repeatedly")
    p.add_argument("--metric", default="hit_at_1",
                   choices=["hit_at_1", "hit_at_k", "reciprocal_rank"])
    args = p.parse_args()

    cells = [json.loads(Path(x).read_text()) for x in args.inp]
    cells.sort(key=lambda c: float(c["vector_weight"]))
    weights = [float(c["vector_weight"]) for c in cells]

    # Build by_k[key][vw] = record. If q_idx is missing on legacy cells,
    # synthesize one from the per_query order within (sample_id, category)
    # so the join is still deterministic.
    by_k: dict[tuple, dict[float, dict]] = {}
    for c in cells:
        vw = float(c["vector_weight"])
        seen_per_pair: dict[tuple, int] = {}
        for r in c["per_query"]:
            sid = r["sample_id"]
            cat = str(r["category"])
            if "q_idx" in r:
                qi = int(r["q_idx"])
            else:
                qi = seen_per_pair.get((sid, cat), 0)
                seen_per_pair[(sid, cat)] = qi + 1
            k = (sid, cat, qi)
            by_k.setdefault(k, {})[vw] = r

    # Filter to keys present in all cells
    keys = [k for k, vmap in by_k.items() if all(w in vmap for w in weights)]
    n = len(keys)
    if n == 0:
        raise SystemExit("no overlapping keys across vw cells; check q_idx joins")
    if 0.0 not in weights:
        raise SystemExit("need vw=0.0 cell as the BM25-only stream")
    bm25_by_k = {k: by_k[k][0.0] for k in keys}
    print(f"queries: {n}, vws: {weights}, metric: {args.metric}")

    # Static fixed-vw policies
    print("\n=== Static fixed-vw policies ===")
    static_means: dict[float, float] = {}
    for vw in weights:
        s = sum(by_k[k][vw][args.metric] for k in keys) / n
        static_means[vw] = s
        print(f"  vw={vw}: {args.metric}={s:.4f}")

    # Oracle
    oracle = sum(max(by_k[k][vw][args.metric] for vw in weights) for k in keys) / n
    print(f"\n=== Oracle (per-query best vw, upper bound) ===\n  oracle {args.metric}={oracle:.4f}")

    # Static best
    best_vw = max(static_means, key=static_means.get)
    static_best_per_q = [by_k[k][best_vw][args.metric] for k in keys]
    print(f"\n=== Static-best baseline: vw={best_vw} {args.metric}={static_means[best_vw]:.4f} ===")

    # Signals: read from bm25-only (vw=0) record. Treat missing values as
    # zero-confidence (gap=0, ng=0, crowd=large). Some queries return zero
    # FTS hits (BM25 scored nothing) — those land in the "trust BM25 less"
    # bucket of every signal, which is the right semantics.
    def _has_any(field):
        return any(bm25_by_k[k].get(field) is not None for k in keys)

    has_gap = _has_any("bm25_gap")
    has_norm = _has_any("bm25_norm_gap")
    has_crowd = _has_any("bm25_crowd_95")
    if not (has_gap or has_norm or has_crowd):
        print("\n(no BM25 signals on vw=0 cell; re-run sweep with --save-bm25-signals)")
        return

    n_missing = sum(1 for k in keys if bm25_by_k[k].get("bm25_gap") is None)
    if n_missing:
        print(f"  ({n_missing}/{n} queries had no BM25 hits — treated as zero-confidence)")

    verdicts = []
    if has_gap:
        sig = {k: float(bm25_by_k[k].get("bm25_gap") or 0.0) for k in keys}
        verdicts.append(_signal_verdict(keys, by_k, bm25_by_k, weights,
                                        sig, static_best_per_q, args.metric, "raw_gap"))
    if has_norm:
        sig = {k: float(bm25_by_k[k].get("bm25_norm_gap") or 0.0) for k in keys}
        verdicts.append(_signal_verdict(keys, by_k, bm25_by_k, weights,
                                        sig, static_best_per_q, args.metric, "normalized_gap"))
    if has_crowd:
        # Lower crowdedness == more BM25 confidence; signal is -crowd.
        # Missing => treat as very large crowd (low confidence) => use vw fallback.
        max_c = max(
            (int(bm25_by_k[k]["bm25_crowd_95"]) for k in keys
             if bm25_by_k[k].get("bm25_crowd_95") is not None),
            default=0,
        )
        sig = {
            k: -float(bm25_by_k[k]["bm25_crowd_95"]) if bm25_by_k[k].get("bm25_crowd_95") is not None
            else -float(max_c + 1)
            for k in keys
        }
        verdicts.append(_signal_verdict(keys, by_k, bm25_by_k, weights,
                                        sig, static_best_per_q, args.metric, "neg_crowdedness_95"))

    print("\n=== Signal verdict (paired bootstrap, 95% CI on Δ vs static-best) ===")
    for v in verdicts:
        d = v["delta_vs_static_best"]
        flag = "✅ USEFUL" if v["useful"] else "❌ not useful"
        print(f"  {v['signal']:>22s}: vw_fb={v['best_cell']['vw_fb']} "
              f"tau_q={v['best_cell']['tau_q']:.2f}  "
              f"Δ={d['mean']:+.4f} CI=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  {flag}")


if __name__ == "__main__":
    main()

"""Offline learned-router analysis on LoCoMo per-query data.

Closes the open question in NEXT.md: binary tau-routing on a single BM25
signal is null. Does a multi-feature classifier over per-query BM25
features recover any of the oracle headroom?

Setup
-----
Inputs are the same LoCoMo sweep result JSONs as `locomo_adaptive_vw`,
written by `evals.locomo_sweep_ci --save-bm25-signals`. Every cell
provides a per-query record keyed by (sample_id, category, q_idx); the
vw=0.0 cell additionally provides BM25 signals (bm25_top1, bm25_top2,
bm25_gap, bm25_norm_gap, bm25_crowd_95, rank, reciprocal_rank).

Features per query (all observable at routing time, before vector mix):
    bm25_top1, bm25_top2, bm25_gap, bm25_norm_gap, bm25_crowd_95,
    vw0_rank, vw0_reciprocal_rank, no_bm25_hits indicator,
    + LoCoMo category one-hot.

Target per query: argmax_{vw in cells} metric. Ties broken by smallest
vw (parsimony — prefer cheaper retrieval).

CV
--
Leave-one-conversation-out across the 10 LoCoMo samples (sample_id ==
conversation). Train on 9, predict on 1, concatenate predictions across
folds — every query is held out exactly once.

Verdict
-------
Paired bootstrap CI on (router_per_q - static_best_per_q) across the
held-out predictions. CI strictly above 0 == useful router.

Usage
-----
    python -m evals.locomo_learned_router --metric hit_at_1 \\
        --in bench/results/locomo10_ht_sig_vw0.0.json \\
        --in bench/results/locomo10_ht_sig_vw0.3.json \\
        --in bench/results/locomo10_ht_sig_vw0.5.json \\
        --in bench/results/locomo10_ht_sig_vw0.7.json \\
        --in bench/results/locomo10_ht_sig_vw1.0.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from evals.io_utils import atomic_write_json


def _paired_bootstrap_ci(
    deltas, resamples: int = 2000, seed: int = 13, alpha: float = 0.05
) -> tuple[float, float, float]:
    try:
        import numpy as np
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
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * resamples))]
    hi_idx = min(int(math.ceil((1 - alpha / 2) * resamples)) - 1, resamples - 1)
    return sum(deltas) / n, lo, means[hi_idx]


def _build_join(cells):
    by_k: dict[tuple, dict[float, dict]] = {}
    weights = sorted({float(c["vector_weight"]) for c in cells})
    for c in cells:
        vw = float(c["vector_weight"])
        seen: dict[tuple, int] = {}
        for r in c["per_query"]:
            sid = r["sample_id"]
            cat = str(r["category"])
            if "q_idx" in r:
                qi = int(r["q_idx"])
            else:
                qi = seen.get((sid, cat), 0)
                seen[(sid, cat)] = qi + 1
            by_k.setdefault((sid, cat, qi), {})[vw] = r
    keys = sorted(k for k, vmap in by_k.items() if all(w in vmap for w in weights))
    return keys, by_k, weights


def _features(rec_vw0, cat_list):
    """Leak-free features: only signals computable BEFORE knowing gold.

    rank/reciprocal_rank are EXCLUDED — they're computed against gold
    evidence sessions and would leak the answer at routing time.
    """
    g = lambda fld: rec_vw0.get(fld)
    has = g("bm25_top1") is not None
    feats = [
        float(g("bm25_top1") or 0.0),
        float(g("bm25_top2") or 0.0),
        float(g("bm25_gap") or 0.0),
        float(g("bm25_norm_gap") or 0.0),
        float(g("bm25_crowd_95") or 0.0),
        0.0 if has else 1.0,
    ]
    cat = str(rec_vw0.get("category", ""))
    feats.extend(1.0 if cat == c else 0.0 for c in cat_list)
    return feats


_FEAT_NAMES_BASE = [
    "bm25_top1", "bm25_top2", "bm25_gap", "bm25_norm_gap",
    "bm25_crowd_95", "no_bm25_hits",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", action="append", required=True)
    p.add_argument("--metric", default="hit_at_1",
                   choices=["hit_at_1", "hit_at_k", "reciprocal_rank"])
    p.add_argument("--model", default="gbm",
                   choices=["gbm", "logreg"])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    cells = [json.loads(Path(x).read_text()) for x in args.inp]
    keys, by_k, weights = _build_join(cells)
    if 0.0 not in weights:
        raise SystemExit("need vw=0.0 cell (BM25 signals + bm25-only stream)")
    print(f"queries: {len(keys)}, vws: {weights}, metric: {args.metric}, model: {args.model}")

    n = len(keys)
    static_means = {vw: sum(by_k[k][vw][args.metric] for k in keys) / n for vw in weights}
    print("\n=== Static fixed-vw policies ===")
    for vw, s in static_means.items():
        print(f"  vw={vw}: {args.metric}={s:.4f}")
    oracle = sum(max(by_k[k][vw][args.metric] for vw in weights) for k in keys) / n
    best_vw = max(static_means, key=static_means.get)
    static_best_per_q = [by_k[k][best_vw][args.metric] for k in keys]
    print(f"\n=== Oracle: {oracle:.4f}   Static-best: vw={best_vw} {static_means[best_vw]:.4f} ===")

    cat_list = sorted({k[1] for k in keys})
    bm25_by_k = {k: by_k[k][0.0] for k in keys}
    X = [_features(bm25_by_k[k], cat_list) for k in keys]
    y = []
    for k in keys:
        best = -1.0
        best_vw_q = weights[0]
        for vw in weights:
            v = by_k[k][vw][args.metric]
            if v > best:
                best, best_vw_q = v, vw
        y.append(best_vw_q)

    sample_ids = [k[0] for k in keys]
    unique_sids = sorted(set(sample_ids))
    print(f"\n=== LOCO CV across {len(unique_sids)} conversations ===")
    print(f"  target distribution: {dict(sorted(Counter(y).items()))}")

    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except Exception as e:
        raise SystemExit(f"sklearn required: {e}")

    X_arr = np.asarray(X, dtype=float)
    # sklearn classifiers reject float targets; encode vw as string labels.
    y_arr = np.asarray([f"{float(v):.3f}" for v in y])
    sid_arr = np.asarray(sample_ids)

    pred_vw = np.zeros(n, dtype=float)
    feat_imps_acc = np.zeros(X_arr.shape[1], dtype=float)
    n_folds_with_imp = 0

    for sid in unique_sids:
        test_mask = (sid_arr == sid)
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if args.model == "gbm":
            clf = GradientBoostingClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.08,
                random_state=args.seed,
            )
            clf.fit(X_arr[train_mask], y_arr[train_mask])
            pred_vw[test_mask] = [float(p) for p in clf.predict(X_arr[test_mask])]
            feat_imps_acc += clf.feature_importances_
            n_folds_with_imp += 1
        else:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, random_state=args.seed)),
            ])
            pipe.fit(X_arr[train_mask], y_arr[train_mask])
            pred_vw[test_mask] = [float(p) for p in pipe.predict(X_arr[test_mask])]

    pred_dist = Counter(round(float(v), 3) for v in pred_vw)
    print(f"  predicted vw distribution: {dict(sorted(pred_dist.items()))}")

    adaptive_per_q = [by_k[keys[i]][float(pred_vw[i])][args.metric] for i in range(n)]
    adaptive_mean = sum(adaptive_per_q) / n
    deltas = [a - s for a, s in zip(adaptive_per_q, static_best_per_q)]
    mean_d, lo, hi = _paired_bootstrap_ci(deltas, seed=args.seed)
    flag = "USEFUL" if lo > 0 else "not useful"

    print("\n=== Learned router verdict ===")
    print(f"  adaptive {args.metric} = {adaptive_mean:.4f}  (static-best {static_means[best_vw]:.4f}, oracle {oracle:.4f})")
    print(f"  delta vs static-best = {mean_d:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {flag}")
    if n_folds_with_imp:
        avg_imp = feat_imps_acc / n_folds_with_imp
        feat_names = list(_FEAT_NAMES_BASE) + [f"cat_{c}" for c in cat_list]
        order = np.argsort(avg_imp)[::-1]
        print("  top features by mean GBM importance:")
        for i in order[:8]:
            print(f"    {feat_names[i]:>16s}  {avg_imp[i]:.3f}")

    if args.out:
        artifact = {
            "metric": args.metric,
            "model": args.model,
            "n_queries": n,
            "weights": weights,
            "static_means": {str(k): v for k, v in static_means.items()},
            "static_best_vw": best_vw,
            "static_best_mean": static_means[best_vw],
            "oracle_mean": oracle,
            "adaptive_mean": adaptive_mean,
            "delta_vs_static_best": {"mean": mean_d, "ci_lo": lo, "ci_hi": hi},
            "predicted_vw_dist": {str(k): v for k, v in pred_dist.items()},
            "useful": lo > 0,
            "n_conversations": len(unique_sids),
        }
        atomic_write_json(args.out, artifact)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

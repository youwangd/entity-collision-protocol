"""§D15c-mechanism — sweep hard_distractors_per_fact and measure Δh@1.

Tests the hypothesis from NEXT.md: the §D15c regression of `gated_pref`
on the synthetic preference corpus is driven by hard-distractor density
(entity-aligned distractors that share tokens with the answer fact).
If true, Δh@1 should slope monotonically negative as density rises.

Single seed, single n; this is mechanism, not power. Paired baseline vs
gated_pref over the SAME corpus per density point.

Usage:
    python -m evals.synth_pref_density_sweep \
        --n-facts 240 --seed 42 --k 10 \
        --densities 0,1,2,3,4,5 \
        --out evals/results/synth_pref_density_sweep.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from engram import Engram, Config

from .metrics import find_match_rank
from .synthetic import generate_preference_dataset
from evals.io_utils import atomic_write_json


def _baseline_cfg(path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    return cfg


def _gated_pref_cfg(path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_type_allow = frozenset(
        {"single-session-preference"}
    )
    return cfg


def _run_arm(arm: str, ds, k: int):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _baseline_cfg(tmp) if arm == "baseline" else _gated_pref_cfg(tmp)
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0
            h1, hk, rr = [], [], []
            for q in ds.queries:
                results = eng.recall(q.text, limit=k)
                rank = find_match_rank(results, q.expected_substrings)
                h1.append(1 if (rank is not None and rank < 1) else 0)
                hk.append(1 if (rank is not None and rank < k) else 0)
                rr.append(0.0 if rank is None else 1.0 / (rank + 1))
            return h1, hk, rr, ingest_s
        finally:
            eng.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=240)
    p.add_argument("--distractors-per-fact", type=int, default=6)
    p.add_argument(
        "--densities",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated hard_distractors_per_fact values",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    densities = [int(x) for x in args.densities.split(",") if x.strip()]
    points = []
    for d in densities:
        print(f"[density-sweep] d={d} building corpus...", flush=True)
        ds = generate_preference_dataset(
            n_facts=args.n_facts,
            distractors_per_fact=args.distractors_per_fact,
            hard_distractors_per_fact=d,
            seed=args.seed,
        )
        b_h1, b_hk, b_rr, b_ingest = _run_arm("baseline", ds, args.k)
        t_h1, t_hk, t_rr, t_ingest = _run_arm("gated_pref", ds, args.k)
        b_h1m = statistics.mean(b_h1)
        t_h1m = statistics.mean(t_h1)
        delta = t_h1m - b_h1m
        # paired-diff std for an informal SE (no CI here; mechanism scan)
        diffs = [a - b for a, b in zip(t_h1, b_h1)]
        se = (statistics.pstdev(diffs) / (len(diffs) ** 0.5)) if len(diffs) > 1 else 0.0
        point = {
            "density": d,
            "n_memories": len(ds.memories),
            "n_queries": len(ds.queries),
            "baseline_h1": round(b_h1m, 4),
            "gated_pref_h1": round(t_h1m, 4),
            "delta_h1": round(delta, 4),
            "delta_se": round(se, 4),
            "baseline_hk": round(statistics.mean(b_hk), 4),
            "gated_pref_hk": round(statistics.mean(t_hk), 4),
            "baseline_mrr": round(statistics.mean(b_rr), 4),
            "gated_pref_mrr": round(statistics.mean(t_rr), 4),
        }
        points.append(point)
        print(
            f"[density-sweep] d={d} base_h1={b_h1m:.4f} treat_h1={t_h1m:.4f} "
            f"Δ={delta:+.4f} (±{se:.4f})",
            flush=True,
        )

    out = {
        "n_facts": args.n_facts,
        "seed": args.seed,
        "k": args.k,
        "densities": densities,
        "points": points,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

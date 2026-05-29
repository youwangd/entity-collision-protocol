"""§D15c-mech — sweep ``answer_anchor_tokens`` and measure Δh@1.

Tests the remaining §D15c-mechanism hypothesis: PRF (the ``gated_pref``
arm) regresses on the synthetic preference corpus because the
single-token answer anchor is BM25-dominant, and PRF dilutes its weight
by introducing additional expansion terms.

Prediction (defensible if true):
    Δh@1(gated_pref − baseline) → 0 (or positive) as
    ``answer_anchor_tokens`` rises from 1 → 2 → 3.

If Δh@1 stays negative across all anchor-token counts, the
single-token-anchor mechanism is falsified and we look elsewhere
(e.g. type-classifier coverage, query-side template-vs-fact
distribution shift).

Single seed, paired baseline vs gated_pref over the SAME corpus per
anchor-token point. Mechanism scan; CIs handled separately if a point
warrants it.

Usage::

    python -m evals.synth_pref_anchor_tokens_sweep \\
        --n-facts 240 --seed 42 --k 10 \\
        --anchor-tokens 1,2,3 \\
        --out evals/results/synth_pref_anchor_tokens_sweep.json
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
    p.add_argument("--hard-distractors-per-fact", type=int, default=3)
    p.add_argument(
        "--anchor-tokens",
        type=str,
        default="1,2,3",
        help="Comma-separated answer_anchor_tokens values to sweep",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    tok_values = [int(x) for x in args.anchor_tokens.split(",") if x.strip()]
    points = []
    for tok in tok_values:
        print(f"[anchor-toks] tok={tok} building corpus...", flush=True)
        ds = generate_preference_dataset(
            n_facts=args.n_facts,
            distractors_per_fact=args.distractors_per_fact,
            hard_distractors_per_fact=args.hard_distractors_per_fact,
            seed=args.seed,
            answer_anchor_tokens=tok,
        )
        b_h1, b_hk, b_rr, b_ingest = _run_arm("baseline", ds, args.k)
        t_h1, t_hk, t_rr, t_ingest = _run_arm("gated_pref", ds, args.k)
        b_h1m = statistics.mean(b_h1)
        t_h1m = statistics.mean(t_h1)
        delta = t_h1m - b_h1m
        diffs = [a - b for a, b in zip(t_h1, b_h1)]
        se = (
            (statistics.pstdev(diffs) / (len(diffs) ** 0.5))
            if len(diffs) > 1 else 0.0
        )
        point = {
            "answer_anchor_tokens": tok,
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
            f"[anchor-toks] tok={tok} base_h1={b_h1m:.4f} "
            f"treat_h1={t_h1m:.4f} Δ={delta:+.4f} (±{se:.4f})",
            flush=True,
        )

    out = {
        "n_facts": args.n_facts,
        "seed": args.seed,
        "k": args.k,
        "hard_distractors_per_fact": args.hard_distractors_per_fact,
        "anchor_tokens": tok_values,
        "points": points,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

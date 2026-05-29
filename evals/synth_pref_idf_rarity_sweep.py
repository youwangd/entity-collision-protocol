"""§4.15g — sweep ``query_expansion_idf_min_rarity`` over the synthetic
preference corpus to test if the IDF-rarity filter rescues PRF from the
§D15c multi-token-anchor regression.

Sweep design (paired-baseline; single seed; same corpus per point):

For each idf_min_rarity ∈ {None, 0.0, 0.3, 0.5, 0.7, 0.9}:
    Δh@1 = mean(gated_pref_with_idf - baseline) over identical queries.

Predicts:
    If the multi-token-anchor regression is driven by PRF appending
    corpus-common tokens, Δh@1 should rise (less negative or positive)
    as idf_min_rarity rises from 0 → 0.9. If Δh@1 stays flat-negative,
    the IDF-rarity hypothesis is falsified and v0.3 needs a different
    angle.

Default settings target the §D15c-mech-2 worst point: anchor-tokens=3.

Usage::

    python -m evals.synth_pref_idf_rarity_sweep \\
        --n-facts 240 --seed 42 --k 10 --anchor-tokens 3 \\
        --rarity-points None,0.0,0.3,0.5,0.7,0.9 \\
        --out evals/results/synth_pref_idf_rarity_sweep.json
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


def _gated_pref_cfg(path: str, idf_min_rarity) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_type_allow = frozenset(
        {"single-session-preference"}
    )
    cfg.retrieval.query_expansion_idf_min_rarity = idf_min_rarity
    return cfg


def _run_arm(arm: str, ds, k: int, idf_min_rarity):
    with tempfile.TemporaryDirectory() as tmp:
        if arm == "baseline":
            cfg = _baseline_cfg(tmp)
        else:
            cfg = _gated_pref_cfg(tmp, idf_min_rarity)
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0
            h1, hk = [], []
            for q in ds.queries:
                results = eng.recall(q.text, limit=k)
                rank = find_match_rank(results, q.expected_substrings)
                h1.append(1 if (rank is not None and rank < 1) else 0)
                hk.append(1 if (rank is not None and rank < k) else 0)
            return h1, hk, ingest_s
        finally:
            eng.close()


def _parse_rarity(s: str):
    s = s.strip()
    if s.lower() == "none":
        return None
    return float(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=240)
    p.add_argument("--distractors-per-fact", type=int, default=6)
    p.add_argument("--hard-distractors-per-fact", type=int, default=3)
    p.add_argument("--anchor-tokens", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--rarity-points",
        type=str,
        default="None,0.0,0.3,0.5,0.7,0.9",
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    rarity_values = [_parse_rarity(x) for x in args.rarity_points.split(",")]

    print(f"[idf-sweep] building corpus tok={args.anchor_tokens}...", flush=True)
    ds = generate_preference_dataset(
        n_facts=args.n_facts,
        distractors_per_fact=args.distractors_per_fact,
        hard_distractors_per_fact=args.hard_distractors_per_fact,
        seed=args.seed,
        answer_anchor_tokens=args.anchor_tokens,
    )
    # Baseline once — same per anchor-toks point.
    b_h1, b_hk, b_ingest = _run_arm("baseline", ds, args.k, None)
    b_h1m = statistics.mean(b_h1)
    print(f"[idf-sweep] baseline h1={b_h1m:.4f}", flush=True)

    points = []
    for r in rarity_values:
        t_h1, t_hk, t_ingest = _run_arm("gated_pref", ds, args.k, r)
        t_h1m = statistics.mean(t_h1)
        delta = t_h1m - b_h1m
        diffs = [a - b for a, b in zip(t_h1, b_h1)]
        se = (
            statistics.pstdev(diffs) / (len(diffs) ** 0.5)
            if len(diffs) > 1 else 0.0
        )
        point = {
            "idf_min_rarity": r,
            "baseline_h1": round(b_h1m, 4),
            "gated_pref_h1": round(t_h1m, 4),
            "delta_h1": round(delta, 4),
            "delta_se": round(se, 4),
            "baseline_hk": round(statistics.mean(b_hk), 4),
            "gated_pref_hk": round(statistics.mean(t_hk), 4),
            "ingest_s": round(t_ingest, 2),
        }
        points.append(point)
        print(
            f"[idf-sweep] r={r} t_h1={t_h1m:.4f} Δ={delta:+.4f} (±{se:.4f})",
            flush=True,
        )

    out = {
        "n_facts": args.n_facts,
        "seed": args.seed,
        "k": args.k,
        "anchor_tokens": args.anchor_tokens,
        "n_memories": len(ds.memories),
        "n_queries": len(ds.queries),
        "rarity_points": args.rarity_points,
        "points": points,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

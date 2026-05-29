"""§D15d — sweep ``query_expansion_anchor_share_max`` over the synthetic
preference corpus to test if the anchor-share gate rescues PRF from the
§D15c multi-token-anchor regression.

Sweep design (paired-baseline; single seed; same corpus per point):

For each anchor_share_max ∈ {None, 1.0, 0.7, 0.5, 0.4, 0.3}:
    Δh@1 = mean(gated_pref_with_share - baseline) over identical queries.

Predicts (per §D15c-mech-3):
    The shared-anchor saturation is the failure mode. Lower thresholds
    (e.g. 0.4) should short-circuit PRF on saturated queries, recovering
    Δh@1 toward 0 (i.e. PRF-off behaviour). When `anchor_share_max=1.0`
    the gate is effectively inert and we should reproduce the raw
    PRF regression. When `anchor_share_max=None`, identical to v0.2.

Default settings target the §D15c-mech-2 worst point: anchor-tokens=3.

Usage::

    python -m evals.synth_pref_anchor_share_sweep \\\\
        --n-facts 240 --seed 42 --k 10 --anchor-tokens 3 \\\\
        --share-points None,1.0,0.7,0.5,0.4,0.3 \\\\
        --out evals/results/synth_pref_anchor_share_sweep.json
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


def _gated_pref_cfg(path: str, anchor_share_max) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_type_allow = frozenset(
        {"single-session-preference"}
    )
    cfg.retrieval.query_expansion_anchor_share_max = anchor_share_max
    return cfg


def _run_arm(arm: str, ds, k: int, anchor_share_max):
    with tempfile.TemporaryDirectory() as tmp:
        if arm == "baseline":
            cfg = _baseline_cfg(tmp)
        else:
            cfg = _gated_pref_cfg(tmp, anchor_share_max)
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


def _parse_share(s: str):
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
        "--share-points",
        type=str,
        default="None,1.0,0.7,0.5,0.4,0.3",
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    share_values = [_parse_share(x) for x in args.share_points.split(",")]

    print(
        f"[anchor-share-sweep] building corpus tok={args.anchor_tokens}...",
        flush=True,
    )
    ds = generate_preference_dataset(
        n_facts=args.n_facts,
        distractors_per_fact=args.distractors_per_fact,
        hard_distractors_per_fact=args.hard_distractors_per_fact,
        seed=args.seed,
        answer_anchor_tokens=args.anchor_tokens,
    )
    b_h1, b_hk, b_ingest = _run_arm("baseline", ds, args.k, None)
    b_h1m = statistics.mean(b_h1)
    print(f"[anchor-share-sweep] baseline h1={b_h1m:.4f}", flush=True)

    points = []
    for s in share_values:
        t_h1, t_hk, t_ingest = _run_arm("gated_pref", ds, args.k, s)
        t_h1m = statistics.mean(t_h1)
        delta = t_h1m - b_h1m
        diffs = [a - b for a, b in zip(t_h1, b_h1)]
        se = (
            statistics.pstdev(diffs) / (len(diffs) ** 0.5)
            if len(diffs) > 1 else 0.0
        )
        point = {
            "anchor_share_max": s,
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
            f"[anchor-share-sweep] s={s} t_h1={t_h1m:.4f} "
            f"Δ={delta:+.4f} (±{se:.4f})",
            flush=True,
        )

    out = {
        "n_facts": args.n_facts,
        "seed": args.seed,
        "k": args.k,
        "anchor_tokens": args.anchor_tokens,
        "n_memories": len(ds.memories),
        "n_queries": len(ds.queries),
        "share_points": args.share_points,
        "points": points,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

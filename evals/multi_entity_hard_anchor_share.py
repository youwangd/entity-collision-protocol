"""§D15d — non-pathological inertness check for the anchor-share gate.

On the multi-entity-hard corpus (high lexical collision, high NER
disambiguation), PRF historically shows positive lift. The anchor-share
gate is meant to short-circuit only the §D15c saturated regime; on a
non-saturated corpus, varying the threshold should be approximately
inert.

Compares two PRF configurations:
  - prf_only:                anchor_share_max=None
  - prf_with_anchor_gate_S:  anchor_share_max=S (for several S values)

Reports hit@1 / hit@5 / hit@10 vs baseline and Δ vs prf_only.

Usage::

    python -m evals.multi_entity_hard_anchor_share --n-facts 500 \\\\
        --n-sessions 25 --share-points None,0.7,0.5,0.4,0.3 \\\\
        --out evals/results/multi_entity_hard_anchor_share.json
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from engram import Config, Engram

from evals.corpora.multi_entity_hard import (
    HardFixtureConfig,
    generate_multi_entity_hard,
)
from evals.metrics import find_match_rank, hit_at_k
from evals.io_utils import atomic_write_json


def _build_config(arm: str, share_max) -> Config:
    cfg = Config()
    cfg.security.max_events_per_minute = 0
    if arm == "baseline":
        return cfg
    cfg.retrieval.query_expansion_min_dominance = 0.3
    if arm == "prf_with_gate":
        cfg.retrieval.query_expansion_anchor_share_max = share_max
    return cfg


def _eval(arm: str, ds, k: int, share_max) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_config(arm, share_max)
        cfg.path = tmp
        eng = Engram(config=cfg)
        try:
            for content, _meta in ds.memories:
                eng.remember(content)
            ranks = []
            for q in ds.queries:
                results = eng.recall(q.text, limit=k)
                ranks.append(find_match_rank(results, q.expected_substrings))
            return {
                "arm": arm,
                "share_max": share_max,
                "hit@1": round(hit_at_k(ranks, 1), 4),
                "hit@5": round(hit_at_k(ranks, 5), 4),
                "hit@10": round(hit_at_k(ranks, min(10, k)), 4),
            }
        finally:
            eng.close()


def _parse_share(s: str):
    s = s.strip()
    if s.lower() == "none":
        return None
    return float(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=500)
    p.add_argument("--n-sessions", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--share-points", type=str,
                   default="None,0.7,0.5,0.4,0.3")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg_fix = HardFixtureConfig(
        n_facts=args.n_facts,
        n_sessions=args.n_sessions,
        seed=args.seed,
    )
    ds = generate_multi_entity_hard(cfg_fix)
    print(f"[meh-as] {len(ds.memories)} mem, {len(ds.queries)} q", flush=True)

    base = _eval("baseline", ds, args.k, None)
    prf = _eval("prf_with_gate", ds, args.k, None)
    print(f"[base] {base}", flush=True)
    print(f"[prf-only] {prf}", flush=True)

    points = [
        {**base, "delta_vs_prf_h1": round(base["hit@1"] - prf["hit@1"], 4)},
        {**prf, "delta_vs_prf_h1": 0.0},
    ]
    for s in [_parse_share(x) for x in args.share_points.split(",")]:
        if s is None:
            continue
        r = _eval("prf_with_gate", ds, args.k, s)
        r["delta_vs_prf_h1"] = round(r["hit@1"] - prf["hit@1"], 4)
        r["delta_vs_prf_h5"] = round(r["hit@5"] - prf["hit@5"], 4)
        r["delta_vs_prf_h10"] = round(r["hit@10"] - prf["hit@10"], 4)
        print(f"[gate s={s}] {r}", flush=True)
        points.append(r)

    out = {
        "n_facts": args.n_facts,
        "n_sessions": args.n_sessions,
        "n_memories": len(ds.memories),
        "n_queries": len(ds.queries),
        "k": args.k,
        "share_points": args.share_points,
        "points": points,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"[meh-as] wrote {args.out}")


if __name__ == "__main__":
    main()

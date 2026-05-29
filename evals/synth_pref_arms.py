"""§D15c — Preference-corpus PRF gated_pref vs baseline (paired CIs).

Non-LongMemEval corroboration of the §D15b finding that
`query_expansion_type_allow={'single-session-preference'}` lifts the
preference slice. Builds a preference-heavy synthetic corpus from
`evals.synthetic.generate_preference_dataset` (every query maps to
TYPE_SS_PREF under the heuristic classifier), runs paired baseline vs
treatment over the SAME corpus + SAME query order, and reports
paired-bootstrap CIs on Δhit@1 / Δhit@k / ΔMRR.

Usage:
    python -m evals.synth_pref_arms \
        --n-facts 240 --seed 42 --k 10 \
        --resamples 5000 \
        --out evals/results/synth_pref_arms_n240.json

The corpus is reused across arms; we don't rebuild between arms because
ingest+recall is deterministic given the seed and Engram is single-process.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from engram import Engram, Config

from .bootstrap_ci import _bootstrap_mean_ci, _paired_diff_ci
from .metrics import find_match_rank
from .synthetic import generate_preference_dataset
from evals.io_utils import atomic_write_json


def _build_baseline_cfg(path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    return cfg


def _build_gated_pref_cfg(path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    # PRF on, share_prior off (matches §D15 gated_pref)
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_type_allow = frozenset(
        {"single-session-preference"}
    )
    return cfg


def _run_arm(arm: str, ds, k: int) -> tuple[list[int], list[int], list[float], float, float]:
    """Returns (h1_flags, hk_flags, rr_vals, ingest_s, recall_p50_ms)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = (
            _build_baseline_cfg(tmp)
            if arm == "baseline"
            else _build_gated_pref_cfg(tmp)
        )
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0

            h1: list[int] = []
            hk: list[int] = []
            rr: list[float] = []
            lat: list[float] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=k)
                lat.append((time.monotonic() - t0) * 1000)
                rank = find_match_rank(results, q.expected_substrings)
                h1.append(1 if (rank is not None and rank < 1) else 0)
                hk.append(1 if (rank is not None and rank < k) else 0)
                rr.append(0.0 if rank is None else 1.0 / (rank + 1))
            return h1, hk, rr, ingest_s, statistics.median(lat)
        finally:
            eng.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=240)
    p.add_argument("--distractors-per-fact", type=int, default=6)
    p.add_argument("--hard-distractors-per-fact", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    print(f"[synth_pref] building corpus n_facts={args.n_facts} seed={args.seed}", flush=True)
    ds = generate_preference_dataset(
        n_facts=args.n_facts,
        distractors_per_fact=args.distractors_per_fact,
        hard_distractors_per_fact=args.hard_distractors_per_fact,
        seed=args.seed,
    )
    print(
        f"[synth_pref] corpus: {len(ds.memories)} memories, "
        f"{len(ds.queries)} queries", flush=True
    )

    print("[synth_pref] arm=baseline ...", flush=True)
    b_h1, b_hk, b_rr, b_ingest, b_p50 = _run_arm("baseline", ds, args.k)
    print(
        f"[synth_pref]   baseline ingest={b_ingest:.1f}s p50={b_p50:.2f}ms "
        f"h@1={statistics.mean(b_h1):.4f} h@{args.k}={statistics.mean(b_hk):.4f}",
        flush=True,
    )

    print("[synth_pref] arm=gated_pref ...", flush=True)
    t_h1, t_hk, t_rr, t_ingest, t_p50 = _run_arm("gated_pref", ds, args.k)
    print(
        f"[synth_pref]   gated_pref ingest={t_ingest:.1f}s p50={t_p50:.2f}ms "
        f"h@1={statistics.mean(t_h1):.4f} h@{args.k}={statistics.mean(t_hk):.4f}",
        flush=True,
    )

    out = {
        "n_memories": len(ds.memories),
        "n_queries": len(ds.queries),
        "n_facts": args.n_facts,
        "k": args.k,
        "seed": args.seed,
        "arms": {
            "baseline": {
                "hit_at_1": round(statistics.mean(b_h1), 4),
                "hit_at_k": round(statistics.mean(b_hk), 4),
                "mrr": round(statistics.mean(b_rr), 4),
                "ingest_s": round(b_ingest, 2),
                "recall_p50_ms": round(b_p50, 3),
            },
            "gated_pref": {
                "hit_at_1": round(statistics.mean(t_h1), 4),
                "hit_at_k": round(statistics.mean(t_hk), 4),
                "mrr": round(statistics.mean(t_rr), 4),
                "ingest_s": round(t_ingest, 2),
                "recall_p50_ms": round(t_p50, 3),
            },
        },
    }

    paired = {}
    for name, b, t in (("hit_at_1", b_h1, t_h1),
                       ("hit_at_k", b_hk, t_hk),
                       ("mrr", b_rr, t_rr)):
        bm, blo, bhi = _bootstrap_mean_ci(
            list(map(float, b)), args.resamples, args.seed
        )
        tm, tlo, thi = _bootstrap_mean_ci(
            list(map(float, t)), args.resamples, args.seed + 1
        )
        dm, dlo, dhi = _paired_diff_ci(
            list(map(float, t)), list(map(float, b)),
            args.resamples, args.seed + 2,
        )
        paired[name] = {
            "baseline": {"mean": round(bm, 4), "ci_lo": round(blo, 4), "ci_hi": round(bhi, 4)},
            "treatment": {"mean": round(tm, 4), "ci_lo": round(tlo, 4), "ci_hi": round(thi, 4)},
            "delta": {"mean": round(dm, 4), "ci_lo": round(dlo, 4), "ci_hi": round(dhi, 4)},
            "significant_at_05": (dlo > 0) or (dhi < 0),
        }
    out["paired"] = paired

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))
    print(f"[synth_pref] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

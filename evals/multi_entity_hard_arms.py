"""D1 v0.3 — multi-entity-hard 4-arm A/B sweep.

Runs the four arms {baseline, PRF, share_prior, both} on the new
multi-entity-hard fixture (evals.corpora.multi_entity_hard) and
reports hit@k.

This is the A/B that NEXT.md priority #1 calls for: "Land an A/B with
current PRF+SP wire-in on a non-saturated corpus." If PRF or share_prior
move the needle here, anchors 17–26 generalize beyond LongMemEval-S.
If they don't, the §4.7 wire-in earns its default-OFF position even
on a hard corpus, and the type-aware-PRF prototype (priority 4) becomes
the only remediation path.

Output: bench/results/multi_entity_hard_arms.json + console summary.

Usage:
    python -m evals.multi_entity_hard_arms \\
        --n-facts 500 --n-sessions 25 \\
        --out bench/results/multi_entity_hard_arms.json
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from engram import Config, Engram

from evals.corpora.multi_entity_hard import (
    HardFixtureConfig,
    generate_multi_entity_hard,
)
from evals.metrics import find_match_rank, hit_at_k, mrr
from evals.io_utils import atomic_write_json


ARMS = ["baseline", "prf", "share_prior", "both"]


def _build_config(arm: str, *, qe_dominance: float = 0.3,
                  sp_alpha: float = 0.05, sp_pool: int = 20) -> Config:
    cfg = Config()
    cfg.security.max_events_per_minute = 0
    if arm in ("prf", "both"):
        cfg.retrieval.query_expansion_min_dominance = qe_dominance
    if arm in ("share_prior", "both"):
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.share_prior_alpha = sp_alpha
        cfg.retrieval.rerank_pool_size = sp_pool
    return cfg


def _eval_arm(arm: str, ds, *, k: int, qe_dominance: float,
              sp_alpha: float, sp_pool: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_config(arm, qe_dominance=qe_dominance,
                            sp_alpha=sp_alpha, sp_pool=sp_pool)
        cfg.path = tmp
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0

            ranks: list[int | None] = []
            lat: list[float] = []
            for q in ds.queries:
                t0 = time.monotonic()
                results = eng.recall(q.text, limit=k)
                lat.append((time.monotonic() - t0) * 1000)
                ranks.append(find_match_rank(results, q.expected_substrings))
            return {
                "arm": arm,
                "n_queries": len(ds.queries),
                "hit@1": round(hit_at_k(ranks, 1), 4),
                "hit@5": round(hit_at_k(ranks, 5), 4),
                "hit@10": round(hit_at_k(ranks, min(10, k)), 4),
                "mrr": round(mrr(ranks), 4),
                "p50_ms": round(statistics.median(lat), 3),
                "p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3),
                "ingest_s": round(ingest_s, 2),
                "ranks": ranks,
            }
        finally:
            eng.close()


def _ranks_to_hits(ranks: list[int | None], k: int) -> list[int]:
    return [1 if (r is not None and r <= k) else 0 for r in ranks]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=500)
    p.add_argument("--n-sessions", type=int, default=25)
    p.add_argument("--distractors-per-fact", type=int, default=4)
    p.add_argument("--lexical-collision", type=float, default=1.0)
    p.add_argument("--ner-disambig", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Multiple seeds for paired-bootstrap CI run; "
                        "overrides --seed when provided.")
    p.add_argument("--resamples", type=int, default=5000,
                   help="Bootstrap resample count when --seeds is set.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--qe-dominance", type=float, default=0.3)
    p.add_argument("--sp-alpha", type=float, default=0.05)
    p.add_argument("--sp-pool", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg_fix = HardFixtureConfig(
        n_facts=args.n_facts,
        n_sessions=args.n_sessions,
        distractors_per_fact=args.distractors_per_fact,
        lexical_collision_rate=args.lexical_collision,
        ner_disambig_rate=args.ner_disambig,
        seed=args.seed,
    )
    ds = generate_multi_entity_hard(cfg_fix)
    print(f"[meh] corpus: {len(ds.memories)} mem, {len(ds.queries)} queries")

    seeds = args.seeds if args.seeds else [args.seed]
    multi = len(seeds) > 1

    # Per-arm aggregations across seeds.
    per_seed_hits: dict[str, dict[str, list[list[int]]]] = {
        arm: {"hit@1": [], "hit@5": [], "hit@10": []} for arm in ARMS
    }
    seed_results: list[dict] = []
    base = None
    for s in seeds:
        cfg_fix.seed = s
        ds_s = generate_multi_entity_hard(cfg_fix) if multi else ds
        for arm in ARMS:
            t0 = time.monotonic()
            r = _eval_arm(arm, ds_s, k=args.k, qe_dominance=args.qe_dominance,
                          sp_alpha=args.sp_alpha, sp_pool=args.sp_pool)
            wall = time.monotonic() - t0
            r["seed"] = s
            for k_ in (1, 5, min(10, args.k)):
                key = f"hit@{k_}"
                per_seed_hits[arm][key].append(_ranks_to_hits(r["ranks"], k_))
            if arm == "baseline":
                base = r
                print(f"[s={s}][{arm:11s}] hit@1={r['hit@1']} hit@5={r['hit@5']} "
                      f"hit@10={r['hit@10']} p50={r['p50_ms']}ms wall={wall:.1f}s")
            else:
                d1 = r["hit@1"] - base["hit@1"]
                d5 = r["hit@5"] - base["hit@5"]
                d10 = r["hit@10"] - base["hit@10"]
                print(f"[s={s}][{arm:11s}] hit@1={r['hit@1']} hit@5={r['hit@5']} "
                      f"hit@10={r['hit@10']} Δ@1={d1:+.4f} Δ@5={d5:+.4f} "
                      f"Δ@10={d10:+.4f} p50={r['p50_ms']}ms wall={wall:.1f}s")
            seed_results.append(r)

    out = {
        "fixture": {
            "n_facts": args.n_facts, "n_sessions": args.n_sessions,
            "distractors_per_fact": args.distractors_per_fact,
            "lexical_collision_rate": args.lexical_collision,
            "ner_disambig_rate": args.ner_disambig,
            "seeds": seeds,
            "n_memories": len(ds.memories),
            "n_queries": len(ds.queries),
        },
        "knobs": {
            "qe_dominance": args.qe_dominance,
            "sp_alpha": args.sp_alpha,
            "sp_pool": args.sp_pool,
            "k": args.k,
        },
        "per_seed_arms": seed_results,
    }

    # Paired-bootstrap CI on per-query Δ vs baseline, with paired resamples
    # (same query indices for baseline+arm). Across seeds: concatenate per-query
    # hit vectors so n_queries scales with seed count and the bootstrap
    # respects the per-query pairing.
    if multi:
        from evals.bootstrap_ci import _paired_diff_ci
        ci_block: dict = {}
        # Flatten across seeds
        flat: dict[str, dict[str, list[int]]] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for k_, vecs in per_seed_hits[arm].items():
                # vecs: list[ list[int] ] one per seed
                flat[arm][k_] = [x for v in vecs for x in v]
        for arm in ARMS:
            if arm == "baseline":
                continue
            entry = {}
            for k_ in ("hit@1", "hit@5", "hit@10"):
                base_v = flat["baseline"][k_]
                arm_v = flat[arm][k_]
                m, lo, hi = _paired_diff_ci(arm_v, base_v,
                                            args.resamples, args.seed)
                entry[f"d_{k_}"] = {
                    "mean": round(m, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "baseline": round(sum(base_v) / max(len(base_v), 1), 4),
                    "arm": round(sum(arm_v) / max(len(arm_v), 1), 4),
                    "n_queries_total": len(base_v),
                }
            ci_block[arm] = entry
        out["paired_bootstrap_ci"] = {
            "resamples": args.resamples,
            "seed": args.seed,
            "n_seeds": len(seeds),
            "delta_vs_baseline": ci_block,
        }
        print("\n[meh-ci] paired-bootstrap Δ vs baseline (n_queries × n_seeds):")
        for arm, entry in ci_block.items():
            for k_ in ("hit@1", "hit@5", "hit@10"):
                d = entry[f"d_{k_}"]
                print(f"  [{arm:11s}] Δ{k_:6s}: {d['mean']:+.4f} "
                      f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, out)
        print(f"[meh] wrote {args.out}")


if __name__ == "__main__":
    main()

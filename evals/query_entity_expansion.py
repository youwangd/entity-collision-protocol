"""query-side entity expansion (pseudo-relevance feedback).

Migrated 2026-05-22 to paper/30_methods.md §3.6 (PRF construction) and
paper/40_results.md §4.7 / §A.4.7.1 (dominance-gate sweep). The
PAPER_NOTES §5.4 anchor 5 referenced below is audit-trail only.

Hypothesis (anchor 5 of PAPER_NOTES §5.4 audit trail; canonical: §3.6): the dominant failure mode for
multi-hop pair retrieval at scale is that *gold pair facts never enter the
top-K candidate pool* in the first place — no amount of reranking inside
the pool can fix that. A cheap remedy: do an initial BM25 retrieval, harvest
named entities from the top-K result texts, append them to the original
query, and re-issue retrieval. The bridge entity that connects fact_a and
fact_b is over-represented in the first-pass results (it appears in BOTH
gold facts plus any near-miss distractors), so an expanded query has a
higher chance of pulling the partner fact into the new top-N.

Spike measures Δany_hit@K and Δpair_recall@K on the bridge corpus.

Driver: `python -m evals.query_entity_expansion --seeds 17,42,101`
Outputs: `evals/results/query_entity_expansion.json`, markdown to stdout.

This is a do-no-harm + signal-or-not test. If Δany_hit@20 ≥ +0.02 across
seeds with no h@1 regression on the unique-entity corpus, the angle is
worth landing as an opt-in retrieval flag.
"""

from __future__ import annotations

import argparse
import re
import statistics
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from engram.retrieval.entities import extract_entities
from evals.share_prior_sweep import generate_bridge_corpus
from evals.entity_channel_sweep import generate_entity_corpus
from evals.io_utils import atomic_write_json


def _query_terms(q: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9]+", q.lower()) if len(t) > 1}


def _expand_query(
    query: str,
    first_pass_texts: list[str],
    *,
    top_k_for_prf: int,
    max_entities: int,
    min_dominance: float = 0.0,
    backend: str = "heuristic",
) -> tuple[str, list[str]]:
    """Append the most frequent novel entities from the top-K texts.

    `min_dominance`: gate. Only expand when the most frequent entity appears
    in ≥ min_dominance × top_k_for_prf documents. This filters single-hop
    queries (where no entity dominates) from the expansion path.
    """
    seen_in_q = _query_terms(query)
    counts: dict[str, int] = {}
    pool = first_pass_texts[:top_k_for_prf]
    n_docs = max(len(pool), 1)
    for text in pool:
        ents = extract_entities(text or "", backend=backend)
        for e in ents:
            words = set(re.findall(r"[a-z0-9]+", e))
            if words and words.issubset(seen_in_q):
                continue
            counts[e] = counts.get(e, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return query, []
    # Dominance gate: top entity must appear in ≥ min_dominance × n_docs
    if min_dominance > 0.0 and ranked[0][1] / n_docs < min_dominance:
        return query, []
    chosen = [e for e, _ in ranked[:max_entities]]
    return query + " " + " ".join(chosen), chosen


def _build_engine(path: str) -> Engram:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval = RetrievalConfig()
    return Engram(config=cfg)


def _eval_bridge(ds, *, expand, top_k_for_prf, max_entities,
                 min_dominance=0.0, k_max=20) -> dict:
    rows = []
    expansions = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                first_pass = eng.recall(q.text, limit=k_max)
                first_texts = [r.memory.content for r in first_pass]
                if expand:
                    expanded_q, chosen = _expand_query(
                        q.text, first_texts,
                        top_k_for_prf=top_k_for_prf,
                        max_entities=max_entities,
                        min_dominance=min_dominance,
                    )
                    expansions.append({"q": q.text, "added": chosen})
                    results = eng.recall(expanded_q, limit=k_max) if chosen else first_pass
                else:
                    results = first_pass
                texts = [r.memory.content for r in results]
                a, b = q.expected_substrings[0], q.expected_substrings[1]

                def hit(needle, top):
                    n = needle.lower()
                    return any(n in t.lower() for t in top)

                def pair_at(k):
                    top = texts[:k]
                    return int(hit(a, top) and hit(b, top))

                def any_at(k):
                    top = texts[:k]
                    return int(hit(a, top) or hit(b, top))

                rows.append({
                    "any@5": any_at(5), "any@10": any_at(10), "any@20": any_at(20),
                    "pair@5": pair_at(5), "pair@10": pair_at(10),
                    "pair@20": pair_at(20),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "expand": expand,
        "n_queries": len(rows),
        "any_hit@5": sum(r["any@5"] for r in rows) / n,
        "any_hit@10": sum(r["any@10"] for r in rows) / n,
        "any_hit@20": sum(r["any@20"] for r in rows) / n,
        "pair_recall@5": sum(r["pair@5"] for r in rows) / n,
        "pair_recall@10": sum(r["pair@10"] for r in rows) / n,
        "pair_recall@20": sum(r["pair@20"] for r in rows) / n,
        "_expansions_sample": expansions[:5],
    }


def _eval_unique_donoharm(ds, *, expand, top_k_for_prf, max_entities,
                          min_dominance=0.0, k_max=10) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                first_pass = eng.recall(q.text, limit=k_max)
                first_texts = [r.memory.content for r in first_pass]
                if expand:
                    expanded_q, chosen = _expand_query(
                        q.text, first_texts,
                        top_k_for_prf=top_k_for_prf,
                        max_entities=max_entities,
                        min_dominance=min_dominance,
                    )
                    results = eng.recall(expanded_q, limit=k_max) if chosen else first_pass
                else:
                    results = first_pass
                texts = [r.memory.content for r in results]
                gold = q.expected_substrings[0].lower()
                rows.append({
                    "hit@1": int(bool(texts) and gold in texts[0].lower()),
                    "hit@5": int(any(gold in t.lower() for t in texts[:5])),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "expand": expand,
        "n_queries": len(rows),
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
    }


def run(*, seeds, n_pairs, plain_distractors, n_facts,
        top_k_for_prf, max_entities, min_dominance=0.0) -> dict:
    t0 = time.monotonic()
    bridge_baseline, bridge_expand = [], []
    unique_baseline, unique_expand = [], []
    for s in seeds:
        ds_b = generate_bridge_corpus(
            n_pairs=n_pairs, plain_distractors=plain_distractors, seed=s
        )
        bridge_baseline.append(_eval_bridge(
            ds_b, expand=False,
            top_k_for_prf=top_k_for_prf, max_entities=max_entities,
            min_dominance=min_dominance,
        ))
        bridge_expand.append(_eval_bridge(
            ds_b, expand=True,
            top_k_for_prf=top_k_for_prf, max_entities=max_entities,
            min_dominance=min_dominance,
        ))
        ds_u = generate_entity_corpus(
            n_facts=n_facts, hard_distractors_per_fact=2,
            plain_distractors=plain_distractors, seed=s + 1000,
        )
        unique_baseline.append(_eval_unique_donoharm(
            ds_u, expand=False,
            top_k_for_prf=top_k_for_prf, max_entities=max_entities,
            min_dominance=min_dominance,
        ))
        unique_expand.append(_eval_unique_donoharm(
            ds_u, expand=True,
            top_k_for_prf=top_k_for_prf, max_entities=max_entities,
            min_dominance=min_dominance,
        ))

    def agg(rows, keys):
        return {
            k: {
                "mean": round(statistics.mean(r[k] for r in rows), 4),
                "stdev": (round(statistics.stdev(r[k] for r in rows), 4)
                          if len(rows) > 1 else 0.0),
            }
            for k in keys
        }

    bridge_keys = ["any_hit@5", "any_hit@10", "any_hit@20",
                   "pair_recall@5", "pair_recall@10", "pair_recall@20"]
    unique_keys = ["hit@1", "hit@5"]

    return {
        "config": {
            "seeds": list(seeds), "n_pairs": n_pairs,
            "plain_distractors": plain_distractors, "n_facts": n_facts,
            "top_k_for_prf": top_k_for_prf, "max_entities": max_entities,
            "min_dominance": min_dominance,
        },
        "bridge": {
            "baseline": agg(bridge_baseline, bridge_keys),
            "expand": agg(bridge_expand, bridge_keys),
            "delta": {
                k: round(
                    statistics.mean(e[k] for e in bridge_expand)
                    - statistics.mean(b[k] for b in bridge_baseline),
                    4,
                )
                for k in bridge_keys
            },
            "_per_seed_baseline": bridge_baseline,
            "_per_seed_expand": bridge_expand,
        },
        "unique_donoharm": {
            "baseline": agg(unique_baseline, unique_keys),
            "expand": agg(unique_expand, unique_keys),
            "delta": {
                k: round(
                    statistics.mean(e[k] for e in unique_expand)
                    - statistics.mean(b[k] for b in unique_baseline),
                    4,
                )
                for k in unique_keys
            },
        },
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"Wall: {rep['wall_seconds']}s | seeds={cfg['seeds']} "
        f"n_pairs={cfg['n_pairs']} top_k_prf={cfg['top_k_for_prf']} "
        f"max_entities={cfg['max_entities']}",
        "",
        "### Bridge multi-hop — query expansion vs. baseline",
        "",
        "| metric | baseline (μ±σ) | expand (μ±σ) | Δ |",
        "|:---|---:|---:|---:|",
    ]
    b = rep["bridge"]["baseline"]; e = rep["bridge"]["expand"]; d = rep["bridge"]["delta"]
    for k in ["any_hit@5", "any_hit@10", "any_hit@20",
              "pair_recall@5", "pair_recall@10", "pair_recall@20"]:
        lines.append(
            f"| {k} | {b[k]['mean']:.3f}±{b[k]['stdev']:.3f} "
            f"| {e[k]['mean']:.3f}±{e[k]['stdev']:.3f} | {d[k]:+.3f} |"
        )
    lines += [
        "",
        "### Unique-entity — do-no-harm",
        "",
        "| metric | baseline (μ±σ) | expand (μ±σ) | Δ |",
        "|:---|---:|---:|---:|",
    ]
    b = rep["unique_donoharm"]["baseline"]; e = rep["unique_donoharm"]["expand"]
    d = rep["unique_donoharm"]["delta"]
    for k in ["hit@1", "hit@5"]:
        lines.append(
            f"| {k} | {b[k]['mean']:.3f}±{b[k]['stdev']:.3f} "
            f"| {e[k]['mean']:.3f}±{e[k]['stdev']:.3f} | {d[k]:+.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="17,42,101")
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=80)
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--top-k-for-prf", type=int, default=10)
    p.add_argument("--max-entities", type=int, default=4)
    p.add_argument("--min-dominance", type=float, default=0.0,
                   help="Only expand if top entity occurs in ≥ this fraction "
                        "of PRF docs. 0.0 = always expand. 0.3 = require "
                        "top entity in ≥30%% of docs.")
    p.add_argument("--out", default="evals/results/query_entity_expansion.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rep = run(
        seeds=seeds, n_pairs=args.n_pairs,
        plain_distractors=args.plain_distractors, n_facts=args.n_facts,
        top_k_for_prf=args.top_k_for_prf, max_entities=args.max_entities,
        min_dominance=args.min_dominance,
    )
    print("§5.4 angle 1 — query-side entity expansion (PRF)")
    print(_md(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[query-entity-expansion] wrote {args.out}")


if __name__ == "__main__":
    main()

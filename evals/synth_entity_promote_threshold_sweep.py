"""§4.6.2 — schema_promote_threshold sweep on synth_entity (high-collision regime).

Companion to ``evals/locomo_promote_threshold_sweep.py`` but operating on
the type-paired collider fixture from
``evals.entity_channel_sweep._build_synth_entity_dataset``. Per threshold
``t`` we run a paired (baseline, treatment) arm:

    baseline   : ingest only (eng.remember + eng.capture), no consolidation
    treatment  : same ingest, then ``eng.consolidate(window='999d')``

with::

    cfg.consolidation = ConsolidationConfig(
        schedule='manual',
        window_hours=24*999,
        schema_synthesis_enabled=True,
        schema_synthesis_tau=0.3,
        schema_synthesis_min_supports=3,
        schema_promote_threshold=t,
    )

and the §4.13c retrieval pivot (``vector_weight=0.3``, ``bm25_weight=1.0``,
salience/recency/context = 0). Per-query rank, hit@1, hit@5, MRR are
recorded; paired Δ-bootstrap CIs (5000 resamples, α=0.05) compare each
threshold's *treatment* arm against the t=3 (default) treatment arm.

CLI
---
    python -m evals.synth_entity_promote_threshold_sweep \\
        --n-entities 32 --K 8 --thresholds 1,2,3,5,7,10 --resamples 5000

Output: ``evals/results/synth_entity_promote_threshold_sweep.json``.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from engram import Config, Engram
from engram.core.config import ConsolidationConfig

from evals.ablation import _make_embedder
from evals._embed_cache import CachingEmbeddingProvider
from evals.bootstrap_ci import _paired_diff_ci
from evals.entity_channel_sweep import _build_synth_entity_dataset
from evals.metrics import find_match_rank, hit_at_k, mrr
from evals.io_utils import atomic_write_json


def _build_cfg(*, threshold: int, vector_weight: float, path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    # §4.13c retrieval pivot.
    cfg.retrieval.bm25_weight = 1.0
    cfg.retrieval.vector_weight = float(vector_weight)
    cfg.retrieval.salience_weight = 0.0
    cfg.retrieval.recency_weight = 0.0
    cfg.retrieval.context_weight = 0.0
    cfg.retrieval.use_extraction_confidence = False
    # Consolidation: manual, full window, schema synthesis on.
    cfg.consolidation = ConsolidationConfig(
        schedule="manual",
        window_hours=24 * 999,
        schema_synthesis_enabled=True,
        schema_synthesis_tau=0.3,
        schema_synthesis_min_supports=3,
        schema_promote_threshold=int(threshold),
    )
    return cfg


def _ingest(eng: Engram, ds) -> None:
    """Dual-write: remember() for retrieval, capture() for consolidation."""
    for content, _meta in ds.memories:
        eng.remember(content)
        try:
            eng.capture(content)
        except Exception:
            # capture is best-effort for the consolidation event stream;
            # if rate-limited or rejected we still have remember()'d data.
            pass


def _score(eng: Engram, ds, *, k: int) -> tuple[list[int | None], list[float], list[dict]]:
    ranks: list[int | None] = []
    lats: list[float] = []
    per_query: list[dict] = []
    for q in ds.queries:
        t0 = time.monotonic()
        results = eng.recall(q.text, limit=k)
        lat = (time.monotonic() - t0) * 1000
        lats.append(lat)
        rank = find_match_rank(results, q.expected_substrings)
        ranks.append(rank)
        hit1 = 1 if (rank is not None and rank == 0) else 0
        hit5 = 1 if (rank is not None and rank < 5) else 0
        rr = (1.0 / (rank + 1)) if rank is not None else 0.0
        per_query.append({
            "query": q.text,
            "tag": q.tags[0] if q.tags else None,
            "rank": rank,
            "hit_at_1": hit1,
            "hit_at_5": hit5,
            "reciprocal_rank": rr,
            "latency_ms": round(lat, 3),
        })
    return ranks, lats, per_query


def _run_threshold(t: int, ds, *, k: int, vector_weight: float,
                   embedder) -> dict:
    """Run baseline + treatment arms for a single promote_threshold value."""
    out: dict = {"threshold": int(t)}
    sub_t = time.monotonic()

    # --- baseline arm: ingest only, no consolidation
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(threshold=t, vector_weight=vector_weight, path=tmp)
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if (embedder is not None and cfg.retrieval.vector_weight > 0)
            else Engram(config=cfg)
        )
        try:
            t_in = time.monotonic()
            _ingest(eng, ds)
            ingest_s = time.monotonic() - t_in
            ranks_b, lats_b, pq_b = _score(eng, ds, k=k)
        finally:
            eng.close()
    out["baseline"] = {
        "hit_at_1": round(hit_at_k(ranks_b, 1), 4),
        "hit_at_5": round(hit_at_k(ranks_b, 5), 4),
        "mrr": round(mrr(ranks_b), 4),
        "p50_ms": round(statistics.median(lats_b), 3),
        "ingest_s": round(ingest_s, 2),
        "per_query": pq_b,
    }

    # --- treatment arm: ingest, then consolidate(window='999d')
    schemas_created = 0
    consolidation_error: str | None = None
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(threshold=t, vector_weight=vector_weight, path=tmp)
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if (embedder is not None and cfg.retrieval.vector_weight > 0)
            else Engram(config=cfg)
        )
        try:
            _ingest(eng, ds)
            try:
                rep = eng.consolidate(window="999d")
                # Report can be ConsolidationReport-like or dict.
                st = getattr(rep, "state_transitions", None)
                if st is None and isinstance(rep, dict):
                    st = rep.get("state_transitions", {})
                if st:
                    schemas_created = int(st.get("schemas", 0))
            except Exception as exc:  # pragma: no cover - diagnostic
                consolidation_error = repr(exc)
            ranks_t, lats_t, pq_t = _score(eng, ds, k=k)
        finally:
            eng.close()
    out["treatment"] = {
        "hit_at_1": round(hit_at_k(ranks_t, 1), 4),
        "hit_at_5": round(hit_at_k(ranks_t, 5), 4),
        "mrr": round(mrr(ranks_t), 4),
        "p50_ms": round(statistics.median(lats_t), 3),
        "schemas_created": schemas_created,
        "consolidation_error": consolidation_error,
        "per_query": pq_t,
    }
    out["wall_seconds"] = round(time.monotonic() - sub_t, 2)
    return out


def _paired_block_vs(rows: list[dict], pivot_t: int, *,
                     resamples: int, seed: int) -> dict:
    pivot = next((r for r in rows if r["threshold"] == pivot_t), None)
    if pivot is None:
        return {}
    pv = pivot["treatment"]["per_query"]
    block: dict = {"pivot_threshold": pivot_t, "n_queries": len(pv),
                   "deltas": {}}
    for r in rows:
        if r["threshold"] == pivot_t:
            continue
        cv = r["treatment"]["per_query"]
        if len(cv) != len(pv):
            continue
        entry: dict = {}
        for metric in ("hit_at_1", "hit_at_5", "mrr"):
            key = "reciprocal_rank" if metric == "mrr" else metric
            a = [x[key] for x in cv]
            b = [x[key] for x in pv]
            m, lo, hi = _paired_diff_ci(a, b, resamples, seed)
            entry[metric] = {
                "mean_delta": round(m, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "arm_mean": round(sum(a) / max(len(a), 1), 4),
                "pivot_mean": round(sum(b) / max(len(b), 1), 4),
                "significant": (lo > 0) or (hi < 0),
            }
        block["deltas"][str(r["threshold"])] = entry
    return block


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-entities", type=int, default=32)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--distractors-per-entity", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--vector-weight", type=float, default=0.3)
    p.add_argument("--thresholds", type=str, default="1,2,3,5,7,10")
    p.add_argument("--pivot-threshold", type=int, default=3)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--embed", type=str, default="hash",
                   choices=["hash", "hash128", "hash256", "hash512",
                            "hash1024", "st"])
    p.add_argument("--out", type=str,
                   default="evals/results/synth_entity_promote_threshold_sweep.json")
    args = p.parse_args()

    thr = [int(x) for x in args.thresholds.split(",") if x.strip()]
    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)

    ds, fixture_meta = _build_synth_entity_dataset(
        n_entities=args.n_entities,
        collision_degree=args.K,
        distractors_per_entity=args.distractors_per_entity,
        seed=args.seed,
    )
    print(f"[se-pt] fixture: {fixture_meta['n_memories']} mems, "
          f"{fixture_meta['n_queries']} queries "
          f"(n_entities={args.n_entities}, K={args.K})")

    t_all = time.monotonic()
    rows: list[dict] = []
    for t in thr:
        row = _run_threshold(t, ds, k=args.k,
                             vector_weight=args.vector_weight,
                             embedder=embedder)
        rows.append(row)
        b = row["baseline"]
        tr = row["treatment"]
        print(f"  t={t:>2}  base h@1={b['hit_at_1']:.3f} "
              f"h@5={b['hit_at_5']:.3f} MRR={b['mrr']:.3f}  |  "
              f"treat h@1={tr['hit_at_1']:.3f} h@5={tr['hit_at_5']:.3f} "
              f"MRR={tr['mrr']:.3f}  schemas={tr['schemas_created']}  "
              f"({row['wall_seconds']}s)")
    wall = round(time.monotonic() - t_all, 2)

    paired = _paired_block_vs(rows, args.pivot_threshold,
                              resamples=args.resamples, seed=args.seed)

    out = {
        "config": {
            "fixture": fixture_meta,
            "vector_weight": args.vector_weight,
            "embed": args.embed,
            "k": args.k,
            "thresholds": thr,
            "pivot_threshold": args.pivot_threshold,
            "resamples": args.resamples,
            "alpha": 0.05,
            "seed": args.seed,
        },
        "rows": [
            {
                "threshold": r["threshold"],
                "baseline": {k: v for k, v in r["baseline"].items()
                             if k != "per_query"},
                "treatment": {k: v for k, v in r["treatment"].items()
                              if k != "per_query"},
                "wall_seconds": r["wall_seconds"],
            }
            for r in rows
        ],
        "per_query_treatment_by_threshold": {
            str(r["threshold"]): r["treatment"]["per_query"] for r in rows
        },
        "per_query_baseline_by_threshold": {
            str(r["threshold"]): r["baseline"]["per_query"] for r in rows
        },
        "paired_bootstrap_ci_treatment_vs_pivot": paired,
        "wall_seconds": wall,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(f"[se-pt] wrote {args.out}  (wall={wall}s)")

    # Markdown summary
    md_lines = [
        f"\n### §4.6.2 — promote_threshold on synth_entity "
        f"(n_entities={args.n_entities}, K={args.K}, vw={args.vector_weight}, "
        f"embed={args.embed})\n",
        "| t | base h@1 | treat h@1 | base h@5 | treat h@5 | "
        "base MRR | treat MRR | schemas |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        b = r["baseline"]
        tr = r["treatment"]
        md_lines.append(
            f"| {r['threshold']} | {b['hit_at_1']:.3f} | {tr['hit_at_1']:.3f} | "
            f"{b['hit_at_5']:.3f} | {tr['hit_at_5']:.3f} | "
            f"{b['mrr']:.3f} | {tr['mrr']:.3f} | {tr['schemas_created']} |"
        )
    if paired and paired.get("deltas"):
        md_lines.append(f"\nPaired Δ vs **t={args.pivot_threshold}** "
                        f"(treatment-arm only, {args.resamples} resamples, "
                        f"α=0.05):\n")
        md_lines.append("| t | metric | Δ mean | 95% CI | sig? |")
        md_lines.append("|---:|:---|---:|:---:|:---:|")
        for tname, e in paired["deltas"].items():
            for metric in ("hit_at_1", "hit_at_5", "mrr"):
                d = e[metric]
                md_lines.append(
                    f"| {tname} | {metric} | {d['mean_delta']:+.4f} | "
                    f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | "
                    f"{'**yes**' if d['significant'] else 'no'} |"
                )
    md_text = "\n".join(md_lines) + "\n"
    print(md_text)


if __name__ == "__main__":
    main()

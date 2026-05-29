"""Run a vw sweep on LoCoMo reusing one embedder instance to amortize load.

This is an internal CLI helper to feed bootstrap_ci faster than running 5
separate `python -m evals.locomo_adapter` invocations.
"""
from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from evals.locomo_adapter import load_locomo, _ingest, _session_id_of
from evals.io_utils import atomic_write_json


def _run_one(samples, vector_weight, k, emb_provider, emb_label,
             save_bm25_signals: bool = False):
    overall_h1, overall_hk = [], []
    per_cat_h1, per_cat_hk = {}, {}
    ingest_lat, recall_lat = [], []
    per_query = []
    n_q = n_scored = n_mem = 0

    for sample in samples:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(path=tmp)
            cfg.security.max_events_per_minute = 0
            cfg.retrieval.vector_weight = float(vector_weight)
            eng = Engram(config=cfg, embeddings=emb_provider) if emb_provider else Engram(config=cfg)
            try:
                t0 = time.monotonic()
                m = _ingest(eng, sample)
                ingest_lat.append((time.monotonic() - t0) * 1000)
                n_mem += m

                for q in sample.qa:
                    n_q += 1
                    gold = set(q.evidence_sessions)
                    if not gold:
                        continue
                    n_scored += 1
                    t0 = time.monotonic()
                    results = eng.recall(q.question, limit=k)
                    recall_lat.append((time.monotonic() - t0) * 1000)
                    sids = [_session_id_of(r) for r in results]
                    h1 = 1 if (sids and sids[0] in gold) else 0
                    hk = 1 if any(s in gold for s in sids) else 0
                    rank = 0
                    for i, s in enumerate(sids, start=1):
                        if s in gold:
                            rank = i
                            break
                    rr = 1.0 / rank if rank > 0 else 0.0
                    overall_h1.append(h1); overall_hk.append(hk)
                    per_cat_h1.setdefault(q.category, []).append(h1)
                    per_cat_hk.setdefault(q.category, []).append(hk)
                    per_query.append({
                        "sample_id": sample.sample_id, "category": q.category,
                        "q_idx": n_scored - 1,
                        "rank": rank, "hit_at_1": h1, "hit_at_k": hk,
                        "reciprocal_rank": rr,
                    })
                    if save_bm25_signals:
                        from evals._signals import (
                            compute_bm25_top_gap, normalized_gap, crowdedness,
                        )
                        bm25_scores = [
                            float(r.sources["bm25"])
                            for r in results
                            if isinstance(getattr(r, "sources", None), dict)
                            and "bm25" in r.sources
                        ]
                        b1, b2, gap = compute_bm25_top_gap(bm25_scores)
                        ng = normalized_gap(b1, b2)
                        c95 = crowdedness(bm25_scores, frac=0.95) if bm25_scores else None
                        per_query[-1].update({
                            "bm25_top1": round(b1, 4) if b1 is not None else None,
                            "bm25_top2": round(b2, 4) if b2 is not None else None,
                            "bm25_gap": round(gap, 4) if gap is not None else None,
                            "bm25_norm_gap": round(ng, 4) if ng is not None else None,
                            "bm25_crowd_95": c95,
                        })
            finally:
                eng.close()

    def _agg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "vector_weight": vector_weight,
        "embedder": emb_label,
        "k": k,
        "n_samples": len(samples),
        "n_questions": n_q,
        "n_questions_scored": n_scored,
        "n_memories_total": n_mem,
        "session_hit_at_1": _agg(overall_h1),
        "session_hit_at_k": _agg(overall_hk),
        "per_category_session_hit_at_1": {c: _agg(v) for c, v in per_cat_h1.items()},
        "per_category_session_hit_at_k": {c: _agg(v) for c, v in per_cat_hk.items()},
        "per_category_n": {c: len(v) for c, v in per_cat_h1.items()},
        "per_query": per_query,
        "ingest_ms": {
            "p50": round(statistics.median(ingest_lat), 2),
            "mean": round(statistics.mean(ingest_lat), 2),
            "max": round(max(ingest_lat), 2),
        } if ingest_lat else {},
        "recall_ms": {
            "p50": round(statistics.median(recall_lat), 2),
            "mean": round(statistics.mean(recall_lat), 2),
            "max": round(max(recall_lat), 2),
        } if recall_lat else {},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--max-instances", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", choices=["hashtrigram", "st"], default="hashtrigram")
    p.add_argument("--vws", default="0.0,0.3,0.5,0.7,1.0")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--save-bm25-signals", action="store_true")
    args = p.parse_args()

    samples = load_locomo(args.dataset, max_instances=args.max_instances)
    print(f"[locomo_sweep_ci] loaded {len(samples)} samples")

    if args.embedder == "hashtrigram":
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        emb = HashTrigramEmbeddingProvider(dimension=256)
        emb_label = "HashTrigram-256"
    else:
        from engram.providers.embeddings import SentenceTransformerProvider
        emb = SentenceTransformerProvider()
        emb_label = "SentenceTransformer-MiniLM-384"

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    written = []
    for vw_s in args.vws.split(","):
        vw = float(vw_s)
        t0 = time.monotonic()
        res = _run_one(samples, vw, args.k, emb if vw > 0 else None,
                       emb_label if vw > 0 else "none(BM25-only)",
                       save_bm25_signals=args.save_bm25_signals)
        out = f"{args.out_prefix}_vw{vw}.json"
        atomic_write_json(out, res)
        dt = time.monotonic() - t0
        print(f"[vw={vw}] h@1={res['session_hit_at_1']} h@{args.k}={res['session_hit_at_k']} "
              f"n={res['n_questions_scored']} t={dt:.1f}s -> {out}")
        written.append(out)
    print(f"[locomo_sweep_ci] wrote: {written}")


if __name__ == "__main__":
    main()

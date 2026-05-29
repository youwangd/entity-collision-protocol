"""LoCoMo c5 adversarial-distractor analyzer.

§4.5 of our paper draft showed c5 (open-domain unanswerable) is a damage
driver as vw increases. Hypothesis: vector retrieval pulls the
*adversarial_answer* (the LoCoMo-provided plausible-but-wrong answer) into
top-1 more readily than BM25. If true, the c5 damage is not retrieval
quality but a property of the dataset's distractor design.

This tool re-runs LoCoMo on c5-only with two vw settings, captures the
top-1 memory's text per query, and computes: for each top-1, does it
share ≥k content tokens with the adversarial_answer (after stoplisting)?

Output: per-query record with `top1_content`, `top1_session`,
`adv_overlap_tokens`, `adv_overlap_frac`, `bm25_or_vector_dominant`.
Aggregates: c5 hit@1 by vw, conditional adversarial-overlap rate
restricted to top-1=non-gold (i.e. the misses).

Usage:
    python -m evals.locomo_c5_adversarial \\
        --dataset bench/data/locomo10.json \\
        --vws 0.0,0.5 --max-instances 10 --k 10 \\
        --out bench/results/locomo10_c5_adv.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time

from engram import Engram, Config
from evals.locomo_adapter import (
    load_locomo, _untag, _ingest, _session_id_of,
)
from evals.io_utils import atomic_write_json

_STOP = set(
    "a an the and or but of in on at to for from with by is are was were be been being have has had do does did i you he she it we they my your his her our their this that these those s t not no yes about as if then so very can could should would will just like just it's i'm don't won't didn't because what which who whose where when why how all any some many much more most few less also too either neither such only own same than into out up down off over under after before during while".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if len(t) > 3 and t not in _STOP}


def _make_embedder(name: str | None):
    if name is None:
        return None, "none(BM25-only)"
    if name == "hashtrigram":
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        return HashTrigramEmbeddingProvider(dimension=256), "HashTrigram-256"
    if name in ("st", "minilm", "sentence_transformer"):
        from engram.providers.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider(), "SentenceTransformer-MiniLM-384"
    raise ValueError(f"unknown embedder: {name!r}")


def run(dataset: str, vws: list[float], max_instances: int, k: int,
        embedder: str | None = "hashtrigram") -> dict:
    samples = load_locomo(dataset, max_instances=max_instances)
    emb_provider, emb_label = _make_embedder(embedder)

    # Per-vw, per-query record. We index queries deterministically.
    out_per_vw: dict[float, list[dict]] = {vw: [] for vw in vws}

    for sample in samples:
        # build per-vw engines, ingest once each
        engines: dict[float, Engram] = {}
        tmpdirs: list[tempfile.TemporaryDirectory] = []
        try:
            for vw in vws:
                td = tempfile.TemporaryDirectory()
                tmpdirs.append(td)
                cfg = Config(path=td.name)
                cfg.security.max_events_per_minute = 0
                cfg.retrieval.vector_weight = float(vw)
                eng = (
                    Engram(config=cfg, embeddings=emb_provider)
                    if emb_provider is not None
                    else Engram(config=cfg)
                )
                _ingest(eng, sample)
                engines[vw] = eng

            for q in sample.qa:
                if str(q.category) != "5":
                    continue
                gold = set(q.evidence_sessions)
                if not gold:
                    continue
                adv = q.adversarial_answer or ""
                adv_toks = _content_tokens(adv)
                for vw, eng in engines.items():
                    results = eng.recall(q.question, limit=k)
                    sids = [_session_id_of(r) for r in results]
                    h1 = 1 if (sids and sids[0] in gold) else 0
                    rank = 0
                    for i, s in enumerate(sids, start=1):
                        if s in gold:
                            rank = i
                            break
                    if results:
                        top1 = results[0]
                        mem = getattr(top1, "memory", top1)
                        sid_t, content_t = _untag(getattr(mem, "content", "") or "")
                        top1_toks = _content_tokens(content_t)
                        if adv_toks:
                            ov = adv_toks & top1_toks
                            ov_frac = len(ov) / len(adv_toks)
                        else:
                            ov, ov_frac = set(), 0.0
                        srcs = getattr(top1, "sources", None) or {}
                        bm25_s = float(srcs["bm25"]) if isinstance(srcs, dict) and "bm25" in srcs else None
                        vec_s = float(srcs["vector"]) if isinstance(srcs, dict) and "vector" in srcs else None
                    else:
                        sid_t, content_t = None, ""
                        ov, ov_frac = set(), 0.0
                        bm25_s, vec_s = None, None

                    out_per_vw[vw].append({
                        "sample_id": sample.sample_id,
                        "question": q.question,
                        "gold_session": next(iter(gold), None),
                        "top1_session": sid_t,
                        "top1_content": content_t[:160],
                        "hit_at_1": h1,
                        "rank": rank,
                        "adv_answer": adv,
                        "adv_overlap_tokens": sorted(ov),
                        "adv_overlap_frac": round(ov_frac, 3),
                        "bm25_score": round(bm25_s, 4) if bm25_s is not None else None,
                        "vector_score": round(vec_s, 4) if vec_s is not None else None,
                    })
        finally:
            for eng in engines.values():
                try: eng.close()
                except Exception: pass
            for td in tmpdirs:
                try: td.cleanup()
                except Exception: pass

    # aggregate
    agg = {}
    for vw, rows in out_per_vw.items():
        if not rows:
            agg[str(vw)] = {"n": 0}
            continue
        h1 = sum(r["hit_at_1"] for r in rows) / len(rows)
        misses = [r for r in rows if r["hit_at_1"] == 0]
        miss_with_adv = [r for r in misses if r["adv_overlap_frac"] >= 0.5]
        miss_adv_rate = len(miss_with_adv) / len(misses) if misses else 0.0
        # Among misses, how often does top1 sit in the GOLD session anyway (right session, wrong turn)?
        miss_in_gold_sess = sum(1 for r in misses if r["top1_session"] == r["gold_session"])
        agg[str(vw)] = {
            "n": len(rows),
            "hit_at_1": round(h1, 4),
            "n_misses": len(misses),
            "miss_with_adv_overlap_ge_0.5": len(miss_with_adv),
            "miss_adv_overlap_rate": round(miss_adv_rate, 4),
            "miss_top1_in_gold_session": miss_in_gold_sess,
        }

    return {"per_vw_summary": agg, "per_query": out_per_vw}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--vws", default="0.0,0.5", help="comma-separated vector_weight values")
    p.add_argument("--max-instances", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=["hashtrigram", "st", "minilm", "sentence_transformer", "none"])
    args = p.parse_args()
    vws = [float(x) for x in args.vws.split(",") if x.strip()]
    emb = None if args.embedder == "none" else args.embedder
    t0 = time.monotonic()
    res = run(args.dataset, vws, args.max_instances, args.k, embedder=emb)
    dt = time.monotonic() - t0
    print(json.dumps(res["per_vw_summary"], indent=2))
    print(f"# elapsed {dt:.1f}s")
    if args.out:
        # write a JSON-stringifiable version
        out = {
            "per_vw_summary": res["per_vw_summary"],
            "per_query": {str(k): v for k, v in res["per_query"].items()},
            "elapsed_s": round(dt, 1),
        }
        atomic_write_json(args.out, out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

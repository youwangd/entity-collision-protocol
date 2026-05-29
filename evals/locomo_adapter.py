"""LoCoMo adapter.

LoCoMo (Maharana et al., arXiv:2402.17753) is a long-conversation memory
benchmark where each *sample* is a multi-session dialogue between two
speakers over weeks/months, plus a list of QA pairs. Each QA carries an
`evidence` field naming which session(s) — and sometimes which turn(s) —
support the answer. Categories include single-hop, multi-hop, temporal,
open-domain, and adversarial.

Public release shape (snap-research/locomo on GitHub):

    [
      {
        "sample_id": "...",
        "conversation": {
          "session_1_date_time": "...",
          "session_1": [{"speaker": "Alice", "text": "...", "dia_id": "D1:1"}, ...],
          "session_2_date_time": "...",
          "session_2": [...],
          ...
        },
        "qa": [
          {"question": "...", "answer": "...", "category": 1,
           "evidence": ["D3:5", "D7:2"], "adversarial_answer": null}
        ]
      }, ...
    ]

`evidence` items look like `D<session>:<turn>`. We score session-level
recall: hit@k = the gold *session* appears among the top-k retrieved
memories' session ids.

We don't score answer correctness — answer generation is downstream of
memory and out of v0.2 scope. Same rationale as LongMemEval.

Usage (CLI):
    python -m evals.locomo_adapter --dataset $LOCOMO_PATH \
        --max-instances 5 --k 10 --out bench/results/locomo_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram import Engram, Config
from evals.io_utils import atomic_write_json


@dataclass
class LoCoMoQA:
    question: str
    answer: str
    category: str
    evidence_sessions: list[str] = field(default_factory=list)
    adversarial_answer: str | None = None


@dataclass
class LoCoMoSample:
    sample_id: str
    sessions: list[dict]  # each: {"id": str, "date": str|None, "turns": [{"speaker","content"}]}
    qa: list[LoCoMoQA]


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_DATE_KEY_RE = re.compile(r"^session_(\d+)_date_time$")
_EVIDENCE_RE = re.compile(r"^D(\d+)(?::\d+)?$")


def _parse_conversation(conv: dict) -> list[dict]:
    """Convert the snap-research conversation dict to an ordered list of
    sessions. Tolerant of either {"session_N": [...]} flat layout or a
    "sessions" list."""
    if "sessions" in conv and isinstance(conv["sessions"], list):
        # already normalized
        out = []
        for i, s in enumerate(conv["sessions"]):
            sid = str(s.get("id", f"D{i+1}"))
            turns = []
            for t in s.get("turns", []):
                if isinstance(t, dict):
                    turns.append({
                        "speaker": str(t.get("speaker", t.get("role", "user"))),
                        "content": str(t.get("text", t.get("content", ""))),
                    })
                else:
                    turns.append({"speaker": "user", "content": str(t)})
            out.append({"id": sid, "date": s.get("date"), "turns": turns})
        return out

    # flat: session_N + session_N_date_time
    by_idx: dict[int, dict] = {}
    for k, v in conv.items():
        m = _SESSION_KEY_RE.match(k)
        if m:
            idx = int(m.group(1))
            d = by_idx.setdefault(idx, {"id": f"D{idx}", "date": None, "turns": []})
            turns = []
            for t in v or []:
                if isinstance(t, dict):
                    turns.append({
                        "speaker": str(t.get("speaker", t.get("role", "user"))),
                        "content": str(t.get("text", t.get("content", ""))),
                    })
                else:
                    turns.append({"speaker": "user", "content": str(t)})
            d["turns"] = turns
            continue
        m = _DATE_KEY_RE.match(k)
        if m:
            idx = int(m.group(1))
            d = by_idx.setdefault(idx, {"id": f"D{idx}", "date": None, "turns": []})
            d["date"] = v
    return [by_idx[i] for i in sorted(by_idx)]


def _evidence_to_sessions(evidence) -> list[str]:
    """Map evidence items like 'D3:5' or 'D3' to session ids 'D3'."""
    if not evidence:
        return []
    if isinstance(evidence, str):
        evidence = [evidence]
    out = []
    for e in evidence:
        e = str(e).strip()
        m = _EVIDENCE_RE.match(e)
        if m:
            out.append(f"D{int(m.group(1))}")
        elif e.startswith("D") and e[1:].isdigit():
            out.append(e)
    # dedup, preserve order
    seen, dedup = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def load_locomo(path: str | os.PathLike, max_instances: int | None = None) -> list[LoCoMoSample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {p}. Set LOCOMO_PATH or pass --dataset."
        )
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        raw = raw.get("samples") or raw.get("data") or [raw]
    if not isinstance(raw, list):
        raise ValueError(f"expected top-level list, got {type(raw).__name__}")

    out: list[LoCoMoSample] = []
    for entry in raw:
        conv = entry.get("conversation") or {}
        sessions = _parse_conversation(conv)
        qas = []
        for q in entry.get("qa", []) or []:
            qas.append(LoCoMoQA(
                question=str(q.get("question", "")),
                answer=str(q.get("answer", "") if q.get("answer") is not None else ""),
                category=str(q.get("category", "unknown")),
                evidence_sessions=_evidence_to_sessions(q.get("evidence")),
                adversarial_answer=q.get("adversarial_answer"),
            ))
        out.append(LoCoMoSample(
            sample_id=str(entry.get("sample_id", entry.get("id", f"sample_{len(out)}"))),
            sessions=sessions,
            qa=qas,
        ))
        if max_instances is not None and len(out) >= max_instances:
            break
    return out


_TAG_PREFIX = "[locomo_session="
_TAG_SUFFIX = "] "


def _tag(sid: str, content: str) -> str:
    return f"{_TAG_PREFIX}{sid}{_TAG_SUFFIX}{content}"


def _untag(content: str) -> tuple[str | None, str]:
    if content.startswith(_TAG_PREFIX):
        end = content.find(_TAG_SUFFIX, len(_TAG_PREFIX))
        if end != -1:
            return content[len(_TAG_PREFIX):end], content[end + len(_TAG_SUFFIX):]
    return None, content


def _ingest(eng: Engram, sample: LoCoMoSample) -> int:
    n = 0
    for sess in sample.sessions:
        sid = sess["id"]
        for turn in sess["turns"]:
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker") or "user"
            line = f"{speaker}: {text}"
            eng.remember(_tag(sid, line))
            n += 1
    return n


def _session_id_of(result) -> str | None:
    mem = getattr(result, "memory", result)
    sid, _ = _untag(getattr(mem, "content", "") or "")
    return sid


def _build_arm_config(
    arm: str,
    qe_dominance: float | None,
    sp_alpha: float,
    sp_pool: int,
    qe_type_purity_min: float | None = None,
    qe_backend: str = "heuristic",
) -> Config | None:
    """Mirror of evals.longmemeval_adapter._build_config so LoCoMo can run
    matched arms (baseline | prf | share_prior | both) with identical knob
    semantics. Returns None for the baseline so callers can fall back to a
    fresh Config(path=tmp) per sample (preserves existing behavior)."""
    if arm == "baseline":
        return None
    cfg = Config()
    if arm in ("prf", "both"):
        cfg.retrieval.query_expansion_min_dominance = qe_dominance
        if qe_type_purity_min is not None:
            cfg.retrieval.query_expansion_type_purity_min = qe_type_purity_min
        cfg.retrieval.entity_ner = qe_backend
    if arm in ("share_prior", "both"):
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.share_prior_alpha = sp_alpha
        cfg.retrieval.rerank_pool_size = sp_pool
    return cfg


def run_locomo(
    dataset_path: str | os.PathLike,
    max_instances: int = 5,
    k: int = 10,
    config: Config | None = None,
    vector_weight: float | None = None,
    embedder: str | None = None,
    save_bm25_signals: bool = False,
) -> dict:
    """Run LoCoMo against Engram. One Engram instance per *sample* (since
    each sample's haystack is its own conversation). All QAs in that
    sample query the same instance."""
    samples = load_locomo(dataset_path, max_instances=max_instances)
    if not samples:
        return {"error": "no samples", "n_samples": 0}

    # Resolve embedding provider once (cold-load is amortized across samples).
    emb_provider = None
    emb_label = None
    if embedder == "hashtrigram":
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        emb_provider = HashTrigramEmbeddingProvider(dimension=256)
        emb_label = "HashTrigram-256"
    elif embedder in ("st", "sentence_transformer", "minilm"):
        from engram.providers.embeddings import SentenceTransformerProvider
        emb_provider = SentenceTransformerProvider()
        emb_label = "SentenceTransformer-MiniLM-384"
    elif embedder in ("bge_large", "bge-large"):
        from engram.providers.embeddings import SentenceTransformerProvider
        emb_provider = SentenceTransformerProvider("BAAI/bge-large-en-v1.5")
        emb_label = "BGE-large-1024"
    elif embedder is None:
        emb_label = "none(BM25-only)"
    else:
        raise ValueError(f"unknown embedder: {embedder!r}")

    per_cat_h1: dict[str, list[int]] = {}
    per_cat_hk: dict[str, list[int]] = {}
    overall_h1: list[int] = []
    overall_hk: list[int] = []
    ingest_lat: list[float] = []
    recall_lat: list[float] = []
    per_query: list[dict] = []
    n_q = 0
    n_q_with_evidence = 0
    n_memories_total = 0

    for sample in samples:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config or Config(path=tmp)
            cfg.path = tmp
            cfg.security.max_events_per_minute = 0
            if vector_weight is not None:
                cfg.retrieval.vector_weight = float(vector_weight)
            eng = Engram(config=cfg, embeddings=emb_provider) if emb_provider is not None else Engram(config=cfg)
            try:
                t0 = time.monotonic()
                n_mem = _ingest(eng, sample)
                ingest_lat.append((time.monotonic() - t0) * 1000)
                n_memories_total += n_mem

                for q in sample.qa:
                    n_q += 1
                    gold = set(q.evidence_sessions)
                    if not gold:
                        # No evidence — can't score recall. Skip but still
                        # report it in the summary so we know coverage.
                        continue
                    n_q_with_evidence += 1

                    t0 = time.monotonic()
                    results = eng.recall(q.question, limit=k)
                    recall_lat.append((time.monotonic() - t0) * 1000)

                    sids = [_session_id_of(r) for r in results]
                    h1 = 1 if (sids and sids[0] in gold) else 0
                    hk = 1 if any(s in gold for s in sids) else 0
                    # 1-indexed rank of first gold hit, 0 if miss
                    rank = 0
                    for i, s in enumerate(sids, start=1):
                        if s in gold:
                            rank = i
                            break
                    rr = (1.0 / rank) if rank > 0 else 0.0
                    overall_h1.append(h1)
                    overall_hk.append(hk)
                    per_cat_h1.setdefault(q.category, []).append(h1)
                    per_cat_hk.setdefault(q.category, []).append(hk)
                    per_query.append({
                        "sample_id": sample.sample_id,
                        "category": q.category,
                        "rank": rank,
                        "hit_at_1": h1,
                        "hit_at_k": hk,
                        "reciprocal_rank": rr,
                    })
                    if save_bm25_signals:
                        from evals._signals import (
                            compute_bm25_top_gap,
                            normalized_gap,
                            crowdedness,
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

    def _agg(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_samples": len(samples),
        "n_questions": n_q,
        "n_questions_scored": n_q_with_evidence,
        "k": k,
        "vector_weight": vector_weight,
        "embedder": emb_label,
        "n_memories_total": n_memories_total,
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
    p.add_argument("--dataset", type=str, default=os.environ.get("LOCOMO_PATH"))
    p.add_argument("--max-instances", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--vector-weight", type=float, default=None,
                   help="override cfg.retrieval.vector_weight")
    p.add_argument("--embedder", type=str, default=None,
                   choices=[None, "hashtrigram", "st", "minilm", "sentence_transformer",
                            "bge_large", "bge-large"],
                   help="embedding provider; None means BM25-only (no vector path)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-bm25-signals", action="store_true",
                   help="capture per-query BM25 top1/top2/gap/norm_gap/crowd@95 for adaptive-vw analysis")
    p.add_argument(
        "--arm",
        choices=["baseline", "prf", "share_prior", "both"],
        default="baseline",
        help="Treatment arm (mirrors longmemeval_adapter): baseline | prf | share_prior | both",
    )
    p.add_argument("--qe-dominance", type=float, default=0.3)
    p.add_argument("--qe-type-purity-min", type=float, default=None,
                   help="Type-purity gate for typed PRF (default: None = legacy heuristic)")
    p.add_argument("--qe-backend", type=str, default="heuristic",
                   choices=["heuristic", "spacy_sm"],
                   help="NER backend for query expansion")
    p.add_argument("--sp-alpha", type=float, default=0.10)
    p.add_argument("--sp-pool", type=int, default=20)
    args = p.parse_args()
    if not args.dataset:
        raise SystemExit("--dataset or $LOCOMO_PATH required")
    arm_cfg = _build_arm_config(
        args.arm, args.qe_dominance, args.sp_alpha, args.sp_pool,
        qe_type_purity_min=args.qe_type_purity_min,
        qe_backend=args.qe_backend,
    )
    metrics = run_locomo(args.dataset, max_instances=args.max_instances,
                         k=args.k, vector_weight=args.vector_weight,
                         embedder=args.embedder,
                         config=arm_cfg,
                         save_bm25_signals=args.save_bm25_signals)
    metrics["arm"] = args.arm
    if args.arm != "baseline":
        metrics["arm_config"] = {
            "qe_dominance": args.qe_dominance,
            "qe_type_purity_min": args.qe_type_purity_min,
            "qe_backend": args.qe_backend,
            "sp_alpha": args.sp_alpha,
            "sp_pool": args.sp_pool,
        }
    print(json.dumps(metrics, indent=2))
    if args.out:
        atomic_write_json(args.out, metrics)
        print(f"[locomo] wrote {args.out}")


if __name__ == "__main__":
    main()

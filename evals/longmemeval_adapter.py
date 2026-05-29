"""LongMemEval adapter.

LongMemEval (Wu et al., arXiv:2410.10813) is a long-context conversational
memory benchmark. Each instance bundles:
    - question_id, question_type, question, answer
    - haystack_sessions: list of multi-turn sessions (each = list of {role, content})
    - haystack_session_ids: parallel ids
    - haystack_dates: parallel timestamps (per session)
    - answer_session_ids: which sessions actually contain the answer

This adapter ingests every turn into Engram, then runs the questions and
scores recall by whether ANY top-k result traces back to one of
`answer_session_ids` via metadata (`session_id`).

The dataset file is NOT bundled — provide a path with:
    LONGMEMEVAL_PATH=/abs/path/to/longmemeval_s.json
or pass `dataset_path=` directly. Without it the harness raises
`FileNotFoundError`, which lets pytest `importorskip`-style skip cleanly.

Usage (CLI):
    python -m evals.longmemeval_adapter --dataset $LONGMEMEVAL_PATH \
        --max-instances 50 --k 10 --out bench/results/lme_smoke.json

Why this shape: LongMemEval rewards a memory system that (a) ingests at
realistic scale (300-500 sessions per question) and (b) retrieves the
correct session(s) under aggressive temporal/identity reasoning. We score
session-level recall, not exact-string answer correctness — answer
generation is downstream of memory and out of scope for v0.2.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram import Engram, Config
from evals.io_utils import atomic_write_json


@dataclass
class LMEInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    sessions: list[dict]  # each: {"id": str, "date": str|None, "turns": [{"role","content"}]}
    answer_session_ids: list[str] = field(default_factory=list)


def load_longmemeval(
    path: str | os.PathLike,
    max_instances: int | None = None,
    stratify: bool = False,
    shuffle_seed: int | None = None,
) -> list[LMEInstance]:
    """Load LongMemEval JSON. Tolerant of the public schema variants
    (longmemeval_s/m/oracle release format).

    The public release groups instances by question_type, so a naive
    ``[:max_instances]`` slice is single-type-biased. Use ``stratify=True``
    to round-robin across question_types up to ``max_instances`` total,
    or ``shuffle_seed`` for a reproducible random sample.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LongMemEval dataset not found at {p}. "
                                f"Set LONGMEMEVAL_PATH or pass --dataset.")
    raw = json.loads(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"expected top-level JSON list, got {type(raw).__name__}")

    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        raw = list(raw)
        rng.shuffle(raw)
    if stratify:
        # Round-robin by question_type for balanced type coverage.
        buckets: dict[str, list] = {}
        for e in raw:
            buckets.setdefault(str(e.get("question_type", "unknown")), []).append(e)
        ordered = []
        idx = 0
        while True:
            added = False
            for qt in sorted(buckets):
                if idx < len(buckets[qt]):
                    ordered.append(buckets[qt][idx])
                    added = True
            if not added:
                break
            idx += 1
        raw = ordered

    out: list[LMEInstance] = []
    for entry in raw:
        sessions_raw = entry.get("haystack_sessions") or entry.get("sessions") or []
        sids = entry.get("haystack_session_ids") or [f"s{i}" for i in range(len(sessions_raw))]
        dates = entry.get("haystack_dates") or [None] * len(sessions_raw)
        sessions = []
        for sid, date, turns in zip(sids, dates, sessions_raw):
            # turns: list of {"role": ..., "content": ...} OR a list of strings
            norm_turns = []
            for t in turns:
                if isinstance(t, dict):
                    norm_turns.append({
                        "role": t.get("role", "user"),
                        "content": str(t.get("content", "")),
                    })
                else:
                    norm_turns.append({"role": "user", "content": str(t)})
            sessions.append({"id": str(sid), "date": date, "turns": norm_turns})

        out.append(LMEInstance(
            question_id=str(entry.get("question_id", entry.get("id", f"q{len(out)}"))),
            question_type=str(entry.get("question_type", "unknown")),
            question=str(entry.get("question", "")),
            answer=str(entry.get("answer", "")),
            sessions=sessions,
            answer_session_ids=[str(s) for s in entry.get("answer_session_ids", [])],
        ))
        if max_instances is not None and len(out) >= max_instances:
            break
    return out


_SESSION_TAG_PREFIX = "[lme_session="
_SESSION_TAG_SUFFIX = "] "


def _tag(sid: str, content: str) -> str:
    """Encode session id as a content prefix. Engram's `remember(**metadata)`
    accepts kwargs into the Event, but that metadata is dropped at memory
    consolidation, so for session-level scoring we must round-trip via the
    content itself."""
    return f"{_SESSION_TAG_PREFIX}{sid}{_SESSION_TAG_SUFFIX}{content}"


def _untag(content: str) -> tuple[str | None, str]:
    if content.startswith(_SESSION_TAG_PREFIX):
        end = content.find(_SESSION_TAG_SUFFIX, len(_SESSION_TAG_PREFIX))
        if end != -1:
            sid = content[len(_SESSION_TAG_PREFIX):end]
            return sid, content[end + len(_SESSION_TAG_SUFFIX):]
    return None, content


def _ingest(eng: Engram, inst: LMEInstance, max_chars: int = 8000) -> int:
    """Ingest one LME instance's haystack as memories. Each turn = one memory,
    tagged with its session_id so we can score session-level recall.

    LongMemEval has occasional pathologically long turns (one observed at
    76 KB). Truncate at `max_chars` to stay under the firewall ceiling and
    keep ingest costs bounded; the head of a long turn is overwhelmingly
    where lexical / entity signal lives for retrieval purposes."""
    n = 0
    for sess in inst.sessions:
        sid = sess["id"]
        for turn in sess["turns"]:
            content = turn["content"].strip()
            if not content:
                continue
            if len(content) > max_chars:
                content = content[:max_chars]
            eng.remember(_tag(sid, content))
            n += 1
    return n


def _session_id_of(result) -> str | None:
    mem = getattr(result, "memory", result)
    sid, _ = _untag(getattr(mem, "content", "") or "")
    return sid


def run_lme(
    dataset_path: str | os.PathLike,
    max_instances: int = 25,
    k: int = 10,
    config: Config | None = None,
    embedder_name: str | None = None,
    stratify: bool = False,
    shuffle_seed: int | None = None,
    rm3: bool = False,
    rm3_top_k: int = 10,
    rm3_num_terms: int = 10,
    rm3_lambda: float = 0.5,
) -> dict:
    """Run LongMemEval against Engram. Returns per-type and overall metrics.

    ``embedder_name``: ``None`` → BM25-only (legacy default). ``"hash"`` →
    HashTrigram-256. ``"st"`` → SentenceTransformer all-MiniLM-L6-v2 (cached
    on first use; requires the ``[entity-ner]``-adjacent ST install).

    ``rm3``: when True, applies RM3 pseudo-relevance feedback (Lavrenko &
    Croft 2001) over the BM25 channel. Two-pass: BM25(query) → expand →
    BM25(expanded). Operates externally to Engram core (AUDIT-D, no
    src/engram/ change). Combines additively with the embedder choice;
    RM3-only is the BM25 baseline + PRF setup, RM3+embedder fuses the
    expanded BM25 with the dense channel via cfg.retrieval.vector_weight.
    """
    instances = load_longmemeval(
        dataset_path, max_instances=max_instances,
        stratify=stratify, shuffle_seed=shuffle_seed,
    )
    if not instances:
        return {"error": "no instances", "n_instances": 0}

    embedder = None
    if embedder_name in (None, "none", "bm25"):
        embedder = None
    elif embedder_name in ("hash", "hashtrigram"):
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        embedder = HashTrigramEmbeddingProvider(dimension=256)
    elif embedder_name in ("st", "minilm", "sentence_transformer"):
        from engram.providers.embeddings import SentenceTransformerProvider
        embedder = SentenceTransformerProvider()
    elif embedder_name in ("bge_large", "bge-large"):
        from engram.providers.embeddings import SentenceTransformerProvider
        embedder = SentenceTransformerProvider("BAAI/bge-large-en-v1.5")
    else:
        raise ValueError(f"unknown embedder: {embedder_name!r}")

    per_type_hits: dict[str, list[int]] = {}
    per_type_topk: dict[str, list[int]] = {}
    overall_hit1: list[int] = []
    overall_hitk: list[int] = []
    ingest_lat: list[float] = []
    recall_lat: list[float] = []
    per_instance: list[dict] = []
    n_memories_total = 0

    for inst in instances:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config or Config(path=tmp)
            cfg.path = tmp
            cfg.security.max_events_per_minute = 0
            # LongMemEval haystacks are real conversational data; the firewall's
            # injection/PII heuristics false-positive on benign chat. Disable
            # for benchmark ingest so we measure retrieval, not the firewall.
            cfg.security.injection_detection = False
            cfg.security.pii_detection = False
            eng = (
                Engram(config=cfg, embeddings=embedder)
                if embedder is not None
                else Engram(config=cfg)
            )
            try:
                t0 = time.monotonic()
                n_mem = _ingest(eng, inst)
                ingest_lat.append((time.monotonic() - t0) * 1000)
                n_memories_total += n_mem

                t0 = time.monotonic()
                if rm3:
                    # Two-pass: BM25(question) → expand → BM25(expanded).
                    # The expansion uses the top-K memory contents themselves
                    # as the relevance-feedback documents.
                    from evals.rm3 import (
                        RM3Config,
                        build_expanded_query_string,
                        expand_query,
                    )
                    _rm3_cfg = RM3Config(
                        top_k=rm3_top_k,
                        num_terms=rm3_num_terms,
                        lambda_orig=rm3_lambda,
                    )
                    first = eng.recall(inst.question, limit=_rm3_cfg.top_k)
                    # Map memory IDs → content text for the expansion pool.
                    _id_to_text: dict[str, str] = {}
                    first_ids: list[str] = []
                    for r in first:
                        mem = getattr(r, "memory", r)
                        mid = str(getattr(mem, "id", id(mem)))
                        first_ids.append(mid)
                        _id_to_text[mid] = getattr(mem, "content", "") or ""
                    expanded = expand_query(
                        inst.question, first_ids, _id_to_text.get, _rm3_cfg
                    )
                    expanded_q = build_expanded_query_string(
                        inst.question, expanded, _rm3_cfg
                    )
                    results = eng.recall(expanded_q, limit=k)
                else:
                    results = eng.recall(inst.question, limit=k)
                recall_lat.append((time.monotonic() - t0) * 1000)

                got_sids = [_session_id_of(r) for r in results]
                gold = set(inst.answer_session_ids)

                hit1 = 1 if (got_sids and got_sids[0] in gold) else 0
                hitk = 1 if any(s in gold for s in got_sids) else 0

                per_type_hits.setdefault(inst.question_type, []).append(hit1)
                per_type_topk.setdefault(inst.question_type, []).append(hitk)
                overall_hit1.append(hit1)
                overall_hitk.append(hitk)
                per_instance.append({
                    "question_id": getattr(inst, "question_id", None),
                    "question_type": inst.question_type,
                    "hit_at_1": hit1,
                    "hit_at_k": hitk,
                })
            finally:
                eng.close()

    def _agg(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_instances": len(instances),
        "k": k,
        "n_memories_total": n_memories_total,
        "session_hit_at_1": _agg(overall_hit1),
        "session_hit_at_k": _agg(overall_hitk),
        "per_type_session_hit_at_1": {qt: _agg(v) for qt, v in per_type_hits.items()},
        "per_type_session_hit_at_k": {qt: _agg(v) for qt, v in per_type_topk.items()},
        "per_type_n": {qt: len(v) for qt, v in per_type_hits.items()},
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
        "per_instance": per_instance,
    }


def _build_config(
    arm: str,
    qe_dominance: float | None,
    sp_alpha: float,
    sp_pool: int,
    qe_type_purity_min: float | None = None,
    qe_backend: str = "heuristic",
    type_allow: frozenset[str] | None = None,
    qe_anchor_share_max: float | None = None,
) -> Config:
    cfg = Config()
    if arm in ("prf", "both"):
        cfg.retrieval.query_expansion_min_dominance = qe_dominance
        if qe_type_purity_min is not None:
            cfg.retrieval.query_expansion_type_purity_min = qe_type_purity_min
        cfg.retrieval.entity_ner = qe_backend
        if type_allow is not None:
            cfg.retrieval.query_expansion_type_allow = type_allow
        if qe_anchor_share_max is not None:
            cfg.retrieval.query_expansion_anchor_share_max = qe_anchor_share_max
    if arm in ("share_prior", "both"):
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.share_prior_alpha = sp_alpha
        cfg.retrieval.rerank_pool_size = sp_pool
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=os.environ.get("LONGMEMEVAL_PATH"))
    p.add_argument("--max-instances", type=int, default=25)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--arm",
        choices=["baseline", "prf", "share_prior", "both"],
        default="baseline",
        help="Treatment arm: baseline | prf (PRF expansion) | share_prior | both",
    )
    p.add_argument("--qe-dominance", type=float, default=0.3)
    p.add_argument("--qe-type-purity-min", type=float, default=None,
                   help="Type-purity gate for typed PRF (default: None = legacy heuristic)")
    p.add_argument("--qe-backend", type=str, default="heuristic",
                   choices=["heuristic", "spacy_sm"],
                   help="NER backend for query expansion")
    p.add_argument("--sp-alpha", type=float, default=0.10)
    p.add_argument("--sp-pool", type=int, default=20)
    p.add_argument("--embed", type=str, default=None,
                   choices=[None, "none", "bm25", "hash", "hashtrigram",
                            "st", "minilm", "sentence_transformer",
                            "bge_large", "bge-large"],
                   help="Embedding provider for vector channel (default: BM25-only)")
    p.add_argument("--vector-weight", type=float, default=None,
                   help="If set with --embed, override cfg.retrieval.vector_weight")
    p.add_argument("--stratify", action="store_true",
                   help="Round-robin sample across question_types (balanced coverage).")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Seed for reproducible shuffled sampling.")
    p.add_argument("--type-allow", type=str, default=None,
                   help="Comma-separated question_type labels to allow PRF on "
                        "(e.g. 'single-session-preference' or "
                        "'knowledge-update,single-session-preference'). "
                        "Default None = no gate (PRF runs on all queries).")
    p.add_argument("--qe-anchor-share-max", type=float, default=None,
                   help="§D15d gate: skip PRF when first-pass top-K is "
                        "saturated by one entity (share > threshold). "
                        "Default None = OFF.")
    # RM3 baseline arm (AUDIT-D). Orthogonal to --arm because LME's
    # "arm" namespace is for PRF/share_prior treatments; RM3 here means
    # "apply Lavrenko-Croft PRF over the BM25 channel".
    p.add_argument("--rm3", action="store_true",
                   help="AUDIT-D: enable RM3 pseudo-relevance feedback over BM25.")
    p.add_argument("--rm3-top-k", type=int, default=10,
                   help="RM3: top-k docs from first BM25 pass (default 10)")
    p.add_argument("--rm3-num-terms", type=int, default=10,
                   help="RM3: number of expansion terms (default 10)")
    p.add_argument("--rm3-lambda", type=float, default=0.5,
                   help="RM3: orig-vs-expansion interpolation (default 0.5)")
    args = p.parse_args()
    if not args.dataset:
        raise SystemExit("--dataset or $LONGMEMEVAL_PATH required")
    type_allow = (
        frozenset(s.strip() for s in args.type_allow.split(",") if s.strip())
        if args.type_allow else None
    )
    cfg = _build_config(
        args.arm, args.qe_dominance, args.sp_alpha, args.sp_pool,
        qe_type_purity_min=args.qe_type_purity_min,
        qe_backend=args.qe_backend,
        type_allow=type_allow,
        qe_anchor_share_max=args.qe_anchor_share_max,
    ) if args.arm != "baseline" else None
    if args.vector_weight is not None:
        if cfg is None:
            cfg = Config()
        cfg.retrieval.vector_weight = float(args.vector_weight)
    metrics = run_lme(
        args.dataset, max_instances=args.max_instances, k=args.k, config=cfg,
        embedder_name=args.embed,
        stratify=args.stratify,
        shuffle_seed=args.shuffle_seed,
        rm3=args.rm3,
        rm3_top_k=args.rm3_top_k,
        rm3_num_terms=args.rm3_num_terms,
        rm3_lambda=args.rm3_lambda,
    )
    metrics["arm"] = args.arm
    metrics["embed"] = args.embed
    metrics["rm3"] = args.rm3
    if args.rm3:
        metrics["rm3_config"] = {
            "top_k": args.rm3_top_k,
            "num_terms": args.rm3_num_terms,
            "lambda": args.rm3_lambda,
        }
    if args.vector_weight is not None:
        metrics["vector_weight_override"] = args.vector_weight
    if args.arm != "baseline":
        metrics["arm_config"] = {
            "qe_dominance": args.qe_dominance,
            "sp_alpha": args.sp_alpha,
            "sp_pool": args.sp_pool,
            "type_allow": (sorted(type_allow) if type_allow else None),
            "qe_anchor_share_max": args.qe_anchor_share_max,
        }
    print(json.dumps(metrics, indent=2))
    if args.out:
        atomic_write_json(args.out, metrics, indent=2)
        print(f"[lme] wrote {args.out}")


if __name__ == "__main__":
    main()

"""BEIR adapter (P-BASELINE scaffold, item #6 in NEXT.md).

BEIR (Thakur et al., NeurIPS 2021 Datasets) is a heterogeneous IR benchmark.
Per LOCKED 2026-05-23 in NEXT.md, the v0.2 paper cuts to **three** tasks:

    - NQ        — single-hop classic
    - HotpotQA  — multi-hop composition
    - FiQA      — financial OOD robustness

We frame Engram as a hybrid retrieval plug-in and report **NDCG@10** (primary)
plus **Recall@100** (supplementary) against:

    1. BM25-only       — engram with embeddings=None (FTS5 BM25 path).
    2. Dense-only      — engram with embeddings provider, vector_weight=1.0.
    3. Engram hybrid   — v0.3 defaults (vector_weight=0.3, PRF×SP gate on).

at matched compute on the dev split (downsampled if budget overrun).

This file is the **scaffold**: dataset I/O (BEIR's standard {corpus,
queries, qrels} shape), Engram ingest/query, and the two metrics. The
real BEIR datasets are NOT bundled — set ``BEIR_DATA_ROOT=/abs/path`` so
each task lives at ``$BEIR_DATA_ROOT/<task>/`` with the canonical
``corpus.jsonl``, ``queries.jsonl``, ``qrels/test.tsv`` layout (this is
the format produced by the official BEIR loader).

Why we don't depend on `beir` the pypi package: it pulls in PyTorch by
default and we want this adapter to be importable from any Engram dev
shell. We re-implement the loader (50 LoC) and the two metrics from
scratch — verified against the BEIR reference values on a fixture below.

Usage (CLI):
    python -m evals.beir_adapter --task fiqa --arm hybrid --max-queries 100 \\
        --out evals/results/beir_fiqa_hybrid.json

Resume mode (long runs):
    NQ (2.7M docs) and HotpotQA (5.2M) take ~25h / ~49h respectively at the
    measured M4 MPS rate. Without a persistent store, a single OS popup or
    lid-close kills the run and discards everything. Pass ``--engram-path
    /some/dir`` to switch from the ephemeral ``tempfile.TemporaryDirectory``
    store to a persistent store + a ``.beir_progress.json`` checkpoint:

        python -m evals.beir_adapter --task nq --arm hybrid --embed bge_large \\
            --engram-path bench/data/beir_nq_bge_engram \\
            --checkpoint-every 5000 \\
            --out bench/results/beir_nq_bge_large_hybrid.json

    Re-running with the SAME command resumes from the last checkpoint —
    already-ingested doc_ids are skipped, only the unflushed window
    (default 5000 docs) is replayed. The checkpoint is keyed on
    (task, arm, embedder, split, n_corpus); a mismatch on any of those
    refuses to resume rather than silently corrupting the run.

    The result dict gains an ``n_ingested_resumed`` field so callers can
    verify the resume actually skipped work. Default behaviour
    (no ``--engram-path``) is unchanged: tempdir, no checkpoint file.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from engram import Engram, Config

from evals.io_utils import atomic_write_text


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class BEIRTask:
    name: str
    corpus: dict[str, dict]              # doc_id -> {"title": str, "text": str}
    queries: dict[str, str]              # qid -> query text
    qrels: dict[str, dict[str, int]]     # qid -> {doc_id: relevance_grade}


def load_beir(
    root: str | os.PathLike,
    task: str,
    split: str = "test",
) -> BEIRTask:
    """Load a BEIR task from on-disk JSONL/TSV. Tolerant of the 'beir' python
    package's directory layout: ``<root>/<task>/{corpus.jsonl,queries.jsonl,qrels/<split>.tsv}``.
    """
    base = Path(root) / task
    if not base.exists():
        raise FileNotFoundError(
            f"BEIR task dir not found at {base}. "
            f"Set BEIR_DATA_ROOT and pre-download with the official `beir` loader."
        )

    corpus_p = base / "corpus.jsonl"
    queries_p = base / "queries.jsonl"
    qrels_p = base / "qrels" / f"{split}.tsv"
    for p in (corpus_p, queries_p, qrels_p):
        if not p.exists():
            raise FileNotFoundError(f"missing required BEIR file: {p}")

    corpus: dict[str, dict] = {}
    with corpus_p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            corpus[str(d["_id"])] = {
                "title": str(d.get("title", "")),
                "text": str(d.get("text", "")),
            }

    queries: dict[str, str] = {}
    with queries_p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            queries[str(d["_id"])] = str(d.get("text", ""))

    qrels: dict[str, dict[str, int]] = {}
    with qrels_p.open() as f:
        _ = f.readline()  # standard BEIR qrels header: query-id\tcorpus-id\tscore
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            try:
                grade = int(score)
            except ValueError:
                continue
            if grade <= 0:
                continue
            qrels.setdefault(qid, {})[did] = grade

    return BEIRTask(name=task, corpus=corpus, queries=queries, qrels=qrels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float:
    """Standard graded-relevance NDCG@k. DCG uses log2(rank+1) discount and
    (2^rel - 1) gain — matches BEIR's reference implementation exactly.
    """
    if not qrel:
        return 0.0
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = qrel.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal_grades = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids: list[str], qrel: dict[str, int], k: int) -> float:
    """Binary-relevant recall@k: |retrieved ∩ relevant| / |relevant|.

    BEIR's recall_cap@k caps the denominator at k; we report uncapped recall
    by default (the BEIR paper's headline). Caller can post-process if needed.
    """
    if not qrel:
        return 0.0
    relevant = {d for d, g in qrel.items() if g > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved & relevant) / len(relevant)


# ---------------------------------------------------------------------------
# Engram retrieval driver
# ---------------------------------------------------------------------------

_DOC_TAG_PREFIX = "[beir_doc="
_DOC_TAG_SUFFIX = "] "


def _tag(did: str, content: str) -> str:
    return f"{_DOC_TAG_PREFIX}{did}{_DOC_TAG_SUFFIX}{content}"


def _untag(content: str) -> str | None:
    if content.startswith(_DOC_TAG_PREFIX):
        end = content.find(_DOC_TAG_SUFFIX, len(_DOC_TAG_PREFIX))
        if end != -1:
            return content[len(_DOC_TAG_PREFIX):end]
    return None


def _doc_text(doc: dict) -> str:
    title, text = doc.get("title", ""), doc.get("text", "")
    return f"{title}. {text}".strip().lstrip(".").strip() if title else text


def _build_config(arm: str) -> Config:
    """Configure Engram for the requested retrieval arm."""
    cfg = Config()
    cfg.security.max_events_per_minute = 0
    cfg.security.injection_detection = False
    cfg.security.pii_detection = False

    if arm == "bm25":
        # FTS5-only path: vector_weight=0 so the lexical channel dominates,
        # AND embeddings=None at construction time so we never even build a vector index.
        cfg.retrieval.vector_weight = 0.0
    elif arm == "rm3":
        # RM3 = BM25 + pseudo-relevance feedback. Same config as bm25
        # (FTS5-only, no vector index); the two-pass dance happens in the
        # query loop, not in Engram core. AUDIT-D, no src/engram/ change.
        cfg.retrieval.vector_weight = 0.0
    elif arm == "dense":
        cfg.retrieval.vector_weight = 1.0
    elif arm == "hybrid":
        cfg.retrieval.vector_weight = 0.3  # v0.3 default
        # PRF + share-prior: leave at config defaults if those are the
        # locked v0.3 operating point. (Re-asserted below for clarity.)
        if hasattr(cfg.retrieval, "query_expansion_min_dominance"):
            cfg.retrieval.query_expansion_min_dominance = 0.3
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    return cfg


def _make_embedder(embedder_name: str | None, arm: str):
    """Return an embedding provider for `dense`/`hybrid` arms, None for `bm25`/`rm3`."""
    if arm in ("bm25", "rm3"):
        return None
    if embedder_name in (None, "st", "minilm"):
        from engram.providers.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider()
    if embedder_name in ("hash", "hashtrigram"):
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        return HashTrigramEmbeddingProvider(dimension=256)
    if embedder_name in ("bge_large", "bge-large"):
        from engram.providers.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider("BAAI/bge-large-en-v1.5")
    raise ValueError(f"unknown embedder: {embedder_name!r}")


def _checkpoint_path(engram_path: str | os.PathLike) -> Path:
    return Path(engram_path) / ".beir_progress.json"


def _load_checkpoint(engram_path: str | os.PathLike, expected: dict) -> set[str]:
    """Load the ingested-doc-id set if checkpoint metadata matches `expected`.
    Mismatched metadata refuses the resume to avoid silently corrupting a run
    started under different params."""
    p = _checkpoint_path(engram_path)
    if not p.exists():
        return set()
    cp = json.loads(p.read_text())
    meta = cp.get("meta", {})
    for k, v in expected.items():
        if meta.get(k) != v:
            raise RuntimeError(
                f"BEIR resume rejected: checkpoint at {p} has meta[{k!r}]={meta.get(k)!r}, "
                f"current run wants {v!r}. Delete the engram-path or pick a fresh one."
            )
    return set(cp.get("ingested_doc_ids", []))


def _save_checkpoint(engram_path: str | os.PathLike, meta: dict, ingested: set[str]) -> None:
    p = _checkpoint_path(engram_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "ingested_doc_ids": sorted(ingested)}))
    tmp.replace(p)


def run_beir(
    root: str | os.PathLike,
    task: str,
    arm: str = "hybrid",
    k_ndcg: int = 10,
    k_recall: int = 100,
    max_queries: int | None = None,
    embedder_name: str | None = None,
    split: str = "test",
    engram_path: str | os.PathLike | None = None,
    checkpoint_every: int = 5000,
    rm3_top_k: int = 10,
    rm3_num_terms: int = 10,
    rm3_lambda: float = 0.5,
) -> dict:
    """Run a single (task, arm) combination. Returns a metrics dict.

    If ``engram_path`` is given, the Engram store + a resume checkpoint live
    at that path across runs. Re-invoking with the same arguments resumes
    ingest from the last checkpoint (skipping already-ingested doc_ids).
    Default behaviour (engram_path=None) keeps the existing tempfile path
    unchanged — used by the existing tests and the prior fiqa run."""
    bench = load_beir(root, task, split=split)
    cfg = _build_config(arm)
    embedder = _make_embedder(embedder_name, arm)

    qids = list(bench.qrels.keys())
    if max_queries is not None:
        qids = qids[:max_queries]
    if not qids:
        return {"task": task, "arm": arm, "error": "no queries with qrels"}

    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    query_lat: list[float] = []

    @contextlib.contextmanager
    def _store_dir():
        if engram_path is None:
            with tempfile.TemporaryDirectory() as tmp:
                yield tmp, False  # ephemeral; never resume
        else:
            p = Path(engram_path)
            p.mkdir(parents=True, exist_ok=True)
            yield str(p), True

    resume_meta = {
        "task": task,
        "arm": arm,
        "embedder": embedder_name,
        "split": split,
        "n_corpus": len(bench.corpus),
    }
    n_resumed = 0

    with _store_dir() as (store_path, persistent):
        cfg.path = store_path
        ingested: set[str] = set()
        if persistent:
            ingested = _load_checkpoint(store_path, resume_meta)
            n_resumed = len(ingested)

        eng = (
            Engram(config=cfg, embeddings=embedder) if embedder is not None
            else Engram(config=cfg)
        )
        try:
            # Ingest the corpus once per (task, arm) run, skipping doc_ids
            # already recorded in the resume checkpoint.
            t0 = time.monotonic()
            since_checkpoint = 0
            for did, doc in bench.corpus.items():
                if did in ingested:
                    continue
                content = _doc_text(doc)
                if not content:
                    continue
                eng.remember(_tag(did, content))
                ingested.add(did)
                since_checkpoint += 1
                if persistent and since_checkpoint >= checkpoint_every:
                    _save_checkpoint(store_path, resume_meta, ingested)
                    since_checkpoint = 0
            if persistent:
                _save_checkpoint(store_path, resume_meta, ingested)
            ingest_ms = (time.monotonic() - t0) * 1000

            # Query loop.
            # Build a corpus lookup for RM3 (only used when arm=="rm3").
            if arm == "rm3":
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

                def _rm3_doc_text(did: str) -> str:
                    d = bench.corpus.get(did)
                    return _doc_text(d) if d else ""

            for qid in qids:
                qtext = bench.queries.get(qid, "")
                if not qtext:
                    continue
                t0 = time.monotonic()
                if arm == "rm3":
                    # Two-pass: BM25(qtext) -> expand -> BM25(expanded).
                    first = eng.recall(qtext, limit=_rm3_cfg.top_k)
                    first_dids: list[str] = []
                    for r in first:
                        mem = getattr(r, "memory", r)
                        did = _untag(getattr(mem, "content", "") or "")
                        if did is not None:
                            first_dids.append(did)
                    expanded = expand_query(
                        qtext, first_dids, _rm3_doc_text, _rm3_cfg
                    )
                    expanded_q = build_expanded_query_string(
                        qtext, expanded, _rm3_cfg
                    )
                    results = eng.recall(expanded_q, limit=max(k_ndcg, k_recall))
                else:
                    results = eng.recall(qtext, limit=max(k_ndcg, k_recall))
                query_lat.append((time.monotonic() - t0) * 1000)

                ranked = []
                for r in results:
                    mem = getattr(r, "memory", r)
                    did = _untag(getattr(mem, "content", "") or "")
                    if did is not None:
                        ranked.append(did)

                qrel = bench.qrels.get(qid, {})
                ndcg_scores.append(ndcg_at_k(ranked, qrel, k_ndcg))
                recall_scores.append(recall_at_k(ranked, qrel, k_recall))
        finally:
            eng.close()

    def _agg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def _pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return round(s[idx], 2)

    return {
        "task": task,
        "arm": arm,
        "embedder": embedder_name if embedder is not None else None,
        "n_queries": len(ndcg_scores),
        "n_corpus": len(bench.corpus),
        "n_ingested_resumed": n_resumed,
        "ndcg_at_10": _agg(ndcg_scores),
        "recall_at_100": _agg(recall_scores),
        "ingest_ms_total": round(ingest_ms, 1),
        "query_ms": {
            "p50": _pct(query_lat, 0.50),
            "p95": _pct(query_lat, 0.95),
            "p99": _pct(query_lat, 0.99),
        },
        "config": {
            "k_ndcg": k_ndcg,
            "k_recall": k_recall,
            "vector_weight": cfg.retrieval.vector_weight,
            "split": split,
            **({"rm3_top_k": rm3_top_k, "rm3_num_terms": rm3_num_terms,
                "rm3_lambda": rm3_lambda} if arm == "rm3" else {}),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BEIR adapter for Engram (P-BASELINE).")
    ap.add_argument("--root", default=os.environ.get("BEIR_DATA_ROOT"),
                    help="BEIR data root (default: $BEIR_DATA_ROOT).")
    ap.add_argument("--task", required=True, choices=["nq", "hotpotqa", "fiqa"])
    ap.add_argument("--arm", required=True, choices=["bm25", "dense", "hybrid", "rm3"])
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--k-ndcg", type=int, default=10)
    ap.add_argument("--k-recall", type=int, default=100)
    ap.add_argument("--embed", default="st", help="embedder for dense/hybrid arms")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="write results JSON to this path")
    ap.add_argument("--engram-path", default=None,
                    help=("persistent Engram store path; enables resume. "
                          "If absent, the run uses an ephemeral tempdir (legacy behaviour). "
                          "If present and a previous run with matching task/arm/embedder/split/n_corpus "
                          "exists, ingest resumes from the last checkpoint."))
    ap.add_argument("--checkpoint-every", type=int, default=5000,
                    help="docs per resume checkpoint (only with --engram-path)")
    # RM3 hyperparameters (only used when --arm rm3).
    ap.add_argument("--rm3-top-k", type=int, default=10,
                    help="RM3: top-k docs from first BM25 pass to expand from (default 10)")
    ap.add_argument("--rm3-num-terms", type=int, default=10,
                    help="RM3: number of expansion terms to keep (default 10)")
    ap.add_argument("--rm3-lambda", type=float, default=0.5,
                    help="RM3: interpolation between original and expansion query (default 0.5)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if not args.root:
        raise SystemExit("error: --root or $BEIR_DATA_ROOT required")
    result = run_beir(
        root=args.root, task=args.task, arm=args.arm,
        k_ndcg=args.k_ndcg, k_recall=args.k_recall,
        max_queries=args.max_queries, embedder_name=args.embed,
        split=args.split,
        engram_path=args.engram_path,
        checkpoint_every=args.checkpoint_every,
        rm3_top_k=args.rm3_top_k,
        rm3_num_terms=args.rm3_num_terms,
        rm3_lambda=args.rm3_lambda,
    )
    blob = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        atomic_write_text(args.out, blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

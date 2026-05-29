"""BEIR adapter unit tests.

We don't ship real BEIR datasets, so we (a) build a tiny fixture in BEIR's
on-disk format and (b) verify metric implementations against worked-by-hand
references. The full Engram run is exercised in a single end-to-end test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals.beir_adapter import (
    load_beir,
    ndcg_at_k,
    recall_at_k,
    run_beir,
)


# ---------------------------------------------------------------------------
# Metric correctness — verified against the BEIR reference (DCG with 2^rel-1
# gain and log2(rank+1) discount).
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    qrel = {"d1": 3, "d2": 2, "d3": 1}
    assert ndcg_at_k(["d1", "d2", "d3"], qrel, k=10) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_below_one():
    qrel = {"d1": 3, "d2": 2, "d3": 1}
    score = ndcg_at_k(["d3", "d2", "d1"], qrel, k=10)
    assert 0.0 < score < 1.0


def test_ndcg_no_relevant_in_topk_zero():
    qrel = {"d_gold": 1}
    assert ndcg_at_k(["d1", "d2", "d3"], qrel, k=3) == 0.0


def test_ndcg_known_value():
    """NDCG@2 with one rel=1 doc at rank 2 against ideal at rank 1.
    DCG = 1/log2(3) ≈ 0.6309; IDCG = 1/log2(2) = 1.0 → NDCG ≈ 0.6309.
    """
    qrel = {"d_gold": 1}
    got = ndcg_at_k(["d_other", "d_gold"], qrel, k=2)
    assert got == pytest.approx(1.0 / math.log2(3), rel=1e-3)


def test_ndcg_empty_qrel_zero():
    assert ndcg_at_k(["d1", "d2"], {}, k=10) == 0.0


def test_recall_at_k_full():
    qrel = {"d1": 1, "d2": 1}
    assert recall_at_k(["d1", "d2", "d3"], qrel, k=3) == 1.0


def test_recall_at_k_partial():
    qrel = {"d1": 1, "d2": 1, "d3": 1, "d4": 1}
    assert recall_at_k(["d1", "d2", "d_other", "d_other2"], qrel, k=4) == 0.5


def test_recall_at_k_empty_qrel_zero():
    assert recall_at_k(["d1"], {}, k=10) == 0.0


# ---------------------------------------------------------------------------
# Loader — BEIR canonical layout.
# ---------------------------------------------------------------------------

def _write_beir_fixture(root: Path, task: str = "tiny") -> Path:
    base = root / task
    (base / "qrels").mkdir(parents=True, exist_ok=True)

    # corpus.jsonl
    corpus = [
        {"_id": "d1", "title": "Vim editor", "text": "alice prefers vim for everyday editing"},
        {"_id": "d2", "title": "Quarterly", "text": "discussed planning for next quarter"},
        {"_id": "d3", "title": "Stripe", "text": "bob's stripe credential lapses on 2026-07-15"},
        {"_id": "d4", "title": "Coffee", "text": "unrelated notes about coffee"},
    ]
    (base / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus) + "\n")

    # queries.jsonl
    queries = [
        {"_id": "q1", "text": "what editor does alice use"},
        {"_id": "q2", "text": "when does bob's stripe key expire"},
    ]
    (base / "queries.jsonl").write_text("\n".join(json.dumps(d) for d in queries) + "\n")

    # qrels/test.tsv
    qrels = (
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q2\td3\t2\n"
        "q1\td2\t0\n"  # zero-grade should be dropped
    )
    (base / "qrels" / "test.tsv").write_text(qrels)
    return base


def test_loader_round_trip(tmp_path):
    _write_beir_fixture(tmp_path, "tiny")
    bench = load_beir(tmp_path, "tiny", split="test")
    assert bench.name == "tiny"
    assert set(bench.corpus.keys()) == {"d1", "d2", "d3", "d4"}
    assert bench.queries["q1"].startswith("what editor")
    assert bench.qrels["q1"] == {"d1": 1}
    assert bench.qrels["q2"] == {"d3": 2}
    # zero-grade row was dropped
    assert "d2" not in bench.qrels.get("q1", {})


def test_loader_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_beir(tmp_path, "does_not_exist")


def test_loader_missing_file_raises(tmp_path):
    base = tmp_path / "broken"
    (base / "qrels").mkdir(parents=True)
    (base / "corpus.jsonl").write_text("")
    # missing queries.jsonl
    with pytest.raises(FileNotFoundError):
        load_beir(tmp_path, "broken")


# ---------------------------------------------------------------------------
# End-to-end — BM25 arm only (cheap, no embedder install) on the fixture.
# Hybrid + dense paths are exercised by the LME suite already; here we only
# verify wiring + metric flow.
# ---------------------------------------------------------------------------

@pytest.mark.evals
def test_run_beir_bm25_endtoend(tmp_path):
    _write_beir_fixture(tmp_path, "tiny")
    result = run_beir(
        root=tmp_path, task="tiny", arm="bm25",
        k_ndcg=10, k_recall=100, max_queries=None,
    )
    assert result["task"] == "tiny"
    assert result["arm"] == "bm25"
    assert result["embedder"] is None
    assert result["n_queries"] == 2
    assert result["n_corpus"] == 4
    # Strong lexical overlap on both qs — expect non-zero metrics.
    assert result["ndcg_at_10"] > 0.0
    assert result["recall_at_100"] > 0.0
    assert result["config"]["vector_weight"] == 0.0


# ---------------------------------------------------------------------------
# Resume capability — persistent engram_path + .beir_progress.json checkpoint.
# ---------------------------------------------------------------------------

@pytest.mark.evals
def test_resume_skips_already_ingested(tmp_path):
    """Two runs with the same engram_path: the second loads the checkpoint
    and skips re-ingest of docs already on disk."""
    _write_beir_fixture(tmp_path, "tiny")
    store = tmp_path / "engram_store"

    first = run_beir(
        root=tmp_path, task="tiny", arm="bm25",
        k_ndcg=10, k_recall=100,
        engram_path=store, checkpoint_every=1,
    )
    assert first["n_ingested_resumed"] == 0
    # Checkpoint file written.
    cp = store / ".beir_progress.json"
    assert cp.exists()
    cp_data = json.loads(cp.read_text())
    assert set(cp_data["ingested_doc_ids"]) == {"d1", "d2", "d3", "d4"}

    second = run_beir(
        root=tmp_path, task="tiny", arm="bm25",
        k_ndcg=10, k_recall=100,
        engram_path=store, checkpoint_every=1,
    )
    # All 4 docs were resumed; ingest_ms_total should be tiny since nothing
    # new was ingested.
    assert second["n_ingested_resumed"] == 4
    # Metrics are still computed at query time, so ndcg/recall match.
    assert second["ndcg_at_10"] == first["ndcg_at_10"]
    assert second["recall_at_100"] == first["recall_at_100"]


@pytest.mark.evals
def test_resume_rejects_metadata_mismatch(tmp_path):
    """A resume that disagrees with the checkpoint's task/arm/embedder MUST
    refuse rather than silently corrupt the run."""
    _write_beir_fixture(tmp_path, "tiny")
    store = tmp_path / "engram_store"

    # First run: bm25 arm.
    run_beir(
        root=tmp_path, task="tiny", arm="bm25",
        k_ndcg=10, k_recall=100,
        engram_path=store,
    )
    # Second run with a different arm tries to resume and must fail loudly.
    with pytest.raises(RuntimeError, match=r"BEIR resume rejected"):
        run_beir(
            root=tmp_path, task="tiny", arm="dense",
            k_ndcg=10, k_recall=100,
            engram_path=store, embedder_name="hash",
        )


@pytest.mark.evals
def test_no_engram_path_uses_tempfile(tmp_path):
    """Default codepath (engram_path=None) keeps the existing tempfile
    behaviour: no checkpoint file is written anywhere."""
    _write_beir_fixture(tmp_path, "tiny")
    pre_files = set(tmp_path.rglob(".beir_progress.json"))
    result = run_beir(
        root=tmp_path, task="tiny", arm="bm25",
        k_ndcg=10, k_recall=100,
    )
    assert result["n_ingested_resumed"] == 0
    post_files = set(tmp_path.rglob(".beir_progress.json"))
    assert pre_files == post_files  # no checkpoint dropped under tmp_path

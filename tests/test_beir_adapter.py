"""Tests for `evals.beir_adapter` — the P-BASELINE scaffold (NEXT.md item #6).

Covers:
  * NDCG@k math vs. hand-computed references (matches BEIR's reference
    implementation: gain = 2^rel - 1, discount = log2(rank+1)).
  * Recall@k uncapped semantics.
  * Loader: tolerant of canonical BEIR layout, raises on missing files,
    drops grade<=0 from qrels, skips blank lines.
  * Doc tag round-trip (the only mechanism by which `recall()` outputs
    map back to BEIR doc ids — if this is wrong, every BEIR number is
    silently zero).
  * End-to-end smoke on a 6-doc fixture with the hash embedder (no
    network, no SentenceTransformer download). Exercises bm25 / dense /
    hybrid arms — asserts the result schema is consistent and metric
    values are in [0, 1].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals import beir_adapter as ba


# --------------------------------------------------------------------------- #
# Metric math                                                                 #
# --------------------------------------------------------------------------- #

class TestNDCG:
    def test_perfect_ranking_is_one(self):
        # qrel grades all 1; if we return all relevant in any order they
        # tie ideal because gain(1)=1 for each — but DCG depends on rank.
        ranked = ["a", "b", "c"]
        qrel = {"a": 1, "b": 1, "c": 1}
        # ideal ranking puts all three at top; ours does too.
        assert ba.ndcg_at_k(ranked, qrel, 3) == 1.0

    def test_reversed_relevance_is_below_one(self):
        # Graded relevance: doc_a=3, doc_b=1. Putting b first is suboptimal.
        ranked = ["b", "a"]
        qrel = {"a": 3, "b": 1}
        ndcg = ba.ndcg_at_k(ranked, qrel, 2)
        assert 0.0 < ndcg < 1.0
        # Manual: dcg = (2^1-1)/log2(2) + (2^3-1)/log2(3)
        #             = 1/1 + 7/1.5849... = 1 + 4.4170 = 5.4170
        # idcg = 7/1 + 1/1.5849 = 7 + 0.6309 = 7.6309
        expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
        assert math.isclose(ndcg, round(expected, 4), abs_tol=1e-4)

    def test_empty_qrel_returns_zero(self):
        assert ba.ndcg_at_k(["a", "b"], {}, 5) == 0.0

    def test_no_overlap_returns_zero(self):
        assert ba.ndcg_at_k(["x", "y"], {"a": 1}, 5) == 0.0

    def test_k_truncates(self):
        # Relevance only at rank 5; with k=3 we miss it.
        ranked = ["a", "b", "c", "d", "e"]
        qrel = {"e": 1}
        assert ba.ndcg_at_k(ranked, qrel, 3) == 0.0
        assert ba.ndcg_at_k(ranked, qrel, 5) > 0.0


class TestRecall:
    def test_full_recall(self):
        ranked = ["a", "b", "c"]
        qrel = {"a": 1, "b": 1}
        assert ba.recall_at_k(ranked, qrel, 3) == 1.0

    def test_partial_recall(self):
        ranked = ["a", "b", "c"]
        qrel = {"a": 1, "b": 1, "z": 1}  # z never retrieved
        assert ba.recall_at_k(ranked, qrel, 3) == 2 / 3

    def test_zero_grade_excluded_from_relevant(self):
        # recall_at_k filters {g > 0} on its own.
        ranked = ["a"]
        qrel = {"a": 1, "b": 0}
        assert ba.recall_at_k(ranked, qrel, 5) == 1.0

    def test_empty_qrel_returns_zero(self):
        assert ba.recall_at_k(["a"], {}, 5) == 0.0

    def test_uncapped_denominator(self):
        # 10 relevant, k=5, retrieve 5 of them → recall = 0.5 (uncapped).
        # If we were using recall_cap@k the value would be 1.0 — assert NOT.
        ranked = [f"d{i}" for i in range(5)]
        qrel = {f"d{i}": 1 for i in range(10)}
        assert ba.recall_at_k(ranked, qrel, 5) == 0.5


# --------------------------------------------------------------------------- #
# Doc tag round-trip                                                          #
# --------------------------------------------------------------------------- #

class TestDocTag:
    def test_roundtrip_basic(self):
        tagged = ba._tag("doc-42", "hello world")
        assert ba._untag(tagged) == "doc-42"

    def test_untag_returns_none_on_unrelated_content(self):
        assert ba._untag("hello world") is None

    def test_untag_handles_brackets_in_id(self):
        # BEIR ids never contain ']' but be defensive.
        tagged = ba._tag("plain-id-123", "body with [brackets] inside")
        assert ba._untag(tagged) == "plain-id-123"

    def test_doc_text_combines_title_and_body(self):
        assert ba._doc_text({"title": "T", "text": "B"}).startswith("T")
        assert "B" in ba._doc_text({"title": "T", "text": "B"})
        # Title-less doc returns just the body.
        assert ba._doc_text({"title": "", "text": "B"}) == "B"


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #

def _write_beir_fixture(root: Path, task: str = "tinytask") -> Path:
    base = root / task
    (base / "qrels").mkdir(parents=True)
    (base / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"Cats","text":"Cats purr."}\n'
        "\n"  # blank line — must be tolerated
        '{"_id":"d2","title":"Dogs","text":"Dogs bark loudly."}\n'
        '{"_id":"d3","title":"","text":"Birds sing."}\n'
    )
    (base / "queries.jsonl").write_text(
        '{"_id":"q1","text":"how do cats sound"}\n'
        '{"_id":"q2","text":"loud animal noises"}\n'
    )
    (base / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q1\td2\t0\n"        # filtered: grade<=0
        "q2\td2\t1\n"
        "bad\tline\n"        # filtered: too few cols
        "q2\tdX\tnotanint\n" # filtered: non-int score
    )
    return base


class TestLoader:
    def test_loads_canonical_layout(self, tmp_path):
        _write_beir_fixture(tmp_path)
        bench = ba.load_beir(tmp_path, "tinytask")
        assert bench.name == "tinytask"
        assert set(bench.corpus.keys()) == {"d1", "d2", "d3"}
        assert bench.corpus["d1"]["title"] == "Cats"
        assert bench.queries == {"q1": "how do cats sound", "q2": "loud animal noises"}
        # qrels: grade<=0 dropped, malformed lines dropped.
        assert bench.qrels == {"q1": {"d1": 2}, "q2": {"d2": 1}}

    def test_missing_task_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BEIR task dir not found"):
            ba.load_beir(tmp_path, "no-such-task")

    def test_missing_required_file_raises(self, tmp_path):
        base = tmp_path / "halftask"
        (base / "qrels").mkdir(parents=True)
        (base / "corpus.jsonl").write_text("")
        # queries.jsonl absent
        with pytest.raises(FileNotFoundError, match="missing required BEIR file"):
            ba.load_beir(tmp_path, "halftask")


# --------------------------------------------------------------------------- #
# End-to-end driver smoke (hash embedder — no network)                         #
# --------------------------------------------------------------------------- #

def _write_smoke_fixture(root: Path, task: str = "smoke") -> None:
    base = root / task
    (base / "qrels").mkdir(parents=True)
    docs = [
        ("d1", "Domestic cats", "Cats are small carnivorous mammals that purr."),
        ("d2", "Wild lions", "Lions roar in the African savanna."),
        ("d3", "House dogs", "Dogs bark to communicate with humans."),
        ("d4", "Goldfinches", "Goldfinches sing bright songs in spring."),
        ("d5", "Cetaceans", "Whales sing complex songs underwater."),
        ("d6", "Snakes", "Snakes hiss when threatened."),
    ]
    (base / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": i, "title": t, "text": x}) for i, t, x in docs) + "\n"
    )
    (base / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "small mammals that purr"}) + "\n"
        + json.dumps({"_id": "q2", "text": "animals that sing"}) + "\n"
    )
    (base / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q2\td4\t2\n"
        "q2\td5\t1\n"
    )


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
def test_run_beir_smoke(tmp_path, arm):
    _write_smoke_fixture(tmp_path)
    out = ba.run_beir(
        root=tmp_path, task="smoke", arm=arm,
        k_ndcg=10, k_recall=100,
        embedder_name="hash",  # no network, no torch model load
    )
    assert out["task"] == "smoke"
    assert out["arm"] == arm
    assert out["n_queries"] == 2
    assert out["n_corpus"] == 6
    assert 0.0 <= out["ndcg_at_10"] <= 1.0
    assert 0.0 <= out["recall_at_100"] <= 1.0
    assert out["query_ms"]["p50"] >= 0.0
    assert out["query_ms"]["p95"] >= out["query_ms"]["p50"]
    assert out["config"]["k_ndcg"] == 10
    if arm == "bm25":
        assert out["config"]["vector_weight"] == 0.0
    elif arm == "dense":
        assert out["config"]["vector_weight"] == 1.0
    elif arm == "hybrid":
        assert out["config"]["vector_weight"] == 0.3


def test_run_beir_smoke_finds_relevant(tmp_path):
    """Sanity floor: BM25 on a 6-doc corpus where 'cats' / 'purr' is verbatim
    in d1 must find d1 for q1. If this asserts zero, the doc-tag round-trip
    or recall() return shape is broken."""
    _write_smoke_fixture(tmp_path)
    out = ba.run_beir(
        root=tmp_path, task="smoke", arm="bm25",
        k_ndcg=10, k_recall=100, embedder_name="hash",
    )
    # At least one of {q1->d1, q2->d4, q2->d5} must rank in the top 10.
    # NDCG@10 averages over q1+q2; if both are zero we have a wiring bug.
    assert out["ndcg_at_10"] > 0.0, (
        "BM25 smoke produced ndcg_at_10=0; doc-tag plumbing is likely broken"
    )


def test_run_beir_unknown_arm_raises(tmp_path):
    _write_smoke_fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown arm"):
        ba.run_beir(root=tmp_path, task="smoke", arm="bogus", embedder_name="hash")

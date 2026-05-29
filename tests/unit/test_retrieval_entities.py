"""Tests for the lightweight entity extractor (D1 retrieval channel).

Covers:
- Multi-word capitalized spans
- Single-token capitalized names (mid-sentence)
- Sentence-initial single capitals are rejected
- Stopwords filtered when alone
- Acronyms picked up
- Empty / None inputs are safe
- Jaccard helper bounds and symmetry
"""
from __future__ import annotations

from engram.retrieval.entities import extract_entities, jaccard


def test_extract_multiword_proper_noun():
    out = extract_entities("I met Alice Smith yesterday in New York.")
    assert "alice smith" in out
    assert "new york" in out


def test_extract_acronym():
    out = extract_entities("She works at NASA on the JWST project.")
    assert "nasa" in out
    assert "jwst" in out


def test_sentence_initial_single_capital_rejected():
    # "The" at offset 0, single token → must not appear as entity.
    out = extract_entities("The cat sat on the mat.")
    assert "the" not in out


def test_midsentence_single_capital_kept_unless_stopword():
    out = extract_entities("yesterday I saw Alice")
    assert "alice" in out


def test_stopword_alone_is_filtered():
    out = extract_entities("she went to Monday meetings on Friday")
    assert "monday" not in out
    assert "friday" not in out


def test_empty_input_is_safe():
    assert extract_entities("") == set()
    assert extract_entities(None) == set()  # type: ignore[arg-type]


def test_jaccard_bounds():
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_jaccard_symmetric():
    a = {"alice", "bob"}
    b = {"bob", "carol"}
    assert jaccard(a, b) == jaccard(b, a)


def test_jaccard_accepts_iterables():
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3

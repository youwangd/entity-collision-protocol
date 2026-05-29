"""Unit tests for evals/rm3.py — AUDIT-D RM3 baseline arm."""
from collections import Counter

import pytest

from evals.rm3 import (
    RM3Config,
    _tokenize,
    build_expanded_query_string,
    expand_query,
)


def test_tokenize_lowercases_and_splits():
    assert _tokenize("Hello, World!") == ["hello", "world"]
    assert _tokenize("FiQA-Q-1") == ["fiqa", "q", "1"]
    assert _tokenize("") == []
    assert _tokenize(None) == []  # robustness for missing doc text


def test_expand_query_empty_input_returns_empty():
    assert expand_query("any query", [], lambda _: "") == Counter()


def test_expand_query_picks_high_idf_terms():
    docs = {
        "d1": "mortgage rate refinance interest payment",
        "d2": "mortgage refinance interest annual",
        "d3": "mortgage interest tax deduction",
    }
    cfg = RM3Config(top_k=3, num_terms=3, lambda_orig=0.5)
    expanded = expand_query(
        original_query="mortgage rate",
        first_pass_doc_ids=["d1", "d2", "d3"],
        get_doc_text=docs.get,
        cfg=cfg,
    )
    # Original-query terms must be excluded (RM3 convention).
    assert "mortgage" not in expanded
    assert "rate" not in expanded
    # 'interest' appears in all 3 docs and should rank highly.
    assert "interest" in expanded
    # Weights sum to 1.0 within float tolerance.
    assert sum(expanded.values()) == pytest.approx(1.0, abs=1e-9)
    # All weights non-negative.
    assert all(w >= 0 for w in expanded.values())


def test_build_expanded_query_string_includes_original_and_expansion():
    expanded = Counter({"interest": 0.6, "refinance": 0.4})
    result = build_expanded_query_string(
        "mortgage rate",
        expanded,
        cfg=RM3Config(lambda_orig=0.5),
        repetition_scale=10,
    )
    tokens = result.split()
    # Original-query terms appear at least once.
    assert "mortgage" in tokens
    assert "rate" in tokens
    # Expansion terms appear at least once.
    assert "interest" in tokens
    assert "refinance" in tokens
    # Higher-weight expansion term repeats more.
    assert tokens.count("interest") >= tokens.count("refinance")


def test_top_k_clamps_to_provided_doc_ids():
    docs = {"d1": "alpha beta gamma"}
    cfg = RM3Config(top_k=10, num_terms=2)
    expanded = expand_query(
        original_query="alpha",
        first_pass_doc_ids=["d1"],
        get_doc_text=docs.get,
        cfg=cfg,
    )
    # With 1 doc and "alpha" excluded as orig-term, only "beta"/"gamma" remain.
    assert set(expanded.keys()) <= {"beta", "gamma"}
    assert sum(expanded.values()) == pytest.approx(1.0, abs=1e-9)


def test_stopwords_filter_excludes_terms():
    docs = {"d1": "the quick brown fox the the"}
    cfg = RM3Config(top_k=1, num_terms=5, stopwords=frozenset({"the"}))
    expanded = expand_query(
        original_query="fox",
        first_pass_doc_ids=["d1"],
        get_doc_text=docs.get,
        cfg=cfg,
    )
    assert "the" not in expanded
    # "fox" is original-query term; "quick"/"brown" should remain.
    assert "fox" not in expanded
    assert set(expanded.keys()) == {"quick", "brown"}


def test_missing_doc_text_does_not_crash():
    """If get_doc_text returns None or empty string for some IDs, skip them."""
    docs = {"d1": "real text here", "d2": ""}
    cfg = RM3Config(top_k=2, num_terms=2)
    expanded = expand_query(
        original_query="missing",
        first_pass_doc_ids=["d1", "d2", "d3"],  # d3 not in docs
        get_doc_text=lambda did: docs.get(did, ""),
        cfg=cfg,
    )
    # Only d1 contributes; "missing" is orig-term excluded.
    assert "real" in expanded or "text" in expanded or "here" in expanded

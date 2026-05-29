"""Tests for HashTrigramEmbeddingProvider — the lightweight zero-dep embedder
used in the evals harness as a non-MiniLM signal source.

Properties we care about (paper-grade defensibility):
  - L2-normalized output
  - Deterministic
  - Paraphrase pairs > unrelated pairs in cosine similarity
  - Empty / short / unicode inputs don't crash
"""
from __future__ import annotations

import math

import pytest

from engram.providers.embeddings import HashTrigramEmbeddingProvider


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_dimension_and_norm():
    p = HashTrigramEmbeddingProvider(dimension=128)
    v = p.embed("hello world")
    assert len(v) == 128
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_determinism():
    p = HashTrigramEmbeddingProvider(dimension=64)
    a = p.embed("the quick brown fox")
    b = p.embed("the quick brown fox")
    assert a == b


def test_paraphrase_beats_unrelated():
    p = HashTrigramEmbeddingProvider(dimension=512)
    v_dark1 = p.embed("user prefers dark mode")
    v_dark2 = p.embed("user likes dark theme")
    v_other = p.embed("the cat sat on the mat")
    # paraphrase should be substantially closer than unrelated content
    assert cos(v_dark1, v_dark2) > cos(v_dark1, v_other) + 0.2


def test_self_similarity_is_one():
    p = HashTrigramEmbeddingProvider(dimension=256)
    v = p.embed("anything goes here")
    assert cos(v, v) == pytest.approx(1.0, abs=1e-9)


def test_empty_and_short():
    p = HashTrigramEmbeddingProvider(dimension=64)
    # empty/short inputs produce some vector and don't raise
    assert len(p.embed("")) == 64
    assert len(p.embed("a")) == 64
    assert len(p.embed("ab")) == 64


def test_unicode_safe():
    p = HashTrigramEmbeddingProvider(dimension=64)
    v = p.embed("café — naïve résumé 日本語")
    assert len(v) == 64
    norm = math.sqrt(sum(x * x for x in v))
    assert 0.0 < norm <= 1.0 + 1e-9


def test_batch_matches_single():
    p = HashTrigramEmbeddingProvider(dimension=128)
    texts = ["alpha", "beta gamma", "δ ε ζ"]
    batch = p.embed_batch(texts)
    singles = [p.embed(t) for t in texts]
    assert batch == singles


def test_invalid_args():
    with pytest.raises(ValueError):
        HashTrigramEmbeddingProvider(dimension=0)
    with pytest.raises(ValueError):
        HashTrigramEmbeddingProvider(ngram=1)

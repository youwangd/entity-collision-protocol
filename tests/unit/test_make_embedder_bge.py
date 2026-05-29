"""Unit test: _make_embedder('bge_large') wires BGE-large-en-v1.5 correctly.

Locks in the BGE-large embedder choice as the third tier
(hash-trigram / MiniLM / BGE-large) for the entity-collision protocol.
"""
from __future__ import annotations

import pytest


def test_make_embedder_bge_large_returns_provider_with_dim_1024():
    """`bge_large` must return a SentenceTransformerProvider whose
    embedding dimension is 1024 (BGE-large-en-v1.5 spec)."""
    pytest.importorskip("sentence_transformers")

    from evals.ablation import _make_embedder

    provider = _make_embedder("bge_large")
    assert provider is not None, "bge_large should return a provider, not None"
    # SentenceTransformerProvider exposes `.dimension`.
    assert provider.dimension == 1024, (
        f"BGE-large-en-v1.5 must have dim=1024, got {provider.dimension}"
    )


def test_make_embedder_unknown_still_raises():
    """Sanity: extending the factory didn't break the unknown-name guard."""
    from evals.ablation import _make_embedder

    with pytest.raises(ValueError, match="unknown embedder"):
        _make_embedder("not_a_real_embedder")


def test_make_embedder_bge_distinct_from_st_minilm():
    """BGE and st (MiniLM) must produce different-dimension providers."""
    pytest.importorskip("sentence_transformers")

    from evals.ablation import _make_embedder

    bge = _make_embedder("bge_large")
    st = _make_embedder("st")
    assert bge.dimension != st.dimension, (
        f"BGE-large ({bge.dimension}) and MiniLM ({st.dimension}) "
        "should have distinct dimensions"
    )
    assert st.dimension == 384, f"MiniLM-L6 should be dim=384, got {st.dimension}"

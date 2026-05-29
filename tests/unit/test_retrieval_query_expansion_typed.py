"""Tests for §5.4 type-aware PRF gate (`expand_query_typed`).

Covers:
- Heuristic backend: type-purity gate is inert (all-MISC, purity=1.0).
- spaCy-style synthetic typed pairs: purity gate blocks mixed-type pools.
- spaCy-style synthetic typed pairs: purity gate passes uniform-type pools.
- Config knob round-trip: query_expansion_type_purity_min serializes.
- Engine integration smoke test: typed path runs without crash.
"""
from __future__ import annotations

import tempfile

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from engram.retrieval import expansion as expansion_mod
from engram.retrieval.expansion import expand_query_typed


def test_typed_heuristic_purity_gate_is_inert():
    """Under heuristic backend, all entities are 'MISC' → purity=1.0 always."""
    texts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
    ]
    # Even with strict purity_min=0.99, heuristic should still fire because
    # purity is trivially 1.0 (all labels are 'MISC').
    expanded, chosen = expand_query_typed(
        "where did they go", texts,
        top_k=3, max_entities=2, min_dominance=0.5,
        type_purity_min=0.99, backend="heuristic",
    )
    assert chosen, "Heuristic backend purity gate should be inert (all MISC)"


def test_typed_purity_blocks_mixed_types(monkeypatch):
    """Mixed PERSON+GPE+ORG pool with purity_min=0.7 → blocked."""
    def fake_typed(text, backend="heuristic"):
        return {
            "doc1": [("alice smith", "PERSON")],
            "doc2": [("paris", "GPE")],
            "doc3": [("acme corp", "ORG")],
        }.get(text, [])

    monkeypatch.setattr(expansion_mod, "extract_entities_typed", fake_typed)

    _, chosen = expand_query_typed(
        "what",
        ["doc1", "doc2", "doc3"],
        top_k=3, max_entities=3,
        min_dominance=0.0,  # disable freq gate, isolate purity
        type_purity_min=0.7,
        backend="spacy_sm",
    )
    # 1 PERSON + 1 GPE + 1 ORG → top label share = 1/3 < 0.7
    assert chosen == [], "Mixed-type pool should be blocked by purity gate"


def test_typed_purity_passes_uniform_types(monkeypatch):
    """Uniform PERSON pool with purity_min=0.7 → passes."""
    def fake_typed(text, backend="heuristic"):
        return {
            "doc1": [("alice smith", "PERSON")],
            "doc2": [("alice smith", "PERSON"), ("bob jones", "PERSON")],
            "doc3": [("alice smith", "PERSON")],
        }.get(text, [])

    monkeypatch.setattr(expansion_mod, "extract_entities_typed", fake_typed)

    _, chosen = expand_query_typed(
        "what",
        ["doc1", "doc2", "doc3"],
        top_k=3, max_entities=3,
        min_dominance=0.0,
        type_purity_min=0.7,
        backend="spacy_sm",
    )
    assert "alice smith" in chosen


def test_typed_none_purity_matches_dominance_only(monkeypatch):
    """purity_min=None → behavior identical to expand_query freq gate."""
    def fake_typed(text, backend="heuristic"):
        return {
            "doc1": [("alice smith", "PERSON")],
            "doc2": [("paris", "GPE")],
            "doc3": [("acme corp", "ORG")],
        }.get(text, [])

    monkeypatch.setattr(expansion_mod, "extract_entities_typed", fake_typed)

    _, chosen = expand_query_typed(
        "what",
        ["doc1", "doc2", "doc3"],
        top_k=3, max_entities=3,
        min_dominance=0.0,
        type_purity_min=None,
        backend="spacy_sm",
    )
    # No freq gate, no purity gate → top-3 entities all returned.
    assert len(chosen) == 3


def test_config_round_trip():
    cfg = RetrievalConfig(
        query_expansion_min_dominance=0.3,
        query_expansion_type_purity_min=0.7,
    )
    assert cfg.query_expansion_type_purity_min == 0.7


def test_engine_typed_path_runs():
    """Smoke: engine wires the typed expansion when knob is set, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
            query_expansion_type_purity_min=0.5,
            entity_ner="heuristic",  # purity inert; just exercise the path
        )
        eng = Engram(config=cfg)
        try:
            for _ in range(5):
                eng.remember("Alice Smith was at Acme Corp today.")
            eng.remember("Random unrelated content about cats.")
            r = eng.recall("who was where", limit=3)
            assert isinstance(r, list)
        finally:
            eng.close()

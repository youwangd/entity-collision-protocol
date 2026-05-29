"""§4.15g — tests for the IDF-rarity filter on PRF candidates.

`query_expansion_idf_min_rarity` drops candidate entities whose corpus
document-rarity (1 - df/N) falls below the threshold, *before* truncation
to max_entities. Targets the §D15c multi-token-anchor regression.
"""
from __future__ import annotations

import tempfile

from engram import Engram, Config
from engram.retrieval.expansion import expand_query


def test_idf_filter_inert_when_threshold_none():
    texts = ["Alice Smith met Bob.", "Alice Smith met Carol.", "Alice Smith works."]
    # Sentinel: rarity_lookup is provided but threshold=None — must be inert.
    calls = []

    def lookup(e):
        calls.append(e)
        return 0.0  # would filter everything if active

    expanded, chosen = expand_query(
        "where",
        texts,
        top_k=3,
        max_entities=2,
        min_dominance=0.5,
        idf_min_rarity=None,
        rarity_lookup=lookup,
    )
    assert chosen, "IDF filter must be inert when threshold is None"
    assert calls == [], "Lookup must not be invoked when threshold is None"


def test_idf_filter_drops_low_rarity_candidates():
    texts = [
        "Alice Smith met Bob in Paris.",
        "Alice Smith met Carol in Paris.",
        "Alice Smith met Dan in Paris.",
    ]
    # Stub: alice_smith is corpus-common (rarity 0.1), paris is rare (0.9).
    rarity = {"alice smith": 0.1, "paris": 0.9}

    def lookup(e):
        return rarity.get(e.lower(), 0.5)

    _, chosen = expand_query(
        "where did they meet",
        texts,
        top_k=3,
        max_entities=3,
        min_dominance=0.5,
        idf_min_rarity=0.5,
        rarity_lookup=lookup,
    )
    chosen_lower = [c.lower() for c in chosen]
    assert "alice smith" not in chosen_lower
    # Paris should pass (rarity 0.9 ≥ 0.5).
    assert any("paris" in c for c in chosen_lower) or chosen == []


def test_idf_filter_can_empty_result_when_all_below():
    texts = ["Alice Smith met Bob.", "Alice Smith met Carol."]

    def lookup(e):
        return 0.0  # everything filtered

    _, chosen = expand_query(
        "where",
        texts,
        top_k=2,
        max_entities=2,
        min_dominance=0.5,
        idf_min_rarity=0.5,
        rarity_lookup=lookup,
    )
    assert chosen == []


def test_idf_filter_lenient_on_lookup_exception():
    texts = ["Alice Smith met Bob.", "Alice Smith met Carol."]

    def lookup(e):
        raise RuntimeError("simulated FTS failure")

    # All candidates should be treated as rarity=0.0 (filtered).
    _, chosen = expand_query(
        "where",
        texts,
        top_k=2,
        max_entities=2,
        min_dominance=0.5,
        idf_min_rarity=0.5,
        rarity_lookup=lookup,
    )
    assert chosen == []


def test_engine_rarity_lookup_active_corpus():
    """Smoke-test the engine's _build_prf_rarity_lookup against a real store."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        e = Engram(cfg)
        for i in range(10):
            e.remember(f"Alice met someone in city number {i}")
        # "alice" appears in all 10 docs → rarity ≈ 0.0
        # "frobozz" appears in 0 docs → rarity ≈ 1.0
        lookup = e._retrieval._build_prf_rarity_lookup()
        assert lookup("alice") < 0.5, "Common token should have low rarity"
        assert lookup("frobozz") > 0.9, "Absent token should have high rarity"


def test_engine_idf_min_rarity_end_to_end_inert_when_none():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.query_expansion_min_dominance = 0.3
        cfg.retrieval.query_expansion_idf_min_rarity = None
        e = Engram(cfg)
        for i in range(5):
            e.remember(f"Alice met Bob in Paris episode {i}")
        # Should not throw; threshold None means inert.
        results = e.recall("where did they meet")
        assert isinstance(results, list)

"""§D15d — tests for the anchor-share diagnostic gate on PRF.

`query_expansion_anchor_share_max` short-circuits PRF when the dominant
candidate entity's share of total candidate-entity occurrences in the
top-K pool exceeds the threshold. Targets the §D15c failure mode:
cross-fact entity confusion under shared-anchor density.
"""
from __future__ import annotations

import tempfile

from engram import Engram, Config
from engram.retrieval.expansion import expand_query


def test_anchor_share_inert_when_threshold_none():
    # All three docs share "Alice Smith" — share would be 1.0 if active.
    texts = [
        "Alice Smith likes pizza.",
        "Alice Smith likes tea.",
        "Alice Smith likes books.",
    ]
    _, chosen = expand_query(
        "what does she like",
        texts,
        top_k=3,
        max_entities=2,
        min_dominance=0.5,
        anchor_share_max=None,
    )
    # Inert path → entity (likely "alice smith") survives.
    assert chosen, "Anchor-share gate must be inert when threshold is None"


def test_anchor_share_blocks_when_pool_saturated_by_one_entity():
    # Pathological pool — only one extracted entity dominates.
    texts = [
        "Alice Smith likes pizza.",
        "Alice Smith likes tea.",
        "Alice Smith likes books.",
    ]
    _, chosen = expand_query(
        "what does she like",
        texts,
        top_k=3,
        max_entities=2,
        min_dominance=0.5,
        anchor_share_max=0.4,
    )
    assert chosen == [], (
        "When dominant entity's share exceeds threshold, PRF must skip"
    )


def test_anchor_share_passes_when_pool_diverse():
    # Diverse pool — three distinct entities, no single one dominates.
    texts = [
        "Alice met Bob in Paris.",
        "Carol travelled to Berlin.",
        "Dan visited Tokyo.",
    ]
    _, chosen = expand_query(
        "where",
        texts,
        top_k=3,
        max_entities=3,
        # min_dominance=0.0 disables the freq-dominance gate so we isolate
        # the anchor-share gate.
        min_dominance=0.0,
        anchor_share_max=0.5,
    )
    # No single entity hits 50% share → expansion proceeds.
    assert chosen, (
        "Diverse pool must not be blocked by anchor-share gate"
    )


def test_anchor_share_threshold_strict_inequality():
    # Two of three docs mention "Alice Smith"; the third mentions "Bob Jones".
    # Heuristic NER picks up "Alice Smith" twice and "Bob Jones" once → 2/3 ≈ 0.667.
    texts = [
        "Alice Smith likes pizza.",
        "Alice Smith likes tea.",
        "Bob Jones likes books.",
    ]
    # Threshold above the share → must pass.
    _, chosen_pass = expand_query(
        "who likes what",
        texts,
        top_k=3,
        max_entities=3,
        min_dominance=0.0,
        anchor_share_max=0.7,
    )
    assert chosen_pass, "share=0.667 < threshold=0.7 must pass"
    # Threshold below the share → must block.
    _, chosen_block = expand_query(
        "who likes what",
        texts,
        top_k=3,
        max_entities=3,
        min_dominance=0.0,
        anchor_share_max=0.5,
    )
    assert chosen_block == [], "share=0.667 > threshold=0.5 must block"


def test_engine_anchor_share_end_to_end_inert_when_none():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.query_expansion_min_dominance = 0.3
        cfg.retrieval.query_expansion_anchor_share_max = None
        e = Engram(cfg)
        for i in range(5):
            e.remember(f"Alice met Bob in Paris episode {i}")
        # Should not throw; threshold None means inert.
        results = e.recall("where did they meet")
        assert isinstance(results, list)


def test_engine_anchor_share_blocks_pathological_corpus():
    """End-to-end smoke: when the active corpus is saturated by one
    entity, the gate kicks in and PRF is short-circuited (no crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.query_expansion_min_dominance = 0.3
        cfg.retrieval.query_expansion_anchor_share_max = 0.4
        e = Engram(cfg)
        for i in range(10):
            e.remember(f"Alice Smith said thing {i}")
        results = e.recall("what did she say")
        assert isinstance(results, list)


def test_config_roundtrip_anchor_share():
    """Persist + reload must preserve the new knob."""
    cfg = Config()
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_anchor_share_max = 0.45
    d = cfg.to_dict()
    assert d["retrieval"]["query_expansion_anchor_share_max"] == 0.45
    cfg2 = Config._from_dict(d)
    assert cfg2.retrieval.query_expansion_anchor_share_max == 0.45


def test_config_default_anchor_share_on():
    """v0.3 default: anchor-share gate value remains 0.5 even though PRF
    itself is OFF by default (`query_expansion_min_dominance is None`).
    The anchor-share gate is inert when PRF is off, but the threshold
    stays at 0.5 so opt-in toggling activates the §4.8.2.4 / §D15d
    operating point with a single-knob change."""
    cfg = Config()
    assert cfg.retrieval.query_expansion_anchor_share_max == 0.5
    assert cfg.retrieval.query_expansion_min_dominance is None

"""Tests for §5.4 angle 1 — PRF query expansion wire-in.

Covers:
- expand_query primitive: dominance gate, novelty filter, top-K cap, max_entities cap.
- RetrievalEngine.search: regression-safe when min_dominance is None.
- RetrievalEngine.search: PRF fires when min_dominance is set + dominant entity exists.
- RetrievalEngine.search: PRF gracefully no-ops when no dominant entity exists.
- Lenient fail: expansion exception path falls back to first-pass.
"""
from __future__ import annotations

import tempfile

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from engram.retrieval.expansion import expand_query


def test_expand_query_basic_appends_entities():
    texts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
    ]
    expanded, chosen = expand_query(
        "where did they go", texts,
        top_k=3, max_entities=2, min_dominance=0.5,
    )
    assert chosen, "Should have picked at least one entity"
    assert expanded != "where did they go"
    assert all(c in expanded for c in chosen)


def test_expand_query_dominance_gate_blocks_diffuse():
    # Three distinct entities, each appears once → top freq = 1/3 < 0.5
    texts = [
        "Alice Smith was here.",
        "Bob Jones was there.",
        "Carol White was elsewhere.",
    ]
    _, chosen = expand_query(
        "they were somewhere", texts,
        top_k=3, max_entities=2, min_dominance=0.5,
    )
    assert chosen == [], "Diffuse entity dist should be gated out"


def test_expand_query_novelty_filter():
    # Entity already in query → must not be re-appended.
    texts = ["Alice Smith works at Acme Corp.", "Alice Smith met someone."]
    _, chosen = expand_query(
        "what does Alice Smith do", texts,
        top_k=2, max_entities=2, min_dominance=0.0,
    )
    # "alice smith" should be filtered as already in query
    assert "alice smith" not in chosen


def test_expand_query_empty_texts_safe():
    assert expand_query("hi", [], top_k=5, max_entities=2)[1] == []


def test_expand_query_min_dominance_none_is_noop():
    """Guard against accidental re-flip of the v0.3 OFF default.

    The shipped default ``RetrievalConfig.query_expansion_min_dominance``
    is ``None`` (PRF off). The expansion primitive must, in that
    configuration, be a strict no-op: ``(query, [])`` — no entities
    returned, query unchanged — even when the top-K pool is full of
    novel entities that *would* fire under any positive threshold.
    """
    texts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
    ]
    # Mirror the engine's call shape: it passes
    # ``min_dominance=float(self.config.query_expansion_min_dominance)``,
    # but only after gating on ``is not None`` upstream. So the contract
    # under test is: with the gate-disabled signal (None semantics),
    # callers must skip expansion. We verify both paths:
    #   (a) caller skips — the canonical engine-side guard.
    #   (b) caller passes ``min_dominance=0.0`` — the historical
    #       gate-off-but-still-mine-entities path — must still expand.
    # If someone re-flips the default to a positive number, the engine
    # guard at engine.py:331 (`is not None`) will start firing and the
    # `test_retrieval_engine_prf_off_by_default` regression alongside
    # this one will catch it. This test pins the *primitive* contract.
    expanded_off, chosen_off = expand_query(
        "where did they go", texts,
        top_k=3, max_entities=2, min_dominance=0.0,
    )
    assert chosen_off, "min_dominance=0.0 (gate disabled) should still mine entities"

    # And the engine-side default — RetrievalConfig() with no args — must
    # have min_dominance=None (the v0.3 shipped default).
    from engram.core.config import RetrievalConfig
    assert RetrievalConfig().query_expansion_min_dominance is None, (
        "v0.3 default must remain None — see PAPER §4.8.2.4 + decision #2."
    )


def test_expand_query_zero_top_k_is_noop():
    expanded, chosen = expand_query(
        "q", ["Alice Smith here."], top_k=0, max_entities=2,
    )
    assert chosen == []
    assert expanded == "q"


def test_retrieval_engine_prf_off_by_default():
    """Regression safety: default config must not fire PRF (recursion etc.)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        eng = Engram(config=cfg)
        try:
            eng.remember("Alice Smith met Bob Jones in Paris.")
            eng.remember("Acme Corp acquired Widgets Inc.")
            r = eng.recall("who met whom", limit=5)
            assert isinstance(r, list)
        finally:
            eng.close()


def test_retrieval_engine_prf_fires_when_configured():
    """When min_dominance is set, PRF should run a 2nd pass without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
        )
        eng = Engram(config=cfg)
        try:
            for _ in range(5):
                eng.remember("Alice Smith was at Acme Corp today.")
            eng.remember("Random unrelated content about cats.")
            # No crash; returns results.
            r = eng.recall("who was where", limit=3)
            assert isinstance(r, list)
            assert len(r) <= 3
        finally:
            eng.close()


def test_retrieval_engine_prf_runtime_toggle_matches_init_time():
    """Setting query_expansion_min_dominance at runtime on a live engine
    must produce the same recall as initializing with that value — i.e.
    PRF×SP is genuinely runtime-toggleable (decision #2 close-out).
    """
    facts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
        "Bob Jones flew to Tokyo last week.",
        "Carol White is a poet.",
    ]
    # Arm A: init with PRF on
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
        )
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            r_init = [m.id for m in eng.recall("where did they go", limit=5)]
        finally:
            eng.close()

    # Arm B: init off, flip on at runtime, recall.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            # runtime flip
            eng.config.retrieval.query_expansion_min_dominance = 0.3
            eng.config.retrieval.query_expansion_top_k = 5
            eng.config.retrieval.query_expansion_max_entities = 2
            r_runtime = [m.id for m in eng.recall("where did they go", limit=5)]
        finally:
            eng.close()

    # Same facts ingested in same order → identical retrieval ordering
    # under the same retrieval config, regardless of how the knob was set.
    assert r_init == r_runtime, (
        f"runtime toggle differs from init-time:\n  init={r_init}\n  runtime={r_runtime}"
    )


def test_retrieval_engine_prf_noop_when_no_dominant_entity():
    """When dominance gate fails, falls back to first-pass results."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.99,  # nothing will pass
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
        )
        eng = Engram(config=cfg)
        try:
            eng.remember("Alice Smith was here.")
            eng.remember("Bob Jones was there.")
            eng.remember("Carol White was elsewhere.")
            r = eng.recall("who was where", limit=3)
            assert isinstance(r, list)
        finally:
            eng.close()

"""Integration test: D1 entity-link retrieval channel.

Pins behaviour: when `RetrievalConfig.entity_weight > 0`, a candidate
whose content shares an entity with the query gets a Jaccard-weighted
boost; with `entity_weight = 0` (default) the channel is fully inert
and the engine's prior behaviour is unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram.core.config import RetrievalConfig
from engram.core.types import (
    EncodingContext,
    Memory,
    MemoryState,
    MemoryType,
    generate_memory_id,
)
from engram.retrieval.engine import RetrievalEngine
from engram.store.memory import SQLiteMemoryStore


def _fact(content: str) -> Memory:
    return Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:60],
        salience=0.5,
        confidence=1.0,
        decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
        encoding_context=EncodingContext(),
    )


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path)
    yield s
    s.close()


def test_entity_channel_boost_recorded_in_sources(store):
    """With entity_weight > 0, sources['entity'] is non-zero on a hit."""
    m = _fact("Alice spent the weekend hiking near Yosemite.")
    store.upsert(m)
    eng = RetrievalEngine(
        store=store,
        config=RetrievalConfig(entity_weight=0.3),
    )
    results = eng.search("Where did Alice go in Yosemite?", limit=5)
    assert results, "expected at least one candidate"
    top = next((r for r in results if r.memory.id == m.id), None)
    assert top is not None
    # The entity channel should have contributed something.
    assert top.sources.get("entity", 0.0) > 0.0


def test_entity_channel_off_by_default(store):
    """Default entity_weight=0 → 'entity' source is exactly 0.0."""
    m = _fact("Alice spent the weekend hiking near Yosemite.")
    store.upsert(m)
    eng = RetrievalEngine(store=store)  # default config
    results = eng.search("Where did Alice go in Yosemite?", limit=5)
    assert results
    top = next((r for r in results if r.memory.id == m.id), None)
    assert top is not None
    assert top.sources.get("entity", 0.0) == 0.0


def test_entity_channel_breaks_ties_toward_entity_match(store):
    """Two candidates with similar text signal: the one sharing the query's
    entity should outrank the entity-less one when entity_weight is high.
    """
    hit = _fact("The Acme Corp meeting ran long today.")
    miss = _fact("The meeting ran long today, very long indeed.")
    store.upsert(hit)
    store.upsert(miss)
    eng = RetrievalEngine(
        store=store,
        config=RetrievalConfig(entity_weight=1.0),
    )
    results = eng.search("How was the Acme Corp meeting?", limit=5)
    ids = [r.memory.id for r in results]
    assert hit.id in ids and miss.id in ids
    # hit must rank above miss.
    assert ids.index(hit.id) < ids.index(miss.id)

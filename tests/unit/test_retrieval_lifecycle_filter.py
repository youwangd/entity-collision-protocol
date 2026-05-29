"""Retrieval-side schema-lifecycle filter.

Pins NEXT.md priority #2: SCHEMA candidates whose lifecycle status is
DEPRECATED in the buffer's event stream must not appear in retrieval
results. Lifecycle event format lives in
`engram.consolidation.lifecycle_projection.make_lifecycle_event`.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram.consolidation.lifecycle_projection import make_lifecycle_event
from engram.consolidation.schema_lifecycle import EventKind
from engram.core.config import RetrievalConfig
from engram.core.types import (
    EncodingContext,
    Memory,
    MemoryState,
    MemoryType,
    generate_memory_id,
)
from engram.retrieval.engine import RetrievalEngine
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore


def _schema_memory(*, content: str, mid: str | None = None) -> Memory:
    return Memory(
        id=mid or generate_memory_id(MemoryType.SCHEMA),
        type=MemoryType.SCHEMA,
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


@pytest.fixture
def buffer(tmp_path):
    return JSONLBufferStore(base_path=tmp_path / "buf")


def test_active_schema_is_returned(store, buffer):
    """Sanity: with no lifecycle events, schemas pass through unfiltered."""
    sm = _schema_memory(content="users tend to prefer postgres pattern")
    store.upsert(sm)
    eng = RetrievalEngine(
        store=store, buffer=buffer,
        config=RetrievalConfig(respect_schema_lifecycle=True),
    )
    results = eng.search("pattern users prefer postgres", limit=5)
    assert any(r.memory.id == sm.id for r in results)


def test_deprecated_schema_is_filtered(store, buffer):
    """A schema with a CREATE then DEPRECATE in the buffer must not surface."""
    sm = _schema_memory(content="users tend to prefer postgres pattern")
    store.upsert(sm)
    buffer.append(make_lifecycle_event(
        schema_id=sm.id, kind=EventKind.CREATE, window_id="w1",
    ))
    buffer.append(make_lifecycle_event(
        schema_id=sm.id, kind=EventKind.DEPRECATE, window_id="w2",
    ))
    eng = RetrievalEngine(
        store=store, buffer=buffer,
        config=RetrievalConfig(respect_schema_lifecycle=True),
    )
    results = eng.search("pattern users prefer postgres", limit=5)
    assert all(r.memory.id != sm.id for r in results)


def test_recovered_schema_resurfaces(store, buffer):
    """DEPRECATE → RECOVER (with fresh window_id) puts the schema back in
    INFERRED state, so it should be retrievable again."""
    sm = _schema_memory(content="users tend to prefer postgres pattern")
    store.upsert(sm)
    for kind, win in [
        (EventKind.CREATE, "w1"),
        (EventKind.DEPRECATE, "w2"),
        (EventKind.RECOVER, "w3"),
    ]:
        buffer.append(make_lifecycle_event(
            schema_id=sm.id, kind=kind, window_id=win,
        ))
    eng = RetrievalEngine(
        store=store, buffer=buffer,
        config=RetrievalConfig(respect_schema_lifecycle=True),
    )
    results = eng.search("pattern users prefer postgres", limit=5)
    assert any(r.memory.id == sm.id for r in results)


def test_filter_can_be_disabled(store, buffer):
    """`respect_schema_lifecycle=False` keeps deprecated schemas visible.
    Useful for audit/debug tooling and for proving the filter is the
    cause of the previous test's filtering, not some unrelated dropout."""
    sm = _schema_memory(content="users tend to prefer postgres pattern")
    store.upsert(sm)
    buffer.append(make_lifecycle_event(
        schema_id=sm.id, kind=EventKind.CREATE, window_id="w1",
    ))
    buffer.append(make_lifecycle_event(
        schema_id=sm.id, kind=EventKind.DEPRECATE, window_id="w2",
    ))
    eng = RetrievalEngine(
        store=store, buffer=buffer,
        config=RetrievalConfig(respect_schema_lifecycle=False),
    )
    results = eng.search("pattern users prefer postgres", limit=5)
    assert any(r.memory.id == sm.id for r in results)


def test_no_buffer_is_noop(store):
    """When no buffer is wired, the filter is a silent no-op."""
    sm = _schema_memory(content="users tend to prefer postgres pattern")
    store.upsert(sm)
    eng = RetrievalEngine(
        store=store,
        config=RetrievalConfig(respect_schema_lifecycle=True),
    )
    results = eng.search("pattern users prefer postgres", limit=5)
    assert any(r.memory.id == sm.id for r in results)


def test_only_deprecated_schemas_filtered_not_facts(store, buffer):
    """A FACT memory whose id collides (hypothetically) with a deprecated
    schema_id must NOT be filtered — the filter is type-scoped to SCHEMA.
    """
    fact = Memory(
        id="m_collide",
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content="postgres is the preferred database",
        summary="postgres preferred",
        salience=0.5, confidence=1.0, decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0, encoding_context=EncodingContext(),
    )
    store.upsert(fact)
    # Lifecycle says "m_collide is a deprecated schema". Wrong type → filter
    # must ignore.
    buffer.append(make_lifecycle_event(
        schema_id="m_collide", kind=EventKind.CREATE, window_id="w1",
    ))
    buffer.append(make_lifecycle_event(
        schema_id="m_collide", kind=EventKind.DEPRECATE, window_id="w2",
    ))
    eng = RetrievalEngine(
        store=store, buffer=buffer,
        config=RetrievalConfig(respect_schema_lifecycle=True),
    )
    results = eng.search("postgres preferred database", limit=5)
    assert any(r.memory.id == "m_collide" for r in results)

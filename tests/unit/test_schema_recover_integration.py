"""End-to-end RECOVER integration through SchemaUpdate (Stage 6).

Until run #25 the lifecycle DAG's recovery edge (DEPRECATED → INFERRED)
was only exercised at the pure-policy layer (`schema_decision.decide`)
and the reducer (`schema_lifecycle._apply`). This module closes the
loop: a re-emitted pattern whose schema is currently DEPRECATED, when
seen in a *fresh* consolidation window with enough supports, must
cause Stage 6 to append a RECOVER lifecycle event referencing the
existing schema_id.

Three cases:
  1. Happy path — DEPRECATED → RECOVER → INFERRED, version preserved,
     no duplicate CREATE.
  2. Same-window (stale) re-emission — supports clear recover but
     window_id matches `last_window_id`; no RECOVER (invariant #5).
  3. Sub-threshold supports in a fresh window — no RECOVER.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engram.consolidation.lifecycle_projection import (
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.pipeline import SchemaUpdate, StageContext
from engram.consolidation.schema_lifecycle import EventKind, SchemaStatus
from engram.core.types import (
    Event,
    EventType,
    Memory,
    MemoryType,
    generate_event_id,
)
from engram.providers.llm import LLMProvider
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore


_PATTERN = (
    "users prefer postgres for transactional workloads with strict "
    "ACID guarantees and pg_partman partitioning"
)
assert len(_PATTERN) >= 80


class _StubLLM(LLMProvider):
    """Returns the canonical pattern with N supporting facts."""

    def __init__(self, supports: int = 3):
        self._payload = {
            "schemas": [
                {"pattern": _PATTERN,
                 "facts": [f"user-{i} uses postgres" for i in range(supports)]}
            ]
        }

    def extract_json(self, prompt: str, *, system: str = "", **_kw):
        return self._payload

    def complete(self, prompt: str, **_kw) -> str:  # pragma: no cover
        return ""


def _seed_facts(store: SQLiteMemoryStore, n: int = 3) -> None:
    for i in range(n):
        ev = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.EVENT_CAPTURE,
            content=f"user-{i} uses postgres",
        )
        store.upsert(Memory.from_event(ev, memory_type=MemoryType.FACT))


def _persist_schema_in_state(
    buf: JSONLBufferStore,
    store: SQLiteMemoryStore,
    *,
    deprecated_window: str,
) -> Memory:
    """Synthesize a DEPRECATED schema with a known last_window_id by
    pre-seeding the buffer with CREATE+DEPRECATE events."""
    schema_event = Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_SCHEMA_UPDATE,
        content=_PATTERN,
    )
    schema_mem = Memory.from_event(schema_event, memory_type=MemoryType.SCHEMA)
    schema_mem.summary = _PATTERN[:80]
    store.upsert(schema_mem)
    buf.append(make_lifecycle_event(
        schema_id=schema_mem.id, kind=EventKind.CREATE,
        window_id=deprecated_window, content=_PATTERN[:200],
    ))
    buf.append(make_lifecycle_event(
        schema_id=schema_mem.id, kind=EventKind.DEPRECATE,
        window_id=deprecated_window, content=_PATTERN[:200],
    ))
    return schema_mem


def test_recover_fires_in_fresh_window_with_sufficient_supports(tmp_path: Path):
    """DEPRECATED schema + fresh window + supports>=recover → RECOVER
    appended; snapshot lands at INFERRED."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)
    schema_mem = _persist_schema_in_state(
        buf, store, deprecated_window="cycle-bad",
    )
    # Sanity: snapshot says DEPRECATED before we run Stage 6.
    pre = snapshot_from_buffer(buf, strict=False)
    assert pre[schema_mem.id].status is SchemaStatus.DEPRECATED

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(supports=3),
        consolidation_id="cycle-fresh",
    )
    out = SchemaUpdate().run(ctx)
    # No new schema memory; we re-used the existing one.
    assert out.stats["schemas_created"] == 0
    assert out.stats["schemas_recovered"] == 1
    assert out.stats["schemas_bumped"] == 0  # same content as stored

    snap = snapshot_from_buffer(buf, strict=False)
    state = snap[schema_mem.id]
    assert state.status is SchemaStatus.INFERRED
    assert state.recover_count == 1
    assert state.last_window_id == "cycle-fresh"
    # Version unchanged across DEPRECATE/RECOVER (only BUMP_VERSION
    # advances version per reducer).
    assert state.version == 1


def test_recover_does_not_fire_in_stale_window(tmp_path: Path):
    """DEPRECATED + supports>=recover BUT window_id == last_window_id
    → reducer rejects RECOVER (invariant #5). Stage 6 still calls
    decide(); decide() returns None for the stale window so no event
    is emitted."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)
    schema_mem = _persist_schema_in_state(
        buf, store, deprecated_window="cycle-stale",
    )

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(supports=3),
        consolidation_id="cycle-stale",  # SAME window_id
    )
    out = SchemaUpdate().run(ctx)
    assert out.stats["schemas_recovered"] == 0
    snap = snapshot_from_buffer(buf, strict=False)
    assert snap[schema_mem.id].status is SchemaStatus.DEPRECATED


def test_recover_does_not_fire_below_threshold(tmp_path: Path):
    """DEPRECATED + fresh window + supports < recover → no RECOVER."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    # Seed 3 facts to clear Stage 6's len(facts)<3 early-return; the
    # LLM still claims only 2 supporting facts for the schema, which
    # is what we want decide() to see.
    _seed_facts(store, n=3)
    schema_mem = _persist_schema_in_state(
        buf, store, deprecated_window="cycle-bad",
    )

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(supports=2),  # < recover=3
        consolidation_id="cycle-fresh",
    )
    out = SchemaUpdate().run(ctx)
    assert out.stats["schemas_recovered"] == 0
    snap = snapshot_from_buffer(buf, strict=False)
    assert snap[schema_mem.id].status is SchemaStatus.DEPRECATED

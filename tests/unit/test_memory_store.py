"""Tests for SQLite memory store."""

import pytest
from datetime import datetime, timezone

from engram.core.types import (
    Memory, MemoryType, MemoryState, Appraisal, SomaticMarker,
    EmotionTag, EncodingContext, DataClassification,
)
from engram.store.memory import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path)
    yield s
    s.close()


def _make_memory(content="test memory", memory_type=MemoryType.FACT, salience=0.5, **kwargs):
    from engram.core.types import generate_memory_id
    return Memory(
        id=kwargs.get("id", generate_memory_id(memory_type)),
        type=memory_type,
        state=kwargs.get("state", MemoryState.ACTIVE),
        content=content,
        summary=kwargs.get("summary", content[:50]),
        salience=salience,
        confidence=1.0,
        decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
    )


class TestUpsert:
    def test_insert(self, store):
        mem = _make_memory("User prefers PostgreSQL")
        store.upsert(mem)
        got = store.get(mem.id)
        assert got is not None
        assert got.content == "User prefers PostgreSQL"

    def test_update(self, store):
        mem = _make_memory("version 1")
        store.upsert(mem)
        mem.content = "version 2"
        mem.salience = 0.9
        store.upsert(mem)
        got = store.get(mem.id)
        assert got.content == "version 2"
        assert got.salience == pytest.approx(0.9)

    def test_preserves_metadata(self, store):
        mem = _make_memory("with metadata")
        mem.appraisal = Appraisal(relevance=1.5, novelty=1.8, goal_conduciveness=1.2)
        mem.somatic = SomaticMarker(valence=0.6, bias="recommend postgres", trigger="database choice")
        mem.emotion = EmotionTag(primary="joy", intensity=0.7, compound="pride")
        mem.encoding_context = EncodingContext(mood_valence=0.3, mood_arousal=0.5, emotions=["curiosity"], task="db-migration")
        mem.classification = DataClassification.INTERNAL
        store.upsert(mem)
        got = store.get(mem.id)
        assert got.appraisal.relevance == pytest.approx(1.5)
        assert got.somatic.bias == "recommend postgres"
        assert got.emotion.primary == "joy"
        assert got.encoding_context.task == "db-migration"
        assert got.classification == DataClassification.INTERNAL


class TestSearch:
    def test_bm25_basic(self, store):
        store.upsert(_make_memory("User prefers PostgreSQL for databases"))
        store.upsert(_make_memory("The weather is sunny today"))
        store.upsert(_make_memory("Database migration completed successfully"))
        results = store.search_text("database", limit=10)
        assert len(results) >= 1
        contents = [r.memory.content for r in results]
        assert any("database" in c.lower() or "postgresql" in c.lower() for c in contents)

    def test_bm25_no_results(self, store):
        store.upsert(_make_memory("hello world"))
        results = store.search_text("xyznonexistent", limit=10)
        assert len(results) == 0

    def test_respects_state_filter(self, store):
        active = _make_memory("active memory", state=MemoryState.ACTIVE)
        suppressed = _make_memory("suppressed memory", state=MemoryState.SUPPRESSED)
        store.upsert(active)
        store.upsert(suppressed)
        results = store.search_text("memory", states=["active"])
        contents = [r.memory.content for r in results]
        assert "active memory" in contents
        assert "suppressed memory" not in contents


class TestStateTransitions:
    def test_update_state(self, store):
        mem = _make_memory("test")
        store.upsert(mem)
        store.update_state(mem.id, MemoryState.SUPPRESSED)
        got = store.get(mem.id)
        assert got.state == MemoryState.SUPPRESSED

    def test_mark_accessed_revives_faded(self, store):
        mem = _make_memory("faded memory", state=MemoryState.FADED)
        store.upsert(mem)
        store.mark_accessed(mem.id)
        got = store.get(mem.id)
        assert got.state == MemoryState.ACTIVE
        assert got.access_count == 1

    def test_mark_accessed_increments_count(self, store):
        mem = _make_memory("test")
        store.upsert(mem)
        store.mark_accessed(mem.id)
        store.mark_accessed(mem.id)
        got = store.get(mem.id)
        assert got.access_count == 2


class TestDelete:
    def test_hard_delete(self, store):
        mem = _make_memory("to delete")
        store.upsert(mem)
        assert store.delete(mem.id)
        assert store.get(mem.id) is None

    def test_delete_nonexistent(self, store):
        assert not store.delete("nonexistent-id")


class TestStats:
    def test_empty(self, store):
        s = store.stats()
        assert s["total_memories"] == 0

    def test_counts(self, store):
        store.upsert(_make_memory("m1", MemoryType.FACT))
        store.upsert(_make_memory("m2", MemoryType.EPISODE))
        store.upsert(_make_memory("m3", MemoryType.FACT, state=MemoryState.SUPPRESSED))
        s = store.stats()
        assert s["total_memories"] == 3
        assert s["by_state"]["active"] == 2
        assert s["by_state"]["suppressed"] == 1
        assert s["by_type"]["fact"] == 1  # only active ones
        assert s["by_type"]["episode"] == 1


class TestPins:
    def test_add_and_get(self, store):
        store.add_pin("pin-1", "important fact")
        pins = store.get_pins()
        assert len(pins) == 1
        assert pins[0]["content"] == "important fact"

    def test_remove(self, store):
        store.add_pin("pin-1", "temp")
        assert store.remove_pin("pin-1")
        assert len(store.get_pins()) == 0

    def test_remove_nonexistent(self, store):
        assert not store.remove_pin("nonexistent")


class TestRebuild:
    def test_drop_all(self, store):
        store.upsert(_make_memory("m1"))
        store.upsert(_make_memory("m2"))
        store.add_pin("pin-1", "pin")
        store.drop_all()
        assert store.stats()["total_memories"] == 0
        assert len(store.get_pins()) == 0

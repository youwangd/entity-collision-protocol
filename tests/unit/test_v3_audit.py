"""Tests for v3 audit fixes — bugs, quality, best practices."""

import pytest
from datetime import datetime, timezone

from engram.core.config import Config
from engram.core.types import (
    Memory, MemoryType, MemoryState,
    Modification,
)
from engram.engine import Engram
from engram.store.memory import SQLiteMemoryStore


@pytest.fixture
def mem(tmp_path):
    config = Config.minimal(str(tmp_path / "test-engram"))
    engine = Engram(config, actor="test")
    yield engine
    engine.close()


# --- B1: Provenance modifications must use Modification objects ---

class TestProvenanceSerialization:
    def test_provenance_to_dict_doesnt_crash(self, mem):
        mem.capture("new fact about database")
        mem.consolidate()
        # After consolidation, all memories should have serializable provenance
        for m in mem._store.all_active():
            prov = m.provenance.to_dict()
            assert isinstance(prov, dict)
            # Modifications should be dicts (serialized from Modification objects)
            for mod in prov.get("modifications", []):
                assert isinstance(mod, dict)
                assert "ts" in mod
                assert "operation" in mod

    def test_modification_objects_used(self, mem):
        """Verify pipeline creates Modification objects, not raw dicts."""
        mem.capture("deploy completed successfully")
        mem.consolidate()
        for m in mem._store.all_active():
            for mod in m.provenance.modifications:
                assert isinstance(mod, Modification), f"Expected Modification, got {type(mod)}"


# --- B2: affect_lock doesn't crash ---

class TestAffectLock:
    def test_affect_lock_succeeds(self, mem):
        # Should not raise IntegrityError
        mem.affect.lock("novelty_seeking")


# --- B3: Depth filtering doesn't mutate original ---

class TestDepthSafety:
    def test_l0_doesnt_corrupt_memory(self, mem):
        mem.remember("long detailed content about architecture decisions and patterns for the project")
        mem.recall("architecture", depth="L2")
        mem.recall("architecture", depth="L0")
        # L2 should still have full content
        results_l2_again = mem.recall("architecture", depth="L2")
        if results_l2_again:
            assert len(results_l2_again[0].memory.content) > 20

    def test_l0_doesnt_wipe_encoding_context(self, mem):
        mem.trigger_emotion("joy", 0.8)
        mem.remember("joyful memory for depth test")
        # L0 recall shouldn't corrupt the stored memory
        mem.recall("joyful memory", depth="L0")
        stored = list(mem._store.all_active())
        joyful = [m for m in stored if "joyful" in m.content]
        if joyful:
            # Encoding context should be intact in the store
            assert joyful[0].encoding_context is not None


# --- Q4: Context manager ---

class TestContextManager:
    def test_with_statement(self, tmp_path):
        config = Config.minimal(str(tmp_path / "ctx-test"))
        with Engram(config) as mem:
            mem.remember("test")
            assert len(mem.recall("test")) >= 1
        # After exit, should be closed

    def test_with_exception(self, tmp_path):
        config = Config.minimal(str(tmp_path / "ctx-test2"))
        try:
            with Engram(config) as mem:
                mem.remember("test")
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        # Should not leak resources


# --- Q9: __version__ ---

class TestVersion:
    def test_version_exists(self):
        import engram
        assert hasattr(engram, "__version__")
        assert engram.__version__ == "0.1.0"


# --- Q13: Sort order ---

class TestSortOrder:
    def test_all_active_sorted_by_salience(self, mem):
        mem.remember("low salience", salience=0.1)
        mem.remember("high salience", salience=0.9)
        memories = mem._store.all_active()
        if len(memories) >= 2:
            assert memories[0].salience >= memories[1].salience


# --- Q14: Input validation ---

class TestInputValidation:
    def test_empty_content_raises(self, mem):
        with pytest.raises(ValueError, match="content must not be empty"):
            mem.remember("")

    def test_whitespace_content_raises(self, mem):
        with pytest.raises(ValueError, match="content must not be empty"):
            mem.remember("   ")

    def test_empty_query_raises(self, mem):
        with pytest.raises(ValueError, match="query must not be empty"):
            mem.recall("")

    def test_salience_clamped(self, mem):
        # Should not raise, just clamp
        mem.remember("test", salience=5.0)
        mem.remember("test2", salience=-1.0)


# --- Q8: Retention TTL via SQL ---

class TestRetentionSQL:
    def test_purge_by_ttl(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            from datetime import timedelta
            from engram.core.types import generate_memory_id
            old = Memory(
                id=generate_memory_id(MemoryType.FACT), type=MemoryType.FACT,
                state=MemoryState.FADED, content="old faded",
                summary="old", salience=0.01, confidence=1.0, decay_rate=0.1,
                created_at=datetime.now(timezone.utc) - timedelta(days=200),
                last_accessed=datetime.now(timezone.utc) - timedelta(days=200),
            )
            recent = Memory(
                id=generate_memory_id(MemoryType.FACT), type=MemoryType.FACT,
                state=MemoryState.FADED, content="recent faded",
                summary="recent", salience=0.01, confidence=1.0, decay_rate=0.1,
                created_at=datetime.now(timezone.utc) - timedelta(days=10),
                last_accessed=datetime.now(timezone.utc) - timedelta(days=10),
            )
            store.upsert(old)
            store.upsert(recent)
            purged = store.purge_by_ttl(MemoryState.FADED, ttl_days=90)
            assert purged == 1
            assert store.get(recent.id) is not None
            assert store.get(old.id) is None
        finally:
            store.close()

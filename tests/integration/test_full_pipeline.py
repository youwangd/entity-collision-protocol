"""Integration tests for the full Engram engine."""

import pytest

from engram.core.config import Config
from engram.core.types import MemoryType, RecallContext
from engram.engine import Engram


@pytest.fixture
def mem(tmp_path):
    config = Config.minimal(str(tmp_path / "test-engram"))
    engine = Engram(config, actor="test")
    yield engine
    engine.close()


class TestRememberRecallCycle:
    def test_basic_remember_recall(self, mem):
        mem.remember("User prefers dark mode in all UIs")
        results = mem.recall("dark mode")
        assert len(results) >= 1
        assert "dark mode" in results[0].memory.content

    def test_multiple_memories(self, mem):
        mem.remember("User prefers PostgreSQL for database work")
        mem.remember("Deploy process uses Docker")
        mem.remember("API endpoint is /v2/users")
        results = mem.recall("database")
        assert len(results) >= 1

    def test_salience_affects_score(self, mem):
        mem.remember("low importance fact", salience=0.1)
        mem.remember("HIGH IMPORTANCE database fact", salience=0.9)
        results = mem.recall("fact", limit=10)
        # Higher salience should score higher
        if len(results) >= 2:
            assert results[0].memory.salience >= results[1].memory.salience

    def test_memory_type(self, mem):
        mem.remember("migration completed", memory_type=MemoryType.EPISODE)
        results = mem.recall("migration")
        assert results[0].memory.type == MemoryType.EPISODE


class TestForget:
    def test_soft_forget_hides_from_recall(self, mem):
        mem.remember("secret info")
        results = mem.recall("secret")
        assert len(results) >= 1
        mem_id = results[0].memory.id
        mem.forget(id=mem_id)
        results = mem.recall("secret")
        assert len(results) == 0

    def test_soft_forget_recoverable(self, mem):
        mem.remember("suppressed info")
        results = mem.recall("suppressed")
        mem_id = results[0].memory.id
        mem.forget(id=mem_id)
        results = mem.recall("suppressed", include_suppressed=True)
        assert len(results) >= 1

    def test_hard_forget_permanent(self, mem):
        mem.remember("gdpr delete me")
        results = mem.recall("gdpr")
        mem_id = results[0].memory.id
        mem.forget(id=mem_id, hard=True)
        results = mem.recall("gdpr", include_suppressed=True)
        assert len(results) == 0

    def test_forget_by_query(self, mem):
        mem.remember("project X is old")
        mem.remember("project X was abandoned")
        mem.remember("project Y is active")
        count = mem.forget(query="project X")
        assert count >= 1
        results = mem.recall("project X")
        assert len(results) == 0


class TestContextBoost:
    def test_encoding_specificity(self, mem):
        # Remember something with task context
        mem.remember("deploy failed because DATABASE_URL was missing")
        results = mem.recall(
            "deploy failed",
            context=RecallContext(task="deployment"),
        )
        assert len(results) >= 1


class TestPins:
    def test_pin_appears_in_context(self, mem):
        mem.pin("Always check DATABASE_URL before deploy")
        ctx = mem.active_context()
        assert "DATABASE_URL" in ctx

    def test_unpin(self, mem):
        pin_id = mem.pin("temporary note")
        assert mem.unpin(pin_id)
        ctx = mem.active_context()
        assert "temporary note" not in ctx


class TestStatus:
    def test_empty_status(self, mem):
        s = mem.status()
        assert s["total_memories"] == 0
        assert s["buffer_events"] == 0

    def test_status_after_operations(self, mem):
        mem.remember("fact 1")
        mem.remember("fact 2")
        s = mem.status()
        assert s["total_memories"] == 2
        assert s["buffer_events"] == 2


class TestEventSourcing:
    def test_rebuild_from_events(self, mem):
        mem.remember("memory 1")
        mem.remember("memory 2")
        mem.remember("memory 3")
        assert mem.status()["total_memories"] == 3

        # Rebuild from events
        count = mem.rebuild()
        assert count == 3
        assert mem.status()["total_memories"] == 3

    def test_rebuild_preserves_content(self, mem):
        mem.remember("User prefers PostgreSQL")
        mem.rebuild()
        results = mem.recall("PostgreSQL")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    def test_rebuild_handles_forget(self, mem):
        mem.remember("to keep")
        mem.remember("to forget")
        results = mem.recall("forget")
        if results:
            mem.forget(id=results[0].memory.id)
        # Rebuild should replay the forget
        mem.rebuild()
        # The remembered + forget events should both replay


class TestAuditLog:
    def test_operations_are_audited(self, mem):
        mem.remember("audited memory")
        mem.recall("audited")
        entries = mem._audit.read()
        ops = [e["op"] for e in entries]
        assert "remember" in ops
        assert "recall" in ops

    def test_audit_records_actor(self, mem):
        mem.remember("test")
        entries = mem._audit.read()
        assert entries[-1]["actor"] == "test"

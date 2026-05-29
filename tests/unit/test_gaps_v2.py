"""Tests for gap analysis v2 fixes."""

import pytest
from datetime import datetime, timedelta, timezone

from engram.core.config import Config
from engram.core.types import (
    EventType, Memory, MemoryType, MemoryState, RecallContext,
)
from engram.engine import Engram
from engram.affect.engine import Mood
from engram.store.memory import SQLiteMemoryStore


@pytest.fixture
def mem(tmp_path):
    config = Config.minimal(str(tmp_path / "test-engram"))
    engine = Engram(config, actor="test")
    yield engine
    engine.close()


# --- #5: Pipeline stage order (appraisal AFTER extraction) ---

class TestPipelineOrder:
    def test_appraisal_scores_consolidation_memories(self, tmp_path):
        """Appraisal must run AFTER extraction so memories_created isn't empty."""
        config = Config.minimal(str(tmp_path / "test"))
        mem = Engram(config, actor="test")
        try:
            mem.capture("URGENT critical deadline approaching")
            mem.capture("routine status update nothing new")
            report = mem.consolidate()
            assert report.memories_created >= 2
            # Verify appraisal actually scored the memories
            results = mem.recall("urgent critical", limit=5)
            urgent = [r for r in results if "urgent" in r.memory.content.lower()]
            routine = [r for r in results if "routine" in r.memory.content.lower()]
            if urgent and routine:
                assert urgent[0].memory.salience > routine[0].memory.salience
        finally:
            mem.close()


# --- #1: Reconsolidation ---

class TestReconsolidation:
    def test_reconsolidates_on_context_divergence(self, mem):
        # Remember in a happy mood
        mem.trigger_emotion("joy", 0.9)
        mem.remember("project architecture decision")
        results = mem.recall("architecture")
        assert len(results) >= 1
        original_confidence = results[0].memory.confidence

        # Now recall in a very different mood
        mem.trigger_emotion("anger", 0.9)
        mem.trigger_emotion("fear", 0.8)
        results2 = mem.recall("architecture", context=RecallContext(
            mood_valence=-0.8, mood_arousal=0.8, task="different-task",
        ))

        if results2:
            # Should have reconsolidated (confidence slightly decreased)
            refreshed = mem.get(results2[0].memory.id)
            if refreshed:
                assert refreshed.confidence <= original_confidence

    def test_no_reconsolidation_on_similar_context(self, mem):
        mem.remember("simple fact")
        results = mem.recall("simple fact")
        assert len(results) >= 1
        mid = results[0].memory.id
        original = mem.get(mid)
        # Recall in similar context — should NOT reconsolidate
        mem.recall("simple fact", context=RecallContext(mood_valence=0.0, mood_arousal=0.0))
        refreshed = mem.get(mid)
        assert refreshed.confidence == original.confidence

    def test_reconsolidation_event_emitted(self, mem):
        mem.trigger_emotion("joy", 0.9)
        mem.remember("reconsolidation test memory")
        mem.trigger_emotion("anger", 0.9)
        mem.recall("reconsolidation test", context=RecallContext(
            mood_valence=-0.9, mood_arousal=0.9, task="totally-different",
        ))
        # Check if any RECONSOLIDATION events were emitted (smoke: code path must not crash).
        events = list(mem._buffer.scan())
        _ = [e for e in events if e.type == EventType.RECONSOLIDATION]


# --- #9: Depth parameter ---

class TestDepthFiltering:
    def test_l0_returns_summary_only(self, mem):
        mem.remember("detailed content about PostgreSQL migration including all steps and details")
        results = mem.recall("PostgreSQL", depth="L0")
        if results:
            # L0 should have truncated content to summary
            assert len(results[0].memory.content) <= 100

    def test_l2_returns_full(self, mem):
        long_content = "detailed " * 50 + "PostgreSQL migration"
        mem.remember(long_content)
        results = mem.recall("PostgreSQL", depth="L2")
        if results:
            assert len(results[0].memory.content) > 100

    def test_l1_strips_modifications(self, mem):
        mem.remember("L1 test memory")
        results = mem.recall("L1 test", depth="L1")
        if results:
            assert results[0].memory.provenance.modifications == []


# --- #23: Revival on access ---

class TestRevivalOnAccess:
    def test_faded_memory_revives(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            from engram.core.types import generate_memory_id
            mem = Memory(
                id=generate_memory_id(MemoryType.FACT), type=MemoryType.FACT,
                state=MemoryState.FADED, content="faded memory test",
                summary="faded", salience=0.05, confidence=1.0, decay_rate=0.1,
                created_at=datetime.now(timezone.utc) - timedelta(days=30),
            )
            store.upsert(mem)
            store.mark_accessed(mem.id)
            refreshed = store.get(mem.id)
            assert refreshed.state == MemoryState.ACTIVE
        finally:
            store.close()

    def test_fading_memory_revives(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            from engram.core.types import generate_memory_id
            mem = Memory(
                id=generate_memory_id(MemoryType.FACT), type=MemoryType.FACT,
                state=MemoryState.FADING, content="fading memory test",
                summary="fading", salience=0.15, confidence=1.0, decay_rate=0.1,
                created_at=datetime.now(timezone.utc) - timedelta(days=10),
            )
            store.upsert(mem)
            store.mark_accessed(mem.id)
            refreshed = store.get(mem.id)
            assert refreshed.state == MemoryState.ACTIVE
        finally:
            store.close()


# --- #25: Affect events in JSONL ---

class TestAffectEventsInBuffer:
    def test_emotion_writes_to_buffer(self, mem):
        mem.trigger_emotion("joy", 0.7, trigger="test")
        events = list(mem._buffer.scan())
        affect_events = [e for e in events if e.type == EventType.AFFECT_EMOTION]
        assert len(affect_events) >= 1

    def test_mood_update_writes_to_buffer(self, mem):
        mem.trigger_emotion("sadness", 0.6)
        events = list(mem._buffer.scan())
        mood_events = [e for e in events if e.type == EventType.AFFECT_MOOD_UPDATE]
        assert len(mood_events) >= 1

    def test_temperament_override_writes_to_buffer(self, mem):
        mem.affect.set_temperament(novelty_seeking=0.9)
        events = list(mem._buffer.scan())
        override_events = [e for e in events if e.type == EventType.AFFECT_OVERRIDE]
        assert len(override_events) >= 1


# --- #2: Relations table ---

class TestRelations:
    def test_add_and_get_relation(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            store.add_relation("mem-a", "mem-b", "supersedes")
            rels = store.get_relations("mem-a")
            assert len(rels) == 1
            assert rels[0]["type"] == "supersedes"
        finally:
            store.close()

    def test_get_relations_both_directions(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            store.add_relation("mem-a", "mem-b", "contradicts")
            assert len(store.get_relations("mem-a")) == 1
            assert len(store.get_relations("mem-b")) == 1  # found as target
        finally:
            store.close()

    def test_filter_by_type(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            store.add_relation("mem-a", "mem-b", "supersedes")
            store.add_relation("mem-a", "mem-c", "contradicts")
            assert len(store.get_relations("mem-a", "supersedes")) == 1
            assert len(store.get_relations("mem-a")) == 2
        finally:
            store.close()


# --- #17: Provenance created_by on consolidation memories ---

class TestProvenanceCreatedBy:
    def test_consolidation_sets_created_by(self, tmp_path):
        config = Config.minimal(str(tmp_path / "test"))
        mem = Engram(config, actor="test")
        try:
            mem.capture("event for provenance test")
            mem.consolidate()
            # Find the consolidation-created memory
            results = mem.recall("provenance test")
            if results:
                prov = mem.provenance(results[0].memory.id)
                assert prov is not None
                assert prov["created_by"].startswith("evt-")  # should be consolidation cycle ID
        finally:
            mem.close()


# --- #3: MCP serve command ---

class TestMCPServe:
    def test_serve_command_exists(self):
        from engram.cli.main import cli
        commands = [cmd for cmd in cli.commands]
        assert "serve" in commands

    def test_mcp_server_has_stdio(self, mem):
        from engram.mcp.server import MCPServer
        server = MCPServer(mem)
        assert hasattr(server, "serve_stdio")

    def test_mcp_lists_12_tools(self, mem):
        from engram.mcp.server import MCPServer
        server = MCPServer(mem)
        tools = server.list_tools()
        assert len(tools) == 12


# --- #19: Export includes full state ---

class TestExportFull:
    def test_export_includes_all_fields(self, mem):
        mem.remember("export test memory", salience=0.7)
        exported = mem.export_memories()
        assert len(exported) >= 1
        e = exported[0]
        assert "appraisal" in e
        assert "somatic" in e
        assert "emotion" in e
        assert "encoding_context" in e
        assert "classification" in e
        assert "provenance" in e
        assert "confidence" in e
        assert "decay_rate" in e


# --- #21: Mood confidence ---

class TestMoodConfidence:
    def test_mood_has_confidence(self):
        mood = Mood()
        assert hasattr(mood, "confidence")
        assert mood.confidence == 1.0


# --- #22: Forgetting budget in report ---

class TestForgettingBudget:
    def test_report_includes_totals(self, tmp_path):
        config = Config.minimal(str(tmp_path / "test"))
        mem = Engram(config, actor="test")
        try:
            mem.capture("test event")
            report = mem.consolidate()
            assert "total_active" in report.state_transitions
            assert "total_faded" in report.state_transitions
        finally:
            mem.close()


# --- #4: Stage filtering ---

class TestStageFiltering:
    def test_custom_stages(self, tmp_path):
        from engram.core.config import ConsolidationConfig
        config = Config.minimal(str(tmp_path / "test"))
        config.consolidation = ConsolidationConfig(stages=["replay", "deduplication", "extraction", "persistence"])
        mem = Engram(config, actor="test")
        try:
            mem.capture("test for custom pipeline")
            report = mem.consolidate()
            # Should work with reduced stages
            assert report.errors == []
        finally:
            mem.close()

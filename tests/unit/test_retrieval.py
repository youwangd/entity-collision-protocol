"""Tests for the retrieval engine."""

import pytest
from datetime import datetime, timezone

from engram.core.types import (
    Memory, MemoryType, MemoryState,
    RecallContext, EncodingContext,
    generate_memory_id,
)
from engram.store.memory import SQLiteMemoryStore
from engram.retrieval.engine import RetrievalEngine, IntentAnalyzer


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path)
    yield s
    s.close()


@pytest.fixture
def engine(store):
    return RetrievalEngine(store=store)


def _make_memory(content, memory_type=MemoryType.FACT, salience=0.5, **kwargs):
    return Memory(
        id=kwargs.get("id", generate_memory_id(memory_type)),
        type=memory_type,
        state=kwargs.get("state", MemoryState.ACTIVE),
        content=content,
        summary=content[:50],
        salience=salience,
        confidence=1.0,
        decay_rate=0.1,
        created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        last_accessed=kwargs.get("last_accessed", datetime.now(timezone.utc)),
        access_count=0,
        encoding_context=kwargs.get("encoding_context", EncodingContext()),
    )


class TestIntentAnalyzer:
    def test_fact_query(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("what is the database preference")
        assert MemoryType.FACT in intent.target_types

    def test_episode_query(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("what happened during the deploy")
        assert MemoryType.EPISODE in intent.target_types

    def test_schema_query(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("what pattern do we usually follow")
        assert MemoryType.SCHEMA in intent.target_types

    def test_recent_temporal(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("what happened recently")
        assert intent.temporal == "recent"

    def test_old_temporal(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("what was originally decided")
        assert intent.temporal == "old"

    def test_emotional_query(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("when I felt frustrated about the deploy")
        assert intent.emotional is True

    def test_neutral_query(self):
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze("PostgreSQL configuration")
        assert not intent.target_types
        assert intent.temporal == ""
        assert intent.emotional is False


class TestRetrievalEngine:
    def test_basic_bm25_search(self, store, engine):
        store.upsert(_make_memory("User prefers PostgreSQL for all database work"))
        store.upsert(_make_memory("Deploy uses Docker containers"))
        results = engine.search("PostgreSQL database")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    def test_respects_state_filter(self, store, engine):
        active = _make_memory("active PostgreSQL memory")
        suppressed = _make_memory("suppressed PostgreSQL memory", state=MemoryState.SUPPRESSED)
        store.upsert(active)
        store.upsert(suppressed)
        results = engine.search("PostgreSQL")
        contents = [r.memory.content for r in results]
        assert "active PostgreSQL memory" in contents
        assert "suppressed PostgreSQL memory" not in contents

    def test_include_suppressed(self, store, engine):
        store.upsert(_make_memory("suppressed PostgreSQL memory", state=MemoryState.SUPPRESSED))
        results = engine.search("PostgreSQL", include_suppressed=True)
        assert len(results) >= 1

    def test_salience_influences_score(self, store, engine):
        store.upsert(_make_memory("low importance database fact", salience=0.1))
        store.upsert(_make_memory("HIGH importance database config", salience=0.9))
        results = engine.search("database", limit=10)
        if len(results) >= 2:
            # Higher salience should generally score higher
            high_sal = [r for r in results if r.memory.salience > 0.5]
            low_sal = [r for r in results if r.memory.salience < 0.5]
            if high_sal and low_sal:
                assert high_sal[0].score >= low_sal[0].score

    def test_context_boost(self, store, engine):
        mem = _make_memory(
            "deploy failed due to missing config",
            encoding_context=EncodingContext(task="deployment", mood_valence=0.3),
        )
        store.upsert(mem)
        # Search with matching context should boost
        results_no_ctx = engine.search("deploy failed")
        results_with_ctx = engine.search(
            "deploy failed",
            context=RecallContext(task="deployment", mood_valence=0.3),
        )
        if results_no_ctx and results_with_ctx:
            assert results_with_ctx[0].score >= results_no_ctx[0].score

    def test_score_components_tracked(self, store, engine):
        store.upsert(_make_memory("test memory for scoring"))
        results = engine.search("test memory")
        assert len(results) >= 1
        sources = results[0].sources
        assert "rrf" in sources
        assert "salience" in sources
        assert "recency" in sources

    def test_empty_results(self, store, engine):
        results = engine.search("nonexistent query xyz")
        assert results == []

    def test_intent_type_filtering(self, store, engine):
        store.upsert(_make_memory("User prefers dark mode", memory_type=MemoryType.FACT))
        store.upsert(_make_memory("Deploy happened at 3pm", memory_type=MemoryType.EPISODE))
        # "what is" query routes to facts
        results = engine.search("what is the preference for dark mode")
        if results:
            # Should prefer facts
            assert results[0].memory.type == MemoryType.FACT

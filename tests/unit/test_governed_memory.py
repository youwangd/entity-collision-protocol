"""Tests for Governed Memory features: dedup, confidence, merge, properties."""
import pytest
from datetime import datetime, timezone

from engram.core import Memory, MemoryType, MemoryState
from engram.store.memory import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh memory store."""
    s = SQLiteMemoryStore(tmp_path)
    yield s
    s.close()


def _make_memory(id="m1", content="test fact", salience=0.5, confidence=1.0, **kwargs):
    """Helper to create a Memory with defaults."""
    return Memory(
        id=id,
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary="",
        salience=salience,
        confidence=confidence,
        decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
        agent_id="test",
        **kwargs,
    )


class TestWriteSideDedup:
    """Test write-side cosine dedup threshold."""

    def test_upsert_returns_true_by_default(self, store):
        """Normal upsert (no dedup) returns True."""
        m = _make_memory()
        result = store.upsert(m)
        assert result is True

    def test_upsert_no_dedup_when_threshold_zero(self, store):
        """Dedup disabled when threshold is 0."""
        m1 = _make_memory(id="m1", content="the cat sat on the mat")
        m2 = _make_memory(id="m2", content="the cat sat on the mat")  # exact dupe
        assert store.upsert(m1, dedup_threshold=0.0) is True
        assert store.upsert(m2, dedup_threshold=0.0) is True  # no dedup, both stored
        assert store.get("m1") is not None
        assert store.get("m2") is not None

    def test_upsert_dedup_without_vector_store(self, store):
        """Dedup gracefully skipped when no vector store provided."""
        m = _make_memory()
        result = store.upsert(m, dedup_threshold=0.92, vector_store=None, embedding_provider=None)
        assert result is True  # stored anyway — no vector store to check


class TestConfidenceInRetrieval:
    """Test that confidence factors into memory retrieval scoring."""

    def test_memory_stores_confidence(self, store):
        """Confidence field persists through store/retrieve."""
        m = _make_memory(confidence=0.75)
        store.upsert(m)
        retrieved = store.get("m1")
        assert retrieved is not None
        assert retrieved.confidence == 0.75

    def test_low_confidence_stored(self, store):
        """Very low confidence memories are stored."""
        m = _make_memory(confidence=0.1)
        store.upsert(m)
        retrieved = store.get("m1")
        assert retrieved.confidence == 0.1

    def test_default_confidence_is_one(self, store):
        """Default confidence should be 1.0."""
        m = _make_memory()
        store.upsert(m)
        retrieved = store.get("m1")
        assert retrieved.confidence == 1.0


class TestMemoryProperties:
    """Test typed property storage (dual extraction)."""

    def test_upsert_properties(self, store):
        """Can store typed properties for a memory."""
        m = _make_memory()
        store.upsert(m)
        
        props = [
            {"key": "topic", "value": "hiring", "type": "text", "confidence": 0.9},
            {"key": "sentiment", "value": "positive", "type": "text", "confidence": 0.8},
            {"key": "deal_value", "value": "450000", "type": "number", "confidence": 0.95},
        ]
        store.upsert_properties("m1", props)
        
        retrieved = store.get_properties("m1")
        assert len(retrieved) == 3
        keys = {p["key"] for p in retrieved}
        assert keys == {"topic", "sentiment", "deal_value"}

    def test_property_confidence(self, store):
        """Property confidence is stored and retrieved."""
        m = _make_memory()
        store.upsert(m)
        store.upsert_properties("m1", [{"key": "topic", "value": "AI", "confidence": 0.7}])
        
        props = store.get_properties("m1")
        assert len(props) == 1
        assert props[0]["confidence"] == 0.7

    def test_property_upsert_overwrites(self, store):
        """Upserting same key updates the value."""
        m = _make_memory()
        store.upsert(m)
        store.upsert_properties("m1", [{"key": "status", "value": "open"}])
        store.upsert_properties("m1", [{"key": "status", "value": "closed", "confidence": 0.95}])
        
        props = store.get_properties("m1")
        assert len(props) == 1
        assert props[0]["value"] == "closed"
        assert props[0]["confidence"] == 0.95

    def test_search_by_property(self, store):
        """Can find memories by property key+value."""
        m1 = _make_memory(id="m1", content="first meeting notes")
        m2 = _make_memory(id="m2", content="second meeting notes")
        m3 = _make_memory(id="m3", content="unrelated")
        store.upsert(m1)
        store.upsert(m2)
        store.upsert(m3)
        
        store.upsert_properties("m1", [{"key": "topic", "value": "hiring"}])
        store.upsert_properties("m2", [{"key": "topic", "value": "hiring"}])
        store.upsert_properties("m3", [{"key": "topic", "value": "budget"}])
        
        results = store.search_by_property("topic", "hiring")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"m1", "m2"}

    def test_search_by_property_key_only(self, store):
        """Can find memories by property key (any value)."""
        m1 = _make_memory(id="m1", content="first")
        m2 = _make_memory(id="m2", content="second")
        store.upsert(m1)
        store.upsert(m2)
        
        store.upsert_properties("m1", [{"key": "priority", "value": "high"}])
        store.upsert_properties("m2", [{"key": "priority", "value": "low"}])
        
        results = store.search_by_property("priority")
        assert len(results) == 2

    def test_empty_properties(self, store):
        """Memory with no properties returns empty list."""
        m = _make_memory()
        store.upsert(m)
        props = store.get_properties("m1")
        assert props == []

    def test_properties_deleted_with_memory(self, store):
        """Properties should be cascade-deleted when memory is deleted."""
        m = _make_memory()
        store.upsert(m)
        store.upsert_properties("m1", [{"key": "x", "value": "y"}])
        assert len(store.get_properties("m1")) == 1
        
        store.delete("m1")
        assert store.get_properties("m1") == []


class TestMechanicalMerge:
    """Test the mechanical merge consolidation stage."""

    def test_merge_stage_exists(self):
        """MechanicalMerge stage is importable."""
        from engram.consolidation.pipeline import MechanicalMerge
        stage = MechanicalMerge()
        assert stage.name == "mechanical_merge"
        assert stage.threshold == 0.95
        assert stage.DEFAULT_MERGE_THRESHOLD == 0.95

    def test_merge_threshold_from_config(self):
        """MechanicalMerge picks up threshold from StorageConfig."""
        from engram.consolidation.pipeline import ConsolidationPipeline, MechanicalMerge
        from engram.core.config import Config, StorageConfig
        from unittest.mock import MagicMock
        config = Config(storage=StorageConfig(merge_threshold=0.88))
        pipeline = ConsolidationPipeline(
            config=config, store=None,
            buffer=MagicMock(), audit=MagicMock(),
        )
        merge_stages = [s for s in pipeline.stages if isinstance(s, MechanicalMerge)]
        assert len(merge_stages) == 1
        assert merge_stages[0].threshold == 0.88

    def test_merge_threshold_explicit_override(self):
        """Explicit threshold arg overrides default."""
        from engram.consolidation.pipeline import MechanicalMerge
        stage = MechanicalMerge(threshold=0.80)
        assert stage.threshold == 0.80

    def test_merge_skips_without_vector_store(self, store):
        """Merge gracefully skips when no vector store."""
        from engram.consolidation.pipeline import MechanicalMerge, StageContext
        stage = MechanicalMerge(vector_store=None)
        ctx = StageContext(store=store)
        result = stage.run(ctx)
        assert result.stats.get("mechanical_merged", 0) == 0

    def test_pipeline_has_15_stages(self):
        """Pipeline now has 15 stages (14 original + mechanical merge)."""
        from engram.consolidation.pipeline import ConsolidationPipeline
        from engram.core.config import Config
        from unittest.mock import MagicMock
        config = Config()
        pipeline = ConsolidationPipeline(
            config=config, store=None,
            buffer=MagicMock(), audit=MagicMock(),
        )
        assert len(pipeline.stages) == 15

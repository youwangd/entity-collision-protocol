"""Tests for the Governed Memory v0.2 features:

- write-side cosine dedup (Storage.write_dedup_threshold)
- per-memory extraction_confidence field + persistence + roundtrip
- StorageConfig YAML loading

Reference: arXiv:2603.17787 ("Governed Memory")
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engram import Engram, Config
from engram.core.config import StorageConfig, RetrievalConfig
from engram.core.types import Memory, MemoryType, MemoryState, generate_memory_id
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore
from engram.providers.embeddings import EmbeddingProvider


class _DeterministicEmbedder(EmbeddingProvider):
    """Trivial deterministic embedder for tests.

    Maps content → 16-d unit vector via hash. Identical content → identical vector.
    Near-identical content (single char diff) → very similar vector.
    """

    def __init__(self):
        self._dim = 16

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        # Deterministic per-character bag-of-chars feature
        v = [0.0] * self._dim
        for ch in text.lower():
            v[ord(ch) % self._dim] += 1.0
        # L2 normalise
        norm = sum(x * x for x in v) ** 0.5
        if norm == 0:
            return v
        return [x / norm for x in v]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def tmpdir_engine(tmp_path: Path):
    """Engine with a deterministic embedder + sqlite-vec store, NO dedup."""
    cfg = Config(path=str(tmp_path / "engram"))
    eng = Engram(config=cfg)
    # Inject deterministic embedder so vector store actually has neighbours to dedup against
    eng._embeddings = _DeterministicEmbedder()
    yield eng
    eng.close()


# --- extraction_confidence persistence ---


def test_extraction_confidence_default_is_one():
    m = Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content="x",
        summary="x",
        salience=0.5,
        confidence=1.0,
        decay_rate=0.005,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    assert m.extraction_confidence == 1.0


def test_extraction_confidence_roundtrips_through_to_from_dict():
    from datetime import datetime, timezone
    m = Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content="user prefers postgres",
        summary="user prefers postgres",
        salience=0.7,
        confidence=0.9,
        decay_rate=0.005,
        created_at=datetime.now(timezone.utc),
        extraction_confidence=0.42,
    )
    d = m.to_dict()
    assert d["extraction_confidence"] == 0.42
    m2 = Memory.from_dict(d)
    assert m2.extraction_confidence == 0.42


def test_extraction_confidence_persists_in_sqlite(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path)
    from datetime import datetime, timezone
    m = Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content="The capital of France is Paris.",
        summary="capital of France",
        salience=0.9,
        confidence=1.0,
        decay_rate=0.005,
        created_at=datetime.now(timezone.utc),
        extraction_confidence=0.73,
    )
    store.upsert(m)
    got = store.get(m.id)
    assert got is not None
    assert abs(got.extraction_confidence - 0.73) < 1e-6
    store.close()


def test_extraction_confidence_migration_on_old_db(tmp_path: Path):
    """Simulate an older DB without the column, verify migration adds it cleanly."""
    import sqlite3
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    # Create a memories table without the extraction_confidence column (legacy schema)
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, type TEXT, state TEXT, content TEXT,
            summary TEXT, salience REAL, confidence REAL, decay_rate REAL,
            created_at TEXT, last_accessed TEXT, access_count INTEGER, agent_id TEXT,
            appraisal_relevance REAL, appraisal_novelty REAL, appraisal_goal_conduciveness REAL,
            somatic_valence REAL, somatic_bias TEXT, somatic_trigger TEXT,
            emotion_primary TEXT, emotion_intensity REAL, emotion_compound TEXT,
            encoding_mood_valence REAL, encoding_mood_arousal REAL,
            encoding_emotions TEXT, encoding_task TEXT,
            classification TEXT, source_events TEXT, schema_id TEXT, provenance TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Now open with our store; migration should add the column
    store = SQLiteMemoryStore(tmp_path)
    cols = {row[1] for row in store._get_conn().execute("PRAGMA table_info(memories)").fetchall()}
    assert "extraction_confidence" in cols
    store.close()


# --- StorageConfig + write-side dedup ---


def test_storage_config_defaults():
    cfg = StorageConfig()
    assert cfg.write_dedup_threshold == 0.0  # disabled by default
    assert cfg.merge_threshold == 0.95


def test_retrieval_config_use_extraction_confidence_default_true():
    rc = RetrievalConfig()
    assert rc.use_extraction_confidence is True


def test_storage_config_loads_from_yaml(tmp_path: Path):
    yml = tmp_path / "engram.yaml"
    yml.write_text(
        f"""
path: {tmp_path}/engram
storage:
  write_dedup_threshold: 0.92
  merge_threshold: 0.97
"""
    )
    cfg = Config.from_yaml(str(yml))
    assert cfg.storage.write_dedup_threshold == 0.92
    assert cfg.storage.merge_threshold == 0.97


def test_dedup_disabled_by_default(tmp_path: Path):
    """With threshold=0, identical content writes succeed (creates two memories)."""
    cfg = Config(path=str(tmp_path / "engram"))
    eng = Engram(config=cfg)
    eng._embeddings = _DeterministicEmbedder()
    eng._vector = SQLiteVecStore(tmp_path / "engram" / "vectors.db", dimension=16)
    try:
        eng.remember("user prefers postgres")
        eng.remember("user prefers postgres")
        status = eng.status()
        assert status["total_memories"] == 2
    finally:
        eng.close()


def test_dedup_blocks_near_duplicate(tmp_path: Path):
    """With threshold=0.92 and a deterministic embedder, identical content is deduped."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.storage.write_dedup_threshold = 0.92
    eng = Engram(config=cfg)
    eng._embeddings = _DeterministicEmbedder()
    eng._vector = SQLiteVecStore(tmp_path / "engram" / "vectors.db", dimension=16)
    try:
        eng.remember("the launch is scheduled for next monday")
        # The exact same content has cosine similarity 1.0 → should be deduped
        eng.remember("the launch is scheduled for next monday")

        status = eng.status()
        assert status["total_memories"] == 1
    finally:
        eng.close()


def test_dedup_does_not_block_distinct_content(tmp_path: Path):
    """Different content should not be deduped even with high threshold."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.storage.write_dedup_threshold = 0.92
    eng = Engram(config=cfg)
    eng._embeddings = _DeterministicEmbedder()
    eng._vector = SQLiteVecStore(tmp_path / "engram" / "vectors.db", dimension=16)
    try:
        eng.remember("the deploy succeeded")
        # Bag-of-chars distance is large; cosine well below 0.92
        eng.remember("xyz pqrst uvw mnokjihg")

        status = eng.status()
        assert status["total_memories"] == 2
    finally:
        eng.close()


def test_dedup_preserves_event_in_buffer(tmp_path: Path):
    """A deduped write should still hit the JSONL event store (source of truth)."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.storage.write_dedup_threshold = 0.92
    eng = Engram(config=cfg)
    eng._embeddings = _DeterministicEmbedder()
    eng._vector = SQLiteVecStore(tmp_path / "engram" / "vectors.db", dimension=16)
    try:
        eng.remember("some fact")
        eng.remember("some fact")  # deduped at projection

        # Both events should be in the buffer
        events = list(eng._buffer.scan())
        remember_events = [e for e in events if e.type.value == "explicit_remember"]
        assert len(remember_events) == 2, "Buffer must keep both events for replay/audit"
    finally:
        eng.close()


# --- Dual extraction (FactExtraction stage) ---


class _StubLLM:
    """Minimal LLM provider that returns a canned dict from extract_json()."""

    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt: str, system: str = "", max_tokens: int = 100) -> str:
        return ""

    def extract_json(self, prompt: str, system: str = "") -> dict:
        return self._payload


def _make_episode_ctx(content: str = "Closed the Acme deal for $450K on Tuesday."):
    """Build a StageContext with one EPISODE memory ready for FactExtraction."""
    from datetime import datetime, timezone
    from engram.consolidation.pipeline import StageContext
    from engram.core.types import Event, EventType, generate_event_id

    ev = Event(
        id=generate_event_id(), ts=datetime.now(timezone.utc),
        type=EventType.EVENT_CAPTURE, content=content,
    )
    mem = Memory.from_event(ev, memory_type=MemoryType.EPISODE)
    ctx = StageContext(memories_created=[mem])
    return ctx, mem


def test_fact_extraction_dual_schema_persists_confidence_and_properties(tmp_path: Path):
    """New schema: per-fact confidence flows to Memory.extraction_confidence; properties persist."""
    from engram.consolidation.pipeline import FactExtraction, MemoryPersistence

    payload = {
        "facts": [
            {
                "text": "Acme deal closed for $450K",
                "confidence": 0.92,
                "properties": [
                    {"key": "deal_value", "value": "$450K", "type": "number", "confidence": 0.95},
                    {"key": "customer", "value": "Acme", "type": "entity", "confidence": 0.99},
                ],
            }
        ]
    }
    ctx, _episode = _make_episode_ctx()
    ctx.llm = _StubLLM(payload)
    ctx.store = SQLiteMemoryStore(tmp_path)
    try:
        FactExtraction().run(ctx)

        facts = [m for m in ctx.memories_created if m.type == MemoryType.FACT]
        assert len(facts) == 1
        f = facts[0]
        assert abs(f.extraction_confidence - 0.92) < 1e-6
        assert getattr(f, "_pending_properties", None), "properties should be stashed for persistence"

        # Persistence stage should call upsert_properties
        MemoryPersistence().run(ctx)
        rows = ctx.store.get_properties(f.id)
        keys = {r["key"]: r for r in rows}
        assert "deal_value" in keys and keys["deal_value"]["value"] == "$450K"
        assert keys["deal_value"]["type"] == "number"
        assert abs(keys["deal_value"]["confidence"] - 0.95) < 1e-6
        assert "customer" in keys and keys["customer"]["type"] == "entity"

        # And the memory's extraction_confidence is durable
        got = ctx.store.get(f.id)
        assert abs(got.extraction_confidence - 0.92) < 1e-6
    finally:
        ctx.store.close()


def test_fact_extraction_legacy_schema_degrades_gracefully(tmp_path: Path):
    """Legacy {"facts": ["...", "..."]} schema still works; default conf=1.0, no props."""
    from engram.consolidation.pipeline import FactExtraction, MemoryPersistence

    payload = {"facts": ["The launch is Monday.", "Pricing is $99 per seat."]}
    ctx, _episode = _make_episode_ctx()
    ctx.llm = _StubLLM(payload)
    ctx.store = SQLiteMemoryStore(tmp_path)
    try:
        FactExtraction().run(ctx)
        facts = [m for m in ctx.memories_created if m.type == MemoryType.FACT]
        assert len(facts) == 2
        for f in facts:
            assert f.extraction_confidence == 1.0
            assert not getattr(f, "_pending_properties", None)

        MemoryPersistence().run(ctx)
        for f in facts:
            assert ctx.store.get_properties(f.id) == []
    finally:
        ctx.store.close()


def test_fact_extraction_clamps_out_of_range_confidence(tmp_path: Path):
    """LLM-supplied confidences outside [0, 1] are clamped, garbage degrades to 1.0."""
    from engram.consolidation.pipeline import FactExtraction

    payload = {
        "facts": [
            {"text": "fact alpha goes here", "confidence": 1.5, "properties": []},
            {"text": "fact beta goes here", "confidence": -0.4, "properties": []},
            {"text": "fact gamma goes here", "confidence": "garbage", "properties": []},
        ]
    }
    ctx, _episode = _make_episode_ctx()
    ctx.llm = _StubLLM(payload)
    FactExtraction().run(ctx)
    facts = [m for m in ctx.memories_created if m.type == MemoryType.FACT]
    confs = sorted(f.extraction_confidence for f in facts)
    assert confs == [0.0, 1.0, 1.0]


# --- Retrieval engine: extraction_confidence multiplier (gated) ---


def _mk_fact(store, content: str, extraction_confidence: float, mid: str, ts=None):
    """Helper: build + persist a FACT memory with a given extraction_confidence."""
    from datetime import datetime, timezone
    from engram.core.types import EncodingContext
    if ts is None:
        ts = datetime.now(timezone.utc)
    m = Memory(
        id=mid,
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:50],
        salience=0.5,
        confidence=1.0,
        decay_rate=0.1,
        created_at=ts,
        last_accessed=ts,
        encoding_context=EncodingContext(),
        extraction_confidence=extraction_confidence,
    )
    store.upsert(m)
    return m


def test_retrieval_extraction_confidence_reorders_when_enabled(tmp_path: Path):
    """Two memories with identical lexical match but different extraction_confidence:
    when use_extraction_confidence=True, the higher-confidence one ranks first."""
    from datetime import datetime, timezone
    from engram.retrieval.engine import RetrievalEngine

    store = SQLiteMemoryStore(tmp_path)
    try:
        # Same content + same timestamps → identical BM25/recency; differ only by extraction_confidence
        ts = datetime.now(timezone.utc)
        _mk_fact(store, "the launch is scheduled for monday", 1.0, "fact-high", ts=ts)
        _mk_fact(store, "the launch is scheduled for monday", 0.5, "fact-low", ts=ts)

        # Flag ON: fact-low should be roughly half fact-high (0.5 multiplier dominates)
        cfg_on = RetrievalConfig(use_extraction_confidence=True)
        eng = RetrievalEngine(store=store, config=cfg_on)
        results_on = eng.search("launch monday", limit=5)
        assert len(results_on) == 2
        scores_on = {r.memory.id: r.score for r in results_on}
        assert scores_on["fact-high"] > scores_on["fact-low"]
        ratio_on = scores_on["fact-low"] / scores_on["fact-high"]
        assert 0.45 < ratio_on < 0.55, f"expected ~0.5 ratio when flag on, got {ratio_on}"

        # Flag OFF: scores should be (nearly) identical — extraction_confidence is ignored
        cfg_off = RetrievalConfig(use_extraction_confidence=False)
        eng_off = RetrievalEngine(store=store, config=cfg_off)
        results_off = eng_off.search("launch monday", limit=5)
        scores_off = {r.memory.id: r.score for r in results_off}
        ratio_off = min(scores_off.values()) / max(scores_off.values())
        assert ratio_off > 0.98, f"expected near-equal scores when flag off, got ratio {ratio_off}"
    finally:
        store.close()


def test_retrieval_extraction_confidence_none_treated_as_one(tmp_path: Path):
    """A None / missing extraction_confidence should not crash and should default to 1.0."""
    from engram.retrieval.engine import RetrievalEngine

    store = SQLiteMemoryStore(tmp_path)
    try:
        m = _mk_fact(store, "alpha bravo charlie", 1.0, "fact-x")
        # Simulate legacy in-memory object with None
        m.extraction_confidence = None  # type: ignore[assignment]
        # We bypass DB — patch what the engine sees by re-upserting after restoring to 1.0
        # (DB column is REAL NOT NULL with default; here we just verify engine tolerates None on the object.)
        # Use a tiny custom store wrapper: easier to call _compute_score directly.
        from engram.core.types import ScoredMemory
        from engram.retrieval.engine import QueryIntent
        from datetime import datetime, timezone
        eng = RetrievalEngine(store=store, config=RetrievalConfig(use_extraction_confidence=True))
        sm = ScoredMemory(memory=m, score=0.0, sources={"bm25_rank": 1})
        score = eng._compute_score(sm, QueryIntent(query="x"), None, datetime.now(timezone.utc))
        assert score >= 0.0
        assert sm.sources["extraction_confidence"] == 1.0
    finally:
        store.close()

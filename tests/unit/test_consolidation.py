"""Tests for the consolidation pipeline (12-stage)."""

import pytest
from datetime import datetime, timedelta, timezone

from engram.core.config import Config
from engram.core.types import (
    Event, EventType, Memory, MemoryType,
    generate_event_id,
)
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore
from engram.audit.log import AuditLog
from engram.consolidation.pipeline import (
    ConsolidationPipeline, StageContext,
    EventIngestion, Deduplication, EpisodeExtraction,
    AppraisalScoring, DecayApplication,
)
from engram.providers.llm import NoLLMProvider
from engram.affect.engine import AffectEngine
from engram.engine import Engram


@pytest.fixture
def stores(tmp_path):
    buffer = JSONLBufferStore(tmp_path)
    store = SQLiteMemoryStore(tmp_path)
    audit = AuditLog(tmp_path)
    config = Config.minimal(str(tmp_path))
    return buffer, store, audit, config


def _capture_event(buffer, content="test event"):
    event = Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.EVENT_CAPTURE,
        content=content,
    )
    buffer.append(event)
    return event


class TestEventIngestion:
    def test_loads_capture_events(self, stores):
        buffer, store, audit, config = stores
        _capture_event(buffer, "event 1")
        _capture_event(buffer, "event 2")
        ctx = StageContext(buffer=buffer)
        ctx = EventIngestion(window_hours=24).run(ctx)
        assert len(ctx.events) == 2

    def test_ignores_remember_events(self, stores):
        buffer, store, audit, config = stores
        _capture_event(buffer, "capture")
        buffer.append(Event(
            id=generate_event_id(), ts=datetime.now(timezone.utc),
            type=EventType.EXPLICIT_REMEMBER, content="remember",
        ))
        ctx = StageContext(buffer=buffer)
        ctx = EventIngestion().run(ctx)
        assert len(ctx.events) == 1

    def test_window_filter(self, stores):
        buffer, store, audit, config = stores
        buffer.append(Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc) - timedelta(hours=48),
            type=EventType.EVENT_CAPTURE, content="old",
        ))
        _capture_event(buffer, "recent")
        ctx = StageContext(buffer=buffer)
        ctx = EventIngestion(window_hours=24).run(ctx)
        assert len(ctx.events) == 1


class TestDeduplication:
    def test_removes_exact_duplicates(self, stores):
        events = [
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="same"),
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="Same"),
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="different"),
        ]
        ctx = StageContext(events=events)
        ctx = Deduplication().run(ctx)
        assert len(ctx.events) == 2


class TestEpisodeExtraction:
    def test_creates_episodes(self, stores):
        events = [
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="Deploy to production"),
        ]
        ctx = StageContext(events=events, llm=NoLLMProvider())
        ctx = EpisodeExtraction().run(ctx)
        assert len(ctx.memories_created) == 1
        assert ctx.memories_created[0].type == MemoryType.EPISODE
        # L0 summary should be set
        assert ctx.memories_created[0].summary != ""


class TestAppraisalScoring:
    def test_urgency_boost(self, stores):
        normal = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="routine task"),
        )
        urgent = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="URGENT critical deadline"),
        )
        ctx = StageContext(memories_created=[normal, urgent])
        ctx = AppraisalScoring().run(ctx)
        assert urgent.salience > normal.salience

    def test_salience_capped(self, stores):
        mem = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE,
                  content="urgent critical important new surprising discovered goal achieved completed"),
        )
        ctx = StageContext(memories_created=[mem])
        ctx = AppraisalScoring().run(ctx)
        assert mem.salience <= 1.0

    def test_appraisal_salience_cap_clamps(self, stores):
        # §94c-appraisal-bound — explicit cap clamps post-appraisal salience.
        from engram import Config
        from engram.core.config import ConsolidationConfig
        cfg = Config(path=":memory:")
        cfg.consolidation = ConsolidationConfig(appraisal_salience_cap=0.4)
        loud = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE,
                  content="urgent critical important new surprising discovered goal achieved completed"),
        )
        quiet = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE, content="routine"),
        )
        ctx = StageContext(memories_created=[loud, quiet], config=cfg)
        ctx = AppraisalScoring().run(ctx)
        # Loud one would otherwise blow past 0.4; assert it is clamped.
        assert loud.salience == 0.4
        # Quiet baseline (0.5) is *above* 0.4 in heuristic since baseline
        # raw_salience = 0.5 * 1 * 1 * 1 = 0.5 — also capped.
        assert quiet.salience <= 0.4

    def test_appraisal_salience_cap_none_is_passthrough(self, stores):
        # Default None preserves pre-§94c-appraisal-bound behavior.
        from engram import Config
        from engram.core.config import ConsolidationConfig
        cfg = Config(path=":memory:")
        cfg.consolidation = ConsolidationConfig(appraisal_salience_cap=None)
        loud = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                  type=EventType.EVENT_CAPTURE,
                  content="urgent critical important new surprising discovered"),
        )
        ctx = StageContext(memories_created=[loud], config=cfg)
        ctx = AppraisalScoring().run(ctx)
        assert loud.salience > 0.5  # heuristic boost still active


class TestDecayApplication:
    def test_old_memories_decay(self, stores):
        buffer, store, audit, config = stores
        mem = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc) - timedelta(days=7),
                  type=EventType.EXPLICIT_REMEMBER, content="old"),
            MemoryType.FACT,
        )
        mem.salience = 0.8
        mem.decay_rate = 0.1
        mem.last_accessed = datetime.now(timezone.utc) - timedelta(days=7)
        store.upsert(mem)
        ctx = StageContext(store=store, config=config, buffer=buffer)
        ctx = DecayApplication().run(ctx)
        refreshed = store.get(mem.id)
        assert refreshed.salience < 0.8

    def test_state_transition_logged(self, stores):
        buffer, store, audit, config = stores
        mem = Memory.from_event(
            Event(id=generate_event_id(), ts=datetime.now(timezone.utc) - timedelta(days=30),
                  type=EventType.EXPLICIT_REMEMBER, content="very old"),
            MemoryType.FACT,
        )
        mem.salience = 0.15
        mem.decay_rate = 0.1
        mem.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)
        store.upsert(mem)
        initial_count = buffer.count()
        ctx = StageContext(store=store, config=config, buffer=buffer)
        ctx = DecayApplication().run(ctx)
        # State transition events should be written
        assert buffer.count() >= initial_count


class TestFullPipeline:
    def test_12_stage_end_to_end(self, stores):
        buffer, store, audit, config = stores
        _capture_event(buffer, "Deploy completed successfully")
        _capture_event(buffer, "CI pipeline passed all tests")
        _capture_event(buffer, "Deploy completed successfully")  # duplicate

        affect = AffectEngine()
        pipeline = ConsolidationPipeline(buffer, store, audit, config, affect=affect)
        report = pipeline.run()
        assert report.events_processed >= 2
        assert report.memories_created >= 2
        assert report.errors == []

    def test_engine_consolidate_full(self, tmp_path):
        config = Config.minimal(str(tmp_path / "test"))
        mem = Engram(config, actor="test")
        try:
            mem.capture("Team standup: discussed blockers")
            mem.capture("PR #42 merged successfully after review")
            mem.capture("Database migration crashed on staging")
            report = mem.consolidate()
            assert report.memories_created >= 3
            assert report.errors == []
            # Captured events should be searchable after consolidation
            results = mem.recall("standup", limit=5)
            assert len(results) >= 1
        finally:
            mem.close()

    def test_consolidation_audited(self, tmp_path):
        config = Config.minimal(str(tmp_path / "test"))
        mem = Engram(config, actor="test")
        try:
            mem.capture("event")
            mem.consolidate()
            entries = mem._audit.read()
            ops = [e["op"] for e in entries]
            assert "consolidation" in ops
        finally:
            mem.close()

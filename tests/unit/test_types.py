"""Tests for core types."""

import pytest
from datetime import datetime, timezone

from engram.core.types import (
    Event, EventType, Memory, MemoryType, MemoryState,
    DataClassification,
    Provenance, Modification,
    generate_event_id, generate_memory_id, content_hash,
)


class TestEventId:
    def test_format(self):
        eid = generate_event_id()
        assert eid.startswith("evt-")
        parts = eid.split("-")
        assert len(parts) == 3

    def test_unique(self):
        ids = {generate_event_id() for _ in range(100)}
        assert len(ids) == 100


class TestMemoryId:
    def test_format(self):
        mid = generate_memory_id(MemoryType.FACT)
        assert mid.startswith("mem-fa-")

    def test_episode_prefix(self):
        mid = generate_memory_id(MemoryType.EPISODE)
        assert mid.startswith("mem-ep-")


class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello")
        h2 = content_hash("hello")
        assert h1 == h2

    def test_different_content(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = content_hash("test")
        assert len(h) == 16


class TestEvent:
    def test_round_trip(self):
        event = Event(
            id="evt-123-abc",
            ts=datetime(2026, 3, 15, tzinfo=timezone.utc),
            type=EventType.EXPLICIT_REMEMBER,
            content="test content",
            metadata={"key": "value"},
            salience_hint=0.8,
            context={"mood_valence": 0.5},
        )
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.id == event.id
        assert restored.type == event.type
        assert restored.content == event.content
        assert restored.metadata == event.metadata
        assert restored.salience_hint == event.salience_hint
        assert restored.context == event.context

    def test_minimal(self):
        event = Event(
            id="evt-1-a",
            ts=datetime.now(timezone.utc),
            type=EventType.EVENT_CAPTURE,
            content="minimal",
        )
        d = event.to_dict()
        restored = Event.from_dict(d)
        assert restored.content == "minimal"
        assert restored.metadata == {}
        assert restored.salience_hint == 0.0


class TestMemory:
    def test_from_event(self):
        event = Event(
            id="evt-1-a",
            ts=datetime.now(timezone.utc),
            type=EventType.EXPLICIT_REMEMBER,
            content="User prefers dark mode",
            salience_hint=0.3,
            context={"mood_valence": 0.5, "active_task": "ui-setup"},
        )
        memory = Memory.from_event(event, MemoryType.FACT)
        assert memory.type == MemoryType.FACT
        assert memory.state == MemoryState.ACTIVE
        assert memory.content == "User prefers dark mode"
        assert memory.salience == pytest.approx(0.8, abs=0.01)  # 0.5 + 0.3
        assert memory.encoding_context.mood_valence == 0.5
        assert memory.encoding_context.task == "ui-setup"
        assert "evt-1-a" in memory.source_events

    def test_salience_clamped(self):
        event = Event(
            id="evt-1-b",
            ts=datetime.now(timezone.utc),
            type=EventType.EXPLICIT_REMEMBER,
            content="test",
            salience_hint=2.0,  # over max
        )
        memory = Memory.from_event(event)
        assert memory.salience <= 1.0


class TestProvenance:
    def test_round_trip(self):
        prov = Provenance(
            source_events=["evt-1", "evt-2"],
            created_by="consolidation-001",
            modifications=[
                Modification(
                    ts=datetime(2026, 3, 15, tzinfo=timezone.utc),
                    operation="decay",
                    old_value={"salience": 0.7},
                    new_value={"salience": 0.65},
                    reason="Ebbinghaus (1d, λ=0.05)",
                )
            ],
        )
        d = prov.to_dict()
        restored = Provenance.from_dict(d)
        assert restored.source_events == ["evt-1", "evt-2"]
        assert restored.created_by == "consolidation-001"
        assert len(restored.modifications) == 1
        assert restored.modifications[0].operation == "decay"


class TestEnums:
    def test_event_types_are_strings(self):
        assert EventType.EXPLICIT_REMEMBER.value == "explicit_remember"
        assert EventType.CONSOLIDATION_START.value == "consolidation_start"

    def test_memory_types(self):
        assert MemoryType.EPISODE.value == "episode"
        assert MemoryType.SCHEMA.value == "schema"

    def test_classification_ordering(self):
        levels = list(DataClassification)
        assert levels[0] == DataClassification.PUBLIC
        assert levels[-1] == DataClassification.RESTRICTED

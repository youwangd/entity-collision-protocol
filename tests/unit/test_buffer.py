"""Tests for JSONL event buffer."""

import pytest
from datetime import datetime, timezone

from engram.core.types import Event, EventType, generate_event_id
from engram.store.buffer import JSONLBufferStore


@pytest.fixture
def buffer(tmp_path):
    return JSONLBufferStore(tmp_path)


def _make_event(content="test", event_type=EventType.EXPLICIT_REMEMBER, ts=None):
    return Event(
        id=generate_event_id(),
        ts=ts or datetime.now(timezone.utc),
        type=event_type,
        content=content,
    )


class TestAppend:
    def test_append_creates_file(self, buffer):
        event = _make_event()
        eid = buffer.append(event)
        assert eid == event.id
        assert buffer.path.exists()

    def test_append_multiple(self, buffer):
        for i in range(10):
            buffer.append(_make_event(f"event-{i}"))
        assert buffer.count() == 10

    def test_append_returns_event_id(self, buffer):
        event = _make_event()
        eid = buffer.append(event)
        assert eid.startswith("evt-")


class TestScan:
    def test_scan_all(self, buffer):
        for i in range(5):
            buffer.append(_make_event(f"event-{i}"))
        events = list(buffer.scan())
        assert len(events) == 5

    def test_scan_since(self, buffer):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2026, 3, 15, tzinfo=timezone.utc)
        buffer.append(_make_event("old", ts=old))
        buffer.append(_make_event("new", ts=recent))
        events = list(buffer.scan(since=datetime(2025, 1, 1, tzinfo=timezone.utc)))
        assert len(events) == 1
        assert events[0].content == "new"

    def test_scan_by_type(self, buffer):
        buffer.append(_make_event("remember", EventType.EXPLICIT_REMEMBER))
        buffer.append(_make_event("capture", EventType.EVENT_CAPTURE))
        buffer.append(_make_event("remember2", EventType.EXPLICIT_REMEMBER))
        events = list(buffer.scan(event_type=EventType.EXPLICIT_REMEMBER))
        assert len(events) == 2

    def test_scan_with_limit(self, buffer):
        for i in range(20):
            buffer.append(_make_event(f"event-{i}"))
        events = list(buffer.scan(limit=5))
        assert len(events) == 5

    def test_scan_empty(self, buffer):
        events = list(buffer.scan())
        assert len(events) == 0

    def test_scan_skips_corrupted_lines(self, buffer):
        buffer.append(_make_event("good1"))
        # Manually write a corrupted line
        with open(buffer.path, "a") as f:
            f.write("this is not json\n")
        buffer.append(_make_event("good2"))
        events = list(buffer.scan())
        assert len(events) == 2


class TestCount:
    def test_empty(self, buffer):
        assert buffer.count() == 0

    def test_after_append(self, buffer):
        buffer.append(_make_event())
        buffer.append(_make_event())
        assert buffer.count() == 2


class TestLastEventId:
    def test_empty(self, buffer):
        assert buffer.last_event_id() is None

    def test_after_append(self, buffer):
        e1 = _make_event("first")
        e2 = _make_event("second")
        buffer.append(e1)
        buffer.append(e2)
        assert buffer.last_event_id() == e2.id


class TestTruncate:
    def test_truncate_old(self, buffer):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        new = datetime(2026, 3, 15, tzinfo=timezone.utc)
        buffer.append(_make_event("old1", ts=old))
        buffer.append(_make_event("old2", ts=old))
        buffer.append(_make_event("new1", ts=new))
        removed = buffer.truncate_before(datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert removed == 2
        assert buffer.count() == 1

    def test_truncate_none(self, buffer):
        now = datetime.now(timezone.utc)
        buffer.append(_make_event("recent", ts=now))
        removed = buffer.truncate_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert removed == 0
        assert buffer.count() == 1


class TestClear:
    def test_clear(self, buffer):
        buffer.append(_make_event())
        buffer.append(_make_event())
        buffer.clear()
        assert buffer.count() == 0

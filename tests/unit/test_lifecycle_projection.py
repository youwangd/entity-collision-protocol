"""Tests for the buffer→reducer projection of schema-lifecycle events.

The reducer is fuzzed in `tests/property/test_schema_lifecycle.py`.
This file pins the wire format and the projection adapter:

  - `make_lifecycle_event` produces buffer-shaped Events that round-trip
    through JSONL append+scan and decode back to the original
    SchemaLifecycleEvent.
  - `snapshot_from_buffer` matches `reduce_events` over the same logical
    sequence (i.e. the projection is just the reducer composed with
    decode).
  - Malformed metadata is dropped silently (lenient projection).
  - Non-lifecycle events sharing the buffer don't perturb the snapshot.
"""
from __future__ import annotations

from pathlib import Path

from engram.consolidation.lifecycle_projection import (
    event_to_lifecycle,
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaLifecycleEvent,
    SchemaStatus,
    reduce_events,
)
from engram.core.types import Event, EventType, generate_event_id
from engram.store.buffer import JSONLBufferStore
from datetime import datetime, timezone


def _bare_buffer(tmp_path: Path) -> JSONLBufferStore:
    return JSONLBufferStore(base_path=tmp_path)


def test_make_event_roundtrips_through_buffer(tmp_path: Path):
    buf = _bare_buffer(tmp_path)
    ev = make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
        content="users prefer postgres",
    )
    buf.append(ev)
    scanned = list(buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE))
    assert len(scanned) == 1
    decoded = event_to_lifecycle(scanned[0])
    assert decoded == SchemaLifecycleEvent(
        schema_id="s1",
        kind=EventKind.CREATE,
        window_id="w1",
        ts=decoded.ts,  # ts is unix-seconds projection of the buffer ts
    )


def test_snapshot_matches_reducer(tmp_path: Path):
    buf = _bare_buffer(tmp_path)
    seq = [
        (EventKind.CREATE, "w1"),
        (EventKind.PROMOTE, "w1"),
        (EventKind.DEPRECATE, "w2"),
        (EventKind.RECOVER, "w3"),
        (EventKind.BUMP_VERSION, "w3"),
    ]
    for kind, win in seq:
        buf.append(make_lifecycle_event(schema_id="s", kind=kind, window_id=win))

    snap = snapshot_from_buffer(buf, strict=False)
    # Build the same logical sequence and reduce directly.
    direct = reduce_events(
        [
            SchemaLifecycleEvent(schema_id="s", kind=k, window_id=w, ts=0)
            for (k, w) in seq
        ],
        strict=False,
    )
    # Status, version, counts must match. (ts differs by construction —
    # the buffer encodes wall-clock ts, the direct path uses 0 — but the
    # reducer's output is `last_window_id`-driven, not ts-driven.)
    assert snap["s"].status == direct["s"].status == SchemaStatus.INFERRED
    assert snap["s"].version == direct["s"].version == 2
    assert snap["s"].promote_count == direct["s"].promote_count == 1
    assert snap["s"].deprecate_count == direct["s"].deprecate_count == 1
    assert snap["s"].recover_count == direct["s"].recover_count == 1


def test_malformed_metadata_is_dropped(tmp_path: Path):
    buf = _bare_buffer(tmp_path)
    # Good event: creates s1.
    buf.append(make_lifecycle_event(schema_id="s1", kind=EventKind.CREATE,
                                    window_id="w1"))
    # Malformed: missing schema_id.
    buf.append(Event(
        id=generate_event_id(), ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE,
        content="", metadata={"kind": "promote", "window_id": "w1"},
    ))
    # Malformed: bogus kind.
    buf.append(Event(
        id=generate_event_id(), ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE,
        content="", metadata={"schema_id": "s1", "kind": "obliterate"},
    ))
    snap = snapshot_from_buffer(buf, strict=False)
    assert set(snap.keys()) == {"s1"}
    assert snap["s1"].status is SchemaStatus.INFERRED
    assert snap["s1"].promote_count == 0  # the malformed promote was dropped


def test_non_lifecycle_events_ignored(tmp_path: Path):
    buf = _bare_buffer(tmp_path)
    # Unrelated events sharing the buffer.
    buf.append(Event(
        id=generate_event_id(), ts=datetime.now(timezone.utc),
        type=EventType.EVENT_CAPTURE, content="hello",
    ))
    buf.append(Event(
        id=generate_event_id(), ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_START, content="",
    ))
    buf.append(make_lifecycle_event(schema_id="s2", kind=EventKind.CREATE,
                                    window_id="w1"))
    snap = snapshot_from_buffer(buf)
    assert list(snap.keys()) == ["s2"]


def test_event_to_lifecycle_returns_none_for_other_types():
    e = Event(
        id="x", ts=datetime.now(timezone.utc),
        type=EventType.EVENT_CAPTURE, content="hi",
    )
    assert event_to_lifecycle(e) is None

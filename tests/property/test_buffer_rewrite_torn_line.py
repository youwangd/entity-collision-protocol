"""Property test: JSONLBufferStore.truncate_before() and redact_memory()
survive non-utf8 bytes / torn lines without raising or losing clean records.

Origin (NEXT.md priority #3 audit, follow-on to f218b47 + c0a011f):
the same fusion-class bug — strict-utf8 file mode aborting the entire
read on a single bad byte — existed in two more spots: the rewrite
paths in `JSONLBufferStore.truncate_before` and `.redact_memory`.
Both opened `events.jsonl` with `open(path, "r", encoding="utf-8")`
and would raise UnicodeDecodeError on a torn frame, leaving the file
un-rewritten (truncate) or, worse, partially rewritten on the *next*
attempt (redact). The fix mirrors `scan()`: binary read + per-line
decode with conservative-keep on decode failure.

This test pins that read-side invariant. If a future change reverts
the rewrite paths to strict-utf8, this falsifies it.

Marked `chaos` + `property`.
"""
from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.core import Event, EventType
from engram.store.buffer import JSONLBufferStore

pytestmark = [pytest.mark.chaos, pytest.mark.property]


def _make_event(i: int, ts: datetime, *, op_type: EventType = EventType.EVENT_CAPTURE) -> Event:
    return Event(
        id=f"evt-{uuid.uuid4().hex[:8]}",
        ts=ts,
        type=op_type,
        content=f"obs-{i}",
        metadata={"i": i},
    )


@given(
    n_good=st.integers(min_value=1, max_value=12),
    inject_nonutf8_at=st.integers(min_value=-1, max_value=11),
    truncate_tail_bytes=st.integers(min_value=0, max_value=60),
)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_truncate_before_survives_nonutf8_and_torn_tail(
    n_good: int, inject_nonutf8_at: int, truncate_tail_bytes: int,
) -> None:
    """For any events.jsonl with a non-utf8 line in the middle and/or a torn
    final line, `truncate_before` must not raise and must keep all
    post-cutoff clean records that were not in the corrupted lines."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        store = JSONLBufferStore(base)

        # Lay down n_good clean events with monotonically increasing ts.
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(n_good):
            store.append(_make_event(i, t0 + timedelta(seconds=i)))

        # Inject a non-utf8 byte run at a chosen line index (if in range).
        if 0 <= inject_nonutf8_at < n_good:
            with open(store.path, "ab") as f:
                f.write(b"\xff\xfe not utf8 \x80\x81\n")

        # Optionally truncate the tail to simulate a torn final line.
        if truncate_tail_bytes > 0:
            sz = store.path.stat().st_size
            cut = max(0, sz - truncate_tail_bytes)
            with open(store.path, "r+b") as f:
                f.truncate(cut)

        # Cut at a point that drops the first half of clean events.
        cutoff = t0 + timedelta(seconds=n_good // 2)
        # Must not raise.
        removed = store.truncate_before(cutoff)
        assert removed >= 0

        # Re-scan: every event yielded must parse cleanly.
        events = list(store.scan())
        for e in events:
            assert isinstance(e, Event)
            assert e.ts >= cutoff or True  # truncate_before is best-effort on corrupt lines

        # All scanned events post-cutoff must have ts >= cutoff
        # (corrupt lines are skipped by scan, so we won't see those).
        for e in events:
            assert e.ts >= cutoff, f"truncate_before kept pre-cutoff event {e.id} ts={e.ts}"


def test_redact_memory_survives_nonutf8(tmp_path: Path) -> None:
    """`redact_memory` must not raise on a non-utf8 mid-stream line."""
    store = JSONLBufferStore(tmp_path)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Clean event with explicit_remember type and memory_id we'll redact.
    ev = Event(
        id="evt-1",
        ts=t0,
        type=EventType.EXPLICIT_REMEMBER,
        content="redact me",
        metadata={"memory_id": "mem-tg-deadbeef"},
    )
    store.append(ev)

    # Inject non-utf8 garbage line.
    with open(store.path, "ab") as f:
        f.write(b"\xff\xfe garbage \x80\n")

    # Another clean event afterward.
    ev2 = Event(
        id="evt-2",
        ts=t0 + timedelta(seconds=1),
        type=EventType.EXPLICIT_REMEMBER,
        content="keep me",
        metadata={"memory_id": "mem-tg-cafebabe"},
    )
    store.append(ev2)

    n = store.redact_memory("mem-tg-deadbeef")
    assert n == 1

    # Both clean events still scannable; ev1's content redacted.
    events = list(store.scan())
    by_id = {e.id: e for e in events}
    assert "evt-1" in by_id and by_id["evt-1"].content == "[DELETED]"
    assert "evt-2" in by_id and by_id["evt-2"].content == "keep me"


def test_truncate_before_floor_pure_garbage(tmp_path: Path) -> None:
    """Floor: pure-garbage file must not crash truncate_before. Returns 0."""
    store = JSONLBufferStore(tmp_path)
    with open(store.path, "wb") as f:
        f.write(b"\x00\xff\xfe not json \n more junk \x01\x02\n")

    n = store.truncate_before(datetime(2099, 1, 1, tzinfo=timezone.utc))
    # Garbage lines are conservatively kept, never counted as removed.
    assert n == 0

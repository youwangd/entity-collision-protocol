"""Property test: buffer→reducer projection equals direct reducer fold.

Pins NEXT.md priority #4. The point is that *no matter what* sequence of
lifecycle events we emit through `make_lifecycle_event` + `JSONLBufferStore`
+ `snapshot_from_buffer`, the resulting state must equal what we'd get by
calling `reduce_events` directly on the same logical sequence.

This locks the wire format: any drift between `make_lifecycle_event`
(writer) and `event_to_lifecycle` (decoder) breaks this property.

Notes:
- We compare `SchemaState` dicts. `SchemaState` does not carry a timestamp,
  so the ts-difference between the two paths (buffer encodes wall-clock,
  direct uses 0) cannot perturb equality.
- `last_window_id` *is* compared. The reducer assigns `last_window_id` from
  the event's `window_id` whenever truthy, so the projection and direct
  fold land on the same value as long as the wire format preserves
  window_id verbatim.
- Lenient mode only: strict mode would raise on illegal sequences and the
  fuzzer naturally generates plenty of those — they're already covered in
  `tests/property/test_schema_lifecycle.py`. The projection's contract is
  precisely "behave like the lenient reducer over a decoded stream."
"""
from __future__ import annotations


from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.lifecycle_projection import (
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaLifecycleEvent,
    reduce_events,
)
from engram.store.buffer import JSONLBufferStore


_SCHEMA_IDS = ["s1", "s2", "s3"]
_WINDOW_IDS = ["w1", "w2", "w3", "w4"]


_event_strategy = st.tuples(
    st.sampled_from(_SCHEMA_IDS),
    st.sampled_from(list(EventKind)),
    st.sampled_from(_WINDOW_IDS),
)


@given(events=st.lists(_event_strategy, max_size=30))
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_projection_equals_direct_reduce(tmp_path_factory, events):
    """For any event sequence: snapshot_from_buffer == reduce_events(direct).

    Uses tmp_path_factory so each Hypothesis example gets a fresh buffer
    directory (function_scoped_fixture would only give us one).
    """
    tmpdir = tmp_path_factory.mktemp("lifecycle_proj")
    buf = JSONLBufferStore(base_path=tmpdir)
    for sid, kind, win in events:
        buf.append(make_lifecycle_event(
            schema_id=sid, kind=kind, window_id=win,
        ))

    direct = reduce_events(
        [
            SchemaLifecycleEvent(schema_id=sid, kind=kind, window_id=win, ts=0)
            for (sid, kind, win) in events
        ],
        strict=False,
    )
    snap = snapshot_from_buffer(buf, strict=False)
    assert set(snap.keys()) == set(direct.keys())
    for sid, st_direct in direct.items():
        st_snap = snap[sid]
        # Compare the storage-relevant fields. (SchemaState.__eq__ would
        # work too — there is no ts field on SchemaState.)
        assert st_snap.status == st_direct.status, sid
        assert st_snap.version == st_direct.version, sid
        assert st_snap.promote_count == st_direct.promote_count, sid
        assert st_snap.deprecate_count == st_direct.deprecate_count, sid
        assert st_snap.recover_count == st_direct.recover_count, sid
        assert st_snap.last_window_id == st_direct.last_window_id, sid


@given(events=st.lists(_event_strategy, max_size=20))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_projection_resumable_via_partial_replay(tmp_path_factory, events):
    """Splitting the buffer's event stream and resuming via `initial=` is
    equivalent to a full replay. Mirrors the snapshot-resume invariant
    from the reducer-level fuzz, but exercised through the projection's
    decoder path.
    """
    if len(events) < 2:
        return
    tmpdir = tmp_path_factory.mktemp("lifecycle_proj_resume")
    buf = JSONLBufferStore(base_path=tmpdir)
    for sid, kind, win in events:
        buf.append(make_lifecycle_event(
            schema_id=sid, kind=kind, window_id=win,
        ))

    full = snapshot_from_buffer(buf, strict=False)

    # Re-decode through the projection's iterator so we use the same
    # decode path, then split and resume via the reducer's initial=.
    from engram.consolidation.lifecycle_projection import iter_lifecycle_events
    from engram.core.types import EventType

    decoded = list(iter_lifecycle_events(
        buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
    ))
    mid = len(decoded) // 2
    snap_a = reduce_events(decoded[:mid], strict=False)
    resumed = reduce_events(decoded[mid:], strict=False, initial=snap_a)
    assert resumed == full

"""Concurrency torture: parallel lifecycle-event appends to the JSONL buffer.

Closes the last open invariant of TODO-RESEARCH §B (Schema lifecycle):

  Invariant #4 — *Schema writes serialize against extraction writes.*
  Concretely: parallel appends of lifecycle events from N writer threads
  must (a) not lose an event, (b) not produce torn JSONL frames, and
  (c) the reduced snapshot must remain a valid fold of *some* total
  ordering of the appended events.

The buffer's append path already takes an exclusive ``fcntl.flock`` (see
``src/engram/store/buffer.py:90``), but the §B invariant has never been
fuzzed against parallel writers. This module is the regression gate.

Why a property test (not a one-off): the failure modes here are timing-
dependent — a partial-write torn frame, a lost event under racing
``O_APPEND`` writes, or a snapshot that disagrees with ``reduce_events``
applied to ``buf.scan()``. Hypothesis drives different (kind, schema_id,
window_id) population so we hit different DAG-state interactions.

Concurrency invariants:

  CL-I1  *Lossless persistence.* The number of well-formed lifecycle
         events visible in ``buf.scan(...)`` after the workers join
         equals the total number of ``buf.append`` calls issued. No
         torn JSONL frames, no lost rows, no truncated tail.

  CL-I2  *Projection consistency.* ``snapshot_from_buffer(buf)`` equals
         ``reduce_events(<events in scan order>, strict=False)``. The
         projection's contract — "behave like the lenient reducer over
         the decoded stream" — must hold under concurrent writes.

  CL-I3  *Per-schema causal order is total.* For every schema_id the
         per-schema subsequence of events recovered from the buffer is
         a valid lenient-reducer trajectory: no event is dropped, no
         event appears twice. Verified by reducing the per-schema
         subsequence and confirming the lenient reducer accepts it
         (no exception) and the resulting status is in the legal DAG.

  CL-I4  *Counts conserved.* The aggregate per-kind event count after
         the writers join equals the per-kind count handed to the writer
         pool. Catches regressions where a kind-specific code path
         (e.g. RECOVER's window-freshness check) interacts badly with a
         partially-written sibling event.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.lifecycle_projection import (
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaLifecycleEvent,
    SchemaStatus,
    reduce_events,
)
from engram.core.types import EventType
from engram.store.buffer import JSONLBufferStore


_SCHEMA_IDS = ["s1", "s2", "s3"]
_WINDOW_IDS = ["w1", "w2", "w3", "w4"]
_KINDS = list(EventKind)


_event_spec = st.tuples(
    st.sampled_from(_SCHEMA_IDS),
    st.sampled_from(_KINDS),
    st.sampled_from(_WINDOW_IDS),
)


def _drain_buffer_lifecycle(buf: JSONLBufferStore) -> list[SchemaLifecycleEvent]:
    """Decode every lifecycle event in scan order, drop malformed."""
    from engram.consolidation.lifecycle_projection import iter_lifecycle_events
    return list(
        iter_lifecycle_events(
            buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
        )
    )


@given(specs=st.lists(_event_spec, min_size=4, max_size=10))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_concurrent_appends_lossless_and_projection_consistent(
    tmp_path_factory, specs
):
    """CL-I1, CL-I2, CL-I4 in one shot.

    Drive ``len(specs)`` writer threads, each appending exactly one
    lifecycle event. After join: every event present, projection == fold.
    """
    tmpdir = tmp_path_factory.mktemp("lifecycle_concurrent")
    buf = JSONLBufferStore(base_path=tmpdir)

    barrier = threading.Barrier(len(specs))

    def _writer(spec):
        sid, kind, win = spec
        # Tighten the race window: every worker stalls on the barrier
        # before issuing its append, so the appends collide as closely
        # as the GIL/scheduler allow.
        barrier.wait()
        buf.append(
            make_lifecycle_event(schema_id=sid, kind=kind, window_id=win)
        )

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [pool.submit(_writer, sp) for sp in specs]
        for fut in as_completed(futures):
            fut.result()  # raise if any worker died

    decoded = _drain_buffer_lifecycle(buf)

    # CL-I1: every append survived, no torn frames.
    assert len(decoded) == len(specs), (
        f"expected {len(specs)} lifecycle events on disk, found {len(decoded)}"
    )

    # CL-I4: per-kind histogram conserved.
    expected_kinds = {k: 0 for k in _KINDS}
    for _sid, k, _win in specs:
        expected_kinds[k] += 1
    got_kinds = {k: 0 for k in _KINDS}
    for ev in decoded:
        got_kinds[ev.kind] += 1
    assert got_kinds == expected_kinds

    # CL-I2: projection equals direct reduce on the buffer's scan order.
    direct = reduce_events(decoded, strict=False)
    snap = snapshot_from_buffer(buf, strict=False)
    assert set(snap.keys()) == set(direct.keys())
    for sid, st_direct in direct.items():
        st_snap = snap[sid]
        assert st_snap.status == st_direct.status, sid
        assert st_snap.version == st_direct.version, sid
        assert st_snap.promote_count == st_direct.promote_count, sid
        assert st_snap.deprecate_count == st_direct.deprecate_count, sid
        assert st_snap.recover_count == st_direct.recover_count, sid


@given(specs=st.lists(_event_spec, min_size=6, max_size=10))
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_concurrent_appends_per_schema_trajectory_legal(
    tmp_path_factory, specs
):
    """CL-I3: every per-schema subsequence is a legal lenient reduction.

    The lenient reducer is total — it never raises — but the resulting
    status must always be one of the four legal states. Concurrency
    must not push a schema into a bogus status (e.g. ``None``-status
    leak from a torn fold).
    """
    tmpdir = tmp_path_factory.mktemp("lifecycle_concurrent_per_schema")
    buf = JSONLBufferStore(base_path=tmpdir)

    barrier = threading.Barrier(len(specs))

    def _writer(spec):
        sid, kind, win = spec
        barrier.wait()
        buf.append(
            make_lifecycle_event(schema_id=sid, kind=kind, window_id=win)
        )

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        for fut in as_completed([pool.submit(_writer, sp) for sp in specs]):
            fut.result()

    decoded = _drain_buffer_lifecycle(buf)
    assert len(decoded) == len(specs)

    # Bucket per schema_id, replay each in scan order under the lenient
    # reducer, and assert the terminal status is in the legal DAG.
    by_schema: dict[str, list] = {}
    for ev in decoded:
        by_schema.setdefault(ev.schema_id, []).append(ev)

    legal = {
        SchemaStatus.INFERRED,
        SchemaStatus.PROMOTED,
        SchemaStatus.DEPRECATED,
    }
    for sid, evs in by_schema.items():
        state = reduce_events(evs, strict=False)
        # Lenient mode may decide a schema never got CREATEd and so
        # absent from the snapshot; that's a legal outcome (e.g. a
        # PROMOTE with no prior CREATE is a no-op in lenient mode).
        if sid in state:
            assert state[sid].status in legal, (
                f"illegal terminal status for {sid}: {state[sid].status}"
            )


@given(
    n_writers=st.integers(min_value=4, max_value=6),
    base_specs=st.lists(_event_spec, min_size=3, max_size=5),
)
@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_concurrent_appends_amplified_writer_count(
    tmp_path_factory, n_writers, base_specs
):
    """Stress amplifier: each base_spec is appended by n_writers threads.

    Where the prior tests use one writer per event, this fans the same
    event out across multiple writers to amplify the GIL/flock contention
    surface. Total expected count = n_writers × len(base_specs).
    """
    tmpdir = tmp_path_factory.mktemp("lifecycle_concurrent_amplified")
    buf = JSONLBufferStore(base_path=tmpdir)

    work = [(w, sp) for w in range(n_writers) for sp in base_specs]
    barrier = threading.Barrier(len(work))

    def _writer(item):
        _w, (sid, kind, win) = item
        barrier.wait()
        buf.append(
            make_lifecycle_event(schema_id=sid, kind=kind, window_id=win)
        )

    with ThreadPoolExecutor(max_workers=min(len(work), 32)) as pool:
        for fut in as_completed([pool.submit(_writer, it) for it in work]):
            fut.result()

    decoded = _drain_buffer_lifecycle(buf)
    assert len(decoded) == len(work), (
        f"expected {len(work)} lifecycle events; got {len(decoded)}"
    )
    # Projection still equals direct fold, even at higher writer count.
    direct = reduce_events(decoded, strict=False)
    snap = snapshot_from_buffer(buf, strict=False)
    assert set(snap.keys()) == set(direct.keys())
    for sid, st_direct in direct.items():
        assert snap[sid].status == st_direct.status, sid

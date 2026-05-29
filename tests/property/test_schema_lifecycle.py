"""Hypothesis property tests for the schema-lifecycle reducer.

Locks down the invariants from TODO-RESEARCH §B before any production
lifecycle code lands. Tests target only the pure reducer in
`engram.consolidation.schema_lifecycle`.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.schema_lifecycle import (
    EventKind,
    LifecycleViolation,
    SchemaLifecycleEvent,
    SchemaState,
    SchemaStatus,
    reduce_events,
)


# A small pool of schema_ids/window_ids — small pool maximizes the chance
# of hitting interesting transitions (multiple events on the same schema).
_SCHEMA_IDS = ["s1", "s2", "s3"]
_WINDOW_IDS = ["w1", "w2", "w3", "w4"]


def _event_strategy() -> st.SearchStrategy[SchemaLifecycleEvent]:
    return st.builds(
        SchemaLifecycleEvent,
        schema_id=st.sampled_from(_SCHEMA_IDS),
        kind=st.sampled_from(list(EventKind)),
        window_id=st.sampled_from(_WINDOW_IDS),
        ts=st.integers(min_value=0, max_value=10_000),
    )


@given(events=st.lists(_event_strategy(), max_size=40))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_reduce_is_deterministic(events):
    """Invariant #2: reducing twice yields the same state."""
    a = reduce_events(events, strict=False)
    b = reduce_events(events, strict=False)
    assert a == b


@given(events=st.lists(_event_strategy(), max_size=40))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_lenient_reduce_respects_dag(events):
    """Invariant #1: every per-schema status path must follow the DAG.

    Reconstructed by replaying the events one at a time and confirming
    that the lenient reducer never lands on an out-of-DAG status.
    """
    allowed = {
        (SchemaStatus.INFERRED, SchemaStatus.PROMOTED),
        (SchemaStatus.INFERRED, SchemaStatus.DEPRECATED),
        (SchemaStatus.PROMOTED, SchemaStatus.DEPRECATED),
        (SchemaStatus.DEPRECATED, SchemaStatus.INFERRED),
    }
    prior: dict[str, SchemaState] = {}
    for ev in events:
        nxt = reduce_events([ev], strict=False, initial=prior)
        before = prior.get(ev.schema_id)
        after = nxt.get(ev.schema_id)
        if before is not None and after is not None and before.status != after.status:
            assert (before.status, after.status) in allowed, (
                f"DAG violation: {before.status} → {after.status} via {ev.kind}"
            )
        prior = nxt


@given(events=st.lists(_event_strategy(), max_size=40))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_initial_snapshot_equivalence(events):
    """`reduce(b, initial=reduce(a)) == reduce(a + b)`.

    Lets us periodically materialize the fold and resume from snapshot
    without changing semantics (invariant #2 corollary).
    """
    if len(events) < 2:
        return
    mid = len(events) // 2
    a, b = events[:mid], events[mid:]
    full = reduce_events(events, strict=False)
    snap = reduce_events(a, strict=False)
    resumed = reduce_events(b, strict=False, initial=snap)
    assert full == resumed


def test_strict_rejects_promote_from_deprecated():
    """Hard-coded smoke test for the most dangerous illegal edge."""
    evs = [
        SchemaLifecycleEvent("s", EventKind.CREATE, "w1"),
        SchemaLifecycleEvent("s", EventKind.PROMOTE, "w1"),
        SchemaLifecycleEvent("s", EventKind.DEPRECATE, "w2"),
    ]
    state = reduce_events(evs, strict=True)
    assert state["s"].status is SchemaStatus.DEPRECATED
    with pytest.raises(LifecycleViolation):
        reduce_events(
            [SchemaLifecycleEvent("s", EventKind.PROMOTE, "w3")],
            strict=True,
            initial=state,
        )


def test_recover_requires_fresh_window():
    """Invariant #5: RECOVER with stale window_id is rejected (strict)
    and no-ops (lenient)."""
    evs = [
        SchemaLifecycleEvent("s", EventKind.CREATE, "w1"),
        SchemaLifecycleEvent("s", EventKind.DEPRECATE, "w2"),
    ]
    state = reduce_events(evs, strict=True)
    # Stale window — same as last DEPRECATE.
    stale = SchemaLifecycleEvent("s", EventKind.RECOVER, "w2")
    with pytest.raises(LifecycleViolation):
        reduce_events([stale], strict=True, initial=state)
    lenient = reduce_events([stale], strict=False, initial=state)
    assert lenient["s"].status is SchemaStatus.DEPRECATED  # unchanged

    # Fresh window — accepted, lands back in INFERRED.
    fresh = SchemaLifecycleEvent("s", EventKind.RECOVER, "w3")
    out = reduce_events([fresh], strict=True, initial=state)
    assert out["s"].status is SchemaStatus.INFERRED
    assert out["s"].recover_count == 1


def test_bump_version_preserves_status_and_counts():
    """Invariant #3: promotion never invalidates stored properties; the
    version-bump event is the migration vehicle and must not perturb
    status or transition counts."""
    evs = [
        SchemaLifecycleEvent("s", EventKind.CREATE, "w1"),
        SchemaLifecycleEvent("s", EventKind.PROMOTE, "w1"),
        SchemaLifecycleEvent("s", EventKind.BUMP_VERSION, "w1"),
        SchemaLifecycleEvent("s", EventKind.BUMP_VERSION, "w1"),
    ]
    state = reduce_events(evs, strict=True)["s"]
    assert state.status is SchemaStatus.PROMOTED
    assert state.version == 3
    assert state.promote_count == 1
    assert state.deprecate_count == 0


def test_create_on_existing_strict_raises():
    evs = [SchemaLifecycleEvent("s", EventKind.CREATE, "w1")]
    state = reduce_events(evs, strict=True)
    with pytest.raises(LifecycleViolation):
        reduce_events(
            [SchemaLifecycleEvent("s", EventKind.CREATE, "w2")],
            strict=True,
            initial=state,
        )


def test_unknown_schema_strict_raises():
    with pytest.raises(LifecycleViolation):
        reduce_events(
            [SchemaLifecycleEvent("ghost", EventKind.PROMOTE, "w1")],
            strict=True,
        )

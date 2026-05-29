"""Integration property test: schema_decision policy × schema_lifecycle reducer.

Locks down that *every* event the policy ever emits is accepted by the
reducer in strict mode. This is the contract between the two halves of
the lifecycle pipeline — if it breaks, one of them is wrong about the
DAG.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.schema_decision import (
    EvidenceWindow,
    Thresholds,
    decide,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    LifecycleViolation,
    SchemaLifecycleEvent,
    SchemaState,
    SchemaStatus,
    reduce_events,
)


def _bootstrap_state(status: SchemaStatus, last_window_id: str | None) -> SchemaState:
    """Build a SchemaState by replaying CREATE + (optional) transitions
    through the strict reducer, so the state we feed `decide` is one
    the reducer itself could have produced."""
    events: list[SchemaLifecycleEvent] = [
        SchemaLifecycleEvent(schema_id="s", kind=EventKind.CREATE,
                             window_id="bootstrap", ts=0)
    ]
    if status == SchemaStatus.PROMOTED:
        events.append(SchemaLifecycleEvent(
            schema_id="s", kind=EventKind.PROMOTE,
            window_id=last_window_id, ts=1))
    elif status == SchemaStatus.DEPRECATED:
        events.append(SchemaLifecycleEvent(
            schema_id="s", kind=EventKind.DEPRECATE,
            window_id=last_window_id, ts=1))
    out = reduce_events(events, strict=True)
    return out["s"]


@given(
    status=st.sampled_from(list(SchemaStatus)),
    last_w=st.sampled_from(["w_old", "w_alpha"]),
    new_w=st.sampled_from(["w_old", "w_alpha", "w_fresh"]),
    supports=st.integers(min_value=0, max_value=8),
    contras=st.integers(min_value=0, max_value=8),
    th=st.builds(
        Thresholds,
        promote=st.integers(min_value=1, max_value=4),
        deprecate=st.integers(min_value=1, max_value=4),
        recover=st.integers(min_value=1, max_value=4),
    ),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_every_decided_event_is_accepted_by_strict_reducer(
    status, last_w, new_w, supports, contras, th,
):
    state = _bootstrap_state(status, last_w)
    ev = EvidenceWindow(window_id=new_w, supports=supports, contradictions=contras)
    kind = decide(state, ev, th)
    if kind is None:
        return  # nothing to feed the reducer

    next_event = SchemaLifecycleEvent(
        schema_id="s", kind=kind, window_id=new_w, ts=99,
    )
    # Strict reducer should never raise on a policy-emitted event.
    try:
        out = reduce_events([next_event], strict=True, initial={"s": state})
    except LifecycleViolation as e:
        raise AssertionError(
            f"policy emitted {kind} from state {state.status} (last_w={last_w}) "
            f"with new_w={new_w}, supports={supports}, contras={contras}, "
            f"thresholds={th}; reducer rejected: {e}"
        )
    # And the resulting state must reflect the transition.
    new_state = out["s"]
    if kind == EventKind.PROMOTE:
        assert new_state.status == SchemaStatus.PROMOTED
    elif kind == EventKind.DEPRECATE:
        assert new_state.status == SchemaStatus.DEPRECATED
    elif kind == EventKind.RECOVER:
        assert new_state.status == SchemaStatus.INFERRED

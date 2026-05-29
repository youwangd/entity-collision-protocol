"""Pure-Python schema lifecycle reducer.

This module is intentionally **storage-free**: it implements only the
event-fold that maps an ordered list of `SchemaLifecycleEvent` records to
a `{schema_id: SchemaState}` snapshot. No DB writes, no I/O, no clocks.

It exists ahead of the production lifecycle pipeline so we can:
  1. lock down the DAG and invariants (TODO-RESEARCH §B, closed
     2026-05-24; promoted to paper §A.4.17 audit trail) by property-fuzzing
     the reducer in isolation; and
  2. reuse the same reducer later as the canonical interpretation of the
     `CONSOLIDATION_SCHEMA_LIFECYCLE` event stream.

Status DAG (from TODO-RESEARCH §B invariant #1, closed 2026-05-24):
    inferred  → promoted
    inferred  → deprecated
    promoted  → deprecated
    deprecated → inferred   (recovery; only via a NEW evidence window —
                              represented by a RECOVER event referencing a
                              fresh window_id distinct from the one that
                              caused the prior demotion)

No `deprecated → promoted` direct edge. No `promoted → inferred`. No
self-loops cause a status change (they no-op deterministically).

Determinism guarantees (invariant #2): the reducer is a pure function of
the event list. No wall-clock, no RNG, no env reads. Reduce-twice
idempotence: reducing `events + events` is **not** the same as reducing
`events` (events are not idempotent — each one carries semantic weight),
BUT `reduce(events)` invoked twice on the same input always returns
equal output. That weaker idempotence is what we test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping


class SchemaStatus(str, Enum):
    INFERRED = "inferred"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"


# Allowed transitions (invariant #1). Encoded as a frozenset of (from, to)
# pairs so the reducer never silently accepts an out-of-DAG move.
_ALLOWED_TRANSITIONS: frozenset[tuple[SchemaStatus, SchemaStatus]] = frozenset({
    (SchemaStatus.INFERRED, SchemaStatus.PROMOTED),
    (SchemaStatus.INFERRED, SchemaStatus.DEPRECATED),
    (SchemaStatus.PROMOTED, SchemaStatus.DEPRECATED),
    (SchemaStatus.DEPRECATED, SchemaStatus.INFERRED),  # recovery only
})


class EventKind(str, Enum):
    CREATE = "create"          # schema first observed → status=inferred, version=1
    PROMOTE = "promote"        # inferred → promoted
    DEPRECATE = "deprecate"    # inferred|promoted → deprecated
    RECOVER = "recover"        # deprecated → inferred (requires fresh window_id)
    BUMP_VERSION = "bump_version"  # version+1, status unchanged (back-compat migration)


@dataclass(frozen=True)
class SchemaLifecycleEvent:
    """A single immutable lifecycle decision.

    Attributes:
      schema_id: stable identifier across versions.
      kind: which transition this event represents.
      window_id: the evidence window that justified the decision. RECOVER
        events MUST carry a window_id distinct from the window_id of the
        DEPRECATE that immediately preceded them — otherwise the reducer
        rejects the recovery (invariant #5).
      ts: monotonic ordering key. The reducer trusts the input list order
        and uses ts only as a tiebreaker / sanity field.
    """
    schema_id: str
    kind: EventKind
    window_id: str | None = None
    ts: int = 0
    # Identity of the agent / consolidator that emitted this event.
    # Optional for back-compat (default-None preserves all pre-quorum
    # call sites). Used only when `reduce_events(deprecate_quorum_k > 1)`
    # — in which case DEPRECATE events without an emitter_id are rejected
    # under strict=True (mitigation against the §6.10 single-actor
    # global-suppression channel).
    emitter_id: str | None = None


@dataclass(frozen=True)
class SchemaState:
    schema_id: str
    status: SchemaStatus
    version: int
    # Counts let downstream consumers (and tests) check fold integrity
    # without re-walking the event log.
    promote_count: int = 0
    deprecate_count: int = 0
    recover_count: int = 0
    last_window_id: str | None = None  # window of the most recent transition
    # Set of distinct emitter_ids that have voted to DEPRECATE this schema
    # since the last status change. Cleared on any other transition
    # (CREATE / PROMOTE / RECOVER) and on the firing of DEPRECATE itself.
    # Always empty when `deprecate_quorum_k <= 1` (the legacy single-emitter
    # path); only populated under explicit quorum gating.
    pending_deprecate_emitters: frozenset[str] = frozenset()


class LifecycleViolation(ValueError):
    """Raised when an event would violate the lifecycle DAG."""


def _apply(state: SchemaState | None, ev: SchemaLifecycleEvent,
           strict: bool, *, deprecate_quorum_k: int = 1) -> SchemaState | None:
    """Apply one event to one schema's state. Returns the new state.

    strict=True raises LifecycleViolation on illegal moves.
    strict=False drops them silently (used for replay-of-untrusted-log
    where we'd rather not crash on garbage; production should be strict).

    deprecate_quorum_k (>=1): number of *distinct* `emitter_id`s required
    on DEPRECATE events before the schema actually transitions to
    DEPRECATED. k=1 preserves legacy behaviour exactly. k>1 is the
    §6.10 mitigation: a single malicious actor can no longer
    unilaterally suppress a schema; they must collude with k-1 other
    distinct emitters.
    """
    if ev.kind == EventKind.CREATE:
        if state is not None:
            if strict:
                raise LifecycleViolation(
                    f"CREATE on existing schema {ev.schema_id}"
                )
            return state
        return SchemaState(
            schema_id=ev.schema_id,
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id=ev.window_id,
        )

    if state is None:
        if strict:
            raise LifecycleViolation(
                f"{ev.kind.value} on unknown schema {ev.schema_id}"
            )
        return None

    if ev.kind == EventKind.BUMP_VERSION:
        # Version bumps preserve status (invariant #3: promotion never
        # invalidates stored properties; version bump is the migration
        # vehicle). last_window_id moves; counts unchanged.
        return replace(state, version=state.version + 1,
                       last_window_id=ev.window_id or state.last_window_id)

    target: SchemaStatus
    if ev.kind == EventKind.PROMOTE:
        target = SchemaStatus.PROMOTED
    elif ev.kind == EventKind.DEPRECATE:
        target = SchemaStatus.DEPRECATED
    elif ev.kind == EventKind.RECOVER:
        target = SchemaStatus.INFERRED
    else:  # pragma: no cover - exhaustive
        raise LifecycleViolation(f"unknown event kind {ev.kind!r}")

    if (state.status, target) not in _ALLOWED_TRANSITIONS:
        if strict:
            raise LifecycleViolation(
                f"illegal transition {state.status.value} → {target.value} "
                f"for schema {ev.schema_id}"
            )
        return state

    # Invariant #5: RECOVER requires a window_id distinct from the
    # last_window_id that produced the DEPRECATE.
    if ev.kind == EventKind.RECOVER:
        if ev.window_id is None or ev.window_id == state.last_window_id:
            if strict:
                raise LifecycleViolation(
                    f"RECOVER for {ev.schema_id} requires a fresh window_id "
                    f"distinct from {state.last_window_id!r}"
                )
            return state

    # §6.10 quorum gate: when k>1, accumulate distinct emitter votes on
    # DEPRECATE rather than firing immediately. Other transitions clear
    # the pending set (a PROMOTE / RECOVER invalidates earlier dissent).
    if ev.kind == EventKind.DEPRECATE and deprecate_quorum_k > 1:
        if ev.emitter_id is None:
            if strict:
                raise LifecycleViolation(
                    f"DEPRECATE under quorum_k={deprecate_quorum_k} requires "
                    f"emitter_id (schema {ev.schema_id})"
                )
            return state
        votes = state.pending_deprecate_emitters | {ev.emitter_id}
        if len(votes) < deprecate_quorum_k:
            # Hold: record the vote, do NOT transition.
            return replace(state, pending_deprecate_emitters=votes)
        # Quorum reached → fire the transition and clear the ballot.
        return replace(
            state,
            status=SchemaStatus.DEPRECATED,
            deprecate_count=state.deprecate_count + 1,
            last_window_id=ev.window_id or state.last_window_id,
            pending_deprecate_emitters=frozenset(),
        )

    counts = {
        EventKind.PROMOTE: ("promote_count", state.promote_count + 1),
        EventKind.DEPRECATE: ("deprecate_count", state.deprecate_count + 1),
        EventKind.RECOVER: ("recover_count", state.recover_count + 1),
    }
    field_name, new_count = counts[ev.kind]
    return replace(
        state,
        status=target,
        last_window_id=ev.window_id or state.last_window_id,
        # Any non-DEPRECATE transition clears any in-flight DEPRECATE
        # ballot — promotion or recovery are evidence the schema is alive.
        pending_deprecate_emitters=frozenset(),
        **{field_name: new_count},
    )


def reduce_events(
    events: Iterable[SchemaLifecycleEvent],
    *,
    strict: bool = True,
    initial: Mapping[str, SchemaState] | None = None,
    deprecate_quorum_k: int = 1,
) -> dict[str, SchemaState]:
    """Fold an ordered event sequence into the current `{schema_id: state}`.

    Pure function: same input → same output. No I/O. No clocks. No RNG.

    `initial` lets you resume from a prior snapshot (e.g. periodic
    materialization of the fold for fast cold-start) without losing
    determinism — the result is identical to `reduce_events(prior_events
    + events)` provided `initial` was itself produced by the reducer.

    `deprecate_quorum_k` (default 1, legacy behaviour): require k
    distinct `emitter_id`s before a DEPRECATE actually fires. Mitigates
    §6.10 (a single malicious actor that learns a schema_id can no
    longer unilaterally suppress the schema). Must be >=1.
    """
    if deprecate_quorum_k < 1:
        raise ValueError("deprecate_quorum_k must be >= 1")
    state: dict[str, SchemaState] = dict(initial) if initial else {}
    for ev in events:
        new = _apply(state.get(ev.schema_id), ev, strict=strict,
                     deprecate_quorum_k=deprecate_quorum_k)
        if new is None:
            state.pop(ev.schema_id, None)
        else:
            state[ev.schema_id] = new
    return state


__all__ = [
    "EventKind",
    "LifecycleViolation",
    "SchemaLifecycleEvent",
    "SchemaState",
    "SchemaStatus",
    "reduce_events",
]

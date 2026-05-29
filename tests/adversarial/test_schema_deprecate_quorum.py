"""§6.10 mitigation: quorum-gated DEPRECATE.

The §6.10 threat (paper/60_threats.md) is that a single actor that
learns a schema_id can append `CREATE+DEPRECATE` and globally suppress
the schema. The mitigation lands as an opt-in `deprecate_quorum_k`
parameter on the lifecycle reducer: DEPRECATE events accumulate distinct
`emitter_id` votes and fire only once k distinct emitters have voted.

This file pins the prototype's behaviour. It is intentionally
reducer-only — no DB integration yet. The aim is to (a) lock the
semantics down so the production wiring can land later without
regressions, and (b) give the paper a concrete defended mitigation.
"""
from __future__ import annotations

import pytest

from engram.consolidation.schema_lifecycle import (
    EventKind,
    LifecycleViolation,
    SchemaLifecycleEvent,
    SchemaStatus,
    reduce_events,
)


def _ev(schema_id, kind, *, emitter=None, window=None, ts=0):
    return SchemaLifecycleEvent(
        schema_id=schema_id,
        kind=kind,
        window_id=window,
        ts=ts,
        emitter_id=emitter,
    )


# ─── Q-1: legacy k=1 path is byte-identical to pre-quorum behaviour. ────
def test_q1_default_k1_is_legacy_path():
    """k=1 (default) preserves the §6.10 attack as the documented baseline.

    This is intentional: changing the default would silently break every
    existing single-emitter consolidation pipeline. The mitigation is
    *available*, not *mandatory*.
    """
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w0", emitter="alice"),
    ]
    snap = reduce_events(events)  # k=1 implicit
    assert snap["sc-1"].status is SchemaStatus.DEPRECATED
    assert snap["sc-1"].deprecate_count == 1
    assert snap["sc-1"].pending_deprecate_emitters == frozenset()


# ─── Q-2: k=2 holds a single attacker's DEPRECATE — schema stays live. ──
def test_q2_single_emitter_cannot_unilaterally_deprecate():
    """The §6.10 attack: malicious Mallory votes DEPRECATE alone.

    Under k=2, schema must remain INFERRED. The vote is recorded
    (audit trail) but does not fire.
    """
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="mallory"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2)
    s = snap["sc-1"]
    assert s.status is SchemaStatus.INFERRED, "single emitter must NOT fire"
    assert s.deprecate_count == 0
    assert s.pending_deprecate_emitters == frozenset({"mallory"})


# ─── Q-3: k=2, two distinct emitters → fires; ballot clears. ────────────
def test_q3_two_distinct_emitters_fire_quorum():
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="alice"),
        _ev("sc-1", EventKind.DEPRECATE, window="w2", emitter="bob"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2)
    s = snap["sc-1"]
    assert s.status is SchemaStatus.DEPRECATED
    assert s.deprecate_count == 1, "quorum fires once, not k times"
    assert s.last_window_id == "w2"
    assert s.pending_deprecate_emitters == frozenset()


# ─── Q-4: same emitter voting twice does NOT count as quorum. ───────────
def test_q4_repeated_votes_from_same_emitter_do_not_satisfy_quorum():
    """Sybil-resistance is not in scope (the reducer trusts emitter_id),
    but at minimum a single `emitter_id` re-voting must not satisfy k>1.
    """
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="mallory"),
        _ev("sc-1", EventKind.DEPRECATE, window="w2", emitter="mallory"),
        _ev("sc-1", EventKind.DEPRECATE, window="w3", emitter="mallory"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2)
    s = snap["sc-1"]
    assert s.status is SchemaStatus.INFERRED
    assert s.deprecate_count == 0
    assert s.pending_deprecate_emitters == frozenset({"mallory"})


# ─── Q-5: PROMOTE clears a pending DEPRECATE ballot. ────────────────────
def test_q5_promote_clears_pending_deprecate_ballot():
    """If the schema gets PROMOTED while a partial DEPRECATE ballot is
    open, the ballot must clear. Otherwise an attacker who voted long
    ago can collude with one fresh emitter to suppress a now-promoted
    schema.
    """
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="alice"),
        # Promotion supervenes — the schema is clearly alive.
        _ev("sc-1", EventKind.PROMOTE, window="w2", emitter="system"),
        # Bob shows up later. Ballot should NOT carry alice's old vote.
        _ev("sc-1", EventKind.DEPRECATE, window="w3", emitter="bob"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2)
    s = snap["sc-1"]
    assert s.status is SchemaStatus.PROMOTED, "still promoted; one vote insufficient"
    assert s.pending_deprecate_emitters == frozenset({"bob"})
    assert s.deprecate_count == 0


# ─── Q-6: RECOVER (deprecated→inferred) clears any future ballot. ───────
def test_q6_recover_clears_pending_ballot():
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="alice"),
        _ev("sc-1", EventKind.DEPRECATE, window="w2", emitter="bob"),  # fires
        _ev("sc-1", EventKind.RECOVER, window="w3"),                   # back to inferred
        _ev("sc-1", EventKind.DEPRECATE, window="w4", emitter="alice"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2)
    s = snap["sc-1"]
    assert s.status is SchemaStatus.INFERRED, "ballot must restart fresh"
    assert s.deprecate_count == 1, "old quorum already fired once"
    assert s.pending_deprecate_emitters == frozenset({"alice"})


# ─── Q-7: missing emitter_id under k>1 is rejected (strict). ────────────
def test_q7_missing_emitter_id_rejected_under_quorum():
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter=None),
    ]
    with pytest.raises(LifecycleViolation, match="emitter_id"):
        reduce_events(events, deprecate_quorum_k=2, strict=True)


# ─── Q-8: missing emitter_id under k>1 is dropped (non-strict). ─────────
def test_q8_missing_emitter_id_dropped_in_non_strict():
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter=None),
    ]
    snap = reduce_events(events, deprecate_quorum_k=2, strict=False)
    assert snap["sc-1"].status is SchemaStatus.INFERRED
    assert snap["sc-1"].pending_deprecate_emitters == frozenset()


# ─── Q-9: positive control — under k=1, the §6.10 attack still works. ──
def test_q9_positive_control_attack_succeeds_at_k1():
    """If this fails, we accidentally changed the default and broke
    back-compat. This is the canary for Q-2 being a real test.
    """
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="mallory"),
    ]
    snap = reduce_events(events, deprecate_quorum_k=1)
    assert snap["sc-1"].status is SchemaStatus.DEPRECATED


# ─── Q-10: parametric k ∈ {2,3,5} — fires exactly at k distinct votes. ─
@pytest.mark.parametrize("k", [2, 3, 5])
def test_q10_fires_exactly_at_k_distinct_votes(k):
    events = [_ev("sc-1", EventKind.CREATE, window="w0")]
    emitters = [f"e{i}" for i in range(k)]
    for i, em in enumerate(emitters):
        events.append(_ev("sc-1", EventKind.DEPRECATE, window=f"w{i+1}",
                          emitter=em))
        snap = reduce_events(events, deprecate_quorum_k=k)
        s = snap["sc-1"]
        if i + 1 < k:
            assert s.status is SchemaStatus.INFERRED, (
                f"fired early at vote {i+1}/{k}"
            )
            assert s.pending_deprecate_emitters == frozenset(emitters[:i+1])
        else:
            assert s.status is SchemaStatus.DEPRECATED
            assert s.deprecate_count == 1
            assert s.pending_deprecate_emitters == frozenset()


# ─── Q-11: invalid k=0 rejected at the API boundary. ────────────────────
def test_q11_invalid_k_rejected():
    with pytest.raises(ValueError):
        reduce_events([], deprecate_quorum_k=0)


# ─── Q-12: determinism — quorum fold is a pure function. ────────────────
def test_q12_quorum_fold_is_deterministic():
    events = [
        _ev("sc-1", EventKind.CREATE, window="w0"),
        _ev("sc-1", EventKind.DEPRECATE, window="w1", emitter="alice"),
        _ev("sc-1", EventKind.DEPRECATE, window="w2", emitter="bob"),
    ]
    a = reduce_events(events, deprecate_quorum_k=2)
    b = reduce_events(events, deprecate_quorum_k=2)
    assert a == b

"""Hypothesis property tests for the schema-lifecycle decision policy.

These pin down the policy invariants (P1-P5) from
`engram.consolidation.schema_decision`. The reducer guards mechanics;
this guards *which* events ever fire.
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
    SchemaState,
    SchemaStatus,
)


_WINDOW_IDS = ["w1", "w2", "w3", "w4", None]


def _state_strategy() -> st.SearchStrategy[SchemaState]:
    return st.builds(
        SchemaState,
        schema_id=st.just("s1"),
        status=st.sampled_from(list(SchemaStatus)),
        version=st.integers(min_value=1, max_value=5),
        promote_count=st.integers(min_value=0, max_value=3),
        deprecate_count=st.integers(min_value=0, max_value=3),
        recover_count=st.integers(min_value=0, max_value=3),
        last_window_id=st.sampled_from(_WINDOW_IDS),
    )


def _evidence_strategy() -> st.SearchStrategy[EvidenceWindow]:
    return st.builds(
        EvidenceWindow,
        window_id=st.sampled_from([w for w in _WINDOW_IDS if w is not None]),
        supports=st.integers(min_value=0, max_value=10),
        contradictions=st.integers(min_value=0, max_value=10),
    )


def _thresholds_strategy() -> st.SearchStrategy[Thresholds]:
    return st.builds(
        Thresholds,
        promote=st.integers(min_value=1, max_value=5),
        deprecate=st.integers(min_value=1, max_value=5),
        recover=st.integers(min_value=1, max_value=5),
    )


@given(state=_state_strategy(), ev=_evidence_strategy(), th=_thresholds_strategy())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_p1_promote_only_from_inferred(state, ev, th):
    """P1: PROMOTE only fires from INFERRED."""
    if decide(state, ev, th) == EventKind.PROMOTE:
        assert state.status == SchemaStatus.INFERRED


@given(state=_state_strategy(), ev=_evidence_strategy(), th=_thresholds_strategy())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_p2_deprecate_only_from_inferred_or_promoted(state, ev, th):
    """P2: DEPRECATE only fires from INFERRED or PROMOTED."""
    if decide(state, ev, th) == EventKind.DEPRECATE:
        assert state.status in (SchemaStatus.INFERRED, SchemaStatus.PROMOTED)


@given(state=_state_strategy(), ev=_evidence_strategy(), th=_thresholds_strategy())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_p3_recover_only_from_deprecated_with_fresh_window(state, ev, th):
    """P3: RECOVER fires only from DEPRECATED with a fresh window_id."""
    if decide(state, ev, th) == EventKind.RECOVER:
        assert state.status == SchemaStatus.DEPRECATED
        assert ev.window_id != state.last_window_id


@given(state=_state_strategy(), th=_thresholds_strategy(),
       window_id=st.sampled_from([w for w in _WINDOW_IDS if w is not None]))
@settings(max_examples=200)
def test_p4_no_evidence_no_decision(state, th, window_id):
    """P4: zero evidence ⇒ no transition, regardless of state/thresholds."""
    ev = EvidenceWindow(window_id=window_id, supports=0, contradictions=0)
    assert decide(state, ev, th) is None


@given(state=_state_strategy(), ev=_evidence_strategy(), th=_thresholds_strategy(),
       extra_supports=st.integers(min_value=1, max_value=5))
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_p5a_promote_monotone_in_supports(state, ev, th, extra_supports):
    """P5: if a PROMOTE fires, more supports keep it firing (or escalate to
    DEPRECATE if contradictions would also dominate, which they wouldn't —
    contradictions weren't increased). It cannot revert to None."""
    base = decide(state, ev, th)
    if base == EventKind.PROMOTE:
        bigger = EvidenceWindow(
            window_id=ev.window_id,
            supports=ev.supports + extra_supports,
            contradictions=ev.contradictions,
        )
        # Adding supports cannot trigger DEPRECATE (which only depends on
        # contradictions). Must remain PROMOTE.
        assert decide(state, bigger, th) == EventKind.PROMOTE


@given(state=_state_strategy(), ev=_evidence_strategy(), th=_thresholds_strategy(),
       extra_contras=st.integers(min_value=1, max_value=5))
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_p5b_deprecate_monotone_in_contradictions(state, ev, th, extra_contras):
    """P5: if a DEPRECATE fires, adding contradictions keeps it firing."""
    base = decide(state, ev, th)
    if base == EventKind.DEPRECATE:
        bigger = EvidenceWindow(
            window_id=ev.window_id,
            supports=ev.supports,
            contradictions=ev.contradictions + extra_contras,
        )
        assert decide(state, bigger, th) == EventKind.DEPRECATE


def test_concrete_inferred_promotes_at_threshold():
    state = SchemaState(schema_id="s", status=SchemaStatus.INFERRED, version=1)
    th = Thresholds(promote=3, deprecate=2, recover=3)
    assert decide(state, EvidenceWindow("w", supports=2), th) is None
    assert decide(state, EvidenceWindow("w", supports=3), th) == EventKind.PROMOTE


def test_concrete_promoted_deprecates_on_contradictions():
    state = SchemaState(schema_id="s", status=SchemaStatus.PROMOTED, version=1)
    th = Thresholds(promote=3, deprecate=2, recover=3)
    assert decide(state, EvidenceWindow("w", contradictions=1), th) is None
    assert decide(state, EvidenceWindow("w", contradictions=2), th) == EventKind.DEPRECATE


def test_concrete_deprecated_requires_fresh_window_for_recover():
    state = SchemaState(
        schema_id="s",
        status=SchemaStatus.DEPRECATED,
        version=1,
        last_window_id="w_old",
    )
    th = Thresholds(promote=3, deprecate=2, recover=3)
    # Same window — no recover even at threshold.
    assert decide(state, EvidenceWindow("w_old", supports=5), th) is None
    # Fresh window with enough supports — recover.
    assert decide(state, EvidenceWindow("w_new", supports=3), th) == EventKind.RECOVER
    # Fresh window but below threshold — no recover.
    assert decide(state, EvidenceWindow("w_new", supports=2), th) is None


def test_concrete_inferred_contradiction_beats_simultaneous_support():
    """When both thresholds cross in the same window, contradictions win
    (conservative). Documents the design choice from the policy module."""
    state = SchemaState(schema_id="s", status=SchemaStatus.INFERRED, version=1)
    th = Thresholds(promote=3, deprecate=2, recover=3)
    ev = EvidenceWindow("w", supports=10, contradictions=10)
    assert decide(state, ev, th) == EventKind.DEPRECATE

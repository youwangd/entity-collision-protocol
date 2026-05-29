"""Hypothesis property tests for the schema-family-aware decision policy.

Locks invariants G1-G6 of `schema_family_decision.decide_with_family`.
The wrapper must be regression-safe (G1, G2), monotone in `share` for
fixed evidence (G3, G4), exact for share=1.0 (G5), and reject bad
inputs (G6).
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.schema_decision import (
    EvidenceWindow,
    Thresholds,
    decide,
)
from engram.consolidation.schema_family_decision import decide_with_family
from engram.consolidation.schema_lifecycle import SchemaState, SchemaStatus


_WINDOW_IDS = ["w1", "w2", "w3"]


def _state_strategy() -> st.SearchStrategy[SchemaState]:
    return st.builds(
        SchemaState,
        schema_id=st.just("s1"),
        status=st.sampled_from(list(SchemaStatus)),
        version=st.integers(min_value=1, max_value=3),
        promote_count=st.integers(min_value=0, max_value=2),
        deprecate_count=st.integers(min_value=0, max_value=2),
        recover_count=st.integers(min_value=0, max_value=2),
        last_window_id=st.sampled_from([*_WINDOW_IDS, None]),
    )


def _ev_strategy() -> st.SearchStrategy[EvidenceWindow]:
    return st.builds(
        EvidenceWindow,
        window_id=st.sampled_from(_WINDOW_IDS),
        supports=st.integers(min_value=0, max_value=8),
        contradictions=st.integers(min_value=0, max_value=8),
    )


def _siblings_strategy() -> st.SearchStrategy[list[EvidenceWindow]]:
    return st.lists(_ev_strategy(), min_size=0, max_size=4)


def _thresholds_strategy() -> st.SearchStrategy[Thresholds]:
    return st.builds(
        Thresholds,
        promote=st.integers(min_value=1, max_value=4),
        deprecate=st.integers(min_value=1, max_value=4),
        recover=st.integers(min_value=1, max_value=4),
    )


# ─── G1: share=0 is byte-identical to decide() ────────────────────────


@given(
    state=_state_strategy(),
    own=_ev_strategy(),
    sibs=_siblings_strategy(),
    th=_thresholds_strategy(),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_g1_share_zero_is_regression_safe(state, own, sibs, th):
    assert decide_with_family(state, own, sibs, th, share=0.0) == decide(
        state, own, th
    )


# ─── G2: empty siblings ⇒ identical for any share ─────────────────────


@given(
    state=_state_strategy(),
    own=_ev_strategy(),
    th=_thresholds_strategy(),
    share=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_g2_empty_siblings_identical(state, own, th, share):
    assert decide_with_family(state, own, [], th, share=share) == decide(
        state, own, th
    )


# ─── G3 / G4: monotonicity in share ───────────────────────────────────
#
# We verify: bumping `share` cannot *remove* the ability to PROMOTE
# (when siblings carry only supports), nor remove DEPRECATE (when siblings
# carry only contradictions). We test by constructing the effective
# evidence at two share levels and asserting the threshold-crossing is
# monotone.


@given(
    state=_state_strategy(),
    own=_ev_strategy(),
    sib_supports=st.lists(
        st.integers(min_value=0, max_value=10), min_size=0, max_size=4
    ),
    th=_thresholds_strategy(),
    share_lo=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    share_hi=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_g3_more_share_more_supports(
    state, own, sib_supports, th, share_lo, share_hi
):
    # Sibling evidence: only supports, no contradictions.
    sibs = [
        EvidenceWindow(window_id=own.window_id, supports=s, contradictions=0)
        for s in sib_supports
    ]
    # The effective `eff_supports` must be monotone non-decreasing in share.
    import math

    s_sum = sum(sib_supports)
    eff_lo = own.supports + math.floor(share_lo * s_sum)
    eff_hi = own.supports + math.floor(share_hi * s_sum)
    assert eff_hi >= eff_lo

    # And: if share_lo already triggered PROMOTE from INFERRED, share_hi
    # cannot un-trigger it (since contradictions are zero).
    if state.status == SchemaStatus.INFERRED and own.contradictions == 0:
        from engram.consolidation.schema_lifecycle import EventKind

        d_lo = decide_with_family(state, own, sibs, th, share=share_lo)
        d_hi = decide_with_family(state, own, sibs, th, share=share_hi)
        if d_lo == EventKind.PROMOTE:
            assert d_hi == EventKind.PROMOTE


@given(
    state=_state_strategy(),
    own=_ev_strategy(),
    sib_contras=st.lists(
        st.integers(min_value=0, max_value=10), min_size=0, max_size=4
    ),
    th=_thresholds_strategy(),
    share_lo=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    share_hi=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_g4_more_share_more_contradictions(
    state, own, sib_contras, th, share_lo, share_hi
):
    from engram.consolidation.schema_lifecycle import EventKind

    sibs = [
        EvidenceWindow(window_id=own.window_id, supports=0, contradictions=c)
        for c in sib_contras
    ]
    # If share_lo already DEPRECATEd a non-DEPRECATED schema, share_hi
    # must too (more contradictions can only reinforce).
    if state.status != SchemaStatus.DEPRECATED:
        d_lo = decide_with_family(state, own, sibs, th, share=share_lo)
        d_hi = decide_with_family(state, own, sibs, th, share=share_hi)
        if d_lo == EventKind.DEPRECATE:
            assert d_hi == EventKind.DEPRECATE


# ─── G5: share=1.0 with sibling==own == doubling own ──────────────────


@given(
    state=_state_strategy(),
    own=_ev_strategy(),
    th=_thresholds_strategy(),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_g5_share_one_doubles_with_clone(state, own, th):
    clone = EvidenceWindow(
        window_id=own.window_id,
        supports=own.supports,
        contradictions=own.contradictions,
    )
    doubled = EvidenceWindow(
        window_id=own.window_id,
        supports=own.supports * 2,
        contradictions=own.contradictions * 2,
    )
    assert decide_with_family(state, own, [clone], th, share=1.0) == decide(
        state, doubled, th
    )


# ─── G6: bad share rejected ───────────────────────────────────────────


@pytest.mark.parametrize("bad", [-0.1, 1.1, -1.0, 2.0])
def test_g6_bad_share_rejected(bad):
    state = SchemaState(schema_id="s1", status=SchemaStatus.INFERRED, version=1)
    own = EvidenceWindow(window_id="w1", supports=1)
    with pytest.raises(ValueError):
        decide_with_family(state, own, [], share=bad)

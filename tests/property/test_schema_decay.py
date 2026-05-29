"""Hypothesis property tests for the multi-window evidence decay aggregator.

Locks the design choices D1-D6 from `engram.consolidation.schema_decay`.
Pairs with the existing single-window decision tests
(`test_schema_decision.py`) — together they cover the full
schema-lifecycle policy stack.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.schema_decay import DecayPolicy, aggregate
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


_WINDOW_IDS = ["w0", "w1", "w2", "w3", "w4", "w5"]


def _evidence_strategy() -> st.SearchStrategy[EvidenceWindow]:
    return st.builds(
        EvidenceWindow,
        window_id=st.sampled_from(_WINDOW_IDS),
        supports=st.integers(min_value=0, max_value=10),
        contradictions=st.integers(min_value=0, max_value=10),
    )


@given(history=st.lists(_evidence_strategy(), min_size=1, max_size=12))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_d1_factor_one_is_cumulative_sum(history):
    """D1: factor=1.0 sums all in-horizon windows with weight 1."""
    policy = DecayPolicy(factor=1.0, horizon=len(history))
    out = aggregate(history, policy)
    assert out.supports == sum(e.supports for e in history)
    assert out.contradictions == sum(e.contradictions for e in history)


@given(history=st.lists(_evidence_strategy(), min_size=1, max_size=12))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_d2_factor_zero_keeps_only_newest(history):
    """D2: factor=0.0 ⇒ aggregate equals the newest window verbatim."""
    policy = DecayPolicy(factor=0.0, horizon=len(history))
    out = aggregate(history, policy)
    newest = history[-1]
    assert out.supports == newest.supports
    assert out.contradictions == newest.contradictions
    assert out.window_id == newest.window_id


@given(
    history=st.lists(_evidence_strategy(), min_size=1, max_size=8),
    bump_idx=st.integers(min_value=0, max_value=7),
    bump=st.integers(min_value=1, max_value=5),
    factor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_d3_supports_monotone_per_window(history, bump_idx, bump, factor):
    """D3: increasing one window's supports cannot decrease aggregate supports.

    Only checks indices that actually fall inside the horizon; bumps
    outside the suffix are no-ops by D4 (which has its own test).
    """
    policy = DecayPolicy(factor=factor, horizon=len(history))
    bump_idx = bump_idx % len(history)
    base = aggregate(history, policy)
    bumped = list(history)
    e = bumped[bump_idx]
    bumped[bump_idx] = EvidenceWindow(
        window_id=e.window_id,
        supports=e.supports + bump,
        contradictions=e.contradictions,
    )
    after = aggregate(bumped, policy)
    assert after.supports >= base.supports
    assert after.contradictions == base.contradictions


@given(
    history=st.lists(_evidence_strategy(), min_size=1, max_size=4),
    extra=st.lists(_evidence_strategy(), min_size=1, max_size=4),
    factor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_d4_horizon_truncation_is_suffix_only(history, extra, factor):
    """D4: prepending windows beyond the horizon is a no-op."""
    horizon = len(history)
    policy = DecayPolicy(factor=factor, horizon=horizon)
    base = aggregate(history, policy)
    # Prepend `extra` so the original history is still the suffix.
    prepended = list(extra) + list(history)
    after = aggregate(prepended, policy)
    assert after == base


@given(history=st.lists(_evidence_strategy(), min_size=1, max_size=8),
       factor=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
       horizon=st.integers(min_value=1, max_value=12))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_d5_newest_window_id_preserved(history, factor, horizon):
    """D5: aggregate.window_id is always the newest input's window_id."""
    policy = DecayPolicy(factor=factor, horizon=horizon)
    out = aggregate(history, policy)
    assert out.window_id == history[-1].window_id


def test_d6_empty_history_raises():
    """D6: empty history is a programming error."""
    with pytest.raises(ValueError):
        aggregate([], DecayPolicy())


def test_decay_policy_validation():
    """DecayPolicy enforces factor ∈ [0,1] and horizon >= 1."""
    with pytest.raises(ValueError):
        DecayPolicy(factor=-0.1)
    with pytest.raises(ValueError):
        DecayPolicy(factor=1.5)
    with pytest.raises(ValueError):
        DecayPolicy(horizon=0)


def test_concrete_decay_compose_with_decide():
    """Smoke test: aggregate → decide() round-trip on a realistic story.

    A schema sees 1 support over 5 windows. Cumulative (factor=1) crosses
    promote=3; full-decay (factor=0) does not.
    """
    state = SchemaState(schema_id="s", status=SchemaStatus.INFERRED, version=1)
    th = Thresholds(promote=3, deprecate=2, recover=3)
    history = [
        EvidenceWindow("w0", supports=1),
        EvidenceWindow("w1", supports=1),
        EvidenceWindow("w2", supports=1),
        EvidenceWindow("w3", supports=0),
        EvidenceWindow("w4", supports=0),
    ]
    cum = aggregate(history, DecayPolicy(factor=1.0, horizon=5))
    assert cum.supports == 3
    assert decide(state, cum, th) == EventKind.PROMOTE

    fast = aggregate(history, DecayPolicy(factor=0.0, horizon=5))
    assert fast.supports == 0
    assert decide(state, fast, th) is None


def test_concrete_decay_floors_to_int():
    """Weighted sum 1*1 + 1*0.5 = 1.5 floors to 1 (not 2)."""
    history = [
        EvidenceWindow("w0", supports=1),
        EvidenceWindow("w1", supports=1),
    ]
    out = aggregate(history, DecayPolicy(factor=0.5, horizon=2))
    # newest weight 1, prior weight 0.5 → 1 + 0.5 = 1.5 → floor 1.
    assert out.supports == 1


def test_concrete_decay_horizon_smaller_than_history():
    """horizon=2 keeps only last 2; ancient supports drop entirely."""
    history = [
        EvidenceWindow("ancient", supports=99),
        EvidenceWindow("w0", supports=1),
        EvidenceWindow("w1", supports=2),
    ]
    out = aggregate(history, DecayPolicy(factor=1.0, horizon=2))
    assert out.supports == 3  # ancient dropped
    assert out.window_id == "w1"

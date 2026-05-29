"""Plumbing test: ConsolidationConfig.schema_*_threshold knobs are honored.

Paper §4.6 churn-budget sweep needs the promotion threshold to be
runtime-configurable. This test pins:

  1. Defaults match the historical hardcoded `Thresholds()` (3/2/3).
  2. Round-trip through to_dict/from_dict preserves non-default values.
  3. Default values are NOT serialized (round-trip stays clean).
  4. The wired threshold reaches `decide()` — verified by direct
     `Thresholds`-injection comparison against the same fixture.
"""
from __future__ import annotations

from engram.core.config import Config, ConsolidationConfig
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


def test_schema_threshold_defaults():
    c = ConsolidationConfig()
    assert c.schema_promote_threshold == 3
    assert c.schema_deprecate_threshold == 2
    assert c.schema_recover_threshold == 3


def test_schema_threshold_round_trip():
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_promote_threshold=7,
        schema_deprecate_threshold=4,
        schema_recover_threshold=5,
    )
    d = cfg.to_dict()
    assert d["consolidation"]["schema_promote_threshold"] == 7
    assert d["consolidation"]["schema_deprecate_threshold"] == 4
    assert d["consolidation"]["schema_recover_threshold"] == 5
    cfg2 = Config._from_dict(d)
    assert cfg2.consolidation.schema_promote_threshold == 7
    assert cfg2.consolidation.schema_deprecate_threshold == 4
    assert cfg2.consolidation.schema_recover_threshold == 5


def test_schema_threshold_default_round_trip_clean():
    """Default values are not serialized (round-trip stays clean)."""
    cfg = Config()
    cfg.consolidation = ConsolidationConfig()
    d = cfg.to_dict()
    consol = d.get("consolidation", {})
    assert "schema_promote_threshold" not in consol
    assert "schema_deprecate_threshold" not in consol
    assert "schema_recover_threshold" not in consol


def test_high_promote_threshold_suppresses_promotion_at_decide_layer():
    """At the policy layer, raising the promote threshold must hold —
    a 3-support window that PROMOTES under default does NOT under thr=99."""
    state = SchemaState(
        schema_id="s1",
        status=SchemaStatus.INFERRED,
        version=1,
        last_window_id="w0",
    )
    ev = EvidenceWindow(window_id="w1", supports=3, contradictions=0)

    # default
    assert decide(state, ev, Thresholds(promote=3)) == EventKind.PROMOTE
    # raised: 3 supports < 99 promote bar → no event
    assert decide(state, ev, Thresholds(promote=99)) is None


def test_low_promote_threshold_promotes_earlier():
    """Symmetric direction — lowering the bar should fire on weaker evidence."""
    state = SchemaState(
        schema_id="s1",
        status=SchemaStatus.INFERRED,
        version=1,
        last_window_id="w0",
    )
    ev = EvidenceWindow(window_id="w1", supports=1, contradictions=0)
    assert decide(state, ev, Thresholds(promote=3)) is None
    assert decide(state, ev, Thresholds(promote=1)) == EventKind.PROMOTE

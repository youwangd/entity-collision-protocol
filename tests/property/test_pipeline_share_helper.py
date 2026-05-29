"""Regression tests for the §8 prior-sharing knob threaded through pipeline.

The pipeline-side helper `_decide_with_share` is a thin shim that:
  - With share=0.0 (default), is byte-identical to bare `decide()`.
  - With share>0.0 but siblings=() (current pre-cluster-wiring state),
    is also byte-identical to bare `decide()` because empty siblings
    contribute zero family evidence (E2/G2 inherited).
  - Reads `share` from `ctx.config.consolidation.schema_family_share`.
  - Out-of-range share is rejected by `decide_with_family` (G6).

These invariants are what make threading the knob now safe — no
pipeline behavior can possibly change at default config.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from engram.consolidation.pipeline import StageContext, _decide_with_share
from engram.consolidation.schema_decision import EvidenceWindow, Thresholds, decide
from engram.consolidation.schema_lifecycle import SchemaState, SchemaStatus
from engram.core.config import Config, ConsolidationConfig


def _make_state(status=SchemaStatus.INFERRED, version=1):
    return SchemaState(
        schema_id="s1",
        status=status,
        version=version,
        last_window_id="w0",
    )


def _ctx_with_share(share: float) -> StageContext:
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(schema_family_share=share)
    return StageContext(config=cfg)


@given(
    supports=st.integers(min_value=0, max_value=20),
    contradictions=st.integers(min_value=0, max_value=20),
    status=st.sampled_from(list(SchemaStatus)),
)
@settings(max_examples=100, deadline=None)
def test_share_zero_identity_to_bare_decide(supports, contradictions, status):
    """share=0.0 ⇒ identical to bare decide() for any state/evidence."""
    ev = EvidenceWindow(window_id="w1", supports=supports, contradictions=contradictions)
    state = _make_state(status=status)
    th = Thresholds()
    expected = decide(state, ev, th)
    actual = _decide_with_share(state, ev, th, _ctx_with_share(0.0))
    assert actual == expected


@given(
    share=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    supports=st.integers(min_value=0, max_value=20),
    contradictions=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_empty_siblings_identity_at_any_share(share, supports, contradictions):
    """With siblings=() (the pre-cluster-wiring state), any share is identity."""
    ev = EvidenceWindow(window_id="w1", supports=supports, contradictions=contradictions)
    state = _make_state()
    th = Thresholds()
    expected = decide(state, ev, th)
    actual = _decide_with_share(state, ev, th, _ctx_with_share(share))
    assert actual == expected


def test_no_config_falls_back_to_bare_decide():
    """ctx.config=None ⇒ share=0.0 default, no crash."""
    ev = EvidenceWindow(window_id="w1", supports=5, contradictions=0)
    state = _make_state()
    th = Thresholds()
    ctx = StageContext(config=None)
    assert _decide_with_share(state, ev, th, ctx) == decide(state, ev, th)


def test_out_of_range_share_raises_at_call_site():
    """share=1.5 ⇒ ValueError from decide_with_family.G6."""
    ev = EvidenceWindow(window_id="w1", supports=5, contradictions=0)
    state = _make_state()
    th = Thresholds()
    with pytest.raises(ValueError):
        _decide_with_share(state, ev, th, _ctx_with_share(1.5))


def test_consolidation_config_yaml_roundtrip(tmp_path):
    """schema_family_share + schema_family_tau survive YAML load."""
    yml = tmp_path / "engram.yaml"
    yml.write_text(
        "path: ~/.engram\n"
        "consolidation:\n"
        "  schedule: manual\n"
        "  window_hours: 12\n"
        "  schema_family_share: 0.75\n"
        "  schema_family_tau: 0.6\n"
    )
    cfg = Config.from_yaml(str(yml))
    assert cfg.consolidation is not None
    assert cfg.consolidation.schema_family_share == 0.75
    assert cfg.consolidation.schema_family_tau == 0.6


def test_consolidation_config_defaults_are_regression_safe():
    """Default ConsolidationConfig() must keep share=0.0 (no behavior change)."""
    cc = ConsolidationConfig()
    assert cc.schema_family_share == 0.0
    assert cc.schema_family_tau == 0.5

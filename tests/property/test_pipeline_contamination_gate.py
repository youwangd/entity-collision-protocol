"""Stage 6 §69 deployment-rule gate: contamination_max in pipeline prepass.

Closes the operational §69 deployment rule by wiring the runtime
contamination meter (`schema_family_contamination.contamination_rate`)
into `_build_schema_family_siblings` as an optional gate.

Invariants:
  C1. cmax=None (default) ⇒ behavior identical to ungated path; no
      stats keys emitted.
  C2. cmax set, observed rate ≤ cmax ⇒ siblings emitted; ctx.stats
      records the rate; no `share_gated` flag.
  C3. cmax set, observed rate > cmax ⇒ empty maps; ctx.stats records
      the rate AND `schema_family_share_gated=True`.
  C4. cmax=0.0 with any non-trivial cluster contamination ⇒ gate trips.
  C5. cmax=1.0 ⇒ gate never trips (rate is always ≤ 1.0).
"""
from __future__ import annotations

from engram.consolidation.pipeline import (
    StageContext,
    _build_schema_family_siblings,
)
from engram.core.config import Config, ConsolidationConfig


def _ctx(share: float, tau: float, cmax: float | None) -> StageContext:
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=share,
        schema_family_tau=tau,
        schema_family_contamination_max=cmax,
    )
    ctx = StageContext(config=cfg)
    ctx.consolidation_id = "w1"
    return ctx


def _tight_schemas():
    """Two patterns sharing dense token vocabulary → cluster together
    at tau=0.3 with high tightness (low contamination)."""
    return [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["alpha bravo charlie"]},
    ]


def _stretchy_schemas():
    """Three patterns connected only by transitive single-link chain:
    A↔B share {alpha,bravo}; B↔C share {bravo,charlie}; A and C share
    only {bravo}. At tau=0.5 only the A↔B and B↔C edges qualify, so
    cluster() unites all three but the A-C direct pair sits below tau
    → contamination > 0."""
    return [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo qq"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern CCCCCCCCCCCC", "facts": ["bravo charlie zz"]},
    ]


def test_c1_cmax_none_no_gate_no_stats():
    sibs, evs = _build_schema_family_siblings(
        _tight_schemas(), "w1", _ctx(share=0.5, tau=0.3, cmax=None),
    )
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert set(evs) == {a, b}
    # Cluster mates emit owner-excluded siblings.
    assert len(sibs[a]) == 1 and sibs[a][0] is evs[b]


def test_c2_clean_clusters_below_cmax_pass():
    ctx = _ctx(share=0.5, tau=0.3, cmax=0.10)
    sibs, evs = _build_schema_family_siblings(_tight_schemas(), "w1", ctx)
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert set(evs) == {a, b}
    assert sibs[a][0] is evs[b]
    # Stats recorded; gate did NOT trip.
    assert "schema_family_contamination_rate" in ctx.stats
    assert ctx.stats["schema_family_contamination_rate"] == 0.0
    assert "schema_family_share_gated" not in ctx.stats


def test_c3_dirty_cluster_trips_gate_returns_empty_maps():
    ctx = _ctx(share=0.5, tau=0.5, cmax=0.10)
    sibs, evs = _build_schema_family_siblings(_stretchy_schemas(), "w1", ctx)
    # Gate tripped: rate > 0.10.
    assert sibs == {}
    assert evs == {}
    rate = ctx.stats["schema_family_contamination_rate"]
    assert rate > 0.10
    assert ctx.stats["schema_family_share_gated"] is True


def test_c4_cmax_zero_trips_on_any_contamination():
    ctx = _ctx(share=0.5, tau=0.5, cmax=0.0)
    sibs, _ = _build_schema_family_siblings(_stretchy_schemas(), "w1", ctx)
    assert sibs == {}
    assert ctx.stats["schema_family_share_gated"] is True


def test_c5_cmax_one_never_trips():
    ctx = _ctx(share=0.5, tau=0.5, cmax=1.0)
    sibs, evs = _build_schema_family_siblings(_stretchy_schemas(), "w1", ctx)
    # rate is in [0, 1] by construction → never strictly > 1.0.
    assert ctx.stats["schema_family_contamination_rate"] <= 1.0
    assert "schema_family_share_gated" not in ctx.stats
    # Single transitive cluster → all three are siblings of each other.
    assert set(evs) == {
        "pattern AAAAAAAAAAAA",
        "pattern BBBBBBBBBBBB",
        "pattern CCCCCCCCCCCC",
    }
    for owner, sib_tuple in sibs.items():
        assert len(sib_tuple) == 2


def test_c_yaml_roundtrip_contamination_max():
    """Config YAML roundtrip preserves the new key when set."""
    import tempfile, os
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=0.75,
        schema_family_tau=0.5,
        schema_family_contamination_max=0.10,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.yaml")
        cfg.save_yaml(path)
        loaded = Config.from_yaml(path)
    assert loaded.consolidation.schema_family_contamination_max == 0.10
    assert loaded.consolidation.schema_family_share == 0.75


def test_c_yaml_roundtrip_default_omits_key():
    """Default cmax=None is omitted from serialized dict (regression-safe)."""
    cfg = Config()
    cfg.consolidation = ConsolidationConfig()
    d = cfg.to_dict()
    cons = d.get("consolidation", {})
    assert "schema_family_contamination_max" not in cons

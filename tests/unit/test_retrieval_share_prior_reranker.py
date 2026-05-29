"""Tests for §96 share_prior reranker.

Invariant under test:
    The original rank-0 candidate must never be demoted by the share_prior
    reranker. Beyond that, the reranker may freely reorder rank ≥ 1 in
    response to the multi-mate signal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram.core.types import (
    EncodingContext,
    Memory,
    MemoryState,
    MemoryType,
    ScoredMemory,
    generate_memory_id,
)
from engram.core.config import Config, RetrievalConfig
from engram.retrieval.engine import RetrievalEngine
from engram.retrieval.rerank import (
    apply_reranker,
    clear_rerankers,
    list_rerankers,
)
from engram.retrieval.rerankers.share_prior import (
    _adaptive_alpha_scale,
    share_prior_reranker,
)
from engram.store.memory import SQLiteMemoryStore


def _mk(content: str, score: float, mid: str | None = None) -> ScoredMemory:
    m = Memory(
        id=mid or generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:50],
        salience=0.5,
        confidence=1.0,
        decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
        encoding_context=EncodingContext(),
    )
    return ScoredMemory(memory=m, score=score, sources={})


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_rerankers()
    yield
    clear_rerankers()


class TestRegistration:
    def test_share_prior_in_builtins(self):
        assert "share_prior" in list_rerankers()


class TestEdgeCases:
    def test_empty_pool(self):
        assert share_prior_reranker([]) == []

    def test_single_item_unchanged(self):
        items = [_mk("Alice met Bob", 0.9)]
        out = share_prior_reranker(items)
        assert out == items
        # Score untouched.
        assert out[0].score == 0.9

    def test_no_entity_overlap_unchanged_order(self):
        items = [
            _mk("alpha thing", 0.9),
            _mk("beta thing", 0.8),
            _mk("gamma thing", 0.7),
        ]
        out = share_prior_reranker(items)
        # Heuristic NER may catch nothing on lower-case nouns → max_deg=0 → bail.
        assert [s.score for s in out] == [0.9, 0.8, 0.7]


class TestRankZeroPreservation:
    def test_rank0_never_demoted_when_others_share_entities(self):
        # rank-0 has no shared entity with anyone; ranks 1..3 all share "Acme Corp".
        items = [
            _mk("In Reykjavik, Solo Cuthbert finished writing.", 0.95),
            _mk("The CFO at Acme Corp bought Foo.", 0.50),
            _mk("Last week Acme Corp launched Bar.", 0.49),
            _mk("Acme Corp said it partnered with Baz.", 0.48),
        ]
        original_top_id = items[0].memory.id
        out = share_prior_reranker(items)
        assert out[0].memory.id == original_top_id, (
            "rank-0 must not be demoted by share_prior"
        )
        # Rank-0 score itself should not have shrunk.
        assert out[0].score >= 0.95 - 1e-9

    def test_boost_capped_below_top(self):
        items = [
            _mk("Reykjavik trip with Solo Cuthbert in winter.", 0.60),
            _mk("Acme Corp acquired Foo.", 0.50),
            _mk("Acme Corp shipped Bar.", 0.49),
            _mk("Acme Corp ended Baz.", 0.48),
        ]
        out = share_prior_reranker(items, alpha=10.0)  # absurd alpha
        # Even with huge alpha, no candidate exceeds rank-0's score.
        assert out[0].score == 0.60
        for sm in out[1:]:
            assert sm.score < 0.60


class TestPromotionWithinTail:
    def test_multi_mate_promoted_among_tail(self):
        # rank-0 anchor with no shared entities. Within the tail, the bridge
        # candidates ("Acme Corp") share with two others; the loner shares
        # with none.
        items = [
            _mk("Reykjavik trip with Solo Cuthbert in winter.", 0.99),  # anchor
            _mk("Quux Industries did Stuff downtown.", 0.80),           # no shares
            _mk("The CFO at Acme Corp acquired Foo.", 0.70),             # shares
            _mk("Last week Acme Corp launched Bar.", 0.65),              # shares
            _mk("Then Acme Corp ended the Baz line.", 0.60),             # shares
        ]
        out = share_prior_reranker(items, alpha=0.20)
        ids = [sm.memory.id for sm in out]
        # Anchor stays first.
        assert ids[0] == items[0].memory.id
        # Among the tail, Acme-bridge candidates should rank higher than
        # Loner Quux even though Loner had a higher fused score.
        loner_pos = ids.index(items[1].memory.id)
        acme_positions = [ids.index(items[k].memory.id) for k in (2, 3, 4)]
        # At least one Acme candidate jumped over Loner.
        assert min(acme_positions) < loner_pos


class TestSourcesAnnotation:
    def test_share_prior_writes_sources(self):
        items = [
            _mk("Reykjavik with Solo Cuthbert in winter.", 0.99),
            _mk("The CFO at Acme Corp acquired Foo.", 0.70),
            _mk("Last week Acme Corp launched Bar.", 0.65),
        ]
        out = share_prior_reranker(items)
        for sm in out:
            assert "share_prior_boost" in sm.sources
            assert "share_prior_degree" in sm.sources


class TestApplyRerankerLenient:
    def test_apply_share_prior_via_registry(self):
        items = [
            _mk("Solo Cuthbert visited Reykjavik.", 0.99),
            _mk("Acme bought Foo.", 0.50),
            _mk("Acme launched Bar.", 0.49),
        ]
        out = apply_reranker("share_prior", items)
        assert isinstance(out, list)
        assert out[0].memory.id == items[0].memory.id


class TestEngineEnd2End:
    def test_engine_with_share_prior_does_not_break(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path)
        try:
            for i, t in enumerate(
                [
                    "Solo Cuthbert visited Reykjavik.",
                    "Acme bought Foo.",
                    "Acme launched Bar.",
                    "Acme merged Baz.",
                ]
            ):
                m = Memory(
                    id=generate_memory_id(MemoryType.FACT),
                    type=MemoryType.FACT,
                    state=MemoryState.ACTIVE,
                    content=t,
                    summary=t[:50],
                    salience=0.6 + i * 0.05,
                    confidence=1.0,
                    decay_rate=0.1,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    access_count=0,
                    encoding_context=EncodingContext(),
                )
                store.upsert(m)
            cfg = RetrievalConfig(reranker="share_prior", rerank_pool_size=10)
            eng = RetrievalEngine(store=store, config=cfg)
            out = eng.search("Acme", limit=4)
            assert isinstance(out, list)
        finally:
            store.close()


class TestRoundTrip:
    def test_share_prior_roundtrips_through_config(self):
        cfg = Config()
        cfg.retrieval.reranker = "share_prior"
        d = cfg.to_dict()
        assert d["retrieval"]["reranker"] == "share_prior"
        cfg2 = Config._from_dict(d)
        assert cfg2.retrieval.reranker == "share_prior"


class TestAdaptiveAlphaSchedule:
    """§5.4 open-angle: max_deg-tapered alpha schedule."""

    def test_schedule_saturates_at_low_max_deg(self):
        assert _adaptive_alpha_scale(0) == 1.0
        assert _adaptive_alpha_scale(1) == 1.0

    def test_schedule_monotone_non_increasing(self):
        prev = 1.0
        for d in range(1, 50):
            cur = _adaptive_alpha_scale(d)
            assert cur <= prev + 1e-12, f"non-monotone at d={d}: {cur}>{prev}"
            assert 0.0 < cur <= 1.0
            prev = cur

    def test_schedule_known_anchors(self):
        # Closed form: 1 / (1 + (d-1)/4)
        assert _adaptive_alpha_scale(2) == pytest.approx(0.8)
        assert _adaptive_alpha_scale(5) == pytest.approx(0.5)
        assert _adaptive_alpha_scale(9) == pytest.approx(1.0 / 3.0)

    def test_schedule_decays_toward_zero(self):
        assert _adaptive_alpha_scale(1000) < 0.005

    def test_adaptive_alpha_off_matches_constant(self):
        items = [
            _mk("Reykjavik with Solo Cuthbert in winter.", 0.99),
            _mk("The CFO at Acme Corp acquired Foo.", 0.70),
            _mk("Last week Acme Corp launched Bar.", 0.65),
            _mk("Then Acme Corp ended the Baz line.", 0.60),
        ]
        items2 = [_mk(s.memory.content, s.score, mid=s.memory.id) for s in items]
        out_const = share_prior_reranker(items, alpha=0.20, adaptive_alpha=False)
        out_adapt_off = share_prior_reranker(items2, alpha=0.20)
        assert [s.score for s in out_const] == [s.score for s in out_adapt_off]

    def test_adaptive_alpha_caps_at_dense_pool(self):
        # Dense pool: every non-anchor candidate shares "Acme Corp" → max_deg high.
        # Adaptive scale tapers the boost vs. constant alpha.
        def _build():
            return [
                _mk("Reykjavik with Solo Cuthbert in winter.", 0.99),
                _mk("The CFO at Acme Corp acquired Foo.", 0.70),
                _mk("Last week Acme Corp launched Bar.", 0.65),
                _mk("Then Acme Corp ended the Baz line.", 0.60),
                _mk("Earlier Acme Corp signed Quux.", 0.55),
                _mk("Yesterday Acme Corp hired someone.", 0.50),
            ]

        out_const = share_prior_reranker(_build(), alpha=0.20)
        out_adapt = share_prior_reranker(_build(), alpha=0.20, adaptive_alpha=True)
        # Non-anchor scores are smaller (or equal) under adaptive schedule.
        for c, a in zip(out_const[1:], out_adapt[1:]):
            assert a.score <= c.score + 1e-9

    def test_adaptive_alpha_preserves_rank0(self):
        items = [
            _mk("Reykjavik with Solo Cuthbert in winter.", 0.99),
            _mk("The CFO at Acme Corp acquired Foo.", 0.70),
            _mk("Last week Acme Corp launched Bar.", 0.65),
            _mk("Then Acme Corp ended the Baz line.", 0.60),
        ]
        anchor_id = items[0].memory.id
        out = share_prior_reranker(items, alpha=10.0, adaptive_alpha=True)
        assert out[0].memory.id == anchor_id
        assert out[0].score == 0.99
        for sm in out[1:]:
            assert sm.score < 0.99

    def test_adaptive_alpha_via_cfg_attribute(self):
        # cfg with share_prior_adaptive_alpha=True should activate the schedule.
        class _Cfg:
            share_prior_adaptive_alpha = True
            entity_ner = "heuristic"

        items = [
            _mk("Reykjavik with Solo Cuthbert in winter.", 0.99),
            _mk("The CFO at Acme Corp acquired Foo.", 0.70),
            _mk("Last week Acme Corp launched Bar.", 0.65),
            _mk("Then Acme Corp ended the Baz line.", 0.60),
            _mk("Earlier Acme Corp signed Quux.", 0.55),
        ]
        items2 = [_mk(s.memory.content, s.score, mid=s.memory.id) for s in items]
        out_cfg = share_prior_reranker(items, cfg=_Cfg(), alpha=0.20)
        out_kwarg = share_prior_reranker(items2, alpha=0.20, adaptive_alpha=True)
        assert [round(s.score, 9) for s in out_cfg] == [
            round(s.score, 9) for s in out_kwarg
        ]


class TestAdaptiveAlphaConfigRoundTrip:
    def test_roundtrip_default_off(self):
        cfg = Config()
        cfg.retrieval.reranker = "share_prior"
        d = cfg.to_dict()
        # Default off → key omitted.
        assert "share_prior_adaptive_alpha" not in d["retrieval"]
        cfg2 = Config._from_dict(d)
        assert cfg2.retrieval.share_prior_adaptive_alpha is False

    def test_roundtrip_explicit_on(self):
        cfg = Config()
        cfg.retrieval.reranker = "share_prior"
        cfg.retrieval.share_prior_adaptive_alpha = True
        d = cfg.to_dict()
        assert d["retrieval"]["share_prior_adaptive_alpha"] is True
        cfg2 = Config._from_dict(d)
        assert cfg2.retrieval.share_prior_adaptive_alpha is True

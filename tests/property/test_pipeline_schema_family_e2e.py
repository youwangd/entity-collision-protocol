"""End-to-end §8 prior-sharing integration test through engine.consolidate().

NEXT.md run #66 next-pickup option (1): proves that the cluster-wiring
shipped in commit 0433329 actually changes lifecycle outcomes when
share>0, not just at the unit-prepass level.

Setup. Inject a stub LLMProvider into a real Engram instance. The stub
returns TWO schema patterns whose supporting_facts share content tokens
above tau=0.5, so `schema_family.cluster()` groups them. Each pattern
has only **2** supporting facts — below the default promote threshold
(K_promote=3) — so a single-schema decision (share=0.0) leaves both
status=INFERRED.

Hypothesis. With share=0.75:

    eff_supports = own.supports + floor(0.75 * Σ siblings.supports)
                 = 2            + floor(0.75 * 2)
                 = 2 + 1 = 3 ≥ promote → PROMOTE

Both patterns should fire PROMOTE in the same window. With share=0.0:

    eff_supports = 2  (bare decide())

→ both stay INFERRED.

Same pipeline, same data, different config knob — the lifecycle
projection should diverge in exactly the way the §8 sweep predicted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engram import Engram, Config
from engram.consolidation.lifecycle_projection import snapshot_from_buffer
from engram.consolidation.schema_lifecycle import SchemaStatus
from engram.core.config import ConsolidationConfig
from engram.core.types import (
    Event,
    EventType,
    Memory,
    MemoryType,
    generate_event_id,
)
from engram.providers.llm import LLMProvider


# Two patterns whose supporting_facts share heavy content-token overlap
# so the `schema_fingerprint` adapter assigns them to the same cluster
# at default tau=0.5. Both patterns have 2 supports (< promote=3).
#
# Fingerprint preview (after stop-words / min-len-3 / lowercased):
#   pattern A facts → {"alice", "uses", "postgres", "production",
#                      "database", "transactional"}
#   pattern B facts → {"bob",   "uses", "postgres", "production",
#                      "database", "transactional"}
# Jaccard = 5 / 7 ≈ 0.71 ≥ 0.5 → cluster together.

_PATTERN_A = (
    "team-A standardizes on postgres for production transactional "
    "databases with strict consistency guarantees and partitioning"
)
_PATTERN_B = (
    "team-B also standardizes on postgres for production transactional "
    "databases with read replicas and connection pooling"
)
assert len(_PATTERN_A) >= 80 and len(_PATTERN_B) >= 80


class _TwoClusteredPatternsLLM(LLMProvider):
    """Returns the two cluster-mate patterns, each with 2 supports."""

    _PAYLOAD = {
        "schemas": [
            {
                "pattern": _PATTERN_A,
                "facts": [
                    "alice uses postgres in production transactional database",
                    "alice uses postgres in production transactional database",
                ],
            },
            {
                "pattern": _PATTERN_B,
                "facts": [
                    "bob uses postgres in production transactional database",
                    "bob uses postgres in production transactional database",
                ],
            },
        ]
    }

    def extract_json(self, prompt: str, *, system: str = "", **_kw):
        return self._PAYLOAD

    def complete(self, prompt: str, **_kw) -> str:  # pragma: no cover
        return ""


def _seed_facts(eng: Engram) -> None:
    """Seed enough fact memories to clear SchemaUpdate's len(facts) >= 3 gate."""
    for i in range(5):
        ev = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.EVENT_CAPTURE,
            content=f"team member {i} uses postgres in production",
        )
        m = Memory.from_event(ev, memory_type=MemoryType.FACT)
        eng._store.upsert(m)


def _build_engine(tmp_path: Path, share: float) -> Engram:
    cfg = Config(path=str(tmp_path / f"engram_share_{share}"))
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=share,
        schema_family_tau=0.5,
    )
    eng = Engram(config=cfg, llm=_TwoClusteredPatternsLLM())
    _seed_facts(eng)
    return eng


def _schema_status_by_summary(eng: Engram) -> dict[str, SchemaStatus]:
    """Map schema-summary → final status from the lifecycle projection."""
    snap = snapshot_from_buffer(eng._buffer, strict=False)
    out: dict[str, SchemaStatus] = {}
    for sid, state in snap.items():
        mem = eng._store.get(sid)
        if mem is not None and mem.type == MemoryType.SCHEMA:
            out[mem.summary] = state.status
    return out


def test_share_zero_neither_pattern_promotes(tmp_path: Path):
    """share=0.0 (regression-safe default): single-schema decide() with
    only 2 supports each. Both schemas stay INFERRED."""
    eng = _build_engine(tmp_path, share=0.0)
    try:
        eng.consolidate()
        statuses = _schema_status_by_summary(eng)
        sa, sb = _PATTERN_A[:80], _PATTERN_B[:80]
        assert sa in statuses and sb in statuses, (
            f"both schemas should be created (got {list(statuses)})"
        )
        assert statuses[sa] is SchemaStatus.INFERRED
        assert statuses[sb] is SchemaStatus.INFERRED
    finally:
        eng.close()


def test_share_seventy_five_clusters_lift_both_to_promoted(tmp_path: Path):
    """share=0.75: the two patterns cluster on overlapping fact-vocab.
    Each owner gets floor(0.75*2)=1 borrowed support → eff=3 ≥ promote.
    Both schemas reach PROMOTED in one window."""
    eng = _build_engine(tmp_path, share=0.75)
    try:
        eng.consolidate()
        statuses = _schema_status_by_summary(eng)
        sa, sb = _PATTERN_A[:80], _PATTERN_B[:80]
        assert sa in statuses and sb in statuses
        assert statuses[sa] is SchemaStatus.PROMOTED, (
            f"pattern A should promote under share=0.75, got {statuses[sa]}"
        )
        assert statuses[sb] is SchemaStatus.PROMOTED, (
            f"pattern B should promote under share=0.75, got {statuses[sb]}"
        )
    finally:
        eng.close()


def test_share_zero_vs_seventy_five_diverge(tmp_path: Path):
    """Composition test: same data, same LLM, only the share knob
    differs → exactly the §8 sweep prediction (no promotes at share=0,
    full promotion at share=0.75) replicates end-to-end."""
    eng0 = _build_engine(tmp_path / "s0", share=0.0)
    eng75 = _build_engine(tmp_path / "s75", share=0.75)
    try:
        eng0.consolidate()
        eng75.consolidate()

        s0 = _schema_status_by_summary(eng0)
        s75 = _schema_status_by_summary(eng75)

        promoted_0 = sum(1 for v in s0.values() if v is SchemaStatus.PROMOTED)
        promoted_75 = sum(1 for v in s75.values() if v is SchemaStatus.PROMOTED)

        assert promoted_0 == 0, f"expected 0 promotes at share=0, got {promoted_0}"
        assert promoted_75 == 2, (
            f"expected 2 promotes at share=0.75, got {promoted_75} "
            f"(statuses={s75})"
        )
    finally:
        eng0.close()
        eng75.close()

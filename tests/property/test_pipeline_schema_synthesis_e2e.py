"""§93 — End-to-end wiring test for the deterministic non-LLM schema synthesizer.

§91's diagnostic: in the no-LLM cron environment, `SchemaUpdate` only
runs the LLM-powered branch, so `consolidate()` produces zero schemas
and the §85/§87 schema-family gate is unreachable. This test pins down
the wiring so a future regression that breaks the synthesis fallback
trips here.

Setup. Real `Engram` instance, default `NoLLMProvider`, but with
`consolidation.schema_synthesis_enabled=True`. Ingest 9 facts that
form 3 clear lexical clusters (3 facts each). Call `consolidate()`.
Assert at least one schema (`MemoryType.SCHEMA`) was created and at
least one lifecycle CREATE event lives in the buffer.

Hard claim: with the flag *off* (regression-safe default), zero
schemas are created from the same corpus — the §93 fallback is the
sole reason schemas appear. We test that too.
"""
from __future__ import annotations

import tempfile

from engram import Engram, Config
from engram.core.config import ConsolidationConfig
from engram.core.types import MemoryType


_FACTS = [
    "Alice loves pizza for dinner",
    "Alice eats pizza on weekends",
    "Pizza is what Alice prefers most",
    "Bob hates spinach in salads",
    "Bob refuses spinach at parties",
    "Spinach is what Bob avoids always",
    "Carol drinks coffee every morning",
    "Carol enjoys coffee with milk",
    "Coffee is Carol favorite beverage",
]


def _build_cfg(tmp: str, *, synthesis: bool) -> Config:
    cfg = Config(path=tmp)
    cfg.security.max_events_per_minute = 0
    cfg.consolidation = ConsolidationConfig(
        schedule="manual",
        window_hours=24 * 999,
        schema_synthesis_enabled=synthesis,
        schema_synthesis_tau=0.3,
        schema_synthesis_min_supports=3,
    )
    return cfg


def test_synthesis_off_creates_zero_schemas():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(tmp, synthesis=False)
        eng = Engram(config=cfg)
        try:
            for f in _FACTS:
                eng.remember(f)
            # capture() so events reach EventReplay
            for f in _FACTS:
                eng.capture(f)
            eng.consolidate(window="999d")
            schemas = eng._store.search_by_type(MemoryType.SCHEMA, limit=100)
            assert len(schemas) == 0
        finally:
            eng.close()


def test_synthesis_on_creates_schemas_no_llm():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(tmp, synthesis=True)
        eng = Engram(config=cfg)
        try:
            for f in _FACTS:
                eng.remember(f)
            for f in _FACTS:
                eng.capture(f)
            eng.consolidate(window="999d")
            schemas = eng._store.search_by_type(MemoryType.SCHEMA, limit=100)
            assert len(schemas) >= 1, (
                "§93 synthesizer should emit ≥1 schema on a corpus with "
                "lexically clustered facts in the no-LLM environment."
            )
            # Each schema's content should start with the synthesizer's
            # canonical "recurring: " prefix.
            assert all(
                s.content.startswith("recurring: ") for s in schemas
            ), [s.content[:40] for s in schemas]
            # Lifecycle CREATE events should have been emitted alongside
            # SCHEMA memories.
            from engram.consolidation.lifecycle_projection import (
                snapshot_from_buffer,
            )
            snap = snapshot_from_buffer(eng._buffer, strict=False)
            schema_ids = {s.id for s in schemas}
            assert any(sid in snap for sid in schema_ids), (
                "synth schemas must appear in lifecycle snapshot"
            )
        finally:
            eng.close()


def test_synthesis_idempotent_on_re_consolidate():
    """Running consolidate() twice with the same facts must not duplicate.

    The summary-slot dedup in SchemaUpdate (existing_by_summary) plus
    the synthesizer's stable pattern output should make this a no-op
    on re-run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_cfg(tmp, synthesis=True)
        eng = Engram(config=cfg)
        try:
            for f in _FACTS:
                eng.remember(f)
            for f in _FACTS:
                eng.capture(f)
            eng.consolidate(window="999d")
            n1 = len(eng._store.search_by_type(MemoryType.SCHEMA, limit=100))
            eng.consolidate(window="999d")
            n2 = len(eng._store.search_by_type(MemoryType.SCHEMA, limit=100))
            assert n1 == n2, f"expected no-op re-consolidate, got {n1} → {n2}"
            assert n1 >= 1
        finally:
            eng.close()

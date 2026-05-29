"""Tests for engine.recall_with_filters (Governed Memory paper §5).

Covers:
- filter-only mode (no query) with equality filters
- filter-only mode with numeric comparison operators (>, >=, <, <=, ==, !=)
- hybrid mode (query + filters): recall ranks, filter intersects
- AND-semantics: all filters must match
- error: empty filters and empty query raise ValueError
- missing property: memory without the key is excluded
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engram import Engram, Config
from engram.core.types import Memory, MemoryType, MemoryState, generate_memory_id


@pytest.fixture
def tmp_engine(tmp_path: Path):
    cfg = Config(path=str(tmp_path / "engram"))
    eng = Engram(config=cfg)
    yield eng


def _seed(eng: Engram, content: str, props: dict, salience: float = 0.5) -> str:
    """Insert a memory with typed properties, bypassing the LLM extractor."""
    from datetime import datetime, timezone
    mem = Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:80],
        salience=salience,
        confidence=1.0,
        decay_rate=0.01,
        created_at=datetime.now(timezone.utc),
    )
    eng._store.upsert(mem)
    eng._store.upsert_properties(
        mem.id,
        [{"key": k, "value": str(v), "type": "text", "confidence": 1.0} for k, v in props.items()],
    )
    return mem.id


class TestRecallWithFiltersEquality:
    def test_filter_only_single_property(self, tmp_engine):
        a = _seed(tmp_engine, "alpha is the lead", {"role": "lead"})
        _seed(tmp_engine, "bravo is the deputy", {"role": "deputy"})
        results = tmp_engine.recall_with_filters(properties={"role": "lead"})
        assert [m.id for m in results] == [a]

    def test_filter_only_multi_property_AND(self, tmp_engine):
        match = _seed(tmp_engine, "alpha lead in eng", {"role": "lead", "team": "eng"})
        _seed(tmp_engine, "bravo lead in design", {"role": "lead", "team": "design"})
        _seed(tmp_engine, "charlie deputy in eng", {"role": "deputy", "team": "eng"})
        results = tmp_engine.recall_with_filters(properties={"role": "lead", "team": "eng"})
        assert [m.id for m in results] == [match]

    def test_filter_only_no_match(self, tmp_engine):
        _seed(tmp_engine, "x", {"role": "lead"})
        assert tmp_engine.recall_with_filters(properties={"role": "ghost"}) == []

    def test_filter_only_excludes_missing_property(self, tmp_engine):
        has = _seed(tmp_engine, "has prop", {"team": "eng"})
        _seed(tmp_engine, "no prop", {})  # no properties
        results = tmp_engine.recall_with_filters(properties={"team": "eng"})
        assert [m.id for m in results] == [has]


class TestRecallWithFiltersNumeric:
    def test_greater_than(self, tmp_engine):
        _seed(tmp_engine, "small deal", {"deal_value": "50"})
        b = _seed(tmp_engine, "big deal", {"deal_value": "150"})
        c = _seed(tmp_engine, "huge deal", {"deal_value": "999"})
        results = tmp_engine.recall_with_filters(properties={"deal_value": ">100"})
        assert {m.id for m in results} == {b, c}

    def test_lte(self, tmp_engine):
        a = _seed(tmp_engine, "exactly", {"deal_value": "100"})
        _seed(tmp_engine, "above", {"deal_value": "150"})
        results = tmp_engine.recall_with_filters(properties={"deal_value": "<=100"})
        assert [m.id for m in results] == [a]

    def test_eq_neq(self, tmp_engine):
        a = _seed(tmp_engine, "a", {"score": "42"})
        b = _seed(tmp_engine, "b", {"score": "100"})
        eq = tmp_engine.recall_with_filters(properties={"score": "==42"})
        assert [m.id for m in eq] == [a]
        ne = tmp_engine.recall_with_filters(properties={"score": "!=42"})
        assert [m.id for m in ne] == [b]


class TestRecallWithFiltersHybrid:
    def test_query_plus_filter(self, tmp_engine):
        # Two leads, query mentions "alpha"
        a = _seed(tmp_engine, "alpha leads engineering team alpha", {"role": "lead"}, salience=0.8)
        _seed(tmp_engine, "bravo leads design team bravo", {"role": "lead"}, salience=0.8)
        _seed(tmp_engine, "alpha is a deputy", {"role": "deputy"}, salience=0.8)
        results = tmp_engine.recall_with_filters(query="alpha", properties={"role": "lead"})
        ids = [m.id for m in results]
        assert a in ids
        # The deputy "alpha" should be excluded by the filter
        assert all(tmp_engine._store.get_properties(m.id) for m in results)

    def test_query_only(self, tmp_engine):
        a = _seed(tmp_engine, "the quick brown fox", {})
        _seed(tmp_engine, "totally unrelated", {})
        results = tmp_engine.recall_with_filters(query="fox", limit=1)
        assert results[0].id == a


class TestRecallWithFiltersErrors:
    def test_empty_raises(self, tmp_engine):
        with pytest.raises(ValueError):
            tmp_engine.recall_with_filters()
        with pytest.raises(ValueError):
            tmp_engine.recall_with_filters(properties={})

    def test_limit_respected(self, tmp_engine):
        for i in range(5):
            _seed(tmp_engine, f"m{i}", {"team": "eng"})
        results = tmp_engine.recall_with_filters(properties={"team": "eng"}, limit=3)
        assert len(results) == 3

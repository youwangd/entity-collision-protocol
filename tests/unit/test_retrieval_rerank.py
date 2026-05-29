"""Tests for the post-rerank stage (§96 hook).

The reranker stage is a registry-keyed callable that runs over the
fused, scored, pre-limit candidate pool. It must:

  - be a no-op when `cfg.reranker` is None (default behaviour, zero
    regression on the rest of the suite).
  - allow custom reranking that reorders within the limit.
  - allow reordering items from outside `limit` into the top-`limit`
    via `rerank_pool_size`.
  - be lenient: a raising reranker must NOT break retrieval.
  - round-trip through EngramConfig.to_dict / from_dict when set.
"""

from datetime import datetime, timezone

import pytest

from engram.core.config import Config, RetrievalConfig
from engram.core.types import (
    EncodingContext,
    Memory,
    MemoryState,
    MemoryType,
    generate_memory_id,
)
from engram.retrieval.engine import RetrievalEngine
from engram.retrieval.rerank import (
    apply_reranker,
    clear_rerankers,
    get_reranker,
    list_rerankers,
    register_reranker,
)
from engram.store.memory import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _clean_registry():
    # Each test starts from the builtins-only registry.
    clear_rerankers()
    yield
    clear_rerankers()


def _mk(content, salience=0.5):
    return Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:50],
        salience=salience,
        confidence=1.0,
        decay_rate=0.1,
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=0,
        encoding_context=EncodingContext(),
    )


class TestRegistry:
    def test_identity_is_builtin(self):
        assert "identity" in list_rerankers()
        assert get_reranker("identity") is not None

    def test_register_then_get(self):
        register_reranker("flip", lambda r, **_: list(reversed(r)))
        assert "flip" in list_rerankers()
        assert get_reranker("flip") is not None

    def test_get_unknown_returns_none(self):
        assert get_reranker("nope") is None
        assert get_reranker(None) is None
        assert get_reranker("") is None

    def test_register_invalid_name_raises(self):
        with pytest.raises(ValueError):
            register_reranker("", lambda r, **_: r)

    def test_clear_rerankers_restores_builtins(self):
        register_reranker("temp", lambda r, **_: r)
        clear_rerankers()
        assert "temp" not in list_rerankers()
        assert "identity" in list_rerankers()


class TestApplyReranker:
    def test_no_name_is_identity(self):
        items = [object(), object()]
        assert apply_reranker(None, items) == items

    def test_unknown_name_is_identity(self):
        items = [object(), object()]
        assert apply_reranker("missing", items) == items

    def test_lenient_on_exception(self):
        register_reranker("boom", lambda r, **_: 1 / 0)
        items = [object(), object()]
        # Returns input order, does not raise.
        out = apply_reranker("boom", items)
        assert out == items

    def test_non_list_return_is_coerced(self):
        register_reranker("gen", lambda r, **_: iter(reversed(r)))
        items = [1, 2, 3]
        out = apply_reranker("gen", items)
        assert out == [3, 2, 1]


class TestEngineWiring:
    def test_default_config_is_no_op(self, store):
        # No reranker → identical results to non-reranked path.
        for i, t in enumerate(["alpha sentence", "beta phrase", "gamma idea"]):
            store.upsert(_mk(t, salience=0.5 + i * 0.1))
        eng = RetrievalEngine(store=store, config=RetrievalConfig())
        out = eng.search("alpha", limit=3)
        assert len(out) >= 1
        assert eng.config.reranker is None  # sanity

    def test_reranker_can_promote_from_outside_limit(self, store):
        # Insert N=5 memories. With limit=2 and pool=5 the reranker
        # can promote a candidate that would otherwise be dropped.
        ids = []
        for i in range(5):
            m = _mk(f"alpha sentence {i}", salience=0.5)
            store.upsert(m)
            ids.append(m.id)

        # Reranker that picks the alphabetically last id and pins it first.
        target = max(ids)

        def pin_target(results, **_):
            results = list(results)
            for i, sm in enumerate(results):
                if sm.memory.id == target:
                    results.insert(0, results.pop(i))
                    break
            return results

        register_reranker("pin_target", pin_target)
        cfg = RetrievalConfig(reranker="pin_target", rerank_pool_size=10)
        eng = RetrievalEngine(store=store, config=cfg)
        out = eng.search("alpha", limit=2)
        assert out, "expected at least one hit"
        assert out[0].memory.id == target

    def test_reranker_exception_does_not_break_search(self, store):
        store.upsert(_mk("alpha sentence"))
        register_reranker("boom", lambda r, **_: (_ for _ in ()).throw(RuntimeError("x")))
        cfg = RetrievalConfig(reranker="boom")
        eng = RetrievalEngine(store=store, config=cfg)
        # Must not raise.
        out = eng.search("alpha", limit=3)
        assert isinstance(out, list)

    def test_reranker_receives_query_context(self, store):
        store.upsert(_mk("alpha sentence"))
        captured = {}

        def capture(results, **kwargs):
            captured.update(kwargs)
            return results

        register_reranker("capture", capture)
        cfg = RetrievalConfig(reranker="capture")
        eng = RetrievalEngine(store=store, config=cfg)
        eng.search("alpha", limit=3)
        assert captured.get("query") == "alpha"
        assert "intent" in captured
        assert "entity_cache" in captured


class TestConfigRoundTrip:
    def test_reranker_roundtrip_preserves_pool_size(self):
        cfg = Config()
        cfg.retrieval.reranker = "identity"
        cfg.retrieval.rerank_pool_size = 42
        d = cfg.to_dict()
        assert d["retrieval"]["reranker"] == "identity"
        assert d["retrieval"]["rerank_pool_size"] == 42
        cfg2 = Config._from_dict(d)
        assert cfg2.retrieval.reranker == "identity"
        assert cfg2.retrieval.rerank_pool_size == 42

    def test_no_reranker_omits_pool_size_from_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        # When reranker is None we don't bother writing pool_size.
        assert "reranker" not in d.get("retrieval", {})
        assert "rerank_pool_size" not in d.get("retrieval", {})

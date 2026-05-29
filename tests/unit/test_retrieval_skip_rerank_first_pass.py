"""§4.15-profile — `query_expansion_skip_rerank_first_pass` lever.

When True AND a reranker is configured AND PRF is active, the FIRST-pass
retrieval (whose results are only used to mine entities for query
expansion) skips the reranker. The second pass still reranks normally.
Targets the §4.15-profile observation that the `both` arm pays ~24% p95
overhead, half of which is wasted reranker work on a first pass whose
ordering only affects entity mining.
"""
from __future__ import annotations

from engram import Config, Engram
from engram.providers.embeddings import HashTrigramEmbeddingProvider


def _make_engine(tmp_path, *, skip: bool, reranker: str | None = "share_prior"):
    cfg = Config(path=str(tmp_path))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.vector_weight = 0.3
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_top_k = 5
    # Disable the anchor-share gate so PRF expansion deterministically
    # fires for the well-known Alice/Earl-Grey corpus.
    cfg.retrieval.query_expansion_anchor_share_max = None
    cfg.retrieval.reranker = reranker
    cfg.retrieval.rerank_pool_size = 10
    cfg.retrieval.query_expansion_skip_rerank_first_pass = skip
    return Engram(config=cfg, embeddings=HashTrigramEmbeddingProvider(dimension=128))


def _seed_corpus(eng):
    # Heuristic NER picks up capitalised tokens. We want the top-K pool to
    # have a clearly dominant entity so PRF expansion fires deterministically.
    rows = [
        "Alice prefers Earl Grey tea on cold mornings.",
        "Alice ordered Earl Grey from the cafe near her office.",
        "Alice bought Earl Grey at the market on Saturday.",
        "Alice noted that Earl Grey pairs well with shortbread.",
        "Alice said Earl Grey is her go-to drink before meetings.",
        "Bob prefers black coffee in tall mugs.",
        "Charlie likes oolong over jasmine.",
        "Diana drinks chamomile at night.",
    ]
    for r in rows:
        eng.remember(r)


def test_default_off(tmp_path):
    cfg = Config(path=str(tmp_path / "default"))
    assert cfg.retrieval.query_expansion_skip_rerank_first_pass is False


def test_inert_when_no_reranker(tmp_path):
    """Skip-flag must be a no-op when no reranker is configured.
    Compares result *contents* (not IDs, which embed timestamps)."""
    eng_a = _make_engine(tmp_path / "a", skip=True, reranker=None)
    eng_b = _make_engine(tmp_path / "b", skip=False, reranker=None)
    _seed_corpus(eng_a)
    _seed_corpus(eng_b)
    a = [r.memory.content for r in eng_a.recall("what does Alice drink", limit=5)]
    b = [r.memory.content for r in eng_b.recall("what does Alice drink", limit=5)]
    eng_a.close()
    eng_b.close()
    assert a == b, "skip-flag must be inert when no reranker is configured"


def test_skip_first_pass_call_count(tmp_path, monkeypatch):
    """When skip=True AND PRF expansion fires, reranker apply must be
    called exactly ONCE per user-facing recall (the second pass).
    """
    eng = _make_engine(tmp_path, skip=True)
    _seed_corpus(eng)
    from engram.retrieval import engine as engine_mod

    calls = {"n": 0}
    real = engine_mod.apply_reranker

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "apply_reranker", spy)
    eng.recall("what does Alice drink", limit=5)
    eng.close()
    assert calls["n"] == 1, f"expected 1 reranker call, got {calls['n']}"


def test_no_skip_double_call(tmp_path, monkeypatch):
    """Without the skip flag, both passes rerank → 2 calls."""
    eng = _make_engine(tmp_path, skip=False)
    _seed_corpus(eng)
    from engram.retrieval import engine as engine_mod

    calls = {"n": 0}
    real = engine_mod.apply_reranker

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "apply_reranker", spy)
    eng.recall("what does Alice drink", limit=5)
    eng.close()
    assert calls["n"] == 2, f"expected 2 reranker calls, got {calls['n']}"


def test_yaml_roundtrip(tmp_path):
    cfg = Config(path=str(tmp_path))
    cfg.retrieval.query_expansion_skip_rerank_first_pass = True
    out = tmp_path / "cfg.yaml"
    cfg.save_yaml(str(out))
    cfg2 = Config.from_yaml(str(out))
    assert cfg2.retrieval.query_expansion_skip_rerank_first_pass is True

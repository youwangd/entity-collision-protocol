"""Adversarial — share_prior reranker ACL side-channel (§D-share-prior-acl).

The §96 share_prior reranker builds an entity-sharing graph over the
retrieved candidate pool and adds a `α · deg / max_deg` boost to each
candidate's fused score. The reranker runs at the `RetrievalEngine` layer
— i.e. BEFORE `Engram.recall()`'s outer ACL filter at engine.py:408.

Without an ACL-aware filter on the reranker pool, the entity-sharing
graph spans cross-agent docs. That means:

  - the multi-mate `degrees[i]` for each Alice doc counts edges to Bob
    docs that share entities;
  - `max_deg` (the normaliser) is global;
  - therefore the boost added to Alice's *own* docs depends on the
    distribution of entities in Bob's private corpus.

The cross-agent docs are dropped at engine.py:408 *after* the reranker
has perturbed the surviving docs' scores. The visible outputs are all
Alice's, but their RANKING is now a function of Bob's private content
— a presence oracle.

This file pins the closure:

  ACL-SP-1  Alice's ranking over her own docs is invariant to Bob's
            private corpus across diversifying / dense / mixed arms.

  ACL-SP-2  Federated `scope='*'` actor still sees the global pool
            (no over-correction).

  ACL-SP-3  ACL-disabled config behaves identically to the pre-fix
            single-actor path (no regression).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engram import Engram
from engram.core.config import Config


def _make_engram(tmp: Path, *, acl: bool, actor: str = "alice") -> Engram:
    cfg = Config(path=str(tmp))
    # Enable the share_prior reranker over a wide pool so cross-agent
    # docs can enter the entity-sharing graph.
    cfg.retrieval.reranker = "share_prior"
    cfg.retrieval.rerank_pool_size = 50
    cfg.retrieval.share_prior_alpha = 0.05
    # Disable PRF so we isolate the reranker side-channel from the
    # already-closed PRF channel.
    cfg.retrieval.query_expansion_min_dominance = None
    if acl:
        cfg.acl = {
            "enabled": True,
            "grants": {
                "alice": {"permissions": ["read", "write"], "scope": "own"},
                "bob": {"permissions": ["read", "write"], "scope": "own"},
            },
        }
    return Engram(config=cfg, actor=actor)


def _seed_alice(e: Engram) -> None:
    """Alice owns 4 docs with bridge-able entities."""
    e.remember("notes about project Apollo with Alice", agent_id="alice")
    e.remember("notes about project Beta with Alice", agent_id="alice")
    e.remember("notes summary stuff Alice", agent_id="alice")
    e.remember("notes neutral random Alice", agent_id="alice")


def _ranks(e: Engram, query: str = "notes Alice", limit: int = 10):
    return [(r.memory.content, round(r.score, 6)) for r in e.recall(query, limit=limit)]


# ---------- ACL-SP-1: rank invariance under Bob's corpus ----------

@pytest.mark.parametrize(
    "bob_signal",
    [
        None,
        "Apollo",         # Bob piles entity that one Alice doc shares
        "Beta",           # Bob piles entity that the other Alice doc shares
        "Apollo Beta",    # Bob bridges Alice's two named docs
        "Gamma",          # Bob's entities orthogonal to Alice
    ],
)
def test_alice_ranking_invariant_to_bob_corpus(tmp_path: Path, bob_signal):
    """share_prior must not let Bob's corpus perturb Alice's own ranking."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    e_base = _make_engram(base_dir, acl=True)
    _seed_alice(e_base)
    base = _ranks(e_base)

    treat_dir = tmp_path / f"treat_{(bob_signal or 'none').replace(' ', '_')}"
    treat_dir.mkdir()
    e_treat = _make_engram(treat_dir, acl=True)
    _seed_alice(e_treat)
    if bob_signal is not None:
        for i in range(20):
            e_treat.remember(
                f"notes {bob_signal} {bob_signal} item {i} Alice",
                agent_id="bob",
            )
    treat = _ranks(e_treat)

    # Order AND scores must match: any score perturbation is a leak.
    base_order = [c for c, _ in base]
    treat_order = [c for c, _ in treat]
    assert treat_order == base_order, (
        f"share_prior reranker leaked Bob's corpus into Alice's ranking.\n"
        f"  bob_signal={bob_signal!r}\n  base:  {base_order}\n  treat: {treat_order}"
    )
    # Tighter: scores match too. share_prior boost is bounded; if it fires
    # off cross-agent edges, scores for Alice's docs change measurably.
    for (c_b, s_b), (c_t, s_t) in zip(base, treat):
        assert c_b == c_t and s_b == s_t, (
            f"share_prior score perturbation under bob_signal={bob_signal!r}: "
            f"{c_b!r} {s_b} vs {c_t!r} {s_t}"
        )


# ---------- ACL-SP-2: federated grant still gets global pool ----------

def test_federated_actor_still_sees_full_pool(tmp_path: Path):
    cfg = Config(path=str(tmp_path))
    cfg.retrieval.reranker = "share_prior"
    cfg.retrieval.rerank_pool_size = 50
    cfg.retrieval.share_prior_alpha = 0.05
    cfg.retrieval.query_expansion_min_dominance = None
    cfg.acl = {
        "enabled": True,
        "grants": {
            "reviewer": {
                "permissions": ["read", "write", "federated"],
                "scope": "*",
            },
            "alice": {"permissions": ["read", "write"], "scope": "own"},
            "bob": {"permissions": ["read", "write"], "scope": "own"},
        },
    }
    e = Engram(config=cfg, actor="reviewer")
    e.remember("notes about project Apollo", agent_id="alice")
    e.remember("notes about project Beta", agent_id="alice")
    for i in range(10):
        e.remember(f"notes Zorbax item {i}", agent_id="bob")

    results = e.recall("notes", limit=8)
    contents = [r.memory.content for r in results]
    assert any("Zorbax" in c for c in contents), (
        f"reviewer with scope='*' should still see Bob's docs; got {contents}"
    )


# ---------- ACL-SP-3: ACL-off no-regression ----------

def test_acl_disabled_no_regression(tmp_path: Path):
    e = _make_engram(tmp_path, acl=False)
    _seed_alice(e)
    for i in range(10):
        e.remember(f"notes Apollo item {i}", agent_id="bob")
    results = e.recall("notes Alice", limit=8)
    assert results, "expected non-empty results with ACL disabled"
    # With ACL off, Bob's docs are visible — no exception, no regression.
    contents = [r.memory.content for r in results]
    assert any("Bob" not in c or "item" in c for c in contents)


# ---------- ACL-SP-4: direct reranker-pool isolation canary ----------

def test_reranker_pool_excludes_cross_agent_when_acl_on(tmp_path: Path, monkeypatch):
    """The pool handed to share_prior must contain only Alice-visible docs."""
    e = _make_engram(tmp_path, acl=True)
    _seed_alice(e)
    for i in range(15):
        e.remember(f"notes Zorbax Zorbax item {i}", agent_id="bob")

    seen_pools: list[list[str]] = []
    import engram.retrieval.engine as eng_mod

    orig_apply = eng_mod.apply_reranker

    def spy_apply(name, pool, **kw):
        seen_pools.append([sm.memory.content for sm in pool])
        return orig_apply(name, pool, **kw)

    monkeypatch.setattr(eng_mod, "apply_reranker", spy_apply)

    e.recall("notes Alice", limit=5)

    assert seen_pools, "share_prior reranker should have been invoked"
    leaks = [c for pool in seen_pools for c in pool if "Zorbax" in c]
    assert not leaks, (
        f"cross-agent docs leaked into share_prior reranker pool: {leaks[:3]}"
    )

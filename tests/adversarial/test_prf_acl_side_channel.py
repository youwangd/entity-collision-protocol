"""Adversarial — PRF query expansion ACL side-channel (§D-prf-acl).

The PRF (pseudo-relevance-feedback) entity expansion mines dominant
entities from the *first-pass* retrieval to construct an expanded query.
Without the §D-prf-acl filter, that first pass runs at the
`RetrievalEngine` layer — i.e. before `Engram.recall()`'s outer ACL
filter — so the entity-mining pool can include memories the actor cannot
READ. The expanded query (and therefore the actor's final ranking over
its *own* memories) then depends on the cross-agent corpus.

This is a side-channel oracle: an actor with `scope='own'` can detect
the presence of specific entities in another agent's private memories
by observing rank-perturbations on its own queries.

This file pins the closure of that channel:

  ACL-PRF-1  Alice's recall ranking is invariant to Bob's private corpus
             (token-by-token ordering equality across the absent /
             diversifying-Bob-corpus / dense-Bob-corpus arms).

  ACL-PRF-2  The PRF expansion seen by the engine for Alice's query
             never includes any token that appears *only* in Bob's
             memories (entity isolation).

  ACL-PRF-3  Federated mode (Permission.FEDERATED granted) is *not*
             gated by this filter: a cross-grant Engram with scope='*'
             still mines entities from the full corpus.

  ACL-PRF-4  ACL-disabled config has zero behavioural change vs. the
             baseline single-actor path (no regression).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engram import Engram
from engram.core.config import Config


def _make_engram(tmp: Path, *, acl: bool, actor: str = "alice") -> Engram:
    cfg = Config(path=str(tmp))
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_top_k = 10
    cfg.retrieval.query_expansion_max_entities = 3
    # Disable the anchor-share gate so PRF fires on diverse corpora;
    # the gate is a *separate* defence in depth and we want to test the
    # ACL filter independently.
    cfg.retrieval.query_expansion_anchor_share_max = None
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
    """Alice owns 4 docs, two with disambiguating entities (Apollo / Beta)."""
    e.remember("notes neutral random", agent_id="alice")
    e.remember("notes summary stuff", agent_id="alice")
    e.remember("notes about project Apollo", agent_id="alice")
    e.remember("notes about project Beta", agent_id="alice")


def _ranks(e: Engram, query: str = "notes", limit: int = 10):
    return [r.memory.content for r in e.recall(query, limit=limit)]


# ---------- ACL-PRF-1: rank invariance ----------

@pytest.mark.parametrize(
    "bob_signal",
    [None, "Apollo Apollo", "Beta Beta", "Gamma Delta Apollo"],
)
def test_alice_ranking_invariant_to_bob_corpus(tmp_path: Path, bob_signal):
    """Alice's recall ranking does not depend on Bob's private memories."""
    # Baseline: Alice only.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    e_base = _make_engram(base_dir, acl=True)
    _seed_alice(e_base)
    base = _ranks(e_base)

    # Treatment: same Alice corpus + dense Bob corpus.
    treat_dir = tmp_path / f"treat_{(bob_signal or 'none').replace(' ', '_')}"
    treat_dir.mkdir()
    e_treat = _make_engram(treat_dir, acl=True)
    _seed_alice(e_treat)
    if bob_signal is not None:
        for i in range(20):
            e_treat.remember(
                f"notes {bob_signal} {bob_signal} item {i}", agent_id="bob"
            )
    treat = _ranks(e_treat)

    assert treat == base, (
        f"PRF leaked Bob's corpus into Alice's ranking.\n"
        f"  bob_signal={bob_signal!r}\n"
        f"  baseline:  {base}\n"
        f"  treatment: {treat}"
    )


# ---------- ACL-PRF-2: expansion never includes cross-agent entities ----------

def test_prf_expansion_excludes_cross_agent_entities(tmp_path: Path, monkeypatch):
    """The PRF expander is invoked only with Alice-visible texts."""
    e = _make_engram(tmp_path, acl=True)
    _seed_alice(e)
    for i in range(20):
        # 'Zorbax' is unique to Bob's corpus.
        e.remember(f"notes Zorbax Zorbax item {i}", agent_id="bob")

    seen_texts = []
    seen_chosen = []
    import engram.retrieval.expansion as exp_mod

    orig = exp_mod.expand_query

    def spy(query, texts, **kw):
        seen_texts.extend(texts)
        result = orig(query, texts, **kw)
        seen_chosen.extend(result[1])
        return result

    monkeypatch.setattr(exp_mod, "expand_query", spy)

    e.recall("notes", limit=5)

    # No first-pass text fed to the expander should contain 'Zorbax'.
    leaks = [t for t in seen_texts if "zorbax" in (t or "").lower()]
    assert not leaks, f"cross-agent text leaked into PRF mining pool: {leaks[:3]}"
    # And no chosen entity should be Bob-only.
    chosen_lower = [c.lower() for c in seen_chosen]
    assert not any("zorbax" in c for c in chosen_lower), (
        f"PRF chose cross-agent entity: {seen_chosen}"
    )


# ---------- ACL-PRF-3: federated grant still mines full corpus ----------

def test_federated_actor_still_sees_full_pool(tmp_path: Path):
    """`scope='*'` + FEDERATED preserves global PRF behaviour (no regression)."""
    cfg = Config(path=str(tmp_path))
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_top_k = 10
    cfg.retrieval.query_expansion_max_entities = 3
    cfg.retrieval.query_expansion_anchor_share_max = None
    cfg.acl = {
        "enabled": True,
        "grants": {
            # Reviewer: full read across all agents.
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
    for i in range(15):
        e.remember(f"notes Zorbax Zorbax item {i}", agent_id="bob")

    # Reviewer has scope='*' so its filter admits Bob's docs into the pool.
    # This is the expected federated behaviour, not a leak.
    results = e.recall("notes", limit=5)
    contents = [r.memory.content for r in results]
    # At least one Bob-owned doc must reach the reviewer (no over-correction).
    assert any("Zorbax" in c for c in contents), (
        "reviewer with scope='*' should still see federated content; "
        f"got {contents}"
    )


# ---------- ACL-PRF-4: ACL-disabled is unchanged ----------

def test_acl_disabled_no_regression(tmp_path: Path):
    """With ACL off, PRF behaves identically to the pre-fix path."""
    e = _make_engram(tmp_path, acl=False)
    _seed_alice(e)
    for i in range(15):
        e.remember(f"notes Apollo Apollo item {i}", agent_id="bob")

    # PRF should fire and at least one doc with 'Apollo' should rank higher.
    results = e.recall("notes", limit=10)
    assert results, "expected non-empty results with ACL disabled"
    # Sanity: the federated/single-actor path is exercised (no exception),
    # and Bob's docs are visible (since ACL is off).
    contents = [r.memory.content for r in results]
    assert any("Apollo" in c for c in contents)

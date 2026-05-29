"""Adversarial — PRF IDF-rarity ACL side-channel (§D-prf-idf-acl).

Sibling channel to §D-prf-acl. The PRF entity expansion has *two* signals
that can leak cross-agent information:

  1. The mining pool (closed in 07d5c35 by `acl_filter` on first_pass).
  2. The §4.15g IDF-rarity gate, whose df/N is computed across the
     *full* FTS index (`engine._build_prf_rarity_lookup`) —
     including memories the actor cannot READ.

This file pins the closure of channel (2): with `idf_min_rarity` enabled,
the *drop / keep* decision for an Alice-pool entity must be invariant to
Bob's private corpus. Otherwise, an adversary can detect the presence of
specific entities in another agent's memories by populating Bob's corpus
with the suspected entity and observing whether Alice's expansion (and
therefore her ranking on her own docs) changes.

  IDF-PRF-1  Alice's recall ranking is invariant to Bob's private corpus
             when `idf_min_rarity` is set (extends ACL-PRF-1).

  IDF-PRF-2  The rarity score for a candidate entity in Alice's PRF is
             computed only over Alice-visible memories.

  IDF-PRF-3  Federated reader (scope='*') still uses global df.

  IDF-PRF-4  ACL-disabled is unchanged (no regression).
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
    cfg.retrieval.query_expansion_anchor_share_max = None
    # Engage the IDF gate (the channel under test).
    cfg.retrieval.query_expansion_idf_min_rarity = 0.5
    if acl:
        cfg.acl = {
            "enabled": True,
            "grants": {
                "alice": {"permissions": ["read", "write"], "scope": "own"},
                "bob": {"permissions": ["read", "write"], "scope": "own"},
            },
        }
    e = Engram(config=cfg, actor=actor)
    e._firewall.config.max_events_per_minute = 1_000_000
    return e


def _seed_alice(e: Engram) -> None:
    e.remember("notes neutral random", agent_id="alice")
    e.remember("notes summary stuff", agent_id="alice")
    e.remember("notes about project Apollo", agent_id="alice")
    e.remember("notes about project Beta", agent_id="alice")


def _ranks(e: Engram, query: str = "notes", limit: int = 10):
    return [r.memory.content for r in e.recall(query, limit=limit)]


# ---------- IDF-PRF-1: rank invariance under IDF gate ----------

@pytest.mark.parametrize(
    "bob_signal",
    [None, "Apollo Apollo", "Beta Beta", "Apollo Beta Gamma"],
)
def test_alice_ranking_invariant_to_bob_under_idf_gate(tmp_path: Path, bob_signal):
    """Alice's recall ranking stays the same when the IDF gate is on,
    regardless of how Bob's private corpus is populated."""
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
        # 100 dense Bob docs containing the suspected entity. With the
        # pre-fix rarity lookup, these inflate df(Apollo|Beta) and
        # rarity drops below 0.5 — flipping the IDF gate's decision on
        # Alice's pool entity, perturbing Alice's ranking.
        for i in range(50):
            e_treat.remember(
                f"notes {bob_signal} {bob_signal} item {i}", agent_id="bob"
            )
    treat = _ranks(e_treat)

    assert treat == base, (
        f"PRF IDF gate leaked Bob's corpus into Alice's ranking.\n"
        f"  bob_signal={bob_signal!r}\n"
        f"  baseline:  {base}\n"
        f"  treatment: {treat}"
    )


# ---------- IDF-PRF-2: rarity computed only over Alice-visible docs ----------

def test_rarity_lookup_excludes_cross_agent_docs(tmp_path: Path):
    """The rarity callable Alice's PRF uses must score Apollo as rare
    even when Bob has 100 Apollo docs (Alice can't see them, so they
    cannot inflate the df denominator from Alice's perspective)."""
    e = _make_engram(tmp_path, acl=True)
    _seed_alice(e)
    for i in range(50):
        e.remember(f"notes Apollo Apollo item {i}", agent_id="bob")

    eng = e._retrieval

    # Build the rarity lookup the way the engine does for Alice's recall.
    # We thread the same actor-scoped agent allow-list that recall() uses.
    allowed = e._prf_rarity_allowed_agents("alice")
    rarity = eng._build_prf_rarity_lookup(allowed_agents=allowed)
    r_apollo = rarity("Apollo")
    r_zorbax = rarity("Zorbax")

    # Apollo: 1 of Alice's 4 visible docs → df/N = 0.25 → rarity = 0.75.
    # Pre-fix: df/N over global ≈ 101/104 → rarity ≈ 0.029, far below
    # the 0.5 gate. Post-fix: ≥ 0.5.
    assert r_apollo >= 0.5, (
        f"rarity(Apollo) leaked Bob's docs into df: got {r_apollo:.3f}"
    )
    # Zorbax appears nowhere in Alice's corpus → max rarity.
    assert r_zorbax == pytest.approx(1.0)


# ---------- IDF-PRF-3: federated scope='*' uses global df ----------

def test_federated_uses_global_df(tmp_path: Path):
    cfg = Config(path=str(tmp_path))
    cfg.retrieval.query_expansion_idf_min_rarity = 0.5
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_top_k = 10
    cfg.retrieval.query_expansion_max_entities = 3
    cfg.retrieval.query_expansion_anchor_share_max = None
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
    for i in range(50):
        e.remember(f"notes Apollo Apollo item {i}", agent_id="bob")

    allowed = e._prf_rarity_allowed_agents("reviewer")
    assert allowed is None, "scope='*' should bypass agent filter"
    rarity = e._retrieval._build_prf_rarity_lookup(allowed_agents=None)
    # Apollo is now common in the global corpus.
    assert rarity("Apollo") < 0.5


# ---------- IDF-PRF-4: ACL-disabled unchanged ----------

def test_acl_disabled_no_regression(tmp_path: Path):
    e = _make_engram(tmp_path, acl=False)
    _seed_alice(e)
    for i in range(15):
        e.remember(f"notes Apollo Apollo item {i}", agent_id="bob")
    # Sanity: no exception, recall returns results, allow-list is None.
    results = e.recall("notes", limit=10)
    assert results
    assert e._prf_rarity_allowed_agents(e._actor) is None

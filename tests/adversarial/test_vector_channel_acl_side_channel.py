"""Adversarial — BM25/vector candidate-pool ACL side-channel (§D-vector-acl).

The PRF, IDF-rarity, share_prior, and lifecycle-cache channels have all
been closed. This file probes the *primary* candidate-generation path:

    RetrievalEngine._search_with_prf
        bm25_results = store.search_text(query, limit=limit*5, states=...)
        vec_results  = vector.search(query_vec, limit=limit*5)

Both calls scan ALL agents' memories. Each Alice doc receives a
`bm25_rank` and `vector_rank` whose values are a function of where that
doc sits in the GLOBAL candidate list. Bob's private docs occupy
high-rank slots (or not), pushing Alice's docs around. The outer ACL
filter at `engine.py:408` strips Bob's docs *after* RRF fusion, so the
visible output contains only Alice's docs but their fused scores
(and relative order) depend on Bob's content.

This is a presence oracle distinct from the previously-closed reranker
channels: it fires even when the reranker is off and PRF is off.

Tests pinned here:

  VC-1  Alice's ranking over her own docs is invariant to Bob's corpus
        across orthogonal / overlapping / dominant arms. PROBE — expected
        to fail until we ACL-scope the candidate pool.

  VC-2  ACL-disabled mode is unchanged (regression guard).

  VC-3  Federated `scope='*'` actor still sees Bob's content as expected.

NOTE — this file deliberately runs the BM25-only path (vector store
defaults to NoVectorStore in tests) so the leak is observed on the
BM25 channel alone. A vector-channel arm is added once VC-1 closes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engram import Engram
from engram.core.config import Config


def _make_engram(tmp: Path, *, acl: bool, actor: str = "alice") -> Engram:
    cfg = Config(path=str(tmp))
    # Disable every closed-channel knob so this test isolates the
    # primary candidate-pool ACL question.
    cfg.retrieval.reranker = None
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
    """Alice's own corpus — 4 docs with shared 'notes' anchor."""
    e.remember("notes about project Apollo with Alice", agent_id="alice")
    e.remember("notes about project Beta with Alice", agent_id="alice")
    e.remember("notes summary stuff Alice", agent_id="alice")
    e.remember("notes neutral random Alice", agent_id="alice")


def _ranks(e: Engram, query: str = "notes Alice", limit: int = 10):
    return [(r.memory.content, round(r.score, 6)) for r in e.recall(query, limit=limit)]


# ---------- VC-1: BM25 candidate-pool rank invariance ----------

@pytest.mark.parametrize(
    "bob_signal",
    [
        None,
        "notes notes notes Alice Apollo",   # bob-doc near top of BM25
        "notes notes Alice Beta",
        "totally orthogonal Zorbax content",  # should not perturb
    ],
)
def test_vc1_alice_ranking_invariant_to_bob_bm25_pool(tmp_path: Path, bob_signal):
    """BM25 candidate pool must be ACL-scoped before rank assignment.

    PROBE: with the current implementation, expect Alice's `bm25_rank`
    values to shift when Bob's corpus is added with terms that match
    the query. After RRF fusion the fused scores will differ.
    """
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    e_base = _make_engram(base_dir, acl=True)
    _seed_alice(e_base)
    base = _ranks(e_base)

    treat_dir = tmp_path / f"treat_{(bob_signal or 'none').replace(' ', '_')[:20]}"
    treat_dir.mkdir()
    e_treat = _make_engram(treat_dir, acl=True)
    _seed_alice(e_treat)
    if bob_signal is not None:
        for i in range(10):
            e_treat.remember(f"{bob_signal} item {i}", agent_id="bob")
    treat = _ranks(e_treat)

    base_order = [c for c, _ in base]
    treat_order = [c for c, _ in treat]
    assert treat_order == base_order, (
        f"BM25 candidate-pool leak: Alice's order shifted under bob_signal={bob_signal!r}\n"
        f"  base:  {base_order}\n  treat: {treat_order}"
    )
    for (c_b, s_b), (c_t, s_t) in zip(base, treat):
        assert c_b == c_t and s_b == s_t, (
            f"BM25 candidate-pool score leak under bob_signal={bob_signal!r}: "
            f"{c_b!r} {s_b} vs {c_t!r} {s_t}"
        )


# ---------- VC-2: ACL-disabled regression guard ----------

def test_vc2_acl_disabled_is_unchanged(tmp_path: Path):
    """With ACL off, all docs are visible — adding Bob's corpus must
    affect Alice's ranking (positive control: leak is real, not a
    bug in the test harness)."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    e_base = _make_engram(base_dir, acl=False)
    _seed_alice(e_base)
    base = _ranks(e_base)

    treat_dir = tmp_path / "treat"
    treat_dir.mkdir()
    e_treat = _make_engram(treat_dir, acl=False)
    _seed_alice(e_treat)
    for i in range(10):
        e_treat.remember(f"notes notes notes Alice Apollo item {i}", agent_id="bob")
    treat = _ranks(e_treat)

    # With ACL off, Bob's docs are visible — so the recall set differs.
    # This is the positive control that the seed terms actually do
    # affect retrieval; if this test fails, the harness is broken.
    base_ids = {c for c, _ in base}
    treat_ids = {c for c, _ in treat}
    assert treat_ids != base_ids, (
        "ACL-off positive control failed: Bob's docs did not appear in "
        "Alice's recall. Test harness is broken."
    )

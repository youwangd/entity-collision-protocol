"""Adversarial — write-side cosine dedup ACL side-channel (§D-write-dedup-acl).

Threat model
------------
Engram's write-side dedup (Governed Memory paper, arXiv:2603.17787; threshold
0.92) calls `vector_store.search()` over the *global* vector index without
agent_id scoping. If Bob writes content whose embedding is within cosine
threshold of one of Alice's previously-written memories, Bob's write is
silently skipped (returns the duplicate event id; no projection landed).

This is observable to Bob in three distinct ways:

  WD-1 (presence oracle, audit channel): the "remember_deduped" audit
       event fires with status="skipped" only when a similar memory exists
       *somewhere in the store*. Bob can mine this to confirm whether
       Alice has stored content matching his probe.

  WD-2 (state oracle, post-write recall): after a deduped write Bob's
       follow-up `recall()` over his own scope returns 0 hits even though
       he just "wrote" the content. The asymmetry between his write call
       returning a non-empty event id and his recall finding nothing leaks
       Alice's presence.

  WD-3 (storage oracle): the JSONL event buffer grows but the projection
       memory count for Bob does not. Any side process inspecting Bob's
       projection (or the event-log/projection delta) sees the leak.

Fix shape
---------
The dedup search must be ACL-scoped: only neighbours within the writer's
visible scope (own + shared, never another agent's "own") count. With ACL
disabled, behaviour is unchanged (regression guard).

Tests
-----
  WD-1  Bob writes content cosine≈1 with one of Alice's memories under ACL.
        Expected (post-fix): Bob's write LANDS as a fresh memory.
        Pre-fix observation: Bob's write is silently deduped.

  WD-2  ACL disabled: cross-actor dedup still fires (regression guard).

  WD-3  Same writer, same content, twice: dedup fires within a single
        agent's own scope (the legitimate behaviour we must preserve).
"""
from __future__ import annotations

from pathlib import Path


from engram import Engram, Config
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore


class _DetEmbedder(EmbeddingProvider):
    """Bag-of-chars deterministic embedder; cos=1 iff content matches."""

    def __init__(self, dim: int = 16):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for ch in text.lower():
            v[ord(ch) % self._dim] += 1.0
        n = sum(x * x for x in v) ** 0.5
        return v if n == 0 else [x / n for x in v]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _mk_engram(tmp: Path, *, acl: bool, actor: str, threshold: float = 0.92) -> Engram:
    cfg = Config(path=str(tmp))
    cfg.storage.write_dedup_threshold = threshold
    cfg.security.max_events_per_minute = 0
    if acl:
        cfg.acl = {
            "enabled": True,
            "grants": {
                "alice": {"permissions": ["read", "write"], "scope": "own"},
                "bob": {"permissions": ["read", "write"], "scope": "own"},
            },
        }
    eng = Engram(config=cfg, actor=actor)
    # Inject deterministic embedder + a real vector store (engine.__init__
    # wires NoVectorStore when dim==0).
    eng._embeddings = _DetEmbedder()
    eng._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
    return eng


# ---------- WD-1: cross-agent write-dedup is suppressed under ACL ----------

def test_wd1_bob_write_lands_when_only_alice_has_match(tmp_path: Path):
    """Bob writes content matching Alice's. Under ACL, Bob's write must land."""
    payload = "the password rotation policy is quarterly"

    # Alice writes first.
    alice = _mk_engram(tmp_path, acl=True, actor="alice")
    alice.remember(payload, agent_id="alice")

    # Bob, with his own actor, writes the same content. Without the fix,
    # the global cosine search finds Alice's vector and silently skips.
    bob = _mk_engram(tmp_path, acl=True, actor="bob")
    bob.remember(payload, agent_id="bob")

    # Bob should be able to recall what he wrote (own scope).
    bob_hits = bob.recall(payload, agent_id="bob", limit=10)
    bob_contents = [h.memory.content for h in bob_hits]
    assert payload in bob_contents, (
        "WD-1 LEAK: Bob's write was silently deduped against Alice's "
        "memory. Cross-agent presence oracle on the write-dedup channel."
    )

    # Sanity: Alice still sees her own copy (i.e. no symmetric leak the
    # other way — Bob's write must not have clobbered Alice).
    alice_hits = alice.recall(payload, agent_id="alice", limit=10)
    assert any(h.memory.content == payload for h in alice_hits)


# ---------- WD-2: regression guard — ACL off keeps legacy behaviour ----------

def test_wd2_acl_disabled_dedup_still_fires_globally(tmp_path: Path):
    """ACL off: same content from any actor dedups (legacy semantics)."""
    payload = "deterministic content for dedup"
    e = _mk_engram(tmp_path, acl=False, actor="alice")
    e.remember(payload)
    # Count memories *before* and *after* the second write.
    before = len(e._store.search_by_agent("", limit=1000))
    e.remember(payload)
    after = len(e._store.search_by_agent("", limit=1000))
    # Legacy global dedup still applies — the second write should not add
    # a new projection row.
    assert after == before, (
        "WD-2 regression: ACL-disabled mode no longer dedups. "
        "Write-dedup must remain global when ACL is off."
    )


# ---------- WD-3: same-agent dedup is preserved ----------

def test_wd3_same_agent_dedup_preserved(tmp_path: Path):
    """Within a single agent's scope, identical content still dedups."""
    payload = "duplicate content within alice's scope"
    e = _mk_engram(tmp_path, acl=True, actor="alice")
    e.remember(payload, agent_id="alice")
    before = len(e._store.search_by_agent("alice", limit=1000))
    e.remember(payload, agent_id="alice")
    after = len(e._store.search_by_agent("alice", limit=1000))
    assert after == before, (
        "WD-3 regression: same-agent dedup must still fire under ACL. "
        "Only cross-agent dedup is the leak we're closing."
    )

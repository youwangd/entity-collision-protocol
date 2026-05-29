"""Adversarial — write-side cosine dedup ACL channel under WRITE-WRITE RACE.

Threat model (§D-write-dedup-acl, race extension; paper §6.14)
--------------------------------------------------------------
The single-threaded fix in 4e31f32 closes the cross-agent dedup leak by
ACL-filtering candidate neighbours inside `MemoryStore.upsert`. The fix
relies on the candidate's `agent_id` being readable via `self.get(...)`
*at the moment of the dedup decision*.

Under a write-write race two separate `Engram` instances (Alice and Bob)
point at the same store. Each holds its OWN `_dedup_lock` — the lock
does NOT serialise across instances. The order of operations in
`Engine.remember()` is:

    1.  store.upsert()           [does dedup search + ACL filter + INSERT]
    2.  vector.upsert(id, vec)   [registers vector for *future* searches]

so a vector becomes visible to other writers only after step 2. Two
hazards we want to pin behaviourally:

  R1. **Stale-row hazard.** vector_store.search returns Alice's vector,
      but `self.get(r.memory_id)` from Bob's connection sees no row yet.
      Current code path treats `cand is None` as "skip this neighbour"
      → safe (Bob's write lands). Pin this so a future refactor that
      flipped the None-policy to "suppress" is caught.

  R2. **Same-payload Alice×Bob storm.** Concurrent identical writes from
      BOTH actors must yield ≥1 surviving memory per actor (cross-actor
      cannot suppress) AND same-actor dedup must still bound the count.

  R3. **No-leak under contention (presence oracle).** Bob's own-scope
      recall after the storm must return his payload regardless of
      Alice's interleaving — the read-side oracle 4e31f32 closed,
      re-tested under contention.

Deterministic 16-d bag-of-chars embeddings → cosine reproducible.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from engram import Engram, Config
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore

pytestmark = pytest.mark.concurrency


class _DetEmbedder(EmbeddingProvider):
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


def _mk_engram(tmp: Path, *, actor: str, threshold: float = 0.92) -> Engram:
    cfg = Config(path=str(tmp))
    cfg.storage.write_dedup_threshold = threshold
    cfg.security.max_events_per_minute = 0
    cfg.acl = {
        "enabled": True,
        "grants": {
            "alice": {"permissions": ["read", "write"], "scope": "own"},
            "bob": {"permissions": ["read", "write"], "scope": "own"},
        },
    }
    eng = Engram(config=cfg, actor=actor)
    eng._embeddings = _DetEmbedder()
    eng._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
    return eng


# ---------- R2: write-write storm with identical payload --------------------

def test_wdrace_r2_alice_bob_identical_storm_both_survive(tmp_path: Path):
    """Concurrent Alice+Bob writes of the same payload: both actors keep ≥1.

    Cross-actor cannot suppress (§D-write-dedup-acl). Same-actor dedup
    must still fire (loose bound — race slack tolerated).
    """
    payload = "the password rotation policy is quarterly"

    alice = _mk_engram(tmp_path, actor="alice")
    bob = _mk_engram(tmp_path, actor="bob")

    barrier = threading.Barrier(20)

    def alice_write():
        barrier.wait()
        alice.remember(payload, agent_id="alice")

    def bob_write():
        barrier.wait()
        bob.remember(payload, agent_id="bob")

    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = []
        for _ in range(10):
            futs.append(ex.submit(alice_write))
            futs.append(ex.submit(bob_write))
        for f in as_completed(futs):
            f.result()

    alice_rows = [m for m in alice._store.search_by_agent("alice", limit=1000)
                  if m.content == payload]
    bob_rows = [m for m in bob._store.search_by_agent("bob", limit=1000)
                if m.content == payload]
    assert len(alice_rows) >= 1, "WD-RACE-R2: Alice's writes all suppressed under storm"
    assert len(bob_rows) >= 1, "WD-RACE-R2: Bob's writes all suppressed under storm"
    # Same-actor dedup must still fire — bound at 50% of writes per actor.
    assert len(alice_rows) <= 5, (
        f"WD-RACE: same-actor dedup degraded under contention "
        f"(alice has {len(alice_rows)} copies; expected ≤5)"
    )
    assert len(bob_rows) <= 5, (
        f"WD-RACE: same-actor dedup degraded under contention "
        f"(bob has {len(bob_rows)} copies; expected ≤5)"
    )


# ---------- R3: presence oracle under contention ----------------------------

def test_wdrace_r3_recall_oracle_holds_under_contention(tmp_path: Path):
    """Bob's own-scope recall must find his payload regardless of Alice."""
    payload = "alpha bravo charlie delta echo foxtrot"

    alice = _mk_engram(tmp_path, actor="alice")
    bob = _mk_engram(tmp_path, actor="bob")

    barrier = threading.Barrier(2)

    def alice_writer():
        barrier.wait()
        for _ in range(5):
            alice.remember(payload, agent_id="alice")

    def bob_writer():
        barrier.wait()
        for _ in range(5):
            bob.remember(payload, agent_id="bob")

    t1 = threading.Thread(target=alice_writer)
    t2 = threading.Thread(target=bob_writer)
    t1.start(); t2.start()
    t1.join(); t2.join()

    bob_hits = bob.recall(payload, agent_id="bob", limit=10)
    assert any(h.memory.content == payload for h in bob_hits), (
        "WD-RACE-R3 LEAK: under write-write contention Bob's recall sees "
        "nothing — cross-agent dedup leaked through the race window."
    )
    alice_hits = alice.recall(payload, agent_id="alice", limit=10)
    assert any(h.memory.content == payload for h in alice_hits), (
        "WD-RACE-R3 LEAK (symmetric): Alice's recall lost her own payload."
    )


# ---------- R1: stale-row hazard (cand is None) is treated as no-match ------

def test_wdrace_r1_missing_candidate_row_does_not_suppress(tmp_path: Path):
    """If `self.get(id)` returns None for a vector hit, the write must land.

    Pins the current None-policy: an unresolvable candidate is "not a
    duplicate" — the safe direction for the ACL channel. A refactor that
    flipped this to "suppressed" would silently reopen §6.14 whenever a
    row materialised after its vector index entry.
    """
    payload = "stale row hazard payload"

    alice = _mk_engram(tmp_path, actor="alice")
    alice.remember(payload, agent_id="alice")

    bob = _mk_engram(tmp_path, actor="bob")
    bob._store.get = lambda _id: None  # type: ignore[assignment]
    bob.remember(payload, agent_id="bob")

    bob_rows = [m for m in bob._store.search_by_agent("bob", limit=100)
                if m.content == payload]
    assert len(bob_rows) == 1, (
        "WD-RACE-R1 REGRESSION: a missing candidate row caused Bob's write "
        "to be suppressed. None-policy must remain 'skip neighbour, don't "
        "suppress' to keep the ACL channel closed under racy row visibility."
    )

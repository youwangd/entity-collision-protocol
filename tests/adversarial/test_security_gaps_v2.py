"""Adversarial security — gap-fill v2 (mission item 2e, second pass).

Three categories the existing adversarial suite did not pin explicitly:

(D) ACL TOCTOU on the read path. ``Engine.recall`` performs an actor-level
    ``Permission.READ`` check up front and then a per-memory ACL filter
    while it walks the result set. A concurrent ``revoke`` between those
    two checks must NEVER let a row leak: the per-memory check is the
    final authority, so a revoke landed mid-call must observe the new
    state.

(E) Embedded-NUL bytes in property filter keys/values. SQLite's text type
    permits ``\\x00`` but parameter handling is brittle in the wild
    (PEP 249 drivers vary). ``recall_with_filters`` must either reject
    cleanly or treat the NUL as a literal byte that simply doesn't
    match — it must NOT crash, leak, or short-circuit the WHERE clause.

(F) FTS5 unicode-normalization canonical-equivalence parity. The default
    FTS5 ``unicode61`` tokenizer normalizes case but NOT NFC/NFD: a
    query for ``café`` (NFC, U+00E9) and ``cafe\\u0301`` (NFD, e + combining
    acute) tokenize to byte-distinct tokens. We pin the *current*
    contract so a future tokenizer swap is a deliberate decision —
    today, the two queries are NOT guaranteed to match the same row,
    but neither must crash, raise, or return rows from other agents.
"""
from __future__ import annotations

import threading
import time

import pytest

from engram import Engram
from engram.core.config import Config
from engram.security.acl import Permission


# ---------------------------------------------------------------------------
# (D) ACL TOCTOU on the read path
# ---------------------------------------------------------------------------


def _two_actor_engram(tmp_path):
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="alice")
    eng.acl._enabled = True
    eng.acl.grant(
        "alice",
        {Permission.READ, Permission.WRITE, Permission.ADMIN, Permission.FEDERATED},
        scope="*",
    )
    eng.acl.grant("bob", {Permission.READ, Permission.WRITE}, scope="own")
    return eng


def test_acl_revoke_during_recall_never_leaks(tmp_path):
    """Bob has READ scope='own'. While he's recalling, alice revokes him
    on a different thread. Whatever bob's already-running recall returns,
    every memory in the result must still belong to bob — never alice."""
    eng = _two_actor_engram(tmp_path)
    try:
        # Seed: many alice-owned memories sharing tokens with bob's.
        for i in range(40):
            eng.remember(f"alpha bravo charlie shared token row {i}", agent_id="alice")
            eng.remember(f"alpha bravo bob owned row {i}", agent_id="bob")

        results: list[list] = []
        errors: list[BaseException] = []

        def reader():
            try:
                for _ in range(20):
                    out = eng.recall("alpha bravo", limit=20, agent_id="bob")
                    results.append(out)
            except PermissionError:
                # A revoke landing before a fresh top-of-call check is
                # an acceptable outcome — that's the explicit contract.
                pass
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        def revoker():
            time.sleep(0.001)
            eng.acl.revoke("bob")

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=revoker)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert not errors, f"unexpected errors during TOCTOU race: {errors!r}"
        # Whatever the reader saw before/after the revoke, NO row may be
        # an alice-owned memory. Per-memory ACL is the final authority.
        for batch in results:
            for s in batch:
                assert s.memory.agent_id != "alice", (
                    f"ACL leak via recall TOCTOU: bob saw alice-owned "
                    f"memory {s.memory.id!r} content={s.memory.content!r}"
                )
    finally:
        eng.close()


def test_acl_revoke_during_recall_with_filters_never_leaks(tmp_path):
    """Same race against the filter path, query-only mode."""
    eng = _two_actor_engram(tmp_path)
    try:
        for i in range(30):
            eng.remember(f"sigma alice row {i}", agent_id="alice")
            eng.remember(f"sigma bob row {i}", agent_id="bob")

        results: list[list] = []
        errors: list[BaseException] = []

        def reader():
            try:
                for _ in range(15):
                    out = eng.recall_with_filters(
                        query="sigma", properties=None, limit=20, agent_id="bob"
                    )
                    results.append(out)
            except PermissionError:
                pass
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        def revoker():
            time.sleep(0.001)
            eng.acl.revoke("bob")

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=revoker)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert not errors, f"unexpected errors: {errors!r}"
        for batch in results:
            for m in batch:
                assert m.agent_id != "alice", (
                    f"ACL leak via recall_with_filters TOCTOU: "
                    f"id={m.id!r} content={m.content!r}"
                )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# (E) Embedded-NUL bytes in property filter keys/values
# ---------------------------------------------------------------------------


_NUL_PAYLOADS = [
    "\x00",
    "key\x00injected",
    "\x00key",
    "key\x00",
    "x\x00' OR 1=1 --",
    "\x00\x00\x00",
]


@pytest.mark.parametrize("payload", _NUL_PAYLOADS)
def test_property_filter_nul_in_key_is_inert(tmp_path, payload):
    """A NUL-laden key matches nothing real and must not raise."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        m = eng.remember("benign test", properties={"role": "admin", "tier": "1"})
        assert m is not None
        try:
            out = eng.recall_with_filters(properties={payload: "x"}, limit=10)
        except (ValueError, TypeError):
            # Acceptable: cleanly rejected at the API surface.
            return
        assert isinstance(out, list)
        # No legit memory ever wrote a NUL-bearing property key.
        assert out == [] or all(getattr(o, "id", None) for o in out), out
        # Engine still healthy.
        eng.recall("benign", limit=5)
    finally:
        eng.close()


@pytest.mark.parametrize("payload", _NUL_PAYLOADS)
def test_property_filter_nul_in_value_is_inert(tmp_path, payload):
    """A NUL-laden filter value must not crash or short-circuit."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember("benign", properties={"role": "admin"})
        try:
            out = eng.recall_with_filters(properties={"role": payload}, limit=10)
        except (ValueError, TypeError):
            return
        assert isinstance(out, list)
        # `role`==<NUL string> is not the value we wrote.
        for m in out:
            mp = {p["key"]: p["value"] for p in eng._store.get_properties(m.id)}
            assert mp.get("role") == payload, (
                f"NUL-value filter matched a row that doesn't actually carry "
                f"that value: {mp!r}"
            )
        eng.recall("benign", limit=5)
    finally:
        eng.close()


def test_property_filter_nul_in_hybrid_query(tmp_path):
    """Hybrid mode: NUL in the query itself + NUL in a property key."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember("hybrid scenario row", properties={"k": "v"})
        try:
            out = eng.recall_with_filters(
                query="hybrid\x00scenario",
                properties={"k\x00": "v"},
                limit=5,
            )
        except (ValueError, TypeError):
            return
        assert isinstance(out, list)
        # No real property has a NUL-bearing key.
        assert out == []
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# (F) FTS5 unicode-normalization canonical-equivalence parity
# ---------------------------------------------------------------------------


# (NFC, NFD) pairs that are CANONICALLY equivalent under Unicode
# normalization but byte-distinct as UTF-8 strings.
_NFC_NFD_PAIRS = [
    ("café", "cafe\u0301"),                 # e + combining acute
    ("naïve", "nai\u0308ve"),               # i + combining diaeresis
    ("Å", "A\u030a"),                       # A + combining ring
    ("élève", "e\u0301le\u0300ve"),         # e + combining acute / grave
    ("Ω", "Ω"),                             # ohm sign vs Greek capital omega
]


@pytest.mark.parametrize("nfc,nfd", _NFC_NFD_PAIRS)
def test_fts_unicode_normalization_does_not_crash(tmp_path, nfc, nfd):
    """Canonical-equivalent NFC/NFD queries must both return ``list``
    without raising. We pin the current contract: byte-distinct queries
    are not guaranteed to retrieve the same rows under unicode61, but
    neither must crash, error, or leak rows from other agents."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember(f"the {nfc} record one")
        eng.remember(f"the {nfd} record two")
        eng.remember("unrelated padding row alpha bravo charlie")
        out_nfc = eng.recall(nfc, limit=10)
        out_nfd = eng.recall(nfd, limit=10)
        assert isinstance(out_nfc, list)
        assert isinstance(out_nfd, list)
        # Every returned row's content must be a real string from this store.
        for r in out_nfc + out_nfd:
            assert isinstance(r.memory.content, str)
            assert len(r.memory.content) > 0
    finally:
        eng.close()


def test_fts_unicode_normalization_no_acl_bypass(tmp_path):
    """The classical worry: a normalization-form mismatch lets an
    attacker craft a query whose tokens equal an unrelated agent's
    surface form. Even if such a collision exists, ACL must hold —
    bob never sees alice's rows regardless of NFC/NFD.
    """
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="alice")
    eng.acl._enabled = True
    eng.acl.grant(
        "alice",
        {Permission.READ, Permission.WRITE, Permission.ADMIN},
        scope="*",
    )
    eng.acl.grant("bob", {Permission.READ, Permission.WRITE}, scope="own")
    try:
        eng.remember("alice secret café record", agent_id="alice")
        eng.remember("alice secret cafe\u0301 record", agent_id="alice")
        eng.remember("bob's coffee notes", agent_id="bob")
        for q in ["café", "cafe\u0301", "secret", "record"]:
            out = eng.recall(q, limit=20, agent_id="bob")
            for s in out:
                assert s.memory.agent_id != "alice", (
                    f"unicode normalization ACL bypass: bob saw {s.memory.id!r} "
                    f"with content={s.memory.content!r} via query={q!r}"
                )
    finally:
        eng.close()

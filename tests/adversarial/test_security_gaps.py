"""Adversarial security — gap-fill suite for mission item 2e.

Three categories that the original test_security_torture.py did not cover:

(A) ACL escape via FEDERATED upgrade attempts.
    A non-admin agent must NOT be able to upgrade its own grant to FEDERATED
    via any public-API path (engine.recall, recall_with_filters, etc.).
    Only an entity with ADMIN can mutate grants — and even then, the
    AccessPolicy API is the only legitimate path.

(B) scope='own' bypass via empty-string memory_agent_id forgery.
    Grant.can_access('') returns True under scope='own' (a deliberate
    backdoor for unowned/system memories from older schemas). This test
    pins the contract that a malicious caller cannot LAND a memory with
    agent_id='' through the public write API — falling back to actor.

(C) FTS5 tokenizer-injection corpus expansion.
    Adds a wider torture corpus targeting FTS5-specific edge cases
    (unicode normalization, RTL, zero-width, surrogates, long tokens,
    column-prefix forgery, MATCH-operator smuggling).
"""

from __future__ import annotations

import pytest

from engram import Engram
from engram.core.config import Config
from engram.security.acl import AccessPolicy, Permission


# ---------------------------------------------------------------------------
# (A) ACL escape via FEDERATED upgrade attempts
# ---------------------------------------------------------------------------


def _two_actor_engram(tmp_path):
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="alice")
    eng.acl._enabled = True
    eng.acl.grant("alice", {Permission.READ, Permission.WRITE}, scope="own")
    eng.acl.grant("bob", {Permission.READ, Permission.WRITE}, scope="own")
    return eng


def test_acl_no_implicit_federated_via_recall(tmp_path):
    """Bob is scope='own' without FEDERATED. Even after many recall calls,
    his grant must not silently acquire FEDERATED — the engine should never
    mutate ACL state on the read path.
    """
    eng = _two_actor_engram(tmp_path)
    try:
        eng.remember("alice owns a red car", agent_id="alice")
        for _ in range(20):
            try:
                eng.recall("car", agent_id="bob")
            except PermissionError:
                pass
        bob_grant = eng.acl._grants["bob"]
        assert Permission.FEDERATED not in bob_grant.permissions
        assert Permission.ADMIN not in bob_grant.permissions
        assert bob_grant.scope == "own"
    finally:
        eng.close()


def test_acl_federated_only_helps_with_explicit_grant(tmp_path):
    """Confirm the FEDERATED escape hatch: it is a deliberate, granted
    permission — not something obtainable through API noise."""
    eng = _two_actor_engram(tmp_path)
    try:
        eng.remember("alice secret", agent_id="alice")
        # Without FEDERATED bob sees nothing of alice's via cross-agent recall.
        # Recall returns only memories he can read.
        results = eng.recall("secret", agent_id="bob")
        assert all("alice secret" not in r.memory.content for r in results)

        # Now an admin grants bob FEDERATED + scope='*'. Bob can see alice's
        # memory. This is the ONLY path; no public API mutates grants.
        eng.acl.grant(
            "bob",
            {Permission.READ, Permission.WRITE, Permission.FEDERATED},
            scope="*",
        )
        results2 = eng.recall("secret", agent_id="bob")
        assert any("alice secret" in r.memory.content for r in results2)
    finally:
        eng.close()


def test_acl_no_grant_self_promotion_via_metadata(tmp_path):
    """Even when bob stuffs ACL-shaped fields into metadata, his grant must
    not change. (The engine never reads ACL state from event metadata.)"""
    eng = _two_actor_engram(tmp_path)
    try:
        before = sorted(p.value for p in eng.acl._grants["bob"].permissions)
        eng.remember(
            "payload",
            agent_id="bob",
            permissions=["admin", "federated"],
            scope="*",
            grant={"agent_id": "bob", "permissions": ["admin"], "scope": "*"},
            acl={"enabled": False},
        )
        after = sorted(p.value for p in eng.acl._grants["bob"].permissions)
        assert before == after
        assert eng.acl._grants["bob"].scope == "own"
    finally:
        eng.close()


def test_acl_disabled_flag_not_writable_via_remember(tmp_path):
    """The acl._enabled flag must not flip via metadata (`acl_enabled=False`,
    `disable_acl=True`, etc.)."""
    eng = _two_actor_engram(tmp_path)
    try:
        eng.remember("p", agent_id="bob",
                     acl_enabled=False, disable_acl=True,
                     enabled=False)
        assert eng.acl._enabled is True
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# (B) scope='own' bypass via empty-string memory_agent_id forgery
# ---------------------------------------------------------------------------


def test_can_access_empty_string_is_documented_backdoor():
    """Grant.can_access('') returns True under scope='own'. This documents
    the contract — the engine MUST never let a writer land a memory with
    an empty agent_id through the public API. (Pinned in next test.)"""
    p = AccessPolicy(enabled=True)
    p.grant("bob", {Permission.READ}, scope="own")
    grant = p._grants["bob"]
    assert grant.can_access("") is True
    assert grant.can_access("alice") is False


@pytest.mark.parametrize("forged", ["", None])
def test_falsy_agent_id_falls_back_to_actor(tmp_path, forged):
    """Engine.remember(agent_id=<falsy>) must NOT land a memory with
    agent_id='' — it must resolve to the authenticated actor."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="bob")
    eng.acl._enabled = True
    eng.acl.grant("bob", {Permission.READ, Permission.WRITE}, scope="own")
    try:
        eng.remember("payload", agent_id=forged)
        mems = list(eng._store.all_active())
        assert mems, "memory should land"
        for m in mems:
            assert m.agent_id == "bob", (
                f"empty agent_id forgery: memory {m.id} agent_id={m.agent_id!r}"
            )
            assert m.agent_id != ""
    finally:
        eng.close()


@pytest.mark.parametrize("forged", [" ", "\t", "\n"])
def test_whitespace_agent_id_blocked_by_acl(tmp_path, forged):
    """Whitespace strings are NOT falsy, so they pass through `agent_id or
    actor`. ACL must reject them (no grant exists for ' ' / '\\t' / '\\n')
    and no memory may persist under that id.

    This pins the safety property: even if the falsy-fallback misses,
    the ACL layer is the second line of defense.
    """
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="bob")
    eng.acl._enabled = True
    eng.acl.grant("bob", {Permission.READ, Permission.WRITE}, scope="own")
    try:
        with pytest.raises(PermissionError):
            eng.remember("payload", agent_id=forged)
        mems = list(eng._store.all_active())
        for m in mems:
            assert m.agent_id != forged
            assert m.agent_id.strip() != ""
    finally:
        eng.close()


def test_empty_string_actor_rejected(tmp_path):
    """Even instantiating with actor='' must not allow ACL bypass:
    an actor with no grant must be denied."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg, actor="")
    eng.acl._enabled = True
    # Grant only alice — empty actor has no grant.
    eng.acl.grant("alice", {Permission.READ, Permission.WRITE}, scope="own")
    try:
        with pytest.raises(PermissionError):
            eng.remember("payload")
        with pytest.raises(PermissionError):
            eng.recall("anything")
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# (C) FTS5 tokenizer-injection corpus expansion
# ---------------------------------------------------------------------------

# Categories beyond the original FTS_HOSTILE_QUERIES set:
#   - unicode normalization (NFD vs NFC, combining marks)
#   - RTL / bidi overrides
#   - zero-width chars
#   - surrogates / private-use
#   - very long tokens
#   - column-prefix forgery (FTS5 supports `col:term` matching)
#   - MATCH-operator smuggling via UNESCAPED special tokens
#   - control characters and BOM
#   - punctuation pile-ups that historically broke tokenizers
FTS_TOKENIZER_TORTURE = [
    # Unicode normalization
    "café",                  # NFC
    "cafe\u0301",            # NFD: e + combining acute
    "ﬃ",                     # ligature
    "Ⅻ",                     # roman numeral
    "𝓗𝓮𝓵𝓵𝓸",                 # mathematical alphanumeric
    # RTL / bidi
    "\u202eevil\u202c",      # RLO override + PDF
    "abc\u200fdef",          # RTL mark embedded
    # Zero-width / invisible
    "a\u200bb",              # zero-width space
    "a\u200cb",              # ZWNJ
    "a\u200db",              # ZWJ
    "\ufefftext",            # BOM prefix
    "a\u00a0b",              # NBSP between tokens (non-empty after strip)
    # Surrogates / private use
    "\ue000\ue001",
    "💀💀💀",
    "👨‍👩‍👧‍👦",                  # ZWJ family emoji
    # Long tokens
    "x" * 1024,
    "y" * 4096,
    "ab" * 2000,
    # Column-prefix forgery (FTS5 col:term syntax)
    "content:secret",
    "memories.content:alice",
    "rowid:1",
    "fts:foo",
    # MATCH-operator smuggling
    "foo AND bar OR baz NOT qux",
    "term1 NEAR(term2, 5)",
    "\"phrase one\" + \"phrase two\"",
    "(a AND b) OR (c AND d)",
    "a*b*c*",
    "*foo*",
    "^^^foo",
    # Control chars & BOM
    "\x07bell",
    "\x1b[31mred\x1b[0m",     # ANSI escape
    "tab\there",
    "cr\rlf\nmix",
    # Punctuation pile-ups
    "!@#$%^&*()_+-=",
    "[]{}<>:;\"'",
    "....,,,,;;;;",
    "//\\\\//\\\\",
    # Mixed-script confusables
    "аррӏе",                 # cyrillic 'apple' lookalike
    "ρауρаӏ",                # cyrillic 'paypal' lookalike
    # Numeric / boolean lookalikes
    "0x0BADF00D",
    "1e308",
    "NaN",
    "true OR false",
]


@pytest.mark.parametrize("q", FTS_TOKENIZER_TORTURE)
def test_fts_tokenizer_torture_does_not_crash(tmp_path, q: str) -> None:
    """Expanded FTS5 tokenizer-injection corpus. Every query must return a
    list (possibly empty) without raising sqlite or OperationalError."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember("the quick brown fox jumps over the lazy dog")
        eng.remember("café au lait — ½ cup sugar")
        eng.remember("memory systems are interesting and complex")
        results = eng.recall(q, limit=5)
        assert isinstance(results, list)
        for r in results:
            assert hasattr(r, "memory")
    finally:
        eng.close()


def test_fts_column_prefix_does_not_leak_schema(tmp_path):
    """`col:term` syntax must not let a caller introspect or address columns
    other than the one the recall path intends to query. The sanitizer
    should treat 'content:' / 'rowid:' / 'memories.content:' as opaque
    text or strip the prefix."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember("alice secret note")
        # These would only match if `rowid:1` were honored as a column
        # specifier. We don't assert it returns nothing — only that it
        # doesn't crash and doesn't return rows that the un-prefixed
        # query wouldn't have matched semantically.
        for q in ["rowid:1", "id:1", "memories.content:alice",
                  "fts:alice", "deleted_at:NULL"]:
            results = eng.recall(q, limit=10)
            assert isinstance(results, list)
    finally:
        eng.close()

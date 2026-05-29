"""Tests for G9 (Access Control) and G10 (Encryption at rest)."""

import pytest
from engram import Engram, Config
from engram.security.acl import AccessPolicy, Permission
from engram.security.encryption import ContentEncryptor, EncryptionError


# ── ACL Unit Tests ──


class TestAccessPolicy:
    def test_disabled_policy_allows_everything(self):
        policy = AccessPolicy(enabled=False)
        # Should not raise
        policy.check("anyone", Permission.READ)
        policy.check("anyone", Permission.WRITE)
        policy.check("anyone", Permission.ADMIN)

    def test_enabled_policy_denies_unknown_agent(self):
        policy = AccessPolicy(enabled=True)
        with pytest.raises(PermissionError, match="no access grants"):
            policy.check("unknown-agent", Permission.READ)

    def test_grant_and_check(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-evan", {"read", "write", "forget"}, scope="own")
        policy.check("agent-evan", Permission.READ)
        policy.check("agent-evan", Permission.WRITE)
        policy.check("agent-evan", Permission.FORGET)

    def test_missing_permission_denied(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-reader", {"read"}, scope="own")
        with pytest.raises(PermissionError, match="lacks 'write'"):
            policy.check("agent-reader", Permission.WRITE)

    def test_admin_grants_all_permissions(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("admin-user", {"admin"}, scope="*")
        # Admin should pass any permission check
        policy.check("admin-user", Permission.READ)
        policy.check("admin-user", Permission.WRITE)
        policy.check("admin-user", Permission.FORGET)
        policy.check("admin-user", Permission.CONSOLIDATE)
        policy.check("admin-user", Permission.EXPORT)

    def test_own_scope_denies_cross_agent(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-a", {"read"}, scope="own")
        # Can read own memories
        policy.check("agent-a", Permission.READ, memory_agent_id="agent-a")
        # Can read unowned (agent_id="") memories
        policy.check("agent-a", Permission.READ, memory_agent_id="")
        # Cannot read another agent's memories
        with pytest.raises(PermissionError, match="cannot access"):
            policy.check("agent-a", Permission.READ, memory_agent_id="agent-b")

    def test_star_scope_allows_cross_agent(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("supervisor", {"read"}, scope="*")
        policy.check("supervisor", Permission.READ, memory_agent_id="agent-a")
        policy.check("supervisor", Permission.READ, memory_agent_id="agent-b")

    def test_federated_permission_overrides_own_scope(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-a", {"read", "federated"}, scope="own")
        # Federated allows cross-agent even with own scope
        policy.check("agent-a", Permission.READ, memory_agent_id="agent-b")

    def test_revoke(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-x", {"read", "write"}, scope="own")
        policy.check("agent-x", Permission.READ)
        policy.revoke("agent-x")
        with pytest.raises(PermissionError):
            policy.check("agent-x", Permission.READ)

    def test_list_grants(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-a", {"read", "write"}, scope="own")
        policy.grant("agent-b", {"read"}, scope="*")
        grants = policy.list_grants()
        assert len(grants) == 2
        ids = {g["agent_id"] for g in grants}
        assert ids == {"agent-a", "agent-b"}

    def test_serialization_roundtrip(self):
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-a", {"read", "write"}, scope="own")
        policy.grant("admin", {"admin"}, scope="*")
        d = policy.to_dict()
        restored = AccessPolicy.from_dict(d)
        assert restored.enabled
        restored.check("agent-a", Permission.READ)
        restored.check("admin", Permission.WRITE)

    def test_design_example(self):
        """The exact example from DESIGN.md §5.3."""
        policy = AccessPolicy(enabled=True)
        policy.grant("agent-evan", {"read", "write", "forget"}, scope="own")
        policy.grant("agent-reviewer", {"read"}, scope="*")
        policy.grant("user-richard", {"read", "write", "forget", "consolidate", "admin", "export"}, scope="*")

        # Evan can read/write own
        policy.check("agent-evan", Permission.READ, memory_agent_id="agent-evan")
        policy.check("agent-evan", Permission.WRITE)
        # Evan cannot read reviewer's
        with pytest.raises(PermissionError):
            policy.check("agent-evan", Permission.READ, memory_agent_id="agent-reviewer")

        # Reviewer can read everyone's
        policy.check("agent-reviewer", Permission.READ, memory_agent_id="agent-evan")
        # Reviewer cannot write
        with pytest.raises(PermissionError):
            policy.check("agent-reviewer", Permission.WRITE)

        # Richard can do everything
        policy.check("user-richard", Permission.CONSOLIDATE)
        policy.check("user-richard", Permission.EXPORT)
        policy.check("user-richard", Permission.READ, memory_agent_id="agent-evan")


# ── Encryption Unit Tests ──


class TestContentEncryptor:
    def test_disabled_passthrough(self):
        enc = ContentEncryptor(enabled=False)
        assert not enc.enabled
        assert enc.encrypt("hello") == "hello"
        assert enc.decrypt("hello") == "hello"

    def test_encrypt_decrypt_roundtrip(self):
        enc = ContentEncryptor(enabled=True, key="test-passphrase-12345", key_source="direct")
        assert enc.enabled
        ct = enc.encrypt("secret data")
        assert ct.startswith("enc:")
        assert ct != "secret data"
        pt = enc.decrypt(ct)
        assert pt == "secret data"

    def test_plaintext_passthrough_on_decrypt(self):
        """Decrypt returns plaintext as-is if no 'enc:' prefix (mixed store support)."""
        enc = ContentEncryptor(enabled=True, key="test-key", key_source="direct")
        assert enc.decrypt("not encrypted") == "not encrypted"

    def test_wrong_key_fails(self):
        enc1 = ContentEncryptor(enabled=True, key="key-one", key_source="direct")
        ct = enc1.encrypt("secret")
        enc2 = ContentEncryptor(enabled=True, key="key-two", key_source="direct")
        with pytest.raises(EncryptionError, match="wrong key"):
            enc2.decrypt(ct)

    def test_encrypted_without_key_fails(self):
        """Trying to decrypt encrypted content without encryption enabled."""
        enc1 = ContentEncryptor(enabled=True, key="test-key", key_source="direct")
        ct = enc1.encrypt("secret")
        enc_disabled = ContentEncryptor(enabled=False)
        with pytest.raises(EncryptionError, match="not enabled"):
            enc_disabled.decrypt(ct)

    def test_env_key_source(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_ENCRYPTION_KEY", "env-test-key-123")
        enc = ContentEncryptor(enabled=True, key_source="env")
        assert enc.enabled
        ct = enc.encrypt("hello")
        assert enc.decrypt(ct) == "hello"

    def test_no_key_disables(self):
        """If no key available, encryption gracefully disables."""
        # Clear env var if set
        enc = ContentEncryptor(enabled=True, key_source="direct", key=None)
        assert not enc.enabled

    def test_fernet_key_direct(self):
        """Direct Fernet key (44 chars base64)."""
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key().decode()
            enc = ContentEncryptor(enabled=True, key=key, key_source="direct")
            assert enc.enabled
            ct = enc.encrypt("test")
            assert enc.decrypt(ct) == "test"
        except ImportError:
            pytest.skip("cryptography not installed")

    def test_generate_key(self):
        try:
            key = ContentEncryptor.generate_key()
            assert len(key) == 44  # Fernet key length
        except EncryptionError:
            pytest.skip("cryptography not installed")


# ── Integration: ACL + Engine ──


class TestACLIntegration:
    def test_engine_without_acl(self, tmp_path):
        """Default: ACL disabled, everything works."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("test fact")
            results = mem.recall("test")
            assert len(results) > 0
        finally:
            mem.close()

    def test_engine_remember_sets_agent_id(self, tmp_path):
        """Remember should set agent_id on the memory."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config, actor="agent-evan")
        try:
            mem.remember("evan's fact", agent_id="agent-evan")
            results = mem.recall("evan")
            assert results[0].memory.agent_id == "agent-evan"
        finally:
            mem.close()


# ── Integration: Encryption + Engine ──


class TestEncryptionIntegration:
    def test_engine_with_encryption(self, tmp_path, monkeypatch):
        """End-to-end: remember with encryption, JSONL is encrypted, SQLite stays plaintext for FTS."""
        monkeypatch.setenv("ENGRAM_ENCRYPTION_KEY", "integration-test-key")
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.encrypt_at_rest = True
        config.security.encryption_key_source = "env"
        mem = Engram(config)
        try:
            mem.remember("top secret database password is hunter2")
            # SQLite stores PLAINTEXT (projection for FTS search)
            raw = mem._store.all_active()
            assert raw[0].content == "top secret database password is hunter2"
            # JSONL stores ENCRYPTED (source of truth at rest)
            with open(tmp_path / "events.jsonl") as f:
                import json
                line = f.readline().strip()
                data = json.loads(line)
                assert data["content"].startswith("enc:")
            # Recall should work (FTS searches plaintext)
            results = mem.recall("database password")
            assert "hunter2" in results[0].memory.content
        finally:
            mem.close()

    def test_engine_without_encryption(self, tmp_path):
        """Default: no encryption, content is plaintext."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("plaintext fact")
            raw = mem._store.all_active()
            assert raw[0].content == "plaintext fact"
        finally:
            mem.close()

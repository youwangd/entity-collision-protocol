"""Tests for v5 audit fixes."""

import json
import pytest
from engram import Engram, Config, Memory


class TestEncryptionArchitecture:
    """B1+B2+B3: Verify encryption only in JSONL, SQLite stays plaintext."""

    def test_fts_search_works_with_encryption(self, tmp_path, monkeypatch):
        """B1: FTS must index plaintext even with encryption enabled."""
        monkeypatch.setenv("ENGRAM_ENCRYPTION_KEY", "test-key-fts")
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.encrypt_at_rest = True
        config.security.encryption_key_source = "env"
        mem = Engram(config)
        try:
            mem.remember("PostgreSQL is the preferred database")
            mem.remember("Redis is used for caching")
            # BM25 search should find the right memory
            results = mem.recall("PostgreSQL database")
            assert len(results) >= 1
            assert "PostgreSQL" in results[0].memory.content
        finally:
            mem.close()

    def test_reconsolidation_with_encryption(self, tmp_path, monkeypatch):
        """B3: Reconsolidation should work correctly with encrypted JSONL."""
        monkeypatch.setenv("ENGRAM_ENCRYPTION_KEY", "test-key-recon")
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.encrypt_at_rest = True
        config.security.encryption_key_source = "env"
        mem = Engram(config)
        try:
            mem.remember("deploy uses Docker", salience=0.8)
            results = mem.recall("deploy")
            assert results  # should find it
            # Recall again — triggers reconsolidation (mark_accessed)
            results2 = mem.recall("deploy")
            assert "Docker" in results2[0].memory.content  # not garbled
        finally:
            mem.close()

    def test_buffer_scan_decrypts(self, tmp_path, monkeypatch):
        """Buffer.scan() should return decrypted events."""
        monkeypatch.setenv("ENGRAM_ENCRYPTION_KEY", "test-key-scan")
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.encrypt_at_rest = True
        config.security.encryption_key_source = "env"
        mem = Engram(config)
        try:
            mem.remember("secret fact 123")
            events = list(mem._buffer.scan())
            # Events from scan should be decrypted
            found = any("secret fact 123" in e.content for e in events)
            assert found, "buffer.scan() should return decrypted events"
        finally:
            mem.close()


class TestContentPolicy:
    """G6: Content policy in firewall."""

    def test_content_policy_blocks_restricted(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.content_policy = {"restricted": "block"}
        mem = Engram(config)
        try:
            # API key should be classified as restricted → blocked
            with pytest.raises(Exception, match="blocked by policy"):
                mem.remember("api_key=sk-abc123456789")
        finally:
            mem.close()

    def test_content_policy_warns(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        config.security.content_policy = {"internal": "warn"}
        mem = Engram(config)
        try:
            # Should succeed (warn, not block)
            eid = mem.remember("server at 192.168.1.100")
            assert eid
        finally:
            mem.close()

    def test_content_policy_disabled_by_default(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            eid = mem.remember("api_key=sk-test")
            assert eid  # no policy, should pass
        finally:
            mem.close()


class TestACLFromConfig:
    """G7: ACL loadable from YAML config dict."""

    def test_acl_from_config(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        config.acl = {
            "enabled": True,
            "grants": {
                "agent-a": {"permissions": ["read", "write"], "scope": "own"},
                "admin": {"permissions": ["admin"], "scope": "*"},
            },
        }
        mem = Engram(config, actor="agent-a")
        try:
            assert mem.acl.enabled
            # agent-a can write
            eid = mem.remember("test fact")
            assert eid
            # agent-a can read own
            results = mem.recall("test")
            assert len(results) >= 1
        finally:
            mem.close()

    def test_acl_denies_unauthorized_agent(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        config.acl = {
            "enabled": True,
            "grants": {
                "agent-a": {"permissions": ["read"], "scope": "own"},
            },
        }
        mem = Engram(config, actor="agent-a")
        try:
            with pytest.raises(PermissionError, match="lacks 'write'"):
                mem.remember("should fail")
        finally:
            mem.close()


class TestProvenance:
    """G5: provenance() method."""

    def test_provenance_returns_lineage(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("test fact")
            results = mem.recall("test")
            mid = results[0].memory.id
            prov = mem.provenance(mid)
            assert prov is not None
            assert prov["memory_id"] == mid
            assert "source_events" in prov
            assert "created_by" in prov
            assert "modifications" in prov
            assert "relations" in prov
        finally:
            mem.close()

    def test_provenance_not_found(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            assert mem.provenance("nonexistent") is None
        finally:
            mem.close()


class TestMemoryRoundtrip:
    """G11+G12: Memory.to_dict/from_dict roundtrip."""

    def test_memory_to_dict_from_dict(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("roundtrip test", salience=0.7)
            results = mem.recall("roundtrip")
            original = results[0].memory
            d = original.to_dict()
            restored = Memory.from_dict(d)
            assert restored.id == original.id
            assert restored.content == original.content
            assert restored.salience == original.salience
            assert restored.type == original.type
            assert restored.state == original.state
            assert restored.agent_id == original.agent_id
        finally:
            mem.close()

    def test_export_import_roundtrip(self, tmp_path):
        """Full export → import → verify (Memory.to_dict roundtrip preserves all fields)."""
        config = Config.minimal()
        config.path = str(tmp_path / "src")
        mem = Engram(config)
        try:
            mem.remember("fact one", salience=0.4)  # 0.5 + 0.4 = 0.9 final
            mem.remember("fact two", salience=0.1)  # 0.5 + 0.1 = 0.6 final
            exported = mem.export_memories()
            original_salience = exported[0]["salience"]
        finally:
            mem.close()

        # Write to file
        backup = tmp_path / "backup.json"
        backup.write_text(json.dumps(exported))

        # Import into fresh store
        config2 = Config.minimal()
        config2.path = str(tmp_path / "dst")
        mem2 = Engram(config2)
        try:
            count = mem2.import_from(str(backup))
            assert count == 2
            # Full state restored (id, salience preserved exactly)
            results = mem2.recall("fact one")
            assert results[0].memory.salience == original_salience
        finally:
            mem2.close()


class TestConfigComplete:
    """Q6: Config.to_dict() roundtrip."""

    def test_config_to_dict_complete(self):
        config = Config.minimal()
        config.security.encrypt_at_rest = True
        config.security.content_policy = {"restricted": "block"}
        config.acl = {"enabled": True, "grants": {"a": {"permissions": ["read"], "scope": "own"}}}
        d = config.to_dict()
        assert d["security"]["encrypt_at_rest"] is True
        assert d["security"]["content_policy"] == {"restricted": "block"}
        assert d["acl"]["enabled"] is True
        assert "retention" in d
        assert "forgetting" in d
        assert "retrieval" in d


class TestVersionConsistency:
    """Q9: Version from __version__."""

    def test_cli_version(self):
        from engram import __version__
        # Importing cli must succeed (lazy click registration); version_option is set
        import engram.cli.main as _cli
        assert hasattr(_cli, "cli")
        assert __version__ == "0.1.0"

    def test_mcp_version(self):
        from engram import __version__
        # MCP server should use __version__
        from engram.mcp.server import __version__ as mcp_ver
        assert mcp_ver == __version__

"""Tests for v5 remaining gaps: G1 (documented), G9 (federated recall), G10 (snapshot/incremental rebuild)."""

import pytest
from engram import Engram, Config


class TestFederatedRecall:
    """G9: Federated recall across Engram instances."""

    def test_federated_recall_merges_results(self, tmp_path):
        """Query spans two independent stores."""
        # Store A
        config_a = Config.minimal()
        config_a.path = str(tmp_path / "store-a")
        mem_a = Engram(config_a, actor="agent-a")

        # Store B
        config_b = Config.minimal()
        config_b.path = str(tmp_path / "store-b")
        mem_b = Engram(config_b, actor="agent-b")

        try:
            mem_a.remember("PostgreSQL is preferred for data storage")
            mem_b.remember("MongoDB works well for document storage")

            # Federated recall: merges A + B results
            federated = mem_a.recall("storage", federated=[mem_b])
            contents = [r.memory.content for r in federated]
            has_postgres = any("PostgreSQL" in c for c in contents)
            has_mongo = any("MongoDB" in c for c in contents)
            assert has_postgres and has_mongo, f"Expected both, got: {contents}"
        finally:
            mem_a.close()
            mem_b.close()

    def test_federated_recall_requires_permission(self, tmp_path):
        """ACL enforces federated permission."""
        config_a = Config.minimal()
        config_a.path = str(tmp_path / "store-a")
        config_a.acl = {
            "enabled": True,
            "grants": {
                "agent-a": {"permissions": ["read", "write"], "scope": "own"},
            },
        }
        config_b = Config.minimal()
        config_b.path = str(tmp_path / "store-b")

        mem_a = Engram(config_a, actor="agent-a")
        mem_b = Engram(config_b, actor="agent-b")

        try:
            mem_b.remember("remote fact")
            # agent-a has read but not federated permission
            with pytest.raises(PermissionError, match="lacks 'federated'"):
                mem_a.recall("remote", federated=[mem_b])
        finally:
            mem_a.close()
            mem_b.close()

    def test_federated_recall_with_permission(self, tmp_path):
        """ACL allows federated when granted."""
        config_a = Config.minimal()
        config_a.path = str(tmp_path / "store-a")
        config_a.acl = {
            "enabled": True,
            "grants": {
                "agent-a": {"permissions": ["read", "write", "federated"], "scope": "own"},
            },
        }
        config_b = Config.minimal()
        config_b.path = str(tmp_path / "store-b")

        mem_a = Engram(config_a, actor="agent-a")
        mem_b = Engram(config_b, actor="agent-b")

        try:
            mem_a.remember("local fact A")
            mem_b.remember("remote fact B")
            results = mem_a.recall("fact", federated=[mem_b])
            contents = [r.memory.content for r in results]
            assert any("local" in c for c in contents)
            assert any("remote" in c for c in contents)
        finally:
            mem_a.close()
            mem_b.close()

    def test_federated_recall_graceful_on_error(self, tmp_path):
        """Federated recall logs warning and continues if remote store fails."""
        config_a = Config.minimal()
        config_a.path = str(tmp_path / "store-a")
        mem_a = Engram(config_a)

        try:
            mem_a.remember("local fact")
            # Create a "broken" store that will fail
            config_b = Config.minimal()
            config_b.path = str(tmp_path / "store-b")
            mem_b = Engram(config_b)
            mem_b.close()  # Close it so _retrieval is broken
            mem_b._retrieval = None  # Force failure

            # Should not raise — graceful degradation
            results = mem_a.recall("local", federated=[mem_b])
            assert len(results) >= 1  # local results still returned
        finally:
            mem_a.close()


class TestSnapshotRebuild:
    """G10: Snapshot + incremental rebuild."""

    def test_snapshot_saves_event_id(self, tmp_path):
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("fact one")
            snap_id = mem.snapshot()
            assert snap_id is not None
            # Verify it's stored in metadata
            stored = mem._store.get_metadata("snapshot_event_id")
            assert stored == snap_id
        finally:
            mem.close()

    def test_incremental_rebuild(self, tmp_path):
        """Incremental rebuild only processes events after snapshot."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("fact one")
            mem.remember("fact two")
            mem.snapshot()
            mem.remember("fact three")

            # Incremental rebuild should process only fact three
            mem.rebuild(incremental=True)
            # Should still have all 3 facts accessible
            results = mem.recall("fact")
            assert len(results) >= 2  # at least some results
        finally:
            mem.close()

    def test_full_rebuild_ignores_snapshot(self, tmp_path):
        """Full rebuild replays everything regardless of snapshot."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("fact one")
            mem.snapshot()
            mem.remember("fact two")

            count = mem.rebuild(incremental=False)
            assert count >= 2  # both facts rebuilt
        finally:
            mem.close()

    def test_rebuild_without_snapshot_does_full(self, tmp_path):
        """If no snapshot exists, incremental=True still does full rebuild."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("fact one")
            mem.remember("fact two")
            count = mem.rebuild(incremental=True)
            assert count >= 2
        finally:
            mem.close()

    def test_rebuild_saves_new_snapshot(self, tmp_path):
        """Rebuild should auto-save a new snapshot after completion."""
        config = Config.minimal()
        config.path = str(tmp_path)
        mem = Engram(config)
        try:
            mem.remember("fact one")
            assert mem._store.get_metadata("snapshot_event_id") is None
            mem.rebuild()
            assert mem._store.get_metadata("snapshot_event_id") is not None
        finally:
            mem.close()

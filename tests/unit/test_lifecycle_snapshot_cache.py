"""Unit tests for `CachedLifecycleSnapshot` (NEXT.md item #1).

The gate-replay amortization probe: validate that the cache keyed on
`(path, size, mtime_ns)` (a) returns equal snapshots to the
uncached path, (b) reuses the cached snapshot when the buffer is
unchanged, and (c) invalidates correctly when the buffer grows or is
externally rewritten.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from engram.consolidation.lifecycle_projection import (
    CachedLifecycleSnapshot,
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import EventKind, SchemaStatus
from engram.store.buffer import JSONLBufferStore


@pytest.fixture
def buffer(tmp_path: Path) -> JSONLBufferStore:
    return JSONLBufferStore(base_path=tmp_path)


def _seed(buf: JSONLBufferStore, n_schemas: int = 5, n_extra: int = 0) -> None:
    for i in range(n_schemas):
        buf.append(make_lifecycle_event(
            schema_id=f"s{i}", kind=EventKind.CREATE,
        ))
    for i in range(n_extra):
        buf.append(make_lifecycle_event(
            schema_id=f"s{i % n_schemas}", kind=EventKind.DEPRECATE,
            window_id=f"w{i}",
        ))


class TestCorrectness:
    def test_empty_buffer_returns_empty_snapshot(self, buffer):
        c = CachedLifecycleSnapshot()
        assert c.get(buffer) == {}

    def test_matches_uncached_snapshot(self, buffer):
        _seed(buffer, n_schemas=4, n_extra=3)
        c = CachedLifecycleSnapshot()
        assert c.get(buffer) == snapshot_from_buffer(buffer)

    def test_deprecated_set_extracted_correctly(self, buffer):
        _seed(buffer, n_schemas=3, n_extra=2)
        c = CachedLifecycleSnapshot()
        snap = c.get(buffer)
        deprecated = {sid for sid, st in snap.items()
                      if st.status is SchemaStatus.DEPRECATED}
        assert deprecated == {"s0", "s1"}


class TestCacheBehavior:
    def test_first_call_is_miss(self, buffer):
        _seed(buffer)
        c = CachedLifecycleSnapshot()
        c.get(buffer)
        assert c.stats["misses"] == 1
        assert c.stats["hits"] == 0

    def test_repeated_call_unchanged_buffer_is_hit(self, buffer):
        _seed(buffer)
        c = CachedLifecycleSnapshot()
        c.get(buffer)
        for _ in range(5):
            c.get(buffer)
        assert c.stats["hits"] == 5
        assert c.stats["misses"] == 1
        assert c.stats["partial_hits"] == 0

    def test_cached_snapshot_object_is_reused(self, buffer):
        _seed(buffer)
        c = CachedLifecycleSnapshot()
        s1 = c.get(buffer)
        s2 = c.get(buffer)
        # Same dict object (cache returns reference, not copy)
        assert s1 is s2

    def test_lifecycle_append_is_partial_hit_not_miss(self, buffer):
        _seed(buffer, n_schemas=3)
        c = CachedLifecycleSnapshot()
        first = c.get(buffer)
        assert len(first) == 3
        buffer.append(make_lifecycle_event(
            schema_id="s_new", kind=EventKind.CREATE,
        ))
        second = c.get(buffer)
        assert "s_new" in second
        # The append grew the file so partial_hits increments,
        # not misses (no full rebuild).
        assert c.stats == {"hits": 0, "partial_hits": 1, "misses": 1}

    def test_non_lifecycle_append_does_not_change_snapshot(self, buffer):
        """Append a RECALL_REQUEST event between two get() calls. The
        cache's incremental scan reads the new bytes but folds nothing
        (wrong event_type), so the snapshot identity is preserved."""
        from datetime import datetime, timezone
        from engram.core.types import Event, EventType, generate_event_id

        _seed(buffer, n_schemas=2)
        c = CachedLifecycleSnapshot()
        s1 = c.get(buffer)
        buffer.append(Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.RECALL_REQUEST,
            content="some query",
            metadata={},
        ))
        s2 = c.get(buffer)
        # Same content (no lifecycle change). partial_hits incremented.
        assert s2 == s1
        assert c.stats["partial_hits"] == 1
        assert c.stats["misses"] == 1

    def test_invalidate_method_forces_reread(self, buffer):
        _seed(buffer)
        c = CachedLifecycleSnapshot()
        c.get(buffer)
        c.invalidate()
        c.get(buffer)
        assert c.stats == {"hits": 0, "partial_hits": 0, "misses": 2}

    def test_truncation_below_offset_triggers_full_rebuild(self, buffer):
        _seed(buffer, n_schemas=3, n_extra=5)
        c = CachedLifecycleSnapshot()
        c.get(buffer)
        # Truncate the file to 0 bytes; the next get() must detect the
        # shrink and rebuild (returning empty).
        with open(buffer.path, "wb"):
            pass
        snap = c.get(buffer)
        assert snap == {}
        assert c.stats["misses"] == 2

    def test_missing_path_falls_back_to_replay(self, tmp_path):
        """If stat fails, get() should still return a (possibly empty)
        snapshot via the replay path rather than crashing."""
        buf = JSONLBufferStore(base_path=tmp_path)
        c = CachedLifecycleSnapshot()
        try:
            os.remove(buf.path)
        except OSError:
            pass
        snap = c.get(buf)
        assert snap == {}


class TestRotationTombstoneRegression:
    """Regression for hypothesis falsifying example
    (test_c1_random_interleave_equivalence): when the first ``get()``
    full-rebuilt against an empty file, ``_prefix`` was captured as
    ``b""``. Subsequent incremental folds advanced ``_offset`` without
    arming the rotation tombstone, so a later inode-recycling rotate +
    append would silently return the stale snapshot. Fixed by
    refreshing ``_prefix`` lazily in the incremental path whenever the
    cached probe is shorter than what's now observable.
    """

    def test_empty_prefix_then_append_then_rotate_returns_fresh(self, tmp_path):
        buf = JSONLBufferStore(base_path=tmp_path)
        c = CachedLifecycleSnapshot()

        # First get() against empty file: full rebuild captures _prefix=b"".
        c.get(buf)

        # Append + get(): incremental fold. Pre-fix this would NOT refresh
        # _prefix, leaving the tombstone disarmed.
        buf.append(make_lifecycle_event(schema_id="s0", kind=EventKind.CREATE))
        c.get(buf)

        # Rotate twice (recycles the inode on most filesystems is unlikely;
        # the fuzz failure relies on at least the prefix bytes diverging
        # after rotate). Then append a *different* event.
        p = buf.path
        for _ in range(2):
            tmp = p.with_suffix(p.suffix + ".rot")
            tmp.write_bytes(b"")
            os.replace(tmp, p)

        buf.append(make_lifecycle_event(schema_id="s0", kind=EventKind.PROMOTE))

        cached = c.get(buf)
        fresh = snapshot_from_buffer(buf)
        assert cached == fresh


class TestStatKey:
    def test_size_changes_invalidate(self, buffer):
        _seed(buffer, n_schemas=2)
        c = CachedLifecycleSnapshot()
        c.get(buffer)
        before_offset = c._offset
        buffer.append(make_lifecycle_event(
            schema_id="s99", kind=EventKind.CREATE,
        ))
        c.get(buffer)
        # Offset advanced past the previous EOF.
        assert c._offset > before_offset


class TestEngineIntegration:
    """The cache is wired into RetrievalEngine; these tests verify that
    the integration preserves observable behavior."""

    def test_engine_partial_hits_dominate_over_full_replays(self, tmp_path):
        """Each recall() appends RECALL_* events but no lifecycle
        events. The cache must therefore stay on the partial-hit path
        (incremental scan, fold nothing) rather than the full-rebuild
        path. The first recall pays the rebuild cost; every subsequent
        one is a partial hit."""
        from engram import Config, Engram
        from engram.core import Memory, MemoryState, MemoryType
        from datetime import datetime, timezone

        cfg = Config(path=str(tmp_path / "eng"))
        cfg.security.max_events_per_minute = 0
        cfg.retrieval.respect_schema_lifecycle = True
        eng = Engram(config=cfg)
        try:
            now = datetime.now(timezone.utc)
            for i in range(3):
                eng._store.upsert(Memory(
                    id=f"sch_{i}", type=MemoryType.SCHEMA,
                    state=MemoryState.ACTIVE,
                    content=f"pattern {i}", summary=f"s{i}",
                    salience=0.5, confidence=0.7, decay_rate=0.1,
                    created_at=now, last_accessed=now,
                ))
                eng._buffer.append(make_lifecycle_event(
                    schema_id=f"sch_{i}", kind=EventKind.CREATE,
                ))
            cache = eng._retrieval._lifecycle_cache
            for _ in range(6):
                eng.recall("pattern", limit=10)
            # Exactly one full rebuild (the first recall).
            assert cache.stats["misses"] == 1
            # The remaining 5 are partial hits (RECALL_* events appended
            # by recall() advance the offset but yield no lifecycle ops).
            assert cache.stats["partial_hits"] >= 5
        finally:
            eng.close()

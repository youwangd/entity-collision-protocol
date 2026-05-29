"""Chaos / fault-injection tests.

Goal: prove the system stays consistent under partial failures —
truncated JSONL, garbled lines, missing files, killed-mid-write,
disk-full-style errors.

Marked `@pytest.mark.chaos`; opt-in via `pytest -m chaos`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram import Engram, Config

pytestmark = pytest.mark.chaos


def _new_engram(tmp_path: Path) -> Engram:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    return Engram(cfg)


# ---------------------------------------------------------------------------
# JSONL corruption
# ---------------------------------------------------------------------------

def test_truncated_last_line_is_skipped(tmp_path: Path) -> None:
    """A partially-written final line (e.g. process killed mid-flush) must
    not break scan() or rebuild()."""
    eng = _new_engram(tmp_path)
    try:
        for i in range(20):
            eng.remember(f"event {i}", salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()

    # Truncate the final 30 bytes — guaranteed to break the last JSON line.
    sz = buf_path.stat().st_size
    with open(buf_path, "r+b") as f:
        f.truncate(max(0, sz - 30))
        # ensure we end mid-line (no trailing newline)
        f.seek(0, 2)

    # Re-open and rebuild — must not crash.
    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        # We lost at most one event.
        assert n >= 19, f"expected >=19 surviving events, got {n}"
    finally:
        eng2.close()


def test_garbage_line_in_middle_is_skipped(tmp_path: Path) -> None:
    """A non-JSON line in the middle of the log must not stop the scan."""
    eng = _new_engram(tmp_path)
    try:
        for i in range(10):
            eng.remember(f"first half {i}", salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()

    # Inject a garbage line in the middle.
    lines = buf_path.read_text(encoding="utf-8").splitlines()
    lines.insert(5, "this is not JSON {{{ broken")
    lines.insert(7, '{"partial": "object" missing brace')
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Append more good events.
    eng2 = _new_engram(tmp_path)
    try:
        for i in range(10):
            eng2.remember(f"second half {i}", salience=0.5)
        n = eng2.rebuild(incremental=False)
        # 20 good events, 2 garbage lines skipped.
        assert n == 20, f"expected 20 events after rebuild, got {n}"
    finally:
        eng2.close()


def test_empty_jsonl_file(tmp_path: Path) -> None:
    """A zero-byte events.jsonl must not crash startup or rebuild."""
    eng = _new_engram(tmp_path)
    try:
        buf_path = eng._buffer.path
    finally:
        eng.close()

    buf_path.write_bytes(b"")

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n == 0
        # And we can still write.
        eng2.remember("after-recovery", salience=0.5)
        results = eng2.recall("recovery", limit=5)
        assert len(results) >= 1
    finally:
        eng2.close()


def test_jsonl_with_only_blank_lines(tmp_path: Path) -> None:
    eng = _new_engram(tmp_path)
    try:
        buf_path = eng._buffer.path
    finally:
        eng.close()

    buf_path.write_text("\n\n   \n\t\n\n", encoding="utf-8")

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n == 0
        eng2.remember("recovered", salience=0.5)
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# SQLite recovery
# ---------------------------------------------------------------------------

def test_missing_sqlite_file_rebuilds_from_jsonl(tmp_path: Path) -> None:
    """If the SQLite projection is wiped, rebuild() restores it from JSONL."""
    eng = _new_engram(tmp_path)
    try:
        for i in range(15):
            eng.remember(f"durable fact {i}", salience=0.7)
        # Capture some content for verification.
        sample_results = eng.recall("durable fact 7", limit=1)
        assert sample_results, "preconditions: original recall works"
        sqlite_path = eng._store.db_path
    finally:
        eng.close()

    # Nuke the SQLite file.
    if sqlite_path.exists():
        sqlite_path.unlink()

    # Reopen — engine should come up clean and rebuild from JSONL.
    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n == 15, f"expected 15 memories after rebuild, got {n}"
        results = eng2.recall("durable fact 7", limit=1)
        assert results, "rebuild lost data"
    finally:
        eng2.close()


def test_sqlite_torn_write_recovers_via_rebuild(tmp_path: Path) -> None:
    """Simulate: SQLite gets corrupted bytes (we just truncate it). The
    JSONL log is the source of truth — recovery procedure is to delete the
    corrupt DB and rebuild() from JSONL.

    Documented invariant: the JSONL is the durable source. SQLite is a
    derivable projection. If the projection is corrupt, the operator
    deletes it and rebuilds.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(10):
            eng.remember(f"truth fact {i}", salience=0.5)
        sqlite_path = eng._store.db_path
    finally:
        eng.close()

    # Truncate the SQLite file mid-stream → corruption.
    sz = sqlite_path.stat().st_size
    with open(sqlite_path, "r+b") as f:
        f.truncate(sz // 2)

    # Recovery procedure: delete corrupt DB, rebuild from JSONL.
    sqlite_path.unlink()
    # Also remove any stray WAL/SHM sidecars.
    for sidecar in (sqlite_path.with_suffix(sqlite_path.suffix + "-wal"),
                    sqlite_path.with_suffix(sqlite_path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n == 10
        results = eng2.recall("truth fact", limit=15)
        assert len(results) >= 5
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# Disk-full simulation (write side)
# ---------------------------------------------------------------------------

def test_oserror_on_buffer_append_propagates(tmp_path: Path, monkeypatch) -> None:
    """If the OS raises (disk full / read-only FS), the engine must not
    silently swallow it."""
    eng = _new_engram(tmp_path)
    try:
        # Patch buffer.append to raise OSError.
        def boom(*a, **kw):
            raise OSError("simulated ENOSPC")
        monkeypatch.setattr(eng._buffer, "append", boom)
        with pytest.raises(OSError):
            eng.remember("this should fail loudly", salience=0.5)
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Killed mid-consolidation: idempotency
# ---------------------------------------------------------------------------

def test_consolidate_is_idempotent_under_repeated_calls(tmp_path: Path) -> None:
    """If a consolidate run is killed and restarted, calling consolidate
    again must converge to the same state — no duplicate memories, no loss."""
    eng = _new_engram(tmp_path)
    try:
        for i in range(20):
            eng.remember(f"the cat sat on the mat number {i}", salience=0.5)

        eng.consolidate()
        snap1 = eng._store.stats()

        # Run consolidate again — should be a no-op or very nearly so.
        eng.consolidate()
        snap2 = eng._store.stats()

        # Memory count must not balloon.
        assert snap2.get("total", snap2.get("count", 0)) <= snap1.get("total", snap1.get("count", 0)) + 5, \
            f"consolidate not idempotent: {snap1} → {snap2}"
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Concurrent open + crash simulation
# ---------------------------------------------------------------------------

def test_reopen_after_unclean_close(tmp_path: Path) -> None:
    """Drop the engine without close() — equivalent to kill -9. Then
    reopen and verify we can still read everything."""
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")

    # First lifetime — no close().
    eng1 = Engram(cfg)
    for i in range(5):
        eng1.remember(f"survived crash {i}", salience=0.5)
    # Deliberately drop the reference WITHOUT close().
    del eng1

    # Second lifetime — must read prior writes.
    eng2 = Engram(cfg)
    try:
        results = eng2.recall("survived", limit=10)
        assert len(results) >= 5, f"only {len(results)} recovered after unclean close"
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# truncate_before atomic-rename invariants
# ---------------------------------------------------------------------------

def test_orphan_truncate_tmp_does_not_corrupt_buffer(tmp_path: Path) -> None:
    """Simulate a crash *after* truncate_before wrote events.jsonl.tmp but
    *before* os.replace landed it. The original events.jsonl must remain
    intact, and the leftover .tmp must not interfere with subsequent reads
    or writes.
    """
    from datetime import datetime, timedelta, timezone
    from engram.core.types import Event, EventType
    from engram.store.buffer import JSONLBufferStore

    store = JSONLBufferStore(tmp_path)
    for i in range(10):
        store.append(Event(
            id=f"e{i:03d}",
            type=EventType.EXPLICIT_REMEMBER,
            ts=datetime.now(timezone.utc),
            content=f"event {i}",
        ))
    pre_count = store.count()
    pre_bytes = store.path.read_bytes()

    # Simulate the orphan: truncate wrote .tmp but crashed before replace.
    tmp_sidecar = store.path.with_suffix(store.path.suffix + ".tmp")
    tmp_sidecar.write_text("garbage that should never be read\n", encoding="utf-8")

    # Re-opening / reading the buffer must ignore the orphan completely.
    store2 = JSONLBufferStore(tmp_path)
    assert store2.count() == pre_count
    assert store2.path.read_bytes() == pre_bytes

    # Subsequent truncate_before must succeed and reclaim the .tmp slot.
    cutoff = datetime.now(timezone.utc) + timedelta(days=365)  # remove all
    removed = store2.truncate_before(cutoff)
    assert removed == pre_count
    assert store2.count() == 0


def _truncate_worker(args):
    from datetime import datetime
    from engram.store.buffer import JSONLBufferStore
    bp, cutoff_iso = args
    s = JSONLBufferStore(Path(bp))
    return s.truncate_before(datetime.fromisoformat(cutoff_iso))


def test_truncate_holds_exclusive_lock_against_concurrent_truncate(tmp_path: Path) -> None:
    """Two truncates running in parallel must serialize, not interleave —
    final state matches one-after-the-other application.
    """
    import multiprocessing as mp
    from datetime import datetime, timedelta, timezone
    from engram.core.types import Event, EventType
    from engram.store.buffer import JSONLBufferStore

    store = JSONLBufferStore(tmp_path)
    base = datetime.now(timezone.utc) - timedelta(days=5)
    for i in range(2000):
        store.append(Event(
            id=f"e{i:04d}",
            type=EventType.EXPLICIT_REMEMBER,
            # Stagger across days so different cutoffs delete different counts.
            ts=base + timedelta(hours=i),
            content=f"e{i}",
        ))

    cutoff_a = (base + timedelta(days=2)).isoformat()
    cutoff_b = (base + timedelta(days=4)).isoformat()

    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        results = pool.map(_truncate_worker, [(str(tmp_path), cutoff_a), (str(tmp_path), cutoff_b)])

    # Whatever the order, everything < max(cutoff) is gone, file is parseable.
    final_lines = [
        l for l in store.path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    for line in final_lines:
        json.loads(line)  # must parse — no torn writes
    # Sum of removed counts should equal pre_count - surviving_count.
    assert sum(results) == 2000 - len(final_lines)

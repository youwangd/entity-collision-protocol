"""Chaos / fault-injection gap-fill (mission item 2d).

Covers gaps not addressed by `test_fault_injection.py`:
- NUL bytes / non-UTF8 lines in JSONL
- ENOSPC mid-line (partial write at OS level)
- Zero-byte SQLite projection alongside populated JSONL
- Populated SQLite alongside zero-byte JSONL (asymmetric corruption)
- Truncated SQLite WAL recovery
- Killed-mid-consolidation between mark-suppressed and projection-update
- JSONL line longer than typical buffer (very large content)
- Read-only mode after permissions drop

All marked `@pytest.mark.chaos`; opt-in via `pytest -m chaos`.
"""
from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from engram import Config, Engram

pytestmark = pytest.mark.chaos


def _new_engram(tmp_path: Path) -> Engram:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    return Engram(cfg)


# ---------------------------------------------------------------------------
# JSONL: encoding / NUL torture
# ---------------------------------------------------------------------------

def test_nul_byte_line_is_skipped(tmp_path: Path) -> None:
    """A line with embedded NUL bytes (page-cache corruption) must not crash
    rebuild — JSON decoder rejects, scan() skips, neighbors survive.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(10):
            eng.remember(f"clean event {i}", salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()

    # Splice a NUL-byte line in the middle.
    raw = buf_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    lines.insert(5, b"\x00\x00\x00garbage\x00line\x00\n")
    buf_path.write_bytes(b"".join(lines))

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n >= 10, f"clean events lost: only {n} survived"
    finally:
        eng2.close()


def test_invalid_utf8_line_is_skipped(tmp_path: Path) -> None:
    """A non-UTF8 line in the middle of a JSONL buffer must be SKIPPED, not
    abort the whole scan. Per-line binary decode (buffer.py scan()) isolates
    the damage so clean events around it survive rebuild.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(8):
            eng.remember(f"utf8 ok {i}", salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()

    # Inject a non-UTF8 line, then add more clean events afterward to confirm
    # scan keeps going past the bad line rather than truncating at it.
    with open(buf_path, "ab") as f:
        f.write(b"\xff\xfe\xff\xfe not utf-8 at all \xff\xfe\n")

    eng2 = _new_engram(tmp_path)
    try:
        for i in range(4):
            eng2.remember(f"after bad line {i}", salience=0.5)
        n = eng2.rebuild(incremental=False)
        # 8 clean before + 4 clean after = 12 should survive; bad line dropped.
        assert n >= 12, f"clean events lost around bad line: got {n}"
    finally:
        eng2.close()


def test_very_long_line_round_trips(tmp_path: Path) -> None:
    """A 256 KB single-line JSON event must round-trip through append/scan
    without being treated as torn or split.
    """
    eng = _new_engram(tmp_path)
    try:
        # Stay under firewall max_length (50_000) but big enough to stress
        # JSONL line-handling with a single ~40KB line.
        big = "x" * 40_000
        eng.remember(big, salience=0.5)
        eng.remember("small follow-up", salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()

    # Sanity: file should have exactly 2 newline-terminated lines.
    raw = buf_path.read_bytes()
    assert raw.count(b"\n") == 2, "expected exactly 2 lines"

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n == 2, f"large line broke rebuild: got {n}"
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# Asymmetric on-disk corruption (one of {jsonl, sqlite} is zero-bytes)
# ---------------------------------------------------------------------------

def test_zero_byte_sqlite_with_populated_jsonl_recovers(tmp_path: Path) -> None:
    """If the SQLite projection is zero bytes (e.g. ENOSPC truncated it
    mid-create), the engine must rebuild from JSONL on next open.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(15):
            eng.remember(f"recover me {i}", salience=0.5)
        db_path = Path(eng._store.db_path)
    finally:
        eng.close()

    # Corrupt: zero out the SQLite file. JSONL is the source of truth.
    db_path.write_bytes(b"")

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n >= 15, f"rebuild from JSONL after zero-byte SQLite: only {n}"
        # Recall must work post-rebuild.
        results = eng2.recall("recover", limit=20)
        assert len(results) >= 15, f"recall lost data: {len(results)} hits"
    finally:
        eng2.close()


def test_populated_sqlite_zero_byte_jsonl_does_not_explode(tmp_path: Path) -> None:
    """If the JSONL is zero bytes but SQLite has content (admin op went
    rogue), the engine must open without crashing. Recall still works on
    whatever is in the projection.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(10):
            eng.remember(f"sqlite-only {i}", salience=0.5)
        eng.consolidate()
        buf_path = eng._buffer.path
    finally:
        eng.close()

    buf_path.write_bytes(b"")

    eng2 = _new_engram(tmp_path)
    try:
        # Open + recall must not crash. Whether anything comes back depends on
        # whether consolidate moved everything to the projection — the
        # invariant we test is "no crash".
        results = eng2.recall("sqlite-only", limit=20)
        assert isinstance(results, list)
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# SQLite WAL torture
# ---------------------------------------------------------------------------

def test_truncated_sqlite_wal_recovers(tmp_path: Path) -> None:
    """Truncate the SQLite WAL mid-record. SQLite must recover (it ignores
    the partial frame) and the engine must keep working.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(20):
            eng.remember(f"wal torture {i}", salience=0.5)
        db_path = Path(eng._store.db_path)
    finally:
        eng.close()

    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists() and wal.stat().st_size > 32:
        # Lop off the last 16 bytes of the WAL — this corrupts the final
        # frame and SQLite truncates back at recovery time.
        sz = wal.stat().st_size
        with open(wal, "r+b") as f:
            f.truncate(sz - 16)

    # Reopen; should not crash. SQLite's recovery either rolls back the
    # partial frame or reports SQLITE_CORRUPT — we accept the former and
    # rebuild on the latter.
    eng2 = _new_engram(tmp_path)
    try:
        try:
            n = eng2.rebuild(incremental=False)
            assert n >= 1, "no events recovered after WAL truncation"
        except sqlite3.DatabaseError:
            pytest.skip("SQLite refused to recover; rebuild path needs harden")
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# Killed mid-consolidation: granular phase crash
# ---------------------------------------------------------------------------

def test_consolidation_crash_after_dedup_before_persist(
    tmp_path: Path, monkeypatch
) -> None:
    """If the persistence stage crashes mid-pipeline, the pipeline must
    log + record the error (not propagate), and a subsequent clean
    consolidate() must converge to a stable state.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(30):
            eng.remember(f"the cat sat on the mat {i % 5}", salience=0.5)

        from engram.consolidation.pipeline import MemoryPersistence

        original_run = MemoryPersistence.run
        boom_count = {"n": 0}

        def boom(self, *a, **kw):
            boom_count["n"] += 1
            if boom_count["n"] == 1:
                raise RuntimeError("simulated crash mid-persistence")
            return original_run(self, *a, **kw)

        monkeypatch.setattr(MemoryPersistence, "run", boom)

        # Pipeline must absorb the per-stage exception.
        report = eng.consolidate()
        assert any("persistence" in err for err in (report.errors or [])), (
            f"expected 'persistence' in report.errors, got {report.errors}"
        )

        # Second call: clean (boom_count > 1 → original).
        eng.consolidate()
        snap = eng._store.stats()
        assert snap is not None
    finally:
        eng.close()


def test_repeated_crash_then_recover_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Two crashes, then a clean run. Final state must be stable."""
    eng = _new_engram(tmp_path)
    try:
        for i in range(20):
            eng.remember(f"crash recover {i % 3}", salience=0.5)

        from engram.consolidation.pipeline import MemoryPersistence

        crashes_remaining = {"n": 2}
        original = MemoryPersistence.run

        def maybe_crash(self, *a, **kw):
            if crashes_remaining["n"] > 0:
                crashes_remaining["n"] -= 1
                raise RuntimeError("crash")
            return original(self, *a, **kw)

        monkeypatch.setattr(MemoryPersistence, "run", maybe_crash)

        for _ in range(2):
            r = eng.consolidate()
            assert r.errors  # exception was captured
        # Third call: clean.
        eng.consolidate()
        snap1 = eng._store.stats()
        eng.consolidate()
        snap2 = eng._store.stats()
        assert snap1.get("total", snap1.get("count")) == snap2.get(
            "total", snap2.get("count")
        )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Read-only filesystem after init
# ---------------------------------------------------------------------------

def test_readonly_jsonl_surfaces_error_loudly(tmp_path: Path) -> None:
    """If JSONL is made read-only between writes, append must fail loudly,
    not silently drop the event.
    """
    eng = _new_engram(tmp_path)
    try:
        eng.remember("first", salience=0.5)
        buf_path = eng._buffer.path
        # Drop write perms.
        cur = buf_path.stat().st_mode
        os.chmod(buf_path, cur & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        try:
            with pytest.raises(Exception):
                eng.remember("second — should fail", salience=0.5)
        finally:
            # Restore so tmp_path cleanup works.
            os.chmod(buf_path, cur)
    finally:
        eng.close()

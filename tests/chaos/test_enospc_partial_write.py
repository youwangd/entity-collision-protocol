"""Chaos: ENOSPC mid-write — partial JSONL write at the OS level.

Existing chaos coverage truncates the buffer *post-hoc*, then asserts
recovery. That exercises the read path but does not cover the more realistic
failure: the kernel returns short on `write()` (e.g. ENOSPC, ECONNRESET on
network FS) leaving a torn line in place. Then the engine must:

  1. Surface the error loudly to the caller (no silent drop).
  2. On the *next* successful append after FS recovery, the new line must
     not fuse onto the torn frame (the buffer's defensive `\\n` prepend in
     `append()`).
  3. A subsequent rebuild() must skip the torn line and recover all
     fully-written prior events plus the post-recovery event.

This nails down the partial-write contract that NEXT.md priority #3 calls
out, beyond what `test_oserror_on_buffer_append_propagates` already covers
(that one only checks the OSError propagates, not torn-frame handling).
"""
from __future__ import annotations

import builtins
import io
from pathlib import Path

import pytest

from engram import Config, Engram

pytestmark = pytest.mark.chaos


def _new_engram(tmp_path: Path) -> Engram:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    return Engram(cfg)


class _ShortWriteFile(io.BufferedRandom):
    """Wraps a real file; the next .write() call writes only the first N
    bytes then raises OSError(ENOSPC). Subsequent writes pass through.
    """


def test_enospc_short_write_leaves_torn_frame_then_recovers(
    tmp_path: Path, monkeypatch
) -> None:
    eng = _new_engram(tmp_path)
    try:
        # Write 5 clean events, capture buffer path.
        for i in range(5):
            eng.remember(f"clean prelude {i}", salience=0.5)
        buf_path = eng._buffer.path
        clean_size = buf_path.stat().st_size

        # Patch open() so the NEXT append() to buf_path performs a short
        # write of half the bytes then raises ENOSPC. We arm exactly once.
        real_open = builtins.open
        armed = {"once": True}

        class _ShortWriter:
            def __init__(self, fh):
                self._fh = fh

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def __enter__(self):
                self._fh.__enter__()
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

            def write(self, b):
                if armed["once"] and isinstance(b, (bytes, bytearray)) and b.endswith(b"\n"):
                    armed["once"] = False
                    half = max(1, len(b) // 2)
                    self._fh.write(b[:half])
                    self._fh.flush()
                    raise OSError(28, "simulated ENOSPC short write")
                return self._fh.write(b)

        def open_with_short_write(path, *a, **kw):
            fh = real_open(path, *a, **kw)
            try:
                if Path(path) == buf_path and (
                    "a" in (a[0] if a else kw.get("mode", ""))
                ):
                    return _ShortWriter(fh)
            except Exception:
                pass
            return fh

        monkeypatch.setattr(builtins, "open", open_with_short_write)

        # The next remember() should fail loudly.
        with pytest.raises(Exception):
            eng.remember("this write hits ENOSPC mid-flush", salience=0.5)

        # File should now be larger than clean_size (torn bytes landed).
        torn_size = buf_path.stat().st_size
        assert torn_size > clean_size, "expected torn bytes to land on disk"
        # Post-fault, append() repairs the broken tail with a newline under
        # exclusive lock so subsequent O_APPEND writes can't fuse onto the
        # half-frame. The torn half-line is now its own (invalid) JSON line,
        # which scan() will skip.
        tail = buf_path.read_bytes()[-1:]
        assert tail == b"\n", "torn frame should be tail-repaired post-fault"

        # Disarm and let the next write succeed.
        # Engram's append() defensive \n-prepend should prevent fusion.
        eng.remember("post-recovery event", salience=0.5)

        n_after = sum(1 for _ in eng._buffer.scan())
        # 5 prelude + 1 post-recovery = 6 valid events; the torn frame
        # must be skipped, not double-counted, not fused.
        assert n_after == 6, f"expected 6 valid events post-recovery, got {n_after}"
    finally:
        eng.close()


def test_enospc_torn_frame_survives_full_rebuild(tmp_path: Path, monkeypatch) -> None:
    """Same setup but exercise the rebuild path: after a torn frame, a fresh
    Engram on the same path must rebuild() to a consistent state with the
    pre-failure events intact.
    """
    eng = _new_engram(tmp_path)
    try:
        for i in range(8):
            eng.remember(f"durable {i}", salience=0.5)
        buf_path = eng._buffer.path
        # Manually write a torn frame: half a JSON line, no newline.
        with open(buf_path, "ab") as f:
            f.write(b'{"id":"torn","type":"remember","timestamp":"2026')
    finally:
        eng.close()

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n >= 8, f"durable events lost after torn frame: only {n}"
        # New writes still work and don't fuse onto the torn tail.
        eng2.remember("after torn", salience=0.5)
        results = eng2.recall("durable", limit=20)
        assert len(results) >= 8
    finally:
        eng2.close()

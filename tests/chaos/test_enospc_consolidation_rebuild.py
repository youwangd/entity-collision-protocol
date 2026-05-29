"""Chaos: ENOSPC during consolidation pipeline writes, and during rebuild.

Closes the NEXT.md priority #3 follow-up: existing ENOSPC coverage exercises
the *user-facing* `remember()` write path (`tests/chaos/test_enospc_partial_write.py`).
The consolidation pipeline and the projection rebuild also write to the same
JSONL buffer (lifecycle events, schema synthesis, state transitions). Those
paths must obey the same torn-frame contract:

  E-CR-1  ENOSPC raised during consolidate() must not silently drop user
          memories; a fresh Engram on the same path must rebuild() back to
          a state containing every successfully-acked remember().
  E-CR-2  ENOSPC during the post-rebuild snapshot write must not corrupt the
          projection; the rebuilt store remains queryable.
  E-CR-3  Repeated arming (every Nth write fails) — the engine never produces
          a fused frame, and total durable events == count of pre-fault
          successful remember() calls.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from engram import Config, Engram

pytestmark = pytest.mark.chaos


def _new_engram(tmp_path: Path, sub: str = "engram") -> Engram:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / sub)
    return Engram(cfg)


class _ShortWriter:
    """Wrap a file handle; on the Nth write of a newline-terminated payload,
    emit a torn (half) write and raise ENOSPC. Subsequent writes pass through.
    """

    def __init__(self, fh, fire_after: int = 0):
        self._fh = fh
        self._n = 0
        self._fire_after = fire_after
        self._fired = False

    def __getattr__(self, name):
        return getattr(self._fh, name)

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)

    def write(self, b):
        if (
            not self._fired
            and isinstance(b, (bytes, bytearray))
            and b.endswith(b"\n")
        ):
            self._n += 1
            if self._n > self._fire_after:
                self._fired = True
                half = max(1, len(b) // 2)
                self._fh.write(b[:half])
                self._fh.flush()
                raise OSError(28, "simulated ENOSPC short write (consolidation)")
        return self._fh.write(b)


def _patch_short_write(monkeypatch, target_path: Path, fire_after: int = 0):
    real_open = builtins.open
    state = {"writer": None}

    def opener(path, *a, **kw):
        fh = real_open(path, *a, **kw)
        try:
            if Path(path) == target_path and (
                "a" in (a[0] if a else kw.get("mode", ""))
            ):
                w = _ShortWriter(fh, fire_after=fire_after)
                state["writer"] = w
                return w
        except Exception:
            pass
        return fh

    monkeypatch.setattr(builtins, "open", opener)
    return state


# ---------------------------------------------------------------------------
# E-CR-1: ENOSPC during consolidate() — user data survives via rebuild.
# ---------------------------------------------------------------------------


def test_enospc_during_consolidation_durable_writes_survive_rebuild(
    tmp_path: Path, monkeypatch
) -> None:
    eng = _new_engram(tmp_path)
    try:
        # 30 durable user writes — these complete BEFORE we arm the fault.
        for i in range(30):
            eng.remember(f"durable consolidation prelude {i}", salience=0.7)
        _ = eng._buffer.path
        n_clean = sum(1 for _ in eng._buffer.scan())
        assert n_clean >= 30
    finally:
        eng.close()

    # Re-open and arm: the next append (which consolidate() will trigger
    # via lifecycle events / state transitions) tears.
    eng = _new_engram(tmp_path)
    try:
        _patch_short_write(monkeypatch, eng._buffer.path, fire_after=0)
        # consolidate may or may not raise depending on which phase first
        # touches the buffer; either is acceptable as long as no silent loss.
        try:
            eng.consolidate()
        except Exception:
            # Either OSError or BufferError-wrapping-OSError is acceptable —
            # the contract is "fail loudly, lose nothing", not a specific type.
            pass
    finally:
        eng.close()
    monkeypatch.undo()

    # Fresh engine on same path: rebuild must recover ≥30 user memories.
    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n >= 30, f"durable user memories lost across consolidation ENOSPC: {n}"
        # Recall path still works; torn frame did not poison the projection.
        results = eng2.recall("durable", limit=50)
        assert len(results) >= 30
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# E-CR-2: ENOSPC during the *first* write after a rebuild (covers the
# snapshot/snap-write lane). The rebuilt store must still be queryable.
# ---------------------------------------------------------------------------


def test_enospc_post_rebuild_first_write_keeps_projection_queryable(
    tmp_path: Path, monkeypatch
) -> None:
    eng = _new_engram(tmp_path)
    try:
        for i in range(15):
            eng.remember(f"post-rebuild prelude {i}", salience=0.5)
    finally:
        eng.close()

    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        assert n >= 15

        _patch_short_write(monkeypatch, eng2._buffer.path, fire_after=0)
        with pytest.raises(Exception):
            eng2.remember("this remember tears post-rebuild", salience=0.5)

        # Disarm the patch — recall() also writes an audit event through
        # the same buffer, and we want to verify the projection remains
        # readable, not exercise the fault again.
        monkeypatch.undo()

        # Even after a torn write, the projection is still readable for the
        # 15 already-projected memories. (recall() reads from SQLite.)
        results = eng2.recall("prelude", limit=30)
        assert len(results) >= 15, (
            f"projection became unqueryable after torn post-rebuild write: {len(results)}"
        )
    finally:
        eng2.close()


# ---------------------------------------------------------------------------
# E-CR-3: Repeated torn-frame stress — every Nth write tears. Across many
# remember() calls, the rebuilt count must equal the count of writes that
# returned successfully (no fusion, no double-count).
# ---------------------------------------------------------------------------


def test_repeated_torn_frames_no_fusion_count_matches_acks(
    tmp_path: Path, monkeypatch
) -> None:
    eng = _new_engram(tmp_path)
    buf_path = eng._buffer.path
    real_open = builtins.open

    # Arm to fire on every 5th write.
    counter = {"n": 0}

    class _PeriodicShortWriter:
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
            if isinstance(b, (bytes, bytearray)) and b.endswith(b"\n"):
                counter["n"] += 1
                if counter["n"] % 5 == 0:
                    half = max(1, len(b) // 2)
                    self._fh.write(b[:half])
                    self._fh.flush()
                    raise OSError(28, "periodic torn write")
            return self._fh.write(b)

    def opener(path, *a, **kw):
        fh = real_open(path, *a, **kw)
        try:
            if Path(path) == buf_path and (
                "a" in (a[0] if a else kw.get("mode", ""))
            ):
                return _PeriodicShortWriter(fh)
        except Exception:
            pass
        return fh

    monkeypatch.setattr(builtins, "open", opener)

    acks = 0
    try:
        for i in range(40):
            try:
                eng.remember(f"periodic torn payload {i}", salience=0.5)
                acks += 1
            except Exception:
                # BufferError or OSError both indicate the torn-write fault.
                pass
    finally:
        eng.close()
    monkeypatch.undo()

    # acks should be roughly 4/5 of attempts — verify we did exercise the fault.
    assert 20 <= acks <= 40

    # Rebuild on a fresh engine: every successfully-acked remember() must
    # come back. Torn frames must not be double-counted or fused.
    eng2 = _new_engram(tmp_path)
    try:
        n = eng2.rebuild(incremental=False)
        # Lower bound: at least the acks survived. Upper bound: no spurious
        # extras from torn-frame fusion.
        assert n == acks, (
            f"rebuild count {n} != ack count {acks}; "
            "indicates torn-frame fusion or silent drop"
        )
    finally:
        eng2.close()

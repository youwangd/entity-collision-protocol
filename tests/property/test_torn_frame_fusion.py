"""Property test: torn-frame fusion never happens, at any fault rate.

Origin (NEXT.md priority #3, 2026-05-23): the torn-frame fusion bug
fixed in `f218b47` surfaced under one specific 5/5 periodic fault rate.
This property test randomizes both the fault period and the number of
remember() attempts, asserting the same invariant the deterministic
chaos test asserts:

    rebuild_count == ack_count

i.e. for any sequence of (success, torn-fault, success, ...) attempts on
a single Engram instance, the post-rebuild durable count must equal the
number of remember() calls that returned without raising — no fused
frames, no silent drops, no double-counts.

Hypothesis-fuzzed analogue of
`tests/chaos/test_enospc_consolidation_rebuild.py
::test_repeated_torn_frames_no_fusion_count_matches_acks`.
Marked `chaos` so it runs with the chaos suite, not the default profile.
"""
from __future__ import annotations

import builtins
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram

pytestmark = [pytest.mark.chaos, pytest.mark.property]


def _new_engram(root: Path) -> Engram:
    cfg = Config.minimal()
    cfg.path = str(root / "engram")
    return Engram(cfg)


@given(
    fault_period=st.integers(min_value=2, max_value=11),
    n_attempts=st.integers(min_value=8, max_value=40),
)
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
def test_torn_frame_fusion_invariant_under_random_fault_rate(
    fault_period: int, n_attempts: int, monkeypatch
) -> None:
    """For any (period, n) with 2 ≤ period ≤ 11 and 8 ≤ n ≤ 40, every Nth
    write tears (half-write + ENOSPC) and the post-rebuild count exactly
    equals the count of acked writes. Falsifies torn-frame fusion."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        eng = _new_engram(root)
        buf_path = eng._buffer.path
        real_open = builtins.open
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
                    if counter["n"] % fault_period == 0:
                        half = max(1, len(b) // 2)
                        self._fh.write(b[:half])
                        self._fh.flush()
                        raise OSError(28, "fuzzed torn write")
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
            for i in range(n_attempts):
                try:
                    eng.remember(
                        f"fuzzed torn payload p={fault_period} i={i}",
                        salience=0.5,
                    )
                    acks += 1
                except Exception:
                    pass
        finally:
            eng.close()
        monkeypatch.undo()

        # Sanity: with period ∈ [2, 11], we must have hit at least one fault
        # (counter increments per *attempted* line write, including faults).
        # acks bound: n_attempts - floor(n_attempts/period) ≤ acks ≤ n_attempts.
        assert 0 < acks <= n_attempts

        eng2 = _new_engram(root)
        try:
            n = eng2.rebuild(incremental=False)
            assert n == acks, (
                f"rebuild count {n} != ack count {acks} "
                f"(period={fault_period}, n_attempts={n_attempts}); "
                "torn-frame fusion or silent drop detected"
            )
        finally:
            eng2.close()

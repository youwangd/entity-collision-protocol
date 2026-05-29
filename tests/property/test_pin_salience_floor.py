"""Pinned-memory salience-floor invariant.

Pin namespace is disjoint from memory namespace (S-I8), so a salience-threshold
forget — which iterates the memories table — must NOT remove or suppress any
pin, regardless of how aggressive the floor is. This test pins arbitrary content,
ingests low-salience memories alongside, and asserts:

  P-S1  forget(below=θ) for any θ ∈ [0, 1] never removes a pin.
  P-S2  forget(below=θ, hard=True) likewise leaves pins intact.
  P-S3  Pinned content still surfaces in active_context() after a salience-floor
        forget that would suppress every memory.
  P-S4  delete(query=...) on text matching pin content does NOT remove the pin
        (delete operates on the memories table only).
"""
from __future__ import annotations

import string
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram


_alphabet = st.text(
    alphabet=string.ascii_lowercase + " ", min_size=4, max_size=20
).filter(lambda s: bool(s.strip()))


@pytest.fixture()
def engram_factory():
    tmps: list[Path] = []

    def _make() -> Engram:
        tmp = Path(tempfile.mkdtemp(prefix="engram-pin-floor-"))
        tmps.append(tmp)
        cfg = Config.minimal()
        cfg.path = str(tmp)
        cfg.security.max_events_per_minute = 0
        return Engram(config=cfg)

    yield _make
    import shutil

    for t in tmps:
        shutil.rmtree(t, ignore_errors=True)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    pin_contents=st.lists(_alphabet, min_size=1, max_size=4, unique=True),
    mem_contents=st.lists(_alphabet, min_size=1, max_size=6, unique=True),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    hard=st.booleans(),
)
def test_salience_floor_forget_leaves_pins_intact(
    engram_factory, pin_contents, mem_contents, threshold, hard
):
    """P-S1/P-S2: forget(below=θ) never touches pins."""
    e = engram_factory()
    try:
        # Pin first so pins exist before the noise.
        pin_ids = [e.pin(c) for c in pin_contents]
        # Ingest memories at low (≤ 0.5) salience so threshold sweeps reach them.
        for c in mem_contents:
            e.remember(c, salience=0.0)
        e.rebuild()

        before = {p["id"] for p in e._store.get_pins()}
        assert set(pin_ids) <= before, "P-S0: pin add failed pre-condition"

        e.forget(below=threshold, hard=hard)

        after = {p["id"] for p in e._store.get_pins()}
        assert after == before, (
            f"P-S1/P-S2 violated: forget(below={threshold}, hard={hard}) "
            f"removed pins. before={before} after={after}"
        )

        # Active context must still surface every pin.
        ctx = e.active_context(max_tokens=8192)
        for c in pin_contents:
            assert c in ctx, (
                f"P-S3 violated: pinned content {c!r} missing from active_context "
                f"after forget(below={threshold})"
            )
    finally:
        try:
            e.close()
        except Exception:
            pass


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(content=_alphabet)
def test_delete_query_does_not_remove_pin(engram_factory, content):
    """P-S4: delete(query=content) on pin-content text leaves the pin alive.

    delete()/forget() operate on the memories table; pins live in a separate
    namespace and must be unaffected even when the query exactly matches.
    """
    e = engram_factory()
    try:
        pid = e.pin(content)
        # Add a memory with the same content so delete has something to match.
        e.remember(content, salience=0.5)
        e.rebuild()

        e.delete(query=content)

        pins_after = {p["id"] for p in e._store.get_pins()}
        assert pid in pins_after, (
            f"P-S4 violated: delete(query={content!r}) removed pin {pid}"
        )
        # And it still surfaces in active_context.
        assert content in e.active_context(max_tokens=8192)
    finally:
        try:
            e.close()
        except Exception:
            pass

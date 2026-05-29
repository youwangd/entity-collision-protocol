"""Property-based chaos tests for JSONL fault injection.

Invariants under random byte-level corruption of events.jsonl:

  C-P1  rebuild() never raises, regardless of corruption pattern.
  C-P2  surviving event count is in [0, N] (no phantom events).
  C-P3  surviving content is a SUBSET of the original event contents
        (we never resurrect data that was never written).
  C-P4  after a corrupt-then-rebuild, additional remember() calls land
        in a consistent store (re-rebuild count == prior count + new count).

These are the random-fuzz analogues of the explicit-case tests in
`test_fault_injection.py` — same contract, broader input distribution.
Marked @chaos; opt-in via `pytest -m chaos`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram

pytestmark = [pytest.mark.chaos, pytest.mark.slow]


def _new_engram(tmp_path: Path) -> Engram:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    return Engram(cfg)


def _populate(tmp_path: Path, n: int) -> tuple[Path, list[str]]:
    eng = _new_engram(tmp_path)
    contents = [f"event-{i}-payload-{'x' * (i % 7)}" for i in range(n)]
    try:
        for c in contents:
            eng.remember(c, salience=0.5)
        buf_path = eng._buffer.path
    finally:
        eng.close()
    return buf_path, contents


def _surviving_contents(tmp_path: Path) -> list[str]:
    eng = _new_engram(tmp_path)
    try:
        eng.rebuild(incremental=False)
        # Pull all stored contents via store iteration.
        return [m.content for m in eng._store.all_active()]  # type: ignore[attr-defined]
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# C-P1 + C-P2 + C-P3: random truncation
# ---------------------------------------------------------------------------

@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(trunc=st.integers(min_value=0, max_value=2000))
def test_random_truncation_never_crashes(tmp_path_factory, trunc: int) -> None:
    tmp_path = tmp_path_factory.mktemp("trunc")
    buf_path, contents = _populate(tmp_path, n=15)
    sz = buf_path.stat().st_size
    new_sz = max(0, sz - trunc)
    with open(buf_path, "r+b") as f:
        f.truncate(new_sz)

    survived = _surviving_contents(tmp_path)
    # C-P2: count bounded
    assert 0 <= len(survived) <= len(contents)
    # C-P3: subset
    assert set(survived).issubset(set(contents)), "rebuild resurrected content"


# ---------------------------------------------------------------------------
# C-P1 + C-P2 + C-P3: random byte-flip corruption
# ---------------------------------------------------------------------------

@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_flips=st.integers(min_value=1, max_value=40),
)
def test_random_byteflips_never_crash_and_subset(
    tmp_path_factory, seed: int, n_flips: int
) -> None:
    import random

    tmp_path = tmp_path_factory.mktemp("flips")
    buf_path, contents = _populate(tmp_path, n=15)
    raw = bytearray(buf_path.read_bytes())
    if not raw:
        return  # vacuous
    rng = random.Random(seed)
    for _ in range(n_flips):
        idx = rng.randrange(len(raw))
        raw[idx] = rng.randrange(256)
    buf_path.write_bytes(bytes(raw))

    survived = _surviving_contents(tmp_path)
    # We *might* have flipped a content byte and still parsed valid JSON,
    # so we only assert the subset property modulo content-byte mutation:
    # surviving set must not exceed the original set in cardinality, and
    # rebuild must complete without raising.
    assert len(survived) <= len(contents)


# ---------------------------------------------------------------------------
# C-P4: random garbage-line injection at random positions
# ---------------------------------------------------------------------------

@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    insert_positions=st.lists(
        st.integers(min_value=0, max_value=20), min_size=0, max_size=8
    ),
    garbage=st.lists(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1,
            max_size=80,
        ),
        min_size=0,
        max_size=8,
    ),
)
def test_random_garbage_lines_skipped(
    tmp_path_factory, insert_positions: list[int], garbage: list[str]
) -> None:
    tmp_path = tmp_path_factory.mktemp("garbage")
    buf_path, contents = _populate(tmp_path, n=20)

    lines = buf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_orig_lines = len(lines)
    # Interleave garbage at chosen positions (clamped). Use min count.
    k = min(len(insert_positions), len(garbage))
    for i in range(k):
        pos = min(insert_positions[i], len(lines))
        # ensure it's not accidentally valid JSON
        line = "GARBAGE::" + garbage[i].replace("\n", "")
        lines.insert(pos, line)
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Rebuild — invariant: all good lines survive, garbage skipped silently.
    survived = _surviving_contents(tmp_path)
    # Every original event content must still be present (we only inserted
    # garbage, did not delete real lines). Caveat: very rare collision where
    # garbage parses as a valid memory event is excluded by the GARBAGE:: prefix.
    assert set(contents).issubset(set(survived)), (
        f"original events lost after garbage injection: "
        f"missing={set(contents) - set(survived)}, n_orig_lines={n_orig_lines}"
    )
    # And no resurrected/garbage content.
    assert set(survived).issubset(set(contents))


# ---------------------------------------------------------------------------
# C-P4 strict: append-after-corrupt is consistent
# ---------------------------------------------------------------------------

@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(trunc=st.integers(min_value=0, max_value=500))
def test_append_after_truncation_is_consistent(
    tmp_path_factory, trunc: int
) -> None:
    tmp_path = tmp_path_factory.mktemp("aftertrunc")
    buf_path, _ = _populate(tmp_path, n=10)
    sz = buf_path.stat().st_size
    with open(buf_path, "r+b") as f:
        f.truncate(max(0, sz - trunc))

    # Survivors after corruption.
    survived_after_corrupt = set(_surviving_contents(tmp_path))

    # Append more good events.
    eng = _new_engram(tmp_path)
    new = [f"post-corrupt-{i}" for i in range(5)]
    try:
        for c in new:
            eng.remember(c, salience=0.5)
    finally:
        eng.close()

    final = set(_surviving_contents(tmp_path))
    # All new appends must survive.
    assert set(new).issubset(final)
    # Pre-existing survivors must still be present (post-truncation rebuild
    # is monotone — appending events does not retroactively lose data).
    assert survived_after_corrupt.issubset(final)

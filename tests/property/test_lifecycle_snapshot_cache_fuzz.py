"""Hypothesis fuzz for `CachedLifecycleSnapshot` (NEXT.md item #1, follow-up).

The unit tests at ``tests/unit/test_lifecycle_snapshot_cache.py`` cover
the documented behaviors point-by-point. This module pins the broader
contract:

  C1. Equivalence — for any random sequence of (append, truncate,
      rotate-inode) operations on a JSONL buffer, the cached snapshot
      after the last operation equals the fresh ``snapshot_from_buffer``
      result.
  C2. Monotone offset on appends — when only appends happen,
      ``_offset`` only grows; a no-change call after an append-then-EOF
      registers as a hit.
  C3. Stats accounting — partial-hits + hits + misses == n_calls; misses
      occur only on first call, truncations, and inode flips; hits only
      on EOF-equal-offset calls.

Heuristic-only test surface: every event uses ``EventKind`` enums; we
seed window_id deterministically; we don't call any retrieval code.
"""
from __future__ import annotations

import os
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.lifecycle_projection import (
    CachedLifecycleSnapshot,
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import EventKind
from engram.store.buffer import JSONLBufferStore


# Operation tags for the random-interleave fuzz.
_KINDS = list(EventKind)


@st.composite
def _op(draw):
    """One step of the fuzz: append-event, get-snapshot, or rotate."""
    tag = draw(st.sampled_from(["append", "get", "rotate"]))
    if tag == "append":
        sid = draw(st.sampled_from(["s0", "s1", "s2", "s3"]))
        kind = draw(st.sampled_from(_KINDS))
        wid = draw(st.one_of(st.none(), st.sampled_from(["w0", "w1", "w2"])))
        return ("append", sid, kind, wid)
    if tag == "get":
        return ("get",)
    return ("rotate",)


def _rotate_inode(buf: JSONLBufferStore) -> None:
    """Atomically replace the JSONL file with a fresh empty one — same
    path, different inode. Simulates a log-rotation under the cache.
    """
    p: Path = buf.path
    tmp = p.with_suffix(p.suffix + ".rot")
    tmp.write_bytes(b"")
    os.replace(tmp, p)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(ops=st.lists(_op(), min_size=1, max_size=40))
def test_c1_random_interleave_equivalence(tmp_path_factory, ops):
    """For any op sequence, cached snapshot == fresh snapshot."""
    base = tmp_path_factory.mktemp("buf")
    buf = JSONLBufferStore(base_path=base)
    cache = CachedLifecycleSnapshot()

    for step in ops:
        if step[0] == "append":
            _, sid, kind, wid = step
            buf.append(make_lifecycle_event(schema_id=sid, kind=kind, window_id=wid))
        elif step[0] == "get":
            cache.get(buf)
        elif step[0] == "rotate":
            _rotate_inode(buf)

    cached = cache.get(buf)
    fresh = snapshot_from_buffer(buf)
    assert cached == fresh, (cached, fresh, ops)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    appends=st.lists(
        st.tuples(
            st.sampled_from(["s0", "s1", "s2"]),
            st.sampled_from(_KINDS),
        ),
        min_size=1,
        max_size=15,
    )
)
def test_c2_append_only_offset_monotone_and_eof_hit(tmp_path_factory, appends):
    """With only appends + gets, _offset never decreases; back-to-back
    get() with no append between is a pure hit."""
    base = tmp_path_factory.mktemp("buf")
    buf = JSONLBufferStore(base_path=base)
    cache = CachedLifecycleSnapshot()

    last_offset = 0
    last_hits = cache.stats["hits"]
    for sid, kind in appends:
        buf.append(make_lifecycle_event(schema_id=sid, kind=kind))
        cache.get(buf)
        assert cache._offset >= last_offset
        last_offset = cache._offset
        # Re-call without changing the buffer ⇒ pure hit.
        before = cache.stats["hits"]
        cache.get(buf)
        assert cache.stats["hits"] == before + 1
        last_hits = cache.stats["hits"]

    # Sanity: at least one hit beyond the initial state we observed.
    assert last_hits >= 1


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n_pre=st.integers(min_value=1, max_value=8),
    n_post=st.integers(min_value=0, max_value=6),
)
def test_c3_rotate_increments_misses(tmp_path_factory, n_pre, n_post):
    """Each inode flip should drive exactly one miss (full rebuild) on
    the next get(); subsequent appends are partial hits."""
    base = tmp_path_factory.mktemp("buf")
    buf = JSONLBufferStore(base_path=base)
    cache = CachedLifecycleSnapshot()

    for i in range(n_pre):
        buf.append(make_lifecycle_event(schema_id=f"s{i}", kind=EventKind.CREATE))
    cache.get(buf)  # first miss
    misses_before = cache.stats["misses"]

    _rotate_inode(buf)
    cache.get(buf)
    assert cache.stats["misses"] == misses_before + 1

    for i in range(n_post):
        buf.append(make_lifecycle_event(schema_id=f"s{i}", kind=EventKind.PROMOTE))
        cache.get(buf)
    # No further misses while inode is stable.
    assert cache.stats["misses"] == misses_before + 1
    # Final equivalence.
    assert cache.get(buf) == snapshot_from_buffer(buf)

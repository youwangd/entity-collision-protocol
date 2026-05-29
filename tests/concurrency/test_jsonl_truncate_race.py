"""Race-condition gate for ``JSONLBufferStore.truncate_before``.

Background
----------
Unlike ``append`` — which rides on POSIX O_APPEND atomicity for writes
``<= PIPE_BUF`` (see ``test_jsonl_buffer_concurrency.py``) —
``truncate_before`` is read-then-rewrite. Naively that races against
concurrent ``append``s:

  (a) Append lands after the read snapshot but before the rewrite
      publishes → bytes physically present, then clobbered. Silent
      data loss.
  (b) Append interleaves into the in-flight rewriter handle → byte-level
      corruption.

The fix (commit 57c03f0, reinforced by 774924e) holds an exclusive
``fcntl.flock`` across read+rewrite and publishes via ``os.replace``
into a ``.tmp`` sidecar; appenders block on the same flock. Both modes
are now closed.

This module is the regression gate that pins that behavior. Both tests
must stay green — a regression here is data loss.

Marked ``concurrency``; opt-in via ``pytest -m concurrency``.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.core.types import Event, EventType
from engram.store.buffer import JSONLBufferStore

pytestmark = pytest.mark.concurrency


def _hammer_appender(args):
    """Append ``count`` fresh-timestamp events as fast as possible. Returns ids."""
    base_path, worker_id, count, start_barrier_path = args
    # Spin until barrier file appears (cheap cross-process sync).
    while not os.path.exists(start_barrier_path):
        time.sleep(0.001)
    store = JSONLBufferStore(Path(base_path))
    ids = []
    for i in range(count):
        ev = Event(
            id=f"new-w{worker_id:02d}-e{i:05d}",
            type=EventType.EXPLICIT_REMEMBER,
            ts=datetime.now(timezone.utc),
            content=f"fresh worker {worker_id} ev {i}",
            metadata={"phase": "during_truncate", "worker": worker_id},
        )
        store.append(ev)
        ids.append(ev.id)
    return ids


def _truncator(args):
    """Call truncate_before(cutoff) once. Returns count removed."""
    base_path, cutoff_iso, start_barrier_path = args
    while not os.path.exists(start_barrier_path):
        time.sleep(0.001)
    store = JSONLBufferStore(Path(base_path))
    cutoff = datetime.fromisoformat(cutoff_iso)
    return store.truncate_before(cutoff)


def _seed_old_events(path: Path, count: int) -> list[str]:
    """Seed the buffer with N events timestamped 2 days in the past.
    Those are the events the truncate is targeting."""
    store = JSONLBufferStore(path)
    old_ts = datetime.now(timezone.utc) - timedelta(days=2)
    ids = []
    for i in range(count):
        ev = Event(
            id=f"old-e{i:05d}",
            type=EventType.EXPLICIT_REMEMBER,
            ts=old_ts,
            content=f"stale event {i}",
            metadata={"phase": "pre_truncate"},
        )
        # Bypass auto-now timestamp — JSONLBufferStore.append uses ev.ts as-is.
        store.append(ev)
        ids.append(ev.id)
    return ids


def test_serial_truncate_no_loss(tmp_path: Path):
    """Sanity: serialized truncate (no concurrency) loses zero fresh events.

    This is the property we *wish* held under concurrency too.
    """
    n_old = 500
    n_fresh = 500

    old_ids = _seed_old_events(tmp_path, n_old)
    store = JSONLBufferStore(tmp_path)
    fresh_ids = []
    for i in range(n_fresh):
        ev = Event(
            id=f"fresh-e{i:05d}",
            type=EventType.EXPLICIT_REMEMBER,
            ts=datetime.now(timezone.utc),
            content=f"fresh {i}",
        )
        store.append(ev)
        fresh_ids.append(ev.id)

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    removed = store.truncate_before(cutoff)
    assert removed == n_old

    log = tmp_path / "events.jsonl"
    surviving = {json.loads(l)["id"] for l in log.read_text().splitlines() if l.strip()}
    assert surviving == set(fresh_ids), (
        f"serial truncate dropped/added events: missing={len(set(fresh_ids) - surviving)} "
        f"extra={len(surviving - set(fresh_ids))}"
    )
    # And every old id must be gone.
    assert not (set(old_ids) & surviving)


def test_concurrent_truncate_vs_append_no_loss(tmp_path: Path):
    """Hammer append() in N processes while one process calls truncate_before.

    *Property*: every fresh-timestamp event written by appenders survives
    the truncate. The truncate is targeting old events (2-day-stale) only.
    The exclusive flock + atomic rename in ``truncate_before`` is what
    makes this hold; without that, appends landing inside the rewrite
    window get clobbered. A regression here is silent data loss.
    """
    n_appenders = 8
    appends_per_worker = 200
    expected_fresh = n_appenders * appends_per_worker

    # Seed enough old events that the rewrite step takes measurable time —
    # widens the race window so the test is more deterministic.
    n_old = 5000
    old_ids = _seed_old_events(tmp_path, n_old)

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    barrier = tmp_path / "GO"

    ctx = mp.get_context("fork")
    appender_args = [
        (str(tmp_path), wid, appends_per_worker, str(barrier))
        for wid in range(n_appenders)
    ]
    truncator_args = [(str(tmp_path), cutoff.isoformat(), str(barrier))]

    with ctx.Pool(n_appenders + 1) as pool:
        appender_handle = pool.map_async(_hammer_appender, appender_args)
        truncator_handle = pool.map_async(_truncator, truncator_args)
        # Release the barrier — all workers race from here.
        time.sleep(0.05)
        barrier.touch()
        written_batches = appender_handle.get(timeout=60)
        [removed] = truncator_handle.get(timeout=60)

    written_fresh = {i for batch in written_batches for i in batch}
    assert len(written_fresh) == expected_fresh, "writer-side id collision (test bug)"
    assert removed >= 0  # truncator ran

    log = tmp_path / "events.jsonl"
    raw_lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    surviving_ids: set[str] = set()
    for n, line in enumerate(raw_lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"corrupted line {n}: {e}: {line[:120]!r}")
        surviving_ids.add(obj["id"])

    lost = written_fresh - surviving_ids
    assert not lost, (
        f"truncate_before clobbered {len(lost)}/{expected_fresh} fresh appends. "
        f"sample={list(lost)[:5]}"
    )
    # And no old ids should have leaked back in.
    assert not (set(old_ids) & surviving_ids)

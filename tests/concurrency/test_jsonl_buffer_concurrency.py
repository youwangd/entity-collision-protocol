"""POSIX append-atomicity torture for JSONLBufferStore.

Background
----------
``JSONLBufferStore.append`` opens the events.jsonl file in ``"a"`` mode and
issues a single ``write()`` of one line.  POSIX guarantees that, for a file
opened with ``O_APPEND``, writes of size ``<= PIPE_BUF`` (typically 4096
bytes on Linux) are atomic with respect to other ``O_APPEND`` writers — no
interleaving, no truncation, no torn lines.

That guarantee is the load-bearing assumption behind Engram's "the JSONL
log is the source of truth" claim.  This test pins it from two sides:

  1. **Atomicity property:** N processes (real fork()ed processes, not
     threads — the GIL would mask the race) hammer the same buffer
     concurrently.  Every line must round-trip cleanly through
     ``json.loads`` and the read-back set of event ids must equal the
     write-side set.  No torn lines, no lost lines, no duplicates.

  2. **Size invariant:** every event Engram emits in practice must
     serialize to ``<= PIPE_BUF`` bytes (4096 on Linux/tmpfs).  We pin
     a worst-case-ish event (long content, dense metadata, full Event
     fields populated) and assert it stays under the cap.  If a future
     change blows past 4 KB, the atomicity guarantee silently degrades
     and this test fails loudly instead.

Marked ``concurrency``; opt-in via ``pytest -m concurrency``.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram.core.types import Event, EventType
from engram.store.buffer import JSONLBufferStore

pytestmark = pytest.mark.concurrency


# PIPE_BUF on Linux. POSIX-mandated minimum is 512; Linux gives 4096.
PIPE_BUF = os.pathconf("/tmp", "PC_PIPE_BUF")


def _writer(args):
    """Write ``count`` events to the shared buffer. Returns list of ids."""
    base_path, worker_id, count = args
    store = JSONLBufferStore(Path(base_path))
    ids = []
    for i in range(count):
        ev = Event(
            id=f"w{worker_id:02d}-e{i:05d}",
            type=EventType.EXPLICIT_REMEMBER,
            ts=datetime.now(timezone.utc),
            # ~200 bytes of content; full event line stays well under PIPE_BUF.
            content=f"worker {worker_id} event {i} " + ("x" * 100),
            metadata={"worker": worker_id, "seq": i},
        )
        store.append(ev)
        ids.append(ev.id)
    return ids


def test_concurrent_append_no_torn_lines(tmp_path: Path):
    """Many forked writers + one shared events.jsonl: every line valid JSON,
    set-equality between written and read event ids."""
    n_workers = 16
    per_worker = 250
    expected_total = n_workers * per_worker

    # Use fork so workers inherit nothing weird from pytest.
    ctx = mp.get_context("fork")
    args = [(str(tmp_path), wid, per_worker) for wid in range(n_workers)]
    with ctx.Pool(n_workers) as pool:
        written = pool.map(_writer, args)

    written_ids = {i for batch in written for i in batch}
    assert len(written_ids) == expected_total, "writer-side id collision (test bug)"

    # Read back. Bypass JSONLBufferStore.scan() (which silently drops
    # corrupted lines) — we want to *see* corruption, not paper over it.
    log = tmp_path / "events.jsonl"
    raw_lines = log.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == expected_total, (
        f"line-count mismatch: wrote {expected_total}, read {len(raw_lines)} "
        f"(possible lost or merged writes)"
    )

    read_ids = set()
    for n, line in enumerate(raw_lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"torn line at {n}: {e}: {line[:120]!r}")
        read_ids.add(obj["id"])

    assert read_ids == written_ids, (
        f"id-set mismatch: missing={len(written_ids - read_ids)} "
        f"extra={len(read_ids - written_ids)}"
    )


def test_event_serialized_size_under_pipe_buf():
    """Worst-case-ish event must serialize to <= PIPE_BUF bytes — otherwise
    POSIX no longer guarantees atomic appends and the concurrent-write
    invariant above is unsound.

    'Worst case' here = a content payload up near our practical cap plus a
    metadata dict roughly the size of what consolidation stages emit
    (extraction confidence, schema_id, source_event_id, salience, etc.).
    """
    # 1500-char content is already an aggressive upper bound for a single
    # remembered fact; consolidation typically chunks long inputs.
    content = "the quick brown fox jumps over the lazy dog. " * 33  # ~1485
    metadata = {
        "schema_id": "schema-" + "a" * 32,
        "source_event_id": "evt-" + "b" * 32,
        "extraction_confidence": 0.87654321,
        "salience": 0.5,
        "tags": ["pii:none", "lang:en", "ext:fact-v2", "stage:6"],
        "props": {f"k{i}": f"v{i}-{'x' * 8}" for i in range(8)},
    }
    ev = Event(
        id="evt-" + "c" * 32,
        type=EventType.EXPLICIT_REMEMBER,
        ts=datetime.now(timezone.utc),
        content=content,
        metadata=metadata,
        salience_hint=0.9,
        context={"session": "s-" + "d" * 16, "user": "u-" + "e" * 16},
    )
    line = json.dumps(ev.to_dict(), separators=(",", ":")) + "\n"
    size = len(line.encode("utf-8"))
    assert size <= PIPE_BUF, (
        f"event line is {size} bytes > PIPE_BUF ({PIPE_BUF}); POSIX no longer "
        f"guarantees atomic concurrent appends. Either chunk the field "
        f"causing growth or document the loss of atomicity."
    )

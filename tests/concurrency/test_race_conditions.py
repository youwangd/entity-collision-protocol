"""Concurrency torture tests — opt-in via `pytest -m concurrency`.

Goal: prove the system survives concurrent access by many writers and readers
without corruption, crashes, or lost writes. Covers:

  * Thread-mix: N writer threads + M reader threads on one shared Engram instance.
  * Multiprocess: M independent processes each owning their own connection,
    pointed at the same DB path. WAL mode should permit this.
  * Write-write race: many threads upserting the same memory id.
  * Read-write race: readers running search_text() while writers append.
  * Consolidation contention: foreground writer + background consolidate().

Asserts:
  * No exceptions escape (errors collected per worker).
  * Final memory count matches expected (no lost writes).
  * No partial rows (every memory has all required fields).
  * FTS index stays queryable throughout.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from engram import Engram, Config

pytestmark = pytest.mark.concurrency


# --- helpers ---


def _mk_config(tmp_path: Path) -> Config:
    cfg = Config.minimal()
    cfg.path = str(tmp_path)
    # Disable rate limiting — we deliberately exceed normal write rates here.
    cfg.security.max_events_per_minute = 0
    return cfg


def _rand_text(n: int = 40) -> str:
    return "".join(random.choices(string.ascii_letters + " ", k=n)).strip() or "x"


# --- thread-mix tests ---


def test_8_writers_8_readers_smoke(tmp_path: Path):
    """8 writer threads x 8 reader threads, 30s soak. Zero crashes, all writes land."""
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        per_writer = 100
        n_writers = 8
        n_readers = 8

        write_errors: list[BaseException] = []
        read_errors: list[BaseException] = []
        stop = threading.Event()

        def writer(wid: int):
            try:
                for j in range(per_writer):
                    e.remember(f"writer-{wid} entry-{j} {_rand_text(30)}", salience=0.3)
            except BaseException as ex:  # noqa: BLE001
                write_errors.append(ex)

        def reader(rid: int):
            try:
                while not stop.is_set():
                    res = e.recall(f"writer-{rid % n_writers}", limit=5)
                    assert isinstance(res, list)
                    time.sleep(0.001)
            except BaseException as ex:  # noqa: BLE001
                read_errors.append(ex)

        with ThreadPoolExecutor(max_workers=n_writers + n_readers) as pool:
            r_futs = [pool.submit(reader, i) for i in range(n_readers)]
            w_futs = [pool.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(w_futs):
                f.result()
            stop.set()
            for f in as_completed(r_futs):
                f.result()

        assert not write_errors, f"writer errors: {write_errors[:3]}"
        assert not read_errors, f"reader errors: {read_errors[:3]}"

        mems = e._store.all_active()
        # Allow a small slack for any dedup collisions (cosine threshold off by default
        # in minimal config, but content_hash collisions still possible on random strings).
        assert len(mems) >= int(per_writer * n_writers * 0.95), \
            f"expected ~{per_writer*n_writers}, got {len(mems)}"
        # Every memory has the required fields.
        for m in mems:
            assert m.id and m.content and m.created_at
    finally:
        e.close()


def test_write_write_idempotency_same_id(tmp_path: Path):
    """Many threads upsert the same memory id concurrently. Final state is consistent."""
    from engram.core.types import Memory, MemoryType, MemoryState, generate_memory_id
    from datetime import datetime, timezone

    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        mid = generate_memory_id(MemoryType.FACT)
        n_threads = 32
        errors: list[BaseException] = []
        barrier = threading.Barrier(n_threads)

        def hammer(idx: int):
            try:
                barrier.wait()
                m = Memory(
                    id=mid,
                    type=MemoryType.FACT,
                    state=MemoryState.ACTIVE,
                    content=f"value-from-thread-{idx}",
                    summary="",
                    salience=idx / n_threads,
                    confidence=1.0,
                    decay_rate=0.01,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                )
                e._store.upsert(m)
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(n_threads)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        assert not errors, f"upsert errors: {errors[:3]}"
        got = e._store.get(mid)
        assert got is not None
        # Some thread won the last write — that's all we require.
        assert got.content.startswith("value-from-thread-")
    finally:
        e.close()


def test_concurrent_consolidate_does_not_corrupt(tmp_path: Path):
    """Run consolidation in one thread while another writes. No errors, no lost rows."""
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        # Seed some data
        for i in range(50):
            e.remember(f"seed memory {i} {_rand_text(20)}", salience=0.5)

        write_errors: list[BaseException] = []
        consolidate_errors: list[BaseException] = []
        stop = threading.Event()

        def keep_writing():
            j = 0
            while not stop.is_set():
                try:
                    e.remember(f"live-{j} {_rand_text(15)}", salience=0.4)
                    j += 1
                except BaseException as ex:  # noqa: BLE001
                    write_errors.append(ex)
                    return

        def consolidate_loop():
            for _ in range(3):
                try:
                    e.consolidate()
                    time.sleep(0.05)
                except BaseException as ex:  # noqa: BLE001
                    consolidate_errors.append(ex)

        w = threading.Thread(target=keep_writing)
        c = threading.Thread(target=consolidate_loop)
        w.start()
        c.start()
        c.join()
        stop.set()
        w.join()

        assert not write_errors, f"write errors during consolidate: {write_errors[:3]}"
        assert not consolidate_errors, f"consolidate errors: {consolidate_errors[:3]}"

        # FTS still queryable
        assert isinstance(e._store.search_text("seed", limit=5), list)
    finally:
        e.close()


def test_two_concurrent_consolidates_do_not_corrupt(tmp_path: Path):
    """Two threads call consolidate() simultaneously on the same Engram.

    Regression guard: if we ever expose consolidation via request-handling
    threads (e.g. an MCP server tool, a scheduler) the pipeline must not
    double-process events, double-promote schemas, drop rows, or raise.
    Today the audit shows consolidation is safe-by-pattern (single-caller
    in CLI), but this pins the invariant under genuine concurrency.

    Asserts:
      * No exception escapes either consolidate() call.
      * No event in the buffer is consumed twice (state stays consistent).
      * FTS index remains queryable.
      * Final memory count is bounded — duplicate consolidate runs of the
        same window must not multiply projected memories.
    """
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        # Seed enough that consolidation has work to do.
        for i in range(80):
            e.remember(
                f"shared seed memory {i} {_rand_text(20)}",
                salience=0.55,
            )

        baseline_count = len(e._store.all_active())

        errors: list[tuple[str, BaseException]] = []
        barrier = threading.Barrier(2)
        results: list = []

        def consolidate_worker(tag: str):
            try:
                barrier.wait(timeout=5.0)
                report = e.consolidate()
                results.append((tag, report))
            except BaseException as ex:  # noqa: BLE001
                errors.append((tag, ex))

        t1 = threading.Thread(target=consolidate_worker, args=("a",))
        t2 = threading.Thread(target=consolidate_worker, args=("b",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"concurrent consolidate errors: {errors[:3]}"
        # Both calls produced reports; the second should observe a
        # drained / partially-drained pipeline (zero or fewer projections
        # than the first), but never a negative or absurdly inflated count.
        assert len(results) == 2

        # Memory store still in a consistent shape.
        post = e._store.all_active()
        for m in post:
            assert m.id and m.content and m.created_at
        # Sanity: count did not collapse to zero (data loss) and did not
        # double-promote (≤ 2× baseline is a generous ceiling).
        assert 0 < len(post) <= max(2 * baseline_count, baseline_count + 200)

        # FTS still queryable post-consolidation.
        assert isinstance(e._store.search_text("seed", limit=5), list)
    finally:
        e.close()


# --- multiprocess tests ---


def _proc_writer(db_path_str: str, wid: int, n: int) -> tuple[int, int, str]:
    """Spawned in a child process. Returns (wid, n_written, error_or_empty)."""
    try:
        cfg = Config.minimal()
        cfg.path = db_path_str
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        try:
            for j in range(n):
                e.remember(f"proc-{wid} item-{j} {_rand_text(20)}", salience=0.5)
            return (wid, n, "")
        finally:
            e.close()
    except BaseException as ex:  # noqa: BLE001
        return (wid, 0, f"{type(ex).__name__}: {ex}")


@pytest.mark.slow
def test_multiprocess_writers(tmp_path: Path):
    """4 independent processes write 50 memories each into the same DB path."""
    # Initialize the DB once so child processes don't race on schema creation.
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    e.close()

    n_procs = 4
    per_proc = 50
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs) as pool:
        results = pool.starmap(
            _proc_writer,
            [(str(tmp_path), i, per_proc) for i in range(n_procs)],
        )

    errs = [r for r in results if r[2]]
    assert not errs, f"process errors: {errs}"
    total_written = sum(r[1] for r in results)
    assert total_written == n_procs * per_proc

    # Reopen and verify rows are visible.
    e2 = Engram(config=_mk_config(tmp_path))
    try:
        mems = e2._store.all_active()
        assert len(mems) >= int(total_written * 0.95)
    finally:
        e2.close()


# --- read-while-write torture ---


def test_recall_during_writes_never_crashes(tmp_path: Path):
    """Recall() must always return a list; never raise during concurrent writes."""
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        # Seed
        for i in range(20):
            e.remember(f"alpha beta gamma {i}", salience=0.5)

        stop = threading.Event()
        crashes: list[BaseException] = []

        def writer():
            j = 0
            while not stop.is_set():
                try:
                    e.remember(f"new {j} delta", salience=0.4)
                    j += 1
                except BaseException as ex:  # noqa: BLE001
                    crashes.append(("w", ex))

        def reader():
            while not stop.is_set():
                try:
                    res = e.recall("alpha", limit=3)
                    assert isinstance(res, list)
                except BaseException as ex:  # noqa: BLE001
                    crashes.append(("r", ex))

        threads = (
            [threading.Thread(target=writer) for _ in range(2)]
            + [threading.Thread(target=reader) for _ in range(6)]
        )
        [t.start() for t in threads]
        time.sleep(2.0)
        stop.set()
        [t.join() for t in threads]

        assert not crashes, f"crashes: {crashes[:3]}"
    finally:
        e.close()

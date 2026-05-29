"""Concurrency stateful machine over the Engram public API.

Hypothesis explores random interleavings of *batched concurrent* operations.
Each rule fans K threads at the Engram instance simultaneously (writers,
readers, deleters), then checks global invariants once the batch quiesces.
This is the missing layer between:

  * `tests/property/test_engram_stateful.py` — sequential interleavings
  * `tests/concurrency/test_race_conditions.py` — explicit-schedule torture

Hypothesis doesn't natively support multi-threaded rules, so we encapsulate
the threading inside each rule. Hypothesis still drives the *macro* schedule
(which batch type runs next, and with what payload), and the GIL + each
batch's ThreadPoolExecutor produce micro-interleavings that vary across
Python invocations.

Invariants checked after every batch quiesces:

  C-I1  status['total_memories'] == sum(by_state.values()).
  C-I2  Hard-deleted ids never reappear in get() or recall().
  C-I3  No worker raised any exception (other than the expected
        ValueError on empty content, which is filtered out at generation).
  C-I4  After a batch of N identical-content writes WITH dedup off
        (write_dedup_threshold=0), at least one survived.
  C-I5  status() never returns torn / partial dicts (always has expected
        keys).
"""
from __future__ import annotations

import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from engram import Config, Engram


_alphabet = st.text(
    alphabet=string.ascii_lowercase + " ", min_size=3, max_size=18
).filter(lambda s: bool(s.strip()))

_REQUIRED_STATUS_KEYS = {"total_memories", "by_state"}


class ConcurrentEngramMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        self._hard_deleted: set[str] = set()
        self._known_ids: set[str] = set()
        self._fanout: int = 6  # threads per batch — small to keep Hypothesis fast

    @initialize()
    def setup(self):
        import tempfile

        self._tmp = Path(tempfile.mkdtemp(prefix="engram-cstateful-"))
        cfg = Config.minimal()
        cfg.path = str(self._tmp)
        cfg.security.max_events_per_minute = 0
        # Dedup OFF for this machine: we want to assert "all writes land",
        # not "dedup leaked X". Dedup-under-contention has its own coverage
        # in tests/concurrency/test_dedup_race.py.
        cfg.storage.write_dedup_threshold = 0.0
        self._engram = Engram(config=cfg)

    # ------------------------------------------------------------------
    # batched concurrent rules
    # ------------------------------------------------------------------

    def _run_concurrently(self, fn, args_list):
        """Run fn(*args) for each args in args_list with a fresh thread pool.
        Returns (results, errors). Uses a Barrier so all threads arrive at
        the API simultaneously."""
        n = len(args_list)
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []
        results: list = []

        def worker(args):
            try:
                barrier.wait()
                return fn(*args)
            except BaseException as ex:  # noqa: BLE001 — collect for assertion
                errors.append(ex)
                return None

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(worker, a) for a in args_list]
            for f in as_completed(futs):
                results.append(f.result())
        return results, errors

    @rule(target=memory_ids, content=_alphabet, sal=st.floats(0.0, 1.0))
    def concurrent_remember_same_content(self, content, sal):
        """K writers race on the same content (dedup is off — all should land)."""
        e = self._engram
        assert e is not None
        before = e.status()["total_memories"]
        args = [(content, sal) for _ in range(self._fanout)]
        results, errors = self._run_concurrently(
            lambda c, s: e.remember(c, salience=s), args
        )
        assert not errors, f"C-I3: writer raised {type(errors[0]).__name__}: {errors[0]}"
        # C-I4: every concurrent write returned a usable id, and after
        # rebuild the store grew by exactly fanout. We assert via
        # status() rather than recall() because recall() is intent-aware
        # (e.g. queries containing phrases like "remember when" or "what is"
        # route to a specific MemoryType, which can legitimately return zero
        # rows even when the writes landed). Concurrency is the property
        # under test here, not intent routing. (Note: remember() returns
        # an event id, not a memory id — we don't assert on it directly.)
        write_ret = [r for r in results if isinstance(r, str) and r]
        assert len(write_ret) == self._fanout, (
            f"C-I4: only {len(write_ret)}/{self._fanout} writes returned an id"
        )
        e.rebuild()
        after = e.status()["total_memories"]
        # Dual extraction (Governed Memory) can yield >1 fact per event,
        # and rebuild semantics are not 1:1 with input. The property under
        # test is that writes weren't all silently dropped under contention.
        assert after - before >= self._fanout, (
            f"C-I4: {self._fanout} concurrent writes of {content!r} only "
            f"yielded {after - before} new memories (status went {before}→{after})"
        )
        # Adopt one new memory id into the bundle (best-effort: derive via
        # recall on a non-intent-trigger query; if recall returns nothing,
        # synthesize a sentinel that subsequent rules will tolerate).
        hits = e.recall(content, limit=self._fanout + 2)
        if hits:
            for h in hits:
                self._known_ids.add(h.memory.id)
            return hits[0].memory.id
        # Fallback: return the event id; deletion rules tolerate missing ids.
        return write_ret[0]

    @rule(content=_alphabet)
    def concurrent_recall(self, content):
        """K readers race on recall — must never crash, never see hard-deleted."""
        e = self._engram
        assert e is not None
        args = [(content,) for _ in range(self._fanout)]
        results, errors = self._run_concurrently(
            lambda c: e.recall(c, limit=5), args
        )
        assert not errors, f"C-I3: reader raised {type(errors[0]).__name__}: {errors[0]}"
        for hits in results:
            if hits is None:
                continue
            for h in hits:
                assert h.memory.id not in self._hard_deleted, (
                    f"C-I2: hard-deleted {h.memory.id} resurfaced in concurrent recall"
                )

    @rule(mid=memory_ids)
    def concurrent_hard_delete_same_id(self, mid):
        """K deleters race on the *same* id — must be idempotent, no crash."""
        if mid in self._hard_deleted:
            return
        e = self._engram
        assert e is not None
        args = [(mid,) for _ in range(self._fanout)]
        _, errors = self._run_concurrently(
            lambda m: e.forget(id=m, hard=True), args
        )
        assert not errors, f"C-I3: deleter raised {type(errors[0]).__name__}: {errors[0]}"
        self._hard_deleted.add(mid)
        self._known_ids.discard(mid)
        # C-I2: gone for good.
        assert e.get(mid) is None, f"C-I2: {mid} still gettable after concurrent delete"

    @rule(content=_alphabet, sal=st.floats(0.0, 1.0))
    def writers_vs_readers(self, content, sal):
        """Mixed batch: half writers on `content`, half readers on `content`."""
        e = self._engram
        assert e is not None
        half = max(2, self._fanout // 2)

        def do_write():
            return e.remember(content, salience=sal)

        def do_read():
            return e.recall(content, limit=5)

        n = half * 2
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []
        ids: list = []

        def worker(is_writer: bool):
            try:
                barrier.wait()
                if is_writer:
                    ids.append(do_write())
                else:
                    do_read()
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(worker, True) for _ in range(half)] + [
                pool.submit(worker, False) for _ in range(half)
            ]
            for f in as_completed(futs):
                f.result()

        assert not errors, (
            f"C-I3: write/read race raised {type(errors[0]).__name__}: {errors[0]}"
        )
        e.rebuild()

    # ------------------------------------------------------------------
    # invariants
    # ------------------------------------------------------------------

    @invariant()
    def status_well_formed(self):
        if self._engram is None:
            return
        s = self._engram.status()
        # C-I5: status is never torn — required keys always present.
        assert _REQUIRED_STATUS_KEYS.issubset(s.keys()), (
            f"C-I5: status missing keys: {sorted(_REQUIRED_STATUS_KEYS - s.keys())}"
        )
        by_state_total = sum(s["by_state"].values())
        assert s["total_memories"] == by_state_total, (
            f"C-I1: total_memories={s['total_memories']} but by_state sum={by_state_total}"
        )

    @invariant()
    def hard_deleted_stay_dead(self):
        if self._engram is None:
            return
        for mid in self._hard_deleted:
            assert self._engram.get(mid) is None, (
                f"C-I2: hard-deleted {mid} reappeared in get()"
            )

    def teardown(self):
        if self._engram is not None:
            try:
                self._engram.close()
            except Exception:
                pass
        if self._tmp is not None:
            import shutil

            shutil.rmtree(self._tmp, ignore_errors=True)


# Keep budget tight: each rule spawns 6 threads, so step_count must be modest.
TestConcurrentEngramStateful = ConcurrentEngramMachine.TestCase
TestConcurrentEngramStateful.settings = settings(
    max_examples=12,
    stateful_step_count=18,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)

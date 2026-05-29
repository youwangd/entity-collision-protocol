"""Concurrency × FTS5 overlay state machine.

NEXT.md priority #1. Companion to:

  - tests/property/test_fts_index_stateful.py  (sequential FTS5 invariants)
  - tests/property/test_engram_concurrent_stateful.py (general concurrency)

This machine specifically races writers and deleters against FTS5 MATCH
readers, then asserts — once the batch quiesces — that the trigger-maintained
`memories_fts` table is consistent with the durable `memories` row state.

Invariants (post-quiesce; mid-batch torn reads are tolerated as long as
they are subsets of "ever-existed" content, never crashes):

  CF-I1  No worker raised an exception during a concurrent batch.
  CF-I2  After the batch, every active mid (token-tagged) is FTS5-reachable
         via its distinctive token.
  CF-I3  After the batch, every hard-forgotten mid is invisible to FTS5
         under any state filter (no ghost row left by the trigger).
  CF-I4  Mid-batch reads never returned a hard-forgotten mid that was
         already hard-forgotten *before* the batch began. (We don't make
         claims about deletes happening *inside* the batch — only about
         the durable invariant from the previous tick.)
  CF-I5  status() never returns torn dicts under contention (must always
         have the standard keys).
"""
from __future__ import annotations

import tempfile
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


# Disjoint, FTS5-friendly tokens — alphabetic, length>2, not stopwords,
# not FTS5 operators. Distinct from those in test_fts_index_stateful.py
# so a failure here cannot be confused with the sequential machine's.
_TOKENS = [
    "vermillion",
    "hexagon",
    "nautilus",
    "pyrite",
    "calliope",
    "thorax",
    "wisteria",
    "basalt",
]
_REQUIRED_STATUS_KEYS = {"total_memories", "by_state"}


class FTSConcurrentMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        self._counter = 0
        self._counter_lock = threading.Lock()
        # mid -> token (so we know what query should retrieve it)
        self._mid_token: dict[str, str] = {}
        # mid -> "active" | "hard_deleted"
        self._mid_state: dict[str, str] = {}
        self._fanout = 3  # writers per batch; readers add another fanout.

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-fts-cstateful-"))
        cfg = Config(path=str(self._tmp))
        cfg.security.max_events_per_minute = 0
        # Dedup off — every concurrent remember() lands its own row, so
        # when we look up by counter we get exactly one match.
        cfg.storage.write_dedup_threshold = 0.0
        self._engram = Engram(config=cfg)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _next_counter(self) -> int:
        with self._counter_lock:
            self._counter += 1
            return self._counter

    def _run(self, tagged_fns):
        """Run a list of (tag, zero-arg callable) pairs on a fresh thread
        pool with a Barrier so they all hit the API at once. Returns
        (tagged_results, errors), where tagged_results is a list of
        (tag, result) preserving the input pairing (not completion order).
        """
        n = len(tagged_fns)
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []
        results: list = [None] * n

        def worker(idx, fn):
            try:
                barrier.wait()
                return fn()
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)
                return None

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {
                pool.submit(worker, i, fn): i
                for i, (_tag, fn) in enumerate(tagged_fns)
            }
            for f in as_completed(futs):
                idx = futs[f]
                results[idx] = f.result()
        return [(tag, results[i]) for i, (tag, _fn) in enumerate(tagged_fns)], errors

    # ------------------------------------------------------------------
    # rules
    # ------------------------------------------------------------------

    @rule(
        target=memory_ids,
        ti=st.integers(min_value=0, max_value=len(_TOKENS) - 1),
    )
    def writers_vs_readers(self, ti):
        """K writers race K readers on the same token.

        Each writer remember()s a unique row tagged with `token` + a
        global counter. Each reader runs `search_text(token)`. After the
        batch quiesces, we look up the freshly-landed mids by counter
        and adopt one into the bundle.
        """
        e = self._engram
        assert e is not None
        token = _TOKENS[ti]
        # Snapshot of "ever hard-deleted before this batch" for CF-I4.
        pre_dead = {
            mid for mid, st_ in self._mid_state.items() if st_ == "hard_deleted"
        }

        # Build write payloads (each gets a unique counter).
        counters = [self._next_counter() for _ in range(self._fanout)]
        contents = [f"the {token} entry number {c} stands alone" for c in counters]

        def make_writer(content):
            return lambda: e.remember(content, salience=0.5)

        def make_reader():
            return lambda: e._store.search_text(
                token, limit=200, states=["active", "fading"]
            )

        fns = [("w", make_writer(c)) for c in contents] + [
            ("r", make_reader()) for _ in range(self._fanout)
        ]
        results, errors = self._run(fns)
        assert not errors, (
            f"CF-I1: writers/readers race raised "
            f"{type(errors[0]).__name__}: {errors[0]}"
        )

        # CF-I4: scan reader results for any pre-batch hard-deleted mid.
        for tag, res in results:
            if tag != "r" or res is None:
                continue
            for h in res:
                assert h.memory.id not in pre_dead, (
                    f"CF-I4: hard-deleted mid {h.memory.id} resurfaced "
                    f"in concurrent FTS5 search"
                )

        # Adopt the freshly-landed mids by counter.
        post_hits = e._store.search_text(
            token, limit=500, states=["active", "fading"]
        )
        new_mid = None
        for c in counters:
            for h in post_hits:
                if str(c) in h.memory.content and h.memory.id not in self._mid_token:
                    self._mid_token[h.memory.id] = token
                    self._mid_state[h.memory.id] = "active"
                    if new_mid is None:
                        new_mid = h.memory.id
                    break
        return new_mid or "noop"

    @rule(mid=memory_ids)
    def concurrent_hard_delete(self, mid):
        """K deleters race on the same mid against K readers.

        Idempotent delete + concurrent MATCH; afterward the mid must
        not appear under any state filter.
        """
        if mid == "noop":
            return
        e = self._engram
        assert e is not None
        if self._mid_state.get(mid) == "hard_deleted":
            return
        token = self._mid_token.get(mid)
        if token is None:
            return

        def make_deleter():
            return lambda: e.forget(id=mid, hard=True)

        def make_reader():
            return lambda: e._store.search_text(
                token, limit=200, states=["active", "fading"]
            )

        fns = [("d", make_deleter()) for _ in range(self._fanout)] + [
            ("r", make_reader()) for _ in range(self._fanout)
        ]
        _, errors = self._run(fns)
        assert not errors, (
            f"CF-I1: delete/read race raised "
            f"{type(errors[0]).__name__}: {errors[0]}"
        )
        self._mid_state[mid] = "hard_deleted"
        # CF-I3 immediately, every state filter:
        for state_set in (
            ["active", "fading"],
            ["active", "fading", "faded", "suppressed"],
        ):
            hits = e._store.search_text(token, limit=200, states=state_set)
            assert mid not in {h.memory.id for h in hits}, (
                f"CF-I3: hard-deleted mid {mid} still in FTS5 with "
                f"states={state_set}"
            )

    # ------------------------------------------------------------------
    # invariants
    # ------------------------------------------------------------------

    @invariant()
    def status_well_formed(self):
        # CF-I5
        e = self._engram
        if e is None:
            return
        s = e.status()
        assert _REQUIRED_STATUS_KEYS.issubset(s.keys()), (
            f"CF-I5: status missing keys: "
            f"{sorted(_REQUIRED_STATUS_KEYS - s.keys())}"
        )

    @invariant()
    def active_rows_searchable(self):
        # CF-I2
        e = self._engram
        if e is None:
            return
        for mid, token in self._mid_token.items():
            if self._mid_state.get(mid) != "active":
                continue
            hits = e._store.search_text(
                token, limit=500, states=["active", "fading"]
            )
            assert mid in {h.memory.id for h in hits}, (
                f"CF-I2: active mid {mid} (token={token!r}) not "
                f"FTS5-reachable post-batch"
            )

    @invariant()
    def hard_deleted_invisible(self):
        # CF-I3 (durable)
        e = self._engram
        if e is None:
            return
        for mid, token in self._mid_token.items():
            if self._mid_state.get(mid) != "hard_deleted":
                continue
            hits = e._store.search_text(
                token, limit=500,
                states=["active", "fading", "faded", "suppressed"],
            )
            assert mid not in {h.memory.id for h in hits}, (
                f"CF-I3: hard-deleted mid {mid} resurfaced in FTS5"
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


# Each rule fans out 6 threads (3 writers + 3 readers, or 3 deleters + 3
# readers). Keep examples × steps modest.
TestFTSConcurrentStateful = FTSConcurrentMachine.TestCase
TestFTSConcurrentStateful.settings = settings(
    max_examples=10,
    stateful_step_count=14,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)

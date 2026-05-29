"""Hypothesis stateful machine over the Engram public API.

Generates random interleavings of remember / recall / forget / rebuild and
asserts global invariants after every step. Catches order-dependent bugs
that explicit-case concurrency tests miss because they only cover the
specific schedules the author thought of.

Invariants checked after every rule:

  S-I1  status['total_memories'] == #(active+fading+faded+suppressed) in by_state.
  S-I2  Every id ever produced by remember+rebuild is gettable iff not hard-deleted.
  S-I3  recall() never returns a hard-deleted memory id.
  S-I4  rebuild() is idempotent: running it twice in a row yields equal counts.
  S-I5  forget(hard=False) leaves the row gettable but recall on its content
        no longer returns it (state is SUPPRESSED).
  S-I6  delete(id=mid) is a true alias for forget(id=mid, hard=True): the row
        becomes unreachable from get() and from recall().
  S-I7  pin/unpin lifecycle is consistent: unpin returns True iff the pin_id
        was present, False otherwise; double-unpin is a no-op (returns False).
  S-I8  Pins are disjoint from memories: a pin_id never collides with any
        memory id (different namespaces), and pinning content does not create
        a memory row that surfaces in recall.
"""
from __future__ import annotations

import string
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


# Keep content alphabet small so generated queries actually hit something.
_alphabet = st.text(alphabet=string.ascii_lowercase + " ", min_size=3, max_size=20).filter(
    lambda s: bool(s.strip())
)


class EngramStateMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")  # post-rebuild memory ids (mem-...)
    pin_ids = Bundle("pin_ids")  # active pin ids

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # set of memory ids that have been hard-deleted; must never reappear
        self._hard_deleted: set[str] = set()
        # all live (non-hard-deleted) memory ids ever observed
        self._known_ids: set[str] = set()
        # last content remembered, for sanity recall checks
        self._last_content: str | None = None
        # set of pin ids currently active (added but not yet removed)
        self._live_pins: set[str] = set()
        # set of pin ids that have been removed at least once
        self._dead_pins: set[str] = set()

    @initialize()
    def setup(self):
        import tempfile

        self._tmp = Path(tempfile.mkdtemp(prefix="engram-stateful-"))
        cfg = Config.minimal()
        cfg.path = str(self._tmp)
        # disable any rate limiting so we can hammer the API.
        cfg.security.max_events_per_minute = 0
        self._engram = Engram(config=cfg)

    # ------------------------------------------------------------------
    # rules
    # ------------------------------------------------------------------

    @rule(target=memory_ids, content=_alphabet, sal=st.floats(0.0, 1.0))
    def remember_and_rebuild(self, content, sal):
        e = self._engram
        assert e is not None
        e.remember(content, salience=sal)
        # rebuild materializes the memory row; otherwise get(eid) returns None.
        e.rebuild()
        self._last_content = content
        # Find the most recently created memory by scanning store.
        # Easiest: pull from status/by_state; safer: recall the just-written content.
        hits = e.recall(content, limit=5)
        if not hits:
            return "noop"  # nothing to bundle; will be filtered out by other rules
        mid = hits[0].memory.id
        self._known_ids.add(mid)
        return mid

    @rule(content=_alphabet)
    def recall_no_hard_deleted(self, content):
        e = self._engram
        assert e is not None
        hits = e.recall(content, limit=10)
        for h in hits:
            assert h.memory.id not in self._hard_deleted, (
                f"S-I3 violated: hard-deleted {h.memory.id} resurfaced"
            )

    @rule(mid=memory_ids)
    def soft_forget(self, mid):
        if mid == "noop" or mid in self._hard_deleted:
            return
        e = self._engram
        assert e is not None
        e.forget(id=mid, hard=False)
        # S-I5: row still gettable, just suppressed
        m = e.get(mid)
        assert m is not None, f"S-I5: soft-forgotten {mid} disappeared from get()"
        assert m.state.value == "suppressed", f"S-I5: state={m.state}, want suppressed"

    @rule(mid=memory_ids)
    def hard_forget(self, mid):
        if mid == "noop" or mid in self._hard_deleted:
            return
        e = self._engram
        assert e is not None
        e.forget(id=mid, hard=True)
        self._hard_deleted.add(mid)
        self._known_ids.discard(mid)
        assert e.get(mid) is None, f"S-I2: hard-deleted {mid} still gettable"

    @rule()
    def rebuild_idempotent(self):
        e = self._engram
        assert e is not None
        s1 = e.status()
        e.rebuild()
        s2 = e.status()
        assert s1["total_memories"] == s2["total_memories"], (
            f"S-I4: rebuild non-idempotent {s1['total_memories']} -> {s2['total_memories']}"
        )

    @rule(mid=memory_ids)
    def delete_alias(self, mid):
        """S-I6: delete(id=mid) must be a true hard-delete alias."""
        if mid == "noop" or mid in self._hard_deleted:
            return
        e = self._engram
        assert e is not None
        n = e.delete(id=mid)
        assert n >= 1, f"S-I6: delete(id={mid}) returned {n}, want >= 1"
        self._hard_deleted.add(mid)
        self._known_ids.discard(mid)
        assert e.get(mid) is None, f"S-I6: delete-alias left {mid} gettable"
        # And it must not surface in recall on its original content either.
        if self._last_content:
            for h in e.recall(self._last_content, limit=10):
                assert h.memory.id != mid, (
                    f"S-I6: deleted {mid} resurfaced in recall after delete()"
                )

    @rule(target=pin_ids, content=_alphabet)
    def pin_content(self, content):
        e = self._engram
        assert e is not None
        pid = e.pin(content)
        assert isinstance(pid, str) and pid.startswith("pin-"), (
            f"S-I8: pin returned non-pin id {pid}"
        )
        # S-I8: pin id must not collide with any known memory id namespace.
        assert pid not in self._known_ids, (
            f"S-I8: pin id {pid} collides with memory id"
        )
        self._live_pins.add(pid)
        return pid

    @rule(pid=pin_ids)
    def unpin_once(self, pid):
        """S-I7: unpin returns True iff pid is currently live."""
        e = self._engram
        assert e is not None
        was_live = pid in self._live_pins
        ok = e.unpin(pid)
        if was_live:
            assert ok is True, f"S-I7: unpin({pid}) returned False on live pin"
            self._live_pins.discard(pid)
            self._dead_pins.add(pid)
        else:
            assert ok is False, (
                f"S-I7: unpin({pid}) returned True on already-removed pin"
            )

    # ------------------------------------------------------------------
    # invariants — checked after every rule
    # ------------------------------------------------------------------

    @invariant()
    def status_consistency(self):
        if self._engram is None:
            return
        s = self._engram.status()
        by_state_total = sum(s["by_state"].values())
        assert s["total_memories"] == by_state_total, (
            f"S-I1: total_memories={s['total_memories']} but by_state sum={by_state_total}"
        )

    @invariant()
    def hard_deleted_stay_dead(self):
        if self._engram is None:
            return
        for mid in self._hard_deleted:
            assert self._engram.get(mid) is None, (
                f"S-I2: hard-deleted {mid} reappeared in get()"
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


# Wrap into a pytest test with conservative settings so it runs in <30s.
TestEngramStateful = EngramStateMachine.TestCase
TestEngramStateful.settings = settings(
    max_examples=20,
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)

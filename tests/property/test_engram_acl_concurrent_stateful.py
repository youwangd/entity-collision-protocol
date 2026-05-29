"""Concurrent ACL stateful machine — readers and writers interleave on a
shared agent pool while grants are reloaded mid-flight.

This is the layer between:

  * `tests/property/test_engram_acl_stateful.py` — sequential ACL ops,
    one rule at a time on a single Engram.
  * `tests/concurrency/test_race_conditions.py` — explicit-schedule
    multi-thread torture without ACL coverage.

Hypothesis drives the *macro* schedule (which kind of burst happens
next: grant, revoke, mixed-burst) and a ThreadPoolExecutor explores
the *micro* interleavings inside each burst. The invariant we are
hunting for is a stale-grant leak: a permission decision that races
with `acl.grant()` / `acl.revoke()` and produces an outcome that
disagrees with both the pre- and post-state of the policy.

Concretely, each `mixed_burst` rule:

  1. Snapshots the policy ground-truth `g_before`.
  2. Fans K worker threads. Each worker picks a random agent from the
     current pool and a random op (`remember` / `recall`), executes
     it, and records `(agent, op, ok_or_PermissionError)`.
  3. Waits for the burst to drain.
  4. Snapshots `g_after` (which equals `g_before` for `mixed_burst` —
     no grants/revokes inside the burst).
  5. For every recorded outcome, asserts the outcome is *consistent
     with* `g_before == g_after` policy. There is no policy mutation
     during the burst, so any disagreement is a real leak.

The `reload_burst` rule is the interesting one: it interleaves
`grant`/`revoke` calls with concurrent reader/writer ops on the same
agent pool. Here the invariant is weaker — a worker's outcome must be
consistent with *some* policy state observed between burst-start and
burst-end, monotone in the obvious sense (grants only go on/off once
each per burst). We track the burst's grant log and check the
outcome against the union of (pre-policy, post-policy) — a stale-grant
leak would produce an outcome consistent with *neither*.

C-A-I1  no worker raised an unexpected exception (only PermissionError
        and the well-known empty-content ValueError are filtered).
C-A-I2  in mixed_burst, every outcome matches g_before exactly.
C-A-I3  in reload_burst, every outcome is consistent with at least one
        of {g_before, g_after}.
C-A-I4  ghost-agent denial holds at every quiescent invariant point.
"""
from __future__ import annotations

import random
import string
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
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

_agent_id = st.text(
    alphabet=string.ascii_lowercase + string.digits, min_size=4, max_size=8
).filter(lambda s: s.strip() and not s.startswith("_") and s != "system")


def _has(perms_scope, perm: str) -> bool:
    return bool(perms_scope and perm in perms_scope[0])


class ConcurrentACLEngramMachine(RuleBasedStateMachine):
    agent_ids = Bundle("agent_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # agent_id -> (frozenset(perms), scope) ground truth.
        self._grants: dict[str, tuple[frozenset[str], str]] = {}

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-acl-conc-"))
        cfg = Config.minimal()
        cfg.path = str(self._tmp)
        cfg.security.max_events_per_minute = 0
        cfg.acl = {
            "enabled": True,
            "grants": {
                "system": {
                    "permissions": [
                        "read", "write", "forget", "consolidate", "admin", "export"
                    ],
                    "scope": "*",
                },
            },
        }
        self._engram = Engram(config=cfg, actor="system")

    # ------------------------------------------------------------------
    # rules — agent pool management
    # ------------------------------------------------------------------

    @rule(target=agent_ids, agent=_agent_id, scope=st.sampled_from(["own", "*"]),
          perms=st.sampled_from([("read",), ("write",), ("read", "write"), ("read", "write", "forget")]))
    def grant(self, agent, scope, perms):
        e = self._engram
        assert e is not None
        e.acl.grant(agent, set(perms), scope=scope)
        self._grants[agent] = (frozenset(perms), scope)
        return agent

    @rule(agent=agent_ids)
    def revoke(self, agent):
        if agent == "skip":
            return
        e = self._engram
        assert e is not None
        e.acl.revoke(agent)
        self._grants.pop(agent, None)

    # ------------------------------------------------------------------
    # rule — burst with no policy changes
    # ------------------------------------------------------------------

    @rule(seed=st.integers(0, 2**31 - 1),
          k=st.integers(min_value=4, max_value=12))
    def mixed_burst(self, seed, k):
        """Quiescent burst: K threads do random read/write while policy
        is frozen. Every outcome must match the snapshotted policy."""
        e = self._engram
        assert e is not None
        if not self._grants:
            return
        _ = random.Random(seed)
        agents = list(self._grants.keys())
        # Snapshot policy. No policy mutation happens during this burst,
        # so g_before == g_after.
        g_snap = dict(self._grants)

        def worker(i: int):
            r = random.Random(seed + i * 7919)
            agent = r.choice(agents + ["ghost-never-granted"])
            op = r.choice(["read", "write"])
            content = "".join(r.choices(string.ascii_lowercase + " ", k=20)).strip() or "x"
            try:
                if op == "write":
                    e.remember(content, agent_id=agent)
                else:
                    e.recall(content, limit=3, agent_id=agent)
                ok = True
            except PermissionError as ex:
                ok = False
                _ = ex
            except ValueError as ex:
                # known: empty content. Very unlikely with our generator.
                return ("skip", agent, op, str(ex))
            except Exception as ex:  # noqa: BLE001
                return ("error", agent, op, repr(ex))
            return ("ok" if ok else "denied", agent, op, None)

        with ThreadPoolExecutor(max_workers=k) as ex_:
            results = list(ex_.map(worker, range(k * 2)))

        for status, agent, op, detail in results:
            assert status != "error", f"C-A-I1 worker raised: {detail}"
            if status == "skip":
                continue
            truth = g_snap.get(agent)
            needed = "write" if op == "write" else "read"
            should_pass = _has(truth, needed)
            if should_pass:
                assert status == "ok", (
                    f"C-A-I2: agent={agent} op={op} should pass per snapshot "
                    f"{truth} but was denied")
            else:
                assert status == "denied", (
                    f"C-A-I2: agent={agent} op={op} should be denied per snapshot "
                    f"{truth} but succeeded")

    # ------------------------------------------------------------------
    # rule — burst that interleaves grant/revoke with read/write
    # ------------------------------------------------------------------

    @rule(seed=st.integers(0, 2**31 - 1),
          k=st.integers(min_value=4, max_value=10))
    def reload_burst(self, seed, k):
        """K reader/writer threads run concurrently with grant/revoke
        on the same agent pool. Every outcome must be consistent with
        AT LEAST ONE of (g_before, g_after)."""
        e = self._engram
        assert e is not None
        if len(self._grants) < 2:
            return
        rng = random.Random(seed)
        pool = list(self._grants.keys())
        # Pick one agent to flip mid-burst.
        flip_agent = rng.choice(pool)
        _ = self._grants.get(flip_agent)
        # Decide flip direction at random.
        flip_op = rng.choice(["revoke", "regrant_read", "regrant_write"])

        g_before = dict(self._grants)

        def policy_mutator():
            # Small jitter so the mutation lands inside the burst, not
            # before / after.
            import time
            time.sleep(0.001)
            if flip_op == "revoke":
                e.acl.revoke(flip_agent)
            elif flip_op == "regrant_read":
                e.acl.grant(flip_agent, {"read"}, scope="own")
            else:
                e.acl.grant(flip_agent, {"write"}, scope="own")

        def worker(i: int):
            r = random.Random(seed + i * 7919 + 31)
            agent = r.choice(pool)
            op = r.choice(["read", "write"])
            content = "".join(r.choices(string.ascii_lowercase + " ", k=18)).strip() or "x"
            try:
                if op == "write":
                    e.remember(content, agent_id=agent)
                else:
                    e.recall(content, limit=3, agent_id=agent)
                return ("ok", agent, op, None)
            except PermissionError:
                return ("denied", agent, op, None)
            except ValueError as ex:
                return ("skip", agent, op, str(ex))
            except Exception as ex:  # noqa: BLE001
                return ("error", agent, op, repr(ex))

        with ThreadPoolExecutor(max_workers=k + 1) as ex_:
            mutator_fut = ex_.submit(policy_mutator)
            futs = [ex_.submit(worker, i) for i in range(k * 2)]
            mutator_fut.result()
            results = [f.result() for f in futs]

        # Apply the mutation to ground truth (post-burst snapshot).
        if flip_op == "revoke":
            self._grants.pop(flip_agent, None)
        elif flip_op == "regrant_read":
            self._grants[flip_agent] = (frozenset({"read"}), "own")
        else:
            self._grants[flip_agent] = (frozenset({"write"}), "own")

        g_after = dict(self._grants)

        def consistent(agent, op, status):
            needed = "write" if op == "write" else "read"
            allow_before = _has(g_before.get(agent), needed)
            allow_after = _has(g_after.get(agent), needed)
            if status == "ok":
                # Must have been allowed under at least one snapshot.
                return allow_before or allow_after
            else:
                # Must have been denied under at least one snapshot.
                return (not allow_before) or (not allow_after)

        for status, agent, op, detail in results:
            assert status != "error", f"C-A-I1 worker raised: {detail}"
            if status == "skip":
                continue
            assert consistent(agent, op, status), (
                f"C-A-I3 stale-grant LEAK: agent={agent} op={op} "
                f"status={status} but g_before={g_before.get(agent)!r} "
                f"g_after={g_after.get(agent)!r} (flip_agent={flip_agent}, "
                f"flip_op={flip_op})")

    # ------------------------------------------------------------------
    # invariants (quiescent)
    # ------------------------------------------------------------------

    @invariant()
    def ghost_agent_always_denied(self):
        """C-A-I4: a never-granted agent_id is always denied."""
        if self._engram is None:
            return
        ghost = "ghost-never-granted"
        assert ghost not in self._grants
        with pytest.raises(PermissionError):
            self._engram.remember("ghost-write", agent_id=ghost)
        with pytest.raises(PermissionError):
            self._engram.recall("ghost", agent_id=ghost)

    def teardown(self):
        if self._engram is not None:
            try:
                self._engram.close()
            except Exception:
                pass
        if self._tmp is not None:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)


TestConcurrentACLEngramStateful = ConcurrentACLEngramMachine.TestCase
TestConcurrentACLEngramStateful.settings = settings(
    max_examples=15,
    stateful_step_count=14,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)

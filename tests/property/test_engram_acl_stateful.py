"""Stateful machine over the Engram ACL surface (Engram + AccessPolicy).

Hypothesis explores random interleavings of grant / revoke / remember /
recall keyed by random agent_ids on a SINGLE Engram with ACL enabled.
The system invariant is: at every step, an agent's API access matches
the policy's ground-truth grant table — no leaks from stale grants,
no false denials, no scope upgrades.

Invariants (checked after every rule):

  A-I1  Disabled-policy bypass: not exercised here (policy is enabled
        for the whole machine; bypass is covered by ACL invariants
        in test_acl_invariants.py I1).
  A-I2  An agent with NO grant can neither remember nor recall — both
        must raise PermissionError. This is the security-critical leak
        condition; we run it after every rule against a fresh "ghost"
        agent id Hypothesis hasn't granted.
  A-I3  An agent with READ but not WRITE may recall but cannot remember.
  A-I4  An agent with WRITE but not READ may remember but cannot recall.
  A-I5  After revoke(agent_id), neither remember nor recall succeeds for
        that agent_id — i.e. revoke is an absolute denial, no stale
        grant survives.
  A-I6  scope='own' agent never sees memories owned by another agent
        (results filtered to its own owner_id) unless it also holds
        FEDERATED.
"""
from __future__ import annotations

import string
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
    alphabet=string.ascii_lowercase + string.digits, min_size=4, max_size=10
).filter(lambda s: s.strip() and not s.startswith("_"))


class ACLEngramMachine(RuleBasedStateMachine):
    agent_ids = Bundle("agent_ids")  # ids that *currently* have a grant

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # Ground-truth: agent_id -> (perms_set, scope)
        self._grants: dict[str, tuple[frozenset[str], str]] = {}
        # Anything ever revoked (for A-I5 follow-up checks)
        self._ever_revoked: set[str] = set()

    @initialize()
    def setup(self):
        import tempfile

        self._tmp = Path(tempfile.mkdtemp(prefix="engram-acl-"))
        cfg = Config.minimal()
        cfg.path = str(self._tmp)
        cfg.security.max_events_per_minute = 0
        # Seed an admin so the machine itself can drive setup ops.
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
    # rules
    # ------------------------------------------------------------------

    @rule(target=agent_ids, agent=_agent_id, scope=st.sampled_from(["own", "*"]))
    def grant_read_write(self, agent, scope):
        """Grant the basic read+write+forget bundle to an agent."""
        e = self._engram
        assert e is not None
        if agent in ("system",):
            return "skip"
        perms = {"read", "write", "forget"}
        e.acl.grant(agent, perms, scope=scope)
        self._grants[agent] = (frozenset(perms), scope)
        return agent

    @rule(target=agent_ids, agent=_agent_id, scope=st.sampled_from(["own", "*"]))
    def grant_read_only(self, agent, scope):
        e = self._engram
        assert e is not None
        if agent in ("system",):
            return "skip"
        e.acl.grant(agent, {"read"}, scope=scope)
        self._grants[agent] = (frozenset({"read"}), scope)
        return agent

    @rule(target=agent_ids, agent=_agent_id, scope=st.sampled_from(["own", "*"]))
    def grant_write_only(self, agent, scope):
        e = self._engram
        assert e is not None
        if agent in ("system",):
            return "skip"
        e.acl.grant(agent, {"write"}, scope=scope)
        self._grants[agent] = (frozenset({"write"}), scope)
        return agent

    @rule(agent=agent_ids)
    def revoke(self, agent):
        if agent == "skip" or agent == "system":
            return
        e = self._engram
        assert e is not None
        e.acl.revoke(agent)
        self._grants.pop(agent, None)
        self._ever_revoked.add(agent)

    @rule(agent=agent_ids, content=_alphabet, sal=st.floats(0.0, 1.0))
    def try_remember(self, agent, content, sal):
        if agent == "skip":
            return
        e = self._engram
        assert e is not None
        truth = self._grants.get(agent)
        has_write = bool(truth and "write" in truth[0])
        try:
            e.remember(content, salience=sal, agent_id=agent)
            ok = True
        except PermissionError:
            ok = False
        if has_write:
            assert ok, f"A-I3/4: agent={agent} has write but remember raised"
        else:
            assert not ok, f"A-I2/5: agent={agent} lacks write but remember succeeded"

    @rule(agent=agent_ids, content=_alphabet)
    def try_recall(self, agent, content):
        if agent == "skip":
            return
        e = self._engram
        assert e is not None
        truth = self._grants.get(agent)
        has_read = bool(truth and "read" in truth[0])
        try:
            results = e.recall(content, limit=5, agent_id=agent)
            ok = True
        except PermissionError:
            results = []
            ok = False
        if has_read:
            assert ok, f"A-I3: agent={agent} has read but recall raised"
            # A-I6: scope='own' without FEDERATED — every result must be
            # owned by this agent (or by anonymous "" / system seed).
            if truth and truth[1] == "own" and "federated" not in truth[0]:
                for r in results:
                    owner = r.memory.agent_id or ""
                    assert owner in (agent, ""), (
                        f"A-I6: scope=own agent {agent} saw memory owned by {owner!r}"
                    )
        else:
            assert not ok, f"A-I2/5: agent={agent} lacks read but recall succeeded"

    # ------------------------------------------------------------------
    # invariants (ghost-agent + revoked-agent leak checks)
    # ------------------------------------------------------------------

    @invariant()
    def ghost_agent_always_denied(self):
        """A-I2: a never-granted agent_id must always be denied."""
        if self._engram is None:
            return
        ghost = "ghost-never-granted"
        assert ghost not in self._grants
        with pytest.raises(PermissionError):
            self._engram.remember("ghost write", agent_id=ghost)
        with pytest.raises(PermissionError):
            self._engram.recall("ghost", agent_id=ghost)

    @invariant()
    def revoked_agents_stay_denied(self):
        """A-I5: anything ever revoked stays denied unless re-granted."""
        if self._engram is None:
            return
        for agent in self._ever_revoked:
            if agent in self._grants:
                continue  # was re-granted; covered by per-rule asserts
            with pytest.raises(PermissionError):
                self._engram.remember("post-revoke write", agent_id=agent)
            with pytest.raises(PermissionError):
                self._engram.recall("post-revoke", agent_id=agent)

    def teardown(self):
        if self._engram is not None:
            try:
                self._engram.close()
            except Exception:
                pass
        if self._tmp is not None:
            import shutil

            shutil.rmtree(self._tmp, ignore_errors=True)


TestACLEngramStateful = ACLEngramMachine.TestCase
TestACLEngramStateful.settings = settings(
    max_examples=20,
    stateful_step_count=22,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)

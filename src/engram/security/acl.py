"""Access control — per-agent memory isolation (Design §5.3).

Grants are permission sets scoped to either 'own' (agent's own memories)
or '*' (all memories). Each Engram instance can optionally enforce ACL
via an AccessPolicy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Operations that can be granted."""

    READ = "read"
    WRITE = "write"
    FORGET = "forget"
    CONSOLIDATE = "consolidate"
    ADMIN = "admin"
    EXPORT = "export"
    FEDERATED = "federated"  # cross-agent queries


@dataclass
class Grant:
    """A permission grant for an agent/user."""

    agent_id: str
    permissions: set[Permission]
    scope: str = "own"  # "own" = only agent's memories; "*" = all

    def has(self, perm: Permission) -> bool:
        return perm in self.permissions or Permission.ADMIN in self.permissions

    def can_access(self, memory_agent_id: str) -> bool:
        """Check if this grant allows access to a memory owned by memory_agent_id."""
        if self.scope == "*":
            return True
        if self.scope == "own":
            return memory_agent_id == self.agent_id or memory_agent_id == ""
        return False


class AccessPolicy:
    """Manages per-agent access grants.

    Usage:
        policy = AccessPolicy()
        policy.grant("agent-evan", {"read", "write", "forget"}, scope="own")
        policy.grant("agent-reviewer", {"read"}, scope="*")
        policy.grant("user-richard", {"read", "write", "forget", "consolidate", "admin", "export"}, scope="*")

        # Check:
        policy.check("agent-evan", Permission.READ, memory_agent_id="agent-evan")  # OK
        policy.check("agent-evan", Permission.READ, memory_agent_id="agent-other")  # raises
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._grants: dict[str, Grant] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def grant(self, agent_id: str, permissions: set[str | Permission], scope: str = "own") -> None:
        """Grant permissions to an agent/user."""
        perms = set()
        for p in permissions:
            if isinstance(p, str):
                perms.add(Permission(p))
            else:
                perms.add(p)
        self._grants[agent_id] = Grant(agent_id=agent_id, permissions=perms, scope=scope)
        logger.info("granted %s to %s (scope=%s)", perms, agent_id, scope)

    def revoke(self, agent_id: str) -> None:
        """Revoke all permissions for an agent/user."""
        self._grants.pop(agent_id, None)
        logger.info("revoked all grants for %s", agent_id)

    def check(self, agent_id: str, permission: Permission, memory_agent_id: str = "") -> None:
        """Check if agent has permission. Raises PermissionError if denied.

        If ACL is disabled, always passes.
        If agent has no grant at all, denied.
        """
        if not self._enabled:
            return

        grant = self._grants.get(agent_id)
        if grant is None:
            raise PermissionError(f"agent '{agent_id}' has no access grants")

        if not grant.has(permission):
            raise PermissionError(
                f"agent '{agent_id}' lacks '{permission.value}' permission"
            )

        if permission == Permission.READ and not grant.can_access(memory_agent_id):
            if Permission.FEDERATED not in grant.permissions:
                raise PermissionError(
                    f"agent '{agent_id}' cannot access memories owned by '{memory_agent_id}' "
                    f"(scope='{grant.scope}', missing 'federated' permission)"
                )

    def check_write(self, agent_id: str) -> None:
        """Shortcut: check write permission."""
        self.check(agent_id, Permission.WRITE)

    def check_read(self, agent_id: str, memory_agent_id: str = "") -> None:
        """Shortcut: check read permission with scope."""
        self.check(agent_id, Permission.READ, memory_agent_id=memory_agent_id)

    def list_grants(self) -> list[dict]:
        """List all grants."""
        return [
            {
                "agent_id": g.agent_id,
                "permissions": sorted(p.value for p in g.permissions),
                "scope": g.scope,
            }
            for g in self._grants.values()
        ]

    def to_dict(self) -> dict:
        """Serialize for config/export."""
        return {
            "enabled": self._enabled,
            "grants": {
                g.agent_id: {
                    "permissions": sorted(p.value for p in g.permissions),
                    "scope": g.scope,
                }
                for g in self._grants.values()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> AccessPolicy:
        """Deserialize from config."""
        policy = cls(enabled=data.get("enabled", False))
        for agent_id, grant_data in data.get("grants", {}).items():
            policy.grant(
                agent_id,
                set(grant_data.get("permissions", [])),
                scope=grant_data.get("scope", "own"),
            )
        return policy

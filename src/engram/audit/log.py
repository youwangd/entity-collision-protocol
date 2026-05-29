"""Audit log — append-only JSONL for operation tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only audit log. Separate from the event store.

    Tracks operations and access, not memory content.
    Content is hashed (not stored) for PII safety.
    """

    def __init__(self, base_path: Path):
        self.path = base_path / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(
        self,
        operation: str,
        actor: str,
        details: dict,
        outcome: str = "success",
        duration_ms: int = 0,
    ) -> None:
        """Log an audit entry."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": operation,
            "actor": actor,
            "details": details,
            "outcome": outcome,
            "ms": duration_ms,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as e:
            logger.error("failed to write audit log: %s", e)

    def read(self, limit: int = 100, operation: str | None = None) -> list[dict]:
        """Read recent audit entries.

        Resilient to torn lines and non-utf8 byte runs — opens binary and
        decodes per-line with `errors="replace"`, mirroring the buffer's
        post-f218b47 scan path. A single bad byte cannot abort the whole
        scan (which is what would happen under strict utf-8 file mode and
        was the failure mode buffer.py had pre-f218b47).
        """
        entries: list[dict] = []
        if not self.path.exists():
            return entries
        with open(self.path, "rb") as f:
            for raw in f:
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    # Damaged byte run — skip just this line, keep scanning.
                    continue
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if operation and entry.get("op") != operation:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]

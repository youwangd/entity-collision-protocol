"""JSONL event buffer — the source of truth (event store)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from engram.core.types import Event, EventType
from engram.core.errors import BufferError

try:  # POSIX advisory locking; fall back to no-op on Windows.
    import fcntl  # type: ignore[import-not-found]
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

if TYPE_CHECKING:
    from engram.security.encryption import ContentEncryptor

logger = logging.getLogger(__name__)


class JSONLBufferStore:
    """Append-only JSONL event store. Source of truth for all operations.

    Every operation in Engram produces an event that is appended here.
    SQLite is a projection (read model) that can be rebuilt from this log.

    Concurrency model
    -----------------
    A sidecar lockfile (``events.jsonl.lock``) coordinates between
    appenders and rewriters via POSIX advisory ``fcntl.flock``:

      * ``append`` takes a SHARED lock — many appenders can run in
        parallel; O_APPEND keeps individual writes atomic up to
        ``PIPE_BUF`` (4096 on Linux/tmpfs).
      * ``truncate_before`` / ``clear`` / ``redact_memory`` take an
        EXCLUSIVE lock — they wait for in-flight appenders to drain,
        then perform the read-rewrite in isolation.

    This closes the previously-documented race where an append landing
    inside a truncate's read-then-rewrite window was clobbered.
    """

    def __init__(self, base_path: Path, encryptor: ContentEncryptor | None = None):
        self.path = base_path / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._encryptor = encryptor
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        # Touch the file if it doesn't exist
        if not self.path.exists():
            self.path.touch()
        if not self._lock_path.exists():
            self._lock_path.touch()
        # One-shot tail repair on open. If the file ends mid-line (external
        # truncation, killed mid-write, etc.), append a newline under an
        # exclusive lock so subsequent O_APPEND writes can't fuse onto a
        # broken tail. Doing this once at init avoids a per-append seek/read
        # that races other appenders (each appender's `seek(-1, 2); read(1)`
        # could land in the *middle* of another process's atomic O_APPEND
        # line, see torn-line discussion in test_jsonl_buffer_concurrency).
        self._repair_tail_if_needed()

    def _repair_tail_if_needed(self) -> None:
        """If the file is non-empty and does not end in '\\n', append one.

        Held under exclusive flock so no concurrent appender can race the
        seek/read. Idempotent: a clean tail is a no-op.
        """
        try:
            with self._flock(exclusive=True):
                with open(self.path, "a+b") as f:
                    f.seek(0, 2)
                    if f.tell() == 0:
                        return
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
        except OSError:
            # Best-effort; surface real problems on the next append.
            pass

    @contextlib.contextmanager
    def _flock(self, exclusive: bool):
        """Acquire a shared (read) or exclusive (write) advisory lock.

        No-op on platforms without ``fcntl`` (Windows). Always closes the
        fd on exit, releasing the lock.
        """
        if not _HAS_FCNTL:
            yield
            return
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def append(self, event: Event) -> str:
        """Append an event. Returns event ID.

        Each line is a complete JSON object — atomic on most filesystems
        for reasonable line lengths. Partial writes are detectable.
        """
        try:
            data = event.to_dict()
            # Encrypt content at rest in JSONL (Design §5.5)
            # SQLite projection stores plaintext for FTS; JSONL is the on-disk source of truth.
            if self._encryptor and self._encryptor.enabled:
                data["content"] = self._encryptor.encrypt(data["content"])
            line = json.dumps(data, separators=(",", ":")) + "\n"
            with self._flock(exclusive=False):
                # POSIX O_APPEND guarantees atomic writes up to PIPE_BUF
                # (4096 on Linux). Each event line stays under that cap by
                # construction (pinned by test_event_serialized_size_under_pipe_buf).
                # Tail-repair runs once at __init__; we do NOT seek/read here
                # because that would race concurrent appenders (their atomic
                # writes could land between our seek and read, returning a
                # mid-line byte and triggering a spurious newline insertion).
                with open(self.path, "ab") as f:
                    f.write(line.encode("utf-8"))
            logger.debug("event appended: %s type=%s", event.id, event.type.value)
            return event.id
        except (OSError, IOError) as e:
            # A torn write — short-write that raises mid-line — leaves the
            # file ending without a newline. The next O_APPEND from any
            # caller would then fuse onto that broken tail, corrupting both
            # frames. Repair the tail under exclusive lock before surfacing
            # the error so subsequent appends land on a clean boundary.
            try:
                self._repair_tail_if_needed()
            except Exception:  # pragma: no cover - best effort
                pass
            raise BufferError(f"Failed to append event {event.id}: {e}") from e

    def scan(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        event_type: EventType | None = None,
        limit: int | None = None,
    ) -> Iterator[Event]:
        """Scan events with optional filters.

        Yields events in chronological order. Skips corrupted lines.
        """
        count = 0
        try:
            # Open in binary mode + decode per-line with errors='replace'.
            # Strict-utf8 file mode aborts the entire scan on a single bad byte
            # (e.g. mid-write torn frame); per-line decode isolates the damage
            # to the offending line only. Replacement chars then trip the
            # JSONDecodeError path below and the line is skipped+logged.
            with open(self.path, "rb") as f:
                for line_num, raw in enumerate(f, 1):
                    try:
                        line = raw.decode("utf-8").strip()
                    except UnicodeDecodeError as e:
                        logger.warning("skipping non-utf8 line %d: %s", line_num, e)
                        continue
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Decrypt content from JSONL (Design §5.5)
                        if self._encryptor and self._encryptor.enabled and "content" in data:
                            data["content"] = self._encryptor.decrypt(data["content"])
                        event = Event.from_dict(data)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning("skipping corrupted event at line %d: %s", line_num, e)
                        continue
                    if since and event.ts < since:
                        continue
                    if until and event.ts > until:
                        continue
                    if event_type and event.type != event_type:
                        continue
                    yield event
                    count += 1
                    if limit and count >= limit:
                        return
        except FileNotFoundError:
            return

    def count(self) -> int:
        """Count total events in the log."""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path, "rb") as f:
            for raw in f:
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    count += 1
        return count

    def last_event_id(self) -> str | None:
        """Get the ID of the most recent event."""
        if not self.path.exists():
            return None
        last_line = None
        # Binary mode + per-line decode: a non-utf8 byte deeper in the file
        # must not abort the lookup of the most recent valid line.
        with open(self.path, "rb") as f:
            for raw in f:
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    last_line = line
        if last_line is None:
            return None
        try:
            return json.loads(last_line)["id"]
        except (json.JSONDecodeError, KeyError):
            return None

    def truncate_before(self, before: datetime) -> int:
        """Remove events older than `before`. Returns count removed.

        Atomic w.r.t. concurrent ``append`` — holds an exclusive flock
        across the read-then-rewrite, then publishes via ``os.replace``
        so any reader sees either the pre- or post-truncate file but
        never a torn intermediate.
        """
        removed = 0
        with self._flock(exclusive=True):
            # Binary read + per-line decode mirrors scan(): a single non-utf8
            # byte (torn frame, FS corruption) must not abort the whole
            # rewrite — that would either propagate the error (worst case
            # the caller never recovers) or, if the tmp swap had partially
            # progressed, risk silent data loss. Conservative: keep raw
            # bytes for any line we can't parse.
            kept_chunks: list[bytes] = []
            with open(self.path, "rb") as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        line = raw.decode("utf-8").strip()
                        data = json.loads(line)
                        ts = datetime.fromisoformat(data["ts"])
                        if ts < before:
                            removed += 1
                            continue
                        kept_chunks.append(line.encode("utf-8") + b"\n")
                    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
                        # keep corrupted lines verbatim (conservative)
                        kept_chunks.append(raw if raw.endswith(b"\n") else raw + b"\n")

            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(b"".join(kept_chunks))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)

        logger.info("truncated %d events before %s", removed, before.isoformat())
        return removed

    def clear(self) -> None:
        """Clear all events (used in rebuild)."""
        with self._flock(exclusive=True):
            with open(self.path, "w", encoding="utf-8") as f:
                f.truncate(0)

    def redact_memory(self, memory_id: str) -> int:
        """Redact content from events that created a specific memory (GDPR Art. 17).

        Replaces content with [DELETED] but preserves metadata for audit trail.
        Returns count of events redacted. Atomic w.r.t. concurrent appends
        via the same exclusive-flock + atomic-rename strategy as
        ``truncate_before``.
        """
        redacted = 0
        with self._flock(exclusive=True):
            # Same binary-read + per-line-decode strategy as truncate_before
            # and scan(): a torn frame must not abort the rewrite.
            chunks: list[bytes] = []
            with open(self.path, "rb") as f:
                for raw in f:
                    if not raw.strip():
                        chunks.append(raw)
                        continue
                    try:
                        line = raw.decode("utf-8").strip()
                        data = json.loads(line)
                        meta = data.get("metadata", {})
                        if (data.get("type") == "explicit_remember" and
                                memory_id in str(meta.get("memory_id", ""))):
                            data["content"] = "[DELETED]"
                            redacted += 1
                        chunks.append(json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n")
                    except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
                        chunks.append(raw if raw.endswith(b"\n") else raw + b"\n")

            if redacted > 0:
                tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    f.write(b"".join(chunks))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.path)
                logger.info("redacted %d events for memory %s", redacted, memory_id)
        return redacted

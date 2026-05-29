"""Read-side projection: replay schema-lifecycle events from the buffer.

The pure reducer in `schema_lifecycle.py` is the source of truth for what
a schema's status is. This module is the **adapter** between the JSONL
event log (`engram.store.buffer.JSONLBufferStore`) and that reducer.

Design:
- Lifecycle decisions are emitted into the buffer as
  `EventType.CONSOLIDATION_SCHEMA_LIFECYCLE` events. The decision is
  encoded entirely in `Event.metadata` so we don't need a new DB column.
- `metadata` shape:
      {
        "schema_id": "<stable id>",
        "kind": "create" | "promote" | "deprecate" | "recover" | "bump_version",
        "window_id": "<evidence window id>",
        "emitter_id": "<consolidator identity>",  # optional; back-compat
      }
  `emitter_id` is the identity of the consolidator / agent that decided
  to emit this lifecycle event. It is **only** consumed by the reducer
  when running under `reduce_events(deprecate_quorum_k>1)` (the §6.16
  quorum gate). Pre-quorum logs lack the field; the decoder treats
  missing/null as `emitter_id=None` for full back-compat.
- Reading is just `scan(event_type=CONSOLIDATION_SCHEMA_LIFECYCLE)`,
  decode, hand the list to `reduce_events`. The reducer is pure, so the
  projection is deterministic and snapshot-resumable.

This is intentionally minimal: write-side emission lives in the
pipeline (one helper used by `SchemaUpdate` once we move it to the new
contract). The projection is what tests, the audit CLI, and any future
debugger consume.
"""
from __future__ import annotations

from typing import Iterable

from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaLifecycleEvent,
    SchemaState,
    reduce_events,
)
from engram.core.types import Event, EventType
from engram.store.buffer import JSONLBufferStore


_VALID_KINDS = {k.value for k in EventKind}


def event_to_lifecycle(ev: Event) -> SchemaLifecycleEvent | None:
    """Decode a buffer Event into a SchemaLifecycleEvent, or None if malformed.

    Pure: no I/O, no clocks. Returns None for anything that doesn't look
    like a well-formed lifecycle event so the projection can keep going
    in lenient mode (we don't want one bad row to kill the snapshot).
    """
    if ev.type != EventType.CONSOLIDATION_SCHEMA_LIFECYCLE:
        return None
    meta = getattr(ev, "metadata", None) or {}
    schema_id = meta.get("schema_id")
    kind_str = meta.get("kind")
    if not isinstance(schema_id, str) or not schema_id:
        return None
    if kind_str not in _VALID_KINDS:
        return None
    window_id = meta.get("window_id")
    if window_id is not None and not isinstance(window_id, str):
        return None
    emitter_id = meta.get("emitter_id")
    if emitter_id is not None and not isinstance(emitter_id, str):
        return None
    # ts: events store datetime; reducer wants an int. Use unix seconds.
    try:
        ts_int = int(ev.ts.timestamp())
    except Exception:
        ts_int = 0
    return SchemaLifecycleEvent(
        schema_id=schema_id,
        kind=EventKind(kind_str),
        window_id=window_id,
        ts=ts_int,
        emitter_id=emitter_id,
    )


def iter_lifecycle_events(
    events: Iterable[Event],
) -> Iterable[SchemaLifecycleEvent]:
    """Filter+decode a stream of buffer events. Drops malformed rows."""
    for ev in events:
        decoded = event_to_lifecycle(ev)
        if decoded is not None:
            yield decoded


def snapshot_from_buffer(
    buffer: JSONLBufferStore, *, strict: bool = False
) -> dict[str, SchemaState]:
    """Replay the lifecycle event stream from a buffer into a snapshot.

    Lenient by default — production audit log can have legacy rows or
    rows from a future schema version we don't recognize, and the
    projection should degrade gracefully rather than crash.
    """
    decoded = list(
        iter_lifecycle_events(
            buffer.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
        )
    )
    return reduce_events(decoded, strict=strict)


class CachedLifecycleSnapshot:
    """Incremental-replay cache for the schema-lifecycle snapshot.

    Hot-path optimization for ``RetrievalEngine`` when
    ``respect_schema_lifecycle=True``: a naive implementation replays
    the entire ``CONSOLIDATION_SCHEMA_LIFECYCLE`` event stream on every
    ``recall()``. The ``bench/bench_lifecycle_gate_cost.py`` 5-point
    sweep showed a +6.5 ms p50 steady-state floor (empty buffer) and
    near-linear cost above 1k events.

    Cache strategy: the JSONL buffer is **append-only** between
    truncations, so we maintain ``(byte_offset, snapshot)`` and on each
    ``get()`` we read only bytes ``[byte_offset, EOF)``, decode any
    lifecycle events found there, fold them into the running snapshot,
    and advance the offset. Non-lifecycle appends (RECALL_REQUEST,
    RECALL_HIT, MEMORY_UPSERT, …) cost a single line-decode each;
    lifecycle appends additionally cost one reducer step.

    Truncation safety: if the file shrinks below our recorded offset,
    or the path's ``inode``/``dev`` change, the cache rebuilds from
    scratch by full ``scan()``. The ``(dev, inode)`` pair detects file
    rotation under us; ``size < offset`` detects in-place truncation.

    Thread-safety: each ``RetrievalEngine`` owns its own cache and
    ``recall()`` is the only caller, so no locks. Cross-process
    freshness is naturally enforced by inode/size checks.
    """

    # Bytes of file prefix used as a rotation tombstone. ``os.replace``
    # can recycle a freed inode number, so ``(dev, inode)`` plus
    # ``size >= offset`` is not sufficient to prove the cached
    # ``[0, offset)`` prefix is still intact. We capture up to this
    # many leading bytes at full-rebuild time and re-verify them on
    # every subsequent ``get()``. ``32`` is enough to cover the JSON
    # ``{"id": "...", "ts": ..., "type": "consolidation_schema_lifec``
    # opening of any lifecycle line and is essentially free to read.
    _PREFIX_PROBE_BYTES: int = 32

    __slots__ = (
        "_offset",
        "_inode_key",
        "_prefix",
        "_snap",
        "_quorum_k",
        "_hits",
        "_partial_hits",
        "_misses",
    )

    def __init__(self) -> None:
        self._offset: int = 0
        self._inode_key: tuple[int, int] | None = None
        # Up to ``_PREFIX_PROBE_BYTES`` bytes of the file head as
        # captured at the last full rebuild. Acts as a rotation
        # tombstone independent of inode/dev/mtime — the only way
        # this matches under a rotate is if the rotated file
        # *literally byte-equals* the original, which we accept as
        # observationally-equivalent.
        self._prefix: bytes = b""
        self._snap: dict[str, SchemaState] = {}
        # The reducer's `deprecate_quorum_k` argument is part of the
        # snapshot's identity — different k yields different snapshots
        # for the same event log. Cache must invalidate on k change.
        # ``None`` means "no get() yet": next call seeds it.
        self._quorum_k: int | None = None
        self._hits: int = 0  # offset == EOF, no work done
        self._partial_hits: int = 0  # advanced offset, folded ≥0 new events
        self._misses: int = 0  # full rebuild

    def _file_key(
        self, buffer: JSONLBufferStore
    ) -> tuple[tuple[int, int] | None, int]:
        try:
            st = buffer.path.stat()
        except OSError:
            return (None, 0)
        return ((st.st_dev, st.st_ino), st.st_size)

    def _read_prefix(self, buffer: JSONLBufferStore) -> bytes:
        try:
            with open(buffer.path, "rb") as f:
                return f.read(self._PREFIX_PROBE_BYTES)
        except OSError:
            return b""

    def _full_rebuild(
        self, buffer: JSONLBufferStore, *, strict: bool, deprecate_quorum_k: int = 1
    ) -> dict[str, SchemaState]:
        # Full rebuild reads the whole event log and re-runs the reducer
        # under the *current* k, so the cached snapshot is consistent
        # with whatever k the caller is now requesting.
        decoded = list(
            iter_lifecycle_events(
                buffer.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
            )
        )
        self._snap = reduce_events(
            decoded, strict=strict, deprecate_quorum_k=deprecate_quorum_k
        )
        try:
            st = buffer.path.stat()
            self._offset = st.st_size
            self._inode_key = (st.st_dev, st.st_ino)
        except OSError:
            self._offset = 0
            self._inode_key = None
        self._prefix = self._read_prefix(buffer)
        self._quorum_k = deprecate_quorum_k
        self._misses += 1
        return self._snap

    def get(
        self,
        buffer: JSONLBufferStore,
        *,
        strict: bool = False,
        deprecate_quorum_k: int = 1,
    ) -> dict[str, SchemaState]:
        if deprecate_quorum_k < 1:
            raise ValueError("deprecate_quorum_k must be >= 1")
        inode_key, size = self._file_key(buffer)
        if inode_key is None:
            # Stat failed (missing file etc.) — fall through to full
            # replay, which scan() handles by returning empty.
            return self._full_rebuild(
                buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
            )

        # Quorum-k change invalidates the snapshot identity: a stream
        # that produced DEPRECATED under k=1 may yield INFERRED under
        # k=2 (vote pending) and vice versa. Force a full rebuild.
        if self._quorum_k is not None and self._quorum_k != deprecate_quorum_k:
            return self._full_rebuild(
                buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
            )

        if self._inode_key != inode_key or size < self._offset:
            # First call, file rotated to a different inode, or
            # in-place truncation.
            return self._full_rebuild(
                buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
            )

        # Rotation-tombstone check: even if (dev, inode) match, an
        # ``os.replace`` could have recycled the inode number. Verify
        # the cached prefix still matches the file head. We only
        # apply this when the cache actually holds a non-empty prefix
        # (otherwise the cache is in its initial state and there is
        # nothing to compare against — the next branches do the right
        # thing).
        if self._prefix:
            probe_len = min(len(self._prefix), size)
            if probe_len > 0:
                current_prefix = self._read_prefix(buffer)[:probe_len]
                if current_prefix != self._prefix[:probe_len]:
                    return self._full_rebuild(
                        buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
                    )

        # Refresh the rotation tombstone if our cached prefix is shorter
        # than what's now observable. This closes a hole where the first
        # full rebuild captured an empty file (prefix == b""), then
        # incremental folds advanced _offset without populating prefix —
        # leaving the rotation tombstone disarmed for any subsequent
        # inode-recycling rotate. We re-read enough bytes to fill the
        # probe and verify they're consistent with the current cached
        # prefix; on mismatch, force a full rebuild.
        if size > 0 and len(self._prefix) < min(self._PREFIX_PROBE_BYTES, size):
            current_prefix = self._read_prefix(buffer)
            if self._prefix and not current_prefix.startswith(self._prefix):
                return self._full_rebuild(
                    buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
                )
            self._prefix = current_prefix

        if size == self._offset:
            # No new bytes since last call.
            self._hits += 1
            return self._snap

        # Incremental: read [offset, size) and fold any lifecycle
        # events found. Decode is per-line; non-lifecycle lines are
        # skipped cheaply (a JSON parse + a type check).
        new_events: list[SchemaLifecycleEvent] = []
        try:
            with open(buffer.path, "rb") as f:
                f.seek(self._offset)
                tail = f.read(size - self._offset)
        except OSError:
            return self._full_rebuild(
                buffer, strict=strict, deprecate_quorum_k=deprecate_quorum_k
            )

        import json as _json
        from engram.core.types import Event as _Event

        for raw in tail.splitlines():
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue
            try:
                data = _json.loads(line)
                # Only decode lifecycle events; cheap fast-path on type field.
                if data.get("type") != EventType.CONSOLIDATION_SCHEMA_LIFECYCLE.value:
                    continue
                # Path through the encryptor for content if needed; here
                # we don't need .content at all, only .metadata + .ts.
                if buffer._encryptor and buffer._encryptor.enabled and "content" in data:
                    data["content"] = buffer._encryptor.decrypt(data["content"])
                ev = _Event.from_dict(data)
            except (_json.JSONDecodeError, KeyError, ValueError):
                continue
            decoded = event_to_lifecycle(ev)
            if decoded is not None:
                new_events.append(decoded)

        if new_events:
            self._snap = reduce_events(
                new_events,
                strict=strict,
                initial=self._snap,
                deprecate_quorum_k=deprecate_quorum_k,
            )
        self._offset = size
        self._inode_key = inode_key
        self._quorum_k = deprecate_quorum_k
        self._partial_hits += 1
        return self._snap

    def invalidate(self) -> None:
        """Force a full rebuild on the next ``get()``."""
        self._offset = 0
        self._inode_key = None
        self._snap = {}
        self._quorum_k = None

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "partial_hits": self._partial_hits,
            "misses": self._misses,
        }


def make_lifecycle_event(
    *,
    schema_id: str,
    kind: EventKind,
    window_id: str | None = None,
    content: str = "",
    emitter_id: str | None = None,
) -> Event:
    """Build a buffer Event encoding one lifecycle decision.

    Centralized so the wire format only lives in one place. The
    `content` field is informational (e.g. the schema pattern string);
    the reducer ignores it. `emitter_id` (optional) identifies the
    consolidator that emitted this decision; consumed only when the
    reducer runs under `deprecate_quorum_k>1` (§6.16 quorum gate).
    Defaulting it to None preserves the legacy single-emitter wire
    format byte-for-byte (the metadata key is omitted entirely).
    """
    from datetime import datetime, timezone

    from engram.core.types import generate_event_id

    metadata: dict = {
        "schema_id": schema_id,
        "kind": kind.value,
        "window_id": window_id,
    }
    if emitter_id is not None:
        metadata["emitter_id"] = emitter_id

    return Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE,
        content=content,
        metadata=metadata,
    )


__all__ = [
    "CachedLifecycleSnapshot",
    "event_to_lifecycle",
    "iter_lifecycle_events",
    "make_lifecycle_event",
    "snapshot_from_buffer",
]

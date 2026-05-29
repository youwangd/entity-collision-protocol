"""Wire-format tests for `emitter_id` on the lifecycle event stream.

Pins the §6.16 quorum-gate plumbing end-to-end: pipeline writes
`StageContext.consolidator_id` into `make_lifecycle_event(emitter_id=...)`,
which lands in `Event.metadata["emitter_id"]`, which `event_to_lifecycle`
decodes back into `SchemaLifecycleEvent.emitter_id`. Under
`reduce_events(deprecate_quorum_k=2)`, two distinct emitters are needed
to actually fire DEPRECATE; one alone parks in `pending_deprecate_emitters`.

Back-compat invariant (W-3): events emitted *without* `emitter_id` (the
legacy single-emitter wire format) MUST NOT contain the metadata key —
verified against the on-disk JSONL so we don't silently bloat existing
event logs.
"""
from __future__ import annotations

import json
from pathlib import Path


from engram.consolidation.lifecycle_projection import (
    event_to_lifecycle,
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaStatus,
    reduce_events,
)
from engram.core.types import EventType
from engram.store.buffer import JSONLBufferStore


def _buf(tmp_path: Path) -> JSONLBufferStore:
    return JSONLBufferStore(base_path=tmp_path)


# ---------------------------------------------------------------------------
# W-1: emitter_id round-trips through buffer and decoder
# ---------------------------------------------------------------------------
def test_w1_emitter_id_roundtrip(tmp_path: Path):
    buf = _buf(tmp_path)
    ev = make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
        emitter_id="consolidator-A",
    )
    buf.append(ev)
    scanned = list(buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE))
    assert len(scanned) == 1
    decoded = event_to_lifecycle(scanned[0])
    assert decoded is not None
    assert decoded.emitter_id == "consolidator-A"
    assert decoded.schema_id == "s1"
    assert decoded.kind == EventKind.CREATE


# ---------------------------------------------------------------------------
# W-2: missing emitter_id decodes to None (legacy log compatibility)
# ---------------------------------------------------------------------------
def test_w2_missing_emitter_id_is_none(tmp_path: Path):
    buf = _buf(tmp_path)
    ev = make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
    )
    buf.append(ev)
    decoded = event_to_lifecycle(
        next(iter(buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)))
    )
    assert decoded is not None
    assert decoded.emitter_id is None


# ---------------------------------------------------------------------------
# W-3: legacy wire format is byte-stable — no `emitter_id` key written when
# the field is None. This protects existing event logs from accidentally
# acquiring a new metadata key (and the on-disk diff that comes with it).
# ---------------------------------------------------------------------------
def test_w3_no_emitter_id_key_when_unset(tmp_path: Path):
    buf = _buf(tmp_path)
    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
    ))
    raw = buf.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    payload = json.loads(raw[0])
    assert "emitter_id" not in payload.get("metadata", {})


# ---------------------------------------------------------------------------
# W-4: malformed emitter_id (non-string) is rejected by the decoder
# (drops the row in lenient projection mode).
# ---------------------------------------------------------------------------
def test_w4_malformed_emitter_id_drops_row(tmp_path: Path):
    from engram.core.types import Event, generate_event_id
    from datetime import datetime, timezone

    buf = _buf(tmp_path)
    bad = Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE,
        content="",
        metadata={
            "schema_id": "s1",
            "kind": "create",
            "window_id": "w1",
            "emitter_id": 12345,  # not a string
        },
    )
    buf.append(bad)
    snap = snapshot_from_buffer(buf, strict=False)
    assert snap == {}  # malformed row dropped, nothing accumulated


# ---------------------------------------------------------------------------
# W-5: end-to-end §6.16 quorum gate via the buffer projection.
# Two distinct emitter_ids → DEPRECATE fires; one alone → pending only.
# ---------------------------------------------------------------------------
def test_w5_quorum_k2_via_buffer(tmp_path: Path):
    buf = _buf(tmp_path)
    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
        emitter_id="A",
    ))
    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.DEPRECATE, window_id="w2",
        emitter_id="A",
    ))
    decoded = [
        d for ev in buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
        if (d := event_to_lifecycle(ev)) is not None
    ]
    snap1 = reduce_events(decoded, deprecate_quorum_k=2, strict=True)
    assert snap1["s1"].status == SchemaStatus.INFERRED
    assert snap1["s1"].pending_deprecate_emitters == frozenset({"A"})

    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.DEPRECATE, window_id="w3",
        emitter_id="B",
    ))
    decoded = [
        d for ev in buf.scan(event_type=EventType.CONSOLIDATION_SCHEMA_LIFECYCLE)
        if (d := event_to_lifecycle(ev)) is not None
    ]
    snap2 = reduce_events(decoded, deprecate_quorum_k=2, strict=True)
    assert snap2["s1"].status == SchemaStatus.DEPRECATED
    assert snap2["s1"].pending_deprecate_emitters == frozenset()
    assert snap2["s1"].deprecate_count == 1


# ---------------------------------------------------------------------------
# W-6: under k=1 (legacy default), emitter_id is ignored — DEPRECATE fires
# on a single event regardless of who emitted it. Pure regression guard.
# ---------------------------------------------------------------------------
def test_w6_k1_ignores_emitter_id(tmp_path: Path):
    buf = _buf(tmp_path)
    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.CREATE, window_id="w1",
        emitter_id="A",
    ))
    buf.append(make_lifecycle_event(
        schema_id="s1", kind=EventKind.DEPRECATE, window_id="w2",
        emitter_id="A",
    ))
    snap = snapshot_from_buffer(buf, strict=True)
    assert snap["s1"].status == SchemaStatus.DEPRECATED
    assert snap["s1"].deprecate_count == 1


# ---------------------------------------------------------------------------
# W-7: pipeline integration smoke — StageContext.consolidator_id propagates
# to the wire when SchemaUpdate runs. We exercise this by inspecting the
# default helper value rather than spinning up the full pipeline (covered
# by `test_schema_recover_integration.py` etc.). This pins the dataclass
# default so a future refactor doesn't silently drop the field.
# ---------------------------------------------------------------------------
def test_w7_stage_context_default():
    from engram.consolidation.pipeline import StageContext
    ctx = StageContext()
    assert ctx.consolidator_id == ""
    ctx2 = StageContext(consolidator_id="node-7")
    assert ctx2.consolidator_id == "node-7"

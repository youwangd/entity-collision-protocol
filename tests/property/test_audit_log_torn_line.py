"""Property test: AuditLog.read() survives torn lines and corrupt mid-stream
entries.

Origin (NEXT.md priority #3 audit, 2026-05-23): the JSONL buffer's torn-frame
fusion bug (fixed in f218b47) prompted an audit of every other JSONL writer in
the codebase. `src/engram/audit/log.py` uses `open(self.path, "a")` (POSIX
O_APPEND) without flock and without a tail-repair pass. Audit entries with
large `details` dicts can exceed PIPE_BUF (4096 on Linux) and tear under
concurrent writers or process kills, leaving a half-line at EOF.

This test pins the *read-side* invariant: regardless of what garbage lives
in audit.jsonl — torn final line, garbage byte run in the middle, mixed
empty lines — `AuditLog.read()` must:

  1. Never raise.
  2. Return every valid entry that precedes the corruption.
  3. Skip the corrupt line(s) silently.

If a future change to AuditLog.read() loses this property (e.g. switches to
strict-utf8 file mode that aborts the whole scan on a single bad byte, the
exact failure mode buffer.py had pre-f218b47), this test falsifies it.

Marked `chaos` + `property`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.audit.log import AuditLog

pytestmark = [pytest.mark.chaos, pytest.mark.property]


@given(
    n_good=st.integers(min_value=0, max_value=20),
    truncate_bytes=st.integers(min_value=1, max_value=80),
    inject_garbage=st.booleans(),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_audit_read_survives_torn_tail_and_garbage(
    n_good: int, truncate_bytes: int, inject_garbage: bool
) -> None:
    """For any audit.jsonl ending in a half-written line and/or containing a
    garbage line in the middle, AuditLog.read() returns every preceding
    valid entry without raising."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        log = AuditLog(base)

        # Lay down n_good clean entries.
        for i in range(n_good):
            log.log(
                operation="test_op",
                actor=f"actor_{i}",
                details={"i": i, "payload": "x" * (i % 7)},
            )

        # Optionally inject a non-JSON garbage line mid-stream.
        if inject_garbage and n_good > 0:
            with open(log.path, "a", encoding="utf-8") as f:
                f.write("this is not json at all\n")

        # Write a partial trailing line by truncating the last appended record.
        log.log(operation="will_be_torn", actor="x", details={"big": "y" * 200})
        sz = log.path.stat().st_size
        cut = max(0, sz - truncate_bytes)
        with open(log.path, "r+b") as f:
            f.truncate(cut)

        # Read must not crash, must return all initial good entries.
        log2 = AuditLog(base)
        entries = log2.read(limit=10_000)

        # Every returned entry parsed successfully.
        for e in entries:
            assert isinstance(e, dict)
            assert "op" in e and "actor" in e

        # The first n_good entries (clean ops) must all survive — torn-tail
        # corruption at EOF cannot eat earlier records.
        clean_ops = [e for e in entries if e.get("op") == "test_op"]
        assert len(clean_ops) == n_good, (
            f"lost clean entries to torn-tail: got {len(clean_ops)} of {n_good} "
            f"(truncate_bytes={truncate_bytes}, garbage={inject_garbage})"
        )

        # Filter on op also works through corruption.
        filtered = log2.read(limit=10_000, operation="test_op")
        assert len(filtered) == n_good


def test_audit_read_skips_pure_garbage_file(tmp_path: Path) -> None:
    """Stress floor: even an audit.jsonl that is pure binary garbage must
    not crash read(). Returns []."""
    base = tmp_path
    log = AuditLog(base)
    with open(log.path, "wb") as f:
        f.write(b"\x00\xff\xfe not json \n more junk \x01\x02\n")
    entries = log.read()
    assert entries == []


def test_audit_round_trip_basic(tmp_path: Path) -> None:
    """Sanity: with no corruption, every logged entry is readable verbatim."""
    log = AuditLog(tmp_path)
    log.log("op1", "alice", {"k": 1})
    log.log("op2", "bob", {"k": 2}, outcome="failure", duration_ms=42)

    entries = AuditLog(tmp_path).read()
    assert len(entries) == 2
    assert entries[0]["op"] == "op1" and entries[0]["actor"] == "alice"
    assert entries[1]["op"] == "op2" and entries[1]["outcome"] == "failure"
    assert entries[1]["ms"] == 42
    # JSON details preserved.
    assert json.loads(json.dumps(entries[0]["details"])) == {"k": 1}

"""Atomic-write helpers for evals/ result writers.

Motivation
----------
Most ``evals/*.py`` sweep drivers do something like::

    Path(out).write_text(json.dumps(report, indent=2))

at the very end of a multi-hour run. If the process is killed (Ctrl-C,
OOM, the box reboots, or a cron tick is reaped) anywhere inside that
single ``write_text`` call, the file on disk is left half-written —
truncated mid-string with no closing brace. The next ``json.load`` of
that artifact throws ``JSONDecodeError`` and the whole sweep is lost.

This module provides a write-to-tmp + ``os.replace`` wrapper so the
on-disk artifact is *either* the previous version (or absent) *or* a
complete, parsable JSON document — never a torn frame. ``os.replace``
is atomic on POSIX within a single filesystem, which is what every
``evals/`` writer uses.

The helpers are deliberately tiny and dependency-free — they live in
``evals/`` because that's where the unsafe writers are; ``src/`` JSONL
writers were already hardened in a separate audit (see
``src/engram/store/buffer.py`` and ``src/engram/audit/log.py``).

Public API
----------
- :func:`atomic_write_text` — write a string atomically.
- :func:`atomic_write_json` — ``json.dumps`` + atomic write.
- :func:`atomic_write_bytes` — for figure PNGs / pickles.

All three accept :class:`os.PathLike` or :class:`str` and create
parent directories on demand (matching ``write_text`` ergonomics so
call-site migration is mechanical).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write_impl(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically.

    Strategy: open ``NamedTemporaryFile`` in the *same directory* as
    the destination so ``os.replace`` is a same-filesystem rename,
    flush + fsync the tmp before the rename, and clean up the tmp on
    error so a killed run doesn't leave ``*.tmp.XXXXXX`` litter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False so we can rename; we manage cleanup manually.
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync can fail on some pseudo-FSes (tmpfs in
                # containers, etc.); the rename is still atomic
                # within the FS, so we tolerate this.
                pass
        os.replace(tmp_path, path)
    except BaseException:
        # KeyboardInterrupt / SystemExit included — clean up the tmp
        # before re-raising so we don't leave debris behind.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Atomically write ``data`` to ``path``. Returns the resolved Path."""
    p = Path(path)
    _atomic_write_impl(p, data)
    return p


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write ``text`` to ``path``. Returns the resolved Path."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: str | os.PathLike[str],
    obj: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    default: Any = None,
) -> Path:
    """Serialize ``obj`` to JSON and write atomically.

    The trailing newline matches the convention used by the existing
    ``Path.write_text(json.dumps(..., indent=2))`` call sites in
    ``evals/`` (most omit the newline, but some pipe through ``jq``
    and benefit from one — including it is harmless either way).
    """
    payload = json.dumps(obj, indent=indent, sort_keys=sort_keys, default=default)
    if not payload.endswith("\n"):
        payload += "\n"
    return atomic_write_text(path, payload)


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
]

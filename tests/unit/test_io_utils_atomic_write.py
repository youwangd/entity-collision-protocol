"""Tests for evals.io_utils — atomic-write helpers.

These tests exercise the three failure modes that motivated the
helper:

1. **Happy path** — the file lands with exactly the bytes we asked
   for and no leftover ``*.tmp.*`` siblings.
2. **Crash mid-write** — if the writer raises after the tmp file is
   created but before ``os.replace``, the destination must be either
   (a) absent (first write) or (b) the previous version (overwrite),
   never a torn frame. The tmp file must also be cleaned up.
3. **Same-filesystem rename** — the tmp file must be created in the
   destination's parent directory so ``os.replace`` is atomic. We
   verify by asserting tmp parent equals destination parent.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evals.io_utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)


class AtomicWriteHappyPathTests(unittest.TestCase):
    def test_write_text_creates_file_with_exact_content(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            atomic_write_text(p, "hello\nworld")
            self.assertEqual(p.read_text(), "hello\nworld")

    def test_write_bytes_creates_file_with_exact_content(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.bin"
            atomic_write_bytes(p, b"\x00\x01\x02\xff")
            self.assertEqual(p.read_bytes(), b"\x00\x01\x02\xff")

    def test_write_json_roundtrips(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.json"
            obj = {"a": 1, "b": [1, 2, 3], "nested": {"x": True}}
            atomic_write_json(p, obj)
            self.assertEqual(json.loads(p.read_text()), obj)
            # Default indent=2 + trailing newline.
            self.assertTrue(p.read_text().endswith("\n"))
            self.assertIn("  ", p.read_text())  # indented

    def test_write_json_sort_keys(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.json"
            atomic_write_json(p, {"b": 2, "a": 1}, sort_keys=True, indent=None)
            self.assertEqual(p.read_text().rstrip(), '{"a": 1, "b": 2}')

    def test_creates_parent_dirs(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "deep" / "nested" / "out.txt"
            atomic_write_text(p, "ok")
            self.assertTrue(p.exists())

    def test_no_leftover_tmp_files(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            atomic_write_text(p, "x")
            siblings = list(Path(td).iterdir())
            self.assertEqual([s.name for s in siblings], ["out.txt"])

    def test_overwrite_existing(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            p.write_text("old")
            atomic_write_text(p, "new")
            self.assertEqual(p.read_text(), "new")


class AtomicWriteCrashSafetyTests(unittest.TestCase):
    def test_crash_before_replace_leaves_no_torn_destination(self):
        """If write fails, the destination must not exist (first write)."""
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"

            real_replace = os.replace

            def boom(*args, **kwargs):
                raise RuntimeError("simulated crash before rename")

            with mock.patch("evals.io_utils.os.replace", side_effect=boom):
                with self.assertRaises(RuntimeError):
                    atomic_write_text(p, "should-not-land")

            self.assertFalse(p.exists(), "destination file should not exist")
            # And no tmp debris.
            siblings = list(Path(td).iterdir())
            self.assertEqual(siblings, [], f"tmp left behind: {siblings}")

            # Recovery: a subsequent successful write works.
            assert real_replace is os.replace  # sanity
            atomic_write_text(p, "second-attempt")
            self.assertEqual(p.read_text(), "second-attempt")

    def test_crash_overwrite_preserves_previous_version(self):
        """If overwrite fails, the destination keeps the prior content."""
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            p.write_text("v1-stable")

            def boom(*args, **kwargs):
                raise RuntimeError("simulated crash")

            with mock.patch("evals.io_utils.os.replace", side_effect=boom):
                with self.assertRaises(RuntimeError):
                    atomic_write_text(p, "v2-half-written")

            self.assertEqual(p.read_text(), "v1-stable")
            siblings = list(Path(td).iterdir())
            self.assertEqual([s.name for s in siblings], ["out.txt"])

    def test_keyboard_interrupt_cleans_up_tmp(self):
        """KeyboardInterrupt mid-write should not leave tmp debris."""
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"

            def boom(*args, **kwargs):
                raise KeyboardInterrupt

            with mock.patch("evals.io_utils.os.replace", side_effect=boom):
                with self.assertRaises(KeyboardInterrupt):
                    atomic_write_text(p, "interrupted")

            self.assertFalse(p.exists())
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_tmp_in_destination_parent(self):
        """The tmp file must be a sibling of the destination so
        os.replace is a same-filesystem atomic rename."""
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"

            captured = {}

            def fake_replace(src, dst):
                captured["src"] = Path(src)
                captured["dst"] = Path(dst)
                # Don't actually replace — bail with a sentinel.
                raise RuntimeError("captured")

            with mock.patch("evals.io_utils.os.replace", side_effect=fake_replace):
                with self.assertRaises(RuntimeError):
                    atomic_write_text(p, "x")

            self.assertEqual(captured["src"].parent, p.parent)
            self.assertTrue(captured["src"].name.startswith(p.name + "."))
            self.assertTrue(captured["src"].name.endswith(".tmp"))


class AtomicWriteFsyncTests(unittest.TestCase):
    def test_fsync_failure_is_tolerated(self):
        """fsync can fail on some pseudo-FSes; we should still write."""
        with TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            with mock.patch("evals.io_utils.os.fsync", side_effect=OSError):
                atomic_write_text(p, "ok")
            self.assertEqual(p.read_text(), "ok")


if __name__ == "__main__":
    unittest.main()

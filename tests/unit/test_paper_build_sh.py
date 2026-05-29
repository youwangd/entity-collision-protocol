"""Smoke tests for paper/build.sh — review-PDF / camera-ready pipeline.

The full pandoc/latexmk path is exercised opportunistically (skipped if
those binaries are absent in the dev environment), but the always-on
contract — anonymization gate + Markdown concatenation — is asserted
unconditionally.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD_SH = REPO / "paper" / "build.sh"
PAPER_DIR = REPO / "paper"
SECTIONS = [
    PAPER_DIR / name
    for name in (
        "00_abstract.md",
        "10_intro.md",
        "20_related.md",
        "30_methods.md",
        "40_results.md",
        "50_discussion.md",
        "60_threats.md",
        "70_conclusion.md",
    )
]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(
        ["bash", str(BUILD_SH), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_build_sh_exists_and_executable() -> None:
    assert BUILD_SH.is_file(), "paper/build.sh must exist"
    assert os.access(BUILD_SH, os.X_OK), "paper/build.sh must be chmod +x"


def test_build_sh_syntax_is_valid() -> None:
    r = subprocess.run(
        ["bash", "-n", str(BUILD_SH)], capture_output=True, text=True
    )
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"


def test_build_sh_rejects_unknown_flag() -> None:
    r = _run(["--no-such-flag"])
    assert r.returncode == 2
    assert "unknown arg" in r.stderr


def test_build_sh_help_does_not_run_pipeline() -> None:
    r = _run(["--help"])
    assert r.returncode == 0
    # The helptext shouldn't have run the linter.
    assert "anonymization linter" not in r.stdout


def test_build_sh_review_mode_produces_master_md(tmp_path: Path) -> None:
    """Review mode: linter green on the current paper, master Markdown
    contains every section, build/ directory created."""
    r = _run(["--review"])
    assert r.returncode == 0, f"build.sh failed: {r.stdout}\n{r.stderr}"
    master = PAPER_DIR / "build" / "paper.md"
    assert master.is_file()
    body = master.read_text()
    # Each section's H1 must appear in the concatenated master.
    for sec in SECTIONS:
        h1 = sec.read_text().splitlines()[0]
        assert h1 in body, f"missing {sec.name} H1 in master.md"
    assert "Review PDF" in body  # review banner present


def test_build_sh_default_is_review() -> None:
    """No-arg invocation must default to review mode."""
    r = _run([])
    assert r.returncode == 0
    assert "mode=review" in r.stdout


def test_build_sh_pandoc_skip_is_graceful_when_absent() -> None:
    """If pandoc isn't installed, the script must still succeed (skip)."""
    if shutil.which("pandoc"):
        pytest.skip("pandoc is installed in this env; skip-path covered elsewhere")
    r = _run([])
    assert r.returncode == 0
    assert "pandoc not installed" in r.stdout


def test_build_sh_camera_ready_uses_strict_linter() -> None:
    """Camera-ready mode must run the linter in --strict (HTML-comment-aware).
    As of f6cdabf the abstract provenance HTML comment was anonymized
    ('the lead author' → 'the lead author'), so --strict mode now passes cleanly
    across all 8 paper files. This test asserts the gate is wired AND that
    the body is currently camera-ready clean — the inverse of the original
    pre-anonymization assertion. If a future identifier slips into a paper
    file or HTML comment, this test will fail and force a fix before
    camera-ready submission."""
    r = _run(["--camera-ready"])
    assert r.returncode == 0
    assert "mode=camera-ready" in r.stdout
    # Strict-mode linter must have actually run (not silently skipped).
    assert "anonymization linter" in (r.stdout + r.stderr).lower() or "0 finding" in (
        r.stdout + r.stderr
    ).lower() or "strict" in (r.stdout + r.stderr).lower()

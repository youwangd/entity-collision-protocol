"""Test scripts/check_doc_refs.py — the doc cross-ref linter.

Smoke-test that the repo currently passes its own linter, plus exercise
the core detection paths against synthetic inputs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_doc_refs.py"

# Build the broken-path token at runtime so this test file itself doesn't
# contain a static backticked path that the linter would flag against the
# real repo.
BAD_PATH_TOKEN = "nope" + "/" + "missing.py"


def _setup_fake_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo skeleton with the linter installed at
    ``scripts/check_doc_refs.py`` (so ``REPO_ROOT`` resolves to ``tmp_path``)."""
    (tmp_path / "paper").mkdir()
    (tmp_path / "scripts").mkdir()
    fake_script = tmp_path / "scripts" / "check_doc_refs.py"
    fake_script.write_text(SCRIPT.read_text())
    return fake_script


def test_repo_passes_linter_default_mode():
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, (
        f"check_doc_refs.py reports unresolved refs:\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "[paths] 0 unresolved ref" in r.stdout
    assert "[anchors] 0 unresolved ref" in r.stdout


def test_linter_detects_synthetic_broken_path(tmp_path: Path):
    fake_script = _setup_fake_repo(tmp_path)
    (tmp_path / "good.py").write_text("# real file\n")
    (tmp_path / "doc.md").write_text(
        f"Reference to `good.py` (resolves) and `{BAD_PATH_TOKEN}` (does not).\n"
    )
    r = subprocess.run(
        [sys.executable, str(fake_script), "--mode=paths"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 1
    assert BAD_PATH_TOKEN in r.stdout


def test_linter_skips_glob_patterns(tmp_path: Path):
    fake_script = _setup_fake_repo(tmp_path)
    (tmp_path / "doc.md").write_text(
        "Glob `bench/results/*.json` and brace `bench/results/{a,b}.json`.\n"
    )
    r = subprocess.run(
        [sys.executable, str(fake_script), "--mode=paths"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    assert "[paths] 0 unresolved" in r.stdout


def test_linter_detects_dangling_anchor(tmp_path: Path):
    fake_script = _setup_fake_repo(tmp_path)
    (tmp_path / "paper" / "30_methods.md").write_text(
        "## 3.5 Real heading\n\nSee §3.5 (resolves) and §9.99 (dangling).\n"
    )
    r = subprocess.run(
        [sys.executable, str(fake_script), "--mode=anchors"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 1
    assert "§9.99" in r.stdout


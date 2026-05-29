#!/usr/bin/env python3
"""Doc cross-reference linter for the Engram repo.

Mechanises the manual reference audits that have been recurring in the
research-loop cron (see NEXT.md "Recently done" entries from 2026-05-24).

Two checks, both run by default; either can be selected explicitly with
``--mode={paths,anchors,all}``.

1. **Backticked path references** (``--mode=paths``):
   For every Markdown / Python / shell file under the repo, find inline
   ``code spans`` that look like a relative file path — i.e. they
   contain a ``/`` and end in one of a small whitelist of extensions
   (``.py .md .sh .json .jsonl .txt .yml .yaml .png``). Verify the path
   resolves either as-is from repo root, or under ``src/engram/``
   (a common pre-v0.2-layout-move drift). Report unresolved refs.

2. **Intra-paper section anchors** (``--mode=anchors``):
   Inside ``paper/*.md``, every ``§N`` / ``§N.M`` / ``§N.M.X`` /
   ``§A.N.M`` reference must point at a defined heading anchor across
   the paper file set. Defined anchors come from heading lines whose
   trimmed text starts with a numeric-like token (``4.5`` / ``A.4.7.6``
   / ``6.9``). Report dangling refs.

Exit codes
----------
0  no findings
1  one or more unresolved refs
2  usage error

Usage
-----
    python scripts/check_doc_refs.py                # both checks
    python scripts/check_doc_refs.py --mode=paths
    python scripts/check_doc_refs.py --mode=anchors
    python scripts/check_doc_refs.py --quiet        # only counts + nonzero exit

By design this is *advisory*: a few refs are intentionally hypothetical
(e.g. a "future ``store/dedup.py``" forward-pointer) or are inside
historical "Recently done" prose in NEXT.md. These are emitted but the
caller can grep them out. CI can wire ``--strict`` to fail on any hit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Path-ref check
# ---------------------------------------------------------------------------

PATH_EXT_WHITELIST = {
    ".py", ".md", ".sh", ".json", ".jsonl", ".txt",
    ".yml", ".yaml", ".png", ".tex", ".cfg", ".toml",
}

# Inline code span: `...` not spanning newlines. Avoid triple-backtick fences
# by requiring exactly one backtick on each side (no preceding/following ``).
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")

# Files we audit for path refs.
PATH_AUDIT_GLOBS = [
    "paper/*.md",
    "*.md",
    "src/engram/**/*.py",
    "tests/**/*.py",
    "evals/**/*.py",
    "scripts/**/*.py",
]

# Substrings that mark a hit as a known-historical / by-design ref to skip.
PATH_HIT_SKIP_SUBSTRINGS = (
    # Historical "Recently done" log entries in NEXT.md preserve old commit
    # message audit trails; we don't want to keep rewriting them.
    "NEXT.md",
    # HANDOFF_*.md docs are forward-looking SOPs for off-host agents — they
    # name artifacts/scripts the M4 agent is expected to produce. Refs there
    # legitimately don't resolve on the Linux x86_64 host checkout until the
    # handoff agent uploads results. Audit those manually if needed.
    "HANDOFF_",
)

# Forward-pointer / hypothetical refs allowed.
PATH_KNOWN_HYPOTHETICAL = {
    # explicit "or a new store/dedup.py" call-out in TODO-GOVERNED-MEMORY
    "store/dedup.py",
}


def _looks_like_path(s: str) -> bool:
    """Return True iff ``s`` plausibly names a *concrete* file path under the
    repo. Reject glob/brace patterns and abstract placeholders — those are
    expected to not resolve and aren't useful to lint here.
    """
    if "/" not in s:
        return False
    if any(c.isspace() for c in s):
        return False
    # Reject shell glob patterns (brace expansion, wildcards, placeholders).
    if any(c in s for c in "*{}<>"):
        return False
    # Reject home-relative paths and absolute /tmp/-style paths — not repo refs.
    if s.startswith(("~", "/tmp/", "/var/", "/etc/")):
        return False
    # Reject paths under build/ artifact directories (gitignored, may not exist
    # on a fresh checkout).
    if "/build/" in s or s.startswith("build/"):
        return False
    s = s.lstrip("./")
    suffix = Path(s).suffix
    if suffix not in PATH_EXT_WHITELIST:
        return False
    if "://" in s or "@" in s:
        return False
    return True


def _resolve_path(ref: str) -> Path | None:
    """Try to resolve ``ref`` against the repo root, with the v0.2 src/engram
    fallback. Returns the matched Path, or None if not found.
    """
    ref = ref.lstrip("./")
    candidates = [
        REPO_ROOT / ref,
        REPO_ROOT / "src" / "engram" / ref,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def audit_paths(strict: bool) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    seen_files: set[Path] = set()
    for glob in PATH_AUDIT_GLOBS:
        for p in REPO_ROOT.glob(glob):
            if not p.is_file() or p in seen_files:
                continue
            seen_files.add(p)
            rel = p.relative_to(REPO_ROOT)
            if not strict and any(s in str(rel) for s in PATH_HIT_SKIP_SUBSTRINGS):
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            in_fence = False
            for lineno, line in enumerate(lines, start=1):
                stripped = line.lstrip()
                if stripped.startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for m in INLINE_CODE_RE.finditer(line):
                    span = m.group(1).strip()
                    if not _looks_like_path(span):
                        continue
                    if span in PATH_KNOWN_HYPOTHETICAL:
                        continue
                    if _resolve_path(span) is None:
                        findings.append((rel, lineno, span))
    return findings


# ---------------------------------------------------------------------------
# Anchor-ref check
# ---------------------------------------------------------------------------

PAPER_DIR = REPO_ROOT / "paper"

# A defined anchor heading: "## 4.5 Title" or "### A.4.7.6 Title"
HEADING_RE = re.compile(r"^#{1,6}\s+(?:§\s*)?([A-Za-z]?\d+(?:\.[A-Za-z\d]+)*[a-z]?)\b")

# A reference: §-prefixed numeric token, possibly with a leading letter.
# We require at least one dot to limit collisions with prose like "§5".
ANCHOR_REF_RE = re.compile(r"§\s*([A-Za-z]?\d+(?:\.[A-Za-z\d]+)+[a-z]?)")

# Bare §N refs (no dot).
BARE_REF_RE = re.compile(r"§\s*(\d+)(?![\w.])")

# Bare §N anchors documented as external / legacy non-paper anchors.
# 8/65/92/93/96 are spelled out in the SCALE_REPORT glossary; 87 is the
# legacy SCALE_REPORT anchor for the consolidation pipeline (§87 stages,
# §87 schema-lifecycle controller).
KNOWN_BARE_REFS_OK = {"8", "65", "87", "92", "93", "96"}

# Skip §-refs that are actually arXiv-id tail digits (§17787 ← arXiv:2603.17787)
# or template placeholders (§4.X with literal X).
ANCHOR_REF_SKIP_RE = re.compile(r"arXiv:\d{4}\.")


def collect_paper_anchors() -> set[str]:
    anchors: set[str] = set()
    if not PAPER_DIR.is_dir():
        return anchors
    for md in sorted(PAPER_DIR.glob("*.md")):
        for line in md.read_text(encoding="utf-8").splitlines():
            m = HEADING_RE.match(line)
            if m:
                anchors.add(m.group(1))
    return anchors


def audit_anchors() -> list[tuple[Path, int, str]]:
    if not PAPER_DIR.is_dir():
        return []
    anchors = collect_paper_anchors()
    findings: list[tuple[Path, int, str]] = []
    for md in sorted(PAPER_DIR.glob("*.md")):
        rel = md.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            md.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for m in ANCHOR_REF_RE.finditer(line):
                ref = m.group(1)
                if ref in anchors:
                    continue
                # Skip template placeholders (§N.X with literal uppercase X).
                segs = ref.split(".")
                if any(len(s) == 1 and s.isupper() for s in segs[1:]):
                    continue
                alt = ref[2:] if ref.startswith("A.") else f"A.{ref}"
                if alt in anchors:
                    continue
                findings.append((rel, lineno, f"§{ref}"))
            for m in BARE_REF_RE.finditer(line):
                ref = m.group(1)
                if ref in anchors or ref in KNOWN_BARE_REFS_OK:
                    continue
                # Skip arXiv-id tail digits (§17787 ← arXiv:2603.17787) — any
                # 4+ digit bare § is not a paper section, every real section
                # number is single- or low-double-digit.
                if len(ref) >= 4:
                    continue
                findings.append((rel, lineno, f"§{ref} (bare)"))
    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("paths", "anchors", "all"), default="all")
    ap.add_argument("--strict", action="store_true",
                    help="Don't skip the historical NEXT.md log block.")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print counts.")
    args = ap.parse_args(argv)

    n_path = n_anchor = 0

    if args.mode in ("paths", "all"):
        hits = audit_paths(strict=args.strict)
        n_path = len(hits)
        if not args.quiet:
            for p, ln, span in hits:
                print(f"{p}:{ln}: unresolved path ref `{span}`")
        print(f"[paths] {n_path} unresolved ref(s)")

    if args.mode in ("anchors", "all"):
        hits = audit_anchors()
        n_anchor = len(hits)
        if not args.quiet:
            for p, ln, span in hits:
                print(f"{p}:{ln}: unresolved anchor ref {span}")
        print(f"[anchors] {n_anchor} unresolved ref(s)")

    return 0 if (n_path + n_anchor) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

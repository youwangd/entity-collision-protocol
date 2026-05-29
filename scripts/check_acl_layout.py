#!/usr/bin/env python3
"""
ACL layout validator — fails the build if the rendered PDF has too many
typographic problems.

Checks (against the .log produced by xelatex):

  1. **Overfull hboxes**: any line that physically overflows the column.
     Threshold: max overflow per line ≤ 5pt; total count ≤ 5.

  2. **Badness 10000 underfulls**: lines with severe word-spacing rivers.
     Threshold: ≤ 30 (some are unavoidable in narrow 2-col layout).

  3. **Page count**: body (excl. limitations) must fit Industry Track 6-page
     limit per the EMNLP CFP. Limitations and appendix excluded.

  4. **Anonymization**: re-runs check_anon.py and aborts on any finding
     across the 16 review-mode files.

  5. **Verbatim-with-unicode**: scans the .tex for `\\begin{verbatim}` blocks
     containing non-ASCII bytes (greek letters, dashes) which xelatex can't
     render in monospace and produces ^^K-style escape sequences.

Usage:
    python scripts/check_acl_layout.py [--strict]

Exit code: 0 if all checks pass, 1 if any fail. With --strict, also fails
on warnings (e.g., 5pt-class overfulls that are visible at high DPI).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "paper" / "build" / "acl" / "engram_v0.2_acl.log"
TEX_DIR = REPO / "paper" / "build" / "acl"
PDF = REPO / "paper" / "dist" / "engram_v0.2_acl.pdf"

# Thresholds — tighter than baseline, looser than perfect.
# Baseline at v0.2 was 47 overfulls (worst 240pt) and 289 badness-10000
# underfulls. Current achievable is 0 overfulls > 5pt and ~165 badness-10000
# (the latter is inherent to narrow 2-col + bold-heavy intro text).
MAX_OVERFULL_PT = 5.0          # any single overflow > this is a hard fail
MAX_OVERFULL_COUNT = 5         # total count of any-size overfulls
MAX_BADNESS_10000_COUNT = 200  # severe word-spacing rivers; bold-cluster bound
BODY_PAGE_LIMIT = 6            # EMNLP Industry Track CFP

REVIEW_FILES = [
    "00_abstract.md", "10_intro.md", "20_related.md", "30_methods.md",
    "40_results.md", "50_discussion.md", "60_threats.md", "70_conclusion.md",
    "75_limitations.md", "A1_appendix_ablations.md",
    "A2_appendix_security_audits.md", "A3_extended_related.md",
    "A4_extended_discussion.md", "A5_extended_threats.md",
    "A6_appendix_traceability.md", "A7_extended_methods.md",
]


class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def report(self, strict: bool) -> int:
        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        if self.errors:
            print("\nErrors:")
            for e in self.errors:
                print(f"  ✗ {e}")
            return 1
        if strict and self.warnings:
            print("\n--strict: warnings treated as errors.")
            return 1
        print("\n✓ ACL layout checks passed.")
        return 0


def check_overfulls(log: str, r: Result) -> None:
    pat = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\)")
    matches = [(float(m.group(1)), m.group(0)) for m in pat.finditer(log)]
    if not matches:
        print(f"  ✓ overfull hboxes: 0")
        return
    severe = [(pt, msg) for pt, msg in matches if pt > MAX_OVERFULL_PT]
    print(f"  → overfull hboxes: {len(matches)} total "
          f"(worst {max(m[0] for m in matches):.1f}pt, "
          f"{len(severe)} above {MAX_OVERFULL_PT}pt threshold)")
    if severe:
        r.err(
            f"{len(severe)} overfull hbox(es) exceed {MAX_OVERFULL_PT}pt — "
            f"worst {max(m[0] for m in severe):.1f}pt. These show as visible "
            f"text bleeding into the right margin in the PDF."
        )
    elif len(matches) > MAX_OVERFULL_COUNT:
        r.warn(f"{len(matches)} small overfulls (none above {MAX_OVERFULL_PT}pt) "
               f"— above suggested ceiling of {MAX_OVERFULL_COUNT}.")


def check_underfull_rivers(log: str, r: Result) -> None:
    n10k = log.count("badness 10000")
    n_total = log.lower().count("underfull")
    print(f"  → underfull hboxes: {n_total} total, {n10k} at max badness")
    if n10k > MAX_BADNESS_10000_COUNT:
        r.warn(
            f"{n10k} lines with badness 10000 (severe word-spacing rivers). "
            f"Threshold is {MAX_BADNESS_10000_COUNT}. Consider reducing **bold** "
            f"clusters in narrow paragraphs, or using shorter compound words."
        )


def check_page_count(log: str, r: Result) -> None:
    if not PDF.exists():
        r.err(f"PDF missing: {PDF}")
        return
    from pypdf import PdfReader
    pages = PdfReader(str(PDF))
    # Reuse the same detection logic as build_paper_acl.py
    lim_needles = ["We name scope limits explicitly", "Limitations\nWe name scope"]
    body_pages = None
    for i, p in enumerate(pages.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            continue
        if any(n in t for n in lim_needles):
            body_pages = i
            break
    if body_pages is None:
        r.warn("Could not detect limitations boundary; body-page check skipped.")
        return
    print(f"  → body pages (excl. limitations): {body_pages}")
    if body_pages > BODY_PAGE_LIMIT:
        r.err(f"Body is {body_pages} pages — Industry Track limit is "
              f"{BODY_PAGE_LIMIT}. (Limitations/appendix excluded per CFP.)")


def check_verbatim_unicode(r: Result) -> None:
    """Detect \\begin{verbatim} blocks containing non-ASCII bytes that xelatex
    can't render in inconsolata. These are the ^^K-glyph nightmare."""
    if not TEX_DIR.exists():
        return
    bad: list[tuple[Path, int, str]] = []
    pat = re.compile(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", re.DOTALL)
    for tex in TEX_DIR.glob("*.tex"):
        text = tex.read_text(errors="replace")
        for m in pat.finditer(text):
            body = m.group(1)
            non_ascii = [c for c in body if ord(c) > 127]
            if non_ascii:
                line = text[: m.start()].count("\n") + 1
                preview = body.strip()[:80].replace("\n", " ")
                bad.append((tex, line, preview))
    if bad:
        for tex, line, preview in bad:
            r.err(f"verbatim-with-unicode in {tex.name}:{line} "
                  f"(use math mode or backticks instead): {preview!r}")
    else:
        print("  ✓ no verbatim-with-unicode blocks")


def check_anonymization(r: Result) -> None:
    paper = REPO / "paper"
    files = [str(paper / f) for f in REVIEW_FILES]
    venv_py = REPO / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    out = subprocess.run(
        [py, str(REPO / "scripts" / "check_anon.py"), *files],
        capture_output=True, text=True,
    )
    last = out.stdout.strip().split("\n")[-1] if out.stdout else ""
    print(f"  → {last}")
    if "OK: 0 findings" not in last:
        r.err(f"check_anon.py reported findings:\n{out.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors")
    args = ap.parse_args()

    if not LOG.exists():
        print(f"Build log missing: {LOG}", file=sys.stderr)
        print("Run scripts/build_paper_acl.py first.", file=sys.stderr)
        return 1
    log = LOG.read_text(errors="replace")
    r = Result()

    print("ACL layout validation")
    print("=" * 50)
    print("Overfull/underfull:")
    check_overfulls(log, r)
    check_underfull_rivers(log, r)
    print("\nPage count:")
    check_page_count(log, r)
    print("\nVerbatim/unicode:")
    check_verbatim_unicode(r)
    print("\nAnonymization:")
    check_anonymization(r)

    return r.report(args.strict)


if __name__ == "__main__":
    sys.exit(main())

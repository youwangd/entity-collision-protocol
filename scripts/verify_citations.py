#!/usr/bin/env python3
"""Verify every arXiv ID cited in a paper file against Semantic Scholar.

Pre-flight check for the EMNLP 2026 Paper Integrity Policy
(https://2026.emnlp.org/paper-integrity-policy/) which uses GPTZero to
detect unverifiable references.  Runs before any camera-ready or arXiv
submission.

Usage:
    python scripts/verify_citations.py paper/20_related.md
    python scripts/verify_citations.py paper/*.md  # check whole paper

Exit codes:
    0  — all arXiv IDs resolved
    1  — at least one ID failed to resolve (non-existent paper)
    2  — network or parsing failure (transient; re-run)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ARXIV_RE = re.compile(r"arXiv:?\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,authors,year,externalIds"


def collect_arxiv_ids(paths: list[Path]) -> dict[str, list[Path]]:
    """Return {arxiv_id: [files referencing it]}; ignores HTML comments."""
    out: dict[str, list[Path]] = {}
    for p in paths:
        text = p.read_text()
        # Strip HTML comments so the audit changelog doesn't false-trigger.
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        for aid in ARXIV_RE.findall(text):
            out.setdefault(aid, []).append(p)
    return out


def verify_batch(arxiv_ids: list[str]) -> list[dict | None]:
    """POST the batch lookup; returns the list aligned with input order."""
    payload = json.dumps({"ids": [f"ARXIV:{a}" for a in arxiv_ids]}).encode()
    req = urllib.request.Request(
        f"{BATCH_URL}?fields={FIELDS}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    return json.loads(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--quiet", action="store_true",
                    help="only print failures")
    args = ap.parse_args()

    by_id = collect_arxiv_ids(args.paths)
    if not by_id:
        print("[verify_citations] no arXiv IDs found — nothing to do.")
        return 0

    ids = sorted(by_id)
    if not args.quiet:
        print(f"[verify_citations] checking {len(ids)} unique arXiv IDs "
              f"across {len({p for ps in by_id.values() for p in ps})} file(s)…")

    try:
        results = verify_batch(ids)
    except Exception as exc:
        print(f"[verify_citations] network/parse failure: {exc}", file=sys.stderr)
        return 2

    bad: list[tuple[str, list[Path]]] = []
    for aid, entry in zip(ids, results):
        files = by_id[aid]
        if entry is None:
            bad.append((aid, files))
            print(f"❌ arXiv:{aid}  NOT FOUND  (in {', '.join(map(str, files))})")
            continue
        if args.quiet:
            continue
        title = entry.get("title", "?")
        authors = entry.get("authors") or []
        first = authors[0]["name"] if authors else "?"
        year = entry.get("year", "?")
        print(f"✅ arXiv:{aid}  ({year}) {first[:25]:<25s} {title[:70]}")

    if bad:
        print(f"\n[verify_citations] FAILURE: {len(bad)} unresolved ID(s).",
              file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\n[verify_citations] OK: {len(ids)} ID(s) verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

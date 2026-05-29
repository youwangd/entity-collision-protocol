#!/usr/bin/env python3
"""scripts/diff_results.py — compare a freshly-regenerated result JSON against
a committed reference and report whether headline metrics agree within a
configurable tolerance (default ±0.5pp absolute on rate-style metrics, and
±25% relative on latency-style metrics).

Designed to be the single acceptance gate for `scripts/reproduce.sh` outputs.
Exit code: 0 = pass, 1 = fail (out of tolerance), 2 = usage / file error.

Currently understands two result-file shapes:

  1. LongMemEval adapter output (`session_hit_at_1`, `session_hit_at_k`,
     `recall_ms.p50`, `ingest_ms.p50`).
  2. 1M-ingest JSON (`writes_per_sec`, `latency_ms.{p50,p95,p99}`).

Usage:
    diff_results.py REF NEW [--rate-tol-abs 0.005] [--latency-tol-rel 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# --- metric-shape detection --------------------------------------------------

def _flatten_lme(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ("session_hit_at_1", "session_hit_at_k"):
        if k in d:
            out[k] = float(d[k])
    for parent in ("recall_ms", "ingest_ms"):
        block = d.get(parent) or {}
        for sub in ("p50", "p95", "p99", "mean"):
            if sub in block:
                out[f"{parent}.{sub}"] = float(block[sub])
    return out


def _flatten_ingest(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "writes_per_sec" in d:
        out["writes_per_sec"] = float(d["writes_per_sec"])
    lat = d.get("latency_ms") or {}
    for sub in ("p50", "p95", "p99", "mean", "max"):
        if sub in lat:
            out[f"latency_ms.{sub}"] = float(lat[sub])
    return out


def flatten(d: dict[str, Any]) -> dict[str, float]:
    flat = _flatten_lme(d)
    flat.update(_flatten_ingest(d))
    return flat


_RATE_KEYS = {"session_hit_at_1", "session_hit_at_k"}


def is_rate(key: str) -> bool:
    return key in _RATE_KEYS


# --- comparison --------------------------------------------------------------

def compare(
    ref: dict[str, float],
    new: dict[str, float],
    rate_tol_abs: float,
    latency_tol_rel: float,
) -> tuple[bool, list[tuple[str, str]]]:
    rows: list[tuple[str, str]] = []
    ok = True
    keys = sorted(set(ref) | set(new))
    for k in keys:
        rv = ref.get(k)
        nv = new.get(k)
        if rv is None or nv is None:
            rows.append((k, f"MISSING ref={rv} new={nv}"))
            ok = False
            continue
        delta = nv - rv
        if is_rate(k):
            tol = rate_tol_abs
            within = abs(delta) <= tol
            rows.append(
                (k, f"ref={rv:.4f} new={nv:.4f} Δ={delta:+.4f} tol=±{tol:.4f} {'✓' if within else '✗'}")
            )
        else:
            tol = latency_tol_rel * abs(rv) if rv else float("inf")
            within = abs(delta) <= tol
            rel = (delta / rv) if rv else float("inf")
            rows.append(
                (k, f"ref={rv:.3f} new={nv:.3f} Δ={delta:+.3f} ({rel:+.1%}) tol=±{latency_tol_rel:.0%} {'✓' if within else '✗'}")
            )
        if not within:
            ok = False
    return ok, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="canonical reference JSON")
    ap.add_argument("new", type=Path, help="freshly-regenerated JSON")
    ap.add_argument("--rate-tol-abs", type=float, default=0.005,
                    help="absolute tolerance for rate metrics (default 0.005 = ±0.5pp)")
    ap.add_argument("--latency-tol-rel", type=float, default=0.25,
                    help="relative tolerance for latency metrics (default 0.25 = ±25%%)")
    args = ap.parse_args()

    for p in (args.ref, args.new):
        if not p.exists():
            print(f"diff_results: file not found: {p}", file=sys.stderr)
            return 2

    ref = flatten(json.loads(args.ref.read_text()))
    new = flatten(json.loads(args.new.read_text()))

    if not ref and not new:
        print("diff_results: no recognised metrics in either file", file=sys.stderr)
        return 2

    ok, rows = compare(ref, new, args.rate_tol_abs, args.latency_tol_rel)
    width = max((len(r[0]) for r in rows), default=10)
    print(f"{'metric'.ljust(width)}  detail")
    print("-" * (width + 2 + 60))
    for k, line in rows:
        print(f"{k.ljust(width)}  {line}")
    print()
    print("PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

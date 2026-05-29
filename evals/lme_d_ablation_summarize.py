"""Summarize the LongMemEval PRF query_expansion_min_dominance ablation.

Reads:
  bench/results/lme_full500_k10_baseline.json    (d = ∞ / off)
  bench/results/lme_full500_k10_prf.json         (d = 0.30, prior run)
  bench/results/lme_d_ablation/prf_d0.2.json
  bench/results/lme_d_ablation/prf_d0.4.json
  bench/results/lme_d_ablation/prf_d0.5.json

Reports session_hit@1 and session_hit@10 vs. d, plus paired Δ-CI vs. baseline
on session_hit@1 using a 5000-resample paired bootstrap on per_instance hits.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"

PATHS = {
    "off": RESULTS / "lme_full500_k10_baseline.json",
    "0.20": RESULTS / "lme_d_ablation" / "prf_d0.2.json",
    "0.30": RESULTS / "lme_full500_k10_prf.json",
    "0.40": RESULTS / "lme_d_ablation" / "prf_d0.4.json",
    "0.50": RESULTS / "lme_d_ablation" / "prf_d0.5.json",
}


def _hits1(d: dict) -> list[int]:
    pi = d.get("per_instance") or d.get("per_instance_hits") or []
    out = []
    for r in pi:
        if isinstance(r, dict):
            v = r.get("session_hit_at_1")
            if v is None:
                v = r.get("hit_at_1")
            out.append(int(bool(v)))
        else:
            out.append(int(bool(r)))
    return out


def _paired_delta_ci(treat: list[int], base: list[int], n_boot: int = 5000, seed: int = 0):
    assert len(treat) == len(base)
    n = len(treat)
    rng = random.Random(seed)
    diffs = [t - b for t, b in zip(treat, base)]
    point = sum(diffs) / n
    samples = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return point, lo, hi


def main():
    rows = {}
    for tag, p in PATHS.items():
        if not p.exists():
            print(f"[skip] {tag}: {p} missing")
            continue
        d = json.loads(p.read_text())
        rows[tag] = {
            "h1": d.get("session_hit_at_1"),
            "hk": d.get("session_hit_at_k"),
            "hits1": _hits1(d),
        }

    if "off" not in rows:
        raise SystemExit("baseline (off) required")
    base = rows["off"]["hits1"]

    print(f"{'d':>6} {'h@1':>8} {'h@10':>8} {'Δh@1':>10} {'CI95':>22}")
    print("-" * 60)
    for tag in ["off", "0.20", "0.30", "0.40", "0.50"]:
        if tag not in rows:
            continue
        r = rows[tag]
        if tag == "off":
            print(f"{tag:>6} {r['h1']:>8.4f} {r['hk']:>8.4f} {'—':>10} {'—':>22}")
            continue
        if r["hits1"] and len(r["hits1"]) == len(base):
            point, lo, hi = _paired_delta_ci(r["hits1"], base)
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            print(f"{tag:>6} {r['h1']:>8.4f} {r['hk']:>8.4f} {point:>+10.4f} {ci:>22}")
        else:
            print(f"{tag:>6} {r['h1']:>8.4f} {r['hk']:>8.4f} {'(n/a)':>10} {'(per_instance n mismatch)':>22}")


if __name__ == "__main__":
    main()

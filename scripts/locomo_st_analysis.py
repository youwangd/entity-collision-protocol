"""Paired-bootstrap CIs for the real LoCoMo ST-embedder 4-arm sweep.

Run after `scripts/run_locomo_st_arms.sh` finishes. Reports Δh@1 / Δh@k vs ST
baseline for {prf, share_prior, both}, plus per-category breakdown for PRF.

Usage:
    python scripts/locomo_st_analysis.py
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench" / "results"


def _load_pq(arm: str) -> list[dict]:
    path = RESULTS_DIR / f"locomo_real_n10_st_{arm}.json"
    return json.loads(path.read_text())["per_query"]


def _summary(arm: str) -> dict:
    d = json.loads((RESULTS_DIR / f"locomo_real_n10_st_{arm}.json").read_text())
    return {
        "h@1": d["session_hit_at_1"],
        "h@k": d["session_hit_at_k"],
        "p50": d["recall_ms"]["p50"],
        "p95": d["recall_ms"].get("p95", d["recall_ms"]["max"]),
    }


def paired_diff_ci(a: list[float], b: list[float], B: int = 10000, seed: int = 42,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    assert len(a) == len(b)
    diffs = [a[i] - b[i] for i in range(len(a))]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * B)]
    hi = means[int((1 - alpha / 2) * B) - 1]
    return statistics.fmean(diffs), lo, hi


def main() -> None:
    arms = ["baseline", "prf", "share_prior", "both"]
    pq = {}
    for a in arms:
        path = RESULTS_DIR / f"locomo_real_n10_st_{a}.json"
        if not path.exists():
            print(f"[skip] {a}: not yet")
            continue
        pq[a] = _load_pq(a)

    print("\n=== Real LoCoMo, ST-MiniLM-384 embedder, n_qa_scored=996 ===\n")
    print(f"{'arm':<12} {'h@1':>7} {'h@k':>7} {'p50_ms':>8} {'p95_ms':>8}")
    for a in arms:
        if a in pq:
            s = _summary(a)
            p95 = s["p95"]
            print(f"{a:<12} {s['h@1']:>7.4f} {s['h@k']:>7.4f} {s['p50']:>8.2f} {p95:>8.2f}")

    if "baseline" not in pq:
        return
    base = pq["baseline"]
    print("\n=== Paired bootstrap (B=10000, α=0.05) Δ vs ST baseline ===\n")
    for a in ("prf", "share_prior", "both"):
        if a not in pq:
            continue
        ot = pq[a]
        m1, lo1, hi1 = paired_diff_ci([r["hit_at_1"] for r in ot],
                                      [r["hit_at_1"] for r in base])
        mk, lok, hik = paired_diff_ci([r["hit_at_k"] for r in ot],
                                      [r["hit_at_k"] for r in base])
        sig1 = "*" if (lo1 > 0 or hi1 < 0) else "ns"
        sigk = "*" if (lok > 0 or hik < 0) else "ns"
        print(f"  {a:<12} Δh@1 = {m1:+.4f} [{lo1:+.4f}, {hi1:+.4f}] {sig1}")
        print(f"  {a:<12} Δh@k = {mk:+.4f} [{lok:+.4f}, {hik:+.4f}] {sigk}")

    if "prf" in pq:
        print("\n=== Per-category Δh@1 (PRF − baseline, ST) ===\n")
        cb, cp = defaultdict(list), defaultdict(list)
        for rb, rp in zip(base, pq["prf"]):
            cb[rb["category"]].append(rb["hit_at_1"])
            cp[rp["category"]].append(rp["hit_at_1"])
        for c in sorted(cb):
            m, lo, hi = paired_diff_ci(cp[c], cb[c], B=5000)
            sig = "*" if (lo > 0 or hi < 0) else "ns"
            print(f"  cat {c} (n={len(cb[c]):>4}): Δh@1={m:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    main()

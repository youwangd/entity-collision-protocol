"""Paired-bootstrap CIs for the LoCoMo type-purity-gate ablation (§4.15n).

Compares baseline vs. {prf (heuristic), prf+spacy_sm, prf+spacy_sm+tp>=0.5}.
Hypothesis under test: does the type-purity gate (require dominant NER label
share >= 0.5 among first-pass entities) cure the cat-4 PRF regression
observed in §4.15m by suppressing type-salad expansions?

Verdict (n=1986 questions, B=10000): the gate REJECTS the hypothesis.
Activating tp>=0.5 *amplifies* the cat-4 regression (Δh@1 -0.0155 ungated
→ -0.0523 gated, both significant) and the overall Δh@1 worsens from
-0.0212 to -0.0445. Type-coherent PRF expansions are not safer; they
crowd out non-entity discriminative terms in cat-4 (knowledge-update)
queries even more than mixed-type expansions do.

Run after locomo_real_n10_{prf_spacy,prf_spacy_tp50}.json exist.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench" / "results"


def _load(arm: str) -> list[tuple[str, int]]:
    path = RESULTS_DIR / f"locomo_real_n10_{arm}.json"
    d = json.loads(path.read_text())
    return [(q["category"], int(q["hit_at_1"])) for q in d["per_query"]]


def _bootstrap_diff(base, treat, *, B=10000, seed=0):
    rng = random.Random(seed)
    diffs = [t[1] - b[1] for b, t in zip(base, treat)]
    n = len(diffs)
    samples = []
    for _ in range(B):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(s) / n)
    samples.sort()
    return sum(diffs) / n, samples[int(0.025 * B)], samples[int(0.975 * B)]


def _per_cat(base, treat, *, B=10000, seed=0):
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for (cb, vb), (_ct, vt) in zip(base, treat):
        by_cat.setdefault(cb, []).append(vt - vb)
    out = {}
    for c, ds in by_cat.items():
        n = len(ds)
        samples = []
        for _ in range(B):
            s = [ds[rng.randrange(n)] for _ in range(n)]
            samples.append(sum(s) / n)
        samples.sort()
        out[c] = (sum(ds) / n, samples[int(0.025 * B)], samples[int(0.975 * B)], n)
    return out


def main() -> int:
    base = _load("baseline")
    print("baseline n =", len(base))
    print()
    print("=== Aggregate Δh@1 vs baseline (paired bootstrap, B=10000, α=0.05) ===")
    for arm in ("prf", "prf_spacy", "prf_spacy_tp50"):
        try:
            t = _load(arm)
        except FileNotFoundError:
            print(f"  {arm:18s} (missing)")
            continue
        m, lo, hi = _bootstrap_diff(base, t)
        sig = "*" if lo * hi > 0 else "ns"
        print(f"  {arm:18s} Δ={m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    print()
    for arm in ("prf_spacy", "prf_spacy_tp50"):
        try:
            t = _load(arm)
        except FileNotFoundError:
            continue
        print(f"=== Per-category ({arm} vs baseline) ===")
        pc = _per_cat(base, t)
        for c in sorted(pc):
            m, lo, hi, n = pc[c]
            sig = "*" if lo * hi > 0 else "ns"
            print(f"  cat{c} n={n:4d}  Δ={m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

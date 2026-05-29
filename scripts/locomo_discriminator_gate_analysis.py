"""§4.15o — Non-entity-discriminator PRF gate (offline replay, BM25).

Hypothesis: PRF regresses on LoCoMo because for entity-only queries
("Who is X?", "What is Y's job?") the 3-token entity expansion crowds
out nothing useful — but it doesn't *hurt* either, since the entities
already dominate the BM25 signal. The per-category breakdown of D19/D21
shows the regression is concentrated where the query carries non-entity
discriminators (verbs, temporal markers) — *those* are what get diluted
when the expansion injects more entity tokens.

So: fire PRF *only* when the query contains a non-entity VERB or
temporal token. If the gate is correct, gate-on should look like the
PRF arm on "discriminator" queries and like the baseline elsewhere,
strictly dominating both global arms.

This is an offline replay: we already ran baseline and prf_spacy on
the same paired stream of 1986 questions in D21. We just choose the
per-query outcome based on the gate predicate.

Three gate variants tested:
  * verb_only:     ≥1 token with POS=VERB outside named-entity spans
  * temporal_only: ≥1 token whose lemma matches a closed-class temporal vocab
  * verb_or_temp:  union of the above (the version proposed in NEXT.md)

Verdict to be filled in after running.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench" / "results"
DATA_PATH = ROOT / "data" / "locomo" / "locomo10.json"

# Closed-class temporal lemmas — covers most LoCoMo cat-2 / cat-4 patterns.
TEMPORAL_LEMMAS = {
    "when", "before", "after", "during", "while", "until", "since",
    "first", "last", "recent", "recently", "ago", "yesterday", "today",
    "tomorrow", "morning", "afternoon", "evening", "night", "week",
    "weekend", "month", "year", "day", "date", "time",
    "earlier", "later", "soon", "now", "then", "ever",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "spring", "summer", "fall", "autumn", "winter",
}


def load_arm(arm: str):
    path = RESULTS_DIR / f"locomo_real_n10_{arm}.json"
    d = json.loads(path.read_text())
    return d["per_query"]


_EVIDENCE_RE = __import__("re").compile(r"^D(\d+)(?::\d+)?$")


def _evidence_to_sessions(evidence) -> list[str]:
    if not evidence:
        return []
    if isinstance(evidence, str):
        evidence = [evidence]
    out = []
    for e in evidence:
        e = str(e).strip()
        m = _EVIDENCE_RE.match(e)
        if m:
            out.append(f"D{int(m.group(1))}")
        elif e.startswith("D") and e[1:].isdigit():
            out.append(e)
    seen, dedup = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def iter_questions_in_per_query_order():
    """Mirror evals/locomo_adapter.py: skip QAs whose evidence list
    parses to empty session set (not just empty `evidence` field)."""
    raw = json.loads(DATA_PATH.read_text())
    for entry in raw:
        sid = entry["sample_id"]
        for q in entry.get("qa", []) or []:
            sess = _evidence_to_sessions(q.get("evidence"))
            if not sess:
                continue
            yield sid, q


def classify_questions(nlp):
    """Return list[dict] aligned with per_query, with gate flags."""
    flags = []
    for sid, q in iter_questions_in_per_query_order():
        question = q["question"]
        doc = nlp(question)
        ent_token_idxs = set()
        for ent in doc.ents:
            for tok in ent:
                ent_token_idxs.add(tok.i)
        has_verb_outside = any(
            t.pos_ == "VERB" and t.lemma_.lower() not in {"be", "do", "have"}
            and t.i not in ent_token_idxs
            for t in doc
        )
        has_temp_outside = any(
            t.lemma_.lower() in TEMPORAL_LEMMAS and t.i not in ent_token_idxs
            for t in doc
        )
        flags.append({
            "sample_id": sid,
            "category": str(q.get("category", "?")),
            "question": question,
            "has_verb_outside_ent": has_verb_outside,
            "has_temp_outside_ent": has_temp_outside,
            "n_ents": len(doc.ents),
            "n_tokens": len(doc),
        })
    return flags


def _bootstrap_diff(diffs, *, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    return sum(diffs) / n, samples[int(0.025 * B)], samples[int(0.975 * B)]


def gate_arm(base_pq, prf_pq, gate_mask):
    """Return per-query hits where gate_mask[i]=True picks PRF, else baseline."""
    out = []
    for i, (b, p) in enumerate(zip(base_pq, prf_pq)):
        chosen = p if gate_mask[i] else b
        out.append((chosen["category"], int(chosen["hit_at_1"])))
    return out


def report(name, base, gated, *, by_cat=True):
    diffs = [g[1] - b[1] for b, g in zip(base, gated)]
    m, lo, hi = _bootstrap_diff(diffs)
    sig = "*" if lo * hi > 0 else "ns"
    print(f"  {name:24s} n={len(diffs):4d}  Δ={m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    if by_cat:
        per_cat = {}
        for (cb, _vb), (cg, _vg), d in zip(base, gated, diffs):
            per_cat.setdefault(cb, []).append(d)
        for c in sorted(per_cat):
            ds = per_cat[c]
            m, lo, hi = _bootstrap_diff(ds)
            sig = "*" if lo * hi > 0 else "ns"
            print(f"      cat{c} n={len(ds):4d}  Δ={m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")


def main() -> int:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("loading per_query streams...")
    base_pq = [{"category": r["category"], "hit_at_1": r["hit_at_1"], "sample_id": r["sample_id"]}
               for r in load_arm("baseline")]
    prf_pq = [{"category": r["category"], "hit_at_1": r["hit_at_1"], "sample_id": r["sample_id"]}
              for r in load_arm("prf_spacy")]
    assert len(base_pq) == len(prf_pq) == 1978, (len(base_pq), len(prf_pq))

    flags = classify_questions(nlp)
    assert len(flags) == len(base_pq), (len(flags), len(base_pq))
    # sanity: sample_id alignment
    mismatches = sum(1 for f, b in zip(flags, base_pq) if f["sample_id"] != b["sample_id"])
    print(f"  sample_id alignment mismatches: {mismatches}/{len(flags)} (should be 0)")

    nv = sum(1 for f in flags if f["has_verb_outside_ent"])
    nt = sum(1 for f in flags if f["has_temp_outside_ent"])
    nvt = sum(1 for f in flags if f["has_verb_outside_ent"] or f["has_temp_outside_ent"])
    print(f"\n  fire-rates: verb_only={nv}/{len(flags)} "
          f"({100*nv/len(flags):.1f}%), temp_only={nt}/{len(flags)} "
          f"({100*nt/len(flags):.1f}%), verb_or_temp={nvt}/{len(flags)} "
          f"({100*nvt/len(flags):.1f}%)")
    # category-conditional fire rate
    by_cat_total = Counter(f["category"] for f in flags)
    by_cat_fire = Counter(f["category"] for f in flags if f["has_verb_outside_ent"] or f["has_temp_outside_ent"])
    for c in sorted(by_cat_total):
        tot = by_cat_total[c]
        fir = by_cat_fire[c]
        print(f"      cat{c}: fire {fir}/{tot} ({100*fir/tot:.1f}%)")

    base = [(r["category"], int(r["hit_at_1"])) for r in base_pq]
    prf = [(r["category"], int(r["hit_at_1"])) for r in prf_pq]

    print("\n=== Reference arms vs baseline ===")
    report("prf_spacy (always-on)", base, prf)

    for name, mask in [
        ("gate=verb_only", [f["has_verb_outside_ent"] for f in flags]),
        ("gate=temp_only", [f["has_temp_outside_ent"] for f in flags]),
        ("gate=verb_or_temp", [f["has_verb_outside_ent"] or f["has_temp_outside_ent"] for f in flags]),
    ]:
        gated = gate_arm(base_pq, prf_pq, mask)
        print(f"\n=== {name} ===")
        report(name, base, gated)

    print("\n=== Sanity: gate=NEVER should equal baseline (Δ=0) ===")
    gated = gate_arm(base_pq, prf_pq, [False] * len(flags))
    report("gate=never", base, gated, by_cat=False)
    print("=== Sanity: gate=ALWAYS should equal prf_spacy ===")
    gated = gate_arm(base_pq, prf_pq, [True] * len(flags))
    report("gate=always", base, gated, by_cat=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

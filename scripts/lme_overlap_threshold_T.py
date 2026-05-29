"""Stratify LongMemEval Δhit@1(ST − BM25/Hash) by query↔gold-session lexical overlap.

Tests TODO-RESEARCH §C empirically: "vector retrieval is paraphrase-robustness;
it pays exactly when query–document lexical overlap is below threshold T."

Prediction: Δhit@1(ST_vw=0.3 − Hash_vw=0.3) should be ≥0 in every overlap bin
and grow monotonically as overlap drops. A flat curve falsifies the framing
(replicates the synthetic-paraphrase null on a real corpus).

Inputs:
  - data/longmemeval/longmemeval_s.json (500 instances)
  - bench/results/lme_n100_hash_vw0.3_baseline.json   (BM25/Hash, vw=0.3)
  - bench/results/lme_n100_st_vw0.3_baseline.json     (ST MiniLM, vw=0.3)

Output:
  - bench/results/lme_overlap_threshold_T.json
    {
      "n": 100, "stop_words": [...], "bins": [{"q": .., "n":, "overlap_lo":, "overlap_hi":,
                "hit1_hash":, "hit1_st":, "delta":, "ci": [lo, hi]}, ...],
      "spearman_overlap_vs_delta": ...,
      "global": {"hit1_hash": .., "hit1_st": .., "delta": .., "ci": [..]}
    }

Method: for each question, define overlap = Jaccard(tok(question) \ stop,
tok(concat(gold_sessions)) \ stop). Bin by quartile (n≈25 each). Per bin:
report hit@1 for each arm, paired-bootstrap Δ with B=10000 seed=42.

CIs are paired across questions because both arms scored the same questions.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "longmemeval" / "longmemeval_s.json"
HASH_R = REPO / "bench" / "results" / "lme_n100_hash_vw0.3_baseline.json"
ST_R = REPO / "bench" / "results" / "lme_n100_st_vw0.3_baseline.json"
OUT = REPO / "bench" / "results" / "lme_overlap_threshold_T.json"

# Minimal English stop list — keeping it short avoids overfitting the bin
# structure; the substantive content tokens are what matter.
STOP = set("""
a an the of in on at to for and or but if then so is are was were be been being
have has had do does did i me my we our you your he she it they them this that
these those there here what when where which who whom whose why how
do can could should would may might will shall not no nor as by with from
about into over under between against during through above below up down out off
""".split())

TOK = re.compile(r"[A-Za-z]{2,}")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOK.findall(text or "")} - STOP


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def gold_text(ex: dict) -> str:
    gold_ids = set(ex.get("answer_session_ids", []))
    sids = ex.get("haystack_session_ids", [])
    sessions = ex.get("haystack_sessions", [])
    chunks: list[str] = []
    for sid, sess in zip(sids, sessions):
        if sid in gold_ids:
            for turn in sess:
                if isinstance(turn, dict) and "content" in turn:
                    chunks.append(turn["content"])
    return " ".join(chunks)


def paired_bootstrap(deltas: list[float], B: int = 10000, seed: int = 42, alpha: float = 0.05):
    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    samples = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(deltas[i] for i in idxs) / n)
    samples.sort()
    lo = samples[int(B * alpha / 2)]
    hi = samples[int(B * (1 - alpha / 2)) - 1]
    return (sum(deltas) / n, lo, hi)


def main() -> int:
    raw = json.loads(DATA.read_text())
    by_qid = {ex["question_id"]: ex for ex in raw}

    hash_pq = {x["question_id"]: x for x in json.loads(HASH_R.read_text())["per_instance"]}
    st_pq = {x["question_id"]: x for x in json.loads(ST_R.read_text())["per_instance"]}

    qids = [q for q in hash_pq if q in st_pq and q in by_qid]
    rows = []
    for qid in qids:
        ex = by_qid[qid]
        q_tok = tokens(ex["question"])
        g_tok = tokens(gold_text(ex))
        ov = jaccard(q_tok, g_tok)
        rows.append({
            "qid": qid,
            "qtype": ex["question_type"],
            "overlap": ov,
            "hit1_hash": int(hash_pq[qid]["hit_at_1"]),
            "hit1_st": int(st_pq[qid]["hit_at_1"]),
            "delta": int(st_pq[qid]["hit_at_1"]) - int(hash_pq[qid]["hit_at_1"]),
        })

    rows.sort(key=lambda r: r["overlap"])
    n = len(rows)
    Q = 4
    bins = []
    for q in range(Q):
        a = (n * q) // Q
        b = (n * (q + 1)) // Q
        chunk = rows[a:b]
        if not chunk:
            continue
        deltas = [r["delta"] for r in chunk]
        m, lo, hi = paired_bootstrap(deltas)
        bins.append({
            "q": q + 1,
            "n": len(chunk),
            "overlap_lo": round(chunk[0]["overlap"], 4),
            "overlap_hi": round(chunk[-1]["overlap"], 4),
            "overlap_mean": round(mean(r["overlap"] for r in chunk), 4),
            "hit1_hash": round(mean(r["hit1_hash"] for r in chunk), 4),
            "hit1_st": round(mean(r["hit1_st"] for r in chunk), 4),
            "delta": round(m, 4),
            "ci": [round(lo, 4), round(hi, 4)],
        })

    # Global Spearman ρ between overlap and delta (rough, no library).
    def spearman(xs, ys):
        def ranks(arr):
            order = sorted(range(len(arr)), key=lambda i: arr[i])
            r = [0.0] * len(arr)
            i = 0
            while i < len(arr):
                j = i
                while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        rx, ry = ranks(xs), ranks(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = sum((a - mx) ** 2 for a in rx) ** 0.5
        dy = sum((b - my) ** 2 for b in ry) ** 0.5
        return num / (dx * dy) if dx and dy else 0.0

    rho = spearman([r["overlap"] for r in rows], [r["delta"] for r in rows])

    deltas_global = [r["delta"] for r in rows]
    g_m, g_lo, g_hi = paired_bootstrap(deltas_global)

    out = {
        "n": n,
        "B": 10000,
        "alpha": 0.05,
        "seed": 42,
        "embedders": {"lex": "HashTrigram-256 vw=0.3", "vec": "ST MiniLM-384 vw=0.3"},
        "stop_words_count": len(STOP),
        "bins": bins,
        "spearman_overlap_vs_delta": round(rho, 4),
        "global": {
            "hit1_hash": round(mean(r["hit1_hash"] for r in rows), 4),
            "hit1_st": round(mean(r["hit1_st"] for r in rows), 4),
            "delta": round(g_m, 4),
            "ci": [round(g_lo, 4), round(g_hi, 4)],
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

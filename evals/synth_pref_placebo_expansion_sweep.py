"""§D15c-mech-3 — placebo query-expansion falsifier.

Hypothesis under test (the remaining live one after §D15c-mech-1/2 and
§4.15g all falsified the entity-content angles): the PRF Δh@1 < 0
regression on the synthetic preference corpus is driven by **BM25 score
dilution from query-length expansion itself**, not by *which* tokens
PRF chooses. If true, *any* 3-token expansion — including content-free
stopwords — should reproduce the regression magnitude.

Design (paired baseline; single seed; same corpus per arm):

  arm                          | how the query is built
  -----------------------------+------------------------------------------
  baseline                     | original query (no expansion)
  prf_real                     | gated_pref via engine (current PRF)
  placebo_stopword             | append fixed stopwords: "the for and"
  placebo_high_df              | append 3 most corpus-common tokens
                               |   (computed from the corpus itself)
  placebo_low_df               | append 3 corpus-rare tokens
  placebo_off_topic_entity     | append 3 random entity-shaped tokens
                               |   from OTHER queries' anchors

For arms 3–6, expansion is done by string-concatenation against the
**baseline** engine (no PRF config). This isolates the BM25-dilution
effect from the PRF mining logic. Predicted outcome map:

  Δh@1(real)        ≈ Δh@1(stopword)   → length-dilution mechanism
  Δh@1(real)        ≈ Δh@1(high_df)    → length+commonness mechanism
  Δh@1(low_df) ≈ 0  (or positive)      → rarity is what matters
  Δh@1(off_topic)   ≪ Δh@1(real)       → real PRF picks better than rand

Usage::

    python -m evals.synth_pref_placebo_expansion_sweep \\
        --n-facts 240 --seed 42 --k 10 --anchor-tokens 3 \\
        --out evals/results/synth_pref_placebo_expansion_sweep.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import tempfile
import time
from pathlib import Path

from engram import Engram, Config

from .metrics import find_match_rank
from .synthetic import generate_preference_dataset
from evals.io_utils import atomic_write_json


_TOK = re.compile(r"[A-Za-z0-9]+")


def _baseline_cfg(path: str) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    return cfg


def _gated_pref_cfg(path: str) -> Config:
    cfg = _baseline_cfg(path)
    cfg.retrieval.query_expansion_min_dominance = 0.3
    cfg.retrieval.query_expansion_type_allow = frozenset(
        {"single-session-preference"}
    )
    return cfg


def _corpus_token_df(memories) -> dict[str, int]:
    df: dict[str, int] = collections.Counter()
    for content, _meta in memories:
        toks = {t.lower() for t in _TOK.findall(content) if len(t) > 1}
        for t in toks:
            df[t] += 1
    return df


def _pick_high_df(df: dict[str, int], n: int = 3) -> list[str]:
    # Most-corpus-common tokens.
    return [t for t, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def _pick_low_df(df: dict[str, int], n: int = 3, rng=None) -> list[str]:
    # Tokens appearing in exactly 1 doc (corpus-rare).
    rares = [t for t, c in df.items() if c == 1]
    rng = rng or random
    rng.shuffle(rares)
    return rares[:n]


def _other_query_anchors(ds, this_qi: int, n: int = 3, rng=None) -> list[str]:
    rng = rng or random
    pool = []
    for j, q in enumerate(ds.queries):
        if j == this_qi:
            continue
        if q.expected_substrings:
            pool.append(q.expected_substrings[0])
    rng.shuffle(pool)
    return pool[:n]


def _run_arm(arm_name: str, ds, k: int, *, expand_fn=None, use_prf: bool = False):
    """Run a single arm.

    expand_fn(qi, query) -> str: pre-expanded query (for placebo arms).
    use_prf: when True, use gated_pref engine config (for prf_real arm).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _gated_pref_cfg(tmp) if use_prf else _baseline_cfg(tmp)
        eng = Engram(config=cfg)
        try:
            t0 = time.monotonic()
            for content, _meta in ds.memories:
                eng.remember(content)
            ingest_s = time.monotonic() - t0
            h1, hk, rr = [], [], []
            for qi, q in enumerate(ds.queries):
                qtxt = expand_fn(qi, q.text) if expand_fn else q.text
                results = eng.recall(qtxt, limit=k)
                rank = find_match_rank(results, q.expected_substrings)
                h1.append(1 if (rank is not None and rank < 1) else 0)
                hk.append(1 if (rank is not None and rank < k) else 0)
                rr.append(0.0 if rank is None else 1.0 / (rank + 1))
            return h1, hk, rr, ingest_s
        finally:
            eng.close()


def _delta(treat: list[int], base: list[int]):
    diffs = [a - b for a, b in zip(treat, base)]
    delta = statistics.mean(diffs) if diffs else 0.0
    se = (
        (statistics.pstdev(diffs) / (len(diffs) ** 0.5))
        if len(diffs) > 1 else 0.0
    )
    return round(delta, 4), round(se, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=240)
    p.add_argument("--distractors-per-fact", type=int, default=6)
    p.add_argument("--hard-distractors-per-fact", type=int, default=3)
    p.add_argument("--anchor-tokens", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    print(f"[placebo] building corpus (n={args.n_facts}, anchor_tokens={args.anchor_tokens})...", flush=True)
    ds = generate_preference_dataset(
        n_facts=args.n_facts,
        distractors_per_fact=args.distractors_per_fact,
        hard_distractors_per_fact=args.hard_distractors_per_fact,
        seed=args.seed,
        answer_anchor_tokens=args.anchor_tokens,
    )

    df = _corpus_token_df(ds.memories)
    high_df_toks = _pick_high_df(df, n=3)
    _ = random.Random(args.seed + 1)
    print(
        f"[placebo] |memories|={len(ds.memories)} |queries|={len(ds.queries)} "
        f"|vocab|={len(df)} top3_DF={high_df_toks}",
        flush=True,
    )

    # Pre-pick low-DF and off-topic expansions per-query (deterministic).
    low_df_per_q: list[list[str]] = []
    off_topic_per_q: list[list[str]] = []
    rng_lo = random.Random(args.seed + 2)
    rng_ot = random.Random(args.seed + 3)
    for qi in range(len(ds.queries)):
        low_df_per_q.append(_pick_low_df(df, n=3, rng=rng_lo))
        off_topic_per_q.append(_other_query_anchors(ds, qi, n=3, rng=rng_ot))

    arms: list[tuple[str, dict]] = [
        ("baseline",                {"expand_fn": None,                                              "use_prf": False}),
        ("prf_real",                {"expand_fn": None,                                              "use_prf": True}),
        ("placebo_stopword",        {"expand_fn": lambda qi, q: q + " the for and",                  "use_prf": False}),
        ("placebo_high_df",         {"expand_fn": lambda qi, q: q + " " + " ".join(high_df_toks),    "use_prf": False}),
        ("placebo_low_df",          {"expand_fn": lambda qi, q: q + " " + " ".join(low_df_per_q[qi]),"use_prf": False}),
        ("placebo_off_topic_entity",{"expand_fn": lambda qi, q: q + " " + " ".join(off_topic_per_q[qi]), "use_prf": False}),
    ]

    raw: dict[str, dict] = {}
    for name, kwargs in arms:
        print(f"[placebo] arm={name} running...", flush=True)
        h1, hk, rr, ingest_s = _run_arm(name, ds, args.k, **kwargs)
        raw[name] = {
            "h1": h1, "hk": hk, "rr": rr, "ingest_s": round(ingest_s, 2),
            "h1_mean": round(statistics.mean(h1), 4),
            "hk_mean": round(statistics.mean(hk), 4),
            "mrr_mean": round(statistics.mean(rr), 4),
        }
        print(
            f"[placebo] arm={name} h1={raw[name]['h1_mean']:.4f} "
            f"hk={raw[name]['hk_mean']:.4f} mrr={raw[name]['mrr_mean']:.4f} "
            f"ingest={ingest_s:.1f}s",
            flush=True,
        )

    base_h1 = raw["baseline"]["h1"]
    deltas = {}
    for name in raw:
        if name == "baseline":
            continue
        d, se = _delta(raw[name]["h1"], base_h1)
        deltas[name] = {"delta_h1": d, "delta_se": se}

    out = {
        "n_facts": args.n_facts,
        "seed": args.seed,
        "k": args.k,
        "anchor_tokens": args.anchor_tokens,
        "high_df_top3": high_df_toks,
        "arms": {
            n: {kk: vv for kk, vv in raw[n].items() if kk not in ("h1", "hk", "rr")}
            for n in raw
        },
        "deltas_vs_baseline": deltas,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

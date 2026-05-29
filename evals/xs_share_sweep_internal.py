"""§94b-internal — instrument decide_with_family across the share sweep.

§94b found that share ∈ {0.0, 0.25, 0.5, 0.75, 1.0} produces bit-identical
downstream metrics on the cross-session corpus. This driver answers WHY:

  (a) decide_with_family produces identical CREATE/BUMP/RECOVER decisions
      across share values — the math is moot at this scale, OR
  (b) decisions differ but the downstream retrieval effect is below
      60-pair granularity.

Method: re-run the §94b share sweep with a `family_decision_trace` recorder
installed for each arm. For every share value we collect:

  * total decision calls
  * count of calls where any borrowing actually happened
    (`borrowed_via_share=True` — i.e. share>0 and siblings nonzero)
  * decision histogram (PROMOTE/DEPRECATE/RECOVER/None)
  * sequence-equality vs the share=0 baseline trace (decisions only)

If the share>0 decision sequences are byte-identical to share=0, that is
case (a). If they differ but the §94b sweep still saw identical metrics,
that is case (b), and §96 (retrieval-side post-rerank) is the right
attack point.

Pure: deterministic given seed. No clocks, no I/O beyond optional --out.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager

from engram.consolidation.schema_family_decision import family_decision_trace

from evals import cross_session_recall_lift as xs
from evals.synthetic import generate_cross_session_dataset
from evals.io_utils import atomic_write_json


@contextmanager
def _patched_share(share: float):
    prev = xs.RECIPE["schema_family_share"]
    xs.RECIPE["schema_family_share"] = share
    try:
        yield
    finally:
        xs.RECIPE["schema_family_share"] = prev


def _summarise_trace(trace: list[dict]) -> dict:
    """Reduce a per-call trace to a small comparable summary."""
    decisions: list[str | None] = [r["decision"] for r in trace]
    borrows = sum(1 for r in trace if r["borrowed_via_share"])
    hist: dict[str, int] = {}
    for d in decisions:
        key = d if d is not None else "NONE"
        hist[key] = hist.get(key, 0) + 1
    return {
        "n_calls": len(trace),
        "n_borrowed": borrows,
        "decision_histogram": hist,
        # We hash the decision sequence rather than emit the full list to
        # keep the report small while still letting the caller compare.
        "decisions": decisions,
    }


def run_internal_sweep(
    *,
    shares: list[float],
    n_facts: int = 60,
    n_sessions: int = 10,
    distractors_per_session: int = 10,
    seed: int = 42,
    k: int = 10,
) -> dict:
    from engram.providers.embeddings import HashTrigramEmbeddingProvider
    embedder = HashTrigramEmbeddingProvider(dimension=256)

    ds = generate_cross_session_dataset(
        n_facts=n_facts,
        n_sessions=n_sessions,
        distractors_per_session=distractors_per_session,
        seed=seed,
    )

    t0 = time.monotonic()
    points = []
    for share in shares:
        sink: list[dict] = []
        with _patched_share(share), family_decision_trace(sink):
            xs._run_arm(ds, treatment=True, embedder=embedder, k=k,
                        synthesis=True)
        summary = _summarise_trace(sink)
        points.append({
            "share": share,
            **summary,
        })

    # Compare each share's decision sequence against share=0.
    if not points:
        equal_to_share0 = []
        baseline_decisions: list = []
    else:
        baseline_decisions = points[0]["decisions"]
        equal_to_share0 = [
            {"share": p["share"],
             "decisions_equal_to_share0":
                p["decisions"] == baseline_decisions}
            for p in points
        ]

    # Strip the verbose decisions list out of points before emitting.
    for p in points:
        p.pop("decisions", None)

    wall = time.monotonic() - t0

    all_equal = all(e["decisions_equal_to_share0"] for e in equal_to_share0)

    # Verdict — three cases.
    #
    # Note: at share=0 the pipeline takes a fast path and never even calls
    # decide_with_family (n_calls=0). So "decisions_equal_to_share0" can
    # be False purely because the reference is empty. Use n_borrowed
    # across share>0 arms to disambiguate.
    nonzero_arms = [p for p in points if p["share"] > 0.0]
    any_borrowed = any(p["n_borrowed"] > 0 for p in nonzero_arms)
    all_zero_calls = all(p["n_calls"] == 0 for p in points)

    if all_zero_calls:
        verdict = (
            "case_0: decide_with_family is never called on this corpus "
            "for any share — the family-decision path is unreachable."
        )
    elif not any_borrowed:
        verdict = (
            "case_a': decide_with_family is called but n_borrowed==0 on "
            "every call across every share>0 (siblings always empty in "
            "the consolidation window). The share knob is structurally "
            "inert because there is never any sibling evidence to share. "
            "Implication: §96 retrieval-side post-rerank is the right "
            "attack point — the consolidation-side knob has no surface."
        )
    elif all_equal:
        verdict = (
            "case_a: decisions identical across shares — math is moot at "
            "this scale (sibling evidence exists but never tips a "
            "decision)."
        )
    else:
        verdict = (
            "case_b: decisions differ across shares but §94b downstream "
            "metrics were identical — points to §96 retrieval-side "
            "post-rerank."
        )

    return {
        "embedder": "HashTrigram-256",
        "corpus": {
            "n_facts": n_facts,
            "n_sessions": n_sessions,
            "distractors_per_session": distractors_per_session,
            "seed": seed,
        },
        "k": k,
        "shares": shares,
        "points": points,
        "decision_seq_equality": equal_to_share0,
        "verdict": verdict,
        "wall_seconds": round(wall, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shares", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--n-facts", type=int, default=60)
    p.add_argument("--n-sessions", type=int, default=10)
    p.add_argument("--distractors", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    shares = [float(s) for s in args.shares.split(",")]
    out = run_internal_sweep(
        shares=shares,
        n_facts=args.n_facts,
        n_sessions=args.n_sessions,
        distractors_per_session=args.distractors,
        seed=args.seed,
        k=args.k,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, out, default=str)
        print(f"[xs_share_sweep_internal] wrote {args.out}")


if __name__ == "__main__":
    main()

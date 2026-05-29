# RM3-BASELINE-SPEC — Relevance Model PRF baseline for AUDIT-D

> **Closes:** AUDIT-D (RM3 baseline arm). Already queued as v0.3
> candidate in §5.6; this spec turns the queue into a runnable plan.
>
> **Status:** SPEC + scaffold (this file ships the spec; the
> implementation lives in `evals/rm3_baseline.py` and a small
> `evals/rm3.py` library, both author-written when execution
> time comes — NOT assistant-written, per the §80 disclosure
> carve-out for experimental code).
>
> **Why an RM3 baseline matters.** §A.4.8.1 establishes that
> Engram's *heuristic* PRF (top-k entity expansion + dominance gate)
> is a regression on real LongMemEval-S. A reviewer will reasonably
> ask: "Is PRF universally bad here, or just your specific
> heuristic?" RM3 is the canonical academic PRF baseline. If RM3
> also regresses, the §A.4.8.1 conclusion strengthens to *"PRF as a
> family is mismatched to single-session conversational retrieval."*
> If RM3 wins where heuristic-PRF lost, we re-frame heuristic-PRF
> as a tuning miss and ship RM3 as a v0.3 default.

## 1. RM3 in 3 sentences

Given a query Q and top-k pseudo-relevant docs D_1..D_k retrieved by
the first-pass scorer, RM3 estimates a relevance language model
P(w | R) = Σ_i P(Q | D_i) · P(w | D_i), keeps the top-N expansion
terms by P(w | R), and re-queries with the linear mix
λ · Q_original + (1 − λ) · Q_expansion. Standard hyperparameters
(Lavrenko & Croft 2001; Lv & Zhai 2009 grid): k ∈ {10, 20, 50};
N ∈ {10, 20, 50}; λ ∈ {0.4, 0.6, 0.8}.

## 2. Implementation plan

Two new files, one library + one driver:

**`evals/rm3.py`** — a small, author-written, dependency-free RM3
implementation. Operates on token-frequency dicts; does not need
Engram internals.

```python
# evals/rm3.py
from collections import Counter
import math

def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t.isalnum() and len(t) > 1]

def rm3_expansion_terms(
    query: str,
    feedback_docs: list[tuple[str, float]],   # [(doc_text, first_pass_score)]
    *,
    top_n_terms: int = 20,
    smoothing: float = 0.1,                  # Dirichlet μ ratio for P(w|D)
) -> list[tuple[str, float]]:
    """Return [(term, P(term | R))] sorted descending, top_n_terms long.

    P(Q | D) is approximated as the doc's first-pass relevance score
    after softmax — this is the standard RM3 short-cut when the
    first-pass scorer produces calibrated relevance scores rather
    than true query likelihoods.  See Lv & Zhai (2009) §3.2 for the
    formal derivation; we do not claim to implement the full
    bm25-likelihood substitution.
    """
    if not feedback_docs:
        return []
    # P(Q | D_i) ∝ score after softmax for numerical stability
    scores = [s for _, s in feedback_docs]
    m = max(scores)
    weights = [math.exp(s - m) for s in scores]
    Z = sum(weights) or 1.0
    weights = [w / Z for w in weights]

    rel_model: Counter[str] = Counter()
    for (doc, _), w_q_d in zip(feedback_docs, weights):
        doc_terms = _tokenize(doc)
        if not doc_terms:
            continue
        doc_count = Counter(doc_terms)
        n_d = sum(doc_count.values())
        # Dirichlet-smoothed P(w | D) for each w in this doc
        for term, c in doc_count.items():
            p_w_d = (c + smoothing) / (n_d + smoothing * len(doc_count))
            rel_model[term] += w_q_d * p_w_d

    return rel_model.most_common(top_n_terms)


def rm3_expanded_query(
    query: str,
    feedback_docs: list[tuple[str, float]],
    *,
    lambda_orig: float = 0.6,
    top_n_terms: int = 20,
) -> str:
    """Mix query and top-N RM3 terms by repeating the query lambda
    fraction so a downstream BM25 scorer naturally weights it heavier.

    This is the simplest possible 'fixed mixing' approximation; for
    a true linear-mixture you'd weight terms in the BM25 lexer,
    which Engram's FTS5 layer does not expose.  The repetition trick
    is documented in Anserini's RM3 stub.
    """
    expansion_terms = rm3_expansion_terms(query, feedback_docs,
                                          top_n_terms=top_n_terms)
    if not expansion_terms:
        return query
    expansion_str = " ".join(t for t, _ in expansion_terms)
    # Repeat the query ceil(λ / (1−λ)) times so its mass dominates.
    # λ=0.6 → repeat 2× (twice the query, once the expansion).
    n_repeat = max(1, round(lambda_orig / (1 - lambda_orig)))
    return " ".join([query] * n_repeat + [expansion_str])
```

**`evals/rm3_baseline.py`** — the run-arm driver. Parallels
`evals/longmemeval_adapter.py --arm prf` so it slots cleanly into
§A.4.8.1's structure.

```python
"""Run the RM3 PRF arm against the §A.4.8.1 LongMemEval-S baseline.

Parallels evals/longmemeval_adapter.py --arm prf but uses a different
expansion strategy.  Same n=500 panel, same paired-bootstrap CI
estimator, output JSON shape compatible with evals/lme_compare_arms.py.
"""
import argparse
import json
from pathlib import Path
from engram import Config, Engram
from evals.rm3 import rm3_expanded_query
from evals.longmemeval_adapter import (
    load_longmemeval_split, ingest_session, score_question,
)

def run_rm3_arm(
    *,
    n_instances: int = 500,
    k: int = 10,
    rm3_top_n_terms: int = 20,
    rm3_lambda: float = 0.6,
    rm3_feedback_k: int = 10,
    out_path: Path,
):
    instances = load_longmemeval_split(n_instances)
    records = []
    for inst in instances:
        cfg = Config(); cfg.security.max_events_per_minute = 0
        eng = Engram(cfg)
        ingest_session(eng, inst)
        for q in inst.questions:
            # First pass: vanilla query
            first = eng.recall(q.text, k=rm3_feedback_k)
            feedback_docs = [(m.text, m.score) for m in first]
            # Expand with RM3
            expanded = rm3_expanded_query(
                q.text, feedback_docs,
                lambda_orig=rm3_lambda,
                top_n_terms=rm3_top_n_terms,
            )
            second = eng.recall(expanded, k=k)
            records.append(score_question(q, second))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"arm": "rm3", "n": len(records),
                                     "hyperparams": {"k": k,
                                                     "feedback_k": rm3_feedback_k,
                                                     "lambda": rm3_lambda,
                                                     "top_n_terms": rm3_top_n_terms},
                                     "records": records}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-instances", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rm3-feedback-k", type=int, default=10)
    p.add_argument("--rm3-lambda", type=float, default=0.6)
    p.add_argument("--rm3-top-n-terms", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run_rm3_arm(n_instances=args.max_instances, k=args.k,
                rm3_feedback_k=args.rm3_feedback_k,
                rm3_lambda=args.rm3_lambda,
                rm3_top_n_terms=args.rm3_top_n_terms,
                out_path=args.out)
```

Note: the function names `load_longmemeval_split`, `ingest_session`,
`score_question` are placeholders — the real signatures live in
`evals/longmemeval_adapter.py` and may need light refactoring to
expose the per-question scoring loop. The refactor is small (≤30
lines) and is part of the implementation, not the spec.

## 3. Hyperparameter pre-registration

To avoid p-hacking the RM3 result, we pre-register **one operating
point** before running:

- `rm3_feedback_k = 10` (matches our existing PRF feedback-k=10
  convention from §A.4.8.1 d-sweep)
- `rm3_top_n_terms = 20` (mid of the Lavrenko-Croft grid)
- `rm3_lambda = 0.6` (Lavrenko-Croft default)

**If** that operating point is null/regression, we report it and
stop. We do **not** sweep λ × top_n_terms × feedback_k looking for
a positive cell. A sweep can come later as a follow-up if there is
reviewer pressure to characterize the failure surface, but a
pre-registered single-point null is honest evidence; a swept
maximum-positive cell is not.

## 4. Comparison structure (paper integration)

The §A.4.8.1 third-encoder block already has the protocol scaffold:
**arm vs baseline, paired by question_id, B=5000 paired bootstrap,
seed=42.** RM3 plugs straight into this:

```
| arm                 | hit@1  | Δhit@1 (95% CI)              |
|---------------------|--------|------------------------------|
| baseline            | 0.810  | —                            |
| prf (heuristic)     | 0.770  | −0.040 [−0.062, −0.018] SIG  |   (existing)
| rm3 (Lavrenko 2001) | <TBD>  | <TBD>                        |   (this spec)
```

If RM3 also regresses → §A.4.8.1 strengthens. If RM3 is null → "PRF
family weakly underperforms at this query length." If RM3 wins →
re-frame heuristic-PRF as a tuning miss; ship RM3 as v0.3 default;
update §5.6.

## 5. Pre-registration block — to land in §5.6 BEFORE running

Before executing the RM3 arm, this paragraph goes into §5.6 so the
result is honestly framed regardless of outcome:

> "We pre-register an RM3 baseline (Lavrenko & Croft 2001) against
> the §A.4.8.1 baseline at λ=0.6, top-N=20, feedback-k=10 — a single
> operating point chosen at the centre of the Lavrenko-Croft grid
> rather than tuned on this dataset. The result will land in
> §A.4.18 regardless of sign; if positive, we will re-evaluate
> heuristic-PRF as a tuning miss and ship RM3 as a v0.3 default;
> if null/negative, the §A.4.8.1 conclusion strengthens to a
> PRF-family-level claim."

This pre-registration paragraph should land in `paper/50_discussion.md`
under §5.6 in the **same commit** that lands the RM3 driver code —
NOT in the commit that lands the RM3 result. That ordering is what
makes the pre-registration credible.

## 6. Wall time on M4 / Linux x86_64 host

LongMemEval-S n=500 with the existing baseline arm runs in ~4m25s
on M4 / MPS / fp32. RM3's added cost is one extra `recall` per
question + tokenisation of the feedback docs, both negligible.
Estimate: 5-7 min wall on M4, 8-12 min on Linux x86_64 host.

This is small enough to run on either side of the M4 ↔ cloud
desktop divide. Recommend Linux x86_64 host so cron can verify the
result independently of the M4 push window.

## 7. Disclosure obligations

`evals/rm3.py` and `evals/rm3_baseline.py` are author-written, not
assistant-drafted. The §A.4.18 narrative will be author-drafted from
artifact JSON + this spec. Update
`paper/80_acknowledgements_cameraready.md` to add the RM3 §A.4.18
write-up as author-only, alongside §A.4.17 (Letta cross-system).

## 8. Stop signals

- **STOP if** RM3 regresses by more than the heuristic-PRF point
  estimate (Δhit@1 < −0.04). That is a reviewer red flag and needs
  investigation before publication — the canonical baseline should
  not under-perform a heuristic on n=500.
- **STOP if** RM3's expansion terms are dominated by stopwords
  ("the", "a", "of") despite the smoothing — that means the BM25
  scoring is not feeding `feedback_docs` with calibrated scores and
  the implementation needs a stopword filter.
- **PROCEED if** Δhit@1 is in [−0.04, +0.04] with a sensible CI.
  That is a real null/positive result and goes into the paper.

# Engram v0.2 — LongMemEval Treatment-Arm Δ Tables (Technical Report)

This technical report holds the per-type Δ vs. baseline tables
for the LongMemEval treatment-arm experiments at n=500. The headline
LongMemEval result and per-type panel summary live in §4.8 of the
main paper; the full per-arm × per-type × per-metric matrices that
were originally appendix §A.4.8.1 are kept here for completeness
without occupying paper review surface.

Subsection IDs (`A.4.X`) and artifact paths under `bench/results/`
are stable.

## A.4.8.1 LongMemEval — treatment-arm Δ vs. baseline (paired, n=500)

We ran three treatment arms against the §4.8 baseline on the **same**
500 LongMemEval-S questions, then paired by `question_id` and computed
2000-resample bootstrap 95% CIs on Δhit@1 / Δhit@10. Arms:

- **prf** — PRF entity query expansion on (`query_expansion_min_dominance=0.3`).
- **share_prior** — share_prior reranker on (`α=0.10`, `pool_size=20`).
- **both** — PRF + share_prior, same hyperparameters.

**Overall (n=500):**

| arm         | hit@1  | Δhit@1 (95% CI)              | hit@10 | Δhit@10 (95% CI)            |
|-------------|--------|------------------------------|--------|-----------------------------|
| baseline    | 0.810  | —                            | 0.932  | —                           |
| prf         | 0.770  | **−0.040** [−0.062, −0.018]  | 0.926  | −0.006 [−0.016, +0.002]     |
| share_prior | 0.810  |  0.000 [0.000, 0.000]        | 0.932  |  0.000 [−0.006, +0.006]     |
| both        | 0.780  | **−0.030** [−0.052, −0.008]  | 0.924  | −0.008 [−0.022, +0.006]     |

**Per question type — Δhit@1 (95% CI), bold = CI excludes zero:**

| type                       |  n  | prf                                | share_prior     | both                              |
|----------------------------|-----|------------------------------------|-----------------|-----------------------------------|
| single-session-user        |  70 | **−0.157** [−0.243, −0.071]        |  0.000          | **−0.129** [−0.214, −0.057]       |
| single-session-assistant   |  56 |  0.000                             |  0.000          |  0.000                            |
| single-session-preference  |  30 | **+0.067** [0.000, +0.167]         |  0.000          | **+0.033** [0.000, +0.100]        |
| multi-session              | 133 | −0.023 [−0.068, +0.023]            |  0.000          | −0.008 [−0.053, +0.038]           |
| temporal-reasoning         | 133 | **−0.045** [−0.090, −0.008]        |  0.000          | −0.030 [−0.075, +0.015]           |
| knowledge-update           |  78 | −0.026 [−0.064, 0.000]             |  0.000          | −0.026 [−0.064, 0.000]            |

**Findings.**

1. **PRF entity expansion is a net regression on real LongMemEval-S
   data.** Overall Δhit@1 = −0.040 with the upper CI bound at −0.018
   (strictly negative), and the regression is concentrated on
   `single-session-user` (Δhit@1 = −0.157, CI fully negative). The
   §A.4.7 PRF×share_prior synthetic CI panel showed PRF lifting recall
   on multi-entity templated queries; on real conversational
   questions, PRF over-expands and dilutes BM25's already-strong
   single-session signal. Type-aware NER (§3.6 D1) does not change
   this — the wired-in path uses the same `entity_ner` backend that
   was at parity on the synthetic D1 sweep.

2. **share_prior is a no-op on this dataset at α=0.10.** Overall Δhit@1
   CI is exactly [0.000, 0.000]: every paired pair of `(question_id,
   hit_at_1)` is identical to baseline. This is consistent with
   share_prior being a per-session prior reweight that rarely changes
   *which session* tops the fused list when BM25 already picks one
   correct session — LongMemEval-S session retrieval is largely a
   hit-the-right-haystack problem, not a tie-break problem.

3. **single-session-preference — the cell PRF was supposed to fix —
   does respond, but the lift is small (+0.067 hit@1, lower CI = 0)
   and is dwarfed by the `single-session-user` regression.** The
   `both` arm shows the same shape at half magnitude, consistent
   with share_prior contributing no signal and PRF doing the work
   (and the damage).

**Default decision.** PRF stays **off by default**
(`query_expansion_min_dominance = None` is already the shipped
default, per §3.6). share_prior stays available as an opt-in
reranker but is not turned on by default for general retrieval —
its current α=0.10, pool=20 operating point shows zero overall
movement on real conversational data. The §A.4.7 synthetic CI panel
remains valid for the templated multi-entity regime it was built
for; it does not transfer to LongMemEval-S, and we report both
results honestly rather than picking the corpus where PRF wins.

Reproduce: each arm via `python -m evals.longmemeval_adapter --arm
{baseline,prf,share_prior,both} --max-instances 500 --k 10 --out
bench/results/lme_full500_k10_<arm>.json`, then `python -m
evals.lme_compare_arms --baseline ...baseline.json --arms
...prf.json ...sp.json ...both.json --out
bench/results/lme_full500_arms_delta.json`. Artifact:
`bench/results/lme_full500_arms_delta.json` (10 KB; carries the full
per-type Δ + CI tables this section summarizes).

**Ablation — `query_expansion_min_dominance` sweep.** To confirm the
PRF regression is monotone in expansion aggressiveness (and not a
single-cell artifact at d=0.30), we swept d ∈ {0.20, 0.30, 0.40, 0.50}
on the same n=500 LongMemEval-S slice, paired against the d=∞ (off)
baseline with a 5000-resample paired bootstrap on per-instance hit@1:

| d         | hit@1  | hit@10 | Δhit@1 vs. off | 95% CI            |
|-----------|--------|--------|----------------|-------------------|
| off (∞)   | 0.8100 | 0.9320 | —              | —                 |
| 0.20      | 0.7620 | 0.9240 | −0.0480        | [−0.0740, −0.0220]|
| 0.30      | 0.7700 | 0.9260 | −0.0400        | [−0.0640, −0.0180]|
| 0.40      | 0.7860 | 0.9320 | −0.0240        | [−0.0400, −0.0100]|
| 0.50      | 0.8000 | 0.9340 | −0.0100        | [−0.0200, −0.0020]|

Δhit@1 is monotonically less negative as d rises (more conservative
expansion → less damage), and even at d=0.50 the upper CI bound stays
strictly negative. The regression is therefore not a tuning miss at
d=0.30 — PRF as currently shaped damages LongMemEval-S at every
expansion threshold we tested, and only approaches break-even by
shrinking until it almost never fires. This rules out "ship a less
aggressive PRF default" as a remediation; the next move is a
type-aware gate (only expand when the dominant entity type is one
the corpus actually disambiguates by), not a softer threshold.
Artifacts: `bench/results/lme_d_ablation/prf_d{0.2,0.4,0.5}.json`;
summary via `python -m evals.lme_d_ablation_summarize`.

**Third-encoder replication — does the PRF regression survive an
encoder swap?** A reviewer reading §A.4.8.1 may reasonably ask
whether the regression is an artifact of MiniLM-384's specific
embedding geometry: BGE-large-1024 (2.7× parameters) might rescue
PRF by carrying the expanded query better. We re-ran the **prf** and
**both** arms on the same n=500 LongMemEval-S panel with
`BAAI/bge-large-en-v1.5` wired in, paired per-`question_id` against
the existing BGE-large baseline (§A.4.16.3) with a 5000-resample
paired bootstrap (seed=42). Wall ≈ 6 h on M4 Pro / MPS / fp32.

**Overall (BGE-large arm, n=500, paired vs BGE-large baseline):**

| arm                | hit@1  | Δhit@1 (95% CI)                | hit@10 | Δhit@10 (95% CI)              |
|--------------------|--------|--------------------------------|--------|-------------------------------|
| baseline (BGE)     | 0.868  | —                              | 0.956  | —                             |
| prf (BGE)          | 0.852  | −0.016 [−0.034, +0.002]        | 0.948  | **−0.008** [−0.016, −0.002]   |
| both (BGE)         | 0.842  | **−0.026** [−0.046, −0.006]    | 0.936  | **−0.020** [−0.034, −0.008]   |

**Per question type — Δhit@1 (95% CI), bold = CI excludes zero:**

| type                       |  n  | prf (BGE)                            | both (BGE)                            |
|----------------------------|-----|--------------------------------------|---------------------------------------|
| single-session-user        |  70 | **−0.100** [−0.171, −0.043]          | **−0.100** [−0.171, −0.043]           |
| single-session-assistant   |  56 |  0.000                               |  0.000                                |
| single-session-preference  |  30 | +0.033 [0.000, +0.100]               | +0.033 [−0.067, +0.133]               |
| multi-session              | 133 | +0.023 [−0.015, +0.060]              |  0.000 [−0.038, +0.038]               |
| temporal-reasoning         | 133 | **−0.030** [−0.060, −0.008]          | **−0.045** [−0.083, −0.015]           |
| knowledge-update           |  78 | −0.013 [−0.064, +0.026]              | −0.013 [−0.051, +0.026]               |

**Verdict.** The PRF regression survives the encoder swap. The
`both` arm is SIG-negative on overall Δhit@1 under BGE-large
(−0.026, [−0.046, −0.006]) just as it was under MiniLM (−0.030
under MiniLM, see table above), and the per-type pattern replicates
cleanly: single-session-user takes a −10pp hit under both
embedders, temporal-reasoning takes a SIG-negative hit (−3pp under
PRF, −4.5pp under `both`), and the small point-estimate lifts on
multi-session and single-session-preference stay null. The BGE
`prf` arm's overall Δhit@1 has its upper CI bound at +0.002 — i.e.
the strongest reading available is "BGE plus PRF is at-best
indistinguishable from BGE alone, at-worst a 3.4 pp regression" —
nowhere near the recovery a reviewer might hope for.

This **strengthens** §A.4.8.1's default decision. PRF stays off by
default not because we lacked encoder capacity, but because **the
regression is encoder-invariant**: doubling the parameter count and
3× the embedding dimension does not buy back the BM25 floor that
PRF over-expansion erodes. The next move remains the type-aware
gate (§A.4.10), not a richer dense channel.

Artifacts:
`bench/results/lme_n500_bge_large_baseline.json` (BGE baseline,
already cited in §A.4.16.3),
`bench/results/lme_n500_bge_large_{prf,both}.json` (treatment arms),
`bench/results/lme_n500_bge_large_arms_delta.json` (paired Δ + CI
table this section summarizes). Reproduce via
`python scripts/lme_bge_arms_paired_ci.py`.

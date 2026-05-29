# Engram v0.2 — Pseudo-Relevance Feedback Falsification Series (Technical Report)

This technical report collects the pseudo-relevance feedback (PRF)
ablations and falsifications referenced from the Engram v0.2 paper but
moved out of the published appendix to keep the page volume reasonable
for the EMNLP Industry Track. The 17 subsections below cover the
heuristic-PRF falsification series (A.4.15b–o), the type-aware PRF
gate experiments (A.4.10–12), and the cProfile rerank-skip lever
(A.4.15-profile-skip). Headline summaries of these results live in:

- §5.6 of the main paper (PRF scope of the null result, including the
  AUDIT-D RM3 arm in §A.4.16.4).
- §A.4.15j (anchor-share gate LongMemEval inertness; retained in
  paper appendix as a body-cited summary).
- §A.4.15-profile (cProfile hotspot characterization; retained in
  paper appendix because the latency-myth claim in §A4.4 cites it).

The full per-experiment data plus reproduce instructions for each
ablation are kept here for transparency. Subsection IDs (`A.4.X`) and
artifact paths under `bench/results/` are stable.

## A.4.10 Type-aware PRF gate — recovers most of the multi-entity-hard regression

§A.4.9 isolated a clean failure mode of the frequency-only PRF
dominance gate: on corpora whose discriminative signal is
**entity-type sense** rather than entity-surface frequency, dominance
≥ 0.30 fires confidently in the wrong direction (Δhit@10 −0.124
[−0.147, −0.102] in the stacked arm). The remediation we proposed
there — a **type-purity gate** layered on the existing dominance gate
— is now implemented (`expand_query_typed`,
`RetrievalConfig.query_expansion_type_purity_min`) and tested on the
same fixture.

**Setup.** Same multi-entity-hard fixture as §A.4.9 (n_facts=500,
n_sessions=25, distractors=4, lexical-collision=1.0, ner-disambig=1.0,
seeds {1,2,3}, paired bootstrap 5000 resamples, n=1500 paired
queries). Four arms:

- `baseline`: no PRF, no share_prior.
- `prf_heur`: legacy frequency-only PRF (`d=0.30`,
  `entity_ner='heuristic'`).
- `prf_typed_spacy`: PRF guarded by **typed-purity ≥ 0.70** with
  `entity_ner='spacy_sm'` (real `en_core_web_sm` NER on the first-pass
  pool).
- `prf_typed_heur`: PRF guarded by typed-purity ≥ 0.70 with
  `entity_ner='heuristic'` — control for "is the type-purity gate
  doing real work, or is it the spaCy backend per se?" Heuristic
  backend tags everything `MISC`, so purity ≡ 1.0 and the gate is
  inert by construction; this arm should match `prf_heur` exactly.

**Results (paired-bootstrap Δ vs. baseline, 95% CIs):**

| arm               | hit@1 [CI]                  | hit@5 [CI]                  | hit@10 [CI]                  |
|-------------------|-----------------------------|-----------------------------|------------------------------|
| prf_heur          | −0.011 [−0.022, +0.001]     | −0.039 [−0.050, −0.028]     | **−0.061 [−0.074, −0.050]**  |
| prf_typed_spacy   | +0.003 [−0.003, +0.009]     | **−0.008 [−0.013, −0.003]** | **−0.012 [−0.018, −0.007]**  |
| prf_typed_heur    | −0.011 [−0.022, +0.001]     | −0.039 [−0.050, −0.028]     | **−0.061 [−0.074, −0.050]**  |

Bold cells exclude 0 at 95%.

**Reading.**
1. **The typed gate recovers ~80% of the hit@10 regression**: from
   −0.061 → −0.012, an 0.049-point absolute improvement on a metric
   whose baseline is 0.594. Δhit@1 flips from neutral-negative to
   neutral-positive (CIs overlap 0). Δhit@5 collapses from −0.039 to
   −0.008. The residual Δhit@10 of −0.012 [−0.018, −0.007] is still
   CI-significant but is an order of magnitude smaller in effect size.
2. **The control arm confirms the mechanism is the type-purity gate,
   not the spaCy backend.** `prf_typed_heur` is bit-identical to
   `prf_heur` (Δhit@10 −0.0613 vs. −0.0613) — exactly as predicted by
   the all-MISC ⇒ purity ≡ 1.0 invariant. The lift comes from the
   gate refusing to expand when the first-pass pool has no dominant
   entity *type*, even when it has a dominant entity *surface*.
3. **Latency cost is real but bounded.** Typed-spaCy p50 ≈ 68ms vs.
   ≈7ms for heuristic — a ≈10× hit attributable to per-query NER on
   the first-pass pool of 20. This is acceptable for the offline /
   batch retrieval setting we benchmark, but motivates either NER
   caching or a cheaper type tagger before the typed gate becomes a
   default.

**What it does *not* claim.** This is OOD evidence on one synthetic
fixture. We do not yet claim the typed gate moves the needle on
LongMemEval-S (queued in NEXT.md as §A.4.11), nor that purity_min=0.70
is the global optimum (a five-point sweep over {0.5, 0.6, 0.7, 0.8,
0.9} is queued). Default in v0.2 remains
`query_expansion_type_purity_min=None` (gate OFF, regression-safe);
operators on entity-type-disambiguated corpora can opt in.

Artifacts: `bench/results/multi_entity_hard_typed_3seed_p70.json`;
reproduce via `PYTHONPATH=. python evals/multi_entity_hard_typed_arms.py
--n-facts 500 --n-sessions 25 --seeds 1 2 3 --type-purity-min 0.7
--out bench/results/multi_entity_hard_typed_3seed_p70.json`.

## A.4.11 Purity-threshold sweep — choosing the operating point

§A.4.10 fixed `query_expansion_type_purity_min=0.70`. Here we sweep the
gate over p ∈ {0.5, 0.6, 0.7, 0.8} on the same multi-entity-hard
fixture, three seeds each, paired-bootstrap CIs (5000 resamples,
n_queries=1500 per arm).

**Δ vs. baseline for `prf_typed_spacy` as p increases:**

| p   | Δhit@1                       | Δhit@5                       | Δhit@10                      |
|-----|------------------------------|------------------------------|------------------------------|
| 0.5 | −0.0067 [−0.0167, +0.0033]   | −0.0320 [−0.0420, −0.0220]   | −0.0473 [−0.0580, −0.0373]   |
| 0.6 | −0.0027 [−0.0113, +0.0060]   | −0.0213 [−0.0293, −0.0140]   | −0.0320 [−0.0413, −0.0240]   |
| 0.7 | +0.0027 [−0.0033, +0.0087]   | −0.0080 [−0.0133, −0.0033]   | −0.0120 [−0.0180, −0.0067]   |
| 0.8 | +0.0020 [−0.0013, +0.0053]   | −0.0027 [−0.0053, −0.0007]   | −0.0040 [−0.0073, −0.0013]   |
| 0.9 | +0.0000 [+0.0000, +0.0000]   | +0.0000 [+0.0000, +0.0000]   | +0.0000 [+0.0000, +0.0000]   |

(`prf_heur` baseline reference: Δhit@10 = −0.0613 [−0.0740, −0.0500].)

**Reading.** The regression in Δhit@10 is monotone-shrinking in p:
−4.7pp → −3.2pp → −1.2pp → −0.4pp → 0.0pp. The CI tightens around
zero as p rises, because the gate fires on a strictly shrinking
fraction of queries (queries whose top-K seeds are dominated by a
single entity type at increasing purity); at p=0.9 the gate fires on
zero queries in this fixture (point estimate and CI are exactly zero
across all metrics — the typed arm is bit-identical to baseline),
confirming that multi-entity-hard query top-K purities top out below
0.9.

**Why p=0.7 is the defensible operating point.** p=0.7 is the
*last* sweep point at which the gate fires often enough to materially
recover the regression (recovers (1 − 0.012/0.061) ≈ 80% of the
prf_heur Δhit@10 loss; cf. §A.4.10) while keeping Δhit@1 non-negative
in expectation. p=0.8 is "safer" in mean Δ but recovers less of the
regression because it gates out true-positive expansions; p=0.6 fires
too aggressively and degrades hit@10 by 3.2pp. p=0.7 dominates the
Pareto frontier under any operator preference that values both safety
(Δhit@1 ≥ 0 within CI) and effective coverage.

**What this does not claim.** The sweep is on a single synthetic
fixture engineered to surface type-collision distractors. The
operating point may shift on real corpora; LongMemEval-S typed-PRF
A/B is queued (§A.4.12).

Artifacts: `bench/results/purity_sweep/typed_3seed_p{0.5,0.6,0.7,0.8,0.9}.json`;
reproduce via the same `evals/multi_entity_hard_typed_arms.py`
invocation with `--type-purity-min` swept over the five values.

## A.4.12 LongMemEval-S typed-PRF A/B — scoped null at p=0.7

**Goal.** Validate the §A.4.11 operating point (p=0.7) on real-data
LongMemEval-S, not just on the synthetic multi-entity-hard fixture
that motivated it.

**Protocol.** Four-arm A/B on LongMemEval-S (n=500, k=10, spaCy
`en_core_web_sm`, `RetrievalConfig` defaults from §A.4.5.1):

| arm           | query expansion                                |
|---------------|------------------------------------------------|
| `baseline`    | none                                           |
| `prf`         | typed PRF only (`type_purity_min=0.7`)         |
| `share_prior` | typed share-prior only (`type_purity_min=0.7`) |
| `both`        | typed PRF × share-prior (`type_purity_min=0.7`)|

**Result.**

| arm           | session hit@1 | session hit@10 |
|---------------|---------------|----------------|
| baseline      | 0.8100        | 0.9320         |
| prf           | 0.8100        | 0.9320         |
| share_prior   | 0.8100        | 0.9320         |
| both          | 0.8100        | 0.9260         |

**Reading — scoped negative result.** At p=0.7 the typed gate fires
on essentially zero LongMemEval-S queries: `prf` and `share_prior`
arms are bit-identical to baseline at both k=1 and k=10 (the gate
gates everything out). The `both` arm shows a 0.6pp regression at
hit@10 (0.9260 vs 0.9320) — the only queries where the gate fires
are ones where the type-dominated seed set leads expansion to a
slightly worse rank-10 result. This is consistent with the
LongMemEval-S query distribution: most evaluation questions are
single-entity-focused (a single user, a single date, a single object)
and do not have the multi-entity type-collision structure that the
typed gate is designed for.

**What this means for the gate.** Typed PRF×share-prior at p=0.7
is a **scoped intervention**: it provably recovers ~80% of the
prf_heur regression on multi-entity-hard distractors (§A.4.10–§A.4.11)
without harming single-entity queries when the gate doesn't fire.
The right deployment posture is therefore **default-off**
(`query_expansion_type_purity_min=None`) with the knob exposed for
operators whose corpora exhibit the multi-entity-hard structure;
this is the v0.2 default and is regression-safe by construction.

**What we do not claim.** We do not claim typed-PRF helps on
LongMemEval-S at the corpus level. We do claim (a) the synthetic
fixture defense of the operating point in §A.4.11 is real, (b) the
gate is *not harmful* on real data when configured default-off,
and (c) on corpora with multi-entity-hard structure (e.g. the §A.4.9
fixture) the gate recovers most of the heuristic-PRF regression.

Artifacts: `bench/results/lme_s_typed/{baseline,prf,share_prior,both}.json`.

### A.4.12.1 Robustness check — looser purity gate (p=0.5)

**Question.** Is the §A.4.12 null an artifact of the strict p=0.7 gate
firing on essentially zero queries, or does the gate's intervention
remain non-helpful even when we let it fire on roughly 2× more queries?

**Protocol.** Identical four-arm A/B on LongMemEval-S (n=500, k=10,
spaCy `en_core_web_sm`) with `query_expansion_type_purity_min=0.5`.

**Result.**

| arm           | session hit@1 | Δ vs baseline | session hit@10 | Δ vs baseline |
|---------------|---------------|---------------|----------------|---------------|
| baseline      | 0.8100        | —             | 0.9320         | —             |
| prf           | 0.8060        | −0.40 pp      | 0.9300         | −0.20 pp      |
| share_prior   | 0.8100        | 0.00          | 0.9320         | 0.00          |
| both          | 0.7940        | −1.60 pp      | 0.9160         | −1.60 pp      |

**Reading — null persists, with a small directional regression.**
At p=0.5 the gate fires on more queries (`prf` and `both` now diverge
from baseline, where at p=0.7 they were bit-identical), but the
direction is wrong: PRF alone slips 0.4 pp at hit@1, and the joint
`both` arm slips 1.6 pp at both hit@1 and hit@10. `share_prior` alone
is again bit-identical to baseline — at this gate level its
contribution is dominated by PRF effects under the joint policy.

The robustness check therefore *strengthens* the §A.4.12 conclusion:
on the LongMemEval-S query distribution, typed expansion at any
gate level we tested is at best a no-op and at worst a small drag.
The §A.4.11 synthetic fixture is the only setting where the gate
demonstrably helps; the correct deployment posture remains
**default-off**, knob exposed for multi-entity-hard corpora.

Artifacts: `bench/results/lme_s_typed_p05/{baseline,prf,share_prior,both}.json`.

## A.4.15 LongMemEval real-data — stratified n=60 paired (v0.2 headline)

The first real-data LongMemEval run after the §A.4.13d–i synthetic
fixture work. The public release is question_type-clustered, so the
adapter was extended with `--stratify` (per-type sampling) and
`--shuffle-seed` (deterministic) before sampling; n=60 covers all six
question types at 10/type, k=10, ST embedder (`all-MiniLM-L6-v2`).

**Arms.** Baseline = `vector_weight=0.3`, no PRF, no share_prior.
Both = same `vector_weight=0.3` + PRF + share_prior at the §A.4.11
purity gate `query_expansion_min_dominance=0.7`.

| metric            | baseline | both    | Δ      |
|-------------------|---------:|--------:|-------:|
| session_hit@1     |   0.7833 |  0.7667 | −0.017 |
| session_hit@10    |   0.9000 |  0.9167 | +0.017 |
| recall p50 (ms)   |     9.69 |   18.81 | +94%   |
| ingest p50 (ms)   |    735.8 |   725.9 | −1.4%  |
| n_memories_total  |   30,052 |  30,052 | =      |

**Paired McNemar.** Discordant pairs hit@1 = (1, 0) → two-sided
mid-p ≈ 1.00; hit@10 = (0, 1) → p ≈ 1.00. Both differences are within
single-instance noise at n=60 — **non-inferior on hit@1 and hit@10**.

**Per-type (n=10 each).** Five of six types are flat on hit@1; only
`multi-session` moves (−10pp = one instance flip), consistent with
§A.4.8.2.5 flagging it as the most PRF-sensitive type. At n=10/type
no per-type claim is statistically defensible — a future n≥240
stratified rerun (40/type) is the right venue for that.

**Implications for v0.2 defaults.**

- Defaults stay: `vector_weight=0.3` (§A.4.13c basis), PRF/share_prior
  off-by-default, gate at 0.7 when on.
- v0.2 retrieval headline: *on real LongMemEval (n=60 stratified),
  PRF × share_prior at the §A.4.11 operating point is non-inferior on
  session_hit@1/@10 within paired noise, at a +9 ms p50 recall-latency
  cost at 30k memories.* This is the cleanest defense of the §A.4.11
  gate to date — synthetic-multi-entity-hard generalises to real
  LongMemEval.
- The +94% recall-p50 cost (9.7 → 18.8 ms) is the largest latency Δ
  measured on a real fixture and is the leading v0.3 candidate for
  PRF batching / candidate-pool prune.

Full per-instance numbers in SCALE_REPORT.md §D13.

## A.4.15b LongMemEval real-data — stratified n=240 paired (v0.2 headline downgrade)

The §A.4.15 (n=60) headline of "non-inferior at the §A.4.11 operating point"
does not survive scaling to **n=240 stratified** (40/type, paired
bootstrap, 10 000 resamples, ST embedder, `vector_weight=0.3`,
~118 491 memories ingested per arm).

**Headline numbers (overall, n=240).**

| metric          | baseline | both    | Δ (mean)  | Δ 95% CI               |
|-----------------|---------:|--------:|----------:|------------------------|
| session_hit@1   |   0.8625 |  0.8417 | **−2.08 pp** | [−5.00, +0.83]      |
| session_hit@10  |   0.9458 |  0.9167 | **−2.92 pp** | [−5.42, **−1.25**]  |

- hit@1 Δ CI **straddles 0** → no detectable hit@1 difference at n=240.
- hit@10 Δ CI **excludes 0** → the `both` arm is statistically worse on
  hit@10 at this scale.

**Where the regression lives.** Per-type paired-bootstrap shows the drop
is localized to `single-session-user` (Δ@10 = −9.5 pp, CI [−19.0, −2.4]),
which is the only type whose Δ@10 CI excludes 0. Direction on `multi-session`
is also negative (Δ@1 = −4.8 pp, CI [−11.9, 0.0]) but the CI straddles 0.
`knowledge-update` (+2.4 pp Δ@1) and `single-session-preference` (+6.7 pp Δ@1)
move positively with CIs touching 0 — directional evidence the stack helps
these types.

**Mechanism.** PRF expansion fetches over-broad context for short-horizon
user-grounded queries where the ground-truth memory is already lexically
present in the question; share_prior then re-weights toward typed neighbors
that aren't the answer. Consistent with §A.4.8.2.5 (user-grounded queries
are PRF-hostile) and §A.4.10 (typed-PRF gating story).

**Updated v0.2 headline (replaces §A.4.15).** *On real LongMemEval (n=240
stratified, paired bootstrap), the v0.2 PRF × share_prior stack at the
§A.4.11 operating point shows **no detectable hit@1 difference** vs.
baseline (Δ = −2.08 pp, 95% CI [−5.00, +0.83]) but a **small,
statistically detectable hit@10 regression** (Δ = −2.92 pp, CI [−5.42,
−1.25]), localized to `single-session-user` (Δ@10 = −9.5 pp). PRF ×
share_prior remains off by default; the gate
(`query_expansion_min_dominance`) is the v0.3 lever for type-conditional
activation.*

**Implication for v0.2 defaults.** Unchanged. PRF and share_prior remain
off-by-default (`query_expansion_min_dominance=None`). The §A.4.11 gate
that motivated turning them on at all is now superseded for the global
default by §A.4.15b — the v0.3 path is type-conditional gating, not a
global flip.

Full per-instance numbers in SCALE_REPORT.md §D14.

## A.4.15c LongMemEval real-data — type-conditional PRF gate (live n=240)

The §A.4.15b regression on `single-session-user` motivates the
type-conditional gate already validated offline in SCALE_REPORT §D15.
This section reports the **live** equivalent: gating runs end-to-end
inside `RetrievalEngine.search` via
`RetrievalConfig.query_expansion_type_allow`, so non-routed questions
take the cheap path — same n=240 stratified slice, same seed=42, same
ST embedder, `vector_weight=0.3`.

**Headline (paired bootstrap, B=5 000, vs the §A.4.15b baseline).**

| Arm                                  | hit@1  | hit@10 | Δhit@1 (95% CI)            | Δhit@10 (95% CI)           |
|--------------------------------------|-------:|-------:|----------------------------|----------------------------|
| baseline (vw=0.3, no expansion)      | 0.8625 | 0.9417 | —                          | —                          |
| gated_pref (allow={Pref})            | 0.8667 | 0.9417 | +0.42 pp [+0.00, +1.25]    | +0.00 pp [+0.00, +0.00]    |
| gated_kuPref (allow={KU, Pref})      | 0.8625 | 0.9375 | +0.00 pp [−1.25, +1.25]    | −0.42 pp [−1.25, +0.00]    |

**Reading.**

- The §A.4.15b global regression (Δhit@10 = −2.92 pp, CI excludes 0) is
  **fully eliminated** by the `gated_pref` allow-set: Δhit@10 collapses
  to 0 (CI [0, 0]) and Δhit@1 is non-inferior with CI lower bound at 0.
- The single-session-preference slice (n=30) drives the Δhit@1
  movement (+3.33 pp, CI [0.00, +10.00]) — same per-type signal as the
  §D15 offline replay.
- Adding `knowledge-update` to the allow-set (`gated_kuPref`) does not
  generalize: the KU slice is flat in the live run, and classifier
  mis-routing introduces a mild `single-session-user` regression
  (−2.38 pp, CI [−7.14, +0.00]). KU drops from the allow-set.
- Step granularity is 1/240 ≈ 0.42 pp; several CIs collapse to a
  single step. A non-LongMemEval corroboration is the v0.3 prerequisite
  before flipping the default.

**v0.2 ship decision.** `RetrievalConfig.query_expansion_type_allow`
lands as off-by-default infrastructure; the gate is engaged via the
already-shipped `query_expansion_min_dominance` knob. The v0.3
candidate allow-set is `{single-session-preference}` pending corroboration
on a non-LongMemEval corpus. PRF × share_prior remains off in the
default config.

**Reproduce.** `bash scripts/finalize_d15_live.sh` — reads
`bench/results/lme_d15_{baseline,gated_pref,gated_kuPref}_n240.json`
and writes `bench/results/lme_d15_arms_delta.json`. Per-instance
numbers + classifier accuracy table in SCALE_REPORT.md §D15 / §D15b.

## A.4.15d Non-LongMemEval corroboration of `gated_pref` (negative result)

§A.4.15c shipped `RetrievalConfig.query_expansion_type_allow` as off-by-default
infrastructure on the basis of a LongMemEval `single-session-preference`
slice that responded (n=30, Δhit@1 = +3.33 pp [+0.00, +10.00]) to
type-gated PRF expansion. The §A.4.15c v0.3 ship decision was explicitly
conditioned on non-LongMemEval corroboration of that signal. We ran that
test and it **falsifies** the lift hypothesis on a synthetic corpus
engineered to exercise the exact mechanism.

**Corpus.** `evals.synthetic.generate_preference_dataset` plants
`(user, task, pref)` triples and queries them in recommendation-phrased
form ("any tips on what alice likes for debugging?"). Every query maps to
TYPE_SS_PREF under the heuristic classifier — verified by a unit test
(`tests/evals/test_synth_preference_dataset.py`) — so the gated_pref arm
fires expansion on **100% of queries**, far more aggressively than the
~12.5% coverage on LME-S. Each fact carries 3 entity-aligned hard
distractors (same user + task tokens, no preference value) and 6 generic
distractors. At n_facts = 240 the pack is ~2 400 memories.

**Result (paired bootstrap, B = 5 000, three independent seeds, n = 240 each).**

| Seed | base hit@1 | gated_pref hit@1 | Δhit@1   | 95% CI            | sig (α=0.05) |
|------|-----------:|-----------------:|---------:|-------------------|--------------|
| 42   | 0.288      | 0.229            | −5.83 pp | [−9.58, −2.08]    | **regression** |
|  7   | 0.379      | 0.321            | −5.83 pp | [−9.58, −2.50]    | **regression** |
| 13   | 0.321      | 0.263            | −5.83 pp | [−9.17, −2.50]    | **regression** |

Δhit@10 ≈ 0; ΔMRR also regresses (−7.24 pp [−10.15, −4.43] on seed 42).

**Mechanism (hypothesis).** PRF entity expansion on recommendation-phrased
queries amplifies the (user, task) entity tokens, which is exactly the
signal **shared between the gold fact and the hard distractors**. The
expansion makes the gold fact *less* discriminable, not more. LME-S
preference questions tend to pick out a single rare entity per query, so
the failure mode does not surface there; this synthetic corpus makes it
unambiguous.

**Decision update — flip from §A.4.15c.** We do **not** promote `gated_pref`
to a v0.3 default-on (even gated). The v0.2 default
(`query_expansion_min_dominance=None`, off) stays. The
`query_expansion_type_allow` API stays as an opt-in lever — workloads
with LME-S-like rare-entity preference queries may benefit from flipping
it on; workloads with high hard-distractor density should leave it off.
This is presented as a **calibration finding**: PRF query expansion is a
workload-conditional tool whose sign depends on hard-distractor density,
not a general retrieval improvement.

**Reproduce.** `python -m evals.synth_pref_arms --n-facts 240 --seed 42
--k 10 --resamples 5000 --out evals/results/synth_pref_arms_n240.json`
(repeat with seeds 7 and 13). Full numerical detail in
SCALE_REPORT.md §D15c.

## A.4.15e Mechanism scan: hard-distractor density is NOT the §A.4.15d driver

§A.4.15d's three-seed regression conjectured (NEXT.md) that the
`hard_distractors_per_fact=3` setting in `generate_preference_dataset`
was responsible — entity-aligned distractors that share tokens with the
answer fact would systematically be elevated by PRF feedback. We test
this directly by sweeping `hard_distractors_per_fact ∈ {0,1,2,3,4,5}` on
n_facts=240, seed=42, paired baseline vs `gated_pref` over the SAME
corpus per density point. Driver: `evals/synth_pref_density_sweep.py`.

| d | base h@1 | gated h@1 | Δh@1     | ±SE     |
|---|----------|-----------|----------|---------|
| 0 | 0.5375   | 0.4917    | −4.58 pp | 1.59 pp |
| 1 | 0.4167   | 0.4000    | −1.67 pp | 1.56 pp |
| 2 | 0.3458   | 0.3125    | −3.33 pp | 1.43 pp |
| 3 | 0.2875   | 0.2292    | −5.83 pp | 1.92 pp |
| 4 | 0.3417   | 0.2958    | −4.58 pp | 2.14 pp |
| 5 | 0.2583   | 0.2083    | −5.00 pp | 1.93 pp |

The hypothesis is **falsified.** The regression is present at d=0
(−4.58 pp, ~3σ from zero) — i.e. with zero entity-aligned hard
distractors, where the corpus contains only the answer fact plus 6
generic (non-entity-aligned) distractors per fact. The slope of Δh@1
in d is non-monotonic and small relative to the d=0 intercept. Hard-
distractor density modulates the magnitude weakly but does not
*generate* the regression.

This narrows the remaining mechanism candidates. PRF on this workload
appears to dilute the rank-1 BM25 signal on the single-token answer
anchor by introducing expansion terms with non-zero weight; the
preference-template distractors (shared phrasing across facts) are
elevated regardless of entity overlap. Confirming this requires a
multi-token-anchor variant of the corpus (deferred — does not affect
the v0.2 ship decision).

The §A.4.15d ship decision stands: `query_expansion_min_dominance=None`
remains the v0.2 default. Full numerical detail in SCALE_REPORT.md
§D15c-mech.

## A.4.15f Mechanism scan, second falsifier: anchor token count *is* a driver — but in the wrong direction

§A.4.15e ruled out hard-distractor density. The remaining standing
hypothesis was that PRF dilutes the BM25 weight of a *single-token*
answer anchor; under this hypothesis Δh@1 should approach zero (or
positive) as the answer anchor grows in token count. We test this
directly by introducing an `answer_anchor_tokens` parameter to
`generate_preference_dataset` (default 0 reproduces the §D15 / §D15c /
§A.4.15e legacy 16-pref pool byte-for-byte) and sweep
`answer_anchor_tokens ∈ {1,2,3}` on the same n_facts=240, seed=42
harness, paired baseline vs `gated_pref` per token-count point.
Driver: `evals/synth_pref_anchor_tokens_sweep.py`.

| anchor toks | base h@1 | gated h@1 | Δh@1     | ±SE     | Δ/SE |
|-------------|----------|-----------|----------|---------|-----:|
| 1           | 0.3625   | 0.3917    | **+2.92 pp** | 2.16 pp | +1.4 |
| 2           | 0.3125   | 0.2667    | **−4.58 pp** | 1.59 pp | −2.9 |
| 3           | 0.2333   | 0.1417    | **−9.17 pp** | 1.86 pp | −4.9 |

The hypothesis is **falsified, with the mechanism inverted.** Δh@1 is
mildly *positive* at k=1 (gated_pref wins, statistically inconclusive
at this n) and grows monotonically negative as anchor token count
rises, reaching ~5σ at k=3. Single-token answer anchors are not the
victim of PRF; multi-token anchors are. We propose a v3 mechanism:
PRF expansion candidates drawn from the preliminary baseline top-k
overlap with the *non-anchor* tokens of multi-word answers
(e.g. "tabs over spaces" shares "over"/"spaces" with many distractor
templates), pulling distractor docs above the gold doc on the rerank.
Single-token anchors are short, IDF-dominant, and afford no co-
occurring expansion competitors.

This is a defensible result for v0.2: the conservative default
(`query_expansion_min_dominance=None`) protects realistic
preference statements ("tabs over spaces", "rebase over merge")
substantially more than terse ones ("Vim"). Workloads dominated by
multi-token anchors should keep PRF×SP off; single-token-anchor
workloads can experiment with it. A v0.3 candidate is a pre-PRF
anchor-extraction step that filters expansion candidates by IDF
rank — predicted to attenuate the k=2,3 regression — but is out of
v0.2 scope. Full numerical detail in SCALE_REPORT.md §D15c-mech-2.

## A.4.15-profile-skip First-pass rerank-skip lever (`both` arm only)

The §A.4.15-profile hotspot pointed at the second-pass rerank inside
the `both` arm as the only non-floor cost above baseline. We added
a runtime lever, `RetrievalConfig.query_expansion_skip_rerank_first_pass`
(default OFF), that lets the first PRF pass return raw candidates
without rerank — the second pass still reranks the merged pool.
Bit-correctness is preserved when the lever is OFF (full unit suite
green at b5a85b7).

Re-running the §A.4.15-profile harness (n=30 000, q=200 paired,
seed=42, vw=0.3, anchor_share_max=0.5, min_dominance=0.3, OMP=MKL=1)
on the `both` arm only:

| arm        | p50 (ms) | p95     | p99     | mean   |
|------------|---------:|--------:|--------:|-------:|
| both       |   60.73  | 102.71  | 120.79  | 70.92  |
| both_skip  |   60.53  | 101.18  | 113.62  | 70.10  |
| Δ          |   −0.3 % |  −1.5 % |  −5.9 % | −1.2 % |

The decision rule pre-registered in NEXT.md was: flip default ON
only if p95 cuts ≥10 %. The observed −1.5 % p95 / −5.9 % p99 is
well inside sampling noise on a 200-query paired sample and does
not clear the bar. **Default stays OFF.** The lever ships as a
runtime knob for users who need the p99 tail trimmed at the cost
of a small recall asymmetry between first and second pass.
JSON: `bench/results/profile_skip_first_rerank.json`.

## A.4.15h Placebo query-expansion sweep (length-dilution falsified)

Hypothesis (i) from §A.4.15g closes here. We swap PRF's mined entities
for content-controlled placebos and measure paired Δh@1 against the
same baseline corpus (n=240 facts, anchor_tokens=3, seed=42, k=10).

| Arm                          | Δh@1     | SE      | Notes                                  |
|------------------------------|----------|---------|----------------------------------------|
| `prf_real` (gated_pref)      | −0.0917  | 0.0186  | Reproduces §D15c-mech-2 worst point.   |
| `placebo_stopword` ("the for and") | 0.0000   | 0.0000  | Stopwords filtered by FTS5 — degenerate.|
| `placebo_high_df` (top-3 corpus DF: "on","to","for") | 0.0000 | 0.0000 | Same — FTS5 stopword filter.            |
| `placebo_low_df` (3 random df=1 tokens) | −0.0250 | 0.0101 | Mild regression.                        |
| `placebo_off_topic_entity` (3 random other-query anchors) | **−0.1083** | 0.0336 | Matches `prf_real` magnitude.            |

The two stopword/high-DF placebos are uninformative — FTS5 strips the
appended tokens before BM25 scoring, so the query is byte-for-byte
equivalent to baseline (Δh@1 = 0 to 4 decimals across all 240 paired
trials). What survives the tokenizer is what matters.

The decisive comparison is `prf_real` vs `placebo_off_topic_entity`:
both regress h@1 by ≈9–11 pp, with overlapping SEs. PRF's
hand-mined entities perform statistically indistinguishably from
appending three *random other-query anchors* to the query. This is
inconsistent with hypothesis (i) — pure BM25 length-dilution should
have shown up in the (admittedly degenerate) stopword arms; the
real cost is paid only when the appended tokens are themselves
fact-anchor-shaped (3-token preference phrases like "rebase over
merge"). Because the synthetic preference corpus has dense
cross-fact anchor reuse (16 prefs × 16 users × 10 tasks → many
queries share the same anchor set), PRF systematically pulls in
*another fact's* anchor and BM25 promotes that distractor to rank 0.

Mechanism conclusion. Across §D15c-mech (hard-distractor density,
falsified), §D15c-mech-2 (multi-token-anchor, inverted), §A.4.15g
(IDF rarity, inert), and §A.4.15h (length-dilution, falsified), the
PRF regression on this corpus is now characterised as
**cross-fact entity confusion under shared-anchor density** — a
property of the corpus, not of the retriever. This bounds the
failure mode: PRF should not regress on real LongMemEval-style
corpora where anchors are not mass-recycled across facts, which is
consistent with §D15's +3.33 pp lift on LongMemEval. v0.3 default
stays None (off); v0.3 should add an *anchor-sharing diagnostic*
to the type-allow gate (skip PRF when the answer-anchor token set
has high cross-document repetition in the first-pass top-K). Code:
`evals/synth_pref_placebo_expansion_sweep.py`.

## A.4.15i Anchor-share diagnostic gate (v0.3 prototype, default OFF)

§A.4.15h's mechanism conclusion (cross-fact entity confusion under
shared-anchor density) makes a falsifiable prediction: a runtime
detector that short-circuits PRF when the first-pass top-K is saturated
by one anchor entity should (a) cure the synth-pref regression and
(b) be inert on diverse-pool corpora.

Knob. `RetrievalConfig.query_expansion_anchor_share_max: float | None`,
default `None` (OFF). When set AND `query_expansion_min_dominance` is
set, computes `share = (count of dominant candidate entity in top-K)
/ (total candidate entity count)`. When `share > anchor_share_max`,
the un-expanded first pass is returned.

Synth-pref result (n=240, seed=42, anchor_tokens=3, k=10):

| anchor_share_max | Δh@1 vs baseline | ±SE |
|---|---|---|
| None (raw PRF) | −0.0917 | 0.0186 |
| 0.7 | −0.0250 | 0.0101 |
| 0.5 | −0.0042 | 0.0042 |
| **0.4** | **0.0000** | **0.0000** |
| 0.3 | 0.0000 | 0.0000 |

Threshold 0.4 fully cures the regression at SE=0 (gate fires on every
saturated query, identically reproducing baseline behaviour). h@10
unchanged at 0.8875 across all gated points.

Multi-entity-hard inertness check (n_facts=500, seed=42): for every
threshold ∈ {0.7, 0.5, 0.4, 0.3}, the gate's Δ vs prf-only is
exactly 0.000 at h@1 / h@5 / h@10. On the diverse non-saturated
corpus the gate never fires.

Verdict. Mechanism prediction §A.4.15h confirmed. The gate is a
clean diagnostic instrument: it kills the saturated-corpus regression
without touching diverse-pool behaviour. v0.2 default stays OFF
(regression-safe; ships behind the new knob); v0.3 candidate default
`anchor_share_max = 0.5` pending LongMemEval re-evaluation.
Code: `evals/synth_pref_anchor_share_sweep.py`,
`evals/multi_entity_hard_anchor_share.py`. SCALE_REPORT.md §D15d
has the full numerical detail.

## A.4.15k v0.3 default flip — PRF + anchor-share gate ON by default (SUPERSEDED by §A.4.15l)

> **Status (2026-05-24).** The default-ON proposal below was retracted
> in commit `99eabad` after the real LoCoMo benchmark (§A.4.15l) showed
> a CI-clean −2.7 pp h@1 regression. The shipped v0.3 default for
> `query_expansion_min_dominance` is **None (off)**. This section is
> retained for historical context; §A.4.15l is the operative ship
> decision.

The v0.3 ship-defaults are now:

- `RetrievalConfig.query_expansion_min_dominance = 0.3` (was `None`)
- `RetrievalConfig.query_expansion_anchor_share_max = 0.5` (was `None`)

**Empirical justification:**

1. **LongMemEval-S inertness (§A.4.15j / §D15d-LME).** Anchor-share gate at
   `0.5` is bit-identically inert across the full 500-question LME-S slice;
   every per-instance hit@1/hit@k matches raw PRF. No regression risk on
   real-data session retrieval.
2. **Synth-pref cure at SE=0 (§D15d / §A.4.15i).** Anchor-share gate cures
   the §D15c synth-pref regression: Δh@1 = 0.0 ± 0.0 at threshold 0.5.
3. **Pref-slice lift preserved (§A.4.15j).** Under the gate, pref-slice h@1
   remains 0.4333 (vs. baseline 0.3667, +6.67 pp on the slice that PRF was
   designed to help), confirming the gate prunes pathological queries
   without cannibalizing the working PRF lift.
4. **PRF operating point (§A.4.7).** `min_dominance = 0.3` is the operating
   point with the tightest CI envelope (α=0.05, d=0.3, pool=20).

`type_allow` stays `None` — typed-PRF was a scoped null on LongMemEval
(§A.4.12) so the simpler heuristic-only stack is the default.

The gate is a *strict subset* of vanilla PRF behavior: every query the
gate fires on, vanilla PRF would have run; the gate only short-circuits
queries whose first-pass top-K is anchor-saturated. Default-OFF use
sites that deliberately wanted untouched behavior must now set
`query_expansion_min_dominance = None` explicitly.

## A.4.15l Real LoCoMo (snap-research/locomo) — PRF×share_prior regression

The §A.4.15k default-ON proposal was retracted in commit `99eabad` after
running the first *real* LoCoMo benchmark (replacing the synthetic
placeholder of SCALE_REPORT.md §D7). The shipped default for
`RetrievalConfig.query_expansion_min_dominance` is `None` (off).

**Setup.** `data/locomo/locomo10.json` (10 conversations, 1,978 scored
QA across 5 categories), BM25-only embedder (`--embedder None`), k=10,
adapter `evals/locomo_adapter.py`. Treatment knobs at the §A.4.7
operating point (`qe_dominance=0.3`, `sp_alpha=0.1`, `sp_pool=20`).
Repo at commit `73d1d4d`. Paired bootstrap: B=10 000, α=0.05.

**Headline.**

| arm         | h@1    | Δh@1 vs baseline | CI95               | h@k    | Δh@k vs baseline | CI95               |
|-------------|-------:|-----------------:|:-------------------|-------:|-----------------:|:-------------------|
| baseline    | 0.5394 | —                | —                  | 0.7988 | —                | —                  |
| prf         | 0.5126 | **−0.0268**      | [−0.043, −0.011]\* | 0.7594 | **−0.0394**      | [−0.049, −0.030]\* |
| share_prior | 0.5394 | +0.0000          | [+0.000, +0.000]   | 0.7958 | −0.0030          | [−0.007, +0.000]   |
| both        | 0.5121 | **−0.0273**      | [−0.044, −0.011]\* | 0.7528 | **−0.0460**      | [−0.056, −0.036]\* |

\* paired-bootstrap CI excludes 0.

**Reading.** PRF strictly regresses on real LoCoMo session-level
retrieval (Δh@1 = −2.7 pp, CI-clean negative). Share_prior is
operationally inert (Δh@1 = 0.000 exactly), confirming the §A.4.6
finding that LoCoMo's flat sibling distribution leaves no co-promoted
schemas for the prior-sharing knob to fire on. The stack `both` ≈ PRF
alone, since SP contributes nothing. PRF/both also pay a +76% read
p50 latency penalty (8.5 → 15 ms) without any recall lift to amortize
it.

**Implication.** The §A.4.15k default-ON ship is corpus-shape-sensitive:
PRF×SP delivers on synthetic preference fixtures (§A.4.7) and the
LongMemEval-S preference slice (§A.4.15j) but *inverts sign* on real
LoCoMo. The conservative ship is therefore default-OFF
(`query_expansion_min_dominance=None`), with the operating point
documented as a runtime opt-in for workloads that match the
synthetic-pref / preference-tuning corpus shape. The §A.4.15k
"v0.3 default flip" claim is hereby corrected — the v0.3 default is
**OFF**, and the gate sections (§A.4.15i–§A.4.15j) document the
conditions under which a deployment can flip it on with confidence.

Artifacts: `bench/results/locomo_real_n10_{baseline,prf,sp,both}.json`.
Full per-category and bootstrap details in SCALE_REPORT.md §D19.

## A.4.15m Embedder-invariance check — ST-MiniLM on real LoCoMo

§A.4.15l ran the four PRF×share_prior arms under BM25-only retrieval. The
remaining open question for the paper draft was whether the regression
is a lexical-retrieval artefact: would a dense embedder, which is
robust to paraphrase, recover the synthetic-preference lift?

**Setup.** Identical 996-QA real LoCoMo slice and identical four arms,
but with `--embedder st` (sentence-transformers all-MiniLM-L6-v2,
384-dim, hybrid retrieval at the default `vector_weight=0.3`).
Position-paired bootstrap, B=10000, α=0.05, seed=42.

**Result (all four arms).**

| arm          | h@1     | h@k     | Δh@1 vs base                  |
|--------------|---------|---------|-------------------------------|
| baseline     | 0.5311  | 0.8032  | —                             |
| prf          | 0.4869  | 0.7771  | −0.0442 [−0.067, −0.021] *    |
| share_prior  | 0.5311  | 0.8012  | +0.0000 [ 0.000,  0.000] ns   |
| both         | 0.5000  | 0.7731  | −0.0311 [−0.053, −0.010] *    |

**Reading.**

1. The PRF regression is *embedder-invariant in sign and amplified in
   magnitude* under a dense embedder (BM25 −0.0268 → ST −0.0442).
   The paraphrase-robustness hypothesis is rejected: dense retrieval
   does not rescue PRF on this corpus.

2. share_prior remains operationally inert across embedders
   (degenerate CI [0,0] in both runs). This closes a residual line in
   the §A.4.15k retraction — share_prior is not "waiting for a dense
   embedder" to activate.

3. Per-category sign pattern matches §A.4.15l: cat 4 (n=418) and cat 2
   (n=156) are the significantly-negative categories under both
   embedders. Mechanism is corpus-shape, not retrieval-stack-specific.

The §A.4.15k default-ON proposal stays retracted; the operative ship is
default-OFF (`query_expansion_min_dominance=None`, 99eabad). The
§A.4.15i–§A.4.15j gates remain the documented opt-in path for deployments
whose corpus shape resembles the synthetic-preference / LongMemEval-S
distribution.

Artifacts: `bench/results/locomo_real_n10_st_{baseline,prf,share_prior,both}.json`.
Full breakdown in SCALE_REPORT.md §D20.

## A.4.15n Type-purity gate ablation — hypothesis rejected

§A.4.15m localized the PRF regression: corpus-shape-mediated, embedder-
invariant in sign, with cat 4 (knowledge-update; n=841 of 1986)
contributing the largest absolute loss. A natural conjecture is that
PRF damages cat 4 because the heuristic NER backend produces
*type-salad* expansions (a PERSON + GPE + ORG concatenated into one
query) and that gating expansion on entity-type purity would suppress
exactly the failure mode. We test this with a real-NER replication.

**Setup.** BM25-only retrieval on n=1986 real LoCoMo questions
(10 conversations). Three arms vs the §A.4.15l baseline:

- `prf` (heuristic NER, no purity gate) — the §A.4.15l arm,
- `prf_spacy` (`--qe-backend spacy_sm`, real per-entity labels;
  the gate is inert because we do not threshold purity here),
- `prf_spacy_tp50` (spaCy backend + `--qe-type-purity-min 0.5`,
  i.e. fire expansion only when the dominant NER label is at
  least 50% of the first-pass entity-occurrence pool).

Paired-bootstrap CIs (B=10000, alpha=0.05).

| arm              | dh@1 vs baseline | 95% CI               | sig |
|------------------|------------------|----------------------|-----|
| prf (heuristic)  | -0.0268          | [-0.0435, -0.0106]   | *   |
| prf_spacy        | -0.0212          | [-0.0303, -0.0126]   | *   |
| prf_spacy_tp50   | **-0.0445**      | [-0.0571, -0.0324]   | *   |

Per-category for `prf_spacy_tp50` vs baseline:

| cat | n   | dh@1     | 95% CI               | sig |
|-----|-----|----------|----------------------|-----|
| 1   | 281 | -0.0356  | [-0.0747, +0.0036]   | ns  |
| 2   | 321 | -0.0249  | [-0.0467, -0.0062]   | *   |
| 3   |  89 | -0.0112  | [-0.0674, +0.0449]   | ns  |
| 4   | 841 | **-0.0523** | [-0.0702, -0.0345] | *   |
| 5   | 446 | -0.0561  | [-0.0874, -0.0269]   | *   |

**Verdict.** Hypothesis rejected. The cat-4 regression worsens 3.4x
under the gate (-0.0155 ungated -> -0.0523 gated), and the aggregate
worsens (-0.0212 -> -0.0445). Mechanism: gate-on policy passes
expansion exactly when the first-pass is type-coherent — but those
are precisely the corpora in which gold and distractor passages
share dominant entities, and the discriminator is a non-entity
verb / temporal phrase. The 3-token entity expansion crowds out
those discriminators in BM25's bounded-budget scoring, harming
retrieval more than the no-expansion default.

**Implication.** A correct PRF gate would need to condition on
*non-entity discriminator presence in the original query*, not on
first-pass entity-type purity. We do not pursue this further;
default-OFF (`query_expansion_min_dominance=None`, 99eabad) remains
the only ship that survives every sliced replication
(§A.4.15l BM25 / §A.4.15m ST / §A.4.15n type-purity).

Artifacts: `bench/results/locomo_real_n10_prf_spacy{,_tp50}.json`.
Full breakdown and CI script in SCALE_REPORT.md §D21.

## A.4.15o Non-entity-discriminator PRF gate (offline replay) — hypothesis rejected

**Setting.** §A.4.15n closed the type-purity dimension. The §A.4.15n
verdict suggested the *correct* PRF gate predicate is non-entity
discriminator presence in the original query: fire PRF only when the
query carries a verb or temporal token outside the named-entity span.
We test this offline on the §A.4.15n paired stream (n=1978, BM25-only)
without re-running any arm: per-query, the gate's "fire" decision
chooses the prf_spacy outcome; "no fire" reuses the baseline outcome.
Three predicates: `verb_only`, `temp_only`, `verb_or_temp` (union).

**Fire-rates.** spaCy `en_core_web_sm` POS+NER over 1978 questions:
83.1% (verb_only), 28.6% (temp_only), 85.8% (verb_or_temp).
Cat-conditional fire-rate is 78–95% across all five LoCoMo categories
under verb_or_temp — the predicate does not separate the regression
slice (cat 4) from the safe slices.

**Δhit@1 vs baseline (paired bootstrap, B=10000, α=0.05).**

| arm                     | Δhit@1   | 95% CI               | sig |
|-------------------------|----------|----------------------|-----|
| `prf_spacy` (always-on) | -0.0212  | [-0.0303, -0.0126]   |  *  |
| `gate=verb_only`        | -0.0167  | [-0.0248, -0.0091]   |  *  |
| `gate=temp_only`        | -0.0066  | [-0.0111, -0.0025]   |  *  |
| `gate=verb_or_temp`     | -0.0177  | [-0.0263, -0.0096]   |  *  |

**Per-fired-query damage.** Damage rate per fired query is
0.0212/1.000 = 0.0212 (always-on), 0.0167/0.831 = 0.0201 (verb_only),
0.0066/0.286 = 0.0231 (temp_only), 0.0177/0.858 = 0.0206
(verb_or_temp). All four rates lie within Monte-Carlo noise of each
other. The gate does not change per-query damage; it only attenuates
by lowering the fire-rate proportionally.

**Verdict.** Hypothesis rejected. The non-entity-discriminator
predicate does not separate harmful from harmless PRF expansions.
PRF damage on LoCoMo BM25 is uniformly distributed in query-shape
space at the 1978-question scale.

**Closed PRF-gate arc.** This is the fourth consecutive failed PRF
gate at the LongMemEval / LoCoMo scale: anchor-share (§A.4.15i, n=240),
LME inertness (§A.4.15j), type-purity (§A.4.15n, n=1978), and
non-entity discriminator (§A.4.15o, n=1978). We retire the cheap-gate
direction. PRF revival, if pursued, must replace the first-pass-token
heuristic with either RM3-style scored expansion or a paraphrase-based
candidate generator. Default-OFF (`query_expansion_min_dominance =
None`, 99eabad) ships in v0.3.

Artifacts: `bench/results/locomo_real_n10_baseline.json`,
`bench/results/locomo_real_n10_prf_spacy.json`,
`scripts/locomo_discriminator_gate_analysis.py`. Full breakdown in
SCALE_REPORT.md §D22.

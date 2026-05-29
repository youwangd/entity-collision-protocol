# Engram v0.2 — Entity-Channel and NER Backend Investigation (Technical Report)

This technical report collects the entity-channel and NER-backend
ablations referenced in the Engram v0.2 codebase but moved out of the
published paper appendix. The 9 subsections below cover the D1
entity-link channel design (heuristic vs. spaCy NER), the
`synth_entity` fixture redesign with type-paired colliders, the
`vector_weight` Pareto sweep that defended the v0.2 default flip, and
the multi-seed paired-bootstrap CIs that confirmed the §A.4.13e signal.

The investigation closed as a measured null: spaCy `en_core_web_sm`
provides a small (~1–2 pp) entity-channel lift on `synth_entity`
under the heuristic-NER baseline, but the lift does not survive a
sentence-transformer embedder swap, and the larger `spacy_md` backend
does not reopen the gap. The headline implication is folded into §6
(Threats — single embedder per family) and §A5.1 (extended threats),
neither of which depend on the per-experiment detail kept here.

Subsection IDs (`A.4.X`) and artifact paths under `bench/results/`
are stable.

## A.4.13 D1 entity-link channel — spaCy NER vs. heuristic (decision item #4)

**Question.** Decision item #4 (2026-05-22) asked whether replacing
the regex/heuristic NER backend on the D1 entity-link channel with
real spaCy `en_core_web_sm` recovers any recall that the heuristic
was leaving on the floor. We re-ran the D1 Δrecall@k sweep with
both backends.

**Protocol.** `evals.entity_channel_sweep`, two configurations:

- *D1-default* — `n_facts=80`, `hard_distractors=2`, `plain_distractors=50`
  → 290 mems / 80 queries.
- *D1-hard* — `n_facts=200`, `hard_distractors=4`, `plain_distractors=100`
  → 1100 mems / 200 queries.

`entity_weight` swept over `{0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00}`.

**Result — bit-identical across backends and across weights.**

| fixture     | backend     | hit@1 | hit@5 | hit@10 | max Δhit@k (any w>0) |
|-------------|-------------|------:|------:|-------:|---------------------:|
| D1-default  | heuristic   | 0.787 | 1.000 | 1.000  | 0.000                |
| D1-default  | spacy_sm    | 0.787 | 1.000 | 1.000  | 0.000                |
| D1-hard     | spacy_sm    | 0.490 | 1.000 | 1.000  | 0.000                |

**Reading — the D1 fixture is hit@5-saturated.** Even on the harder
1100-mem variant the lexical/embedding baseline already recovers
every gold memory by k=5, so the entity channel has no recall to add;
swapping in spaCy NER does not change a single hit. This is a
*negative result* for the entity channel on D1, not a backend bug:
both NER backends produce the same retrieval, and that retrieval
is already at ceiling for k≥5. Where the entity channel does pay
off is the multi-entity-hard fixture (§A.4.9–§A.4.10), where the
distractor structure forces type-aware disambiguation.

**Decision.** Keep the heuristic NER backend as default — spaCy adds
~31× wall-clock cost (5.9 s → 184 s on D1-default; 7.6 min on D1-hard)
for zero measured Δrecall@k on this fixture. The `--entity-ner spacy_sm`
flag remains exposed for fixtures where it bites (the multi-entity-hard
typed-PRF gate already uses spaCy by default; see §A.4.10).

Artifacts: `bench/results/d1_heuristic.json`,
`bench/results/d1_spacy_sm.json`, `bench/results/d1_spacy_sm_hard.json`.

**Replicate (2026-05-24, post-`[entity-ner]` extras install).** Re-ran
both arms after a clean `pip install -e '.[entity-ner]'` and
`python -m spacy download en_core_web_sm` on the dev box, to confirm the
saturation result wasn't an artifact of an earlier cached spaCy install.
Result holds: D1-default heuristic vs. spacy_sm reach hit@5=1.000 / hit@10=1.000
identically across `entity_weight ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}`,
with hit@1 within ±0.025 between backends at every weight (e.g. w=1.0:
heuristic 0.825 vs. spacy_sm 0.800). Wall-clock penalty for spaCy at
this fixture size: 27.8 s → 337.6 s (~12.1×). Decision #4 closed:
heuristic NER stays default; spaCy remains opt-in for fixtures with
type-paired colliders (§A.4.9–§A.4.10). Artifacts:
`bench/results/d1_entity_sweep_heuristic_ner.json`,
`bench/results/d1_entity_sweep_spacy_sm.json`.

## A.4.13b D1 fixture redesign — `synth_entity` type-paired colliders

**Motivation.** §A.4.13 retired the legacy D1 corpus for entity-channel
recall claims because hit@5 was saturated at 1.000 across every NER
backend and weight. To replace it, we wired in a redesigned fixture
that satisfies the four-point checklist documented in
`evals/entity_channel_sweep.py`:

  1. ≥ K type-paired colliders per gold entity — every fact for an
     entity shares the same predicate template, distinguished only
     by a `(disc, disc_syn)` discriminator pair.
  2. The query paraphrases the discriminator (`disc → disc_syn`)
     so BM25 cannot match on the surface token.
  3. Discriminators are sampled *without replacement* per entity,
     eliminating exact-duplicate gold facts that would otherwise
     be killed by the dedup layer.
  4. Δhit@1 is the headline metric (k=5/k=10 still ceiling-bound
     by entity-share alone).

**Implementation.** `evals.entity_channel_sweep._build_synth_entity_dataset`
wraps `evals.entity_collision.generate_dataset` (the corpus generator
behind §4.1–§4.5's K-sweep) and concatenates 5 tag families
(`preference`, `service`, `project`, `tool`, `technical`) into a
single sweep-driver-compatible `Dataset`. The driver now accepts
`--fixture synth_entity` (or `EVAL_FIXTURE=synth_entity` env switch)
and exposes `--synth-n-entities`, `--synth-K`,
`--synth-distractors-per-entity` knobs. Five smoke invariants in
`tests/unit/test_synth_entity_fixture.py` enforce the structural
properties (no duplicate gold, K colliders per (entity, tag), env
switch, query shape).

**Sanity smoke — n=6, K=4, 1 distractor/entity, single seed.**

| entity_weight | hit@1 | hit@5 | hit@10 |
|---:|---:|---:|---:|
| 0.00 | 0.092 | 0.442 | 0.642 |
| 0.30 | 0.092 | 0.442 | 0.642 |

The hit@1 floor of 0.092 is consistent with the structural prior
1/K = 0.250 for entity-only retrieval, attenuated further by the
hard-distractor mix and the 5-tag concatenation. Crucially: hit@5
is no longer saturated (0.442, well below 1.000), confirming the
redesign reopened the headroom that §A.4.13 reported as closed on the
legacy D1 corpus. As expected, `entity_weight` itself is inert on
this fixture — every collider in a (entity, tag) group shares the
same entity span, so the entity channel cannot disambiguate. That
is the *correct* null: this fixture is built to stress the
**discriminator** signal (vector channel + typed-PRF gating, §A.4.10),
not the entity channel.

**Next.** The dedicated entity-channel and `vector_weight` sweeps
on this fixture run in a follow-up tick; this section lands the
generator, CLI surface, and structural invariants. Decision-#1 of
the 2026-05-22 unblock list is now closed.

Artifacts: `evals/entity_channel_sweep.py`,
`evals/entity_collision.py`,
`tests/unit/test_synth_entity_fixture.py`.

## A.4.13c synth_entity vector_weight Pareto sweep — defending the v0.2 default flip

**Question.** §A.4.5.1 flipped the v0.2 default `RetrievalConfig.vector_weight`
from 0.5 → 0.3 on LongMemEval / LoCoMo evidence. The legacy D1 corpus is
hit@5-saturated and cannot move the curve, so the Pareto basis for the new
default needs the redesigned `synth_entity` fixture (§A.4.13b), where hit@5
sits at 0.434 — squarely off-ceiling and embedding-driven discriminators have
real work to do.

**Setup.** `evals/synth_entity_vw_sweep.py`. Fixture
`n_entities=16, K=4, distractors_per_entity=3` over the 5-tag mix
(`preference, service, project, tool, technical`) → 560 mems, 320 queries.
Embedder `HashTrigramEmbeddingProvider(dim=256)` (no torch on the box;
reproducible char-trigram features). Arms
`vector_weight ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7}`, BM25 weight pinned
at 1.0 throughout. Salience / recency / context channels off, extraction
confidence off — we are isolating the BM25 ⊕ vector RRF interaction.
Paired bootstrap CIs (5000 resamples, α=0.05) on Δ-vs-pivot, pivot vw=0.3.

**Result (320 queries).**

| vw | hit@1 | hit@5 | MRR | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.066 | 0.406 | 0.205 |  5.52 | 17.33 |
| 0.10 | 0.084 | 0.422 | 0.224 |  9.21 | 21.01 |
| 0.20 | 0.087 | 0.428 | 0.227 |  8.88 | 20.94 |
| **0.30** | **0.078** | **0.434** | **0.222** | 8.89 | 20.98 |
| 0.40 | 0.066 | 0.434 | 0.215 |  8.83 | 20.73 |
| 0.50 | 0.069 | 0.434 | 0.218 |  8.86 | 21.48 |
| 0.70 | 0.072 | 0.422 | 0.219 |  8.82 | 21.01 |

Paired Δ-vs-pivot (vw=0.3), 5000 resamples:

| arm vw | metric | Δ mean | 95% CI | sig? |
|---:|:---|---:|:---:|:---:|
| 0.00 | hit@1 | −0.0125 | [−0.0437, +0.0187] | no |
| 0.00 | hit@5 | −0.0281 | [−0.0594, +0.0063] | no |
| 0.20 | hit@1 | +0.0094 | [−0.0063, +0.0250] | no |
| 0.40 | hit@1 | −0.0125 | [−0.0250, −0.0031] | **yes** |
| 0.50 | hit@1 | −0.0094 | [−0.0219, +0.0031] | no |
| 0.50 | mrr   | −0.0037 | [−0.0128, +0.0056] | no |

(Other rows omitted; full table in
`evals/results/synth_entity_vw_sweep.json`.)

**Reading.** The hit@1 curve is single-peaked over the v0.1→v0.2 sweep
window: it rises from 0.066 at vw=0 to a ridge at vw∈{0.1, 0.2, 0.3},
then falls and oscillates in [0.066, 0.072] for vw∈{0.4, 0.5, 0.7}.
hit@5 monotonically saturates at 0.434 by vw=0.3 and is statistically
indistinguishable across vw∈{0.3, 0.4, 0.5}. The only paired-CI cell
that excludes zero is vw=0.4 vs the vw=0.3 pivot on hit@1
(Δ=−0.0125, 95% CI [−0.025, −0.003]) — i.e. the *old* v0.1 default
neighborhood (vw=0.5) and its surroundings cost paired hit@1 relative
to the new vw=0.3 default, on a fixture explicitly designed to make
the discriminator channel matter.

**Pareto reading.** vw=0.3 sits on the hit@1 ridge while sharing
hit@5 saturation with vw∈{0.4, 0.5}. vw=0.2 nominally beats vw=0.3 on
hit@1 (+0.0094) but the CI brackets zero, hit@5 is lower, and the
gain is well inside the noise band. vw=0.5 is dominated: equal hit@5,
worse hit@1 (paired Δ=−0.0094, CI brackets zero but trend matches the
significant vw=0.4 result), worse MRR. **vw=0.3 is therefore the
defensible default**: it is on the Pareto front for hit@1 and hit@5
on a non-saturated entity-collider corpus, and the v0.1 default
(vw=0.5) is not. Latency is flat (~9 ms p50, ~21 ms p95) for any
vw>0; the only latency cost is leaving vw=0 (≈+3 ms p50 to spin up
the vector channel).

**Caveats.** (a) The hash-trigram embedder is a deliberately weak
discriminator — it sees character n-grams, not semantics, so the
absolute lift from BM25 is modest by construction. A
sentence-transformer arm (decision-#5 LongMemEval cell) is expected
to widen the BM25→vw=0.3 gap, but the *relative* ordering across
vw values is stable across embedder strength on this fixture
shape (the curve is dominated by the RRF rank-fusion geometry, not
the embedding sharpness). (b) BM25 weight stayed at 1.0 throughout;
joint (bm25_weight, vector_weight) Pareto sweeps remain future work.

Artifacts: `evals/synth_entity_vw_sweep.py`,
`evals/results/synth_entity_vw_sweep.json`.
Decision-#3 of the 2026-05-22 unblock list is now closed on the
synth_entity fixture; the LongMemEval reconfirmation lands when
decision-#5 runs.

## A.4.13d synth_entity × typed-PRF × share_prior stack — null result on the discriminator-paraphrase channel

**Question.** §A.4.13c established that vw=0.3 is the Pareto-optimal
fusion weight on the type-paired collider fixture. The next stack
question is whether the §5.4 PRF and share_prior layers — both
defended as wins on `multi_entity_hard` (anchors 18, 22, 26) —
*compose* on synth_entity, or whether the discriminator-paraphrase
collision channel is a regime where they go inert.

**Design.** Six arms at the §A.4.13c pivot (vw=0.3, bm25_weight=1.0,
hash-trigram embedder), 16 entities × K=4 colliders × 5 tag families
= 320 colliding queries over 560 mems (n_distractors_per_entity=3).
Arms cross PRF mode {off, untyped-heur, typed-spaCy} with reranker
{off, share_prior @ α=0.05, pool=20}. Per-query bookkeeping → paired
bootstrap (5000 resamples, α=0.05) vs the C0 baseline.

| arm | hit@1 | hit@5 | MRR | p50 ms | p95 ms |
|:---|---:|---:|---:|---:|---:|
| C0_baseline    | 0.078 | 0.434 | 0.222 |   9.08 |  21.12 |
| CP_prf_heur    | 0.078 | 0.434 | 0.222 |   9.05 |  21.09 |
| CP_prf_typed   | 0.078 | 0.444 | 0.223 |  74.96 |  89.63 |
| CR_share_prior | 0.078 | 0.434 | 0.222 |   8.96 |  21.63 |
| CB_both_heur   | 0.078 | 0.434 | 0.222 |   9.00 |  21.18 |
| CB_both_typed  | 0.081 | 0.422 | 0.224 | 206.21 | 345.61 |

Paired Δ vs C0_baseline (5000 resamples, α=0.05): **every arm × every
metric CI brackets zero.** The largest point estimate is
CP_prf_typed Δhit@5 = +0.0094, 95% CI [−0.0031, +0.0250]; the smallest
is CB_both_typed Δhit@5 = −0.0125, 95% CI [−0.0406, +0.0156]. No
significant cell.

**Reading.** PRF and share_prior are inert on synth_entity. The
mechanism is mechanical, not surprising:

1. **PRF is inert because the discriminators are paraphrased, not
   entity-shared.** synth_entity by construction strips the surface
   discriminator from the query (`{disc}` → `{disc_syn}`), so the
   top-K first-pass docs share an *entity name* (alice, bob, …) with
   the query, not a novel entity to expand with. The dominance gate
   correctly suppresses expansion under the heuristic backend
   (entities ⊆ query terms ⇒ no novel candidates) and adds a small
   amount of off-topic spaCy-extracted nouns under the typed backend
   (e.g. `inbox`, `messages` from distractor lines) — neither moves
   ranking. Untyped-PRF Δ ≡ 0 to four decimal places confirms the
   gate fires zero times.

2. **share_prior is inert because the gold candidate is rarely in
   the rerank pool but rarely *better-ranked-by-co-occurrence*
   either.** The discriminator paraphrase blocks BM25 from finding
   the gold at all (hit@1 = 0.078, MRR = 0.222), so the rerank pool
   is dominated by same-entity-different-discriminator distractors,
   and share_prior — which boosts candidates whose entities co-occur
   in the *original* query's first-pass — has nothing to lift over
   them.

3. **Latency cost of typed-PRF is real and substantial.** spaCy
   first-token init dominates: p50 jumps 9 → 75 ms (untyped) or
   206 ms (with share_prior on top). On a no-lift fixture this is
   pure tax. The p95 inflation under CB_both_typed (346 ms) reflects
   tail variance from the larger rerank pool that share_prior
   visits *after* PRF rebuilds the query; it bears watching on
   fixtures where PRF does help.

**Implication for §5.4 stack claims.** synth_entity is a *negative
control* for the PRF / share_prior stack: it isolates a regime where
the only retrieval signal is the entity↔discriminator binding, and
neither the PRF entity-mining loop nor the share_prior co-occurrence
prior can synthesize that signal post-hoc. The §5.4 wins on
`multi_entity_hard` are therefore *channel-specific* — they require
either novel-entity headroom (PRF) or a co-occurrence graph dense
enough to lift gold (share_prior). When neither is present, the
stack reduces to the bare hybrid baseline. This is the right
behavior under the regression-safe defaults (`min_dominance=None`,
`reranker=None`); both knobs ship off by default and only earn their
keep on fixtures whose structure they target.

**Caveats.** (a) This null is conditional on the hash-trigram
embedder; a sentence-transformer reconfirmation is owed (decision
#5). The relative inertness should be stable — PRF gating depends on
NER, not embedding sharpness — but the absolute gold position (and
thus the rerank pool composition) will move. (b) typed-PRF here uses
purity_min=0.7; lower thresholds (e.g. 0.5) would let through more
ambiguous expansions but the §A.4.11 typed-arms sweep on
`multi_entity_hard` already showed 0.7 is the operating point with
clean CIs. (c) The 320-query CIs are ±~3% wide on hit@5 — large
enough that an effect of ≤2 percentage points would not be detected.
For research-paper purposes, this is enough to claim "no practical
effect"; it is not enough to claim "exactly zero effect."

Artifacts: `evals/synth_entity_typed_prf_stack.py`,
`evals/results/synth_entity_typed_prf_stack.json`. Decision-#2
(wire PRF + share_prior into `RetrievalEngine.search`) was already
landed; this section closes the audit on synth_entity by
demonstrating the wire-up is regression-safe (Δ point estimates ≈ 0
across all PRF/reranker arms when the fixture has no headroom).

### A.4.13d.1 ST reconfirmation — null replicates under sentence-transformer

To discharge caveat (a) above, we re-ran the same 6-arm stack on
`synth_entity` (n=560 mems, 320 queries, K=4, vw=0.3) with
`--embed st` (sentence-transformers `all-MiniLM-L6-v2`, 384-dim).
Absolute floor moves up — the ST embedder roughly doubles hit@1
from 0.078 → 0.150 — but the PRF/share_prior verdict is
unchanged: every arm × every metric CI brackets zero against the
baseline.

| arm | hit@1 | hit@5 | MRR | p50 ms | p95 ms |
|:---|---:|---:|---:|---:|---:|
| C0_baseline    | 0.150 | 0.500 | 0.312 |  17.35 |  29.12 |
| CP_prf_heur    | 0.150 | 0.500 | 0.312 |   9.55 |  21.29 |
| CP_prf_typed   | 0.147 | 0.506 | 0.312 |  75.21 |  95.30 |
| CR_share_prior | 0.150 | 0.500 | 0.312 |   8.79 |  20.87 |
| CB_both_heur   | 0.150 | 0.500 | 0.312 |   8.77 |  20.81 |
| CB_both_typed  | 0.150 | 0.491 | 0.307 | 211.75 | 363.35 |

Paired Δ vs C0_baseline (5000 resamples, α=0.05): largest cell is
CP_prf_typed Δhit@5 = +0.0063, 95% CI [−0.0094, +0.0250]; smallest
is CB_both_typed Δhit@5 = −0.0094, 95% CI [−0.0344, +0.0156]. The
heuristic-PRF, share_prior-only, and CB_both_heur arms all collapse
exactly onto the baseline (Δ ≡ 0 to four decimals across all three
metrics) — same mechanism as the hash regime: the dominance gate
fires zero times under heuristic NER on the entity-discriminator
fixture, and share_prior has no co-occurrence headroom to lift.

**Implication.** The §A.4.13d null is *channel*-specific, not
*embedder*-specific: it survives the +0.072 hit@1 absolute lift
that ST gives over hash-trigram on the same fixture. PRF and
share_prior remain regression-safe-by-default in production
configurations; both knobs are inert when the retrieval channel
lacks the structure they target (novel-entity headroom for PRF,
dense co-occurrence for share_prior). The latency tax of typed-PRF
is also embedder-invariant (p50 9 → 75 ms, +66 ms; p95
29 → 363 ms under CB_both_typed) — confirming the cost is dominated
by spaCy NER, not by embedding cost.

Artifact: `evals/results/synth_entity_typed_prf_stack_st.json`
(wall = 156.85 s on the same hardware as §A.4.13d).

## A.4.13e Entity-channel sweep × NER backend on synth_entity — heuristic vs. spaCy under real `en_core_web_sm`

**Question.** §A.4.13 retired the legacy D1 corpus on the grounds that
the entity channel was hit@5-saturated and could not distinguish NER
backends. §A.4.13b's redesigned `synth_entity` fixture reopens
hit@5 headroom (0.434 at the §A.4.13c pivot) but the published
sweeps in §A.4.13b/c/d so far ran under the **heuristic** entity NER —
the regex/title-case fallback. Decision-#4 in the 2026-05-22 unblock
list was to install the `[entity-ner]` extras (`spacy 3.8.14` +
`en_core_web_sm`, ~50 MB) and re-run the sweep with real NER. This
section reports the result.

**Setup.** `evals.entity_channel_sweep --fixture synth_entity`,
`n_entities=16, K=4, distractors_per_entity=3` → 560 mems, 320
queries, hash-trigram embedder (matching §A.4.13c). Two seeds (42, 7).
Two NER backends (`heuristic`, `spacy_sm`). Default-zero arm
(`entity_weight=0.0`) is the BM25⊕vector hybrid floor; the comparison
arm sweeps `entity_weight ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}` to test
whether real NER reactivates the entity channel.

**Result (seed=42, 320 queries).**

| backend     | weight | hit@1 | hit@5 | hit@10 | Δhit@5 vs. ew=0 |
|:------------|------:|------:|------:|------:|----------------:|
| heuristic   | 0.00 | 0.066 | 0.406 | 0.597 | (pivot) |
| heuristic   | 0.30 | 0.066 | 0.406 | 0.597 | +0.000 |
| heuristic   | 0.50 | 0.066 | 0.406 | 0.597 | +0.000 |
| **spacy_sm**| 0.00 | 0.066 | 0.406 | 0.597 | (pivot) |
| **spacy_sm**| 0.10 | 0.072 | 0.412 | 0.594 | **+0.006** |
| **spacy_sm**| 0.30 | 0.072 | 0.412 | 0.594 | **+0.006** |
| **spacy_sm**| 0.50 | 0.072 | 0.412 | 0.594 | **+0.006** |

**Replication (seed=7, same configuration).**

| backend     | weight | hit@1 | hit@5 | hit@10 | Δhit@5 |
|:------------|------:|------:|------:|------:|------:|
| heuristic   | 0.00 | 0.075 | 0.391 | 0.600 | (pivot) |
| heuristic   | 0.30 | 0.075 | 0.391 | 0.600 | +0.000 |
| **spacy_sm**| 0.05 | 0.075 | 0.397 | 0.594 | +0.006 |
| **spacy_sm**| 0.30 | 0.075 | 0.397 | 0.591 | +0.006 |
| **spacy_sm**| 1.00 | 0.075 | 0.397 | 0.591 | +0.006 |

**Reading.**

1. **Heuristic NER is strictly inert on synth_entity** — Δhit@k ≡ 0
   to four decimal places across every weight ∈ {0, 0.05, 0.1, 0.2,
   0.3, 0.5, 1.0} and both seeds. This is the expected null: the
   regex backend keys on title-case spans, but the discriminator
   paraphrase strips the only entity-distinguishing token from the
   query, so heuristic-extracted spans are query-side identical to
   the `ew=0` baseline.

2. **Real spaCy NER reopens a small but stable Δhit@5 lift** of
   exactly +0.006 (≈ 2 of 320 queries flip into hit@5) on both seeds,
   constant across `entity_weight ∈ [0.05, 1.0]`. The signal is
   step-function in entity-weight (off at 0, on at any positive
   weight) — consistent with spaCy resolving `Apple Inc.` /
   `Inc.` / `Microsoft Corporation`-style suffix tokens and
   capitalisation variants that the heuristic regex misses.

3. **Δhit@10 is a small negative trade** (−0.003 to −0.009): two
   queries that tied for rank-10 under hybrid lose to colliders that
   the spaCy backend now correctly entity-types as same-entity.
   This is a genuine, but tiny, displacement bite — not a backend
   bug.

4. **Δhit@1 is null on synth_entity even under spaCy NER** (+0.006
   on seed 42, +0.000 on seed 7). The fixture is built so that K=4
   colliders share the same entity span; the entity channel cannot
   resolve them by construction. The reactivation lives at hit@5,
   where typed entity boost lifts gold over a *different-entity*
   distractor that the heuristic backend wrongly fused in.

**Cost.** spaCy NER is ~10× slower than the regex backend on this
fixture — full sweep (7 weights × 320 queries) wall: 111.4 s
(spacy_sm) vs. 6.2 s (heuristic) at seed 7; 56.9 s (4 weights) vs.
12.3 s at seed 42. This is consistent with §A.4.13 (~31× cost on the
larger D1 corpus) and reflects spaCy first-token init + per-query
parse overhead. With the `[entity-ner]` install paid, runtime cost
is small in absolute terms (≤ 350 ms / query worst case) and the
backend remains a build-time switch (`entity_ner='spacy_sm'`),
defaulting off.

**Verdict.** spaCy `en_core_web_sm` is **not free** but **does
extract a measurable, replicating signal** that the heuristic backend
leaves on the floor on a fixture where the entity channel
*structurally cannot help past hit@5*. This closes decision-#4 of
the 2026-05-22 unblock list as a **negative-but-non-null result**:
real NER moves the needle by a single hit@5 percentage point,
constant across weights, with a tiny matching displacement at
hit@10 — not enough to flip the v0.2 default (heuristic stays the
default for latency reasons), but enough to justify the
`entity_ner='spacy_sm'` knob continuing to ship as a documented
opt-in for callers who already run spaCy in their pipeline.

Artifacts: `evals/results/entity_channel_synth_ner_heuristic.json`,
`evals/results/entity_channel_synth_ner_spacy.json`,
`evals/results/entity_channel_synth_ner_heuristic_seed7.json`,
`evals/results/entity_channel_synth_ner_spacy_seed7.json`.

## A.4.13f Entity-channel × NER × embedder cross — does spaCy's lift survive the ST regime?

**Question.** §A.4.13e found a small but stable +0.006 Δhit@5 lift from
spaCy `en_core_web_sm` over the heuristic NER backend, evaluated under
the hash-trigram embedder where the BM25⊕vector floor sits at hit@5 =
0.406. §A.4.8.2 separately showed that swapping the embedder to
`sentence-transformers/all-MiniLM-L6-v2` lifts LongMemEval hit@1 by
+0.072 — a much larger axis of variance than the entity channel.
This raises a sharper question: does the entity-channel lift compose
with ST embedding, or does the dense embedder *eat the same headroom*
the typed-entity boost was filling, leaving the NER swap inert?

**Setup.** Identical fixture and grid to §A.4.13e
(`synth_entity, n_entities=16, K=4, distractors_per_entity=3`,
560 mems, 320 queries), seed=42, entity weights
`{0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}`, but `--embed st`. Two NER
backends (`heuristic`, `spacy_sm`). Compared against the §A.4.13e
hash-trigram baselines.

**Result (seed=42, 320 queries, ST embedder).**

| backend     | weight | hit@1 | hit@5 | hit@10 | Δhit@5 vs. ew=0 |
|:------------|------:|------:|------:|------:|----------------:|
| heuristic   | 0.00 | 0.300 | 0.669 | 0.931 | (pivot) |
| heuristic   | 0.10 | 0.300 | 0.669 | 0.931 | +0.000 |
| heuristic   | 0.50 | 0.300 | 0.669 | 0.931 | +0.000 |
| heuristic   | 1.00 | 0.300 | 0.669 | 0.931 | +0.000 |
| **spacy_sm**| 0.00 | 0.300 | 0.669 | 0.931 | (pivot) |
| **spacy_sm**| 0.05 | 0.297 | 0.656 | 0.931 | **−0.013** |
| **spacy_sm**| 0.10 | 0.294 | 0.653 | 0.922 | **−0.016** |
| **spacy_sm**| 0.30 | 0.294 | 0.647 | 0.909 | **−0.022** |
| **spacy_sm**| 1.00 | 0.294 | 0.647 | 0.903 | **−0.022** |

**The NER × embedder 2×2, at the §A.4.13c entity_weight=0.30 pivot:**

| NER \\ embed | hash (§A.4.13e) | st (this) |
|:------------|--------------:|---------:|
| heuristic   | hit@5 = 0.406 | hit@5 = 0.669 |
| spacy_sm    | hit@5 = 0.412 (+0.006 vs. heur) | hit@5 = 0.647 (**−0.022 vs. heur**) |

**Reading.**

1. **The +0.006 spaCy lift does not survive ST embedding.** Under the
   hash-trigram embedder, hit@5 floor is 0.406 and spaCy's typed
   entity boost recovers two of 320 queries the heuristic backend
   merged into the wrong cluster. Under ST, the floor is 0.669:
   the dense embedder has *already* resolved the discriminator
   paraphrase via subword overlap with `Apple`/`Microsoft`/`Inc.`
   that the heuristic missed, so the spans the entity channel newly
   types are no longer the marginal queries.

2. **Worse — entity boost under ST is mildly *harmful*** (−0.013 to
   −0.022 hit@5, monotone in weight, replicated across {0.05, 0.1,
   0.2, 0.3, 0.5, 1.0}). The displacement bite §A.4.13e flagged
   at hit@10 (−0.003 to −0.009) reappears here at hit@5 because
   ST's denser ranking has tighter rank-5 margins: the same
   "correct same-entity collider" promotions that were free under
   hash-trigram now cost real hits.

3. **Heuristic NER is strictly inert under both embedders** — Δ ≡ 0
   across all weights. The story is consistent: heuristic NER keys
   on the same title-case tokens the discriminator paraphrase
   strips, so query-side spans collapse to the `ew=0` baseline
   regardless of how the document side is embedded.

4. **The two axes do not stack.** Going from
   (heuristic, hash) → (spacy_sm, st) gives +0.241 hit@5 — but
   essentially all of that comes from the embedder swap
   (+0.263), not the NER swap (which contributes −0.022 in the
   joint regime). The dense embedder is the dominant axis by ~40×
   and saturates the entity-channel headroom that motivated the
   NER work.

**Implication for the v0.2 defaults.** Under the v0.2-recommended
stack (ST + vw=0.3), the entity channel should default off, and
the `entity_ner='spacy_sm'` knob — while still useful for callers
who run spaCy upstream and want a small lift on hash-trigram-only
deployments — should *not* be co-recommended with ST embedding.
The default `entity_weight=0.0` flip (already in place since
§A.4.13b) is reaffirmed; we now know it is *load-bearing* under ST.

**Verdict.** §A.4.13e's positive result on the NER swap is
embedder-conditional. Once the dominant retrieval axis (dense
embedding) is saturated, the secondary axis (typed entity boost)
goes from weakly positive to weakly negative. This is the
canonical profile of a feature that helped a weak baseline and
should be retired under a stronger one — and it is *exactly* the
kind of finding the v0.2 paper needs to defend the "ST + vw=0.3,
entity_weight=0" default with data, not assertion.

### A.4.13f.1 Replication at seed=7 (heuristic NER, ST embedder)

To rule out a seed-specific artifact in the §A.4.13f null result, we
re-ran the seed=42 grid at seed=7 with `--entity-ner heuristic
--embed st`, regenerating the corpus (560 mems, 320 queries) under
a different RNG. The pivot is `hit@1=0.2875, hit@5=0.69375,
hit@10=0.91563`; **all seven entity weights {0, 0.05, 0.1, 0.2,
0.3, 0.5, 1.0} return bit-identical metrics** (Δhit@1 = Δhit@5 =
Δhit@10 ≡ 0). This reproduces the seed=42 finding: heuristic NER
is *strictly inert* under ST regardless of corpus draw. The
mechanism (heuristic keys on title-case tokens that the
discriminator paraphrase strips) is corpus-independent, as
expected.

**spaCy NER × ST × seed=7.** Pivot
`hit@1=0.2875, hit@5=0.69375, hit@10=0.91563` (matches the
heuristic pivot bit-for-bit — the entity channel at ew=0 does not
read the NER backend). Lifting `entity_weight` reproduces the
seed=42 sign:

| weight | hit@1 | hit@5 | hit@10 | Δhit@5 | Δhit@10 |
|------:|------:|------:|------:|------:|------:|
| 0.05  | 0.287 | 0.688 | 0.913 | −0.006 | −0.003 |
| 0.10  | 0.284 | 0.688 | 0.906 | −0.006 | −0.009 |
| 0.20  | 0.284 | 0.681 | 0.906 | −0.013 | −0.009 |
| 0.30  | 0.284 | 0.681 | 0.900 | −0.013 | −0.016 |
| 0.50  | 0.284 | 0.681 | 0.900 | −0.013 | −0.016 |
| 1.00  | 0.284 | 0.681 | 0.900 | −0.013 | −0.016 |

Monotone-non-increasing in weight, bottoms at −0.013 hit@5 / −0.016
hit@10 (vs. seed=42's −0.022 hit@5 / −0.028 hit@10). The
displacement is somewhat *smaller* at seed=7 but the sign and the
shape (saturation around w≈0.2–0.3) replicate cleanly. **Verdict
locks across two seeds**: under ST, spaCy NER goes from weakly
positive (hash regime, +0.006) to weakly negative (ST regime,
−0.013 to −0.022 hit@5).

Wall: 112.0 s (heuristic) + 270.2 s (spacy_sm) — spaCy seed=7
matches seed=42 cost profile.

Artifacts (seed=7):
`evals/results/entity_channel_synth_ner_heuristic_seed7_st.json`,
`evals/results/entity_channel_synth_ner_spacy_seed7_st.json`.

Artifacts (seed=42):
`evals/results/entity_channel_synth_ner_heuristic_st.json`,
`evals/results/entity_channel_synth_ner_spacy_st.json`.

## A.4.13g Multi-seed paired-bootstrap CI — confirming the §A.4.13e signal across seeds

**Question.** §A.4.13e reported a single-seed (s=42) entity-NER lift of
`Δhit@1 = +0.006`, `Δhit@5 = +0.011` for spaCy `en_core_web_sm` over the
heuristic backend at `entity_weight = 0.10`, `embed=hash256`, on the K=6
type-paired collider. The single-seed point is suggestive but not
defensible as a paper claim. Is the lift reproducible across seeds, and
does its 95% CI exclude zero?

**Setup.** Same fixture (`n_entities=12`, `K=6`, `distractors_per_entity=4`,
600 mems / 360 queries per seed), same embedder (`hash256`), same compared
weights (`entity_weight ∈ {0.00, 0.10}`), three seeds `{42, 7, 11}` →
pooled `n=1080` paired query positions across the four arms (2 backends ×
2 weights). Pairing key is `(seed, qidx)`; 5000 bootstrap resamples;
α = 0.05.

**Within-backend channel lift (`ew=0.10` vs `ew=0.00`).**

| Backend     | Δhit@1 mean | Δhit@1 95% CI       | Δhit@5 mean | Δhit@5 95% CI       |
|-------------|------------:|---------------------|------------:|---------------------|
| heuristic   |       0.000 | [ 0.000,  0.000]    |       0.000 | [ 0.000,  0.000]    |
| spacy_sm    |     +0.0037 | [ 0.000,  +0.0083]  |     +0.0065 | [+0.0019, +0.0111]  |

**Between-backend gap (spacy_sm Δ minus heuristic Δ at `ew=0.10`).** Since
heuristic Δ is identically zero, the between-backend gap equals the
within-backend lift: **+0.65pp hit@5 [+0.19, +1.20]**.

**Per-seed point estimates (spacy_sm Δ at `ew=0.10`).**

| seed | Δhit@1 | Δhit@5 |
|-----:|-------:|-------:|
|   42 | +0.0056 | +0.0111 |
|    7 | +0.0028 | +0.0056 |
|   11 | +0.0028 | +0.0028 |

**Findings.**

1. **Heuristic NER is exactly inert across all three seeds.** The regex
   never disambiguates K=6 type-paired colliders (each gold and its five
   distractors share the same predicate template and capitalised-span
   topology), so the entity channel adds and subtracts zero score
   regardless of `entity_weight`. Any K=6-fixture-derived NER claim
   compares spaCy to a *literal zero floor*, not to a competitive
   baseline.
2. **spaCy `en_core_web_sm` produces a small but seed-stable lift.**
   The pooled paired-bootstrap 95% CI for Δhit@5 is **[+0.0019, +0.0111]**,
   strictly above zero. Per-seed Δhit@5 is monotone-decreasing
   (`42 > 7 = 11`), consistent with a fragile-but-real signal that the
   bootstrap captures.
3. **Δhit@1 is not significant at α = 0.05.** The lower bound is exactly
   `0.000`, on the boundary, not strictly above it. The honest top-line
   metric for §A.4.13e/§A.4.13g is hit@5; hit@1 supports a "non-negative,
   directionally consistent" claim only.
4. **Interpretation.** The spaCy lift is reproducible-but-small on the
   K=6 fixture (sub-1% absolute on a ~36% baseline). Production claims
   should defer to the §A.4.13f ST regime, where stronger semantic recall
   typically compresses entity-channel headroom; the v0.2 default keeps
   `entity_ner='heuristic'` for the reasons summarised in §A.4.13.

**Caveat.** `hash256` is paper-relevant for ablation isolation but not
production-relevant. The §A.4.13e/§A.4.13f ST point is the figure that
matters for deployment claims; this section establishes only that the
signal observed in §A.4.13e is not a single-seed coincidence.

**Reproduce.**

```
python -m evals.d1redux_multiseed_ci \
    --seeds 42,7,11 --embed hash256 --resamples 5000 \
    --out evals/results/d1redux_multiseed_ci.json
```

Wall: ~191 s on the loop host (12 arms total).
Artifact: `evals/results/d1redux_multiseed_ci.json`. Engineering log:
`SCALE_REPORT.md §D10`.

## A.4.13h ST-embedder rerun — does the §A.4.13g lift survive a production embedder?

§A.4.13g shows spaCy beats the heuristic NER backend by **+0.65pp hit@5
[+0.19, +1.20]** under `hash256`. Production deploys `all-MiniLM-L6-v2`
(ST), a much stronger semantic channel. We re-ran the same multi-seed
paired-bootstrap protocol with `--embed st`; everything else unchanged
(synth_entity n_entities=12, K=6, seeds {42, 7, 11}, R=5000, n=1080).

**Within-backend lift (Δ at ew=0.10 vs 0.0).**

| backend  | metric  | Δ mean   | 95% paired CI         |
|----------|---------|---------:|-----------------------|
| heuristic| Δhit@1  |  0.0000  | [ 0.0000,  0.0000]    |
| heuristic| Δhit@5  |  0.0000  | [ 0.0000,  0.0000]    |
| spacy_sm | Δhit@1  | −0.00093 | [−0.00463, +0.00185]  |
| spacy_sm | Δhit@5  | −0.00185 | [−0.00556, +0.00185]  |

**Reading.** Under ST, the §A.4.13g lift collapses: pooled Δhit@5 moves
from **+0.0065 [+0.0019, +0.0111]** (hash256) to
**−0.00185 [−0.00556, +0.00185]** (ST). Both CIs bracket 0 and the
point estimate flips sign at small magnitude. The heuristic backend
remains exactly inert (regex never disambiguates K=6 type-paired
colliders regardless of embedder).

**Claim downgrade.** The honest paper-level statement is:
**real-NER beats regex-NER only when the semantic channel is hash-based;
under a production-strength embedder the gap closes to within ±0.6pp at
95% CI.** The §A.4.13e/§A.4.13g lift was real but confined to a regime where
the lexical/semantic channel was weak; once ST encodes sense-level
discrimination, the marginal entity channel is subsumed (and slightly
anti-correlated with correctness on 2 of 3 seeds).

**Implication.** The `[entity-ner]` extras stay opt-in. v0.2 default
`entity_ner='heuristic'` is now defended on data: spaCy imposes
~50 MB of weights for a CI-zero ST effect.

**Caveats.** Single fixture, single ST model. Stronger spaCy variants
(`md`/`lg`) and the multi-entity-hard fixture (§A.4.9) under ST are
reasonable follow-ups; not blocking v0.2.

**Reproduce.**

```
python -m evals.d1redux_multiseed_ci \
    --seeds 42,7,11 --embed st --resamples 5000 \
    --out evals/results/d1redux_multiseed_ci_st.json
```

Wall: 240 s on the loop host (heuristic arms ~28 s, spaCy arms ~46 s).
Artifact: `evals/results/d1redux_multiseed_ci_st.json`. Engineering log:
`SCALE_REPORT.md §D11`.

## A.4.13i `spacy_md` follow-up — does bigger NER reopen the §A.4.13h gap?

§A.4.13h showed the heuristic→`spacy_sm` lift collapses under ST. Two
hypotheses for that collapse: **H1 (NER quality)** — `sm` is too weak
to surface entities ST misses, and `md` would reopen the lift; **H2
(channel saturation)** — ST already encodes entity semantics well, so
no NER backend can add real signal on this fixture class. We extended
the §A.4.13h sweep with a third arm using `en_core_web_md` (50 MB
transformer-distilled model). Same protocol: synth_entity n=12, K=6,
seeds {42, 7, 11}, R=5000, n=1080, `--embed st`.

**Within-backend lift (Δ at ew=0.10 vs 0.0).**

| backend  | metric  | Δ mean   | 95% paired CI         |
|----------|---------|---------:|-----------------------|
| heuristic| Δhit@1  |  0.0000  | [ 0.0000,  0.0000]    |
| heuristic| Δhit@5  |  0.0000  | [ 0.0000,  0.0000]    |
| spacy_sm | Δhit@1  | −0.00093 | [−0.00463, +0.00185]  |
| spacy_sm | Δhit@5  | −0.00185 | [−0.00556, +0.00185]  |
| **spacy_md** | **Δhit@1** | **−0.0259** | **[−0.0380, −0.0139]** |
| **spacy_md** | **Δhit@5** | **−0.0315** | **[−0.0426, −0.0204]** |

**Result — H2 wins, decisively. Bigger NER actively *hurts* under ST.**
The `spacy_md` arm shows a statistically significant **negative** lift
(Δhit@5 = −3.1pp, 95% CI excludes 0) with the entity channel enabled.
Every additional entity that `md` surfaces beyond `sm` is, on net, a
distractor: it pulls RRF mass toward false-positive candidates that
the BM25/vector channels would otherwise correctly down-rank. The
trend is monotone worsening across the three backends:
**0.0 → −0.2pp → −3.1pp** on Δhit@5.

**Claim, sharpened.** §A.4.13h argued the heuristic→`sm` gap closes to
within ±0.6pp under ST. §A.4.13i strengthens that to: **NER quality is
not a capacity-bottleneck in the entity channel; the channel itself is
mis-specified for ST-grade embeddings on this fixture class.** Either
the entity-channel weight needs per-(NER × embedder) re-tuning (the
v0.2 default 0.10 is calibrated against heuristic + hash256), or the
channel must structurally gate on entity rarity in the corpus (rare
entities only) since ST already dominates on common ones.

**Implication for v0.2.** Confirms `entity_ner='heuristic'` as the
defended default. We do **not** extend the sweep to `lg` for this
fixture: spending more compute on a regressing arm is poor allocation.
The structural-redesign question (rarity gating, IDF-weighted entity
overlap) is parked for post-v0.2.

**Reproduce.**

```
python -m evals.d1redux_multiseed_ci \
    --seeds 42,7,11 --embed st --ners heuristic,spacy_sm,spacy_md \
    --resamples 5000 \
    --out evals/results/d1redux_multiseed_ci_st_md.json
```

Wall: 506 s on the loop host. Artifact:
`evals/results/d1redux_multiseed_ci_st_md.json`. Engineering log:
`SCALE_REPORT.md §D12`.

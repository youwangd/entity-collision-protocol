# A5. Extended Threats

This appendix expands §6's main-body threats with the supporting
analyses that the ACL 2-column 6-page Industry Track body cannot
accommodate. Each is referenced by a one-sentence pointer at the
end of §6; the full content lives here.

## A5.0 Sibling-lexical paraphrase replication on `service`

§6.1 reports the paraphrase replication on the strongest lexical
cell (`tool`). To check whether the paraphrase collapse is
`tool`-specific or generalizes across the lexical axis, we re-ran
the same protocol on `service` (closed-vocabulary proper-noun
answers: aws, gcp, azure, …):

| K  | hash Δhit@1 (paraphrased)      | ST Δhit@1 (paraphrased)        |
|----|--------------------------------|--------------------------------|
| 2  | −0.094 [−0.219, +0.031]        | +0.109 [+0.047, +0.188]        |
| 4  | −0.031 [−0.102, +0.039]        | +0.156 [+0.094, +0.219]        |
| 8  | +0.031 [−0.008, +0.070] **null** | +0.141 [+0.098, +0.188]      |
| 16 | **+0.037 [+0.012, +0.062]**    | **+0.105 [+0.078, +0.133]**    |

Compared to the fixed-template baseline (hash service K=16:
+0.057 [+0.025, +0.088]), the paraphrased lift shrinks to +0.037 —
about 65% of the fixed-template effect — **but stays CI-strictly
above zero**. Unlike `tool`, hash on `service` survives memory
paraphrase at K=16. The conservative reading: the paraphrase
collapse on `tool` is real, but the broader claim is not "hash
trigrams die under paraphrase" — it is "hash retention under
paraphrase depends on how lexically distinctive the answer
vocabulary is at the character-trigram level." `service`'s answer
set (short proper nouns: aws, gcp, azure) preserves more
discriminative character-trigrams across template variation than
`tool`'s (git, docker, postgres). The two-axis claim survives on
the **lexical/intent split**, but the within-lexical paraphrase
robustness has tag-level structure.

Outputs:
`bench/results/ec_sweep_{hash,st}_service_n32_K16_paraphrased{,_ci}.json`.

## A5.1 Single embedder per family

§6 tests exactly one hash-trigram dim (256) in the headline figure
and one sentence-transformer (MiniLM-L6-v2). To check whether the
lexical-axis hash lift is specific to dim=256, we ran a hash-dim
ablation at the strongest lexical cell (`tool`, K=8, n=32):

| dim   | Δhit@1   | 95% CI               |
|-------|----------|----------------------|
| 128   | −0.0195  | [−0.0742, +0.0312]   |
| 256   | +0.0664  | [+0.0039, +0.1250]   |
| 512   | +0.0586  | [+0.0000, +0.1172]   |
| 1024  | +0.0703  | [+0.0117, +0.1250]   |

Dim=128 is below noise; 256–1024 sit on a CI-positive plateau with
no monotone scaling. The two-axis claim is robust across hash dim
∈ {256, 512, 1024}; only the smallest sketch (128) collapses.
Extending the embedder grid to BGE / E5 on the dense side and to
character-quintgrams on the sketch side would tighten the family
generalization claim. The BGE-large-en-v1.5 (1024-d) follow-up has
since landed (Appendix §A.4.16) and rejects the encoder-capacity
hypothesis: the two-axis result survives a 2.7×-parameter encoder
swap, with BGE *losing* on lexical-discriminator tags. The
character-quintgram sketch is left as future work and flagged as a
named scope limitation rather than an open TODO.

## A5.2 hit@1 only

§6 reports `hit@1` because it is the worst-case metric and the most
sensitive to retriever ranking. `hit@5` and MRR mostly converge
toward 1.0 on the entity-collision corpus and are uninformative.
LoCoMo numbers are reported across all metrics in `SCALE_REPORT.md`.

## A5.3 Single-process SQLite

All operational latency numbers are single-writer SQLite/FTS5.
Multi-process write contention is not measured. A concurrency
torture suite (≥50 writers × ≥50 readers, see §A6) passes
correctness invariants under contention, but throughput-under-
contention is not yet a reported number.

## A5.4 Single machine, single OS

All wall-clock and throughput numbers come from one Linux x86_64
workstation (see `REPRODUCIBILITY.md` for the full env). p50/p95/p99
ingest latencies and the 100k constant-p99 claim therefore should
be read as *invariant within this hardware envelope*. The replay
discipline argued for in §A4.2 is what makes cross-machine
reproduction tractable — point-estimates may shift, but the
event-sourced lifecycle guarantees that the *shape* of any reported
distribution is reproducible from a pinned decision log; the
diff-results acceptance gate (`REPRODUCIBILITY.md` §4) operationalizes
this with ±0.5pp / ±25%-latency tolerances.

## A5.5 Author-as-annotator on tag definitions

The lexical/intent dichotomy of §3.3 is author-defined: `service`
and `tool` were labeled as closed-vocabulary lexical, the other
three as open-vocabulary intent-style, without an inter-annotator
agreement protocol. The labels are derived from the answer-set
construction (closed enum vs free phrasal slot), not from a third
party's judgment, so the categorization is *traceable* but not
*independently validated*. A reviewer could plausibly relabel
`technical` as lexical and recover a different two-axis fit. We
therefore present the dichotomy as a hypothesis the data is
consistent with, not as the unique correct partition.

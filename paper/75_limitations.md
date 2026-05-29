# Limitations

We name scope limits explicitly so deployments and follow-up research
can plan around them.

**Memory-side paraphrase robustness.** The protocol generates
synthetic memories from a fixed sentence template; real
conversational memories are noisier and more paraphrased. To check
whether the headline two-axis claim survives memory-side paraphrase,
we re-ran the strongest lexical cell (`tool`, n=32, K∈{1,2,4,8,16})
with paired 95% CIs against ≥4 paraphrased templates per fact. The
hash lift collapses to CI-null at K∈{8,16} once templates vary
(K=16: +0.023 [−0.006, +0.053], n.s.); ST retains all four cells
CI-strictly above zero, with K=16 lift *growing* from +0.043 (fixed)
to **+0.096 [+0.070, +0.121]** (paraphrased). Memory-side paraphrase
strengthens the two-axis claim: semantic embedders are
paraphrase-robust, hash-trigram retrievers are template-bound.
Replications on `service`, `preference`, and `project` confirm the
pattern (within-lexical `service` retains **+0.037 [+0.012, +0.062]**
at K=16; intent-tag hash null and MiniLM lift both survive
paraphrase, point estimates within ±0.02 pp of fixed-template).
Full per-tag tables in §A5.0; supporting threat analyses
(hash-dim ablation §A5.1, hit@1-only metric §A5.2, single-process
SQLite §A5.3, single-machine/single-OS §A5.4, author-as-annotator
§A5.5) live in the extended-threats appendix.

**Single-system instantiation.** We exercise the protocol on one
open-source agent-memory testbed — the artifact through which we run
the experiments and which we release for reproducibility. The
protocol is system-agnostic by construction (any retriever exposing
a per-document score qualifies), but cross-system replication on
Letta, Mem0, or another governed-memory implementation is queued
for a follow-up release. A pattern that does not replicate across
systems would weaken the methodological claim; we explicitly mark
this as the largest open risk to the protocol's external validity.

**Encoder coverage.** The grid covers three single-vector encoders
spanning a 4× parameter range (HashTrigram-256, MiniLM-384,
BGE-large-1024). Cross-encoder rerankers (ColBERT, SPLADE) require
a separate freeze (different latency budget, different deployment
story); §A3.4 documents the non-inclusion rationale and queues the
comparison as v0.3.

**External-validity coverage.** BEIR-3 results are reported on
FiQA (57k docs, ndcg@10=0.341, recall@100=0.695) and NQ (2.68M
docs, ndcg@10=0.355, recall@100=0.812); both runs use BGE-large +
hybrid (`vw=0.3`), no reranker, no expansion. HotpotQA (5.23M docs)
is deferred pending a batched-ingest helper (§5.1, §A.4.16.5):
the measured single-writer ingest rate of 30.6 ms/doc on NQ
projects to ≈44 h on HotpotQA at a constant rate, with
super-linear corpus-size penalty making the realized wall-clock
higher; we do not ship a multi-day single-pass run inside this
paper's window when the v0.3 batched-encode path will re-run it
in ≈8 h on the same hardware. The deferral does not affect the
headline claim, which is established on the synthetic grid +
LongMemEval n=500 + LoCoMo per-category + 2-of-3 BEIR-3.

**Statistical power.** Per-cell n=32 entity-collision results at
K=1 (n=32 paired trials) have minimum detectable effect ≈ 8–10 pp
Δhit@1 at α=0.05; headline two-axis claims are made at K ≥ 4
(n ≥ 128, MDE ≈ 4–5 pp). The §A.4.16.3 LongMemEval n=100 → n=500
inversion is a worked example of underpowering being mistaken for
null; readers should treat any "n.s." cell at n ≤ 56 as
power-limited rather than evidence of true null.

**Embedder train-test contamination.** Off-the-shelf MiniLM-L6-v2
and BGE-large-en-v1.5 were trained on web corpora that may overlap
with LongMemEval source data and the public-personae prompts used
to generate LoCoMo. We have no leakage-free zero-shot guarantee for
the natural-data results in §4.5 and §A.4.16.3. The synthetic
entity-collision corpus (§3.2) is constructed from disjoint
synthetic entity strings and is therefore leakage-free by
construction; the synthetic → natural transfer claim rests on the
synthetic side.

**Hardware envelope.** Latency and throughput numbers are reported
on consumer-grade hardware: a single Linux workstation (CPU only),
a single Apple Silicon laptop with unified memory, and a single
consumer CUDA laptop with 8 GB VRAM. Production GPU servers (L40S,
H100, A100) should improve the per-doc forward-pass cost linearly
with FLOPS, but the specific constants are not measured here. The
qualitative two-axis pattern is a property of the encoder family,
not the host, but we flag this as a measurement gap rather than
asserting it.

**Domain coverage.** Synthetic queries target five tag categories
(preference, project, technical, service, tool); natural-data
anchors (LongMemEval, LoCoMo) cover conversational long-context
recall. Specialized domains — legal, medical, multilingual,
code-search — are out of scope and may exhibit different per-tag
patterns. Practitioners deploying the protocol in such domains
should re-derive the tag schema for their distribution.

**Author-as-annotator on tag schema.** The lexical/intent dichotomy
of §3.3 is author-defined: `service`/`tool` were labeled
closed-vocabulary lexical, the other three open-vocabulary
intent-style, without an inter-annotator agreement protocol. Labels
are derived from answer-set construction (closed enum vs free
phrasal slot), so the categorization is *traceable* but not
*independently validated*. We present the dichotomy as a hypothesis
the data is consistent with, not as the unique correct partition.

**Operational contingencies.** The deterministic-replay framing
assumes the testbed's event-sourced log is durably persisted and
the schema-lifecycle DAG is monotonically advanced. Production
deployments that introduce non-monotone state mutations (e.g.,
user-driven deletion of memory entries) fall outside the invariant
set verified under property-based testing in §A6, and the
replay guarantee that supports paired-CI reproducibility no longer
holds without additional engineering.

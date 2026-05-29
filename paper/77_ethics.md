# Ethics Statement

**Data provenance.** All evaluation data come from publicly released
benchmarks: LongMemEval (Wu et al., 2024) and LoCoMo (Maharana et al.,
2024) for natural-language memory tasks, and BEIR-3 (Thakur et al.,
2021) for external-validity sweeps (FiQA, NQ, deferred HotpotQA). No
human-subjects data were collected for this work; no proprietary or
user-derived corpora were ingested. Synthetic stress sets were
generated programmatically with seeded RNGs (§A6) and contain no
real-person identifiers. The release tarball ships only public data
references and synthetic-generation scripts — no scraped corpora,
no de-anonymized records, no chat logs.

**Generative-AI use during authorship** is disclosed in full at
camera-ready time per the ACL Policy on Publication Ethics; this
disclosure is intentionally omitted from the review version under
double-blind policy.

**Broader impact.** Agent memory systems sit at the boundary
between long-context retrieval and personalisation: as deployments
scale, retrieval lift (and its absence) directly governs how often
an agent surfaces stale facts, conflated entities, or hallucinated
attributions to the wrong user. The protocol introduced here aims
to make those failure modes measurable rather than narrative,
which we view as net-positive for downstream deployments. The
protocol does not itself produce new corpora, new model weights,
or new user-facing capabilities; it produces an evaluation
methodology and a falsifiable scorecard.

**Risks of misuse.** Reproducing the experiments requires only the
public benchmarks and an open-source encoder (HashTrigram-256,
MiniLM-384, or BGE-large-1024). We do not foresee dual-use risks
beyond those already inherent to public IR benchmarks. The
schema-lifecycle invariants and security audits documented in §A2
were authored to surface, not exploit, retrieval-manipulation
attack surfaces in agent-memory pipelines; we explicitly enumerate
the eight invariants so that downstream builders can pin them
rather than rediscover them after a production incident.

**Limitations on generality.** External validity is bounded as
documented in §A4.16 (single-instantiation testbed, three
single-vector encoders, two natural-language benchmarks plus
two of three BEIR-3 corpora). Conclusions about absolute lift
magnitudes should not be transported to corpora or query
distributions outside this evidence base without re-running the
protocol; the protocol itself is designed precisely to make such
re-runs cheap.

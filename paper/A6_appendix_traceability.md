# A6 Implementation and Testbed Traceability

This appendix is a one-stop reference for the implementation paths
and reproduce scripts cited by abstraction in the body. Reviewers
who do not need to inspect the testbed code can skip it; reviewers
who *do* — particularly when checking the deterministic-replay
arguments in §A7.4, §A4.2, and §75 Limitations — will find every cited file
here, with what it pins and how many cases / properties it
exercises.

We treat this index as version-controlled supplementary material.
Every path below is verified to exist at the tagged release by
`scripts/verify_repro_artifacts.sh`, the same script that gates the
artifact registry (§REPRODUCIBILITY).

## A6.1 Pure-reducer invariants for the schema lifecycle (§A4.2)

The three lifecycle invariants in §A4.2 — event-sourced state,
permutation-stable family clustering, real-time monotone decay —
correspond to the following property-test files:

| Invariant (§A4.2)                               | Test file (`tests/property/`)                                                                         | Notes                                                |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Lifecycle state = fold of decision log         | `test_schema_lifecycle.py`, `test_schema_decision.py`, `test_schema_decision_x_reducer.py`            | Includes RECOVER edge fuzzing                        |
| Family clustering invariant under permutation  | `test_schema_family.py`, `test_schema_family_decision.py`, `test_schema_family_window.py`             | Permutes the input fact stream; asserts assignment   |
| Decay monotone in real time, not arrival       | `test_schema_decay.py`                                                                                | Fuzzes interleaved `tick()` / `update()` calls       |

Hypothesis is configured at `max_examples=200` per property in CI;
the lifecycle reducer is therefore exercised over ≈1.6k randomized
event traces per CI run.

## A6.2 Write-path independence (§A7.4.2)

The two invariants on deduplication × extraction-confidence
independence (I1/I2 in §A7.4.2) are pinned by:

| Invariant (§A7.4.2)                                                   | Test file                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| I1: dedup outcome invariant under keeper-side confidence              | `tests/property/test_dedup_extraction_confidence_indep.py`         |
| I2: deduped write does not mutate keeper's confidence                 | (same)                                                             |

## A6.3 Concurrency torture suite (§75 Limitations)

The ≥50 writers × ≥50 readers correctness suite referenced as
testbed sanity in §75 Limitations lives at:

| Concern                                       | Test file (`tests/concurrency/`)                          |
| --------------------------------------------- | --------------------------------------------------------- |
| Generic write/read race coverage              | `test_race_conditions.py`                                 |
| Dedup race                                    | `test_dedup_race.py`                                      |
| High-fanout writer/reader load                | `test_high_fanout.py`                                     |
| JSONL append-buffer concurrency               | `test_jsonl_buffer_concurrency.py`                        |
| JSONL truncate race                           | `test_jsonl_truncate_race.py`                             |

The suite asserts correctness invariants (no torn writes, no lost
events, ACL never lifts under contention) but does not yet report
throughput; this is the §75 Limitations limitation.

## A6.4 Adversarial / ACL invariants (Appendix A2)

The path-by-path enumeration of the 11 ACL side-channel audits is
already in §A2 (Security Audits). §A6 is intended for body-level
abstractions; readers tracing §A2 references should consult that
appendix directly. The full list of `tests/adversarial/test_*.py`
audit files is duplicated below for convenience:

| Audit subject (§A2)                                  | Test file (`tests/adversarial/`)                                |
| ---------------------------------------------------- | --------------------------------------------------------------- |
| PRF expansion ACL                                    | `test_prf_acl_side_channel.py`                                  |
| PRF + IDF gate ACL                                   | `test_prf_idf_acl_side_channel.py`                              |
| Share-prior ACL                                      | `test_share_prior_acl_side_channel.py`                          |
| Lifecycle cache ACL                                  | `test_lifecycle_cache_acl_side_channel.py`                      |
| Schema-id targeted suppression                       | `test_schema_id_targeted_suppression.py`                        |
| Vector-channel ACL                                   | `test_vector_channel_acl_side_channel.py`                       |
| Mechanical-merge ACL                                 | `test_mechanical_merge_acl_side_channel.py`                     |
| Fact-extraction ACL inheritance                      | `test_fact_extraction_acl_inheritance.py`                       |
| Write-dedup ACL                                      | `test_write_dedup_acl_side_channel.py`                          |
| Write-dedup × ACL race                               | `test_write_dedup_acl_race.py`                                  |
| Extraction-confidence ACL                            | `test_extraction_confidence_acl_side_channel.py`                |
| Schema deprecate quorum                              | `test_schema_deprecate_quorum.py`                               |
| Mech-merge × write-dedup composition (CXE)           | `test_cxe_mech_merge_x_write_dedup_compose.py`                  |
| Mech-merge × extraction-confidence (CXF)             | `test_cxf_mech_merge_x_extraction_confidence.py`                |

## A6.5 Scale evidence (§A.4.14, §6.4)

The 1M-memory single-writer characterization referenced in §1, §A.4.14,
and §6.4 corresponds to:

| Claim                                              | Test file (`tests/scale/`)                          |
| -------------------------------------------------- | --------------------------------------------------- |
| Tail-100k p99 write latency at 100k                | `test_ingest_scale.py::test_scale_recall_after_100k` |
| Same-harness 1M extension                          | `test_ingest_scale.py` (1M fixture)                 |

## A6.6 Why this index lives here, not in the body

Three of Engram's invariants — event-sourced lifecycle, mechanical
merge, write dedup — are implementation-level guarantees that
support the measurement claims rather than being measurement claims
in their own right. The body of the paper therefore states each
invariant in academic abstraction (deterministic fold over events,
permutation-stable cluster identity, real-time monotone decay) and
defers per-property file paths and case counts to this appendix.
This separation matches the standard distinction in systems papers
between *what is true of the system* (body) and *how the truth is
verified at the implementation* (appendix), and it leaves the body
sections short enough that a reviewer focused on the
entity-collision protocol can read the methodology subsections in
linear time.

# Acknowledgements (camera-ready only — anonymized for review)

> **Anonymization status.** This file is **NOT** included in the
> review build (`paper/build.sh --review`). Camera-ready build
> (`paper/build.sh --camera-ready`) appends this section after §7.
> Reviewer-visible builds intentionally lack acknowledgements per
> ACL double-blind policy.

## Generative-AI authorship disclosure

Per the ACL Policy on Publication Ethics (Guidelines for Generative
Assistance in Authorship; https://www.aclweb.org/adminwiki/index.php/
ACL_Policy_on_Publication_Ethics#Guidelines_for_Generative_Assistance_in_Authorship)
we disclose the following.

A long-context coding/research assistant (Anthropic's Claude family,
accessed through a custom internal harness called Hermes Agent) was
used as an authorship-adjacent collaborator across the following
scopes:

1. **Language polishing and paraphrasing** of author-drafted prose
   throughout the paper (covered by ACL §a — does not require
   disclosure under the ACL policy, but disclosed here for full
   transparency).

2. **Drafting expansions** of §2 (Related Work) from a list of
   anchor citations and topic clusters provided by the author. All
   citations were verified by the author against Semantic Scholar
   and arXiv before final submission (covered by ACL §d — automatic
   generation of low-novelty text about pre-existing ideas; flagged
   here per policy).

3. **Drafting** of the §A6 "Implementation and Testbed Traceability"
   appendix as a refactor of in-line test-file mentions previously
   scattered across §3.7 / §5.3 / §6.6 (covered by ACL §a/§d —
   restructuring of pre-existing content; flagged here per policy).

4. **Drafting** of the §A.4.8.1 third-encoder block (PRF arms × BGE
   results) from author-provided experimental artifacts. Numerical
   results are the author's; the prose summary was assistant-drafted
   (covered by ACL §a/§d).

5. **Verdict / Implication** scan-grammar lead-ins layered onto §4 /
   §5 / §6 subsections (covered by ACL §a — light editorial
   refactoring).

6. **Operational and editorial commentary** during multiple
   reviewer-pass simulations (Gemini-Pro v1–v5, an internal cycle of
   simulated peer review). Resulting changes were applied by the
   author with assistant drafting; the simulations themselves did
   not contribute new research ideas, only structural and rhetorical
   feedback (covered by ACL §a, with the simulation framing
   disclosed for full transparency).

Categories the assistant **did NOT** contribute to:

- The **entity-collision protocol design**, including the BM25-floor
  formalisation, two-axis stratification, and per-tag CI estimator,
  which are the author's research contribution.
- The **experimental design and execution**, all numerical results,
  paired-bootstrap CI computation, and the underlying
  `evals/entity_collision_*` and `evals/longmemeval_adapter` code
  paths.
- The **schema-lifecycle invariant set** (event-sourced fold,
  permutation-stable family clustering, real-time monotone decay)
  and the property-test substrate that pins them.
- The **two-axis empirical finding**, the **adaptive vector-weight
  null result**, and the **encoder-capacity falsification**, all of
  which are the author's research conclusions; the assistant
  surfaced them in prose but did not derive them.
- **Citation accuracy.** The author independently verified every
  arXiv ID and venue string against Semantic Scholar and the arXiv
  abstract pages prior to submission (per EMNLP 2026 Paper
  Integrity Policy on unverifiable references).

Per ACL policy §f ("New ideas + new text"), no generative model is
listed or could be listed as an author of this paper.

## Other acknowledgements

[TODO at camera-ready time: thank human reviewers, collaborators,
hardware / cluster providers, and any conference-specific support.]

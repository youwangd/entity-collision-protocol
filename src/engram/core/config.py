"""Configuration for Engram."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RetrievalConfig:
    """Retrieval engine weights and thresholds."""

    bm25_weight: float = 0.3
    vector_weight: float = 0.3
    salience_weight: float = 0.2
    recency_weight: float = 0.1
    context_weight: float = 0.1
    confidence_threshold: float = 0.3
    reranker: str | None = None
    # When a reranker is configured, run it over the top-N fused candidates
    # (where N = max(limit, rerank_pool_size)) BEFORE truncating to `limit`.
    # Larger pool → more reorder headroom but more compute. 0 means "use
    # limit only" (no extra pool). Inert when reranker is None.
    rerank_pool_size: int = 20
    # §96 share_prior reranker boost magnitude. Inert unless
    # reranker == "share_prior". Capped per-candidate by the
    # rank-0-preservation guard, so larger values just saturate
    # rather than hurt h@1. Sweepable from evals/share_prior_sweep.py.
    share_prior_alpha: float = 0.05
    # §96 share_prior — adaptive alpha schedule. When True, the effective
    # alpha is scaled by 1/(1 + (max_deg-1)/4), saturating at 1.0 for
    # max_deg ≤ 1 and tapering as the entity-sharing graph densifies.
    # See paper/30_methods.md §3.5 / paper/A1_appendix_ablations.md §A.4.7.6.
    # Default OFF: shipping behind a flag while we collect a Δ
    # characterization vs. constant-alpha.
    share_prior_adaptive_alpha: bool = False
    # Use extraction_confidence as a multiplier on final retrieval score.
    # From "Governed Memory" paper (arXiv:2603.17787). Default: True.
    use_extraction_confidence: bool = True
    # D1 — entity-link channel (Mem0 v3-inspired). When > 0, Jaccard
    # similarity between the query's extracted entity set and each
    # candidate memory's entity set is added to the fused score with this
    # weight. 0.0 disables the channel entirely (default OFF: shipping
    # behind a flag while we collect a Δrecall@k characterization).
    entity_weight: float = 0.0
    # D1 — NER backend selector. "heuristic" (default, dependency-free regex)
    # or one of "spacy_sm" / "spacy_md" / "spacy_lg" (each requires the
    # `entity-ner` extras + the corresponding spaCy model download).
    # Inert when entity_weight == 0.0. Lazy-imported on first use.
    entity_ner: str = "heuristic"
    # §5.4 angle 1 — pseudo-relevance-feedback (PRF) entity-based query
    # expansion. When non-None, RetrievalEngine.search runs a first-pass
    # retrieval, mines the most frequent novel entities from the top-k
    # texts, and (if the top entity's document-frequency / k ≥
    # query_expansion_min_dominance) appends them to the query and runs a
    # second pass. Recommended operating point: 0.3 (anchor 18 + 22 CIs,
    # α=0.05, d=0.3, pool=20). Default None = OFF (regression-safe; same
    # behavior as v0.1). Inert when entity_ner backend is unavailable.
    # v0.3 default flip: ON at 0.3 (operating point defended in §4.15 with
    # CIs at α=0.05, d=0.3, pool=20). Combined with query_expansion_anchor_
    # share_max=0.5 below, this is bit-identically inert on LongMemEval-S
    # (§D15d-LME) and cures the §D15c synth-pref regression at SE=0.
    # 2026-05-24 SECOND FLIP — back to None (OFF). paper/40_results.md
    # §4.8.2.4 paired n=500 LongMemEval-S re-bench shows PRF×SP at this
    # operating point regresses Δhit@1 = −0.022 [−0.042, −0.002] and
    # Δhit@5 = −0.012 [−0.024, −0.002] (CIs exclude zero). Decision #2 in
    # the v0.2 plan ("default=None (off), runtime-toggleable") is now
    # backed by data; ship OFF. Knob remains for opt-in on multi-entity-
    # hard / type-disambiguated corpora (§4.10 typed gate).
    query_expansion_min_dominance: float | None = None
    # PRF top-K pool size: how many first-pass docs to mine entities from.
    # Anchor 29/30 sweep showed k=10 is the sweet spot; k=5 underfits,
    # k=40 overfits.
    query_expansion_top_k: int = 10
    # PRF max entities to append. Anchor 14 breadth sweep.
    query_expansion_max_entities: int = 3
    # §4.15g — IDF-rarity filter on PRF candidate entities (v0.3 prototype).
    # When non-None, each candidate entity is scored by corpus document-
    # rarity = 1 - df/N, where df is the # of memories whose FTS index
    # contains the entity surface form and N is the active corpus size.
    # Candidates with rarity < threshold are dropped *before* the
    # max_entities truncation. Targets the §D15c multi-token-anchor
    # regression: when PRF appends common tokens, BM25 scores dilute
    # toward the answer anchor; filtering low-IDF candidates preserves
    # only corpus-rare expansion terms.
    # Default None = OFF (preserves v0.2 PRF behaviour). Inert when
    # query_expansion_min_dominance is None.
    query_expansion_idf_min_rarity: float | None = None
    # §D15d — anchor-share diagnostic gate (v0.3 prototype).
    # When non-None AND query_expansion_min_dominance is set, count the
    # surface-form occurrences of the *dominant* candidate entity across
    # the first-pass top-K texts, divided by the total entity occurrences
    # in that pool. When this "anchor share" exceeds the threshold, the
    # pool is saturated by the anchor entity (the §D15c failure mode:
    # cross-fact entity confusion under shared-anchor density). PRF is
    # then short-circuited for that query and the un-expanded first pass
    # is returned. Default None = OFF (preserves v0.2 PRF behaviour).
    # Inert when query_expansion_min_dominance is None.
    # v0.3 default flip: 0.5 (LME bit-identically inert at this threshold,
    # synth-pref Δh@1 = 0.0 at SE=0; see §D15d-LME / §4.15j).
    query_expansion_anchor_share_max: float | None = 0.5
    # §5.4 follow-up — type-aware PRF gate. When non-None *and*
    # entity_ner == "spacy_sm", PRF additionally requires that the
    # dominant entity-type's share of total entity occurrences in the
    # top-K pool is ≥ this threshold. Inert under "heuristic" backend
    # (which has no real types) and inert when None. Default None = OFF.
    query_expansion_type_purity_min: float | None = None
    # §D15 type-conditional PRF gate. When non-None AND
    # query_expansion_min_dominance is set, PRF expansion is short-
    # circuited (skipped) for queries whose heuristic question-type
    # is NOT in this allow-set. Lets us route PRF only to types that
    # benefit (per §D14 directional evidence). When None, the gate is
    # inert: PRF runs for every query (preserving v0.2 behaviour).
    # Type strings must come from `engram.retrieval.type_classifier`.
    # Default None = OFF.
    query_expansion_type_allow: frozenset[str] | None = None
    # §4.15-profile follow-up — `both` arm latency lever (v0.3 prototype).
    # When True AND a reranker is configured AND PRF expansion is active,
    # the FIRST-pass retrieval (whose results are only used to mine
    # entities for query expansion) skips the reranker. The SECOND pass
    # (whose results are returned to the caller) still reranks normally.
    # Targets the §4.15-profile observation that the `both` arm pays
    # ~24% p95 overhead, half of which is wasted reranker work on the
    # first pass. Inert when reranker is None or PRF is OFF.
    # Default False (regression-safe; preserves v0.2/v0.3 default behaviour).
    query_expansion_skip_rerank_first_pass: bool = False
    # Suppress SCHEMA-typed candidates whose lifecycle status is DEPRECATED
    # (replayed from the buffer's CONSOLIDATION_SCHEMA_LIFECYCLE event stream
    # via `engram.consolidation.lifecycle_projection.snapshot_from_buffer`).
    # No-op when no buffer is wired to the RetrievalEngine. Default: True.
    respect_schema_lifecycle: bool = True
    # §6.16 quorum gate. When >1, DEPRECATE lifecycle events require k
    # *distinct* `emitter_id`s before a schema actually transitions to
    # DEPRECATED (the pure reducer accumulates votes; see
    # `engram.consolidation.schema_lifecycle.reduce_events`). k=1 (default)
    # preserves legacy single-emitter behaviour byte-for-byte. The
    # RetrievalEngine plumbs this into `CachedLifecycleSnapshot.get`,
    # which re-runs the reducer on cache miss / partial-hit. Changing k
    # at runtime invalidates the cache (the snapshot is a function of k).
    deprecate_quorum_k: int = 1


@dataclass
class StorageConfig:
    """Storage / write-path settings."""

    # Cosine similarity above which a write is treated as a near-duplicate and skipped.
    # 0.0 disables. Governed Memory paper recommends 0.92.
    write_dedup_threshold: float = 0.0
    # Cosine threshold for the mechanical-merge consolidation stage (no LLM).
    # Stricter than write-side dedup. Governed Memory paper uses 0.95.
    merge_threshold: float = 0.95


@dataclass
class ForgettingConfig:
    """Forgetting parameters."""

    decay_enabled: bool = True
    fade_threshold: float = 0.1
    auto_suppress: bool = True
    suppress_threshold: float = -0.7


@dataclass
class SecurityConfig:
    """Security settings."""

    pii_detection: bool = False
    auto_redact: list[str] = field(default_factory=list)
    encrypt_at_rest: bool = False
    encryption_key_source: str = "env"
    encryption_key_path: str | None = None
    redact_in_logs: bool = True
    content_policy: dict[str, str] = field(default_factory=dict)  # classification → action: "block"|"warn"
    # Rate limit on writes per minute. 0 disables. Default 100 matches FirewallConfig.
    max_events_per_minute: int = 100
    # Toggle injection-pattern detection (default on)
    injection_detection: bool = True


@dataclass
class RetentionConfig:
    """Data retention TTLs."""

    buffer_ttl_days: int = 30
    faded_ttl_days: int = 90
    suppressed_ttl_days: int = 180
    audit_ttl_days: int = 365


@dataclass
class AffectConfig:
    """Affect engine configuration."""

    temperament: str | dict = "neutral"
    mutation_rate: float = 0.005
    mood_window_hours: int = 4


@dataclass
class ConsolidationConfig:
    """Consolidation pipeline configuration."""

    schedule: str = "manual"
    window_hours: int = 24
    stages: list[str] | None = None  # None = default pipeline
    # Personize §8 prior-sharing knob. 0.0 = single-schema decision
    # (regression-safe default, identical to bare decide()). >0.0 lets
    # cluster-mate evidence credit toward an owner's promote/deprecate
    # threshold via decide_window(). Sweep evidence (run #62) shows
    # share≈0.75 is the sweet spot on sparse evidence; >0.85 starts
    # leaking false promotes.
    schema_family_share: float = 0.0
    # Jaccard threshold for schema_family.cluster() when share>0.
    # 0.5 is the documented default in the schema_family module.
    schema_family_tau: float = 0.5
    # Operational §69 deployment-rule gate. When set, the consolidation
    # prepass computes `contamination_rate(features, clusters, tau)` over
    # the just-built clusters; if the rate exceeds this threshold, the
    # window falls back to ``effective_share = 0.0`` (bare decide()).
    # Default ``None`` = disabled (no gate, share applies as configured).
    # The §69 synthetic threshold is 0.10. Real-corpus calibration on
    # LoCoMo (SCALE_REPORT §78, §80) found no gateable tau under the
    # synthetic-derived 0.10 cutoff (the meter sits at 0 in the
    # generative regime, see §72/§74); the fragmentation_max signal
    # is the operational successor (`schema_family_fragmentation_max`,
    # default 0.10 from §76 calibration).
    schema_family_contamination_max: float | None = None
    # Operational §74 fragmentation gate. SCALE_REPORT §74 showed the
    # contamination meter is identically ≈0.0 across the realistic
    # generative regime because cluster() expels outsiders as
    # singletons (K1 zero-weight) rather than gluing them in;
    # ``singletons/n_schemas`` is the actually-monotone signal of
    # cluster-quality stress in that regime. When set, the prepass
    # computes ``fragmentation_rate(features, clusters)``; if it
    # exceeds this cap, the window falls back to ``effective_share=0.0``.
    # Default ``None`` = disabled. Either gate (contamination_max OR
    # fragmentation_max) tripping collapses share. See SCALE_REPORT §74.
    schema_family_fragmentation_max: float | None = None
    # §D3 (Mem0 v3 ablation primitive). When True, the interference stage
    # is bypassed: no supersede / no conflict-flag mutation. Memories
    # accumulate ADD-only (mechanical-merge cosine dedup still applies).
    # Lets us test the Mem0 v3 "single-pass ADD-only" hypothesis against
    # Engram's default supersede-on-overlap behavior on the same corpus.
    # Default False = current Engram semantics. See SCALE_REPORT §D3.
    add_only: bool = False
    # §D3-collateral-(b) entity-aware interference detector. When True,
    # `_detect_interference` requires high overlap on *content tokens*
    # (non-stop-word, length≥3) in addition to the existing Jaccard gate.
    # This collapses the cross-slot false-positive FADE rate exposed by
    # the §D3-collateral n_slots sweep without disabling supersede.
    # Threshold (`interference_entity_overlap_min`, default 0.7) is the
    # minimum fraction of content-token Jaccard required. §D3-collateral-(d)
    # swept {0.3, 0.5, 0.7, 0.9} on the n_slots=200 supersede corpus and
    # found 0.7 is the unique knee that preserves Δhit@1 = +9pp (p<1e-4)
    # while collapsing Δhit@k from −44.5pp (at 0.5) to 0pp (p=1.0). Lower
    # values over-fire on template overlap; 0.9 fires too rarely
    # (Δhit@1 = +0.5pp, p=0.81).
    # Default True since §D3-collateral-(c): paired-bootstrap CI on full
    # LoCoMo10 (n=1978, 10k resamples) shows all five Δ-metrics land at
    # exactly [0,0] p=1.0 vs entity_aware=False — clean null, zero
    # regression risk on real corpora — while the synthetic supersede
    # corpus shows entity-aware fully eliminates the −58.5pp Δhit@k
    # cross-slot collateral. Flag preserved for ablations.
    interference_entity_aware: bool = True
    interference_entity_overlap_min: float = 0.7
    # §93 deterministic non-LLM schema synthesizer. When True (and the
    # configured LLM is None / NoLLMProvider), SchemaUpdate falls back
    # to `engram.consolidation.schema_synthesis.synthesize_schemas`
    # over the fact corpus and feeds the result through the same
    # CREATE/BUMP/RECOVER + family-gate machinery the LLM path uses.
    # This unblocks §85 / §87 / §90 / §91 from network/LLM dependence.
    # Default False: regression-safe; matches pre-§93 behavior on the
    # no-LLM cron exactly (zero schemas synthesized).
    schema_synthesis_enabled: bool = False
    # §93 synthesizer knobs (pure passthroughs; see schema_synthesis.py).
    schema_synthesis_tau: float = 0.3
    schema_synthesis_min_supports: int = 3
    # §94c-appraisal-bound. After appraisal computes a per-memory salience
    # in [0.0, 1.0], clamp to ``min(salience, salience_cap)`` if set. The
    # §94c-appraisal-inspect-CI evidence shows appraisal's salience-driven
    # rerank inside top-k displaces more golds than it surfaces (lone Δgrk
    # bite, p=0.038 over 7 stages). Capping bounds appraisal's contribution
    # to the retrieval score (which is salience * salience_weight) without
    # disabling the appraisal stage. Default None = no cap, current
    # behavior. See SCALE_REPORT §94c-appraisal-bound.
    appraisal_salience_cap: float | None = None

    # Schema lifecycle thresholds (Design §4.7 / paper §4.6 churn budget).
    # These plumb directly into `schema_decision.Thresholds`. Defaults match
    # the historical hardcoded `Thresholds()` values; exposed so the
    # L0→schema-promotion churn-budget sweep (paper §4.6) can vary the
    # promotion bar and measure the recall × churn frontier without
    # forking the pipeline. All counts; rate-based policies layer on top.
    schema_promote_threshold: int = 3
    schema_deprecate_threshold: int = 2
    schema_recover_threshold: int = 3


@dataclass
class Config:
    """Main Engram configuration."""

    path: str = "~/.engram"
    buffer: str = "jsonl"
    memory: str = "sqlite"
    vector: str | None = None
    embedding: str | None = None
    llm: str | None = None
    affect: AffectConfig | None = None
    consolidation: ConsolidationConfig | None = None
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    acl: dict | None = None  # ACL grants config (Design §5.3)
    # §6.16 quorum gate — identity of *this* consolidator/process, threaded
    # into every CONSOLIDATION_SCHEMA_LIFECYCLE event's metadata as
    # `emitter_id`. Empty (default) preserves the legacy pre-quorum wire
    # format byte-for-byte (the metadata key is omitted entirely).
    # Production multi-node deployments should set this to a stable
    # per-consolidator id (e.g. `socket.gethostname()` or a uuid4 baked at
    # boot). Counted by `reduce_events(deprecate_quorum_k>1)` to gate
    # DEPRECATE on k distinct emitters.
    consolidator_id: str = ""

    @property
    def resolved_path(self) -> Path:
        """Resolve ~ and return Path."""
        return Path(self.path).expanduser()

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load config from YAML file."""
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Config:
        """Create Config from a dictionary."""
        config = cls(
            path=d.get("path", "~/.engram"),
            buffer=d.get("buffer", "jsonl"),
            memory=d.get("memory", "sqlite"),
            vector=d.get("vector"),
            embedding=d.get("embedding"),
            llm=d.get("llm"),
        )
        if "affect" in d and d["affect"] is not None:
            a = d["affect"]
            config.affect = AffectConfig(
                temperament=a.get("temperament", "neutral"),
                mutation_rate=a.get("mutation_rate", 0.005),
                mood_window_hours=a.get("mood_window_hours", 4),
            )
        if "consolidation" in d and d["consolidation"] is not None:
            c = d["consolidation"]
            config.consolidation = ConsolidationConfig(
                schedule=c.get("schedule", "manual"),
                window_hours=c.get("window_hours", 24),
                stages=c.get("stages"),
                schema_family_share=float(c.get("schema_family_share", 0.0)),
                schema_family_tau=float(c.get("schema_family_tau", 0.5)),
                schema_family_contamination_max=(
                    float(c["schema_family_contamination_max"])
                    if c.get("schema_family_contamination_max") is not None
                    else None
                ),
                schema_family_fragmentation_max=(
                    float(c["schema_family_fragmentation_max"])
                    if c.get("schema_family_fragmentation_max") is not None
                    else None
                ),
                add_only=bool(c.get("add_only", False)),
                interference_entity_aware=bool(c.get("interference_entity_aware", True)),
                interference_entity_overlap_min=float(c.get("interference_entity_overlap_min", 0.7)),
                schema_synthesis_enabled=bool(c.get("schema_synthesis_enabled", False)),
                schema_synthesis_tau=float(c.get("schema_synthesis_tau", 0.3)),
                schema_synthesis_min_supports=int(c.get("schema_synthesis_min_supports", 3)),
                appraisal_salience_cap=(
                    float(c["appraisal_salience_cap"])
                    if c.get("appraisal_salience_cap") is not None
                    else None
                ),
                schema_promote_threshold=int(c.get("schema_promote_threshold", 3)),
                schema_deprecate_threshold=int(c.get("schema_deprecate_threshold", 2)),
                schema_recover_threshold=int(c.get("schema_recover_threshold", 3)),
            )
        if "forgetting" in d:
            fg = d["forgetting"]
            config.forgetting = ForgettingConfig(
                decay_enabled=fg.get("decay_enabled", True),
                fade_threshold=fg.get("fade_threshold", 0.1),
                auto_suppress=fg.get("auto_suppress", True),
                suppress_threshold=fg.get("suppress_threshold", -0.7),
            )
        if "retrieval" in d:
            r = d["retrieval"]
            config.retrieval = RetrievalConfig(
                bm25_weight=r.get("bm25_weight", 0.3),
                vector_weight=r.get("vector_weight", 0.3),
                salience_weight=r.get("salience_weight", 0.2),
                recency_weight=r.get("recency_weight", 0.1),
                context_weight=r.get("context_weight", 0.1),
                confidence_threshold=r.get("confidence_threshold", 0.3),
                reranker=r.get("reranker"),
                rerank_pool_size=r.get("rerank_pool_size", 20),
                share_prior_alpha=r.get("share_prior_alpha", 0.05),
                share_prior_adaptive_alpha=r.get(
                    "share_prior_adaptive_alpha", False
                ),
                use_extraction_confidence=r.get("use_extraction_confidence", True),
                entity_weight=r.get("entity_weight", 0.0),
                entity_ner=r.get("entity_ner", "heuristic"),
                query_expansion_min_dominance=r.get("query_expansion_min_dominance"),
                query_expansion_top_k=r.get("query_expansion_top_k", 10),
                query_expansion_max_entities=r.get("query_expansion_max_entities", 3),
                query_expansion_idf_min_rarity=r.get(
                    "query_expansion_idf_min_rarity"
                ),
                query_expansion_anchor_share_max=r.get(
                    "query_expansion_anchor_share_max"
                ),
                query_expansion_type_purity_min=r.get(
                    "query_expansion_type_purity_min"
                ),
                query_expansion_type_allow=(
                    frozenset(r["query_expansion_type_allow"])
                    if r.get("query_expansion_type_allow") is not None
                    else None
                ),
                respect_schema_lifecycle=r.get("respect_schema_lifecycle", True),
                deprecate_quorum_k=int(r.get("deprecate_quorum_k", 1)),
                query_expansion_skip_rerank_first_pass=r.get(
                    "query_expansion_skip_rerank_first_pass", False
                ),
            )
        if "storage" in d:
            st = d["storage"]
            config.storage = StorageConfig(
                write_dedup_threshold=st.get("write_dedup_threshold", 0.0),
                merge_threshold=st.get("merge_threshold", 0.95),
            )
        if "security" in d:
            s = d["security"]
            config.security = SecurityConfig(
                pii_detection=s.get("pii_detection", False),
                auto_redact=s.get("auto_redact", []),
                encrypt_at_rest=s.get("encrypt_at_rest", False),
                encryption_key_source=s.get("encryption_key_source", "env"),
                encryption_key_path=s.get("encryption_key_path"),
                redact_in_logs=s.get("redact_in_logs", True),
                content_policy=s.get("content_policy", {}),
                max_events_per_minute=s.get("max_events_per_minute", 100),
                injection_detection=s.get("injection_detection", True),
            )
        if "acl" in d:
            config.acl = d["acl"]
        if "consolidator_id" in d and d["consolidator_id"] is not None:
            config.consolidator_id = str(d["consolidator_id"])
        if "retention" in d:
            rt = d["retention"]
            config.retention = RetentionConfig(
                buffer_ttl_days=rt.get("buffer_ttl_days", 30),
                faded_ttl_days=rt.get("faded_ttl_days", 90),
                suppressed_ttl_days=rt.get("suppressed_ttl_days", 180),
                audit_ttl_days=rt.get("audit_ttl_days", 365),
            )
        return config

    @classmethod
    def minimal(cls, path: str = "~/.engram") -> Config:
        """Zero-dependency config: BM25 only, no LLM, no vectors, no affect."""
        return cls(path=path)

    def to_dict(self) -> dict:
        """Serialize to dict (for YAML export)."""
        d: dict[str, Any] = {"path": self.path}
        if self.vector:
            d["vector"] = self.vector
        if self.embedding:
            d["embedding"] = self.embedding
        if self.llm:
            d["llm"] = self.llm
        if self.affect:
            d["affect"] = {
                "temperament": self.affect.temperament,
                "mutation_rate": self.affect.mutation_rate,
                "mood_window_hours": self.affect.mood_window_hours,
            }
        if self.consolidation:
            d["consolidation"] = {
                "schedule": self.consolidation.schedule,
                "window_hours": self.consolidation.window_hours,
            }
            if self.consolidation.stages:
                d["consolidation"]["stages"] = self.consolidation.stages
            if self.consolidation.schema_family_share != 0.0:
                d["consolidation"]["schema_family_share"] = self.consolidation.schema_family_share
            if self.consolidation.schema_family_tau != 0.5:
                d["consolidation"]["schema_family_tau"] = self.consolidation.schema_family_tau
            if self.consolidation.schema_family_contamination_max is not None:
                d["consolidation"]["schema_family_contamination_max"] = (
                    self.consolidation.schema_family_contamination_max
                )
            if self.consolidation.schema_family_fragmentation_max is not None:
                d["consolidation"]["schema_family_fragmentation_max"] = (
                    self.consolidation.schema_family_fragmentation_max
                )
            if self.consolidation.add_only:
                d["consolidation"]["add_only"] = True
            if self.consolidation.schema_synthesis_enabled:
                d["consolidation"]["schema_synthesis_enabled"] = True
                d["consolidation"]["schema_synthesis_tau"] = self.consolidation.schema_synthesis_tau
                d["consolidation"]["schema_synthesis_min_supports"] = self.consolidation.schema_synthesis_min_supports
            if self.consolidation.appraisal_salience_cap is not None:
                d["consolidation"]["appraisal_salience_cap"] = (
                    self.consolidation.appraisal_salience_cap
                )
            # Only emit non-default schema thresholds to keep round-trip clean.
            if self.consolidation.schema_promote_threshold != 3:
                d["consolidation"]["schema_promote_threshold"] = (
                    self.consolidation.schema_promote_threshold
                )
            if self.consolidation.schema_deprecate_threshold != 2:
                d["consolidation"]["schema_deprecate_threshold"] = (
                    self.consolidation.schema_deprecate_threshold
                )
            if self.consolidation.schema_recover_threshold != 3:
                d["consolidation"]["schema_recover_threshold"] = (
                    self.consolidation.schema_recover_threshold
                )
        d["forgetting"] = {
            "decay_enabled": self.forgetting.decay_enabled,
            "fade_threshold": self.forgetting.fade_threshold,
            "auto_suppress": self.forgetting.auto_suppress,
            "suppress_threshold": self.forgetting.suppress_threshold,
        }
        d["retrieval"] = {
            "bm25_weight": self.retrieval.bm25_weight,
            "vector_weight": self.retrieval.vector_weight,
            "salience_weight": self.retrieval.salience_weight,
            "recency_weight": self.retrieval.recency_weight,
            "context_weight": self.retrieval.context_weight,
            "confidence_threshold": self.retrieval.confidence_threshold,
            "entity_weight": self.retrieval.entity_weight,
            "entity_ner": self.retrieval.entity_ner,
        }
        if self.retrieval.query_expansion_min_dominance is not None:
            d["retrieval"]["query_expansion_min_dominance"] = (
                self.retrieval.query_expansion_min_dominance
            )
            d["retrieval"]["query_expansion_top_k"] = (
                self.retrieval.query_expansion_top_k
            )
            d["retrieval"]["query_expansion_max_entities"] = (
                self.retrieval.query_expansion_max_entities
            )
            if self.retrieval.query_expansion_type_purity_min is not None:
                d["retrieval"]["query_expansion_type_purity_min"] = (
                    self.retrieval.query_expansion_type_purity_min
                )
            if self.retrieval.query_expansion_anchor_share_max is not None:
                d["retrieval"]["query_expansion_anchor_share_max"] = (
                    self.retrieval.query_expansion_anchor_share_max
                )
            if self.retrieval.query_expansion_type_allow is not None:
                # Persist as sorted list (frozensets aren't YAML-friendly).
                d["retrieval"]["query_expansion_type_allow"] = sorted(
                    self.retrieval.query_expansion_type_allow
                )
        # `query_expansion_skip_rerank_first_pass` is an independent
        # latency-tuning knob and must round-trip even when PRF is off
        # (operators may flip PRF on later, and the YAML should preserve
        # their explicit `True`). Persist outside the PRF-on guard.
        if self.retrieval.query_expansion_skip_rerank_first_pass:
            d["retrieval"]["query_expansion_skip_rerank_first_pass"] = True
        if self.retrieval.reranker:
            d["retrieval"]["reranker"] = self.retrieval.reranker
            d["retrieval"]["rerank_pool_size"] = self.retrieval.rerank_pool_size
            if self.retrieval.reranker == "share_prior":
                d["retrieval"]["share_prior_alpha"] = self.retrieval.share_prior_alpha
                if self.retrieval.share_prior_adaptive_alpha:
                    d["retrieval"]["share_prior_adaptive_alpha"] = True
        d["security"] = {
            "pii_detection": self.security.pii_detection,
            "encrypt_at_rest": self.security.encrypt_at_rest,
        }
        if self.security.auto_redact:
            d["security"]["auto_redact"] = self.security.auto_redact
        if self.security.encryption_key_source != "env":
            d["security"]["encryption_key_source"] = self.security.encryption_key_source
        if self.security.encryption_key_path:
            d["security"]["encryption_key_path"] = self.security.encryption_key_path
        if self.security.content_policy:
            d["security"]["content_policy"] = self.security.content_policy
        d["retention"] = {
            "buffer_ttl_days": self.retention.buffer_ttl_days,
            "faded_ttl_days": self.retention.faded_ttl_days,
            "suppressed_ttl_days": self.retention.suppressed_ttl_days,
            "audit_ttl_days": self.retention.audit_ttl_days,
        }
        if self.acl:
            d["acl"] = self.acl
        return d

    def save_yaml(self, path: str | None = None) -> None:
        """Save config to YAML."""
        p = Path(path or self.resolved_path / "config.yaml").expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

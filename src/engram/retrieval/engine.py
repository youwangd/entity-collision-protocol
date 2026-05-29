"""Retrieval engine — orchestrates hybrid search pipeline.

Stage 0: Intent Analysis (query → type/time/emotion routing)
Stage 1: Candidate Generation (BM25 + vector + metadata)
Stage 2: Scoring (RRF + salience + recency + context + somatic)
Stage 3: Optional LLM rerank (future)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from engram.core.types import (
    Memory,
    MemoryType,
    RecallContext,
    ScoredMemory,
)
from engram.core.config import RetrievalConfig
from engram.retrieval.entities import extract_entities, jaccard
from engram.retrieval.rerank import apply_reranker
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import VectorStore, NoVectorStore
from engram.providers.embeddings import EmbeddingProvider, NoEmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Parsed intent from a query (Stage 0)."""
    query: str
    target_types: list[MemoryType] = field(default_factory=list)  # empty = all
    temporal: str = ""  # "recent", "old", "specific_date", ""
    emotional: bool = False
    depth: str = "L1"
    keywords: list[str] = field(default_factory=list)


class IntentAnalyzer:
    """Stage 0: Classify query intent to route retrieval.

    Without LLM: heuristic keyword matching.
    With LLM: structured intent extraction (future).
    """

    TEMPORAL_RECENT_WORDS = {"recent", "recently", "lately", "today", "yesterday", "just"}
    TEMPORAL_RECENT_PHRASES = {"this week"}
    TEMPORAL_OLD_WORDS = {"before", "originally", "initially"}
    TEMPORAL_OLD_PHRASES = {"long ago", "first time"}
    FACT_SIGNALS = {"what is", "what does", "preference", "prefers", "likes", "always", "rule", "how to"}
    EPISODE_SIGNALS = {"what happened", "when did", "last time", "remember when", "that time"}
    SCHEMA_SIGNALS = {"pattern", "tendency", "usually", "often", "generally", "theme"}
    EMOTION_SIGNALS = {"felt", "feeling", "angry", "happy", "frustrated", "excited", "worried"}

    def analyze(self, query: str) -> QueryIntent:
        q_lower = query.lower()
        words = set(q_lower.split())

        intent = QueryIntent(query=query, keywords=q_lower.split())

        # Temporal hints
        if (words & self.TEMPORAL_RECENT_WORDS) or any(p in q_lower for p in self.TEMPORAL_RECENT_PHRASES):
            intent.temporal = "recent"
        elif (words & self.TEMPORAL_OLD_WORDS) or any(p in q_lower for p in self.TEMPORAL_OLD_PHRASES):
            intent.temporal = "old"

        # Type routing
        if any(sig in q_lower for sig in self.FACT_SIGNALS):
            intent.target_types.append(MemoryType.FACT)
        if any(sig in q_lower for sig in self.EPISODE_SIGNALS):
            intent.target_types.append(MemoryType.EPISODE)
        if any(sig in q_lower for sig in self.SCHEMA_SIGNALS):
            intent.target_types.append(MemoryType.SCHEMA)

        # Emotional queries
        if words & self.EMOTION_SIGNALS:
            intent.emotional = True

        return intent


class RetrievalEngine:
    """Hybrid retrieval with RRF fusion.

    Combines multiple signals:
    - BM25 text search (always available)
    - Vector similarity (when embeddings configured)
    - Metadata filters (type, state, temporal)
    - Context boost (encoding specificity)
    - Salience and recency weighting
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: RetrievalConfig | None = None,
        buffer=None,
    ):
        self.store = store
        self.vector = vector_store or NoVectorStore()
        self.embeddings = embedding_provider or NoEmbeddingProvider()
        self.config = config or RetrievalConfig()
        self.intent_analyzer = IntentAnalyzer()
        # Optional buffer for replaying schema-lifecycle events. When present
        # AND config.respect_schema_lifecycle is True, deprecated SCHEMA
        # candidates are suppressed from results.
        self._buffer = buffer
        # Mtime-keyed cache for the lifecycle snapshot. The gate's hot
        # path replays the entire CONSOLIDATION_SCHEMA_LIFECYCLE stream
        # on every recall(); when the buffer is unchanged across calls
        # (the common case) the snapshot is invariant. See
        # `CachedLifecycleSnapshot` for the cache-key invariants.
        from engram.consolidation.lifecycle_projection import CachedLifecycleSnapshot
        self._lifecycle_cache = CachedLifecycleSnapshot()

    def _build_prf_rarity_lookup(self, *, allowed_agents=None):
        """§4.15g — return a callable ``(entity:str)->float`` giving corpus
        rarity = 1 - df/N for the given surface form, computed against
        the active+fading FTS index of the underlying store.

        Lookup is per-call (cached to memoize repeated entities within
        one PRF expansion). Lenient: any sqlite/store error returns
        rarity=0.0 for that entity so the IDF gate behaves conservatively
        (drops it rather than admitting it).

        ``allowed_agents`` (§D-prf-idf-acl): when not None, restrict both
        the corpus-size denominator (N) and the document-frequency
        numerator (df) to memories whose ``agent_id`` is in the set.
        Closes the IDF-rarity ACL side-channel: without this, Alice's
        keep/drop decision for an entity in her PRF pool depends on
        Bob's private corpus (df is computed globally), letting an
        adversary detect Bob's vocabulary by observing rank perturbations
        on Alice's own queries. None = global (federated / ACL-off).
        """
        cache: dict[str, float] = {}
        store = self.store
        # Build the agent-scope clause once. Use a parameterized
        # placeholder list so SQLite handles quoting.
        if allowed_agents is None:
            scope_sql = ""
            scope_params: tuple = ()
        else:
            allowed = tuple(sorted(set(allowed_agents)))
            if not allowed:
                # Empty allow-list → empty corpus from this actor's view.
                def _empty(_e: str) -> float:
                    return 1.0  # vacuous — gate will still drop unless e in pool
                return _empty
            placeholders = ",".join("?" for _ in allowed)
            scope_sql = f" AND m.agent_id IN ({placeholders})"
            scope_params = allowed

        def _rarity(entity: str) -> float:
            if not entity:
                return 0.0
            if entity in cache:
                return cache[entity]
            try:
                conn = store._get_conn()
                # Active corpus size — denominator (scoped).
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories m "
                    "WHERE m.state IN ('active','fading')" + scope_sql,
                    scope_params,
                ).fetchone()
                n = int(row["c"]) if row else 0
                if n <= 0:
                    cache[entity] = 1.0
                    return 1.0
                fts_q = store._sanitize_fts_query(entity)
                if not fts_q.strip():
                    cache[entity] = 0.0
                    return 0.0
                df_row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories_fts fts "
                    "JOIN memories m ON m.rowid = fts.rowid "
                    "WHERE memories_fts MATCH ? "
                    "AND m.state IN ('active','fading')" + scope_sql,
                    (fts_q, *scope_params),
                ).fetchone()
                df = int(df_row["c"]) if df_row else 0
                rarity = max(0.0, 1.0 - df / n)
            except Exception:
                rarity = 0.0
            cache[entity] = rarity
            return rarity

        return _rarity

    def _search_with_prf(
        self,
        query: str,
        limit: int,
        depth: str,
        context: RecallContext | None,
        include_faded: bool,
        include_suppressed: bool,
        acl_filter=None,
        rarity_allowed_agents=None,
    ) -> list[ScoredMemory]:
        """PRF entity-based query expansion path (extracted for §D15 gate).

        Runs a first-pass retrieval, mines dominant entities from the
        top-K texts, and recurses (via `search(..., _prf_recursing=True)`)
        with the expanded query. Falls back to the un-expanded first
        pass when no entity passes the dominance gate or any error
        occurs in expansion.
        """
        top_k = max(1, int(self.config.query_expansion_top_k))
        # §4.15-profile lever: optionally skip the reranker on the first
        # pass (whose results are only used to mine entities). The second
        # pass below still reranks. Inert when reranker is None or the
        # config flag is False.
        skip_first_rerank = bool(
            self.config.query_expansion_skip_rerank_first_pass
            and self.config.reranker
        )
        first_pass = self.search(
            query,
            limit=max(limit, top_k),
            depth=depth,
            context=context,
            include_faded=include_faded,
            include_suppressed=include_suppressed,
            _prf_recursing=True,
            _suppress_rerank=skip_first_rerank,
        )
        # §D-prf-acl: drop cross-agent docs from the entity-mining pool.
        # Without this, PRF mines entities from memories the actor can't
        # READ (ACL side-channel: Alice's expanded query — and therefore
        # her final ranking over her *own* docs — depends on Bob's
        # private corpus). The outer `Engram.recall()` filter still
        # strips cross-agent results from the user-visible output, but
        # the PRF feedback loop already happened upstream of that point.
        if acl_filter is not None:
            first_pass = [r for r in first_pass if acl_filter(r.memory)]
        try:
            from engram.retrieval.expansion import expand_query, expand_query_typed

            texts = [(r.memory.content or "") for r in first_pass]
            if self.config.query_expansion_type_purity_min is not None:
                expanded, chosen = expand_query_typed(
                    query,
                    texts,
                    top_k=top_k,
                    max_entities=int(self.config.query_expansion_max_entities),
                    min_dominance=float(
                        self.config.query_expansion_min_dominance
                    ),
                    type_purity_min=float(
                        self.config.query_expansion_type_purity_min
                    ),
                    backend=self.config.entity_ner,
                )
            else:
                rarity_lookup = None
                idf_min_rarity = self.config.query_expansion_idf_min_rarity
                if idf_min_rarity is not None:
                    rarity_lookup = self._build_prf_rarity_lookup(
                        allowed_agents=rarity_allowed_agents
                    )
                expanded, chosen = expand_query(
                    query,
                    texts,
                    top_k=top_k,
                    max_entities=int(self.config.query_expansion_max_entities),
                    min_dominance=float(
                        self.config.query_expansion_min_dominance
                    ),
                    backend=self.config.entity_ner,
                    idf_min_rarity=idf_min_rarity,
                    rarity_lookup=rarity_lookup,
                    anchor_share_max=self.config.query_expansion_anchor_share_max,
                )
            if chosen:
                return self.search(
                    expanded,
                    limit=limit,
                    depth=depth,
                    context=context,
                    include_faded=include_faded,
                    include_suppressed=include_suppressed,
                    _prf_recursing=True,
                )
        except Exception:
            # Expansion must never break retrieval. Lenient fail.
            pass
        return first_pass[:limit]

    def search(
        self,
        query: str,
        limit: int = 5,
        depth: str = "L1",
        context: RecallContext | None = None,
        include_faded: bool = False,
        include_suppressed: bool = False,
        _prf_recursing: bool = False,
        _suppress_rerank: bool = False,
        _acl_filter=None,
        _rarity_allowed_agents=None,
    ) -> list[ScoredMemory]:
        """Full hybrid retrieval pipeline.

        ``_acl_filter`` (private): optional callable
        ``(Memory) -> bool`` used by the PRF expansion path to drop
        cross-agent docs from the entity-mining pool. Closes the §D-prf-acl
        side-channel where Alice's expanded query depended on Bob's
        private corpus. None = no filtering (federated / single-actor mode).

        ``_rarity_allowed_agents`` (private): optional iterable of
        allowed ``agent_id`` values. Threaded into the PRF rarity lookup
        so the §4.15g IDF gate's df/N is computed only over actor-visible
        memories. Closes the §D-prf-idf-acl sibling side-channel.
        """

        # Stage 0: PRF query expansion (§5.4 angle 1). When the
        # min_dominance gate is configured, run a first-pass search,
        # mine entities from the top-K texts, and recurse once with the
        # expanded query. The `_prf_recursing` guard prevents infinite
        # recursion. No-op when min_dominance is None or recursing.
        if (
            not _prf_recursing
            and self.config.query_expansion_min_dominance is not None
        ):
            # §D15 type-conditional gate. When configured, only run PRF
            # expansion for queries whose heuristic type is in the
            # allow-set. Out-of-allow queries fall through to the normal
            # (non-expanded) pipeline below. Failure-tolerant: any
            # classifier exception falls through to "expand as before".
            if self.config.query_expansion_type_allow is not None:
                try:
                    from engram.retrieval.type_classifier import (
                        classify_question_type,
                    )

                    label = classify_question_type(query).label
                    if (
                        label is None
                        or label
                        not in self.config.query_expansion_type_allow
                    ):
                        # Skip the entire PRF block; fall through to the
                        # standard pipeline at Stage 0/1 below.
                        pass
                    else:
                        return self._search_with_prf(
                            query,
                            limit=limit,
                            depth=depth,
                            context=context,
                            include_faded=include_faded,
                            include_suppressed=include_suppressed,
                            acl_filter=_acl_filter,
                            rarity_allowed_agents=_rarity_allowed_agents,
                        )
                except Exception:
                    # Classifier must never break retrieval. Fall through
                    # to the un-gated PRF path (preserves prior behaviour).
                    return self._search_with_prf(
                        query,
                        limit=limit,
                        depth=depth,
                        context=context,
                        include_faded=include_faded,
                        include_suppressed=include_suppressed,
                        acl_filter=_acl_filter,
                        rarity_allowed_agents=_rarity_allowed_agents,
                    )
            else:
                return self._search_with_prf(
                    query,
                    limit=limit,
                    depth=depth,
                    context=context,
                    include_faded=include_faded,
                    include_suppressed=include_suppressed,
                    acl_filter=_acl_filter,
                    rarity_allowed_agents=_rarity_allowed_agents,
                )

        # Stage 0: Intent Analysis
        intent = self.intent_analyzer.analyze(query)
        intent.depth = depth

        # Build state filter
        states = ["active", "fading"]
        if include_faded:
            states.append("faded")
        if include_suppressed:
            states.append("suppressed")

        # Stage 1: Candidate Generation
        candidates: dict[str, ScoredMemory] = {}

        # BM25 candidates
        bm25_results = self.store.search_text(query, limit=limit * 5, states=states)
        for i, result in enumerate(bm25_results):
            mid = result.memory.id
            if mid not in candidates:
                candidates[mid] = result
            candidates[mid].sources["bm25_rank"] = i + 1

        # Vector candidates (if available)
        if self.embeddings.dimension > 0:
            query_vec = self.embeddings.embed(query)
            vec_results = self.vector.search(query_vec, limit=limit * 5)
            for i, vr in enumerate(vec_results):
                if vr.memory_id not in candidates:
                    # Need to fetch the memory
                    memory = self.store.get(vr.memory_id)
                    if memory and memory.state.value in states:
                        candidates[vr.memory_id] = ScoredMemory(
                            memory=memory, score=0.0, sources={}
                        )
                if vr.memory_id in candidates:
                    candidates[vr.memory_id].sources["vector_rank"] = i + 1
                    candidates[vr.memory_id].sources["vector_score"] = vr.score

        # §D-vector-acl: drop cross-agent candidates from the BM25/vector
        # pool BEFORE RRF fusion, then re-number ranks so they are
        # contiguous over the actor-visible candidates. Without this,
        # Alice's surviving docs carry `bm25_rank`/`vector_rank` values
        # whose positions were determined by a global pool that includes
        # Bob's private docs — a presence oracle distinct from the PRF /
        # share_prior / lifecycle channels.
        #
        # Residual (documented in paper §6.11): FTS5's BM25 score itself
        # is computed over global corpus statistics (avgdl, df, N).
        # Re-numbering closes the rank-position channel; the score-magnitude
        # channel is a much weaker signal (FTS5 BM25 only enters scoring
        # via the rank, not the raw score).
        if _acl_filter is not None and candidates:
            kept = {
                mid: sm for mid, sm in candidates.items()
                if _acl_filter(sm.memory)
            }
            # Re-number bm25_rank in the surviving pool, preserving the
            # original BM25 order. Ditto vector_rank.
            for key in ("bm25_rank", "vector_rank"):
                ordered = sorted(
                    (sm for sm in kept.values() if key in sm.sources),
                    key=lambda s: s.sources[key],
                )
                for new_rank, sm in enumerate(ordered, start=1):
                    sm.sources[key] = new_rank
            candidates = kept

        # Filter by intent target types
        if intent.target_types:
            type_values = {t for t in intent.target_types}
            candidates = {
                mid: sm for mid, sm in candidates.items()
                if sm.memory.type in type_values
            }

        # Schema-lifecycle filter: drop SCHEMA candidates whose status has
        # been DEPRECATED in the buffer's lifecycle event stream. No-op when
        # no buffer is wired or no SCHEMA candidates are in the result set.
        # See engram/consolidation/lifecycle_projection.py for the wire format.
        if (
            self.config.respect_schema_lifecycle
            and self._buffer is not None
            and any(sm.memory.type == MemoryType.SCHEMA for sm in candidates.values())
        ):
            try:
                from engram.consolidation.schema_lifecycle import SchemaStatus

                snap = self._lifecycle_cache.get(
                    self._buffer,
                    strict=False,
                    deprecate_quorum_k=getattr(self.config, "deprecate_quorum_k", 1),
                )
                deprecated_ids = {
                    sid for sid, st in snap.items()
                    if st.status is SchemaStatus.DEPRECATED
                }
                if deprecated_ids:
                    candidates = {
                        mid: sm for mid, sm in candidates.items()
                        if not (
                            sm.memory.type == MemoryType.SCHEMA
                            and sm.memory.id in deprecated_ids
                        )
                    }
            except Exception:
                # Lifecycle replay must never break retrieval. Lenient fail.
                pass

        # Stage 2: Scoring
        now = datetime.now(timezone.utc)
        # Pre-extract query entities once (only if entity channel is on).
        query_entities: set[str] = set()
        # Per-call memoization: extracting entities from the same memory
        # content twice in a single retrieve() is pure waste. Cache by
        # memory.id (memories are immutable within a retrieve call). Only
        # populated when the entity channel is on.
        entity_cache: dict[str, set[str]] = {}
        if self.config.entity_weight > 0.0:
            query_entities = extract_entities(query, backend=self.config.entity_ner)
        for mid, sm in candidates.items():
            score = self._compute_score(sm, intent, context, now, query_entities, entity_cache)
            sm.score = score

        # Sort and limit
        results = sorted(candidates.values(), key=lambda r: r.score, reverse=True)

        # Stage 3: Optional post-rerank over a wider pool than `limit`.
        # Configured via cfg.reranker (registry name) + cfg.rerank_pool_size.
        # No-op + lenient on errors when no reranker is registered.
        if self.config.reranker and not _suppress_rerank:
            # §D-share-prior-acl: drop cross-agent docs from the reranker's
            # candidate pool. Without this, the §96 share_prior reranker's
            # entity-sharing graph spans memories the actor can't READ —
            # so the multi-mate degree counted for Alice's own docs (and
            # the global `max_deg` normaliser) depend on Bob's private
            # corpus. The outer `Engram.recall()` ACL filter strips
            # cross-agent docs from the user-visible output, but the
            # rerank-induced score perturbations on Alice's surviving
            # docs already happened. Closes the side-channel oracle.
            # None = federated / single-actor mode (no filtering).
            if _acl_filter is not None:
                results = [r for r in results if _acl_filter(r.memory)]
            pool_size = max(limit, int(self.config.rerank_pool_size or 0))
            pool = results[:pool_size]
            tail = results[pool_size:]
            reranked = apply_reranker(
                self.config.reranker,
                pool,
                query=query,
                intent=intent,
                entity_cache=entity_cache,
                query_entities=query_entities,
                cfg=self.config,
                alpha=getattr(self.config, "share_prior_alpha", 0.05),
            )
            results = list(reranked) + tail

        results = results[:limit]

        # Confidence assessment (Design §4.2: "When top_score < confidence_threshold,
        # return partial results with confidence='low' flag")
        confidence = "high"
        if not results:
            confidence = "none"
        elif results[0].score < self.config.confidence_threshold:
            confidence = "low"

        # Store confidence on each result
        for r in results:
            r.sources["confidence"] = confidence

        # Apply depth filtering (Design §4.6: L0/L1/L2 tiered loading)
        for r in results:
            r.memory = self._apply_depth(r.memory, intent.depth)

        return results

    def _apply_depth(self, memory: Memory, depth: str) -> Memory:
        """Filter memory fields based on requested depth. Returns a COPY — never mutates original.

        L0 (~20 tokens): summary + type + salience only
        L1 (~100 tokens): + somatic bias + key metadata
        L2 (full): everything
        """
        if depth == "L2":
            return memory  # full detail — no copy needed

        m = copy.copy(memory)

        if depth == "L0":
            m.content = m.summary or m.content[:80]
            m.encoding_context = type(m.encoding_context)()
            m.provenance = type(m.provenance)()
            m.source_events = []
            return m

        # L1 (default): keep content + somatic + emotion but strip modification history
        m.provenance = type(m.provenance)(
            source_events=m.provenance.source_events,
            created_by=m.provenance.created_by,
            modifications=[],
        )
        return m

    def _compute_score(
        self,
        sm: ScoredMemory,
        intent: QueryIntent,
        context: RecallContext | None,
        now: datetime,
        query_entities: set[str] | None = None,
        entity_cache: dict[str, set[str]] | None = None,
    ) -> float:
        """Compute final score using RRF + weighted signals."""
        memory = sm.memory
        sources = sm.sources
        cfg = self.config

        # Reciprocal Rank Fusion for search signals
        rrf_k = 60  # standard RRF constant
        rrf_score = 0.0

        if "bm25_rank" in sources:
            rrf_score += cfg.bm25_weight * (1.0 / (rrf_k + sources["bm25_rank"]))
        if "vector_rank" in sources:
            rrf_score += cfg.vector_weight * (1.0 / (rrf_k + sources["vector_rank"]))

        # If only one signal, don't penalize
        if "bm25_rank" in sources and "vector_rank" not in sources:
            rrf_score += cfg.vector_weight * (1.0 / (rrf_k + 100))  # gentle penalty
        if "vector_rank" in sources and "bm25_rank" not in sources:
            rrf_score += cfg.bm25_weight * (1.0 / (rrf_k + 100))

        # Normalize RRF to ~0-1 range
        max_rrf = (cfg.bm25_weight + cfg.vector_weight) * (1.0 / (rrf_k + 1))
        rrf_normalized = rrf_score / max_rrf if max_rrf > 0 else 0

        # Salience
        salience_score = memory.salience * cfg.salience_weight

        # Recency
        last = memory.last_accessed or memory.created_at
        hours_ago = max((now - last).total_seconds() / 3600, 0.01)
        recency_score = cfg.recency_weight * (1.0 / (1.0 + math.log1p(hours_ago)))

        # Context boost (encoding specificity)
        context_boost = 1.0
        if context:
            context_boost = self._context_boost(memory, context)
        context_score = cfg.context_weight * (context_boost - 1.0)

        # Somatic marker boost
        somatic_boost = 0.0
        if memory.somatic and memory.somatic.valence != 0:
            somatic_boost = abs(memory.somatic.valence) * 0.1

        # Temporal boost from intent
        temporal_boost = 0.0
        if intent.temporal == "recent":
            temporal_boost = recency_score * 0.5  # double-weight recency
        elif intent.temporal == "old":
            temporal_boost = -recency_score * 0.3  # de-weight recency

        # Entity-link channel (D1, Mem0 v3-inspired).
        # Jaccard(query_entities, memory_entities) * cfg.entity_weight.
        # Per-call cache keyed by memory.id avoids redundant extraction
        # when the same memory is scored more than once.
        entity_score = 0.0
        if cfg.entity_weight > 0.0 and query_entities:
            if entity_cache is not None and memory.id in entity_cache:
                mem_entities = entity_cache[memory.id]
            else:
                mem_entities = extract_entities(memory.content or "", backend=cfg.entity_ner)
                if entity_cache is not None:
                    entity_cache[memory.id] = mem_entities
            j = jaccard(query_entities, mem_entities)
            entity_score = cfg.entity_weight * j

        final = (
            rrf_normalized
            + salience_score
            + recency_score
            + context_score
            + somatic_boost
            + temporal_boost
            + entity_score
        )

        # Confidence factor — memories with low extraction confidence get penalized
        confidence_factor = memory.confidence if memory.confidence > 0 else 1.0
        final *= confidence_factor

        # Extraction confidence (Governed Memory paper, arXiv:2603.17787) — gated multiplier
        # Multiplied into the final score when RetrievalConfig.use_extraction_confidence is True.
        # Penalises facts the extractor was uncertain about (e.g. inferred vs explicit).
        extraction_factor = 1.0
        if cfg.use_extraction_confidence:
            ec = getattr(memory, "extraction_confidence", 1.0)
            if ec is None:
                ec = 1.0
            extraction_factor = max(0.0, min(1.0, float(ec)))
            final *= extraction_factor

        # Store component scores for debugging
        sm.sources["rrf"] = round(rrf_normalized, 4)
        sm.sources["salience"] = round(salience_score, 4)
        sm.sources["recency"] = round(recency_score, 4)
        sm.sources["context"] = round(context_score, 4)
        sm.sources["somatic"] = round(somatic_boost, 4)
        sm.sources["entity"] = round(entity_score, 4)
        sm.sources["confidence"] = round(confidence_factor, 4)
        sm.sources["extraction_confidence"] = round(extraction_factor, 4)

        return max(final, 0.0)

    def _context_boost(self, memory: Memory, context: RecallContext) -> float:
        """Encoding specificity boost (Tulving).

        Memories encoded in similar mood/task/emotion get boosted.
        """
        boost = 1.0
        ec = memory.encoding_context

        # Mood similarity
        if context.mood_valence is not None and ec.mood_valence is not None:
            mood_diff = abs(context.mood_valence - ec.mood_valence)
            arousal_diff = abs((context.mood_arousal or 0.5) - (ec.mood_arousal or 0.5))
            mood_sim = 1.0 - (mood_diff + arousal_diff) / 2.0
            boost *= (0.7 + 0.3 * mood_sim)

        # Task match
        if context.task and ec.task:
            if context.task.lower() == ec.task.lower():
                boost *= 1.3
            elif context.task.lower() in ec.task.lower() or ec.task.lower() in context.task.lower():
                boost *= 1.15

        # Emotion overlap
        if context.emotions and ec.emotions:
            # Normalize: emotions can be strings or dicts with "primary" key
            ctx_emo = {(e["primary"] if isinstance(e, dict) else str(e)) for e in context.emotions}
            enc_emo = {(e["primary"] if isinstance(e, dict) else str(e)) for e in ec.emotions}
            overlap = len(ctx_emo & enc_emo)
            total = max(len(ctx_emo), 1)
            emotion_sim = overlap / total
            boost *= (0.8 + 0.2 * emotion_sim)

        return boost

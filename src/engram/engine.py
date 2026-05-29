"""Engram — main engine class that wires all components together."""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engram.core.types import (
    ConsolidationReport,
    DataClassification,
    Event,
    EventType,
    Memory,
    MemoryState,
    MemoryType,
    Modification,
    RecallContext,
    ScoredMemory,
    content_hash,
    generate_event_id,
)
from engram.core.config import Config, ConsolidationConfig
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import VectorStore, NoVectorStore, SQLiteVecStore
from engram.audit.log import AuditLog
from engram.consolidation.pipeline import ConsolidationPipeline
from engram.retrieval.engine import RetrievalEngine
from engram.providers.llm import LLMProvider, NoLLMProvider
from engram.providers.embeddings import EmbeddingProvider, NoEmbeddingProvider
from engram.security.firewall import MemoryFirewall, FirewallConfig
from engram.security.acl import AccessPolicy, Permission
from engram.security.encryption import ContentEncryptor
from engram.affect.engine import AffectEngine, Temperament

logger = logging.getLogger(__name__)


class Engram:
    """Main entry point for the Engram memory system.

    Wires event store, memory store, retrieval engine, consolidation pipeline,
    affect engine, and security firewall into a single coherent API.
    """

    def __init__(self, config: Config | str | None = None, actor: str = "system",
                 llm: LLMProvider | None = None,
                 embeddings: EmbeddingProvider | None = None):
        if config is None:
            config = Config.minimal()
        elif isinstance(config, str):
            config = Config.from_yaml(config)
        elif isinstance(config, Path):
            config = Config.from_yaml(str(config))

        self.config = config
        self._actor = actor
        self._base_path = config.resolved_path
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Core stores
        # Encryption at rest (Design §5.5)
        # Encrypts JSONL events only — SQLite stays plaintext for FTS indexing.
        self._encryptor = ContentEncryptor(
            enabled=config.security.encrypt_at_rest,
            key_source=config.security.encryption_key_source,
            key_path=config.security.encryption_key_path,
        )

        self._buffer = JSONLBufferStore(self._base_path, encryptor=self._encryptor)
        self._store = SQLiteMemoryStore(self._base_path)
        self._audit = AuditLog(self._base_path)
        self._llm = llm or NoLLMProvider()
        self._embeddings = embeddings or NoEmbeddingProvider()
        # Serialises the write-side cosine-dedup critical section
        # (cosine check → projection insert → vector upsert) so concurrent
        # remember() calls cannot all pass an empty-vector-store dedup check
        # before any of them have committed. Only contended when dedup is on.
        self._dedup_lock = threading.RLock()

        # Vector store (Tier 0 if no embeddings, Tier 1 if embeddings available)
        if self._embeddings.dimension > 0:
            self._vector: VectorStore = SQLiteVecStore(
                self._base_path / "vectors.db", dimension=self._embeddings.dimension
            )
        else:
            self._vector = NoVectorStore()

        # Retrieval engine
        self._retrieval = RetrievalEngine(
            store=self._store,
            vector_store=self._vector,
            embedding_provider=self._embeddings,
            config=config.retrieval,
            buffer=self._buffer,
        )

        # Security firewall
        fw_config = FirewallConfig(
            pii_detection=config.security.pii_detection,
            pii_action="redact" if config.security.auto_redact else "warn",
            auto_redact_patterns=config.security.auto_redact,
            content_policy=config.security.content_policy,
            max_events_per_minute=config.security.max_events_per_minute,
            injection_detection=config.security.injection_detection,
        )
        self._firewall = MemoryFirewall(fw_config)

        # Affect engine — load persisted state if available
        if config.affect and config.affect.temperament:
            if isinstance(config.affect.temperament, str):
                temperament = Temperament.preset(config.affect.temperament)
            else:
                temperament = Temperament.from_dict(config.affect.temperament)
        else:
            temperament = Temperament()

        # Try to restore persisted temperament
        saved_temperament = self._store.get_latest_affect("temperament")
        if saved_temperament:
            temperament = Temperament.from_dict(saved_temperament)

        self._affect = AffectEngine(temperament=temperament)

        # Restore persisted mood
        saved_mood = self._store.get_latest_affect("mood")
        if saved_mood:
            self._affect.mood.valence = saved_mood.get("valence", 0.0)
            self._affect.mood.arousal = saved_mood.get("arousal", 0.0)

        logger.info("engram initialized at %s", self._base_path)

        # Access control (Design §5.3)
        if config.acl and isinstance(config.acl, dict):
            self._acl = AccessPolicy.from_dict(config.acl)
        else:
            self._acl = AccessPolicy(enabled=False)

        # Public sub-API (Design §8: mem.affect.status(), mem.affect.mood(), etc.)
        self.affect = AffectAPI(self)

    @property
    def acl(self) -> AccessPolicy:
        """Access the ACL policy (Design §5.3)."""
        return self._acl

    @property
    def encryptor(self) -> ContentEncryptor:
        """Access the content encryptor (Design §5.5)."""
        return self._encryptor

    def close(self) -> None:
        """Clean up resources."""
        self._store.close()
        self._vector.close()

    def __enter__(self) -> Engram:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- Memory Operations ---

    def remember(self, content: str, salience: float = 0.0, memory_type: MemoryType = MemoryType.FACT,
                 agent_id: str | None = None, **metadata: Any) -> str:
        """Store a memory. Returns event ID.

        Raises ValueError if content is empty or salience out of range.
        Raises PermissionError if ACL denies write.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        salience = max(0.0, min(1.0, salience))

        # ACL check
        actor = agent_id or self._actor
        self._acl.check(actor, Permission.WRITE)

        start = time.monotonic()

        # Security: validate content through firewall
        content = self._firewall.validate(content, actor=self._actor)

        event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.EXPLICIT_REMEMBER,
            content=content,
            metadata=metadata,
            salience_hint=salience,
            context={
                "mood_valence": self._affect.mood.valence,
                "mood_arousal": self._affect.mood.arousal,
                "active_emotions": [e.primary for e in self._affect.active_emotions],
            },
        )

        # Append to event store (source of truth)
        self._buffer.append(event)

        # Write-through to SQLite for immediate availability
        # NOTE: SQLite stores PLAINTEXT (it's a projection for FTS search).
        # Encryption is applied only to the JSONL event store (source of truth on disk).
        memory = Memory.from_event(event, memory_type=memory_type)
        memory.agent_id = actor

        # Auto-classify content (Design §5.2)
        classification = self._firewall.classify(content)
        try:
            memory.classification = DataClassification(classification)
        except ValueError:
            memory.classification = DataClassification.PUBLIC

        # Auto-redact RESTRICTED content if configured (Design §5.2)
        if memory.classification == DataClassification.RESTRICTED and self.config.security.auto_redact:
            redact_fw = MemoryFirewall(FirewallConfig(
                pii_detection=True, pii_action="redact",
                auto_redact_patterns=self.config.security.auto_redact,
            ))
            memory.content = redact_fw.validate(content, actor=self._actor)

        # Write-side cosine dedup (Governed Memory paper, arXiv:2603.17787).
        # If write_dedup_threshold > 0, store.upsert() returns False when a near-duplicate
        # already exists, and we skip the rest of the write path. The event still hits
        # the JSONL buffer above (source of truth), so consolidation can still process it.
        # Lock only when dedup is active (avoid serialising the hot path otherwise).
        dedup_active = (
            self.config.storage.write_dedup_threshold > 0
            and self._embeddings.dimension > 0
        )
        if dedup_active:
            self._dedup_lock.acquire()
        try:
            # ACL-scope dedup: only neighbours visible to the writer count
            # as duplicates. Without this, vector_store.search() runs
            # globally and Bob's write can be silently deduped against
            # Alice's memory — a presence oracle on the audit + recall
            # channels (§D-write-dedup-acl).
            if self._acl.enabled:
                acl_filter = lambda cand_agent_id: self._acl_allows_read(actor, cand_agent_id)
            else:
                acl_filter = None
            written = self._store.upsert(
                memory,
                dedup_threshold=self.config.storage.write_dedup_threshold,
                vector_store=self._vector,
                embedding_provider=self._embeddings,
                acl_filter=acl_filter,
            )
            if not written:
                elapsed = int((time.monotonic() - start) * 1000)
                self._audit.log(
                    "remember_deduped", self._actor,
                    {"content_hash": content_hash(content), "memory_id": memory.id},
                    "skipped", elapsed,
                )
                logger.debug("remember: deduped (cosine > %.2f) — skipped projection write for %s",
                             self.config.storage.write_dedup_threshold, memory.id)
                return event.id

            # Store vector embedding if available — must happen INSIDE the dedup lock
            # so the next concurrent remember() sees this vector during its cosine check.
            if self._embeddings.dimension > 0:
                try:
                    vec = self._embeddings.embed(content)
                    self._vector.upsert(memory.id, vec)
                except Exception as e:
                    logger.warning("failed to embed memory %s: %s", memory.id, e)
        finally:
            if dedup_active:
                self._dedup_lock.release()

        elapsed = int((time.monotonic() - start) * 1000)
        self._audit.log(
            "remember", self._actor,
            {"content_hash": content_hash(content), "salience": salience, "type": memory_type.value, "memory_id": memory.id},
            "success", elapsed,
        )
        logger.debug("remembered: %s (salience=%.2f, type=%s)", memory.id, salience, memory_type.value)
        return event.id

    def capture(self, content: str, event_type: EventType = EventType.EVENT_CAPTURE, **metadata: Any) -> str:
        """Capture a raw event to the buffer. Not immediately searchable."""
        content = self._firewall.validate(content, actor=self._actor)
        event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=event_type,
            content=content,
            metadata=metadata,
        )
        self._buffer.append(event)
        return event.id

    def recall(
        self,
        query: str,
        limit: int = 5,
        depth: str = "L1",
        context: RecallContext | None = None,
        include_faded: bool = False,
        include_suppressed: bool = False,
        agent_id: str | None = None,
        federated: list[Engram] | None = None,
    ) -> list[ScoredMemory]:
        """Search memories using hybrid retrieval.

        Args:
            federated: Optional list of other Engram instances to query across.
                       Requires 'federated' permission in ACL. (Design §9)
        Raises ValueError if query is empty.
        Raises PermissionError if ACL denies read.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # ACL check
        actor = agent_id or self._actor
        self._acl.check(actor, Permission.READ)

        start = time.monotonic()

        # Auto-fill context from current affect state if not provided
        if context is None:
            state = self._affect.get_current_state()
            active_emotions = state.get("active_emotions", [])
            emotion_names = [e["primary"] if isinstance(e, dict) else str(e) for e in active_emotions]
            context = RecallContext(
                mood_valence=state["mood_valence"],
                mood_arousal=state["mood_arousal"],
                emotions=emotion_names,
            )

        # Build the PRF-side ACL filter for this actor. The retrieval
        # engine consumes it inside `_search_with_prf` to drop cross-agent
        # docs from the entity-mining pool. Closes the §D-prf-acl
        # side-channel: without this, the expanded query (and therefore
        # the actor's own ranking) depends on memories the actor cannot
        # READ. Inert when ACL is disabled. Federated cross-agent reads
        # remain governed by the existing FEDERATED grant — see below.
        if self._acl.enabled:
            def _prf_acl_filter(mem) -> bool:
                return self._acl_allows_read(actor, getattr(mem, "agent_id", None))
        else:
            _prf_acl_filter = None

        # §D-prf-idf-acl: scope the §4.15g IDF rarity df/N to actor-visible
        # memories. Without this, Alice's keep/drop decision for a pool
        # entity depends on Bob's private corpus (df is global), enabling
        # an entity-presence oracle. None = global (federated/scope='*' or
        # ACL disabled).
        _prf_rarity_allowed = self._prf_rarity_allowed_agents(actor)

        results = self._retrieval.search(
            query, limit=limit, depth=depth, context=context,
            include_faded=include_faded, include_suppressed=include_suppressed,
            _acl_filter=_prf_acl_filter,
            _rarity_allowed_agents=_prf_rarity_allowed,
        )

        # Track local memory IDs before merging federated results
        local_memory_ids = {r.memory.id for r in results}

        # Federated recall: query across other Engram instances (Design §9)
        if federated:
            self._acl.check(actor, Permission.FEDERATED)
            for other in federated:
                try:
                    remote_results = other._retrieval.search(
                        query, limit=limit, depth=depth, context=context,
                        include_faded=include_faded, include_suppressed=include_suppressed,
                    )
                    results.extend(remote_results)
                except Exception as e:
                    logger.warning("federated recall failed for %s: %s", getattr(other, '_base_path', '?'), e)
            # Re-sort merged results and take top limit
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:limit]

        # Mark accessed (spaced repetition — resets decay clock)
        # + Reconsolidation: update encoding context if current context diverges (Nader et al., 2000)
        # Only for local memories (not federated results from other stores)
        for r in results:
            if r.memory.id not in local_memory_ids:
                continue
            self._store.mark_accessed(r.memory.id)
            original = self._store.get(r.memory.id)
            if original:
                self._reconsolidate(original, context)

        # Log recall events to event store (Design §3.1: recall_request + recall_hit)
        self._buffer.append(Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.RECALL_REQUEST,
            content=query,
            metadata={"limit": limit, "depth": depth, "results": len(results)},
        ))
        for r in results:
            self._buffer.append(Event(
                id=generate_event_id(),
                ts=datetime.now(timezone.utc),
                type=EventType.RECALL_HIT,
                content=r.memory.id,
                metadata={"score": round(r.score, 3), "memory_id": r.memory.id},
            ))

        # ACL: filter results by agent scope (must happen before audit logging)
        if self._acl.enabled:
            filtered = []
            for r in results:
                try:
                    self._acl.check(actor, Permission.READ, memory_agent_id=r.memory.agent_id)
                    filtered.append(r)
                except PermissionError:
                    continue
            results = filtered

        # Audit (after ACL filtering so counts reflect what caller actually receives)
        elapsed = int((time.monotonic() - start) * 1000)
        self._audit.log(
            "recall", self._actor,
            {
                "query": query,
                "results": len(results),
                "top_score": round(results[0].score, 3) if results else 0,
                "depth": depth,
            },
            "success", elapsed,
        )

        return results

    def recall_with_filters(
        self,
        query: str | None = None,
        properties: dict[str, str] | None = None,
        limit: int = 10,
        agent_id: str | None = None,
    ) -> list[Memory]:
        """Filter-aware recall using typed properties (Governed Memory paper §5).

        Two modes:

        1. Filter-only (query=None): pure SQL filter on `memory_properties`.
           Returns memories ranked by salience desc.

        2. Hybrid (query + properties): runs hybrid recall, then filters the
           returned memories to those whose properties match every constraint.
           Order is preserved (recall score order).

        Filter values support comparison operators on numeric properties:
            ">100", ">=100", "<100", "<=100", "==42", "!=42"
        Plain strings do equality match.

        Args:
            query: optional natural-language query for hybrid recall.
            properties: dict of {key: value-or-expression} that ALL must match.
            limit: max results.
            agent_id: actor for ACL check (defaults to engram actor).

        Returns:
            List of Memory (not ScoredMemory) — score is irrelevant for
            filter-only mode and ambiguous when intersected.

        Raises:
            ValueError if both query and properties are empty.
            PermissionError if ACL denies read.
        """
        if not query and not properties:
            raise ValueError("recall_with_filters requires query or properties")

        actor = agent_id or self._actor
        self._acl.check(actor, Permission.READ)

        # Filter-only: SQL-level intersection
        if not query:
            results = self._store.filter_by_properties(properties or {}, limit=limit)
            if self._acl.enabled:
                results = [
                    m for m in results
                    if self._acl_allows_read(actor, m.agent_id)
                ]
            return results[:limit]

        # Hybrid: recall then property-filter
        scored = self.recall(query=query, limit=max(limit * 4, 20), agent_id=agent_id)
        if not properties:
            return [s.memory for s in scored][:limit]

        # In-memory filter using store.get_properties
        kept: list[Memory] = []
        op_re = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+)$")
        for s in scored:
            mem_props = {p["key"]: p["value"] for p in self._store.get_properties(s.memory.id)}
            ok = True
            for k, raw in properties.items():
                if k not in mem_props:
                    ok = False
                    break
                m = op_re.match(str(raw))
                if m:
                    op, num = m.group(1), m.group(2).strip()
                    try:
                        actual = float(mem_props[k])
                        target = float(num)
                    except (ValueError, TypeError):
                        ok = False
                        break
                    matched = (
                        (op == ">" and actual > target)
                        or (op == ">=" and actual >= target)
                        or (op == "<" and actual < target)
                        or (op == "<=" and actual <= target)
                        or (op == "==" and actual == target)
                        or (op == "!=" and actual != target)
                    )
                    if not matched:
                        ok = False
                        break
                else:
                    if mem_props[k] != str(raw):
                        ok = False
                        break
            if ok:
                kept.append(s.memory)
            if len(kept) >= limit:
                break
        return kept

    def _acl_allows_read(self, actor: str, memory_agent_id: str | None) -> bool:
        try:
            self._acl.check(actor, Permission.READ, memory_agent_id=memory_agent_id)
            return True
        except PermissionError:
            return False

    def _prf_rarity_allowed_agents(self, actor: str) -> set[str] | None:
        """Build the agent-id allow-list used to scope the PRF IDF
        rarity df/N for ``actor``. Returns:

          * ``None`` when ACL is disabled, when the actor has scope='*'
            (or FEDERATED), or when the actor has no grant — meaning
            "no scoping; use the global corpus" (preserves prior
            single-actor / federated behaviour).
          * a ``set[str]`` of allowed agent_ids otherwise (typically
            ``{actor, ''}`` for scope='own' — empty agent_id is the
            unset / system bucket that scope='own' admits).

        Closes §D-prf-idf-acl: see RetrievalEngine._build_prf_rarity_lookup.
        """
        if not self._acl.enabled:
            return None
        grant = self._acl._grants.get(actor)
        if grant is None:
            # No grant → all reads will fail anyway; rarity won't be
            # consulted because recall() will raise. Be permissive here
            # rather than empty-allow; the outer ACL check is the gate.
            return None
        if grant.scope == "*" or Permission.FEDERATED in grant.permissions:
            return None
        # scope='own' — actor sees only its own memories plus the
        # unset/system bucket (mirrors Grant.can_access).
        return {actor, ""}

    def forget(self, id_or_query: str | None = None, *, id: str | None = None, query: str | None = None, hard: bool = False, below: float | None = None) -> int:
        """Suppress or delete memories. Returns count affected.

        Can be called as:
            mem.forget("memory-id")           # positional ID
            mem.forget(id="memory-id")         # keyword ID
            mem.forget(query="project X")      # query match
            mem.forget(below=0.1)              # salience threshold
        """
        # Support positional: mem.forget("memory-id")
        if id_or_query and not id and not query:
            if id_or_query.startswith("mem-"):
                id = id_or_query
            else:
                query = id_or_query

        # ACL check
        self._acl.check(self._actor, Permission.FORGET)

        start = time.monotonic()
        count = 0

        if id:
            if hard:
                if self._store.delete(id):
                    self._vector.delete(id)
                    self._buffer.redact_memory(id)
                    count = 1
            else:
                self._store.update_state(id, MemoryState.SUPPRESSED)
                count = 1
        elif query:
            matches = self._store.search_text(query, limit=100, states=["active", "fading", "faded", "suppressed"])
            for m in matches:
                if hard:
                    self._store.delete(m.memory.id)
                    self._vector.delete(m.memory.id)
                    self._buffer.redact_memory(m.memory.id)
                else:
                    self._store.update_state(m.memory.id, MemoryState.SUPPRESSED)
                count += 1
        elif below is not None:
            # Suppress/delete all memories below a salience threshold
            for m in self._store.all_active():
                if m.salience < below:
                    if hard:
                        self._store.delete(m.id)
                        self._vector.delete(m.id)
                        self._buffer.redact_memory(m.id)
                    else:
                        self._store.update_state(m.id, MemoryState.SUPPRESSED)
                    count += 1

        # Log the forget as an event
        event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.FORGET_REQUEST,
            content=f"forget: id={id} query={query} hard={hard} affected={count}",
            metadata={"id": id, "query": query, "hard": hard, "affected": count},
        )
        self._buffer.append(event)

        elapsed = int((time.monotonic() - start) * 1000)
        self._audit.log(
            "forget", self._actor,
            {"id": id, "query": query, "hard": hard, "affected": count},
            "success", elapsed,
        )
        return count

    def delete(self, id: str | None = None, query: str | None = None) -> int:
        """Hard-delete memories (GDPR). Alias for forget(hard=True).

        Design §8: mem.delete("memory-id", hard=True)
        """
        return self.forget(id=id, query=query, hard=True)

    # --- Active Context ---

    def pin(self, content: str) -> str:
        """Pin a fact to active context."""
        pin_id = f"pin-{uuid.uuid4().hex[:8]}"
        self._store.add_pin(pin_id, content)

        event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.PIN_ADD,
            content=content,
            metadata={"pin_id": pin_id},
        )
        self._buffer.append(event)
        return pin_id

    def unpin(self, pin_id: str) -> bool:
        """Remove a pin from active context."""
        removed = self._store.remove_pin(pin_id)
        if removed:
            event = Event(
                id=generate_event_id(),
                ts=datetime.now(timezone.utc),
                type=EventType.PIN_REMOVE,
                content=pin_id,
                metadata={"pin_id": pin_id},
            )
            self._buffer.append(event)
        return removed

    def active_context(self, max_tokens: int = 4096) -> str:
        """Get position-aware active context for injection into prompts.

        Returns: pins + mood + L0 summaries of high-salience memories + schemas.
        High-salience at top and bottom (Lost-in-the-Middle mitigation).
        """
        lines: list[str] = []

        # Current mood (Design §4.6: "Additional context included: Current mood")
        state = self._affect.get_current_state()
        lines.append(f"[MOOD] {state['mood_label']} (valence={state['mood_valence']:.2f}, arousal={state['mood_arousal']:.2f})")

        # Pins always included at L1
        pins = self._store.get_pins()
        for p in pins:
            lines.append(f"[PIN] {p['content']}")

        # Active schemas (Design §4.6: "Active schemas: relevant patterns for the current task")
        schemas = self._store.search_by_type(MemoryType.SCHEMA, limit=5)
        for s in schemas:
            lines.append(f"[SCHEMA] {s.summary or s.content[:80]} (salience={s.salience:.2f})")

        # Top active memories by salience — L0 summaries
        memories = self._store.all_active()
        # Filter out schemas (already included above)
        memories = [m for m in memories if m.type != MemoryType.SCHEMA]

        budget = max_tokens - (len(lines) * 30)
        token_per_memory = 25
        available = min(len(memories), budget // token_per_memory) if token_per_memory > 0 else 0
        selected = memories[:available]

        if not selected:
            return "\n".join(lines)

        # Position-aware: high salience at top and bottom (Liu et al., 2024)
        mid = len(selected) // 2
        top_half = selected[:mid]
        bottom_half = selected[mid:]

        for m in top_half:
            lines.append(f"[{m.type.value.upper()}] {m.summary or m.content[:80]} (salience={m.salience:.2f})")

        if bottom_half:
            lines.append("---")
            for m in reversed(bottom_half):
                lines.append(f"[{m.type.value.upper()}] {m.summary or m.content[:80]} (salience={m.salience:.2f})")

        return "\n".join(lines)

    # --- Consolidation ---

    def consolidate(self, window: str | None = None) -> ConsolidationReport:
        """Run the consolidation pipeline (the brain's 'sleep cycle').

        Args:
            window: Override consolidation window, e.g. "24h", "7d". Default from config.
        """
        # ACL check
        self._acl.check(self._actor, Permission.CONSOLIDATE)

        # Enforce retention TTLs before consolidation (Design §5.4)
        self._enforce_retention()

        # Parse optional window override
        config = self.config
        if window:
            import dataclasses as _dc
            config = copy.copy(config)
            hours = self._parse_window(window)
            base = config.consolidation or ConsolidationConfig()
            config.consolidation = _dc.replace(base, window_hours=hours)

        pipeline = ConsolidationPipeline(
            buffer=self._buffer,
            store=self._store,
            audit=self._audit,
            config=config,
            llm=self._llm,
            affect=self._affect,
            vector_store=self._vector,
            embedding_provider=self._embeddings,
        )
        return pipeline.run(actor=self._actor)

    # --- Affect (delegated to self.affect proxy; kept for backward compat) ---

    def trigger_emotion(self, primary: str, intensity: float = 0.5, trigger: str = "") -> None:
        """Trigger an emotion. Shortcut for self.affect.trigger(...)."""
        self.affect.trigger(primary, intensity, trigger)

    def status(self) -> dict[str, Any]:
        """Memory stats + health."""
        store_stats = self._store.stats()
        store_stats["buffer_events"] = self._buffer.count()
        return store_stats

    def provenance(self, memory_id: str) -> dict | None:
        """Get provenance for a memory (Design §8: mem.provenance("memory-id")).

        Returns the lineage chain: source events, created_by, modifications.
        ACL: gated by READ on the memory's owning agent (no metadata leak).
        """
        memory = self._store.get(memory_id)
        if memory is None:
            return None
        if not self._acl_allows_read(self._actor, memory.agent_id):
            return None
        return {
            "memory_id": memory.id,
            "source_events": memory.provenance.source_events,
            "created_by": memory.provenance.created_by,
            "modifications": [m.to_dict() for m in memory.provenance.modifications],
            "relations": self._store.get_relations(memory_id),
        }

    def trace(self, memory_id: str) -> dict | None:
        """Full lineage trace: source events → memory → all modifications.

        Design §6.1: Shows event → appraisal → extraction → modifications.
        ACL: gated by READ on the memory's owning agent (no metadata leak).
        """
        memory = self._store.get(memory_id)
        if memory is None:
            return None
        if not self._acl_allows_read(self._actor, memory.agent_id):
            return None

        # Find source events — single scan with ID set lookup
        target_ids = set(memory.source_events)
        source_events = []
        if target_ids:
            for event in self._buffer.scan():
                if event.id in target_ids:
                    source_events.append(event.to_dict())
                    target_ids.discard(event.id)
                    if not target_ids:
                        break  # found all

        return {
            "memory_id": memory.id,
            "type": memory.type.value,
            "state": memory.state.value,
            "content": memory.content,
            "created_at": memory.created_at.isoformat(),
            "source_events": source_events,
            "provenance": memory.provenance.to_dict(),
            "appraisal": {
                "relevance": memory.appraisal.relevance,
                "novelty": memory.appraisal.novelty,
                "goal_conduciveness": memory.appraisal.goal_conduciveness,
            },
            "somatic": {
                "valence": memory.somatic.valence,
                "bias": memory.somatic.bias,
                "trigger": memory.somatic.trigger,
            },
            "emotion": {
                "primary": memory.emotion.primary,
                "intensity": memory.emotion.intensity,
                "compound": memory.emotion.compound,
            },
            "encoding_context": {
                "mood_valence": memory.encoding_context.mood_valence,
                "mood_arousal": memory.encoding_context.mood_arousal,
                "emotions": memory.encoding_context.emotions,
                "task": memory.encoding_context.task,
            },
        }

    def schemas(self) -> list[dict]:
        """List auto-generated schemas (Design §8)."""
        schema_mems = self._store.search_by_type(MemoryType.SCHEMA, limit=50)
        return [{
            "id": s.id,
            "content": s.content,
            "summary": s.summary,
            "salience": s.salience,
            "created_at": s.created_at.isoformat(),
        } for s in schema_mems]

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        ACL: gated by READ on the memory's owning agent. Returns None if the
        actor lacks READ scope for that memory's owner — pinned by
        tests/adversarial/test_acl_escape_ops.py to prevent metadata-channel
        cross-agent leaks.
        """
        memory = self._store.get(memory_id)
        if memory is None:
            return None
        if not self._acl_allows_read(self._actor, memory.agent_id):
            return None
        return memory

    # --- Export / Import ---

    def export_memories(self, subject: str | None = None) -> list[dict]:
        """Export memories as dicts. With subject filter for GDPR DSAR (Design §5.4)."""
        self._acl.check(self._actor, Permission.EXPORT)
        memories = self._store.all_active()
        exported = []
        for m in memories:
            d = self._memory_to_export(m)
            if subject and subject.lower() not in m.content.lower():
                continue
            exported.append(d)
        return exported

    def export_dsar(self, subject: str) -> dict:
        """GDPR Data Subject Access Request — export all data for a subject.

        Returns memories + audit trail (Design §5.4).
        """
        memories = self.export_memories(subject=subject)
        audit = [e for e in self._audit.read(limit=10000) if subject.lower() in str(e).lower()]
        return {
            "subject": subject,
            "memories": memories,
            "audit_entries": audit,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def import_from(self, path: str) -> int:
        """Import memories from a JSON backup file. Returns count imported.

        Supports both full Memory dicts (roundtrip from export) and simple
        {content, salience, type} dicts. Design §8: mem.import_from("backup.json")
        """
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            memories = data
        elif isinstance(data, dict) and "memories" in data:
            memories = data["memories"]
        else:
            raise ValueError("Expected JSON array or object with 'memories' key")

        count = 0
        for m in memories:
            if "id" in m and "state" in m:
                # Full Memory dict — restore directly via upsert
                memory = Memory.from_dict(m)
                self._store.upsert(memory)
            else:
                # Simple dict — use remember()
                self.remember(
                    m["content"],
                    salience=m.get("salience", 0.0),
                    memory_type=MemoryType(m.get("type", "fact")),
                )
            count += 1
        return count

    # --- Rebuild ---

    def snapshot(self) -> str | None:
        """Save a snapshot marker for incremental rebuild (Design §4.1).

        Records the last processed event ID. On next rebuild(incremental=True),
        only events after this point are replayed.
        """
        last_id = self._buffer.last_event_id()
        if last_id:
            self._store.set_metadata("snapshot_event_id", last_id)
            logger.info("snapshot saved at event %s", last_id)
        return last_id

    def rebuild(self, incremental: bool = True) -> int:
        """Rebuild SQLite from JSONL events. Returns count of memories created.

        Args:
            incremental: If True and a snapshot exists, only replay events
                        since the snapshot. If False, full replay from scratch.
        """
        start = time.monotonic()

        # Check for snapshot (incremental rebuild)
        since_event_id: str | None = None
        if incremental:
            since_event_id = self._store.get_metadata("snapshot_event_id")
            if since_event_id:
                logger.info("incremental rebuild from snapshot event %s", since_event_id)

        if not since_event_id:
            self._store.drop_all()
            # Full rebuild — also drop the vector store so write-side dedup
            # during replay (below) compares against a clean slate, otherwise
            # stale vectors from the prior populated state would suppress
            # legitimate replays. Incremental rebuilds keep vectors intact
            # since they share the snapshot horizon.
            try:
                self._vector.drop_all()
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("vector store drop_all failed during rebuild: %s", e)

        count = 0
        past_snapshot = since_event_id is None
        # Replay with write-side cosine dedup plumbed in (Governed Memory paper,
        # arXiv:2603.17787 §4.2). Previously rebuild() bypassed dedup entirely,
        # so two identical remember() events absorbed to 1 row at write-time
        # produced 2 rows after rebuild. Now we route EXPLICIT_REMEMBER through
        # the same dedup-aware upsert path used by remember().
        dedup_threshold = self.config.storage.write_dedup_threshold
        dedup_active = (
            dedup_threshold > 0
            and self._embeddings is not None
            and self._embeddings.dimension > 0
        )
        for event in self._buffer.scan():
            # Skip events before snapshot
            if not past_snapshot:
                if event.id == since_event_id:
                    past_snapshot = True
                continue
            if event.type == EventType.EXPLICIT_REMEMBER:
                memory = Memory.from_event(event)
                if dedup_active:
                    written = self._store.upsert(
                        memory,
                        dedup_threshold=dedup_threshold,
                        vector_store=self._vector,
                        embedding_provider=self._embeddings,
                        acl_filter=None,  # rebuild runs as system, no ACL scope
                    )
                    if not written:
                        continue
                    # Mirror the remember() hot-path: persist the vector so
                    # subsequent replays can see it during their cosine check.
                    try:
                        vec = self._embeddings.embed(event.content)
                        self._vector.upsert(memory.id, vec)
                    except Exception as e:
                        logger.warning("rebuild: failed to embed %s: %s", memory.id, e)
                else:
                    self._store.upsert(memory)
                count += 1
            elif event.type == EventType.FORGET_REQUEST:
                meta = event.metadata
                if meta.get("id"):
                    if meta.get("hard"):
                        self._store.delete(meta["id"])
                    else:
                        self._store.update_state(meta["id"], MemoryState.SUPPRESSED)
            elif event.type == EventType.PIN_ADD:
                pin_id = event.metadata.get("pin_id", "")
                if pin_id:
                    self._store.add_pin(pin_id, event.content)
            elif event.type == EventType.PIN_REMOVE:
                pin_id = event.metadata.get("pin_id", event.content)
                self._store.remove_pin(pin_id)
            elif event.type == EventType.STATE_TRANSITION:
                meta = event.metadata
                mid = meta.get("memory_id", "")
                new_state = meta.get("new_state", "")
                if mid and new_state:
                    try:
                        self._store.update_state(mid, MemoryState(new_state))
                    except (ValueError, KeyError):
                        pass
            elif event.type in (EventType.AFFECT_EMOTION, EventType.AFFECT_MOOD_UPDATE,
                                EventType.AFFECT_TEMPERAMENT_DRIFT, EventType.AFFECT_OVERRIDE):
                # Replay affect events to restore affect_log
                affect_type_map = {
                    EventType.AFFECT_EMOTION: "emotion",
                    EventType.AFFECT_MOOD_UPDATE: "mood",
                    EventType.AFFECT_TEMPERAMENT_DRIFT: "temperament",
                    EventType.AFFECT_OVERRIDE: "temperament",
                }
                self._store.log_affect(
                    affect_type_map[event.type],
                    event.metadata,
                    cause="rebuild",
                )

        elapsed = int((time.monotonic() - start) * 1000)
        self._audit.log(
            "rebuild", self._actor,
            {"memories_created": count, "incremental": since_event_id is not None},
            "success", elapsed,
        )
        # Save snapshot for future incremental rebuilds
        self.snapshot()
        logger.info("rebuilt %d memories from events in %dms", count, elapsed)
        return count

    # --- Internal ---

    def _memory_to_export(self, m: Memory) -> dict:
        """Export a memory as a full dict (uses Memory.to_dict for roundtrip fidelity)."""
        return m.to_dict()

    @staticmethod
    def _parse_window(window: str) -> int:
        """Parse window string like '24h', '7d' into hours."""
        window = window.strip().lower()
        if window.endswith("h"):
            return int(window[:-1])
        elif window.endswith("d"):
            return int(window[:-1]) * 24
        else:
            return int(window)  # assume hours

    def _enforce_retention(self) -> None:
        """Enforce data retention TTLs (Design §5.4, §7)."""
        retention = self.config.retention
        purged = 0

        # Purge faded memories past TTL (done in SQL for efficiency)
        purged += self._store.purge_by_ttl(MemoryState.FADED, retention.faded_ttl_days)
        purged += self._store.purge_by_ttl(MemoryState.SUPPRESSED, retention.suppressed_ttl_days)

        # Truncate old buffer events
        if retention.buffer_ttl_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention.buffer_ttl_days)
            self._buffer.truncate_before(cutoff)

        if purged:
            logger.info("retention: purged %d expired memories", purged)

    def _reconsolidate(self, memory: Memory, context: RecallContext) -> None:
        """Reconsolidation: update memory when recalled in divergent context (Nader et al., 2000).

        When retrieval context is significantly different from encoding context,
        the memory is flagged and its encoding context is blended with current context.
        """
        ec = memory.encoding_context
        if ec.mood_valence is None or context.mood_valence is None:
            return

        # Calculate context divergence
        mood_diff = abs((context.mood_valence or 0) - (ec.mood_valence or 0))
        arousal_diff = abs((context.mood_arousal or 0.5) - (ec.mood_arousal or 0.5))
        divergence = (mood_diff + arousal_diff) / 2.0

        # Task mismatch adds to divergence
        if context.task and ec.task and context.task.lower() != ec.task.lower():
            divergence += 0.2

        # Threshold: only reconsolidate on significant divergence
        if divergence < 0.4:
            return

        old_context = {
            "mood_valence": ec.mood_valence,
            "mood_arousal": ec.mood_arousal,
            "emotions": ec.emotions,
            "task": ec.task,
        }

        # Blend: 70% old + 30% new (memories are resistant to wholesale overwrite)
        ec.mood_valence = ec.mood_valence * 0.7 + (context.mood_valence or 0) * 0.3
        ec.mood_arousal = (ec.mood_arousal or 0.5) * 0.7 + (context.mood_arousal or 0.5) * 0.3
        if context.emotions:
            # Merge emotion sets
            existing = set(ec.emotions)
            existing.update(context.emotions)
            ec.emotions = list(existing)[:5]  # cap at 5
        if context.task and not ec.task:
            ec.task = context.task

        # Record modification in provenance
        memory.provenance.modifications.append(Modification(
            ts=datetime.now(timezone.utc),
            operation="reconsolidation",
            old_value=old_context,
            new_value={
                "mood_valence": ec.mood_valence,
                "mood_arousal": ec.mood_arousal,
                "emotions": ec.emotions,
                "task": ec.task,
            },
            reason=f"context divergence={divergence:.2f}",
        ))

        # Slight confidence decrease (reconsolidation makes memories less certain)
        memory.confidence = max(0.5, memory.confidence - 0.05)

        # Persist updated memory
        self._store.upsert(memory)

        # Emit reconsolidation event
        self._buffer.append(Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.RECONSOLIDATION,
            content=f"reconsolidated {memory.id}: divergence={divergence:.2f}",
            metadata={"memory_id": memory.id, "divergence": round(divergence, 3)},
        ))

        logger.debug("reconsolidated %s (divergence=%.2f)", memory.id, divergence)


class AffectAPI:
    """Proxy for affect operations on Engram (Design §8: mem.affect.*)."""

    def __init__(self, engine: Engram):
        self._engine = engine

    def status(self) -> dict:
        """Get current affect state (mood, emotions, temperament)."""
        return self._engine._affect.get_current_state()

    def mood(self) -> dict:
        """Get current mood (Russell's circumplex)."""
        m = self._engine._affect.mood
        return {"valence": m.valence, "arousal": m.arousal, "label": m.label, "confidence": m.confidence}

    def trigger(self, primary: str, intensity: float = 0.5, trigger: str = "") -> None:
        """Trigger an emotion. Updates mood and encoding context. Persists."""
        emotion = self._engine._affect.trigger_emotion(primary, intensity, trigger)
        store = self._engine._store
        buffer = self._engine._buffer
        affect = self._engine._affect

        store.log_affect("emotion", {
            "primary": primary, "intensity": intensity, "trigger": trigger,
            "compound": emotion.compound,
        }, cause=trigger)
        store.log_affect("mood", {
            "valence": affect.mood.valence, "arousal": affect.mood.arousal,
            "label": affect.mood.label,
        }, cause=f"emotion:{primary}")

        buffer.append(Event(
            id=generate_event_id(), ts=datetime.now(timezone.utc),
            type=EventType.AFFECT_EMOTION,
            content=f"{primary} ({intensity:.2f})",
            metadata={"primary": primary, "intensity": intensity, "trigger": trigger, "compound": emotion.compound},
        ))
        buffer.append(Event(
            id=generate_event_id(), ts=datetime.now(timezone.utc),
            type=EventType.AFFECT_MOOD_UPDATE,
            content=f"mood: {affect.mood.label} (v={affect.mood.valence:.2f}, a={affect.mood.arousal:.2f})",
            metadata={"valence": affect.mood.valence, "arousal": affect.mood.arousal},
        ))

    def set_temperament(self, **kwargs: float) -> None:
        """Set temperament dimensions directly. Persists."""
        old = self._engine._affect.temperament.to_dict()
        for dim, val in kwargs.items():
            if hasattr(self._engine._affect.temperament, dim):
                setattr(self._engine._affect.temperament, dim, max(0.0, min(1.0, val)))
        self._engine._store.log_affect("temperament", self._engine._affect.temperament.to_dict(), cause="manual_set")
        self._engine._buffer.append(Event(
            id=generate_event_id(), ts=datetime.now(timezone.utc),
            type=EventType.AFFECT_OVERRIDE,
            content=f"temperament override: {kwargs}",
            metadata={"old": old, "new": self._engine._affect.temperament.to_dict(), "changes": kwargs},
        ))

    def lock(self, dimension: str) -> None:
        """Lock a temperament dimension from mutation."""
        self._engine._store.log_affect("temperament_lock", {"dimension": dimension, "locked": True}, cause="user_lock")

    def reset_mood(self) -> None:
        """Reset mood to temperament baseline."""
        affect = self._engine._affect
        affect.mood.valence = affect.temperament.baseline_valence
        affect.mood.arousal = affect.temperament.baseline_arousal
        self._engine._store.log_affect("mood", {
            "valence": affect.mood.valence, "arousal": affect.mood.arousal,
            "label": affect.mood.label,
        }, cause="manual_reset")

    def history(self, affect_type: str | None = None, limit: int = 100, days: int | None = None) -> list[dict]:
        """Get affect history. Use days= to filter by time range."""
        if days is not None:
            # Convert days to limit estimate (rough: ~10 entries/day)
            limit = max(limit, days * 10)
        entries = self._engine._store.get_affect_history(affect_type, limit)
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            entries = [e for e in entries if e.get("ts", "") >= cutoff]
        return entries

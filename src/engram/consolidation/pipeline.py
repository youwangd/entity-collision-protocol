"""Consolidation pipeline — the brain's 'sleep cycle'.

Processes buffered events into structured memories via a pluggable
pipeline of 12 stages matching DESIGN.md §4.3.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from engram.core.types import (
    ConsolidationReport,
    Event,
    EventType,
    Memory,
    MemoryType,
    MemoryState,
    Modification,
    generate_event_id,
    EMOTIONAL_DECAY_RATE,
    SOMATIC_MARKED_DECAY_RATE,
)
from engram.core.config import Config
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore
from engram.audit.log import AuditLog
from engram.providers.llm import LLMProvider, NoLLMProvider
from engram.affect.engine import AffectEngine, PLUTCHIK_COMPOUNDS

logger = logging.getLogger(__name__)


@dataclass
class StageContext:
    """Shared context passed between consolidation stages."""

    events: list[Event] = field(default_factory=list)
    memories_created: list[Memory] = field(default_factory=list)
    memories_updated: list[Memory] = field(default_factory=list)
    memories_decayed: list[str] = field(default_factory=list)
    memories_suppressed: list[str] = field(default_factory=list)
    schemas_created: list[Memory] = field(default_factory=list)
    interference_actions: list[dict] = field(default_factory=list)
    emotions_triggered: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    consolidation_id: str = ""  # ID of this consolidation cycle
    # Identity of the consolidator/agent emitting lifecycle events from
    # this cycle. Threaded into `make_lifecycle_event(emitter_id=...)`
    # so that under `deprecate_quorum_k>1` the §6.16 reducer can count
    # distinct emitters before firing DEPRECATE. Default "" preserves
    # legacy wire format (no metadata key written).
    consolidator_id: str = ""
    # Stores for stages to use
    buffer: JSONLBufferStore | None = None
    store: SQLiteMemoryStore | None = None
    llm: LLMProvider | None = None
    config: Config | None = None
    affect: AffectEngine | None = None


class ConsolidationStage:
    """Base class for consolidation pipeline stages."""

    name: str = "unnamed"

    def run(self, ctx: StageContext) -> StageContext:
        """Process and return the context. Override in subclasses."""
        return ctx


# --- Stage 1: Replay ---

class EventIngestion(ConsolidationStage):
    """Stage 1: Load unprocessed events from the buffer."""

    name = "replay"

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.buffer is None:
            return ctx
        since = None
        if self.window_hours > 0:
            since = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)

        # Get last consolidated event ID to skip already-processed events
        last_consolidated = None
        if ctx.store:
            last_consolidated = ctx.store.get_metadata("last_consolidated_event_id")

        events = list(ctx.buffer.scan(since=since))
        # Filter to only capture events (remembers are already in SQLite)
        ctx.events = [e for e in events if e.type == EventType.EVENT_CAPTURE]

        # Skip events already consolidated (Q12)
        if last_consolidated:
            ctx.events = [e for e in ctx.events if e.id > last_consolidated]

        ctx.stats["events_loaded"] = len(ctx.events)
        logger.info("replay: ingested %d events (window=%dh)", len(ctx.events), self.window_hours)
        return ctx


# --- Stage 2: Appraisal (Scherer CPM) ---

class AppraisalScoring(ConsolidationStage):
    """Stage 2: Score memories using Scherer's Component Process Model.

    Without LLM: heuristic scoring based on content signals.
    With LLM: EmotionPrompt-framed appraisal for better extraction.
    """

    name = "appraisal"

    URGENCY_WORDS = {"urgent", "critical", "asap", "important", "deadline", "emergency", "now", "immediately", "breaking"}
    NOVELTY_WORDS = {"new", "first", "never", "surprising", "unexpected", "discovered", "breakthrough", "novel", "unique"}
    GOAL_WORDS = {"goal", "objective", "milestone", "achieved", "completed", "shipped", "launched", "failed", "blocked", "error", "bug", "fixed"}
    NEGATIVE_WORDS = {"failed", "error", "bug", "broken", "crash", "wrong", "mistake", "problem", "issue"}

    EMOTION_PROMPT_SYSTEM = """You are a memory appraisal agent. Your job is deeply important — you determine which memories matter and which fade.

Score each memory on three dimensions (1.0 = baseline, up to 2.0 = exceptional):
- relevance: How relevant is this to the agent's current goals and tasks?
- novelty: How new or surprising is this information? Does it contradict existing knowledge?
- goal_conduciveness: How much does this help or hinder active goals? Errors and corrections score high.

This is the difference between remembering what matters and forgetting everything. Take it seriously."""

    def run(self, ctx: StageContext) -> StageContext:
        all_memories = ctx.memories_created + ctx.memories_updated
        cap: float | None = None
        if ctx.config and ctx.config.consolidation:
            cap = ctx.config.consolidation.appraisal_salience_cap
        for memory in all_memories:
            if ctx.llm and not isinstance(ctx.llm, NoLLMProvider):
                self._llm_appraise(ctx.llm, memory)
            else:
                self._heuristic_appraise(memory)

            # §94c-appraisal-bound — clamp post-appraisal salience.
            if cap is not None and memory.salience > cap:
                memory.salience = float(cap)

            # Emit consolidation_appraisal event (Design §3.1)
            if ctx.buffer:
                ctx.buffer.append(Event(
                    id=generate_event_id(),
                    ts=datetime.now(timezone.utc),
                    type=EventType.CONSOLIDATION_APPRAISAL,
                    content=f"appraised {memory.id}: rel={memory.appraisal.relevance:.2f} nov={memory.appraisal.novelty:.2f} gc={memory.appraisal.goal_conduciveness:.2f}",
                    metadata={"memory_id": memory.id, "salience": round(memory.salience, 3)},
                ))

        ctx.stats["appraised"] = len(all_memories)
        return ctx

    def _heuristic_appraise(self, memory: Memory) -> None:
        words = set(memory.content.lower().split())
        urgency_hits = len(words & self.URGENCY_WORDS)
        novelty_hits = len(words & self.NOVELTY_WORDS)
        goal_hits = len(words & self.GOAL_WORDS)

        memory.appraisal.relevance = min(1.0 + urgency_hits * 0.3, 2.0)
        memory.appraisal.novelty = min(1.0 + novelty_hits * 0.4, 2.0)
        memory.appraisal.goal_conduciveness = min(1.0 + goal_hits * 0.3, 2.0)

        raw_salience = (
            0.5
            * memory.appraisal.relevance
            * memory.appraisal.novelty
            * memory.appraisal.goal_conduciveness
        )
        memory.salience = min(max(raw_salience, 0.0), 1.0)

    def _llm_appraise(self, llm: LLMProvider, memory: Memory) -> None:
        """Use EmotionPrompt-framed LLM appraisal (Li et al., 2023)."""
        result = llm.extract_json(
            f"""Appraise this memory. Return JSON:
{{"relevance": 1.0-2.0, "novelty": 1.0-2.0, "goal_conduciveness": 1.0-2.0}}

Memory: {memory.content}""",
            system=self.EMOTION_PROMPT_SYSTEM,
        )
        if result:
            memory.appraisal.relevance = min(max(result.get("relevance", 1.0), 1.0), 2.0)
            memory.appraisal.novelty = min(max(result.get("novelty", 1.0), 1.0), 2.0)
            memory.appraisal.goal_conduciveness = min(max(result.get("goal_conduciveness", 1.0), 1.0), 2.0)
            raw = 0.5 * memory.appraisal.relevance * memory.appraisal.novelty * memory.appraisal.goal_conduciveness
            memory.salience = min(max(raw, 0.0), 1.0)
        else:
            self._heuristic_appraise(memory)


# --- Stage 3: Emotion Tagging (Plutchik) ---

class EmotionTagging(ConsolidationStage):
    """Stage 3: Map appraisal scores to Plutchik emotions. Update mood."""

    name = "emotion_tagging"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.affect is None:
            return ctx

        tagged = 0
        for memory in ctx.memories_created:
            emotion, intensity = self._classify_emotion(memory)
            if emotion:
                memory.emotion.primary = emotion
                memory.emotion.intensity = intensity

                # Trigger the emotion in the affect engine
                em = ctx.affect.trigger_emotion(emotion, intensity, trigger=memory.content[:80])
                if em.compound:
                    memory.emotion.compound = em.compound

                # Check for compound emotions from co-occurring active emotions
                if not memory.emotion.compound and len(ctx.affect.active_emotions) >= 2:
                    active_primaries = [e.primary for e in ctx.affect.active_emotions if e.intensity > 0.2]
                    for i, p1 in enumerate(active_primaries):
                        for p2 in active_primaries[i+1:]:
                            compound = PLUTCHIK_COMPOUNDS.get((p1, p2)) or PLUTCHIK_COMPOUNDS.get((p2, p1))
                            if compound:
                                memory.emotion.compound = compound
                                break

                # Override decay rate for emotional memories
                if intensity > 0.5:
                    memory.decay_rate = EMOTIONAL_DECAY_RATE

                ctx.emotions_triggered.append({
                    "primary": emotion, "intensity": intensity, "memory_id": memory.id,
                })
                tagged += 1

        ctx.stats["emotions_tagged"] = tagged
        if tagged:
            logger.info("emotion_tagging: tagged %d memories", tagged)
        return ctx

    def _classify_emotion(self, memory: Memory) -> tuple[str, float]:
        """Map appraisal dimensions to primary emotion + intensity."""
        content_lower = memory.content.lower()
        appr = memory.appraisal

        # High goal conduciveness → Joy (success/completion)
        if appr.goal_conduciveness >= 1.3:
            if any(w in content_lower for w in ("completed", "success", "shipped", "achieved", "fixed")):
                return "joy", min(appr.goal_conduciveness / 2.0, 1.0)

        # High relevance + high goal conduciveness → Anticipation
        if appr.relevance > 1.3 and appr.goal_conduciveness > 1.0:
            return "anticipation", 0.5

        # High novelty → Surprise
        if appr.novelty > 1.5:
            return "surprise", min(appr.novelty / 2.0, 1.0)

        # Failure/error words → Anger or Sadness
        if any(w in content_lower for w in ("failed", "error", "broken", "crash", "wrong")):
            if appr.relevance > 1.3:  # high relevance failure = anger
                return "anger", min(appr.relevance / 2.0, 1.0)
            return "sadness", 0.5

        # Blocked/stuck → Fear
        if any(w in content_lower for w in ("blocked", "stuck", "deadline", "urgent")):
            return "fear", 0.5

        # User correction → Trust (acceptance)
        if any(w in content_lower for w in ("correction", "actually", "instead", "prefer")):
            return "trust", 0.4

        return "", 0.0


# --- Stage 4: Deduplication + Extraction ---

class Deduplication(ConsolidationStage):
    """Stage 4a: Deduplicate near-identical events."""

    name = "deduplication"

    def run(self, ctx: StageContext) -> StageContext:
        if len(ctx.events) <= 1:
            return ctx
        unique: list[Event] = []
        seen: set[str] = set()
        for event in ctx.events:
            normalized = event.content.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(event)
        removed = len(ctx.events) - len(unique)
        ctx.events = unique
        ctx.stats["dedup_removed"] = removed
        if removed:
            logger.info("dedup: removed %d duplicates", removed)
        return ctx


class EpisodeExtraction(ConsolidationStage):
    """Stage 4b: Convert events into episode memories with L0 summaries."""

    name = "extraction"

    def run(self, ctx: StageContext) -> StageContext:
        for event in ctx.events:
            memory = Memory.from_event(event, memory_type=MemoryType.EPISODE)
            memory.provenance.created_by = ctx.consolidation_id  # track which cycle created this

            # Generate L0 summary (Design §4.6: "During consolidation, every memory gets a one-line L0 summary")
            if ctx.llm and not isinstance(ctx.llm, NoLLMProvider):
                summary = ctx.llm.complete(
                    f"Summarize in ≤20 tokens, one line: {event.content}",
                    system="You are a memory consolidation agent. Be extremely concise.",
                    max_tokens=50,
                )
                if summary:
                    memory.summary = summary.strip()[:100]
            else:
                # Heuristic L0: first sentence or first 80 chars
                first_sentence = event.content.split(".")[0]
                memory.summary = first_sentence[:80]

            ctx.memories_created.append(memory)

        ctx.stats["episodes_created"] = len(ctx.memories_created)
        logger.info("extraction: created %d episodes", len(ctx.memories_created))
        return ctx


class FactExtraction(ConsolidationStage):
    """Stage 4c: Extract facts from episodes using LLM. Degrades gracefully."""

    name = "fact_extraction"

    # Dual-extraction prompt schema (Governed Memory paper, arXiv:2603.17787):
    # In a single LLM call, extract both fact text AND typed structured properties
    # plus per-fact extraction confidence. Backward-compatible with legacy schema.
    _PROMPT_TEMPLATE = """Extract factual statements from this episode. Return STRICT JSON:
{{"facts": [
  {{"text": "fact text", "confidence": 0.95,
    "properties": [
      {{"key": "deal_value", "value": "$450K", "type": "number", "confidence": 0.9}}
    ]}}
]}}

Rules:
- "confidence" is your confidence the fact is correct (0.0–1.0); 1.0 = direct/explicit.
- "properties" capture typed structured attributes; type ∈ {{text, number, date, entity}}.
- Empty properties list is fine.
- Return only clear, concrete facts. If none, return {{"facts": []}}.

Episode: {content}"""

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.llm is None or isinstance(ctx.llm, NoLLMProvider):
            ctx.stats["facts_extracted"] = 0
            return ctx

        facts_created = 0
        for memory in list(ctx.memories_created):
            if memory.type != MemoryType.EPISODE:
                continue
            result = ctx.llm.extract_json(
                self._PROMPT_TEMPLATE.format(content=memory.content),
                system="Extract only clear, concrete facts with structured properties. Return empty list if none.",
            )
            for parsed in self._parse_facts(result):
                fact_text = parsed["text"]
                if not fact_text or len(fact_text) <= 5:
                    continue
                fact_event = Event(
                    id=generate_event_id(),
                    ts=datetime.now(timezone.utc),
                    type=EventType.CONSOLIDATION_EXTRACT,
                    content=fact_text,
                    metadata={"source_memory": memory.id},
                )
                fact_memory = Memory.from_event(fact_event, memory_type=MemoryType.FACT)
                fact_memory.source_events = [memory.id]
                # Inherit agent_id from the source episode. Without this,
                # facts extracted from Alice's episode default to
                # agent_id='' and are globally readable — Bob's recall
                # would surface Alice's distilled facts. The consolidation
                # event has no actor context (synthetic), so the source
                # memory's owner is the only correct attribution.
                fact_memory.agent_id = memory.agent_id
                fact_memory.summary = fact_text[:80]
                fact_memory.provenance.created_by = ctx.consolidation_id
                # Clamp extraction_confidence to [0, 1]
                conf = parsed.get("confidence", 1.0)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 1.0
                fact_memory.extraction_confidence = max(0.0, min(1.0, conf))
                # Stash typed properties to be persisted alongside the memory
                # (transient attribute consumed by MemoryPersistence stage)
                props = parsed.get("properties") or []
                if props:
                    fact_memory._pending_properties = self._normalize_properties(props)
                ctx.memories_created.append(fact_memory)
                facts_created += 1

        ctx.stats["facts_extracted"] = facts_created
        if facts_created:
            logger.info("fact_extraction: extracted %d facts", facts_created)
        return ctx

    @staticmethod
    def _parse_facts(result: Any) -> list[dict]:
        """Parse the LLM response with tolerance for both new and legacy schemas.

        Accepts:
          - {"facts": [{"text": "...", "confidence": 0.9, "properties": [...]}]}  (new)
          - {"facts": ["fact 1", "fact 2"]}                                         (legacy)
          - ["fact 1", "fact 2"]                                                    (very legacy)
        Returns a list of normalized {"text", "confidence", "properties"} dicts.
        """
        if not result:
            return []
        if isinstance(result, list):
            facts_iter = result
        elif isinstance(result, dict):
            facts_iter = result.get("facts", []) or []
        else:
            return []
        normalized: list[dict] = []
        for item in facts_iter:
            if isinstance(item, str):
                normalized.append({"text": item, "confidence": 1.0, "properties": []})
            elif isinstance(item, dict):
                text = item.get("text") or item.get("fact") or ""
                if not isinstance(text, str):
                    continue
                normalized.append({
                    "text": text,
                    "confidence": item.get("confidence", 1.0),
                    "properties": item.get("properties") or [],
                })
        return normalized

    @staticmethod
    def _normalize_properties(props: list) -> list[dict]:
        """Coerce LLM-supplied properties into the upsert_properties() shape."""
        out: list[dict] = []
        for p in props:
            if not isinstance(p, dict):
                continue
            key = p.get("key")
            value = p.get("value")
            if not key or value is None:
                continue
            try:
                conf = float(p.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            out.append({
                "key": str(key),
                "value": str(value),
                "type": str(p.get("type", "text")),
                "confidence": max(0.0, min(1.0, conf)),
            })
        return out


# --- Stage 5: Interference Detection ---

class InterferenceDetection(ConsolidationStage):
    """Stage 5: Check new memories against existing for contradictions.

    Supersede, update, or flag conflicts. Without LLM: exact content overlap detection.
    With LLM: semantic contradiction analysis.
    """

    name = "interference"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None:
            return ctx

        # §D3 add-only mode: skip interference entirely (no supersede,
        # no conflict-flag mutation). Mem0-v3 ablation primitive.
        cfg_cons = getattr(ctx.config, "consolidation", None) if ctx.config else None
        if cfg_cons is not None and getattr(cfg_cons, "add_only", False):
            ctx.stats["interference_actions"] = 0
            ctx.stats["add_only_skipped"] = True
            return ctx

        actions = 0
        # §D3-collateral-(b) gating: read entity-aware flag once per run.
        entity_aware = bool(cfg_cons and getattr(cfg_cons, "interference_entity_aware", False))
        entity_min = float(cfg_cons.interference_entity_overlap_min) if (cfg_cons and entity_aware) else 0.0
        for new_mem in list(ctx.memories_created):
            if new_mem.type not in (MemoryType.FACT, MemoryType.SKILL):
                continue  # only facts and skills can interfere

            # Find potentially conflicting existing memories
            existing = ctx.store.search_text(new_mem.content, limit=5, states=["active", "fading"])
            for scored in existing:
                old_mem = scored.memory
                if old_mem.id == new_mem.id:
                    continue

                # Check for interference
                action = self._detect_interference(
                    new_mem, old_mem, ctx.llm,
                    entity_aware=entity_aware,
                    entity_min=entity_min,
                )
                if action == "supersede":
                    ctx.store.update_state(old_mem.id, MemoryState.FADED)
                    ctx.store.add_relation(new_mem.id, old_mem.id, "supersedes")
                    new_mem.provenance.modifications.append(Modification(
                        ts=datetime.now(timezone.utc),
                        operation="supersede",
                        old_value={"id": old_mem.id},
                        reason="newer information supersedes",
                    ))
                    ctx.interference_actions.append({
                        "action": "supersede", "new": new_mem.id, "old": old_mem.id,
                    })
                    if ctx.buffer:
                        ctx.buffer.append(Event(
                            id=generate_event_id(), ts=datetime.now(timezone.utc),
                            type=EventType.CONSOLIDATION_INTERFERENCE,
                            content=f"supersede: {new_mem.id} supersedes {old_mem.id}",
                            metadata={"action": "supersede", "new": new_mem.id, "old": old_mem.id},
                        ))
                    actions += 1
                elif action == "conflict":
                    # Flag both as lower confidence
                    new_mem.confidence = min(new_mem.confidence, 0.7)
                    old_mem.confidence = min(old_mem.confidence, 0.7)
                    ctx.store.upsert(old_mem)
                    ctx.store.add_relation(new_mem.id, old_mem.id, "contradicts")
                    ctx.interference_actions.append({
                        "action": "conflict", "new": new_mem.id, "old": old_mem.id,
                    })
                    if ctx.buffer:
                        ctx.buffer.append(Event(
                            id=generate_event_id(), ts=datetime.now(timezone.utc),
                            type=EventType.CONSOLIDATION_INTERFERENCE,
                            content=f"conflict: {new_mem.id} contradicts {old_mem.id}",
                            metadata={"action": "conflict", "new": new_mem.id, "old": old_mem.id},
                        ))
                    actions += 1

        ctx.stats["interference_actions"] = actions
        if actions:
            logger.info("interference: %d actions taken", actions)
        return ctx

    # §D3-collateral-(b) — minimal English stop-word list for the
    # entity-aware overlap gate. Intentionally small and dependency-free
    # so the production codepath stays NER-free; we only need to strip
    # template boilerplate ("user", "now", "for", "of", etc.) so that
    # cross-slot template overlap is no longer enough to fire FADE.
    # Tokens of length < 3 are also dropped (catches "a", "i", "is").
    _STOP_TOKENS = frozenset({
        "the", "and", "for", "with", "from", "that", "this", "these", "those",
        "into", "onto", "upon", "about", "over", "under", "out", "off", "down",
        "now", "then", "than", "but", "not", "yes", "you", "your", "yours",
        "his", "her", "its", "our", "their", "they", "them", "him", "she", "he",
        "are", "was", "were", "been", "being", "have", "has", "had", "having",
        "will", "would", "could", "should", "shall", "may", "might", "must",
        "can", "do", "does", "did", "doing", "done",
        "what", "when", "where", "who", "why", "how", "which",
        "user", "users", "uses", "use", "using", "used",
        "currently", "daily", "work", "version", "owns", "owned", "runs",
        "lives", "lived", "likes", "prefers", "preferred", "prefer",
    })

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        """Tokenize to non-stop, length≥3 lowercase tokens for entity-aware match."""
        out: set[str] = set()
        for raw in text.lower().split():
            tok = "".join(c for c in raw if c.isalnum())
            if len(tok) < 3:
                continue
            if tok in InterferenceDetection._STOP_TOKENS:
                continue
            out.add(tok)
        return out

    def _detect_interference(
        self,
        new: Memory,
        old: Memory,
        llm: LLMProvider | None,
        *,
        entity_aware: bool = False,
        entity_min: float = 0.0,
    ) -> str:
        """Detect type of interference between two memories."""
        # Same type required for interference
        if new.type != old.type:
            return ""

        # Simple heuristic: high content overlap but different key terms
        new_words = set(new.content.lower().split())
        old_words = set(old.content.lower().split())
        overlap = len(new_words & old_words)
        total = max(len(new_words | old_words), 1)
        similarity = overlap / total

        if similarity > 0.6:
            # High overlap — likely an update/supersede
            if new.created_at > old.created_at:
                # §D3-collateral-(b): require non-trivial entity-token overlap
                # to avoid cross-slot template-overlap false positives.
                if entity_aware:
                    new_ent = self._content_tokens(new.content)
                    old_ent = self._content_tokens(old.content)
                    union = new_ent | old_ent
                    if not union:
                        return ""
                    ent_jacc = len(new_ent & old_ent) / len(union)
                    if ent_jacc < entity_min:
                        return ""
                return "supersede"
            return ""

        if similarity > 0.3:
            # Moderate overlap — might be contradictory
            contradiction_signals = {"not", "no", "instead", "actually", "but", "however", "wrong", "changed"}
            if (new_words | old_words) & contradiction_signals:
                return "conflict"

        return ""


# --- Stage 6: Schema Update ---

def _count_schema_contradictions(
    supporting_facts: list[str],
    ctx: "StageContext",
) -> int:
    """Count contradictions for a schema in the current consolidation window.

    Heuristic (paper §3.7 / TODO-RESEARCH §B, closed 2026-05-24):
    Stage 5 (InterferenceDetection) has
    already classified memory pairs as ``"conflict"`` when their
    contents disagree. We use those upstream actions as ground-truth
    contradiction signals and attribute one contradiction to the
    schema for each conflict whose ``new`` memory's content overlaps
    a supporting fact (Jaccard ≥ 0.3 on lowercased word sets — same
    threshold Stage 5 itself uses for moderate overlap).

    Pure-ish: depends on ctx state, but no I/O, no clocks, no RNG.
    Returns 0 when there's no store/no actions/no overlap, which is
    the conservative default that keeps existing tests green.
    """
    actions = [
        a for a in ctx.interference_actions
        if a.get("action") == "conflict"
    ]
    if not actions or ctx.store is None or not supporting_facts:
        return 0

    fact_word_sets: list[set[str]] = [
        set(f.lower().split()) for f in supporting_facts if f
    ]
    if not fact_word_sets:
        return 0

    contradictions = 0
    for act in actions:
        new_id = act.get("new")
        if not new_id:
            continue
        mem = ctx.store.get(new_id)
        if mem is None:
            continue
        new_words = set(mem.content.lower().split())
        if not new_words:
            continue
        for fact_words in fact_word_sets:
            if not fact_words:
                continue
            overlap = len(new_words & fact_words)
            total = len(new_words | fact_words) or 1
            if overlap / total >= 0.3:
                contradictions += 1
                break  # one contradiction per conflict, not per fact
    return contradictions


def _decide_with_share(cur_state, ev_window, thresholds, ctx, siblings=()):
    """Apply Personize §8 prior-sharing if config requests it.

    Default share=0.0 ⇒ byte-identical to bare ``decide()`` (G1
    regression-safety). With ``share>0`` and a non-empty ``siblings``
    tuple the call delegates to ``decide_with_family`` which credits
    ``share * Σsiblings`` to owner counts before policy. ``siblings``
    is computed once-per-window by ``_build_schema_family_siblings``;
    when that prepass yields nothing for an owner we pass ``()`` and
    decide_with_family is identity (E2/G2).
    """
    from engram.consolidation.schema_decision import decide
    cfg = ctx.config.consolidation if ctx.config else None
    share = float(cfg.schema_family_share) if cfg else 0.0
    if share == 0.0:
        return decide(cur_state, ev_window, thresholds)
    from engram.consolidation.schema_family_decision import decide_with_family
    return decide_with_family(
        cur_state, ev_window, siblings=siblings, thresholds=thresholds, share=share,
    )


def _build_schema_family_siblings(
    schemas_data,
    window_id,
    ctx,
):
    """Pre-pass: cluster the LLM-returned patterns by supporting-facts
    fingerprint and return a per-summary map of sibling EvidenceWindows.

    Keyed on ``pattern[:80]`` (stable across re-emission and new
    creation within a single window). Returns ``({}, {})`` when the
    feature is off (share=0.0) or window_id is None — both branches
    short-circuit the caller back to bare ``decide()``.

    Returns:
      (siblings_by_summary, evidence_by_summary)
    """
    cfg = ctx.config.consolidation if ctx.config else None
    share = float(cfg.schema_family_share) if cfg else 0.0
    if share == 0.0 or window_id is None:
        return {}, {}
    from engram.consolidation.schema_decision import EvidenceWindow
    from engram.consolidation.schema_family import cluster
    from engram.consolidation.schema_family_evidence import all_owner_siblings
    from engram.consolidation.schema_fingerprint import fingerprints
    tau = float(cfg.schema_family_tau) if cfg else 0.5
    facts_by_summary: dict[str, list[str]] = {}
    ev_by_summary: dict[str, EvidenceWindow] = {}
    for sd in schemas_data:
        pattern = sd.get("pattern", "") or ""
        if not pattern or len(pattern) <= 10:
            continue
        summary = pattern[:80]
        if summary in ev_by_summary:
            continue  # first-wins, mirrors existing_by_summary semantics
        sf = sd.get("facts") or []
        contradictions = _count_schema_contradictions(sf, ctx)
        facts_by_summary[summary] = list(sf)
        ev_by_summary[summary] = EvidenceWindow(
            window_id=window_id,
            supports=len(sf),
            contradictions=contradictions,
        )
    if not ev_by_summary:
        return {}, {}
    fps = fingerprints(facts_by_summary)
    clusters = cluster(fps, tau=tau)
    # §69 deployment-rule gate: if cluster contamination exceeds the
    # configured cap, fall back to bare decide() for this window. The
    # rate is non-monotone in the §69 generative `c` knob (high c
    # fragments clusters → singletons drop pair-weight) but it correctly
    # detects single-link transitivity stress on tight regimes. See
    # SCALE_REPORT §72; the operational form lives here.
    cmax = cfg.schema_family_contamination_max if cfg else None
    fmax = cfg.schema_family_fragmentation_max if cfg else None
    if cmax is not None or fmax is not None:
        from engram.consolidation.schema_family_contamination import (
            contamination_rate,
            fragmentation_rate,
        )
        gated = False
        if cmax is not None:
            rate = contamination_rate(fps, clusters, tau)
            ctx.stats["schema_family_contamination_rate"] = rate
            if rate > float(cmax):
                gated = True
        if fmax is not None:
            frag = fragmentation_rate(fps, clusters)
            ctx.stats["schema_family_fragmentation_rate"] = frag
            if frag > float(fmax):
                gated = True
                ctx.stats["schema_family_fragmentation_gated"] = True
        if gated:
            ctx.stats["schema_family_share_gated"] = True
            return {}, {}
    siblings_by_summary = all_owner_siblings(clusters, ev_by_summary)
    return siblings_by_summary, ev_by_summary


class SchemaUpdate(ConsolidationStage):
    """Stage 6: Update or create schemas from repeated patterns.

    Without LLM: simple pattern detection via content similarity.
    With LLM: semantic pattern analysis.
    """

    name = "schema_update"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None:
            return ctx

        # Group memories by content similarity to detect patterns
        facts = [m for m in ctx.store.search_by_type(MemoryType.FACT, limit=200)]
        if len(facts) < 3:
            ctx.stats["schemas_created"] = 0
            return ctx

        # Simple pattern detection: find groups of related facts
        schemas_created = 0
        schemas_bumped = 0
        schemas_recovered = 0
        # Threshold knobs (paper §4.6 churn budget). Default to historical
        # Thresholds() values when unset.
        from engram.consolidation.schema_decision import Thresholds as _Thresholds
        _cc = ctx.config.consolidation if ctx.config is not None else None
        _thr = _Thresholds(
            promote=getattr(_cc, "schema_promote_threshold", 3) if _cc else 3,
            deprecate=getattr(_cc, "schema_deprecate_threshold", 2) if _cc else 2,
            recover=getattr(_cc, "schema_recover_threshold", 3) if _cc else 3,
        )
        # Identity index: schema summary (== pattern[:80]) → existing
        # schema memory. Lets us recognise a re-emitted pattern as the
        # same schema and emit BUMP_VERSION instead of a duplicate
        # CREATE. Summary is stable: it's a deterministic prefix of
        # pattern, and SchemaUpdate is the only writer of SCHEMA-typed
        # memories.
        existing_by_summary: dict[str, Memory] = {}
        for s in ctx.store.search_by_type(MemoryType.SCHEMA, limit=500):
            if s.summary:
                existing_by_summary.setdefault(s.summary, s)
        if ctx.llm and not isinstance(ctx.llm, NoLLMProvider):
            # LLM-powered schema extraction
            fact_texts = [f.content for f in facts[:50]]  # limit to prevent huge prompts
            result = ctx.llm.extract_json(
                f"""Analyze these facts and identify recurring patterns or themes.
Return JSON: {{"schemas": [{{"pattern": "description", "facts": ["fact1", "fact2"]}}]}}

Facts:
{chr(10).join(f'- {t}' for t in fact_texts)}""",
                system="You identify patterns in memories. Only report patterns with 3+ supporting facts.",
            )
        elif (
            ctx.config is not None
            and ctx.config.consolidation is not None
            and getattr(ctx.config.consolidation, "schema_synthesis_enabled", False)
        ):
            # §93 deterministic non-LLM schema synthesis. Same downstream
            # contract as the LLM path: produce a `result["schemas"]`
            # list of {pattern, facts}, then fall through into the
            # shared CREATE/BUMP/RECOVER machinery below.
            from engram.consolidation.schema_synthesis import synthesize_schemas
            cc = ctx.config.consolidation
            fact_texts = [f.content for f in facts[:50]]
            synth = synthesize_schemas(
                fact_texts,
                tau=cc.schema_synthesis_tau,
                min_supports=cc.schema_synthesis_min_supports,
            )
            result = {"schemas": synth}
        else:
            result = None

        if result is not None:
            # Lazy-load the lifecycle snapshot once per stage run so we
            # can decide RECOVER for re-emitted patterns whose schema is
            # currently DEPRECATED. Pure read; no mutation.
            _lifecycle_snap: dict | None = None
            def _snap():
                nonlocal _lifecycle_snap
                if _lifecycle_snap is None:
                    if ctx.buffer is None:
                        _lifecycle_snap = {}
                    else:
                        from engram.consolidation.lifecycle_projection import (
                            snapshot_from_buffer,
                        )
                        _lifecycle_snap = snapshot_from_buffer(
                            ctx.buffer, strict=False,
                        )
                return _lifecycle_snap

            schemas_data_list = result.get("schemas", [])
            # Personize §8 prepass: build siblings map keyed on
            # pattern-summary so cluster-mates discovered in this same
            # LLM batch can prior-share evidence. share=0.0 → empty.
            _siblings_by_summary, _ = _build_schema_family_siblings(
                schemas_data_list, ctx.consolidation_id or None, ctx,
            )

            for schema_data in schemas_data_list:
                pattern = schema_data.get("pattern", "")
                if pattern and len(pattern) > 10:
                    summary = pattern[:80]
                    existing = existing_by_summary.get(summary)
                    if existing is not None:
                        # Re-emission path. The schema already exists.
                        # Two sub-cases (refined vs idempotent), but
                        # BOTH need a RECOVER check first: if the
                        # current state is DEPRECATED and this fresh
                        # window provides enough supports, fire
                        # RECOVER before any BUMP_VERSION/no-op.
                        if ctx.buffer is not None:
                            from engram.consolidation.lifecycle_projection import (
                                make_lifecycle_event,
                            )
                            from engram.consolidation.schema_decision import (
                                EvidenceWindow,
                            )
                            from engram.consolidation.schema_lifecycle import (
                                EventKind,
                                SchemaStatus,
                            )
                            window_id = ctx.consolidation_id or None
                            cur_state = _snap().get(existing.id)
                            if (
                                cur_state is not None
                                and cur_state.status == SchemaStatus.DEPRECATED
                                and window_id is not None
                            ):
                                supporting_facts = schema_data.get("facts") or []
                                supports = len(supporting_facts)
                                contradictions = _count_schema_contradictions(
                                    supporting_facts, ctx,
                                )
                                ev_window = EvidenceWindow(
                                    window_id=window_id,
                                    supports=supports,
                                    contradictions=contradictions,
                                )
                                kind = _decide_with_share(
                                    cur_state, ev_window, _thr, ctx,
                                    siblings=_siblings_by_summary.get(summary, ()),
                                )
                                if kind == EventKind.RECOVER:
                                    ctx.buffer.append(make_lifecycle_event(
                                        schema_id=existing.id,
                                        kind=EventKind.RECOVER,
                                        window_id=window_id,
                                        content=pattern[:200],
                                        emitter_id=ctx.consolidator_id or None,
                                    ))
                                    schemas_recovered += 1
                        if existing.content != pattern:
                            # Refined content → BUMP_VERSION on top of
                            # whatever status RECOVER (or its absence)
                            # left us in.
                            if ctx.buffer is not None:
                                from engram.consolidation.lifecycle_projection import (
                                    make_lifecycle_event,
                                )
                                from engram.consolidation.schema_lifecycle import EventKind
                                ctx.buffer.append(make_lifecycle_event(
                                    schema_id=existing.id,
                                    kind=EventKind.BUMP_VERSION,
                                    window_id=ctx.consolidation_id or None,
                                    content=pattern[:200],
                                    emitter_id=ctx.consolidator_id or None,
                                ))
                                schemas_bumped += 1
                        # Same content + non-DEPRECATED: silent no-op.
                        continue
                    schema_event = Event(
                        id=generate_event_id(),
                        ts=datetime.now(timezone.utc),
                        type=EventType.CONSOLIDATION_SCHEMA_UPDATE,
                        content=pattern,
                    )
                    schema_mem = Memory.from_event(schema_event, memory_type=MemoryType.SCHEMA)
                    schema_mem.summary = summary
                    ctx.schemas_created.append(schema_mem)
                    ctx.memories_created.append(schema_mem)
                    schemas_created += 1
                    existing_by_summary[summary] = schema_mem
                    # Emit a lifecycle CREATE event so the buffer→reducer
                    # projection can track this schema's status from
                    # birth. Window id is the consolidation cycle id.
                    if ctx.buffer is not None:
                        from engram.consolidation.lifecycle_projection import (
                            make_lifecycle_event,
                        )
                        from engram.consolidation.schema_decision import (
                            EvidenceWindow,
                        )
                        from engram.consolidation.schema_lifecycle import (
                            EventKind,
                            SchemaState,
                            SchemaStatus,
                        )
                        window_id = ctx.consolidation_id or None
                        ctx.buffer.append(make_lifecycle_event(
                            schema_id=schema_mem.id,
                            kind=EventKind.CREATE,
                            window_id=window_id,
                            content=pattern[:200],
                            emitter_id=ctx.consolidator_id or None,
                        ))
                        # Evidence = LLM-returned supporting facts for
                        # this pattern in THIS window. Contradictions
                        # come from Stage 5 InterferenceDetection: any
                        # memory flagged 'conflict' in this window
                        # whose content textually overlaps a supporting
                        # fact contributes one contradiction. This is
                        # the simplest defensible heuristic: it reuses
                        # an upstream signal that's already explicitly
                        # "this fact disagrees with an existing one"
                        # without needing typed-property machinery yet.
                        supporting_facts = schema_data.get("facts") or []
                        supports = len(supporting_facts)
                        contradictions = _count_schema_contradictions(
                            supporting_facts, ctx,
                        )
                        if window_id is not None:
                            ev_window = EvidenceWindow(
                                window_id=window_id,
                                supports=supports,
                                contradictions=contradictions,
                            )
                            inferred_state = SchemaState(
                                schema_id=schema_mem.id,
                                status=SchemaStatus.INFERRED,
                                version=1,
                                last_window_id=window_id,
                            )
                            kind = _decide_with_share(
                                inferred_state, ev_window, _thr, ctx,
                                siblings=_siblings_by_summary.get(summary, ()),
                            )
                            if kind is not None:
                                ctx.buffer.append(make_lifecycle_event(
                                    schema_id=schema_mem.id,
                                    kind=kind,
                                    window_id=window_id,
                                    content=pattern[:200],
                                    emitter_id=ctx.consolidator_id or None,
                                ))

        ctx.stats["schemas_created"] = schemas_created
        ctx.stats["schemas_bumped"] = schemas_bumped
        ctx.stats["schemas_recovered"] = schemas_recovered
        if schemas_created or schemas_bumped or schemas_recovered:
            logger.info(
                "schema_update: created %d schemas, bumped %d, recovered %d",
                schemas_created, schemas_bumped, schemas_recovered,
            )
        return ctx


# --- Stage 7: Somatic Marking (Damasio) ---

class SomaticMarking(ConsolidationStage):
    """Stage 7: Attach behavioral biases to strong-outcome episodes.

    Based on Damasio's Somatic Marker Hypothesis: strong emotional experiences
    create 'gut feelings' that shortcut future decisions.
    """

    name = "somatic_marking"

    def run(self, ctx: StageContext) -> StageContext:
        marked = 0
        for memory in ctx.memories_created + ctx.memories_updated:
            marker = self._generate_marker(memory)
            if marker:
                memory.somatic.valence = marker["valence"]
                memory.somatic.bias = marker["bias"]
                memory.somatic.trigger = marker["trigger"]
                memory.decay_rate = SOMATIC_MARKED_DECAY_RATE  # somatic-marked memories persist longer
                marked += 1

        ctx.stats["somatic_marked"] = marked
        if marked:
            logger.info("somatic_marking: marked %d memories", marked)
        return ctx

    def _generate_marker(self, memory: Memory) -> dict | None:
        """Generate somatic marker from memory content and emotion."""
        content_lower = memory.content.lower()

        # Strong positive outcomes → positive somatic marker
        positive_triggers = {
            "completed": ("recommend this approach", 0.6),
            "success": ("this works well", 0.7),
            "fixed": ("this solution works", 0.5),
            "shipped": ("this deployment approach is reliable", 0.6),
            "achieved": ("this strategy is effective", 0.7),
        }
        for keyword, (bias, valence) in positive_triggers.items():
            if keyword in content_lower:
                return {"valence": valence, "bias": bias, "trigger": keyword}

        # Strong negative outcomes → negative somatic marker
        negative_triggers = {
            "failed": ("avoid this approach", -0.6),
            "crash": ("this is dangerous", -0.8),
            "broken": ("check carefully before trying", -0.5),
            "error": ("verify before proceeding", -0.4),
            "wrong": ("double-check this assumption", -0.5),
            "lost data": ("always backup first", -0.9),
        }
        for keyword, (bias, valence) in negative_triggers.items():
            if keyword in content_lower:
                return {"valence": valence, "bias": bias, "trigger": keyword}

        # High-emotion memories without explicit markers
        if memory.emotion.intensity > 0.6:
            if memory.emotion.primary in ("anger", "fear", "sadness", "disgust"):
                return {"valence": -0.4, "bias": "approach with caution", "trigger": memory.emotion.primary}
            elif memory.emotion.primary in ("joy", "trust", "anticipation"):
                return {"valence": 0.4, "bias": "positive experience", "trigger": memory.emotion.primary}

        return None


# --- Stage 8: Decay (Ebbinghaus) ---

class DecayApplication(ConsolidationStage):
    """Stage 8: Apply Ebbinghaus decay to existing memories.

    salience *= exp(-decay_rate * days_since_access)
    Memories that drop below fade_threshold transition state.
    """

    name = "decay"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None or ctx.config is None:
            return ctx

        fade_threshold = ctx.config.forgetting.fade_threshold
        now = datetime.now(timezone.utc)
        decayed = 0
        state_changes = 0

        for memory in ctx.store.all_active():
            last = memory.last_accessed or memory.created_at
            hours = max((now - last).total_seconds() / 3600, 0)

            if hours < 0.1:
                continue  # too recent

            old_salience = memory.salience
            memory.salience *= math.exp(-memory.decay_rate * hours / 24)
            memory.salience = max(memory.salience, 0.0)

            # State transitions
            new_state = None
            if memory.salience < fade_threshold:
                if memory.state == MemoryState.ACTIVE:
                    new_state = MemoryState.FADING
                elif memory.state == MemoryState.FADING:
                    new_state = MemoryState.FADED

            if new_state:
                memory.state = new_state
                state_changes += 1
                # Log state transition event (Design: "Transitions are logged as events")
                if ctx.buffer:
                    ctx.buffer.append(Event(
                        id=generate_event_id(),
                        ts=datetime.now(timezone.utc),
                        type=EventType.STATE_TRANSITION,
                        content=f"{memory.id}: {memory.state.value} → {new_state.value}",
                        metadata={"memory_id": memory.id, "old_state": memory.state.value, "new_state": new_state.value},
                    ))

            if abs(old_salience - memory.salience) > 0.001 or new_state:
                ctx.store.upsert(memory)
                ctx.memories_decayed.append(memory.id)
                decayed += 1
                # Emit consolidation_decay event (Design §3.1)
                if ctx.buffer and abs(old_salience - memory.salience) > 0.01:
                    ctx.buffer.append(Event(
                        id=generate_event_id(),
                        ts=datetime.now(timezone.utc),
                        type=EventType.CONSOLIDATION_DECAY,
                        content=f"decay {memory.id}: {old_salience:.3f} → {memory.salience:.3f}",
                        metadata={"memory_id": memory.id, "old": round(old_salience, 3), "new": round(memory.salience, 3)},
                    ))

        ctx.stats["decayed"] = decayed
        ctx.stats["state_changes"] = state_changes
        if decayed:
            logger.info("decay: %d memories decayed (%d state changes)", decayed, state_changes)
        return ctx


# --- Stage 9: Motivated Suppression ---

class MotivatedSuppression(ConsolidationStage):
    """Stage 9: Auto-suppress memories that are negative + low utility + rarely accessed.

    Also: superseded facts older than 30 days.
    Based on motivated forgetting research.
    """

    name = "suppression"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None or ctx.config is None:
            return ctx
        if not ctx.config.forgetting.auto_suppress:
            return ctx

        suppress_threshold = ctx.config.forgetting.suppress_threshold
        now = datetime.now(timezone.utc)
        suppressed = 0

        for memory in ctx.store.all_active():
            should_suppress = False
            reason = ""

            # Negative somatic marker + low utility + rarely accessed
            if (memory.somatic.valence < suppress_threshold
                    and memory.access_count < 3
                    and memory.salience < 0.2):
                should_suppress = True
                reason = f"negative somatic ({memory.somatic.valence:.2f}) + low utility"

            # Faded memories that haven't been accessed in 30+ days
            if memory.state == MemoryState.FADED:
                last = memory.last_accessed or memory.created_at
                days_inactive = (now - last).total_seconds() / 86400
                if days_inactive > 30:
                    should_suppress = True
                    reason = f"faded + {days_inactive:.0f} days inactive"

            if should_suppress:
                ctx.store.update_state(memory.id, MemoryState.SUPPRESSED)
                ctx.memories_suppressed.append(memory.id)
                suppressed += 1

                if ctx.buffer:
                    ctx.buffer.append(Event(
                        id=generate_event_id(),
                        ts=datetime.now(timezone.utc),
                        type=EventType.CONSOLIDATION_SUPPRESS,
                        content=f"auto-suppressed {memory.id}: {reason}",
                        metadata={"memory_id": memory.id, "reason": reason},
                    ))

        ctx.stats["suppressed"] = suppressed
        if suppressed:
            logger.info("suppression: auto-suppressed %d memories", suppressed)
        return ctx


# --- Stage 10: Temperament Drift (Cloninger) ---

class TemperamentDrift(ConsolidationStage):
    """Stage 10: Review accumulated outcomes → nudge Cloninger dimensions.

    Self-mutation: experience slowly reshapes personality over weeks/months.
    """

    name = "temperament_drift"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.affect is None:
            return ctx

        # Count outcome types from this consolidation cycle
        positive = sum(1 for e in ctx.emotions_triggered if e["primary"] in ("joy", "trust", "anticipation"))
        negative = sum(1 for e in ctx.emotions_triggered if e["primary"] in ("anger", "sadness", "fear", "disgust"))
        novel = sum(1 for m in ctx.memories_created if m.appraisal.novelty > 1.3)
        total = max(len(ctx.events), 1)

        # Feed back into affect engine
        ctx.affect.consolidation_feedback(
            positive_outcomes=positive,
            negative_outcomes=negative,
            novel_discoveries=novel,
            total_events=total,
        )

        # Persist temperament
        if ctx.store:
            ctx.store.log_affect("temperament", ctx.affect.temperament.to_dict(), cause="consolidation_drift")

        # Emit event to JSONL (Design §3.1: affect_temperament_drift)
        if ctx.buffer:
            ctx.buffer.append(Event(
                id=generate_event_id(),
                ts=datetime.now(timezone.utc),
                type=EventType.AFFECT_TEMPERAMENT_DRIFT,
                content=f"temperament drift: +{positive}/−{negative} outcomes, {novel} novel",
                metadata=ctx.affect.temperament.to_dict(),
            ))

        ctx.stats["temperament_drifted"] = True
        ctx.stats["positive_outcomes"] = positive
        ctx.stats["negative_outcomes"] = negative
        logger.info("temperament_drift: +%d/-%d outcomes, %d novel", positive, negative, novel)
        return ctx


# --- Stage 11: Mood Update ---

class MoodUpdate(ConsolidationStage):
    """Stage 11: Update rolling mood from recent emotion history."""

    name = "mood_update"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.affect is None:
            return ctx

        # Decay mood toward temperament baseline
        ctx.affect.mood.decay_toward_baseline(
            ctx.affect.temperament.baseline_valence,
            ctx.affect.temperament.baseline_arousal,
            hours=1.0,
        )

        # Persist mood
        if ctx.store:
            ctx.store.log_affect("mood", {
                "valence": ctx.affect.mood.valence,
                "arousal": ctx.affect.mood.arousal,
                "label": ctx.affect.mood.label,
            }, cause="consolidation_update")

        # Emit event to JSONL (Design §3.1: affect_mood_update)
        if ctx.buffer:
            ctx.buffer.append(Event(
                id=generate_event_id(),
                ts=datetime.now(timezone.utc),
                type=EventType.AFFECT_MOOD_UPDATE,
                content=f"mood: {ctx.affect.mood.label} (v={ctx.affect.mood.valence:.2f}, a={ctx.affect.mood.arousal:.2f})",
                metadata={"valence": ctx.affect.mood.valence, "arousal": ctx.affect.mood.arousal},
            ))

        ctx.stats["mood_label"] = ctx.affect.mood.label
        logger.info("mood_update: %s (v=%.2f, a=%.2f)",
                     ctx.affect.mood.label, ctx.affect.mood.valence, ctx.affect.mood.arousal)
        return ctx


# --- Stage 12: Mechanical Merge (Governed Memory paper, threshold 0.95) ---

class MechanicalMerge(ConsolidationStage):
    """Stage 12: Merge near-duplicate memories mechanically (no LLM).
    
    Finds memory pairs with cosine similarity > 0.95 and merges them:
    keeps the higher-salience memory, suppresses the other.
    """

    name = "mechanical_merge"
    DEFAULT_MERGE_THRESHOLD = 0.95

    def __init__(self, vector_store=None, embedding_provider=None, threshold: float | None = None):
        self._vector = vector_store
        self._embeddings = embedding_provider
        self.threshold = threshold if threshold is not None else self.DEFAULT_MERGE_THRESHOLD

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None or self._vector is None or self._embeddings is None:
            return ctx
        if self._embeddings.dimension == 0:
            return ctx

        merged = 0
        active = ctx.store.all_active()
        seen_ids = set()

        for memory in active:
            if memory.id in seen_ids:
                continue
            try:
                vec = self._embeddings.embed(memory.content)
                if not vec:
                    continue
                # Over-fetch and filter by ACL scope. Cross-agent merges are an
                # ACL side-channel: Bob's near-duplicate of Alice's content
                # would otherwise let Alice's salience suppress Bob's memory
                # (existence-leak + DoS). System memories (agent_id='', e.g.
                # SCHEMA prototypes) remain mergeable globally — that is the
                # intended behaviour for shared system patterns.
                results = self._vector.search(vec, limit=20)
                owner = memory.agent_id or ""
                for r in results:
                    if r.memory_id == memory.id or r.memory_id in seen_ids:
                        continue
                    if r.score > self.threshold:
                        other = ctx.store.get(r.memory_id)
                        if other is None or other.state.value not in ("active", "fading"):
                            continue
                        # ACL scope: only merge if both memories have the
                        # same agent_id (incl. both '' for system memories).
                        other_owner = other.agent_id or ""
                        if owner != other_owner:
                            continue
                        # Keep higher salience, suppress lower
                        if memory.salience >= other.salience:
                            ctx.store.update_state(other.id, MemoryState.SUPPRESSED)
                            seen_ids.add(other.id)
                        else:
                            ctx.store.update_state(memory.id, MemoryState.SUPPRESSED)
                            seen_ids.add(memory.id)
                            break  # This memory is suppressed, move on
                        merged += 1
            except Exception as e:
                logger.warning("mechanical merge failed for %s: %s", memory.id, e)

        if merged > 0:
            logger.info("mechanical merge: %d near-duplicates suppressed", merged)
        ctx.stats["mechanical_merged"] = merged
        return ctx


# --- Stage 13: Persistence + Report ---

class MemoryPersistence(ConsolidationStage):
    """Stage 13: Write new/updated memories to SQLite + vectors."""

    name = "persistence"

    def __init__(self, vector_store=None, embedding_provider=None):
        self._vector = vector_store
        self._embeddings = embedding_provider

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.store is None:
            return ctx

        persisted = 0
        for memory in ctx.memories_created:
            ctx.store.upsert(memory)
            # Persist any typed properties stashed by FactExtraction (Governed Memory dual extraction)
            pending = getattr(memory, "_pending_properties", None)
            if pending:
                try:
                    ctx.store.upsert_properties(memory.id, pending)
                except Exception as e:
                    logger.warning("failed to persist properties for %s: %s", memory.id, e)
            # Generate vector embeddings for consolidation-created memories (Gap #20)
            if self._embeddings and self._vector and self._embeddings.dimension > 0:
                try:
                    vec = self._embeddings.embed(memory.content)
                    self._vector.upsert(memory.id, vec)
                except Exception as e:
                    logger.warning("failed to embed memory %s: %s", memory.id, e)
            persisted += 1

        for memory in ctx.memories_updated:
            ctx.store.upsert(memory)
            persisted += 1

        ctx.stats["persisted"] = persisted
        if persisted:
            logger.info("persistence: wrote %d memories", persisted)
        return ctx


# --- Pipeline Orchestrator ---

class ConsolidationPipeline:
    """Orchestrates the 12-stage consolidation pipeline (Design §4.3)."""

    def __init__(
        self,
        buffer: JSONLBufferStore,
        store: SQLiteMemoryStore,
        audit: AuditLog,
        config: Config,
        llm: LLMProvider | None = None,
        affect: AffectEngine | None = None,
        vector_store=None,
        embedding_provider=None,
    ):
        self.buffer = buffer
        self.store = store
        self.audit = audit
        self.config = config
        self.llm = llm or NoLLMProvider()
        self.affect = affect

        window = 24
        if config.consolidation:
            window = config.consolidation.window_hours

        # Full 12-stage pipeline — corrected order per DESIGN.md §4.3
        # Appraisal + EmotionTagging MUST come AFTER extraction (they score memories_created)
        self.stages: list[ConsolidationStage] = [
            EventIngestion(window_hours=window),       # Stage 1: Replay
            Deduplication(),                           # Stage 2: Dedup
            EpisodeExtraction(),                       # Stage 3: Episode extraction + L0
            FactExtraction(),                          # Stage 4: Fact extraction
            AppraisalScoring(),                        # Stage 5: Appraisal (Scherer CPM)
            EmotionTagging(),                          # Stage 6: Emotion (Plutchik)
            InterferenceDetection(),                   # Stage 7: Interference
            SchemaUpdate(),                            # Stage 8: Auto-schemas
            SomaticMarking(),                          # Stage 9: Somatic markers (Damasio)
            DecayApplication(),                        # Stage 10: Ebbinghaus decay
            MotivatedSuppression(),                    # Stage 11: Auto-suppression
            TemperamentDrift(),                        # Stage 12a: Cloninger drift
            MoodUpdate(),                              # Stage 12b: Russell mood
            MechanicalMerge(vector_store, embedding_provider,
                            threshold=config.storage.merge_threshold if config.storage else None),  # Stage 12c
            MemoryPersistence(vector_store, embedding_provider),  # Stage 13: Persist + report
        ]

        # Filter stages if custom pipeline configured (Design §4.3)
        if config.consolidation and config.consolidation.stages:
            allowed = set(config.consolidation.stages)
            allowed.update({"replay", "persistence"})  # mandatory stages
            self.stages = [s for s in self.stages if s.name in allowed]

    def run(self, actor: str = "consolidation") -> ConsolidationReport:
        """Run the full consolidation pipeline."""
        start = time.monotonic()

        start_event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.CONSOLIDATION_START,
            content="consolidation cycle started",
        )
        self.buffer.append(start_event)

        ctx = StageContext(
            buffer=self.buffer,
            store=self.store,
            llm=self.llm,
            config=self.config,
            affect=self.affect,
            consolidation_id=start_event.id,
            consolidator_id=getattr(self.config, "consolidator_id", "") or "",
        )

        for stage in self.stages:
            stage_start = time.monotonic()
            try:
                ctx = stage.run(ctx)
                stage_ms = int((time.monotonic() - stage_start) * 1000)
                ctx.stats[f"{stage.name}_ms"] = stage_ms
            except Exception as e:
                ctx.errors.append(f"{stage.name}: {e}")
                logger.error("stage %s failed: %s", stage.name, e, exc_info=True)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        end_event = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.CONSOLIDATION_COMPLETE,
            content=f"consolidation complete: {len(ctx.memories_created)} created, {len(ctx.memories_decayed)} decayed, {len(ctx.memories_suppressed)} suppressed",
            metadata=ctx.stats,
        )
        self.buffer.append(end_event)

        # Get store stats for forgetting budget
        store_stats = {}
        if ctx.store:
            store_stats = ctx.store.stats()

        report = ConsolidationReport(
            consolidation_id=start_event.id,
            events_processed=ctx.stats.get("events_loaded", 0),
            memories_created=len(ctx.memories_created),
            facts_extracted=ctx.stats.get("facts_extracted", 0),
            state_transitions={
                "decayed": len(ctx.memories_decayed),
                "suppressed": len(ctx.memories_suppressed),
                "schemas": len(ctx.schemas_created),
                "interference": len(ctx.interference_actions),
                "total_active": store_stats.get("by_state", {}).get("active", 0),
                "total_faded": store_stats.get("by_state", {}).get("faded", 0),
                "total_suppressed": store_stats.get("by_state", {}).get("suppressed", 0),
            },
            duration_ms=elapsed_ms,
            errors=ctx.errors,
        )

        self.audit.log(
            "consolidation", actor,
            {
                "events": report.events_processed,
                "created": report.memories_created,
                "facts": report.facts_extracted,
                "decayed": len(ctx.memories_decayed),
                "suppressed": len(ctx.memories_suppressed),
                "schemas": len(ctx.schemas_created),
                "interference": len(ctx.interference_actions),
                "emotions": len(ctx.emotions_triggered),
                "errors": len(report.errors),
                "mood": ctx.stats.get("mood_label", ""),
                **{k: v for k, v in ctx.stats.items() if k.endswith("_ms")},
            },
            "success" if not report.errors else "partial",
            elapsed_ms,
        )

        # Track last consolidated event to avoid re-processing (Q12)
        if ctx.events and self.store:
            self.store.set_metadata("last_consolidated_event_id", ctx.events[-1].id)

        return report

"""Core data types for Engram."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All event types in the system."""

    # Memory operations
    EXPLICIT_REMEMBER = "explicit_remember"
    EVENT_CAPTURE = "event_capture"
    FORGET_REQUEST = "forget_request"
    PIN_ADD = "pin_add"
    PIN_REMOVE = "pin_remove"

    # Retrieval
    RECALL_REQUEST = "recall_request"
    RECALL_HIT = "recall_hit"
    RECONSOLIDATION = "reconsolidation"

    # Consolidation
    CONSOLIDATION_START = "consolidation_start"
    CONSOLIDATION_COMPLETE = "consolidation_complete"
    CONSOLIDATION_APPRAISAL = "consolidation_appraisal"
    CONSOLIDATION_EXTRACT = "consolidation_extract"
    CONSOLIDATION_INTERFERENCE = "consolidation_interference"
    CONSOLIDATION_DECAY = "consolidation_decay"
    CONSOLIDATION_SUPPRESS = "consolidation_suppress"
    CONSOLIDATION_SCHEMA_UPDATE = "consolidation_schema_update"
    CONSOLIDATION_SCHEMA_LIFECYCLE = "consolidation_schema_lifecycle"
    STATE_TRANSITION = "state_transition"

    # Affect
    AFFECT_EMOTION = "affect_emotion"
    AFFECT_MOOD_UPDATE = "affect_mood_update"
    AFFECT_TEMPERAMENT_DRIFT = "affect_temperament_drift"
    AFFECT_OVERRIDE = "affect_override"

    # System
    CONFIG_CHANGE = "config_change"


class MemoryType(str, Enum):
    """Types of long-term memory."""

    EPISODE = "episode"
    FACT = "fact"
    SKILL = "skill"
    SCHEMA = "schema"


# Decay rates per memory type (from neuroscience research)
# These determine forgetting speed: lower = slower decay = longer memory
DECAY_RATES: dict[MemoryType, float] = {
    MemoryType.EPISODE: 0.1,    # ~7 day half-life — "what happened Monday" fades fast
    MemoryType.FACT: 0.005,     # ~140 day half-life — "user prefers postgres" persists
    MemoryType.SKILL: 0.001,    # ~700 day half-life — procedural knowledge is nearly permanent
    MemoryType.SCHEMA: 0.001,   # ~700 day half-life — patterns are nearly permanent
}

# Emotional episodes get a special rate (overridden during consolidation)
EMOTIONAL_DECAY_RATE = 0.01     # ~70 day half-life
SOMATIC_MARKED_DECAY_RATE = 0.002  # ~350 day half-life


class MemoryState(str, Enum):
    """Lifecycle states for memories."""

    ACTIVE = "active"
    FADING = "fading"
    FADED = "faded"
    SUPPRESSED = "suppressed"


class DataClassification(str, Enum):
    """Data sensitivity levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


def generate_event_id() -> str:
    """Generate a unique event ID: evt-{timestamp_ms}-{random_hex}."""
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().hex[:8]
    return f"evt-{ts}-{rand}"


def generate_memory_id(memory_type: MemoryType) -> str:
    """Generate a unique memory ID: mem-{type_prefix}-{random_hex}."""
    prefix = memory_type.value[:2]
    rand = uuid.uuid4().hex[:12]
    return f"mem-{prefix}-{rand}"


def content_hash(content: str) -> str:
    """SHA-256 hash of content (for audit log — avoids logging PII)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Event:
    """An immutable event in the event store (source of truth)."""

    id: str
    ts: datetime
    type: EventType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    salience_hint: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat(),
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "salience_hint": self.salience_hint,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            type=EventType(d["type"]),
            content=d["content"],
            metadata=d.get("metadata", {}),
            salience_hint=d.get("salience_hint", 0.0),
            context=d.get("context", {}),
        )


@dataclass
class Appraisal:
    """Multi-dimensional appraisal scores (Scherer CPM)."""

    relevance: float = 1.0  # 1.0–2.0
    novelty: float = 1.0  # 1.0–2.0
    goal_conduciveness: float = 1.0  # 0.5–2.0


@dataclass
class SomaticMarker:
    """Behavioral bias attached to a memory (Damasio)."""

    valence: float = 0.0  # -1.0 (avoid) to +1.0 (approach)
    bias: str = ""  # human-readable behavioral bias
    trigger: str = ""  # what triggers this marker


@dataclass
class EmotionTag:
    """Emotion associated with a memory (Plutchik)."""

    primary: str = ""  # joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    intensity: float = 0.0  # 0.0–1.0 (mild/moderate/intense)
    compound: str = ""  # e.g., "curiosity", "pride", "frustration"


@dataclass
class EncodingContext:
    """Context at the time the memory was formed (Tulving)."""

    mood_valence: float | None = None
    mood_arousal: float | None = None
    emotions: list[str] = field(default_factory=list)
    task: str = ""


@dataclass
class Modification:
    """A single modification in a memory's provenance chain."""

    ts: datetime
    operation: str  # "reconsolidation", "decay", "suppress", "merge", "update"
    consolidation_id: str = ""
    old_value: dict[str, Any] = field(default_factory=dict)
    new_value: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "operation": self.operation,
            "consolidation_id": self.consolidation_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Modification:
        return cls(
            ts=datetime.fromisoformat(d["ts"]),
            operation=d["operation"],
            consolidation_id=d.get("consolidation_id", ""),
            old_value=d.get("old_value", {}),
            new_value=d.get("new_value", {}),
            reason=d.get("reason", ""),
        )


@dataclass
class Provenance:
    """Full lineage of a memory."""

    source_events: list[str] = field(default_factory=list)
    created_by: str = ""  # consolidation cycle ID
    modifications: list[Modification] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_events": self.source_events,
            "created_by": self.created_by,
            "modifications": [m.to_dict() for m in self.modifications],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Provenance:
        return cls(
            source_events=d.get("source_events", []),
            created_by=d.get("created_by", ""),
            modifications=[Modification.from_dict(m) for m in d.get("modifications", [])],
        )


@dataclass
class Memory:
    """A consolidated memory in the long-term store."""

    id: str
    type: MemoryType
    state: MemoryState
    content: str
    summary: str  # L0 one-line abstract
    salience: float
    confidence: float
    decay_rate: float
    created_at: datetime
    last_accessed: datetime | None = None
    access_count: int = 0
    agent_id: str = ""  # owner agent for ACL (Design §5.3)

    # Appraisal (Scherer CPM)
    appraisal: Appraisal = field(default_factory=Appraisal)

    # Somatic marker (Damasio)
    somatic: SomaticMarker = field(default_factory=SomaticMarker)

    # Emotion (Plutchik)
    emotion: EmotionTag = field(default_factory=EmotionTag)

    # Encoding context (Tulving)
    encoding_context: EncodingContext = field(default_factory=EncodingContext)

    # Classification
    classification: DataClassification = DataClassification.PUBLIC

    # Lineage
    source_events: list[str] = field(default_factory=list)
    schema_id: str = ""
    provenance: Provenance = field(default_factory=Provenance)

    # Extraction confidence (Governed Memory paper, arXiv:2603.17787)
    # How confident is the extractor that this fact is correct? 1.0 = direct/explicit, <1.0 = inferred.
    extraction_confidence: float = 1.0

    @classmethod
    def from_event(cls, event: Event, memory_type: MemoryType = MemoryType.FACT) -> Memory:
        """Create a basic memory from an explicit_remember event."""
        now = datetime.now(timezone.utc)
        mid = generate_memory_id(memory_type)
        return cls(
            id=mid,
            type=memory_type,
            state=MemoryState.ACTIVE,
            content=event.content,
            summary=event.content[:100],  # basic L0 until consolidation generates better
            salience=max(0.1, min(1.0, 0.5 + event.salience_hint)),
            confidence=1.0,
            decay_rate=DECAY_RATES.get(memory_type, 0.1),
            created_at=now,
            last_accessed=now,
            access_count=0,
            encoding_context=EncodingContext(
                mood_valence=event.context.get("mood_valence"),
                mood_arousal=event.context.get("mood_arousal"),
                emotions=event.context.get("active_emotions", []),
                task=event.context.get("active_task", ""),
            ),
            source_events=[event.id],
            provenance=Provenance(source_events=[event.id], created_by="direct"),
        )

    def to_dict(self) -> dict:
        """Full serialization for export/import roundtrip."""
        return {
            "id": self.id,
            "type": self.type.value,
            "state": self.state.value,
            "content": self.content,
            "summary": self.summary,
            "salience": self.salience,
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "agent_id": self.agent_id,
            "appraisal": {"relevance": self.appraisal.relevance, "novelty": self.appraisal.novelty, "goal_conduciveness": self.appraisal.goal_conduciveness},
            "somatic": {"valence": self.somatic.valence, "bias": self.somatic.bias, "trigger": self.somatic.trigger},
            "emotion": {"primary": self.emotion.primary, "intensity": self.emotion.intensity, "compound": self.emotion.compound},
            "encoding_context": {"mood_valence": self.encoding_context.mood_valence, "mood_arousal": self.encoding_context.mood_arousal, "emotions": self.encoding_context.emotions, "task": self.encoding_context.task},
            "classification": self.classification.value,
            "source_events": self.source_events,
            "schema_id": self.schema_id,
            "provenance": self.provenance.to_dict(),
            "extraction_confidence": self.extraction_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memory:
        """Deserialize from dict (import/export roundtrip)."""
        return cls(
            id=d["id"],
            type=MemoryType(d["type"]),
            state=MemoryState(d["state"]),
            content=d["content"],
            summary=d.get("summary", ""),
            salience=d.get("salience", 0.5),
            confidence=d.get("confidence", 1.0),
            decay_rate=d.get("decay_rate", 0.1),
            created_at=datetime.fromisoformat(d["created_at"]),
            last_accessed=datetime.fromisoformat(d["last_accessed"]) if d.get("last_accessed") else None,
            access_count=d.get("access_count", 0),
            agent_id=d.get("agent_id", ""),
            appraisal=Appraisal(**d["appraisal"]) if "appraisal" in d else Appraisal(),
            somatic=SomaticMarker(**d["somatic"]) if "somatic" in d else SomaticMarker(),
            emotion=EmotionTag(**d["emotion"]) if "emotion" in d else EmotionTag(),
            encoding_context=EncodingContext(**d["encoding_context"]) if "encoding_context" in d else EncodingContext(),
            classification=DataClassification(d["classification"]) if "classification" in d else DataClassification.PUBLIC,
            source_events=d.get("source_events", []),
            schema_id=d.get("schema_id", ""),
            provenance=Provenance.from_dict(d["provenance"]) if "provenance" in d else Provenance(),
            extraction_confidence=d.get("extraction_confidence", 1.0),
        )


@dataclass
class ScoredMemory:
    """A memory with retrieval scores."""

    memory: Memory
    score: float
    sources: dict[str, float] = field(default_factory=dict)  # which retrieval paths contributed


@dataclass
class RecallContext:
    """Context for encoding specificity matching at recall time.

    Can be created with individual fields or from a mood dict:
        RecallContext(mood=mem.affect.mood(), task="deploy")
    """

    mood_valence: float | None = None
    mood_arousal: float | None = None
    task: str | None = None
    emotions: list[str] = field(default_factory=list)
    mood: dict | None = field(default=None, repr=False)  # convenience: unpacks valence/arousal

    def __post_init__(self):
        # Allow passing mood dict for convenience (Design §8)
        if self.mood and isinstance(self.mood, dict):
            if self.mood_valence is None:
                self.mood_valence = self.mood.get("valence")
            if self.mood_arousal is None:
                self.mood_arousal = self.mood.get("arousal")
            self.mood = None  # clear after unpacking

    @classmethod
    def from_mood(cls, mood: dict, task: str | None = None, emotions: list[str] | None = None) -> RecallContext:
        """Create from a mood dict (from mem.affect.mood())."""
        return cls(
            mood_valence=mood.get("valence"),
            mood_arousal=mood.get("arousal"),
            task=task,
            emotions=emotions or [],
        )


@dataclass
class ConsolidationReport:
    """Report from a consolidation cycle."""

    consolidation_id: str = ""
    events_processed: int = 0
    memories_created: int = 0
    facts_extracted: int = 0
    state_transitions: dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0
    errors: list[str] = field(default_factory=list)

# Engram — Neuroscience-Inspired Memory for AI Agents

> The first agent memory system built from how human brains actually work.

An engram is the hypothesized physical trace of a memory in the brain — the pattern of neural connections that stores a specific memory. Fitting for a memory system built from neuroscience principles.

## 1. Vision

Current agent memory is flat, dumb, and static. Vectors go in, vectors come out. No prioritization, no forgetting, no consolidation, no learning. It's where computer memory was before virtual memory was invented.

Human memory is none of those things. It's hierarchical, adaptive, importance-weighted, and self-organizing. It forgets on purpose, consolidates during downtime, updates memories when recalled, and builds schemas automatically from experience.

**Engram** brings these properties to AI agents — as a pluggable, secure, zero-infra library.

### Design Principles

| Principle | Meaning |
|---|---|
| **Neuroscience-grounded** | Every mechanism maps to a researched brain process |
| **Pluggable** | Every component has an interface; swap, extend, or disable independently |
| **Event-sourced** | JSONL event log is the source of truth; everything else is rebuildable |
| **Secure by default** | Memory firewall, PII detection, access control, audit from day one |
| **Zero infra** | Works with `pip install engram` on a Raspberry Pi. External services optional. |
| **Graceful degradation** | If embedding/LLM/vector service fails, falls back to BM25 + metadata |
| **Observable** | Every operation is audited, traceable, and measurable |

### Neuroscience Mapping

| Human Brain | Engram Equivalent | Key Reference |
|---|---|---|
| Working memory (prefrontal cortex) | Active Context — L0/L1/L2 tiered loading + position-aware injection | Baddeley & Hitch (1974) |
| Hippocampus (fast episodic capture) | Event Buffer — raw JSONL, zero processing delay | McClelland et al. (1995) |
| Neocortex (slow structured knowledge) | Long-Term Store — SQLite projection with schemas | McClelland et al. (1995) |
| Sleep consolidation | Consolidation Pipeline — periodic LLM-powered distillation | Sharp wave-ripples (2024) |
| Amygdala (importance tagging) | Appraisal System — multi-dimensional scoring | Scherer (2001) CPM |
| Somatic markers (gut feelings) | Behavioral Bias — outcome-tagged decision shortcuts | Damasio (1994) SMH |
| Forgetting curve | 5-type Forgetting — decay, interference, suppression, etc. | Ebbinghaus (1885) |
| Memory reconsolidation | Mutable Retrieval — memories update on recall | Nader et al. (2000) |
| Schema formation | Auto-Schemas — emergent patterns from repeated experience | Tse et al. (2007) |
| Temperament / mood / emotion | Affect Engine — self-mutating personality from experience | Cloninger (1993), Russell (1980), Plutchik (1980) |
| Encoding specificity | Context-Matched Retrieval — mood/task boost at recall | Tulving & Thomson (1973) |

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Agent / Client                             │
│                  (Claude, Cline, sage, any MCP client)               │
└────────────┬────────────────────┬──────────────────┬─────────────────┘
             │ write              │ read             │ inspect
             ▼                   ▼                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Engram API                                 │
│     remember()  recall()  consolidate()  forget()  affect.*          │
│                                                                      │
│  ┌─────────────┐    ┌──── MEMORY FIREWALL ────┐                     │
│  │ Audit Log   │◄───│ injection · PII · ACL   │                     │
│  │ (audit.jsonl)│    │ rate-limit · classify   │                     │
│  └─────────────┘    └─────────┬───────────────┘                     │
│                               │ validated writes                     │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │                     EVENT STORE (source of truth)              │  │
│  │                     Append-only JSONL                          │  │
│  │  Every operation → immutable event → rebuildable projection    │  │
│  └────────────┬───────────────────────────────────────────────────┘  │
│               │                                                      │
│    ┌──────────▼──────────┐        ┌──────────────────────────────┐  │
│    │  CONSOLIDATION      │        │     RETRIEVAL ENGINE         │  │
│    │  PIPELINE           │        │                              │  │
│    │                     │        │  ┌──────┐ ┌──────┐ ┌──────┐ │  │
│    │  replay → appraise  │───────►│  │ BM25 │ │Vector│ │ Meta │ │  │
│    │  → extract → decay  │        │  │(FTS5)│ │(tier)│ │filter│ │  │
│    │  → forget → affect  │        │  └──┬───┘ └──┬───┘ └──┬───┘ │  │
│    │                     │        │     └────┬───┘────┬───┘     │  │
│    └──────────┬──────────┘        │     ┌────▼────────▼───┐     │  │
│               │                   │     │  RRF + Salience  │     │  │
│               ▼                   │     │  + Recency       │     │  │
│    ┌──────────────────────┐       │     │  + Context boost │     │  │
│    │  SQLite PROJECTION   │       │     └──────────────────┘     │  │
│    │  (rebuildable)       │◄──────│                              │  │
│    │  episodes · facts    │       └──────────────────────────────┘  │
│    │  schemas · relations │                                         │
│    │  affect · provenance │        ┌──────────────────────────────┐ │
│    └──────────────────────┘        │     AFFECT ENGINE            │ │
│                                    │  temperament → mood → emotion│ │
│                                    │  (biases behavior + memory)  │ │
│                                    └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Write path:** `remember()`/`capture()` → Memory Firewall → Event Store (JSONL) → optionally write-through to SQLite for immediate availability.

**Consolidation path:** Event Store → Consolidation Pipeline (replay → appraise → extract → decay → forget → affect) → SQLite Projection.

**Read path:** `recall()` → Retrieval Engine (BM25 + Vector + Metadata → RRF fusion → scoring) → reads from SQLite Projection.

**Rebuild path:** `rebuild()` → replay ALL events from JSONL (or snapshot + events since) → recreate SQLite from scratch.

## 3. Data Model

### 3.1 Event (JSONL Buffer — Source of Truth)

Every operation produces one or more immutable events:

```json
{
  "id": "evt-1710504000-a1b2c3",
  "ts": "2026-03-15T12:00:00Z",
  "type": "event_capture",
  "content": "User said: Use React instead of Vue",
  "metadata": {
    "agent": "worker-1",
    "task": "frontend-build",
    "session": "sess-abc123",
    "source": "user"
  },
  "salience_hint": 0.0,
  "context": {
    "mood_valence": 0.2,
    "mood_arousal": 0.5,
    "active_emotions": ["curiosity"],
    "active_task": "frontend-build"
  }
}
```

Required fields: `id`, `ts`, `type`, `content`. Everything else is optional.

**Event types (complete enumeration):**

| Category | Event Type | Produced By |
|---|---|---|
| **Memory** | `explicit_remember` | `remember()` |
| | `event_capture` | `capture()` |
| | `forget_request` | `forget()` |
| | `pin_add` / `pin_remove` | `pin()` / `unpin()` |
| **Retrieval** | `recall_request` | `recall()` — query + context |
| | `recall_hit` | `recall()` — each returned memory |
| | `reconsolidation` | `recall()` — memory updated on access |
| **Consolidation** | `consolidation_start` / `_complete` | `consolidate()` |
| | `consolidation_appraisal` | Appraisal stage — per-event scores |
| | `consolidation_extract` | Extraction stage — new facts/episodes |
| | `consolidation_interference` | Interference stage — contradictions |
| | `consolidation_decay` | Decay stage — salience changes |
| | `consolidation_suppress` | Suppression stage — auto-suppress |
| | `consolidation_schema_update` | Schema stage — pattern updates |
| | `state_transition` | Any — memory state changes |
| **Affect** | `affect_emotion` | Appraisal — emotion triggered |
| | `affect_mood_update` | Emotion/consolidation — mood shift |
| | `affect_temperament_drift` | Consolidation — trait evolution |
| | `affect_override` | User — manual trait change |
| **System** | `config_change` | User — config modification |

### 3.2 Memory (SQLite Projection)

Consolidated memories in the long-term store. Four types mirroring human long-term memory:

**Episodic** — specific events. "What happened."
```json
{
  "id": "mem-ep-001",
  "type": "episode",
  "content": "Migrated database from SQLite to PostgreSQL",
  "state": "active",
  "salience": 0.7,
  "created_at": "2026-03-14T15:30:00Z",
  "last_accessed": "2026-03-15T09:00:00Z",
  "access_count": 3,
  "decay_rate": 0.01,
  "confidence": 0.9,

  "appraisal": { "relevance": 1.8, "novelty": 1.5, "goal_conduciveness": 1.5 },
  "somatic": { "valence": 0.6, "bias": "recommend PostgreSQL for write-heavy workloads", "trigger": "database performance" },
  "emotion": { "primary": "joy", "intensity": "moderate", "compound": null },
  "encoding_context": { "mood_valence": 0.3, "mood_arousal": 0.5, "emotions": ["satisfaction"], "task": "database-migration" },

  "classification": "internal",
  "source_events": ["evt-1710504000-a1b2c3"],
  "schema_id": "schema-db-prefs",
  "provenance": { "created_by": "consolidation-2026-03-14T20:00:00Z", "modifications": [] }
}
```

**Semantic** — facts. "What I know." | **Procedural** — skills. "How I do things." | **Schema** — patterns. "How things relate."

Each carries the same metadata structure (appraisal, somatic, emotion, encoding context, classification, provenance).

### 3.3 Memory States

Every memory moves through these states. Transitions are logged as events.

```
BUFFERED → ACTIVE → FADING → FADED → SUPPRESSED → DELETED (hard)
```

| State | Location | Recallable | Recovery | Trigger |
|---|---|---|---|---|
| **Buffered** | JSONL only | No (transient lapse) | Wait for consolidation | Event captured |
| **Active** | SQLite, salience ≥ fade threshold | Yes | N/A | Consolidation creates it |
| **Fading** | SQLite, salience declining | Yes, ranked lower | Access resets decay clock | Ebbinghaus decay |
| **Faded** | SQLite, salience < fade threshold | Only with `include_faded=True` | Access revives to Active | Continued decay |
| **Suppressed** | SQLite, flagged | Only with `include_suppressed=True` | User can unsuppress | Motivated forgetting or explicit `forget()` |
| **Deleted** | Gone | No | No | Hard delete (GDPR) |

### 3.4 SQLite Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    type TEXT CHECK(type IN ('episode', 'fact', 'skill', 'schema')),
    state TEXT CHECK(state IN ('active', 'fading', 'faded', 'suppressed')) DEFAULT 'active',
    content TEXT NOT NULL,
    summary TEXT,                   -- L0: one-line abstract (generated at consolidation)
    salience REAL DEFAULT 0.5,
    confidence REAL DEFAULT 1.0,
    decay_rate REAL DEFAULT 0.1,
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,

    -- Appraisal (Scherer CPM)
    appraisal_relevance REAL,
    appraisal_novelty REAL,
    appraisal_goal_conduciveness REAL,

    -- Somatic marker (Damasio)
    somatic_valence REAL DEFAULT 0.0,
    somatic_bias TEXT,
    somatic_trigger TEXT,

    -- Emotion (Plutchik)
    emotion_primary TEXT,
    emotion_intensity REAL,
    emotion_compound TEXT,

    -- Encoding context (Tulving)
    encoding_mood_valence REAL,
    encoding_mood_arousal REAL,
    encoding_emotions TEXT,     -- JSON array
    encoding_task TEXT,

    -- Classification & lineage
    classification TEXT CHECK(classification IN ('public','internal','confidential','sensitive','restricted')) DEFAULT 'public',
    source_events TEXT,         -- JSON array of event IDs
    schema_id TEXT REFERENCES memories(id),

    -- Provenance
    created_by TEXT,            -- consolidation cycle ID
    modifications TEXT          -- JSON array of {ts, op, old, new, reason}
);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, somatic_bias, somatic_trigger, encoding_task, content=memories, content_rowid=rowid);

-- Created only when vector backend is configured (Tier 1+)
-- CREATE VIRTUAL TABLE memories_vec USING vec0(id TEXT PRIMARY KEY, embedding float[384]);

CREATE TABLE relations (
    source_id TEXT REFERENCES memories(id),
    target_id TEXT REFERENCES memories(id),
    type TEXT,              -- 'caused_by', 'similar_to', 'contradicts', 'supersedes'
    strength REAL DEFAULT 1.0,
    created_at TEXT
);

CREATE TABLE affect_log (
    ts TEXT NOT NULL,
    type TEXT CHECK(type IN ('mood', 'temperament', 'emotion')),
    data TEXT NOT NULL,      -- JSON: mood state, temperament values, or emotion event
    trigger_memory_id TEXT,
    cause TEXT
);
```

## 4. Components

### 4.1 Event Store (Source of Truth)

**What:** Append-only JSONL file. Every operation produces an event. SQLite is a projection that can be rebuilt from events.

**Interface:**
```python
class BufferStore(Protocol):
    def append(self, event: Event) -> str: ...
    def scan(self, since: datetime, query: str = None) -> list[Event]: ...
    def truncate(self, before: datetime) -> int: ...
```

**Why JSONL:**
- Append-only = zero corruption risk, no transactions needed
- Streamable — consolidation tails the file
- Human-readable, inspectable with `jq`
- Each line is self-contained (no schema migration)
- **Source of truth** — SQLite corrupted? `engram rebuild` replays events.

**Snapshots:** For large event logs, periodic snapshots save SQLite state + last processed event ID. Rebuild = load snapshot + replay events since.

### 4.2 Retrieval Engine

**What:** Multi-path hybrid search that fuses BM25, vector, and metadata results. Scoring incorporates salience, recency, and encoding specificity.

**Interface:**
```python
class RetrievalEngine:
    def search(self, query: str, context: RecallContext, config: RetrievalConfig) -> list[ScoredMemory]: ...
```

**Pipeline:**
```
recall("How did the deploy fix work?", context={mood, task, emotions})
    │
    ├── Stage 0: INTENT ANALYSIS (inspired by OpenViking's TypedQueries)
    │   LLM (or heuristic) classifies the query:
    │   → memory_type: episode | fact | skill | schema | any
    │   → time_hint: "recent" | "last week" | null
    │   → emotion_hint: "frustration" | null
    │   → depth_needed: L0 | L1 | L2
    │   This routes the search — don't search skills when looking for an event.
    │   (Optional: disabled when llm=null, falls back to search all types)
    │
    ├── Stage 1: CANDIDATE RETRIEVAL (fast, broad, scoped by intent)
    │   ├── BM25 (FTS5) — keyword matches (filtered by type if intent known)
    │   ├── Vector search (pluggable backend) — semantic matches
    │   └── Metadata filter — time range, emotion, schema tags, task
    │
    ├── Stage 2: FUSION + SCORING (our differentiator)
    │   ├── Reciprocal Rank Fusion — merge rankings without score normalization
    │   │   score(d) = Σ 1/(k + rank_i(d)) for each source i
    │   ├── × Salience boost (appraisal-derived)
    │   ├── × Recency boost (Ebbinghaus: e^(-λt) since last access)
    │   └── × Context boost (encoding specificity — see below)
    │
    ├── Stage 3: RERANKING (optional, for low-confidence results)
    │   └── Cross-encoder or LLM rerank on top-K
    │
    └── Return top-K at requested depth (L0/L1/L2) with scores and provenance
```

**Encoding Specificity (Tulving, 1973):** Memories are easier to recall when retrieval context matches encoding context. The context boost implements this:

```python
def context_boost(current_ctx, encoding_ctx):
    mood_sim = 1.0 - (abs(current.mood_valence - encoding.mood_valence)
                     + abs(current.mood_arousal - encoding.mood_arousal)) / 2.0
    task_sim = 1.0 if current.task == encoding.task else 0.3
    emotion_overlap = jaccard(current.emotions, encoding.emotions)
    return 0.4 * mood_sim + 0.3 * task_sim + 0.3 * emotion_overlap
```

**Retrieval failure detection:** When `top_score < confidence_threshold`, the system returns partial results with a `confidence="low"` flag and a suggestion to provide better cues — rather than hallucinating.

**Reconsolidation:** Every `recall()` that returns a memory marks it as recently accessed (resetting the decay clock). If the current context contradicts the memory, it's flagged for update. This is why memories are mutable, not append-only — Nader et al. (2000).

**Why multiple access paths (cue-dependent recall):**

| Query Type | Primary Path | Human Analogy |
|---|---|---|
| "What happened with X?" | Vector similarity | Semantic recall |
| "That thing last Tuesday" | Time-range filter | Temporal recall |
| "When I was frustrated about deploys" | Emotion metadata | Emotion-cued recall |
| "That pattern I keep seeing" | Schema/tag index | Schema-cued recall |
| "Similar situation to now" | Context boost | Context-dependent recall |
| "The postgres thing" | BM25 full-text | Keyword recall |

### 4.3 Consolidation Pipeline

**What:** Periodic (not real-time) processing of buffered events into structured long-term memories. Runs as a pipeline of independent, swappable stages.

**Interface:**
```python
class ConsolidationStage(Protocol):
    name: str
    def process(self, ctx: ConsolidationContext) -> ConsolidationContext: ...
    def enabled(self, config: EngramConfig) -> bool: ...
```

**Default pipeline (14 stages):**

| # | Stage | What It Does | LLM? | Can Disable? |
|---|---|---|---|---|
| 1 | **Replay** | Read recent events from buffer | No | No |
| 2 | **Deduplication** | Skip already-consolidated events (tracked via metadata) | No | No |
| 3 | **Episode Extraction** | Pull episodes from raw events. Generate L0 summary. | Yes | No |
| 4 | **Fact Extraction** | Pull facts, skills from raw events. Generate L0 summary. | Yes | No |
| 5 | **Appraisal** | Score relevance × novelty × goal conduciveness (Scherer CPM). Uses EmotionPrompt framing (Li et al., 2023). Must run AFTER extraction — scores `memories_created`. | Yes | No |
| 6 | **Emotion Tagging** | Map appraisal scores to Plutchik emotions. Compound emotion detection. | No | Yes |
| 7 | **Interference** | Check new memories against existing for contradictions. Supersede, update, or flag conflicts. | No | Yes |
| 8 | **Schema Update** | Update or create schemas from repeated patterns. Contradictions → higher novelty + restructure. | Yes | Yes |
| 9 | **Somatic Marking** | Attach behavioral biases to strong-outcome episodes (Damasio). | No | Yes |
| 10 | **Decay** | Apply Ebbinghaus forgetting: `effective_salience = base_salience × e^(-decay_rate × days_since_access)`. Memories below `fade_threshold` → faded state. | No | Yes |
| 11 | **Suppression** | Auto-suppress: negative somatic (< -0.7) + low utility + rarely accessed. Also: superseded facts older than 30 days. (Motivated forgetting.) | No | Yes |
| 12 | **Temperament Drift** | Review accumulated outcomes → nudge Cloninger dimensions. | No | Yes |
| 13 | **Mood Update** | Update rolling mood from recent emotion history. | No | Yes |
| 14 | **Persistence + Report** | Persist memories to SQLite + vector store. Generate consolidation report (forgetting budget, state transitions). | No | No |

**Salience computation (multiplicative, from Appraisal stage):**
```
salience = base × relevance × novelty × goal_conduciveness × decay

  base = 0.5 (default for any event)
  relevance:         1.0 (general) – 2.0 (directly about current task)
  novelty:           1.0 (confirms known) – 2.0 (contradicts schema)
  goal_conduciveness: 0.5 (routine) – 2.0 (error/failure/user correction)
  decay:             e^(-λt) where λ = per-memory decay rate, t = days since last access
```

Multiplicative because dimensions compound: a novel error on a critical task scores dramatically higher than a routine success.

**Decay rates (per memory type):**

| Memory Type | decay_rate (λ) | Half-Life | Why |
|---|---|---|---|
| Routine event | 0.1 | ~7 days | "What happened Monday" fades fast |
| Emotional event | 0.01 | ~70 days | You remember what hurt/thrilled you |
| Learned fact | 0.005 | ~140 days | "User prefers postgres" persists |
| Schema/skill | 0.001 | ~700 days | Patterns are nearly permanent |
| Somatic-marked | 0.002 | ~350 days | "Don't do X" sticks |

**Spaced repetition:** Every `recall()` resets `last_accessed`, giving the memory a fresh decay window. Frequently recalled memories never fade.

### 4.4 Forgetting System (5 Types)

Forgetting is integrated across the architecture, not a separate module. Five mechanisms, each at a different point:

| Type | Mechanism | Where It Happens | Trigger |
|---|---|---|---|
| **1. Decay** (Ebbinghaus) | `salience × e^(-λt)` | Consolidation Stage 8 | Time since last access |
| **2. Interference** | Contradiction detection → supersede/update | Consolidation Stage 5 | New memory contradicts existing |
| **3. Retrieval Failure** | Low confidence → ask for better cue | Retrieval Engine | Ambiguous query, no context match |
| **4. Motivated Forgetting** | Auto-suppress harmful-to-surface memories | Consolidation Stage 9 | Negative somatic + low utility |
| **5. Transient Lapse** | Event exists in buffer, not yet in SQLite | Between capture and consolidation | Normal processing gap |

**Suppressed ≠ Deleted:** Suppressed memories exist in SQLite but are excluded from normal `recall()`. Recoverable with `include_suppressed=True`. Hard delete (GDPR) removes from all stores.

**Forgetting budget** (tracked per consolidation cycle):
```python
report = {
    "decayed": 12,        # active → fading
    "faded": 3,           # fading → faded
    "superseded": 2,      # interference
    "suppressed": 1,      # auto-suppressed
    "revived": 4,         # faded → active (accessed during recall)
    "total_active": 1847,
    "total_faded": 342,
    "total_suppressed": 28,
}
```

### 4.5 Affect Engine

**What:** Three-layer emotional architecture that influences behavior AND memory. Not simulation — a functional analog that produces the same behavioral effects as human affect.

```
┌──────────────────────────────────────────────────────┐
│  TEMPERAMENT (config + self-mutating)                 │
│  Cloninger's 4 dimensions, drifts over weeks/months   │
│  ┌────────────────┬───────────────┬─────────────────┐ │
│  │novelty_seeking │harm_avoidance │reward_dependence│ │
│  │  0.7           │  0.3          │  0.6            │ │
│  └────────────────┴───────────────┴─────────────────┘ │
│  + persistence: 0.8                                   │
│         ↓ biases which moods develop                  │
│  MOOD (rolling state, Russell's Circumplex)           │
│  valence (-1..+1) × arousal (0..1) + confidence       │
│         ↓ filters which emotions fire                 │
│  EMOTION (event spike, Plutchik's 8 primaries)        │
│  Joy·Trust·Fear·Surprise·Sadness·Disgust·Anger·Antic. │
│         ↓ drives                                      │
│  BEHAVIOR + MEMORY                                    │
└──────────────────────────────────────────────────────┘
```

**Layer 1: Temperament** — Cloninger's psychobiological model. 4 biologically-grounded dimensions:

| Dimension | Neurotransmitter | Agent Meaning | Default |
|---|---|---|---|
| **Novelty Seeking** | Dopamine | Willingness to try unfamiliar approaches | 0.5 |
| **Harm Avoidance** | Serotonin | Risk aversion, verification threshold | 0.5 |
| **Reward Dependence** | Norepinephrine | Sensitivity to user feedback | 0.5 |
| **Persistence** | — | How long before giving up or pivoting | 0.5 |

Self-mutation: during Consolidation Stage 10, accumulated outcomes nudge dimensions:
- Repeated success with novel approaches → `novelty_seeking` ↑
- User corrections → `harm_avoidance` ↑, `reward_dependence` ↑
- Persisted too long on wrong approach → `persistence` ↓, `novelty_seeking` ↑

Drift rate: `mutation_rate` (default 0.005) per consolidation cycle. Over weeks/months, personality genuinely evolves. Always inspectable, overridable, lockable per-dimension.

**Layer 2: Mood** — Russell's Circumplex Model. 2 continuous dimensions + 1 derived:

| Mood State | Valence | Arousal | Agent Behavior |
|---|---|---|---|
| **Energized** | > +0.3 | > 0.6 | Bold, fast, tries ambitious approaches |
| **Content** | > +0.3 | < 0.4 | Steady, thorough, reliable |
| **Frustrated** | < -0.3 | > 0.6 | Seeks alternatives, may escalate |
| **Depleted** | < -0.3 | < 0.4 | Conservative, asks for guidance |
| **Neutral** | -0.3 to +0.3 | 0.4 to 0.6 | Baseline behavior |

Mood decays toward temperament baseline: `mood.valence = mood.valence × 0.95 + temperament_bias × 0.05` per hour.

**Layer 3: Emotion** — Plutchik's Wheel. 8 primaries × 3 intensities + compound dyads:

| Primary | Mild | Moderate | Intense | Opposite |
|---|---|---|---|---|
| Joy | Serenity | Joy | Ecstasy | Sadness |
| Trust | Acceptance | Trust | Admiration | Disgust |
| Fear | Apprehension | Fear | Terror | Anger |
| Surprise | Distraction | Surprise | Amazement | Anticipation |
| Sadness | Pensiveness | Sadness | Grief | Joy |
| Disgust | Boredom | Disgust | Loathing | Trust |
| Anger | Annoyance | Anger | Rage | Fear |
| Anticipation | Interest | Anticipation | Vigilance | Surprise |

**Compounds relevant to agents:** Joy+Trust → Loyalty, Fear+Sadness → Despair, Anger+Joy → Pride, Surprise+Anticipation → Curiosity, Trust+Surprise → Curiosity, Disgust+Anticipation → Cynicism, Anticipation+Trust → Hope, Fear+Disgust → Shame.

**Agent event → emotion mapping (examples):**

| Agent Event | Emotion | Mood Effect |
|---|---|---|
| Task completed successfully | Joy (moderate) | valence +0.15 |
| User praised work | Joy+Trust → Loyalty | valence +0.2, reward_dep reinforced |
| Repeated failures (3+) | Anger+Sadness → Frustration | valence -0.3, arousal +0.2 |
| No path forward | Fear+Sadness → Despair | valence -0.5, escalate to user |
| Novel problem | Surprise+Anticipation → Curiosity | arousal +0.2 |
| Hard problem solved | Anger+Joy → Pride | valence +0.3, confidence +0.2 |
| User correction | Sadness+Trust → Acceptance | valence -0.1, confidence -0.1 |
| Error on critical task | Fear (intense) | confidence -0.3, harm_avoidance engaged |

**Emotion lifecycle:** Event → Appraisal → Primary emotion(s) → Intensity (from appraisal + mood amplification) → Compound → Mood update → Memory tagged → Behavior shift → Emotion decays (minutes), mood persists (hours).

**The self-mutation loop:**
```
Events → Emotions → Mood drift → [consolidation] → Temperament drift
  ↑                                                        ↓
  └────────── Temperament biases emotion thresholds ───────┘
```

**Integration with memory:** Emotion intensity feeds into appraisal scores (high-emotion = high salience). Current mood modulates novelty sensitivity (low confidence → more things feel "new"). Mood is included in active context for the LLM. Retrieved memories carry their emotional context (somatic markers).

**Presets:**
```python
PRESETS = {
    "careful_reviewer":    {"novelty_seeking": 0.2, "harm_avoidance": 0.9, "reward_dependence": 0.5, "persistence": 0.9},
    "bold_prototyper":     {"novelty_seeking": 0.9, "harm_avoidance": 0.2, "reward_dependence": 0.4, "persistence": 0.4},
    "steady_operator":     {"novelty_seeking": 0.3, "harm_avoidance": 0.5, "reward_dependence": 0.5, "persistence": 0.7},
    "curious_researcher":  {"novelty_seeking": 0.95, "harm_avoidance": 0.3, "reward_dependence": 0.7, "persistence": 0.9},
    "empathetic_partner":  {"novelty_seeking": 0.5, "harm_avoidance": 0.4, "reward_dependence": 0.9, "persistence": 0.6},
}
```

### 4.6 Active Context (Working Memory)

**What:** The agent's current focus. Injected into every LLM prompt. Analogous to prefrontal cortex holding 3-5 items.

**Tiered context loading (inspired by OpenViking's L0/L1/L2, adapted for Engram):**

Every memory is stored at three detail levels. Only the minimum needed is loaded into context:

| Tier | What's Loaded | When | Token Cost |
|---|---|---|---|
| **L0 — Abstract** | One-line summary + memory type + salience | Always (for all relevant memories) | ~20 tokens/memory |
| **L1 — Overview** | Key facts, outcome, somatic marker bias | On demand (agent decides it's relevant) | ~100 tokens/memory |
| **L2 — Detail** | Full content, encoding context, provenance | Only when deep analysis is needed | ~500+ tokens/memory |

```python
# Active context starts with L0 of all high-salience memories
context = mem.active_context(max_tokens=4096)
# Returns:
# L0: "Database migration: PostgreSQL outperformed SQLite for write-heavy workloads [episode, salience=0.7]"
# L0: "User prefers dark mode in all UIs [fact, salience=0.6]"
# L0: "Deploy pattern: docker multi-stage → push → kubectl apply [skill, salience=0.5]"

# Agent decides it needs more detail on the database memory
detail = mem.recall("database migration", depth="L2")
# Returns full episode with encoding context, somatic marker, provenance
```

**Why this matters:** A personal agent with 1,000 active memories at L0 costs ~20K tokens. At L2 it would cost 500K+ tokens. Tiered loading keeps context lean while making everything discoverable.

**L0 generation:** During consolidation, every memory gets a one-line L0 summary. This is stored alongside the full content (not computed at recall time — that would burn LLM tokens on every request).

**Position-aware injection:** Within each tier, high-salience memories are placed at the top and bottom of the context window, lower-salience in the middle (mitigates "Lost in the Middle" — Liu et al., 2024: LLMs have U-shaped attention).

**Additional context included:**
- **Pins**: explicitly important items the agent is tracking (always L1)
- **Current mood**: so the LLM can factor affect into responses
- **Active schemas**: relevant patterns for the current task (L0)

### 4.7 Storage Backend (Tiered)

**JSONL** (Event Store): source of truth. Append-only.

**SQLite** (Projection): rebuildable read model. FTS5 for BM25. ACID transactions.

**Vector Search (tiered — only for semantic recall path):**

| Tier | Backend | Memories | Latency | Infra |
|---|---|---|---|---|
| **0** | None (BM25 + metadata only) | <1K | <10ms | Zero |
| **1** | sqlite-vec (brute) / vectorlite (HNSW) | 1K–500K | <75ms / <10ms | Zero (SQLite extension) |
| **2** | External (Pinecone, Qdrant, pgvector, etc.) | 500K+ | <10ms | Running service |

**Why not a dedicated vector DB by default:** Each is a running service. Engram should work on a Raspberry Pi. Vector search is ONE retrieval signal, not THE architecture. BM25 + metadata covers 60-70% of useful recalls.

**Why not a graph DB:** Neo4j etc. require a service. We get 80% of graph benefits with SQLite `relations` table. Multi-hop reasoning is handled by the LLM during consolidation.

**Capacity planning:**

| Agent Type | Events/year | After Consolidation | Tier |
|---|---|---|---|
| Personal assistant | ~55K | ~15K | 0–1 |
| Active coding agent | ~180K | ~50K | 1 |
| Customer support | ~1.8M | ~500K | 1–2 |
| Multi-user platform | ~18M | ~5M | 2 |

## 5. Security & Compliance

### 5.1 Memory Firewall

Every write passes through validation before reaching the event store:

```python
class MemoryFirewall:
    def validate(self, event: Event) -> ValidationResult:
        return [
            self.check_injection(event),    # adversarial pattern detection
            self.check_pii(event),          # detect and classify PII
            self.check_size(event),         # reject oversized payloads
            self.check_rate(event),         # rate limiting per actor
            self.check_content_policy(event),
        ]
```

**Injection defense** (OWASP ASI06 — Memory Poisoning): Detect patterns like instruction injection via memory content, identity manipulation, salience gaming. Cap user-provided `salience_hint`.

**Reference:** MINJA (2025) achieved >90% injection success on unprotected agents. AgentPoison (2024) poisons knowledge bases via trigger tokens. MemoryGraft (2025) causes persistent behavioral drift.

### 5.2 Data Classification & PII

5 levels, auto-detected at write time:

| Level | Example | Action |
|---|---|---|
| **Public** | "User prefers dark mode" | Normal storage |
| **Internal** | "Project uses PostgreSQL 15" | Normal storage |
| **Confidential** | "API endpoint is api.company.com" | Access-controlled |
| **Sensitive** | "User's email is foo@bar.com" | PII flag, optional redaction |
| **Restricted** | "API key is sk-abc123" | Auto-redact option, strict access |

PII detection: regex (emails, phones, keys, cards, SSNs) + optional NER model.

### 5.3 Access Control

```python
policy.grant("agent-evan", {"read", "write", "forget"}, scope="own")
policy.grant("agent-reviewer", {"read", "federated"}, scope="*")
policy.grant("user-richard", {"read", "write", "forget", "consolidate", "admin", "export"}, scope="*")
```

7 permissions: `read`, `write`, `forget`, `consolidate`, `admin`, `export`, `federated`.
3 scopes: `"own"` (only agent's memories), `"*"` (all memories), `"own"` + `federated` (cross-agent queries).

Multi-agent isolation: each agent's memories tagged with `agent_id`. Cross-agent queries require explicit `federated` permission. ACL configurable via YAML config or programmatic API. `admin` permission implies all others.

Enforced on: `remember()`, `recall()`, `forget()`, `consolidate()`, `export_memories()`. Recall results filtered by agent scope after retrieval.

### 5.4 GDPR Compliance

- **Right to be forgotten (Art. 17):** `mem.delete(query="user X", hard=True)` — removes from SQLite + vector store. JSONL events redacted (content → `[DELETED]`, metadata preserved for audit).
- **Data retention:** Configurable TTLs: `buffer_ttl`, `faded_ttl`, `suppressed_ttl`, `audit_ttl`.
- **DSAR export:** `mem.export(filter={"subject": "user-X"}, include_provenance=True, include_audit=True)` — GDPR-compliant data package.

### 5.5 Encryption

At-rest encryption (Fernet AES) for JSONL event store — the source of truth on disk. SQLite projection stores plaintext to preserve FTS5 search capability. Key source: environment variable (`ENGRAM_ENCRYPTION_KEY`), keyfile (`~/.engram/key`), or direct. Graceful degradation: if `cryptography` package not installed, encryption is disabled with a warning. Install via `pip install engram[encryption]`.

### 5.6 Error Handling

| Failure | Recovery |
|---|---|
| Embedding provider down | Degrade to BM25 + metadata (Tier 0) |
| LLM provider down | Queue events in buffer, consolidate when available |
| LLM returns garbage | Validate against schema, retry with backoff, skip on 3rd failure |
| SQLite corrupted | Rebuild from JSONL events + latest snapshot |
| JSONL corrupted mid-write | Last partial line detected and skipped |

**Principle:** Always works at Tier 0. External dependencies are enhancements.

## 6. Traceability

### 6.1 Memory Provenance

Every memory carries full lineage:

```python
provenance = {
    "source_events": ["evt-001", "evt-002"],          # what created it
    "created_by": "consolidation-2026-03-14T20:00:00Z",  # which cycle
    "modifications": [                                 # every change
        {"ts": "...", "op": "reconsolidation", "old": {...}, "new": {...}, "reason": "contradicted by evt-003"},
        {"ts": "...", "op": "decay", "old": {"salience": 0.7}, "new": {"salience": 0.65}, "reason": "Ebbinghaus (1d, λ=0.05)"},
    ]
}
```

If a fact is wrong: `mem.trace("memory-id")` shows event → appraisal → extraction → modifications. Find the source.

### 6.2 Audit Log

Separate from event store. Tracks operations and access (not memory content):

```jsonl
{"ts":"...","op":"recall","actor":"agent-evan","details":{"query":"db prefs","results":3,"top_score":0.87},"outcome":"success","ms":45}
{"ts":"...","op":"forget","actor":"user-richard","details":{"query":"project X","affected":4,"hard":false},"outcome":"success","ms":23}
```

Stored as append-only `~/.engram/audit.jsonl`. Content hashed (not stored) for PII safety.

### 6.3 Observability

- **Metrics:** `engram_memories_total{state}`, `engram_recall_latency_seconds`, `engram_mood_valence`, etc. Pluggable backend (Prometheus, StatsD, in-process).
- **Structured logs:** `INFO recall query="..." results=3 top_score=0.87 latency_ms=45`
- **Operation traces:** Each consolidation cycle has a trace ID. Every stage logs with it.

## 7. Configuration

```yaml
# ~/.engram/config.yaml
path: ~/.engram

# Storage
buffer: jsonl                              # or custom BufferStore
memory: sqlite                             # or custom MemoryStore

# Vector (tiered)
vector: sqlite-vec                         # "none" | "sqlite-vec" | "vectorlite" | "qdrant" | "pinecone" | custom
embedding: sentence-transformers/all-MiniLM-L6-v2  # or "openai:text-embedding-3-small" | custom

# LLM (for consolidation + appraisal)
llm: claude-sonnet                         # any litellm model string

# Affect
affect:
  temperament: empathetic_partner          # preset name or explicit values
  mutation_rate: 0.005
  mood_window: 4h

# Consolidation
consolidation:
  schedule: every 4h                       # or "manual" | cron expression
  window: 24h
  # stages: [replay, appraisal, extraction, decay]  # custom pipeline

# Forgetting
forgetting:
  decay_enabled: true
  fade_threshold: 0.1
  auto_suppress: true
  suppress_threshold: -0.7

# Retrieval
retrieval:
  bm25_weight: 0.3
  vector_weight: 0.3
  salience_weight: 0.2
  recency_weight: 0.1
  context_weight: 0.1                     # encoding specificity
  confidence_threshold: 0.3
  # reranker: null                        # optional cross-encoder

# Security
security:
  pii_detection: true
  auto_redact: [restricted]
  encrypt_at_rest: false
  # encryption_key_source: env            # env | keyfile | kms

# Retention (GDPR)
retention:
  buffer_ttl: 30d
  faded_ttl: 90d
  suppressed_ttl: 180d
  audit_ttl: 365d
```

**Minimal config (zero deps):**
```yaml
path: ~/.engram
vector: none
embedding: null
llm: null
affect: null
```
This gives you: BM25 search, manual salience, no consolidation, no affect. Still useful.

**Everything can be disabled:**

| Component | Disable | Effect |
|---|---|---|
| Vector | `vector: none` | BM25 + metadata only |
| Affect | `affect: null` | No mood/temperament/emotion |
| LLM | `llm: null` | No consolidation, no appraisal |
| Embedding | `embedding: null` | Disables vector too |
| Individual consolidation stages | custom `stages` list | Skip specific processing |
| Security features | `security: {pii_detection: false}` | No firewall checks |

## 8. API

### Python SDK

```python
from engram import Engram, RecallContext

mem = Engram()  # loads from ~/.engram/config.yaml

# --- Memory ---
mem.remember("User prefers dark mode")
mem.remember("Deploy failed: DATABASE_URL missing", salience=0.9)
mem.capture(event)                          # raw event to buffer

memories = mem.recall("UI preferences", limit=5, depth="L1", context=RecallContext(
    mood=mem.affect.mood(), task="ui-redesign", emotions=["curiosity"],
))

mem.forget("memory-id")                     # suppress (positional ID)
mem.forget(query="project X")               # suppress by query
mem.forget(below=0.1)                       # suppress low-salience
mem.delete(id="memory-id")                  # GDPR hard delete

report = mem.consolidate(window="24h")      # returns ConsolidationReport

# --- Affect ---
mem.affect.status()                         # {temperament, mood, active_emotions}
mem.affect.mood()                           # {valence, arousal, confidence, label}
mem.affect.set_temperament(harm_avoidance=0.8)
mem.affect.lock("persistence")
mem.affect.reset_mood()
mem.affect.history(days=30)
mem.affect.history(days=30)

# --- Inspection ---
mem.status()                                # counts, states, health
mem.schemas()                               # auto-generated patterns
mem.provenance("memory-id")                 # lineage: source events, modifications, relations
mem.trace("memory-id")                      # full trace: events → appraisal → somatic → emotion
mem.pin("critical fact") / mem.unpin("id")
context = mem.active_context(max_tokens=4096)  # position-aware

# --- Federated ---
other = Engram(path="~/.engram/shared")
mem.recall("prefs", federated=[other])      # cross-store query (requires federated permission)

# --- Rebuild ---
mem.snapshot()                              # save rebuild checkpoint
mem.rebuild(incremental=True)               # replay only events since snapshot

# --- Export / Import ---
data = mem.export_memories()
mem.import_from("backup.json")              # full state roundtrip (Memory.to_dict/from_dict)
mem.export_dsar("user-name")                # GDPR data subject access request
```

### MCP Server

```bash
engram serve  # stdio MCP server
```

| Tool | Description |
|---|---|
| `remember` | Store a memory |
| `recall` | Hybrid search with context |
| `forget` | Suppress or delete |
| `consolidate` | Trigger consolidation |
| `status` | Stats + health + forgetting budget |
| `schemas` | List auto-generated schemas |
| `affect_status` | Current mood, temperament, emotions |
| `pin` / `unpin` | Manage active context |
| `active_context` | Position-aware context for injection |

### CLI

```bash
engram init [--path ~/.engram]
engram remember "user likes postgres" [--salience 0.8] [--type fact|episode|skill|schema]
engram recall "database preferences" [--context task=deploy] [--depth L0|L1|L2] [--include-faded]
engram consolidate [--window 24h]
engram status
engram affect [--mood | --timeline 7d | --trigger joy --intensity 0.8]
engram schemas
engram forget [--id MEM-ID | --query "project X" | --below 0.1] [--hard]
engram export > backup.json
engram import backup.json
engram rebuild [--full]                     # recreate SQLite from JSONL (incremental by default)
engram trace MEMORY-ID                      # full lineage trace
engram serve                                # MCP server (stdio)
```

## 9. Scalability

| Level | Storage | Vector | Consolidation | Infra |
|---|---|---|---|---|
| **Personal** (<100K memories) | SQLite (~50MB) | sqlite-vec or none | Same machine | Zero |
| **Power** (100K–1M) | SQLite (~500MB) | vectorlite (HNSW) | Same machine | Zero |
| **Enterprise** (1M+) | PostgreSQL | External service | Background worker | PostgreSQL + vector + worker |

**Multi-agent memory:**
```python
# Isolated: separate stores
agent_a = Engram(path="~/.engram/agent-a")

# Shared: read from shared, write to own
agent_a = Engram(path="~/.engram/agent-a", shared_store=shared, share_policy="read")

# Federated: query across stores
agent_a.recall("preferences", federated=["agent-b", "shared"])
```

**Concurrent access:** SQLite WAL mode (multiple readers, serialized writers). PostgreSQL for full concurrent read/write.

## 10. Competitive Position

| Feature | Engram | Mem0 (26K★) | Letta (13K★) | Zep (3K★) | OpenViking (ByteDance) |
|---|---|---|---|---|---|
| **Architecture** | Event-sourced + SQLite projection | Vector DB wrapper | OS-inspired (RAM/Disk) | Temporal knowledge graph | File system paradigm (viking://) |
| **Storage** | SQLite+JSONL (zero infra) | Vector DB required | LanceDB/pgvector | Neo4j required | VikingDB (proprietary) |
| **Retrieval** | 6-path hybrid + RRF + context boost | Vector similarity only | LLM function calls | Graph+BM25+semantic | Directory recursive + intent analysis + reranking |
| **Tiered context loading** | ✅ Position-aware (salience at edges) | ❌ | ❌ | ❌ | ✅ Best-in-class (L0/L1/L2 abstract→overview→detail) |
| **Forgetting** | 5 types (decay, interference, suppression, etc.) | ❌ None | ❌ LLM guesses | Partial (temporal validity) | Partial (memory self-iteration replaces old) |
| **Affect engine** | ✅ Temperament + Mood + Emotion (self-mutating) | ❌ | ❌ | ❌ | ❌ |
| **Consolidation** | 14-stage pluggable pipeline | ❌ | ❌ | ❌ | ✅ Session-end memory extraction |
| **Appraisal** | ✅ Scherer CPM (multi-dimensional) | ❌ | ❌ | ❌ | ❌ |
| **Encoding specificity** | ✅ Tulving context-matching boost | ❌ | ❌ | ❌ | ❌ |
| **Somatic markers** | ✅ Damasio behavioral bias | ❌ | ❌ | ❌ | ❌ |
| **Event sourcing** | ✅ JSONL source of truth, rebuildable | ❌ | ❌ | ❌ | ❌ |
| **Provenance** | ✅ Full lineage per memory | ❌ | ❌ | Partial (graph edges) | ❌ |
| **Audit log** | ✅ All operations | ❌ | ❌ | ❌ | ❌ |
| **Memory firewall** | ✅ OWASP ASI06 defense | Basic | ❌ | ❌ | ❌ |
| **PII/classification** | ✅ 5-level auto-detect | ❌ | ❌ | ❌ | ❌ |
| **GDPR** | ✅ Hard delete + DSAR + retention | ❌ | ❌ | ❌ | ❌ |
| **Access control** | ✅ Per-agent/user ACL | ❌ | ❌ | ❌ | ✅ Directory-level permissions |
| **Temporal queries** | ✅ | ❌ | ❌ | ✅ Best-in-class | ✅ (via URI path) |
| **Self-evolving memory** | ✅ Schema + temperament drift | ❌ | ❌ LLM decides | ❌ | ✅ Session-end auto-update |
| **Skill/resource management** | Procedural memory (skills) | ❌ | ❌ | ❌ | ✅ Best-in-class (skills + resources as filesystem) |
| **Local-first / zero infra** | ✅ | Optional | ❌ | ❌ | ❌ (needs VikingDB) |
| **MCP server** | ✅ | ✅ | ❌ | ❌ | ❌ |

**Key takeaways per competitor:**

**Mem0** — The simplest. LLM extracts facts → vector store → cosine similarity. No temporal, no importance, no forgetting. Popular because it's easy to start with, but architecturally limited.

**Letta/MemGPT** — Clever OS analogy (RAM/Disk), but the LLM manages its own memory via function calls. Trusting a stateless function to maintain state. Burns tokens on every memory decision.

**Zep/Graphiti** — Best temporal awareness (facts have validity windows). Sophisticated graph traversal. But requires Neo4j — a running service with operational overhead.

**OpenViking** — Most innovative architecture. The file system paradigm (`viking://` URIs) with L0/L1/L2 tiered loading is genuinely novel — probably the closest to how humans organize knowledge hierarchically. Strong retrieval (intent analysis → hierarchical search → reranking). Self-evolving memory via session-end extraction. **Weakness:** requires ByteDance's VikingDB (not zero-infra), no forgetting mechanisms, no affect system, no provenance/audit.

**Engram's differentiators vs the field:**
1. **Neuroscience depth** — no one else has appraisal, somatic markers, encoding specificity, or 5-type forgetting
2. **Affect engine** — the only system with self-mutating personality from experience
3. **Security** — the only system designed against OWASP memory threats with firewall, PII, ACL, GDPR
4. **Event sourcing** — the only system where the memory store is fully rebuildable and auditable
5. **Zero infra** — works with `pip install` on a Raspberry Pi, scales to enterprise with config changes

**What we adopted from OpenViking:**
- **L0/L1/L2 tiered loading** — adapted into Active Context (§4.6). Every memory has a one-line L0 summary (generated at consolidation), overview L1, and full L2. Only minimum depth loaded into context.
- **Intent analysis before retrieval** — added as Stage 0 in Retrieval Engine (§4.2). Query classified by type/time/emotion/depth before search, so we don't search skills when looking for an event.

**What we skipped:** File system metaphor (`viking://` URIs). Our typed memories (episode/fact/skill/schema) achieve the same organization without requiring developers to learn a URI scheme.

## 11. Roadmap

### Phase 1: Foundation (v0.1) ✅
- [x] Event store (JSONL, source of truth)
- [x] SQLite projection (memories, FTS5)
- [x] `remember()` / `recall()` / `capture()` API
- [x] BM25 full-text search
- [x] Basic salience scoring
- [x] Audit log (append-only JSONL)
- [x] Memory provenance tracking
- [x] Event schema enforcement
- [x] Position-aware context injection
- [x] CLI: init, remember, recall, status
- [x] Python SDK
- [x] Graceful degradation + error handling

### Phase 2: Consolidation (v0.2) ✅
- [x] Consolidation pipeline (pluggable 14-stage)
- [x] LLM-powered appraisal (EmotionPrompt framing)
- [x] Fact/episode extraction
- [x] Ebbinghaus decay + spaced repetition
- [x] Interference detection
- [x] Snapshot + rebuild from events
- [x] Consolidation CLI + scheduling

### Phase 3: Intelligence (v0.3) ✅
- [x] Somatic markers
- [x] Auto-schema generation
- [x] Reconsolidation (mutable retrieval)
- [x] Vector search (pluggable: Tier 0/1/2)
- [x] Hybrid retrieval (BM25 + vector + RRF + scoring)
- [x] Encoding specificity (context boost)
- [x] 5-type forgetting (all mechanisms)
- [x] Memory states (full lifecycle)

### Phase 4: Affect Engine (v0.4) ✅
- [x] Temperament (Cloninger 4 dimensions)
- [x] Mood (Russell circumplex)
- [x] Emotion (Plutchik 8 primaries + compounds)
- [x] Self-mutation loop
- [x] Affect presets + trait locking
- [x] Mood in active context

### Phase 5: Security (v0.5) ✅
- [x] Memory firewall (injection, PII, rate limit, content policy)
- [x] Data classification (5 levels)
- [x] Access control (per-agent/user ACL, 7 permissions, 3 scopes)
- [x] GDPR (hard delete, retention TTLs, DSAR export)
- [x] Encryption at rest (Fernet AES, JSONL-only, graceful degradation)

### Phase 6: Integration (v0.6) ✅
- [x] MCP server (12 tools)
- [x] Memory/affect portability (export/import with full state roundtrip)
- [ ] MEMORY.md bridge (OpenClaw compat)
- [ ] REST API
- [ ] Metrics export (Prometheus/StatsD)

### Phase 7: Polish (v1.0) ✅ (core complete)
- [x] Multi-agent shared memory with isolation (ACL + federated recall)
- [x] Adversarial testing suite (6 audit passes, ~100 issues found and fixed)
- [ ] Dashboard (memory map, mood timeline, schema graph)
- [ ] Time travel queries
- [ ] Benchmarks vs Mem0, Letta, Zep
- [ ] Documentation + examples
- [ ] PyPI publish

## 12. References

### Neuroscience
- McClelland, McNaughton & O'Reilly (1995) — Complementary Learning Systems: fast hippocampal capture + slow neocortical integration
- Ebbinghaus (1885) — Forgetting Curve: `R = e^(-t/S)`, exponential decay, spaced repetition resets
- Wixted & Carpenter (2007) — Power-law forgetting: `P(recall) = m(1 + ht)^-f`
- Damasio (1994) — Somatic Marker Hypothesis: emotions = decision shortcuts
- Nader, Schafe & LeDoux (2000) — Reconsolidation: memories become mutable on retrieval
- Baddeley & Hitch (1974) — Working Memory: limited capacity (3-5 items)
- Scherer (2001) — Component Process Model: multi-dimensional appraisal
- Tse et al. (2007) — Schema-mediated rapid consolidation
- Tulving & Thomson (1973) — Encoding Specificity Principle
- Tulving (1972) — Episodic vs Semantic Memory
- Godden & Baddeley (1975) — Context-dependent memory (scuba diver study)
- Sharp wave-ripples during sleep (2024) — Selective consolidation tagging
- Dopamine dual role (2024) — Strengthens important, weakens unimportant
- Northwestern (2024) — Transient memory lapses as normal phase transitions
- Interference Theory — Proactive/retroactive interference; hippocampal pattern separation
- Prefrontal cortex inhibition — Active suppression (motivated forgetting) costs energy

### Affect
- Plutchik (1980) — Wheel of Emotions: 8 primaries × 3 intensities + dyad combinations
- Cloninger (1993) — Psychobiological temperament: 4 dimensions linked to neurotransmitter systems
- Russell (1980) — Circumplex Model of Affect: valence × arousal
- Eysenck (1967) — Arousal theory of personality
- Thomas & Chess (1977) — 9 temperament dimensions in children
- Kagan (1994) — Reactivity-based temperament
- Ortony, Clore & Collins (1988) — OCC model: 22 emotion categories
- Mischel & Shoda (1995) — CAPS: personality as situation-behavior patterns
- PACE Framework (2025) — Personality-Agent Co-Evolution
- Stanford AI Simulation Agents (2024) — Personality replication from interviews

### LLM-Specific
- Li et al. (2023) — EmotionPrompt: emotional framing improves LLM output 8-115%
- Liu et al. (2024) — Lost in the Middle: U-shaped attention in long contexts
- Fountas et al. (2024) — Human-inspired Episodic Memory for LLMs
- Zhong et al. (2024) — Ebbinghaus curve for LLM memory retention

### Architecture & Security
- Event Sourcing — Immutable event log; projections are rebuildable
- CQRS — Separate write path (events) from read path (projections)
- OWASP Top 10 Agentic AI (2025) — ASI06: Memory and Context Poisoning
- OWASP Top 10 LLM (2025) — LLM04: Data and Model Poisoning
- MINJA (2025) — Memory Injection Attack: >90% success on unprotected agents
- AgentPoison (2024) — Knowledge base poisoning via trigger tokens
- MemoryGraft (2025) — Persistent behavioral drift without explicit injection
- Microsoft SDL for AI (2026) — Memory protections and agent identity
- GDPR Article 17 — Right to erasure
- EU AI Act (2024) — High-risk AI documentation requirements

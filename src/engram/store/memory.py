"""SQLite memory store — the projection (read model), rebuildable from events."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engram.core.types import (
    Appraisal,
    DataClassification,
    EmotionTag,
    EncodingContext,
    Memory,
    MemoryState,
    MemoryType,
    Provenance,
    ScoredMemory,
    SomaticMarker,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('episode', 'fact', 'skill', 'schema')),
    state TEXT NOT NULL CHECK(state IN ('active', 'fading', 'faded', 'suppressed')) DEFAULT 'active',
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    salience REAL DEFAULT 0.5,
    confidence REAL DEFAULT 1.0,
    decay_rate REAL DEFAULT 0.1,
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    agent_id TEXT DEFAULT '',

    -- Appraisal (Scherer CPM)
    appraisal_relevance REAL DEFAULT 1.0,
    appraisal_novelty REAL DEFAULT 1.0,
    appraisal_goal_conduciveness REAL DEFAULT 1.0,

    -- Somatic marker (Damasio)
    somatic_valence REAL DEFAULT 0.0,
    somatic_bias TEXT DEFAULT '',
    somatic_trigger TEXT DEFAULT '',

    -- Emotion (Plutchik)
    emotion_primary TEXT DEFAULT '',
    emotion_intensity REAL DEFAULT 0.0,
    emotion_compound TEXT DEFAULT '',

    -- Encoding context (Tulving)
    encoding_mood_valence REAL,
    encoding_mood_arousal REAL,
    encoding_emotions TEXT DEFAULT '[]',
    encoding_task TEXT DEFAULT '',

    -- Classification & lineage
    classification TEXT DEFAULT 'public',
    source_events TEXT DEFAULT '[]',
    schema_id TEXT DEFAULT '',
    -- Provenance: Design §3.4 specifies separate created_by + modifications columns.
    -- We use a single JSON column for simplicity — same data, easier to extend.
    provenance TEXT DEFAULT '{}',

    -- Extraction confidence (Governed Memory paper, arXiv:2603.17787)
    -- 1.0 = explicit/direct, <1.0 = inferred. Multiplied into retrieval score.
    extraction_confidence REAL DEFAULT 1.0
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, summary, somatic_bias, somatic_trigger, encoding_task,
    content=memories, content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, somatic_bias, somatic_trigger, encoding_task)
    VALUES (new.rowid, new.content, new.summary, new.somatic_bias, new.somatic_trigger, new.encoding_task);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, somatic_bias, somatic_trigger, encoding_task)
    VALUES ('delete', old.rowid, old.content, old.summary, old.somatic_bias, old.somatic_trigger, old.encoding_task);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, somatic_bias, somatic_trigger, encoding_task)
    VALUES ('delete', old.rowid, old.content, old.summary, old.somatic_bias, old.somatic_trigger, old.encoding_task);
    INSERT INTO memories_fts(rowid, content, summary, somatic_bias, somatic_trigger, encoding_task)
    VALUES (new.rowid, new.content, new.summary, new.somatic_bias, new.somatic_trigger, new.encoding_task);
END;

-- Relations (lightweight graph — no FK enforcement for flexibility)
CREATE TABLE IF NOT EXISTS relations (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, type)
);

-- Pins (active context)
CREATE TABLE IF NOT EXISTS pins (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Affect log (temperament/mood/emotion persistence)
CREATE TABLE IF NOT EXISTS affect_log (
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    trigger_memory_id TEXT DEFAULT '',
    cause TEXT DEFAULT ''
);

-- Metadata
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Typed properties (dual extraction: schema-enforced key-value pairs)
CREATE TABLE IF NOT EXISTS memory_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'text',
    confidence REAL NOT NULL DEFAULT 1.0,
    UNIQUE(memory_id, key)
);
"""


class SQLiteMemoryStore:
    """SQLite projection of consolidated memories.

    This is NOT the source of truth — the JSONL event store is.
    This store can be rebuilt from events at any time via rebuild().
    """

    def __init__(self, base_path: Path):
        self.db_path = base_path / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Thread-local connection cache. SQLite WAL mode permits concurrent readers
        # across threads/processes; each thread gets its own connection so no cross-thread
        # cursor sharing can occur. Writes serialize at the SQLite layer via busy_timeout.
        self._tls = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._tls.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        # Migrations for extending existing DBs
        self._migrate(conn)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent migrations that ALTER TABLE for additive columns."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "extraction_confidence" not in existing:
            conn.execute("ALTER TABLE memories ADD COLUMN extraction_confidence REAL DEFAULT 1.0")
            logger.info("migration: added memories.extraction_confidence")

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    # --- CRUD ---

    def upsert(self, memory: Memory, dedup_threshold: float = 0.0, vector_store=None,
               embedding_provider=None, acl_filter=None) -> bool:
        """Insert or update a memory.

        If dedup_threshold > 0 and vector_store + embedding_provider are provided,
        checks for near-duplicate memories before writing. Returns False if skipped.

        acl_filter, if provided, is a callable taking a candidate memory's
        agent_id and returning True iff the writer is allowed to "see" that
        candidate for dedup purposes. This closes the cross-agent write-dedup
        side-channel (§D-write-dedup-acl): without it, vector_store.search()
        scans globally and Bob's write can be silently suppressed by Alice's
        memory whose embedding sits within `dedup_threshold` cosine.
        """
        # Write-side cosine dedup (from Governed Memory paper, threshold 0.92).
        # Note: vector_store.search() returns a backend-specific score (e.g. for
        # sqlite-vec it is 1/(1+L2_distance), NOT cosine). For dedup we must
        # compare the *cosine* between the candidate's embedding and each
        # nearest neighbour's stored embedding, so we re-compute it explicitly
        # against the top-k result. This decouples the dedup decision from the
        # search backend's ranking metric.
        if dedup_threshold > 0 and vector_store is not None and embedding_provider is not None:
            try:
                query_vec = embedding_provider.embed(memory.content)
                if query_vec:
                    # Pull a small candidate pool (K=5) — backend-ranked, so any
                    # neighbour with cosine ≥ threshold will be in the top few.
                    # When acl_filter is active we widen the pool because top-K
                    # may all be cross-agent neighbours that get filtered out.
                    k = 5 if acl_filter is None else 32
                    results = vector_store.search(query_vec, limit=k)
                    if results:
                        import math
                        q_norm = math.sqrt(sum(x * x for x in query_vec))
                        if q_norm > 0:
                            for r in results:
                                # ACL-scope the dedup decision: skip neighbours
                                # the writer is not allowed to "see". Without
                                # this, cross-agent writes get silently
                                # suppressed by another agent's memory — a
                                # presence oracle on the audit + recall
                                # channels (§D-write-dedup-acl).
                                if acl_filter is not None:
                                    cand = self.get(r.memory_id)
                                    if cand is None:
                                        continue
                                    if not acl_filter(getattr(cand, "agent_id", "")):
                                        continue
                                stored = None
                                getter = getattr(vector_store, "get_vector", None)
                                if callable(getter):
                                    stored = getter(r.memory_id)
                                if stored is None:
                                    # Backend doesn't expose stored vectors —
                                    # fall back to the search score (legacy
                                    # behaviour, only correct when the backend
                                    # score IS cosine, e.g. brute-force path).
                                    if r.score > dedup_threshold:
                                        return False
                                    continue
                                s_norm = math.sqrt(sum(x * x for x in stored))
                                if s_norm == 0:
                                    continue
                                dot = sum(a * b for a, b in zip(query_vec, stored))
                                cos = dot / (q_norm * s_norm)
                                if cos > dedup_threshold:
                                    return False
            except Exception:
                pass  # If embedding fails, proceed with normal write
        
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memories (
                id, type, state, content, summary, salience, confidence, decay_rate,
                created_at, last_accessed, access_count, agent_id,
                appraisal_relevance, appraisal_novelty, appraisal_goal_conduciveness,
                somatic_valence, somatic_bias, somatic_trigger,
                emotion_primary, emotion_intensity, emotion_compound,
                encoding_mood_valence, encoding_mood_arousal, encoding_emotions, encoding_task,
                classification, source_events, schema_id, provenance, extraction_confidence
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                state=excluded.state, content=excluded.content, summary=excluded.summary,
                salience=excluded.salience, confidence=excluded.confidence,
                last_accessed=excluded.last_accessed, access_count=excluded.access_count,
                agent_id=excluded.agent_id,
                appraisal_relevance=excluded.appraisal_relevance,
                appraisal_novelty=excluded.appraisal_novelty,
                appraisal_goal_conduciveness=excluded.appraisal_goal_conduciveness,
                somatic_valence=excluded.somatic_valence, somatic_bias=excluded.somatic_bias,
                somatic_trigger=excluded.somatic_trigger,
                emotion_primary=excluded.emotion_primary, emotion_intensity=excluded.emotion_intensity,
                emotion_compound=excluded.emotion_compound,
                encoding_mood_valence=excluded.encoding_mood_valence,
                encoding_mood_arousal=excluded.encoding_mood_arousal,
                encoding_emotions=excluded.encoding_emotions, encoding_task=excluded.encoding_task,
                classification=excluded.classification, source_events=excluded.source_events,
                schema_id=excluded.schema_id, provenance=excluded.provenance,
                extraction_confidence=excluded.extraction_confidence
            """,
            self._memory_to_row(memory),
        )
        conn.commit()
        return True

    def upsert_properties(self, memory_id: str, properties: list[dict]) -> None:
        """Store typed properties for a memory (dual extraction).
        
        Each property: {"key": str, "value": str, "type": str, "confidence": float}
        """
        conn = self._get_conn()
        for prop in properties:
            conn.execute(
                """INSERT INTO memory_properties (memory_id, key, value, type, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, key) DO UPDATE SET
                    value=excluded.value, type=excluded.type, confidence=excluded.confidence""",
                (memory_id, prop["key"], str(prop["value"]), prop.get("type", "text"), prop.get("confidence", 1.0)),
            )
        conn.commit()

    def get_properties(self, memory_id: str) -> list[dict]:
        """Get typed properties for a memory."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key, value, type, confidence FROM memory_properties WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [{"key": r["key"], "value": r["value"], "type": r["type"], "confidence": r["confidence"]} for r in rows]

    def search_by_property(self, key: str, value: str = None, limit: int = 10) -> list[Memory]:
        """Find memories by typed property key (and optionally value)."""
        conn = self._get_conn()
        if value is not None:
            rows = conn.execute(
                """SELECT m.* FROM memories m
                JOIN memory_properties p ON m.id = p.memory_id
                WHERE p.key = ? AND p.value = ? AND m.state IN ('active', 'fading')
                ORDER BY m.salience DESC LIMIT ?""",
                (key, value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT m.* FROM memories m
                JOIN memory_properties p ON m.id = p.memory_id
                WHERE p.key = ? AND m.state IN ('active', 'fading')
                ORDER BY m.salience DESC LIMIT ?""",
                (key, limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def filter_by_properties(
        self,
        filters: dict[str, str],
        limit: int = 50,
    ) -> list[Memory]:
        """Find memories matching ALL property filters.

        Filter values support comparison operators on numeric properties:
            ">100", ">=100", "<100", "<=100", "==42", "!=42"
        Plain string values do equality match.

        Args:
            filters: dict mapping property key -> value-or-expression.
            limit: max results.

        Returns:
            Memories satisfying every filter (intersection), ordered by salience desc.
        """
        if not filters:
            return []
        conn = self._get_conn()
        clauses = []
        params: list = []
        op_re = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+)$")
        for key, raw in filters.items():
            m = op_re.match(str(raw))
            if m:
                op, num = m.group(1), m.group(2).strip()
                # Numeric compare on CAST(value AS REAL)
                clauses.append(
                    f"EXISTS (SELECT 1 FROM memory_properties p WHERE p.memory_id = m.id "
                    f"AND p.key = ? AND CAST(p.value AS REAL) {op} ?)"
                )
                params.extend([key, float(num)])
            else:
                clauses.append(
                    "EXISTS (SELECT 1 FROM memory_properties p WHERE p.memory_id = m.id "
                    "AND p.key = ? AND p.value = ?)"
                )
                params.extend([key, str(raw)])
        sql = (
            "SELECT m.* FROM memories m WHERE m.state IN ('active', 'fading') "
            f"AND {' AND '.join(clauses)} ORDER BY m.salience DESC LIMIT ?"
        )
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get(self, id: str) -> Memory | None:
        """Get a memory by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def delete(self, id: str) -> bool:
        """Hard delete a memory. Returns True if found and deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0

    def update_state(self, id: str, state: MemoryState) -> None:
        """Update a memory's state."""
        conn = self._get_conn()
        conn.execute("UPDATE memories SET state = ? WHERE id = ?", (state.value, id))
        conn.commit()

    def mark_accessed(self, id: str) -> None:
        """Mark a memory as accessed (spaced repetition — resets decay clock)."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            """UPDATE memories SET last_accessed = ?, access_count = access_count + 1,
               state = CASE WHEN state IN ('fading', 'faded') THEN 'active' ELSE state END
               WHERE id = ?""",
            (now, id),
        )
        conn.commit()

    # --- Search ---

    @staticmethod
    def _sanitize_fts_query(query: str, max_terms: int = 10) -> str:
        """Sanitize a query string for FTS5 MATCH.

        FTS5 chokes on long text, special characters, and operator-like words.
        Extract meaningful keywords and join with implicit AND.
        """
        import re
        # Strip FTS5 operators and special chars
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        # Split into words, filter short/stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                "on", "with", "at", "by", "from", "as", "into", "through", "during",
                "before", "after", "above", "below", "between", "out", "off", "over",
                "under", "again", "further", "then", "once", "that", "this", "these",
                "those", "and", "but", "or", "nor", "not", "so", "if", "it", "its",
                "i", "me", "my", "you", "your", "he", "she", "we", "they", "them",
                "what", "which", "who", "whom", "when", "where", "why", "how"}
        # FTS5 reserved operator keywords — case-insensitive in MATCH.
        # If we let `AND`/`OR`/`NOT`/`NEAR` through as bare terms, FTS5 parses
        # them as operators and either errors or returns surprising results.
        fts_ops = {"and", "or", "not", "near"}
        words = [
            w for w in cleaned.lower().split()
            if len(w) > 2 and w not in stop and w not in fts_ops
        ]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        # Take top N terms, join with OR for broader matching
        terms = unique[:max_terms]
        if not terms:
            # All words were stop words — use original words (minus very short ones)
            fallback = [
                w for w in cleaned.lower().split()
                if len(w) > 1 and w not in fts_ops
            ][:max_terms]
            if fallback:
                return " OR ".join(fallback)
            # Last resort: extract any sequence of word chars from the original input.
            # NEVER return the raw query — it may contain FTS5 operators or control
            # chars (e.g. \x1b) that break MATCH syntax. If nothing salvageable
            # remains, return "" and let the caller fall back to LIKE.
            salvage = re.findall(r"\w+", query.lower())
            salvage = [
                w for w in salvage if len(w) > 1 and w not in fts_ops
            ][:max_terms]
            return " OR ".join(salvage)
        return " OR ".join(terms)

    def search_text(self, query: str, limit: int = 10, states: list[str] | None = None) -> list[ScoredMemory]:
        """BM25 full-text search via FTS5."""
        if states is None:
            states = ["active", "fading"]
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in states)

        fts_query = self._sanitize_fts_query(query)
        try:
            rows = conn.execute(
                f"""SELECT m.*, fts.rank as fts_rank
                    FROM memories_fts fts
                    JOIN memories m ON m.rowid = fts.rowid
                    WHERE memories_fts MATCH ?
                    AND m.state IN ({placeholders})
                    ORDER BY fts.rank
                    LIMIT ?""",
                (fts_query, *states, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 query syntax error — fall back to LIKE
            logger.warning("FTS5 query failed, falling back to LIKE: %s", query)
            rows = conn.execute(
                f"""SELECT *, 0 as fts_rank FROM memories
                    WHERE content LIKE ?
                    AND state IN ({placeholders})
                    LIMIT ?""",
                (f"%{query}%", *states, limit),
            ).fetchall()

        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            # BM25 rank is negative (lower = better), normalize to 0-1
            bm25_score = 1.0 / (1.0 + abs(row["fts_rank"])) if row["fts_rank"] else 0.5
            score = bm25_score * memory.salience
            results.append(ScoredMemory(memory=memory, score=score, sources={"bm25": bm25_score}))
        return results

    def search_by_type(self, memory_type: MemoryType, limit: int = 10) -> list[Memory]:
        """Get memories by type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE type = ? AND state IN ('active', 'fading') ORDER BY salience DESC LIMIT ?",
            (memory_type.value, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_by_state(self, state: MemoryState, limit: int = 100) -> list[Memory]:
        """Get memories by state."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE state = ? ORDER BY salience DESC LIMIT ?",
            (state.value, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def purge_by_ttl(self, state: MemoryState, ttl_days: int) -> int:
        """Delete memories in given state older than TTL days. Returns count deleted."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
        cursor = conn.execute(
            """DELETE FROM memories WHERE state = ?
               AND COALESCE(last_accessed, created_at) < ?""",
            (state.value, cutoff),
        )
        conn.commit()
        return cursor.rowcount

    def all_active(self) -> list[Memory]:
        """Get all active + fading memories (for consolidation)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE state IN ('active', 'fading') ORDER BY salience DESC"
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_by_agent(self, agent_id: str, limit: int = 100) -> list[Memory]:
        """Get memories owned by a specific agent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND state IN ('active', 'fading') ORDER BY salience DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        """Memory store statistics."""
        conn = self._get_conn()
        counts = {}
        for state in MemoryState:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE state = ?", (state.value,)
            ).fetchone()
            counts[state.value] = row["c"]

        type_counts = {}
        for mt in MemoryType:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE type = ? AND state IN ('active', 'fading')",
                (mt.value,),
            ).fetchone()
            type_counts[mt.value] = row["c"]

        total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        pins = conn.execute("SELECT COUNT(*) as c FROM pins").fetchone()["c"]

        return {
            "total_memories": total,
            "by_state": counts,
            "by_type": type_counts,
            "pins": pins,
        }

    # --- Pins ---

    def add_pin(self, id: str, content: str) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO pins (id, content, created_at) VALUES (?, ?, ?)",
            (id, content, now),
        )
        conn.commit()

    def remove_pin(self, id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM pins WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_pins(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM pins ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    # --- Affect Persistence ---

    def log_affect(self, affect_type: str, data: dict, trigger_memory_id: str = "", cause: str = "") -> None:
        """Log an affect state change (mood/temperament/emotion)."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO affect_log (ts, type, data, trigger_memory_id, cause) VALUES (?, ?, ?, ?, ?)",
            (now, affect_type, json.dumps(data), trigger_memory_id, cause),
        )
        conn.commit()

    def get_latest_affect(self, affect_type: str) -> dict | None:
        """Get the most recent affect state of a given type."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM affect_log WHERE type = ? ORDER BY ts DESC LIMIT 1",
            (affect_type,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def get_affect_history(self, affect_type: str | None = None, limit: int = 100) -> list[dict]:
        """Get affect history entries."""
        conn = self._get_conn()
        if affect_type:
            rows = conn.execute(
                "SELECT * FROM affect_log WHERE type = ? ORDER BY ts DESC LIMIT ?",
                (affect_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM affect_log ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            entry["data"] = json.loads(entry["data"])
            results.append(entry)
        return results

    # --- Rebuild ---

    def drop_all(self) -> None:
        """Drop all data (for rebuild from events)."""
        conn = self._get_conn()
        conn.executescript("""
            DELETE FROM memories;
            DELETE FROM relations;
            DELETE FROM pins;
            DELETE FROM affect_log;
            DELETE FROM metadata;
        """)
        conn.commit()
        logger.info("all data dropped for rebuild")

    # --- Relations (lightweight graph) ---

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key."""
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata key-value pair."""
        conn = self._get_conn()
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

    def add_relation(self, source_id: str, target_id: str, rel_type: str, strength: float = 1.0) -> None:
        """Add a relation between two memories."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO relations (source_id, target_id, type, strength, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, target_id, rel_type, strength, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def get_relations(self, memory_id: str, rel_type: str | None = None) -> list[dict]:
        """Get all relations for a memory (both directions)."""
        conn = self._get_conn()
        if rel_type:
            rows = conn.execute(
                """SELECT source_id, target_id, type, strength, created_at FROM relations
                   WHERE (source_id = ? OR target_id = ?) AND type = ?""",
                (memory_id, memory_id, rel_type),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT source_id, target_id, type, strength, created_at FROM relations
                   WHERE source_id = ? OR target_id = ?""",
                (memory_id, memory_id),
            ).fetchall()
        return [{"source_id": r[0], "target_id": r[1], "type": r[2], "strength": r[3], "created_at": r[4]} for r in rows]

    # --- Internal helpers ---

    def _memory_to_row(self, m: Memory) -> tuple:
        return (
            m.id, m.type.value, m.state.value, m.content, m.summary,
            m.salience, m.confidence, m.decay_rate,
            m.created_at.isoformat(), m.last_accessed.isoformat() if m.last_accessed else None,
            m.access_count, m.agent_id,
            m.appraisal.relevance, m.appraisal.novelty, m.appraisal.goal_conduciveness,
            m.somatic.valence, m.somatic.bias, m.somatic.trigger,
            m.emotion.primary, m.emotion.intensity, m.emotion.compound,
            m.encoding_context.mood_valence, m.encoding_context.mood_arousal,
            json.dumps(m.encoding_context.emotions), m.encoding_context.task,
            m.classification.value, json.dumps(m.source_events), m.schema_id,
            json.dumps(m.provenance.to_dict()),
            m.extraction_confidence,
        )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        prov_data = json.loads(row["provenance"]) if row["provenance"] else {}
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            state=MemoryState(row["state"]),
            content=row["content"],
            summary=row["summary"] or "",
            salience=row["salience"],
            confidence=row["confidence"],
            decay_rate=row["decay_rate"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            access_count=row["access_count"],
            agent_id=row["agent_id"] if "agent_id" in row.keys() else "",
            appraisal=Appraisal(
                relevance=row["appraisal_relevance"],
                novelty=row["appraisal_novelty"],
                goal_conduciveness=row["appraisal_goal_conduciveness"],
            ),
            somatic=SomaticMarker(
                valence=row["somatic_valence"],
                bias=row["somatic_bias"] or "",
                trigger=row["somatic_trigger"] or "",
            ),
            emotion=EmotionTag(
                primary=row["emotion_primary"] or "",
                intensity=row["emotion_intensity"],
                compound=row["emotion_compound"] or "",
            ),
            encoding_context=EncodingContext(
                mood_valence=row["encoding_mood_valence"],
                mood_arousal=row["encoding_mood_arousal"],
                emotions=json.loads(row["encoding_emotions"]) if row["encoding_emotions"] else [],
                task=row["encoding_task"] or "",
            ),
            classification=DataClassification(row["classification"]) if row["classification"] else DataClassification.PUBLIC,
            source_events=json.loads(row["source_events"]) if row["source_events"] else [],
            schema_id=row["schema_id"] or "",
            provenance=Provenance.from_dict(prov_data),
            extraction_confidence=row["extraction_confidence"] if "extraction_confidence" in row.keys() else 1.0,
        )

"""Vector store interface and implementations.

Tier 0: No vectors (BM25 only) — NoVectorStore
Tier 1: sqlite-vec (local, zero-infra) — SQLiteVecStore
Tier 2: External service (Pinecone, Qdrant, etc.) — future
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """A vector search result."""
    memory_id: str
    distance: float
    score: float  # normalized 0-1 (1 = best match)


class VectorStore(ABC):
    """Abstract vector store."""

    @abstractmethod
    def upsert(self, memory_id: str, vector: list[float]) -> None:
        """Store or update a vector."""
        ...

    @abstractmethod
    def search(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]:
        """Search for nearest neighbors."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a vector."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Count stored vectors."""
        ...

    def drop_all(self) -> None:
        """Drop all stored vectors (for full rebuild). Default: no-op."""
        return None

    def get_vector(self, memory_id: str) -> list[float] | None:
        """Return the stored vector for `memory_id`, or None if absent.

        Default implementation returns None; subclasses with materialised
        storage override this so the write-side cosine dedup can compute
        a true cosine (independent of the search backend's distance metric).
        """
        return None

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


class NoVectorStore(VectorStore):
    """Tier 0: No vector search. BM25 only."""

    def upsert(self, memory_id: str, vector: list[float]) -> None:
        pass

    def search(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]:
        return []

    def delete(self, memory_id: str) -> None:
        pass

    def count(self) -> int:
        return 0


class SQLiteVecStore(VectorStore):
    """Tier 1: sqlite-vec based vector store (local, zero-infra).

    Uses the sqlite-vec extension for efficient vector similarity search.
    Falls back to brute-force cosine similarity if sqlite-vec is unavailable.
    """

    def __init__(self, db_path: Path, dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        # Thread-local connections — sqlite3 connections are not thread-safe.
        # Without this, concurrent remember() calls trip SQLite's thread-affinity
        # check, the engine swallows the exception, and write-side cosine dedup
        # is silently bypassed under contention.
        self._tls = threading.local()
        self._has_vec_ext: bool | None = None  # detected on first connect
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            # Try to load sqlite-vec extension
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                if self._has_vec_ext is None:
                    self._has_vec_ext = True
                    logger.info("sqlite-vec extension loaded")
            except (ImportError, Exception) as e:
                if self._has_vec_ext is None:
                    self._has_vec_ext = False
                    logger.info("sqlite-vec not available, using brute-force fallback: %s", e)
            self._tls.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        if self._has_vec_ext:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors
                USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{self.dimension}])
            """)
        else:
            # Fallback: store vectors as JSON blobs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL
                )
            """)
        conn.commit()

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    def upsert(self, memory_id: str, vector: list[float]) -> None:
        if not vector:
            return
        conn = self._get_conn()
        if self._has_vec_ext:
            import struct
            blob = struct.pack(f"{len(vector)}f", *vector)
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (memory_id, embedding) VALUES (?, ?)",
                (memory_id, blob),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (memory_id, embedding) VALUES (?, ?)",
                (memory_id, json.dumps(vector)),
            )
        conn.commit()

    def search(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]:
        if not query_vector:
            return []

        conn = self._get_conn()

        if self._has_vec_ext:
            import struct
            blob = struct.pack(f"{len(query_vector)}f", *query_vector)
            rows = conn.execute(
                """SELECT memory_id, distance
                   FROM memory_vectors
                   WHERE embedding MATCH ?
                   ORDER BY distance
                   LIMIT ?""",
                (blob, limit),
            ).fetchall()
            results = []
            for row in rows:
                dist = row["distance"]
                score = 1.0 / (1.0 + dist)  # convert distance to 0-1 score
                results.append(VectorResult(memory_id=row["memory_id"], distance=dist, score=score))
            return results
        else:
            # Brute-force cosine similarity fallback
            return self._brute_force_search(query_vector, limit)

    def _brute_force_search(self, query_vector: list[float], limit: int) -> list[VectorResult]:
        """Fallback: compute cosine similarity against all stored vectors."""
        import math

        conn = self._get_conn()
        rows = conn.execute("SELECT memory_id, embedding FROM memory_vectors").fetchall()

        results = []
        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0:
            return []

        for row in rows:
            stored = json.loads(row["embedding"])
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vector, stored))
            s_norm = math.sqrt(sum(x * x for x in stored))
            if s_norm == 0:
                continue
            sim = dot / (q_norm * s_norm)
            distance = 1.0 - sim
            score = max(sim, 0.0)
            results.append(VectorResult(memory_id=row["memory_id"], distance=distance, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def delete(self, memory_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
        conn.commit()

    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) as c FROM memory_vectors").fetchone()["c"]

    def drop_all(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_vectors")
        conn.commit()

    def get_vector(self, memory_id: str) -> list[float] | None:
        """Return the stored vector for `memory_id`, decoded to a Python list."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM memory_vectors WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        emb = row["embedding"]
        if self._has_vec_ext:
            import struct
            n = len(emb) // 4
            return list(struct.unpack(f"{n}f", emb))
        return list(json.loads(emb))

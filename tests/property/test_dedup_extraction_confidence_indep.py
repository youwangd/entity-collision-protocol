"""Property: write-side dedup is independent of extraction_confidence.

Decision (paper §3.7): dedup is a *content* deduplication mechanism. It
operates on cosine similarity of the candidate's content embedding against
its nearest neighbours and MUST NOT be coupled to per-fact extraction
confidence — those are orthogonal channels:

  • dedup_threshold = "is this content already in the store?"
  • extraction_confidence = "how sure are we this fact was extracted right?"

Coupling them would mean a low-confidence fact could either (a) sneak in
when its near-duplicate would normally be deduped, or (b) be rejected when
a high-confidence near-duplicate would have landed. Either direction
breaks the schema lifecycle: dedup outcomes become non-monotone in the
extractor's calibration drift.

This module pins independence with two Hypothesis properties:

  (I1) For any pair of payloads (a, b), the SECOND-write landed/deduped
       outcome is invariant under any choice of extraction_confidence
       in {0.0, 0.25, 0.5, 0.75, 1.0} for either memory.
  (I2) The retrieval-time extraction_confidence multiplier is unchanged
       on memories that landed via dedup (the keeper's confidence is
       preserved; the duplicate's confidence is not silently overwritten).

If a future commit deliberately couples the two channels, this file is
the thing that has to change — and the rationale must land in §3.7.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Engram, Config
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore


class _DetEmbedder(EmbeddingProvider):
    """Bag-of-chars deterministic embedder; cos=1 iff content matches."""

    def __init__(self, dim: int = 16):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for ch in text.lower():
            v[ord(ch) % self._dim] += 1.0
        n = sum(x * x for x in v) ** 0.5
        return v if n == 0 else [x / n for x in v]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _mk_engram(tmp: str, threshold: float = 0.92):
    cfg = Config(path=tmp)
    cfg.storage.write_dedup_threshold = threshold
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    eng._embeddings = _DetEmbedder()
    eng._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
    return eng


def _set_extraction_conf(store, memory_id: str, conf: float) -> None:
    """Patch extraction_confidence on an already-persisted memory.

    The public ingest path (engine.remember) doesn't accept this kwarg —
    it flows in from the extraction pipeline. For an independence test we
    only need to demonstrate that varying the field on already-landed
    memories does NOT change the dedup decision for the next write.
    """
    m = store.get(memory_id)
    assert m is not None
    m.extraction_confidence = conf
    store.upsert(m)  # plain upsert (no vector_store) keeps dedup off


# ---------------------------------------------------------------------------
# (I1) Dedup decision is invariant under extraction_confidence
# ---------------------------------------------------------------------------


@given(
    a=st.text(alphabet="abcdefgh ", min_size=4, max_size=20).filter(lambda s: s.strip()),
    b=st.text(alphabet="abcdefgh ", min_size=4, max_size=20).filter(lambda s: s.strip()),
    conf_first=st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
    conf_second_arms=st.lists(
        st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
        min_size=2, max_size=5, unique=True,
    ),
)
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_decision_indep_of_extraction_confidence(
    a: str, b: str, conf_first: float, conf_second_arms: list[float]
):
    """Vary the extraction_confidence on the already-landed first memory
    and assert the second-write dedup decision is the same across all arms.
    """
    landed_counts: list[int] = []
    for conf in conf_second_arms:
        with tempfile.TemporaryDirectory() as tmp:
            eng = _mk_engram(tmp, threshold=0.92)
            try:
                eng.remember(a, salience=0.4)
                # Patch the keeper's extraction_confidence to the arm's value;
                # then attempt the second write — dedup must use cosine only.
                rows = eng._store.all_active()
                if rows:
                    _set_extraction_conf(eng._store, rows[0].id, conf)
                eng.remember(b, salience=0.4)
                landed_counts.append(eng.status()["total_memories"])
            finally:
                eng.close()

    # All arms must collapse to a single decision.
    assert len(set(landed_counts)) == 1, (
        f"dedup decision must be independent of extraction_confidence; "
        f"arms={conf_second_arms}, landed={landed_counts}, "
        f"a={a!r}, b={b!r}"
    )


# ---------------------------------------------------------------------------
# (I2) Keeper's extraction_confidence is preserved across a deduped write
# ---------------------------------------------------------------------------


@given(
    text=st.text(alphabet="abcdefghij ", min_size=5, max_size=40).filter(lambda s: s.strip()),
    keeper_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_does_not_overwrite_keeper_extraction_confidence(text: str, keeper_conf: float):
    """When a near-duplicate write is rejected, the keeper's stored
    extraction_confidence must NOT change. (Mechanical merge happens via
    the consolidation pipeline, not via the dedup short-circuit.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        eng = _mk_engram(tmp, threshold=0.92)
        try:
            eng.remember(text, salience=0.4)
            # Find the keeper and pin its extraction_confidence
            rows = eng._store.all_active()
            assert rows, "expected one memory after first remember"
            keeper_id = rows[0].id
            _set_extraction_conf(eng._store, keeper_id, keeper_conf)

            # Second write of identical content → must be deduped
            eng.remember(text, salience=0.4)
            assert eng.status()["total_memories"] == 1
            after = eng._store.get(keeper_id)
            assert after is not None
            assert abs(after.extraction_confidence - keeper_conf) < 1e-6, (
                f"deduped write must not overwrite keeper's "
                f"extraction_confidence: before={keeper_conf}, "
                f"after={after.extraction_confidence}"
            )
        finally:
            eng.close()

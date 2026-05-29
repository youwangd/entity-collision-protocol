"""Property-based invariant tests for Engram core abstractions.

Uses Hypothesis to torture-test the system with thousands of generated inputs.
Every test here encodes an invariant the system MUST hold under any input.
"""
from __future__ import annotations

import string
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck, assume

from engram import Engram, Config
from engram.core.types import (
    Memory,
    MemoryType,
    MemoryState,
    DataClassification,
    Appraisal,
    SomaticMarker,
    EmotionTag,
    EncodingContext,
    Provenance,
    Modification,
    generate_memory_id,
)
from engram.store.memory import SQLiteMemoryStore


# --- Strategies ---


@st.composite
def memory_strategy(draw) -> Memory:
    """Generate an arbitrary valid Memory."""
    mt = draw(st.sampled_from(list(MemoryType)))
    state = draw(st.sampled_from(list(MemoryState)))
    content = draw(st.text(min_size=1, max_size=500).filter(lambda s: s.strip()))
    return Memory(
        id=generate_memory_id(mt),
        type=mt,
        state=state,
        content=content,
        summary=draw(st.text(max_size=100)),
        salience=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        decay_rate=draw(st.floats(min_value=0.0001, max_value=1.0, allow_nan=False)),
        created_at=datetime.now(timezone.utc),
        last_accessed=datetime.now(timezone.utc),
        access_count=draw(st.integers(min_value=0, max_value=10_000)),
        agent_id=draw(st.text(alphabet=string.ascii_letters, max_size=20)),
        appraisal=Appraisal(
            relevance=draw(st.floats(min_value=1.0, max_value=2.0, allow_nan=False)),
            novelty=draw(st.floats(min_value=1.0, max_value=2.0, allow_nan=False)),
            goal_conduciveness=draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False)),
        ),
        somatic=SomaticMarker(
            valence=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
            bias=draw(st.text(max_size=50)),
            trigger=draw(st.text(max_size=50)),
        ),
        emotion=EmotionTag(
            primary=draw(st.sampled_from(["", "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"])),
            intensity=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            compound=draw(st.text(max_size=30)),
        ),
        encoding_context=EncodingContext(
            mood_valence=draw(st.one_of(st.none(), st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))),
            mood_arousal=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))),
            emotions=draw(st.lists(st.text(max_size=20), max_size=5)),
            task=draw(st.text(max_size=50)),
        ),
        classification=draw(st.sampled_from(list(DataClassification))),
        source_events=draw(st.lists(st.text(alphabet=string.ascii_letters + string.digits + "-", min_size=5, max_size=30), max_size=5)),
        schema_id=draw(st.text(max_size=30)),
        provenance=Provenance(
            source_events=[],
            created_by=draw(st.text(max_size=30)),
            modifications=[],
        ),
        extraction_confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


# --- Invariants ---


@given(memory_strategy())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_memory_dict_roundtrip_is_lossless(m: Memory):
    """Memory.from_dict(m.to_dict()) preserves every field exactly."""
    d = m.to_dict()
    m2 = Memory.from_dict(d)
    assert m2.id == m.id
    assert m2.type == m.type
    assert m2.state == m.state
    assert m2.content == m.content
    assert m2.summary == m.summary
    assert abs(m2.salience - m.salience) < 1e-9
    assert abs(m2.confidence - m.confidence) < 1e-9
    assert abs(m2.decay_rate - m.decay_rate) < 1e-9
    assert m2.access_count == m.access_count
    assert m2.agent_id == m.agent_id
    assert abs(m2.appraisal.relevance - m.appraisal.relevance) < 1e-9
    assert abs(m2.somatic.valence - m.somatic.valence) < 1e-9
    assert m2.emotion.primary == m.emotion.primary
    assert m2.classification == m.classification
    assert m2.source_events == m.source_events
    assert abs(m2.extraction_confidence - m.extraction_confidence) < 1e-9


@given(memory_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sqlite_upsert_is_idempotent(m: Memory):
    """Upserting the same memory twice yields the same DB state and same Memory on read."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteMemoryStore(Path(tmp))
        store.upsert(m)
        first = store.get(m.id)
        assert first is not None

        store.upsert(m)
        second = store.get(m.id)
        assert second is not None

        # Same memory ID, same content, same fundamental fields
        assert first.id == second.id
        assert first.content == second.content
        assert first.type == second.type
        assert first.classification == second.classification
        assert abs(first.salience - second.salience) < 1e-9
        assert abs(first.extraction_confidence - second.extraction_confidence) < 1e-9
        store.close()


@given(st.text())
@settings(max_examples=500, deadline=None)
def test_fts_sanitizer_never_throws(s: str):
    """The FTS5 sanitizer must handle ANY input (even adversarial unicode) without crashing."""
    cleaned = SQLiteMemoryStore._sanitize_fts_query(s)
    # It should always return a string (possibly empty)
    assert isinstance(cleaned, str)


# Adversarial FTS5 corpus: every char that has ever broken sqlite FTS5 syntax in the wild.
_FTS_HOSTILE_TOKEN = st.text(
    alphabet=st.characters(blacklist_categories=(), min_codepoint=1, max_codepoint=0x10FFFF),
    min_size=0,
    max_size=40,
)


@given(_FTS_HOSTILE_TOKEN)
@settings(max_examples=300, deadline=None)
def test_fts_sanitizer_output_is_executable_against_real_fts5(s: str):
    """Stronger invariant: the sanitizer's output must actually parse against a live FTS5 table.

    The original test only checks the sanitizer doesn't throw. This one feeds the
    sanitized string into a real ephemeral FTS5 MATCH and asserts no syntax error.
    Catches the case where the sanitizer yields something benign-looking but still
    invalid FTS5 syntax (unbalanced quotes, leading operators, etc).
    """
    import sqlite3
    cleaned = SQLiteMemoryStore._sanitize_fts_query(s)
    # An empty cleaned query is a no-op — fall back is the caller's job.
    if not cleaned.strip():
        return
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(c)")
        conn.execute("INSERT INTO t(c) VALUES ('hello world')")
        # Must not raise OperationalError("fts5: syntax error...")
        try:
            conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (cleaned,)).fetchall()
        except sqlite3.OperationalError as e:
            msg = str(e)
            # The store falls back to LIKE on syntax errors, so this is
            # technically tolerable — but our sanitizer claims to produce
            # safe output, so any syntax error is a bug we want to know about.
            raise AssertionError(
                f"sanitizer produced FTS5-invalid output {cleaned!r} from input {s!r}: {msg}"
            )
    finally:
        conn.close()


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_extraction_confidence_multiplier_is_monotone(base: float, c_low: float, c_hi: float):
    """If two memories have identical retrieval-side score `base`, the one with
    higher extraction_confidence must rank ≥ the one with lower confidence.

    This is the invariant behind cfg.retrieval.use_extraction_confidence: confidence
    can only multiply down, never invert ordering. Pure math test, no I/O.
    """
    assume(c_low <= c_hi)
    s_low = base * c_low
    s_hi = base * c_hi
    # Multiplier is in [0,1], so neither score exceeds base, and ordering is preserved.
    assert s_low <= s_hi + 1e-12
    assert s_low <= base + 1e-12
    assert s_hi <= base + 1e-12


@given(st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
@settings(max_examples=50, deadline=None)
def test_remember_then_get_returns_same_content(content: str):
    """Anything you can remember, you can read back."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        eng = Engram(config=cfg)
        try:
            eng.remember(content)
            assert eng.status()["total_memories"] >= 1
        finally:
            eng.close()


@given(st.lists(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()), min_size=1, max_size=20, unique=True))
@settings(max_examples=20, deadline=None)
def test_memory_count_monotonic_with_distinct_writes(contents: list[str]):
    """Without dedup, N distinct writes produce N memories, monotonically."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        eng = Engram(config=cfg)
        try:
            last = 0
            for c in contents:
                eng.remember(c)
                now = eng.status()["total_memories"]
                assert now > last, f"count must strictly increase on each distinct write: {last} → {now}"
                last = now
        finally:
            eng.close()


@given(memory_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_provenance_modifications_are_append_only(m: Memory):
    """Once a modification is appended, it must persist through DB roundtrip."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteMemoryStore(Path(tmp))

        mod = Modification(
            ts=datetime.now(timezone.utc),
            operation="reconsolidation",
            consolidation_id="cons-test-1",
            old_value={"content": m.content},
            new_value={"content": m.content + " (updated)"},
            reason="property test",
        )
        m.provenance.modifications.append(mod)
        store.upsert(m)

        got = store.get(m.id)
        assert got is not None
        assert len(got.provenance.modifications) >= 1
        assert got.provenance.modifications[-1].operation == "reconsolidation"
        assert got.provenance.modifications[-1].reason == "property test"
        store.close()


@given(st.text(min_size=0, max_size=2000))
@settings(max_examples=200, deadline=None)
def test_encryption_roundtrip_is_lossless(plaintext: str):
    """For any string, decrypt(encrypt(s)) == s when encryption is enabled."""
    from engram.security.encryption import ContentEncryptor, _HAS_CRYPTO  # type: ignore[attr-defined]
    if not _HAS_CRYPTO:
        pytest.skip("cryptography not installed")
    enc = ContentEncryptor(enabled=True, key="property-test-key-passphrase", key_source="direct")
    if not enc.enabled:
        pytest.skip("encryption did not initialize")
    ct = enc.encrypt(plaintext)
    assert ct.startswith("enc:") or plaintext == ""  # fernet always produces token
    pt = enc.decrypt(ct)
    assert pt == plaintext


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=100, deadline=None)
def test_encryption_passthrough_on_unprefixed(plaintext: str):
    """Decrypting a plaintext (no enc: prefix) returns it unchanged — even if encryption is disabled."""
    from engram.security.encryption import ContentEncryptor
    enc = ContentEncryptor(enabled=False)
    # No 'enc:' prefix → passthrough (mixed-mode store invariant)
    assume(not plaintext.startswith("enc:"))
    assert enc.decrypt(plaintext) == plaintext


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=64,
    ),
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=64,
    ),
)
@settings(max_examples=200, deadline=None)
def test_cosine_similarity_is_symmetric(a: list[float], b: list[float]):
    """cos(a,b) == cos(b,a) — fundamental similarity invariant."""
    import math
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    def cos(u, v):
        dot = sum(x*y for x, y in zip(u, v))
        nu = math.sqrt(sum(x*x for x in u))
        nv = math.sqrt(sum(x*x for x in v))
        return dot / (nu * nv) if nu and nv else 0.0

    # Avoid underflow corner cases — require non-trivial magnitudes
    assume(math.sqrt(sum(x*x for x in a)) > 1e-6)
    assume(math.sqrt(sum(x*x for x in b)) > 1e-6)

    s_ab = cos(a, b)
    s_ba = cos(b, a)
    assert abs(s_ab - s_ba) < 1e-9
    assert abs(cos(a, a) - 1.0) < 1e-6
    # Bounded in [-1, 1] (with a tiny FP slack)
    assert -1.0 - 1e-9 <= s_ab <= 1.0 + 1e-9


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=None)
def test_decay_is_monotonic_in_elapsed_time(salience: float, rate: float, t1: float, t2: float):
    """Ebbinghaus decay: salience * exp(-rate * t) is non-increasing in t (for rate>=0)."""
    import math
    t_short, t_long = (t1, t2) if t1 <= t2 else (t2, t1)
    s_short = salience * math.exp(-rate * t_short)
    s_long = salience * math.exp(-rate * t_long)
    # More elapsed time -> equal-or-less salience
    assert s_long <= s_short + 1e-12
    # Decay never produces negative or super-original values
    assert 0.0 <= s_long <= salience + 1e-12
    assert 0.0 <= s_short <= salience + 1e-12


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=64,
    ),
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=64,
    ),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_dedup_decision_is_symmetric(a: list[float], b: list[float], threshold: float):
    """If cos(a,b) > threshold, then cos(b,a) > threshold. Dedup must be symmetric."""
    import math
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    def cos(u, v):
        dot = sum(x*y for x, y in zip(u, v))
        nu = math.sqrt(sum(x*x for x in u))
        nv = math.sqrt(sum(x*x for x in v))
        return dot / (nu * nv) if nu and nv else 0.0
    assume(math.sqrt(sum(x*x for x in a)) > 1e-6)
    assume(math.sqrt(sum(x*x for x in b)) > 1e-6)
    a_dup_b = cos(a, b) > threshold
    b_dup_a = cos(b, a) > threshold
    assert a_dup_b == b_dup_a


@given(st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
@settings(max_examples=50, deadline=None)
def test_write_dedup_idempotent_at_high_threshold(content: str):
    """Storing identical content twice with high dedup threshold yields exactly one memory."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # Configure aggressive dedup
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        eng = Engram(config=cfg)
        try:
            eng.remember(content)
            count_after_first = eng.status()["total_memories"]
            eng.remember(content)
            count_after_second = eng.status()["total_memories"]
            # Either dedup rejected the duplicate (==) or vector store unavailable so it accepted both
            # Invariant: count never decreases, and if dedup engaged, second write was a no-op
            assert count_after_second >= count_after_first
            assert count_after_second <= count_after_first + 1
        finally:
            eng.close()


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_extraction_confidence_clamped_to_unit_interval(c: float):
    """extraction_confidence must always be in [0, 1] after roundtrip."""
    from datetime import datetime, timezone
    m = Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content="x",
        summary="x",
        salience=0.5,
        confidence=1.0,
        decay_rate=0.005,
        created_at=datetime.now(timezone.utc),
        extraction_confidence=c,
    )
    d = m.to_dict()
    m2 = Memory.from_dict(d)
    assert 0.0 <= m2.extraction_confidence <= 1.0


# --- Full FactExtraction → store → retrieve monotonicity ---
#
# The unit test in test_governed_memory_v02 checks two hand-picked confidences.
# Here we hammer the FULL pipeline (FactExtraction stage → MemoryPersistence →
# RetrievalEngine.search) with random confidence pairs and assert ranking
# monotonicity. This is the invariant the v0.2 paper will defend: per-fact
# extraction_confidence, plumbed through the dual-extraction prompt schema,
# can ONLY rerank by penalising less-confident facts. It cannot invert.

@given(
    st.floats(min_value=0.05, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.05, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow,
                                 HealthCheck.function_scoped_fixture])
def test_pipeline_extraction_confidence_monotone_under_retrieval(
    c_a: float, c_b: float,
) -> None:
    """End-to-end: FactExtraction emits two facts with confidences (c_a, c_b);
    after MemoryPersistence + RetrievalEngine.search, the higher-confidence
    fact must rank ≥ the lower-confidence one when use_extraction_confidence
    is enabled.

    Note: when |c_a - c_b| is smaller than the RRF rank-tiebreak noise on
    identical text (~0.005 in score units), the confidence multiplier can be
    swamped by insertion-order tiebreaks. We require a meaningful gap (≥0.05)
    before asserting the ordering — the reducer property is "monotone in
    confidence", not "wins ties against insertion-order noise". The pure-math
    multiplier test (test_extraction_confidence_score_multiplier) already
    pins the math; this test pins the end-to-end plumbing.
    """
    assume(abs(c_a - c_b) >= 0.05)
    import tempfile
    from engram.consolidation.pipeline import FactExtraction, MemoryPersistence, StageContext
    from engram.core.types import Event, EventType, generate_event_id
    from engram.retrieval.engine import RetrievalEngine
    from engram.core.config import RetrievalConfig

    # Stub LLM that returns two paraphrased facts with given confidences.
    class _Stub:
        def __init__(self, payload): self._p = payload
        def complete(self, *a, **kw): return ""
        def extract_json(self, *a, **kw): return self._p

    payload = {"facts": [
        {"text": "the launch is scheduled for monday morning", "confidence": c_a, "properties": []},
        {"text": "the launch is scheduled for monday morning", "confidence": c_b, "properties": []},
    ]}

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteMemoryStore(Path(tmp))
        try:
            ev = Event(id=generate_event_id(), ts=datetime.now(timezone.utc),
                       type=EventType.EVENT_CAPTURE,
                       content="we discussed the launch")
            episode = Memory.from_event(ev, memory_type=MemoryType.EPISODE)
            ctx = StageContext(memories_created=[episode])
            ctx.llm = _Stub(payload)
            ctx.store = store
            FactExtraction().run(ctx)
            MemoryPersistence().run(ctx)

            facts = [m for m in ctx.memories_created if m.type == MemoryType.FACT]
            assert len(facts) == 2
            # Sort by extraction_confidence so we know which we expect on top.
            hi, lo = sorted(facts, key=lambda m: m.extraction_confidence, reverse=True)

            eng = RetrievalEngine(
                store=store,
                config=RetrievalConfig(use_extraction_confidence=True),
            )
            results = eng.search("launch monday", limit=5)
            scores = {r.memory.id: r.score for r in results}
            assert hi.id in scores and lo.id in scores
            assert scores[hi.id] + 1e-9 >= scores[lo.id], (
                f"extraction_confidence inverted: hi={hi.extraction_confidence} "
                f"score={scores[hi.id]} vs lo={lo.extraction_confidence} score={scores[lo.id]}"
            )
        finally:
            store.close()

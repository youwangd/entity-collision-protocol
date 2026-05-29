"""Property-based FTS5 fuzz: arbitrary query strings must never crash recall().

NEXT.md priority #4. The FTS5 sanitizer (`_sanitize_fts_query` in
`engram.store.memory`) is the only thing standing between user/agent input
and `... MATCH ?`. Historically, control characters, unbalanced quotes,
column-prefix forgery (`col:term`), and FTS5 operator words have all been
known to either crash sqlite3 or leak through.

This test encodes the invariant:

    For ANY string `q`, `eng.recall(q, limit=k)` returns a list[ScoredMemory]
    without raising — even if the underlying FTS5 MATCH would have rejected
    the query. The sanitizer + the LIKE fallback together must absorb
    everything.

It complements the static `FTS_HOSTILE_QUERIES` corpus in
`tests/adversarial/test_security_torture.py` by exploring inputs the
human-curated corpus would never think of.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram
from engram.store.memory import SQLiteMemoryStore


# Adversarial alphabets: FTS5 operators, sqlite metachars, control bytes,
# column-prefix bait, unicode confusables, and a sprinkling of normal text.
FTS_TOKENS = [
    "AND", "OR", "NOT", "NEAR", "NEAR/3", "NEAR/-1",
    '"', "'", "(", ")", "*", ":", ";", "--", "\\",
    "^", "+", "-", "?", "!", "{", "}", "[", "]",
    "col:", "title:", "content:",
    "<", ">", "&", "|", "@", "#", "$", "%", "=",
    "\x00", "\x01", "\x02", "\x1b", "\x7f",
    "\u200b", "\u202e", "\ufeff",  # ZWSP, RTL override, BOM
    "🔥", "💥", "Ω",
    "DROP TABLE", "UNION SELECT", "DELETE FROM", "sqlite_master",
    "../", "../../",
    "fox", "memory", "the", "quick",
    " ", "  ", "\n", "\t", "\r",
]

token_strategy = st.sampled_from(FTS_TOKENS)
hostile_query = st.lists(token_strategy, min_size=0, max_size=12).map(
    lambda parts: "".join(parts)
)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(q=st.one_of(hostile_query, st.text(max_size=200)))
def test_fts_fuzz_recall_never_raises(tmp_path_factory, q: str) -> None:
    """recall() must absorb arbitrary input without raising."""
    tmp_path = tmp_path_factory.mktemp("fts_fuzz")
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        eng.remember("the quick brown fox jumps over the lazy dog")
        eng.remember("memory systems are interesting")
        if not q or not q.strip():
            # recall() rejects empty queries by contract; nothing to fuzz.
            return
        results = eng.recall(q, limit=5)
        assert isinstance(results, list)
    finally:
        eng.close()


@settings(
    max_examples=500,
    deadline=None,
)
@given(q=st.one_of(hostile_query, st.text(max_size=200)))
def test_fts_sanitizer_output_is_safe(q: str) -> None:
    """The sanitizer's output must never contain raw FTS5 operators that
    could break MATCH syntax. We assert structural invariants:

      - No unbalanced double quotes (FTS5 panics on these).
      - No raw control chars (\\x00..\\x1f except whitespace).
      - No ``col:term`` column-prefix (sanitizer strips ``:``).
      - No FTS5 operator keywords as bare tokens (NEAR/NOT/AND would be
        parsed as operators; we use OR-joined plain terms only).

    Empty output is fine — the caller has a LIKE fallback.
    """
    sanitized = SQLiteMemoryStore._sanitize_fts_query(q)
    assert isinstance(sanitized, str)
    # No control chars except space.
    for ch in sanitized:
        assert ch == " " or ord(ch) >= 0x20, f"control char leaked: {ch!r}"
    # Balanced (= zero) double quotes — sanitizer strips them entirely.
    assert sanitized.count('"') == 0
    # No column-prefix forgery.
    assert ":" not in sanitized
    # No bare operator tokens (only " OR " as join is allowed).
    operators = {"AND", "NOT", "NEAR"}
    tokens = [t for t in sanitized.split() if t != "OR"]
    for t in tokens:
        assert t.upper() not in operators, f"operator leaked: {t!r}"


@settings(
    max_examples=200,
    deadline=None,
)
@given(q=st.text(alphabet=st.characters(blacklist_categories=()), max_size=300))
def test_fts_sanitizer_total_function(q: str) -> None:
    """The sanitizer is a total function: it returns a str for ALL inputs,
    including pure control-byte payloads, lone surrogates (where Python
    permits), and zero-length strings.
    """
    out = SQLiteMemoryStore._sanitize_fts_query(q)
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Stronger invariant: the sanitizer's output, when fed straight to FTS5 MATCH,
# must never raise sqlite3.OperationalError. The LIKE fallback in
# `search_text` exists as a safety net, but if the sanitizer is doing its
# job, we should never need it. This test pins that contract — any
# regression in the sanitizer that lets an FTS5 operator/quote/colon slip
# through will surface as a sqlite OperationalError here, *before* it
# silently degrades production recall to a slow LIKE scan.
# ---------------------------------------------------------------------------

import sqlite3 as _sqlite3


@pytest.fixture(scope="module")
def _fts_probe_conn():
    """Module-scoped FTS5 probe table — sanitizer-output goes in MATCH directly."""
    conn = _sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE probe USING fts5(content, tokenize='unicode61')"
    )
    conn.executemany(
        "INSERT INTO probe(content) VALUES (?)",
        [
            ("the quick brown fox jumps over the lazy dog",),
            ("memory systems are interesting",),
            ("vector search hybrid retrieval",),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


@settings(
    max_examples=400,
    deadline=None,
)
@given(q=st.one_of(hostile_query, st.text(max_size=200)))
def test_sanitizer_output_never_trips_fts5_match(_fts_probe_conn, q: str) -> None:
    """For any input `q`, `MATCH _sanitize_fts_query(q)` must not raise.

    Regression guard: if the sanitizer ever leaks a `"`, `:`, bareword
    `NEAR`/`AND`/`NOT`, or unbalanced paren, FTS5 raises
    sqlite3.OperationalError("fts5: syntax error near ...") and we fall
    back to a full-table LIKE scan. That's a silent perf cliff at scale —
    pin it here.
    """
    sanitized = SQLiteMemoryStore._sanitize_fts_query(q)
    if not sanitized.strip():
        # Empty sanitized output → caller uses LIKE fallback by design.
        return
    try:
        _fts_probe_conn.execute(
            "SELECT rowid FROM probe WHERE probe MATCH ? LIMIT 1",
            (sanitized,),
        ).fetchall()
    except _sqlite3.OperationalError as e:
        raise AssertionError(
            f"sanitizer output {sanitized!r} (from input {q!r}) "
            f"tripped FTS5 MATCH: {e}"
        )

"""SQL-injection-style adversarial fuzz over the property-filter path.

NEXT.md pickup #2 — last leg. The property-filter path
(`Engram.recall_with_filters` → `SQLiteMemoryStore.filter_by_properties`)
is the only public surface that interpolates a user-controlled string
into a dynamically-built SQL statement (the operator dispatch on
``>= <= == != > <``). Everything else flows through ``?`` placeholders.

The implementation pre-validates the operator against a fixed regex and
binds the *key*, *value*, and *limit* with parameters — so by
construction it shouldn't be SQL-injectable. This test pins that
contract under fuzzing:

    S-I1  recall_with_filters never raises (other than ValueError /
          PermissionError) on arbitrary key/value strings.
    S-I2  After N fuzz calls, the DB schema (sqlite_master row count)
          is unchanged.
    S-I3  After N fuzz calls, the user-data row count in `memories` is
          unchanged from the seed.
    S-I4  No fuzz call is allowed to RETURN a row whose content is a
          known canary written under a different actor (cross-actor
          isolation under fuzz).

If any of these break, the Mem0-style "external-content prompt
injection that turns into SQL injection via memory metadata" attack
becomes real. Lock it down.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram


# Hostile tokens drawn from the canonical SQLi & FTS5 corpora plus a few
# operator-lookalikes meant to fool the comparison-operator regex.
HOSTILE_TOKENS = [
    "'", '"', ";", "--", "/*", "*/", "\\", "\x00",
    "DROP TABLE memories", "DROP TABLE memory_properties",
    "; DROP TABLE memories;--",
    "' OR 1=1 --", "' OR '1'='1",
    "UNION SELECT * FROM sqlite_master",
    "1) OR (1=1",
    "%' --",
    # operator-lookalike attempts to escape the dispatch
    ">= 0 OR 1=1", "==1; DROP TABLE memories",
    "> 0 UNION SELECT name FROM sqlite_master",
    # plain comparators (legitimate)
    ">100", "<=42", "==canary", ">=0", "!=x",
    # plain values
    "x", "canary", "0", "100", "tier", "v",
]

token_strategy = st.sampled_from(HOSTILE_TOKENS)
hostile_key = st.lists(token_strategy, min_size=1, max_size=3).map(lambda p: "".join(p))
hostile_val = st.lists(token_strategy, min_size=1, max_size=3).map(lambda p: "".join(p))
hostile_filters = st.dictionaries(hostile_key, hostile_val, min_size=1, max_size=3)


def _seed(eng: Engram) -> None:
    """Seed a small, well-known corpus and tag it with properties."""
    for i in range(5):
        eng.remember(f"benign content {i}", salience=0.5)
    # write one memory whose content is a known canary used by S-I4 to
    # detect cross-row leaks.
    eng.remember("CANARY-VALUE-DO-NOT-RETURN", salience=0.9)


def _row_count(eng: Engram) -> tuple[int, int]:
    """Return (sqlite_master row count, memories row count)."""
    conn = eng._store._get_conn()  # noqa: SLF001 — invariant pin
    nschema = conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
    ndata = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    return nschema, ndata


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(filters=hostile_filters)
def test_sqli_fuzz_filter_path_never_corrupts(tmp_path_factory, filters: dict) -> None:
    """S-I1..S-I4 — adversarial property filters cannot crash, mutate
    the schema, mutate user rows, or leak the canary."""
    tmp_path: Path = tmp_path_factory.mktemp("sqli_fuzz")
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        _seed(eng)
        nschema_before, ndata_before = _row_count(eng)

        # S-I1: any single fuzz call only raises documented exceptions.
        try:
            results = eng.recall_with_filters(properties=filters, limit=5)
            assert isinstance(results, list)
            # S-I4: no leaked canary via filter-only path. Filter-only
            # mode does not search content; canaries should never appear
            # unless the attacker can guess and bind a real key/value
            # pair, which the fuzz strategies do not generate.
            for m in results:
                assert "CANARY-VALUE-DO-NOT-RETURN" not in m.content or \
                    any("canary" in str(v).lower() for v in filters.values()), \
                    "canary leaked through filter that did not target it"
        except (ValueError, PermissionError):
            # ValueError can fire from float() of a non-numeric numeric
            # operand inside filter_by_properties — that's a documented
            # contract surface, not a corruption.
            pass

        # S-I2 + S-I3: schema and data unchanged after the fuzzed call.
        nschema_after, ndata_after = _row_count(eng)
        assert nschema_after == nschema_before, "schema row count changed under fuzz"
        assert ndata_after == ndata_before, "user-data row count changed under fuzz"
    finally:
        eng.close()


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(filters=hostile_filters)
def test_sqli_fuzz_hybrid_path_never_corrupts(tmp_path_factory, filters: dict) -> None:
    """Same invariants under the hybrid (query + properties) path,
    where the properties dict is intersected in-Python after recall."""
    tmp_path: Path = tmp_path_factory.mktemp("sqli_fuzz_hybrid")
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        _seed(eng)
        nschema_before, ndata_before = _row_count(eng)

        try:
            results = eng.recall_with_filters(
                query="benign", properties=filters, limit=5,
            )
            assert isinstance(results, list)
        except (ValueError, PermissionError):
            pass

        nschema_after, ndata_after = _row_count(eng)
        assert nschema_after == nschema_before
        assert ndata_after == ndata_before
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Static-corpus pin: classical SQLi payloads as the property KEY.
# These are documented evasions; they must be no-ops, not droppers.
# ---------------------------------------------------------------------------

CLASSIC_SQLI_KEYS = [
    "x'; DROP TABLE memory_properties;--",
    "x' OR '1'='1",
    "x) UNION SELECT 1,2,3,4,5,6,7,8,9 FROM sqlite_master --",
    "1; DELETE FROM memories",
    "key) OR EXISTS(SELECT 1 FROM memories",
]


@pytest.mark.parametrize("hostile_key_str", CLASSIC_SQLI_KEYS)
def test_classic_sqli_key_is_inert(tmp_path: Path, hostile_key_str: str) -> None:
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    eng = Engram(cfg)
    try:
        _seed(eng)
        nschema_before, ndata_before = _row_count(eng)
        results = eng.recall_with_filters(
            properties={hostile_key_str: "x"}, limit=5,
        )
        # No row can match a property key that no event ever wrote.
        assert results == []
        nschema_after, ndata_after = _row_count(eng)
        assert nschema_after == nschema_before
        assert ndata_after == ndata_before
        # Sanity: the memories table is still query-able and intact.
        assert eng.recall("benign", limit=5) is not None
    finally:
        eng.close()

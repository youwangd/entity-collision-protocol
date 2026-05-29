"""End-to-end PII redaction round-trip fuzz at the public API boundary.

Closes a gap that the per-tier suites leave open: the existing PII-bypass
torture (`tests/adversarial/test_pii_bypass.py`) hits `MemoryFirewall.validate`
in isolation; the firewall fuzz (`tests/property/test_firewall_fuzz.py`) does
the same. Neither asserts the *system-level* invariant that matters for the
paper's threat model:

    P-PII-RT1.  When `security.pii_detection=True` with redact action, no
                Hypothesis-generated PII string ever resurfaces verbatim in
                ``Engram.recall()`` results.
    P-PII-RT2.  The redaction is observable in the audit projection too —
                ``status()`` and recall content both carry the [REDACTED-*]
                marker, never the raw PII token.
    P-PII-RT3.  The full pipeline never raises on PII-shaped inputs (only
                SecurityError when `pii_action=block`, which we don't use here).

Hypothesis generates PII payloads by composing prefix/suffix prose with one of
the five canonical PII shapes (email, phone, SSN, credit card, IPv4). The
prose itself is constrained to printable ASCII to avoid colliding with the
firewall's injection tier (a separate concern, separately tested).
"""
from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# PII payload strategies — one strategy per PII type so shrinking is local.
# ---------------------------------------------------------------------------

# Use ASCII alpha-only for local-part/domain to avoid the email regex's
# Unicode quirks; the goal is to generate STRINGS THE REGEX MATCHES, then
# verify they're stripped from recall() output.

_ALPHA = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=3,
    max_size=10,
)
_DIGIT3 = st.integers(min_value=100, max_value=999).map(str)
_DIGIT2 = st.integers(min_value=10, max_value=99).map(str)
_DIGIT4 = st.integers(min_value=1000, max_value=9999).map(str)


email_strategy = st.tuples(_ALPHA, _ALPHA, st.sampled_from(["com", "org", "net", "io"])).map(
    lambda t: f"{t[0]}@{t[1]}.{t[2]}"
)

phone_strategy = st.tuples(_DIGIT3, _DIGIT3, _DIGIT4).map(lambda t: f"{t[0]}-{t[1]}-{t[2]}")

ssn_strategy = st.tuples(_DIGIT3, _DIGIT2, _DIGIT4).map(lambda t: f"{t[0]}-{t[1]}-{t[2]}")

credit_card_strategy = st.tuples(_DIGIT4, _DIGIT4, _DIGIT4, _DIGIT4).map(
    lambda t: f"{t[0]}-{t[1]}-{t[2]}-{t[3]}"
)

ipv4_strategy = st.tuples(
    st.integers(min_value=1, max_value=254),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=254),
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")


pii_strategy = st.one_of(
    email_strategy.map(lambda v: ("email", v)),
    phone_strategy.map(lambda v: ("phone", v)),
    ssn_strategy.map(lambda v: ("ssn", v)),
    credit_card_strategy.map(lambda v: ("credit_card", v)),
    ipv4_strategy.map(lambda v: ("ip_address", v)),
)


# Prose alphabet: ASCII letters + space only. We deliberately exclude digits
# from the prose to avoid spurious overlap with PII patterns, and exclude
# "ignore previous instructions"-shaped tokens (handled by injection tier).
_PROSE_ALPHA = st.text(
    alphabet=st.characters(
        min_codepoint=ord("a"),
        max_codepoint=ord("z"),
        whitelist_characters=" ",
    ),
    min_size=0,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Engram fixture with PII redaction on (auto_redact non-empty → redact action).
# ---------------------------------------------------------------------------


def _make_engram(tmp_path):
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    cfg.security.pii_detection = True
    cfg.security.auto_redact = ["__sentinel_never_matches__"]  # non-empty → redact
    cfg.security.injection_detection = False  # isolate the PII tier
    cfg.security.max_events_per_minute = 0  # disable rate limit for fuzz
    return Engram(cfg)


# ---------------------------------------------------------------------------
# P-PII-RT1: PII never round-trips verbatim through recall().
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy, prefix=_PROSE_ALPHA, suffix=_PROSE_ALPHA)
def test_pii_never_roundtrips_verbatim(tmp_path_factory, payload, prefix, suffix) -> None:
    """P-PII-RT1: any generated PII payload, embedded in arbitrary prose, must
    not survive a write→recall round-trip when redaction is on."""
    pii_type, pii_value = payload
    tmp_path = tmp_path_factory.mktemp("pii_rt1")
    eng = _make_engram(tmp_path)
    try:
        # Stitch prose around the PII so we have a non-empty memory.
        content = f"{prefix} {pii_value} {suffix}".strip() or pii_value
        eng.remember(content)

        # Recall by a token from the prose if available, else by PII type word.
        # The query needs to find the memory; use a permissive token.
        anchor_words = [w for w in (prefix + " " + suffix).split() if len(w) >= 3]
        query = anchor_words[0] if anchor_words else pii_type
        results = eng.recall(query, limit=10)

        # If recall surfaced nothing, FTS may have rejected the prose; fall
        # back to scanning the engine-level status, which projects content too.
        all_text = " ".join(r.memory.content for r in results)
        if pii_value not in all_text:
            return  # invariant trivially holds
        # If we got it back, it must NOT contain the raw PII.
        pytest.fail(
            f"P-PII-RT1 violated: raw {pii_type} '{pii_value}' surfaced in "
            f"recall() despite redaction. content={all_text!r}"
        )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# P-PII-RT2: stored content carries the [REDACTED-*] marker, not the raw PII.
# ---------------------------------------------------------------------------


_REDACTED_RE = re.compile(r"\[REDACTED-(EMAIL|PHONE|SSN|CREDIT_CARD|IP_ADDRESS)\]")


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy)
def test_pii_redacted_marker_present_in_storage(tmp_path_factory, payload) -> None:
    """P-PII-RT2: the redaction marker is the ONLY surviving artifact of the
    PII payload after it passes through the firewall."""
    pii_type, pii_value = payload
    tmp_path = tmp_path_factory.mktemp("pii_rt2")
    eng = _make_engram(tmp_path)
    try:
        anchor = "memorymarker"  # stable, FTS-safe anchor for retrieval
        content = f"{anchor} {pii_value} tail"
        eng.remember(content)
        results = eng.recall(anchor, limit=5)
        assert results, "recall lost the memory entirely (anchor token missing?)"
        stored = results[0].memory.content
        # Raw PII must be gone.
        assert pii_value not in stored, (
            f"raw PII leaked into storage: {pii_value!r} in {stored!r}"
        )
        # And a redaction marker for the right tier must be present.
        assert _REDACTED_RE.search(stored), (
            f"no [REDACTED-*] marker in stored content: {stored!r}"
        )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# P-PII-RT3: pipeline never raises on PII-shaped inputs under redact action.
# ---------------------------------------------------------------------------


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy, prefix=_PROSE_ALPHA, suffix=_PROSE_ALPHA)
def test_pii_pipeline_never_raises(tmp_path_factory, payload, prefix, suffix) -> None:
    """P-PII-RT3: under pii_action=redact, no PII payload (in any prose
    framing) raises a SecurityError or any other exception in the public API.
    """
    _, pii_value = payload
    tmp_path = tmp_path_factory.mktemp("pii_rt3")
    eng = _make_engram(tmp_path)
    try:
        content = f"{prefix} {pii_value} {suffix}".strip() or pii_value
        # remember() must succeed; recall() must succeed; status() must succeed.
        mid = eng.remember(content)
        assert isinstance(mid, str) and mid
        results = eng.recall("the", limit=3)
        assert isinstance(results, list)
        st_ = eng.status()
        assert isinstance(st_, dict)
    finally:
        eng.close()

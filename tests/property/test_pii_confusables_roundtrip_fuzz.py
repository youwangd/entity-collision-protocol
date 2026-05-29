"""End-to-end PII confusables-fold round-trip fuzz at the public API boundary.

Closes the optional gap flagged in NEXT.md: `test_pii_bypass.py` exercises
homoglyph folding at the firewall layer in isolation, but no test pins the
system-level invariant that a Cyrillic/Greek confusable inside a PII pattern
*also* survives the full ``Engram.remember() → Engram.recall()`` round trip
without leaking the raw confusable-bearing PII.

Invariants (paper §6 threat model — Unicode-confusables tier):

  P-PII-CR1.  Any PII pattern whose ASCII letters are randomly perturbed
              with a curated subset of `firewall._CONFUSABLES` (Cyrillic/
              Greek lookalikes) is redacted in storage AND in recall(),
              never resurfacing verbatim with its homoglyphs.
  P-PII-CR2.  After redaction the stored content carries the correct
              [REDACTED-*] marker for the PII tier whose regex matched.
  P-PII-CR3.  The confusables-fold roundtrip never raises; the public API
              degrades gracefully on the full Hypothesis search space.

Strategy: Hypothesis builds (a) one of the five canonical PII shapes whose
characters are independently sampled from {ASCII original, confusable
twin}, and (b) ASCII prose around it. The canonical-letter mapping below
is the *inverse* of `_CONFUSABLES` restricted to letters used in the
PII patterns (a, e, o, p, c, etc.). Single-character maps are picked at
generation time so each fuzz example produces a fresh perturbation.
"""
from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Config, Engram


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Per-letter ASCII → confusable lookup. Inverse of firewall._CONFUSABLES,
# restricted to the lowercase Latin letters that appear in the PII patterns
# we exercise here (e.g. e/x/a/m/p/l/o for email locals/domains, plus the
# uppercase analogues). Every twin lives in firewall._CONFUSABLES so the
# normaliser must fold it back to ASCII.
# ---------------------------------------------------------------------------

_CONFUSABLE_TWINS: dict[str, str] = {
    "a": "а",  # Cyrillic а (U+0430)
    "e": "е",  # Cyrillic е (U+0435)
    "o": "о",  # Cyrillic о (U+043E)
    "p": "р",  # Cyrillic р (U+0440)
    "c": "с",  # Cyrillic с (U+0441)
    "x": "х",  # Cyrillic х (U+0445)
    "i": "і",  # Cyrillic і (U+0456)
    "m": "м",  # Cyrillic м
    "k": "к",  # Cyrillic к
    "A": "А",  # Cyrillic А (U+0410)
    "E": "Е",
    "M": "М",
    "K": "К",
    "O": "О",
    "P": "Р",
    "C": "С",
    "X": "Х",
}


def _perturb_letter(letter: str, choose_twin: bool) -> str:
    """Return either the original letter or its confusable twin (if known)."""
    if choose_twin and letter in _CONFUSABLE_TWINS:
        return _CONFUSABLE_TWINS[letter]
    return letter


# Hypothesis strategy: a list of booleans, one per letter, selecting the twin
# pointwise. Length-flexible because the PII shapes have varying letter counts.
_TWIN_FLAGS = st.lists(st.booleans(), min_size=0, max_size=64)


def _apply_flags(s: str, flags: list[bool]) -> str:
    """Map each ASCII letter to either itself or its twin, indexed by flags."""
    out = []
    fi = 0
    for ch in s:
        if ch.isalpha() and ch in _CONFUSABLE_TWINS:
            flag = flags[fi % len(flags)] if flags else False
            out.append(_perturb_letter(ch, flag))
            fi += 1
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# PII payload strategies — same shapes as test_pii_roundtrip_fuzz, but with
# a confusables-flag list pointwise applied to every letter.
# ---------------------------------------------------------------------------

# Use only ASCII letters that have entries in _CONFUSABLE_TWINS (avoids
# generating letters that have no twin and so silently degrade the test).
_LETTERS_WITH_TWIN = "".join(c for c in "aeopcxim" if c in _CONFUSABLE_TWINS)

_ALPHA_TWINNABLE = st.text(
    alphabet=_LETTERS_WITH_TWIN,
    min_size=3,
    max_size=8,
)
_DIGIT3 = st.integers(min_value=100, max_value=999).map(str)
_DIGIT2 = st.integers(min_value=10, max_value=99).map(str)
_DIGIT4 = st.integers(min_value=1000, max_value=9999).map(str)


email_strategy = st.tuples(
    _ALPHA_TWINNABLE, _ALPHA_TWINNABLE, st.sampled_from(["com", "org", "net", "io"])
).map(lambda t: f"{t[0]}@{t[1]}.{t[2]}")

phone_strategy = st.tuples(_DIGIT3, _DIGIT3, _DIGIT4).map(
    lambda t: f"{t[0]}-{t[1]}-{t[2]}"
)

ssn_strategy = st.tuples(_DIGIT3, _DIGIT2, _DIGIT4).map(
    lambda t: f"{t[0]}-{t[1]}-{t[2]}"
)

credit_card_strategy = st.tuples(_DIGIT4, _DIGIT4, _DIGIT4, _DIGIT4).map(
    lambda t: f"{t[0]}-{t[1]}-{t[2]}-{t[3]}"
)


# Note: ip_address has no letters to perturb; we still include it as a control
# (must match unchanged) so the suite covers every PII tier the firewall ships.
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


# ASCII-letter prose (no digits, no PII-shaped tokens).
_PROSE_ALPHA = st.text(
    alphabet=st.characters(
        min_codepoint=ord("a"),
        max_codepoint=ord("z"),
        whitelist_characters=" ",
    ),
    min_size=0,
    max_size=20,
)


def _make_engram(tmp_path):
    cfg = Config.minimal()
    cfg.path = str(tmp_path / "engram")
    cfg.security.pii_detection = True
    cfg.security.auto_redact = ["__sentinel_never_matches__"]  # non-empty → redact
    cfg.security.injection_detection = False  # isolate the PII tier
    cfg.security.max_events_per_minute = 0
    return Engram(cfg)


_REDACTED_RE = re.compile(r"\[REDACTED-(EMAIL|PHONE|SSN|CREDIT_CARD|IP_ADDRESS)\]")


# ---------------------------------------------------------------------------
# P-PII-CR1: confusable-bearing PII never round-trips verbatim.
# ---------------------------------------------------------------------------


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy, flags=_TWIN_FLAGS, prefix=_PROSE_ALPHA, suffix=_PROSE_ALPHA)
def test_confusable_pii_never_roundtrips_verbatim(
    tmp_path_factory, payload, flags, prefix, suffix
) -> None:
    """P-PII-CR1: any PII payload, with letters randomly substituted for their
    Cyrillic/Greek confusable twins, must not survive a write→recall trip.
    """
    pii_type, ascii_value = payload
    perturbed = _apply_flags(ascii_value, flags)
    tmp_path = tmp_path_factory.mktemp("pii_cr1")
    eng = _make_engram(tmp_path)
    try:
        content = f"{prefix} {perturbed} {suffix}".strip() or perturbed
        eng.remember(content)
        anchor_words = [w for w in (prefix + " " + suffix).split() if len(w) >= 3]
        query = anchor_words[0] if anchor_words else pii_type
        results = eng.recall(query, limit=10)
        all_text = " ".join(r.memory.content for r in results)
        # The perturbed PII (with twins) must NOT appear verbatim.
        if perturbed in all_text:
            pytest.fail(
                f"P-PII-CR1 violated: confusable-bearing {pii_type} "
                f"'{perturbed}' surfaced in recall(). content={all_text!r}"
            )
        # And the original ASCII form must not appear either (the firewall
        # rewrites the redacted span, not just folds confusables in place).
        if ascii_value in all_text:
            pytest.fail(
                f"P-PII-CR1 violated: ASCII {pii_type} '{ascii_value}' "
                f"resurfaced post-fold in recall(). content={all_text!r}"
            )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# P-PII-CR2: stored content carries the correct [REDACTED-*] marker.
# ---------------------------------------------------------------------------


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy, flags=_TWIN_FLAGS)
def test_confusable_pii_redacted_marker_present(tmp_path_factory, payload, flags) -> None:
    """P-PII-CR2: the redaction marker for the matched tier appears in
    storage even when the PII pattern was hidden behind confusables.
    """
    pii_type, ascii_value = payload
    perturbed = _apply_flags(ascii_value, flags)
    tmp_path = tmp_path_factory.mktemp("pii_cr2")
    eng = _make_engram(tmp_path)
    try:
        anchor = "memorymarker"
        content = f"{anchor} {perturbed} tail"
        eng.remember(content)
        results = eng.recall(anchor, limit=5)
        assert results, "recall lost the memory entirely (anchor token missing?)"
        stored = results[0].memory.content
        assert perturbed not in stored, (
            f"confusable PII leaked into storage: {perturbed!r} in {stored!r}"
        )
        assert ascii_value not in stored, (
            f"ASCII PII leaked into storage post-fold: {ascii_value!r} in {stored!r}"
        )
        assert _REDACTED_RE.search(stored), (
            f"no [REDACTED-*] marker in stored content: {stored!r} "
            f"(perturbed PII was {perturbed!r})"
        )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# P-PII-CR3: confusable-fold round-trip never raises.
# ---------------------------------------------------------------------------


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=pii_strategy, flags=_TWIN_FLAGS, prefix=_PROSE_ALPHA, suffix=_PROSE_ALPHA)
def test_confusable_pii_pipeline_never_raises(
    tmp_path_factory, payload, flags, prefix, suffix
) -> None:
    """P-PII-CR3: under pii_action=redact, no confusable-bearing PII payload
    in any prose framing raises a SecurityError or other exception in the
    public API. Pins graceful degradation across the homoglyph search space.
    """
    _, ascii_value = payload
    perturbed = _apply_flags(ascii_value, flags)
    tmp_path = tmp_path_factory.mktemp("pii_cr3")
    eng = _make_engram(tmp_path)
    try:
        content = f"{prefix} {perturbed} {suffix}".strip() or perturbed
        mid = eng.remember(content)
        assert isinstance(mid, str) and mid
        results = eng.recall("the", limit=3)
        assert isinstance(results, list)
        st_ = eng.status()
        assert isinstance(st_, dict)
    finally:
        eng.close()

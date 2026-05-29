"""Property-based fuzz of MemoryFirewall — never-crash + monotonicity invariants.

Companion to:
- `tests/adversarial/test_security_torture.py` (curated injection / PII corpus)
- `tests/adversarial/test_pii_bypass.py` (regex-tier evasion contract)
- `tests/property/test_fts_fuzz.py` (sqlite-side never-crash)

The firewall is the *write-side* gate — every adversarial input ultimately
hits `validate()` or `scan()` first. Static corpora cannot enumerate the
combinatorial space of:

  injection-canonical phrase  ×  unicode insertions (ZWJ, BIDI, NFKC pairs)
                              ×  case mutations
                              ×  whitespace mutations
                              ×  surrounding garbage prose

Hypothesis is the right tool. Invariants encoded:

  F-I1  validate() raises ONLY SecurityError. Any other exception (TypeError,
        UnicodeError, AttributeError, re.error, RecursionError) is a bug.
  F-I2  scan() NEVER raises on any unicode input — it must always return a
        well-formed dict {pii, injection, length, classification}.
  F-I3  When injection_detection=False, NO SecurityError is ever raised due
        to an injection pattern (only size/rate/pii/policy can fire).
  F-I4  validate() is idempotent in the no-op regime: with all detections
        off and rate-limit=0, validate(x) == x for any string ≤ max length.
  F-I5  scan() is monotone wrt prefixing: if scan(s)["injection"] is True,
        then scan(prefix + s)["injection"] is also True for any short prefix
        (the trigger is content-positional, not anchored).
  F-I6  Length is always reported in code points (== len(content)), even when
        normalisation would shorten the string — scan() must reflect the
        original input length.
"""
from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.core.errors import SecurityError
from engram.security.firewall import (
    INJECTION_PATTERNS,
    FirewallConfig,
    MemoryFirewall,
)

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Canonical injection seeds — Hypothesis will mutate these.
INJECTION_SEEDS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "you are now ",  # trailing space — pattern `you\s+are\s+now\s+` requires it
    "system prompt:",
    "<system>",
    "</system>",
    "[INST]",
]

# Unicode insertions: zero-width, BIDI controls, combining marks, confusables.
UNICODE_INSERTS = [
    "",  # baseline (no insertion)
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u2060",  # word joiner
    "\ufeff",  # BOM
    "\u202e",  # RTL override
    "\u0301",  # combining acute
    "\u0307",  # combining dot above
]

WS_TOKENS = [" ", "  ", "\t", "\n", " \t ", "\u00a0"]  # incl. NBSP

GARBAGE = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),  # no surrogates
        max_codepoint=0x2FFFF,
    ),
    max_size=80,
)


def _mutate(seed: str, inserts: list[str], ws: list[str], upper_mask: list[bool]) -> str:
    """Inject unicode + whitespace + case noise into a canonical phrase."""
    out = []
    for i, ch in enumerate(seed):
        if upper_mask and upper_mask[i % len(upper_mask)]:
            ch = ch.upper()
        if inserts:
            out.append(inserts[i % len(inserts)])
        if ws and ch == " ":
            ch = ws[i % len(ws)]
        out.append(ch)
    return "".join(out)


mutated_injection = st.builds(
    _mutate,
    seed=st.sampled_from(INJECTION_SEEDS),
    inserts=st.lists(st.sampled_from(UNICODE_INSERTS), min_size=0, max_size=6),
    ws=st.lists(st.sampled_from(WS_TOKENS), min_size=0, max_size=4),
    upper_mask=st.lists(st.booleans(), min_size=0, max_size=8),
)

# Final input: mutated injection optionally wrapped in random garbage prose.
fuzzed_content = st.one_of(
    mutated_injection,
    st.builds(
        lambda pre, mid, post: pre + mid + post,
        pre=GARBAGE,
        mid=mutated_injection,
        post=GARBAGE,
    ),
    GARBAGE,  # pure random prose — no injection seed at all
)


# ---------------------------------------------------------------------------
# F-I1 / F-I2: never-crash on any input
# ---------------------------------------------------------------------------

@settings(max_examples=400, deadline=None)
@given(content=fuzzed_content)
def test_validate_only_raises_securityerror(content: str) -> None:
    """F-I1: validate() must raise SecurityError or return a string. Nothing else."""
    fw = MemoryFirewall(
        FirewallConfig(injection_detection=True, max_events_per_minute=0)
    )
    try:
        out = fw.validate(content, actor="fuzz")
        assert isinstance(out, str)
    except SecurityError:
        pass  # expected family
    # Any other exception type bubbles up and fails the test → real bug.


@settings(max_examples=400, deadline=None)
@given(content=fuzzed_content)
def test_scan_never_raises(content: str) -> None:
    """F-I2: scan() must return a well-formed dict for any unicode input."""
    fw = MemoryFirewall(FirewallConfig(injection_detection=True))
    findings = fw.scan(content)
    assert isinstance(findings, dict)
    assert set(findings.keys()) >= {"pii", "injection", "length", "classification"}
    assert isinstance(findings["pii"], dict)
    assert isinstance(findings["injection"], bool)
    assert isinstance(findings["length"], int)
    assert isinstance(findings["classification"], str)


# ---------------------------------------------------------------------------
# F-I3: injection_detection=False fully disables the injection tier
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=None)
@given(content=mutated_injection)
def test_injection_disabled_never_blocks(content: str) -> None:
    """F-I3: With injection_detection=False, injection seeds must pass."""
    fw = MemoryFirewall(
        FirewallConfig(
            injection_detection=False,
            pii_detection=False,
            max_events_per_minute=0,
            max_content_length=100_000,
        )
    )
    # Should never raise SecurityError-from-injection. Other tiers all off.
    out = fw.validate(content, actor="fuzz")
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# F-I4: full no-op idempotence
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=None)
@given(content=GARBAGE)
def test_validate_no_op_idempotent(content: str) -> None:
    """F-I4: All detections off → validate is identity on bounded input."""
    fw = MemoryFirewall(
        FirewallConfig(
            injection_detection=False,
            pii_detection=False,
            max_events_per_minute=0,
            max_content_length=10_000_000,
        )
    )
    assert fw.validate(content, actor="fuzz") == content


# ---------------------------------------------------------------------------
# F-I5: prefix monotonicity for injection detection
# ---------------------------------------------------------------------------

@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    content=mutated_injection,
    prefix=st.text(
        alphabet=string.ascii_letters + string.digits + " .,!?",
        max_size=40,
    ),
)
def test_injection_prefix_monotone(content: str, prefix: str) -> None:
    """F-I5: Prefixing benign prose must not hide an injection signal.

    If the bare content trips an injection pattern, the prefixed content
    must trip too. (None of INJECTION_PATTERNS is anchored with ^.)
    """
    fw = MemoryFirewall(FirewallConfig(injection_detection=True))
    bare = fw.scan(content)["injection"]
    if not bare:
        return  # vacuous — only assert the implication when LHS true
    prefixed = fw.scan(prefix + content)["injection"]
    assert prefixed, (
        f"prefix={prefix!r} hid injection in content={content!r}"
    )


# ---------------------------------------------------------------------------
# F-I6: scan reports raw length
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=None)
@given(content=fuzzed_content)
def test_scan_length_is_raw_codepoints(content: str) -> None:
    """F-I6: findings['length'] equals len(content) regardless of normalisation."""
    fw = MemoryFirewall(FirewallConfig())
    assert fw.scan(content)["length"] == len(content)


# ---------------------------------------------------------------------------
# Sanity: every canonical INJECTION_PATTERNS seed survives a clean trip.
# (Non-property; pins the assumption F-I5 / F-I3 lean on.)
# ---------------------------------------------------------------------------

def test_canonical_seeds_each_match_some_pattern() -> None:
    fw = MemoryFirewall(FirewallConfig(injection_detection=True))
    for seed in INJECTION_SEEDS:
        # Lower-case canonical form must be detectable by at least one
        # compiled pattern; otherwise INJECTION_SEEDS is stale.
        assert any(p.search(seed) for p in INJECTION_PATTERNS), (
            f"seed not covered by any INJECTION_PATTERNS: {seed!r}"
        )
        # And the firewall must reject the bare seed.
        with pytest.raises(SecurityError):
            fw.validate(seed, actor="fuzz")

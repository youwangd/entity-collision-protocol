"""Memory firewall — validates all writes before they reach the event store.

Checks: injection detection, PII detection, size limits, rate limiting.
"""

from __future__ import annotations

import base64
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field

from engram.core.errors import SecurityError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unicode normalisation helpers for the PII tier.
#
# Common bypass tricks (zero-width joiners, BIDI marks, combining diacritics)
# rely on the regex tier matching against a raw byte sequence that contains
# invisible code points. We normalise to NFKC and strip Unicode "format"
# (Cf) and combining-mark (Mn) categories before regex matching. This catches
# the majority of zero-width / homoglyph payloads without requiring an LLM
# classifier (paper §5.2).
# ---------------------------------------------------------------------------

# Pre-computed for hot path: zero-width and BIDI control marks we explicitly
# strip even when their unicodedata category isn't Cf (defence-in-depth).
_ZERO_WIDTH_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\ufeff\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
)

# Confusables fold — Cyrillic/Greek look-alikes that a human reads as Latin.
# Curated subset (~30 most common). Bidirectional case preserved. Keep small;
# the full Unicode confusables.txt is ~10k entries and most aren't relevant
# to PII patterns (which only care about ASCII letters + digits anyway).
# Reference: TR39 / unicode.org/Public/security/latest/confusables.txt
_CONFUSABLES = {
    # --- Cyrillic lowercase ---
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "м": "m", "т": "t",
    "к": "k", "н": "h", "в": "B",  # lowercase в visually like Latin B; rare but cheap
    # --- Cyrillic uppercase ---
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J",
    # --- Greek ---
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "ν": "v", "Α": "A", "Β": "B",
    "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
    # --- Mathematical alphanumerics not folded by NFKC bold/italic etc. are
    # already handled by NFKC. Don't duplicate. ---
}


# Pre-computed regex: base64-shaped tokens (covers email/ssn/cc payloads).
# Lower bound 12 alpha chars + up to 2 `=` pads — just below the shortest
# meaningful encoded PII payload ("MTIzLTQ1LTY3ODk=" = 15 alpha + 1 pad,
# decoding to "123-45-6789"). Cap 512 to bound work on pathological inputs.
_BASE64_TOKEN_RE = re.compile(r"[A-Za-z0-9+/]{12,512}={0,2}")


# Verbal-obfuscation rewrites: humans defeat regex by spelling tokens.
# Conservatively rewrite the bracketed/parenthesised forms only — bare " at "
# and " dot " are too risky (false positives in normal prose). This closes the
# `verbal-obfuscation-email` documented bypass.
# Examples normalised:
#   "alice [at] example [dot] com"  → "alice@example.com"
#   "alice (at) example (dot) com"  → "alice@example.com"
#   "alice {at} example {dot} com"  → "alice@example.com"
_VERBAL_AT_RE = re.compile(r"\s*[\[\(\{]\s*at\s*[\]\)\}]\s*", re.IGNORECASE)
_VERBAL_DOT_RE = re.compile(r"\s*[\[\(\{]\s*dot\s*[\]\)\}]\s*", re.IGNORECASE)


# Spelled-out-digit fold. Closes the `spelled-out-ssn` documented bypass:
#   "ssn one two three four five six seven eight nine"
#     → "ssn 123456789" → SSN regex matches.
# Conservative: only rewrites runs of ≥3 consecutive digit-words separated by
# whitespace, hyphens, or dots. A single "one" or "two" in prose is left alone.
_DIGIT_WORDS = {
    "zero": "0", "oh": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_DIGIT_WORD_RUN_RE = re.compile(
    r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
    r"(?:[\s.\-]+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)){2,}\b",
    re.IGNORECASE,
)


def _fold_digit_word_runs(text: str) -> str:
    """Replace runs of ≥3 digit-words with their concatenated digits."""
    def _sub(m: re.Match[str]) -> str:
        toks = re.split(r"[\s.\-]+", m.group(0).lower())
        return "".join(_DIGIT_WORDS[t] for t in toks if t in _DIGIT_WORDS)
    return _DIGIT_WORD_RUN_RE.sub(_sub, text)


def _normalise_for_pii(text: str) -> str:
    """Return a regex-friendly form of ``text`` for PII matching.

    - NFKC compatibility composition (folds full-width digits, ligatures).
    - Strip Unicode format chars (category Cf) — zero-width joiners, BIDI marks.
    - Strip combining marks (category Mn) — diacritics inserted between digits.
    - Fold a curated set of Cyrillic/Greek confusables to their ASCII
      look-alikes (closes homoglyph e-mail / domain bypass; paper §5.2).
    Idempotent.
    """
    if not text:
        return text
    nfkc = unicodedata.normalize("NFKC", text)
    out_chars: list[str] = []
    for ch in nfkc:
        if ch in _ZERO_WIDTH_CHARS:
            continue
        cat = unicodedata.category(ch)
        # Cf = format (zero-width, BIDI), Mn = combining mark (diacritics),
        # Me = enclosing mark (e.g. COMBINING ENCLOSING KEYCAP — closes the
        # emoji-keycap phone bypass: 4️⃣1️⃣5️⃣-555-1234 → 415-555-1234).
        if cat == "Cf" or cat == "Mn" or cat == "Me":
            continue
        # Confusables fold — applied AFTER Cf/Mn strip so a stray combining
        # mark on a Cyrillic letter still folds.
        out_chars.append(_CONFUSABLES.get(ch, ch))
    folded = "".join(out_chars)
    # Verbal-obfuscation rewrite (after confusables fold so `[аt]` works too).
    folded = _VERBAL_AT_RE.sub("@", folded)
    folded = _VERBAL_DOT_RE.sub(".", folded)
    # Spelled-out-digit fold (last so confusables/NFKC normalise `οne`→`one`).
    folded = _fold_digit_word_runs(folded)
    return folded


# Common PII patterns
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}

# Injection patterns (prompt injection attempts in memory content)
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
]


@dataclass
class FirewallConfig:
    """Configuration for the memory firewall."""
    max_content_length: int = 50_000  # chars
    max_events_per_minute: int = 100
    pii_detection: bool = False
    pii_action: str = "warn"  # "warn", "redact", "block"
    injection_detection: bool = True
    auto_redact_patterns: list[str] = field(default_factory=list)
    content_policy: dict[str, str] = field(default_factory=dict)  # classification → action: "block"|"warn"


class MemoryFirewall:
    """Validates content before it enters the memory system."""

    def __init__(self, config: FirewallConfig | None = None):
        self.config = config or FirewallConfig()
        self._event_timestamps: list[float] = []

    def validate(self, content: str, actor: str = "unknown") -> str:
        """Validate and optionally redact content. Returns cleaned content.

        Raises SecurityError if content is blocked.
        """
        # Size check
        if len(content) > self.config.max_content_length:
            raise SecurityError(
                f"Content exceeds max length ({len(content)} > {self.config.max_content_length})"
            )

        # Rate limit
        self._check_rate_limit()

        # Injection detection
        if self.config.injection_detection:
            for pattern in INJECTION_PATTERNS:
                if pattern.search(content):
                    logger.warning("injection attempt detected from %s: %s", actor, pattern.pattern)
                    raise SecurityError("Content contains potential injection pattern")

        # PII detection
        if self.config.pii_detection:
            content = self._handle_pii(content)

        # Content policy (Design §5.1: check_content_policy)
        if self.config.content_policy:
            classification = self.classify(content)
            action = self.config.content_policy.get(classification)
            if action == "block":
                raise SecurityError(
                    f"Content blocked by policy: classification '{classification}' is not allowed"
                )
            elif action == "warn":
                logger.warning("content policy warning: classification=%s actor=%s", classification, actor)

        return content

    def _check_rate_limit(self) -> None:
        """Simple sliding window rate limit. 0 disables."""
        if self.config.max_events_per_minute <= 0:
            return
        now = time.time()
        cutoff = now - 60
        self._event_timestamps = [t for t in self._event_timestamps if t > cutoff]
        if len(self._event_timestamps) >= self.config.max_events_per_minute:
            raise SecurityError(
                f"Rate limit exceeded ({self.config.max_events_per_minute}/min)"
            )
        self._event_timestamps.append(now)

    def _unmask_base64_pii(self, content: str) -> str:
        """Replace base64-encoded PII payloads with their decoded form so the
        regex tier sees them.

        We extract every base64-shaped token (≥16 chars), attempt strict
        base64 decode, require valid UTF-8, and if the decoded text contains
        any known PII pattern we substitute it inline. Conservative: a token
        that doesn't decode to valid UTF-8 with PII inside is left alone.

        This closes the documented `base64-email`, `base64-ssn`, `base64-cc`
        evasions in `tests/adversarial/test_pii_bypass.py`.
        """
        def _decode_match(m: re.Match[str]) -> str:
            tok = m.group(0)
            # Length must be a multiple of 4 (with padding) to be valid base64.
            if len(tok) % 4 != 0:
                return tok
            try:
                raw = base64.b64decode(tok, validate=True)
                decoded = raw.decode("utf-8")
            except (ValueError, UnicodeDecodeError, base64.binascii.Error):
                return tok
            # Only substitute if the decoded payload actually contains PII.
            for pat in PII_PATTERNS.values():
                if pat.search(decoded):
                    return decoded
            return tok

        return _BASE64_TOKEN_RE.sub(_decode_match, content)

    def _handle_pii(self, content: str) -> str:
        """Detect and handle PII based on config.

        Pre-normalises to NFKC and strips zero-width / format / combining
        characters before regex matching, so common bypass tricks (zero-width
        joiner inside an SSN, combining marks, BIDI overrides) cannot evade
        the regex tier. The normalised string is what gets persisted on
        redact, since the original may itself be the evasion payload.

        Base64 evasion: any standalone base64-shaped token whose decoded
        bytes are valid UTF-8 and contain a known PII pattern is treated
        as if the decoded payload were inline (block / redact / warn).
        """
        normalised = _normalise_for_pii(content)
        if normalised != content:
            content = normalised
        # Pass 1: base64-encoded payload sniff (defence against the
        # documented base64-email/base64-ssn/base64-cc evasions).
        content = self._unmask_base64_pii(content)
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(content)
            if matches:
                if self.config.pii_action == "block":
                    raise SecurityError(f"Content contains {pii_type} PII")
                elif self.config.pii_action == "redact":
                    content = pattern.sub(f"[REDACTED-{pii_type.upper()}]", content)
                    logger.info("redacted %d %s instances", len(matches), pii_type)
                else:  # warn
                    logger.warning("PII detected (%s): %d instances", pii_type, len(matches))

        # Custom redact patterns
        for pat_str in self.config.auto_redact_patterns:
            try:
                pat = re.compile(pat_str)
                content = pat.sub("[REDACTED]", content)
            except re.error:
                logger.warning("invalid redact pattern: %s", pat_str)

        return content

    def scan(self, content: str) -> dict:
        """Scan content and return findings without blocking."""
        normalised = _normalise_for_pii(content)
        normalised = self._unmask_base64_pii(normalised)
        findings: dict = {"pii": {}, "injection": False, "length": len(content), "classification": "public"}

        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(normalised)
            if matches:
                findings["pii"][pii_type] = len(matches)

        for pattern in INJECTION_PATTERNS:
            if pattern.search(content):
                findings["injection"] = True
                break

        # Auto-classify based on content (Design §5.2)
        findings["classification"] = self._auto_classify(normalised, findings["pii"])
        return findings

    def classify(self, content: str) -> str:
        """Auto-detect data classification level (Design §5.2)."""
        findings = self.scan(content)
        return findings["classification"]

    def _auto_classify(self, content: str, pii_found: dict) -> str:
        """Determine classification from content and PII findings."""
        # Restricted: API keys, secrets, passwords
        restricted_patterns = [
            re.compile(r"(sk-|api[_-]?key|secret|password|token)\s*[:=]\s*\S+", re.IGNORECASE),
            re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----"),
        ]
        for pat in restricted_patterns:
            if pat.search(content):
                return "restricted"

        # Sensitive: PII (SSN, credit card)
        if "ssn" in pii_found or "credit_card" in pii_found:
            return "sensitive"

        # Confidential: email, phone, IP
        if "email" in pii_found or "phone" in pii_found:
            return "confidential"

        # Internal: mentions of internal systems, endpoints, IPs
        if "ip_address" in pii_found:
            return "internal"

        return "public"

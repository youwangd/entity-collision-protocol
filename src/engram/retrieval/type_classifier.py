"""Cheap heuristic question-type classifier (§D15).

Designed for the LongMemEval-S taxonomy. Goal: route PRF × share_prior
on/off **per-query** based on the §D14 finding that the gain is
**type-conditional** — `single-session-user` regresses heavily while
`knowledge-update` and `single-session-preference` are directionally
positive.

Constraints:
  * No LLM calls. Pure regex + lexical features. Microseconds per call.
  * Conservative: when uncertain, return None so the caller falls back
    to the (safe) default of "no expansion".
  * Output labels match the LongMemEval taxonomy exactly so callers can
    use a simple set-membership gate.

The classifier is a small ordered cascade of patterns. Order matters:
the most distinctive types (single-session-assistant, temporal-reasoning,
single-session-preference) are matched first, falling through to the
ambiguous `single-session-user` / `knowledge-update` / `multi-session`
bucket using lexical heuristics from manual sample inspection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Public taxonomy strings. Mirrors the dataset labels exactly.
TYPE_SS_USER = "single-session-user"
TYPE_SS_PREF = "single-session-preference"
TYPE_SS_AST = "single-session-assistant"
TYPE_MULTI = "multi-session"
TYPE_TEMPORAL = "temporal-reasoning"
TYPE_KNOW_UPD = "knowledge-update"

ALL_TYPES: tuple[str, ...] = (
    TYPE_SS_USER,
    TYPE_SS_PREF,
    TYPE_SS_AST,
    TYPE_MULTI,
    TYPE_TEMPORAL,
    TYPE_KNOW_UPD,
)

# Strong cues for single-session-assistant: explicit reference to a prior
# conversation with the assistant. These phrases are highly specific.
_RX_SS_AST = re.compile(
    r"\b(remind me|previous (chat|conversation|discussion)|earlier( chat| conversation)?|"
    r"you (told|mentioned|recommended|suggested|created|wrote|made) me|"
    r"i remember you|going back to (our )?previous|looking back at (our )?previous)\b",
    re.IGNORECASE,
)

# Strong cues for temporal-reasoning: ordering, durations, "how many days/weeks",
# "first/last", "before/after", "since X when Y".
_RX_TEMPORAL = re.compile(
    r"\b("
    r"how (many|much) (days?|weeks?|months?|years?|hours?) "
    r"(ago|did|had|since|between|elapsed|passed)|"
    r"how (long|much time)|"
    r"which .* (first|last|earlier|later|before|after)|"
    r"(happened|came|came up) (first|last|before|after)|"
    r"days passed|weeks passed|months passed|years passed|"
    r"in (the )?(order|sequence)|in what order|"
    r"between (my|the) .* and (my|the)"
    r")\b",
    re.IGNORECASE,
)

# Strong cues for single-session-preference: recommendation/advice asks
# expressed as a preference. Distinctive: "recommend", "suggestions", "any tips".
_RX_PREFERENCE = re.compile(
    r"\b("
    r"(can you |could you |do you have any |any |got any |got some )"
    r"(recommend|recommendation|suggestion|suggest|tips?|advice|ideas?)|"
    r"recommend(ation)?s?\b|"
    r"any (tips|ideas|advice|thoughts|recommendations)|"
    r"do you (think|have any|have some)|"
    r"what (do you think|would you (recommend|suggest))"
    r")\b",
    re.IGNORECASE,
)

# Multi-session: aggregate / cross-session counting. Cues: "total", "in total",
# "how many ... have I" (when paired with broad scope), "across".
_RX_MULTI_AGG = re.compile(
    r"\b(total|in total|altogether|overall|combined|cumulative|across|"
    r"on average|average)\b",
    re.IGNORECASE,
)

# Knowledge-update: state-change / "currently" / "now" / "recent" patterns —
# the answer is the *latest* fact, superseding earlier ones.
_RX_KNOW_UPD = re.compile(
    r"\b("
    r"current(ly)?|right now|these days|nowadays|"
    r"most recent(ly)?|recently|latest|new(est)?|"
    r"have i (tried|been|started|stopped|switched|moved|changed)|"
    r"do i (currently|now|still)|"
    r"how (many|much|long|often) (do|have) i\b"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TypeGuess:
    """Predicted question type with a coarse confidence flag."""

    label: str | None
    confidence: float  # in [0, 1]; 0 = unknown, 1 = high


def classify_question_type(query: str) -> TypeGuess:
    """Return a heuristic type guess for `query`.

    Cascade order matches descending pattern specificity:
      1. single-session-assistant (very distinctive phrasing)
      2. temporal-reasoning       (durations / ordering)
      3. single-session-preference (recommendation asks)
      4. multi-session            (aggregate cues)
      5. knowledge-update         (currently / recently)
      6. fallback → single-session-user with low confidence
    """
    if not query or not query.strip():
        return TypeGuess(None, 0.0)

    q = query.strip()

    if _RX_SS_AST.search(q):
        return TypeGuess(TYPE_SS_AST, 0.9)
    if _RX_TEMPORAL.search(q):
        return TypeGuess(TYPE_TEMPORAL, 0.85)
    if _RX_PREFERENCE.search(q):
        return TypeGuess(TYPE_SS_PREF, 0.8)
    if _RX_MULTI_AGG.search(q):
        return TypeGuess(TYPE_MULTI, 0.7)
    if _RX_KNOW_UPD.search(q):
        return TypeGuess(TYPE_KNOW_UPD, 0.6)

    # Fallback: assume single-session-user (low confidence). This is the
    # type that REGRESSED most under PRF×SP at n=240, so it's the safe
    # default for an off-by-default gate.
    return TypeGuess(TYPE_SS_USER, 0.3)


def should_expand_for_type(
    label: str | None,
    allow_set: frozenset[str] | set[str] | None = None,
) -> bool:
    """Gate decision: True iff `label` is in the allow-set.

    `allow_set=None` → never expand (safe default; equivalent to OFF).
    """
    if not label or not allow_set:
        return False
    return label in allow_set


# Default allow-set per §D14 evidence: the two types that were
# directionally positive (knowledge-update, single-session-preference).
# Conservative: temporal-reasoning regressed; single-session-user
# regressed hard; multi-session regressed mildly; single-session-assistant
# was flat.
DEFAULT_PRF_ALLOW: frozenset[str] = frozenset({TYPE_KNOW_UPD, TYPE_SS_PREF})

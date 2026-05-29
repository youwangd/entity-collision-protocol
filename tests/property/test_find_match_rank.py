"""Property-based tests for evals.metrics.find_match_rank.

find_match_rank is the workhorse of every eval — it scans a ranked result list
and returns the 0-indexed position of the first item whose content contains
ALL expected substrings (case-insensitive). Bugs here silently corrupt every
hit@k / MRR / nDCG number we report, so we fuzz the contract:

  - returns None iff no item contains all needles
  - when a match exists, the returned index points to a matching item
  - the returned index is the *first* matching item (no earlier match exists)
  - case-insensitivity is symmetric across needle and haystack
  - empty needles list -> None (by current convention; locks the contract)
  - permutation: moving the matching item earlier never increases rank
"""
from __future__ import annotations

from dataclasses import dataclass
from hypothesis import given, settings
from hypothesis import strategies as st

from evals.metrics import find_match_rank


@dataclass
class _FakeMem:
    content: str


@dataclass
class _FakeScored:
    memory: _FakeMem


# Substrings: short alpha-only tokens to keep the search space tractable.
_token = st.text(alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
                 min_size=1, max_size=6)
_doc = st.text(alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"),
                                       whitelist_characters=" 0123456789"),
               min_size=0, max_size=40)


def _contains_all(content: str, needles: list[str]) -> bool:
    c = content.lower()
    return all(n.lower() in c for n in needles if n)


def _wrap(strs: list[str], use_scored: bool) -> list:
    if use_scored:
        return [_FakeScored(memory=_FakeMem(content=s)) for s in strs]
    return [_FakeMem(content=s) for s in strs]


@given(docs=st.lists(_doc, min_size=0, max_size=10),
       needles=st.lists(_token, min_size=0, max_size=3),
       use_scored=st.booleans())
@settings(max_examples=300, deadline=None)
def test_first_match_contract(docs: list[str], needles: list[str], use_scored: bool) -> None:
    results = _wrap(docs, use_scored)
    rank = find_match_rank(results, needles)

    # Empty needles -> None by current convention
    if not [n for n in needles if n]:
        assert rank is None
        return

    # Compute expected rank by brute force
    expected = next(
        (i for i, d in enumerate(docs) if _contains_all(d, needles)),
        None,
    )
    assert rank == expected


@given(docs=st.lists(_doc, min_size=1, max_size=8),
       needle=_token,
       use_scored=st.booleans())
@settings(max_examples=200, deadline=None)
def test_case_insensitivity(docs: list[str], needle: str, use_scored: bool) -> None:
    # Inject the needle into a known position so a match exists
    pos = len(docs) // 2
    docs = list(docs)
    docs[pos] = f"prefix {needle.upper()} suffix"
    results = _wrap(docs, use_scored)
    rank_lower = find_match_rank(results, [needle.lower()])
    rank_upper = find_match_rank(results, [needle.upper()])
    rank_mixed = find_match_rank(results, [needle.swapcase()])
    assert rank_lower is not None
    assert rank_lower == rank_upper == rank_mixed
    # And the rank points to a doc that actually contains the needle
    assert needle.lower() in docs[rank_lower].lower()


@given(docs=st.lists(_doc, min_size=2, max_size=8),
       needle=_token)
@settings(max_examples=200, deadline=None)
def test_moving_match_earlier_never_increases_rank(docs: list[str], needle: str) -> None:
    # Plant a match at the END so original rank is len(docs)-1
    docs = list(docs)
    docs[-1] = f"only-here {needle}"
    # Strip any accidental earlier occurrences. Use a sentinel that cannot
    # equal the needle (otherwise the "scrub" is a no-op and breaks the test).
    nl = needle.lower()
    # Pick a sentinel char that does NOT appear in the needle, so substitution
    # cannot accidentally re-create the needle by overlapping with adjacent
    # source chars (e.g. 'ttq'.replace('tq','qq') == 'tqq' still has 'tq').
    sentinel = next((c for c in "qzx0._-" if c not in nl), "\x01")
    for i in range(len(docs) - 1):
        d = docs[i].lower()
        # Loop until no residual occurrences (single-pass replace can leave
        # overlapping matches when sentinel*len(nl) shares chars with nl).
        while nl in d:
            d = d.replace(nl, sentinel * len(nl))
        docs[i] = d

    results = _wrap(docs, use_scored=True)
    r0 = find_match_rank(results, [needle])
    assert r0 == len(docs) - 1

    # Move the match to position 0; rank must drop to 0
    docs2 = [docs[-1]] + docs[:-1]
    r1 = find_match_rank(_wrap(docs2, use_scored=True), [needle])
    assert r1 == 0
    assert r1 <= r0


def test_known_examples() -> None:
    docs = ["Apple pie", "Banana split", "apple BANANA"]
    results = _wrap(docs, use_scored=True)
    assert find_match_rank(results, ["apple"]) == 0
    assert find_match_rank(results, ["banana"]) == 1
    # AND semantics
    assert find_match_rank(results, ["apple", "banana"]) == 2
    # No match
    assert find_match_rank(results, ["zzzz"]) is None
    # Empty needles
    assert find_match_rank(results, []) is None
    assert find_match_rank(results, [""]) is None

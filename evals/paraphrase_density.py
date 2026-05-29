"""Paraphrase-density synthetic generator.

Motivation
----------
Adaptive vector-weight is closed (binary tau and learned router both null).
The remaining open question for v0.2 is *when does vector retrieval pay?*

Hypothesis (the paraphrase-robustness reframe):
    Lexical retrievers (BM25/FTS5) and dense vector retrievers tie when
    query and memory share most non-entity content tokens, but as that
    overlap drops, lexical recall collapses while vector recall degrades
    gracefully. There should be a phase transition in
    Δhit@1(vector − bm25) as the overlap fraction T sweeps from 1.0
    (verbatim) to 0.0 (full paraphrase, entities only).

This module generates a controlled corpus where T is an explicit knob.

Design
------
Each fact has:
  - a *memory* sentence with entity placeholders + content words
  - a *query* template that asks for the answer, also using content words

We define, per tag, the **non-entity content tokens** that appear in BOTH
the memory and the query (the "shared lexical surface"), each paired with
a synonym. A target overlap ``T ∈ [0, 1]`` means: for each shared token,
keep it with probability T, otherwise replace its query occurrence with
the synonym. Entity tokens (user, service, etc.) are always kept — they
are the unavoidable answer anchors.

Why deterministic per-fact? So that the *empirical* overlap matches the
*requested* T in expectation, and so a sweep over T produces monotonically
increasing lexical-mismatch difficulty. We measure realized overlap and
report it alongside the requested T.

Usage
-----
    from evals.paraphrase_density import generate_dataset
    ds = generate_dataset(n_facts=100, overlap_target=0.3, seed=42)
    # ds.memories, ds.queries, ds.realized_overlap (mean per-query)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    expected_substrings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    realized_overlap: float = 0.0  # actual jaccard on non-entity content tokens


@dataclass
class Dataset:
    memories: list[tuple[str, dict]] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    @property
    def realized_overlap(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.realized_overlap for q in self.queries) / len(self.queries)


# Per-tag spec:
#   memory:  template with {entity} placeholders
#   query:   template with {entity} placeholders
#   synonyms: dict mapping a token in BOTH memory & query → its query-side
#             replacement when we want to break overlap. Memory text is
#             never altered — that's the corpus. Query is what we vary.
#
# We avoid touching entity tokens. Synonyms are chosen so the query stays
# grammatical and unambiguously asks the same question.
_SPECS: dict[str, dict] = {
    "preference": {
        "memory":   "User {user} prefers {pref} for {task}.",
        "query":    "what does {user} prefer for {task}?",
        # Shared content tokens: prefer (verb), for
        "synonyms": {"prefer": "favor", "prefers": "favors", "for": "during"},
    },
    "ownership": {
        "memory":   "Project {proj} is owned by team {team}.",
        "query":    "who owns project {proj}?",
        # Shared: owns/owned, project
        "synonyms": {"owns": "is responsible for", "owned": "managed",
                     "project": "initiative"},
    },
    "fact-temporal": {
        "memory":   "{user}'s API key for {service} expires on {date}.",
        "query":    "when does {user}'s {service} key expire?",
        # Shared: key, expire(s)
        "synonyms": {"key": "credential", "expire": "lapse", "expires": "lapses"},
    },
    "bug-fix": {
        "memory":   "Bug {bug_id} in {component} was fixed by commit {sha}.",
        "query":    "what fixed bug {bug_id}?",
        # Shared: fixed, bug
        "synonyms": {"fixed": "remediated", "bug": "issue"},
    },
    "bio": {
        "memory":   "{user} lives in {city} and works at {company}.",
        "query":    "where does {user} live?",
        # Shared: live(s)
        "synonyms": {"live": "reside", "lives": "resides"},
    },
    "config": {
        "memory":   "Server {host} runs version {ver} of {software}.",
        "query":    "what version of {software} runs on {host}?",
        # Shared: version, runs, of
        "synonyms": {"version": "release", "runs": "executes", "of": "for"},
    },
}

_DISTRACTORS = [
    "Daily standup: discussed sprint priorities and blockers.",
    "Reviewed PR with minor style nits, approved.",
    "Coffee break — chatted about weekend plans.",
    "Read article on distributed consensus, took notes.",
    "Lunch meeting moved from Tuesday to Thursday.",
    "Reorganized inbox, archived 200 messages.",
    "Pair programmed on feature flag refactor.",
    "Compiled quarterly status update for leadership.",
]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def _bind(rng: random.Random) -> dict[str, str]:
    users = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
    prefs = ["dark mode", "vim keybindings", "JSON logs", "cosine sim", "L2 distance"]
    tasks = ["debugging", "review", "design docs", "incident response"]
    services = ["github", "aws", "stripe", "datadog", "pagerduty"]
    projs = ["atlas", "borealis", "cipher", "delta", "echo"]
    teams = ["platform", "infra", "ml", "security", "growth"]
    components = ["auth", "billing", "search", "scheduler", "ingest"]
    cities = ["seattle", "nyc", "berlin", "tokyo", "lisbon"]
    companies = ["acme", "globex", "initech", "cyberdyne"]
    softwares = ["postgres", "redis", "nginx", "envoy"]
    versions = ["1.4.2", "2.0.1", "3.7.0", "0.9.5"]
    return {
        "user": rng.choice(users),
        "pref": rng.choice(prefs),
        "task": rng.choice(tasks),
        "service": rng.choice(services),
        "date": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "proj": rng.choice(projs),
        "team": rng.choice(teams),
        "bug_id": f"BUG-{rng.randint(1000,9999)}",
        "component": rng.choice(components),
        "sha": f"{rng.randint(0, 0xFFFFFFFF):08x}",
        "city": rng.choice(cities),
        "company": rng.choice(companies),
        "host": f"host-{rng.randint(0,99):02d}",
        "ver": rng.choice(versions),
        "software": rng.choice(softwares),
    }


def _entity_token_set(template: str, bindings: dict[str, str]) -> set[str]:
    """Return the set of lowercased tokens contributed by entity values
    in this template (so we never count them toward overlap-shaping)."""
    out: set[str] = set()
    for k, v in bindings.items():
        if "{" + k + "}" in template:
            out.update(_tokens(str(v)))
    return out


def _apply_query_swaps(
    query_filled: str,
    synonyms: dict[str, str],
    keep_prob: float,
    entity_tokens: set[str],
    rng: random.Random,
) -> str:
    """Walk the filled query token-by-token. For tokens that are swap
    candidates (in synonyms, not entity), keep with prob ``keep_prob``,
    else swap to the listed synonym. Whitespace/punct preserved."""
    pieces: list[str] = []
    last = 0
    for m in _TOKEN_RE.finditer(query_filled):
        # Append intervening punct/space verbatim
        pieces.append(query_filled[last:m.start()])
        tok = m.group(0)
        low = tok.lower()
        if low in entity_tokens:
            pieces.append(tok)
        elif low in synonyms and rng.random() >= keep_prob:
            # Swap: preserve case style of original where simple
            repl = synonyms[low]
            if tok[:1].isupper() and repl:
                repl = repl[:1].upper() + repl[1:]
            pieces.append(repl)
        else:
            pieces.append(tok)
        last = m.end()
    pieces.append(query_filled[last:])
    return "".join(pieces)


def _measure_overlap(
    memory: str, query: str, entity_tokens: set[str]
) -> float:
    """Jaccard overlap on non-entity content tokens. Stopwords stay in
    because BM25/FTS5 sees them too — this is the lexical surface that
    the retriever actually consumes."""
    m_toks = set(_tokens(memory)) - entity_tokens
    q_toks = set(_tokens(query)) - entity_tokens
    if not m_toks and not q_toks:
        return 1.0
    inter = m_toks & q_toks
    union = m_toks | q_toks
    return len(inter) / len(union) if union else 0.0


def generate_dataset(
    n_facts: int = 60,
    distractors_per_fact: int = 3,
    overlap_target: float = 0.5,
    seed: int = 42,
    tags: list[str] | None = None,
) -> Dataset:
    """Generate a paraphrase-density-controlled benchmark.

    Args:
        n_facts: number of (memory, query) pairs to plant.
        distractors_per_fact: easy-noise memories per fact (no entity overlap).
        overlap_target: T ∈ [0,1]. Probability of keeping each shared
            non-entity content token in the query (vs. swapping to synonym).
            T=1.0 → verbatim queries; T=0.0 → fully paraphrased.
        seed: RNG seed.
        tags: restrict to these tags (default: all).
    """
    if not 0.0 <= overlap_target <= 1.0:
        raise ValueError(f"overlap_target must be in [0,1], got {overlap_target}")
    rng = random.Random(seed)
    keys = tags if tags else list(_SPECS.keys())
    ds = Dataset()

    for i in range(n_facts):
        tag = keys[i % len(keys)]
        spec = _SPECS[tag]
        bindings = _bind(rng)
        mem_text = spec["memory"].format(**bindings)
        query_filled = spec["query"].format(**bindings)
        entity_toks = _entity_token_set(spec["memory"], bindings) | \
                      _entity_token_set(spec["query"], bindings)
        query_text = _apply_query_swaps(
            query_filled, spec["synonyms"], overlap_target,
            entity_toks, rng,
        )
        # Answer anchors: entity-bound tokens that appear in the memory but
        # NOT in the original query template — these are what we expect the
        # answer text to contain.
        anchors: list[str] = []
        for k, v in bindings.items():
            if "{" + k + "}" in spec["memory"] and "{" + k + "}" not in spec["query"]:
                anchors.append(str(v))
        if not anchors:
            anchors = [mem_text.split()[-1].rstrip(".")]

        realized = _measure_overlap(mem_text, query_text, entity_toks)
        ds.memories.append((mem_text, {"tag": tag, "kind": "fact", "fact_idx": i}))
        ds.queries.append(Query(
            text=query_text,
            expected_substrings=anchors,
            tags=[tag],
            realized_overlap=realized,
        ))

        for _ in range(distractors_per_fact):
            d = rng.choice(_DISTRACTORS)
            d = f"{d} (n{rng.randint(0, 99999)})"
            ds.memories.append((d, {"kind": "distractor"}))

    rng.shuffle(ds.memories)
    return ds

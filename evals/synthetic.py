"""Synthetic benchmark generator (LongMemEval / LoCoMo-style).

Generates a stream of "session" memories with planted ground-truth facts
that we later query for. Each query has a known correct memory_id (or set of ids).

Why synthetic? It's reproducible, free, and we can vary distractor density,
session length, and recency to ablate every retrieval mechanism we ship.

Usage:
    from evals.synthetic import generate_dataset
    ds = generate_dataset(n_sessions=10, distractors_per_session=20, seed=42)
    # ds.memories: list[(content, metadata)]
    # ds.queries:  list[Query(text, expected_ids, expected_substrings)]
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    expected_substrings: list[str] = field(default_factory=list)  # match-by-content fallback
    tags: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    memories: list[tuple[str, dict]] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.memories)


# Planted-fact templates: (memory_template, query_template, tag)
# Each generates a unique, recoverable fact and a query that should retrieve it.
_FACT_TEMPLATES = [
    ("User {user} prefers {pref} for {task}.",
     "what does {user} prefer for {task}?",
     "preference"),
    ("{user}'s API key for {service} expires on {date}.",
     "when does {user}'s {service} key expire?",
     "fact-temporal"),
    ("Project {proj} is owned by team {team}.",
     "who owns project {proj}?",
     "ownership"),
    ("Bug {bug_id} in {component} was fixed by commit {sha}.",
     "what fixed bug {bug_id}?",
     "bug-fix"),
    ("{user} lives in {city} and works at {company}.",
     "where does {user} live?",
     "bio"),
    ("Server {host} runs version {ver} of {software}.",
     "what version of {software} runs on {host}?",
     "config"),
]

# Paraphrased query templates — lexically minimal overlap with the memory.
# Keys must match memory templates above (by tag) so we can swap in.
_PARAPHRASE_QUERIES = {
    "preference":
        "regarding {user} and {task}, which option do they like?",
    "fact-temporal":
        "tell me the expiration date associated with {user} on {service}",
    "ownership":
        "which group is responsible for {proj}?",
    "bug-fix":
        "the patch that resolved {bug_id} — which commit was it?",
    "bio":
        "in which place does {user} reside?",
    "config":
        "{software} build identifier on {host}?",
}

# Strict-paraphrase queries — designed for ZERO non-entity token overlap
# with the memory template. Entity tokens ({user}, {service}, etc.) are
# unavoidable since they're the anchors, but every other word is replaced
# with a synonym or rephrasing. This is the hardest case for lexical retrieval
# (BM25/FTS5) and exposes the value of a semantic embedder.
#
# Memory template words → strict synonyms:
#   "prefers"  → "favors"     "expires"    → "lapses"
#   "owned"    → "belongs to" "fixed"      → "remediated"
#   "lives"    → "resides"    "runs"       → "executes"
#   "version"  → "release"    "team"       → "squad"
#   "API key"  → "credential" "commit"     → "patchset"
_STRICT_PARAPHRASE_QUERIES = {
    "preference":
        "what does {user} favor when {task}?",
    "fact-temporal":
        "lapse moment of {user}'s {service} credential?",
    "ownership":
        "what squad does {proj} belong to?",
    "bug-fix":
        "remediating patchset for {bug_id}?",
    "bio":
        "{user}'s residence?",
    "config":
        "release of {software} executing on {host}?",
}

_DISTRACTOR_TEMPLATES = [
    "Daily standup: discussed sprint priorities and blockers.",
    "Reviewed PR with minor style nits, approved.",
    "Coffee break — chatted about weekend plans.",
    "Read article on distributed consensus, took notes.",
    "Lunch meeting moved from Tuesday to Thursday.",
    "Reorganized inbox, archived 200 messages.",
    "Pair programmed on feature flag refactor.",
    "Compiled quarterly status update for leadership.",
]

# Adversarial-distractor templates: each shares entity tokens with a planted
# fact (the same {user}, {task}, {service}, etc.) but does NOT contain the
# answer anchor. This forces the retriever to discriminate semantically rather
# than relying on lexical overlap. Indexed by tag so we can plant N hard
# distractors for each fact of that tag.
_HARD_DISTRACTOR_TEMPLATES = {
    "preference": [
        "{user} mentioned {task} in passing during standup.",
        "Reviewed {user}'s PR — touched the {task} flow but no preference noted.",
        "Open question: should {user} own {task} this sprint? Deferred.",
    ],
    "fact-temporal": [
        "{user} rotated their {service} credentials last quarter; no expiry recorded.",
        "{service} dashboard is bookmarked for {user}, kept for reference.",
        "{user} filed a ticket about {service} latency — unrelated to keys.",
    ],
    "ownership": [
        "Project {proj} was discussed at the planning sync; team assignment TBD.",
        "Saw {proj} mentioned in the roadmap deck, no owner listed there.",
        "Slack thread referenced {proj} alongside three other initiatives.",
    ],
    "bug-fix": [
        "Triaged {bug_id} — reproduced locally, pending owner.",
        "{bug_id} was reopened twice last month, root cause still unclear.",
        "Customer ping referenced {bug_id} but no fix landed in that window.",
    ],
    "bio": [
        "{user} sent regards from a conference last week, location unspecified.",
        "{user}'s calendar shows travel; no home base recorded in this note.",
        "Met {user} at the offsite, didn't ask where they're based.",
    ],
    "config": [
        "Host {host} appeared in the alerting list; software stack not enumerated.",
        "Saw {software} mentioned in the runbook but not on {host} specifically.",
        "Inventory entry for {host} is stale and missing version metadata.",
    ],
}


# Cross-session paraphrase pairs — for each tag, two semantically equivalent
# statements of the same fact that share answer-relevant content. Each pair is
# planted in *different* sessions; the gold evidence set is therefore
# {session_a, session_b}. Designed for §91 cross-session-evidence corpus,
# where consolidation's schema-family gate might cluster the two paraphrases
# and (in principle) help retrieval surface both members.
_CROSS_SESSION_PAIRS = {
    "preference": [
        "User {user} prefers {pref} for {task}.",
        "When doing {task}, {user} consistently picks {pref} as their go-to.",
    ],
    "fact-temporal": [
        "{user}'s API key for {service} expires on {date}.",
        "On {date}, the {service} credential issued to {user} will lapse.",
    ],
    "ownership": [
        "Project {proj} is owned by team {team}.",
        "Team {team} carries the on-call rotation for project {proj}.",
    ],
    "bug-fix": [
        "Bug {bug_id} in {component} was fixed by commit {sha}.",
        "Commit {sha} closes out {bug_id} on the {component} subsystem.",
    ],
    "bio": [
        "{user} lives in {city} and works at {company}.",
        "{user}'s base of operations is {city}, where {company} employs them.",
    ],
    "config": [
        "Server {host} runs version {ver} of {software}.",
        "On {host}, the active {software} install is at release {ver}.",
    ],
}


def generate_cross_session_dataset(
    n_facts: int = 50,
    n_sessions: int = 10,
    distractors_per_session: int = 10,
    seed: int = 42,
) -> Dataset:
    """Generate a corpus where each gold fact lives in TWO sessions.

    For every fact, we plant two semantically-equivalent paraphrases in
    two distinct, randomly-chosen sessions. The query (canonical, not
    paraphrased) has gold evidence = {sess_a, sess_b}. This is the
    corpus shape that should let a working schema-family gate
    demonstrate operational lift: clustering the two paraphrases under
    one schema means a retriever that exploits cluster membership
    should surface both members and improve session_hit@k.

    Distractors are session-local noise (not entity-aligned). The
    haystack is non-trivial but the planted pairs are the entire
    discriminative signal.
    """
    rng = random.Random(seed)
    ds = Dataset()
    if n_sessions < 2:
        raise ValueError("cross-session corpus needs n_sessions >= 2")

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

    tag_to_query = {
        # Use the canonical (non-paraphrased) query — neither planted half
        # is a verbatim match, so retrieval has to do real work.
        "preference": "what does {user} prefer for {task}?",
        "fact-temporal": "when does {user}'s {service} key expire?",
        "ownership": "who owns project {proj}?",
        "bug-fix": "what fixed bug {bug_id}?",
        "bio": "where does {user} live?",
        "config": "what version of {software} runs on {host}?",
    }
    tags = list(_CROSS_SESSION_PAIRS.keys())

    for f_i in range(n_facts):
        tag = tags[f_i % len(tags)]
        bindings = {
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
        a_t, b_t = _CROSS_SESSION_PAIRS[tag]
        a_text = a_t.format(**bindings)
        b_text = b_t.format(**bindings)
        # Two distinct sessions for the pair
        sess_a, sess_b = rng.sample(range(n_sessions), 2)

        ds.memories.append((a_text, {
            "session": sess_a, "tag": tag, "kind": "fact",
            "pair_id": f"pair_{f_i}", "pair_half": "a",
        }))
        ds.memories.append((b_text, {
            "session": sess_b, "tag": tag, "kind": "fact",
            "pair_id": f"pair_{f_i}", "pair_half": "b",
        }))

        query_text = tag_to_query[tag].format(**bindings)
        # Answer anchors: the values that appear in the memory but not
        # in the canonical query template — these are the discriminative
        # tokens a recall-correct answer must surface.
        canonical_q = tag_to_query[tag]
        anchors = []
        for k, v in bindings.items():
            if "{" + k + "}" in (a_t + b_t) and "{" + k + "}" not in canonical_q:
                anchors.append(str(v))
        if not anchors:
            anchors = [a_text.split()[-1].rstrip(".")]

        ds.queries.append(Query(
            text=query_text,
            expected_substrings=anchors,
            tags=[tag, f"sess_a={sess_a}", f"sess_b={sess_b}",
                  f"pair_id=pair_{f_i}"],
        ))

    # Session-local distractors (not entity-aligned)
    for sess in range(n_sessions):
        for _ in range(distractors_per_session):
            d_text = rng.choice(_DISTRACTOR_TEMPLATES)
            d_text = f"{d_text} (s{sess}, n{rng.randint(0, 9999)})"
            ds.memories.append((d_text, {
                "session": sess, "kind": "distractor",
            }))

    rng.shuffle(ds.memories)
    return ds


# --- Supersede corpus (§D3-real) -----------------------------------
# Planted contradicting facts: for each slot we plant N=`updates`
# successive statements about the same entity, where every later
# statement *replaces* the earlier value. The gold answer is the LAST
# value. The memory templates are intentionally word-overlap > 0.6 so
# the heuristic interference detector (similarity > 0.6) actually
# fires and produces supersede transitions. This is the workload §D3
# was meant to probe but LoCoMo10 doesn't exercise.
_SUPERSEDE_TEMPLATES = [
    # (memory_template, query_template, value_field, tag)
    # All templates engineered for word-overlap > 0.6 across two
    # versions that differ only in {value}. Verified empirically.
    ("User {user} now prefers using {value} for daily {task} work.",
     "what does {user} prefer for {task}?",
     "value", "preference"),
    ("Project {proj} is owned by team {value}.",
     "who owns project {proj}?",
     "value", "ownership"),
    ("Server {host} runs version {value} of {software}.",
     "what version of {software} runs on {host}?",
     "value", "config"),
    ("{user} currently lives in the city of {value}.",
     "where does {user} live?",
     "value", "bio"),
]


def generate_supersede_dataset(
    n_slots: int = 50,
    updates_per_slot: int = 2,
    distractors: int = 100,
    seed: int = 42,
) -> Dataset:
    """Generate a corpus of contradicting facts that exercises supersede.

    For each slot:
      - Pick a template + entity binding.
      - Emit ``updates_per_slot`` memories where the value field changes
        each time. Templates have >0.6 word overlap by construction, so
        the interference heuristic should classify update#k as
        superseding update#k-1.
      - Emit ONE query whose gold answer is the *last* value
        (the only one that's true at query time).

    Distractors are non-entity-aligned noise (session-level chatter).
    The discriminative signal lives entirely in the planted conflicts.

    The expected_substrings of each Query is exactly the latest value,
    so a recall-correct response under default consolidation (where
    older versions are FADED) should hit. ADD-only retains every
    version active, so the retriever ranks by content similarity
    alone — which on these templates is roughly tied across versions.
    """
    rng = random.Random(seed)
    ds = Dataset()

    users = [f"user_{i:03d}" for i in range(200)]
    prefs = ["dark mode", "vim keybindings", "JSON logs", "cosine sim",
             "L2 distance", "rofi", "tmux", "fish shell"]
    tasks = ["debugging", "review", "design docs", "incident response"]
    projs = [f"proj_{i:03d}" for i in range(200)]
    teams = ["platform", "infra", "ml", "security", "growth", "core"]
    hosts = [f"host-{i:03d}" for i in range(200)]
    softwares = ["postgres", "redis", "nginx", "envoy"]
    versions = ["1.4.2", "2.0.1", "3.7.0", "0.9.5", "4.1.0", "5.2.3"]
    cities = ["seattle", "nyc", "berlin", "tokyo", "lisbon", "amsterdam"]

    pools = {
        "preference": prefs,
        "ownership": teams,
        "config": versions,
        "bio": cities,
    }

    if updates_per_slot < 2:
        raise ValueError("supersede needs >= 2 updates per slot")

    for s_i in range(n_slots):
        mem_t, query_t, _, tag = _SUPERSEDE_TEMPLATES[s_i % len(_SUPERSEDE_TEMPLATES)]
        bindings = {
            "user": rng.choice(users),
            "task": rng.choice(tasks),
            "proj": rng.choice(projs),
            "host": rng.choice(hosts),
            "software": rng.choice(softwares),
        }
        # Pick distinct values across updates so each later one truly
        # contradicts the prior.
        pool = pools[tag]
        if updates_per_slot > len(pool):
            raise ValueError(f"updates_per_slot={updates_per_slot} > pool size {len(pool)} for tag {tag}")
        values = rng.sample(pool, updates_per_slot)
        slot_id = f"slot_{s_i:04d}"

        for u_i, v in enumerate(values):
            mtext = mem_t.format(value=v, **bindings)
            ds.memories.append((mtext, {
                "kind": "fact",
                "tag": tag,
                "slot_id": slot_id,
                "update_idx": u_i,
                "value": v,
                "is_latest": (u_i == updates_per_slot - 1),
            }))

        gold_value = values[-1]
        stale_values = values[:-1]
        ds.queries.append(Query(
            text=query_t.format(**bindings),
            expected_substrings=[gold_value],
            tags=[tag, slot_id, f"stale={'|'.join(stale_values)}"],
        ))

    for _ in range(distractors):
        d_text = rng.choice(_DISTRACTOR_TEMPLATES)
        d_text = f"{d_text} (n{rng.randint(0, 99999)})"
        ds.memories.append((d_text, {"kind": "distractor"}))

    # NOTE: do NOT shuffle. Supersede only fires when the newer
    # memory enters consolidation *after* the older — with shuffled
    # ingest order the heuristic sees old.created_at > new.created_at
    # and bails. We preserve update_idx ordering; distractors will
    # bracket but won't reorder slot updates among themselves.
    return ds


def generate_dataset(
    n_sessions: int = 10,
    facts_per_session: int = 5,
    distractors_per_session: int = 20,
    seed: int = 42,
    paraphrase: bool = False,
    strict_paraphrase: bool = False,
    hard_distractors_per_fact: int = 0,
) -> Dataset:
    """Generate a reproducible synthetic memory benchmark.

    Each "session" contains:
      - facts_per_session planted facts (ground truth for queries)
      - distractors_per_session noise memories
    The query set is one query per planted fact.
    """
    rng = random.Random(seed)
    ds = Dataset()

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

    def _fill(s: str) -> str:
        return s.format(
            user=rng.choice(users),
            pref=rng.choice(prefs),
            task=rng.choice(tasks),
            service=rng.choice(services),
            date=f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            proj=rng.choice(projs),
            team=rng.choice(teams),
            bug_id=f"BUG-{rng.randint(1000,9999)}",
            component=rng.choice(components),
            sha=f"{rng.randint(0, 0xFFFFFFFF):08x}",
            city=rng.choice(cities),
            company=rng.choice(companies),
            host=f"host-{rng.randint(0,99):02d}",
            ver=rng.choice(versions),
            software=rng.choice(softwares),
        )

    # Use a deterministic substitution so memory and query reference the SAME entities
    for sess in range(n_sessions):
        for _ in range(facts_per_session):
            mem_t, query_t, tag = rng.choice(_FACT_TEMPLATES)
            # Bind variables once so memory + query agree on entities
            bindings = {
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
            mem_text = mem_t.format(**bindings)
            if strict_paraphrase and tag in _STRICT_PARAPHRASE_QUERIES:
                query_text = _STRICT_PARAPHRASE_QUERIES[tag].format(**bindings)
            elif paraphrase and tag in _PARAPHRASE_QUERIES:
                query_text = _PARAPHRASE_QUERIES[tag].format(**bindings)
            else:
                query_text = query_t.format(**bindings)

            # Distinctive substrings the answer must contain
            # Pull the values that appear in BOTH memory and query as anchors
            answer_anchors = []
            for k, v in bindings.items():
                if "{" + k + "}" in mem_t and "{" + k + "}" not in query_t:
                    answer_anchors.append(str(v))
            if not answer_anchors:
                answer_anchors = [mem_text.split()[-1].rstrip(".")]

            ds.memories.append((mem_text, {"session": sess, "tag": tag, "kind": "fact"}))
            ds.queries.append(Query(
                text=query_text,
                expected_substrings=answer_anchors,
                tags=[tag],
            ))

            # Plant hard distractors that share entity tokens with this fact
            # but don't contain the answer anchor.
            if hard_distractors_per_fact > 0 and tag in _HARD_DISTRACTOR_TEMPLATES:
                pool = _HARD_DISTRACTOR_TEMPLATES[tag]
                for _ in range(hard_distractors_per_fact):
                    hd_t = rng.choice(pool)
                    try:
                        hd_text = hd_t.format(**bindings)
                    except KeyError:
                        continue
                    # Sanity: drop any candidate that accidentally contains an
                    # answer anchor (rare, but keeps ground truth clean).
                    if any(a and a in hd_text for a in answer_anchors):
                        continue
                    ds.memories.append((hd_text, {
                        "session": sess, "tag": tag, "kind": "hard_distractor",
                    }))

        for _ in range(distractors_per_session):
            d_text = rng.choice(_DISTRACTOR_TEMPLATES)
            # Add slight perturbation so dedup doesn't collapse them
            d_text = f"{d_text} (s{sess}, n{rng.randint(0, 9999)})"
            ds.memories.append((d_text, {"session": sess, "kind": "distractor"}))

    rng.shuffle(ds.memories)
    return ds


# --- Preference-heavy synthetic corpus (§D15c corroboration) -------------
#
# Built specifically to exercise the `single-session-preference` slice that
# §D15 / §D15b found responds to PRF query expansion (Δhit@1 = +3.33 pp on
# LongMemEval, n=30). The point of THIS generator is to give us a
# **non-LongMemEval** corpus where queries trigger the type classifier's
# `_RX_PREFERENCE` regex and the answer requires surfacing a fact buried
# under entity-aligned hard distractors. If gated_pref also lifts here, we
# have independent corroboration of the v0.3 ship decision.

_PREF_MEMORY_TEMPLATES = [
    "{user} prefers {pref} when working on {task}.",
    "For {task}, {user}'s go-to choice is {pref}.",
    "{user} consistently picks {pref} for {task} sessions.",
    "When {user} does {task}, they reach for {pref} every time.",
]

# Each maps to label='single-session-preference' under classify_question_type.
_PREF_QUERY_TEMPLATES = [
    "any tips on what {user} likes for {task}?",
    "do you have any recommendations for what {user} uses for {task}?",
    "what would you suggest {user} reaches for during {task}?",
    "can you recommend what {user} prefers for {task}?",
    "got any ideas about {user}'s pick for {task}?",
]

_PREF_HARD_DISTRACTORS = [
    "{user} mentioned {task} during yesterday's standup but didn't elaborate.",
    "Reviewed {user}'s PR touching the {task} flow; no preference noted.",
    "Open thread: should {user} drive {task} this sprint? Deferred.",
    "{user} filed a ticket about {task} latency, unrelated to tooling choices.",
    "Saw {user} pair on {task} with another engineer — no decision reached.",
]


def generate_preference_dataset(
    n_facts: int = 80,
    distractors_per_fact: int = 6,
    hard_distractors_per_fact: int = 3,
    seed: int = 42,
    answer_anchor_tokens: int = 0,
) -> Dataset:
    """Preference-heavy corpus that exercises type=single-session-preference.

    Each planted fact yields one memory and one preference-phrased query.
    Hard distractors share the (user, task) tokens but lack the {pref}
    value, so a lexical-only retriever cannot trivially win. The
    `_RX_PREFERENCE` classifier maps every query to TYPE_SS_PREF, which
    makes this corpus the ideal stress test for the §D15 gated_pref arm.
    """
    rng = random.Random(seed)
    ds = Dataset()

    users = [
        "alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry",
        "iris", "jack", "kate", "liam", "mia", "noah", "olivia", "paul",
    ]
    # Original 16-pref list — preserved verbatim for back-compat with prior
    # §D15 / §D15c sweep seeds. Mixed token counts.
    prefs_legacy = [
        "dark mode", "vim keybindings", "JSON logs", "cosine sim",
        "L2 distance", "tabs over spaces", "iterative debugging",
        "pair review", "rebase over merge", "structured logging",
        "feature flags", "dependency injection", "TDD", "monorepos",
        "type hints", "async I/O",
    ]
    # Token-count-controlled pools for the §D15c-mech anchor-token sweep.
    prefs_by_tok = {
        1: [
            "TDD", "monorepos", "Vim", "Emacs", "Rust", "Python",
            "Go", "Kotlin", "PostgreSQL", "SQLite", "Redis", "Docker",
            "Kubernetes", "Terraform", "Ansible", "Bazel",
        ],
        2: [
            "dark mode", "vim keybindings", "JSON logs", "cosine sim",
            "iterative debugging", "pair review", "L2 distance",
            "structured logging", "feature flags", "dependency injection",
            "type hints", "hexagonal architecture",
            "onion architecture", "async I/O",
        ],
        3: [
            "tabs over spaces", "rebase over merge",
            "test driven development", "behaviour driven development",
            "single page applications", "domain driven design",
            "convention over configuration", "infrastructure as code",
            "continuous integration pipelines",
        ],
        5: [
            "command query responsibility segregation",
        ],
    }
    if answer_anchor_tokens <= 0:
        prefs = prefs_legacy
    else:
        prefs = prefs_by_tok.get(answer_anchor_tokens)
        if not prefs:
            raise ValueError(
                f"No prefs with exactly {answer_anchor_tokens} tokens available"
            )
    tasks = [
        "debugging", "code review", "design docs", "incident response",
        "refactoring", "load testing", "API design", "schema migrations",
        "post-mortems", "on-call rotations",
    ]

    for f_i in range(n_facts):
        bindings = {
            "user": rng.choice(users),
            "pref": rng.choice(prefs),
            "task": rng.choice(tasks),
        }
        mem_t = rng.choice(_PREF_MEMORY_TEMPLATES)
        q_t = rng.choice(_PREF_QUERY_TEMPLATES)
        mem_text = mem_t.format(**bindings)
        query_text = q_t.format(**bindings)

        anchor = bindings["pref"]
        ds.memories.append((mem_text, {
            "kind": "fact", "tag": "preference",
            "fact_id": f"pref_{f_i}",
        }))
        ds.queries.append(Query(
            text=query_text,
            expected_substrings=[anchor],
            tags=["preference", f"fact_id=pref_{f_i}"],
        ))

        for _ in range(hard_distractors_per_fact):
            hd_t = rng.choice(_PREF_HARD_DISTRACTORS)
            hd_text = hd_t.format(**bindings)
            if anchor.lower() in hd_text.lower():
                continue
            ds.memories.append((hd_text, {
                "kind": "hard_distractor", "tag": "preference",
                "fact_id": f"pref_{f_i}",
            }))

        for _ in range(distractors_per_fact):
            d_text = rng.choice(_DISTRACTOR_TEMPLATES)
            d_text = f"{d_text} (n{rng.randint(0, 99999)})"
            ds.memories.append((d_text, {
                "kind": "distractor", "tag": "preference",
            }))

    rng.shuffle(ds.memories)
    return ds

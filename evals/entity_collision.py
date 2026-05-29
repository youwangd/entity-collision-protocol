"""Entity-collision synthetic generator.

Reframe of the paraphrase-density null
--------------------------------------
The paraphrase-density sweep showed *no* phase transition on entity-rich
corpora because FTS5 retrieves on entity anchors (`{user}`, `{bug_id}`),
not content verbs. Swapping ``prefer → favor`` doesn't move BM25 because
each entity binding is unique in a small corpus.

The hypothesis under test here:

    Vector retrieval pays when **entity disambiguation** fails, not when
    content-words paraphrase. Plant K colliding facts per entity (alice
    has K different services, K different preferences, ...). The entity
    token alone now picks one of K uniformly at random in BM25.
    The *discriminator* — a non-entity content token — becomes the only
    surface signal; if we additionally paraphrase that discriminator in
    the query, BM25 has nothing left and collapses to 1/K, while a dense
    vector retains semantic similarity.

Design
------
For each entity ``u`` and a chosen tag, we plant ``K`` memories that all
share the entity but have a unique *discriminator* token from a small
vocabulary. The query asks about one specific discriminator, paraphrased
to a synonym so BM25 can no longer match on it directly.

Per-tag spec contains:
  * ``memory``: template with ``{entity}`` and ``{disc}`` (and answer).
  * ``query``:  template with ``{entity}`` and ``{disc_syn}`` only.
  * ``discs``:  list of (token, synonym, answer-distractor) triples.

Hypothesis under sweep over K ∈ {1,2,4,8}:
  - BM25 hit@1 ≈ 1/K  (purely entity-driven)
  - Vector hit@1 ≫ 1/K (semantic discriminator persists under paraphrase)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    expected_substrings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    collision_degree: int = 1


@dataclass
class Dataset:
    memories: list[tuple[str, dict]] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)


# Each spec defines:
#   * a memory template with {entity}, {disc}, {answer}
#   * a query template with {entity}, {disc_syn} only
#   * a discriminator vocabulary: list of (token, synonym, answer)
#   * memory_variants: optional list of paraphrased memory templates, all
#     of which must use the same set of placeholders. When
#     ``generate_dataset(paraphrase_memory=True)`` is set, each memory
#     samples one variant. This addresses the §6.1 threat (fixed-template
#     synthetic corpus) without changing the discriminator/answer signal.
#
# The discriminator and its synonym share NO surface-form tokens.
# The answer token is unique per memory and is what we score recall on.
_SPECS: dict[str, dict] = {
    "preference": {
        "memory": "User {entity} prefers {answer} when {disc}.",
        "query": "what does {entity} like while {disc_syn}?",
        "memory_variants": [
            "User {entity} prefers {answer} when {disc}.",
            "While {disc}, {entity} tends to reach for {answer}.",
            "{entity}'s go-to during {disc} is {answer}.",
            "When {disc}, the choice {entity} usually makes is {answer}.",
        ],
        "discs": [
            ("debugging",     "troubleshooting",       "JSON logs"),
            ("reviewing",     "auditing",              "inline comments"),
            ("designing",     "architecting",          "whiteboard sketches"),
            ("deploying",     "shipping",              "canary rollouts"),
            ("documenting",   "writing up",            "diataxis structure"),
            ("oncall",        "paged",                 "runbook links"),
            ("planning",      "roadmapping",           "OKR alignment"),
            ("interviewing",  "screening",             "system-design rubrics"),
            ("refactoring",   "restructuring",         "small reversible diffs"),
            ("benchmarking",  "profiling",             "flamegraph traces"),
            ("pairing",       "collaborating live",    "shared tmux sessions"),
            ("estimating",    "sizing tickets",        "T-shirt sizes"),
            ("retrospecting", "looking back",          "blameless format"),
            ("learning",      "studying",              "annotated notebooks"),
            ("commuting",     "travelling to work",    "podcast queues"),
            ("focusing",      "concentrating",         "noise-cancelling headphones"),
        ],
    },
    "service": {
        "memory": "{entity}'s account on {answer} is used for {disc}.",
        "query": "which provider does {entity} use to {disc_syn}?",
        "memory_variants": [
            "{entity}'s account on {answer} is used for {disc}.",
            "For {disc}, {entity} relies on their {answer} account.",
            "{entity} runs {disc} through {answer}.",
            "The provider {entity} has set up for {disc} is {answer}.",
        ],
        "discs": [
            ("payments",       "charge customers",       "stripe"),
            ("alerts",         "page on incidents",      "pagerduty"),
            ("monitoring",     "watch service health",   "datadog"),
            ("source control", "host code",              "github"),
            ("infrastructure", "run servers",            "aws"),
            ("messaging",      "broadcast updates",      "slack"),
            ("dns",            "resolve domains",        "cloudflare"),
            ("auth",           "verify identity",        "okta"),
            ("ci",             "build pipelines",        "circleci"),
            ("error tracking", "capture exceptions",     "sentry"),
            ("feature flags",  "gate rollouts",          "launchdarkly"),
            ("analytics",      "measure usage",          "amplitude"),
            ("email",          "send transactional mail","sendgrid"),
            ("storage",        "keep object blobs",      "s3"),
            ("cdn",            "cache assets globally",  "fastly"),
            ("video",          "stream meetings",        "zoom"),
        ],
    },
    "project": {
        "memory": "Project {answer} owned by {entity} targets {disc}.",
        "query": "which initiative of {entity} aims to {disc_syn}?",
        "memory_variants": [
            "Project {answer} owned by {entity} targets {disc}.",
            "{entity} leads project {answer}, which is focused on {disc}.",
            "The {disc} effort under {entity} is project {answer}.",
            "{answer}, run by {entity}, has the goal of {disc}.",
        ],
        "discs": [
            ("latency reduction",    "cut response time",        "atlas"),
            ("cost reduction",       "lower spend",              "borealis"),
            ("reliability",          "improve uptime",           "cipher"),
            ("security hardening",   "tighten access",           "delta"),
            ("developer experience", "speed up engineers",       "echo"),
            ("data quality",         "clean up records",         "foxtrot"),
            ("user growth",          "acquire customers",        "golf"),
            ("mobile parity",        "match desktop features",   "hotel"),
            ("accessibility",        "support assistive tech",   "india"),
            ("internationalization", "ship localized strings",   "juliet"),
            ("observability",        "expose service signals",   "kilo"),
            ("billing accuracy",     "fix invoice drift",        "lima"),
            ("schema migration",     "evolve table layouts",     "mike"),
            ("onboarding flow",      "guide new signups",        "november"),
            ("compliance",           "meet audit requirements",  "oscar"),
            ("offline support",      "work without network",     "papa"),
        ],
    },
    "tool": {
        # Sibling-lexical tag to `service`: proper-noun answers, action-
        # style discriminators. Tests whether the hash-K=16 lift on
        # `service` replicates on a sibling lexical-discriminator corpus.
        "memory": "{entity} uses {answer} for {disc}.",
        "query": "what does {entity} rely on to {disc_syn}?",
        "memory_variants": [
            "{entity} uses {answer} for {disc}.",
            "For {disc}, {entity} reaches for {answer}.",
            "{entity}'s tool of choice when doing {disc} is {answer}.",
            "When it comes to {disc}, {entity} sticks with {answer}.",
        ],
        "discs": [
            ("version control",      "track code revisions",       "git"),
            ("text editing",         "edit source files",          "vim"),
            ("terminal multiplexing","manage shell panes",         "tmux"),
            ("containerization",     "package services",           "docker"),
            ("orchestration",        "schedule containers",        "kubernetes"),
            ("database",             "persist relational data",    "postgres"),
            ("cache layer",          "speed up reads",             "redis"),
            ("search engine",        "index documents",            "elasticsearch"),
            ("message broker",       "pass async events",          "rabbitmq"),
            ("build automation",     "compile artifacts",          "bazel"),
            ("package management",   "install dependencies",       "poetry"),
            ("http server",          "serve web traffic",          "nginx"),
            ("reverse proxy",        "route frontend calls",       "envoy"),
            ("secret store",         "keep credentials safe",      "vault"),
            ("object store",         "hold large blobs",           "minio"),
            ("workflow engine",      "chain pipeline steps",       "airflow"),
        ],
    },
    "technical": {
        "memory": "{entity}'s service handles {disc} via {answer}.",
        "query": "what does {entity}'s service use to {disc_syn}?",
        "memory_variants": [
            "{entity}'s service handles {disc} via {answer}.",
            "For {disc}, {entity}'s service relies on {answer}.",
            "{entity}'s system implements {disc} using {answer}.",
            "Inside {entity}'s service, {disc} is implemented through {answer}.",
        ],
        "discs": [
            ("rate limiting",     "throttle traffic",            "token bucket"),
            ("caching",           "memoize hot reads",           "lru cache"),
            ("queueing",          "buffer async jobs",           "kafka topics"),
            ("sharding",          "split data horizontally",     "consistent hashing"),
            ("replication",       "duplicate writes",            "raft consensus"),
            ("compression",       "shrink payload size",         "zstd"),
            ("encryption",        "secure data at rest",         "aes-gcm"),
            ("hashing",           "fingerprint records",         "blake3"),
            ("indexing",          "accelerate lookups",          "b-tree"),
            ("deduplication",     "drop repeated entries",       "bloom filter"),
            ("scheduling",        "order pending tasks",         "priority heap"),
            ("backpressure",      "slow upstream producers",     "credit-based flow"),
            ("retry",             "recover transient errors",    "exponential backoff"),
            ("tracing",           "follow request paths",        "opentelemetry"),
            ("authentication",    "establish user identity",     "jwt tokens"),
            ("serialization",     "encode messages on the wire", "protobuf"),
        ],
    },
}

_DISTRACTORS = [
    "Daily standup: discussed sprint priorities and blockers.",
    "Reviewed PR with minor style nits, approved.",
    "Coffee break — chatted about weekend plans.",
    "Read article on distributed consensus, took notes.",
    "Lunch meeting moved from Tuesday to Thursday.",
    "Reorganized inbox, archived 200 messages.",
]

_ENTITIES = [
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry",
    "ivy", "jack", "kate", "leo", "mia", "noah", "olivia", "peter",
    "quinn", "rachel", "sam", "tina", "uma", "victor", "wendy", "xavier",
    "yara", "zach", "amelia", "bruno", "clara", "diego", "elena", "felix",
]


def generate_dataset(
    n_entities: int = 8,
    collision_degree: int = 4,
    distractors_per_entity: int = 3,
    seed: int = 42,
    tag: str = "preference",
    paraphrase_memory: bool = False,
) -> Dataset:
    """Plant a corpus where each entity has ``collision_degree`` facts
    of the same tag, distinguished only by a discriminator token whose
    synonym is what appears in the query.

    Args:
        n_entities: number of distinct entities (each gets K facts).
        collision_degree: K — facts per entity (1=no collision).
        distractors_per_entity: easy-noise memories per entity.
        seed: RNG seed.
        tag: which spec to use ("preference", "service", "project").
        paraphrase_memory: if True, sample a memory template per fact
            from ``spec["memory_variants"]`` instead of using the fixed
            ``spec["memory"]``. Addresses the §6.1 fixed-template threat.
            All variants share the same {entity}/{disc}/{answer} slots,
            so ground truth is unchanged.
    """
    if tag not in _SPECS:
        raise ValueError(f"unknown tag {tag!r}; choose from {list(_SPECS)}")
    if collision_degree < 1:
        raise ValueError(f"collision_degree must be >=1, got {collision_degree}")
    spec = _SPECS[tag]
    if collision_degree > len(spec["discs"]):
        raise ValueError(
            f"collision_degree {collision_degree} exceeds discriminator "
            f"vocab size {len(spec['discs'])} for tag {tag!r}"
        )
    if n_entities > len(_ENTITIES):
        raise ValueError(
            f"n_entities {n_entities} exceeds entity pool {len(_ENTITIES)}"
        )

    rng = random.Random(seed)
    ds = Dataset()
    variants = spec.get("memory_variants") if paraphrase_memory else None
    if paraphrase_memory and not variants:
        raise ValueError(
            f"tag {tag!r} has no memory_variants; cannot paraphrase memory"
        )

    for ei in range(n_entities):
        entity = _ENTITIES[ei]
        # Sample K discriminators for this entity (without replacement).
        discs = rng.sample(spec["discs"], collision_degree)
        for disc, disc_syn, answer in discs:
            mem_template = (
                variants[rng.randrange(len(variants))]
                if variants else spec["memory"]
            )
            mem_text = mem_template.format(
                entity=entity, disc=disc, answer=answer,
            )
            ds.memories.append((
                mem_text,
                {"tag": tag, "kind": "fact", "entity": entity, "disc": disc},
            ))
            # One query per (entity, discriminator) pair, paraphrased.
            q_text = spec["query"].format(entity=entity, disc_syn=disc_syn)
            ds.queries.append(Query(
                text=q_text,
                expected_substrings=[answer],
                tags=[tag],
                collision_degree=collision_degree,
            ))

        # Easy distractors per entity (no entity overlap).
        for _ in range(distractors_per_entity):
            d = rng.choice(_DISTRACTORS)
            d = f"{d} (n{rng.randint(0, 99999)})"
            ds.memories.append((d, {"kind": "distractor"}))

    rng.shuffle(ds.memories)
    return ds

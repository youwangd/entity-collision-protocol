"""D1 v0.3 — non-saturated multi-entity-query hard fixture.

Goal: build a corpus where BM25 alone cannot disambiguate the gold
memory from a swarm of lexical-collision distractors, but a
*type-aware* retriever (NER + entity-typed PRF / share_prior) can.


For each fact we plant ONE gold memory of the form

    "{person} works at {org}."          (PERSON × ORG)
    "{person} lives in {location}."     (PERSON × LOC)
    "{person} met {person2} in {year}." (PERSON × PERSON × YEAR)

and a configurable number of *lexical-collision* distractors that
re-use ALL the query's content tokens but bind them to the WRONG
entity type. Concretely, for a PERSON×ORG fact

    gold:        "Alice works at Apple."
    collision-1: "Alice ate an apple at the picnic."     (apple = food, not ORG)
    collision-2: "Alice owns an Apple device."           (apple = product, not employer)
    collision-3: "Alice's friend works at the orchard."  (orchard ≠ Apple, but high BM25 overlap with 'works at')

A pure BM25 retriever sees all four as competitive matches because
they share {alice, apple, works, at} or close variants. Only a
retriever that knows "Apple here is an ORG, the query asks about
employment, the collisions are FOOD/PRODUCT senses" recovers the gold.

We expose two knobs:

* lexical_collision_rate — how aggressively each distractor re-uses
  the gold's content words (0.0 = no overlap → easy; 1.0 = full
  overlap → BM25 cannot win on lexical features alone).
* ner_disambig_rate — fraction of facts where the discriminator is
  an entity TYPE (vs. a different surface form). When 1.0 the corpus
  is maximally NER-shaped; when 0.0 it degenerates into the existing
  saturated synthetic suite.

The fixture is deliberately compact (default 1k facts, ~5k memories
including distractors) so a sweep fits inside the cron's 30-min
budget.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from evals.synthetic import Dataset, Query


# -- Entity pools ------------------------------------------------------
# Each pool has a clean "canonical" sense and at least one "collision"
# sense that NER can distinguish from the canonical.

_PERSONS = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olive", "Pete",
]

# (canonical-ORG, food-or-product-collision-noun)
_ORG_COLLISIONS = [
    ("Apple", "apple"),         # company ↔ fruit
    ("Amazon", "amazon"),       # company ↔ rainforest/river
    ("Tesla", "tesla"),         # company ↔ unit / scientist
    ("Oracle", "oracle"),       # company ↔ prophecy
    ("Square", "square"),       # company ↔ shape
    ("Salesforce", "sales force"),  # company ↔ business unit
    ("Stripe", "stripe"),       # company ↔ pattern
    ("Bumble", "bumble"),       # company ↔ verb
    ("Slack", "slack"),         # company ↔ adjective
    ("Box", "box"),             # company ↔ container
]

# (canonical-LOCATION, common-noun-collision)
_LOC_COLLISIONS = [
    ("Jordan", "jordan"),       # country ↔ person name
    ("Georgia", "georgia"),     # state/country ↔ person name
    ("Brooklyn", "brooklyn"),   # borough ↔ person name
    ("Phoenix", "phoenix"),     # city ↔ mythical bird
    ("Sandwich", "sandwich"),   # town ↔ food
    ("Hamilton", "hamilton"),   # city ↔ person name
    ("Reading", "reading"),     # city ↔ activity
    ("Mobile", "mobile"),       # city ↔ adjective
    ("Bath", "bath"),           # city ↔ activity
    ("Lima", "lima"),           # city ↔ bean
]

_YEARS = [str(y) for y in range(2010, 2026)]

# -- Templates ---------------------------------------------------------

# Gold templates emit ONE memory per fact; the canonical query asks for
# the "{answer}" slot which is the entity-typed token.
_GOLD_PERSON_ORG = (
    "{person} works at {org_canon}.",
    "where does {person} work?",
    "org",
)
_GOLD_PERSON_LOC = (
    "{person} lives in {loc_canon}.",
    "where does {person} live?",
    "loc",
)

# Collision templates re-use the same surface form but in a non-ORG /
# non-LOC sense. NER (real or rule-based) sees these tokens as a
# different entity type than the gold and can downweight them.
_ORG_COLLISION_TEMPLATES = [
    "{person} ate an {org_collide} at lunch.",
    "{person} bought an {org_collide} at the market.",
    "{person} mentioned {org_collide} in passing during standup.",
    "{person} likes {org_collide} as a snack.",
    "{person} reviewed a doc about {org_collide} pricing trends.",
]
_LOC_COLLISION_TEMPLATES = [
    "{person} met {loc_collide} at the conference.",
    "{person} introduced {loc_collide} to the team.",
    "{person} mentioned {loc_collide} in the retro.",
    "{person} has a coworker named {loc_collide}.",
    "{person} saw {loc_collide} at the offsite.",
]

# Adversarial high-BM25-overlap distractors: re-use the query verbs
# ('works at', 'lives in') but with no entity match. These exist to
# raise BM25 baseline noise without giving NER an easy win.
_HIGH_OVERLAP_DISTRACTORS = [
    "{person}'s sibling works at a different place entirely.",
    "{person}'s neighbor lives in a town nobody remembers.",
    "{person} once worked at three companies in a single year.",
    "{person} has lived in many places over the past decade.",
]


# -- Generator ---------------------------------------------------------

@dataclass
class HardFixtureConfig:
    n_facts: int = 1000
    n_sessions: int = 50
    distractors_per_fact: int = 4
    high_overlap_per_fact: int = 1
    lexical_collision_rate: float = 1.0   # 1.0 = always reuse query tokens
    ner_disambig_rate: float = 1.0        # 1.0 = always type-collision
    seed: int = 42


def generate_multi_entity_hard(cfg: HardFixtureConfig | None = None) -> Dataset:
    cfg = cfg or HardFixtureConfig()
    rng = random.Random(cfg.seed)
    ds = Dataset()

    n_org = cfg.n_facts // 2
    n_loc = cfg.n_facts - n_org

    fact_id = 0

    def _pick_session() -> int:
        return rng.randrange(cfg.n_sessions)

    # PERSON × ORG facts
    for _ in range(n_org):
        person = rng.choice(_PERSONS)
        org_canon, org_collide = rng.choice(_ORG_COLLISIONS)
        gold_t, q_t, _slot = _GOLD_PERSON_ORG
        gold_text = gold_t.format(person=person, org_canon=org_canon)
        sess = _pick_session()
        ds.memories.append((gold_text, {
            "session": sess, "kind": "fact", "fact_id": f"meh_{fact_id}",
            "tag": "person_org", "answer": org_canon,
        }))

        # type-collision distractors (only fire when ner_disambig_rate hits)
        for _ in range(cfg.distractors_per_fact):
            if rng.random() < cfg.ner_disambig_rate:
                tmpl = rng.choice(_ORG_COLLISION_TEMPLATES)
                token = org_collide if rng.random() < cfg.lexical_collision_rate else "thing"
                d_text = tmpl.format(person=person, org_collide=token)
            else:
                d_text = rng.choice(_HIGH_OVERLAP_DISTRACTORS).format(person=person)
            ds.memories.append((d_text, {
                "session": _pick_session(), "kind": "distractor",
                "fact_id": f"meh_{fact_id}", "distractor_kind": "type_collision",
            }))
        for _ in range(cfg.high_overlap_per_fact):
            d_text = rng.choice(_HIGH_OVERLAP_DISTRACTORS).format(person=person)
            ds.memories.append((d_text, {
                "session": _pick_session(), "kind": "distractor",
                "fact_id": f"meh_{fact_id}", "distractor_kind": "high_overlap",
            }))

        ds.queries.append(Query(
            text=q_t.format(person=person),
            expected_substrings=[org_canon],
            tags=["person_org", f"fact_id=meh_{fact_id}"],
        ))
        fact_id += 1

    # PERSON × LOC facts
    for _ in range(n_loc):
        person = rng.choice(_PERSONS)
        loc_canon, loc_collide = rng.choice(_LOC_COLLISIONS)
        gold_t, q_t, _slot = _GOLD_PERSON_LOC
        gold_text = gold_t.format(person=person, loc_canon=loc_canon)
        sess = _pick_session()
        ds.memories.append((gold_text, {
            "session": sess, "kind": "fact", "fact_id": f"meh_{fact_id}",
            "tag": "person_loc", "answer": loc_canon,
        }))
        for _ in range(cfg.distractors_per_fact):
            if rng.random() < cfg.ner_disambig_rate:
                tmpl = rng.choice(_LOC_COLLISION_TEMPLATES)
                token = loc_collide if rng.random() < cfg.lexical_collision_rate else "someone"
                d_text = tmpl.format(person=person, loc_collide=token)
            else:
                d_text = rng.choice(_HIGH_OVERLAP_DISTRACTORS).format(person=person)
            ds.memories.append((d_text, {
                "session": _pick_session(), "kind": "distractor",
                "fact_id": f"meh_{fact_id}", "distractor_kind": "type_collision",
            }))
        for _ in range(cfg.high_overlap_per_fact):
            d_text = rng.choice(_HIGH_OVERLAP_DISTRACTORS).format(person=person)
            ds.memories.append((d_text, {
                "session": _pick_session(), "kind": "distractor",
                "fact_id": f"meh_{fact_id}", "distractor_kind": "high_overlap",
            }))

        ds.queries.append(Query(
            text=q_t.format(person=person),
            expected_substrings=[loc_canon],
            tags=["person_loc", f"fact_id=meh_{fact_id}"],
        ))
        fact_id += 1

    rng.shuffle(ds.memories)
    return ds


__all__ = ["HardFixtureConfig", "generate_multi_entity_hard"]

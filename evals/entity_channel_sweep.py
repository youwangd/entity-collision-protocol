"""D1 — entity-link channel Δrecall@k sweep.

DEPRECATION NOTE (2026-05-23, decision item from §4.13)
-------------------------------------------------------
**This fixture is hit@5-saturated and is being retired for
entity-channel recall claims.** See `paper/40_results.md §4.13`:
even on the D1-hard variant (1100 mems / 200 queries) the
lexical+embedding baseline already recovers every gold memory by
k=5, so the entity channel has zero Δrecall@k headroom to capture
across either NER backend (heuristic, spaCy `en_core_web_sm`) or
across `entity_weight ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00}`.

What D1 is still good for:
  * **Entity-weight safety regression.** Confirming that turning
    entity_weight up does not *hurt* recall on a benign fixture
    (i.e. no false collisions, no degenerate scoring). Keep the
    sweep wired into CI for this purpose only.
  * **NER-backend smoke.** Cheap end-to-end check that swapping
    `entity_ner_backend` at config time doesn't crash the engine.

What D1 is **not** good for, and what to use instead:
  * Entity-disambiguation recall claims → `evals/corpora/multi_entity_hard.py`
    (§4.9, §4.10). The hard distractors there are *type-paired*
    (other people, other projects with the same predicate), so
    BM25 cannot float the gold to top-k by entity-token alone and
    the entity channel actually has work to do.
  * Type-aware NER claims → `evals/multi_entity_hard_typed_arms.py`
    (§4.11). D1 has no type structure to exploit.

Concrete fixture-redesign checklist (if anyone revives a D1-style
single-entity sweep):
  1. **Plant ≥ K=4 type-paired colliders per gold entity** — one
     gold "Project Atlas migrated to GCP" plus three distractors
     "Project Borealis / Cipher / Delta migrated to GCP". This
     is the entity-collision generator pattern (`evals/entity_collision.py`)
     and it does produce a phase transition.
  2. **Paraphrase the discriminator in the query** — otherwise
     BM25 on the non-entity content token still picks gold.
  3. **Match `n_facts` to `len(_PEOPLE) × len(_PROJECTS)`** so
     each (person, project) tuple is unique; the current D1
     samples with replacement and accidentally creates exact
     duplicates that the dedup layer kills.
  4. **Report Δrecall@1**, not @5/@10 — k=5 ceiling is the whole
     reason D1 is retired.

For now, only the regression role is exercised in CI. Do not cite
D1 numbers as evidence of entity-channel value.

Original docstring follows.
-------------------------------------------------------

Sweeps `RetrievalConfig.entity_weight` over a grid and reports
hit@1/@5/@10 on a corpus that intentionally exercises the channel:
multi-word capitalized proper-noun entities shared between memory
and query, surrounded by lexically-similar hard distractors that
mention *different* entities.

The current heuristic NER (`engram.retrieval.entities.extract_entities`)
fires on multi-word capitalized spans + ALL-CAPS acronyms. So the
corpus uses titles like "Project Atlas", "Server NIMBUS-7", "Alice
Smith" as anchors; distractors share the surface verbs/topic but
swap in *other* capitalized entities. Lexical-only retrieval
(BM25/FTS5) confuses these; an entity-aware fusion signal should
discriminate.

Outputs
-------
- evals/results/entity_channel_sweep.json   (full grid)
- a Markdown table on stdout
- if --update-report is passed, appends to ENTITY_CHANNEL_REPORT.md

Wall budget: ~30-90 s for default corpus (n_facts=80, weights=7).
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from evals.synthetic import Dataset, Query
from evals import entity_collision as _ec
from evals.io_utils import atomic_write_json, atomic_write_text


# --- Corpus ---------------------------------------------------------

# Capitalized entity pools. Multi-word spans guarantee the heuristic
# NER fires (single-cap-token spans at sentence start are stripped).
_PEOPLE = [
    "Alice Chen", "Bob Martinez", "Carol Singh", "Dave Park",
    "Eve Thompson", "Frank Okafor", "Grace Liu", "Henry Vasquez",
    "Iris Nakamura", "Jack Holloway", "Kara Petrov", "Liam Becker",
]
_PROJECTS = [
    "Project Atlas", "Project Borealis", "Project Cipher",
    "Project Delta", "Project Echo", "Project Fortuna",
    "Project Gemini", "Project Helios", "Project Iris",
    "Project Juno", "Project Kestrel", "Project Lumen",
]
_TEAMS = [
    "Platform Core", "Infra Edge", "ML Research", "Security Ops",
    "Growth Cloud", "Data Pipelines", "Search Quality",
    "Identity Trust",
]
_CITIES = [
    "New York", "San Francisco", "Buenos Aires", "Cape Town",
    "Hong Kong", "Tel Aviv", "Sao Paulo", "Stockholm",
]
_COMPANIES = [
    "Acme Logistics", "Globex Industries", "Initech Software",
    "Cyberdyne Robotics", "Stark Engineering", "Wayne Holdings",
]


# Each fact: (memory_template, paraphrased_query_template, tag).
# Crucially: queries are PARAPHRASED — minimal non-entity word overlap
# with the memory. BM25 alone struggles because (a) the verbs differ
# and (b) hard distractors are verb-rich on the query side. The entity
# channel must do the disambiguation.
_TEMPLATES = [
    # (memory, query, tag)
    ("{person} leads the engineering effort on {project}.",
     "what initiative does {person} oversee?",
     "leadership"),
    ("{project} is owned by team {team}.",
     "which group is responsible for stewardship of {project}?",
     "ownership"),
    ("{person} relocated from {city_a} to {city_b} last quarter.",
     "to which destination did {person} move?",
     "relocation"),
    ("{company} acquired {project} in a strategic deal.",
     "which buyer purchased {project}?",
     "acquisition"),
    ("{person} joined {company} as a principal engineer.",
     "where did {person} take a new role?",
     "hiring"),
]

# Hard distractors: lexically aligned with the QUERY (so BM25 promotes
# them) but mention *different* entities. They share the same surface
# verbs/topic words used in the query template but neither the gold
# person nor the gold project. Only the entity-link channel breaks
# the tie.
_HARD = {
    "leadership": "{other_person} oversees the strategic initiative roadmap weekly.",
    "ownership": "{other_team} carries stewardship responsibility across the group.",
    "relocation": "{other_person} considered moving to a new destination, eventually staying put.",
    "acquisition": "{other_company} is the buyer in a separate purchase transaction.",
    "hiring": "{other_person} took a new role at a different firm last cycle.",
}


def generate_entity_corpus(
    n_facts: int = 80,
    hard_distractors_per_fact: int = 2,
    plain_distractors: int = 50,
    seed: int = 42,
) -> Dataset:
    """Build a paraphrase-query corpus.

    Each fact gets a freshly-minted Person / Project / Company so the
    gold entity for a query is unambiguous (no other planted fact
    mentions that same entity). Hard distractors swap in *other*
    entities and align lexically with the QUERY template — they share
    surface verbs ('oversees', 'stewardship', 'destination', 'buyer',
    'new role') with the query template but mention different entities.

    Reading the channel: an ideal entity-aware retriever should beat
    a lexical-only retriever here, since the gold memory uniquely
    contains the query's entity while hard distractors contain
    semantically-aligned but entity-mismatched content.
    """
    rng = random.Random(seed)
    ds = Dataset()

    # Generate enough unique entities; we mint them deterministically
    # so the corpus is reproducible across runs.
    def _person(i: int) -> str:
        first = ["Aria", "Brennan", "Cyra", "Devon", "Elara", "Finn",
                "Gita", "Hugo", "Ines", "Jonas", "Kavi", "Lior",
                "Mei", "Nuri", "Olin", "Petra", "Quinn", "Rasmus",
                "Saoirse", "Tariq", "Una", "Viggo", "Wren", "Xiulan",
                "Yotam", "Zara"]
            # 26 first names
        last = ["Alvarez", "Bjornsen", "Cattaneo", "Devarakonda",
                "Eriksson", "Fontaine", "Goyal", "Hartwell", "Iniesta",
                "Jovanovic", "Kowalski", "Lindqvist", "Marchetti",
                "Nagasawa", "Olufsen", "Pereira", "Quirke", "Rasmussen",
                "Stoltzfus", "Takahashi", "Underhill", "Vasconcelos",
                "Worthington", "Xanthopoulos", "Ymir", "Zelenka"]
        return f"{first[i % len(first)]} {last[(i // len(first)) % len(last)]}"

    def _project(i: int) -> str:
        codenames = ["Atlas", "Borealis", "Cipher", "Delta", "Echo",
                     "Fortuna", "Gemini", "Helios", "Iris", "Juno",
                     "Kestrel", "Lumen", "Mimir", "Nova", "Orion",
                     "Pyxis", "Quasar", "Rigel", "Sirius", "Triton",
                     "Umbra", "Vega", "Wraith", "Xerus", "Ymir",
                     "Zephyr", "Aegis", "Beacon", "Castor", "Drake"]
        return f"Project {codenames[i % len(codenames)]}"

    def _company(i: int) -> str:
        names = ["Acme Logistics", "Globex Industries", "Initech Software",
                 "Cyberdyne Robotics", "Stark Engineering", "Wayne Holdings",
                 "Soylent Foods", "Tyrell Genetics", "Umbrella Sciences",
                 "Veridian Dynamics", "Weyland Corp", "Xanadu Media"]
        return names[i % len(names)]

    def _team(i: int) -> str:
        names = ["Platform Core", "Infra Edge", "ML Research", "Security Ops",
                 "Growth Cloud", "Data Pipelines", "Search Quality",
                 "Identity Trust", "Reliability Wing", "Storage North",
                 "Storage South", "Network Pods"]
        return names[i % len(names)]

    def _city(i: int) -> str:
        return _CITIES[i % len(_CITIES)]

    for f_i in range(n_facts):
        mem_t, q_t, tag = _TEMPLATES[f_i % len(_TEMPLATES)]
        bindings = {
            "person": _person(f_i),
            "project": _project(f_i),
            "team": _team(f_i),
            "city_a": _city(f_i),
            "city_b": _city(f_i + 3),
            "company": _company(f_i),
        }

        mem_text = mem_t.format(**bindings)
        q_text = q_t.format(**bindings)

        anchors = []
        for k, v in bindings.items():
            if "{" + k + "}" in mem_t and "{" + k + "}" not in q_t:
                anchors.append(str(v))
        if not anchors:
            anchors = [mem_text.rstrip(".").split()[-1]]

        ds.memories.append((mem_text, {
            "tag": tag, "kind": "fact", "fact_id": f"fact_{f_i:04d}",
        }))
        ds.queries.append(Query(
            text=q_text,
            expected_substrings=anchors,
            tags=[tag, f"fact_{f_i:04d}"],
        ))

        hard_t = _HARD[tag]
        for d_i in range(hard_distractors_per_fact):
            other = {
                "other_person": _person(f_i + 100 + d_i * 7),
                "other_project": _project(f_i + 100 + d_i * 7),
                "other_team": _team(f_i + 100 + d_i * 7),
                "other_company": _company(f_i + 100 + d_i * 7),
            }
            try:
                hd_text = hard_t.format(**other)
            except KeyError:
                continue
            if any(a and a in hd_text for a in anchors):
                continue
            ds.memories.append((hd_text, {
                "tag": tag, "kind": "hard_distractor",
            }))

    for i in range(plain_distractors):
        ds.memories.append((
            f"Routine note #{i}: standup discussed sprint priorities and blockers.",
            {"kind": "distractor"},
        ))

    rng.shuffle(ds.memories)
    return ds


# --- Sweep ----------------------------------------------------------

def _build_engine(path: str, entity_weight: float, entity_ner: str = "heuristic",
                  embedder=None) -> Engram:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    # Replace retrieval config preserving defaults except entity_weight + entity_ner.
    cfg.retrieval = RetrievalConfig(entity_weight=entity_weight, entity_ner=entity_ner)
    if embedder is not None:
        return Engram(config=cfg, embeddings=embedder)
    return Engram(config=cfg)


def _value_in(text: str, value: str) -> bool:
    return value.lower() in (text or "").lower()


def _eval_arm(ds: Dataset, entity_weight: float, k_max: int = 10, entity_ner: str = "heuristic",
              embedder=None) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp, entity_weight, entity_ner=entity_ner, embedder=embedder)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                gold = q.expected_substrings[0]
                hit_at_1 = bool(texts) and _value_in(texts[0], gold)
                hit_at_5 = any(_value_in(t, gold) for t in texts[:5])
                hit_at_10 = any(_value_in(t, gold) for t in texts[:10])
                rows.append({
                    "tag": q.tags[0] if q.tags else "",
                    "fact_id": next((t for t in q.tags if t.startswith("fact_")), ""),
                    "hit@1": int(hit_at_1),
                    "hit@5": int(hit_at_5),
                    "hit@10": int(hit_at_10),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "entity_weight": entity_weight,
        "n_queries": len(rows),
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
        "rows": rows,
    }


def _build_synth_entity_dataset(
    *,
    n_entities: int = 16,
    collision_degree: int = 4,
    distractors_per_entity: int = 3,
    seed: int = 42,
    tags: list[str] | None = None,
) -> tuple[Dataset, dict]:
    """Type-paired collider fixture (decision-#1 redesign).

    Wraps ``evals.entity_collision.generate_dataset`` for use by the
    D1 sweep driver. The resulting corpus satisfies the four-point
    checklist documented at the top of this module:

      1. K=collision_degree type-paired colliders per gold entity
         (every disc shares the same predicate template).
      2. Discriminator is paraphrased in the query
         (``disc`` → ``disc_syn``), so BM25 has no surface hook.
      3. ``rng.sample(spec['discs'], K)`` is without-replacement
         per entity, so no exact-duplicate gold facts.
      4. Δrecall@1 is the headline metric (k=5/@10 saturate fast).

    Each call concatenates `tags` so candidate-schema density is
    ~10× LoCoMo10's: the default 5-tag mix produces
    n_entities × K × 5 = 16·4·5 = 320 colliding facts in a fixture
    of ~640 mems (incl. distractors).
    """
    if tags is None:
        tags = ["preference", "service", "project", "tool", "technical"]
    ds = Dataset()
    n_gold = 0
    for t_i, tag in enumerate(tags):
        sub = _ec.generate_dataset(
            n_entities=n_entities,
            collision_degree=collision_degree,
            distractors_per_entity=distractors_per_entity,
            seed=seed + 1000 * t_i,
            tag=tag,
        )
        for content, meta in sub.memories:
            ds.memories.append((content, dict(meta)))
        for q in sub.queries:
            qq = Query(
                text=q.text,
                expected_substrings=list(q.expected_substrings),
                tags=[tag, f"K{q.collision_degree}"],
            )
            ds.queries.append(qq)
            n_gold += 1
    rng = random.Random(seed ^ 0xC0FFEE)
    rng.shuffle(ds.memories)
    meta = {
        "fixture": "synth_entity",
        "tags": list(tags),
        "n_entities": n_entities,
        "collision_degree": collision_degree,
        "distractors_per_entity": distractors_per_entity,
        "seed": seed,
        "n_memories": len(ds.memories),
        "n_queries": n_gold,
    }
    return ds, meta


def run_sweep(
    *,
    weights: list[float],
    n_facts: int = 80,
    hard_distractors_per_fact: int = 2,
    plain_distractors: int = 50,
    seed: int = 42,
    entity_ner: str = "heuristic",
    fixture: str = "d1",
    synth_n_entities: int = 16,
    synth_collision_degree: int = 4,
    synth_distractors_per_entity: int = 3,
    embedder=None,
    embed_name: str | None = None,
    save_per_query: bool = False,
) -> dict:
    t0 = time.monotonic()
    fixture = (os.environ.get("EVAL_FIXTURE") or fixture).lower()
    fixture_meta: dict
    if fixture in ("synth_entity", "synth-entity", "synth"):
        ds, fixture_meta = _build_synth_entity_dataset(
            n_entities=synth_n_entities,
            collision_degree=synth_collision_degree,
            distractors_per_entity=synth_distractors_per_entity,
            seed=seed,
        )
    else:
        ds = generate_entity_corpus(
            n_facts=n_facts,
            hard_distractors_per_fact=hard_distractors_per_fact,
            plain_distractors=plain_distractors,
            seed=seed,
        )
        fixture_meta = {
            "fixture": "d1",
            "n_facts": n_facts,
            "hard_distractors_per_fact": hard_distractors_per_fact,
            "plain_distractors": plain_distractors,
            "seed": seed,
            "n_memories": len(ds.memories),
            "n_queries": len(ds.queries),
        }
    arms = [_eval_arm(ds, w, entity_ner=entity_ner, embedder=embedder) for w in weights]

    # Δ vs baseline (entity_weight=0).
    baseline = next((a for a in arms if a["entity_weight"] == 0.0), arms[0])
    for a in arms:
        a["d_hit@1"] = round(a["hit@1"] - baseline["hit@1"], 6)
        a["d_hit@5"] = round(a["hit@5"] - baseline["hit@5"], 6)
        a["d_hit@10"] = round(a["hit@10"] - baseline["hit@10"], 6)

    return {
        "corpus": fixture_meta,
        "weights": weights,
        "entity_ner": entity_ner,
        "embed": embed_name or "hash",
        "arms": [{k: v for k, v in a.items() if (k != "rows" or save_per_query)} for a in arms],
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md_table(rep: dict) -> str:
    c = rep["corpus"]
    if c.get("fixture") == "synth_entity":
        header = (
            f"Corpus: synth_entity n_entities={c['n_entities']} "
            f"K={c['collision_degree']} tags={','.join(c['tags'])} "
            f"→ {c['n_memories']} mems, {c['n_queries']} queries  "
            f"(wall={rep['wall_seconds']}s, ner={rep.get('entity_ner', 'heuristic')})"
        )
    else:
        header = (
            f"Corpus: n_facts={c.get('n_facts')} "
            f"hard/{c.get('hard_distractors_per_fact')} "
            f"plain/{c.get('plain_distractors')} "
            f"→ {c['n_memories']} mems, {c['n_queries']} queries  "
            f"(wall={rep['wall_seconds']}s, ner={rep.get('entity_ner', 'heuristic')})"
        )
    lines = [
        header,
        "",
        "| entity_weight | hit@1 | hit@5 | hit@10 | Δhit@1 | Δhit@5 | Δhit@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in rep["arms"]:
        lines.append(
            f"| {a['entity_weight']:.2f} | {a['hit@1']:.3f} | {a['hit@5']:.3f} | {a['hit@10']:.3f} "
            f"| {a['d_hit@1']:+.3f} | {a['d_hit@5']:+.3f} | {a['d_hit@10']:+.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str,
                   default="0.0,0.05,0.1,0.2,0.3,0.5,1.0",
                   help="Comma-separated entity_weight values to sweep.")
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--hard-distractors", type=int, default=2)
    p.add_argument("--plain-distractors", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entity-ner", type=str, default="heuristic",
                   choices=["heuristic", "spacy_sm", "spacy_md", "spacy_lg"],
                   help="NER backend (heuristic regex or spaCy en_core_web_sm)")
    p.add_argument("--fixture", type=str, default="d1",
                   choices=["d1", "synth_entity"],
                   help="d1=legacy hit@5-saturated, synth_entity=type-paired colliders")
    p.add_argument("--synth-n-entities", type=int, default=16)
    p.add_argument("--synth-K", type=int, default=4,
                   help="collision_degree per entity (synth_entity only)")
    p.add_argument("--synth-distractors-per-entity", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--embed", type=str, default="hash",
                   help="Embedder backend: hash|hash128|hash256|hash512|hash1024|st|none")
    p.add_argument("--update-report", action="store_true",
                   help="Append the markdown table to ENTITY_CHANNEL_REPORT.md")
    p.add_argument("--save-per-query", action="store_true",
                   help="Persist per-query rows in --out JSON (for paired bootstrap CIs).")
    args = p.parse_args()

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    from evals.ablation import _make_embedder
    from evals._embed_cache import CachingEmbeddingProvider
    embedder = _make_embedder(args.embed)
    if embedder is not None:
        embedder = CachingEmbeddingProvider(embedder)
    rep = run_sweep(
        weights=weights,
        n_facts=args.n_facts,
        hard_distractors_per_fact=args.hard_distractors,
        plain_distractors=args.plain_distractors,
        seed=args.seed,
        entity_ner=args.entity_ner,
        fixture=args.fixture,
        synth_n_entities=args.synth_n_entities,
        synth_collision_degree=args.synth_K,
        synth_distractors_per_entity=args.synth_distractors_per_entity,
        embedder=embedder,
        embed_name=args.embed,
        save_per_query=args.save_per_query,
    )

    md = _md_table(rep)
    print("D1  entity-link channel  Δrecall@k sweep")
    print(md)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[entity-sweep] wrote {args.out}")

    if args.update_report:
        report = Path("ENTITY_CHANNEL_REPORT.md")
        c = rep["corpus"]
        if c.get("fixture") == "synth_entity":
            label = (f"fixture=synth_entity n_entities={c['n_entities']} "
                     f"K={c['collision_degree']} seed={c['seed']}")
        else:
            label = f"n_facts={c.get('n_facts')} seed={c.get('seed')}"
        header = (f"\n## Sweep run ({label}, "
                  f"ner={rep.get('entity_ner', 'heuristic')})\n\n")
        text = header + md + "\n"
        if report.exists():
            atomic_write_text(report, report.read_text() + text)
        else:
            atomic_write_text(report,
                "# Entity-Link Channel — Δrecall@k Report\n\n"
                "Driver: `evals/entity_channel_sweep.py`\n"
                + text
            )
        print(f"[entity-sweep] appended to {report}")


if __name__ == "__main__":
    main()

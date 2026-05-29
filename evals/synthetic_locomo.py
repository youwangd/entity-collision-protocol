"""Synthetic LoCoMo-style dataset generator.

Emits JSON in the exact snap-research/locomo public-release shape consumed
by `evals.locomo_adapter.load_locomo`, so the same adapter (and therefore
the same paper §4 codepath) can run end-to-end without the upstream
dataset on disk. The real `data/locomo/` is not yet provisioned; this
gives us a placeholder so paper-bound results have a structural mirror
and so changes to the adapter / retrieval stack don't regress in the
gap.

Generates samples with:

* `n_sessions` sessions per sample, each populated with K turns drawn
  from a small per-sample fact bag (atoms about the user's life:
  pets, jobs, hobbies, places). Each fact is anchored in exactly one
  session and re-mentioned (paraphrased) in 0+ later sessions.
* Filler turns interleaved between facts so retrieval can't trivially
  rank by length / position.
* QA pairs with three categories:
    - "single_hop" : evidence = one session
    - "multi_hop"  : evidence = 2 sessions (composition of two facts)
    - "adversarial": evidence = one session + adversarial_answer set,
                     question is paraphrased so a naïve embedder might
                     match a wrong session. Used to shape category-
                     conditional recall in §4.

Seeded; identical seed → byte-identical JSON. Default seed=2026.

Usage:

    python -m evals.synthetic_locomo \\
        --n-samples 50 --n-sessions 8 \\
        --out data/synthetic_locomo/synthetic_n50_s8.json
    python -m evals.locomo_adapter \\
        --dataset data/synthetic_locomo/synthetic_n50_s8.json \\
        --max-instances 50 --k 10 --arm both \\
        --out evals/results/synth_locomo_smoke.json
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from evals.io_utils import atomic_write_json


# Atomic facts (subject, predicate, object) — keyed templates so we can
# generate a base assertion + paraphrase + question + adversarial.
_FACT_TEMPLATES = [
    {
        "key": "pet_name",
        "assert": "I just adopted a {kind} named {obj}.",
        "para":   "{obj} (my {kind}) has been an absolute joy this week.",
        "q":      "What did I name my {kind}?",
        "a":      "{obj}",
        "adv_q":  "Is my {kind}'s name {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("dog", "Mochi"), ("cat", "Pepper"), ("rabbit", "Clover"),
                   ("cat", "Biscuit"), ("dog", "Tofu"), ("parrot", "Mango")],
    },
    {
        "key": "job_title",
        "assert": "Started a new role today as a {obj}.",
        "para":   "Day {day} of being a {obj} — pacing myself.",
        "q":      "What's my current job title?",
        "a":      "{obj}",
        "adv_q":  "Am I working as a {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("staff engineer",), ("ML researcher",), ("product manager",),
                   ("data scientist",), ("design lead",), ("infra SRE",)],
    },
    {
        "key": "city",
        "assert": "We're moving to {obj} next month — lease is signed.",
        "para":   "Boxes everywhere; {obj} is real now.",
        "q":      "What city did I move to?",
        "a":      "{obj}",
        "adv_q":  "Did I move to {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("Seattle",), ("Brooklyn",), ("Austin",), ("Lisbon",),
                   ("Berlin",), ("Kyoto",), ("Vancouver",)],
    },
    {
        "key": "hobby",
        "assert": "Picked up {obj} this season — it's wrecking my forearms.",
        "para":   "Two more {obj} sessions this week. Slowly improving.",
        "q":      "What new hobby did I pick up?",
        "a":      "{obj}",
        "adv_q":  "Did I take up {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("bouldering",), ("fencing",), ("pottery",), ("violin",),
                   ("calligraphy",), ("kendo",), ("skateboarding",)],
    },
    {
        "key": "allergy",
        "assert": "Doctor confirmed I'm allergic to {obj}. No more for me.",
        "para":   "Reading every label now — looking out for {obj}.",
        "q":      "What am I allergic to?",
        "a":      "{obj}",
        "adv_q":  "Am I allergic to {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("peanuts",), ("shellfish",), ("dairy",), ("cinnamon",),
                   ("sesame",), ("kiwi",)],
    },
    {
        "key": "milestone_year",
        "assert": "I graduated from {place} back in {obj}.",
        "para":   "Class of {obj}, {place} — feels like a different life.",
        "q":      "What year did I graduate from {place}?",
        "a":      "{obj}",
        "adv_q":  "Did I graduate from {place} in {wrong}?",
        "adv_a":  "{obj}",
        "objs":   [("2014",), ("2017",), ("2019",), ("2021",), ("2009",), ("2012",)],
        "extra":  {"place": ["Berkeley", "MIT", "CMU", "Waterloo", "ETH", "Cambridge"]},
    },
]


_FILLER = [
    "Coffee was great this morning.",
    "Need to send that follow-up email.",
    "The weather has been wild lately.",
    "Got stuck behind a parade on the way home.",
    "Slept poorly — neighbors were loud.",
    "Trying a new bread recipe this weekend.",
    "Bookstore had a 30% off sale, picked up two novels.",
    "Watched a really weird film last night.",
    "My back is killing me from yesterday's run.",
    "Rebooted the router; internet is fine again.",
]


@dataclass
class _PlannedFact:
    key: str
    template: dict
    fields: dict          # rendered fields ({obj, kind?, place?, day?})
    anchor_sid: str       # session id where the fact is first stated
    paraphrase_sids: list[str]  # later sessions that re-mention it


def _render(s: str, fields: dict) -> str:
    try:
        return s.format(**fields)
    except KeyError:
        return s


def _build_sample(rng: random.Random, sample_idx: int, n_sessions: int,
                  turns_per_session: int, n_facts: int) -> dict:
    # Pick distinct fact templates for this sample
    tmpls = rng.sample(_FACT_TEMPLATES, k=min(n_facts, len(_FACT_TEMPLATES)))
    planned: list[_PlannedFact] = []

    for t in tmpls:
        choice = rng.choice(t["objs"])
        fields: dict[str, str] = {}
        if len(choice) == 2:
            fields["kind"], fields["obj"] = choice
        else:
            fields["obj"] = choice[0]
        if "extra" in t:
            for ek, evals_list in t["extra"].items():
                fields[ek] = rng.choice(evals_list)
        fields["day"] = str(rng.randint(2, 30))
        # wrong-answer for adversarial: pick another obj from same template
        other = rng.choice([o for o in t["objs"] if o != choice])
        fields["wrong"] = other[-1] if isinstance(other, tuple) else str(other)

        anchor_idx = rng.randint(0, n_sessions - 1)
        anchor_sid = f"D{anchor_idx + 1}"
        # paraphrase in a later session ~50% of the time
        para_sids: list[str] = []
        if anchor_idx + 1 < n_sessions and rng.random() < 0.5:
            j = rng.randint(anchor_idx + 1, n_sessions - 1)
            para_sids.append(f"D{j + 1}")
        planned.append(_PlannedFact(
            key=t["key"], template=t, fields=fields,
            anchor_sid=anchor_sid, paraphrase_sids=para_sids,
        ))

    # Build sessions
    base_date = datetime(2025, 1, 1) + timedelta(days=sample_idx * 90)
    sessions: dict[str, object] = {}
    for s_idx in range(n_sessions):
        sid = f"D{s_idx + 1}"
        when = base_date + timedelta(days=s_idx * 4)
        sessions[f"session_{s_idx + 1}_date_time"] = when.strftime("%Y-%m-%d %H:%M")
        turns: list[dict] = []
        # collect anchored or paraphrased facts for this session
        anchored = [p for p in planned if p.anchor_sid == sid]
        paraphrased = [p for p in planned if sid in p.paraphrase_sids]
        # ensure each session has at least 1 fact-bearing turn if it owns one
        n_filler = max(1, turns_per_session - len(anchored) - len(paraphrased))
        # interleave: build a slot list and shuffle deterministically
        slots = (
            [("anchor", p) for p in anchored]
            + [("para", p) for p in paraphrased]
            + [("filler", None)] * n_filler
        )
        rng.shuffle(slots)
        for tnum, (kind, p) in enumerate(slots, start=1):
            speaker = "user" if (tnum % 2 == 1) else "assistant"
            if kind == "anchor":
                text = _render(p.template["assert"], p.fields)
            elif kind == "para":
                text = _render(p.template["para"], p.fields)
            else:
                text = rng.choice(_FILLER)
            turns.append({
                "speaker": speaker,
                "text": text,
                "dia_id": f"{sid}:{tnum}",
            })
        sessions[f"session_{s_idx + 1}"] = turns

    # Build QA
    qa: list[dict] = []
    # one single_hop per fact
    for p in planned:
        qa.append({
            "question": _render(p.template["q"], p.fields),
            "answer":   _render(p.template["a"], p.fields),
            "category": "single_hop",
            "evidence": [p.anchor_sid],
            "adversarial_answer": None,
        })
    # one adversarial per fact (paraphrased question + wrong-answer field)
    for p in planned:
        qa.append({
            "question": _render(p.template["adv_q"], p.fields),
            "answer":   _render(p.template["adv_a"], p.fields),
            "category": "adversarial",
            "evidence": [p.anchor_sid],
            "adversarial_answer": p.fields.get("wrong"),
        })
    # multi_hop: pair two facts whose answers compose ("In <city> I <hobby>")
    by_key = {p.key: p for p in planned}
    if "city" in by_key and "hobby" in by_key:
        c = by_key["city"]
        h = by_key["hobby"]
        qa.append({
            "question": "Where am I doing my new hobby, and what is it?",
            "answer": f"{h.fields['obj']} in {c.fields['obj']}",
            "category": "multi_hop",
            "evidence": sorted({c.anchor_sid, h.anchor_sid}),
            "adversarial_answer": None,
        })
    if "job_title" in by_key and "city" in by_key:
        j = by_key["job_title"]
        c = by_key["city"]
        qa.append({
            "question": "What's my current job title and which city did I move to?",
            "answer": f"{j.fields['obj']} in {c.fields['obj']}",
            "category": "multi_hop",
            "evidence": sorted({j.anchor_sid, c.anchor_sid}),
            "adversarial_answer": None,
        })

    return {
        "sample_id": f"synth_{sample_idx:04d}",
        "conversation": sessions,
        "qa": qa,
    }


def generate(n_samples: int, n_sessions: int, turns_per_session: int,
             n_facts: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        _build_sample(rng, i, n_sessions, turns_per_session, n_facts)
        for i in range(n_samples)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic LoCoMo-shape JSON.")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--turns-per-session", type=int, default=12)
    ap.add_argument("--n-facts", type=int, default=5,
                    help=f"Distinct fact templates per sample (≤ {len(_FACT_TEMPLATES)}).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    samples = generate(
        n_samples=args.n_samples,
        n_sessions=args.n_sessions,
        turns_per_session=args.turns_per_session,
        n_facts=args.n_facts,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, samples)
    n_qa = sum(len(s["qa"]) for s in samples)
    print(f"[synthetic_locomo] wrote {out} : "
          f"{len(samples)} samples, {n_qa} QA "
          f"(seed={args.seed}, n_sessions={args.n_sessions})")


if __name__ == "__main__":
    main()

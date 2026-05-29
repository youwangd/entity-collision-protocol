"""§96 — share_prior reranker Δrecall@k sweep.

Tests the rank-0 preservation invariant in practice (h@1 must not drop
relative to baseline) and measures whether multi-mate prior sharing
moves h@5 / multi-hop pair_recall.

Two recipes
-----------
1. **unique-entity** (reuses `evals.entity_channel_sweep.generate_entity_corpus`).
   Each gold memory has a unique entity. Multi-mate signal is a
   priori weak here (no entity sharing among golds), so this is
   primarily a *do-no-harm* test: alpha must not regress h@1 or h@5.

2. **bridge-entity multi-hop**. Pairs of facts share a bridge entity.
   The query asks a question that is best answered by retrieving BOTH
   facts in the pair. We measure pair_recall@k = fraction of queries
   where both gold facts appear in the top-k. This is the recipe where
   share_prior is hypothesized to help.

Outputs
-------
- evals/results/share_prior_sweep.json
- Markdown table on stdout
- if --update-report, writes/extends SHARE_PRIOR_REPORT.md
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from evals.synthetic import Dataset, Query
from evals.entity_channel_sweep import generate_entity_corpus
from evals.io_utils import atomic_write_json, atomic_write_text


# --- Bridge-entity multi-hop corpus ---------------------------------

_BRIDGE_TEMPLATES = [
    # (fact_a_template, fact_b_template, query_template)
    ("{person} leads engineering on {bridge}.",
     "{bridge} reports its OKRs to executive {exec}.",
     "which executive ultimately owns the work {person} leads?"),
    ("{person} relocated to {city} for the new role.",
     "{city} hosts the {bridge} regional headquarters.",
     "what regional headquarters serves {person}'s new city?"),
    ("{person} was promoted into the {bridge} division last year.",
     "{bridge} ships the flagship product {product}.",
     "what flagship product does {person}'s division ship?"),
]


def _person(i: int) -> str:
    first = ["Aria", "Brennan", "Cyra", "Devon", "Elara", "Finn",
            "Gita", "Hugo", "Ines", "Jonas", "Kavi", "Lior", "Mei",
            "Nuri", "Olin", "Petra", "Quinn", "Rasmus", "Saoirse",
            "Tariq", "Una", "Viggo", "Wren", "Xiulan", "Yotam", "Zara"]
    last = ["Alvarez", "Bjornsen", "Cattaneo", "Devarakonda", "Eriksson",
            "Fontaine", "Goyal", "Hartwell", "Iniesta", "Jovanovic",
            "Kowalski", "Lindqvist", "Marchetti", "Nagasawa", "Olufsen",
            "Pereira", "Quirke", "Rasmussen", "Stoltzfus", "Takahashi"]
    return f"{first[i % len(first)]} {last[(i // len(first)) % len(last)]}"


def _bridge(i: int) -> str:
    names = ["Project Atlas", "Project Borealis", "Project Cipher",
             "Project Delta", "Project Echo", "Project Fortuna",
             "Project Gemini", "Project Helios", "Project Iris",
             "Project Juno", "Project Kestrel", "Project Lumen"]
    return names[i % len(names)]


def _exec(i: int) -> str:
    names = ["Marcus Reeves", "Pamela Hsu", "Linus Erikson", "Naomi Park",
             "Vincent Albright", "Karina Solberg", "Theo Donnelly",
             "Imogen Whitley"]
    return names[i % len(names)]


def _city(i: int) -> str:
    names = ["Lisbon Centre", "Osaka Bay", "Auckland North", "Calgary West",
             "Mumbai Pier", "Reykjavik", "Buenos Aires", "Cape Town"]
    return names[i % len(names)]


def _product(i: int) -> str:
    names = ["NimbusOS", "ZephyrCloud", "AegisGuard", "BeaconSync",
             "QuasarKit", "PyxisLens", "LumenStack", "OrionMesh"]
    return names[i % len(names)]


def generate_bridge_corpus(
    n_pairs: int = 60,
    plain_distractors: int = 80,
    seed: int = 17,
) -> Dataset:
    """Each pair shares a bridge entity. Query needs BOTH facts of a pair."""
    rng = random.Random(seed)
    ds = Dataset()
    pair_idx = 0
    for p_i in range(n_pairs):
        a_t, b_t, q_t = _BRIDGE_TEMPLATES[p_i % len(_BRIDGE_TEMPLATES)]
        bindings = {
            "person": _person(p_i),
            "bridge": _bridge(p_i),
            "exec": _exec(p_i),
            "city": _city(p_i),
            "product": _product(p_i),
        }
        try:
            a_text = a_t.format(**bindings)
            b_text = b_t.format(**bindings)
            q_text = q_t.format(**bindings)
        except KeyError:
            continue

        a_id = f"pair_{p_i:04d}_a"
        b_id = f"pair_{p_i:04d}_b"
        ds.memories.append((a_text, {"kind": "fact", "fact_id": a_id,
                                     "pair_id": f"pair_{p_i:04d}"}))
        ds.memories.append((b_text, {"kind": "fact", "fact_id": b_id,
                                     "pair_id": f"pair_{p_i:04d}"}))

        # Gold substrings: pick the value bound for each unique slot
        # so we can cheaply identify whether each fact made the topk.
        # We use the fact_id token in metadata for exact attribution
        # via Memory.metadata, but for content-only matching we use
        # a distinctive slot value from each fact.
        a_anchor = bindings["person"]            # appears only in fact_a
        # Pick a b_anchor that (a) appears in b_text, (b) does NOT appear in
        # a_text, and (c) is not the bridge itself (which appears in both).
        # Order candidates by specificity: exec / product / city (city is the
        # fallback for the relocation template, where both a_text and b_text
        # mention the same city — in that case there is no clean per-fact
        # anchor and we must accept that pair_recall is unmeasurable; skip).
        b_anchor_candidates = [bindings.get("exec"), bindings.get("product"),
                               bindings.get("city")]
        b_anchor = next((v for v in b_anchor_candidates
                         if v and v != bindings["bridge"]
                         and v in b_text and v not in a_text), None)
        if b_anchor is None:
            # No measurable per-fact anchor in b_text — drop the pair from
            # the eval rather than scoring against a phantom string.
            ds.memories.pop()
            ds.memories.pop()
            continue

        ds.queries.append(Query(
            text=q_text,
            expected_substrings=[a_anchor, b_anchor],
            tags=[f"pair_{p_i:04d}", "multi_hop"],
        ))
        pair_idx += 1

    for i in range(plain_distractors):
        ds.memories.append((
            f"Routine note #{i}: weekly team standup; sprint goals on track.",
            {"kind": "distractor"},
        ))
    rng.shuffle(ds.memories)
    return ds


# --- Sweep arms -----------------------------------------------------

def _build_engine(path: str, *, reranker: str | None,
                  alpha: float, pool_size: int) -> Engram:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval = RetrievalConfig(
        reranker=reranker,
        rerank_pool_size=pool_size,
        share_prior_alpha=alpha,
    )
    return Engram(config=cfg)


def _hits(text: str, anchors: list[str]) -> bool:
    t = (text or "").lower()
    return all(a and a.lower() in t for a in anchors)


def _any_hit(text: str, anchors: list[str]) -> bool:
    t = (text or "").lower()
    return any(a and a.lower() in t for a in anchors)


def _eval_unique(ds: Dataset, *, reranker, alpha, pool_size, k_max=10) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp, reranker=reranker, alpha=alpha,
                            pool_size=pool_size)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                gold = q.expected_substrings[0]
                rows.append({
                    "hit@1": int(bool(texts) and gold.lower() in texts[0].lower()),
                    "hit@5": int(any(gold.lower() in t.lower() for t in texts[:5])),
                    "hit@10": int(any(gold.lower() in t.lower() for t in texts[:10])),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "recipe": "unique_entity",
        "reranker": reranker,
        "alpha": alpha,
        "n_queries": len(rows),
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
    }


def _eval_bridge(ds: Dataset, *, reranker, alpha, pool_size, k_max=10) -> dict:
    """pair_recall@k = fraction of queries where BOTH anchors appear in top-k."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp, reranker=reranker, alpha=alpha,
                            pool_size=pool_size)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                a, b = q.expected_substrings[0], q.expected_substrings[1]

                def pair_at(k: int) -> int:
                    top = texts[:k]
                    has_a = any(a.lower() in t.lower() for t in top)
                    has_b = any(b.lower() in t.lower() for t in top)
                    return int(has_a and has_b)

                def any_at(k: int) -> int:
                    top = texts[:k]
                    return int(any(a.lower() in t.lower()
                                   or b.lower() in t.lower() for t in top))

                rows.append({
                    "pair@5": pair_at(5),
                    "pair@10": pair_at(10),
                    "any@1": any_at(1),
                    "any@5": any_at(5),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "recipe": "bridge_multi_hop",
        "reranker": reranker,
        "alpha": alpha,
        "n_queries": len(rows),
        "pair_recall@5": sum(r["pair@5"] for r in rows) / n,
        "pair_recall@10": sum(r["pair@10"] for r in rows) / n,
        "any_hit@1": sum(r["any@1"] for r in rows) / n,
        "any_hit@5": sum(r["any@5"] for r in rows) / n,
    }


def _delta(arm: dict, baseline: dict, keys: list[str]) -> None:
    for k in keys:
        if k in arm and k in baseline:
            arm[f"d_{k}"] = round(arm[k] - baseline[k], 6)


def run_sweep(*, alphas, n_facts, n_pairs, plain_distractors, seed) -> dict:
    t0 = time.monotonic()
    ds_unique = generate_entity_corpus(
        n_facts=n_facts, hard_distractors_per_fact=2,
        plain_distractors=plain_distractors, seed=seed)
    ds_bridge = generate_bridge_corpus(
        n_pairs=n_pairs, plain_distractors=plain_distractors, seed=seed + 1)

    # Baseline: reranker=None, alpha=0 (alpha is inert when reranker=None
    # but we set it explicitly to make the table read clearly).
    arms_unique = [
        _eval_unique(ds_unique, reranker=None, alpha=0.0, pool_size=20)
    ]
    arms_bridge = [
        _eval_bridge(ds_bridge, reranker=None, alpha=0.0, pool_size=20)
    ]
    for a in alphas:
        arms_unique.append(_eval_unique(ds_unique, reranker="share_prior",
                                        alpha=a, pool_size=20))
        arms_bridge.append(_eval_bridge(ds_bridge, reranker="share_prior",
                                        alpha=a, pool_size=20))

    # Δ vs baseline arm (reranker=None).
    _delta(arms_unique[0], arms_unique[0], ["hit@1", "hit@5", "hit@10"])
    for a in arms_unique[1:]:
        _delta(a, arms_unique[0], ["hit@1", "hit@5", "hit@10"])
    _delta(arms_bridge[0], arms_bridge[0],
           ["pair_recall@5", "pair_recall@10", "any_hit@1", "any_hit@5"])
    for a in arms_bridge[1:]:
        _delta(a, arms_bridge[0],
               ["pair_recall@5", "pair_recall@10", "any_hit@1", "any_hit@5"])

    return {
        "alphas": alphas,
        "corpus": {"n_facts": n_facts, "n_pairs": n_pairs,
                   "plain_distractors": plain_distractors, "seed": seed,
                   "n_unique_memories": len(ds_unique.memories),
                   "n_bridge_memories": len(ds_bridge.memories)},
        "unique_entity": arms_unique,
        "bridge_multi_hop": arms_bridge,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep: dict) -> str:
    lines = [
        f"Wall: {rep['wall_seconds']}s  "
        f"unique={rep['corpus']['n_unique_memories']} mems, "
        f"bridge={rep['corpus']['n_bridge_memories']} mems "
        f"(seed={rep['corpus']['seed']})",
        "",
        "### Unique-entity (do-no-harm test for h@1)",
        "",
        "| reranker | alpha | hit@1 | hit@5 | hit@10 | Δhit@1 | Δhit@5 | Δhit@10 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in rep["unique_entity"]:
        lines.append(
            f"| {a['reranker'] or '—'} | {a['alpha']:.2f} "
            f"| {a['hit@1']:.3f} | {a['hit@5']:.3f} | {a['hit@10']:.3f} "
            f"| {a.get('d_hit@1', 0):+.3f} | {a.get('d_hit@5', 0):+.3f} "
            f"| {a.get('d_hit@10', 0):+.3f} |"
        )
    lines += [
        "",
        "### Bridge multi-hop (pair_recall is the target signal)",
        "",
        "| reranker | alpha | pair@5 | pair@10 | any@1 | any@5 | Δpair@5 | Δpair@10 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in rep["bridge_multi_hop"]:
        lines.append(
            f"| {a['reranker'] or '—'} | {a['alpha']:.2f} "
            f"| {a['pair_recall@5']:.3f} | {a['pair_recall@10']:.3f} "
            f"| {a['any_hit@1']:.3f} | {a['any_hit@5']:.3f} "
            f"| {a.get('d_pair_recall@5', 0):+.3f} "
            f"| {a.get('d_pair_recall@10', 0):+.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=str, default="0.02,0.05,0.10,0.20")
    p.add_argument("--n-facts", type=int, default=80)
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="evals/results/share_prior_sweep.json")
    p.add_argument("--update-report", action="store_true")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rep = run_sweep(alphas=alphas, n_facts=args.n_facts, n_pairs=args.n_pairs,
                    plain_distractors=args.plain_distractors, seed=args.seed)
    md = _md(rep)
    print("§96 share_prior reranker — Δrecall@k sweep")
    print(md)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[share-prior-sweep] wrote {args.out}")

    if args.update_report:
        report = Path("SHARE_PRIOR_REPORT.md")
        header = (f"\n## Sweep run "
                  f"(n_facts={rep['corpus']['n_facts']}, "
                  f"n_pairs={rep['corpus']['n_pairs']}, "
                  f"seed={rep['corpus']['seed']})\n\n")
        text = header + md + "\n"
        if report.exists():
            atomic_write_text(report, report.read_text() + text)
        else:
            atomic_write_text(report,
                "# §96 share_prior Reranker — Δrecall@k Report\n\n"
                "Driver: `evals/share_prior_sweep.py`\n"
                "Mechanism: undirected entity-sharing graph over the\n"
                "candidate pool; per-candidate boost = α · (deg / max_deg),\n"
                "capped to preserve original rank-0.\n"
                + text
            )
        print(f"[share-prior-sweep] appended to {report}")


if __name__ == "__main__":
    main()

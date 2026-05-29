"""§96 stacking sweep — share_prior × entity_weight × rerank_pool_size.

Hypothesis (per NEXT.md after the share_prior_sweep run):
the bridge multi-hop signal collapsed at larger corpus because BM25
couldn't surface gold pair facts into the rerank pool. Two knobs to
investigate:

1. **entity_weight** — gives the entity-link channel a chance to pull
   gold pair facts (which share a bridge entity) into the top-K
   *before* the share_prior reranker runs.
2. **rerank_pool_size** — simply enlarging the pool we rerank over
   gives share_prior more material to work with.

This driver runs the bridge recipe only (multi-hop is the regime we
care about) on a bigger corpus and sweeps both. Baseline is
`reranker=None, entity_weight=0.0`.

Output
------
- evals/results/share_prior_stack_sweep.json
- Markdown table on stdout
- if --update-report, appends to SHARE_PRIOR_REPORT.md
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from engram.core.config import RetrievalConfig
from evals.entity_channel_sweep import generate_entity_corpus
from evals.share_prior_sweep import generate_bridge_corpus
from evals.io_utils import atomic_write_json, atomic_write_text


def _build_engine(path: str, *, reranker, alpha, entity_weight, pool_size):
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.retrieval = RetrievalConfig(
        reranker=reranker,
        rerank_pool_size=pool_size,
        share_prior_alpha=alpha,
        entity_weight=entity_weight,
    )
    return Engram(config=cfg)


def _eval_arm_unique(ds, *, reranker, alpha, entity_weight, pool_size, k_max=10):
    """Unique-entity recipe: rank-0 hit on the gold anchor."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp, reranker=reranker, alpha=alpha,
                            entity_weight=entity_weight, pool_size=pool_size)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                gold = q.expected_substrings[0].lower()
                rows.append({
                    "hit@1": int(bool(texts) and gold in texts[0].lower()),
                    "hit@5": int(any(gold in t.lower() for t in texts[:5])),
                    "hit@10": int(any(gold in t.lower() for t in texts[:10])),
                })
        finally:
            eng.close()
    n = max(len(rows), 1)
    return {
        "recipe": "unique_entity",
        "reranker": reranker,
        "alpha": alpha,
        "entity_weight": entity_weight,
        "pool_size": pool_size,
        "n_queries": len(rows),
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
    }


def _eval_arm(ds, *, reranker, alpha, entity_weight, pool_size, k_max=10):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _build_engine(tmp, reranker=reranker, alpha=alpha,
                            entity_weight=entity_weight, pool_size=pool_size)
        try:
            for content, meta in ds.memories:
                clean = {k: v for k, v in meta.items()
                         if isinstance(v, (str, int, float, bool))}
                eng.remember(content, **clean)
            for q in ds.queries:
                results = eng.recall(q.text, limit=k_max)
                texts = [r.memory.content for r in results]
                a, b = q.expected_substrings[0], q.expected_substrings[1]

                def pair_at(k):
                    top = texts[:k]
                    has_a = any(a.lower() in t.lower() for t in top)
                    has_b = any(b.lower() in t.lower() for t in top)
                    return int(has_a and has_b)

                def any_at(k):
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
        "reranker": reranker,
        "alpha": alpha,
        "entity_weight": entity_weight,
        "pool_size": pool_size,
        "n_queries": len(rows),
        "pair_recall@5": sum(r["pair@5"] for r in rows) / n,
        "pair_recall@10": sum(r["pair@10"] for r in rows) / n,
        "any_hit@1": sum(r["any@1"] for r in rows) / n,
        "any_hit@5": sum(r["any@5"] for r in rows) / n,
    }


def _delta(arm, base, keys):
    for k in keys:
        if k in arm and k in base:
            arm[f"d_{k}"] = round(arm[k] - base[k], 6)


def run_sweep(*, alpha, entity_weights, pool_sizes,
              n_pairs, plain_distractors, seed, recipe="bridge"):
    t0 = time.monotonic()
    if recipe == "bridge":
        ds = generate_bridge_corpus(n_pairs=n_pairs,
                                    plain_distractors=plain_distractors,
                                    seed=seed)
        evaluator = _eval_arm
        delta_keys = ["pair_recall@5", "pair_recall@10", "any_hit@1", "any_hit@5"]
    elif recipe == "unique":
        ds = generate_entity_corpus(n_facts=n_pairs,
                                    hard_distractors_per_fact=2,
                                    plain_distractors=plain_distractors,
                                    seed=seed)
        evaluator = _eval_arm_unique
        delta_keys = ["hit@1", "hit@5", "hit@10"]
    else:
        raise ValueError(f"unknown recipe {recipe!r}")

    arms = []
    base = evaluator(ds, reranker=None, alpha=0.0,
                     entity_weight=0.0, pool_size=20)
    arms.append(base)
    _delta(base, base, delta_keys)

    # entity_weight only (no reranker)
    for ew in entity_weights:
        if ew == 0.0:
            continue
        arm = evaluator(ds, reranker=None, alpha=0.0,
                        entity_weight=ew, pool_size=20)
        _delta(arm, base, delta_keys)
        arms.append(arm)

    # share_prior × entity_weight × pool_size grid
    for ew in entity_weights:
        for ps in pool_sizes:
            arm = evaluator(ds, reranker="share_prior", alpha=alpha,
                            entity_weight=ew, pool_size=ps)
            _delta(arm, base, delta_keys)
            arms.append(arm)

    return {
        "recipe": recipe,
        "alpha": alpha,
        "entity_weights": entity_weights,
        "pool_sizes": pool_sizes,
        "corpus": {"n_pairs": n_pairs,
                   "plain_distractors": plain_distractors,
                   "seed": seed,
                   "n_memories": len(ds.memories)},
        "arms": arms,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep):
    recipe = rep.get("recipe", "bridge")
    lines = [
        f"Wall: {rep['wall_seconds']}s  "
        f"recipe={recipe}  "
        f"corpus={rep['corpus']['n_memories']} mems "
        f"({rep['corpus']['n_pairs']} {'pairs' if recipe == 'bridge' else 'facts'} + "
        f"{rep['corpus']['plain_distractors']} distractors, "
        f"seed={rep['corpus']['seed']})  "
        f"alpha={rep['alpha']}",
        "",
    ]
    if recipe == "bridge":
        lines += [
            "| reranker | alpha | ew | pool | pair@5 | pair@10 | any@1 | "
            "Δpair@5 | Δpair@10 |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for a in rep["arms"]:
            lines.append(
                f"| {a['reranker'] or '—'} | {a['alpha']:.2f} "
                f"| {a['entity_weight']:.2f} | {a['pool_size']} "
                f"| {a['pair_recall@5']:.3f} | {a['pair_recall@10']:.3f} "
                f"| {a['any_hit@1']:.3f} "
                f"| {a.get('d_pair_recall@5', 0):+.3f} "
                f"| {a.get('d_pair_recall@10', 0):+.3f} |"
            )
    else:
        lines += [
            "| reranker | alpha | ew | pool | hit@1 | hit@5 | hit@10 | "
            "Δhit@1 | Δhit@5 |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for a in rep["arms"]:
            lines.append(
                f"| {a['reranker'] or '—'} | {a['alpha']:.2f} "
                f"| {a['entity_weight']:.2f} | {a['pool_size']} "
                f"| {a['hit@1']:.3f} | {a['hit@5']:.3f} | {a['hit@10']:.3f} "
                f"| {a.get('d_hit@1', 0):+.3f} "
                f"| {a.get('d_hit@5', 0):+.3f} |"
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--entity-weights", type=str, default="0.0,0.1,0.2,0.3")
    p.add_argument("--pool-sizes", type=str, default="20,40,80")
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipe", choices=["bridge", "unique", "both"],
                   default="bridge")
    p.add_argument("--out", default="evals/results/share_prior_stack_sweep.json")
    p.add_argument("--update-report", action="store_true")
    args = p.parse_args()

    ews = [float(x) for x in args.entity_weights.split(",") if x.strip()]
    pss = [int(x) for x in args.pool_sizes.split(",") if x.strip()]

    recipes = ["bridge", "unique"] if args.recipe == "both" else [args.recipe]
    reports = []
    for recipe in recipes:
        rep = run_sweep(alpha=args.alpha, entity_weights=ews, pool_sizes=pss,
                        n_pairs=args.n_pairs,
                        plain_distractors=args.plain_distractors,
                        seed=args.seed, recipe=recipe)
        md = _md(rep)
        print(f"§96 share_prior × entity_weight × pool_size — {recipe}")
        print(md)
        reports.append((recipe, rep, md))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if len(reports) == 1:
            payload = reports[0][1]
        else:
            payload = {r: rep for r, rep, _ in reports}
        atomic_write_json(args.out, payload, default=str)
        print(f"[stack-sweep] wrote {args.out}")

    if args.update_report:
        report_path = Path("SHARE_PRIOR_REPORT.md")
        text_blocks = []
        for recipe, rep, md in reports:
            header = (f"\n## Stacking sweep — {recipe} "
                      f"(α={rep['alpha']}, "
                      f"n_pairs={rep['corpus']['n_pairs']}, "
                      f"seed={rep['corpus']['seed']})\n\n")
            text_blocks.append(header + md + "\n")
        text = "".join(text_blocks)
        if report_path.exists():
            atomic_write_text(report_path, report_path.read_text() + text)
        else:
            atomic_write_text(report_path, "# §96 share_prior — Stacking Report\n" + text)
        print(f"[stack-sweep] appended to {report_path}")


if __name__ == "__main__":
    main()

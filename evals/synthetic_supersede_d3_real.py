"""§D3-real — supersede ablation on a synthetic conflicting-facts corpus.

Probes the question §D3-add-only-CI-full could not answer on LoCoMo10
(where the interference stage was inert at the default recipe):

    "Does ADD-only consolidation under-recall versus default supersede
    when the workload actually contains conflicts?"

Corpus
------
`evals.synthetic.generate_supersede_dataset` plants ``n_slots`` slots,
each with ``updates_per_slot`` successive contradicting statements
about the same entity (lexical overlap > 0.6 by construction so the
default interference detector classifies later→earlier as
``supersede``). Gold answer is the LATEST value; older values are
"stale". A retrieval system that respects supersede should return the
latest value; one that does not should sometimes surface a stale
value first.

Arms
----
  arm_default : full pipeline, supersede ON
                (older versions transitioned to FADED, excluded from
                normal recall).
  arm_addonly : full pipeline, ``consolidation.add_only=True``
                (interference stage no-ops, all versions remain
                ACTIVE and recallable).

Metrics (per query)
-------------------
  hit@1_latest    : top-1 result contains the gold (latest) value.
  hit@k_latest    : any of top-k contains the gold value.
  stale_at_1      : top-1 contains a stale (older-version) value.
  stale_at_k      : any of top-k contains a stale value.
  Δstaleness@1    : stale_at_1 ∈ {0, 1}; >0 means retriever surfaced
                    a stale value at rank 1.

Headlines
---------
  hit@1, hit@k, stale@1, stale@k for each arm; paired-bootstrap CI on
  (default − addonly) for each.

Determinism
-----------
Default embedder = HashTrigram-256 (no clocks, no IO). Memory ingest
preserves insertion order so that update_idx ordering is preserved
into store created_at. Both arms call ``consolidate(window="999d")``
to push events through the pipeline.

Wall budget: ~5–15 s for n_slots=50, scales linearly.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from engram import Engram, Config
from engram.core.config import ConsolidationConfig

from evals.synthetic import generate_supersede_dataset
from evals.locomo_recall_lift_decompose_ci import _bootstrap_mean_ci
from evals.io_utils import atomic_write_json


def _build_config(path: str, *, add_only: bool, entity_aware: bool = False,
                  entity_min: float = 0.5) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    cfg.consolidation = ConsolidationConfig(
        schedule="manual",
        window_hours=24 * 999,
        add_only=add_only,
        interference_entity_aware=entity_aware,
        interference_entity_overlap_min=entity_min,
    )
    return cfg


def _value_in_text(value: str, text: str) -> bool:
    return value.lower() in text.lower()


def _run_arm(ds, *, add_only: bool, k: int = 10, entity_aware: bool = False,
             entity_min: float = 0.5) -> list[dict]:
    """Ingest, run InterferenceDetection directly, recall every query.

    Why direct-stage invocation? The default consolidation pipeline
    routes facts through ``EpisodeExtraction → FactExtraction``, and
    FactExtraction is a no-op without an LLM provider. With
    ``NoLLMProvider`` the interference stage never sees a single FACT
    memory in ``memories_created`` and is structurally inert
    (mirroring the §D3-add-only-CI-full LoCoMo10 finding). For the
    §D3-real probe we want to measure what interference would do
    *given* a stream of conflicting facts — so we synthesise that
    stream by reading the store's FACT memories ordered by
    ``created_at`` and feeding them into a hand-built StageContext.

    The ADD-only branch checks ``cfg.consolidation.add_only`` exactly
    the same way the production stage does, so this harness shares
    the production code path inside ``InterferenceDetection.run``.
    """
    from engram.consolidation.pipeline import InterferenceDetection, StageContext

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_config(tmp, add_only=add_only, entity_aware=entity_aware,
                            entity_min=entity_min)
        eng = Engram(config=cfg)
        try:
            for content, meta in ds.memories:
                clean_meta = {k_: v_ for k_, v_ in meta.items()
                              if isinstance(v_, (str, int, float, bool))}
                eng.remember(content, **clean_meta)

            # Stream all FACT memories through interference in
            # insertion (created_at) order. The store doesn't expose
            # an order-preserving "all FACT" iterator directly, so
            # collect via search_by_state(ACTIVE) and sort.
            from engram.core import MemoryState, MemoryType
            active_facts = [
                m for m in eng._store.search_by_state(MemoryState.ACTIVE, limit=10_000)
                if m.type == MemoryType.FACT
            ]
            active_facts.sort(key=lambda m: m.created_at)

            ctx = StageContext(
                store=eng._store,
                config=eng.config,
                buffer=None,
                memories_created=active_facts,
            )
            stage = InterferenceDetection()
            stage.run(ctx)

            interference_actions = ctx.stats.get("interference_actions", 0)

            for q in ds.queries:
                results = eng.recall(q.text, limit=k)
                texts = [r.memory.content for r in results]
                gold = q.expected_substrings[0]
                stale_tag = next((t for t in q.tags if t.startswith("stale=")), "stale=")
                stale_values = stale_tag[len("stale="):].split("|") if stale_tag else []

                hit_at_1 = bool(texts) and _value_in_text(gold, texts[0])
                hit_at_k = any(_value_in_text(gold, t) for t in texts)
                stale_at_1 = bool(texts) and any(_value_in_text(s, texts[0]) for s in stale_values)
                stale_at_k = any(any(_value_in_text(s, t) for s in stale_values) for t in texts)

                slot_id = next((t for t in q.tags if t.startswith("slot_")), "")
                tag = q.tags[0] if q.tags else ""
                rows.append({
                    "slot_id": slot_id,
                    "tag": tag,
                    "query": q.text,
                    "gold": gold,
                    "stale_values": stale_values,
                    "hit_at_1": int(hit_at_1),
                    "hit_at_k": int(hit_at_k),
                    "stale_at_1": int(stale_at_1),
                    "stale_at_k": int(stale_at_k),
                    "n_results": len(results),
                    "_interference_actions": interference_actions,
                })
        finally:
            eng.close()
    return rows


def _arm_summary(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "hit_at_1": sum(r["hit_at_1"] for r in rows) / n,
        "hit_at_k": sum(r["hit_at_k"] for r in rows) / n,
        "stale_at_1": sum(r["stale_at_1"] for r in rows) / n,
        "stale_at_k": sum(r["stale_at_k"] for r in rows) / n,
        "interference_actions": rows[0].get("_interference_actions", 0) if rows else 0,
    }


def run_d3_real(
    *,
    n_slots: int = 50,
    updates_per_slot: int = 2,
    distractors: int = 100,
    seed: int = 42,
    k: int = 10,
    resamples: int = 10000,
    boot_seed: int = 42,
    entity_aware: bool = False,
    entity_min: float = 0.5,
) -> dict:
    t0 = time.monotonic()
    ds = generate_supersede_dataset(
        n_slots=n_slots, updates_per_slot=updates_per_slot,
        distractors=distractors, seed=seed,
    )

    rows_def = _run_arm(ds, add_only=False, k=k, entity_aware=entity_aware, entity_min=entity_min)
    rows_add = _run_arm(ds, add_only=True, k=k, entity_aware=entity_aware, entity_min=entity_min)

    # Pair by slot_id (one query per slot).
    by_slot_def = {r["slot_id"]: r for r in rows_def}
    by_slot_add = {r["slot_id"]: r for r in rows_add}
    common = sorted(set(by_slot_def) & set(by_slot_add))

    diffs = {"d_hit_at_1": [], "d_hit_at_k": [],
             "d_stale_at_1": [], "d_stale_at_k": []}
    for sid in common:
        a, b = by_slot_def[sid], by_slot_add[sid]
        diffs["d_hit_at_1"].append(a["hit_at_1"] - b["hit_at_1"])
        diffs["d_hit_at_k"].append(a["hit_at_k"] - b["hit_at_k"])
        diffs["d_stale_at_1"].append(a["stale_at_1"] - b["stale_at_1"])
        diffs["d_stale_at_k"].append(a["stale_at_k"] - b["stale_at_k"])

    summary = {}
    for key, vals in diffs.items():
        m, lo, hi, p = _bootstrap_mean_ci(vals, resamples, boot_seed)
        summary[key] = {
            "mean_diff_default_minus_addonly": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_bootstrap_two_sided": round(p, 6),
            "n_paired": len(vals),
        }

    return {
        "corpus": {
            "n_slots": n_slots,
            "updates_per_slot": updates_per_slot,
            "distractors": distractors,
            "seed": seed,
            "n_memories": len(ds.memories),
            "n_queries": len(ds.queries),
        },
        "k": k,
        "arms": {
            "default": _arm_summary(rows_def),
            "addonly": _arm_summary(rows_add),
        },
        "ci_config": {"resamples": resamples, "seed": boot_seed,
                      "alpha": 0.05, "method": "percentile_paired_diff"},
        "summary": summary,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-slots", type=int, default=50)
    p.add_argument("--updates-per-slot", type=int, default=2)
    p.add_argument("--distractors", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--entity-aware", action="store_true",
                   help="§D3-collateral-(b): require entity-token Jaccard match")
    p.add_argument("--entity-min", type=float, default=0.5,
                   help="entity-token Jaccard threshold (default 0.5)")
    args = p.parse_args()

    rep = run_d3_real(
        n_slots=args.n_slots,
        updates_per_slot=args.updates_per_slot,
        distractors=args.distractors,
        seed=args.seed,
        k=args.k,
        resamples=args.resamples,
        boot_seed=args.boot_seed,
        entity_aware=args.entity_aware,
        entity_min=args.entity_min,
    )

    print("§D3-real  supersede synthetic corpus")
    print(f"  corpus: n_slots={rep['corpus']['n_slots']} updates={rep['corpus']['updates_per_slot']} "
          f"mems={rep['corpus']['n_memories']} queries={rep['corpus']['n_queries']}")
    print(f"  wall={rep['wall_seconds']}s")
    a, b = rep["arms"]["default"], rep["arms"]["addonly"]
    print(f"  ARM default : hit@1={a['hit_at_1']:.3f} hit@k={a['hit_at_k']:.3f} "
          f"stale@1={a['stale_at_1']:.3f} stale@k={a['stale_at_k']:.3f} "
          f"interference_actions={a['interference_actions']}")
    print(f"  ARM addonly : hit@1={b['hit_at_1']:.3f} hit@k={b['hit_at_k']:.3f} "
          f"stale@1={b['stale_at_1']:.3f} stale@k={b['stale_at_k']:.3f} "
          f"interference_actions={b['interference_actions']}")
    for k_ in ("d_hit_at_1", "d_hit_at_k", "d_stale_at_1", "d_stale_at_k"):
        c = rep["summary"][k_]
        print(f"  Δ({k_:>14}): mean={c['mean_diff_default_minus_addonly']:+.4f}  "
              f"95% CI=[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
              f"p={c['p_bootstrap_two_sided']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[D3-real] wrote {args.out}")


if __name__ == "__main__":
    main()

"""§90 — End-to-end recall-lift on LoCoMo at the §87 recommended recipe.

Closes NEXT pickup #1 (post-§89). §87 picked the paper recipe on
*cluster-topology* grounds (tau=0.20, share=0.75, fmax=0.9021,
contam_max=0.15). This driver answers the natural follow-up: does
flipping the consolidation gate at that recipe move ``session_hit@k``
on real LoCoMo recall, or is the gate operationally inert?

Design
------
Per-sample paired comparison. For each LoCoMo sample we:

  1. Build engram, ingest all turns, recall every QA → BASELINE.
  2. Tear down. Build a *fresh* engram with the same config but
     ``schema_family_share=0.75``, ``schema_family_tau=0.20``,
     ``schema_family_fragmentation_max=0.9021``,
     ``schema_family_contamination_max=0.15``. Ingest, then
     ``consolidate(window="999d")`` to flush all events through the
     pipeline (so the schema-family gate gets to fire), and recall
     every QA → TREATMENT.

Both arms share the same embedder (default HashTrigram-256), the
same vector_weight, the same RNG seed (none — deterministic). The
*only* delta is the four schema_family knobs.

Metrics
-------
We report:
  * baseline / treatment ``session_hit_at_1`` and ``session_hit_at_k``.
  * paired delta (treatment − baseline) overall.
  * per-category delta.
  * a ``per_query_pairs`` field (gold session set, baseline rank,
    treatment rank, delta in 1/rank) for downstream sign-test or
    bootstrap CI.

The driver is *expensive* (consolidation + double-ingest), so it
defaults to ``max_instances=2`` for the smoke variant. The full-
corpus run is intended to be invoked manually.

Pure: deterministic given the input json + embedder. No clocks
participate in scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

from engram import Engram, Config
from engram.core.config import ConsolidationConfig

from evals.locomo_adapter import (
    load_locomo,
    _ingest,
    _session_id_of,
)
from evals.io_utils import atomic_write_json


# §87 recommended recipe (paper-locked). Source: SCALE_REPORT §87.
RECIPE = {
    "schema_family_share": 0.75,
    "schema_family_tau": 0.20,
    "schema_family_fragmentation_max": 0.9021,
    "schema_family_contamination_max": 0.15,
}


def _build_config(
    path: str,
    treatment: bool,
    *,
    synthesis: bool = False,
    stages: list[str] | None = None,
    appraisal_salience_cap: float | None = None,
    schema_synthesis_tau: float = 0.3,
    schema_synthesis_min_supports: int = 3,
    add_only: bool = False,
    interference_entity_aware: bool = False,
    interference_entity_overlap_min: float = 0.5,
    schema_promote_threshold: int | None = None,
) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    if treatment:
        cfg.consolidation = ConsolidationConfig(
            schedule="manual",
            window_hours=24 * 999,  # huge window, captures all events
            stages=stages,
            schema_family_share=RECIPE["schema_family_share"],
            schema_family_tau=RECIPE["schema_family_tau"],
            schema_family_fragmentation_max=RECIPE[
                "schema_family_fragmentation_max"
            ],
            schema_family_contamination_max=RECIPE[
                "schema_family_contamination_max"
            ],
            add_only=add_only,
            schema_synthesis_enabled=synthesis,
            schema_synthesis_tau=schema_synthesis_tau,
            schema_synthesis_min_supports=schema_synthesis_min_supports,
            appraisal_salience_cap=appraisal_salience_cap,
            interference_entity_aware=interference_entity_aware,
            interference_entity_overlap_min=interference_entity_overlap_min,
        )
        if schema_promote_threshold is not None:
            cfg.consolidation.schema_promote_threshold = schema_promote_threshold
    elif synthesis:
        # §94c baseline-with-synthesis: gate disabled, synthesizer
        # populates SCHEMA table — isolates the gate's effect from
        # the synthesis-on-vs-off confound (mirrors xs §94 design).
        cfg.consolidation = ConsolidationConfig(
            schedule="manual",
            window_hours=24 * 999,
            schema_synthesis_enabled=True,
            schema_synthesis_tau=schema_synthesis_tau,
            schema_synthesis_min_supports=schema_synthesis_min_supports,
        )
    return cfg


def _run_arm(
    sample,
    *,
    treatment: bool,
    embedder,
    k: int,
    synthesis: bool = False,
    stages: list[str] | None = None,
    appraisal_salience_cap: float | None = None,
    schema_synthesis_tau: float = 0.3,
    schema_synthesis_min_supports: int = 3,
    add_only: bool = False,
    interference_entity_aware: bool = False,
    interference_entity_overlap_min: float = 0.5,
    schema_promote_threshold: int | None = None,
    reports_out: list | None = None,
) -> list[dict]:
    """Returns one per-QA record for this sample under the chosen arm."""
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_config(
            tmp, treatment=treatment, synthesis=synthesis, stages=stages,
            appraisal_salience_cap=appraisal_salience_cap,
            schema_synthesis_tau=schema_synthesis_tau,
            schema_synthesis_min_supports=schema_synthesis_min_supports,
            add_only=add_only,
            interference_entity_aware=interference_entity_aware,
            interference_entity_overlap_min=interference_entity_overlap_min,
            schema_promote_threshold=schema_promote_threshold,
        )
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if embedder is not None
            else Engram(config=cfg)
        )
        try:
            _ = _ingest(eng, sample)
            # §94c: dual-write. _ingest only does eng.remember(), which
            # makes content searchable but doesn't push events through
            # EventReplay. Without capture(), the synthesizer (and any
            # consolidation stage that reads the event stream) sees no
            # facts and is vacuous. We capture in BOTH arms so the
            # baseline-vs-treatment contrast cleanly isolates the
            # consolidate(window=999d) call rather than confounding
            # with capture-on-vs-off.
            from evals.locomo_adapter import _tag
            for sess in sample.sessions:
                sid = sess["id"]
                for turn in sess["turns"]:
                    text = (turn.get("content") or "").strip()
                    if not text:
                        continue
                    speaker = turn.get("speaker") or "user"
                    eng.capture(_tag(sid, f"{speaker}: {text}"))
            if treatment:
                # Flush every buffered event through the schema-family
                # gate. window=999d means "everything we've seen this
                # session", which is the right semantics for an offline
                # benchmark replay.
                try:
                    rep = eng.consolidate(window="999d")
                    if reports_out is not None:
                        reports_out.append(rep)
                except Exception as exc:  # pragma: no cover - diagnostic
                    out.append({"_consolidation_error": repr(exc)})
            for q in sample.qa:
                gold = set(q.evidence_sessions)
                if not gold:
                    continue
                results = eng.recall(q.question, limit=k)
                sids = [_session_id_of(r) for r in results]
                rank = 0
                for i, s in enumerate(sids, start=1):
                    if s in gold:
                        rank = i
                        break
                # §95 — pair_recall@k semantics. For LoCoMo,
                # `evidence_sessions` may list multiple gold sessions
                # (multi-hop questions). pair_recall@k = 1 iff *all*
                # gold sessions are present in the top-k retrieved set;
                # gold_recall@k is the *fraction* of gold sessions
                # recovered. The §94c +h@1 number doesn't see the
                # displacement cost on multi-hop questions — these
                # two metrics do.
                retrieved_set = {s for s in sids if s is not None}
                covered = gold & retrieved_set
                pair_recall_at_k = 1 if gold.issubset(retrieved_set) else 0
                gold_recall_at_k = (len(covered) / len(gold)) if gold else 0.0
                out.append({
                    "sample_id": sample.sample_id,
                    "category": q.category,
                    "question": q.question,
                    "gold_sessions": sorted(gold),
                    "n_gold": len(gold),
                    "rank": rank,
                    "hit_at_1": 1 if rank == 1 else 0,
                    "hit_at_k": 1 if (0 < rank <= k) else 0,
                    "pair_recall_at_k": pair_recall_at_k,
                    "gold_recall_at_k": round(gold_recall_at_k, 6),
                    "reciprocal_rank": (1.0 / rank) if rank > 0 else 0.0,
                })
        finally:
            eng.close()
    return out


def run_recall_lift(
    dataset_path: str | os.PathLike,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = False,
    stages: list[str] | None = None,
    appraisal_salience_cap: float | None = None,
    schema_synthesis_tau: float = 0.3,
    schema_synthesis_min_supports: int = 3,
    add_only: bool = False,
    interference_entity_aware: bool = False,
    interference_entity_overlap_min: float = 0.5,
    schema_promote_threshold: int | None = None,
) -> dict:
    samples = load_locomo(dataset_path, max_instances=max_instances)
    if not samples:
        return {"error": "no samples", "n_samples": 0}

    if embedder_name == "hashtrigram":
        from engram.providers.embeddings import HashTrigramEmbeddingProvider
        embedder = HashTrigramEmbeddingProvider(dimension=256)
        emb_label = "HashTrigram-256"
    elif embedder_name in ("st", "minilm", "sentence_transformer"):
        from engram.providers.embeddings import SentenceTransformerProvider
        embedder = SentenceTransformerProvider()
        emb_label = "SentenceTransformer-MiniLM-384"
    elif embedder_name is None:
        embedder = None
        emb_label = "none(BM25-only)"
    else:
        raise ValueError(f"unknown embedder: {embedder_name!r}")

    baseline_rows: list[dict] = []
    treatment_rows: list[dict] = []
    consolidation_errors: list[str] = []
    treatment_reports: list = []
    t0 = time.monotonic()
    for sample in samples:
        baseline_rows.extend(_run_arm(
            sample, treatment=False, embedder=embedder, k=k, synthesis=synthesis,
            stages=stages, appraisal_salience_cap=appraisal_salience_cap,
            schema_synthesis_tau=schema_synthesis_tau,
            schema_synthesis_min_supports=schema_synthesis_min_supports,
            add_only=add_only,
            interference_entity_aware=interference_entity_aware,
            interference_entity_overlap_min=interference_entity_overlap_min,
            schema_promote_threshold=schema_promote_threshold,
        ))
        for r in _run_arm(
            sample, treatment=True, embedder=embedder, k=k, synthesis=synthesis,
            stages=stages, appraisal_salience_cap=appraisal_salience_cap,
            schema_synthesis_tau=schema_synthesis_tau,
            schema_synthesis_min_supports=schema_synthesis_min_supports,
            add_only=add_only,
            interference_entity_aware=interference_entity_aware,
            interference_entity_overlap_min=interference_entity_overlap_min,
            schema_promote_threshold=schema_promote_threshold,
            reports_out=treatment_reports,
        ):
            if "_consolidation_error" in r:
                consolidation_errors.append(r["_consolidation_error"])
            else:
                treatment_rows.append(r)
    wall = time.monotonic() - t0

    # Pair on (sample_id, question). LoCoMo questions are unique per
    # sample, so this is well-defined.
    bkey = lambda r: (r["sample_id"], r["question"])
    bmap = {bkey(r): r for r in baseline_rows}
    pairs: list[dict] = []
    for tr in treatment_rows:
        b = bmap.get(bkey(tr))
        if b is None:
            continue
        pairs.append({
            "sample_id": tr["sample_id"],
            "question": tr.get("question"),
            "category": tr["category"],
            "gold_sessions": tr["gold_sessions"],
            "n_gold": tr.get("n_gold", len(tr["gold_sessions"])),
            "baseline_rank": b["rank"],
            "treatment_rank": tr["rank"],
            "baseline_h1": b["hit_at_1"],
            "treatment_h1": tr["hit_at_1"],
            "baseline_hk": b["hit_at_k"],
            "treatment_hk": tr["hit_at_k"],
            "baseline_prk": b.get("pair_recall_at_k", 0),
            "treatment_prk": tr.get("pair_recall_at_k", 0),
            "baseline_grk": b.get("gold_recall_at_k", 0.0),
            "treatment_grk": tr.get("gold_recall_at_k", 0.0),
            "baseline_rr": b["reciprocal_rank"],
            "treatment_rr": tr["reciprocal_rank"],
            "delta_h1": tr["hit_at_1"] - b["hit_at_1"],
            "delta_hk": tr["hit_at_k"] - b["hit_at_k"],
            "delta_prk": tr.get("pair_recall_at_k", 0) - b.get("pair_recall_at_k", 0),
            "delta_grk": tr.get("gold_recall_at_k", 0.0) - b.get("gold_recall_at_k", 0.0),
            "delta_rr": tr["reciprocal_rank"] - b["reciprocal_rank"],
        })

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    per_cat: dict[str, dict[str, float]] = {}
    for p in pairs:
        c = p["category"]
        per_cat.setdefault(c, {"n": 0, "dh1": 0, "dhk": 0, "dprk": 0,
                               "dgrk": 0.0, "drr": 0.0})
        per_cat[c]["n"] += 1
        per_cat[c]["dh1"] += p["delta_h1"]
        per_cat[c]["dhk"] += p["delta_hk"]
        per_cat[c]["dprk"] += p["delta_prk"]
        per_cat[c]["dgrk"] += p["delta_grk"]
        per_cat[c]["drr"] += p["delta_rr"]
    for c, v in per_cat.items():
        n = v["n"] or 1
        v["mean_delta_h1"] = round(v["dh1"] / n, 4)
        v["mean_delta_hk"] = round(v["dhk"] / n, 4)
        v["mean_delta_prk"] = round(v["dprk"] / n, 4)
        v["mean_delta_grk"] = round(v["dgrk"] / n, 4)
        v["mean_delta_rr"] = round(v["drr"] / n, 4)

    # §95 — multi-hop subset (n_gold >= 2). pair_recall is only
    # informative when the question has multiple gold sessions.
    multi = [p for p in pairs if p["n_gold"] >= 2]

    # §4.6 churn proxy — aggregate schema-create counts across treatment
    # consolidation reports. Each sample produces one report; sum schemas
    # newly created and report mean per-sample.
    churn = {
        "n_treatment_reports": len(treatment_reports),
        "schemas_created_total": sum(
            int(r.state_transitions.get("schemas", 0)) for r in treatment_reports
        ),
    }
    if treatment_reports:
        churn["schemas_created_mean_per_sample"] = round(
            churn["schemas_created_total"] / len(treatment_reports), 4
        )
    else:
        churn["schemas_created_mean_per_sample"] = 0.0

    return {
        "recipe": dict(RECIPE),
        "synthesis": synthesis,
        "stages": stages,
        "schema_promote_threshold": schema_promote_threshold,
        "add_only": add_only,
        "interference_entity_aware": interference_entity_aware,
        "interference_entity_overlap_min": interference_entity_overlap_min,
        "appraisal_salience_cap": appraisal_salience_cap,
        "schema_synthesis_tau": schema_synthesis_tau,
        "schema_synthesis_min_supports": schema_synthesis_min_supports,
        "n_samples": len(samples),
        "n_pairs": len(pairs),
        "k": k,
        "embedder": emb_label,
        "wall_seconds": round(wall, 2),
        "baseline": {
            "session_hit_at_1": _mean([p["baseline_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["baseline_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["baseline_prk"] for p in pairs]),
            "gold_recall_at_k": _mean([p["baseline_grk"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["baseline_rr"] for p in pairs]),
        },
        "treatment": {
            "session_hit_at_1": _mean([p["treatment_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["treatment_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["treatment_prk"] for p in pairs]),
            "gold_recall_at_k": _mean([p["treatment_grk"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["treatment_rr"] for p in pairs]),
        },
        "delta": {
            "session_hit_at_1": _mean([p["delta_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["delta_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["delta_prk"] for p in pairs]),
            "gold_recall_at_k": _mean([p["delta_grk"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["delta_rr"] for p in pairs]),
        },
        "multi_hop": {
            "n_pairs": len(multi),
            "baseline_pair_recall_at_k": _mean([p["baseline_prk"] for p in multi]),
            "treatment_pair_recall_at_k": _mean([p["treatment_prk"] for p in multi]),
            "delta_pair_recall_at_k": _mean([p["delta_prk"] for p in multi]),
            "baseline_gold_recall_at_k": _mean([p["baseline_grk"] for p in multi]),
            "treatment_gold_recall_at_k": _mean([p["treatment_grk"] for p in multi]),
            "delta_gold_recall_at_k": _mean([p["delta_grk"] for p in multi]),
        },
        "per_category": per_cat,
        "churn": churn,
        "consolidation_errors": consolidation_errors,
        "per_query_pairs": pairs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("LOCOMO_PATH",
                   "bench/data/locomo10.json"))
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--out", default=None)
    p.add_argument("--synthesis", action="store_true",
                   help="§94c — enable §93 deterministic non-LLM schema synthesis")
    p.add_argument("--stages", default=None,
                   help="§94c-decompose — comma-separated stage names to run "
                        "(replay+persistence are always included). e.g. "
                        "'extraction,fact_extraction,interference'")
    p.add_argument("--add-only", action="store_true",
                   help="§D3 — disable supersede in the interference stage "
                        "(Mem0-style ADD-only ablation).")
    args = p.parse_args()
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out = run_recall_lift(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
        stages=stages,
        add_only=args.add_only,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, out, default=str)
        print(f"[recall_lift] wrote {args.out}")


if __name__ == "__main__":
    main()

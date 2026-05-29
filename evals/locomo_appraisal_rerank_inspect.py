"""§94c-decompose-suffix-CI follow-up — characterize the `appraisal` mover.

§94c-decompose-suffix-CI showed that of the 7 stages bundled into the
S6→S7 jump, only `appraisal` produces non-zero per-pair diffs — and the
pattern is the H1↑ / HK↓ / grk↓ "displacement" signature: appraisal
re-ranks one gold session up to rank 1 (Δh@1 +0.0133, n.s. but largest
non-zero) while pushing other gold sessions out of the top-k window
(Δh@k −0.0100, Δgrk −0.0075 ★).

This driver answers the natural follow-up: *which retrievals are
moving, and what's the salience profile of the items that displace
them?* For each LoCoMo question we run two arms (S6, S6+appraisal),
record the full top-k retrieved set with per-memory salience and
Scherer appraisal scores, and emit per-question rank deltas plus —
for questions where appraisal hurts the gold rank — the salience
profile of the displacing item that took the slot.

Outputs:
  - per-question records with (rank_a, rank_b, gold_session_set,
    top_k_a, top_k_b, displacing_item) where each top_k entry carries
    {session_id, score, salience, rel, nov, gc}.
  - aggregated movement summary binned by category.
  - "salience deciles" of displacing items vs gold items — confirms
    or refutes the "appraisal over-rewards opening-turn salience"
    hypothesis.

Usage:
    python -m evals.locomo_appraisal_rerank_inspect \\
        --dataset bench/data/locomo10.json \\
        --max-instances 2 \\
        --out bench/results/locomo_appraisal_rerank_inspect.json \\
        --md-out bench/results/locomo_appraisal_rerank_inspect.md
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

from evals.locomo_adapter import (
    _ingest,
    _session_id_of,
    _tag,
    load_locomo,
)
from evals.locomo_recall_lift import _build_config
from evals.locomo_recall_lift_decompose_ci import SUBSET_PRESETS
from evals.io_utils import atomic_write_json, atomic_write_text

# Anchor + probe.
S6_NAME = "S6_+merge_persist"
PROBE_STAGE = "appraisal"


def _stages_with(stage: str | None) -> list[str]:
    base = list(SUBSET_PRESETS[S6_NAME] or [])
    if stage and stage not in base:
        base.append(stage)
    return base


def _topk_record(r) -> dict:
    """Extract the fields we need to characterize an appraisal re-rank."""
    mem = getattr(r, "memory", r)
    appr = getattr(mem, "appraisal", None)
    return {
        "memory_id": getattr(mem, "id", None),
        "session_id": _session_id_of(r),
        "score": round(float(getattr(r, "score", 0.0)), 6),
        "salience": round(float(getattr(mem, "salience", 0.0)), 6),
        "rel": round(float(getattr(appr, "relevance", 0.0) if appr else 0.0), 6),
        "nov": round(float(getattr(appr, "novelty", 0.0) if appr else 0.0), 6),
        "gc": round(float(getattr(appr, "goal_conduciveness", 0.0) if appr else 0.0), 6),
        "content_head": (getattr(mem, "content", "") or "")[:80],
    }


def _run_arm_with_topk(
    samples,
    *,
    stages: list[str],
    embedder,
    k: int,
) -> list[dict]:
    """Run one arm and capture top-k metadata for every gold question."""
    rows: list[dict] = []
    for sample in samples:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _build_config(tmp, treatment=True, stages=stages)
            from engram import Engram
            eng = (
                Engram(config=cfg, embeddings=embedder)
                if embedder is not None
                else Engram(config=cfg)
            )
            try:
                _ingest(eng, sample)
                # Dual-write: capture for the consolidation pipeline.
                for sess in sample.sessions:
                    sid = sess["id"]
                    for turn in sess["turns"]:
                        text = (turn.get("content") or "").strip()
                        if not text:
                            continue
                        speaker = turn.get("speaker") or "user"
                        eng.capture(_tag(sid, f"{speaker}: {text}"))
                try:
                    eng.consolidate(window="999d")
                except Exception:
                    pass
                for q in sample.qa:
                    gold = set(q.evidence_sessions)
                    if not gold:
                        continue
                    results = eng.recall(q.question, limit=k)
                    topk = [_topk_record(r) for r in results]
                    sids = [t["session_id"] for t in topk]
                    rank = 0
                    for i, s in enumerate(sids, start=1):
                        if s in gold:
                            rank = i
                            break
                    retrieved_set = {s for s in sids if s is not None}
                    covered = gold & retrieved_set
                    rows.append({
                        "sample_id": sample.sample_id,
                        "category": q.category,
                        "question": q.question,
                        "gold_sessions": sorted(gold),
                        "n_gold": len(gold),
                        "rank": rank,
                        "covered": sorted(covered),
                        "n_covered": len(covered),
                        "topk": topk,
                    })
            finally:
                eng.close()
    return rows


def _bin(rank_a: int, rank_b: int, k: int) -> str:
    """Categorize the per-question rank movement A→B."""
    in_a = 0 < rank_a <= k
    in_b = 0 < rank_b <= k
    if rank_a == 1 and rank_b == 1:
        return "stable_rank1"
    if rank_a == 1 and rank_b != 1:
        return "lost_rank1"
    if rank_a != 1 and rank_b == 1:
        return "gained_rank1"
    if not in_a and in_b:
        return "entered_topk"
    if in_a and not in_b:
        return "left_topk"
    if in_a and in_b and rank_b < rank_a:
        return "improved_within_topk"
    if in_a and in_b and rank_b > rank_a:
        return "worsened_within_topk"
    if not in_a and not in_b:
        return "absent_both"
    return "stable_within_topk"


def _displacing(topk_a: list[dict], topk_b: list[dict], gold: set[str]) -> dict | None:
    """If S6+appraisal pushed a non-gold item to rank 1 and a gold item
    fell out, return the displacing rank-1 item from B and the displaced
    gold from A's top result (if any).
    """
    if not topk_a or not topk_b:
        return None
    b1 = topk_b[0]
    if b1["session_id"] in gold:
        return None  # rank-1 is gold under treatment, nothing displaced
    # Try to identify the gold item that moved out of (or down from) rank 1
    a_gold = next((t for t in topk_a if t["session_id"] in gold), None)
    return {
        "displacing": b1,
        "displaced_gold_in_a": a_gold,
        "salience_gap": (
            round(b1["salience"] - (a_gold["salience"] if a_gold else 0.0), 6)
        ),
    }


def _aggregate(per_q: list[dict]) -> dict:
    """Bucket per-question diffs and surface the salience-gap distribution."""
    buckets: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    salience_gaps: list[float] = []
    rel_gaps: list[float] = []
    displacing_saliences: list[float] = []
    gold_saliences: list[float] = []
    for q in per_q:
        b = q["movement_bin"]
        buckets[b] = buckets.get(b, 0) + 1
        by_category.setdefault(q["category"], {})
        by_category[q["category"]][b] = by_category[q["category"]].get(b, 0) + 1
        d = q.get("displacing")
        if d:
            salience_gaps.append(d["salience_gap"])
            displacing_saliences.append(d["displacing"]["salience"])
            if d.get("displaced_gold_in_a"):
                gold_saliences.append(d["displaced_gold_in_a"]["salience"])
                rel_gaps.append(
                    d["displacing"]["rel"] - d["displaced_gold_in_a"]["rel"]
                )

    def _summ(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        xs_s = sorted(xs)
        return {
            "n": len(xs),
            "mean": round(statistics.fmean(xs), 6),
            "median": round(statistics.median(xs), 6),
            "min": round(xs_s[0], 6),
            "p25": round(xs_s[len(xs) * 1 // 4], 6),
            "p75": round(xs_s[min(len(xs) - 1, len(xs) * 3 // 4)], 6),
            "max": round(xs_s[-1], 6),
        }

    return {
        "movement_bins_overall": buckets,
        "movement_bins_by_category": by_category,
        "salience_gap_displacing_minus_gold": _summ(salience_gaps),
        "rel_gap_displacing_minus_gold": _summ(rel_gaps),
        "displacing_item_salience": _summ(displacing_saliences),
        "displaced_gold_salience": _summ(gold_saliences),
    }


def run_appraisal_rerank_inspect(
    dataset_path: str,
    *,
    max_instances: int = 2,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
) -> dict:
    t0 = time.monotonic()
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

    arm_a = _run_arm_with_topk(
        samples, stages=_stages_with(None), embedder=embedder, k=k,
    )
    arm_b = _run_arm_with_topk(
        samples, stages=_stages_with(PROBE_STAGE), embedder=embedder, k=k,
    )
    a_map = {(r["sample_id"], r["question"]): r for r in arm_a}

    per_q: list[dict] = []
    for r in arm_b:
        a = a_map.get((r["sample_id"], r["question"]))
        if a is None:
            continue
        gold = set(r["gold_sessions"])
        bin_ = _bin(a["rank"], r["rank"], k)
        disp = _displacing(a["topk"], r["topk"], gold)
        per_q.append({
            "sample_id": r["sample_id"],
            "question": r["question"],
            "category": r["category"],
            "gold_sessions": r["gold_sessions"],
            "n_gold": r["n_gold"],
            "rank_a": a["rank"],
            "rank_b": r["rank"],
            "delta_rank": (
                (1.0 / r["rank"] if r["rank"] else 0.0)
                - (1.0 / a["rank"] if a["rank"] else 0.0)
            ),
            "n_covered_a": a["n_covered"],
            "n_covered_b": r["n_covered"],
            "delta_n_covered": r["n_covered"] - a["n_covered"],
            "movement_bin": bin_,
            "displacing": disp,
            "topk_a": a["topk"],
            "topk_b": r["topk"],
        })

    return {
        "dataset_path": str(dataset_path),
        "max_instances": max_instances,
        "k": k,
        "embedder": emb_label,
        "anchor": S6_NAME,
        "probe_stage": PROBE_STAGE,
        "n_questions": len(per_q),
        "aggregate": _aggregate(per_q),
        "per_question": per_q,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def render_markdown(report: dict) -> str:
    agg = report["aggregate"]
    lines = []
    lines.append(
        f"### §94c-decompose-suffix-CI follow-up — `{report['probe_stage']}` "
        f"re-rank inspector "
        f"(max_instances={report['max_instances']}, k={report['k']}, "
        f"embedder={report['embedder']}, n_questions={report['n_questions']})"
    )
    lines.append("")
    lines.append(
        f"Anchor: **{report['anchor']}** vs `{report['anchor']} + "
        f"{report['probe_stage']}`. Each row is a per-question rank "
        f"movement bin (A=anchor, B=anchor+probe). Salience gap = "
        f"displacing rank-1 item's salience minus the displaced gold's "
        f"salience under A."
    )
    lines.append("")
    lines.append("| movement_bin | count |")
    lines.append("| --- | ---: |")
    bins = sorted(agg["movement_bins_overall"].items(), key=lambda x: -x[1])
    for b, c in bins:
        lines.append(f"| `{b}` | {c} |")
    lines.append("")
    lines.append("**Salience gap (displacing − displaced_gold)** — "
                 "if positive, appraisal rewards the wrong item.")
    g = agg["salience_gap_displacing_minus_gold"]
    if g["n"]:
        lines.append(
            f"  n={g['n']}  mean={g['mean']:+.4f}  median={g['median']:+.4f}  "
            f"min={g['min']:+.4f}  p25={g['p25']:+.4f}  p75={g['p75']:+.4f}  "
            f"max={g['max']:+.4f}"
        )
    else:
        lines.append("  (no displacing events on this fixture)")
    lines.append("")
    r = agg["rel_gap_displacing_minus_gold"]
    if r["n"]:
        lines.append(
            f"**Scherer relevance gap (displacing − gold).** "
            f"n={r['n']}  mean={r['mean']:+.4f}  median={r['median']:+.4f}"
        )
    lines.append("")

    # Category breakdown — does lost_rank1 cluster in any one category?
    by_cat = agg.get("movement_bins_by_category", {})
    if by_cat:
        # Collect all bins seen across categories so columns are stable.
        all_bins: list[str] = []
        for cat_bins in by_cat.values():
            for b in cat_bins:
                if b not in all_bins:
                    all_bins.append(b)
        # Order columns by overall frequency desc.
        overall = agg.get("movement_bins_overall", {})
        all_bins.sort(key=lambda b: -overall.get(b, 0))

        lines.append("**Movement bins by category** — does any bin "
                     "(esp. `lost_rank1`) cluster in one category?")
        lines.append("")
        header = "| category | total | " + " | ".join(f"`{b}`" for b in all_bins) + " |"
        sep = "| --- | ---: | " + " | ".join("---:" for _ in all_bins) + " |"
        lines.append(header)
        lines.append(sep)
        # Sort categories by total count desc.
        cat_totals = {c: sum(b.values()) for c, b in by_cat.items()}
        for cat in sorted(by_cat, key=lambda c: -cat_totals[c]):
            row_bins = by_cat[cat]
            cells = [str(row_bins.get(b, 0)) for b in all_bins]
            lines.append(f"| `{cat}` | {cat_totals[cat]} | " + " | ".join(cells) + " |")
        lines.append("")
        # Highlight: lost_rank1 vs gained_rank1 per category.
        lost_per_cat = {c: by_cat[c].get("lost_rank1", 0) for c in by_cat}
        gained_per_cat = {c: by_cat[c].get("gained_rank1", 0) for c in by_cat}
        if any(lost_per_cat.values()) or any(gained_per_cat.values()):
            lines.append(
                "**Lost vs gained rank-1 by category** "
                "(asymmetry = displacing − surfacing):"
            )
            lines.append("")
            lines.append("| category | lost_rank1 | gained_rank1 | net |")
            lines.append("| --- | ---: | ---: | ---: |")
            for cat in sorted(by_cat, key=lambda c: -(lost_per_cat[c] + gained_per_cat[c])):
                lost = lost_per_cat[cat]
                gained = gained_per_cat[cat]
                if lost == 0 and gained == 0:
                    continue
                net = gained - lost
                lines.append(f"| `{cat}` | {lost} | {gained} | {net:+d} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get(
        "LOCOMO_PATH", "bench/data/locomo10.json"))
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_appraisal_rerank_inspect(
        args.dataset,
        max_instances=args.max_instances,
        k=args.k,
        embedder_name=args.embedder,
    )
    print(f"§94c-appraisal-inspect  wall={rep.get('wall_seconds', '?')}s  "
          f"n={rep.get('n_questions', 0)}")
    bins = rep.get("aggregate", {}).get("movement_bins_overall", {})
    for b, c in sorted(bins.items(), key=lambda x: -x[1]):
        print(f"  {b:<25s} {c}")
    g = rep.get("aggregate", {}).get("salience_gap_displacing_minus_gold", {})
    if g.get("n"):
        print(f"  salience_gap n={g['n']} mean={g['mean']:+.4f} "
              f"median={g['median']:+.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[appraisal-inspect] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep))
        print(f"[appraisal-inspect] wrote {args.md_out}")


if __name__ == "__main__":
    main()

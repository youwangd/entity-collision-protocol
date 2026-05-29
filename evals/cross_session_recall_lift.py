"""§91 — End-to-end recall-lift on the *cross-session* synthetic corpus.

§90 found the §87 schema-family gate operationally inert on LoCoMo because
LoCoMo evidence is session-localized: BM25 already finds the right session
for nearly every QA, leaving no headroom for cluster-aware retrieval. This
driver answers the natural follow-up: when evidence *crosses* sessions —
i.e. the gold answer is supported by paraphrased planted facts in two
distinct sessions — does flipping the §87 gate produce a positive Δ?

Design
------
Per-fact paired comparison on a synthetic corpus from
``evals.synthetic.generate_cross_session_dataset``. For each query:

  - gold = {sess_a, sess_b}
  - baseline arm: default Config + ingest + recall.
  - treatment arm: Config with §87 recipe + ingest + ``consolidate(window=999d)``
    + recall.

Metrics
-------
  * ``session_hit_at_1``  — top-1 hits any gold session.
  * ``session_hit_at_k``  — top-k hits any gold session.
  * ``pair_recall_at_k``  — both gold sessions in top-k. This is the
    most diagnostic metric for cross-session evidence: a session-local
    BM25 retriever can satisfy hit@1 by surfacing only one half;
    pair_recall@k is what cluster-aware retrieval can plausibly help.
  * ``mean_reciprocal_rank`` of the *first* gold session.

Pure: deterministic given seed. No clocks participate in scoring.
Intent: deliver a "yes-it-works-when-the-corpus-needs-it" data point
next to §90's "no-effect-on-LoCoMo" finding.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time

from engram import Engram, Config
from engram.core.config import ConsolidationConfig

from evals.synthetic import generate_cross_session_dataset
from evals.io_utils import atomic_write_json


# §87 recommended recipe (paper-locked).
RECIPE = {
    "schema_family_share": 0.75,
    "schema_family_tau": 0.20,
    "schema_family_fragmentation_max": 0.9021,
    "schema_family_contamination_max": 0.15,
}

_TAG_PREFIX = "[xs_session="
_TAG_SUFFIX = "] "


def _tag(sess: int, content: str) -> str:
    return f"{_TAG_PREFIX}{sess}{_TAG_SUFFIX}{content}"


def _untag(content: str) -> int | None:
    if content.startswith(_TAG_PREFIX):
        end = content.find(_TAG_SUFFIX, len(_TAG_PREFIX))
        if end != -1:
            try:
                return int(content[len(_TAG_PREFIX):end])
            except ValueError:
                return None
    return None


def _build_config(path: str, treatment: bool, *, synthesis: bool = False) -> Config:
    cfg = Config(path=path)
    cfg.security.max_events_per_minute = 0
    if treatment:
        cfg.consolidation = ConsolidationConfig(
            schedule="manual",
            window_hours=24 * 999,
            stages=None,
            schema_family_share=RECIPE["schema_family_share"],
            schema_family_tau=RECIPE["schema_family_tau"],
            schema_family_fragmentation_max=RECIPE[
                "schema_family_fragmentation_max"
            ],
            schema_family_contamination_max=RECIPE[
                "schema_family_contamination_max"
            ],
            add_only=False,
            schema_synthesis_enabled=synthesis,
            schema_synthesis_tau=0.3,
            schema_synthesis_min_supports=3,
        )
    elif synthesis:
        # §94 baseline-with-synthesis variant: keep §87 gate disabled
        # but run the synthesizer so we measure the gate's effect
        # *given* a populated schema table (rather than confounding
        # with synthesis-on-vs-off).
        cfg.consolidation = ConsolidationConfig(
            schedule="manual",
            window_hours=24 * 999,
            schema_synthesis_enabled=True,
            schema_synthesis_tau=0.3,
            schema_synthesis_min_supports=3,
        )
    return cfg


def _run_arm(ds, *, treatment: bool, embedder, k: int, synthesis: bool = False) -> list[dict]:
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _build_config(tmp, treatment=treatment, synthesis=synthesis)
        eng = (
            Engram(config=cfg, embeddings=embedder)
            if embedder is not None
            else Engram(config=cfg)
        )
        try:
            for content, meta in ds.memories:
                tagged = _tag(int(meta["session"]), content)
                # §94: dual write — remember() makes it searchable;
                # capture() lets EventReplay process it for schema
                # synthesis. The §91 driver was remember-only and so
                # vacuously had events_processed=0.
                eng.remember(tagged)
                if synthesis or treatment:
                    eng.capture(tagged)
            if treatment:
                try:
                    eng.consolidate(window="999d")
                except Exception as exc:  # pragma: no cover
                    out.append({"_consolidation_error": repr(exc)})
            for q in ds.queries:
                # Extract gold sessions from query.tags
                gold = set()
                for t in q.tags:
                    if t.startswith("sess_a=") or t.startswith("sess_b="):
                        gold.add(int(t.split("=", 1)[1]))
                if not gold:
                    continue
                results = eng.recall(q.text, limit=k)
                sids: list[int | None] = []
                for r in results:
                    mem = getattr(r, "memory", r)
                    sids.append(_untag(getattr(mem, "content", "") or ""))
                # First-gold-hit rank
                rank = 0
                for i, s in enumerate(sids, start=1):
                    if s in gold:
                        rank = i
                        break
                # Both-gold-hit (pair recall)
                hit_both = 1 if gold.issubset(set(s for s in sids if s is not None)) else 0
                out.append({
                    "pair_id": next((t.split("=", 1)[1] for t in q.tags
                                     if t.startswith("pair_id=")), None),
                    "tag": q.tags[0] if q.tags else "",
                    "gold": sorted(gold),
                    "rank": rank,
                    "hit_at_1": 1 if rank == 1 else 0,
                    "hit_at_k": 1 if (0 < rank <= k) else 0,
                    "pair_recall_at_k": hit_both,
                    "reciprocal_rank": (1.0 / rank) if rank > 0 else 0.0,
                })
        finally:
            eng.close()
    return out


def run_cross_session_recall_lift(
    *,
    n_facts: int = 60,
    n_sessions: int = 10,
    distractors_per_session: int = 10,
    seed: int = 42,
    k: int = 10,
    embedder_name: str | None = "hashtrigram",
    synthesis: bool = False,
) -> dict:
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

    ds = generate_cross_session_dataset(
        n_facts=n_facts,
        n_sessions=n_sessions,
        distractors_per_session=distractors_per_session,
        seed=seed,
    )

    t0 = time.monotonic()
    baseline = _run_arm(ds, treatment=False, embedder=embedder, k=k, synthesis=synthesis)
    treatment_raw = _run_arm(ds, treatment=True, embedder=embedder, k=k, synthesis=synthesis)
    consolidation_errors: list[str] = []
    treatment: list[dict] = []
    for r in treatment_raw:
        if "_consolidation_error" in r:
            consolidation_errors.append(r["_consolidation_error"])
        else:
            treatment.append(r)
    wall = time.monotonic() - t0

    bmap = {r["pair_id"]: r for r in baseline}
    pairs: list[dict] = []
    for tr in treatment:
        b = bmap.get(tr["pair_id"])
        if b is None:
            continue
        pairs.append({
            "pair_id": tr["pair_id"],
            "tag": tr["tag"],
            "baseline_rank": b["rank"],
            "treatment_rank": tr["rank"],
            "delta_h1": tr["hit_at_1"] - b["hit_at_1"],
            "delta_hk": tr["hit_at_k"] - b["hit_at_k"],
            "delta_pair_recall_k": tr["pair_recall_at_k"] - b["pair_recall_at_k"],
            "delta_rr": tr["reciprocal_rank"] - b["reciprocal_rank"],
            "baseline_h1": b["hit_at_1"],
            "treatment_h1": tr["hit_at_1"],
            "baseline_hk": b["hit_at_k"],
            "treatment_hk": tr["hit_at_k"],
            "baseline_prk": b["pair_recall_at_k"],
            "treatment_prk": tr["pair_recall_at_k"],
            "baseline_rr": b["reciprocal_rank"],
            "treatment_rr": tr["reciprocal_rank"],
        })

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "recipe": dict(RECIPE),
        "synthesis": synthesis,
        "corpus": {
            "n_facts": n_facts,
            "n_sessions": n_sessions,
            "distractors_per_session": distractors_per_session,
            "seed": seed,
            "n_memories": len(ds.memories),
        },
        "k": k,
        "embedder": emb_label,
        "wall_seconds": round(wall, 2),
        "n_pairs": len(pairs),
        "baseline": {
            "session_hit_at_1": _mean([p["baseline_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["baseline_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["baseline_prk"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["baseline_rr"] for p in pairs]),
        },
        "treatment": {
            "session_hit_at_1": _mean([p["treatment_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["treatment_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["treatment_prk"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["treatment_rr"] for p in pairs]),
        },
        "delta": {
            "session_hit_at_1": _mean([p["delta_h1"] for p in pairs]),
            "session_hit_at_k": _mean([p["delta_hk"] for p in pairs]),
            "pair_recall_at_k": _mean([p["delta_pair_recall_k"] for p in pairs]),
            "mean_reciprocal_rank": _mean([p["delta_rr"] for p in pairs]),
        },
        "n_nonzero_pairs": sum(
            1 for p in pairs
            if p["delta_h1"] != 0 or p["delta_hk"] != 0
            or p["delta_pair_recall_k"] != 0 or p["delta_rr"] != 0
        ),
        "consolidation_errors": consolidation_errors,
        "per_pair": pairs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-facts", type=int, default=60)
    p.add_argument("--n-sessions", type=int, default=10)
    p.add_argument("--distractors", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedder", default="hashtrigram",
                   choices=[None, "hashtrigram", "st", "minilm",
                            "sentence_transformer"])
    p.add_argument("--out", default=None)
    p.add_argument("--synthesis", action="store_true",
                   help="§93 deterministic non-LLM schema synthesis on")
    args = p.parse_args()
    out = run_cross_session_recall_lift(
        n_facts=args.n_facts,
        n_sessions=args.n_sessions,
        distractors_per_session=args.distractors,
        seed=args.seed,
        k=args.k,
        embedder_name=args.embedder,
        synthesis=args.synthesis,
    )
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        atomic_write_json(args.out, out, default=str)
        print(f"[xs_recall_lift] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Paired bootstrap CIs for BGE-large vs MiniLM (Δ vector_fusion across embedders).

Pairs per-query records from two embedders' sweep JSONs by query text and
collision degree, then bootstraps the per-cell mean difference of hit@1.

Usage::

    python -m scripts.ec_bge_vs_minilm_ci \
        --bge-glob 'bench/results/ec_bge_large_{tag}_n32_K16.json' \
        --st-glob 'bench/results/ec_sweep_st_{tag}_n32_K16.json' \
        --st-extra 'bench/results/ec_sweep_st_{tag}_n32_K124.json' \
        --tags service,tool,preference,project,technical \
        --out bench/results/ec_bge_vs_minilm_ci.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.bootstrap_ci import _paired_diff_ci


def _load_per_query_by_K(path: Path) -> dict[int, list[dict]]:
    """Return {K: per_query rows from vector_fusion} from a sweep JSON."""
    data = json.loads(path.read_text())
    out = {}
    for row in data.get("sweep", []):
        K = row["collision_degree"]
        out[K] = row["vector_fusion"]["per_query"]
    return out


def _index_by_query(rows: list[dict]) -> dict[str, dict]:
    return {r["query"]: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bge-glob", required=True,
                   help="format string with {tag} for BGE sweep files")
    p.add_argument("--st-glob", required=True,
                   help="format string with {tag} for MiniLM sweep files")
    p.add_argument("--st-extra", default=None,
                   help="optional second format string for MiniLM (covers Ks "
                        "missing from --st-glob); merges keys")
    p.add_argument("--tags", required=True,
                   help="comma-separated tags")
    p.add_argument("--out", required=True)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    out_obj = {"config": {"resamples": args.resamples, "seed": args.seed},
               "tags": {}}

    for tag in tags:
        bge_path = Path(args.bge_glob.format(tag=tag))
        st_path = Path(args.st_glob.format(tag=tag))
        st_by_K = _load_per_query_by_K(st_path)
        if args.st_extra:
            st_extra_path = Path(args.st_extra.format(tag=tag))
            if st_extra_path.exists():
                extra = _load_per_query_by_K(st_extra_path)
                # Don't clobber existing
                for K, pq in extra.items():
                    if K not in st_by_K:
                        st_by_K[K] = pq
        bge_by_K = _load_per_query_by_K(bge_path)

        rows = []
        Ks = sorted(set(bge_by_K) & set(st_by_K))
        print(f"\nTAG: {tag}  Ks={Ks}")
        print(f"{'K':>3} | {'n':>3} | "
              f"{'BGE hit@1':>9} | {'ST hit@1':>9} | "
              f"{'Δ(BGE-ST) hit@1 [CI]':>30}")
        print("-" * 78)
        for K in Ks:
            bge_q = _index_by_query(bge_by_K[K])
            st_q = _index_by_query(st_by_K[K])
            shared = sorted(set(bge_q) & set(st_q))
            if not shared:
                continue
            bge_v = [bge_q[q]["hit_at_1"] for q in shared]
            st_v = [st_q[q]["hit_at_1"] for q in shared]
            mean_bge = sum(bge_v) / len(bge_v)
            mean_st = sum(st_v) / len(st_v)
            m, lo, hi = _paired_diff_ci(bge_v, st_v, args.resamples, args.seed)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"{K:>3} | {len(shared):>3} | "
                  f"{mean_bge:>9.3f} | {mean_st:>9.3f} | "
                  f"{m:>+8.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")
            rows.append({
                "K": K,
                "n_paired": len(shared),
                "bge_hit_at_1_mean": round(mean_bge, 4),
                "st_hit_at_1_mean": round(mean_st, 4),
                "delta_hit_at_1_mean": round(m, 4),
                "delta_hit_at_1_ci_lo": round(lo, 4),
                "delta_hit_at_1_ci_hi": round(hi, 4),
                "ci_excludes_zero": (lo > 0 or hi < 0),
            })
        out_obj["tags"][tag] = rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"\n[ec_bge_vs_minilm_ci] wrote {out_path}")


if __name__ == "__main__":
    main()

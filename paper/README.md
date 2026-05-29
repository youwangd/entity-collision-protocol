# Engram Paper — Working Draft

> Status: skeleton bootstrapped 2026-05-21. Framing FROZEN 2026-05-23
> (two-axis: lexical vs intent-style discriminators). Venue FROZEN
> 2026-05-23: EMNLP 2026 main, deadline ~2026-06-15 (see VENUE.md).
> Reproducibility scaffold landed 2026-05-23 (see REPRODUCIBILITY.md).

## Working title

**Hash trigrams partially serve lexical-discriminator memory retrieval; semantic embedders dominate intent-style retrieval — an entity-collision controlled study.**

## Layout

| file | purpose | status |
|---|---|---|
| `00_abstract.md` | 200-word abstract | locked 2026-05-24 (measurement-first) |
| `10_intro.md` | motivation + 3 contributions | locked 2026-05-24 (measurement-first) |
| `20_related.md` | LongMemEval, LoCoMo, Letta, Mem0, Personize | locked 2026-05-24 (camera-ready citation pass deferred to venue lock) |
| `30_methods.md` | entity-collision protocol + retriever cell + governed-memory §3.7 | locked 2026-05-24 |
| `40_results.md` | headline §4: per-cell CIs, two-axis interp, ops + 1M ingest | locked 2026-05-24 (sliced; appendix split out) |
| `50_discussion.md` | when does vector pay? | locked 2026-05-23 |
| `60_threats.md` | threats to validity | locked 2026-05-23 |
| `70_conclusion.md` | one paragraph | locked 2026-05-23 |
| `A1_appendix_ablations.md` | secondary ablations + falsified-hypothesis log | locked 2026-05-24 (was §A.4.6/§A.4.7/§A.4.9–§A.4.13/§A.4.15-series) |
| `figures/` | symlinks to `bench/results/*.png` | populated |

## Sources of truth

- Headline figure: `bench/results/ec_paper_figure.png` (do **not** edit; regenerate via `python -m evals.entity_collision_plot`).
- Per-cell CIs: `bench/results/ec_sweep_{hash,st}_{tag}_n32_K16_ci.json`.
- LoCoMo per-category CIs: `bench/results/locomo10_{ht,st}_sweep_ci_percat.json`.
- Adaptive-vw null result: SCALE_REPORT §"Adaptive-vw on real LoCoMo".
- Schema lifecycle invariants: `TODO-RESEARCH.md §B`.

## Build

Markdown-first draft. LaTeX export targets the ACL 2023 template
(`acl.sty`) per VENUE.md (EMNLP 2026 main, ~2026-06-15 deadline).

`paper/build.sh` runs the gated pipeline:

| phase | tool | always-on? |
|---|---|---|
| 0 | `scripts/check_anon.py` (review) or `--strict` (camera-ready) | yes |
| 1 | concatenate `00_…70_*.md` → `paper/build/paper.md` | yes |
| 2 | `pandoc -t latex` → `paper/build/paper.tex` | only if pandoc present |
| 3 | `latexmk -pdf` → `paper/build/paper.pdf` | only if latexmk present |

Default mode is `--review` (anon linter green, HTML comments stripped
by pandoc). `--camera-ready` runs the linter in `--strict` mode so even
HTML-comment-only identifying strings are caught before the de-anonymized
author block goes back in.

Anonymization is enforced at export time (see VENUE.md §"Anonymization
checklist"). HTML comments (`<!-- ... -->`) in the numbered files are
stripped by pandoc and are safe places to leave camera-ready TODOs
without leaking into the review PDF.

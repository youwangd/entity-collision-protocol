#!/usr/bin/env bash
# paper/build.sh — review-PDF / camera-ready build pipeline.
#
# Phases (each gated; failure aborts):
#   0. Anonymization linter (paper/VENUE.md mandatory checklist).
#   1. Concatenate numbered sections into a single Markdown master.
#   2. Pandoc → LaTeX (skipped with a notice if pandoc is missing).
#   3. LaTeX → PDF via latexmk (skipped with a notice if latexmk missing).
#
# Modes:
#   --review        (default) strict-strip HTML comments, anon-strip enabled.
#   --camera-ready  keep author block, skip --strict on the linter.
#   --arxiv         like camera-ready but inserts a public author block
#                   (paper/05_authors_arxiv.md) right after the title;
#                   uses the real GitHub link, includes acknowledgements.
#
# Outputs land under paper/build/ which is gitignored.
#
# This script is intentionally pure bash + standard tools. No pip deps.

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/.." && pwd)"
BUILD_DIR="$PAPER_DIR/build"
MODE="review"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --review) MODE="review"; shift ;;
        --camera-ready) MODE="camera-ready"; shift ;;
        --arxiv) MODE="arxiv"; shift ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "build.sh: unknown arg '$1'" >&2
            exit 2
            ;;
    esac
done

SECTIONS=(
    "$PAPER_DIR/00_abstract.md"
    "$PAPER_DIR/10_intro.md"
    "$PAPER_DIR/20_related.md"
    "$PAPER_DIR/30_methods.md"
    "$PAPER_DIR/40_results.md"
    "$PAPER_DIR/50_discussion.md"
    "$PAPER_DIR/75_limitations.md"
    "$PAPER_DIR/A1_appendix_ablations.md"
    "$PAPER_DIR/A2_appendix_security_audits.md"
    "$PAPER_DIR/A3_extended_related.md"
    "$PAPER_DIR/A4_extended_discussion.md"
    "$PAPER_DIR/A5_extended_threats.md"
    "$PAPER_DIR/A6_appendix_traceability.md"
    "$PAPER_DIR/A7_extended_methods.md"
)

# Camera-ready / arxiv: append the AI-disclosure Acknowledgements after §7.
# In review mode this file is intentionally excluded (double-blind).
# arXiv mode also inserts an author block right after the title page.
if [[ "$MODE" == "camera-ready" || "$MODE" == "arxiv" ]]; then
    NEW_SECTIONS=()
    for entry in "${SECTIONS[@]}"; do
        NEW_SECTIONS+=("$entry")
        # Author block sits between abstract and §1 in arXiv builds only.
        if [[ "$MODE" == "arxiv" && "$entry" == *"00_abstract.md" ]]; then
            if [[ -f "$PAPER_DIR/05_authors_arxiv.md" ]]; then
                NEW_SECTIONS+=("$PAPER_DIR/05_authors_arxiv.md")
            else
                echo "build.sh: --arxiv mode requires paper/05_authors_arxiv.md" >&2
                exit 1
            fi
        fi
        if [[ "$entry" == *"77_ethics.md" ]]; then
            NEW_SECTIONS+=("$PAPER_DIR/80_acknowledgements_cameraready.md")
        fi
    done
    SECTIONS=("${NEW_SECTIONS[@]}")
fi

for f in "${SECTIONS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "build.sh: missing section file: $f" >&2
        exit 1
    fi
done

mkdir -p "$BUILD_DIR"

# ── Phase 0: anonymization linter ────────────────────────────────────────
echo "[0/3] anonymization linter (mode=$MODE)"
LINTER_ARGS=()
if [[ "$MODE" == "review" ]]; then
    # In --review we are about to strip comments before pandoc, so HTML
    # comments are safe — default (non-strict) mode mirrors what reviewers
    # will see in the rendered PDF.
    :
elif [[ "$MODE" == "arxiv" ]]; then
    # arXiv: real names go in 05_authors_arxiv.md by design, but the rest
    # of the paper sections must NOT leak corp hostnames or identifying
    # internal infra. Run the linter against everything EXCEPT the author
    # block and the acknowledgements.
    LINTER_TARGETS=()
    for s in "${SECTIONS[@]}"; do
        case "$s" in
            *05_authors_arxiv.md|*80_acknowledgements_cameraready.md) : ;;
            *) LINTER_TARGETS+=("$s") ;;
        esac
    done
    python "$REPO_ROOT/scripts/check_anon.py" "${LINTER_TARGETS[@]}"
    # Phase 0 done early for arxiv; skip the catch-all linter call below.
else
    # Camera-ready: run --strict so even comment-only secrets are caught
    # before the de-anonymized author block goes back in.
    LINTER_ARGS+=(--strict)
fi
if [[ "$MODE" != "arxiv" ]]; then
    python "$REPO_ROOT/scripts/check_anon.py" ${LINTER_ARGS[@]+"${LINTER_ARGS[@]}"} "${SECTIONS[@]}"
fi

# ── Phase 1: concatenate ─────────────────────────────────────────────────
echo "[1/3] concatenate sections → $BUILD_DIR/paper.md"
MASTER="$BUILD_DIR/paper.md"
{
    if [[ "$MODE" == "review" ]]; then
        echo "% Review PDF — author block anonymized per VENUE.md."
        echo
    elif [[ "$MODE" == "arxiv" ]]; then
        echo "% arXiv preprint — author block public; same content as review/camera-ready."
        echo
    else
        echo "% Camera-ready — author block restored."
        echo
    fi
    for f in "${SECTIONS[@]}"; do
        cat "$f"
        echo; echo
    done
} > "$MASTER"

# ── Phase 2: pandoc → latex (optional) ───────────────────────────────────
TEX="$BUILD_DIR/paper.tex"
if command -v pandoc >/dev/null 2>&1; then
    echo "[2/3] pandoc: $MASTER → $TEX"
    PANDOC_ARGS=(-f markdown -t latex --standalone)
    # pandoc strips HTML comments by default, which matches our review-mode
    # convention. Camera-ready uses the same flag — in-section TODO comments
    # never reach the published PDF.
    pandoc "${PANDOC_ARGS[@]}" -o "$TEX" "$MASTER"
else
    echo "[2/3] pandoc not installed — skipping LaTeX export."
    echo "      install: 'sudo dnf install pandoc' or 'apt install pandoc'."
fi

# ── Phase 3: latexmk → pdf (optional) ────────────────────────────────────
if [[ -f "$TEX" ]] && command -v latexmk >/dev/null 2>&1; then
    echo "[3/3] latexmk: $TEX → $BUILD_DIR/paper.pdf"
    (cd "$BUILD_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null)
elif [[ -f "$TEX" ]]; then
    echo "[3/3] latexmk not installed — skipping PDF render."
else
    echo "[3/3] no .tex artifact — skipping PDF render."
fi

echo "build.sh: done (mode=$MODE)."

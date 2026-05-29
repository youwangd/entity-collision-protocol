#!/usr/bin/env python3
"""
Build paper/dist/engram_v0.2_acl.pdf using the official ACL 2-column style.

Pipeline: markdown sections → pandoc (markdown → latex body) → wrap into the
ACL acl_latex.tex skeleton → pdflatex (×2 for refs) → strip metadata.

Runs everything inside the pandoc/latex Docker image so the AL2 cloud desktop
doesn't need pandoc or LaTeX installed locally.

Usage:
    python scripts/build_paper_acl.py [--no-bib] [--keep-tex]

Outputs:
    paper/dist/engram_v0.2_acl.pdf      ACL 2-column rendering, review mode
    paper/dist/engram_v0.2_acl.tex      (with --keep-tex) the wrapped .tex
    paper/build/acl/                    intermediate artifacts (aux, log, etc.)

Page-budget reading: the ACL Industry Track allows 6 pages of body content
+ unlimited appendix + mandatory Limitations. Body = sections 1-7 + 75
Limitations + bibliography. Pages 1-N where the first 'Appendix A' heading
appears = body+limitations page count. Read the build log's "body ends on
page X" line for the count.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO / "paper"
ACL_DIR = PAPER_DIR / "acl"
BUILD_DIR = PAPER_DIR / "build" / "acl"
DIST_DIR = PAPER_DIR / "dist"

# Same SECTIONS order as paper/build.sh — keep in sync if anything changes
BODY_SECTIONS = [
    "00_abstract.md",
    "10_intro.md",
    "20_related.md",
    "30_methods.md",
    "40_results.md",
    "50_discussion.md",
    "75_limitations.md",
    "77_ethics.md",
]
APPENDIX_SECTIONS = [
    "A1_appendix_ablations.md",
    "A2_appendix_security_audits.md",
    "A3_extended_related.md",
    "A4_extended_discussion.md",
    "A5_extended_threats.md",
    "A6_appendix_traceability.md",
    "A7_extended_methods.md",
]

DOCKER_IMAGE = "pandoc-acl:engram"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def docker_run(args: list[str], workdir: str = "/data") -> None:
    """Run a command inside the pandoc/latex image with REPO mounted at /data."""
    import os
    base = [
        "docker", "run", "--rm",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{REPO}:/data",
        "-w", workdir,
        "--entrypoint", args[0],
        DOCKER_IMAGE,
    ] + args[1:]
    run(base)


def md_to_tex(md_path: Path, tex_path: Path) -> None:
    """Convert a single markdown section to a LaTeX body fragment via pandoc.

    The acl_tables.lua filter rewrites every Table element as a
    \\begin{table*}\\begin{tabular}...\\end{tabular}\\end{table*} block, which
    is what ACL's 2-column page mode requires (it rejects longtable).
    It also strips leading "N.M.K " numeric prefixes from headings (so
    LaTeX auto-numbering doesn't collide) and wraps long Code spans in
    \\seqsplit{} for column-break friendliness.
    """
    rel_in = md_path.relative_to(REPO)
    rel_out = tex_path.relative_to(REPO)
    docker_run([
        "pandoc",
        "-f", "markdown+raw_tex+pipe_tables+grid_tables+tex_math_dollars",
        "-t", "latex",
        "--wrap=preserve",
        "--top-level-division=section",
        "--lua-filter=/data/scripts/acl_tables.lua",
        f"/data/{rel_in}",
        "-o", f"/data/{rel_out}",
    ])
    # Post-process: \seqsplit is fragile in moving arguments (\section et al).
    # The Lua filter can't reliably tell when Code is inside a Header (filters
    # run top-down and the Header filter sees the already-rewritten RawInline,
    # not the original Code). So we strip \seqsplit{...} inside any
    # \section{} / \subsection{} / \subsubsection{} / \paragraph{} arg here.
    _strip_seqsplit_from_headings(tex_path)
    # Allow line-breaks inside arXiv identifiers (arXiv:NNNN.NNNNN). In ACL's
    # narrow 2-column layout, the unbreakable token causes wide rivers of
    # whitespace on the line preceding it. Wrap with \seqsplit so xelatex
    # can break between digits if needed. Phase-1 fix; Phase-2 will replace
    # these inline tags with proper \citep{} calls + a References section.
    _seqsplit_arxiv_ids(tex_path)
    # Promote wide multi-panel figures (e.g. ec_paper_figure.png — 3 side-by-side
    # encoder panels) from single-column figure to two-column figure*. Pandoc
    # always emits \begin{figure}; ACL 2-column papers need \begin{figure*}
    # for any image whose three sub-panels would be unreadable at column width.
    _promote_wide_figures(tex_path)


def _seqsplit_arxiv_ids(tex_path: Path) -> None:
    """Wrap bare ``arXiv:NNNN.NNNNN`` tokens in ``\\seqsplit{...}`` so xelatex
    can break them mid-token in narrow columns. Skips occurrences already
    inside a \\seqsplit{...} or \\texttt{...} group (idempotent)."""
    import re
    src = tex_path.read_text()
    # Match bare 'arXiv:NNNN.NNNNN' not preceded by '\seqsplit{' or '\texttt{'.
    # The arXiv ID format: 4 digits, dot, 4-5 digits, optional vN suffix.
    pat = re.compile(r"(?<!\\seqsplit\{)(?<!\\texttt\{)arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)")
    new = pat.sub(r"\\seqsplit{arXiv:\1}", src)
    if new != src:
        tex_path.write_text(new)


# Images that should span both columns (figure*) instead of one (figure).
# Match by basename — these have three or more sub-panels packed into one PNG.
_WIDE_FIGURE_BASENAMES = {
    "ec_paper_figure.png",  # 3 encoder panels of Δhit@1 vs K
}


def _promote_wide_figures(tex_path: Path) -> None:
    """Convert ``\\begin{figure}...\\end{figure}`` to ``\\begin{figure*}...\\end{figure*}``
    when the figure body references an image listed in _WIDE_FIGURE_BASENAMES.
    Also swap \\pandocbounded → \\pandocboundedwide so the resizebox targets
    \\textwidth instead of \\linewidth.
    """
    import re
    src = tex_path.read_text()
    pat = re.compile(r"\\begin\{figure\}(.*?)\\end\{figure\}", re.DOTALL)

    def repl(m):
        body = m.group(1)
        for name in _WIDE_FIGURE_BASENAMES:
            if name in body:
                body = body.replace(r"\pandocbounded", r"\pandocboundedwide")
                return r"\begin{figure*}" + body + r"\end{figure*}"
        return m.group(0)

    new = pat.sub(repl, src)
    if new != src:
        tex_path.write_text(new)


_HEADING_CMDS = ("section", "subsection", "subsubsection", "paragraph",
                 "subparagraph", "section*", "subsection*", "subsubsection*")


def _strip_seqsplit_from_headings(tex_path: Path) -> None:
    """Replace `\\seqsplit{X}` with `X` inside any heading-command argument.

    Algorithm: scan for `\\<heading-cmd>{` markers, brace-match the argument,
    then replace any `\\seqsplit{...}` inside that argument with its inner
    content (the `\\texttt{...}` wrapping is preserved).
    """
    import re
    src = tex_path.read_text()

    def find_matching_brace(s: str, open_pos: int) -> int:
        """Given s[open_pos] == '{', return index of matching '}'."""
        depth = 1
        i = open_pos + 1
        while i < len(s):
            ch = s[i]
            if ch == '\\' and i + 1 < len(s):
                i += 2
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    def strip_seqsplit(arg_body: str) -> str:
        # Repeatedly replace \seqsplit{X} with X (X may contain braces, so we
        # do a brace-matched extraction).
        out = []
        i = 0
        while i < len(arg_body):
            if arg_body.startswith(r"\seqsplit{", i):
                inner_start = i + len(r"\seqsplit{")
                inner_end = find_matching_brace(arg_body, inner_start - 1)
                if inner_end < 0:
                    out.append(arg_body[i])
                    i += 1
                    continue
                inner = arg_body[inner_start:inner_end]
                # Recurse for nested cases
                out.append(strip_seqsplit(inner))
                i = inner_end + 1
            else:
                out.append(arg_body[i])
                i += 1
        return "".join(out)

    pattern = re.compile(r"\\(" + "|".join(re.escape(c) for c in _HEADING_CMDS) + r")\{")
    out_chunks = []
    pos = 0
    for m in pattern.finditer(src):
        out_chunks.append(src[pos:m.start()])
        cmd = m.group(0)  # \section{
        arg_open = m.end() - 1  # index of '{'
        arg_close = find_matching_brace(src, arg_open)
        if arg_close < 0:
            out_chunks.append(src[m.start():])
            pos = len(src)
            break
        arg_body = src[arg_open + 1:arg_close]
        cleaned = strip_seqsplit(arg_body)
        out_chunks.append(cmd + cleaned + "}")
        pos = arg_close + 1
    out_chunks.append(src[pos:])
    new_src = "".join(out_chunks)
    if new_src != src:
        tex_path.write_text(new_src)


def assemble_tex(body_frags: list[Path], appendix_frags: list[Path]) -> Path:
    """Build the wrapped .tex by injecting fragments into the ACL skeleton.

    The skeleton has placeholder commentary; we replace its body+appendix
    region with our generated fragments while keeping documentclass, ACL
    package, and bibliography wiring intact.
    """
    skel = (ACL_DIR / "acl_latex.tex").read_text()

    body_tex = "\n\n".join(p.read_text() for p in body_frags)
    appendix_tex = "\n\n".join(p.read_text() for p in appendix_frags)

    # Build a clean ACL-conformant document. We don't try to splice into
    # the example skeleton's body — we keep its preamble + author block,
    # then replace everything between \begin{document} and \end{document}.
    preamble = r"""\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{fontspec}
\setmainfont{texgyretermes-regular.otf}[
  BoldFont = texgyretermes-bold.otf,
  ItalicFont = texgyretermes-italic.otf,
  BoldItalicFont = texgyretermes-bolditalic.otf]
\setsansfont{texgyreheros-regular.otf}[
  BoldFont = texgyreheros-bold.otf,
  ItalicFont = texgyreheros-italic.otf,
  BoldItalicFont = texgyreheros-bolditalic.otf]
\setmonofont{Inconsolatazi4-Regular.otf}[
  BoldFont = Inconsolatazi4-Bold.otf]
\usepackage{microtype}
\usepackage{graphicx}
\graphicspath{{/data/}{/data/bench/results/}{/data/paper/}{/data/paper/build/acl/}{../}{../../}{../bench/results/}}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{colortbl}
\usepackage{pdflscape}
\usepackage{tabu}
\usepackage{threeparttable}
\usepackage{hyperref}
\usepackage{calc}
\usepackage{enumitem}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{seqsplit}
\usepackage{url}
\Urlmuskip=0mu plus 1mu  % allow url-line-breaks

% Justification tuning for ACL narrow 2-column layout. The defaults produce
% rivers of white-space ("loose lines") on lines that are hard to break.
% Tradeoff: relax tolerance for slightly tighter typography. \sloppy was
% over-aggressive (caused the wide gaps user reported); these settings let
% TeX hyphenate more aggressively before resorting to glue stretching.
\tolerance=1500           % default 200 — accept slightly worse boxes
\emergencystretch=1.5em   % was 3em — smaller = less rivers
\hyphenpenalty=50         % default 50 — keep hyphenation cheap
\exhyphenpenalty=50       % allow breaking at existing hyphens
\linepenalty=10           % default 10 — neutral
\hbadness=8000            % suppress warnings below this badness
\hfuzz=2pt                % allow 2pt overrun before "Overfull hbox" warns

% Unicode characters our markdown emits — provide ensuremath fallbacks
\providecommand{\textgreaterequal}{\ensuremath{\geq}}
\providecommand{\textlessequal}{\ensuremath{\leq}}
% xelatex declares some chars; for ones the Termes font lacks, force ensuremath
\usepackage{newunicodechar}
\newunicodechar{≡}{\ensuremath{\equiv}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{′}{\ensuremath{'}}

% pandoc emits \tightlist for compact bullet lists
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

% pandoc emits these for tables
\providecommand{\toprule}{\hline}
\providecommand{\midrule}{\hline}
\providecommand{\bottomrule}{\hline}

% pandoc emits \pandocbounded{\includegraphics{...}} for images.
% Default: force every image to fit \linewidth (single-column gutter)
% while keeping aspect ratio. Wide multi-panel figures use \pandocboundedwide
% (set up below) to fit \textwidth instead — see _promote_wide_figures().
\providecommand{\pandocbounded}[1]{#1}
\renewcommand{\pandocbounded}[1]{%
  \resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{#1}%
}
\providecommand{\pandocboundedwide}[1]{%
  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{#1}%
}

% Anonymized metadata (review mode)
\title{Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory}

\author{Anonymous Authors \\ \texttt{anonymous@anonymous.org}}

\begin{document}
\maketitle

"""

    document_close = r"""

\end{document}
"""

    # Strip the leading "Abstract" heading from 00_abstract.md output and
    # wrap its content in \begin{abstract}...\end{abstract}. The first body
    # fragment is always the abstract; pandoc emits it as
    # \section{Abstract}\nabstract content. We cut that.
    abstract_frag = body_frags[0].read_text()
    # crude but reliable: find first \section{Abstract...} and replace
    import re
    m = re.search(r"\\section\*?\{[^}]*[Aa]bstract[^}]*\}", abstract_frag)
    if m:
        abstract_body = abstract_frag[m.end():].strip()
        abstract_block = f"\\begin{{abstract}}\n{abstract_body}\n\\end{{abstract}}\n"
        rest_body = "\n\n".join(p.read_text() for p in body_frags[1:])
    else:
        # Fallback: treat first frag as-is
        abstract_block = ""
        rest_body = body_tex

    appendix_marker = r"""

\bibliography{references}

\appendix

"""

    full = preamble + abstract_block + "\n" + rest_body + appendix_marker + appendix_tex + document_close

    out = BUILD_DIR / "engram_v0.2_acl.tex"
    out.write_text(full)
    return out


def find_body_end_page(log_path: Path) -> int | None:
    """Parse pdflatex .log for the last page number on which a body section
    appears (heuristic: search for 'Appendix' / 'A.1' anchor).

    Returns None if we can't tell.
    """
    if not log_path.exists():
        return None
    return None  # leave as a TODO — pypdf-based check below is more reliable


def measure_pages(pdf_path: Path) -> tuple[int, int | None, int | None]:
    """Return (total_pages, limitations_start_page_1indexed, appendix_start_page_1indexed).

    EMNLP Industry Track CFP: "References and limitations sections do
    not count toward the page limit". So the 6-page check should be on
    body-without-limitations, not body+limitations.

    We detect:
      - limitations_start via the §75 Limitations opening sentence
      - appendix_start via the §A.4.6 prose (first ## block of A1)
    """
    from pypdf import PdfReader
    r = PdfReader(str(pdf_path))
    total = len(r.pages)
    lim_needles = [
        "We name scope limits explicitly",  # §75 opening
        "Limitations\nWe name scope",        # heading + body
    ]
    appx_needles = [
        "L0 promotion",
        "Where the consolidation lift",
        "Supplementary Ablations and Mechanism",
    ]
    def find_first(needles):
        for n in needles:
            for i, p in enumerate(r.pages):
                try:
                    t = p.extract_text() or ""
                except Exception:
                    continue
                if n in t:
                    return i
        return None
    return total, find_first(lim_needles), find_first(appx_needles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bib", action="store_true", help="skip bibtex pass")
    ap.add_argument("--keep-tex", action="store_true", help="copy generated .tex into paper/dist/")
    args = ap.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity: ACL style files present
    for required in ["acl.sty", "acl_natbib.bst"]:
        if not (ACL_DIR / required).exists():
            print(f"Missing {ACL_DIR / required}; cannot proceed.", file=sys.stderr)
            return 1

    # Copy ACL .sty + .bst into build dir so pdflatex finds them
    for f in ACL_DIR.iterdir():
        if f.suffix in (".sty", ".bst", ".cls", ".bib"):
            shutil.copy2(f, BUILD_DIR / f.name)

    # 1. Convert each markdown section to a .tex fragment
    body_frags: list[Path] = []
    for name in BODY_SECTIONS:
        md = PAPER_DIR / name
        tex = BUILD_DIR / (md.stem + ".tex")
        md_to_tex(md, tex)
        body_frags.append(tex)

    appendix_frags: list[Path] = []
    for name in APPENDIX_SECTIONS:
        md = PAPER_DIR / name
        tex = BUILD_DIR / (md.stem + ".tex")
        md_to_tex(md, tex)
        appendix_frags.append(tex)

    # 2. Assemble wrapped .tex
    main_tex = assemble_tex(body_frags, appendix_frags)
    print(f"wrote {main_tex} ({main_tex.stat().st_size} bytes)")

    # 3. Build sequence: xelatex → bibtex → xelatex → xelatex
    # bibtex is needed to resolve \citep{} references against references.bib.
    # Three xelatex passes after bibtex ensures all forward refs are resolved.
    rel_build = main_tex.parent.relative_to(REPO)
    rel_tex_name = main_tex.name
    rel_aux_name = rel_tex_name.replace(".tex", "")  # bibtex takes basename without .tex

    print(f"\n=== xelatex pass 1/3 ===")
    docker_run([
        "xelatex", "-interaction=nonstopmode", "-halt-on-error", rel_tex_name,
    ], workdir=f"/data/{rel_build}")

    if not args.no_bib:
        print(f"\n=== bibtex pass ===")
        docker_run([
            "bibtex", rel_aux_name,
        ], workdir=f"/data/{rel_build}")

    for pass_num in (2, 3):
        print(f"\n=== xelatex pass {pass_num}/3 ===")
        docker_run([
            "xelatex", "-interaction=nonstopmode", "-halt-on-error", rel_tex_name,
        ], workdir=f"/data/{rel_build}")

    pdf_in_build = main_tex.with_suffix(".pdf")
    if not pdf_in_build.exists():
        print(f"ERROR: pdflatex did not produce {pdf_in_build}", file=sys.stderr)
        return 2

    final_pdf = DIST_DIR / "engram_v0.2_acl.pdf"
    shutil.copy2(pdf_in_build, final_pdf)
    print(f"\ncopied to {final_pdf}")

    if args.keep_tex:
        shutil.copy2(main_tex, DIST_DIR / "engram_v0.2_acl.tex")
        print(f"copied {main_tex} → {DIST_DIR / 'engram_v0.2_acl.tex'}")

    # 4. Measure pages
    try:
        total, lim_start_idx, appendix_start_idx = measure_pages(final_pdf)
        print(f"\n=== ACL 2-column page count ===")
        print(f"total pages = {total}")
        if lim_start_idx is None and appendix_start_idx is None:
            print("WARN: could not detect limitations or appendix start.")
        else:
            # body-without-limitations = pages 1..lim_start_idx (lim_start_idx is 0-indexed → 1-indexed boundary = lim_start_idx)
            if lim_start_idx is not None:
                body_pages = lim_start_idx
                print(f"body (excl. limitations) spans pages 1-{body_pages}")
            else:
                body_pages = appendix_start_idx if appendix_start_idx else total
                print(f"body+limitations spans pages 1-{body_pages} (no limitations marker found)")
            if appendix_start_idx is not None:
                lim_end = appendix_start_idx
                if lim_start_idx is not None:
                    print(f"limitations spans pages {lim_start_idx+1}-{lim_end}")
                appendix_pages = total - appendix_start_idx
                print(f"appendix spans pages {appendix_start_idx+1}-{total} ({appendix_pages} pages)")
            if body_pages <= 6:
                print(f"OK: body fits in {body_pages} pages (Industry Track 6-page limit; limitations excluded per CFP)")
            else:
                print(f"WARN: body is {body_pages} pages "
                      f"(Industry Track limit is 6; over by {body_pages - 6}; limitations excluded per CFP)")
    except ImportError:
        print("(pypdf not installed - skipping page measurement)")

    # 5. Run layout validator (overfull/underfull/anon/page-count checks)
    print()
    validator = REPO / "scripts" / "check_acl_layout.py"
    if validator.exists():
        # Find a python with pypdf available — prefer current interpreter,
        # fall back to mise python3.11 (where pypdf is typically installed
        # on this AL2 cloud-desktop setup).
        py_candidates = [sys.executable]
        mise_py = Path.home() / ".local/share/mise/installs/python/3.11.13/bin/python3"
        if mise_py.exists():
            py_candidates.append(str(mise_py))
        rc = 1
        for py in py_candidates:
            check = subprocess.run(
                [py, "-c", "import pypdf"], capture_output=True
            )
            if check.returncode == 0:
                rc = subprocess.run([py, str(validator)]).returncode
                break
        else:
            print("(no python with pypdf found - running validator anyway, page-count check will skip)")
            rc = subprocess.run([sys.executable, str(validator)]).returncode
        if rc != 0:
            print("\n⚠ Layout validation failed. Fix issues above before submitting.")
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build paper PDF from paper/build/paper.md.

Pipeline:
  1. paper/build.sh (concatenates paper/00..A2_*.md → paper/build/paper.md)
  2. pypandoc-binary: markdown (gfm + tex_math) → HTML5 + TOC
  3. chromium headless --print-to-pdf: HTML → PDF

Output: paper/build/paper.pdf

Modes (default: review):
  --review        anonymized double-blind PDF (no author block, no ack)
  --camera-ready  venue camera-ready (ack restored, author block from venue template)
  --arxiv         arXiv preprint (public author block + ack + real GitHub link)

Why not weasyprint: AL2 Linux x86_64 host's libpango is too old (missing
pango_context_set_round_glyph_positions). Chromium handles it fine.

Why not pandoc → LaTeX: no pandoc/latexmk installed system-wide;
pypandoc-binary ships its own pandoc.
"""
import pypandoc
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--mode", choices=["review", "camera-ready", "arxiv"], default="review",
                help="build mode (default: review)")
args = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
md_path = REPO / "paper/build/paper.md"
out_dir = REPO / "paper/build"
html_path = out_dir / "paper.html"
pdf_path = out_dir / f"paper_{args.mode}.pdf" if args.mode != "review" else out_dir / "paper.pdf"

# Step 1: regenerate concatenated paper.md if needed (idempotent)
build_sh = REPO / "paper/build.sh"
if build_sh.exists():
    subprocess.run(["bash", str(build_sh), f"--{args.mode}"], cwd=REPO, check=True)

if not md_path.exists():
    sys.exit(f"missing {md_path}; run paper/build.sh first")

import re
md = md_path.read_text()

# Strip leading `% ` LaTeX-style comments (build.sh emits one as a header
# marker; pandoc's gfm reader passes them through as literal text).
md = re.sub(r'(?m)^%[^\n]*\n', '', md)

# Rewrite relative image paths to absolute. Markdown lives in paper/build/, but
# image refs are like ../bench/results/foo.png (correct from paper/, not paper/build/).
# Repoint to absolute file:// URLs so chromium can find them.
md = re.sub(r'!\[([^\]]*)\]\(\.\./bench/', r'![\1](file://' + str(REPO) + '/bench/', md)
md = re.sub(r'!\[([^\]]*)\]\((bench/)', r'![\1](file://' + str(REPO) + r'/\2', md)
md = re.sub(r'!\[([^\]]*)\]\(\.\./figures/', r'![\1](file://' + str(REPO) + '/paper/figures/', md)

# pypandoc-binary ships its own pandoc; convert markdown → standalone HTML
html_body = pypandoc.convert_text(
    md,
    to="html5",
    format="gfm+tex_math_dollars",
    extra_args=["--mathjax", "--toc", "--toc-depth=3"],
)

css = """
@page { size: Letter; margin: 0.9in 0.85in; @bottom-center { content: counter(page); font-family: serif; font-size: 9pt; color: #555; } }
body { font-family: "Liberation Serif", "DejaVu Serif", serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 18pt; margin-top: 1.2em; border-bottom: 1px solid #999; padding-bottom: 0.15em; page-break-before: auto; }
h1.title-page { page-break-before: always; }
h2 { font-size: 13pt; margin-top: 1.1em; }
h3 { font-size: 11.5pt; margin-top: 1em; }
h4 { font-size: 10.5pt; margin-top: 0.9em; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.88em; background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 0.82em; background: #f6f6f6; padding: 8px 10px; border-left: 2px solid #aaa; overflow-wrap: break-word; white-space: pre-wrap; page-break-inside: avoid; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 0.92em; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #efefef; }
blockquote { border-left: 3px solid #888; padding-left: 10px; color: #444; margin-left: 0; }
a { color: #0a4; text-decoration: none; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
nav#TOC { font-size: 0.95em; }
nav#TOC ul { list-style: none; padding-left: 1em; }
nav#TOC > ul { padding-left: 0; }
img { max-width: 100%; display: block; margin: 1em auto; }
figcaption, p > em:only-child { display: block; text-align: center; font-size: 0.9em; color: #555; margin-top: 0.3em; }
"""

# Build the full HTML doc with TOC + title
title = "Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory"
title_page_subtitle = {
    "review": "Working draft (anonymized for review). Compiled from paper/ Markdown sources.",
    "camera-ready": "Camera-ready. Compiled from paper/ Markdown sources.",
    "arxiv": "arXiv preprint. Compiled from paper/ Markdown sources.",
}[args.mode]
full_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style>
</head>
<body>
<div style="text-align:center; margin-top:2.5in;">
<h1 class="title-page" style="border:none; font-size:20pt;">{title}</h1>
<p style="font-size:11pt; color:#555;">{title_page_subtitle}<br>
Branch: feat/v0.2-governed-memory-and-scale-tests</p>
</div>
{html_body}
</body></html>
"""
html_path.write_text(full_html)
print(f"wrote {html_path} ({len(full_html):,} bytes)")

# Step 3: chromium headless --print-to-pdf
chromium_candidates = [
    Path.home() / ".cache/ms-playwright/chromium-1217/chrome-linux64/chrome",
    Path.home() / ".cache/ms-playwright/chromium-1208/chrome-linux64/chrome",
]
chrome = next((p for p in chromium_candidates if p.exists()), None)
if chrome is None:
    chrome_str = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    if not chrome_str:
        sys.exit("no chromium found; install playwright chromium or chrome")
    chrome = Path(chrome_str)

print(f"using chromium: {chrome}")
subprocess.run(
    [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--virtual-time-budget=15000",
        f"file://{html_path}",
    ],
    check=True,
    capture_output=True,
)
print(f"wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

# Step 3b: strip identifying metadata for review-PDF anonymization (AUDIT-O)
# chromium leaves /Title, /Producer="Skia/PDF mNNN", /Creator="Mozilla/5.0 (X11; Linux x86_64)"
# in the PDF info dict. The /Creator string especially can hint at OS/author setup.
# We keep /Title (it's the public paper title, not author-identifying) and clear the rest.
# In --arxiv mode we ALSO write /Author for proper preprint metadata.
# Uses pypdf (pure-python, AL2-friendly) rather than pikepdf (C deps fail to build on AL2).
try:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter(clone_from=reader)
    keep_title = (reader.metadata or {}).get("/Title", "")
    writer.metadata = {}  # nuke everything
    new_meta = {}
    if keep_title:
        new_meta["/Title"] = str(keep_title)
    if args.mode == "arxiv":
        # Match what 05_authors_arxiv.md declares. Update both places when adding co-authors.
        new_meta["/Author"] = "Anonymous Authors"
    if new_meta:
        writer.add_metadata(new_meta)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    print(f"stripped identifying metadata from {pdf_path} (mode={args.mode})")
except ImportError:
    print("pypdf not installed — skipping metadata strip (AUDIT-O)")
except Exception as e:
    print(f"metadata strip failed (non-fatal): {e}")

# Step 4: copy to paper/dist/ for git-tracked distribution
dist_dir = REPO / "paper/dist"
dist_dir.mkdir(exist_ok=True)
dist_filename = {
    "review": "engram_v0.2_draft.pdf",
    "camera-ready": "engram_v0.2_camera_ready.pdf",
    "arxiv": "engram_v0.2_arxiv.pdf",
}[args.mode]
shutil.copy(pdf_path, dist_dir / dist_filename)
print(f"copied to {dist_dir / dist_filename}")
shutil.copy(md_path, dist_dir / "engram_v0.2_draft.md")
print(f"updated {dist_dir}/")

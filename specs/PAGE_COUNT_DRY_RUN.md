# PAGE-COUNT-DRY-RUN — EMNLP 2026 9-page main-track fit analysis

> **Generated:** 2026-05-26 (this commit). LaTeX is not available on
> the Linux x86_64 host without root, so this dry-run is a calibrated
> word-count → ACL-page-count estimate, not a rendered PDF. A real
> LaTeX render must happen before submission; the README at the end
> of this file has the install + build invocation.

## 1. Current word counts (body only)

| File                | Words | Notes                                    |
|---------------------|-------|------------------------------------------|
| 00_abstract.md      |  271  | trimmed to ~165w in body, 271 incl HTML comments |
| 10_intro.md         |  728  | with new §1 paragraph 4 (governance bridge) |
| 20_related.md       | 1921  | post-citation-fix; 28 refs               |
| 30_methods.md       | 2254  | with §3.1.1 latency + §3.7 governance preamble |
| 40_results.md       | 2560  | with 9 verdict markers                   |
| 50_discussion.md    | 1946  | with §5.7 effect-size + §5.8 power       |
| 60_threats.md       | 1511  | with §6.6/6.7/6.8 + §6.8 paraphrase table|
| 70_conclusion.md    |  505  | protocol-first rewrite                   |
| **Body total**      | **11,696** |                                     |

## 2. ACL/EMNLP page-density calibration

ACL `acl_template.tex` (10pt, two-column, ~7.0in usable width per
page after margins). Empirical wpp by content type:

| Content mix                   | Words / page | Source                        |
|-------------------------------|-------------:|-------------------------------|
| Pure prose                    |     ~800     | ACL FAQ                       |
| Prose + 1 small table         |     ~700     | typical Resource paper        |
| Prose + tables + figures      |     ~600     | empirical IR / NLP-eval paper |
| Prose + heavy math + tables   |     ~500     | theory-heavy paper            |

Our paper sits in the **third bucket** — many CI tables (§4.2, §4.4,
§4.5.1, §4.8, §A.4.8.1's third column), 2 figures (§4.1 hash vs ST
vs BGE three-panel; §4.5 BM25 saturation), no display math.

**Estimate: 600 wpp.** Body 11,696 / 600 = **19.5 pages.**

Even at the most prose-heavy 800 wpp, body = 14.6 pages. Either way
we are **5-11 pages over** the EMNLP main-track 9-page long-paper
limit. Not submittable to main track without major restructuring.

## 3. Three options for getting under the limit

### Option A — EMNLP Findings (no extra work)

Findings of EMNLP uses the same review pool, same template, same
page limit (9 pages long, 5 pages short). Same problem. **Findings
does not help with our page-limit issue.**

### Option B — Move §5 + §6 to appendix; reframe paper as "protocol + headline result"

The §A appendix has no page limit. We can move ~half of the body to
an extended appendix and keep the main body crisp:

| Section            | Stay in body? | Trim target (words) | Action                                        |
|--------------------|---------------|---------------------|-----------------------------------------------|
| §1 Intro           | yes           | 600 (from 728)      | Trim contributions list to 3 bullets, defer specifics to §3 |
| §2 Related work    | yes (½)       | 800 (from 1921)     | Keep 2.1 + 2.2; move 2.3 ecosystem + 2.4 ES-lit to §A.5 (new) |
| §3 Methods         | yes           | 1500 (from 2254)    | Trim §3.6 (PRF) + §3.7 (governance) by half; move long subsection bodies to §A6 |
| §4 Results         | yes           | 2000 (from 2560)    | Drop §4.5.1 BGE-only inversion; defer detail to §A.4.16; keep verdicts in body |
| §5 Discussion      | half (½)      | 1000 (from 1946)    | Move §5.4 + §5.5 + §5.6 + §5.7 + §5.8 to §A.5; keep §5.1 + §5.2 + §5.3 in body |
| §6 Threats         | half (½)      | 700 (from 1511)     | Keep §6.1 + §6.4 + §6.6 + §6.8 in body; move 6.2/6.3/6.5/6.7 to §A.5 |
| §7 Conclusion      | yes           | 400 (from 505)      | Trim final operational corollary by 1/3       |
| **Body total**     | —             | **7000**            |                                               |

7000 words / 600 wpp = **11.7 pages.** Still over.

Need a deeper trim. Realistic Option B target: **5500 words** body =
9.2 pages, with appendix carrying everything we cut.

### Option C — Reframe as a short paper

EMNLP short paper limit: 5 pages + 1 page response = 6 pages total.
At 600 wpp that's 3000 words. **Too tight** unless we strip the
paper to pure protocol + headline two-axis table + cross-system
replication.

Short paper version is plausible but kills the systems-side
contribution and most of the related-work positioning. Recommended
only if we want a fast first publication and a longer journal /
follow-up paper later.

## 4. Recommended plan

**Target: long paper, EMNLP main, Option B aggressive.** Sequence:

1. **First pass (this commit-cycle):** the spec lives here; no
   actual deletion yet. We need the M4 P4 BEIR-3 result before
   we know whether §A grows further.
2. **Once P4 lands:** decide which §5 / §6 subsections are
   load-bearing for the headline two-axis claim. Move the rest
   to a new `paper/A5_appendix_extended.md` that picks up where
   §A2 / §A6 leave off.
3. **Render LaTeX** (see §6 of this file) and measure actual page
   count on the ACL template.
4. **Iterate trimming** until the body is ≤9 pages of compiled
   PDF, not estimated pages.

## 5. Trim candidates ranked by safety-of-removal

The body subsections that contribute least to the headline claim and
are safest to relocate to appendix:

1. **§2.4 Schema-lifecycle and event-sourced memory** — 8 cites,
   ~500w, all positioned as background for §3.7's governance
   framing. Reviewers focused on the protocol will not need it.
   → move to §A.5
2. **§5.4 The honest version of "consolidation lifts retrieval"** —
   ~480w of methodological prescription. Important but not
   protocol-relevant. → §A.5
3. **§5.5 The PRF latency myth** — ~300w of latency-microbench
   methodology. → §A.5
4. **§5.6 Scope of the PRF null result** — ~250w. → §A.5
5. **§5.7 + §5.8** — effect-size translation + statistical-power
   disclosure (~600w combined). Audit-mandated; keep partially in
   body but trim verbose tables to §A.5.
6. **§6.2, §6.3, §6.5, §6.7** — the threats-section subsections that
   are not paraphrase-related. ~600w combined. → §A.5
7. **§3.7.1-§3.7.4** — most of the governance-mechanism descriptions
   can move to §A6 (which we already created for traceability).
   Keep the §3.7 preamble in body.
8. **§3.1.1 encoder latency-cost trade-off table** — audit-mandated
   so cannot move; keep in body.

Conservative removal: §2.4 + §5.4 + §5.5 + §5.6 + §6.2 + §6.5 + §6.7
= ~2500w out of body. New body total: 11,696 − 2500 = 9200w =
**15.3 pages** at 600 wpp. Still over.

Aggressive removal: above + half of §3.7 + §5.7/5.8 collapse +
§4.5.1 detail = ~5500w out of body. New body total: 6200w =
**10.3 pages**. Close.

Brutal removal: above + tighten every section's prose by 15-20% =
~7000w out of body. New body total: 4700w = **7.8 pages**. Fits.

The brutal pass is achievable but takes 2-3 days of focused
editing. The aggressive pass is one full day.

## 6. Real LaTeX render — how to do it when the time comes

### 6.1 Install (Linux x86_64 host, sudo required)

```bash
sudo dnf install texlive-scheme-medium texlive-acl pandoc
# OR if texlive-acl is unavailable:
sudo dnf install texlive-scheme-medium pandoc
git clone https://github.com/acl-org/acl-style-files /tmp/acl-style
```

ACL templates are at https://github.com/acl-org/acl-style-files —
clone, not pip install.

### 6.2 Convert markdown body to LaTeX

```bash
cd ~/projects/engram
mkdir -p paper/latex_dryrun
cp /tmp/acl-style/latex/acl_latex.{cls,sty} paper/latex_dryrun/

# concatenate body sections through pandoc → tex fragments
for f in paper/{00_abstract,10_intro,20_related,30_methods,40_results,50_discussion,60_threats,70_conclusion}.md; do
  pandoc -f markdown -t latex "$f" >> paper/latex_dryrun/_body.tex
  echo "" >> paper/latex_dryrun/_body.tex
done
```

Then wrap in a minimal driver:

```latex
\documentclass[11pt]{acl_latex}
\usepackage{microtype}
\usepackage{booktabs}
\title{Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory}
\author{Anonymous}
\begin{document}
\maketitle
\input{_body}
\bibliographystyle{acl_natbib}
\bibliography{refs}
\end{document}
```

### 6.3 Compile and measure

```bash
cd paper/latex_dryrun
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
pdfinfo paper.pdf | grep Pages
```

If `Pages > 9`, trim per §5 of this file.

## 7. Submission deadline math

EMNLP 2026 submission deadline: **August 20, 2026** (per
https://2026.emnlp.org/calls/main_conference_papers/).
Current date: 2026-05-26. We have **86 days**.

| Task                                                | Days | Cumulative |
|-----------------------------------------------------|-----:|-----------:|
| Wait on M4 P4 (BEIR-3 × BGE)                        |   ?  |   ?        |
| Wire P4 into §A appendix                            |   1  |            |
| Run RM3 baseline (per RM3_BASELINE_SPEC.md)         |   1  |            |
| Run Letta cross-system replication                  |   5  |            |
| Aggressive Option-B trim of body to <9 LaTeX pages  |   2  |            |
| LaTeX template setup + first compile               |   1  |            |
| Iterate on page-count trim                          |   1  |            |
| De-anonymize check + final linter pass              |   0.5|            |
| GPTZero pre-flight re-run on full bibliography      |   0.5|            |
| OpenReview / ARR upload + checklist                 |   0.5|            |
| **Working days needed**                             | **12.5** | T+12.5 days |

We have substantial slack. The constraint is M4 P4 wall time, not
calendar days.

## 8. Stop signals (don't ship until)

- [ ] LaTeX-rendered body ≤9 pages.
- [ ] `python scripts/verify_citations.py paper/*.md` returns 0.
- [ ] `python scripts/check_doc_refs.py` returns 0.
- [ ] `python scripts/check_anon.py paper/[0-9]*.md paper/A1_*.md
      paper/A2_*.md paper/A6_*.md` returns 0.
- [ ] All non-arXiv cites have venues / DOIs / URLs in §2.
- [ ] §A.4.18 RM3 result wired in (even if null) — pre-registered
      result, not a sweep.
- [ ] §A.4.17 Letta cross-system result wired in (or honest scope
      paragraph saying "not run, deferred to follow-up").
- [ ] `paper/80_acknowledgements_cameraready.md` reflects what the
      assistant actually drafted vs what is author-only.

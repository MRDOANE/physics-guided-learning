# Journal and reference audit for the broader manuscript

Checked 12 September 2026. The working title is *Choosing between physical and neural forecasters with imperfect equations*.

## Submission format

The [official Journal of Computational Science author guide](https://www.sciencedirect.com/journal/journal-of-computational-science/publish/guide-for-authors) supplied indexed excerpts for the requirements below. Direct retrieval returned HTTP 403, so this is a documented partial check.

| Item | Verified guidance / action |
| --- | --- |
| Abstract | At most 250 words. |
| Keywords | 1–7; the planned six fit. |
| Highlights | 3–5 bullets, each at most 85 characters including spaces; provide a separate editable file. |
| Review model | Single anonymized; retain the author's name and affiliation. |
| Editable manuscript | DOC/DOCX accepted. Use the existing single-column Word workflow. |
| Citations | Numbers in square brackets. |
| Overall length | No total word or page ceiling was verified. The planned 6,000 words is an editorial target. |
| Reference count | No required count was verified. |

Retain editable tables and equations, separate figure files, captions, funding, competing interests, data availability, author contributions, and an accurate AI-use disclosure. Confirm correspondence email and declaration accuracy before submission. The full live submission checklist remains the final check.

## LaTeX

Word is the requested primary deliverable. Optional LaTeX can use Elsevier's `elsarticle` class and an appropriate numbered bibliography style. If LaTeX is included, the journal-submission source bundle should place all files at one folder level: Elsevier says Editorial Manager cannot process subfolders. A GitHub repository can retain its ordinary directory structure. The [official Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions) were directly accessible. LaTeX is an optional source format; a DOCX submission does not require conversion.

## Reference coverage and count

The existing 40 entries already cover neural operators, transformer recurrence, hybrid physical learning, perturbed training examples, solver errors, numerical time integration, algorithm selection, and weak physical baselines. The new physical-calibration comparison needs established system-identification and model-discrepancy literature. Three verified additions are supplied in `reference_additions.json` and `.bib`:

| Key | Role |
| --- | --- |
| `ljung2010identification` | Introduce fitting a dynamical model to observed trajectories as an established system-identification task. |
| `brynjarsdottir2014discrepancy` | Explain why adjusting coefficients and correcting missing equation terms are different modeling problems; fitted coefficients can compensate for equation errors. |
| `virtanen2020scipy` | Credit the numerical optimization software used by the calibration code. |

Metadata was checked against publisher-deposited Crossref records. Ljung's publisher abstract and author-hosted presentation were checked; the author-hosted document is an earlier presentation, so the citation metadata uses the 2010 journal record. Brynjarsdóttir and O'Hagan's publication was verified on the coauthor's publication page and against its DOI metadata; its complete article was not retrieved during this audit. SciPy's official citation page supplied its preferred author list, including the group author. These checks support the specific background uses above and do not imply a full-text review of every paper.

The 12-paper physical-neural-modeling comparison sample has a median of 51 references, ranging from 23 to 95. All titles, DOI links, counts, selection rules, and raw records are retained in `REFERENCE_COUNT_SAMPLE.md` and `reference_count_sample.json`. Forty-three references is reasonable for the revised manuscript's coverage and lies within this observed range. Reaching exactly 51 has no scientific or journal-policy justification.

A relevant additional journal article, Sikora et al. (2024), *Comparison of Physics Informed Neural Networks and Finite Element Method Solvers for advection-dominated diffusion problems*, DOI [10.1016/j.jocs.2024.102340](https://doi.org/10.1016/j.jocs.2024.102340), was identified and its metadata saved. It supplies a journal-specific comparison precedent. Its full results were not retrieved, so adding detailed claims about its findings would need further reading. It is optional and is absent from the three-addition export.

## Placement and scientific claims

Move `mcgreivy2024weak` into the introductory motivation for calibrating the physical baseline and measuring its execution cost. Its concern about weak baselines is directly relevant to the changed model ranking. Existing `Rice1976AlgorithmSelection` and `Xu2008SATzilla` are sufficient background for choosing among candidate algorithms; the broader manuscript does not need a new cluster of selection citations.

The new contribution is the controlled evidence about which candidate forecasts best after coefficient fitting, combined with the existing augmentation controls and validation decisions. The paper should describe the exact setting where calibration changes the ranking. Parameter fitting, ordinary validation-based selection, and the distinction between coefficient uncertainty and missing equation terms all have substantial prior literature. Avoid claiming a first demonstration of any of those general ideas.

Maintain separate explanations for the new validation-based choice among physical and neural forecasters and the earlier learned augmentation selector. The latter selects a neural training procedure for a fixed architecture. This distinction changes how the reader understands the evidence and the computational costs.

The two 2026 preprints already in the bibliography (`huang2026stablepdenet`, `ma2026operatorcorrection`) were checked again against their arXiv abstract pages. Their current titles, author lists, and preprint status are consistent with the stored entries. They should remain labeled as preprints. No broader novelty guarantee is implied by this targeted revision audit.

## Sources

- [Journal guide](https://www.sciencedirect.com/journal/journal-of-computational-science/publish/guide-for-authors)
- [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)
- [Ljung journal article](https://doi.org/10.1016/j.arcontrol.2009.12.001)
- [Ljung author-hosted presentation](https://isy.gitlab-pages.liu.se/staff/lenlj48/seoul2dvinew/plenary2.pdf)
- [Brynjarsdóttir and O'Hagan journal article](https://doi.org/10.1088/0266-5611/30/11/114007)
- [O'Hagan publication list](https://www.tonyohagan.co.uk/academic/pub.html)
- [SciPy preferred citation](https://scipy.org/citing-scipy/)
- [StablePDENet arXiv record](https://arxiv.org/abs/2601.06472)
- [Physics-guided correction arXiv record](https://arxiv.org/abs/2606.03469)

# Revised manuscript and section review copies

**Choosing between physical and neural forecasters with imperfect equations**

Prepared for Journal of Computational Science. This is an author-review draft.

- [Complete manuscript](Physics_JOCS_Manuscript_Draft.docx)
- [Supplementary material](Physics_JOCS_Supplement_Draft.docx)
- [Highlights](Physics_JOCS_Highlights.docx)
- [Abstract and introduction](01_Abstract_and_Introduction.docx)
- [Methods](02_Methods.docx)
- [Results](03_Results.docx)
- [Discussion and conclusions](04_Discussion_and_Conclusions.docx)
- [Revision notes](REVISION_NOTES.md)
- [Journal preparation check](JOURNAL_FORMAT_CHECK.md)

The four numbered files contain the corresponding prose with tables, captions,
and figures removed. Methods retains editable equations. Citations use the
numbering in the complete paper; its reference list is also exported in
`references.json` and `references.bib`.

The complete Word files are the formatted reading copies. `source/main.md` and
`source/supplement.md` use symbolic citations and table, equation, and figure
markers consumed by `source/build_manuscript.py`. The top-level Markdown copies
expand citations but retain object markers. Tables are generated from saved
evidence, and all Word equations are native editable Office Math.

## Rebuild the Word files

From the repository root, install `manuscript/source/requirements.txt` in a
Python environment, then run:

```text
python manuscript/source/build_manuscript.py
```

The default output is `manuscript/`. Set `MANUSCRIPT_OUTPUT` to a separate
directory when checking a rebuild. This script assembles documents from saved
results; it performs no training or calibration. Word pagination can vary
slightly between font and office-software installations.

## Figures and evidence replay

Standalone figures and their plotted values are in `figures/`. Main Figure 1
uses `figure1_absolute_accuracy`; main Figure 2 uses
`figure_3_architecture_effects`; main Figure 3 uses `figure2_cpu_execution`.
Supplementary Figures S1 and S2 use `figure_1_perturbation_controls` and
`figure_2_grid_confirmation`, respectively.

The new figure generator accepts `--report` and `--output` paths. Its report is
`results/calibration/reports/report.json` at repository level. Plotting requires
NumPy and Matplotlib. `figures/audit_primary_claims.py` independently replays
600 primary and secondary means and all 156 frozen validation choices; pass
`--run results/calibration`, `--archive experiments/calibration/assets`, and an
`--output` JSON path from the repository root. The stored audit records an exact
match for these means and choices. Bootstrap intervals remain the original
analysis's estimates.

Author details and the new DOI need the checks listed in `REVISION_NOTES.md`.
The paper's existing DOI identifies v1.0.0. The new release is v1.1.0.

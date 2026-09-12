# Manuscript revision for v1.1.0

Title: **Choosing between physical and neural forecasters with imperfect equations**

The paper now begins with the practical choice between a physical solver and a learned forecast. The new calibration results supply the main comparison. The original physical-supervision experiments explain differences among the available neural candidates.

## What changed

- Added all 30 completed physical-calibration settings and direct validation-based choices among unchanged physics, calibrated physics, and the archived neural candidates.
- Reported absolute normalized forecast errors, coefficient corrections, missing-term outcomes, and matched CPU execution costs. Primary results show one representative setting per physical family; supplementary tables retain every setting and both test distributions.
- Distinguished the ordinary validation-based forecasting choice from the earlier learned augmentation selector. The former compares completed candidates; the latter predicts which training procedure to continue from short probes.
- Explained the scope of the Fourier operator's advantage over the tested transformer configurations. A separate latent world-model architecture was outside the experiment.
- Retained the original augmentation estimates, confidence intervals, and numerical controls, with their interpretation placed alongside the direct physics baseline.
- Added system-identification, model-discrepancy, and numerical-optimization references. The manuscript has 43 cited references.
- Rounded reported effects to readable precision, generally two or three significant figures. Exact configuration values, counts, and seeds remain precise where reproducibility requires them. Machine-readable result files retain their original precision.

## How to review the documents

`Physics_JOCS_Manuscript_Draft.docx` contains the complete paper, editable tables and equations, figures, captions, declarations, and references. `Physics_JOCS_Supplement_Draft.docx` contains implementation details and complete supplementary tables. Highlights are in a separate editable document.

The four numbered DOCX files reproduce the corresponding manuscript prose. They contain no tables, captions, or figures. The Methods copy retains editable equations. In-text table and figure references are retained so the prose agrees with the full paper. Discussion uses slightly tighter paragraph spacing to avoid a nearly empty final page.

The source Markdown files use symbolic citation and object markers as inputs to `source/build_manuscript.py`. The complete formatted Word files are the primary reading copies. The release includes numbered reference exports in JSON and BibTeX, standalone figures, plotted values, and the document builder.

## Evidence boundaries retained in the manuscript

The calibration extension is exploratory: earlier test results informed its design. Its new coefficients and model choices were frozen before opening the extension's test scores. Its pointwise bootstrap intervals condition on the fitted physical correction and generated training dataset.

The observations are fully available and noiseless, the spatial domains are one-dimensional and periodic, and the imposed coefficient distortion can be represented exactly by the three fitted multipliers. The results establish the observed model ordering under those conditions. The two missing-term cases provide a limited sample of equation errors.

Current CPU timing uses initialized neural architectures because trained checkpoints are absent from the original archives. Trained-model accuracy comes from archived evaluations with verified data identity. Historical GPU training costs remain separate from the new CPU fitting and execution measurements.

The manuscript interprets effect estimates and uncertainty without imposing an arbitrary minimum improvement or a required number of successful systems. Historical status labels and their original rules remain in the preserved archive.

## Author details to finish

Confirm the correspondence email, funding statement, competing-interest statement, and author contributions before submission. Read and approve the AI-use disclosure. DOI 10.5281/zenodo.22728288 identifies v1.0.0. Add the actual v1.1.0 DOI after the new version is published.

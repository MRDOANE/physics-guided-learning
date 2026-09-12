# Changelog

## 1.1.0 — prepared release

- Reframes the manuscript around selecting a physical or neural forecasting model, using absolute accuracy and computational cost.
- Adds 30 completed calibrated-physics settings, with three shared coefficient corrections learned from training trajectories and candidates chosen by validation forecasts.
- Adds frozen validation choices among calibrated physics, unchanged physics, and the archived neural candidates.
- Adds matched CPU execution timing and separately recorded calibration cost.
- Adds a direct audit of the original 1,584 neural fits and a numerical compatibility check for comparing their saved errors with new physical forecasts.
- Adds the revised manuscript and section-by-section review documents.
- Extends the dependency-free integrity checker to the new source, calibration outputs, and frozen decisions.
- Preserves the original experimental source and result files. Generated raw trajectory caches and installed environments are excluded. Their required source, metadata, and data hashes are retained.

The complete calibration run took about 85 seconds on its recorded CPU environment. All 30 settings completed; all 28 generated datasets matched their original archive hashes. The extension performed no neural training. It was designed after inspection of prior results and is exploratory.

The DOI for this release has yet to be assigned. Follow the existing Zenodo record's version chain when publishing it.

## 1.0.0 — published original release

Original development, resolution, and augmentation-selection experiments: 1,584 neural fits across eight synthetic equations. Archived at [DOI 10.5281/zenodo.22728288](https://doi.org/10.5281/zenodo.22728288). Original metadata and documentation are preserved in `provenance/v1.0.0/`.

# Direct model-choice audit

Start with `MODEL_CHOICE_AUDIT.html`: a self-contained report with an interactive comparison table. `MODEL_CHOICE_AUDIT.md` contains the same narrative. The report uses only existing experiment records. No neural models were trained and no RunPod job was started during this audit.

## Reproduce the saved-data audit

Use the published version 1.0.0 release: https://doi.org/10.5281/zenodo.22728288. Its source/result directory should contain `experiments/`, `results/`, and `manifest.json`.

With NumPy and Matplotlib available:

```bash
python audit_model_choice.py --repository /path/to/physics-guided-learning
python build_audit_report.py
```

The analysis script takes seconds to minutes depending on disk and CPU speed. It checks saved hashes and metadata, reads errors, resamples them, and plots them. It never imports training code. Default uncertainty intervals use 5,000 paired bootstrap replicates. The archived selector was separately reconstructed for integrity checking; this small historical ridge-fit replay performs no neural training.

## Optional: fill the missing inference-time measurement

This step has not been run in the analysis environment. It needs an existing PyTorch/NumPy installation. No packages are installed or replaced, and no model is trained. On a functioning GPU pod:

```bash
bash launch_timing_no_training.sh /workspace/physics-guided-learning --device cuda --output /workspace/fresh_inference_timings.json
```

For an explicit CPU measurement, use `--device cpu`. Use `--batch-sizes 1 32` to add a batched throughput workload. The default is batch one, matching the archived latency's batch size. Use `--families allen_cahn` for one equation first. The output path must be new. The script prints each completed case so progress is visible.

The script measures existing physics and neural implementations on the same hardware and inputs. Trained checkpoints are absent from the public release. Neural weights are initialized to time the fixed forward graph; these are **fresh architecture execution timings**, with no new accuracy evaluation. Compare fresh physics and neural timings with each other. Keep them distinct from archived measurements of trained models. Runtime has not been measured here; work consists of warmups and repeated forwards, with no optimizer or full training loop.

## Data and methods

- `absolute_accuracy.csv`: all 1,392 absolute-error summaries, including physical and persistence references, with exploratory pointwise intervals.
- `direct_contrasts.csv`: all 2,280 direct comparisons with paired intervals. Negative difference favors the candidate. Multiple comparisons are unadjusted; this is a post hoc analysis.
- `main_comparison.csv`: one stated stage per equation, with the primary 64-step in-distribution comparisons.
- `archived_costs.csv`: fixed-choice setup, training/validation, and recorded neural inference time for all 288 neural configuration groups.
- `retrospective_validation_selection.csv`: optional full-training validation choices among neural architectures and augmentation procedures. No test values choose a model. The standalone physical solver is outside this selector because its validation score was not saved.
- `numerical_diagnostics.csv`: raw-error and cap diagnostics. All primary 64-step comparisons are finite and uncapped.
- `paired_trajectory_scores.npz` and `score_index.json`: checked per-trajectory scores and identity arrays sufficient to reproduce every reported mean and bootstrap comparison without loading the larger original evaluation arrays.
- `record_metadata.json`: archived candidate settings, normalization, validation results, and timings.
- `integrity_audit.json`, `source_checksums.json`, `release_verification.txt`, and `analysis_console.txt`: integrity results and audit provenance.
- `analysis_results.json`: complete structured analysis used by the report.
- `accuracy_*.png` and `neural_accuracy_latency.png`: result figures generated from the recorded errors.

The original arrays contain normalized squared errors, not full forecast states. The primary metric is the arithmetic mean NMSE across steps, trajectories, and seeds. The manuscript's earlier geometric mean of paired error ratios is a different estimator. The report explains how this affects percentage comparisons.

The manuscript and GitHub release files were preserved. This package is a separate exploratory analysis for deciding whether to broaden the paper.

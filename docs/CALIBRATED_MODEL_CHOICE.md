# Calibrated physics and model choice

## What changed

The original experiments tested how physical guidance changes neural forecasting. The direct forecast audit compared those neural errors with the unchanged physical solver. The new extension gives the physical solver an explicit opportunity to learn its coefficients from the same observed training trajectories. This makes the choice between available forecasting implementations easier to interpret.

Each setting has 128 training trajectories, 24 validation trajectories, 64 ordinary test trajectories, and 64 test trajectories whose coefficients lie outside the training range. A setting is an equation, grid, stage, and physical-information condition. There are 30 settings across eight equations: 14 with correct coefficients, 14 with biased coefficients, and two with omitted terms.

## Calibration and selection

Three positive multipliers correct the supplied physical coefficients. The multipliers are shared across trajectories within a setting. The fitting function receives observed states, actions, and supplied coefficients. No separate ground-truth coefficient field is passed to it; in the biased cases, its coefficient input is the distorted vector. It fits six candidates using one- and four-step training forecasts and three generic starting points. The unchanged physical solver is a seventh candidate. The candidate with the smallest 16-step validation error is retained.

A second validation decision selects among unchanged physics, calibrated physics, and all archived neural candidates available within each setting and training seed. It uses archived neural validation scores and the new physical validation scores. The resulting choices are saved in `results/calibration/freeze.json` before test evaluation. A candidate selected by test error is never used for this comparison.

The earlier guarded selector chose a neural augmentation procedure using a historical validation bank and short target probes. Its study and findings remain separate from this new forecasting-model choice.

## Error and uncertainty

Forecast error is the mean squared error after division by training-derived channel scales. The primary horizon is 64 steps; the second is 96 steps. Full trajectory scores are paired across methods. Pointwise 95% bootstrap intervals resample test trajectories and the archived neural training seeds. They hold the fitted physical model and training data fixed. They omit uncertainty from collecting new training trajectories or repeating calibration on a new training sample.

The calibration extension is exploratory because prior test results informed the design. Its intervals have no multiplicity adjustment. Individual result labels summarize each pointwise interval. No minimum percentage improvement and no required number of successful systems determines whether the experiment is scientifically usable.

## Main results and scope

Physics had lower error in all 14 correct-coefficient settings. Calibration corrected the imposed coefficient bias, and calibrated physics had lower error than the selected neural model in all 14 biased-coefficient settings. Neural forecasts had lower error in both missing-term settings. The pattern held at both horizons and in both coefficient distributions. The validation-based forecasting choice selected a physical solver in the 28 complete-equation settings and a neural model in both missing-term settings.

The equations, observations, and bias construction make exact coefficient correction possible. The physical solver shares the reference generator's equation family. Its near-zero error under successful calibration therefore describes this controlled setting. Claims about noisy sensors, hidden states, uncertain forcing, realistic equation mismatch, higher-dimensional systems, and broad architecture rankings require additional experiments.

## Cost

The archived neural fits retain their original measured training and inference costs. New calibration and execution measurements are stored separately. The complete CPU extension took about 85 seconds. The new matched timing experiment uses the original physical stepping code and initialized copies of the archived neural architectures, with identical state/input layouts, batch size, device, and thread settings. The trained neural checkpoints were unavailable. The timing experiment measures these implementations' forward execution; it adds no new neural accuracy observations.

Selecting from a fully trained neural candidate bank requires paying for that bank when it is first created. Reusing it here removes new training expense for this extension. It does not remove the historical acquisition cost from a prospective model-selection workflow.

## Reproducibility

All 28 regenerated training/validation and test dataset files matched their archived NPZ hashes during the recorded run. All 30 comparisons were eligible through exact data hashes and the recorded identity checks. Three settings exceeded the preset CPU/GPU reference-error replay tolerance; those discrepancies remain recorded. The repository includes those hash records, all fitted multipliers, test error arrays, selection freezes, timing records, and exact experiment source. Regenerated raw trajectories were omitted to keep the release compact. Their generation code and archived hashes are supplied.

Use `scripts/verify_release.py` to check the packaged evidence. Follow `docs/REPRODUCIBILITY.md` to regenerate the data and refit physical coefficients in a new output directory. The archived result directory remains a frozen evidence record.

# Completed results

The source reports contain exact numerical values. Values here are rounded for reading. Positive MSE reduction means lower forecast error. A negative reduction can exceed 100% in magnitude because the augmented candidate can have several times the baseline error.

## Spatial-resolution follow-up

The following comparisons use the 64-cell grid and smooth augmentation versus no augmentation. Each pools the three architectures under the original estimator. All six prespecified effect directions were supported.

| Equation | Physical prior | MSE reduction (%) | Pointwise 95% CI (%) | Holm p | Interpretation |
|---|---|---:|---|---:|---|
| Wave | Correct | 35.323 | [27.433, 44.222] | <0.001 | Expected benefit supported |
| Wave | Biased coefficients | -210.940 | [-268.134, -157.952] | <0.001 | Expected harm supported |
| Advection–diffusion | Correct | 24.642 | [7.510, 43.007] | 0.024 | Expected benefit supported |
| Advection–diffusion | Biased coefficients | -240.877 | [-316.054, -179.961] | <0.001 | Expected harm supported |
| Allen–Cahn | Correct | 83.219 | [80.659, 85.521] | <0.001 | Expected benefit supported |
| Allen–Cahn | Biased coefficients | 75.304 | [69.945, 79.807] | <0.001 | Expected benefit supported |

These are selected follow-up contrasts, not a claim that every architecture benefits. The earlier six-equation experiment contains the smooth-versus-independent and response-matched comparisons. Its full table remains in [the development report](../results/development/reports/report.md).

## Augmentation selection on two new equations

| Guarded selector compared with | MSE reduction (%) | Pointwise 95% CI (%) | Holm p | Interpretation |
|---|---:|---|---:|---|
| No augmentation | 2.343 | [-4.674, 8.831] | 0.510 | Inconclusive |
| Best short validation probe | 19.007 | [13.702, 23.720] | <0.001 | Lower pooled error supported |

The result against probes is stronger for Cahn–Hilliard than FitzHugh–Nagumo. The comparison with no augmentation does not establish equivalence or noninferiority. On the secondary coefficient-shift evaluation, the guarded selector has about 18.07% higher pooled MSE than no augmentation. That limitation should accompany any deployment-oriented interpretation.

## Cost

| Target route | Reconstructed mean seconds |
|---|---:|
| No augmentation | 36.101 |
| Always smooth | 65.918 |
| Best short probe | 70.753 |
| Development-selected fixed arm | 47.518 |
| Original ridge selector | 65.817 |
| Guarded selector | 64.713 |

These routes use measured components from the all-candidate bank. They do not represent independently executed, compute-matched deployment experiments. The guard costs about 1.79 times the unaugmented route and about 8.5% less than the probe-selected route. Actual follow-up wall time was 11.067 hours; development added approximately 13.70 earlier hours.

## Status labels

The original confirmation report states 720/720 verified candidate records and no training failures. GREEN means support for a stated effect direction; YELLOW means inconclusive; RED means evidence in the opposite direction. A green expected-harm comparison does not mean model performance improved. Integrity checks and scientific evidence are separate. No minimum effect size or required count of successful systems is imposed by the follow-up scientific labels.

The development report's historical aggregate traffic-light rules are preserved in its original source. They are not carried forward as paper acceptance or hypothesis requirements.

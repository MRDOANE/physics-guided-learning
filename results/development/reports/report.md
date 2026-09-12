# Physical-family mechanism and selector results

- Completion: **GREEN** (864/864 candidate records)
- Scientific eligibility: **EVALUATED**
- Mechanism: **YELLOW**
- Selector: **GREEN**

## Prespecified mechanism contrasts

Positive reduction favors smooth augmentation. CIs are Bonferroni-adjusted within the mechanism family.

| Physical family | Prior | Baseline | Reduction % | Adjusted CI % | Result |
|---|---|---|---:|---|---|
| wave | correct | iid | 92.1 | [88.6, 95.1] | GREEN |
| wave | correct | response_matched | 56.9 | [46.7, 67.9] | GREEN |
| wave | correct | none | 37.4 | [29.7, 44.6] | GREEN |
| wave | coefficient | none | -217.5 | [-343.4, -117.2] | RED |
| burgers | correct | iid | 15.4 | [-1.1, 31.4] | YELLOW |
| burgers | correct | response_matched | 29.4 | [3.3, 52.1] | GREEN |
| burgers | correct | none | 35.4 | [10.7, 52.5] | GREEN |
| burgers | coefficient | none | 3.5 | [-29.1, 26.7] | YELLOW |
| ks | correct | iid | -13.3 | [-50.5, 17.6] | YELLOW |
| ks | correct | response_matched | -14.8 | [-57.5, 23.6] | YELLOW |
| ks | correct | none | 24.6 | [0.4, 41.3] | GREEN |
| ks | coefficient | none | -8.7 | [-59.2, 30.9] | YELLOW |
| advection_diffusion | correct | iid | 4.2 | [-88.2, 33.2] | YELLOW |
| advection_diffusion | correct | response_matched | 5.2 | [-106.5, 38.5] | YELLOW |
| advection_diffusion | correct | none | 33.4 | [10.4, 50.6] | GREEN |
| advection_diffusion | coefficient | none | -373.7 | [-942.3, -201.8] | RED |
| allen_cahn | correct | iid | 13.6 | [-23.9, 50.6] | YELLOW |
| allen_cahn | correct | response_matched | -11.0 | [-63.4, 20.4] | YELLOW |
| allen_cahn | correct | none | 75.3 | [59.9, 86.9] | GREEN |
| allen_cahn | coefficient | none | 75.4 | [58.8, 85.4] | GREEN |
| gray_scott | correct | iid | 6.0 | [-14.4, 21.6] | YELLOW |
| gray_scott | correct | response_matched | 12.7 | [-2.2, 24.0] | YELLOW |
| gray_scott | correct | none | -0.7 | [-18.9, 16.0] | YELLOW |
| gray_scott | coefficient | none | -194.2 | [-288.4, -129.8] | RED |

GREEN requires smooth > IID in at least 3 families and smooth > response-matched IID in at least 2; this does not imply benefit over no augmentation.

## Prespecified selector contrasts

| Split | Baseline | Reduction % | Adjusted CI % | Result |
|---|---|---:|---|---|
| development_lofo | none | 22.2 | [0.0, 45.8] | YELLOW |
| development_lofo | best_fixed | 48.5 | [13.4, 73.5] | GREEN |
| development_lofo | probe_best | 24.1 | [-8.4, 52.9] | YELLOW |
| confirmation | none | -22.2 | [-62.5, 4.0] | YELLOW |
| confirmation | best_fixed | 29.0 | [9.0, 44.4] | GREEN |
| confirmation | probe_best | -6.2 | [-24.2, 6.9] | YELLOW |

GREEN requires at least 10% confirmed improvement over best fixed, adjusted CI excluding zero, no demonstrated confirmation harm against probe-best, and positive development improvement over best fixed. RED denotes demonstrated confirmation harm against best fixed or probe-best.

Only 5 development physical families are resampled; this interval cannot establish generality over all physical systems.

All candidate fits were executed for the research bank. Policy figures are counterfactual training/selection timing accounts, not actual GPU money saved. Ridge and probe-best pay all candidate probe startup + short probes, then only the selected candidate's continuation startup + remaining training; full-validation pays all complete fits. Legacy records lacking split startup timings allocate setup to the probe. Fixed/random policies pay one full selected fit. Any selected augmentation, and every probe or diagnostic policy, also pays the complete shared augmentation-bank/diagnostic build once. The hand rule is conservatively charged this implemented bundle because a cheaper diagnostic path was not measured. The development-bank subtotal exposes the research cost of obtaining selector training labels; selector fitting is included in the total and disclosed separately. Selected-model inference latency is measured at batch one on validation inputs, including normalization, with warmups excluded and CUDA synchronization. No deployment workload, aggregate inference expense, or energy usage is assumed.

## Interpretation limits

- A successful completed run can yield positive, inconclusive, or negative scientific findings.
- Mechanism support against IID/response matching does not establish an improvement over ordinary supervised training.
- Pure physics and persistence baselines are reported; a correct simulator may outperform every learned model.
- This experiment selects augmentation within each architecture, not a winner among disjoint transformer/physics/world-model categories.
- Matched update counts are not matched compute or parameter counts.
- Scores cap each trajectory's mean MSE at the configured loss cap; failed trajectories remain in the analysis. Log ratios use an additive 1e-15 floor.
- The selection manifest must be frozen before test evaluation. Reports audit declared family membership and evidence hashes, not an external timestamp authority.

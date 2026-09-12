# Calibrated physics and neural forecast comparison

Completion: **GREEN** — 30/30 settings verified.
Scientific evidence: **EXPLORATORY_COMPLETED_SETTINGS**. Profile: `full`; backend: `torch`.

This extension tests whether fitting a small physical-parameter correction to the existing training trajectories changes the choice between a solver and a neural forecaster. Neural weights are reused through their archived prediction errors. No neural model is retrained.

Three positive multipliers are fitted per equation/prior/grid setting. They are shared across trajectories and remain fixed for both test distributions. Structural omissions remain in the equations. The unchanged solver is included in validation selection.

## Reading the evidence

GREEN indicates lower candidate error with a pointwise interval above zero; RED indicates higher error; YELLOW means the interval includes zero; BLUE means identical trajectory scores. Diagnostic profiles use NOT_EVALUATED. Positive differences and positive percentages favor the candidate.

There is no minimum required improvement and no required number of successful equations. Labels describe individual exploratory comparisons. They do not decide whether the paper or a general hypothesis succeeds.

Exploratory pointwise 95% paired bootstrap intervals; no multiplicity adjustment or global acceptance rule.
Shared test trajectories and, for neural options, archived training seeds are resampled. The fitted physical solver and training dataset are held fixed; intervals omit calibration/training-data uncertainty.

## Primary in-distribution accuracy

Errors average the first 64 forecast steps. NMSE uses the original training-derived channel scales. Lower values are better. Equations remain separate.

| Setting | Uncalibrated physics | Calibrated physics | Validation-best neural | Validation-selected route |
| --- | ---: | ---: | ---: | ---: |
| advection_diffusion / coefficient / g32 | 0.5313 | 3.53e-12 | 0.000337 | 3.53e-12 |
| advection_diffusion / correct / g32 | 3.063e-12 | 3.063e-12 | 0.0001202 | 3.063e-12 |
| allen_cahn / coefficient / g32 | 0.004759 | 4.075e-12 | 0.0002658 | 4.075e-12 |
| allen_cahn / correct / g32 | 1.957e-12 | 3.914e-12 | 0.0002364 | 3.914e-12 |
| burgers / coefficient / g32 | 0.01712 | 3.616e-11 | 0.0005206 | 3.616e-11 |
| burgers / correct / g32 | 2.531e-11 | 2.531e-11 | 0.0002041 | 2.531e-11 |
| gray_scott / coefficient / g32 | 0.734 | 2.106e-11 | 0.0004435 | 2.106e-11 |
| gray_scott / correct / g32 | 1.994e-11 | 2.057e-11 | 0.0004188 | 2.057e-11 |
| ks / coefficient / g32 | 0.5865 | 1.806e-07 | 0.02702 | 1.806e-07 |
| ks / correct / g32 | 2.426e-07 | 1.806e-07 | 0.01479 | 1.806e-07 |
| wave / coefficient / g32 | 0.3974 | 4.458e-11 | 0.0004112 | 4.458e-11 |
| wave / correct / g32 | 8.928e-11 | 4.505e-11 | 0.0001421 | 4.505e-11 |
| advection_diffusion / coefficient / g32 | 0.6843 | 4.36e-12 | 0.0003787 | 4.36e-12 |
| advection_diffusion / correct / g32 | 4.236e-12 | 4.236e-12 | 0.0001257 | 4.236e-12 |
| allen_cahn / coefficient / g32 | 0.00638 | 4.771e-12 | 0.0005064 | 4.771e-12 |
| allen_cahn / correct / g32 | 1.887e-12 | 4.798e-12 | 0.0002453 | 4.798e-12 |
| wave / coefficient / g32 | 0.4136 | 6.17e-11 | 0.0003548 | 6.17e-11 |
| wave / correct / g32 | 1.169e-10 | 6.264e-11 | 0.0001476 | 6.264e-11 |
| advection_diffusion / coefficient / g64 | 0.6843 | 2.462e-12 | 0.0004464 | 2.462e-12 |
| advection_diffusion / correct / g64 | 2.372e-12 | 2.38e-12 | 0.0001508 | 2.38e-12 |
| allen_cahn / coefficient / g64 | 0.005892 | 4.459e-12 | 0.0005613 | 4.459e-12 |
| allen_cahn / correct / g64 | 1.587e-12 | 4.327e-12 | 0.0002268 | 4.327e-12 |
| wave / coefficient / g64 | 0.4136 | 4.904e-12 | 0.000373 | 4.904e-12 |
| wave / correct / g64 | 5.343e-12 | 5.117e-12 | 0.000134 | 5.117e-12 |
| cahn_hilliard / coefficient / g64 | 0.1141 | 2.052e-12 | 4.568e-05 | 2.052e-12 |
| cahn_hilliard / correct / g64 | 2.656e-12 | 2.387e-12 | 2.914e-05 | 2.387e-12 |
| cahn_hilliard / structural / g64 | 0.2674 | 0.2674 | 4.697e-05 | 4.697e-05 |
| fitzhugh_nagumo / coefficient / g64 | 0.03142 | 3.542e-11 | 0.0004343 | 3.542e-11 |
| fitzhugh_nagumo / correct / g64 | 3.062e-11 | 3.493e-11 | 0.0004465 | 3.493e-11 |
| fitzhugh_nagumo / structural / g64 | 1.62 | 1.492 | 0.0005038 | 0.0005038 |

Every neural architecture and augmentation option, both distributions, both horizons, pointwise intervals, and archived cost measurements appear in `absolute_accuracy.csv`.

## Calibrated-physics contrasts

| Setting | Distribution / steps | Baseline | Label | MSE reduction, % [95% interval] |
| --- | --- | --- | --- | ---: |
| advection_diffusion / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | test / 64 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | test / 96 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | ood / 64 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | ood / 96 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | test / 64 | uncalibrated | RED | -100 [-222.5, -24.66] |
| allen_cahn / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | test / 96 | uncalibrated | RED | -111 [-216.9, -39.3] |
| allen_cahn / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | ood / 64 | uncalibrated | RED | -15.31 [-28.9, -4.909] |
| allen_cahn / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | ood / 96 | uncalibrated | RED | -31.33 [-51.29, -15.66] |
| allen_cahn / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| burgers / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / correct / g32 | test / 64 | uncalibrated | BLUE | 0 [0, 0] |
| burgers / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / correct / g32 | test / 96 | uncalibrated | BLUE | 0 [0, 0] |
| burgers / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / correct / g32 | ood / 64 | uncalibrated | BLUE | 0 [0, 0] |
| burgers / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| burgers / correct / g32 | ood / 96 | uncalibrated | BLUE | 0 [0, 0] |
| burgers / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| gray_scott / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / correct / g32 | test / 64 | uncalibrated | YELLOW | -3.136 [-16.01, 9.21] |
| gray_scott / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / correct / g32 | test / 96 | uncalibrated | YELLOW | -6.694 [-20.3, 6.237] |
| gray_scott / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / correct / g32 | ood / 64 | uncalibrated | YELLOW | 1.247 [-7.797, 10.46] |
| gray_scott / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| gray_scott / correct / g32 | ood / 96 | uncalibrated | YELLOW | 0.6242 [-13.23, 13.58] |
| gray_scott / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [99.99, 100] |
| ks / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| ks / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 99.99 [99.99, 100] |
| ks / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 99.99 [99.98, 100] |
| ks / correct / g32 | test / 64 | uncalibrated | YELLOW | 25.57 [-38.21, 51.59] |
| ks / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| ks / correct / g32 | test / 96 | uncalibrated | YELLOW | 15.68 [-26.9, 38.19] |
| ks / correct / g32 | test / 96 | validation_best_neural | GREEN | 99.99 [99.99, 100] |
| ks / correct / g32 | ood / 64 | uncalibrated | YELLOW | -1.167 [-7.01, 5.53] |
| ks / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| ks / correct / g32 | ood / 96 | uncalibrated | YELLOW | 7.742 [-3.698, 18.99] |
| ks / correct / g32 | ood / 96 | validation_best_neural | GREEN | 99.99 [99.98, 100] |
| wave / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | test / 64 | uncalibrated | GREEN | 49.54 [47.71, 51.89] |
| wave / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | test / 96 | uncalibrated | GREEN | 50.05 [48.13, 52.52] |
| wave / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | ood / 64 | uncalibrated | GREEN | 19.7 [19, 20.6] |
| wave / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | ood / 96 | uncalibrated | GREEN | 19.57 [18.89, 20.46] |
| wave / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | test / 64 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | test / 96 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | ood / 64 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g32 | ood / 96 | uncalibrated | BLUE | 0 [0, 0] |
| advection_diffusion / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | test / 64 | uncalibrated | RED | -154.3 [-235.5, -85.62] |
| allen_cahn / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | test / 96 | uncalibrated | RED | -194.9 [-279.4, -122.3] |
| allen_cahn / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | ood / 64 | uncalibrated | RED | -10.21 [-21.04, -1.741] |
| allen_cahn / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g32 | ood / 96 | uncalibrated | RED | -24.25 [-39.59, -11.5] |
| allen_cahn / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | test / 64 | uncalibrated | GREEN | 46.44 [45.06, 48.45] |
| wave / correct / g32 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | test / 96 | uncalibrated | GREEN | 46.94 [45.43, 49.12] |
| wave / correct / g32 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | ood / 64 | uncalibrated | GREEN | 20.46 [19.8, 21.23] |
| wave / correct / g32 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g32 | ood / 96 | uncalibrated | GREEN | 20.24 [19.61, 20.98] |
| wave / correct / g32 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| advection_diffusion / coefficient / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g64 | test / 64 | uncalibrated | YELLOW | -0.3182 [-1.163, 0.04125] |
| advection_diffusion / correct / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g64 | test / 96 | uncalibrated | YELLOW | -0.772 [-2.847, 0.1004] |
| advection_diffusion / correct / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g64 | ood / 64 | uncalibrated | YELLOW | -3.435 [-10.44, 0.6013] |
| advection_diffusion / correct / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| advection_diffusion / correct / g64 | ood / 96 | uncalibrated | YELLOW | -2.298 [-8.646, 2.136] |
| advection_diffusion / correct / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| allen_cahn / coefficient / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g64 | test / 64 | uncalibrated | RED | -172.7 [-270.4, -94.52] |
| allen_cahn / correct / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g64 | test / 96 | uncalibrated | RED | -249.8 [-365.4, -158.9] |
| allen_cahn / correct / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g64 | ood / 64 | uncalibrated | RED | -11.39 [-23.79, -2.04] |
| allen_cahn / correct / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| allen_cahn / correct / g64 | ood / 96 | uncalibrated | RED | -28.18 [-46.63, -13.75] |
| allen_cahn / correct / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| wave / coefficient / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g64 | test / 64 | uncalibrated | YELLOW | 4.23 [-6.984, 13.93] |
| wave / correct / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g64 | test / 96 | uncalibrated | YELLOW | 5.853 [-3.29, 13.38] |
| wave / correct / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g64 | ood / 64 | uncalibrated | GREEN | 6.496 [2.797, 9.968] |
| wave / correct / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| wave / correct / g64 | ood / 96 | uncalibrated | GREEN | 9 [6.705, 11.15] |
| wave / correct / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| cahn_hilliard / coefficient / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / correct / g64 | test / 64 | uncalibrated | YELLOW | 10.13 [-2.022, 22.71] |
| cahn_hilliard / correct / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / correct / g64 | test / 96 | uncalibrated | GREEN | 13.76 [0.3033, 27.64] |
| cahn_hilliard / correct / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / correct / g64 | ood / 64 | uncalibrated | GREEN | 9.731 [1.286, 18.7] |
| cahn_hilliard / correct / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / correct / g64 | ood / 96 | uncalibrated | GREEN | 9.854 [0.9705, 18.96] |
| cahn_hilliard / correct / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| cahn_hilliard / structural / g64 | test / 64 | uncalibrated | BLUE | 0 [0, 0] |
| cahn_hilliard / structural / g64 | test / 64 | validation_best_neural | RED | -5.692e+05 [-7.944e+05, -3.827e+05] |
| cahn_hilliard / structural / g64 | test / 96 | uncalibrated | BLUE | 0 [0, 0] |
| cahn_hilliard / structural / g64 | test / 96 | validation_best_neural | RED | -3.034e+06 [-5.113e+06, -1.563e+06] |
| cahn_hilliard / structural / g64 | ood / 64 | uncalibrated | BLUE | 0 [0, 0] |
| cahn_hilliard / structural / g64 | ood / 64 | validation_best_neural | RED | -3.637e+04 [-6.756e+04, -2.391e+04] |
| cahn_hilliard / structural / g64 | ood / 96 | uncalibrated | BLUE | 0 [0, 0] |
| cahn_hilliard / structural / g64 | ood / 96 | validation_best_neural | RED | -2.244e+05 [-4.402e+05, -1.389e+05] |
| fitzhugh_nagumo / coefficient / g64 | test / 64 | uncalibrated | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | test / 96 | uncalibrated | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | ood / 64 | uncalibrated | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | ood / 96 | uncalibrated | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / coefficient / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / correct / g64 | test / 64 | uncalibrated | RED | -14.07 [-19.46, -9.274] |
| fitzhugh_nagumo / correct / g64 | test / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / correct / g64 | test / 96 | uncalibrated | RED | -10.98 [-16.29, -6.31] |
| fitzhugh_nagumo / correct / g64 | test / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / correct / g64 | ood / 64 | uncalibrated | RED | -10.37 [-14.33, -6.415] |
| fitzhugh_nagumo / correct / g64 | ood / 64 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / correct / g64 | ood / 96 | uncalibrated | RED | -10.75 [-14.96, -6.465] |
| fitzhugh_nagumo / correct / g64 | ood / 96 | validation_best_neural | GREEN | 100 [100, 100] |
| fitzhugh_nagumo / structural / g64 | test / 64 | uncalibrated | GREEN | 7.917 [6.796, 9.25] |
| fitzhugh_nagumo / structural / g64 | test / 64 | validation_best_neural | RED | -2.96e+05 [-4.492e+05, -2.096e+05] |
| fitzhugh_nagumo / structural / g64 | test / 96 | uncalibrated | GREEN | 7.051 [6.078, 8.269] |
| fitzhugh_nagumo / structural / g64 | test / 96 | validation_best_neural | RED | -6.922e+05 [-1.041e+06, -4.988e+05] |
| fitzhugh_nagumo / structural / g64 | ood / 64 | uncalibrated | GREEN | 7.932 [6.571, 9.425] |
| fitzhugh_nagumo / structural / g64 | ood / 64 | validation_best_neural | RED | -3.09e+04 [-5.817e+04, -1.971e+04] |
| fitzhugh_nagumo / structural / g64 | ood / 96 | uncalibrated | GREEN | 7.786 [6.349, 9.402] |
| fitzhugh_nagumo / structural / g64 | ood / 96 | validation_best_neural | RED | -6.483e+04 [-1.119e+05, -4.254e+04] |

## Calibration and optimizer diagnostics

| Setting | Selected candidate | Multipliers | Search bounds | Selected validation NMSE | Calibration elapsed, s | Boundary coordinates |
| --- | --- | --- | --- | ---: | ---: | --- |
| advection_diffusion / coefficient / g32 | h1_start1 | 1.42857, 0.769231, 0.714287 | [0.001, 1000.0] | 4.84e-13 | 0.5525 | [] |
| advection_diffusion / correct / g32 | identity | 1, 1, 1 | [0.001, 1000.0] | 5.024e-13 | 0.4008 | [] |
| allen_cahn / coefficient / g32 | h4_start2 | 1.42859, 0.76924, 0.714299 | [0.001, 1000.0] | 1.582e-12 | 0.7472 | [] |
| allen_cahn / correct / g32 | h4_start0 | 1.00001, 1.00001, 1.00002 | [0.001, 1000.0] | 1.598e-12 | 0.4988 | [] |
| burgers / coefficient / g32 | h4_start1 | 1.42856, 0.769238, 0.714293 | [0.001, 1000.0] | 1.051e-11 | 0.7999 | [] |
| burgers / correct / g32 | identity | 1, 1, 1 | [0.001, 1000.0] | 7.396e-12 | 0.6622 | [] |
| gray_scott / coefficient / g32 | h4_start0 | 1.42857, 0.769231, 0.714286 | [0.001, 1000.0] | 1.047e-12 | 1.219 | [] |
| gray_scott / correct / g32 | h4_start1 | 0.999999, 1, 1 | [0.001, 1000.0] | 1.006e-12 | 0.7572 | [] |
| ks / coefficient / g32 | h4_start0 | 1.42858, 0.769214, 0.714284 | [0.001, 1000.0] | 3.308e-09 | 0.8528 | [] |
| ks / correct / g32 | h4_start0 | 1.00001, 0.999978, 0.999997 | [0.001, 1000.0] | 3.309e-09 | 0.5814 | [] |
| wave / coefficient / g32 | h4_start0 | 1.42857, 0.769236, 0.714271 | [0.001, 1000.0] | 6.888e-12 | 0.6686 | [] |
| wave / correct / g32 | h4_start0 | 1, 1.00001, 0.999979 | [0.001, 1000.0] | 6.948e-12 | 0.4024 | [] |
| advection_diffusion / coefficient / g32 | h4_start2 | 1.42857, 0.769231, 0.714286 | [0.001, 1000.0] | 4.771e-13 | 0.5164 | [] |
| advection_diffusion / correct / g32 | identity | 1, 1, 1 | [0.001, 1000.0] | 4.956e-13 | 0.3526 | [] |
| allen_cahn / coefficient / g32 | h4_start2 | 1.42859, 0.769241, 0.7143 | [0.001, 1000.0] | 2.066e-12 | 0.5849 | [] |
| allen_cahn / correct / g32 | h4_start0 | 1.00001, 1.00001, 1.00002 | [0.001, 1000.0] | 2.06e-12 | 0.4882 | [] |
| wave / coefficient / g32 | h4_start1 | 1.42857, 0.769236, 0.714271 | [0.001, 1000.0] | 7.462e-12 | 0.6044 | [] |
| wave / correct / g32 | h4_start0 | 1, 1.00001, 0.999979 | [0.001, 1000.0] | 7.478e-12 | 0.4547 | [] |
| advection_diffusion / coefficient / g64 | h4_start0 | 1.42857, 0.769231, 0.714286 | [0.001, 1000.0] | 2.68e-13 | 0.7697 | [] |
| advection_diffusion / correct / g64 | h4_start1 | 1, 1, 1 | [0.001, 1000.0] | 2.665e-13 | 0.5114 | [] |
| allen_cahn / coefficient / g64 | h4_start1 | 1.42859, 0.769241, 0.714301 | [0.001, 1000.0] | 1.97e-12 | 0.8907 | [] |
| allen_cahn / correct / g64 | h4_start1 | 1.00001, 1.00001, 1.00002 | [0.001, 1000.0] | 1.895e-12 | 0.6912 | [] |
| wave / coefficient / g64 | h4_start2 | 1.42857, 0.769231, 0.714285 | [0.001, 1000.0] | 7.462e-13 | 1.432 | [] |
| wave / correct / g64 | h4_start0 | 1, 1, 0.999999 | [0.001, 1000.0] | 7.504e-13 | 0.9579 | [] |
| cahn_hilliard / coefficient / g64 | h4_start1 | 1.42857, 0.769231, 0.714289 | [0.001, 1000.0] | 1.194e-13 | 1.2 | [] |
| cahn_hilliard / correct / g64 | h4_start0 | 1, 1, 1 | [0.001, 1000.0] | 1.153e-13 | 0.895 | [] |
| cahn_hilliard / structural / g64 | identity | 1, 1, 1 | [0.001, 1000.0] | 0.00411 | 2.022 | [] |
| fitzhugh_nagumo / coefficient / g64 | h4_start1 | 1.42856, 0.76923, 0.714291 | [0.001, 1000.0] | 2.926e-12 | 1.749 | [] |
| fitzhugh_nagumo / correct / g64 | h4_start1 | 0.999994, 0.999999, 1.00001 | [0.001, 1000.0] | 3.003e-12 | 1.333 | [] |
| fitzhugh_nagumo / structural / g64 | h4_start0 | 1.72928, 0.965761, 1.33214 | [0.001, 1000.0] | 0.05297 | 1.891 | [] |

The complete JSON includes every candidate, optimizer termination, finite-difference call count, boundary hit, numerical failure, and Jacobian rank. A boundary hit or exhausted evaluation budget is a diagnostic. No scientific threshold is attached to either. Coefficient-error experiments admit an exact shared multiplicative correction; the structural cases retain their missing terms.

## Computation

Current elapsed time: 85.41 s. Estimated Pod compute cost: unavailable (rate or elapsed time missing).

Historical neural training and inference measurements remain identified as historical in the accuracy CSV. Current solver-calibration time cannot establish a matched-hardware training-cost comparison with those records.

Fresh matched measurements execute resident inputs on one device. Neural weights are initialized because trained checkpoints were omitted from the release. These are architecture-execution measurements; no fresh neural accuracy is inferred. Full block timings and environments are in `matched_timing.csv` and the timing JSON files.

| Setting | Device | Method | Batch | Median ms / batch-step |
| --- | --- | --- | ---: | ---: |
| advection_diffusion / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.1991 |
| advection_diffusion / coefficient / g32 | cpu | physics_calibrated | 1 | 0.1999 |
| advection_diffusion / coefficient / g32 | cpu | transformer | 1 | 1.08 |
| advection_diffusion / coefficient / g32 | cpu | looped | 1 | 1.111 |
| advection_diffusion / coefficient / g32 | cpu | fno | 1 | 0.6475 |
| advection_diffusion / correct / g32 | cpu | physics_uncalibrated | 1 | 0.2188 |
| advection_diffusion / correct / g32 | cpu | physics_calibrated | 1 | 0.2102 |
| advection_diffusion / correct / g32 | cpu | transformer | 1 | 1.087 |
| advection_diffusion / correct / g32 | cpu | looped | 1 | 1.119 |
| advection_diffusion / correct / g32 | cpu | fno | 1 | 0.6725 |
| allen_cahn / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.4557 |
| allen_cahn / coefficient / g32 | cpu | physics_calibrated | 1 | 0.4492 |
| allen_cahn / coefficient / g32 | cpu | transformer | 1 | 1.115 |
| allen_cahn / coefficient / g32 | cpu | looped | 1 | 1.113 |
| allen_cahn / coefficient / g32 | cpu | fno | 1 | 0.6559 |
| allen_cahn / correct / g32 | cpu | physics_uncalibrated | 1 | 0.4535 |
| allen_cahn / correct / g32 | cpu | physics_calibrated | 1 | 0.4633 |
| allen_cahn / correct / g32 | cpu | transformer | 1 | 1.145 |
| allen_cahn / correct / g32 | cpu | looped | 1 | 1.142 |
| allen_cahn / correct / g32 | cpu | fno | 1 | 0.6634 |
| burgers / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.5296 |
| burgers / coefficient / g32 | cpu | physics_calibrated | 1 | 0.5353 |
| burgers / coefficient / g32 | cpu | transformer | 1 | 1.174 |
| burgers / coefficient / g32 | cpu | looped | 1 | 1.288 |
| burgers / coefficient / g32 | cpu | fno | 1 | 0.683 |
| burgers / correct / g32 | cpu | physics_uncalibrated | 1 | 0.6002 |
| burgers / correct / g32 | cpu | physics_calibrated | 1 | 0.5567 |
| burgers / correct / g32 | cpu | transformer | 1 | 1.264 |
| burgers / correct / g32 | cpu | looped | 1 | 1.2 |
| burgers / correct / g32 | cpu | fno | 1 | 0.6982 |
| gray_scott / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.6466 |
| gray_scott / coefficient / g32 | cpu | physics_calibrated | 1 | 0.5908 |
| gray_scott / coefficient / g32 | cpu | transformer | 1 | 1.261 |
| gray_scott / coefficient / g32 | cpu | looped | 1 | 1.208 |
| gray_scott / coefficient / g32 | cpu | fno | 1 | 0.7691 |
| gray_scott / correct / g32 | cpu | physics_uncalibrated | 1 | 0.5945 |
| gray_scott / correct / g32 | cpu | physics_calibrated | 1 | 0.5439 |
| gray_scott / correct / g32 | cpu | transformer | 1 | 1.076 |
| gray_scott / correct / g32 | cpu | looped | 1 | 1.07 |
| gray_scott / correct / g32 | cpu | fno | 1 | 0.6406 |
| ks / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.5113 |
| ks / coefficient / g32 | cpu | physics_calibrated | 1 | 0.5215 |
| ks / coefficient / g32 | cpu | transformer | 1 | 1.072 |
| ks / coefficient / g32 | cpu | looped | 1 | 1.069 |
| ks / coefficient / g32 | cpu | fno | 1 | 0.6308 |
| ks / correct / g32 | cpu | physics_uncalibrated | 1 | 0.5091 |
| ks / correct / g32 | cpu | physics_calibrated | 1 | 0.5157 |
| ks / correct / g32 | cpu | transformer | 1 | 1.082 |
| ks / correct / g32 | cpu | looped | 1 | 1.065 |
| ks / correct / g32 | cpu | fno | 1 | 0.6352 |
| wave / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.2341 |
| wave / coefficient / g32 | cpu | physics_calibrated | 1 | 0.2376 |
| wave / coefficient / g32 | cpu | transformer | 1 | 1.081 |
| wave / coefficient / g32 | cpu | looped | 1 | 1.072 |
| wave / coefficient / g32 | cpu | fno | 1 | 0.6445 |
| wave / correct / g32 | cpu | physics_uncalibrated | 1 | 0.2378 |
| wave / correct / g32 | cpu | physics_calibrated | 1 | 0.2397 |
| wave / correct / g32 | cpu | transformer | 1 | 1.097 |
| wave / correct / g32 | cpu | looped | 1 | 1.076 |
| wave / correct / g32 | cpu | fno | 1 | 0.6298 |
| advection_diffusion / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.1975 |
| advection_diffusion / coefficient / g32 | cpu | physics_calibrated | 1 | 0.2029 |
| advection_diffusion / coefficient / g32 | cpu | transformer | 1 | 1.077 |
| advection_diffusion / coefficient / g32 | cpu | looped | 1 | 1.067 |
| advection_diffusion / coefficient / g32 | cpu | fno | 1 | 0.6276 |
| advection_diffusion / correct / g32 | cpu | physics_uncalibrated | 1 | 0.1962 |
| advection_diffusion / correct / g32 | cpu | physics_calibrated | 1 | 0.2008 |
| advection_diffusion / correct / g32 | cpu | transformer | 1 | 1.064 |
| advection_diffusion / correct / g32 | cpu | looped | 1 | 1.066 |
| advection_diffusion / correct / g32 | cpu | fno | 1 | 0.6247 |
| allen_cahn / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.4318 |
| allen_cahn / coefficient / g32 | cpu | physics_calibrated | 1 | 0.4375 |
| allen_cahn / coefficient / g32 | cpu | transformer | 1 | 1.066 |
| allen_cahn / coefficient / g32 | cpu | looped | 1 | 1.07 |
| allen_cahn / coefficient / g32 | cpu | fno | 1 | 0.6265 |
| allen_cahn / correct / g32 | cpu | physics_uncalibrated | 1 | 0.4323 |
| allen_cahn / correct / g32 | cpu | physics_calibrated | 1 | 0.4407 |
| allen_cahn / correct / g32 | cpu | transformer | 1 | 1.084 |
| allen_cahn / correct / g32 | cpu | looped | 1 | 1.094 |
| allen_cahn / correct / g32 | cpu | fno | 1 | 0.6442 |
| wave / coefficient / g32 | cpu | physics_uncalibrated | 1 | 0.2378 |
| wave / coefficient / g32 | cpu | physics_calibrated | 1 | 0.238 |
| wave / coefficient / g32 | cpu | transformer | 1 | 1.099 |
| wave / coefficient / g32 | cpu | looped | 1 | 1.094 |
| wave / coefficient / g32 | cpu | fno | 1 | 0.6283 |
| wave / correct / g32 | cpu | physics_uncalibrated | 1 | 0.2393 |
| wave / correct / g32 | cpu | physics_calibrated | 1 | 0.2502 |
| wave / correct / g32 | cpu | transformer | 1 | 1.138 |
| wave / correct / g32 | cpu | looped | 1 | 1.095 |
| wave / correct / g32 | cpu | fno | 1 | 0.6548 |
| advection_diffusion / coefficient / g64 | cpu | physics_uncalibrated | 1 | 0.2178 |
| advection_diffusion / coefficient / g64 | cpu | physics_calibrated | 1 | 0.2066 |
| advection_diffusion / coefficient / g64 | cpu | transformer | 1 | 2.51 |
| advection_diffusion / coefficient / g64 | cpu | looped | 1 | 2.406 |
| advection_diffusion / coefficient / g64 | cpu | fno | 1 | 0.6859 |
| advection_diffusion / correct / g64 | cpu | physics_uncalibrated | 1 | 0.21 |
| advection_diffusion / correct / g64 | cpu | physics_calibrated | 1 | 0.2111 |
| advection_diffusion / correct / g64 | cpu | transformer | 1 | 2.435 |
| advection_diffusion / correct / g64 | cpu | looped | 1 | 2.364 |
| advection_diffusion / correct / g64 | cpu | fno | 1 | 0.722 |
| allen_cahn / coefficient / g64 | cpu | physics_uncalibrated | 1 | 0.439 |
| allen_cahn / coefficient / g64 | cpu | physics_calibrated | 1 | 0.4536 |
| allen_cahn / coefficient / g64 | cpu | transformer | 1 | 2.503 |
| allen_cahn / coefficient / g64 | cpu | looped | 1 | 2.409 |
| allen_cahn / coefficient / g64 | cpu | fno | 1 | 0.7257 |
| allen_cahn / correct / g64 | cpu | physics_uncalibrated | 1 | 0.4341 |
| allen_cahn / correct / g64 | cpu | physics_calibrated | 1 | 0.425 |
| allen_cahn / correct / g64 | cpu | transformer | 1 | 2.35 |
| allen_cahn / correct / g64 | cpu | looped | 1 | 2.366 |
| allen_cahn / correct / g64 | cpu | fno | 1 | 0.693 |
| wave / coefficient / g64 | cpu | physics_uncalibrated | 1 | 0.4498 |
| wave / coefficient / g64 | cpu | physics_calibrated | 1 | 0.4623 |
| wave / coefficient / g64 | cpu | transformer | 1 | 2.526 |
| wave / coefficient / g64 | cpu | looped | 1 | 2.55 |
| wave / coefficient / g64 | cpu | fno | 1 | 0.7432 |
| wave / correct / g64 | cpu | physics_uncalibrated | 1 | 0.4894 |
| wave / correct / g64 | cpu | physics_calibrated | 1 | 0.5094 |
| wave / correct / g64 | cpu | transformer | 1 | 2.542 |
| wave / correct / g64 | cpu | looped | 1 | 2.68 |
| wave / correct / g64 | cpu | fno | 1 | 0.8095 |
| cahn_hilliard / coefficient / g64 | cpu | physics_uncalibrated | 1 | 0.4883 |
| cahn_hilliard / coefficient / g64 | cpu | physics_calibrated | 1 | 0.5191 |
| cahn_hilliard / coefficient / g64 | cpu | transformer | 1 | 2.541 |
| cahn_hilliard / coefficient / g64 | cpu | looped | 1 | 2.56 |
| cahn_hilliard / coefficient / g64 | cpu | fno | 1 | 0.7337 |
| cahn_hilliard / correct / g64 | cpu | physics_uncalibrated | 1 | 0.5262 |
| cahn_hilliard / correct / g64 | cpu | physics_calibrated | 1 | 0.5188 |
| cahn_hilliard / correct / g64 | cpu | transformer | 1 | 2.613 |
| cahn_hilliard / correct / g64 | cpu | looped | 1 | 2.62 |
| cahn_hilliard / correct / g64 | cpu | fno | 1 | 0.7722 |
| cahn_hilliard / structural / g64 | cpu | physics_uncalibrated | 1 | 0.4242 |
| cahn_hilliard / structural / g64 | cpu | physics_calibrated | 1 | 0.4328 |
| cahn_hilliard / structural / g64 | cpu | transformer | 1 | 2.38 |
| cahn_hilliard / structural / g64 | cpu | looped | 1 | 2.366 |
| cahn_hilliard / structural / g64 | cpu | fno | 1 | 0.666 |
| fitzhugh_nagumo / coefficient / g64 | cpu | physics_uncalibrated | 1 | 0.5435 |
| fitzhugh_nagumo / coefficient / g64 | cpu | physics_calibrated | 1 | 0.5444 |
| fitzhugh_nagumo / coefficient / g64 | cpu | transformer | 1 | 2.349 |
| fitzhugh_nagumo / coefficient / g64 | cpu | looped | 1 | 2.447 |
| fitzhugh_nagumo / coefficient / g64 | cpu | fno | 1 | 0.676 |
| fitzhugh_nagumo / correct / g64 | cpu | physics_uncalibrated | 1 | 0.5215 |
| fitzhugh_nagumo / correct / g64 | cpu | physics_calibrated | 1 | 0.5202 |
| fitzhugh_nagumo / correct / g64 | cpu | transformer | 1 | 2.339 |
| fitzhugh_nagumo / correct / g64 | cpu | looped | 1 | 2.34 |
| fitzhugh_nagumo / correct / g64 | cpu | fno | 1 | 0.6746 |
| fitzhugh_nagumo / structural / g64 | cpu | physics_uncalibrated | 1 | 0.5459 |
| fitzhugh_nagumo / structural / g64 | cpu | physics_calibrated | 1 | 0.539 |
| fitzhugh_nagumo / structural / g64 | cpu | transformer | 1 | 2.408 |
| fitzhugh_nagumo / structural / g64 | cpu | looped | 1 | 2.379 |
| fitzhugh_nagumo / structural / g64 | cpu | fno | 1 | 0.711 |

## Provenance and limits

Frozen validation choices use archived neural final_val and the new 16-step physics validation scores. Test errors are used only for evaluation. Prior test inspection makes this a post hoc study extension.

development caps each trajectory mean at 1e6; confirmation caps each per-step error at 1e6 before averaging. Nonfinite errors are assigned the cap; failed trajectories are retained.

Calibration-file hashes must match the freeze; the freeze must match the protocol; evaluation arrays and their linked calibration/freeze must match their recorded hashes. Frozen choices are reconstructed from the validation scores and the declared tie order. Full/pilot evaluations require 64 trajectories per distribution, and smoke requires eight.

Incompatible archived outcomes are withheld from neural comparisons. Compatibility requires the original solver source, full counts, ordered trajectory IDs and matching normalization, together with exact development/test archive data hashes or successful numerical reference replay. Exact data hashes establish identical forecast targets even if CPU physical forecasts differ from archived GPU forecasts through floating-point rounding. These replay deviations remain visible in each compatibility record; tolerances are unchanged.

Fully observed, noiseless, one-dimensional simulated fields and known forcing define the evaluated use case.

## Diagnostics requiring attention

- development__allen_cahn__coefficient: exact development/test data hashes establish shared targets; CPU/GPU physical-reference rounding differences remain recorded
- development__ks__correct: exact development/test data hashes establish shared targets; CPU/GPU physical-reference rounding differences remain recorded
- resolution_g32__allen_cahn__coefficient: exact development/test data hashes establish shared targets; CPU/GPU physical-reference rounding differences remain recorded


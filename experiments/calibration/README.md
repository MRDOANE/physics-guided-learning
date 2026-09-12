# Calibrated physics and neural forecast comparison

This package asks whether fitting the available physical equations to observed training trajectories changes the choice between a physical solver and a neural forecaster. It fits three shared coefficient corrections for each physical setting and compares the resulting forecasts with the saved neural results. It also measures the cost of calibration and the execution time of the different forecasting implementations.

**No neural model is trained.** The package contains the archived neural error arrays, metadata, reference errors, configurations, and original solver/model code needed for the comparison. No GitHub checkout or previous output directory is required. The original release is [Zenodo DOI 10.5281/zenodo.22728288](https://doi.org/10.5281/zenodo.22728288), with source at [MRDOANE/physics-guided-learning](https://github.com/MRDOANE/physics-guided-learning).

## Start in RunPod

Upload `physics_calibrated_baseline.zip` and `launch_calibrated_physics.sh` to `/workspace` through JupyterLab. Keep them beside each other. Use an environment with working PyTorch; CPU PyTorch is sufficient. The launcher preserves the installed PyTorch package.

In a JupyterLab terminal:

```bash
bash /workspace/launch_calibrated_physics.sh plan
bash /workspace/launch_calibrated_physics.sh smoke
bash /workspace/launch_calibrated_physics.sh pilot
```

`plan` prints the work without fitting or installing dependencies. `smoke` checks four small settings. `pilot` executes two complete Allen–Cahn settings at grid 64, including test comparisons and timing. To run the complete experiment:

```bash
JOBS=4 THREADS=1 TIMING_DEVICE=auto bash /workspace/launch_calibrated_physics.sh full
```

To include a cost estimate, supply the actual rate shown for your Pod. For example, replace the illustrative rate below:

```bash
POD_HOURLY_RATE=0.50 JOBS=4 THREADS=1 bash /workspace/launch_calibrated_physics.sh full
```

`TIMING_DEVICE=cpu` keeps everything on the CPU. `auto` attempts a real CUDA allocation and falls back to CPU timing if it fails. `TIMING_DEVICE=none` skips the optional timing benchmark. A CUDA configuration problem does not prevent CPU calibration. The script leaves GPU visibility settings alone.

The default installation is `/workspace/physics_calibrated_baseline`. The launcher works from any current directory when invoked by its full path. `INSTALL_ROOT` changes the parent installation directory. `OUTPUT_DIR` changes the result directory. Repeating a command resumes verified completed files. A different protocol, source release, or software environment requires a new output folder. Existing files are preserved.

## Hardware and cost

Start with **8 vCPUs, 16 GB system RAM, four calibration workers and one thread per worker**. A GPU is optional. A large GPU or extra VRAM will give little benefit to the calibration because its optimizer and physical stepping run on the CPU. Data regeneration temporarily uses eight CPU threads to reproduce the archived generator's FFT arithmetic. At these small grids, process startup, data generation and reporting can be substantial fractions of runtime.

The pilot reports its measured calibration time and a simple projection for 30 settings. The projection assumes equal setting costs; nonlinear systems can need different numbers of optimizer evaluations. Runtime estimates from the accompanying validation notes are measurements on the development machine, with no guarantee of equal RunPod performance.

The terminal separates data generation, calibration, evaluation and timing. Pod cost is elapsed run time multiplied by your supplied hourly rate. Storage, idle time, dependency installation and historical neural training are excluded. Interrupted work before the execution summary is committed can be undercounted. The full profile performs its own calibration; the pilot is an independent, small cost check.

## What is compared

The full profile has **30 equation/prior/grid/stage settings across eight physical families** and reuses **1,584 archived neural fits**.

| Stage | Families | Grid | Physical information supplied |
| --- | --- | ---: | --- |
| Development | Wave, Burgers, Kuramoto–Sivashinsky, advection–diffusion, Allen–Cahn, Gray–Scott | 32 | Correct or biased coefficients |
| Resolution | Wave, advection–diffusion, Allen–Cahn | 32 and 64 | Correct or biased coefficients |
| Selector extension | Cahn–Hilliard, FitzHugh–Nagumo | 64 | Correct coefficients, biased coefficients, or an omitted equation term |

Forecast options include unchanged physics, calibrated physics, persistence, the standard transformer, the looped transformer and the Fourier neural operator (FNO). Neural options retain their original augmentation arms. The archive contains these three neural architectures. It contains no separate latent world-model architecture.

Every setting has 128 training trajectories and 24 validation trajectories. In-distribution evaluation uses 64 complete held-out trajectories. Another 64 trajectories have physical coefficients outside the training range. Forecast errors are reported at 64 steps, the primary horizon, and 96 steps. The same training-derived channel scales normalize each method's squared errors. Scores remain separated by equation, stage, prior and distribution.

## Calibration protocol

The solver receives the same approximate physical coefficients that were available to the neural models. Three positive multipliers, shared across every trajectory within a setting, modify those coefficients. The fitting function receives observed states, actions and supplied coefficients. True generating coefficients are excluded from its input object. Initial optimizer guesses are generic and do not contain the known inverse bias.

Each full setting fits six candidates: training rollout lengths of one and four steps, each with three generic starts, `(1,1,1)`, `(0.8,1.2,1)` and `(1.2,0.8,1)`. Each trajectory contributes two evenly spaced training windows. The objective is normalized squared forecast error. SciPy least squares operates on log multipliers with a three-point numerical Jacobian and a budget of 60 function evaluations per candidate. Jacobian evaluations require extra residual calls; those calls and their cost are recorded.

Multipliers can range from 0.001 to 1000. These broad bounds keep numerical optimization finite. A preliminary small structural smoke check reached the original narrower lower bound, which motivated expanding the range before the production run. This design history is disclosed because the study is exploratory. Selected boundary coordinates, Jacobian rank, condition number, failed trials and optimizer termination are saved. These diagnostics have no attached publication or hypothesis acceptance gate.

The unchanged solver is an additional candidate. Validation selects the candidate with the lowest 16-step recursive forecast error. This matches the horizon and aggregation used for the archived neural validation scores. Fitting uses float64 stepping. Validation and forecast evaluation use float32 states and the original solver's internal arithmetic. Validation retains failed trajectories at the original loss cap.

Calibration preserves the supplied equation. The Cahn–Hilliard structural case continues to omit its cubic term. The FitzHugh–Nagumo structural case continues to omit the recovery-variable contribution to the activator equation. Coefficient correction cannot add either term back. The coefficient-bias experiments admit an exact global multiplicative correction, so success there measures how much simple system identification explains the earlier neural advantage. These cases provide a relatively favorable setting for calibration.

All requested calibrations and validation choices are written to a frozen manifest before the runner generates test trajectories or opens archived neural test errors. A second validation choice selects across unchanged physics, calibrated physics and the existing neural candidates, separately for each archived neural training seed. Its test errors are reported alongside those of the validation-best neural candidate. Acquiring all neural candidates would incur historical training/search cost in a fresh deployment; the present reuse incurs no new neural training cost.

## Data identity and numerical checks

The package regenerates the original data from the saved configuration and seeds using the original float64 Torch generator, 32-trajectory chunks, burn-in and final float32 representation. Original data generation used eight Torch CPU threads; the runner preserves that setting and restores the requested fitting thread count afterward.

Before reusing neural test errors, the runner verifies ordered trajectory IDs and normalization. Exact development and test NPZ hashes establish that the underlying arrays match the archive. The runner also compares newly computed unchanged-physics and persistence error curves with archived reference curves. GPU and CPU arithmetic can produce different reference errors even with identical input-data hashes; these differences remain visible in the report.

If exact data hashes differ, numerical reference replay must pass before the archived neural comparison is used. The replay tolerance is absolute `1e-7` plus relative `1e-4` on each normalized squared-error value. This fallback provides numerical compatibility evidence, with less certainty than exact data identity. If neither route establishes compatibility, the report withholds comparisons with saved neural errors and keeps the new calibrated-versus-unchanged physics comparison on common regenerated data.

`BACKEND=numpy` provides a diagnostic smoke fallback. Its equations were independently checked against the original Torch implementation in float64. Float32 intermediate arithmetic differs, so this backend does not produce scientific labels or archived neural comparisons.

## Reading the output

Completion, archive compatibility and the scientific direction of each comparison are reported separately.

| Label | Meaning for a comparison |
| --- | --- |
| GREEN | The candidate has lower error, with a pointwise 95% interval excluding zero in that direction. |
| YELLOW | The interval includes zero. |
| RED | The candidate has higher error, with the interval excluding zero in that direction. |
| BLUE | The compared trajectory scores are identical. |
| NOT_EVALUATED | Diagnostic evidence or an unavailable scientific comparison. |

The report prints both absolute normalized MSE and relative changes. Positive reductions favor the candidate. **There is no minimum required improvement and no required number of successful systems.** Mixed results can explain which physical information makes a solver useful. Very small differences near numerical precision should be read in terms of their absolute errors; a percentage can overstate their practical importance.

Intervals use 5,000 paired bootstrap replicates, resampling shared test trajectories and archived neural training seeds. The physical calibration fit is held fixed. These intervals omit uncertainty from drawing a new calibration training set. They are exploratory pointwise intervals, with no multiple-comparison adjustment or global accept/reject decision. Previously inspected test results informed this extension; its new freeze does not make the overall study an independent preregistered confirmation.

The original scoring conventions remain intact: development caps each trajectory's mean error at `1e6`; the later stages cap per-step errors before averaging. Divergent trajectories remain in the scores. Diagnostic tables disclose failures and cap usage.

Full results appear under `outputs/full/`:

- `reports/report.md` and `report.json`: narrative report, labels, diagnostics and complete machine-readable results.
- `reports/absolute_accuracy.csv`: every neural architecture/arm and physical option, both distributions and horizons.
- `reports/comparisons.csv`: paired differences and intervals, including validation-based selection.
- `reports/matched_timing.csv`: newly measured forward execution cost on one device.
- `jobs/`: every calibration candidate, selected coefficients and raw per-step physics errors.
- `freeze.json`: hashed calibrations and all pre-evaluation choices.
- `execution.json`: runtime components and supplied-rate cost estimate.
- `full_results.zip`: upload this file for analysis. Regenerable raw trajectory caches are excluded; their hashes are included.

To repeat the report in a terminal:

```bash
bash /workspace/launch_calibrated_physics.sh report
```

In a JupyterLab notebook, this also displays a colored completion panel:

```python
%run /workspace/physics_calibrated_baseline/check_results.py
```

For pilot output, add `--output /workspace/physics_calibrated_baseline/outputs/pilot` to the notebook command. The corresponding terminal report uses that path as `OUTPUT_DIR`.

## Timing interpretation

New timing measurements use identical resident float32 inputs on the same CPU or GPU, five warmup calls per implementation and five randomized, interleaved blocks of 20 calls. CUDA timing synchronizes each block. Physics timing includes applying fitted multipliers. Neural timing includes normalization and all model layers. The benchmark excludes host/device transfers, history updates, fitting and error calculation.

The release has saved neural errors and metadata, with no trained weight checkpoints. New neural timing therefore uses initialized models with the archived architecture. All layers execute, including their zero-initialized output heads. These measurements describe architecture execution. They do not measure fresh neural accuracy or latency of the exact historical trained checkpoint.

Historical training and latency records remain clearly identified. They were measured on the original machine and cannot establish a matched-hardware training-cost comparison with new CPU calibration. Current forward timing compares the existing implementations; the solver has no new caching or optimization. Full rollout throughput, observational noise, partial observability, per-test-trajectory identification and additional equation discovery are outside this extension.

## Software and verification

Python 3.10+, NumPy, SciPy and an existing PyTorch installation are required for pilot/full. The launcher creates a virtual environment that can use system packages and installs only missing NumPy/SciPy dependencies. It verifies the complete release manifest and safely extracts the ZIP. It checks signatures and checksums before reusing data, calibration, evaluations and timing.

Independent numerical tests are included in `tests/`. From the installed package:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests check solver equations against an independent NumPy implementation, parameter rounding and archived data hashes. Full data hashes may differ on another numerical stack; the runner's documented compatibility checks determine whether its archived comparisons can be used. See `VALIDATION.md` for the development-machine checks and measured runtime.

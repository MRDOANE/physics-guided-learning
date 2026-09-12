# Physics families and prospective augmentation selection

This is a standalone follow-up to the wave mechanism experiment. It broadens the controlled tests to six physical families and asks whether a small learned rule can choose when, and how, to use physics augmentation on an unseen family. All observations are generated locally by numerical simulators. No external dataset, model download, experiment service, or paid API is required.

## Run on RunPod

Choose a recent **PyTorch GPU template**, upload these two files into the same persistent directory, and open a terminal there:

```bash
bash launch_physics_family_selector.sh smoke
bash launch_physics_family_selector.sh full
```

The files are `physics_family_selector_v1.zip` and `launch_physics_family_selector.sh`. The launcher extracts and verifies the source, creates a small virtual environment that can use the template's existing packages, and preserves its PyTorch/CUDA installation. It only installs NumPy if the installed version does not satisfy `requirements.txt`.

For the full experiment, use **one NVIDIA RTX 4090 with 24 GB VRAM, 8 vCPUs, and 32–64 GB system RAM**. One GPU is sufficient; the program trains candidates sequentially and does not distribute training across multiple GPUs. A persistent volume matters more than extra GPUs because it preserves the dataset cache, resumable checkpoints, and completed runs.

The full profile is substantially larger than the earlier 152-fit experiment: **864 fits at 4,000 updates each**, plus simulation, probe, selection, and evaluation work. Budget approximately **18–36 hours on one RTX 4090**, and reserve **at least 50 GB of persistent disk space**. This is a planning estimate extrapolated from the earlier measured 4090 run, not a GPU timing of this new release. Use the smoke run to check installation and the pilot to measure performance before committing the full GPU budget. Smoke timing does not reliably predict the full run.

To inspect the fixed schedule without starting experiments:

```bash
bash launch_physics_family_selector.sh plan full
```

To use a custom persistent result directory and thread count:

```bash
OUTPUT_DIR=/workspace/family_selector_results THREADS=8 \
  bash launch_physics_family_selector.sh full
```

`DEVICE` accepts `cuda`, `cpu`, or `auto`. Full defaults to `cuda`; smoke and pilot default to `auto`. A CPU smoke check is available:

```bash
DEVICE=cpu THREADS=4 bash launch_physics_family_selector.sh smoke
```

For a persistent terminal session, start the command inside `tmux` if your template provides it. Keep the RunPod instance and volume while it is running. Closing a browser tab does not itself mean a shell process is safely detached.

## What it runs

| Component | Included settings |
|---|---|
| Physical families | Wave, viscous Burgers, Kuramoto–Sivashinsky, advection–diffusion, Allen–Cahn, Gray–Scott |
| Architectures | Ordinary transformer, shared-block looped transformer, Fourier neural operator |
| Physics priors | Correct coefficients; deliberately perturbed coefficients |
| Augmentation choices | None; IID perturbations; smooth perturbations; response-matched perturbations |
| Development transfer | Leave one family out across the first five families |
| Final confirmation family | Gray–Scott, excluded from selector development labels |
| Selector | Regularized linear prediction from observable diagnostics and short validation probes |
| Selection alternatives | Best development-selected fixed choice per architecture, direct short-probe selection, and other report-defined baselines |
| Evaluation | New in-distribution and parameter-shift test trajectories after the selector is frozen |

These are small, controlled, one-dimensional periodic systems with observed states and actions. A loop is repeated computation within a prediction, not an additional physical time step. The models act as learned dynamics/world models; “world model” is not treated as a fourth mutually exclusive architecture. FNO is retained as a strong field-model baseline.

| Profile | Seeds | Fits | Updates per fit | Probe update | Train / validation / ID test / OOD test trajectories per family | Primary / secondary horizon |
|---|---:|---:|---:|---:|---|---|
| Smoke | 1 | 144 | 6 | 2 | 4 / 2 / 3 / 3 | 4 / 8 |
| Pilot | 1 | 144 | 300 | 30 | 32 / 8 / 12 / 12 | 64 / 96 |
| Full | 6 | 864 | 4,000 | 300 | 128 / 24 / 64 / 64 | 64 / 96 |

The full profile uses training seeds 101, 202, 303, 404, 505, and 606. Its state grid has 32 cells, each trajectory has 160 recorded transitions after 24 burn-in steps, and the models use context 4, width 64, and nominal depth 3. The looped model shares block parameters. A common width, depth setting, and update budget do **not** imply equal parameters, FLOPs, or wall time.

## Read the final results

At the end, the program prints mechanism findings, selector transfer results, and cost information directly to the terminal. It also saves a compact results archive:

```text
physics_family_selector/outputs/full/full_results.zip
```

When using `OUTPUT_DIR`, the archive is `OUTPUT_DIR/full_results.zip`. It contains the auditable reports and compact run records; large training arrays and model checkpoints remain in the output directory and are excluded from the sharing ZIP. Keep the original output directory if you want to resume training or retain weights.

Reprint the results later without retraining:

```bash
bash launch_physics_family_selector.sh report
# For a custom output directory:
OUTPUT_DIR=/workspace/family_selector_results \
  bash launch_physics_family_selector.sh report
# For a smoke output:
PROFILE=smoke bash launch_physics_family_selector.sh report
```

The report uses colored status labels when supported and readable words otherwise. **Mechanism status and selector status are separate.** A complete run can provide useful negative evidence even when a proposed selector fails. Green indicates that a specific prespecified numerical criterion was met; it does not prove a universal mechanism or publication readiness. Smoke and pilot are engineering/exploratory runs and cannot produce a confirmatory science pass. Incomplete or invalid runs must be distinguished from completed scientific failures.

## Selection cost and what this experiment can claim

The selector sees training diagnostics and each candidate's short validation probe. Its diagnostics describe prior/observation disagreement, IID and smooth response gains, low/high spatial spectral power, and observed state increments. The available architecture and augmentation choice are also known. It does not see the test trajectories, hidden true parameters, or the experimenter's “correct/coefficient” label as input. Development labels come from completed development-family validation outcomes. Gray–Scott is withheld from those labels; its permitted short probes describe the new selection task.

For full runs, probing all four choices for 300 updates and continuing only the selected choice to 4,000 updates costs **4,900 nominal training updates**, versus 4,000 for one fixed choice and 16,000 for training every choice. That is a 22.5% update overhead relative to one fixed choice. Actual measured time is the meaningful cost because model and augmentation costs differ. The program executes the full candidate bank to measure counterfactual outcomes. Any reported “selected route” saving is reconstructed from that bank, not a GPU saving realized by this experiment. The expensive development bank is also a real cost and must be disclosed rather than treated as free.

The simplest serious competing selector is “pick the best short validation probe.” If the learned rule cannot improve on it at comparable information and cost, the added selection model is not justified by these experiments. Test-best oracle choices are only upper-bound references; they cannot be deployed without test leakage.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) and the JSON protocol saved with each run for the design, leakage boundary, controls, and limitations.

## Resume and reproducibility

Rerun the same command with the same output directory to resume. Keep the same source release, profile, and environment. The source/protocol hashes protect an existing run from being silently mixed with a different design, and checkpoints preserve optimizer and random-number state. A crash can lose work after the most recent checkpoint. A changed PyTorch/CUDA stack is not an exact-resume environment.

The outer launcher never overwrites a differing existing source release. If it detects edited source or another manifest, move the new ZIP and launcher to a new directory; your old outputs remain in place. Do not edit configuration or Python files during an experiment. Use a new output directory for a deliberately changed protocol.

Optional developer checks require `pytest`, which is not a runtime dependency:

```bash
cd physics_family_selector
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

This release's local validation record describes what was actually checked. A locally passing CPU smoke run is not a completed GPU full experiment.

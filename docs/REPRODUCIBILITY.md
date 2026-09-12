# Reproduction guide

## Inspect the saved experiment

Make a copy of `inspect_results.ipynb` in the same repository folder, open the copy in JupyterLab, and select **Run → Run All Cells**. Saving outputs in the copy preserves the original notebook hash. It verifies this release's files and prints the saved confirmation results. This does not train models or rerun statistical calculations. The equivalent terminal command, from the repository root, is:

```text
python scripts/verify_release.py
```

The default checker requires Python 3.10 or later and its standard library. It verifies the packaging inventory, experimental record counts, evaluation hashes, source fingerprints, and the linkage of the filtered development bank. A green integrity result means the archived files agree with their recorded hashes; it is separate from the scientific labels.

The manifest records the initial package. Editing a tracked file later will correctly report a changed release. Keep an unmodified copy for verification. Adding new unlisted files does not invalidate the original inventory.

## Recompute reports from evaluation arrays

Install the matching source's NumPy and PyTorch dependencies in an environment of your choice. The archived versions were Python 3.12.3, NumPy 2.1.2, and PyTorch 2.8.0+cu128. The confirmation requirements accept NumPy >=1.26,<3 and PyTorch >=2.8,<3; the historical source expects PyTorch to be supplied by the environment. A GPU is unnecessary for report replay, but the original modules import PyTorch. Report bootstrap calculations run on the CPU. Use Linux, macOS, or WSL for confirmation replay because the original imports require `fcntl`.

```text
python scripts/replay_reports.py confirmation
python scripts/replay_reports.py development
```

The wrapper copies evidence to a new directory under `outputs/report_replay/` and calls the original analysis for the selected stage. It preserves `results/`. It chooses a new numbered directory when a previous replay exists. The initial release's packaging check does not rerun those full bootstrap reports. See [the validation record](../provenance/VALIDATION.md) for checks actually performed while preparing this repository.

For an optional NumPy-only replay of the frozen guard decisions:

```text
python scripts/verify_release.py --selector-replay
```

Floating-point coefficients can differ slightly across NumPy/BLAS environments. The meaningful portable check is that the selected rule and target choices agree; bitwise model-coefficient equality is not required.

## Fresh model training

Full training is intended for Linux, WSL, or RunPod. The original runners use Linux file locking. Choose an environment with compatible PyTorch and NumPy already installed. The confirmation launcher preserves the installed Torch package. The historical development launcher may create an environment and install NumPy, so the direct Python entry point below is clearer.

From the repository root, inspect the follow-up plan without importing Torch or training:

```text
python experiments/confirmation/run_experiment.py plan
```

Run a small engineering smoke profile:

```text
python experiments/confirmation/run_experiment.py smoke --device cpu --threads 2 --output outputs/confirmation_check
```

Run the completed follow-up design on a new GPU output directory:

```text
bash experiments/confirmation/launch.sh full --output ../../outputs/confirmation
```

The included launcher sets `CUBLAS_WORKSPACE_CONFIG` before Python starts, which is required by the source's deterministic CUDA operations. It changes into the source directory, so `../../outputs/confirmation` resolves to the repository's root output folder.

`resolution` or `selector` can replace `full` to request one stage. `--max-hours` supplies a soft per-invocation pause boundary between committed jobs. It does not stop Pod billing. `--hourly-rate` records the user's actual rate for the runner's estimates. The confirmation plan's printed historical `launch_physics_confirmation.sh` examples refer to an earlier distribution wrapper; use the included `experiments/confirmation/launch.sh` for GPU execution.

To retrain the original development experiment, open a terminal in `experiments/development/` and use:

```text
python -m lpb.cli run --profile full --output ../../outputs/development/full --device cuda --threads 8
```

That stage uses its own original configuration and selector. Its archived validation outcomes were filtered into the bank included with the confirmation source. A fresh stochastic development rerun will not be byte-identical across hardware/software stacks. The confirmation reproduction deliberately uses the archived frozen bank, so it can be run independently. Replacing that bank with new fitted outcomes defines a separate experiment and should be documented as such.

All fits use synthetic observations generated locally. The archive has no model checkpoints, so do not point a training command at `results/`. A newly started run writes resumable checkpoints to its new output directory. Rerun the same command with unchanged source, configuration, and environment to resume that new run.

## Hardware and measured duration

Use one 24 GB GPU, 8–16 vCPUs, 32–64 GB system RAM, and at least 50 GB persistent space when retraining. The sequential runners do not distribute fits across GPUs. On the recorded RTX 3090 Ti, development took approximately 13.70 hours and confirmation took 11.067 hours. Confirmation recorded 10.612 hours of training/validation and 0.012 hours of CPU data generation/audits; elapsed time includes other work. GPU/training throughput was the main measured workload. These measurements do not guarantee the same time on another Pod.

## Existing source checks

With the training dependencies available, open a terminal in `experiments/confirmation/` and run:

```text
python -m unittest discover -s tests -v
```

The existing suite includes statistical-label checks, selector information barriers, physics checks, inventory checks, and a small checkpoint-resume exercise. The development tests use pytest from `experiments/development/`. A smoke or unit-test pass checks implementation; it does not supply full scientific evidence.

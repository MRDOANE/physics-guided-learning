# Package verification

The package was developed and checked with Python 3.12, NumPy 2.3.5, SciPy and CPU PyTorch 2.8.0. No GPU was required and no neural optimizer was run.

An initial complete integration run executed all 30 full-budget calibration settings, both held-out distributions, saved-neural comparisons where archive compatibility passed, and matched CPU timing. It completed in 73.3 seconds: data generation 33.9 seconds, calibration wall time 10.1 seconds, evaluation 11.0 seconds and matched timing 17.8 seconds. Four calibration workers each used one thread. This preliminary run used one data-generation thread; subsequent checks identified and implemented the original eight-thread data-generation setting. The final launcher is tested separately with that setting. These timings are development-machine measurements, not a RunPod service estimate. Allow several minutes on a typical eight-vCPU Pod and use the pilot to measure your machine.

Independent numerical checks covered 96 cases across all eight equation families, both grids, every prior and coarse/fine integration. The NumPy and original Torch float64 implementations agreed to a maximum absolute difference of 5.08e-15 over four steps. A further 144 cases verified that the shared development and confirmation equations produce identical float32/float64 results. Four full wave and Allen–Cahn development/test datasets reproduced their exact archived NPZ hashes using eight data-generation threads.

Actual CPU forward timing executed physical solvers and all three neural architectures. Neural parameter counts matched the archived records. The initialized neural models are used exclusively for execution timing; their outputs supply no new accuracy evidence.

The test suite also checks coefficient recovery from observed transitions, retention of divergent trajectories and the frozen-calibration test-access barrier. Report-side fixture checks rejected modified choices and truncated evaluations. The launcher was checked for fresh installation, preservation of existing outputs, source modification detection, different-release rejection, path traversal and symlink rejection.

Fresh matched GPU timing depends on the user's CUDA installation. The CPU path and automatic CPU fallback remain available. The package keeps the installed Torch distribution intact.

# Release validation

Local validation date: 2026-09-11. Runtime: Python 3.12.14, PyTorch 2.8.0+cpu,
NumPy 2.3.5. No CUDA device was available for release testing.

## Checks completed

- 40 tests pass, including 24 physics subtests. Tests cover analytic Fourier
  propagation, reaction invariants, numerical time refinement, finite gradients,
  family-grouped feature scaling and tuning, held-out-label mutation, complete
  candidate inventories, paired statistics, failure caps, and cost arithmetic.
- A 144-fit end-to-end CPU smoke run completed the data, probe, development,
  frozen selection, Gray–Scott continuation, held-out evaluation, report, and
  result-export phases. Short runs correctly return scientific status
  `NOT_EVALUATED`; execution completion is reported separately.
- Probe-then-continuation training produces exactly the same model tensors and
  validation history as uninterrupted training on the same CPU stack.
- An injected interruption after the authoritative checkpoint write, before
  the record write, resumes and repairs the stale record. Changing the protocol
  refuses resume. An early numerical failure cannot retain a favorable partial
  probe score.
- Full-profile training and validation data were generated for all six
  families: 128 training and 24 validation trajectories, 160 transitions,
  24 burn-in steps, and grid 32. All references were finite. Independent ID/OOD
  32-step fine/check solver audits had maximum normalized error `3.08e-6`,
  below the prespecified `1e-3` threshold. No full-profile test outcomes were
  used to choose the selection rule.
- All 12 smoke augmentation banks and four full-size nonlinear banks
  (Allen–Cahn and Gray–Scott, both priors, 4,096 windows) were independently
  reconstructed. Maximum observed relative input-RMS mismatch was `1.98e-6`;
  maximum operator-response-RMS mismatch was `3.21e-6`. The numerical response
  matching tolerance is `rtol=0.002`, `atol=0.000002`.
- Bash syntax, archive extraction, same-release preservation, edited-source
  refusal, and path-traversal rejection were checked.

## Limits

The 864-fit GPU full experiment has **not** been run here. The 18–36 hour RTX
4090 estimate is an extrapolation from the earlier measured experiment, not a
new GPU benchmark. A passing local smoke run establishes pipeline operation,
not a successful scientific result.

Temporal convergence is checked at one fixed spatial grid. This is a
controlled finite-resolution synthetic benchmark, not evidence of spatial
convergence or validation against physical measurements. Allen–Cahn and
Gray–Scott are distinct equations within the broader reaction–diffusion class.
Selection cost figures reconstruct possible deployment routes from an actual
all-candidate research bank; this run does not realize those potential savings.

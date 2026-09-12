# Prospective augmentation selection

The selector chooses one augmentation for a **fixed architecture**. It does not
choose an architecture, change its training budget, inspect test losses, or use
the generator's prior-mismatch label. Candidate arms are `none`, `iid`, `smooth`,
and `response_matched`, in that order. Ties prefer the earlier arm, so `none` wins
an exact tie. A useful result may be that abstention is often the best decision.

## Information and phase barrier

The full protocol uses five development physical families: wave, Burgers,
Kuramoto–Sivashinsky, advection–diffusion, and Allen–Cahn. Gray–Scott is a locked
confirmation family. The confirmation family is known when defining the
protocol; it is not used to fit or tune the selector.

For each candidate, training runs first reach the common short-probe checkpoint.
The probe loss is the rollout validation loss selected **only from validation
checkpoints at or before that probe budget**. Development candidates then finish
the full budget, providing full-budget validation labels. Before confirmation
candidates continue or any held-out test trajectories are scored, the program
writes and hashes a frozen selection manifest.

Development evaluation uses leave-one-physical-family-out fits. For example, a
decision for a wave task is trained on Burgers, KS, advection–diffusion and
Allen–Cahn only. Its wave full-budget validation outcome is excluded from the
fit, feature scaling, hyperparameter selection and fixed-policy baseline. The
confirmation fit uses all five development families. Family identity controls
these splits; it is never a prediction feature. Prior identity and seed are
metadata, never features.

`lpb.selection.freeze_choices(records, cfg)` accepts a strict whitelist of fields.
Supplying a test field or another unknown field is an error. Confirmation
`final_val` is discarded if present. The caller must enforce the phase barrier:
this function cannot prevent a separate program from reading test files.

## Model

Six common diagnostics are computed from training data:

- `gap_relative`: relative discrepancy between available coarse physics and
  observed one-step increments;
- `gain_iid` and `gain_smooth`: coarse-physics response magnitudes for the two
  perturbation types, under the diagnostic's common normalization;
- `low_power` and `high_power`: observed training-field spectral power fractions;
- `increment_rms`: normalized observed increment RMS.

The selection code requires these diagnostics to be finite and identical across
arms within a task. They must be calculated from observed training trajectories
and the available physics model, without access to hidden true parameters or
test trajectories. Exact diagnostic definitions are implemented in the
experiment's diagnostics module and fixed with the source snapshot.

For each non-`none` arm, the 34-dimensional design vector contains:

1. the six common diagnostics;
2. three architecture indicators;
3. three augmentation indicators;
4. `log(probe_val_arm / probe_val_none)`;
5. each augmentation indicator multiplied by each common diagnostic;
6. each augmentation indicator multiplied by the probe log ratio.

The regression target is
`log(full_budget_validation_arm / full_budget_validation_none)`.
Losses are floored at `1e-12` and capped at `1e6`. NaN and infinite numerical
outcomes count as `1e6`, so failing candidates remain in the training data.
Negative loss values or missing required development labels are errors.

Ridge regression fits an unpenalized intercept and penalized standardized
features. Feature means and scales are estimated within the training fold only.
Each physical family has equal total weight, regardless of its number of tasks.
Ridge strengths `[0.1, 1, 10, 100]` are selected with another leave-one-family-out
validation loop **inside the outer training families**. Its objective is mean
squared log-ratio prediction error, first averaged within a family and then
across held-out inner families. Ties select the smaller ridge strength. No test
metric chooses the ridge strength or the 5% abstention threshold.

The selector predicts each arm's log loss relative to `none`. The prediction for
`none` is fixed at zero. It chooses the smallest prediction only if that
prediction implies at least a **5% reduction** relative to `none`; otherwise it
selects `none`. This threshold is a prespecified decision convention, not a
calibrated confidence interval or a guarantee of benefit.

The manifest stores coefficients, feature scaling, training families, nested
folds, selected ridge strength, predictions, decisions and fitting time. These
records support an audit of which information each choice could use.

Smoke configurations can use two development families. An outer fold then has
only one training family and cannot run nested family validation; it explicitly
uses the prespecified strength closest to 10. Such a run verifies execution and
does not establish generalization. The full five-family schedule uses nested
validation in every fold.

## Baselines

| Rule | Choice | Target information needed |
|---|---|---|
| `none` | Always no augmentation | Selected candidate's ordinary training |
| `always_smooth` | Always smooth physics augmentation | Training and augmentation-bank construction |
| `best_fixed` | Separately for each architecture, the arm with lowest family-balanced mean validation log ratio in the outer training families | Development candidate bank; selected target training |
| `random` | SHA-256-derived deterministic uniform arm | Fixed protocol seed and task ID; selected target training |
| `hand_rule` | Smooth if `gap_relative < 0.25` and `gain_smooth <= gain_iid`; otherwise none | Training diagnostics and selected target training |
| `probe_best` | Lowest validation loss among the four short probes | All four target probes, then selected continuation |
| `ridge_selector` | Ridge prediction with 5% abstention | Development labels, training diagnostics, all four target probes, then selected continuation |

The random rule is reproducible under record ordering and does not learn from
family names. The task ID only supplies a stable randomization key.

`best_fixed` uses architecture identity, matching the selector's access to this
information. It averages across prior conditions and seeds within each training
family, then gives each family equal weight; it does not choose a separate arm
by prior label. The manifest stores `best_fixed_by_kind` and
`best_fixed_scores_by_kind`. Its older global `best_fixed_arm` and
`best_fixed_scores` fields remain descriptive audit metadata and do not determine
the deployed `best_fixed` choices.

A full-budget validation-selection reference can be computed after all candidates
finish, and an oracle can be scored after tests. Those are higher-cost or
nondeployable references, respectively. Neither is an input to the frozen rule.
`probe_best` is a fixed short-probe tournament, not adaptive successive halving.

## Cost accounting and interpretation

For ridge and `probe_best`, target cost includes **all four** probes, their model
setup, any required diagnostics/augmentation bank, and the selected candidate's
remaining training. The selected probe is reused, not retrained. Fixed rules
incur the selected full fit; the hand rule additionally requires diagnostics.
When the implementation constructs diagnostics and all perturbation banks as one
bundle, its measured bundle cost is charged once to any policy that requires it,
even if a more optimized deployment might construct less. Raw training and bank
times remain available for audit.

The frozen manifest's cost entries are preliminary. At the confirmation freeze
barrier, unknown continuation costs remain `null`, rather than being reported as
zero. The final report recalculates costs from the complete measured bank and
adds selector fitting time according to its documented amortization. Report the
entire development-bank cost and the additional offline full-candidate comparison
cost separately from per-target deployment cost. Selection is not free just
because candidate models were needed for the scientific comparison.

Improved accuracy alone is insufficient to claim efficient selection. The
selector should be compared with `none`, `best_fixed`, the hand rule and
`probe_best`, showing both loss and measured cost. A failure to beat a simple rule
is informative and must remain visible. A single confirmation family is one
independent family-level confirmation, regardless of its number of trajectories,
architectures or seeds. The five development-family folds overlap in their
training data and are not five independent replications of the learned rule.

## API contract

Every input record has `task_id`, `family`, `kind`, `prior`, `seed`, `arm`,
`features`, and `probe_val`. Development records also require `final_val`.
Allowed optional timings are `probe_seconds`, `remaining_seconds`,
`setup_seconds`, `train_seconds`, and common `diagnostic_seconds`.
All four arms must be present for every task, all configured families must be
present, and duplicate scientific tasks are rejected. If `cfg` also supplies
`kinds`, `priors`, and `seeds`, the full Cartesian task inventory is checked.

Optional `cfg['selector']` settings are `development_families`,
`confirmation_family`, `ridge_alphas`, `min_predicted_gain`, and `random_seed`.
The defaults implement the full protocol above. An `arms` setting, if supplied,
must equal the fixed four-arm list in its fixed order. Changing these settings
creates a different protocol; it must not be done after reviewing test results.

The output's main keys are `decisions`, `models`, `fit_metadata`, and
`selector_fit_seconds`. Each decision has `task_id`, family/architecture metadata,
`choices` keyed by the seven rule names, predicted log relative losses,
`model_id`, and preliminary `costs`. `manifest_digest` creates a SHA-256 digest of
the JSON-safe frozen manifest for the caller's resume/phase barrier.

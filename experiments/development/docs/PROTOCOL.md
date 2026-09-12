# Controlled family transfer and augmentation selection

## Questions and unit of generalization

The mechanism question is whether the effect of physics augmentation depends on perturbation structure, solver response, and prior mismatch across distinct equations. The selection question is whether cheap, observable diagnostics can prospectively choose a useful augmentation for a new physical family.

The six families are wave propagation, viscous Burgers flow, Kuramoto–Sivashinsky dynamics, advection–diffusion, Allen–Cahn reaction–diffusion, and two-species Gray–Scott reaction–diffusion. These provide different linear/nonlinear, dissipative, reaction, and instability structures. They remain a small, related collection of one-dimensional periodic simulated systems. More trajectories within one family do not create more independent physical families.

## Factors and controls

The candidate bank crosses three architectures, two physics-prior coefficient conditions, four augmentation choices, and the configured training seeds. The ordinary transformer has independent blocks; the looped transformer reuses its block; the Fourier neural operator is a strong field-model baseline. These factors are not a comparison of four disjoint concepts called “transformer,” “loop,” “physics,” and “world model.”

The augmentation menu contains an unaugmented option, IID perturbations, smooth perturbations, and a response-matched control. The smooth condition changes the spatial structure of a perturbation. The response-matched condition controls the size of the prior's predicted response instead of assuming that equal input perturbation size causes equal physical response. Because several new equations are nonlinear, response matching must be checked numerically; the affine scaling identity from the earlier wave experiment is not a general law.

Correct-prior and coefficient-perturbed-prior conditions test whether the method can recognize situations where augmentation should be avoided. The none option is required for meaningful abstention. An augmentation beating a damaging IID condition is weaker evidence than beating ordinary supervised training. Mechanism reports therefore must retain both comparisons and disclose uncertainty and divergence.

Ground-truth trajectories come from the configured reference simulator. The available augmentation prior uses its own numerical approximation and coefficient condition. Solver convergence checks distinguish a reference adequate for this benchmark from a numerically unreliable target. Simulator agreement is numerical validation, not experimental validation against the real world.

## Prospective information boundary

1. Fix source, profile, random seeds, candidate menu, feature definitions, and scoring rules in the saved protocol.
2. Generate training and validation data without generating test trajectories.
3. Train the candidate probes and record training diagnostics, probe validation losses, and measured time.
4. Complete the development candidate bank. Use its final **validation** outcomes as selector-learning targets.
5. Evaluate family transfer across the five development families with leave-one-family-out splits. Keep all tasks and seeds from a held-out family out of that fold's training labels. Choose ridge regularization inside the training-family partition rather than using the outer family's outcomes.
6. Fit the final selector using the five development families. For the confirmation family, Gray–Scott, allow the same observable diagnostics and short validation probes available for deployment. Do not use its completed-training validation labels to fit or tune the selector. Commit the predictions and selector state before completing the confirmation bank.
7. Only after those commitments and required training are complete, generate the independent ID/OOD tests and score the frozen decisions. Test results may explain outcomes but may not revise predictions, feature scaling, hyperparameters, or gate thresholds.

The observed short-probe validation losses are legitimate selection inputs. A final or test loss accidentally substituted for a probe loss would violate this boundary. Family identity, true simulator parameters, the hidden prior-condition label, and any test performance are excluded from the feature vector. Observable state/response summaries can still reveal characteristics of a family; that is the intended task signal, not proof of universality.

## Selector and comparisons

Use a small regularized linear model before a larger selector. With only five development families, complexity is easy to overfit and nested family-held-out tuning is essential. Predict candidate utility from the permitted feature vector, then choose from the fixed four-option menu. The rule is evaluated within the available architecture/prior task; it is not a claim that architecture and augmentation selection have both been solved jointly.

Report the fixed no-augmentation choice, fixed physics choices, the best development-selected fixed choice, and direct short-probe validation selection alongside the learned method. Any random-selection reference is an expectation or a separately seeded rule, never a test-chosen favorable draw. Full-training validation selection is a stronger but more expensive reference. Test-oracle selection is explicitly a retrospective upper bound.

Paired comparisons should preserve the common training seed, physical task, and test trajectory where applicable. Report per-family results so that a pooled average does not conceal a failure on the held-out confirmation family. A confidence interval over many trajectories within a few families does not establish transfer to the population of all physical equations. The saved protocol/report gives the implemented endpoints, uncertainty calculation, practical margins, and pass conditions.

## Cost accounting

For a deployment task, count all candidate probes, feature/selection work, and continuation of the selected checkpoint. Do not charge the selected probe twice. The full-profile update-equivalent cost is `4 * 300 + (4000 - 300) = 4900` updates per selected route, before considering unequal per-update costs. A single fixed arm costs 4000 updates; an all-arm bank costs 16000. Report measured seconds, candidate fit time, and total actually executed bank time. The fixed-arm route is cheaper than the probing route unless performance justifies the overhead.

The package deliberately completes all candidates to make fair paired counterfactual comparisons. It therefore does not realize the predicted deployment saving during this run. Acquiring the development labels also requires the full development bank. Any deployment break-even estimate must include that development cost and state the number of future tasks over which it is amortized.

## Completion and interpretation

Full-profile execution completeness, numerical integrity, mechanism evidence, and selector evidence are separate checks. Missing fits, mismatched hashes, unavailable test records, or non-finite required statistics cannot be silently interpreted as success. Smoke and pilot profiles exercise the pipeline and provide exploratory measurements only.

Mechanism findings may survive a failed selector: for example, smooth perturbations could be consistently safer while the learned rule adds nothing over the best fixed choice or short-probe ranking. Conversely, a selector can improve a practical average without demonstrating a unique physical cause. Negative prior-mismatch or unstable-rollout outcomes should be retained, not discarded to improve summary scores.

The strongest positive conclusion this package can support is that a prespecified selection procedure improved measured accuracy/cost tradeoffs on these particular controlled families, including a held-out family, with the reported uncertainty. It cannot establish universal selection, equal-compute architectural superiority, real-world reliability, or publication acceptance.

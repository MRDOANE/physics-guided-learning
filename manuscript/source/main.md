# Choosing between physical and neural forecasters with imperfect equations

Michael Doane

Independent Researcher, Cary, NC, United States

Correspondence email to be supplied by the author

## Abstract

Choosing a forecaster for a physical system requires knowing whether available equations describe its dynamics well enough and whether their coefficients can be estimated from observations. This study compares physical solvers and neural forecasters across eight simulated, fully observed systems described by partial differential equations. It combines 1,584 archived neural fits with a new calibration of 30 equation, grid, and physical-information settings. Neural candidates include a standard transformer, a transformer that applies a shared block three times, and a Fourier neural operator, each trained with several physical-supervision choices. Three shared coefficient corrections are fitted to training trajectories and selected using validation forecasts. Calibrated physics has lower forecast error than the validation-selected neural candidate in all 14 correct-coefficient and all 14 biased-coefficient settings. Neural forecasts have lower error in both settings with an omitted equation term. These directions persist at 64 and 96 forecast steps and under a shift beyond the training coefficient ranges. Validation selects a physical solver in every complete-equation setting and a Fourier neural operator in both missing-term settings. Calibration is inexpensive, and physical forward execution is faster in all matched CPU comparisons; neural timing uses initialized copies of the archived architectures. Physical supervision also produces useful gains and substantial harms within the neural candidates. The results support including coefficient calibration, direct validation forecasts, and measured execution cost in model choice. Their scope is the tested noiseless systems, whose imposed coefficient errors admit a simple global correction.

Keywords: scientific machine learning; physical model calibration; partial differential equations; neural forecasting; algorithm selection; imperfect physics

## 1 Introduction

A forecast of a physical system may be used to choose an operating condition, evaluate a control action, or compare possible future trajectories. The forecaster could integrate available equations, learn a transition from observed data, or combine physical calculations with a learned model. Each option needs some information about the system. A numerical solver needs equations and coefficients that describe the relevant dynamics. A neural forecaster needs training examples that cover the states and conditions it will encounter. Choosing between them requires a comparison of forecast error, the effort needed to prepare the model, and the cost of using it.

An uncertain physical coefficient creates a different problem from an absent physical mechanism. If the equation form is adequate, observed transitions may contain enough information to estimate its coefficients. If an important term is missing, coefficient fitting can improve the approximation and still leave a substantial forecasting error. System identification provides an established framework for learning dynamical models from data [@ljung2010identification]. Work on simulator calibration also explains why fitted parameters can absorb model discrepancy and lose their usual physical interpretation [@brynjarsdottir2014discrepancy]. These distinctions matter when a neural model is compared with an imperfect physical baseline.

The present study began with a related question about physical supervision. An available solver can label perturbed states and supply additional training examples to a neural forecaster. The usefulness of those examples depends on the equation, the solver's error, the perturbation, and the neural architecture. The resulting experiments retained direct physical forecasts alongside neural errors. A subsequent audit showed that those forecasts could support a broader comparison of the available prediction methods. It also identified a missing baseline: a physical solver whose coefficients had been fitted to the same training trajectories available to the neural models. Concerns about weak numerical baselines make this comparison important when interpreting a reported neural advantage [@mcgreivy2024weak].

The expanded experiment adds that baseline. It asks whether the apparent advantage of learning persists after simple coefficient calibration, whether coefficient fitting helps when an equation term is absent, and whether validation forecasts identify a suitable model before test evaluation. The original physical-supervision experiments remain part of this comparison. They establish which neural candidates are available and explain how physical information changes their performance. Absolute forecast errors then determine how those candidates compare with standalone solvers.

### 1.1 Physical and learned forecasting models

Neural operators learn mappings between functions and provide useful representations for predicting spatial fields. Fourier neural operators perform learned transformations in spectral space [@li2021fno], and DeepONet represents operators through separate input-function and evaluation-location networks [@lu2021deeponet]. Transformer-based simulators extend attention-based representations to physical prediction [@holzschuh2025pdetransformer; @herde2024poseidon]. Repeated application of shared attention blocks has precedents in Universal Transformers [@dehghani2019universal], and recent recurrent-depth simulators examine how additional inference computation changes accuracy [@majid2026recurrsim]. The looped transformer here uses a fixed repetition count of three.

Physical information can also enter neural training directly. Physics-informed neural networks use equation constraints [@raissi2019pinn], and physics-informed neural operators combine operator learning with physical losses [@li2024pino]. Reviews describe a wider range of methods for incorporating physical knowledge into learning [@karniadakis2021piml]. Surrogate-data-enriched training supplies additional examples from an available physical model [@Leiteritz2022SurrogateData]. Solver-in-the-Loop trains learned corrections through a differentiable solver [@um2020solverloop]. The present auxiliary targets are computed in advance by a fixed numerical operator. This gives a controlled way to study the influence of those targets on separately trained forecasters.

Physical-model error is central to these choices. Zou et al. examine correction of misspecified equations in physics-informed neural networks [@zou2024misspecification]. A recent preprint studies physical discrepancy in operator learning [@ma2026operatorcorrection]. PDEBench, APEBench, and The Well provide broader collections of physical prediction tasks [@takamoto2022pdebench; @koehler2024apebench; @ohana2024well]. APEBench includes defective numerical solvers and hybrid methods. These studies establish substantial prior work on both comparison and combination. The contribution examined here is a controlled comparison that connects the type of physical error, coefficient calibration, neural supervision, validation-based choice, and measured computation within one archived experiment sequence.

### 1.2 Perturbations and the choice of training procedure

Physical data augmentation can use equation symmetries, coordinate transformations, or numerical evolution constructions [@brandstetter2022lpsda; @fanaskov2023gcda; @liu2025inverseaugmentation]. Noise injection has a long history in learning [@bishop1995noise] and appears in learned physical simulators [@sanchezgonzalez2020gns]. PDE-Refiner studies poorly learned spectral components through iterative denoising [@lippe2023pderefiner]. StablePDENet, a recent preprint, examines structured perturbations and residual sensitivity [@huang2026stablepdenet]. Sobolev training supplies target derivatives as additional supervision [@czarnecki2017sobolev]. The perturbation control used here measures how much a fixed physical operator's prediction changes after a finite input change. Its interpretation concerns that measured response.

Two state perturbations can have equal root-mean-square amplitude and very different spatial wavelengths. Their physical next-state responses can therefore differ in magnitude. This study compares independent and smooth perturbations of equal input size and adds a control that matches the scalar physical response. Those comparisons help explain neural performance when a solver supplies auxiliary targets. A separate comparison with standalone calibrated physics is needed to decide which forecasting method to use.

Algorithm selection has a longstanding formulation [@Rice1976AlgorithmSelection]. Performance prediction and partial training curves can support selection from a candidate portfolio [@Xu2008SATzilla; @Ruhkopf2023MASIF; @Nguyen2024MetaLC]. Scientific-learning applications include selecting interface conditions and learning reusable loss functions [@Li2023METALIC; @Psaros2022MetaPINNLoss]. ForkMerge uses target validation to control auxiliary-task contributions [@Jiang2023ForkMerge], and recent algorithm-selection work studies changes in problem-feature distributions [@Wu2025RobustAlgorithmSelection]. The present study evaluates two decisions with different information and costs. One uses completed validation forecasts to choose among physical and neural predictors. The other transfers a learned rule for choosing neural augmentation from short training probes.

## 2 Methods

### 2.1 Systems data and the experiment sequence

The systems are one-dimensional periodic fields with fully observed states and known forcing actions. The initial experiment contains the wave, viscous Burgers, Kuramoto–Sivashinsky, advection–diffusion, Allen–Cahn, and Gray–Scott equations. The later selector experiment adds Cahn–Hilliard and FitzHugh–Nagumo. Supplementary Section S1 gives the implemented equations, numerical updates, parameter ranges, and initial conditions. All systems are dimensionless.

Each equation and experimental stage uses 128 training trajectories and 24 validation trajectories. Test evaluation uses 64 independent trajectories from the training coefficient ranges and another 64 from a specified shifted distribution. Each trajectory has 24 burn-in transitions followed by 160 recorded transitions. Coefficients are constant within a trajectory and vary across trajectories. Training coefficients are sampled independently from 0.85–1.15 times their equation-specific centers. Shifted evaluation draws every coordinate from 1.30–1.50 times its center. All candidates within a setting share the observed data and evaluation trajectories.

Table 1 describes the sequence. The initial study has 864 neural fits. A resolution follow-up adds 288 fits for waves, advection–diffusion, and Allen–Cahn at 32 and 64 cells. A selector follow-up adds 432 fits for Cahn–Hilliard and FitzHugh–Nagumo at 64 cells. The total is 1,584 archived neural fits. Different seeds vary neural initialization and training-window sampling. They share the generated training and validation trajectories within each setting.

[[TABLE design]]

The calibration extension uses 30 combinations of experimental stage, equation, grid, and supplied physical information. There are 14 correct-coefficient settings, 14 biased-coefficient settings, and two omitted-term settings. The 14-setting groups include repeated equations under different grids or data draws. They represent eight physical families. No neural weights are retrained in the extension.

The experiment sequence affects the strength of the statistical claims. The original follow-up froze its augmentation directions and selector configuration after inspecting development results and before its new test evaluations. The later calibration study was motivated by inspection of the archived results. It freezes its own fitting and validation choices before new test evaluation, and it remains an exploratory extension of an already observed study. There was no external preregistration.

### 2.2 Physical forecasts and coefficient calibration

A standalone physical forecast starts from the last of the four observed context states and repeatedly applies the available numerical transition under the supplied future actions. A physical prior here means the supplied equation and coefficient information. The correct prior uses the generating equation and coefficients with a coarser temporal update where the implementation has distinct fidelity levels. Advection–diffusion uses an exact affine Fourier-mode transition, so its fidelity levels coincide up to rounding. Persistence, which repeats the last observed state, is retained as an additional baseline.

The biased prior multiplies the three supplied coefficients by 0.70, 1.30, and 1.40. The structural prior omits a specified term. For Cahn–Hilliard it removes the cubic contribution to chemical-potential diffusion and retains the destabilizing linear contribution. For FitzHugh–Nagumo it removes the recovery variable's contribution to the activator equation; the recovery equation remains present. These changes are maintained during calibration.

Let q be the supplied coefficient vector for a trajectory and z a vector of three log corrections shared across all trajectories in one setting. The calibrated coefficients are

[[EQUATION calibration]]

The symbol ⊙ denotes coordinatewise multiplication. The fitting function receives observed training states, actions, and q. No separate ground-truth coefficient field is passed to the fitting function. In biased-coefficient cases, its only coefficient input is the distorted vector q. The correction is global within a setting; test trajectories receive the same fitted correction without any further parameter estimation. The imposed constant coefficient bias can be represented exactly by this calibration family. The structural omissions cannot be added back by multiplying coefficients.

Calibration uses SciPy nonlinear least squares [@virtanen2020scipy] on normalized forecast residuals. Each training trajectory supplies two evenly spaced windows. Six fitted candidates combine rollout lengths of one and four steps with multiplier starts (1, 1, 1), (0.8, 1.2, 1), and (1.2, 0.8, 1). Each fit uses a three-point numerical Jacobian, a budget of 60 objective evaluations, and convergence tolerances of 10⁻⁸. Numerical Jacobian evaluations require additional residual calls, which are recorded. Fitting uses float64 arithmetic. Multipliers are constrained to 0.001–1000 to bound the numerical search and maintain positive coefficients.

An initial diagnostic run reached a narrower lower bound in a structural case. The bounds were expanded before the full calibration run. The complete candidate records preserve this exploratory design choice, the starting points, boundary diagnostics, optimizer termination, and Jacobian information. The unchanged physical solver is included as a seventh candidate. Validation selects the candidate with the smallest mean normalized error over 16 recursively forecast steps on all 24 validation trajectories. Exact ties retain the earlier candidate, beginning with unchanged physics. Validation and final physical forecasts use float32 states with the original solver's internal coefficient arithmetic.

### 2.3 Neural candidates and training

Each neural model predicts the next state from four observed or recursively predicted states, four associated actions, and supplied coefficient information. The standard transformer [@vaswani2017attention] has three separate encoder blocks. The looped transformer applies one shared block three times. Both have width 64 and four attention heads. The Fourier neural operator has width 64, three layers, and eight retained Fourier modes. Each model predicts a state increment added to the final context state. The experiment contains these three neural architectures; a separate latent world-model architecture was outside its design.

The parameter counts differ. One-channel systems use approximately 101,000 parameters for the standard transformer, 34,000 for the looped transformer, and 210,000 real scalar parameters for the Fourier model. Two-channel counts are similar; exact values appear in the release. Complex weights count as two real parameters. The study uses a shared update budget, so architectural comparisons describe the tested configurations and their measured costs.

Every neural candidate completes 4,000 AdamW updates with learning rate 0.0003, weight decay 0.0001, batch size 32, and gradient-norm limit 1 [@loshchilov2019adamw]. Validation occurs every 100 updates using a 16-step recursive forecast. The lowest validation error within the budget selects the checkpoint used for the primary evaluation. Supplementary Section S2 gives the layers, normalization, sampling, seeds, and complete settings.

State means and scales are fitted to training observations. Supplied coefficients are also standardized using their corresponding training values. A positive constant coefficient rescaling therefore cancels in the neural coefficient input when the scale floor is inactive, as it is here. The biased coefficients continue to affect the physical transition and its auxiliary targets. This construction preserves the neural model's coefficient information and creates a correctable physical-parameter error. Both the neural and calibrated physical options learn from the same underlying training trajectories.

### 2.4 Physical supervision and perturbation controls

The neural candidates include ordinary supervised training and several ways of adding physical supervision. For a state history H ending in u, a perturbation δ changes the history, and the physical operator labels the perturbed final state under the final action. Physical targets are precomputed and held fixed during optimization. The neural objective combines the observed-transition loss and the auxiliary physical-target loss:

[[EQUATION auxiliary_loss]]

Both losses average squared errors in training-normalized state coordinates. The physical-loss weight λ is 0.1 for augmented candidates and zero for ordinary supervised training. An augmentation bank contains 4,096 sampled training windows; each update draws 32 auxiliary windows. The comparison with ordinary training includes both the perturbation and the added supervision. An additional unperturbed physical-target arm would be needed to isolate the contribution of perturbation itself.

Independent Gaussian perturbations are scaled by 0.05 times the state scale. Smooth perturbations retain spatial Fourier modes 1–3 and are rescaled to match each paired independent perturbation's input root-mean-square magnitude within each history position and channel. The response-matched arm instead changes the independent perturbation's amplitude until the physical operator's normalized next-state change matches the scalar response to the smooth perturbation. The control uses the actual coarse operator, including its supplied physical error.

Response matching controls one scalar magnitude. It changes the independent perturbation's input amplitude and leaves the physical response direction, spatial covariance, and target error uncontrolled. Perturbations are independently sampled at different history positions, so the perturbed history can differ from a dynamically integrated trajectory. These properties define the interpretation of the augmentation comparisons. The newly calibrated physical solver is used for standalone forecasting; it does not relabel the archived augmentation banks.

### 2.5 Choosing a forecasting model with validation data

The direct model-choice rule compares completed candidates. Within each equation, prior, and archived training seed, it first identifies the neural architecture and augmentation arm with the smallest saved validation error. That checkpoint's validation horizon is the same 16 steps used to select physical calibration. The rule then compares this neural validation score with unchanged and calibrated physics. Ties favor unchanged physics, then calibrated physics, then the neural candidate. Neural ties use a deterministic architecture and arm order.

All calibration files and all model-choice decisions are written to a hashed manifest before test trajectories or archived neural test scores are opened by the extension. The rule makes 156 seed-specific decisions across the 30 settings. The physical fit and training data are reused across seeds; the 156 choices are therefore repeated candidate comparisons within the stated settings. This rule uses no test error to choose a model.

A fresh application of this procedure would pay for fitting and validating its candidate neural models. Selection uses validation error alone. Execution cost is reported separately. The present analysis reuses those completed fits. Its short new runtime measures the extra cost of calibration and analysis. It leaves the historical cost of acquiring the neural candidates in a separate account.

### 2.6 Transferring an augmentation rule from short probes

The earlier guarded rule makes a different decision: it chooses an augmentation procedure for a fixed neural architecture after a short target training run. Ridge regression predicts the final validation log-loss ratio of each augmented arm relative to ordinary supervised training. Its features describe training transitions, the physical operator's discrepancy and response, state spectra, architecture, augmentation arm, and the observed validation ratio after 300 training updates. Equation identity, true coefficients, prior labels, and test errors are excluded from its predictor inputs.

The 864-fit development experiment supplies training labels. The rule is fitted to all six development equations and to six additional subsets that each omit one equation. For an augmented candidate, the guard takes the largest predicted log ratio across these fits. The candidate is eligible when that value is below zero and its target probe error is below the unaugmented probe error. The lowest eligible prediction determines the arm. If no augmented candidate qualifies, the rule selects ordinary supervised training. Development evaluation selects a ridge penalty of 1.

This rule is frozen before candidate continuation and test evaluation on Cahn–Hilliard and FitzHugh–Nagumo. Architecture, prior, and seed define 108 target tasks. Every candidate is eventually continued to 4,000 updates to measure the outcomes of the available choices. The primary baselines are ordinary supervised training and choosing the best short validation probe. Supplementary Section S3 gives the feature construction, development folds, and additional comparison rules.

### 2.7 Forecast errors and statistical comparisons

For trajectory j, forecast step h, channel c, and spatial cell i, the normalized step error is

[[EQUATION main_metric]]

The reference state is u, the forecast is û, the training standard deviation of channel c is sc, and the channel and grid counts are C and N. Step errors are averaged over the first 64 forecast steps for the primary outcome and over 96 steps for the secondary outcome. The two test distributions are reported separately. Errors remain grouped by equation, stage, and physical-information setting. The paper does not pool absolute errors across different equations into a single overall model score.

The calibration extension reports arithmetic mean errors and paired differences between calibrated physics, unchanged physics, the validation-selected neural candidate, and the validation-selected forecasting route. Its 5,000-replicate percentile bootstrap resamples complete shared test trajectories and, for neural comparisons, archived training seeds. The fitted physical correction and generated training data remain fixed. The intervals are exploratory, pointwise 95% intervals, with no multiplicity adjustment or overall hypothesis decision. They omit uncertainty from a fresh calibration training dataset.

The original augmentation analyses use their archived estimators. Candidate-to-baseline error ratios are formed within paired equation, architecture, prior, and seed configurations. Log ratios are averaged over the specified fixed strata and exponentiated. The pooled guarded-selector analysis gives equal weight to each equation. Percentage reduction is 100 times one minus this geometric mean ratio. This quantity differs from a ratio of errors pooled across all equations.

Development augmentation contrasts use 50,000 paired bootstrap replicates and Bonferroni-adjusted intervals across 24 comparisons. The follow-up uses 20,000 paired replicates, pointwise 95% intervals, and centered bootstrap p-values with Holm correction [@holm1979] across six primary 64-cell resolution contrasts and, separately, two guarded-selector contrasts. Secondary horizons and architecture breakdowns are descriptive. The manuscript interprets effect estimates and intervals without imposing a minimum percentage improvement or a required number of successful equations. Legacy development status labels remain in the archive and are excluded from these interpretations.

Protective scoring retains failed trajectories. Development caps each trajectory's mean error at 10⁶. Later stages cap per-step errors before averaging. Primary neural errors and all newly evaluated physical errors are finite and below the cap. Some unselected neural candidates exceed it after step 64; these cases and the stage-specific rules are documented in Supplementary Section S4. Numerical integrity checks and scientific effect estimates are reported separately.

### 2.8 Numerical identity, timing, and reproducibility

The nonlinear spectral systems use the implemented Cox–Matthews exponential time-differencing scheme [@cox2002etd]. Small-argument Taylor expansions address the coefficient-evaluation concern described by Kassam and Trefethen [@kassam2005etd]. Reference trajectories use float64 calculations and are stored in float32. Existing temporal and spatial checks describe the sampled finite-grid problem. Supplementary Section S1 gives their scope. The numerical solver family also generates the observations, which favors an adequate physical model in this controlled setting.

The original neural experiments ran on an NVIDIA RTX 3090 Ti with Python 3.12, NumPy 2.1, PyTorch 2.8, CUDA 12.8, and eight CPU threads. The calibration extension uses CPU PyTorch 2.8, NumPy 2.3, and four fitting processes with one thread each. Data regeneration temporarily restores eight Torch CPU threads because FFT rounding depends on that setting. All 28 regenerated development and test dataset files exactly match the archived hashes. Ordered trajectory identifiers, normalization, source files, frozen choices, and evaluation hashes are also verified.

A further check compares newly computed physical and persistence errors with archived reference errors. Three settings have CPU/GPU physical-reference differences beyond the preset numerical replay tolerance. Their exact input-data hashes establish common evaluation targets, and the discrepancies remain in the record. An independently written NumPy implementation agrees with the Torch float64 solver in 96 short-run checks across equations, grids, priors, and fidelity levels. These are implementation checks on the specified numerical models.

Fresh forward timing uses resident float32 inputs on the same CPU, five warmup calls per method, and five randomly interleaved blocks of 20 calls. The physical benchmark includes applying the fitted correction. Neural timing includes normalization and every model layer. Input transfer, history updates, fitting, and error calculation are excluded. The archive contains trained-model errors and metadata, with trained checkpoints excluded. Fresh neural timing therefore uses initialized copies of the archived architectures. It measures their execution graphs on the current machine; historical trained-model accuracy remains a separate measurement.

Current calibration and execution times are separated from historical neural training times. The older guarded-selector route costs reconstruct startup, probes, augmentation-bank work, and continuation from measured components. Their scope excludes common evaluation and acquisition of the earlier development bank. No matched-hardware claim is made between new CPU calibration cost and historical GPU training cost. The importance of explicit numerical baselines and computational accounting is established in prior analyses of learned simulation [@list2025unrolled; @mcgreivy2024weak].

OpenAI ChatGPT and Codex assisted with code preparation, analysis summaries, figure code, and manuscript preparation under the author's direction. Numerical solvers generated the physical observations, and Matplotlib generated data figures from retained arrays. Exact service model identifiers used during development were not recorded. The author is responsible for reviewing the code, calculations, interpretation, and final manuscript.

## 3 Results

### 3.1 Coefficient calibration changes the accuracy comparison

All 30 requested calibration settings completed, with no failed physical forecast trajectories. All 1,584 neural records and their common evaluation data passed the archive comparison checks. In the 14 biased-coefficient settings, the fitted multipliers closely approximate 1.43, 0.769, and 0.714, the values that undo the imposed scaling. These values were learned from training transitions. No selected calibration candidate reached the search bounds.

Table 2 reports the primary absolute errors for one representative stage and grid per physical family. In the wave case, calibration changes error from 0.414 to approximately 5 × 10⁻¹². The validation-selected neural error is 0.000373. Advection–diffusion changes from 0.684 to approximately 2 × 10⁻¹², compared with 0.000446 for the neural candidate. Allen–Cahn changes from 0.00589 to approximately 4 × 10⁻¹², compared with 0.000561 for the neural candidate. Figure 1 shows the corresponding errors across all eight families.

[[TABLE absolute]]

Kuramoto–Sivashinsky retains a larger residual forecast error than the other calibrated systems. Its primary error decreases from 0.586 to 1.81 × 10⁻⁷, compared with 0.0270 for the validation-selected neural candidate. This setting still favors calibrated physics by several orders of magnitude. The other biased-coefficient settings have calibrated errors close to the numerical differences between the reference and forecast implementations.

Calibrated physics has lower error than the validation-selected neural candidate in all 14 biased-coefficient settings. Every corresponding pointwise interval favors physics. The same directions hold at 96 steps and under the shifted coefficient distribution. The result identifies the fixed coefficient bias as a sufficient explanation for the earlier neural advantage over these uncalibrated physical solvers.

[[FIGURE model_choice]]

### 3.2 Correct coefficients and missing terms give different outcomes

When coefficients are already correct, the unchanged physical forecast is highly accurate. The calibrated procedure also has lower error than the validation-selected neural candidate in all 14 settings, at both horizons and in both distributions. Additional fitting has mixed effects relative to unchanged physics. At 64 steps in-distribution, two settings have lower errors with intervals excluding zero, five have intervals crossing zero, four have higher errors, and three retain identical forecasts. The four higher-error cases concern errors around 10⁻¹²–10⁻¹¹. These small changes provide little practical reason to modify an already adequate solver.

The two omitted-term systems produce a different ordering. In Cahn–Hilliard, validation retains the unchanged solver. Its primary test error is 0.267, compared with 0.0000470 for the selected neural candidate. The fitted candidates therefore supply no validation-supported improvement for this setting. In FitzHugh–Nagumo, calibration reduces primary error from 1.62 to 1.49, a reduction of 7.9% with a pointwise 95% interval of 6.8–9.2%. The selected neural error is 0.000504. Coefficient adjustment improves the approximation and leaves a large forecasting gap.

Neural forecasts remain more accurate in both omitted-term settings at 96 steps and under shifted coefficients. At 64 shifted-distribution steps, calibrated errors are 0.709 for Cahn–Hilliard and 2.41 for FitzHugh–Nagumo. The corresponding neural errors are 0.00194 and 0.00777. These comparisons show that the specified missing terms matter for predicting the observed trajectories even after the remaining coefficients can be adjusted.

### 3.3 Validation chooses the successful model class in these settings

The completed-validation rule chooses a physical solver in all 28 settings with complete equations and a neural model in both omitted-term settings. Across the repeated training seeds, it chooses calibrated physics in all 72 biased-coefficient comparisons. Correct-coefficient comparisons select calibrated physics 56 times and unchanged physics 16 times. The 12 omitted-term comparisons select a neural candidate. These counts describe 156 decisions within the 30 settings.

The distinction between the two physical choices is often numerically small in the correct-coefficient cases. The clear decision is whether to use a physical or neural forecast. That choice agrees with the observed test ordering in every setting and retains the same class-level ordering under the longer horizon and coefficient shift. Supplementary Table S8 records the exact choices and their validation losses.

The Fourier neural operator supplies the lowest validation error in every one of the 156 neural comparisons. In both omitted-term settings, the final choice is its unaugmented version. Smooth physical supervision is frequently selected when the equations and coefficients are correct. The observed preference for Fourier models is specific to these compact periodic fields and the tested model sizes. The experiments provide no corresponding advantage for the fixed-depth looped transformer.

### 3.4 Physical supervision still changes which neural candidate is useful

The augmentation results explain differences within the neural portfolio. At 64 cells, correct-physics smooth augmentation reduces pooled error by 35.3% for waves, 24.6% for advection–diffusion, and 83.2% for Allen–Cahn. Table 3 gives the intervals and Holm-adjusted tests. These are improvements relative to ordinary supervised neural training within the paired configurations.

[[TABLE resolution]]

Architecture changes the interpretation of the pooled transport result. Smooth augmentation increases error by 12.2% for the standard transformer and 17.8% for the looped transformer. The Fourier model improves by 67.6%. Figure 2 shows the architecture-specific effects. A positive average over architectures therefore provides incomplete guidance for an individual forecaster.

Biased physical supervision increases wave and advection–diffusion errors to 3.11 and 3.41 times their unaugmented values. Allen–Cahn still improves by 75.3%, with a 95% interval of 69.9–79.8%. These effects were specified from earlier observations and reproduced in the follow-up. The calibrated standalone solver outperforms the resulting neural candidates in all three systems. Helpful auxiliary supervision and best absolute forecast accuracy are separate measured outcomes.

[[FIGURE architecture]]

The original perturbation controls retain a narrower explanatory role. For correct-prior waves, smooth augmentation has 92.1% lower error than input-matched independent perturbations, with an adjusted interval of 88.6–95.1%. Matching the physical response magnitude reduces that difference to 56.9%, with an adjusted interval of 46.7–67.9%. Burgers retains a 29.4% advantage over the response-matched candidate, with an adjusted interval of 3.3–52.1%. The other four equations have response-matched intervals crossing zero. Supplementary Figure S1 and Table S3 give the full comparisons. Response magnitude alone leaves some of the wave and Burgers differences unexplained.

### 3.5 Short-probe augmentation selection has more limited gains

The earlier guarded augmentation rule reduces primary pooled error by 19.0% relative to choosing the best short validation probe, with a 95% interval of 13.7–23.7% and Holm-adjusted p below 0.001. Relative to ordinary supervised training, the estimate is a 2.3% reduction with an interval of −4.7% to 8.8% and adjusted p = 0.510. The comparison with the simplest neural training option remains uncertain.

The guard chooses no augmentation in 58 of 108 target tasks, smooth augmentation in 36, independent perturbations in 13, and response-matched perturbations in one. It chooses no augmentation in every omitted-term task. Under the 64-step coefficient shift, its pooled error is 18.1% higher than ordinary supervised training and 20.8% lower than the short-probe-selected route. The development information and short probes therefore support only limited claims about the advantage of transferring this augmentation rule.

The direct model-choice result uses more target information. It compares completed validation forecasts and includes standalone physical solvers. Its successful choices cannot be attributed to the learned guard or interpreted as a demonstration that a short probe would make the same decisions. The two analyses answer distinct computational questions about what is available at the time of selection.

### 3.6 Calibration and forecast execution costs

The complete new CPU experiment takes about 85 seconds after setup. Data regeneration takes about 49 seconds, the parallel calibration phase 9 seconds, physical evaluation 11 seconds, and matched forward timing 16 seconds. Individual setting calibration times have a median of 0.72 seconds and range from 0.35 to 2.0 seconds. These include candidate fitting, validation, and local setup. They describe a small three-parameter identification problem.

[[TABLE computation]]

Calibrated physics executes faster than all three neural architectures in each of the 30 matched CPU settings. Across setting-specific medians, its median step time is 0.45 ms, compared with 0.67 ms for the Fourier model, 1.2 ms for the standard transformer, and 1.2 ms for the looped transformer. Figure 3 shows the distributions across settings. These measurements use the existing implementations and initialized neural execution graphs. They establish neither the best attainable solver implementation nor exact trained-checkpoint latency on another device.

[[FIGURE timing]]

The original neural experiments incurred a separate historical cost: approximately 13.7 hours for development and 11.1 hours for the follow-up. The new 85-second experiment reuses those trained-model errors. For the earlier augmentation decision, reconstructed mean target-route times are 36.1 seconds for ordinary supervised training, 70.8 seconds for short-probe selection, and 64.7 seconds for the guarded rule. The guard's additional training cost accompanies an uncertain primary improvement over ordinary supervised training. Supplementary Table S6 provides the full accounting.

## 4 Discussion

The broader comparison changes how the earlier neural results should be read. A neural model can outperform a solver with biased coefficients even when the available equations describe the system adequately. In these experiments, a small calibration using the observed training trajectories removes that advantage. Including fitted physical coefficients is therefore necessary to assess the value of learning a replacement transition in this setting. The uncalibrated comparison describes the consequences of using the supplied coefficients without identification.

The omitted-term cases show why calibration and equation adequacy need separate treatment. Cahn–Hilliard retains its original coefficients because the fitted alternatives fail to improve validation forecasts. FitzHugh–Nagumo achieves a modest forecasting improvement through coefficient changes and still has much larger errors than the neural model. The learned correction factors in that case can compensate for some observed behavior without acquiring the missing mathematical term. They should be interpreted as predictive adjustments to the imperfect model. Simulator-calibration work has long described this difficulty in assigning physical meaning to fitted parameters under discrepancy [@brynjarsdottir2014discrepancy].

For a practitioner, the relevant sequence begins with the information already available. If equations and observed trajectories are available, fitting a small set of uncertain coefficients can be inexpensive enough to include routinely. Validation rollouts then compare the fitted solver with candidate learned forecasts over an operationally meaningful horizon. Execution cost determines whether an accurate candidate is usable at the required frequency. The present measurements illustrate this process for fully observed controlled systems. Applications with hidden states would also need a state-estimation procedure, and observational error would change the fitting objective and its uncertainty.

The validation rule succeeds here because the forecast differences between the physical and neural classes are large. The saved validation scores and choices show how each decision was made. A universal selection rule would require broader variation in equation error, observation quality, training coverage, and computational constraints. The current study tests one fixed joint coefficient shift and two particular omissions. The repeated grids and seeds improve understanding of those settings; they do not create additional independent examples of physical mechanisms.

The neural architecture result also has a clear scope. Fourier neural operators are well suited to representations of periodic spatial fields, and they lead the tested validation comparisons. The standard and looped transformers use their stated widths, depths, and training budgets, with smaller parameter counts than the Fourier models. The study compares these implemented candidates. It leaves open the effects of larger training corpora, longer contexts, alternative transformer tokenizations, adaptive recurrence, and matched-capacity tuning. A claim about world models in general would extend beyond the architectures and observations evaluated here.

Physical supervision remains useful within that restricted portfolio. The wave and transport harms under biased coefficients show how an inaccurate auxiliary target can damage neural forecasts. The Allen–Cahn benefit shows that the same coefficient distortion can still produce useful supervision in another system. Calibrated physics gives these results an absolute-accuracy context: an augmentation gain can coexist with a much more accurate standalone solver. Choosing an augmented neural candidate therefore requires comparison with both ordinary neural training and an appropriately fitted physical baseline.

The response-matching experiment addresses a specific ambiguity about those auxiliary examples. Equal input amplitudes can generate different physical response amplitudes, and matching the latter reduces the apparent smoothness advantage in waves. A remaining advantage persists in waves and Burgers. Response direction, perturbed-history consistency, and physical-target error remain possible contributors. The experiment measures their combined influence after one scalar control. Claims about a unique causal explanation would require controls that isolate those quantities.

Several favorable properties of the data explain the very small calibrated errors. Observations are noiseless, states and actions are fully available, the generating equation belongs to the solver family, and the coefficient distortion is constant across trajectories. Three shared multipliers can undo it. The near-zero errors in many cases are expected after successful identification and provide a useful baseline check. They give limited evidence about calibration under changing parameters, unknown forcing, stochastic dynamics, or incomplete measurements. The larger residual error in Kuramoto–Sivashinsky can include numerical approximation and imperfect coefficient fitting.

The uncertainty estimates have corresponding limits. Each setting has one generated training and validation dataset. The new bootstrap conditions on the fitted correction, and the older bootstrap conditions on its learned augmentation rule. The same test trajectories are reused for paired comparisons. The intervals measure variation across those trajectories and the archived neural training seeds. They leave training-data uncertainty and transfer to new equations unresolved. The calibration extension was designed after examining previous test results, so its findings should retain their exploratory status.

Computational claims require equal care. Current CPU stepping favors the existing physical implementations, and calibration requires little additional time. Historical neural training was performed on a different machine. Fresh neural timing uses initialized models because the original weight files were omitted from the archive. Timing exact trained checkpoints, optimizing the solver, and comparing full forecasting throughput on the intended deployment hardware would answer more specific engineering questions. The present separation of fitting, historical training, and forward execution costs makes those remaining questions visible.

Physical models can generate auxiliary examples, supply direct forecasts, or provide a fitted alternative that changes whether a neural model is needed. The appropriate use depends on which parts of the dynamics are represented, what can be learned from the available observations, and the errors measured on validation trajectories. The completed comparisons provide evidence for making that decision in the tested systems without another neural training campaign.

## 5 Conclusions

At the primary 64-step test horizon, coefficient calibration reverses the neural advantage over biased physical solvers across the eight tested physical families. Calibrated or unchanged physics is selected whenever the available equations are complete, and neural forecasting is selected in both omitted-term settings. Those class-level choices remain favorable under a longer horizon and the specified coefficient shift. Calibration is inexpensive, and physical stepping is faster in the matched CPU implementation benchmark.

The original augmentation experiments add detail about the neural alternatives. Physical supervision can improve their forecasts, cause substantial harm, and produce different effects across architectures. The transferred short-probe augmentation rule has limited gains compared with ordinary supervised training. The combined evidence supports comparing an identified physical model with validated neural candidates and reporting the cost of acquiring and executing each option. The conclusions concern fully observed, noiseless numerical systems and motivate further tests under less favorable observational and modeling conditions.

## Data and code availability

The repository at https://github.com/MRDOANE/physics-guided-learning contains the experiment source, configurations, numerical audits, candidate records, evaluation arrays, and reproduction instructions. Version 1.0.0 is archived at https://doi.org/10.5281/zenodo.22728288 [@doane2026physicsrelease]. The accompanying version 1.1.0 package adds the calibrated physical baselines, matched CPU timing, direct model-choice analysis, updated manuscript, and plotted values. A new version-specific Zenodo DOI will be added after that version is deposited; the existing DOI identifies the earlier release. Raw trajectory caches are regenerated by the included numerical solvers, and archived hashes allow their identity to be checked. Trained neural checkpoints are absent from the shared archives.

## Funding

[Author confirmation required for the funding statement.]

## Declaration of competing interests

[Author confirmation required for the competing-interest declaration.]

## CRediT authorship contribution statement

Michael Doane: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review and editing.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During preparation of this work, the author used OpenAI ChatGPT and Codex to assist with drafting, language editing, literature organization, and code preparation. The author directed the use of these tools and is responsible for reviewing the resulting text, verifying the reported analyses, and approving the final manuscript. Their use in the computational workflow is described in Section 2.8.

## References

[[REFERENCES]]

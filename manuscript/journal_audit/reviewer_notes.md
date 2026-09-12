# Read-only literature and voice review of the revised manuscript

Reviewed `manuscript_revision_work/main.md` on 12 September 2026. This review did not modify the manuscript.

## Overall assessment

The revised scientific story is clear. The abstract, introduction, first three results subsections, and discussion now address choosing a forecaster from physical and neural options. Calibration is the missing comparison that changes the interpretation of earlier results. The augmentation experiments explain the neural candidates and retain a useful subsidiary role. The distinction between the completed-validation forecaster choice and the learned short-probe augmentation rule is explained explicitly.

The conclusions are scoped to the tested fully observed, noiseless systems. The draft acknowledges that the imposed coefficient error admits an exact global correction, that structural evidence comes from two omissions, that checkpoints are unavailable for fresh timing, and that the calibration extension is exploratory. No unsupported claim about latent world models or a universal model-selection method was found.

## Checks passed

- The abstract contains 236 whitespace-separated words, below the 250-word ceiling.
- All 43 citation keys resolve to the retained bibliography and its three verified additions. Every reference is cited.
- No first-person-plural `we` or `our` occurs.
- No `while`, `rather than`, `not only`, `not X but Y`, or the user's named canned transitions occurs.
- The new calibration citations serve their verified purposes: Ljung introduces system identification; Brynjarsdóttir/O'Hagan supports the distinction between physical-parameter fitting and model discrepancy; Virtanen credits SciPy's optimizer.
- Both 2026 preprints remain described as preprints. Their stored titles and authors match the arXiv records checked during the audit.
- Architecture descriptions match their cited precedents at the level stated. The text explains that the tested looped model has a fixed recurrence count.

## Recommended local revisions

| Location | Issue | Suggested change |
| --- | --- | --- |
| Abstract, sentence 2 | Equations themselves are not the observed objects. | “This study compares physical solvers and neural forecasters across eight simulated, fully observed systems described by partial differential equations.” |
| Section 2.2, first paragraph | `Prior` appears without definition and can suggest a Bayesian prior distribution. | At first use, say “The correct physical-information setting uses …”; subsequently use “setting,” “supplied model,” or define the label explicitly. The three labels can still be retained for traceability. |
| Section 2.5, first paragraph | The candidate grouping omits stage and grid from its wording. | “Within each experimental stage, equation, grid, physical-information setting, and archived training seed …” or refer directly to a setting as already defined in 2.1. |
| Section 3.1 | Multiplier 1.429 carries four significant figures. | Use 1.43, 0.769, and 0.714. |
| Section 3.2 | Four-digit structural errors imply more reporting precision than needed. | 1.620 → 1.62; 1.492 → 1.49; 2.408 → 2.41. Preserve full precision in machine-readable tables. |
| Section 3.6 | Four-digit neural timing medians add little practical information. | 1.160 ms → 1.16 ms; 1.171 ms → 1.17 ms. The submillisecond values already have three significant figures. |
| Introduction | The clearest literature rationale for checking the physical baseline is currently deferred to Methods 2.8. | Add a short sentence after describing the missing calibrated baseline: “Published comparisons of learned PDE solvers show how weak numerical baselines can distort apparent advantages [@mcgreivy2024weak].” Avoid repeating an extended account in Methods. |
| Discussion, paragraph on class selection | “An inspectable model-choice result” is an abstract phrase. | “The saved validation scores and choices make each model-selection decision traceable.” |
| Discussion, final paragraph | “A focused extension of the paper's original purpose” describes paper construction rather than the scientific result. | Begin directly: “Physical models can generate auxiliary examples, supply direct forecasts, or provide a fitted alternative that changes whether a neural model is needed.” |

## Optional voice cleanup

The standalone negations in the manuscript generally communicate necessary methodological limits and are not the rhetorical contrast templates the user prohibited. Two sentences can nevertheless be made more concrete:

- Section 3.6: “They establish neither the best attainable solver implementation nor exact trained-checkpoint latency on another device.” A positive formulation is: “These are implementation-specific CPU step times; optimized solvers and trained checkpoints on other devices require separate measurements.”
- Discussion: “The repeated grids and seeds improve understanding of those settings; they do not create additional independent examples of physical mechanisms.” A simpler formulation is: “Repeated grids and seeds characterize these settings in more detail. The number of distinct omitted mechanisms remains two.”

The sentence about the response-matched arm using `instead` makes a literal procedural comparison, not an ornamental pivot. It can be shortened to “The response-matched arm changes …” without changing its meaning.

## Reference metadata improvements

The accessible publisher record for McGreivy and Hakim gives volume 6, pages 1256–1269 (2024). Those fields are absent from the old JSON entry and can be filled in during bibliography export. The publication date is 25 September 2024. DOI: [10.1038/s42256-024-00897-5](https://www.nature.com/articles/s42256-024-00897-5).

The List et al. reference legitimately includes computational performance and numerical/neural comparison. Its author-hosted arXiv version is [2402.12971](https://arxiv.org/abs/2402.12971); the existing 2025 journal DOI and title are correct. That URL can be added as a reading link without changing the publication citation. The current broad methods citation is defensible.

This is a targeted literature and voice review. Numerical estimates should continue to be checked against the retained result arrays by the numerical audit.

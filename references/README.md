# Literature map

`references.bib` preserves the 36 references identified in the September 2026 novelty audit. Inclusion means relevance to the research question; it does not imply that every method was implemented as a baseline. Some entries are preprints. Verify final publication metadata when preparing the manuscript.

The empirical contribution concerns controlled comparisons of perturbation structure, imperfect physical supervision, architecture, and selected confirmation experiments. Individual ingredients have substantial prior work. No first-ever claim is made.

| Area | Closely related work | Relationship |
|---|---|---|
| PDE data augmentation | [Lie Point Symmetry Data Augmentation](https://proceedings.mlr.press/v162/brandstetter22a.html); [General Covariance Data Augmentation](https://proceedings.mlr.press/v202/fanaskov23a.html); [Inverse evolution augmentation](https://arxiv.org/abs/2501.14604) | Existing physical and equation-aware augmentation methods |
| Perturbation and rollout behavior | [PDE-Refiner](https://arxiv.org/abs/2308.05732); [StablePDENet](https://arxiv.org/abs/2601.06472) | Prior work on refinement, stability, and response sensitivity |
| Imperfect physics | [Correcting model misspecification in PINNs](https://doi.org/10.1016/j.jcp.2024.112918); [Physics-guided correction for operator learning](https://arxiv.org/abs/2606.03469) | Prior work already studies misspecified physical guidance |
| Benchmarks | [APEBench](https://arxiv.org/abs/2411.00180) | Broad benchmarking already includes physical-solver defects and hybrid models |
| Choosing physical training components | [Meta Learning of Interface Conditions](https://proceedings.mlr.press/v202/li23w.html) | Existing learning-based choices for PDE training |

The guarded selector here is a development-trained ridge rule with short validation probes. Its results should be judged against the included baselines and data-access costs. The repository's strongest evidence is the structured empirical comparison; it does not introduce a new transformer architecture or a general guarantee for selector performance.

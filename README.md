# Perturbation structure and imperfect physics in neural PDE forecasting

**Michael Doane — Independent Researcher, Cary, NC**

DOI: 10.5281/zenodo.22728288

When does using physics to generate extra training examples help a neural network predict how a physical system changes? This repository studies that question using controlled simulations of waves, transport, diffusion, and pattern formation. It compares three neural network designs and tests both accurate and imperfect physical guidance.

The completed experiments show that the effect depends on the equation, the architecture, and the type of physical error. Smooth perturbations help in several settings and can cause large errors in others. Selected patterns persisted when the spatial grid was refined. A guarded rule for choosing augmentation improved on selection based on short training probes in the primary pooled comparison. Its advantage over using no augmentation remains inconclusive.

## Start here

- [Main findings and interpretation](docs/RESULTS.md)
- [Reproduce or inspect the experiment](docs/REPRODUCIBILITY.md)
- [Study design and limitations](docs/STUDY_DESIGN.md)
- [Data and artifact dictionary](docs/DATA_DICTIONARY.md)
- [GitHub and Zenodo publication guide](docs/PUBLISHING_UI.md)
- [Saved confirmation report](results/confirmation/reports/report.md)
- [Saved development report](results/development/reports/report.md)
- [Jupyter result viewer](inspect_results.ipynb)

**No training is needed to inspect the results.** Make a copy of the Jupyter notebook in the same folder before running it, so saved cell outputs preserve the frozen original. The viewer and the standard-library checker verify the packaged files and display the saved findings:

```text
python scripts/verify_release.py
```

The checker reports file and evidence integrity separately from scientific findings. It does not refit neural networks or recompute confidence intervals. The report replay script recomputes the original analyses from saved arrays; see the reproduction guide.

## What is included

| Stage | Systems and purpose | Candidate fits |
|---|---|---:|
| Development | Wave, Burgers, Kuramoto–Sivashinsky, advection–diffusion, Allen–Cahn, Gray–Scott; perturbation and prior comparisons | 864 |
| Resolution follow-up | Wave, advection–diffusion, Allen–Cahn; separate training on grids of 32 and 64 cells | 288 |
| Selector follow-up | Cahn–Hilliard and FitzHugh–Nagumo; frozen selection rule tested on two additional equations | 432 |
| Total | Eight distinct equations across the study | 1,584 |

The 1,584 fits are repeated experimental configurations, not 1,584 independent physical systems. The equations share some physical structure. All systems are synthetic, one-dimensional, and periodic.

The architectures are an ordinary transformer, a transformer that applies one shared block three times, and a compact Fourier neural operator. Each predicts observed physical states. The study does not compare four mutually exclusive transformer, looped-transformer, physics-model, and world-model categories.

## Repository contents

| Path | Contents |
|---|---|
| `experiments/development/` | Exact source snapshot for the original 864-fit experiment |
| `experiments/confirmation/` | Exact source snapshot for the 720-fit follow-up and its frozen development validation bank |
| `results/development/` | Original development records, evaluation arrays, protocols, and reports |
| `results/confirmation/` | Original follow-up records, evaluation arrays, selection manifests, and reports |
| `scripts/` | Release verification, report replay, and optional selector checks |
| `docs/` | Methods, interpretation, reproduction, and publication instructions |
| `references/` | BibTeX references and a short literature map |
| `provenance/` | Original archive hashes and release validation record |
| `manifest.json` | SHA-256 inventory for the packaged release |

The original source and evidence files are preserved byte for byte. Only their directory layout changes. Earlier source READMEs refer to distribution launchers that belonged to the original RunPod packages. Use this repository's [reproduction guide](docs/REPRODUCIBILITY.md) for the entry points included here.

## Reading the evidence

The follow-up uses confidence intervals and Holm-adjusted tests for stated directions. It has no minimum percentage improvement and no required number of successful systems. A supported prediction of harm is scientific evidence even though predictive performance worsened. An inconclusive comparison establishes neither equivalence nor noninferiority.

The development report retains the original experiment's stricter aggregate status rules for historical reproducibility. Those old traffic-light summaries are not the criteria for interpreting the follow-up or judging the paper. Reported effects, uncertainty, and scope are the primary evidence.

The learned selector used the earlier six-equation validation bank and target training/validation probes. It was frozen before the target test evaluations. This is a local prospective follow-up to observed results; it is not an externally registered preregistration. Complete trajectories and paired training seeds are retained in the statistical analysis. Generalization beyond the tested equations remains an open question.

## Compute and reproducibility

The archived runs used Python 3.12.3, NumPy 2.1.2, PyTorch 2.8.0+cu128, one RTX 3090 Ti, and eight CPU threads. The measured development wall time was about 13.70 hours; the follow-up took 11.067 hours. These are historical timings on that environment, not runtime guarantees. Full runs train candidates sequentially on one GPU. A 24 GB GPU, 8–16 vCPUs, 32–64 GB RAM, and at least 50 GB persistent disk are suitable planning resources. Model training dominated the measured follow-up time.

Saved evaluation arrays and source are included. Model weights and the generated training trajectory caches were excluded by the original archive exporters. The runners regenerate those arrays for fresh training. Use a new output directory; the archived results are not resumable training directories.

## Citation and license

This repository is released under the [MIT License](LICENSE), retaining the original project's license. Citation metadata is in [CITATION.cff](CITATION.cff). A Zenodo DOI is intentionally absent until the first release is published. Cite the exact version DOI for reproducibility once available. This repository contains research software and results; it does not claim a published or accepted journal article.

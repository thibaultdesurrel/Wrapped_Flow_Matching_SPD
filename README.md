# Wrapped Flow Matching for Symmetric Positive Definite Matrices

Code release for an anonymous TMLR submission. This repository is shared for
double-blind review purposes; author information has been withheld.

## Overview

Conditional Flow Matching (CFM) has become a state-of-the-art paradigm for
generative modeling, but extending it to non-Euclidean geometries remains
challenging: existing Riemannian CFM methods rely either on geodesic
interpolation or on a pullback to Euclidean space, which limits their
expressivity.

This work introduces a general framework for CFM on the manifold of
**Symmetric Positive Definite (SPD) matrices**, built around the **wrapped
Gaussian** distribution. The framework recovers existing Riemannian CFM
variants as special cases and yields a new method, **WrappedCFM**, that
constructs conditional probability paths natively on the manifold — without
relying on geodesic interpolation or a Euclidean tangent-space
approximation. SPD matrices arise throughout statistics and machine
learning (Brain-Computer Interfaces, Diffusion Tensor Imaging, radar and
hyperspectral signal processing, covariance-based representations, ...),
making this a broadly applicable setting.

## Methods

| Method | Conditional center $\bar X_t(X_1)$ | Noise injection | Vector field | Working space |
|---|---|---|---|---|
| **WrappedCFM** (proposed) | $X_1^{t}$ | wrapped, on the manifold | closed form | native SPD manifold |
| GeodesicCFM | AIRM geodesic $\gamma(X_0, X_1, t)$ | none (path is deterministic) | autodiff through the geodesic | native SPD manifold |
| DiffeoCFM | $I_d$ (fixed) | linear, in tangent space | closed form | tangent space at $I_d$ |
| EuclideanCFM (baseline) | $t \cdot X_1$ | linear, in flat space | closed form | flat Euclidean space |

All four share the same base training loop (`cfm_models/cfm_base.py`) and
only differ in how the conditional path $p_t(\cdot \mid X_1)$ and its vector
field are defined. `WrappedCFM_with_prior` is an extension of WrappedCFM
that additionally conditions the source distribution on a per-class
Riemannian prior.

## Repository structure

```
runner/
  src/
    train.py                  # training entry point (Hydra + Lightning)
    evaluate.py                # evaluation on real-world datasets (quality + classification metrics)
    evaluate_synthetic.py       # evaluation on synthetic benchmarks (SPD-rate, log-likelihood, W2)
    cfm_models/
      cfm_base.py               # shared CFM training loop / LR scheduling
      wrapped_cfm.py             # WrappedCFM (proposed)
      wrapped_cfm_with_prior.py  # WrappedCFM with a class-conditional prior
      geodesic_cfm.py            # GeodesicCFM baseline (AIRM geodesic paths)
      diffeo_cfm.py               # DiffeoCFM baseline (tangent space at I_d)
      euclidean_cfm.py            # EuclideanCFM baseline (flat Euclidean space)
    backbones/                 # neural network backbones (MLP, tangent-space projection)
    datamodules/                # Lightning data modules
      BNCI2014002.py / BNCI2015001.py   # motor-imagery EEG (via MOABB)
      breizhcrops.py                     # satellite time-series crop classification
      textile.py                         # textile defect detection (image covariances)
      synthetic.py                        # figure-8, swiss roll, condition-number mixture
    solver/                    # Euclidean / Riemannian ODE solvers used at sampling time
    callbacks/                 # EMA callback
    utils/                     # wrapped-Gaussian sampling, AIRM ops, PRDC metrics, logging
  configs/                     # Hydra configs (see below)
  scripts/                     # launch + aggregation scripts for the paper's experiments
  results/                     # aggregated result tables / plotting notebook
```

## Setup

```bash
uv sync
```

(A `requirements.txt` is also provided for a plain `pip install -r requirements.txt` workflow.)

## Training

All commands are run from the `runner/` directory:

```bash
cd runner
```

### Normal training

```bash
uv run src/train.py
```

### Choose a model / dataset

Models (`configs/model/`): `euclidean_cfm` (default), `diffeo_cfm`,
`geodesic_cfm`, `wrapped_cfm`, `wrapped_cfm_with_prior`.

```bash
uv run src/train.py model=wrapped_cfm datamodule=BNCI2015001
```

Pre-defined experiment configs (`configs/experiment/`) bundle a
model/dataset/architecture combination matching the paper's settings, e.g.:

```bash
uv run src/train.py experiment=figure8_wrappedcfm
uv run src/train.py experiment=wg_mixture_wrappedcfm datamodule.kappa=10000
```

### Override config from the command line

```bash
uv run src/train.py model.lr=5e-4 trainer.max_epochs=500 seed=123
```

### Debug modes

Quick sanity check (1 epoch, CPU, anomaly detection, no callbacks/loggers):

```bash
uv run src/train.py debug=default
```

Overfit 3 batches for 20 epochs (verify the model can memorize):

```bash
uv run src/train.py debug=overfit
```

## Datasets

**Synthetic** (no download required): `figure8`, `swiss_roll` /
`swiss_roll_dim_5` / `swiss_roll_dim_10`, and `wg_mixture` — a mixture of
wrapped Gaussians on $\mathcal P_d$ whose class-mean condition number $\kappa$
is a config knob (`datamodule.kappa`), used for the condition-number
robustness sweep in the paper.

**Real-world**:
- `BNCI2014002`, `BNCI2015001` — motor-imagery EEG, downloaded automatically
  through [MOABB](https://github.com/NeuroTechX/moabb) on first run (set
  `datamodule.mne_data_path` to reuse an existing local MNE cache).
- `breizhcrops` — satellite time-series crop classification, downloaded
  automatically through the `breizhcrops` package on first run.
- `textile` — textile defect detection; requires `train64.h5` /
  `train64.csv` to be placed manually under `datamodule.dataset_dir` (see
  `runner/src/datamodules/textile.py` for details, since it is not
  redistributed in this repository).

## Evaluation

```bash
cd runner
uv run src/evaluate.py logs/<experiment>/<timestamp>              # real-world datasets
uv run src/evaluate_synthetic.py logs/<experiment>/seed_<seed>/<timestamp>  # synthetic benchmarks
```

`evaluate.py` reports sample-quality metrics (PRDC), an SPD-validity rate,
and downstream classification metrics (TSTR/TRTS). `evaluate_synthetic.py`
reports the SPD-validity rate, test log-likelihood, and the AIRM
Wasserstein-2 distance between generated and held-out samples.

## Reproducing the paper's experiments

Launch scripts under `runner/scripts/` sweep seeds/GPUs and write into
`logs/`; aggregation scripts then collect everything into `results/`.

```bash
cd runner

# Synthetic benchmark (figure8, swiss_roll, swiss_roll_dim_5/10) x 4 methods x 5 seeds
./scripts/run_synthetic_benchmark.sh
uv run scripts/fetch_results.py            # -> results/all_results.csv
uv run scripts/make_synthetic_table.py     # -> results/synthetic_summary.csv

# Condition-number robustness sweep (kappa in {10, 100, 1000, 10000})
./scripts/run_condition_sweep.sh
uv run scripts/make_synthetic_table.py

# Real-world datasets
./scripts/train_all_groups_BNCI.sh BNCI2014002_wrappedcfm 5
./scripts/train_all_models_breizhcrops.sh
./scripts/train_all_models_textile.sh
```

`results/nice_plots.ipynb` reproduces the paper's figures from the
aggregated CSVs.

## Acknowledgements

- `runner/src/utils/prdc.py` is adapted from [Naeem et al., "Reliable
  Fidelity and Diversity Metrics for Generative Models", ICML
  2020](https://github.com/clovaai/generative-evaluation-prdc) (MIT
  license).
- Real-world data loading relies on [MOABB](https://github.com/NeuroTechX/moabb),
  [MNE](https://mne.tools/), [pyriemann](https://pyriemann.readthedocs.io/),
  and [breizhcrops](https://github.com/dl4sits/BreizhCrops).

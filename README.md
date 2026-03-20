# SPD Wrapped Flow Matching

A cleaner version of the Wrapped Flow Matching on SPD matrices.

## Setup

```bash
uv sync
```

## Training

All commands are run from the `runner/` directory:

```bash
cd runner
```

### Normal training

```bash
uv run src/train.py
```

### Override config from the command line

```bash
uv run src/train.py model.lr=5e-4 trainer.max_epochs=500 seed=123
```

### Switch dataset

```bash
uv run src/train.py datamodule=BNCI2015001
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

## Project structure

```
runner/
  src/
    train.py              # training entry point (Hydra + Lightning)
    cfm_models/           # flow matching models (CFMBase, EuclideanCFM)
    backbones/            # neural network backbones (MLP)
    datamodules/          # Lightning data modules (BNCI2014002, BNCI2015001)
    solver/               # ODE solvers
    utils/                # logging, sampling utilities
  configs/
    train.yaml            # main config
    callbacks/            # early stopping, checkpointing, progress bar
    trainer/              # Lightning Trainer settings
    datamodule/           # dataset configs
    logger/               # logging backends (CSV, etc.)
    debug/                # debug overrides
    hydra/                # Hydra output and logging settings
    paths/                # directory paths
```
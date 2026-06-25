"""Evaluate a trained CFM model on a synthetic dataset (figure8 / swiss_roll).

Focused counterpart to evaluate.py for the single-class synthetic benchmarks.
Computes only the three metrics of interest on the test set and writes them to
``<run_dir>/eval/synthetic_metrics.csv``:

    pct_spd  : % of generated test samples that are symmetric AND positive
               definite (fraction_spd "in M" * 100)
    ll       : mean (+ std) log-likelihood of the true test x1 under the model
    w2       : Wasserstein-2 distance (AIRM ground metric) between the true test
               x1 and the generated test samples

EuclideanCFM returns a density w.r.t. the flat Lebesgue measure on symmetric
matrices; the manifold models (Wrapped/Geodesic/Diffeo) return one w.r.t. the
AIRM Riemannian volume. To make the log-likelihoods comparable we add the log
AIRM volume factor (dim + 1) / 2 * logdet(x1) to the EuclideanCFM result.

Usage:
    cd runner
    uv run src/evaluate_synthetic.py logs/figure8_wrappedcfm/seed_0/2026-06-25_14-33-48
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from ot import emd2

from utils.loading import load_model
from utils.spd_ops import pairwise_distance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

LL_NSTEPS = 1000  # backward-ODE steps for compute_loglikelihood
LL_BATCH = 512


def fraction_in_M(matrices: np.ndarray, tol: float = 1e-12) -> float:
    """Fraction of matrices that are symmetric AND positive definite ("in M")."""
    is_sym = np.array([np.allclose(m, m.T, atol=tol) for m in matrices])
    is_pd = np.all(np.linalg.eigvalsh(matrices) > tol, axis=1)
    return float(np.mean(is_sym & is_pd))


def project_on_spd(matrices: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project matrices onto the SPD cone: A_proj = (1-alpha)A + alpha*I.

    Same convention as evaluate.py. Used only before the AIRM W2 (whose ground
    metric is undefined on non-PD matrices); the %SPD metric is computed on the
    raw, unprojected samples.
    """
    n = matrices.shape[-1]
    flat = matrices.reshape(-1, n, n)
    min_eig = np.linalg.eigvalsh(flat).min(axis=1)
    mask = min_eig < eps
    alpha = np.zeros_like(min_eig)
    alpha[mask] = (eps - min_eig[mask]) / (1 - min_eig[mask])
    eye = np.eye(n)[None]
    projected = (1 - alpha)[:, None, None] * flat + alpha[:, None, None] * eye
    return projected.reshape(matrices.shape)


def wasserstein2(x_real: np.ndarray, x_fake: np.ndarray) -> float:
    """Wasserstein-2 distance with the AIRM ground metric (matches evaluate.py)."""
    d_cross = (
        pairwise_distance(torch.from_numpy(x_real), torch.from_numpy(x_fake))
        .cpu()
        .numpy()
    )
    a = np.ones(len(x_real)) / len(x_real)
    b = np.ones(len(x_fake)) / len(x_fake)
    return float(np.sqrt(emd2(a, b, d_cross**2)))


def generate_samples(model, labels: np.ndarray, cfg) -> np.ndarray:
    """Generate SPD matrices for each label using the model's sample_ode."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    labels_t = torch.from_numpy(labels).long().to(device)
    y = F.one_hot(labels_t, num_classes=cfg.model.num_classes).to(dtype)
    x0 = model.sample_x0(batch_size=len(labels), int_labels=labels_t)
    return model.sample_ode(x0, y).cpu().numpy()


def loglikelihood(model, x_test: np.ndarray, y_test: np.ndarray, cfg) -> tuple:
    """Mean and std test log-likelihood (AIRM-comparable across all models).

    Runs with grad enabled (compute_loglikelihood differentiates through the
    net), batched to bound memory.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    is_euclidean = cfg.model._target_.endswith("EuclideanCFM")

    lls = []
    for i in range(0, len(x_test), LL_BATCH):
        x1 = torch.as_tensor(x_test[i : i + LL_BATCH], dtype=dtype, device=device)
        lbl = torch.from_numpy(y_test[i : i + LL_BATCH]).long().to(device)
        y = F.one_hot(lbl, num_classes=cfg.model.num_classes).to(dtype)

        _, logp = model.compute_loglikelihood(x1, y, nsteps=LL_NSTEPS)
        if is_euclidean:
            # flat Lebesgue -> AIRM volume, so LL is comparable across models
            logp = logp + 0.5 * (x1.shape[-1] + 1) * torch.logdet(x1)
        lls.append(logp.detach().cpu())

    ll = torch.cat(lls).numpy()
    return float(ll.mean()), float(ll.std())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CFM model on synthetic data.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a training run directory (contains .hydra/ and checkpoints/)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename inside checkpoints/ (default: best, fallback last.ckpt)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    ckpt = run_dir / "checkpoints" / args.ckpt if args.ckpt else run_dir
    model, cfg = load_model(ckpt)
    log.info(f"Loaded model from {run_dir}")

    seed_everything(cfg.seed)

    output_dir = run_dir / "eval"
    output_dir.mkdir(exist_ok=True)

    # --- Data ---
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    x_test = datamodule.test_ds.tensors[0].numpy()
    y_test = datamodule.test_ds.tensors[1].numpy()
    log.info(f"Test: {x_test.shape}")

    if hasattr(model, "set_prior_stats"):
        model.set_prior_stats(datamodule.class_means, datamodule.class_vars)

    # --- Generate test samples ---
    log.info("Generating test samples...")
    gen_test = generate_samples(model, y_test, cfg)

    # --- Metrics ---
    log.info("Computing %SPD, log-likelihood and W2...")
    # %SPD on the raw samples; W2 on samples projected onto the SPD cone (the
    # AIRM ground metric is undefined for non-PD matrices), matching evaluate.py.
    pct_spd = 100.0 * fraction_in_M(gen_test)
    ll_mean, ll_std = loglikelihood(model, x_test, y_test, cfg)
    w2 = wasserstein2(x_test, project_on_spd(gen_test))

    row = {
        "experiment": cfg.experiment.name,
        "seed": int(cfg.seed),
        "n_test": int(len(x_test)),
        "pct_spd": pct_spd,
        "ll_mean": ll_mean,
        "ll_std": ll_std,
        "w2": w2,
    }
    df = pd.DataFrame([row])
    df.to_csv(output_dir / "synthetic_metrics.csv", index=False, float_format="%.6f")
    log.info(f"Synthetic metrics:\n{df.to_string(index=False)}")
    log.info(f"Saved to {output_dir / 'synthetic_metrics.csv'}")


if __name__ == "__main__":
    main()

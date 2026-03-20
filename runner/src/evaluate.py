"""Evaluate a trained CFM model.

Loads the config from a training run directory, loads the best checkpoint,
generates SPD matrices, and computes quality + classification metrics.

Usage:
    cd runner
    uv run src/evaluate.py logs/BNCI2014002_diffeocfm/2026-03-20_17-23-08
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from prdc import compute_prdc

from utils import sample_wrapped_gaussian

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPD constraint checks
# ---------------------------------------------------------------------------


def fraction_spd(matrices: np.ndarray, tol: float = 1e-12) -> dict:
    """Check what fraction of matrices are symmetric and positive definite."""
    is_sym = np.array([np.allclose(m, m.T, atol=tol) for m in matrices])
    eigvals = np.linalg.eigvalsh(matrices)
    is_pd = np.all(eigvals > tol, axis=1)
    return {
        "Sym.": float(np.mean(is_sym)),
        "Pos. def.": float(np.mean(is_pd)),
        "in M": float(np.mean(is_sym & is_pd)),
    }


def project_on_spd(matrices: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project matrices onto the SPD cone: A_proj = (1-α)A + αI."""
    n = matrices.shape[-1]
    flat = matrices.reshape(-1, n, n)
    min_eig = np.linalg.eigvalsh(flat).min(axis=1)
    mask = min_eig < eps
    alpha = np.zeros_like(min_eig)
    alpha[mask] = (eps - min_eig[mask]) / (1 - min_eig[mask])
    eye = np.eye(n)[None]
    projected = (1 - alpha)[:, None, None] * flat + alpha[:, None, None] * eye
    return projected.reshape(matrices.shape)


# ---------------------------------------------------------------------------
# Quality metrics (distribution similarity)
# ---------------------------------------------------------------------------


def compute_quality_metrics(X_real: np.ndarray, X_fake: np.ndarray) -> dict:
    """PRDC metrics between real and generated distributions."""
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_fake_flat = X_fake.reshape(len(X_fake), -1)

    min_len = min(len(X_real_flat), len(X_fake_flat))
    X_real_flat = X_real_flat[:min_len]
    X_fake_flat = X_fake_flat[:min_len]

    prdc = compute_prdc(
        real_features=X_real_flat,
        fake_features=X_fake_flat,
        nearest_k=10,
    )
    return {
        "Precision": prdc["precision"],
        "Recall": prdc["recall"],
        "Density": prdc["density"],
        "Coverage": prdc["coverage"],
    }


# ---------------------------------------------------------------------------
# Classification metrics (train-on-synthetic evaluation)
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train LR in tangent space on X_train, evaluate on X_test."""
    clf = make_pipeline(
        TangentSpace(metric="riemann"),
        LogisticRegressionCV(
            cv=5,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        ),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    return {
        "ROC-AUC": roc_auc_score(y_test, y_proba[:, 1]),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def generate_samples(model, labels: np.ndarray, cfg) -> np.ndarray:
    """Generate SPD matrices for each label using the model's sample_ode."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dim = cfg.model.dim
    num_classes = cfg.model.num_classes

    labels_t = torch.from_numpy(labels).long().to(device)
    y_onehot = F.one_hot(labels_t, num_classes=num_classes).to(dtype)

    n = len(labels)
    dim_ts = dim * (dim + 1) // 2
    p0 = torch.eye(dim, dtype=dtype, device=device)
    mu0 = torch.zeros(dim_ts, dtype=dtype, device=device)
    sigma_init = 1.0 / math.sqrt(dim_ts)
    sigma0 = (sigma_init**2) * torch.eye(dim_ts, dtype=dtype, device=device)
    x0 = sample_wrapped_gaussian(n, p0, mu0, sigma0)

    generated = model.sample_ode(x0, y_onehot)
    return generated.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CFM model.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a training run directory (contains .hydra/ and checkpoints/)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename inside checkpoints/ (default: best checkpoint, fallback to last.ckpt)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # --- Load config from the training run ---
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    log.info(f"Loaded config from {config_path}")

    seed_everything(cfg.seed)

    # --- Find checkpoint ---
    ckpt_dir = run_dir / "checkpoints"
    if args.ckpt:
        ckpt_path = ckpt_dir / args.ckpt
    else:
        # Find best checkpoint (not last.ckpt)
        ckpts = sorted([p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"])
        if ckpts:
            ckpt_path = ckpts[-1]
        elif (ckpt_dir / "last.ckpt").exists():
            ckpt_path = ckpt_dir / "last.ckpt"
        else:
            log.error(f"No checkpoints found in {ckpt_dir}")
            sys.exit(1)

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    log.info(f"Using checkpoint: {ckpt_path}")

    # --- Output directory ---
    output_dir = run_dir / "eval"
    output_dir.mkdir(exist_ok=True)

    # --- Load datamodule and set up splits ---
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    x_train = datamodule.train_ds.tensors[0].numpy()
    y_train = datamodule.train_ds.tensors[1].numpy()
    x_val = datamodule.val_ds.tensors[0].numpy()
    y_val = datamodule.val_ds.tensors[1].numpy()

    log.info(f"Train: {x_train.shape}, Val: {x_val.shape}")

    # --- Load model from checkpoint ---
    log.info(f"Instantiating net <{cfg.model.net._target_}>")
    net = hydra.utils.instantiate(cfg.model.net)
    optimizer_partial = hydra.utils.instantiate(cfg.model.optimizer)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.model,
        net=net,
        optimizer=optimizer_partial,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Generate samples ---
    log.info("Generating samples...")
    t0 = time.time()
    gen_train = generate_samples(model, y_train, cfg)
    gen_val = generate_samples(model, y_val, cfg)
    sampling_time = time.time() - t0
    log.info(f"Sampling finished in {sampling_time:.1f}s")

    # --- Save generated samples ---
    np.save(output_dir / "generated_train.npy", gen_train)
    np.save(output_dir / "labels_train.npy", y_train)
    np.save(output_dir / "generated_val.npy", gen_val)
    np.save(output_dir / "labels_val.npy", y_val)
    np.save(output_dir / "real_train.npy", x_train)
    np.save(output_dir / "real_val.npy", x_val)

    # --- SPD constraint check ---
    log.info("Checking SPD constraints...")
    frac_real = fraction_spd(x_train)
    frac_gen = fraction_spd(gen_train)
    log.info(f"  Real train: {frac_real}")
    log.info(f"  Generated:  {frac_gen}")

    all_spd = frac_gen["in M"] == 1.0
    if not all_spd:
        log.warning(
            f"Only {frac_gen['in M']*100:.1f}% of generated samples are SPD. "
            "Projecting onto SPD cone for classification metrics."
        )
        gen_train_clf = project_on_spd(gen_train)
        gen_val_clf = project_on_spd(gen_val)
    else:
        gen_train_clf = gen_train
        gen_val_clf = gen_val

    # --- Quality metrics ---
    log.info("Computing quality metrics (PRDC)...")
    quality_train_gen = compute_quality_metrics(x_train, gen_train_clf)
    quality_val_gen = compute_quality_metrics(x_val, gen_val_clf)
    quality_train_val = compute_quality_metrics(x_train, x_val)

    quality_rows = [
        {"Comparison": "Train vs Val (baseline)", **quality_train_val},
        {"Comparison": "Train vs Gen", **quality_train_gen},
        {"Comparison": "Val vs Gen", **quality_val_gen},
    ]
    df_quality = pd.DataFrame(quality_rows)
    df_quality.to_csv(
        output_dir / "quality_metrics.csv", index=False, float_format="%.4f"
    )
    log.info(f"Quality metrics:\n{df_quality.to_string(index=False)}")

    # --- Classification metrics ---
    log.info("Computing classification metrics...")

    clf_baseline = compute_classification_metrics(x_train, y_train, x_val, y_val)
    clf_tstr = compute_classification_metrics(gen_train_clf, y_train, x_val, y_val)
    clf_trts = compute_classification_metrics(x_train, y_train, gen_val_clf, y_val)

    clf_rows = [
        {"Comparison": "Real vs Real (baseline)", **clf_baseline},
        {"Comparison": "Gen vs Real (TSTR)", **clf_tstr},
        {"Comparison": "Real vs Gen (TRTS)", **clf_trts},
    ]
    df_clf = pd.DataFrame(clf_rows)
    df_clf.to_csv(
        output_dir / "classification_metrics.csv", index=False, float_format="%.4f"
    )
    log.info(f"Classification metrics:\n{df_clf.to_string(index=False)}")

    # --- Constraint summary ---
    constraint_rows = [
        {"Subset": "Real train", **frac_real},
        {"Subset": "Generated train", **frac_gen},
        {"Subset": "Generated val", **fraction_spd(gen_val)},
    ]
    df_constraints = pd.DataFrame(constraint_rows)
    df_constraints.to_csv(
        output_dir / "spd_constraints.csv", index=False, float_format="%.4f"
    )

    # --- Timing ---
    with open(output_dir / "sampling_time.log", "w") as f:
        f.write(f"{sampling_time:.1f}s\n")

    log.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

"""Evaluate a trained CFM model.

Loads the config from a training run directory, loads the best checkpoint,
generates SPD matrices, and computes quality + classification metrics.

Usage:
    cd runner
    uv run src/evaluate.py logs/BNCI2014002_diffeocfm/2026-03-20_17-23-08
"""

import argparse
import logging
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
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from utils.prdc import compute_prdc
from ot import emd2
from utils.spd_ops import pairwise_distance

from utils.loading import load_model

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
    # X_real_flat = X_real.reshape(len(X_real), -1)
    # X_fake_flat = X_fake.reshape(len(X_fake), -1)

    min_len = min(len(X_real), len(X_fake))
    X_real_flat = X_real[:min_len]
    X_fake_flat = X_fake[:min_len]

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


def compute_wasserstein_distance(X_real: np.ndarray, X_fake: np.ndarray) -> dict:
    """Wasserstein-2 distance between real and generated SPD distributions."""
    D_cross = (
        pairwise_distance(
            torch.from_numpy(X_real),
            torch.from_numpy(X_fake),
        )
        .cpu()
        .numpy()
    )

    a = np.ones(len(X_real)) / len(X_real)
    b = np.ones(len(X_fake)) / len(X_fake)
    w2 = np.sqrt(emd2(a, b, D_cross**2))
    return {"Wasserstein-2 distance": w2}


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
    num_classes = cfg.model.num_classes

    labels_t = torch.from_numpy(labels).long().to(device)
    y_onehot = F.one_hot(labels_t, num_classes=num_classes).to(dtype)

    x0 = model.sample_x0(batch_size=len(labels), int_labels=labels_t)

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

    # --- Load model and config ---
    ckpt = run_dir / "checkpoints" / args.ckpt if args.ckpt else run_dir
    model, cfg = load_model(ckpt)
    log.info(f"Loaded model from {run_dir}")

    seed_everything(cfg.seed)

    # --- Output directory ---
    output_dir = run_dir / "eval"
    output_dir.mkdir(exist_ok=True)

    # --- Load datamodule and set up splits ---
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    x_train = datamodule.train_ds.tensors[0].numpy()
    y_train = datamodule.train_ds.tensors[1].numpy()
    x_test = datamodule.test_ds.tensors[0].numpy()
    y_test = datamodule.test_ds.tensors[1].numpy()

    log.info(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # --- Set prior stats for models that need per-class priors ---
    if hasattr(model, "set_prior_stats"):
        model.set_prior_stats(datamodule.class_means, datamodule.class_vars)

    # --- Generate samples ---
    log.info("Generating samples...")
    t0 = time.time()
    gen_train = generate_samples(model, y_train, cfg)
    gen_test = generate_samples(model, y_test, cfg)
    sampling_time = time.time() - t0
    log.info(f"Sampling finished in {sampling_time:.1f}s")

    # --- Save generated samples ---
    np.save(output_dir / "generated_train.npy", gen_train)
    np.save(output_dir / "labels_train.npy", y_train)
    np.save(output_dir / "generated_test.npy", gen_test)
    np.save(output_dir / "labels_test.npy", y_test)
    np.save(output_dir / "real_train.npy", x_train)
    np.save(output_dir / "real_test.npy", x_test)

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
        gen_test_clf = project_on_spd(gen_test)
    else:
        gen_train_clf = gen_train
        gen_test_clf = gen_test

    # --- Quality metrics ---
    log.info("Computing quality metrics (PRDC)...")
    quality_train_gen = compute_quality_metrics(x_train, gen_train_clf)
    quality_test_gen = compute_quality_metrics(x_test, gen_test_clf)
    quality_train_test = compute_quality_metrics(x_train, x_test)

    log.info("Computing Wasserstein-2 distances...")
    w2_train_gen = compute_wasserstein_distance(x_train, gen_train_clf)
    w2_test_gen = compute_wasserstein_distance(x_test, gen_test_clf)
    w2_train_test = compute_wasserstein_distance(x_train, x_test)

    quality_rows = [
        {"Comparison": "Train vs Test (baseline)", **quality_train_test, **w2_train_test},
        {"Comparison": "Train vs Gen", **quality_train_gen, **w2_train_gen},
        {"Comparison": "Test vs Gen", **quality_test_gen, **w2_test_gen},
    ]
    df_quality = pd.DataFrame(quality_rows)
    df_quality.to_csv(
        output_dir / "quality_metrics.csv", index=False, float_format="%.4f"
    )
    log.info(f"Quality metrics:\n{df_quality.to_string(index=False)}")

    # --- Classification metrics ---
    log.info("Computing classification metrics...")

    clf_baseline_test = compute_classification_metrics(x_train, y_train, x_test, y_test)
    clf_tstr_test = compute_classification_metrics(gen_train_clf, y_train, x_test, y_test)
    clf_trts_test = compute_classification_metrics(x_train, y_train, gen_test_clf, y_test)

    clf_rows = [
        {"Comparison": "Real→Test (baseline)", **clf_baseline_test},
        {"Comparison": "Gen→Test (TSTR)", **clf_tstr_test},
        {"Comparison": "Real→Gen-Test (TRTS)", **clf_trts_test},
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
        {"Subset": "Generated test", **fraction_spd(gen_test)},
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

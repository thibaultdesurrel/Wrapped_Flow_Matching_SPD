"""DataModule for the Textile defect dataset.

Loads images from train64.h5, computes 10-channel covariance matrices from
image features (intensity + Gaussian/Laplace/Farid filters). Binary labels:
"cut" → 0, "good" → 1.

Random 80/10/10 train/val/test split (reproducible via seed).
Covariance matrices are cached to disk on first run.
"""

import warnings

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset

from datamodules.base_datamodule import BaseDataModule

warnings.filterwarnings("ignore")


class TextileDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int = 0,
        seed: int = 42,
        dtype: str = "float64",
        recenter: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            dtype=dtype,
            recenter=recenter,
        )
        self.dataset_dir = Path(dataset_dir)

    def _compute_covariances(self):
        """Load raw images and compute 10-channel covariance matrices."""
        import h5py
        import pandas as pd
        from skimage import filters
        from tqdm import tqdm

        cache_dir = self.dataset_dir
        filename = cache_dir / "train64.h5"
        csv_path = cache_dir / "train64.csv"

        if not filename.exists() or not csv_path.exists():
            raise RuntimeError(
                "Failed to open the data, please download it from the link: "
                "https://drive.google.com/drive/folders/13IqVNr3HawAjPfjMqjtync3mDd98WGc9?usp=share_link"
                "cache dir: {}".format(cache_dir)
            )

        with h5py.File(filename, "r") as f:
            images = f["images"]
            labels_raw = pd.read_csv(csv_path)["indication_type"]

            ind = np.where((labels_raw == "good") | (labels_raw == "cut"))[0]
            coo = images[ind, :, :, 0]
            lab = labels_raw.iloc[ind].values

        covs = []
        for i in tqdm(range(len(ind)), desc="Computing textile covariances"):
            g = (coo[i, 1:, 1:] - np.mean(coo[i, 1:, 1:])) / np.std(coo[i, 1:, 1:])
            gauss = filters.gaussian(g, sigma=2)
            gauss1 = filters.gaussian(g, sigma=3)
            val = np.vstack(
                [
                    g.flatten(),
                    filters.gaussian(g, sigma=2).flatten(),
                    filters.gaussian(g, sigma=3).flatten(),
                    filters.gaussian(g, sigma=4).flatten(),
                    filters.laplace(g).flatten(),
                    filters.farid(g).flatten(),
                    filters.laplace(gauss).flatten(),
                    filters.laplace(gauss1).flatten(),
                    filters.farid(gauss).flatten(),
                    filters.farid(gauss1).flatten(),
                ]
            )
            covs.append(np.cov(val))

        all_cov = np.array(covs)

        # Remap string labels to integers (alphabetical: "cut"→0, "good"→1)
        original_classes = np.unique(lab)
        class_mapping = {old: new for new, old in enumerate(original_classes)}
        all_labels = np.vectorize(class_mapping.get)(lab)

        return all_cov, all_labels.astype(int)

    def setup(self, stage=None):
        cache_file = self.dataset_dir / "data_textile_all.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            all_cov = data["all_cov"]
            all_labels = data["all_labels"].astype(int)
        else:
            all_cov, all_labels = self._compute_covariances()
            np.savez(cache_file, all_cov=all_cov, all_labels=all_labels)

        # 80/10/10 train/val/test split (shuffled, reproducible)
        rng = np.random.default_rng(self.seed)
        n = len(all_cov)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        x_train = all_cov[idx[:n_train]]
        y_train = all_labels[idx[:n_train]]
        x_val = all_cov[idx[n_train : n_train + n_val]]
        y_val = all_labels[idx[n_train : n_train + n_val]]
        x_test = all_cov[idx[n_train + n_val :]]
        y_test = all_labels[idx[n_train + n_val :]]

        if self.recenter:
            x_train, x_val, x_test = self._center_and_scale(
                x_train, x_val, x_test, scale=True
            )

        # Compute per-class Riemannian means and variances on training data
        self._compute_class_means_var(x_train, y_train)

        self.train_ds = TensorDataset(
            torch.from_numpy(x_train).to(self.dtype),
            torch.from_numpy(y_train).long(),
        )
        self.val_ds = TensorDataset(
            torch.from_numpy(x_val).to(self.dtype),
            torch.from_numpy(y_val).long(),
        )
        self.test_ds = TensorDataset(
            torch.from_numpy(x_test).to(self.dtype),
            torch.from_numpy(y_test).long(),
        )

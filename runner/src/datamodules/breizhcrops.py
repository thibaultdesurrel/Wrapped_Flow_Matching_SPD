"""DataModule for BreizhCrops crop-type mapping dataset.

Geographic split: frh01+frh02+frh03 = train+val, frh04 = test.
Val is carved from train at 90/10. Classes filtered to [0, 1, 3, 7].
Covariance matrices are cached to disk on first run.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset

from datamodules.base_datamodule import BaseDataModule

CLASSES_TO_KEEP = np.array([0, 1, 3, 7])


class BreizhCropsDataModule(BaseDataModule):
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

    def setup(self, stage=None):
        import breizhcrops as bzh
        from pyriemann.estimation import Covariances
        from tqdm import tqdm

        cache_dir = self.dataset_dir
        train_cache = cache_dir / f"data_breizhcrops_train_{self.seed}.npz"
        test_cache = cache_dir / f"data_breizhcrops_test_{self.seed}.npz"

        if train_cache.exists() and test_cache.exists():
            train_data = np.load(train_cache)
            x_train_val = train_data["all_cov"]
            y_train_val = train_data["all_labels"].astype(int)
            test_data = np.load(test_cache)
            x_test = test_data["all_cov"]
            y_test = test_data["all_labels"].astype(int)
            # Re-encode to contiguous 0-based integers in case the cache was built
            # before the fix (labels could be {0,1,3,7} instead of {0,1,2,3}).
            le = LabelEncoder().fit(y_train_val)
            y_train_val = le.transform(y_train_val)
            y_test = le.transform(y_test)
        else:
            frh01 = bzh.BreizhCrops(region="frh01", root=str(cache_dir))
            frh02 = bzh.BreizhCrops(region="frh02", root=str(cache_dir))
            frh03 = bzh.BreizhCrops(region="frh03", root=str(cache_dir))
            frh04 = bzh.BreizhCrops(region="frh04", root=str(cache_dir))

            splits = {
                "train": (
                    torch.utils.data.ConcatDataset([frh01, frh02, frh03]),
                    train_cache,
                ),
                "test": (frh04, test_cache),
            }

            for split_name, (dataset, cache_file) in splits.items():
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=256, num_workers=self.num_workers, shuffle=False
                )
                all_x_batches, all_y_batches = [], []
                for x, y, _ in tqdm(loader, desc=f"Loading {split_name}"):
                    # x: (B, T, C) → swap to (B, C, T) for covariance estimation
                    all_x_batches.append(x.numpy().swapaxes(1, 2))
                    all_y_batches.append(y.numpy())

                all_x = np.concatenate(all_x_batches, axis=0)
                all_labels = np.concatenate(all_y_batches, axis=0)
                all_cov = Covariances(estimator="oas").fit_transform(all_x)

                # Remap original integer labels to 0-based sequential integers
                all_labels = LabelEncoder().fit_transform(all_labels)

                # Filter to keep only the four classes of interest
                mask = np.isin(all_labels, CLASSES_TO_KEEP)
                all_cov = all_cov[mask]
                all_labels = all_labels[mask]

                # Re-encode filtered labels to contiguous 0-based integers {0,1,2,3}
                all_labels = LabelEncoder().fit_transform(all_labels)

                np.savez(cache_file, all_cov=all_cov, all_labels=all_labels)

            x_train_val = np.load(train_cache)["all_cov"]
            y_train_val = np.load(train_cache)["all_labels"].astype(int)
            x_test = np.load(test_cache)["all_cov"]
            y_test = np.load(test_cache)["all_labels"].astype(int)

        # 90/10 train/val split (shuffled, reproducible)
        rng = np.random.default_rng(self.seed)
        n = len(x_train_val)
        idx = rng.permutation(n)
        split = int(0.9 * n)
        x_train, y_train = x_train_val[idx[:split]], y_train_val[idx[:split]]
        x_val, y_val = x_train_val[idx[split:]], y_train_val[idx[split:]]

        if self.recenter:
            x_train, x_val, x_test = self._center_and_scale(
                x_train, x_val, x_test, scale=False
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

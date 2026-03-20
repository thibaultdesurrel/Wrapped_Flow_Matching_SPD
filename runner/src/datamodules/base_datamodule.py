"""Base DataModule for SPD matrix datasets.

BaseDataModule handles DataLoader creation and SPDBatch collation.
Subclasses implement setup() to populate train_ds, val_ds, and test_ds.

BaseEEGDataModule adds MOABB loading, OAS covariance estimation,
and group-based test splitting on top.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningDataModule

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm, powm


@dataclass
class SPDBatch:
    """Batch format expected by CFMBase.preprocess_batch."""

    x1: torch.Tensor  # (batch_size, n, n)
    labels: torch.Tensor  # (batch_size,)


def _collate_spd(batch):
    """Collate a list of (x1, label) tuples into an SPDBatch."""
    x1 = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return SPDBatch(x1=x1, labels=labels)


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        seed: int = 42,
        dtype: str = "float64",
        recenter: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dtype = DTYPE_MAP[dtype]
        self.recenter = recenter

    @staticmethod
    def _center_and_scale(X_fit, *X_others, scale=True):
        """Center SPD matrices around identity and optionally rescale to unit Fréchet variance.

        Fits on X_fit (training set), applies the same transform to all arrays.
        Returns transformed arrays in input order.
        """
        X_bar = mean_riemann(X_fit)
        inv_sqrt_mean = invsqrtm(X_bar)

        if scale:
            var_frechet = np.mean(distance_riemann(X_fit, X_bar) ** 2)
            power = 1.0 / np.sqrt(var_frechet)

        results = []
        for X in (X_fit, *X_others):
            X_c = inv_sqrt_mean @ X @ inv_sqrt_mean
            if scale:
                X_c = powm(X_c, power)
            results.append(X_c)

        return tuple(results)

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_spd,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_spd,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_spd,
        )


class BaseEEGDataModule(BaseDataModule):
    """Base for EEG datasets: loads raw trials via MOABB, estimates covariances,
    and splits by group index."""

    def __init__(
        self,
        test_group_idx: int,
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
        self.test_group_idx = test_group_idx

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ts, y, groups) from MOABB.

        ts:     (n_samples, n_channels, n_timepoints) — raw EEG trials
        y:      (n_samples,) — integer class labels
        groups: (n_samples,) — integer group indices
        """
        raise NotImplementedError

    @staticmethod
    def _filter_outliers(X, y, groups):
        """Remove covariance matrices with extreme values or high Mahalanobis distance."""
        from sklearn.covariance import OAS
        from scipy.spatial.distance import mahalanobis

        # Filter matrices with absolute values above 1e4
        mask_abs = np.max(np.abs(X), axis=(-2, -1)) < 1e4

        # Filter matrices with Mahalanobis distance above 90th percentile
        X_flat = X.reshape(X.shape[0], -1)
        oas = OAS()
        oas.fit(X_flat)
        cov_inv = np.linalg.inv(oas.covariance_)
        mean = np.mean(X_flat, axis=0)
        maha_dists = np.array(
            [mahalanobis(X_flat[i], mean, cov_inv) for i in range(len(X_flat))]
        )
        mask_maha = maha_dists < np.percentile(maha_dists, 90)

        mask = mask_abs & mask_maha
        return X[mask], y[mask], groups[mask]

    def setup(self, stage=None):
        from pyriemann.estimation import Covariances

        ts, y, groups = self.load_raw_data()
        X = Covariances(estimator="oas").fit_transform(ts)

        # Remove outlier covariance matrices
        X, y, groups = self._filter_outliers(X, y, groups)

        # Split: test = group at test_group_idx, rest = train + val
        test_mask = groups == self.test_group_idx
        x_test, y_test = X[test_mask], y[test_mask]
        x_tv, y_tv = X[~test_mask], y[~test_mask]

        # 90/10 train/val split (shuffled)
        rng = np.random.default_rng(self.seed)
        n = len(x_tv)
        idx = rng.permutation(n)
        split = int(0.9 * n)

        x_train, y_train = x_tv[idx[:split]], y_tv[idx[:split]]
        x_val, y_val = x_tv[idx[split:]], y_tv[idx[split:]]

        # Center around identity and rescale to unit Fréchet variance (fit on train only)
        if self.recenter:
            x_train, x_val, x_test = self._center_and_scale(
                x_train, x_val, x_test, scale=True
            )

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

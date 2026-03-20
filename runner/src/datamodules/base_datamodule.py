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
    def __init__(self, batch_size: int, num_workers: int = 0, seed: int = 42, dtype: str = "float64"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dtype = DTYPE_MAP[dtype]

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

    def __init__(self, test_group_idx: int, batch_size: int, num_workers: int = 0, seed: int = 42, dtype: str = "float64"):
        super().__init__(batch_size=batch_size, num_workers=num_workers, seed=seed, dtype=dtype)
        self.test_group_idx = test_group_idx

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ts, y, groups) from MOABB.

        ts:     (n_samples, n_channels, n_timepoints) — raw EEG trials
        y:      (n_samples,) — integer class labels
        groups: (n_samples,) — integer group indices
        """
        raise NotImplementedError

    def setup(self, stage=None):
        from pyriemann.estimation import Covariances

        ts, y, groups = self.load_raw_data()
        X = Covariances(estimator="oas").fit_transform(ts)

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

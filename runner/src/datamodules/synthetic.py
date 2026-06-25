import torch
from torch.utils.data import TensorDataset
from datamodules.base_datamodule import BaseDataModule


class SwissRollDataModule(BaseDataModule):
    def __init__(
        self,
        n_samples: int,
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
        self.n_samples = n_samples

    def create_swiss_roll(self, n_samples=1000):
        # 1. Generate Spiral parameters
        # torch.rand returns uniform random numbers in [0, 1)
        t = (
            2 * torch.pi * (1 + 2 * torch.rand(n_samples, dtype=self.dtype))
        )  # Angle parameter
        width = torch.rand(n_samples, dtype=self.dtype) * 3.0  # Ribbon width

        # 2. Create Spiral in horizontal plane (u=a-c, v=2b)
        radius = t
        u = radius * torch.cos(t)
        v = radius * torch.sin(t)

        # 3. Lift to vertical axis (w=a+c).
        # The cone boundary is at w = sqrt(u^2 + v^2).
        w_boundary = torch.sqrt(u**2 + v**2)
        w = w_boundary + width + 1.0

        # 4. Map back to Matrix Coordinates (a, b, c)
        b = v / 2.0
        a = (w + u) / 2.0
        c = (w - u) / 2.0

        # Stack into a tensor (n, 2, 2)
        # P = [[a, b], [b, c]]
        x1 = torch.zeros((n_samples, 2, 2), dtype=self.dtype)
        x1[:, 0, 0] = a
        x1[:, 0, 1] = b
        x1[:, 1, 0] = b
        x1[:, 1, 1] = c

        return x1

    def setup(self, stage=None):
        torch.manual_seed(self.seed)
        x1 = self.create_swiss_roll(self.n_samples)
        labels = torch.zeros(self.n_samples, dtype=torch.long)

        n_train = int(0.8 * self.n_samples)
        n_val = int(0.1 * self.n_samples)

        idx = torch.randperm(self.n_samples)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        x_train = x1[train_idx]
        x_val = x1[val_idx]
        x_test = x1[test_idx]

        if self.recenter:
            x_train, x_val, x_test = self._center_and_scale(
                x_train, x_val, x_test, scale=True
            )

        self.train_ds = TensorDataset(x_train, labels[train_idx])
        self.val_ds = TensorDataset(x_val, labels[val_idx])
        self.test_ds = TensorDataset(x_test, labels[test_idx])


class Figure8DataModule(BaseDataModule):
    def __init__(
        self,
        n_samples: int,
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
        self.n_samples = n_samples

    def create_figure8(self, n_samples=1000):
        scale = 4.0
        width_noise = 1.0
        # 1. Generate Parameter t (position along the figure-8)
        # t goes from 0 to 2*pi to trace the complete shape.
        t = 2 * torch.pi * torch.rand(n_samples, dtype=self.dtype)

        # Generate random width for the ribbon effect
        # This adds noise to how far the points are from the exact cone boundary.
        ribbon_width = torch.rand(n_samples, dtype=self.dtype) * width_noise
        # 2. Create Figure-8 in the 'Horizontal' Plane (u, v)
        # u corresponds to (a - c)
        # v corresponds to (2 * b)
        # We use parametric equations for a figure-8 (lemniscate-like shape).
        u = scale * torch.cos(t)
        v = scale * torch.sin(2 * t)

        # 3. Lift to the Cone's 'Vertical' Axis (w)
        # w corresponds to the trace (a + c).
        # The boundary of the SPD cone is defined by w = sqrt(u^2 + v^2).
        w_boundary = torch.sqrt(u**2 + v**2)

        # We add a small positive buffer + ribbon_width to ensure all matrices
        # are strictly positive definite (inside the cone, not on the boundary).
        w = w_boundary + ribbon_width + 0.5
        # 4. Map back to Matrix Coordinates P = [[a, b], [b, c]]
        # Solving the system: u=a-c, v=2b, w=a+c
        b = v / 2.0
        a = (w + u) / 2.0
        c = (w - u) / 2.0

        # 5. Stack into the final tensor (n_samples, 2, 2)
        x1 = torch.zeros((n_samples, 2, 2), dtype=self.dtype)
        x1[:, 0, 0] = a
        x1[:, 0, 1] = b
        x1[:, 1, 0] = b
        x1[:, 1, 1] = c

        return x1

    def setup(self, stage=None):
        torch.manual_seed(self.seed)
        x1 = self.create_figure8(self.n_samples)
        labels = torch.zeros(self.n_samples, dtype=torch.long)

        n_train = int(0.8 * self.n_samples)
        n_val = int(0.1 * self.n_samples)

        idx = torch.randperm(self.n_samples)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        x_train = x1[train_idx]
        x_val = x1[val_idx]
        x_test = x1[test_idx]

        if self.recenter:
            x_train, x_val, x_test = self._center_and_scale(
                x_train.detach().cpu().numpy(),
                x_val.detach().cpu().numpy(),
                x_test.detach().cpu().numpy(),
                scale=True,
            )

        self.train_ds = TensorDataset(
            torch.from_numpy(x_train).to(self.dtype), labels[train_idx]
        )
        self.val_ds = TensorDataset(
            torch.from_numpy(x_val).to(self.dtype), labels[val_idx]
        )
        self.test_ds = TensorDataset(
            torch.from_numpy(x_test).to(self.dtype), labels[test_idx]
        )

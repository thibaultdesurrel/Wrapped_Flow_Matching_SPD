import math

import torch
from torch.utils.data import TensorDataset
from datamodules.base_datamodule import BaseDataModule
from utils import sample_wrapped_gaussian


class SwissRollDataModule(BaseDataModule):
    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        dim: int = 2,
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
        self.dim = dim

    def _random_orthogonal(self, n):
        A = torch.randn(n, n, dtype=self.dtype)
        Q, R = torch.linalg.qr(A)
        # avoid sign ambiguity / improper reflections
        Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
        return Q

    def create_swiss_roll_2d(self, n_samples=1000):
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

    def create_swiss_roll(self, n_samples=1000):
        n = self.dim
        t = 2 * torch.pi * (1 + 2 * torch.rand(n_samples, dtype=self.dtype))
        width = torch.rand(n_samples, dtype=self.dtype) * 3.0

        lam_min = (width + 1.0) / 2.0  # distance to the boundary (rank-deficiency)
        lam_max = t + lam_min  # spiral "radius" growth

        eigvals = lam_min.unsqueeze(1).repeat(1, n)
        eigvals[:, -1] = lam_max
        for k in range(1, n - 1):  # optional: spread remaining eigenvalues
            frac = k / (n - 1)
            eigvals[:, k] = lam_min + frac * (lam_max - lam_min)

        # Generator of the winding. A single 2-plane rotation reproduces n=2
        # exactly; a fixed random skew-symmetric A gives richer winding through
        # SO(n).
        A = torch.zeros(n, n, dtype=self.dtype)
        A[0, 1], A[1, 0] = 1.0, -1.0
        R = torch.matrix_exp(t.view(-1, 1, 1) / 2 * A)

        D = torch.diag_embed(eigvals)
        P = R @ D @ R.transpose(-1, -2)
        Q = self._random_orthogonal(n)
        P_rot = Q @ P @ Q.transpose(-1, -2)
        return P_rot

    def setup(self, stage=None):
        torch.manual_seed(self.seed)
        if self.dim == 2:
            x1 = self.create_swiss_roll_2d(self.n_samples)
        else:
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
        else:
            self.train_ds = TensorDataset(x_train, labels[train_idx])
            self.val_ds = TensorDataset(x_val, labels[val_idx])
            self.test_ds = TensorDataset(x_test, labels[test_idx])


class WGMixtureDataModule(BaseDataModule):
    """Mixture of wrapped Gaussians with controlled condition number.

    Each of the ``num_classes`` components is a wrapped Gaussian
    WG(C_k, 0, sigma_tangent^2 I_D) whose center

        C_k = Q_k diag(lambda_1, ..., lambda_d) Q_k^T

    has eigenvalues log-spaced in [kappa^{-1/2}, kappa^{1/2}] (condition number
    exactly ``kappa``) and an independent random orientation Q_k. ``kappa`` is
    the single knob of the condition-number sweep: the spectral spread of
    log(C_k) is Delta = ln(kappa), which controls the anisotropy that a fixed
    tangent chart at the identity (DiffeoCFM) induces on conditional paths,
    while WrappedCFM's moving-base-point paths remain isotropic.

    By default the data is centered at its Riemannian mean (fit on train) but
    NOT rescaled to unit Frechet variance: the power rescaling X -> X^p
    multiplies the log-eigenvalue spread by p, i.e. it would shrink the very
    conditioning this dataset controls.
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        dim: int = 10,
        num_classes: int = 3,
        kappa: float = 100.0,
        sigma_tangent: float | None = None,
        num_workers: int = 0,
        seed: int = 42,
        dtype: str = "float64",
        recenter: bool = True,
        rescale: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            dtype=dtype,
            recenter=recenter,
        )
        self.n_samples = n_samples
        self.dim = dim
        self.num_classes = num_classes
        self.kappa = float(kappa)
        D = dim * (dim + 1) // 2
        self.sigma_tangent = (
            1.0 / math.sqrt(D) if sigma_tangent is None else float(sigma_tangent)
        )
        self.rescale = rescale

    def _random_orthogonal(self, n):
        A = torch.randn(n, n, dtype=self.dtype)
        Q, R = torch.linalg.qr(A)
        # avoid sign ambiguity / improper reflections
        Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
        return Q

    def create_mixture(self):
        D = self.dim * (self.dim + 1) // 2
        n_per_class = self.n_samples // self.num_classes
        half_log_kappa = math.log10(self.kappa) / 2.0
        eigvals = torch.logspace(
            -half_log_kappa, half_log_kappa, self.dim, dtype=self.dtype
        )

        mu = torch.zeros(D, dtype=self.dtype)
        Sigma = (self.sigma_tangent**2) * torch.eye(D, dtype=self.dtype)

        xs, ys = [], []
        self.centers = []
        for k in range(self.num_classes):
            Q = self._random_orthogonal(self.dim)
            center = Q @ torch.diag(eigvals) @ Q.T
            center = 0.5 * (center + center.T)
            self.centers.append(center)
            xs.append(sample_wrapped_gaussian(n_per_class, center, mu, Sigma))
            ys.append(torch.full((n_per_class,), k, dtype=torch.long))

        return torch.cat(xs), torch.cat(ys)

    def setup(self, stage=None):
        torch.manual_seed(self.seed)
        x1, labels = self.create_mixture()

        n = x1.shape[0]
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        idx = torch.randperm(n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        x_train = x1[train_idx].numpy()
        x_val = x1[val_idx].numpy()
        x_test = x1[test_idx].numpy()

        if self.recenter:
            # Center at the Riemannian mean; keep rescale=False for the
            # condition-number sweep (see class docstring).
            x_train, x_val, x_test = self._center_and_scale(
                x_train, x_val, x_test, scale=self.rescale
            )

        y_train = labels[train_idx]
        self._compute_class_means_var(x_train, y_train.numpy())

        self.train_ds = TensorDataset(torch.from_numpy(x_train).to(self.dtype), y_train)
        self.val_ds = TensorDataset(
            torch.from_numpy(x_val).to(self.dtype), labels[val_idx]
        )
        self.test_ds = TensorDataset(
            torch.from_numpy(x_test).to(self.dtype), labels[test_idx]
        )


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
        else:
            self.train_ds = TensorDataset(x_train, labels[train_idx])
            self.val_ds = TensorDataset(x_val, labels[val_idx])
            self.test_ds = TensorDataset(x_test, labels[test_idx])

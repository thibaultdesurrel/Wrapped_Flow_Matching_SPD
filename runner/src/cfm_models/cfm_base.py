"""Base class for all Conditional Flow Matching models on SPD matrices.

Provides shared infrastructure: LR scheduling and the common training loop skeleton.
Subclasses implement sample_xt(), ut_cond(), and step() for their specific
parameterization of the conditional probability path
p_t(·|X₁) = WG(X̄_t, μ_t, σ_t² I_D).

"""

import torch
import torch.nn.functional as F
from torch.func import jvp, vmap

import math
from lightning.pytorch import LightningModule
from utils import sample_wrapped_gaussian, loglikelihood_wrapped_gaussian
from utils.spd_ops import gradient, logdetG_SPD
from spd_learn.functional import exp_map_airm, clamp_eigvals, get_epsilon

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class CFMBase(LightningModule):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.optimizer_cls = optimizer
        self.data_dtype: torch.dtype = DTYPE_MAP[self.cfg.dtype]
        self.num_classes: int = self.cfg.num_classes
        self.dim: int = self.cfg.dim

        self.sigma_init: float = 1.0 / math.sqrt(self.dim * (self.dim + 1) / 2)
        self.sigma_min: float = 0.0
        self.criterion = torch.nn.MSELoss()

    def sample_x0(self, batch_size, int_labels=None):
        dim_TS = self.dim * (self.dim + 1) // 2
        p0 = torch.eye(self.dim, dtype=self.data_dtype, device=self.device)
        mu0 = torch.zeros(dim_TS, dtype=self.data_dtype, device=self.device)
        sigma0 = (self.sigma_init**2) * torch.eye(
            dim_TS, dtype=self.data_dtype, device=self.device
        )
        x0 = sample_wrapped_gaussian(batch_size, p0, mu0, sigma0)
        return x0

    def preprocess_batch(self, batch, int_labels=None):

        # Get the x1 and labels from the batch
        x1 = batch.x1.to(self.data_dtype)
        labels = batch.labels

        # Convert the labels to one-hot encoding
        labels = F.one_hot(labels, num_classes=self.num_classes).to(self.data_dtype)

        batch_size = x1.shape[0]
        dim = x1.shape[1]
        assert dim == self.dim, f"Expected input dimension {self.dim}, but got {dim}"

        # Sample the random time steps for each sample in the batch
        t = torch.rand(batch_size, dtype=self.data_dtype, device=x1.device)

        # Sample x0
        x0 = self.sample_x0(batch_size, int_labels)

        return x0, x1, t, labels

    def configure_optimizers(self):
        LR = self.cfg.lr
        FACTOR_LR = self.cfg.factor_lr
        WARMUP_EPOCHS = self.cfg.warmup_epochs
        EPOCHS = self.cfg.epochs

        optimizer = self.optimizer_cls(self.parameters(), lr=LR, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=FACTOR_LR,
                    end_factor=1.0,
                    total_iters=WARMUP_EPOCHS,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * FACTOR_LR
                ),
            ],
            milestones=[WARMUP_EPOCHS],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _source_logp(self, x0):
        """Log-density of `x0` under the source distribution.

        The source is the wrapped Gaussian centered at the identity with
        isotropic covariance sigma_init^2 I in the tangent space — the same
        distribution that sample_x0() draws from.
        """
        dim_TS = self.dim * (self.dim + 1) // 2
        p0 = torch.eye(self.dim, dtype=self.data_dtype, device=x0.device)
        mu0 = torch.zeros(dim_TS, dtype=self.data_dtype, device=x0.device)
        sigma0 = (self.sigma_init**2) * torch.eye(
            dim_TS, dtype=self.data_dtype, device=x0.device
        )
        return loglikelihood_wrapped_gaussian(x0, p0, mu0, sigma0)

    def compute_loglikelihood(self, x1, y=None, nsteps=1000, reg_SPD=1e-3):
        raise NotImplementedError(
            "compute_loglikelihood must be implemented by subclasses."
        )

    def _riemannian_loglikelihood(self, x1, y=None, nsteps=1000, reg_SPD=1e-3):
        """Log-likelihood of SPD points ``x1`` under a manifold CFM model.

        Shared by every model whose ``sample_ode`` integrates with the AIRM
        exponential map (WrappedCFM, GeodesicCFM, ...). It is agnostic to the
        conditional probability path used during training: it only uses the
        learned velocity field ``self.net`` and the common source distribution.

        Uses the instantaneous change-of-variables theorem on the manifold:
        the continuity equation is integrated backwards (t=1 -> 0) with a
        Riemannian Euler step (exp map of the learned field v_t), accumulating
        the Riemannian divergence of v_t along the trajectory. The divergence is
        estimated with a single fixed Hutchinson probe ``z`` plus the AIRM metric
        correction 1/2 * d/dt log det G. The endpoint x_0 is scored under the
        source wrapped Gaussian, and

            log p_1(x1) = log p_0(x_0) - \\int_0^1 div_g v_t dt.

        Must be called with grad enabled (it differentiates through ``self.net``).

        Args:
            x1: (N, dim, dim) SPD matrices to score.
            y: optional one-hot conditioning labels (N, num_classes).
            nsteps: number of Euler steps for the backward integration.
            reg_SPD: minimum eigenvalue used to clamp iterates onto the SPD cone.

        Returns:
            (x_0, logp): the integrated source point and the (N,) log-likelihood.
        """
        N = x1.shape[0]

        # Fixed Rademacher probe vector for the Hutchinson trace estimator.
        z = (torch.randn_like(x1) < 0).to(self.data_dtype) * 2.0 - 1.0

        threshold_eps = get_epsilon(x1.dtype, "eigval_clamp")
        if reg_SPD is not None:
            threshold_value = max(float(threshold_eps), float(reg_SPD))
        else:
            threshold_value = float(threshold_eps)

        def augmented_dynamics(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt = xt.requires_grad_(True)
                vt = self.net(xt, t, y)

                # Hutchinson estimate of the Euclidean divergence: z^T (dv/dx) z.
                vt_dot_z = torch.einsum(
                    "ij,ij->i", vt.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_vt_dot_z = gradient(vt_dot_z, xt)
                div = torch.einsum(
                    "ij,ij->i",
                    grad_vt_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

                # Riemannian metric correction: + 1/2 <grad log det G, v_t>.
                def _jvp(x, v):
                    return jvp(logdetG_SPD, (x,), (v,))[1]

                corr = vmap(_jvp)(xt, vt)
                div = div + 0.5 * corr.to(div)

            return (vt.detach(), -div.detach())

        states = (x1, torch.zeros(N, device=x1.device, dtype=self.data_dtype))
        all_t = torch.linspace(
            1.0, 0.0, nsteps + 1, device=x1.device, dtype=self.data_dtype
        )

        for i in range(nsteps):
            t = all_t[i]
            dt = all_t[i + 1] - all_t[i]  # negative (integrating 1 -> 0)
            t_reshaped = t.expand(N)
            dx, dlogp = augmented_dynamics(t_reshaped, states)
            x = exp_map_airm(states[0], dx * dt)
            x = clamp_eigvals.apply(x, threshold_value)
            logp = states[1] + dlogp * dt
            states = (x, logp)

        x_0, log_det = states
        source_logp = self._source_logp(x_0)

        return x_0, source_logp - log_det

    def step(self, batch):
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def training_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("train", loss, batch_size=batch.x1.shape[0])
        return loss

    def validation_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("val", loss, batch_size=batch.x1.shape[0], on_step=False)

    def test_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("test", loss, batch_size=batch.x1.shape[0], on_step=False)

    def log_step(self, prefix, loss, batch_size, on_step=True):
        """Log the training/val/test loss metric."""
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def forward(self, x, t, y):
        return self.net(x, t, y)

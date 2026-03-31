"""WrappedCFM — the proposed method.

Paper parameterization (Table 1):
    X̄_t(X₁) = X₁ᵗ
    μ_t(X₁)  = 0
    σ_t(X₁)  = σ_init - (σ_init - σ_min)t

See Section 5, Theorem 5.1 for the conditional vector field.
"""

import torch
from torch.func import jvp


from cfm_models.cfm_base import CFMBase
from solver.riemannian_ode_solver import RiemannianODESolver
from utils.spd_ops import airm_inner_product, power_matrix, airm_geodesic
from backbones.common import ProjectToTangent
from spd_learn.functional import matrix_log
from utils import sample_wrapped_gaussian


class WrappedCFM_with_prior(CFMBase):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)
        self.net = ProjectToTangent(self.net)
        # Placeholder until datamodule provides per-class Riemannian means
        self.register_buffer(
            "X_init",
            torch.eye(self.dim, dtype=self.data_dtype)
            .unsqueeze(0)
            .expand(self.num_classes, -1, -1)
            .clone(),
        )

    def set_prior_stats(self, class_means, class_vars):
        """Set per-class prior statistics (can be called outside training)."""
        self.X_init = class_means.to(self.device)
        self.sigma_classes = class_vars.to(self.device)
        self.sigma_init = torch.sqrt(
            1 / (self.dim * (self.dim + 1) // 2) * self.sigma_classes
        )

    def on_fit_start(self):
        self.set_prior_stats(
            self.trainer.datamodule.class_means,
            self.trainer.datamodule.class_vars,
        )

    def sample_x0(self, batch_size, int_labels):
        x0 = torch.empty(
            batch_size, self.dim, self.dim, dtype=self.data_dtype, device=self.device
        )
        for c in range(self.num_classes):
            idxs = (int_labels == c).nonzero(as_tuple=True)[0]
            num_c = idxs.shape[0]
            if num_c > 0:
                p0 = self.X_init[c]
                mu0 = torch.zeros(
                    self.dim * (self.dim + 1) // 2,
                    dtype=self.data_dtype,
                    device=self.device,
                )
                sigma0 = (self.sigma_init[c] ** 2) * torch.eye(
                    self.dim * (self.dim + 1) // 2,
                    dtype=self.data_dtype,
                    device=self.device,
                )
                x0[idxs] = sample_wrapped_gaussian(
                    num_c,
                    p0,
                    mu0,
                    sigma0,
                )
        return x0

    def bar_X_t(self, t, x1, int_labels):
        X_init = self.X_init[int_labels]  # (B, dim, dim)
        return airm_geodesic(X_init, x1, t)

    def sigma_t(self, t, int_labels):
        sigma_init_c = self.sigma_init[int_labels]  # (B,)
        return sigma_init_c - (sigma_init_c - self.sigma_min) * t.reshape(-1)  # (B,)

    def sample_xt(self, x0, x1, t, int_labels):
        # Unwrap x0 from WG(X_init[c], ...) to WG(I, ...)
        X_init = self.X_init[int_labels]  # (B, dim, dim)
        X_init_inv_sqrt = power_matrix(X_init, -0.5)

        sigma_init_c = self.sigma_init[int_labels]  # (B,)
        bar_xt = self.bar_X_t(t, x1, int_labels)
        bar_xt_sqrt = power_matrix(bar_xt, 1 / 2)
        x0_pow_sigma = power_matrix(
            X_init_inv_sqrt @ x0 @ X_init_inv_sqrt, self.sigma_t(t, int_labels) / sigma_init_c
        )
        return bar_xt_sqrt @ x0_pow_sigma @ bar_xt_sqrt

    def ut_cond(self, xt, x1, t, int_labels):
        def gamma_sqrt(t):
            return power_matrix(self.bar_X_t(t, x1, int_labels), 1 / 2)

        bar_x_t_sqrt, St = jvp(gamma_sqrt, (t,), (torch.ones_like(t).to(t),))
        bar_x_t_inv_sqrt = torch.linalg.inv(bar_x_t_sqrt)

        Mt = bar_x_t_inv_sqrt @ xt @ bar_x_t_inv_sqrt
        Mt_sqrt = power_matrix(Mt, 0.5)

        sigma_init_c = self.sigma_init[int_labels]  # (B,)
        sigma_t_prime = -(sigma_init_c - self.sigma_min)  # (B,)
        sigma_t = self.sigma_t(t, int_labels)  # (B,)
        sigma_t_prime_over_sigma_t = (sigma_t_prime / sigma_t).reshape(-1, 1, 1)  # (B, 1, 1)

        return (
            St @ bar_x_t_inv_sqrt @ xt
            + xt @ bar_x_t_inv_sqrt @ St
            + sigma_t_prime_over_sigma_t
            * bar_x_t_sqrt
            @ Mt_sqrt
            @ matrix_log.apply(Mt)
            @ Mt_sqrt
            @ bar_x_t_sqrt
        )

    def step(self, batch):
        x0, x1, t, labels_onehot = self.preprocess_batch(batch, int_labels=batch.labels)
        int_labels = batch.labels

        xt = self.sample_xt(x0, x1, t, int_labels)
        ut = self.ut_cond(xt, x1, t, int_labels)
        vt = self.forward(xt, t, labels_onehot)

        diff = vt - ut
        loss = airm_inner_product(xt, diff, diff).mean() / self.dim

        return loss, vt, ut

    @torch.no_grad()
    def sample_ode(self, x0, y):
        was_training = self.net.training
        self.net.eval()
        solver = RiemannianODESolver(self.net)

        results = solver.sample(x0, y)

        if was_training:
            self.net.train()
        return results

"""WrappedCFM — the proposed method.

Paper parameterization (Table 1):
    X̄_t(X₁) = X₁ᵗ
    μ_t(X₁)  = 0
    σ_t(X₁)  = σ_init - (σ_init - σ_min)t

See Section 5, Theorem 5.1 for the conditional vector field.
"""

import torch

from cfm_models.cfm_base import CFMBase
from solver.riemannian_ode_solver import RiemannianODESolver
from utils.spd_ops import airm_inner_product, power_matrix
from backbones.common import ProjectToTangent
from spd_learn.functional import matrix_sqrt, matrix_log


class WrappedCFM(CFMBase):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)
        self.net = ProjectToTangent(self.net)
        print("Initialized WrappedCFM with net wrapped in ProjectToTangent.")

    def sigma_t(self, t):
        return self.sigma_init - (self.sigma_init - self.sigma_min) * t

    def sample_xt(self, x0, x1, t):
        x1_pow_t_over_2 = power_matrix(x1, t / 2)
        x0_pow_sigma = power_matrix(x0, self.sigma_t(t) / self.sigma_init)
        return x1_pow_t_over_2 @ x0_pow_sigma @ x1_pow_t_over_2

    def ut_cond(self, xt, x1, t):
        x1_pow_t_over_2 = power_matrix(x1, t / 2)
        x1_pow_minus_t_over_2 = power_matrix(x1, -t / 2)
        log_x1 = matrix_log.apply(x1)

        Mt = x1_pow_minus_t_over_2 @ xt @ x1_pow_minus_t_over_2
        Mt_sqrt = matrix_sqrt.apply(Mt)

        sigma_t_prime = -(self.sigma_init - self.sigma_min)
        sigma_t = self.sigma_t(t)
        sigma_t_prime_over_sigma_t = (sigma_t_prime / sigma_t).reshape(-1, 1, 1)

        return (
            0.5 * log_x1 @ xt
            + 0.5 * xt @ log_x1
            + sigma_t_prime_over_sigma_t
            * x1_pow_t_over_2
            @ Mt_sqrt
            @ matrix_log.apply(Mt)
            @ Mt_sqrt
            @ x1_pow_t_over_2
        )

    def step(self, batch):
        x0, x1, t, labels = self.preprocess_batch(batch)

        xt = self.sample_xt(x0, x1, t)
        ut = self.ut_cond(xt, x1, t)
        vt = self.forward(xt, t, labels)

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

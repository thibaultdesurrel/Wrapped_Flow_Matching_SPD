"""EuclideanCFM — Euclidean CFM baseline.

Operates in flat Euclidean space without any Riemannian structure.
Uses Gaussian conditional probability paths (Section 2.3 of the paper):
    μ_t(X₁) = t · X₁
    σ_t(X₁) = σ_init - (σ_init - σ_min) · t
and MSE (Frobenius) loss.
"""

import torch

from cfm_models.cfm_base import CFMBase
from solver.euclidean_ode_solver import EuclideanODESolver


class EuclideanCFM(CFMBase):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)

    def sample_xt(self, x0, x1, t):
        t_ = self._expand_t(t, x0)
        sigma_t = self.sigma_init - (self.sigma_init - self.sigma_min) * t_
        return (sigma_t / self.sigma_init) * x0 + t_ * x1

    def ut_cond(self, xt, x1, t):
        t_ = self._expand_t(t, xt)
        sigma_t = self.sigma_init - (self.sigma_init - self.sigma_min) * t_
        sigma_t_prime = -(self.sigma_init - self.sigma_min)
        return (sigma_t_prime / sigma_t) * (xt - t_ * x1) + x1

    def step(self, batch):
        x0, x1, t, labels = self.preprocess_batch(batch)

        xt = self.sample_xt(x0, x1, t)
        vt = self.forward(xt, t, labels)
        ut = self.ut_cond(xt, x1, t)

        loss = self.criterion(vt, ut)
        return loss, vt, ut

    @torch.no_grad()
    def sample_ode(self, x0, y):
        was_training = self.net.training
        self.net.eval()
        solver = EuclideanODESolver(self.net)
        result = solver.sample(x0, y)
        if was_training:
            self.net.train()
        return result

"""EuclideanCFM — Euclidean CFM baseline.

Operates in flat Euclidean space without any Riemannian structure.
Uses Gaussian conditional probability paths (Section 2.3 of the paper):
    μ_t(X₁) = t · X₁
    σ_t(X₁) = σ_init - (σ_init - σ_min) · t
and MSE (Frobenius) loss.
"""

import torch

from src.cfm_models.cfm_base import CFMBase
from src.solver.euclidean_ode_solver import EuclideanODESolver
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher


class EuclideanCFM(CFMBase):
    def __init__(self, cfg, net):
        super().__init__(cfg, net)
        self.fm = TargetConditionalFlowMatcher()

    def step(self, batch):
        x0, x1, t, labels = self.preprocess_batch(batch)
        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1, t)
        vt = self.forward(xt, t, labels)

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

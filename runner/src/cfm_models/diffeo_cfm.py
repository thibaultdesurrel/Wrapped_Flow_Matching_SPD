"""DiffeoCFM.

Paper parameterization (Table 1):
    X̄_t(X₁) = I_d
    μ_t(X₁)  = t · log(X₁)
    σ_t(X₁)  = σ_init - (σ_init - σ_min)t

Equivalent to EuclideanCFM applied in the tangent space at the identity:
points are mapped to the tangent space at I_d via Log, then Euclidean CFM
is run in that flat space, and results are mapped back via Exp.
"""

import torch

from cfm_models.euclidean_cfm import EuclideanCFM
from solver.euclidean_ode_solver import EuclideanODESolver
from spd_learn.functional import matrix_log, matrix_exp


class DiffeoCFM(EuclideanCFM):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)

    def step(self, batch):
        x0, x1, t, labels = self.preprocess_batch(batch)

        # Map to the tangent space at the identity via Log
        x0_tangent = matrix_log.apply(x0)
        x1_tangent = matrix_log.apply(x1)

        # Run Euclidean CFM in the tangent space
        xt_tangent = self.sample_xt(x0_tangent, x1_tangent, t)
        vt = self.forward(xt_tangent, t, labels)
        ut = self.ut_cond(xt_tangent, x1_tangent, t)

        loss = self.criterion(vt, ut)
        return loss, vt, ut

    @torch.no_grad()
    def sample_ode(self, x0, y):
        was_training = self.net.training
        self.net.eval()
        solver = EuclideanODESolver(self.net)

        x0_tangent = matrix_log.apply(x0)
        result_tangent = solver.sample(x0_tangent, y)

        if was_training:
            self.net.train()
        return matrix_exp.apply(result_tangent)

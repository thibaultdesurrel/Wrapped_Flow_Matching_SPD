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
from torch.func import jvp

from cfm_models.cfm_base import CFMBase
from solver.riemannian_ode_solver import RiemannianODESolver
from spd_learn.functional import airm_geodesic
from utils.spd_ops import airm_inner_product
from backbones.common import ProjectToTangent


class GeodesicCFM(CFMBase):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)

        self.net = ProjectToTangent(self.net)

    def step(self, batch):
        x0, x1, t, labels = self.preprocess_batch(batch)

        def SPD_geodesic(t):
            return airm_geodesic(x0, x1, t)

        xt, ut = jvp(SPD_geodesic, (t,), (torch.ones_like(t).to(t),))

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

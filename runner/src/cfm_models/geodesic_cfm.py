"""GeodesicCFM.

Conditional probability path defined by the AIRM geodesic from the source X₀
to the target X₁:
    X_t = γ(t) = X₀^{1/2} (X₀^{-1/2} X₁ X₀^{-1/2})^t X₀^{1/2}

The conditional vector field u_t is the geodesic velocity dγ/dt, obtained here
by differentiating the geodesic through it with a JVP. Training matches the
network to u_t under the AIRM inner product; sampling integrates the learned
field on the manifold with the AIRM exponential map.
"""

import torch
from torch.func import jvp

from cfm_models.cfm_base import CFMBase
from solver.riemannian_ode_solver import RiemannianODESolver
from utils.spd_ops import airm_geodesic, airm_inner_product
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
    def sample_ode(self, x0, y, nsteps=100):
        was_training = self.net.training
        self.net.eval()
        solver = RiemannianODESolver(self.net, nsteps=nsteps)

        results = solver.sample(x0, y)

        if was_training:
            self.net.train()
        return results

    def compute_loglikelihood(self, x1, y=None, nsteps=1000, reg_SPD=1e-3):
        """Manifold log-likelihood via the shared Riemannian CFM estimator.

        GeodesicCFM shares WrappedCFM's source distribution (WG at the identity)
        and AIRM exp-map integration, so the likelihood is identical: only the
        learned velocity field differs. See CFMBase._riemannian_loglikelihood.
        """
        return self._riemannian_loglikelihood(x1, y, nsteps, reg_SPD)

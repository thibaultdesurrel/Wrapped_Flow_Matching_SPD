"""DiffeoCFM.

Paper parameterization (Table 1):
    X̄_t(X₁) = I_d
    μ_t(X₁)  = t · log(X₁)
    σ_t(X₁)  = σ_init - (σ_init - σ_min)t

Equivalent to EuclideanCFM applied in the tangent space at the identity:
points are mapped to the tangent space at I_d via Log, then Euclidean CFM
is run in that flat space, and results are mapped back via Exp.
"""

import math

import torch

from cfm_models.euclidean_cfm import EuclideanCFM
from solver.euclidean_ode_solver import EuclideanODESolver
from spd_learn.functional import matrix_log, matrix_exp, sym_to_upper, vec_to_sym


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

    def _exp_logjac(self, x1):
        """Log-Jacobian of the exp map at the identity, evaluated at log(x1).

        J_exp(s) = prod_{i<j} sinh(d_ij/2) / (d_ij/2), with d_ij the differences
        of eigenvalues of s = log(x1) (i.e. of the log-eigenvalues of x1). This
        is the volume distortion of Exp_I, used to convert a flat tangent-space
        density into one w.r.t. the AIRM Riemannian volume. Same term as in the
        wrapped-Gaussian log-likelihood.
        """
        logw = torch.linalg.eigvalsh(x1).clamp_min(1e-10).log()  # (N, d)
        d = x1.shape[-1]
        idx = torch.triu_indices(d, d, offset=1)
        half = 0.5 * (logw[:, idx[0]] - logw[:, idx[1]])
        sinhc = torch.where(
            half.abs() < 1e-12, torch.ones_like(half), torch.sinh(half) / half
        )
        return torch.log(sinhc).sum(dim=-1)

    def compute_loglikelihood(self, x1, y=None, nsteps=1000, reg_SPD=1e-3):
        """Log-likelihood of SPD points ``x1`` under DiffeoCFM (flat-space CFM).

        DiffeoCFM flows in the tangent space at the identity and maps back with
        Exp, so the density is computed by a *flat* (Euclidean) change of
        variables in the orthonormal vech coordinates of that tangent space,
        followed by the exp-map Jacobian that converts to the AIRM Riemannian
        volume (your convention, matching the WrappedCFM / GeodesicCFM source):

            log p_1(x1) = log N(v0; 0, sigma_init^2 I_D)
                          - \\int_0^1 div_v g dt          (flat, no metric corr.)
                          - log J_exp(log x1).

        Here v = sym_to_upper(s) are the orthonormal tangent coordinates,
        g(v) = sym_to_upper(net(vec_to_sym(v))) is the velocity in those coords,
        and the source in vech space is exactly an isotropic Gaussian (because
        sample_x0 draws Exp(vec_to_sym(z)) with z ~ N(0, sigma_init^2 I)).

        Must be called with grad enabled (it differentiates through ``self.net``).
        ``reg_SPD`` is unused (no manifold clamping in flat space); kept for a
        common signature.

        Returns:
            (x_0, logp): the integrated SPD source point and the (N,) log-likelihood.
        """
        N = x1.shape[0]
        D = self.dim * (self.dim + 1) // 2

        # Fixed Rademacher probe in tangent (vech) space for Hutchinson trace.
        eps = (torch.randn(N, D, dtype=self.data_dtype, device=x1.device) < 0)
        eps = eps.to(self.data_dtype) * 2.0 - 1.0

        def velocity(v, t):
            return sym_to_upper(self.net(vec_to_sym(v), t, y))

        def augmented_dynamics(t, v):
            with torch.set_grad_enabled(True):
                v = v.requires_grad_(True)
                gv = velocity(v, t)
                # Hutchinson estimate of the flat divergence: eps^T (dg/dv) eps.
                gv_dot_eps = (gv * eps).sum(dim=-1)
                grad = torch.autograd.grad(
                    gv_dot_eps.sum(), v, create_graph=True, retain_graph=True
                )[0]
                div = (grad * eps).sum(dim=-1)
            return gv.detach(), -div.detach()

        # Map data to orthonormal tangent (vech) coordinates.
        v1 = sym_to_upper(matrix_log.apply(x1))

        states = (v1, torch.zeros(N, device=x1.device, dtype=self.data_dtype))
        all_t = torch.linspace(
            1.0, 0.0, nsteps + 1, device=x1.device, dtype=self.data_dtype
        )

        for i in range(nsteps):
            t = all_t[i]
            dt = all_t[i + 1] - all_t[i]  # negative (integrating 1 -> 0)
            t_reshaped = t.expand(N)
            dv, dlogp = augmented_dynamics(t_reshaped, states[0])
            v = states[0] + dv * dt  # flat Euler step
            logp = states[1] + dlogp * dt
            states = (v, logp)

        v0, log_det = states

        # Source density: isotropic Gaussian in vech coordinates.
        var = self.sigma_init**2
        source_logp = -0.5 * (
            D * math.log(2.0 * math.pi)
            + D * math.log(var)
            + (v0**2).sum(dim=-1) / var
        )

        # Convert flat tangent density -> AIRM Riemannian volume at the data point.
        jac_exp = self._exp_logjac(x1)

        x_0 = matrix_exp.apply(vec_to_sym(v0))
        return x_0, source_logp - log_det - jac_exp

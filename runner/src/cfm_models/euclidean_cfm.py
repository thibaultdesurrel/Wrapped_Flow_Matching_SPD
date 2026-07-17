"""EuclideanCFM — Euclidean CFM baseline.

Operates in flat Euclidean space without any Riemannian structure.
Uses Gaussian conditional probability paths (Section 2.3 of the paper):
    μ_t(X₁) = t · X₁
    σ_t(X₁) = σ_init - (σ_init - σ_min) · t
and MSE (Frobenius) loss.
"""

import math

import torch

from cfm_models.cfm_base import CFMBase
from solver.euclidean_ode_solver import EuclideanODESolver
from spd_learn.functional import sym_to_upper, vec_to_sym


class EuclideanCFM(CFMBase):
    def __init__(self, cfg, net, optimizer=None):
        super().__init__(cfg, net, optimizer)
        self.sigma_init = 1.0
        self.sigma_min = self.sigma_init / 50.0

    def sample_x0(self, batch_size, int_labels=None):
        """Classical Gaussian source on the space of symmetric matrices.

        Samples z ~ N(0, sigma_init^2 I_D) in the orthonormal vech coordinates
        (D = d(d+1)/2) and reshapes to a symmetric matrix. Unlike CFMBase and
        the manifold models, this does NOT exp-map onto the SPD cone: the source
        is a plain Euclidean Gaussian, so x0 is symmetric but generally not
        positive definite. This is what makes the model a classical Euclidean CFM.
        """
        dim_TS = self.dim * (self.dim + 1) // 2
        z = self.sigma_init * torch.randn(
            batch_size, dim_TS, dtype=self.data_dtype, device=self.device
        )
        return vec_to_sym(z)

    def _expand_t(self, t, x):
        """Reshape t from (B,) to broadcast with x of shape (B, ..., ...)."""
        return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))

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

    def compute_loglikelihood(self, x1, y=None, nsteps=1000, reg_SPD=1e-3):
        """Classical Euclidean CFM log-likelihood (flat change of variables).

        The model lives in the flat space of symmetric matrices (orthonormal
        vech coordinates v = sym_to_upper(x)); x1 is used directly, not log(x1).
        The density is the standard Euclidean instantaneous change of variables:

            log p_1(x1) = log N(v0; 0, sigma_init^2 I_D) - \\int_0^1 div_v g dt,

        with g(v) = sym_to_upper(net(vec_to_sym(v))). There is NO manifold
        structure: no exp-map Jacobian and no metric correction.

        NOTE: this density is taken w.r.t. the flat Lebesgue measure on symmetric
        matrices, NOT the AIRM Riemannian volume used by WrappedCFM / GeodesicCFM
        / DiffeoCFM. To compare NLLs against those models, add
        (dim + 1) / 2 * logdet(x1) (the log AIRM volume factor) to the result.

        Must be called with grad enabled (it differentiates through ``self.net``).
        ``reg_SPD`` is unused (no manifold clamping in flat space).

        Returns:
            (x_0, logp): the integrated source point (symmetric matrix) and the
            (N,) log-likelihood.
        """
        N = x1.shape[0]
        D = self.dim * (self.dim + 1) // 2

        # Fixed Rademacher probe in vech space for the Hutchinson trace.
        eps = torch.randn(N, D, dtype=self.data_dtype, device=x1.device) < 0
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

        # Data in vech coordinates (used directly, no Log).
        v1 = sym_to_upper(x1)

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
            D * math.log(2.0 * math.pi) + D * math.log(var) + (v0**2).sum(dim=-1) / var
        )

        return vec_to_sym(v0), source_logp - log_det

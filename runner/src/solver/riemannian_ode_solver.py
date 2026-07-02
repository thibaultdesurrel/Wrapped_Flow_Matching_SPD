import torch

from torchdiffeq import odeint
from spd_learn.functional import clamp_eigvals, get_epsilon, exp_map_airm


class RiemannianODESolver:
    def __init__(self, net, eucl_then_proj=False, nsteps=100):
        self.net = net
        self.eucl_then_proj = eucl_then_proj
        self.nsteps = nsteps

    def sample(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Solve the CFM for a given x0."""

        if self.eucl_then_proj:
            # Solve the ODE in Euclidean space and then project back to the manifold
            time_grid = torch.linspace(0, 1, 2, device=x0.device, dtype=x0.dtype)

            def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                if y is not None:
                    return self.net(x, t, y)
                else:
                    return self.net(x, t)

            sol = odeint(
                ode_func,
                x0,
                t=time_grid,
                method="dopri5",
                atol=1e-5,
                rtol=1e-5,
                options={"min_step": 1e-5},
            )

            epsilon = get_epsilon(x0.dtype, "eigval_clamp")
            threshold_value = max(1e-5, epsilon)
            x1 = clamp_eigvals.apply(sol[-1], threshold_value)

        else:
            # Solve the ODE directly on the manifold using the Riemannian exponential map
            epsilon = get_epsilon(x0.dtype, "eigval_clamp")
            threshold_value = max(1e-5, epsilon)
            nsteps = self.nsteps
            delta_t = 1.0 / nsteps
            x1 = x0
            time_grid = torch.linspace(0, 1, nsteps + 1, device=x0.device, dtype=x0.dtype).view(
                -1, 1
            )

            for i in range(nsteps):
                if y is not None:
                    vt = self.net(x1, time_grid[i], y)
                else:
                    vt = self.net(x1, time_grid[i])

                # Riemannian Euler step
                x1 = exp_map_airm(x1, delta_t * vt)
                x1 = clamp_eigvals.apply(x1, threshold_value)

        return x1

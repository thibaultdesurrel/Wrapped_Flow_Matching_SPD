import torch

from torchdiffeq import odeint
from spd_learn.functional import clamp_eigvals, get_epsilon


class RiemannianODESolver:
    def __init__(self, net):
        self.net = net

    def sample(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Solve the CFM for a given x0."""
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
        return x1

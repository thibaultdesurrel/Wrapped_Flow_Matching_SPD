import torch

from torchdiffeq import odeint


class EuclideanODESolver:
    def __init__(self, net):
        self.net = net

    def sample(self, x0, y):
        """Solve the CFM for a given x0."""
        time_grid = torch.linspace(0, 1, 2, device=x0.device, dtype=x0.dtype)

        def ode_func(t, x):
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

        return sol[-1]

import torch


class EuclideanODESolver:
    def __init__(self, net, nsteps=100):
        self.net = net
        self.nsteps = nsteps

    def sample(self, x0, y):
        """Solve the CFM for a given x0 with a classical (flat) Euler scheme.

        Uses the same fixed integration budget as RiemannianODESolver (100
        steps by default), with the exponential-map update replaced by the
        flat Euclidean step x <- x + h * v_t(x), so that Euclidean/Diffeo and
        the manifold models are compared under the same solver.
        """
        delta_t = 1.0 / self.nsteps
        x1 = x0
        time_grid = torch.linspace(
            0, 1, self.nsteps + 1, device=x0.device, dtype=x0.dtype
        ).view(-1, 1)

        for i in range(self.nsteps):
            if y is not None:
                vt = self.net(x1, time_grid[i], y)
            else:
                vt = self.net(x1, time_grid[i])

            # Euler step
            x1 = x1 + delta_t * vt

        return x1

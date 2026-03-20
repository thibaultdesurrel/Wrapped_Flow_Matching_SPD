"""Shared building blocks for MLP backbones."""

import torch
from torch import nn
from spd_learn.functional import clamp_eigvals, matrix_sqrt


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input (spd_learn version)."""

    def __init__(self, vecfield):
        super().__init__()
        self.vecfield = vecfield

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        x = clamp_eigvals.apply(x, 1e-5)
        v = self.vecfield(x, t, y)

        # Metric normalization: X^{1/2} V X^{1/2}

        x_sqrt = matrix_sqrt.apply(x)
        v = x_sqrt @ v @ x_sqrt
        return v


class TimeDependentSwish(nn.Module):
    """Time-dependent Swish activation function.

    Extracts t from the last element of x. Used by mlp.py and mlp_film.py
    inside nn.Sequential pipelines where t is concatenated with the input.

    Adapted from https://github.com/facebookresearch/riemannian-fm/
    """

    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4), dtype=self.dtype),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim, dtype=self.dtype),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, -1].reshape(-1, 1)
        beta = self.beta(t)
        return x * torch.sigmoid(x * beta)

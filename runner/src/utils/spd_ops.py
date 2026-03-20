"""AIRM operations on SPD matrices (PyTorch, batched, differentiable)."""

import torch


def airm_inner_product(P, U, V):
    """AIRM inner product: <U, V>_P = tr(P^{-1} U P^{-1} V).

    Uses torch.linalg.solve instead of explicit inverse for numerical stability.

    Args:
        P: (..., d, d) SPD base points
        U: (..., d, d) tangent vectors at P
        V: (..., d, d) tangent vectors at P

    Returns:
        (...,) scalar inner products
    """
    invP_U = torch.linalg.solve(P, U)
    invP_V = torch.linalg.solve(P, V)
    return torch.diagonal(invP_U @ invP_V, dim1=-2, dim2=-1).sum(dim=-1)
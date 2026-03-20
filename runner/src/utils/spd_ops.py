"""AIRM operations on SPD matrices (PyTorch, batched, differentiable)."""

import torch
from spd_learn.functional import get_epsilon


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


def _power_eigh(A, t):
    """Matrix power A^t via eigendecomposition. Compatible with functorch transforms.

    Args:
        A: (..., d, d) SPD matrices
        t: scalar or (...,) exponents

    Returns:
        (..., d, d) A^t
    """
    L, Q = torch.linalg.eigh(A)
    epsilon = get_epsilon(A.dtype, "eigval_clamp")
    L = L.clamp(min=epsilon)
    if isinstance(t, torch.Tensor) and t.ndim >= 1:
        L_pow = torch.pow(L, t[..., None])
    else:
        L_pow = torch.pow(L, t)
    return (Q * L_pow.unsqueeze(-2)) @ Q.mH


def airm_geodesic(A, B, t):
    """AIRM geodesic: γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}.

    Uses only standard torch ops (eigh), so it is compatible with torch.func.jvp.

    Args:
        A: (..., d, d) start SPD matrices
        B: (..., d, d) end SPD matrices
        t: scalar or (...,) interpolation parameter in [0, 1]

    Returns:
        (..., d, d) points on the geodesic
    """
    A_sq = _power_eigh(A, 0.5)
    A_invsq = _power_eigh(A, -0.5)
    mid = A_invsq @ B @ A_invsq
    return A_sq @ _power_eigh(mid, t) @ A_sq

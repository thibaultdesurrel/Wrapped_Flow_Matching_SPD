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


def power_matrix(A, t):
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
    A_sq = power_matrix(A, 0.5)
    A_invsq = power_matrix(A, -0.5)
    mid = A_invsq @ B @ A_invsq
    return A_sq @ power_matrix(mid, t) @ A_sq


def pairwise_distance(X, X2=None):
    """Pairwise Riemannian distance for d×d SPD matrices (GPU-accelerated).

    Loops over rows to keep each eigvalsh batch at size n (not n²),
    avoiding cusolver batch-size limits and O(n² d²) memory.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d, d)
    X2 : torch.Tensor, shape (m, d, d), optional
        If provided, computes cross-distances between X and X2.
        If None, computes pairwise distances within X.

    Returns
    -------
    D : torch.Tensor, shape (n, n) if X2 is None, else (n, m)
    """
    n = X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(X)
    inv_sqrt_eigvals = 1.0 / torch.sqrt(eigvals)
    X_invsqrt = (eigvecs * inv_sqrt_eigvals.unsqueeze(-2)) @ eigvecs.mT

    if X2 is None:
        D = torch.zeros(n, n, dtype=X.dtype, device=X.device)
        for i in range(n - 1):
            Xj = X[i + 1 :]  # (n-i-1, d, d)
            tmp = X_invsqrt[i] @ Xj  # (n-i-1, d, d)
            conj = tmp @ X_invsqrt[i]  # (n-i-1, d, d)
            conj = 0.5 * (conj + conj.mT)  # symmetrise for eigvalsh
            ev = torch.linalg.eigvalsh(conj)  # (n-i-1, d)
            ev = torch.clamp(ev, min=1e-300)
            dists = torch.log(ev).square().sum(dim=-1).sqrt()
            D[i, i + 1 :] = dists
            D[i + 1 :, i] = dists
        return D
    else:
        m = X2.shape[0]
        D = torch.zeros(n, m, dtype=X.dtype, device=X.device)
        for i in range(n):
            tmp = X_invsqrt[i] @ X2  # (m, d, d)
            conj = tmp @ X_invsqrt[i]  # (m, d, d)
            conj = 0.5 * (conj + conj.mT)
            ev = torch.linalg.eigvalsh(conj)  # (m, d)
            ev = torch.clamp(ev, min=1e-300)
            D[i] = torch.log(ev).square().sum(dim=-1).sqrt()
        return D

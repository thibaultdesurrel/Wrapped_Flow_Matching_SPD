import torch
import numpy as np
from sklearn.decomposition import PCA
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann, exp_map_riemann
from pyriemann.utils.tangentspace import upper, unupper


def pga(covs, random_state=None):
    """PGA initialization (CPU, numpy in/out — called once)."""
    mean = mean_riemann(covs)
    all_u = log_map_riemann(covs, mean, False)
    flat_covs = upper(all_u)
    pca = PCA(n_components=3, random_state=random_state)
    covs_embedded = pca.fit_transform(flat_covs)
    covs_embedded = covs_embedded / np.std(covs_embedded[:, 0]) * 1e-4
    return exp_map_riemann(unupper(covs_embedded), np.eye(2))


# ---------------------------------------------------------------------------
# GPU-accelerated pairwise Riemannian distance (general d×d SPD)
# ---------------------------------------------------------------------------

def pairwise_distance(X):
    """Pairwise Riemannian distance for d×d SPD matrices (GPU-accelerated).

    Loops over rows to keep each eigvalsh batch at size n (not n²),
    avoiding cusolver batch-size limits and O(n² d²) memory.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d, d)

    Returns
    -------
    D : torch.Tensor, shape (n, n)
        Symmetric matrix of Riemannian distances.
    """
    n = X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(X)
    inv_sqrt_eigvals = 1.0 / torch.sqrt(eigvals)
    X_invsqrt = (eigvecs * inv_sqrt_eigvals.unsqueeze(-2)) @ eigvecs.mT

    D = torch.zeros(n, n, dtype=X.dtype, device=X.device)
    for i in range(n - 1):
        Xj = X[i + 1 :]                              # (n-i-1, d, d)
        tmp = X_invsqrt[i] @ Xj                      # (n-i-1, d, d)
        conj = tmp @ X_invsqrt[i]                     # (n-i-1, d, d)
        conj = 0.5 * (conj + conj.mT)                # symmetrise for eigvalsh
        ev = torch.linalg.eigvalsh(conj)              # (n-i-1, d)
        ev = torch.clamp(ev, min=1e-300)
        dists = torch.log(ev).square().sum(dim=-1).sqrt()
        D[i, i + 1 :] = dists
        D[i + 1 :, i] = dists
    return D


# ---------------------------------------------------------------------------
# Specialised 2×2 helpers (closed-form, no eigendecomposition calls)
# ---------------------------------------------------------------------------

def pairwise_distance_sq_2x2(Y):
    """Squared pairwise Riemannian distances for 2×2 SPD matrices (GPU-vectorized).

    Parameters
    ----------
    Y : torch.Tensor, shape (n, 2, 2)

    Returns
    -------
    Dsq : torch.Tensor, shape (n, n)
    """
    a = Y[:, 0, 0]
    b = Y[:, 0, 1]
    c = Y[:, 1, 1]
    det = a * c - b * b

    # Trace and det-ratio of Y[i]^{-1} Y[j] for all (i, j)
    tr = (
        c[:, None] * a[None, :]
        - 2.0 * b[:, None] * b[None, :]
        + a[:, None] * c[None, :]
    ) / det[:, None]
    det_ratio = det[None, :] / det[:, None]

    half_tr = 0.5 * tr
    disc = torch.sqrt(torch.clamp(half_tr * half_tr - det_ratio, min=0.0))

    l1 = torch.clamp(half_tr - disc, min=1e-300)
    l2 = torch.clamp(half_tr + disc, min=1e-300)

    Dsq = torch.log(l1).square() + torch.log(l2).square()
    Dsq.fill_diagonal_(0.0)
    return Dsq


def _eigh_2x2_sym(a, b, c):
    """Batch eigendecomposition of 2×2 symmetric matrices [[a,b],[b,c]].

    Returns (l1, l2, vx, vy) where l1 ≤ l2 and the eigenvector columns are
    v1 = (-vy, vx) for l1 and v2 = (vx, vy) for l2.
    """
    ht = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(0.25 * (a - c) ** 2 + b ** 2, min=0.0))
    l1 = ht - disc
    l2 = ht + disc

    vx = b
    vy = l2 - a
    vnorm = torch.sqrt(vx * vx + vy * vy)
    safe = vnorm >= 1e-15
    vx = torch.where(safe, vx / vnorm, torch.zeros_like(vx))
    vy = torch.where(safe, vy / vnorm, torch.ones_like(vy))

    return l1, l2, vx, vy


def _build_2x2_sym(f1, f2, vx, vy):
    """Reconstruct 2×2 symmetric matrices from spectral data.

    M = f1 * v1⊗v1 + f2 * v2⊗v2  with v1=(-vy, vx), v2=(vx, vy).
    Returns (m00, m01, m11).
    """
    m00 = f1 * vy * vy + f2 * vx * vx
    m01 = (f2 - f1) * vx * vy
    m11 = f1 * vx * vx + f2 * vy * vy
    return m00, m01, m11


# ---------------------------------------------------------------------------
# Riemannian gradient (fully vectorized, no loops)
# ---------------------------------------------------------------------------

def riemannian_gradient(Y, P, Q, Dsq):
    """Riemannian gradient of the t-SNE cost for 2×2 SPD matrices.

    Parameters
    ----------
    Y : torch.Tensor, shape (n, 2, 2)
    P, Q, Dsq : torch.Tensor, shape (n, n)

    Returns
    -------
    grad : torch.Tensor, shape (n, 2, 2)
    """
    # --- 1. sqrt / invsqrt of every Y[i] ---
    l1, l2, vx, vy = _eigh_2x2_sym(Y[:, 0, 0], Y[:, 0, 1], Y[:, 1, 1])

    S00, S01, S11 = _build_2x2_sym(torch.sqrt(l1), torch.sqrt(l2), vx, vy)
    IS00, IS01, IS11 = _build_2x2_sym(
        1.0 / torch.sqrt(l1), 1.0 / torch.sqrt(l2), vx, vy
    )

    # --- 2. conjugation: conj[i,j] = IS[i] @ Y[j] @ IS[i]  (all pairs) ---
    yj00 = Y[:, 0, 0]
    yj01 = Y[:, 0, 1]
    yj10 = Y[:, 1, 0]
    yj11 = Y[:, 1, 1]

    # tmp = IS[i] @ Y[j]   (IS is symmetric → IS10 = IS01)
    t00 = IS00[:, None] * yj00[None, :] + IS01[:, None] * yj10[None, :]
    t01 = IS00[:, None] * yj01[None, :] + IS01[:, None] * yj11[None, :]
    t10 = IS01[:, None] * yj00[None, :] + IS11[:, None] * yj10[None, :]
    t11 = IS01[:, None] * yj01[None, :] + IS11[:, None] * yj11[None, :]

    # conj = tmp @ IS[i]
    c00 = t00 * IS00[:, None] + t01 * IS01[:, None]
    c01 = t00 * IS01[:, None] + t01 * IS11[:, None]
    c10 = t10 * IS00[:, None] + t11 * IS01[:, None]
    c11 = t10 * IS01[:, None] + t11 * IS11[:, None]

    # symmetrise
    ca = c00
    cb = 0.5 * (c01 + c10)
    cc = c11

    # --- 3. matrix logarithm of each 2×2 conj[i,j] ---
    el1, el2, evx, evy = _eigh_2x2_sym(ca, cb, cc)
    ll1 = torch.log(torch.clamp(el1, min=1e-300))
    ll2 = torch.log(torch.clamp(el2, min=1e-300))
    lm00, lm01, lm11 = _build_2x2_sym(ll1, ll2, evx, evy)

    # --- 4. parallel-transport: R[i,j] = S[i] @ logm[i,j] @ S[i] ---
    u00 = S00[:, None] * lm00 + S01[:, None] * lm01
    u01 = S00[:, None] * lm01 + S01[:, None] * lm11
    u10 = S01[:, None] * lm00 + S11[:, None] * lm01
    u11 = S01[:, None] * lm01 + S11[:, None] * lm11

    r00 = u00 * S00[:, None] + u01 * S01[:, None]
    r01 = u00 * S01[:, None] + u01 * S11[:, None]
    r10 = u10 * S00[:, None] + u11 * S01[:, None]
    r11 = u10 * S01[:, None] + u11 * S11[:, None]

    # --- 5. weighted sum → Riemannian gradient ---
    coeff = (P - Q) / (1.0 + Dsq)  # (n, n)

    n = Y.shape[0]
    grad = torch.empty((n, 2, 2), dtype=Y.dtype, device=Y.device)
    grad[:, 0, 0] = -4.0 * (coeff * r00).sum(dim=1)
    grad[:, 0, 1] = -4.0 * (coeff * r01).sum(dim=1)
    grad[:, 1, 0] = -4.0 * (coeff * r10).sum(dim=1)
    grad[:, 1, 1] = -4.0 * (coeff * r11).sum(dim=1)
    return grad


# ---------------------------------------------------------------------------
# Retraction & norm on the product SPD manifold
# ---------------------------------------------------------------------------

def retraction(point, tangent_vector):
    """Second-order retraction on SPD(2) (batched)."""
    p_inv_tv = torch.linalg.solve(point, tangent_vector)
    result = point + tangent_vector + tangent_vector @ p_inv_tv / 2
    return 0.5 * (result + result.mT)


def norm_SPD(point, tangent_vector):
    """Riemannian norm √Σᵢ tr(P⁻¹V P⁻¹V) on the product manifold."""
    p_inv_tv = torch.linalg.solve(point, tangent_vector)
    return (p_inv_tv * p_inv_tv.mT).sum().sqrt()

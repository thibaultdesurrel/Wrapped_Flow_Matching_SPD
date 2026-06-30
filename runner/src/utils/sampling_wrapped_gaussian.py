import math

import torch

from spd_learn.functional import exp_map_airm, matrix_sqrt, sym_to_upper, vec_to_sym


def sample_wrapped_gaussian(num, p, mu, Sigma):
    """Sample num points of the wrapped Gaussian with parameters p, mu and Gamma.

    Args:
        num (int): Number of points to generate
        p (PyTorch tensor of size (dim, dim)): point on the SPD manifold
        mu (PyTorch tensor of size (dim*(dim+1)/2)): mean of the multivariate gaussian on the tangent space
        Sigma (PyTorch tensor of size (D, D)): Covariance matrix of the multivariate gaussian on the tangent space
    Returns:
        PyTorch tensor of size (num, dim, dim): The sampled SPD matrices
    """
    # Sample from multivariate normal distribution
    mvn = torch.distributions.MultivariateNormal(mu, Sigma)
    t = mvn.sample((num,))

    # Apply unvect_sigma and exponential map
    mean_sqrt = matrix_sqrt.apply(p)
    samples_ts = mean_sqrt @ vec_to_sym(t) @ mean_sqrt
    return exp_map_airm(p, samples_ts)


def loglikelihood_wrapped_gaussian(sample, p, mu, Sigma):
    """Log-density of the wrapped Gaussian WG(p, mu, Sigma) on the SPD manifold.

    Exact inverse of :func:`sample_wrapped_gaussian`: each ``sample`` is mapped
    back to the tangent space at ``p`` via the AIRM log map, vectorized in the
    same (norm-preserving) basis used for sampling, and scored under the
    multivariate Gaussian N(mu, Sigma). A change-of-variables correction for the
    Riemannian exponential map is then subtracted.

    Args:
        sample: (m, dim, dim) SPD matrices to score.
        p:      (dim, dim) base point of the wrapped Gaussian.
        mu:     (D,) mean of the tangent-space Gaussian, D = dim*(dim+1)/2.
        Sigma:  (D, D) covariance of the tangent-space Gaussian.

    Returns:
        (m,) log-likelihoods.
    """
    p = p.to(sample.device, dtype=sample.dtype)
    mu = mu.to(sample.device, dtype=sample.dtype)
    Sigma = Sigma.to(sample.device, dtype=sample.dtype)

    n = sample.shape[-1]
    D = n * (n + 1) // 2

    # Whiten by p^{-1/2}: X = p^{-1/2} sample p^{-1/2}. Vectorizing log(X) in the
    # norm-preserving basis equals vect_sigma of the AIRM log map at p.
    eigvals_p, eigvecs_p = torch.linalg.eigh(p)
    p_invsqrt = (eigvecs_p * eigvals_p.rsqrt().unsqueeze(-2)) @ eigvecs_p.mH
    X = p_invsqrt @ sample @ p_invsqrt

    w, v = torch.linalg.eigh(X)
    logw = w.clamp_min(1e-10).log()
    logm = (v * logw.unsqueeze(-2)) @ v.mH
    log_vect = sym_to_upper(logm)  # (m, D), matches vec_to_sym used when sampling

    # Multivariate Gaussian log-density in the tangent space.
    centered = log_vect - mu
    sol = torch.linalg.solve(Sigma, centered.unsqueeze(-1)).squeeze(-1)
    quad = -0.5 * torch.sum(centered * sol, dim=-1)
    const_norm = -0.5 * (D * math.log(2.0 * math.pi) + torch.logdet(Sigma))
    log_gauss = const_norm + quad

    # Log-Jacobian of the exp map (AIRM): sum_{i<j} log( sinh(d/2) / (d/2) ),
    # with d = logw_i - logw_j the differences of log-eigenvalues of X.
    idx = torch.triu_indices(n, n, offset=1)
    half = 0.5 * (logw[:, idx[0]] - logw[:, idx[1]])
    sinhc = torch.where(
        half.abs() < 1e-12, torch.ones_like(half), torch.sinh(half) / half
    )
    log_jac = torch.log(sinhc).sum(dim=-1)

    return log_gauss - log_jac

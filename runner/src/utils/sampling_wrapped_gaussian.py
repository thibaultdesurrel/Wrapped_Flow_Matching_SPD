import torch

from spd_learn.functional import exp_map_airm, matrix_sqrt, vec_to_sym


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

import torch
import numpy as np

from runner.src.riemannian_tSNE.utils_R_tSNE import (
    pga,
    pairwise_distance,
    pairwise_distance_sq_2x2,
    riemannian_gradient,
    retraction,
    norm_SPD,
)

from sklearn.manifold._utils import _binary_search_perplexity

import matplotlib.pyplot as plt
import warnings
from time import time


class Riemannian_tSNE:
    """Riemannian version of t-SNE algorithm. Reduces a set of c x c SPD matrices into a set of 2 x 2 SPD matrices.

    Parameters
    ----------
    perplexity : int, default = None
        Perplexity used in the Riemannian t-SNE algorithm. If perplexity = None, it will be set to 0.75*n_matrices
    verbosity : int, default = 1
        Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.
    max_it : int, default = 10000
        Maximum number of iterations used for the Riemannian gradient descent.
    max_time : int, default = 300
        Maximum time on the run time of the Riemannian gradient descent in seconds.
    init : str, default = "pga"
        The initialization method for the embedding. If "random", the embedding is randomly initialized.
    device : str or torch.device, default = None
        Device to run computations on. If None, uses CUDA if available, else CPU.

    Attributes
    ----------
    res_opti : ndarray, shape (n_matrices, 2, 2)
        The result of the optimization process.
    """

    def __init__(
        self,
        perplexity=None,
        verbosity=2,
        max_it=10000,
        max_time=300,
        init="pga",
        seed=None,
        device=None,
        dtype=torch.float64,
    ):
        self.perplexity = perplexity
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.init = init
        self.seed = seed
        self.dtype = dtype
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.loss_evolution = None
        self.res_opti = None
        self.n_iterations = None
        self.runtime = None
        self.final_loss = None

    def compute_similarities(self, X):
        """Computed the high dimensional symmetrized conditional similarities p_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices to reduce.

        Returns
        ----------
        P : torch.Tensor, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        """
        nb_mat = X.shape[0]

        # Pairwise distances on GPU, then transfer to CPU for sklearn binary search
        X_torch = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        D = pairwise_distance(X_torch)
        D_np = D.cpu().numpy().astype(np.float32, copy=False)

        conditional_P = _binary_search_perplexity(D_np, self.perplexity, 0)
        P = conditional_P + conditional_P.T
        P = P / (2 * nb_mat)
        return torch.as_tensor(P, dtype=self.dtype, device=self.device)

    def compute_low_affinities(self, Y):
        """Computed the low dimensional similarities q_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        Y : torch.Tensor, shape (n_matrices, 2, 2)
            Set of SPD matrices.

        Returns
        ----------
        Q : torch.Tensor, shape (n_matrices, n_matrices)
        Dsq : torch.Tensor, shape (n_matrices, n_matrices)
        """
        Dsq = pairwise_distance_sq_2x2(Y)

        inv_dist = 1.0 / (1.0 + Dsq)
        denominator = inv_dist.sum() - inv_dist.diagonal().sum()
        Q = inv_dist / denominator
        Q.fill_diagonal_(0.0)
        return Q, Dsq

    def cost(self, P, Q):
        """Computed the loss of the t-SNE, that is the Kullback-Leibler divergence between P and Q.

        Parameters
        ----------
        P : torch.Tensor, shape (n_matrices, n_matrices)
        Q : torch.Tensor, shape (n_matrices, n_matrices)

        Returns
        ----------
        _ : float
            The cost of the t-SNE.
        """
        eye = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        return (torch.xlogy(P, P + eye) - torch.xlogy(P, Q + eye)).sum()

    def riemannian_grad(self, Y, P, Q, Dsq):
        """Computed the Riemannian gradient of the loss of the t-SNE.

        Parameters
        ----------
        Y : torch.Tensor, shape (n_matrices, 2, 2)
        P : torch.Tensor, shape (n_matrices, n_matrices)
        Q : torch.Tensor, shape (n_matrices, n_matrices)
        Dsq : torch.Tensor, shape (n_matrices, n_matrices)

        Returns
        ----------
        grad : torch.Tensor, shape (n_matrices, 2, 2)
        """
        return riemannian_gradient(Y, P, Q, Dsq)

    def run_minimization(self, P):
        """Run the minimization to solve the t-SNE optimization.

        Parameters
        ----------
        P : torch.Tensor, shape (n_matrices, n_matrices)

        Returns
        ----------
        current_sol : torch.Tensor, shape (n_matrices, 2, 2)
        """
        tol_step = 1e-6
        current_sol = self.initial_point
        self.loss_evolution = []
        initial_time = time()

        for i in range(self.max_it):
            if self.verbosity >= 2 and i % 100 == 0:
                print("Iteration : ", i)

            Q, Dsq = self.compute_low_affinities(current_sol)
            loss = self.cost(P, Q)
            loss_val = loss.item()
            self.loss_evolution.append(loss_val)

            direction = self.riemannian_grad(current_sol, P, Q, Dsq)
            norm_direction = norm_SPD(current_sol, direction)

            # Backtracking line search
            if i == 0:
                alpha = 1.0 / norm_direction.item()
            else:
                alpha = (
                    4
                    * (self.loss_evolution[-2] - loss_val)
                    / (norm_direction.item() ** 2)
                )

            tau = 0.50
            r = 1e-4
            maxiter_linesearch = 25

            retracted = retraction(current_sol, -alpha * direction)
            Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
            loss_retracted = self.cost(P, Q_retracted).item()

            for _ in range(maxiter_linesearch):
                if loss_val - loss_retracted > r * alpha * norm_direction.item() ** 2:
                    break
                alpha = tau * alpha

                retracted = retraction(current_sol, -alpha * direction)
                Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
                loss_retracted = self.cost(P, Q_retracted).item()
            else:
                print("Maximum iteration in linesearched reached.")

            current_sol = retracted

            crit = norm_SPD(current_sol, -alpha * direction).item()
            if crit <= tol_step:
                print("Min stepsize reached")
                break

            if time() - initial_time >= self.max_time:
                warnings.warn("Time limite reached after " + str(i) + " iterations.")
                break

        else:
            warnings.warn("Maximum iterations reached.")
        self.n_iterations = i
        self.final_loss = loss_val
        print(
            f"Optimization done in {time() - initial_time:.2f} seconds and with {i} iterations."
        )
        print(f"Final loss: {loss_val}")
        return current_sol

    def fit(self, X):
        """Fit X to 2x2 SDP matrices using the Riemannian t-SNE algorithm.

        Parameters
        ----------
        X : array_like of shape (n_matrices, n_channels, n_channels)
            Numpy array of SPD matrices.

        Returns
        ----------
        res_opti :  ndarray, shape (n_matrices, 2, 2)
            The solution of the t-SNE problem (numpy array).
        """
        fit_start = time()
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        (n_matrices, _, _) = X.shape

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_matrices)

        # Compute similarities (high-dim distances on GPU, binary search on CPU)
        P = self.compute_similarities(X)

        if self.init == "random":
            sigma = 1e-2
            rng = np.random.default_rng(self.seed)
            L = np.zeros((n_matrices, 2, 2))
            L[:, 0, 0] = np.exp(rng.normal(0, sigma, n_matrices))
            L[:, 1, 0] = rng.normal(0, sigma, n_matrices)
            L[:, 1, 1] = np.exp(rng.normal(0, sigma, n_matrices))
            init_np = L @ np.swapaxes(L, -1, -2)
        elif self.init == "pga":
            init_np = pga(X, random_state=self.seed)
        else:
            raise ValueError("Unknown initialization method: " + str(self.init))

        self.initial_point = torch.as_tensor(
            init_np, dtype=self.dtype, device=self.device
        )

        if self.verbosity >= 1:
            print(f"Optimizing on {self.device}...")
        self.res_opti = self.run_minimization(P)
        self.runtime = time() - fit_start

        # Return numpy for downstream compatibility
        return self.res_opti.cpu().numpy()

    def plot_loss(self):
        """Plot the evolution of the loss during the Riemannian gradient descent in log-log scale."""
        if self.loss_evolution is None:
            raise Exception("You need to fit the T-SNE before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.loss_evolution)
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()

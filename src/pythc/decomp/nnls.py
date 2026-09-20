"""
Non-negative least squares via the Lawson-Hanson active set method.

This module provides the solver underlying the NNLS quadrature grid reweighting of
Hillers-Bendtsen, Lu, Martinez (2026 - DOI: 10.1021/acs.jctc.6c00664). There, the
quadrature weights of a spatial grid are fitted so that numerical integration on the
grid reproduces the AO overlap matrix,

    min_w 1/2 || S_{mu nu} - sum_P phi_mu(r_P) phi_nu(r_P) w_P ||^2,   w >= 0

Non-negativity is required because the weights represent integration volumes, and the
active set of the solver leaves most weights at exactly zero, which prunes the grid as
a side effect of the fit.

Two details drive the implementation:

1. The fitting matrix A (one row per AO pair, one column per grid point) is never
   materialized. Its columns are generated on demand from the collocation matrix and
   the gradient A^T r is evaluated as a contraction over the AO indices, which keeps
   memory at O(n_grid * n_AO) instead of O(n_grid * n_AO^2).
2. The least-squares problem on the passive set is maintained as an incremental QR
   factorization with Givens downdating, rather than being re-solved from scratch in
   every iteration.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

logger = logging.getLogger()


class NNLSOperator(ABC):
    """
    Matrix-free interface to the fitting matrix A of an NNLS problem.

    Only the operations the Lawson-Hanson iteration actually performs are exposed:
    extracting a single column and forming the gradient A^T r.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def column(self, j: int) -> np.ndarray:
        """Column j of A."""
        pass

    @abstractmethod
    def gradient(self, r: np.ndarray) -> np.ndarray:
        """A^T r, the negative gradient of 1/2 ||Aw - b||^2 at the current residual."""
        pass


class DenseOperator(NNLSOperator):
    """Wraps a dense matrix so that any NNLS problem can use the solver."""

    def __init__(self, A: np.ndarray):
        self.A = A

    @property
    def shape(self):
        return self.A.shape

    def column(self, j):
        return self.A[:, j]

    def gradient(self, r):
        return self.A.T @ r


class OverlapFitOperator(NNLSOperator):
    """
    Fitting matrix of the overlap quadrature problem, built from a collocation matrix.

    The rows are indexed by the unique AO pairs (mu <= nu), with off-diagonal pairs
    scaled by sqrt(2) so that the packed norm equals the Frobenius norm of the full
    matrix. Column P is therefore the packed outer product of the AOs evaluated at grid
    point P,

        A[:, P] = T(phi(r_P) phi(r_P)^T)

    which is cheap enough to regenerate on demand, so A is never stored. The gradient
    follows from the same identity without going through the pair index at all:

        (A^T r)_P = phi(r_P)^T U(r) phi(r_P)

    where U(r) is r unpacked into a symmetric matrix. That contraction costs
    O(n_grid * n_AO^2) flops but only needs the O(n_grid * n_AO) collocation matrix.
    """

    def __init__(self, R: np.ndarray):
        """
        :param R: Collocation matrix phi_mu(r_P) of shape (n_grid, n_AO), unweighted.
        """
        self.R = R
        self.n_grid, self.n_ao = R.shape

        self.rows, self.cols = np.triu_indices(self.n_ao)
        self.scale = np.where(self.rows == self.cols, 1.0, np.sqrt(2.0))
        self.n_pair = self.rows.size

    @property
    def shape(self):
        return self.n_pair, self.n_grid

    def pack(self, M: np.ndarray) -> np.ndarray:
        """Pack a symmetric AO matrix into the norm-preserving unique-pair vector."""
        return M[self.rows, self.cols] * self.scale

    def unpack(self, v: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`pack`."""
        M = np.zeros((self.n_ao, self.n_ao))
        M[self.rows, self.cols] = v / self.scale
        return M + M.T - np.diag(np.diag(M))

    def column(self, j):
        phi = self.R[j]
        return phi[self.rows] * phi[self.cols] * self.scale

    def gradient(self, r):
        # (A^T r)_P = phi_P^T U phi_P, contracted over the AO indices so that the
        # n_pair dimension never appears.
        U = self.unpack(r)
        return np.einsum('pm,pm->p', self.R @ U, self.R, optimize=True)


class _IncrementalQR:
    """
    Thin QR factorization of the passive columns, supporting append and delete.

    Appending uses modified Gram-Schmidt with one reorthogonalization pass; deleting a
    column restores the triangular form with Givens rotations. Both keep the cost per
    active set update at O(m * k) instead of the O(m * k^2) of a fresh least-squares
    solve.
    """

    def __init__(self, m: int, max_cols: int):
        self.Q = np.zeros((m, max_cols))
        self.R = np.zeros((max_cols, max_cols))
        self.k = 0

    def append(self, a: np.ndarray) -> bool:
        """
        Append column ``a``. Returns False if it is numerically dependent on the
        columns already present, in which case the factorization is left untouched.
        """
        k = self.k
        Q = self.Q[:, :k]

        u = Q.T @ a
        q = a - Q @ u
        # One reorthogonalization pass (Daniel-Gragg-Kaufman-Stewart): the grid points
        # of a quadrature grid are highly redundant, so a single projection loses
        # orthogonality quickly.
        du = Q.T @ q
        q -= Q @ du
        u += du

        rho = float(np.linalg.norm(q))
        if rho <= 1e-10 * float(np.linalg.norm(a)):
            return False

        self.R[:k, k] = u
        self.R[k, k] = rho
        self.Q[:, k] = q / rho
        self.k += 1

        return True

    def remove(self, j: int):
        """
        Delete column j and restore the upper triangular form of R.

        Deleting a column leaves one subdiagonal entry per column to the right of j,
        which a sweep of Givens rotations removes. The sweep is inherently sequential
        and is the hot loop of the whole solve once the passive set grows into the
        hundreds, so it runs through LAPACK rather than as a Python loop.
        """
        k = self.k

        Q, R = sp.linalg.qr_delete(self.Q[:, :k], self.R[:k, :k], j, which='col',
                                   overwrite_qr=False, check_finite=False)

        self.Q[:, :k - 1] = Q[:, :k - 1]
        self.Q[:, k - 1] = 0.0
        self.R[:k - 1, :k - 1] = R[:k - 1, :k - 1]
        self.R[k - 1, :] = 0.0
        self.R[:, k - 1] = 0.0
        self.k -= 1

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Least-squares solution over the currently held columns."""
        k = self.k
        return sp.linalg.solve_triangular(self.R[:k, :k], self.Q[:, :k].T @ b, lower=False)


def lawson_hanson(op: NNLSOperator | np.ndarray,
                  b: np.ndarray,
                  weight_threshold: float = 1e-4,
                  max_passive: int = None,
                  maxiter: int = None) -> np.ndarray:
    """
    Solve ``argmin_w ||A w - b||^2`` subject to ``w >= 0`` by the Lawson-Hanson active
    set method.

    The iteration stops once every component of the gradient over the active set has
    fallen below ``weight_threshold``, which is the Karush-Kuhn-Tucker condition for the
    NNLS problem and the *weight threshold* of the reference. It is the knob that trades
    the accuracy of the fit against the number of retained points: each outer iteration
    moves at most one variable off its bound, so a looser threshold terminates earlier
    and leaves more weights at exactly zero.

    :param op: Fitting matrix, either an :class:`NNLSOperator` or a dense array.
    :param b: Right-hand side.
    :param weight_threshold: KKT tolerance on the active set gradient.
    :param max_passive: Cap on the number of non-zero weights. Defaults to the rank
        bound of A.
    :param maxiter: Cap on outer iterations. Quadrature grids are highly redundant, so
        the active set churns: a point is often added and dropped again several times
        before the set settles. The default allows for that.
    :return: The solution vector, with the entries never activated left at exactly zero.
    """
    if isinstance(op, np.ndarray):
        op = DenseOperator(op)

    m, n = op.shape

    # The passive set can never exceed the rank of A, so neither can the number of
    # retained grid points.
    rank_bound = min(m, n)
    max_passive = rank_bound if max_passive is None else min(max_passive, rank_bound)
    # Empirically an active set of k points costs roughly 5k iterations to settle on a
    # Becke grid; 20k leaves headroom without letting a pathological case run forever.
    maxiter = 20 * max_passive if maxiter is None else maxiter

    qr = _IncrementalQR(m, max_passive)
    a_passive = np.zeros((m, max_passive))

    passive: list[int] = []
    x_passive = np.zeros(0)
    excluded = np.zeros(n, dtype=bool)

    r = b.astype(np.float64, copy=True)

    logger.info(f"NNLS: {m} equations, {n} variables, weight threshold {weight_threshold:g}, "
                f"at most {max_passive} non-zero weights")

    for it in range(maxiter):
        if len(passive) >= max_passive:
            logger.warning(f"NNLS: hit the cap of {max_passive} non-zero weights before convergence")
            break

        grad = op.gradient(r)
        grad[passive] = -np.inf
        grad[excluded] = -np.inf

        # The residual does not change while we are only rejecting candidates, so keep
        # picking from this gradient rather than recomputing it for each rejection.
        converged = False
        while True:
            j = int(np.argmax(grad))
            if grad[j] <= weight_threshold:
                logger.info(f"NNLS: converged after {it} iterations, "
                            f"max gradient {grad[j]:.3e} <= {weight_threshold:g}")
                converged = True
                break

            a_j = op.column(j)
            if qr.append(a_j):
                break

            # Redundant with the columns already chosen, so it can never enter the
            # passive set: drop it for good instead of picking it again.
            #
            # In exact arithmetic this is unreachable. The least-squares solve leaves
            # the residual orthogonal to every passive column, so anything in their
            # span has gradient exactly zero and is never the argmax. It is a guard
            # against near-dependence at finite precision, where the gradient can sit
            # just above the threshold while the column is numerically dependent - and
            # without it the solver would pick the same point forever.
            excluded[j] = True
            grad[j] = -np.inf

        if converged:
            break

        a_passive[:, len(passive)] = a_j
        passive.append(j)

        s = qr.solve(b)
        # The newly added variable enters at zero, so the feasible point to interpolate
        # from is the previous solution extended by a zero.
        x_full = np.append(x_passive, 0.0)

        while s.size > 0 and s.min() <= 0.0:
            infeasible = s <= 0.0
            alpha = float(np.min(x_full[infeasible] / (x_full[infeasible] - s[infeasible])))
            x_full = x_full + alpha * (s - x_full)

            # Everything that reached the bound leaves the passive set. alpha puts at
            # least one entry there by construction, but rounding can leave it just
            # above the cut-off, so drop the smallest explicitly rather than risk
            # spinning on an unchanged active set.
            dropped = np.flatnonzero(x_full <= 1e-14)
            if dropped.size == 0:
                dropped = np.array([int(np.argmin(x_full))])

            for idx in reversed(dropped.tolist()):
                qr.remove(idx)
                a_passive[:, idx:len(passive) - 1] = a_passive[:, idx + 1:len(passive)]
                del passive[idx]

            x_full = np.delete(x_full, dropped)
            if not passive:
                break

            s = qr.solve(b)

        x_passive = s if passive else np.zeros(0)
        r = b - a_passive[:, :len(passive)] @ x_passive if passive else b.copy()
    else:
        logger.warning(f"NNLS: stopped at the iteration limit of {maxiter}")

    w = np.zeros(n)
    if passive:
        w[passive] = np.maximum(x_passive, 0.0)

    # r is kept in step with x_passive, so it is already the residual of the solution.
    logger.info(f"NNLS: {int(np.count_nonzero(w))} of {n} weights are non-zero, "
                f"residual {float(np.linalg.norm(r)):.3e}")

    return w

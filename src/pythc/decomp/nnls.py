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

:class:`GroupOperator` additionally ties the weights of a group of points together, so
that the same solver selects whole symmetry orbits of the grid instead of individual
points. See :func:`~pythc.grid.octahedral_orbits` for what the groups are and why.

:class:`StackedOperator` goes the other way and ties several *fits* together, so that
one support has to serve a point set in more than one environment at once. That is what
an offline, per-element grid needs: the element is fitted against a range of ghost
neighbours simultaneously rather than in any one of them.

:class:`ERIFitOperator` changes the target rather than the variables: it asks the same
weights to reproduce the two-electron integrals instead of the overlap matrix. That is
still linear in w - the ERI is the integral of a co-density against the electrostatic
potential of another - but it supplies O(n_AO^4) equations where the overlap supplies
O(n_AO^2), which is what lifts the Lawson-Hanson support ceiling off a free-atom fit.
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


class ERIFitOperator(NNLSOperator):
    """
    Fitting matrix of the **ERI** quadrature problem, built from a collocation matrix
    and the electrostatic potentials of the AO co-densities on the same points.

    :class:`OverlapFitOperator` asks a weight vector to reproduce the overlap matrix,

        S_{mu nu} = sum_P w_P phi_mu(r_P) phi_nu(r_P),

    which is one equation per unique AO pair - ``n_AO (n_AO + 1) / 2`` of them. Since
    Lawson-Hanson can never retain more variables than the problem has equations, that
    count is a hard ceiling on the support an overlap fit can select: 15 points for
    hydrogen in cc-pVDZ, whatever the threshold. On an isolated atom, where there is no
    neighbour to add rows, the ceiling binds and the fit fails structurally (see
    ``experiments/atom_centered_grids/ghosts.py``).

    This operator swaps the target for the two-electron integrals, using the exact
    identity

        (mu nu | lambda sigma) = integral dr phi_mu(r) phi_nu(r) V_{lambda sigma}(r),
        V_{lambda sigma}(r) = integral dr' phi_lambda(r') phi_sigma(r') / |r - r'|,

    so that a quadrature rule reproducing every ERI is one satisfying

        (mu nu | lambda sigma) = sum_P w_P phi_mu(r_P) phi_nu(r_P) V_{lambda sigma}(r_P).

    That is still **linear in w**, so the same NNLS solver applies unchanged - but it is
    ``[n_AO (n_AO + 1) / 2]^2`` equations rather than ``n_AO (n_AO + 1) / 2``, i.e.
    ``O(n_AO^4)`` against ``O(n_AO^2)``. Hydrogen goes from 15 equations to 225 and
    carbon from 105 to 11025, which removes the equation-count ceiling outright and
    lets a *free-atom* fit select a support of useful size.

    The target is also physically closer to what LS-THC needs. The overlap target is a
    short-ranged, exponentially decaying quantity; ``V_{lambda sigma}`` falls off as
    ``1/r``, so the ERI target keeps paying attention to the tail region where a bond
    would be - which is exactly what ghost neighbours were introduced to supply.

    Rows are indexed by pairs of AO pairs, each pair packed as in
    :class:`OverlapFitOperator` (unique ``mu <= nu``, off-diagonals scaled by
    ``sqrt(2)``), so the packed 2-norm is the Frobenius norm of the full four-index
    array. Column P is the outer product of the packed co-density and the packed
    potential at ``r_P``, and the gradient contracts the residual against both without
    the pair-pair index ever appearing:

        (A^T r)_P = rho(r_P)^T U(r) v(r_P)

    with ``U(r)`` the residual reshaped to (pair, pair).

    **Rank reduction.** Every column lies in ``range(Rho^T) (x) range(Vp^T)``, so the
    component of the target outside that subspace is a constant of the fit: no weight
    vector can touch it. Projecting both indices onto orthonormal bases of those two
    ranges is therefore an *exact* reduction - the minimiser, the gradient and the
    active set are unchanged - and it shrinks the row count from ``n_pair^2`` to
    ``rank(Rho) * rank(Vp)``. That costs two thin SVDs and is what makes the operator
    usable when a screened environment carries AOs the grid cannot resolve. The part
    dropped is reported by :meth:`dropped_norm_sq` so that a residual can still be
    quoted against the full target.
    """

    def __init__(self, R: np.ndarray, V: np.ndarray, rank_tol: float = 1e-12):
        """
        :param R: Collocation matrix ``phi_mu(r_P)``, shape ``(n_grid, n_AO)``.
        :param V: Co-density potentials ``V_{mu nu}(r_P)``, shape
            ``(n_grid, n_AO, n_AO)`` - PySCF's ``int1e_grids`` evaluated on the same
            points, in the same AO ordering as ``R``.
        :param rank_tol: Relative singular-value cutoff for the exact range projection
            described above. ``0`` keeps every direction and so does nothing; the
            default discards only what is numerically zero. Raising it makes the
            reduction lossy, which is occasionally worth it and never silent - the
            discarded weight shows up in :meth:`dropped_norm_sq`.
        """
        R, V = np.asarray(R), np.asarray(V)
        if V.shape[0] != R.shape[0] or V.shape[1:] != (R.shape[1], R.shape[1]):
            raise ValueError(f"collocation matrix of shape {R.shape} does not go with "
                             f"potentials of shape {V.shape}")

        self.n_grid, self.n_ao = R.shape
        self.rows, self.cols = np.triu_indices(self.n_ao)
        self.scale = np.where(self.rows == self.cols, 1.0, np.sqrt(2.0))
        self.n_pair = self.rows.size

        # Packed co-densities and packed potentials, one row per grid point.
        self.rho = R[:, self.rows] * R[:, self.cols] * self.scale
        self.vpot = V[:, self.rows, self.cols] * self.scale

        self.p_left = _range_basis(self.rho, rank_tol)
        self.p_right = _range_basis(self.vpot, rank_tol)
        self.rho = self.rho @ self.p_left
        self.vpot = self.vpot @ self.p_right
        self.k_left, self.k_right = self.p_left.shape[1], self.p_right.shape[1]

    @property
    def shape(self):
        return self.k_left * self.k_right, self.n_grid

    def pack(self, eri: np.ndarray) -> np.ndarray:
        """
        Pack a four-index ERI array into the right-hand side of this fit.

        :param eri: ``(mu nu | lambda sigma)`` in chemists' notation, shape
            ``(n_AO,) * 4``, in the AO ordering of the collocation matrix.
        """
        return self._project(self._to_pairs(eri)).ravel()

    def dropped_norm_sq(self, eri: np.ndarray) -> float:
        """
        How much of the target the rank reduction put out of reach, as a squared norm.

        ``||b_full||^2 - ||pack(eri)||^2``. At the default ``rank_tol`` this is the
        component no quadrature on these points could have reproduced anyway, so it
        belongs in a quoted residual but not in a judgement of the fit.
        """
        pairs = self._to_pairs(eri)
        return float(np.sum(pairs ** 2) - np.sum(self._project(pairs) ** 2))

    def _to_pairs(self, eri: np.ndarray) -> np.ndarray:
        """Four-index ERI -> the (pair, pair) matrix this fit works in."""
        eri = np.asarray(eri)
        if eri.shape != (self.n_ao,) * 4:
            raise ValueError(f"expected an ERI array of shape {(self.n_ao,) * 4}, "
                             f"got {eri.shape}")
        m = eri[self.rows[:, None], self.cols[:, None], self.rows[None, :], self.cols[None, :]]
        return m * self.scale[:, None] * self.scale[None, :]

    def _project(self, pairs: np.ndarray) -> np.ndarray:
        return self.p_left.T @ pairs @ self.p_right

    def column(self, j):
        return np.outer(self.rho[j], self.vpot[j]).ravel()

    def gradient(self, r):
        # (A^T r)_P = rho_P^T U vpot_P, contracted so that the pair-pair dimension is
        # never materialised per point.
        u = r.reshape(self.k_left, self.k_right)
        return np.einsum('pk,pk->p', self.rho @ u, self.vpot, optimize=True)


def _range_basis(m: np.ndarray, rank_tol: float) -> np.ndarray:
    """
    Orthonormal basis of the row space of ``m``, as a ``(n_cols, rank)`` matrix.

    Directions whose singular value falls below ``rank_tol`` times the largest are
    dropped. With ``rank_tol = 0`` nothing is dropped and the result is square and
    orthogonal, so every operation built on it is an exact change of basis.
    """
    if rank_tol <= 0.0:
        return np.eye(m.shape[1])

    _, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > rank_tol * (s[0] if s.size else 0.0)
    if not keep.any():                      # an all-zero block; keep one direction
        keep[0] = True
    return vt[keep].T


class GroupOperator(NNLSOperator):
    """
    Ties the variables of an NNLS problem into groups that enter and leave the fit
    together, by presenting the group sums of the underlying columns as the variables.

    Given a partition of the columns of A into groups G, this operator is the matrix
    whose columns are ``A_G = sum_{j in G} A[:, j]``. Solving the NNLS problem in the
    group variables is exactly the original problem restricted to weight vectors that
    are constant within a group,

        w_j = v_{G(j)},   v >= 0

    so the solution is automatically **group sparse**: every member of a group is either
    retained with a common weight or dropped, and no group is ever split.

    That is a stronger statement than group sparsity alone, and deliberately so. In the
    grid application the groups are octahedral orbits of a Lebedev shell (see
    :func:`~pythc.grid.octahedral_orbits`), whose points are symmetry-equivalent: a
    common weight is the only assignment that leaves the retained quadrature invariant
    under the point group, which is the property the grouping exists to buy. Allowing
    the weights to vary within an orbit would restore the anisotropy that selecting
    whole orbits was meant to remove.

    The reduction is exact rather than heuristic - the tied problem is itself an NNLS
    problem - so :func:`lawson_hanson` solves it unmodified, with the KKT threshold now
    applied to the group gradient. It also shrinks the problem: a level-0 Becke oxygen
    sub-grid has 858 points but only 58 orbits.
    """

    def __init__(self, base: NNLSOperator | np.ndarray, labels: np.ndarray):
        """
        :param base: Operator (or dense matrix) whose columns are to be grouped.
        :param labels: Group label per column of ``base``. Any labels ``np.unique``
            accepts will do; they are renumbered to ``0 .. n_groups - 1`` in sorted
            order of the label, which is the indexing of the solution vector.
        """
        self.base = DenseOperator(base) if isinstance(base, np.ndarray) else base

        labels = np.asarray(labels)
        if labels.shape != (self.base.shape[1],):
            raise ValueError(f"expected one label per column, got {labels.shape} labels "
                             f"for {self.base.shape[1]} columns")

        _, self.labels = np.unique(labels, return_inverse=True)
        self.labels = self.labels.astype(np.intp).ravel()
        self.n_groups = int(self.labels.max()) + 1 if self.labels.size else 0
        # Members of each group, in column order.
        order = np.argsort(self.labels, kind='stable')
        bounds = np.searchsorted(self.labels[order], np.arange(self.n_groups + 1))
        self.members = [order[bounds[g]:bounds[g + 1]] for g in range(self.n_groups)]

    @property
    def shape(self):
        return self.base.shape[0], self.n_groups

    def column(self, g):
        members = self.members[g]
        a = self.base.column(members[0]).astype(np.float64, copy=True)
        for j in members[1:]:
            a += self.base.column(j)
        return a

    def gradient(self, r):
        return np.bincount(self.labels, weights=self.base.gradient(r),
                           minlength=self.n_groups)

    def expand(self, v: np.ndarray) -> np.ndarray:
        """Spread a group solution back over the underlying columns."""
        return np.asarray(v)[self.labels]


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


class StackedOperator(NNLSOperator):
    """
    Stacks several fitting matrices that share a common set of variables, so that one
    NNLS solve satisfies all of them at once.

    Given operators ``A^(1) .. A^(K)`` with the same number of columns, this is the
    operator

        A = [ c_1 A^(1) ; c_2 A^(2) ; ... ; c_K A^(K) ]

    whose least-squares problem against the correspondingly stacked right-hand side is
    the sum of the individual residuals. The solution is therefore a single weight
    vector - and, because the solver leaves most of it at zero, a single **support** -
    that serves every block.

    The grid application is offline per-element fitting (see
    ``experiments/atom_centered_grids/ghosts.py``): the columns are the points of one
    element's atomic grid, and each block is that element's overlap fit in a different
    ghost environment - a different neighbour, at a different distance, in a different
    direction. A point that only matters when a bond points along ``+z`` has a nonzero
    gradient in the block that puts a ghost there, and so survives a fit that a single
    environment would have pruned it out of.

    The blocks do not have to agree on their row count or even on what their rows mean;
    they only have to agree on the columns. That is what lets each environment screen
    its own AO set and carry its own neighbour basis.

    :param scales: Per-block multipliers ``c_k`` applied to both the block and its share
        of the right-hand side, so the fitted quantity is unchanged and only its weight
        in the objective moves. Rescaling matters here because the blocks are not
        commensurate: an environment with a larger neighbour basis has both more rows
        and a larger target norm, and so would otherwise dominate the fit for reasons
        that have nothing to do with the physics. See :func:`stack_targets`.
    """

    def __init__(self, blocks: list[NNLSOperator | np.ndarray], scales: np.ndarray = None):
        if not blocks:
            raise ValueError("need at least one block to stack")

        blocks = [DenseOperator(b) if isinstance(b, np.ndarray) else b for b in blocks]
        n = blocks[0].shape[1]
        for k, b in enumerate(blocks):
            if b.shape[1] != n:
                raise ValueError(f"block {k} has {b.shape[1]} columns, but block 0 has "
                                 f"{n}; stacked blocks must share their variables")

        self.blocks = blocks
        self.scales = (np.ones(len(blocks)) if scales is None
                       else np.asarray(scales, dtype=float))
        if self.scales.shape != (len(blocks),):
            raise ValueError(f"expected one scale per block, got {self.scales.shape} "
                             f"for {len(blocks)} blocks")

        self.offsets = np.cumsum([0] + [b.shape[0] for b in blocks])
        self.n_var = n

    @property
    def shape(self):
        return int(self.offsets[-1]), self.n_var

    def column(self, j):
        return np.concatenate([c * b.column(j) for c, b in zip(self.scales, self.blocks)])

    def gradient(self, r):
        # A^T r splits over the blocks: each sees only its own slice of the residual.
        g = np.zeros(self.n_var)
        for k, (c, b) in enumerate(zip(self.scales, self.blocks)):
            g += c * b.gradient(r[self.offsets[k]:self.offsets[k + 1]])
        return g


def stack_targets(targets: list[np.ndarray], normalise: bool = True):
    """
    Assemble the right-hand side of a :class:`StackedOperator` and the block scales that
    go with it.

    With ``normalise`` the blocks are put on an equal footing: each is scaled to unit
    target norm and then by ``1/sqrt(K)``, so the stacked right-hand side is a unit
    vector however many blocks there are and whatever their sizes. The objective is then
    the mean *relative* residual over the environments, which is the sense in which a
    ghost-augmented fit should treat them equally.

    Be aware that this rescales the KKT gradient too, so a ``weight_threshold`` for a
    normalised stack is not on the same scale as one for a single unnormalised overlap
    fit; ladders have to be calibrated per mode.

    :return: ``(b, scales)`` for :func:`lawson_hanson` and :class:`StackedOperator`.
    """
    if normalise:
        norms = np.array([max(float(np.linalg.norm(t)), 1e-300) for t in targets])
        scales = 1.0 / (norms * np.sqrt(len(targets)))
    else:
        scales = np.ones(len(targets))

    return np.concatenate([c * t for c, t in zip(scales, targets)]), scales

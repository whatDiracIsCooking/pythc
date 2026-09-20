"""
Tests for the NNLS quadrature grid reweighting of Hillers-Bendtsen, Lu, Martinez
(2026 - DOI: 10.1021/acs.jctc.6c00664).

Two layers. The algebraic tests pin the identities the solver is built on - the packed
pair representation, the matrix-free gradient, and the incremental QR - against direct
evaluations; they need no molecule and run in milliseconds. The grid tests then check
the three claims of the reference on a real system: the fit prunes most of the input
grid, reproduces the AO overlap matrix better than the grid it started from, and
saturates once the input grid is large enough.

Fitting a grid is the expensive part, so each distinct grid is built once in a
module-scoped fixture and shared by every test that needs it.
"""

import os

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2
from scipy.optimize import nnls

from pythc.decomp.nnls import OverlapFitOperator, _IncrementalQR, lawson_hanson
from pythc.grid import BeckeGrid, GridProvider, NNLSGrid, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_nnls import LS_RI_NNLS
from pythc.tracking.experiment_run import ExperimentRun

os.environ["PYTHC_USE_CUDA"] = "False"

WATER = """
H        0.087529        0.023820        0.930805
O        0.657172        0.599414        0.406256
H        0.792448        1.344387        1.004310
"""

AUXBASIS = 'cc-pvdz-ri'


# --------------------------------------------------------------------------------------
# Algebraic identities: no molecule, no grid, microseconds each.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def operator():
    """An OverlapFitOperator over a random collocation matrix, with A built explicitly."""
    rng = np.random.default_rng(0)
    R = rng.normal(size=(30, 7))
    op = OverlapFitOperator(R)
    A = np.stack([np.outer(phi, phi)[op.rows, op.cols] * op.scale for phi in R], axis=1)

    return op, R, A


def test_packing_preserves_the_frobenius_norm(operator):
    """
    The sqrt(2) on off-diagonal pairs is what makes the packed least-squares problem the
    same problem as the full one. Get it wrong and the fit silently optimizes something
    else.
    """
    op, _, _ = operator
    rng = np.random.default_rng(1)
    M = rng.normal(size=(op.n_ao, op.n_ao))
    M = M + M.T

    np.testing.assert_allclose(np.linalg.norm(op.pack(M)), np.linalg.norm(M))
    np.testing.assert_allclose(op.unpack(op.pack(M)), M)


def test_columns_are_the_packed_codensity(operator):
    """Column P of A is the packed outer product of the AOs at grid point P."""
    op, _, A = operator

    for j in (0, 13, 29):
        np.testing.assert_allclose(op.column(j), A[:, j])


def test_gradient_matches_the_explicit_matrix(operator):
    """
    The solver never forms A, and instead contracts the gradient over the AO indices.
    That identity is the load-bearing optimization of the whole module, so check it
    against A^T r directly.
    """
    op, _, A = operator
    rng = np.random.default_rng(2)
    r = rng.normal(size=op.n_pair)

    np.testing.assert_allclose(op.gradient(r), A.T @ r, atol=1e-12)


def test_matrix_free_solve_matches_the_dense_one(operator):
    """The two operator implementations must drive the solver to the same answer."""
    op, _, A = operator
    rng = np.random.default_rng(3)
    b = np.abs(rng.normal(size=op.n_pair))

    np.testing.assert_allclose(lawson_hanson(op, b, weight_threshold=1e-12),
                               lawson_hanson(A, b, weight_threshold=1e-12), atol=1e-10)


@pytest.mark.parametrize("shape", [(40, 25), (25, 40), (60, 60)])
def test_lawson_hanson_matches_scipy(shape):
    """Overdetermined, underdetermined and square, against a reference NNLS."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=shape)
    b = rng.normal(size=shape[0])

    x = lawson_hanson(A, b, weight_threshold=1e-12)
    x_ref, _ = nnls(A, b)

    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, x_ref, atol=1e-10)


def test_incremental_qr_stays_exact_through_updates():
    """
    The passive set churns hard on a redundant grid, so the factorization is appended to
    and deleted from thousands of times per solve, including down to empty and back. Any
    drift shows up as a wrong least-squares solution rather than an error.
    """
    rng = np.random.default_rng(4)
    m, n = 12, 6
    A = rng.normal(size=(m, n))

    qr = _IncrementalQR(m, n)
    for j in range(n):
        assert qr.append(A[:, j].copy())

    cols = list(range(n))
    for drop in (4, 0, 2, 0, 0, 0):
        qr.remove(drop)
        del cols[drop]
        k = qr.k

        np.testing.assert_allclose(qr.Q[:, :k] @ qr.R[:k, :k], A[:, cols], atol=1e-12)
        np.testing.assert_allclose(qr.Q[:, :k].T @ qr.Q[:, :k], np.eye(k), atol=1e-12)

    assert qr.k == 0
    assert qr.append(A[:, 3].copy())


def test_incremental_qr_rejects_dependent_columns():
    """A repeated grid point must not enter the passive set twice."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=8)

    qr = _IncrementalQR(8, 2)
    assert qr.append(a.copy())
    assert not qr.append(2.0 * a)
    assert qr.k == 1


def test_duplicate_grid_points_do_not_disturb_the_solve():
    """
    Real grids contain coincident and near-coincident points. Duplicating one must not
    change the fit: it cannot be selected twice, and it must not cost accuracy either.
    """
    rng = np.random.default_rng(6)
    R = rng.normal(size=(12, 4))
    b = np.abs(rng.normal(size=OverlapFitOperator(R).n_pair))

    clean = OverlapFitOperator(R)
    w_clean = lawson_hanson(clean, b, weight_threshold=1e-10)

    R_dup = np.vstack([R, R[2]])
    dup = OverlapFitOperator(R_dup)
    w_dup = lawson_hanson(dup, b, weight_threshold=1e-10)

    def residual(op, w):
        return np.linalg.norm(b - np.stack([op.column(j) for j in range(op.n_grid)], axis=1) @ w)

    assert np.count_nonzero(w_dup) == np.count_nonzero(w_clean)
    np.testing.assert_allclose(residual(dup, w_dup), residual(clean, w_clean), atol=1e-10)


def test_solver_stops_at_the_iteration_limit():
    """A capped solve returns the best feasible point it reached, not an exception."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(30, 20))
    b = np.abs(rng.normal(size=30))

    w = lawson_hanson(A, b, weight_threshold=1e-12, maxiter=1)

    assert np.all(w >= 0.0)
    assert np.count_nonzero(w) <= 1


# --------------------------------------------------------------------------------------
# Grid behaviour. Each distinct fit is built once and shared.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mol():
    return gto.M(atom=WATER, basis='cc-pvdz')


@pytest.fixture(scope="module")
def overlap(mol):
    return mol.intor('int1e_ovlp_sph')


@pytest.fixture(scope="module")
def becke(mol):
    return BeckeGrid(mol).build()


@pytest.fixture(scope="module")
def fits(mol):
    """Every NNLS grid the tests below need, fitted once each."""
    return {
        'loose': NNLSGrid(mol, weight_threshold=1e-2).build(),
        'default': NNLSGrid(mol, weight_threshold=1e-4).build(),
        'tight': NNLSGrid(mol, weight_threshold=1e-6).build(),
        'tight_l1': NNLSGrid(mol, parent=BeckeGrid(mol, level=1), weight_threshold=1e-6).build(),
        'blocked': NNLSGrid(mol, weight_threshold=1e-4, blocked=True).build(),
        'capped': NNLSGrid(mol, weight_threshold=1e-6, max_points=50).build(),
    }


def rmsd(mol, grid):
    coords, weights = grid
    return _rmsd_overlap(_eval_basefuncs(mol, coords), weights, mol.intor('int1e_ovlp_sph'))


def test_prunes_the_input_grid_to_positive_weights(mol, becke, fits):
    """
    Weights are integration volumes, so non-negativity is physical; the active set
    leaving most of them at exactly zero is what prunes the grid.
    """
    coords, weights = fits['default']

    assert np.all(weights > 0.0)
    assert len(coords) < 0.25 * len(becke[0])


def test_improves_on_the_input_grid(mol, becke, fits):
    """
    At a tight threshold the refitted weights reproduce S better than the input grid.
    A scheme that only selects points cannot do this, and it is the reference's central
    claim against pivoted Cholesky pruning.
    """
    assert rmsd(mol, fits['tight']) < _rmsd_overlap(_eval_basefuncs(mol, becke[0]), becke[1],
                                                    mol.intor('int1e_ovlp_sph'))
    assert len(fits['tight'][0]) < len(becke[0])


def test_threshold_trades_accuracy_for_compactness(mol, fits):
    """The weight threshold is the reference's knob, monotone in both directions."""
    sizes = [len(fits[k][0]) for k in ('loose', 'default', 'tight')]
    errors = [rmsd(mol, fits[k]) for k in ('loose', 'default', 'tight')]

    assert sizes[0] < sizes[1] < sizes[2]
    assert errors[0] > errors[1] > errors[2]


def test_output_size_saturates_in_the_input_grid(mol, fits):
    """
    Growing the input grid buys nothing once the space the fit selects from is
    saturated, so a level 1 input yields essentially the level 0 output.
    """
    small, large = len(fits['tight'][0]), len(fits['tight_l1'][0])

    assert abs(large - small) <= 0.2 * small


def test_max_points_caps_the_grid(fits):
    assert 0 < len(fits['capped'][0]) <= 50


def test_blocked_mode_is_comparable_to_the_global_fit(mol, fits):
    """
    The per-atom fit is an approximation beyond the reference, so it is held to being of
    the same order as the global fit, not to matching it.
    """
    coords, weights = fits['blocked']

    assert np.all(weights > 0.0)
    assert len(coords) < 4 * len(fits['default'][0])
    assert rmsd(mol, fits['blocked']) < 10 * rmsd(mol, fits['default'])


def test_blocked_mode_needs_an_atom_partitioned_parent(mol):
    class FlatGrid(GridProvider):
        def build(self):
            return BeckeGrid(mol).build()

    with pytest.raises(NotImplementedError):
        NNLSGrid(mol, parent=FlatGrid(), blocked=True).build()


@pytest.mark.parametrize("blocked", [False, True])
def test_metrics_reach_an_active_experiment_run(mol, blocked):
    """
    The runner keys benchmark rows off these metric names, so a rename here corrupts
    recorded data silently rather than failing.
    """
    class Recorder:
        def __init__(self):
            self.metrics = {}

        def log_metric(self, key, value):
            self.metrics[key] = value

    recorder = Recorder()
    previous = ExperimentRun._active
    ExperimentRun._active = recorder
    try:
        NNLSGrid(mol, weight_threshold=1e-2, blocked=blocked).build()
    finally:
        ExperimentRun._active = previous

    assert {'grid_points_in', 'grid_points', 'rmsd_S_out', 'sum_weights'} <= set(recorder.metrics)
    assert recorder.metrics['grid_points'] < recorder.metrics['grid_points_in']


def test_grid_names_itself_for_experiment_tracking(mol):
    """THCSCF interpolates the grid's name into run identifiers."""
    grid = NNLSGrid(mol, weight_threshold=1e-4, blocked=True)

    assert str(grid) == "nnls_0.0001_blocked"
    assert "weight_threshold=0.0001" in repr(grid)


def test_the_fit_is_only_paid_for_once(mol):
    """THC builders call build() repeatedly; refitting each time would be ruinous."""
    grid = NNLSGrid(mol, weight_threshold=1e-2)

    assert grid.build() is grid.build()


def test_mp2_on_a_reweighted_grid_matches_the_ri_reference(mol):
    """
    The point of the whole module: a grid this compact still has to give the right
    answer. Exercises LS_RI_NNLS end to end, in the 'ov' mode MP2 requires.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()

    thc = LS_RI_NNLS(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, weight_threshold=1e-4)
    eri = thc.build(mode='ov')

    e_thc = LaplaceRMP2(mol, mf, eri, n_laplace=10).kernel()
    e_ref = DFMP2(mf).kernel()[0]

    assert eri.X.shape[0] < 200
    assert abs(e_thc - e_ref) < 1e-4

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

import itertools
import os

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from pythc.decomp.nnls import (ERIFitOperator, GroupOperator, OverlapFitOperator,
                               StackedOperator, _IncrementalQR, lawson_hanson,
                               stack_targets)
from pythc.grid import (BeckeGrid, GridProvider, NNLSGrid, _eval_basefuncs, _rmsd_overlap,
                        octahedral_orbits)
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
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


# --------------------------------------------------------------------------------------
# Orbit grouping: selecting whole symmetry orbits instead of individual points.
# --------------------------------------------------------------------------------------

# The 48 signed permutation matrices of the octahedral group, as index/sign pairs.
_OCTAHEDRAL = [(perm, signs)
               for perm in itertools.permutations(range(3))
               for signs in itertools.product((1.0, -1.0), repeat=3)]


def test_group_operator_is_the_column_summed_problem():
    """
    Tying variables into groups is not an approximation: it is the original problem
    restricted to weight vectors constant on each group, whose fitting matrix is the
    group-summed one. If that identity slips, the solver is quietly fitting something
    else.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 24))
    # Interleaved rather than contiguous, so a wrong assumption about the members being
    # adjacent would show up.
    labels = np.tile(np.arange(6), 4)
    A_grouped = np.stack([A[:, labels == g].sum(axis=1) for g in range(6)], axis=1)

    op = GroupOperator(A, labels)
    r = rng.normal(size=40)

    assert op.shape == (40, 6)
    np.testing.assert_allclose(np.stack([op.column(g) for g in range(6)], axis=1),
                               A_grouped, atol=1e-13)
    np.testing.assert_allclose(op.gradient(r), A_grouped.T @ r, atol=1e-12)

    b = np.abs(rng.normal(size=40))
    v_ref, _ = nnls(A_grouped, b)
    np.testing.assert_allclose(lawson_hanson(op, b, weight_threshold=1e-12), v_ref,
                               atol=1e-9)


def test_group_operator_expands_to_constant_weights_per_group():
    """Every member of a group is kept with a common weight, or dropped with it."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 18))
    labels = np.repeat(np.arange(6), 3)

    op = GroupOperator(A, labels)
    w = op.expand(lawson_hanson(op, np.abs(rng.normal(size=30)), weight_threshold=1e-12))

    for g in range(6):
        assert len(np.unique(w[labels == g])) == 1


def test_group_operator_rejects_a_mismatched_label_count():
    with pytest.raises(ValueError):
        GroupOperator(np.zeros((4, 5)), np.zeros(4, dtype=int))


# --------------------------------------------------------------------------------------
# The ERI target: the same weights, fitted against O(n_AO^4) equations instead of
# O(n_AO^2). See experiments/atom_centered_grids/atomic_eri.py for what it is for.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eri_operator():
    """An ERIFitOperator over random collocation and potential matrices, with A built
    explicitly. The potentials are symmetrised because real co-density potentials are.
    """
    rng = np.random.default_rng(3)
    R = rng.normal(size=(25, 4))
    V = rng.normal(size=(25, 4, 4))
    V = 0.5 * (V + V.transpose(0, 2, 1))

    op = ERIFitOperator(R, V, rank_tol=0.0)
    A = np.stack([op.column(j) for j in range(R.shape[0])], axis=1)
    return op, R, V, A


def test_eri_columns_are_the_codensity_potential_outer_product(eri_operator):
    """
    Column P is the packed co-density at r_P against the packed potential at r_P. That
    outer-product structure is the whole reason the ERI stays linear in the weights, so
    it is worth pinning against a direct construction rather than against itself.
    """
    op, R, V, A = eri_operator
    for j in (0, 7, 24):
        rho = np.outer(R[j], R[j])[op.rows, op.cols] * op.scale
        v = V[j][op.rows, op.cols] * op.scale
        np.testing.assert_allclose(A[:, j], np.outer(rho, v).ravel(), atol=1e-13)

    assert op.shape == (op.n_pair ** 2, R.shape[0])


def test_eri_packing_preserves_the_frobenius_norm(eri_operator):
    """
    The sqrt(2) now appears on both pair indices, so a four-index array packs to a
    vector of the same 2-norm. Get it wrong and the fit weights some integrals twice
    and others once.
    """
    op, _, _, _ = eri_operator
    rng = np.random.default_rng(4)
    e = rng.normal(size=(op.n_ao,) * 4)
    # An ERI array is symmetric in each pair and under exchanging them.
    e = e + e.transpose(1, 0, 2, 3)
    e = e + e.transpose(0, 1, 3, 2)
    e = e + e.transpose(2, 3, 0, 1)

    np.testing.assert_allclose(np.linalg.norm(op.pack(e)), np.linalg.norm(e))
    assert op.dropped_norm_sq(e) == pytest.approx(0.0, abs=1e-18)


def test_eri_gradient_matches_the_explicit_matrix(eri_operator):
    """A^T r, contracted over the two pair indices rather than through n_pair^2 rows."""
    op, _, _, A = eri_operator
    rng = np.random.default_rng(5)
    r = rng.normal(size=A.shape[0])
    np.testing.assert_allclose(op.gradient(r), A.T @ r, atol=1e-11)


def test_eri_model_is_the_quadrature_it_claims_to_be(eri_operator):
    """
    ``A w`` has to be the four-index array a quadrature with those weights produces -
    that identity is what makes the fit mean anything, and it is the one place a
    packing or ordering slip would hide.
    """
    op, R, V, A = eri_operator
    rng = np.random.default_rng(6)
    w = rng.random(R.shape[0])

    quad = np.einsum('p,pm,pn,pls->mnls', w, R, R, V, optimize=True)
    np.testing.assert_allclose(A @ w, op.pack(quad), atol=1e-11)


def test_eri_rank_reduction_leaves_the_problem_unchanged(eri_operator):
    """
    Every column lies in ``range(Rho^T) x range(Vp^T)``, so projecting both indices onto
    those ranges cannot move the minimiser: the part it discards is a constant of the
    fit. Here the potential side is made deliberately rank-deficient, which is what a
    screened environment does in practice, and the reduced problem must still return the
    same weights.
    """
    _, R, V, _ = eri_operator
    # Collapse the potentials onto two independent point profiles: the row space of the
    # packed potential matrix then has rank 2 rather than n_pair.
    rng = np.random.default_rng(7)
    profiles = rng.normal(size=(2, V.shape[1], V.shape[2]))
    profiles = profiles + profiles.transpose(0, 2, 1)
    mix = rng.normal(size=(V.shape[0], 2))
    V = np.einsum('pk,kmn->pmn', mix, profiles)

    full = ERIFitOperator(R, V, rank_tol=0.0)
    reduced = ERIFitOperator(R, V, rank_tol=1e-10)
    assert reduced.shape[0] < full.shape[0]
    assert reduced.k_right == 2

    w_true = np.zeros(R.shape[0])
    w_true[[2, 11, 19]] = [0.7, 0.2, 1.3]
    quad = np.einsum('p,pm,pn,pls->mnls', w_true, R, R, V, optimize=True)

    for op in (full, reduced):
        np.testing.assert_allclose(
            lawson_hanson(op, op.pack(quad), weight_threshold=1e-13), w_true, atol=1e-9)

    # And the gradients agree exactly, which is the statement that the discarded rows
    # are orthogonal to every column.
    w = rng.random(R.shape[0])
    for op in (full, reduced):
        op._r = op.pack(quad) - np.stack(
            [op.column(j) for j in range(R.shape[0])], axis=1) @ w
    np.testing.assert_allclose(full.gradient(full._r), reduced.gradient(reduced._r),
                               atol=1e-11)


def test_eri_rank_reduction_accounts_for_what_it_dropped(eri_operator):
    """
    A lossy cutoff is allowed, but it must not be silent: the weight it puts out of
    reach is exactly what ``dropped_norm_sq`` reports, so a residual can still be quoted
    against the whole target.
    """
    _, R, V, _ = eri_operator
    rng = np.random.default_rng(8)
    e = rng.normal(size=(R.shape[1],) * 4)
    e = e + e.transpose(2, 3, 0, 1)

    full = ERIFitOperator(R, V, rank_tol=0.0)
    lossy = ERIFitOperator(R, V, rank_tol=0.5)          # absurdly aggressive, on purpose
    assert lossy.shape[0] < full.shape[0]

    np.testing.assert_allclose(
        lossy.dropped_norm_sq(e),
        np.sum(full.pack(e) ** 2) - np.sum(lossy.pack(e) ** 2), atol=1e-10)


def test_eri_operator_rejects_mismatched_inputs(eri_operator):
    op, R, V, _ = eri_operator
    with pytest.raises(ValueError):
        ERIFitOperator(R, V[:, :, :3])
    with pytest.raises(ValueError):
        ERIFitOperator(R, V[:-1])
    with pytest.raises(ValueError):
        op.pack(np.zeros((op.n_ao,) * 3))


def test_eri_quadrature_identity_holds_on_a_real_grid(mol):
    """
    The identity the whole target rests on, on a molecule: a dense Becke grid weighting
    ``phi_mu phi_nu V_{lambda sigma}`` reproduces ``(mu nu | lambda sigma)``. If PySCF's
    ``int1e_grids`` were normalised differently, or ordered differently from
    ``GTOval_sph``, this is where it would show.
    """
    g = gen_grid.Grids(mol)
    g.level = 3
    g.build()

    R = _eval_basefuncs(mol, g.coords)
    V = mol.intor('int1e_grids', grids=g.coords)
    quad = np.einsum('p,pm,pn,pls->mnls', g.weights, R, R, V, optimize=True)
    exact = mol.intor('int2e')

    rel = np.linalg.norm(quad - exact) / np.linalg.norm(exact)
    assert rel < 1e-5


def test_eri_target_lifts_the_free_atom_support_ceiling():
    """
    The result this target exists for, as a regression test.

    Lawson-Hanson cannot retain more points than the fit has equations, and a free
    atom's overlap target has only ``n_AO (n_AO + 1) / 2`` - 15 for hydrogen in cc-pVDZ.
    That ceiling is what makes the free-atom fit of ghosts.py fail structurally, and it
    binds exactly, not approximately. The ERI target supplies ``225`` equations on the
    same atom and the same 392-point grid, and the support it can reach must clear the
    overlap ceiling by a wide margin.
    """
    atom = gto.M(atom="H 0.0 0.0 0.0", basis='cc-pvdz', spin=1, verbose=0)
    g = gen_grid.Grids(atom)
    coords, weights = g.gen_atomic_grids(atom, level=0, prune=treutler_prune)['H'][:2]

    R = _eval_basefuncs(atom, coords)
    V = atom.intor('int1e_grids', grids=coords)
    n_pair = atom.nao_nr() * (atom.nao_nr() + 1) // 2

    ovl = OverlapFitOperator(R)
    b_ovl = ovl.pack((R * weights[:, None]).T @ R)
    n_ovl = np.count_nonzero(lawson_hanson(ovl, b_ovl / np.linalg.norm(b_ovl),
                                           weight_threshold=1e-13))

    eri_op = ERIFitOperator(R, V, rank_tol=1e-10)
    quad = np.einsum('p,pm,pn,pls->mnls', weights, R, R, V, optimize=True)
    b_eri = eri_op.pack(quad)
    w_eri = lawson_hanson(eri_op, b_eri / np.linalg.norm(b_eri), weight_threshold=1e-13)

    assert n_ovl == n_pair == 15               # the ceiling, exactly
    assert eri_op.shape[0] == n_pair ** 2 == 225
    assert np.count_nonzero(w_eri) > 3 * n_ovl

    # And the extra points are not decoration: they reproduce a target the overlap fit
    # could not even express, to seven decades.
    kept = np.flatnonzero(w_eri)
    unit = b_eri / np.linalg.norm(b_eri)
    resid = np.linalg.norm(
        unit - np.stack([eri_op.column(j) for j in kept], axis=1) @ w_eri[kept])
    assert resid < 1e-6


# --------------------------------------------------------------------------------------
# Stacking: one support fitted against several environments at once.
# --------------------------------------------------------------------------------------


def test_stacked_operator_is_the_row_stacked_problem():
    """
    Stacking blocks is the least-squares problem of the vertically concatenated matrix,
    so a solve over the stack has to agree with a solve over that matrix. The blocks
    carry different row counts here because the environments an element is fitted
    against screen their own AO sets and so genuinely differ in size.
    """
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(m, 9)) for m in (12, 5, 20)]
    A = np.vstack(blocks)

    op = StackedOperator(blocks)
    r = rng.normal(size=A.shape[0])

    assert op.shape == A.shape
    np.testing.assert_allclose(np.stack([op.column(j) for j in range(9)], axis=1), A,
                               atol=1e-13)
    np.testing.assert_allclose(op.gradient(r), A.T @ r, atol=1e-12)

    b = np.abs(rng.normal(size=A.shape[0]))
    w_ref, _ = nnls(A, b)
    np.testing.assert_allclose(lawson_hanson(op, b, weight_threshold=1e-12), w_ref,
                               atol=1e-9)


def test_stacked_scales_apply_to_both_sides():
    """
    A per-block scale reweights a block in the objective without changing what it is
    fitting, so a solution that satisfies every block exactly must survive any scaling.
    Here the second block is a thousand times larger than the first, which is roughly
    what a heavy ghost partner does to an overlap target next to a hydrogen one.
    """
    rng = np.random.default_rng(1)
    A1, A2 = rng.random((12, 9)), 1000.0 * rng.random((7, 9))
    w_true = np.zeros(9)
    w_true[[1, 4, 7]] = [0.5, 1.2, 0.3]

    b, scales = stack_targets([A1 @ w_true, A2 @ w_true], normalise=True)

    # Each block normalised to unit target norm, then the stack to a unit vector.
    np.testing.assert_allclose(np.linalg.norm(b), 1.0, atol=1e-12)

    op = StackedOperator([A1, A2], scales)
    np.testing.assert_allclose(lawson_hanson(op, b, weight_threshold=1e-14), w_true,
                               atol=1e-9)


def test_unnormalised_stacking_leaves_the_blocks_alone():
    targets = [np.array([3.0, 4.0]), np.array([1.0])]
    b, scales = stack_targets(targets, normalise=False)

    np.testing.assert_allclose(scales, [1.0, 1.0])
    np.testing.assert_allclose(b, [3.0, 4.0, 1.0])


def test_stacked_operator_rejects_blocks_that_disagree_on_variables():
    """The blocks share their variables - that is the whole point of stacking them."""
    with pytest.raises(ValueError):
        StackedOperator([np.zeros((4, 5)), np.zeros((4, 6))])

    with pytest.raises(ValueError):
        StackedOperator([np.zeros((4, 5))], scales=np.ones(2))

    with pytest.raises(ValueError):
        StackedOperator([])


def test_orbits_of_an_atomic_grid_have_octahedral_sizes(mol):
    """
    A Lebedev grid is a union of octahedral orbits of 6, 8, 12, 24 or 48 points. If the
    labelling produced anything else it has merged or split orbits, and the invariance
    the grouping exists to buy would be gone.
    """
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    coords = atom_grids['O'][0]

    labels = octahedral_orbits(coords)
    sizes = np.bincount(labels)

    assert set(sizes.tolist()) <= {6, 8, 12, 24, 48}
    # Each orbit lies on one radial shell.
    radii = np.linalg.norm(coords, axis=1)
    for label in np.unique(labels):
        shell = radii[labels == label]
        assert np.ptp(shell) <= 1e-9 * shell.mean()


def test_orbit_labels_are_invariant_under_the_octahedral_group(mol):
    """
    The defining property: applying any signed axis permutation to the grid permutes
    the points within their orbits and leaves the labelling unchanged.
    """
    g = gen_grid.Grids(mol)
    coords = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)['O'][0]
    labels = octahedral_orbits(coords)

    for perm, signs in _OCTAHEDRAL:
        rotated = coords[:, perm] * np.array(signs)
        np.testing.assert_array_equal(octahedral_orbits(rotated), labels)


def test_a_general_rotation_destroys_the_orbit_structure(mol):
    """
    The counterpart of the test above: the labelling is not vacuously constant. A
    rotation outside the group scrambles the keys, so equality there would mean the
    function is not looking at the angular structure at all.
    """
    g = gen_grid.Grids(mol)
    coords = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)['O'][0]

    tilted = coords @ Rotation.from_rotvec([0.3, -0.4, 0.2]).as_matrix().T

    assert len(np.unique(octahedral_orbits(tilted))) > len(np.unique(octahedral_orbits(coords)))


@pytest.mark.parametrize("blocked", [False, True])
def test_orbit_grouped_fits_keep_whole_orbits(mol, blocked):
    """
    End to end through ``NNLSGrid``: every retained point belongs to an orbit that was
    retained entire, and carries that orbit's common weight. This is the property that
    makes each atom's grid invariant under the octahedral group about its own nucleus.
    """
    coords, weights = NNLSGrid(mol, weight_threshold=1e-3, blocked=blocked,
                               group_orbits=True).build()

    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)

    matched = 0
    for ia in range(mol.natm):
        parent = atom_grids[mol.atom_symbol(ia)][0]
        labels = octahedral_orbits(parent)

        # Map this atom's retained points back onto its parent sub-grid.
        distance = np.linalg.norm(coords[:, None, :] - (parent + mol.atom_coord(ia))[None, :, :],
                                  axis=2)
        mine = distance.min(axis=1) < 1e-9
        onto = distance[mine].argmin(axis=1)
        matched += int(mine.sum())

        for label in np.unique(labels[onto]):
            assert np.count_nonzero(labels[onto] == label) == np.count_nonzero(labels == label)
            assert len(np.unique(weights[mine][labels[onto] == label])) == 1

    assert matched == len(coords)


def test_orbit_grouping_refuses_a_point_cap(mol):
    """``max_points`` counts points, which an orbit-grouped fit cannot honour."""
    with pytest.raises(ValueError):
        NNLSGrid(mol, group_orbits=True, max_points=100)


def test_orbit_grouped_grids_still_reproduce_the_ri_reference(mol):
    """
    The grouping is a constraint on the fit, so it has to be checked that it does not
    cost the energy: a grid that is invariant but wrong is no use.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()

    grid = NNLSGrid(mol, weight_threshold=1e-3, blocked=True, group_orbits=True)
    thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, grid=grid)

    e_thc = LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel()

    assert abs(e_thc - DFMP2(mf).kernel()[0]) < 1e-4


def test_orbit_grouped_grid_names_itself(mol):
    grid = NNLSGrid(mol, weight_threshold=1e-4, blocked=True, group_orbits=True)

    assert str(grid) == "nnls_0.0001_blocked_orbits"
    assert "group_orbits=True" in repr(grid)

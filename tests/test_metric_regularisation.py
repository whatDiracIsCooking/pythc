"""
Tests for the two ways of regularising a rank-deficient metric.

LS-THC inverts a metric that is always rank deficient. The package does that by
truncating eigenvalues below a relative cutoff, which is accurate but *discrete*: the
number of surviving eigenvalues is an integer function of the matrix, so a matrix that
varies smoothly - as the LS-THC metric does when nuclei move - produces an inverse that
jumps. `lib.ridge_inv` offers the Tikhonov alternative, which damps the same directions
without dropping any.

The algebraic tests below pin the identities that make ridge a drop-in replacement, and
the continuity test pins the property it exists for, by driving an eigenvalue through
the cutoff and comparing the two. Everything here is a few small matrices and runs in
milliseconds; a single end-to-end check at the bottom confirms the option is actually
wired into the THC build.
"""

import os

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc import lib
from pythc.grid import NNLSGrid
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_thc_funcs import invert_metric

os.environ["PYTHC_USE_CUDA"] = "False"

WATER = """
H        0.087529        0.023820        0.930805
O        0.657172        0.599414        0.406256
H        0.792448        1.344387        1.004310
"""

AUXBASIS = 'cc-pvdz-ri'


@pytest.fixture(scope="module")
def spd():
    """A well-conditioned symmetric positive definite matrix."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 9))

    return A @ A.T + np.eye(9)


def sweeping_metric(t, size=8):
    """
    A matrix whose smallest eigenvalue passes through ``pinv``'s cutoff at ``t = 1``.

    Fixed eigenvectors and a single moving eigenvalue is the cleanest model of what
    happens to the LS-THC metric along a bond scan, and it isolates the cutoff crossing
    from everything else that changes with geometry.
    """
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.normal(size=(size, size)))[0]
    eig_vals = 10.0 ** -np.arange(size, dtype=float)
    eig_vals[-1] = 10.0 ** (-9.0 - t)

    return (Q * eig_vals) @ Q.T


def largest_steps(inverse, ts):
    """Largest change in the inverse between neighbouring points of a sweep."""
    values = [inverse(sweeping_metric(t)) for t in ts]

    return np.array([np.abs(b - a).max() for a, b in zip(values, values[1:])])


# --------------------------------------------------------------------------------------
# Ridge as an inverse: the identities that make it a drop-in replacement.
# --------------------------------------------------------------------------------------

def test_ridge_recovers_the_exact_inverse_as_lambda_vanishes(spd):
    """With nothing to regularise, ridge must not be doing anything."""
    np.testing.assert_allclose(lib.ridge_inv(spd, 1e-14), np.linalg.inv(spd), atol=1e-10)


def test_ridge_applies_the_shift_it_advertises(spd):
    """The result is the inverse of the shifted matrix, not an approximation to it."""
    shift = lib.ridge_shift(spd, 1e-3, "trace")

    np.testing.assert_allclose(lib.ridge_inv(spd, 1e-3),
                               np.linalg.inv(spd + shift * np.eye(spd.shape[0])),
                               atol=1e-12)


@pytest.mark.parametrize("scale", ["trace", "max_eig"])
def test_ridge_strength_is_dimensionless(spd, scale):
    """
    ``lam`` must mean the same thing whatever the metric is scaled by.

    The LS-THC metric inherits whatever scaling the collocation matrix carries - the
    weights alone move it by orders of magnitude - so an absolute shift would silently
    become a different regularisation for every grid.
    """
    c = 137.0

    np.testing.assert_allclose(lib.ridge_inv(c * spd, 1e-4, scale) * c,
                               lib.ridge_inv(spd, 1e-4, scale), rtol=1e-10)


def test_ridge_inv_sqrt_squares_to_the_ridge_inverse(spd):
    """The two entry points must regularise identically, or the pipeline is inconsistent."""
    root = lib.ridge_inv_sqrt(spd, 1e-4)

    np.testing.assert_allclose(root @ root, lib.ridge_inv(spd, 1e-4), atol=1e-12)


def test_ridge_survives_an_exactly_singular_matrix():
    """
    A metric with exactly null directions is the normal case, not an edge case: the
    co-density rank caps it at n_occ * n_vir however many points the grid has.
    """
    rng = np.random.default_rng(2)
    A = rng.normal(size=(10, 4))
    singular = A @ A.T

    inv = lib.ridge_inv(singular, 1e-6)

    assert np.all(np.isfinite(inv))
    np.testing.assert_allclose(inv, inv.T, atol=1e-12)


def test_ridge_leaves_its_argument_alone(spd):
    """Unlike ``pinv``, which zeroes small entries of the caller's array in place."""
    before = spd.copy()
    lib.ridge_inv(spd, 1e-6)

    np.testing.assert_array_equal(spd, before)


def test_unknown_ridge_scale_is_rejected(spd):
    with pytest.raises(ValueError, match="unknown ridge scale"):
        lib.ridge_shift(spd, 1e-6, "relative")


# --------------------------------------------------------------------------------------
# The property ridge exists for.
# --------------------------------------------------------------------------------------

def test_truncation_jumps_where_ridge_stays_continuous():
    """
    The whole point. As the smallest eigenvalue crosses the cutoff, the truncated
    pseudoinverse gains a rank-one term of size 1/lambda out of nowhere, so its output
    is discontinuous in the matrix - and therefore in the geometry. Ridge, damping the
    same direction rather than dropping it, moves by an ordinary finite step.
    """
    ts = np.linspace(0.5, 1.5, 201)

    # pinv zeroes small entries of its argument in place, so hand it a copy.
    truncated = largest_steps(lambda M: lib.pinv(M.copy()), ts)
    ridged = largest_steps(lambda M: lib.ridge_inv(M, 1e-6, "max_eig"), ts)

    assert truncated.max() > 1e8
    assert truncated.max() / np.median(truncated) > 100
    assert ridged.max() / np.median(ridged) < 10


def test_invert_metric_defaults_to_the_truncated_pseudoinverse(spd):
    """The existing path must be untouched unless a ridge is asked for explicitly."""
    np.testing.assert_array_equal(invert_metric(spd.copy()), lib.pinv(spd.copy()))
    np.testing.assert_array_equal(invert_metric(spd.copy(), ridge=1e-6),
                                  lib.ridge_inv(spd, 1e-6))


# --------------------------------------------------------------------------------------
# End to end: the option reaches the fit, and costs about what it should.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def water():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()

    return mol, mf, DFMP2(mf).kernel()[0], NNLSGrid(mol, weight_threshold=1e-2, blocked=True)


def thc_mp2(mol, mf, grid, **kwargs):
    thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, grid=grid, **kwargs)

    return float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())


def test_ridge_costs_little_accuracy_in_a_real_fit(water):
    """
    Swapping the regulariser must not move the energy much, or it would be trading one
    problem for another. The tolerance here is well inside this deliberately coarse
    grid's own error of some 38 uHa.
    """
    mol, mf, ref, grid = water

    truncated = thc_mp2(mol, mf, grid)
    ridged = thc_mp2(mol, mf, grid, metric_ridge=1e-8, aux_ridge=1e-8)

    assert abs(truncated - ref) < 1e-4
    assert abs(ridged - truncated) < 2e-5

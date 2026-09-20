"""
Tests for the NNLS quadrature grid reweighting of Hillers-Bendtsen, Lu, Martinez
(2026 - DOI: 10.1021/acs.jctc.6c00664).

The assertions track the three claims of the reference: the fit prunes most of the input
grid, it reproduces the AO overlap matrix better than the input grid it started from, and
the size of the output grid saturates once the input grid is large enough.
"""

import os

import numpy as np
import pytest
from pyscf import gto
from scipy.optimize import nnls

from pythc.decomp.nnls import lawson_hanson
from pythc.grid import BeckeGrid, GridProvider, NNLSGrid, _rmsd_overlap
from pythc.thc.ls_thc_funcs import eval_basefuncs

os.environ["PYTHC_USE_CUDA"] = "False"

WATER = """
H        0.087529        0.023820        0.930805
O        0.657172        0.599414        0.406256
H        0.792448        1.344387        1.004310
"""


@pytest.fixture(scope="module")
def mol():
    return gto.M(atom=WATER, basis='cc-pvdz')


def overlap_rmsd(mol, coords, weights):
    return _rmsd_overlap(eval_basefuncs(mol, coords), weights, mol.intor('int1e_ovlp_sph'))


@pytest.mark.parametrize("shape", [(40, 25), (25, 40), (60, 60)])
def test_lawson_hanson_matches_scipy(shape):
    """Our active set solver has to agree with a reference NNLS implementation."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=shape)
    b = rng.normal(size=shape[0])

    x = lawson_hanson(A, b, weight_threshold=1e-12)
    x_ref, _ = nnls(A, b)

    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, x_ref, atol=1e-10)


def test_weights_are_non_negative(mol):
    """Weights are integration volumes, so the constraint is physical, not cosmetic."""
    _, weights = NNLSGrid(mol, weight_threshold=1e-4).build()

    assert len(weights) > 0
    assert np.all(weights > 0.0)


def test_prunes_most_of_the_input_grid(mol):
    """The active set leaves the great majority of the weights at exactly zero."""
    coords_in, _ = BeckeGrid(mol).build()
    coords_out, _ = NNLSGrid(mol, weight_threshold=1e-4).build()

    assert len(coords_out) < 0.25 * len(coords_in)


def test_improves_on_the_input_grid(mol):
    """
    At a tight threshold the refitted weights reproduce S better than the input grid,
    which is what a grid-point selection scheme alone cannot do.
    """
    coords_in, weights_in = BeckeGrid(mol).build()
    coords_out, weights_out = NNLSGrid(mol, weight_threshold=1e-6).build()

    rmsd_in = overlap_rmsd(mol, coords_in, weights_in)
    rmsd_out = overlap_rmsd(mol, coords_out, weights_out)

    assert rmsd_out < rmsd_in
    assert len(coords_out) < len(coords_in)


def test_looser_threshold_gives_a_smaller_grid(mol):
    """The weight threshold is the knob trading accuracy against compactness."""
    sizes = [len(NNLSGrid(mol, weight_threshold=t).build()[0]) for t in (1e-2, 1e-4, 1e-6)]

    assert sizes[0] < sizes[1] < sizes[2]


def test_output_size_saturates_in_the_input_grid(mol):
    """
    Growing the input grid stops buying anything once the space the fit selects from is
    saturated, so a level 1 input yields essentially the level 0 output.
    """
    small, _ = NNLSGrid(mol, parent=BeckeGrid(mol, level=0), weight_threshold=1e-6).build()
    large, _ = NNLSGrid(mol, parent=BeckeGrid(mol, level=1), weight_threshold=1e-6).build()

    assert len(BeckeGrid(mol, level=1).build()[0]) > 4 * len(BeckeGrid(mol, level=0).build()[0])
    assert abs(len(large) - len(small)) <= 0.2 * len(small)


def test_blocked_mode_is_comparable_to_the_global_fit(mol):
    """
    The per-atom fit is an approximation beyond the reference, so it is only held to
    being of the same order as the global fit, not to matching it.
    """
    coords_g, weights_g = NNLSGrid(mol, weight_threshold=1e-4).build()
    coords_b, weights_b = NNLSGrid(mol, weight_threshold=1e-4, blocked=True).build()

    assert np.all(weights_b > 0.0)
    assert len(coords_b) < 4 * len(coords_g)
    assert overlap_rmsd(mol, coords_b, weights_b) < 10 * overlap_rmsd(mol, coords_g, weights_g)


def test_blocked_mode_needs_an_atom_partitioned_parent(mol):
    class FlatGrid(GridProvider):
        def build(self):
            return BeckeGrid(mol).build()

    with pytest.raises(NotImplementedError):
        NNLSGrid(mol, parent=FlatGrid(), blocked=True).build()

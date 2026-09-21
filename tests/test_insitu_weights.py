"""
Tests for the in-molecule per-element weight fit of
``experiments/atom_centered_grids/insitu.py`` (HANDOFF.md step 8).

Three layers.

The *structural* tests pin what makes the object transferable at all: that a fit returns
indices into the element grid and nothing else, that the same element gets byte-identical
points and weights in every molecule, and that none of it moves when the nuclei do. That
last one is the whole gradient argument - ``dw/dR = 0`` holds because the weights are a
constant of the geometry, and a regression that made them depend on the molecule would
break the analytic gradient silently rather than loudly.

The *target* tests check that the thing being fitted is the thing intended: the per-atom
Becke shares of a real molecule, which sum back to its overlap matrix.

The *result* tests are the experiment's actual claim, reduced to the cheapest form that
still carries it - that in-molecule weights beat both ghost footings on the in-molecule
objective, and that the gain survives on a molecule no fit ever saw. They are the ones
that would fail if the idea were wrong rather than merely mis-coded.

Every fit here runs at a loose threshold on small molecules; the module takes well under
a minute and needs no SCF.
"""

import os
import sys

import numpy as np
import pytest

os.environ["PYTHC_USE_CUDA"] = "False"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments", "atom_centered_grids"))

insitu = pytest.importorskip("insitu")

from insitu import (atom_blocks, build_mol, build_supports, element_grid,
                    fit_element_insitu, matched_size_report, per_atom_for_mode,
                    per_atom_from_supports, stacked_residual)

# Loose enough that a ghost fit is a fraction of a second per element, tight enough that
# the supports are large enough for the residual comparison to mean something.
THRESHOLD = 1e-3

TRAIN = ('water', 'methanol')
HELD_OUT = 'formaldehyde'


@pytest.fixture(scope="module")
def cache():
    """One support cache for the module: every mode reuses the same ghost fits."""
    return {}


@pytest.fixture(scope="module")
def supports(cache):
    """The three transferable weight footings on one shared support, per element."""
    elements = ['H', 'C', 'O']
    return {mode: build_supports(elements, mode, THRESHOLD, TRAIN, cache)
            for mode in ('ghost', 'molw', 'molfit')}


# --------------------------------------------------------------------------------------
# The target: per-atom Becke shares of a real molecule.
# --------------------------------------------------------------------------------------

def test_blocks_cover_every_atom_of_the_element_and_nothing_else():
    blocks, targets, labels = atom_blocks('H', ['water', 'methanol'])

    # water has 2 hydrogens, methanol 4.
    assert len(blocks) == len(targets) == len(labels) == 6
    assert [lab.split(':')[0] for lab in labels] == ['water'] * 2 + ['methanol'] * 4
    for name, ia in (lab.split(':') for lab in labels):
        assert build_mol(name).atom_symbol(int(ia)) == 'H'


def test_targets_are_the_becke_shares_and_sum_to_the_overlap_matrix():
    """The identity the blocked fit rests on: ``sum_A S_A = S``.

    It is what makes a union of per-atom fits a fit of the whole molecule, and so what
    lets a per-element weight set be assembled into a molecular quadrature at all. The
    tolerance is the level-0 grid's own quadrature error, not a fitting error.
    """
    mol = build_mol('water')
    S = mol.intor('int1e_ovlp_sph')

    total = np.zeros_like(S)
    for element in ('H', 'O'):
        blocks, targets, _ = atom_blocks(element, ['water'], screening=1e-14)
        for op, target in zip(blocks, targets):
            assert op.n_ao == S.shape[0], "screening dropped an AO at 1e-14"
            total += op.unpack(target)

    assert np.linalg.norm(total - S) / np.linalg.norm(S) < 0.02


def test_block_columns_are_the_packed_codensity_at_the_support_points():
    """Column j is the outer product of the AOs at support point j, packed."""
    support = np.array([3, 17, 100, 250])
    blocks, _, _ = atom_blocks('O', ['water'], support=support)
    op = blocks[0]

    assert op.shape[1] == len(support)

    mol = build_mol('water')
    from pythc.grid import _eval_basefuncs
    ia = next(i for i in range(mol.natm) if mol.atom_symbol(i) == 'O')
    R = _eval_basefuncs(mol, element_grid('O')[support] + mol.atom_coord(ia))
    # The block screens its AOs, so compare on the columns it kept.
    assert R.shape[1] >= op.n_ao

    for j in range(len(support)):
        phi = R[j, :op.n_ao] if R.shape[1] == op.n_ao else None
        if phi is None:
            pytest.skip("screening removed AOs; the direct comparison needs the full set")
        expected = np.outer(phi, phi)[op.rows, op.cols] * op.scale
        assert np.allclose(op.column(j), expected)


def test_a_subgrid_that_is_not_the_element_grid_is_refused(monkeypatch):
    """A support is an index list, so a block built on different points is meaningless.

    Without this check a mismatch would not raise - it would quietly attach the weights
    to the wrong points at runtime, which is the one failure mode of an index-list grid
    that no energy would obviously reveal.
    """
    truncated = np.asarray(element_grid('O'))[:-1]
    monkeypatch.setattr(insitu, 'element_grid', lambda symbol: truncated)

    with pytest.raises(RuntimeError, match="not the element grid"):
        atom_blocks('O', ['water'])


def test_fitting_an_element_that_is_absent_is_an_error():
    with pytest.raises(ValueError, match="no atom of element"):
        fit_element_insitu('C', ['water'])


# --------------------------------------------------------------------------------------
# The product: an index list and a weight per element, and nothing else.
# --------------------------------------------------------------------------------------

def test_held_support_is_never_widened_by_the_weight_fit(supports):
    """`molw` may zero a point NNLS cannot use, but it can never invent one."""
    for element in ('H', 'C', 'O'):
        ghost_idx = supports['ghost'][element][0]
        molw_idx = supports['molw'][element][0]

        assert set(molw_idx.tolist()) <= set(ghost_idx.tolist())
        assert len(molw_idx) > 0


def test_a_free_fit_selects_from_the_whole_element_grid(supports):
    """Without a held support the same solve selects as well, out of every parent point."""
    for element in ('H', 'C', 'O'):
        idx = supports['molfit'][element][0]
        assert idx.min() >= 0
        assert idx.max() < len(element_grid(element))
        assert len(np.unique(idx)) == len(idx)


def test_weights_are_strictly_positive_on_the_retained_points(supports):
    """Non-negativity is the constraint; a retained point with w = 0 would be a bug.

    ``X = w^(1/4) phi`` needs ``w >= 0`` to be real at all, and a zero weight is a point
    that is not there - so the index list and the weight list have to agree on the
    support exactly.
    """
    for mode in ('ghost', 'molw', 'molfit'):
        for element in ('H', 'C', 'O'):
            idx, w = supports[mode][element]
            assert len(idx) == len(w)
            assert np.all(w > 0.0)


def test_the_same_element_gets_the_same_grid_in_every_molecule(supports):
    """The transferability invariant, stated as an equality rather than an argument.

    Every carbon in every molecule must receive byte-identical nucleus-relative points
    and byte-identical weights. If this ever fails the object is not per-element and the
    scheme has silently become a per-molecule fit.
    """
    per_atom = {name: per_atom_from_supports(build_mol(name), supports['molw'])
                for name in ('methanol', 'formaldehyde', 'propene')}

    carbons = [(name, ia) for name in per_atom
               for ia in range(build_mol(name).natm)
               if build_mol(name).atom_symbol(ia) == 'C']
    assert len(carbons) >= 3

    (n0, i0) = carbons[0]
    rel0, w0 = per_atom[n0][i0]
    for name, ia in carbons[1:]:
        rel, w = per_atom[name][ia]
        assert np.array_equal(rel, rel0)
        assert np.array_equal(w, w0)


def test_the_grid_does_not_move_when_the_nuclei_do(cache):
    """``dw/dR = 0`` and ``d(rel)/dR = 0``: the frozen-grid gradient argument, tested.

    A transferable grid's only geometry dependence is the ``+ atom_coord`` translation.
    `blocked` is included as the contrast - it re-fits at every geometry and so *does*
    move, which is exactly why it has no analytic gradient of this kind.
    """
    mol = build_mol('water')
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() + np.array([0.0, 0.0, 0.03]), unit="Bohr")
    moved.build(False, False)

    for mode in ('ghost', 'ghostw', 'molw', 'molfit', 'molfit1'):
        before = per_atom_for_mode(mol, mode, THRESHOLD, TRAIN, cache)
        after = per_atom_for_mode(moved, mode, THRESHOLD, TRAIN, cache)
        for (rel_b, w_b), (rel_a, w_a) in zip(before, after):
            assert np.array_equal(rel_b, rel_a), f"{mode} moved its points"
            assert np.array_equal(w_b, w_a), f"{mode} moved its weights"

    blocked_before = per_atom_for_mode(mol, 'blocked', THRESHOLD, TRAIN, cache)
    blocked_after = per_atom_for_mode(moved, 'blocked', THRESHOLD, TRAIN, cache)
    assert any(not np.array_equal(b[0], a[0])
               for b, a in zip(blocked_before, blocked_after)), \
        "blocked is supposed to re-fit with the geometry; if it does not, the contrast " \
        "this test draws is vacuous"


def test_the_two_footings_of_one_fit_share_a_support(cache):
    """`molfit1` is `molfit` read at ``w = 1``: same points, different weights."""
    mol = build_mol('methanol')
    weighted = per_atom_for_mode(mol, 'molfit', THRESHOLD, TRAIN, cache)
    ones = per_atom_for_mode(mol, 'molfit1', THRESHOLD, TRAIN, cache)

    for (rel_w, w), (rel_1, o) in zip(weighted, ones):
        assert np.array_equal(rel_w, rel_1)
        assert np.array_equal(o, np.ones(len(o)))
        assert not np.array_equal(w, o)


def test_the_w_equals_one_modes_really_are_w_equals_one(cache):
    mol = build_mol('water')
    for mode in ('ghost', 'molfit1'):
        for _rel, w in per_atom_for_mode(mol, mode, THRESHOLD, TRAIN, cache):
            assert np.array_equal(w, np.ones(len(w)))

    for mode in ('ghostw', 'molw', 'molfit'):
        assert any(not np.array_equal(w, np.ones(len(w)))
                   for _rel, w in per_atom_for_mode(mol, mode, THRESHOLD, TRAIN, cache))


# --------------------------------------------------------------------------------------
# The result: does fitting against the runtime's own objective actually buy anything?
# --------------------------------------------------------------------------------------

def test_in_molecule_weights_beat_both_ghost_footings_on_the_training_objective(supports):
    """Step (8)'s claim, at matched support and without an SCF.

    All three rows carry the *same* ghost-selected points, so the only difference is the
    weight footing: ``w = 1``, the weights the ghost fit produced, and the weights fitted
    against real atoms in real molecules.
    """
    for element in ('H', 'C', 'O'):
        ghost_idx, ghost_w = supports['ghost'][element]
        molw_idx, molw_w = supports['molw'][element]

        r_ones = stacked_residual(element, TRAIN, ghost_idx, np.ones(len(ghost_idx)))
        r_ghost = stacked_residual(element, TRAIN, ghost_idx, ghost_w)
        r_molw = stacked_residual(element, TRAIN, molw_idx, molw_w)

        assert r_molw < r_ghost < r_ones, (
            f"{element}: molw {r_molw:.3e}, ghostw {r_ghost:.3e}, ones {r_ones:.3e}")


def test_the_gain_survives_on_a_molecule_no_fit_ever_saw(supports):
    """The overfitting check, and the only reason the training set is split.

    `molw` minimises the training residual by construction, so that comparison proves
    nothing on its own. Formaldehyde is in neither the training set nor the ghost
    environments, and carries a C=O bond shorter than any ghost shell.
    """
    assert HELD_OUT not in TRAIN

    for element in ('H', 'C', 'O'):
        ghost_idx, ghost_w = supports['ghost'][element]
        molw_idx, molw_w = supports['molw'][element]

        r_ones = stacked_residual(element, [HELD_OUT], ghost_idx, np.ones(len(ghost_idx)))
        r_ghost = stacked_residual(element, [HELD_OUT], ghost_idx, ghost_w)
        r_molw = stacked_residual(element, [HELD_OUT], molw_idx, molw_w)

        assert r_molw < r_ghost, (
            f"{element} held out: molw {r_molw:.3e} did not beat ghostw {r_ghost:.3e}")
        assert r_molw < r_ones


def test_a_weight_set_is_no_better_than_chance_once_it_is_scrambled(supports):
    """A control: the gain has to come from *which* weight sits on which point.

    Permuting the fitted weights across the same points keeps every summary statistic -
    the count, the sum, the distribution - and destroys the fit. Without this, a residual
    improvement could just be a better overall scale.
    """
    rng = np.random.default_rng(0)
    for element in ('H', 'C', 'O'):
        idx, w = supports['molw'][element]
        fitted = stacked_residual(element, TRAIN, idx, w)
        scrambled = np.median([stacked_residual(element, TRAIN, idx, rng.permutation(w))
                               for _ in range(3)])

        assert fitted < scrambled


def test_rescaling_the_fitted_weights_can_only_make_the_residual_worse(supports):
    """The fit really is at a minimum, so a uniform rescale is a step away from it.

    Worth pinning because it separates the two things a weight set carries. The *energy*
    absorbs a positive diagonal rescale of ``w`` exactly under the pseudoinverse - which
    is why FINDINGS section 3 measured bit-identical MP2 energies for random weights -
    but the quadrature objective fitted here does not. A scheme that normalises the
    weights (to unit sum, say) therefore moves this residual while leaving the
    pseudoinverse energy alone, and the two must not be confused.
    """
    element = 'O'
    idx, w = supports['molw'][element]
    base = stacked_residual(element, TRAIN, idx, w)

    for factor in (0.5, 2.0):
        assert stacked_residual(element, TRAIN, idx, factor * w) > base


# --------------------------------------------------------------------------------------
# Reporting helpers.
# --------------------------------------------------------------------------------------

def test_matched_size_report_interpolates_each_mode_onto_shared_point_counts():
    rows = [
        dict(molecule='m', held_out=False, mode='a', threshold=1e-3, n_points=100,
             err_uha=100.0),
        dict(molecule='m', held_out=False, mode='a', threshold=1e-4, n_points=400,
             err_uha=10.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-3, n_points=200,
             err_uha=50.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-4, n_points=800,
             err_uha=5.0),
    ]
    out = matched_size_report(rows, ['a', 'b'], 'err_uha')

    assert out, "the ladders overlap on 200..400 points, so there is a window to quote"
    for row in out:
        assert 200 <= row['n_points'] <= 400
        assert 'a' in row and 'b' in row


def test_matched_size_report_declines_to_compare_ladders_that_do_not_overlap():
    rows = [
        dict(molecule='m', held_out=False, mode='a', threshold=1e-3, n_points=10,
             err_uha=100.0),
        dict(molecule='m', held_out=False, mode='a', threshold=1e-4, n_points=20,
             err_uha=10.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-3, n_points=500,
             err_uha=50.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-4, n_points=900,
             err_uha=5.0),
    ]
    assert matched_size_report(rows, ['a', 'b'], 'err_uha') == []


def test_a_becke_row_carries_no_threshold_and_is_left_out_of_the_interpolation():
    rows = [
        dict(molecule='m', held_out=False, mode='becke', threshold=None, n_points=5000,
             err_uha=0.1),
        dict(molecule='m', held_out=False, mode='a', threshold=1e-3, n_points=100,
             err_uha=100.0),
        dict(molecule='m', held_out=False, mode='a', threshold=1e-4, n_points=400,
             err_uha=10.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-3, n_points=200,
             err_uha=50.0),
        dict(molecule='m', held_out=False, mode='b', threshold=1e-4, n_points=800,
             err_uha=5.0),
    ]
    out = matched_size_report(rows, ['becke', 'a', 'b'], 'err_uha')

    assert out
    assert all('becke' not in row for row in out)

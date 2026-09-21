"""
Tests for the orbital-response layer and the total frozen-grid THC-MP2 gradient.

``tests/test_thc_gradient.py`` checks the fixed-orbital gradient, which can be finite
differenced one adjoint at a time with the orbitals pinned. The response cannot be tested
that way: it only exists relative to a re-converged SCF, so the honest check is the total
gradient against a finite difference of the total energy with the SCF re-solved at every
displaced geometry. That is :func:`test_relaxed_gradient_against_finite_difference`, and
it is why this file is slower than its neighbour.

Three cheaper tests sit in front of it, each pinning one link of the chain:

* :func:`test_loewner_is_the_derivative_of_the_matrix_function` - the divided-difference
  matrix, against an actual matrix exponential;
* :func:`test_fock_adjoint_diagonal_reproduces_eps_bar` and
  :func:`test_fock_adjoints_against_finite_difference` - the new matrix adjoint, against a
  finite difference of a genuine Fock-block perturbation. The diagonal one is free and
  exact: the reformulation must reduce to the old orbital-energy adjoint on the diagonal;
* :func:`test_lagrangian_is_symmetric_in_oo_and_vv` - the invariance the whole
  reformulation rests on. If ``Theta_o = w^(1/4) exp(t F_oo)`` is right then the energy
  cannot see an antisymmetric rotation within the occupied or virtual space, so those
  blocks of the MO Lagrangian must come out symmetric. They do, to 1e-14.
"""
import numpy as np
import pytest
from pyscf import gto, md, scf
from scipy.linalg import expm

from pythc.grad import FrozenGrid, thc_mp2_gradient
from pythc.grad.factorisation import ThcFactorisation
from pythc.grad.laplace_mp2 import (energy, energy_and_adjoints, laplace_factors,
                                    loewner)
from pythc.grad.response import orbital_response_gradient, response_lagrangian
from pythc.grad.total import ThcMP2Gradients
from pythc.grid import BeckeGrid, NNLSGrid

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-ri"

# As in test_thc_gradient: loose enough that the metric is invertible without amplifying
# rounding noise, so a central difference converges.
TEST_RIDGE = 1e-2
N_LAPLACE = 6

ATOM = ("O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; "
        "H 0.0000 -0.7572 -0.4692")


def _scf_at(coords=None):
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    if coords is not None:
        mol = mol.copy()
        mol.set_geom_(coords, unit="Bohr")
        mol.build(False, False)

    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-14
    mf.kernel()

    return mol, mf


@pytest.fixture(scope="module")
def system():
    mol, mf = _scf_at()

    grid = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=1e-3, blocked=True)
    coords, weights = grid.build()
    owner = np.argmin(np.linalg.norm(coords[:, None, :] - mol.atom_coords()[None], axis=2),
                      axis=1)
    per_atom = [(coords[owner == ia] - mol.atom_coord(ia), weights[owner == ia])
                for ia in range(mol.natm)]

    return mol, mf, per_atom


@pytest.fixture(scope="module")
def adjoints(system):
    """One reverse pass, plus the pieces a Fock-block finite difference needs."""
    mol, mf, per_atom = system
    n_occ = mol.nelectron // 2
    coords, weights = FrozenGrid(mol, per_atom).build()

    fac = ThcFactorisation(mol, coords, weights, mf.mo_coeff, n_occ, AUXBASIS,
                           metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE).build()
    eps = np.asarray(mf.mo_energy)
    t, tau_o, tau_v = laplace_factors(eps, n_occ, N_LAPLACE)

    out = energy_and_adjoints(fac.X_o, fac.X_v, fac.Z, tau_o, tau_v, t,
                              want_fock_adjoints=True,
                              eps_o=eps[:n_occ], eps_v=eps[n_occ:])

    return fac, eps, n_occ, t, tau_o, tau_v, out


# --------------------------------------------------------------------------------------
# The divided-difference matrix
# --------------------------------------------------------------------------------------

def test_loewner_is_the_derivative_of_the_matrix_function():
    """
    ``loewner`` against a real directional derivative of ``w^(1/4) exp(t A)``.

    At a diagonal ``A`` the Frechet derivative of a matrix function is a Hadamard product
    with the divided-difference matrix, which is the identity the response layer is built
    on. Checked here rather than assumed.
    """
    eps = np.array([-1.3, -0.44, -0.44, 0.21, 2.6])
    t, amp = 0.73, 0.3 ** 0.25
    f = amp * np.exp(t * eps)
    L = loewner(eps, f, t)

    rng = np.random.default_rng(11)
    D = rng.standard_normal((len(eps),) * 2)
    D = 0.5 * (D + D.T)

    def at(h):
        return amp * expm(t * (np.diag(eps) + h * D))

    # The ratio is what carries the test: a central difference of a correct derivative
    # falls by 100x per decade of step, and one of an incorrect derivative plateaus at
    # the difference. The absolute floor is just the truncation error at the last step.
    prev = None
    for h in (1e-2, 1e-3, 1e-4):
        num = (at(h) - at(-h)) / (2 * h)
        err = np.abs(L * D - num).max()
        if prev is not None:
            assert err < prev / 50, "not converging quadratically"
        prev = err

    assert prev < 1e-7


def test_loewner_survives_a_wide_spectrum_and_a_degeneracy():
    """
    Finite and symmetric across the range a real virtual block spans.

    Neither closed form works everywhere: the stable ``expm1`` branch overflows for large
    ``|d|``, and the plain difference quotient loses its digits for small. This is the
    regression test for using each only where it is good.
    """
    for t, eps in ((40.0, np.array([-20.0, -1.1, -0.5, -0.5, -0.31])),
                   (-40.0, np.array([0.02, 0.4, 0.4, 3.0, 12.0, 120.0]))):
        f = 0.3 ** 0.25 * np.exp(t * eps)
        L = loewner(eps, f, t)

        assert np.all(np.isfinite(L))
        assert np.abs(L - L.T).max() <= 1e-9 * max(1.0, np.abs(L).max())
        assert np.abs(np.diag(L) - t * f).max() <= 1e-9 * max(1.0, np.abs(t * f).max())

        i, j = 0, len(eps) - 1
        ref = (f[i] - f[j]) / (eps[i] - eps[j])
        assert abs(L[i, j] - ref) <= 1e-9 * max(1.0, abs(ref))


# --------------------------------------------------------------------------------------
# The Fock adjoint
# --------------------------------------------------------------------------------------

def test_fock_adjoint_diagonal_reproduces_eps_bar(adjoints):
    """
    ``diag(dE/dF_oo) == dE/de``, exactly.

    The reformulation only changes the energy off the diagonal - on it, ``F_oo[i,i]`` is
    ``e_i``. So this is free, it is exact rather than approximate, and it would catch a
    mis-scaled Loewner diagonal immediately.
    """
    *_, out = adjoints
    _, _, _, _, eps_o_bar, eps_v_bar, F_oo_bar, F_vv_bar = out

    assert np.abs(np.diag(F_oo_bar) - eps_o_bar).max() < 1e-14
    assert np.abs(np.diag(F_vv_bar) - eps_v_bar).max() < 1e-14
    assert np.abs(F_oo_bar - F_oo_bar.T).max() < 1e-14
    assert np.abs(F_vv_bar - F_vv_bar.T).max() < 1e-14


def test_fock_adjoints_against_finite_difference(adjoints):
    """
    ``dE/dF_oo`` and ``dE/dF_vv`` against perturbing the Fock blocks for real.

    The perturbed energy is evaluated by diagonalising ``Theta = w^(1/4) exp(t(diag(e)+D))``
    and feeding ``(X V, theta)`` to the ordinary energy routine - which is exact, since
    ``X Theta X^T = (X V) diag(theta) (X V)^T``. That the two agree at ``D = 0`` is itself
    the check that the invariant rewriting is the same function.
    """
    fac, eps, n_occ, t, tau_o, tau_v, out = adjoints
    e0, _, _, _, _, _, F_oo_bar, F_vv_bar = out

    eps_o, eps_v = eps[:n_occ], eps[n_occ:]
    amp = tau_o[:, 0] * np.exp(-t * eps_o[0])           # w^(1/4) per node
    zo = np.zeros((n_occ, n_occ))
    zv = np.zeros((len(eps_v), len(eps_v)))

    def energy_at(d_o, d_v):
        total = 0.0
        for v in range(N_LAPLACE):
            th_o, Vo = np.linalg.eigh(amp[v] * expm(+t[v] * (np.diag(eps_o) + d_o)))
            th_v, Vv = np.linalg.eigh(amp[v] * expm(-t[v] * (np.diag(eps_v) + d_v)))
            total += energy(fac.X_o @ Vo, fac.X_v @ Vv, fac.Z,
                            th_o[None, :], th_v[None, :])
        return total

    # The rewriting is the same function, not merely a close one.
    assert abs(energy_at(zo, zv) - e0) < 1e-12

    rng = np.random.default_rng(3)
    for bar, nb, occ in ((F_oo_bar, n_occ, True), (F_vv_bar, len(eps_v), False)):
        D = rng.standard_normal((nb, nb))
        D = 0.5 * (D + D.T)
        ana = float(np.sum(bar * D))

        prev = None
        for h in (1e-3, 1e-4):
            num = ((energy_at(h * D, zv) - energy_at(-h * D, zv)) / (2 * h) if occ else
                   (energy_at(zo, h * D) - energy_at(zo, -h * D)) / (2 * h))
            err = abs(num - ana)
            if prev is not None:
                assert err < prev / 50, "not converging quadratically"
            prev = err

        assert prev < 1e-8


# --------------------------------------------------------------------------------------
# The response
# --------------------------------------------------------------------------------------

def test_lagrangian_is_symmetric_in_oo_and_vv(system, adjoints):
    """
    The invariance the reformulation buys, checked on the object that uses it.

    With ``Theta_o`` a function of ``F_oo``, the energy is invariant under an orthogonal
    rotation within the occupied space and within the virtual space. So the MO Lagrangian
    can have no antisymmetric part in either block - and if it does, the Fock adjoint is
    wrong and the response would be contracting ``U`` blocks that a CPHF solver never
    supplies. This is the cheapest test in the file and the most diagnostic.
    """
    mol, mf, per_atom = system
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS,
                           n_laplace=N_LAPLACE, metric_ridge=TEST_RIDGE,
                           aux_ridge=TEST_RIDGE)

    _, _, _, asymmetry = response_lagrangian(mf, res.C_bar, res.F_oo_bar, res.F_vv_bar)

    assert asymmetry < 1e-10


def test_relaxed_gradient_against_finite_difference(system):
    """
    The whole thing: total correlation gradient against a fully relaxed finite difference.

    The SCF is re-converged at every displaced geometry, so this compares the analytic
    gradient against the derivative of the energy a trajectory would actually feel. The
    fixed-orbital gradient is checked too, and must be *worse* - if it is not, the
    response is not doing anything and the test is not measuring what it claims to.
    """
    mol, mf, per_atom = system
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS,
                           n_laplace=N_LAPLACE, metric_ridge=TEST_RIDGE,
                           aux_ridge=TEST_RIDGE)
    resp = orbital_response_gradient(mol, mf, res.C_bar, res.F_oo_bar, res.F_vv_bar)
    total = res.de + resp.de

    def ecorr_at(coords):
        m, mfx = _scf_at(coords)
        return thc_mp2_gradient(m, mfx, FrozenGrid(m, per_atom), AUXBASIS,
                                n_laplace=N_LAPLACE, metric_ridge=TEST_RIDGE,
                                aux_ridge=TEST_RIDGE).energy

    R0 = mol.atom_coords()
    h = 1e-3
    worst_total = worst_fixed = 0.0

    for ia in range(mol.natm):
        for k in range(3):
            dp, dm = R0.copy(), R0.copy()
            dp[ia, k] += h
            dm[ia, k] -= h
            num = (ecorr_at(dp) - ecorr_at(dm)) / (2.0 * h)

            worst_total = max(worst_total, abs(total[ia, k] - num))
            worst_fixed = max(worst_fixed, abs(res.de[ia, k] - num))

    assert worst_total < 1e-7, f"relaxed gradient off by {worst_total:.2e}"
    # The response is worth three orders of magnitude here; without it the gradient is
    # simply not the derivative of this energy.
    assert worst_fixed > 100 * worst_total


def test_response_closes_the_rotational_identity(system):
    """
    ``-sum_A R_A x dE/dR_A == sum_A tau_A``, with no finite difference anywhere.

    For a lab-fixed atom-centred grid the rate at which nuclear angular momentum leaks is
    exactly the grid's own orientation torque - that is FINDINGS section 11's identity,
    and it holds only if the gradient is the true derivative of the propagated energy.
    The Hartree-Fock part is exactly rotationally invariant on its own (density-fitted HF
    has no quadrature grid), so any residual belongs to the correlation gradient.

    This is the sharpest test in the file: it is exact rather than converged, and it is
    what FINDINGS section 13 found violated at ~8300 uHa/rad on the fixed-orbital force -
    fifty times the transferable grid's own torque, and the reason no AIMD could run.
    """
    mol, mf, per_atom = system
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS,
                           n_laplace=N_LAPLACE, metric_ridge=TEST_RIDGE,
                           aux_ridge=TEST_RIDGE)
    resp = orbital_response_gradient(mol, mf, res.C_bar, res.F_oo_bar, res.F_vv_bar)
    hf = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)

    R = mol.atom_coords()
    tau_grid = res.torque.sum(axis=0)

    def leak(de):
        return -np.cross(R, de).sum(axis=0)

    relaxed = np.linalg.norm(leak(hf + res.de + resp.de) - tau_grid)
    fixed = np.linalg.norm(leak(hf + res.de) - tau_grid)
    scale = np.linalg.norm(hf + res.de + resp.de)

    assert relaxed / scale < 1e-7, "the relaxed gradient does not close the identity"
    assert fixed > 1e4 * relaxed


def test_energy_is_conserved_along_a_trajectory(system):
    """
    NVE on the frozen-grid THC-MP2 surface, which is the application the whole programme
    is for.

    A Born-Oppenheimer trajectory conserves total energy only if the force really is minus
    the gradient of the propagated energy. With the response it drifts by well under a
    microhartree per step; on the fixed-orbital force the same trajectory drifts by tens,
    which is the number FINDINGS section 13 reported and the reason it called the response
    a prerequisite rather than a refinement.

    The grid is frozen for the whole run: one nucleus-relative point set per atom,
    translated onto the nuclei at each step and never re-selected.
    """
    mol, mf, per_atom = system
    drifts = {}

    for with_response in (True, False):
        grad = ThcMP2Gradients(per_atom=per_atom, auxbasis=AUXBASIS,
                               n_laplace=N_LAPLACE, metric_ridge=TEST_RIDGE,
                               aux_ridge=TEST_RIDGE, with_response=with_response)

        integrator = md.NVE(grad.as_scanner(mol.copy()), dt=20, steps=6)
        integrator.verbose = 0
        integrator.incore_anyway = True
        integrator.frames = []
        integrator.run()

        e = np.array([f.ekin + f.epot for f in integrator.frames])
        drifts[with_response] = np.abs(np.diff(e)).mean() * 1e6      # uHa per step

    assert drifts[True] < 2.0, f"drift {drifts[True]:.3f} uHa/step with the response"
    assert drifts[False] > 20 * drifts[True]

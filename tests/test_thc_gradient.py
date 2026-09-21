"""
Tests for the analytic nuclear gradient of the frozen-grid LS-THC MP2 energy.

Every formula in :mod:`pythc.grad` was derived by hand, so each stage is checked against
a finite difference of its own inputs rather than only at the end - an error in one
adjoint can hide behind another if only the total is tested.

**A note on step sizes, because it is the whole difficulty of testing this code.** The
LS-THC metric is spectacularly rank deficient: water's 156-point blocked grid has 61 of
156 directions below 1e-14 of the largest eigenvalue. A ridge of ``lambda`` inverts those
directions at ``1/lambda``, so a finite-difference step that moves them by more than the
ridge shift measures the resulting nonlinearity instead of a derivative. The adjoint
formulas do not depend on ``lambda``, so the stage tests use a loose ridge where a
central difference is actually resolvable, and the end-to-end test uses a ridge where the
pipeline is well conditioned. Where the usable window lies is not a testing artefact but
a result in its own right - see ``experiments/atom_centered_grids/gradient.py``.
"""
import numpy as np
import pytest
from pyscf import df, gto, scf

from pythc import lib
from pythc.grad import FrozenGrid, thc_mp2_gradient
from pythc.grad.factorisation import ThcFactorisation
from pythc.grad.geometry import basis_to_atom
from pythc.grad.laplace_mp2 import energy, energy_and_adjoints, laplace_factors
from pythc.grad import linalg as glin
from pythc.grid import BeckeGrid, NNLSGrid
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_thc_funcs import build_S, invert_metric

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-ri"

# Loose enough that the metric is invertible without amplifying rounding noise, so a
# central difference converges. See the module docstring.
TEST_RIDGE = 1e-2


def _water():
    return gto.M(atom="O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; "
                      "H 0.0000 -0.7572 -0.4692",
                 basis=BASIS, verbose=0)


@pytest.fixture(scope="module")
def system():
    mol = _water()
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()

    grid = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=1e-3, blocked=True)
    coords, weights = grid.build()

    # Regroup the flat grid into per-atom sets. The fit is blocked, so each atom's points
    # are contiguous and nearest-nucleus assignment recovers the grouping exactly.
    owner = np.argmin(np.linalg.norm(coords[:, None, :] - mol.atom_coords()[None], axis=2),
                      axis=1)
    per_atom = [(coords[owner == ia] - mol.atom_coord(ia), weights[owner == ia])
                for ia in range(mol.natm)]

    return mol, mf, per_atom


def _directional(f, x, d, h):
    return (f(x + h * d) - f(x - h * d)) / (2.0 * h)


# --------------------------------------------------------------------------------------
# Stage 1: the Laplace MP2 expression
# --------------------------------------------------------------------------------------

def test_laplace_energy_matches_production(system):
    """What the gradient differentiates must be what pythc evaluates."""
    mol, mf, per_atom = system
    grid = FrozenGrid(mol, per_atom)
    coords, weights = grid.build()

    thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                      grid=grid, metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE)
    eri = thc.build(mode="ov")
    expected = float(LaplaceRMP2(mol, mf, eri, n_laplace=10).kernel())

    nocc = mol.nelectron // 2
    X, Z = np.array(eri.get_X_Z()[0]), np.array(eri.get_X_Z()[1])
    _, tau_o, tau_v = laplace_factors(mf.mo_energy, nocc, 10)

    got = energy(X[:, :nocc], X[:, nocc:], Z, tau_o, tau_v)

    assert abs(got - expected) < 1e-12 * max(abs(expected), 1.0)


@pytest.mark.parametrize("which", ["X_o", "X_v", "Z", "eps"])
def test_laplace_adjoints(system, which):
    """dE/dX, dE/dZ and dE/de against a central difference."""
    mol, mf, per_atom = system
    nocc = mol.nelectron // 2

    fac = ThcFactorisation(mol, *FrozenGrid(mol, per_atom).build(), mf.mo_coeff, nocc,
                           AUXBASIS, metric_ridge=TEST_RIDGE).build()
    t, tau_o, tau_v = laplace_factors(mf.mo_energy, nocc, 10)
    _, X_o_bar, X_v_bar, Z_bar, eps_o_bar, eps_v_bar = energy_and_adjoints(
        fac.X_o, fac.X_v, fac.Z, tau_o, tau_v, t)

    rng = np.random.default_rng(0)
    h = 1e-6

    if which == "eps":
        d = rng.standard_normal(mol.nao_nr())

        def f(scale):
            _, to, tv = laplace_factors(mf.mo_energy + scale * d, nocc, 10)
            return energy(fac.X_o, fac.X_v, fac.Z, to, tv)

        num = (f(h) - f(-h)) / (2.0 * h)
        ana = float(eps_o_bar @ d[:nocc] + eps_v_bar @ d[nocc:])
    else:
        ref = {"X_o": fac.X_o, "X_v": fac.X_v, "Z": fac.Z}[which]
        bar = {"X_o": X_o_bar, "X_v": X_v_bar, "Z": Z_bar}[which]
        d = rng.standard_normal(ref.shape)
        if which == "Z":
            d = 0.5 * (d + d.T)          # Z = D^T D is symmetric

        def f(m):
            args = {"X_o": fac.X_o, "X_v": fac.X_v, "Z": fac.Z}
            args[which] = m
            return energy(args["X_o"], args["X_v"], args["Z"], tau_o, tau_v)

        num = _directional(f, ref, d, h)
        ana = float(np.sum(bar * d))

    assert abs(ana - num) < 1e-6 * max(abs(num), 1e-8)


# --------------------------------------------------------------------------------------
# Stage 2: the regularised inversions
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spd():
    rng = np.random.default_rng(3)
    n = 24
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    M = (Q * np.logspace(0, -5, n)) @ Q.T

    return 0.5 * (M + M.T), Q


@pytest.mark.parametrize("scale", ["trace", "absolute", "max_eig"])
def test_ridge_inv_adjoint(spd, scale):
    M, _ = spd
    rng = np.random.default_rng(5)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)
    lam = 1e-3 if scale != "absolute" else 1e-4

    inv = lib.ridge_inv(M.copy(), lam, scale)
    ana = float(np.sum(glin.ridge_inv_adjoint(M.copy(), inv, out_bar, lam, scale) * d))
    num = _directional(lambda m: np.sum(lib.ridge_inv(m, lam, scale) * out_bar),
                       M, d, 1e-9)

    assert abs(ana - num) < 1e-5 * abs(num)


def test_ridge_inv_sqrt_adjoint(spd):
    M, _ = spd
    rng = np.random.default_rng(6)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)

    ana = float(np.sum(glin.ridge_inv_sqrt_adjoint(M.copy(), out_bar, 1e-3) * d))
    num = _directional(lambda m: np.sum(lib.ridge_inv_sqrt(m, 1e-3) * out_bar),
                       M, d, 1e-9)

    assert abs(ana - num) < 1e-5 * abs(num)


def test_pseudo_inv_sqrt_adjoint(spd):
    M, _ = spd
    rng = np.random.default_rng(7)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)

    ana = float(np.sum(glin.pseudo_inv_sqrt_adjoint(M.copy(), out_bar) * d))
    num = _directional(lambda m: np.sum(lib.pseudo_inv_sqrt(m.copy()) * out_bar),
                       M, d, 1e-9)

    assert abs(ana - num) < 1e-5 * abs(num)


@pytest.mark.parametrize("scale", ["trace", "absolute", "max_eig"])
def test_damped_inv_adjoint(spd, scale):
    M, _ = spd
    rng = np.random.default_rng(8)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)
    lam = 1e-3 if scale != "absolute" else 1e-4

    ana = float(np.sum(glin.damped_inv_adjoint(M.copy(), out_bar, lam, scale) * d))
    num = _directional(lambda m: np.sum(lib.damped_inv(m, lam, scale) * out_bar),
                       M, d, 1e-9)

    assert abs(ana - num) < 1e-5 * abs(num)


def test_damped_inv_sqrt_adjoint(spd):
    M, _ = spd
    rng = np.random.default_rng(9)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)

    ana = float(np.sum(glin.damped_inv_sqrt_adjoint(M.copy(), out_bar, 1e-3) * d))
    num = _directional(lambda m: np.sum(lib.damped_inv_sqrt(m, 1e-3) * out_bar),
                       M, d, 1e-9)

    assert abs(ana - num) < 1e-5 * abs(num)


def test_damped_inv_mu_term_is_not_optional():
    """
    The damping ``mu`` depends on ``M``, so it contributes a term of its own.

    Dropping it leaves an adjoint that is wrong by a term proportional to ``lam`` - the
    kind of omission a finite difference converges to a constant on rather than
    quadratically. This pins the term directly by varying ``mu`` alone, which
    ``scale="absolute"`` makes an independent knob.
    """
    rng = np.random.default_rng(10)
    n, n_null = 18, 6
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    sigma = np.concatenate([np.logspace(0, -5, n - n_null), np.zeros(n_null)])
    M = 0.5 * (((Q * sigma) @ Q.T) + ((Q * sigma) @ Q.T).T)
    out_bar = rng.standard_normal((n, n)); sym = 0.5 * (out_bar + out_bar.T)

    for mu in (1e-2, 1e-3):
        eigvals, eigvecs = np.linalg.eigh(M)
        denom = eigvals ** 2 + mu ** 2
        df_dmu = -2.0 * eigvals * mu / denom ** 2
        ana = float(np.sum(df_dmu * np.einsum("ki,kl,li->i", eigvecs, sym, eigvecs)))

        h = mu * 1e-5
        num = float(np.sum(out_bar * (lib.damped_inv(M, mu + h, "absolute")
                                      - lib.damped_inv(M, mu - h, "absolute")))) / (2 * h)

        assert abs(ana - num) < 1e-6 * abs(num)


def test_damped_inv_suppresses_the_null_space_the_ridge_amplifies():
    """
    The whole point of the damped inverse, as a property rather than a derivative.

    Both schemes share a peak gain of ``1/(2 mu)``, so ``lam`` means the same thing on
    each. They differ entirely on the numerically null directions, which the ridge hands
    the *largest* gain in the operator and the damped inverse sends to zero.
    """
    rng = np.random.default_rng(12)
    n, n_null = 40, 15
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    sigma = np.concatenate([np.logspace(0, -6, n - n_null), np.zeros(n_null)])
    M = 0.5 * (((Q * sigma) @ Q.T) + ((Q * sigma) @ Q.T).T)

    lam = 1e-4
    mu = lib.ridge_shift(M, lam)
    ridged, damped = lib.ridge_inv(M.copy(), lam), lib.damped_inv(M.copy(), lam)

    null = Q[:, -1]
    assert float(null @ ridged @ null) == pytest.approx(1.0 / mu, rel=1e-6)
    assert abs(float(null @ damped @ null)) < 1e-8 / mu

    assert np.linalg.norm(damped, 2) == pytest.approx(1.0 / (2.0 * mu), rel=1e-3)


def test_pinv_adjoint_at_fixed_rank(spd):
    """
    Between crossings the truncated pseudoinverse is an ordinary spectral function.

    The cutoff is set loose and the discarded eigenvalues put four decades below it, so
    the retained set survives the finite-difference step. Crossing it is the subject of
    the next test.
    """
    _, Q = spd
    n = Q.shape[0]
    eps = 1e-2
    ev = np.concatenate([np.logspace(0, -1, n - 8), np.full(8, 1e-6)])
    M = (Q * ev) @ Q.T; M = 0.5 * (M + M.T)

    rng = np.random.default_rng(8)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)

    ana = float(np.sum(glin.pinv_adjoint(M.copy(), out_bar, eps) * d))
    num = _directional(lambda m: np.sum(lib.pinv(m.copy(), eps) * out_bar), M, d, 1e-8)

    assert abs(ana - num) < 1e-5 * abs(num)


@pytest.fixture(scope="module")
def spd_scaled():
    """
    An SPD matrix whose *diagonal* spans four decades, so the Jacobi scaling is real.

    The ``spd`` fixture above is built from a random orthogonal basis, which leaves its
    diagonal nearly flat and ``E`` nearly a multiple of the identity - under which a
    preconditioned adjoint would agree with an unpreconditioned one for the wrong reason.
    The row scaling here is exactly the object the LS-THC metric carries: collocation
    weights enter it only as ``S(w) = D S(1) D``.
    """
    rng = np.random.default_rng(3)
    n = 24
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    M = 0.5 * (((Q * np.logspace(0, -5, n)) @ Q.T) + ((Q * np.logspace(0, -5, n)) @ Q.T).T)
    s = np.logspace(-1.0, 1.0, n)
    M = s[:, None] * M * s[None, :]

    return 0.5 * (M + M.T)


@pytest.mark.parametrize("scheme", ["ridge_jacobi", "ridge_eigh_jacobi", "damped_jacobi"])
@pytest.mark.parametrize("scale", ["trace", "absolute", "max_eig"])
def test_jacobi_adjoint(spd_scaled, scheme, scale):
    """
    Adjoint of the Jacobi-preconditioned inversion, against a central difference.

    ``E = diag(1 / sqrt(diag M))`` is a function of ``M``, so this differentiates three
    paths at once - the inner filter, and ``E`` in each of the two places it appears.
    """
    M = spd_scaled
    rng = np.random.default_rng(11)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)
    lam = 1e-3 if scale != "absolute" else 1e-4

    inv = invert_metric(M.copy(), lam, scale, scheme)
    ana = float(np.sum(glin.invert_metric_adjoint(M.copy(), inv, out_bar, lam, scale,
                                                  scheme) * d))
    num = _directional(lambda m: np.sum(invert_metric(m.copy(), lam, scale, scheme)
                                        * out_bar), M, d, 1e-10)

    assert abs(ana - num) < 1e-5 * abs(num)


@pytest.mark.parametrize("scheme", ["ridge_jacobi", "damped_jacobi"])
def test_jacobi_scaling_terms_are_not_optional(spd_scaled, scheme):
    """
    Neither of the two ways of not differentiating ``E`` gives the right answer.

    The forward pass shipped without an adjoint and the dispatch fell through to the
    ridge's, which differentiates ``filter(S)`` where the forward applied
    ``E filter(E S E) E``. Energies were right and gradients silently wrong - net torques
    of ~4e5 uHa/rad against ~1e-1 for the same grids under ``"damped"``. Two candidates
    are checked here against the same finite difference: that historic fall-through, and
    the near miss of taking the inner filter's term at ``Mj`` but holding ``E`` fixed.
    Both are wrong by decades more than the full adjoint's own error, and the
    ``"damped"`` fall-through is wrong by 100%.
    """
    M = spd_scaled
    rng = np.random.default_rng(12)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)
    lam, inner = 1e-3, lib.strip_jacobi(scheme)

    inv = invert_metric(M.copy(), lam, "trace", scheme)
    full = float(np.sum(glin.invert_metric_adjoint(M.copy(), inv, out_bar, lam, "trace",
                                                   scheme) * d))

    # The inner filter's term alone, i.e. E held fixed - everything jacobi_inv_adjoint
    # adds on top of that is what this test is about.
    e = lib.jacobi_scaling(M)
    Mj_bar = glin.invert_metric_adjoint(e[:, None] * M * e[None, :],
                                        inv / np.outer(e, e),
                                        e[:, None] * out_bar * e[None, :],
                                        lam, "trace", inner)
    held = float(np.sum((e[:, None] * Mj_bar * e[None, :]) * d))

    # And the fall-through itself: the unpreconditioned filter, differentiated at M.
    fell_through = float(np.sum(glin.invert_metric_adjoint(M.copy(), inv, out_bar, lam,
                                                           "trace", inner) * d))

    num = _directional(lambda m: np.sum(invert_metric(m.copy(), lam, "trace", scheme)
                                        * out_bar), M, d, 1e-10)

    assert abs(full - num) < 1e-6 * abs(num)
    assert abs(held - num) > 1e-4 * abs(num)
    assert abs(fell_through - num) > 1e-4 * abs(num)


def test_jacobi_cancels_where_the_filter_is_an_exact_inverse(spd_scaled):
    """
    The preconditioner is a similarity transform, so an exact inverse cannot see it.

    ``E (E M E)^-1 E = M^-1`` for any invertible diagonal ``E``, which makes the whole
    construction a no-op at ``lambda = 0`` - and its derivative the plain
    ``-M^-1 B_bar M^-1``. That is an identity rather than a finite difference, so it
    checks the three paths against each other to machine precision: an error in the
    inner term, in either ``E`` term, or in a sign, breaks the cancellation.
    """
    M = spd_scaled
    rng = np.random.default_rng(13)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)

    inv = invert_metric(M.copy(), 0.0, "absolute", "ridge_jacobi")
    exact_inv = np.linalg.inv(M)
    assert np.max(np.abs(inv - exact_inv)) < 1e-8 * np.max(np.abs(exact_inv))

    ana = glin.invert_metric_adjoint(M.copy(), inv, out_bar, 0.0, "absolute",
                                     "ridge_jacobi")
    exact = -exact_inv @ out_bar @ exact_inv

    assert np.max(np.abs(ana - exact)) < 1e-8 * np.max(np.abs(exact))


def test_jacobi_leaves_a_point_with_no_amplitude_alone():
    """
    ``S_PP`` is exactly zero wherever a weight fit dropped a point, and ``1/sqrt(0)``
    would turn the whole metric - and its adjoint - into NaN.

    Those rows are left unscaled, which also makes ``e_P`` a constant there rather than a
    function of the diagonal, so they contribute nothing to the diagonal term.
    """
    rng = np.random.default_rng(14)
    n, dead = 12, 3
    A = rng.standard_normal((n, n))
    A[-dead:] = 0.0
    M = A @ A.T

    assert np.allclose(np.diag(M)[-dead:], 0.0)

    out_bar = rng.standard_normal((n, n)); out_bar = 0.5 * (out_bar + out_bar.T)
    inv = invert_metric(M.copy(), 1e-3, "trace", "damped_jacobi")
    bar = glin.invert_metric_adjoint(M.copy(), inv, out_bar, 1e-3, "trace",
                                     "damped_jacobi")

    assert np.all(np.isfinite(inv))
    assert np.all(np.isfinite(bar))

    # And it is still the derivative. The perturbation leaves the dead rows dead, which
    # is what a frozen support does - a dropped point stays dropped as the nuclei move -
    # so the forward map is smooth along it.
    d = rng.standard_normal((n, n)); d = 0.5 * (d + d.T)
    d[-dead:, :] = 0.0
    d[:, -dead:] = 0.0

    ana = float(np.sum(bar * d))
    num = _directional(lambda m: np.sum(invert_metric(m.copy(), 1e-3, "trace",
                                                      "damped_jacobi") * out_bar),
                       M, d, 1e-10)

    assert abs(ana - num) < 1e-5 * abs(num)


def test_pinv_has_no_derivative_across_a_crossing(spd):
    """
    The truncation's discontinuity, in miniature.

    With an eigenvalue parked on the cutoff, the finite difference straddles a change of
    rank and the fixed-subspace derivative is simply the wrong object. This is HANDOFF
    section 4(1)'s result reduced to one matrix, and it is asserted rather than merely
    noted so that anyone who later "fixes" :func:`pythc.grad.linalg.pinv_adjoint` to
    agree here knows they have papered over a real discontinuity.
    """
    _, Q = spd
    n = Q.shape[0]
    eps = 1e-2
    ev = np.concatenate([np.logspace(0, -1, n - 1), [eps * 1.000001]])
    M = (Q * ev) @ Q.T; M = 0.5 * (M + M.T)

    rng = np.random.default_rng(9)
    out_bar = rng.standard_normal(M.shape); out_bar = 0.5 * (out_bar + out_bar.T)
    d = rng.standard_normal(M.shape); d = 0.5 * (d + d.T)

    ana = float(np.sum(glin.pinv_adjoint(M.copy(), out_bar, eps) * d))
    num = _directional(lambda m: np.sum(lib.pinv(m.copy(), eps) * out_bar), M, d, 1e-6)

    assert abs(ana - num) > 1e-3 * abs(num)


# --------------------------------------------------------------------------------------
# Stage 3: the factorisation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["X_ao", "V", "j2c", "C"])
def test_factorisation_adjoints(system, which):
    """dE/dX_ao, dE/dV, dE/dJ and dE/dC against a central difference."""
    mol, mf, per_atom = system
    nocc = mol.nelectron // 2

    fac = ThcFactorisation(mol, *FrozenGrid(mol, per_atom).build(), mf.mo_coeff, nocc,
                           AUXBASIS, metric_ridge=TEST_RIDGE).build()
    t, tau_o, tau_v = laplace_factors(mf.mo_energy, nocc, 10)
    _, X_o_bar, X_v_bar, Z_bar, _, _ = energy_and_adjoints(fac.X_o, fac.X_v, fac.Z,
                                                           tau_o, tau_v, t)
    adj = fac.backward(Z_bar, X_o_bar, X_v_bar)

    def rebuild(X_ao=None, V=None, j2c=None, C=None):
        X_ao = fac.X_ao if X_ao is None else X_ao
        V = fac.V if V is None else V
        j2c = fac.j2c if j2c is None else j2c
        C = fac.mo_coeff if C is None else C

        X = X_ao @ C
        S_inv = invert_metric(build_S("ov", X, nocc), TEST_RIDGE, "trace")
        Jm = lib.pseudo_inv_sqrt(j2c.copy())
        Xl = X_ao @ (C[:, :nocc] @ C[:, :nocc].T)
        Xr = X_ao @ (C[:, nocc:] @ C[:, nocc:].T)
        D = (Jm @ lib.einsum("ijq,gi,gj->qg", V, Xl, Xr)) @ S_inv

        return energy(X[:, :nocc], X[:, nocc:], D.T @ D, tau_o, tau_v)

    ref = {"X_ao": fac.X_ao, "V": fac.V, "j2c": fac.j2c, "C": fac.mo_coeff}[which]
    bar = {"X_ao": adj.X_ao_bar, "V": adj.V_bar, "j2c": adj.j2c_bar,
           "C": adj.C_bar}[which]

    rng = np.random.default_rng(12)
    d = rng.standard_normal(ref.shape)
    if which == "j2c":
        d = 0.5 * (d + d.T)
    if which == "V":
        d = 0.5 * (d + d.transpose(1, 0, 2))     # (mu nu|M) is symmetric in mu, nu

    h = 1e-6
    num = (rebuild(**{which: ref + h * d}) - rebuild(**{which: ref - h * d})) / (2.0 * h)
    ana = float(np.sum(bar * d))

    assert abs(ana - num) < 1e-5 * max(abs(num), 1e-10)


# --------------------------------------------------------------------------------------
# Stage 4: derivative-integral conventions
# --------------------------------------------------------------------------------------

def test_gto_ip_is_positive_gradient():
    """``GTOval_ip_sph`` is +grad_r phi, so an AO on atom A has d phi/dR_A = -grad_r phi."""
    mol = _water()
    pts = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 1.1]])
    ip = mol.eval_gto("GTOval_ip_sph", pts)

    h, k, g, mu = 1e-5, 1, 0, 3
    p, m = pts.copy(), pts.copy()
    p[g, k] += h
    m[g, k] -= h
    num = (mol.eval_gto("GTOval_sph", p)[g, mu]
           - mol.eval_gto("GTOval_sph", m)[g, mu]) / (2.0 * h)

    assert abs(ip[k, g, mu] - num) < 1e-7


@pytest.mark.parametrize("centre", ["three", "two"])
def test_derivative_integral_contractions(centre):
    """The 3- and 2-centre derivative contractions, against moving the nuclei."""
    from pythc.grad.geometry import three_center_gradient, two_center_gradient

    mol = _water()
    auxmol = df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    rng = np.random.default_rng(1)
    R0 = mol.atom_coords()
    h = 1e-5

    def moved(coords):
        m = mol.copy(); m.set_geom_(coords, unit="Bohr"); m.build(False, False)
        return m, df.addons.make_auxmol(m, auxbasis=AUXBASIS)

    if centre == "three":
        bar = rng.standard_normal((nao, nao, naux))
        ana = three_center_gradient(mol, auxmol, bar)

        def value(coords):
            m, am = moved(coords)
            V = df.incore.aux_e2(m, am, intor="int3c2e", aosym="s1")
            return float(np.sum(bar * V.reshape(nao, nao, naux)))
    else:
        bar = rng.standard_normal((naux, naux)); bar = 0.5 * (bar + bar.T)
        ana = two_center_gradient(mol, auxmol, bar)

        def value(coords):
            return float(np.sum(bar * moved(coords)[1].intor("int2c2e", aosym="s1")))

    for ia in range(mol.natm):
        for k in range(3):
            dp, dm = R0.copy(), R0.copy()
            dp[ia, k] += h
            dm[ia, k] -= h
            num = (value(dp) - value(dm)) / (2.0 * h)
            assert abs(ana[ia, k] - num) < 1e-6 * max(abs(num), 1.0)


def test_basis_to_atom_partitions_all_functions():
    mol = _water()
    index = basis_to_atom(mol)

    assert len(index) == mol.nao_nr()
    assert set(index.tolist()) == set(range(mol.natm))


# --------------------------------------------------------------------------------------
# Stage 5: the assembled gradient
# --------------------------------------------------------------------------------------

def test_nuclear_gradient_against_finite_difference(system):
    """
    The whole thing: analytic gradient against a central difference of the same energy.

    Orbitals are held fixed on both sides, because this gradient deliberately excludes
    the orbital-response term; letting the SCF relax on the numerical side would compare
    two different quantities.
    """
    mol, mf, per_atom = system
    grid = FrozenGrid(mol, per_atom)
    res = thc_mp2_gradient(mol, mf, grid, AUXBASIS, n_laplace=10,
                           metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE)

    class Ref:
        mo_coeff = np.array(mf.mo_coeff)
        mo_energy = np.array(mf.mo_energy)

    R0 = mol.atom_coords()
    h = 1e-4

    def energy_at(coords):
        m = mol.copy(); m.set_geom_(coords, unit="Bohr"); m.build(False, False)
        return thc_mp2_gradient(m, Ref, FrozenGrid(m, per_atom), AUXBASIS, n_laplace=10,
                                metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE,
                                mo_coeff=Ref.mo_coeff, mo_energy=Ref.mo_energy).energy

    for ia in range(mol.natm):
        for k in range(3):
            dp, dm = R0.copy(), R0.copy()
            dp[ia, k] += h
            dm[ia, k] -= h
            num = (energy_at(dp) - energy_at(dm)) / (2.0 * h)
            assert abs(res.de[ia, k] - num) < 1e-7, f"atom {ia} component {k}"


@pytest.mark.parametrize("scheme", ["ridge_jacobi", "damped_jacobi"])
def test_nuclear_gradient_under_jacobi_preconditioning(system, scheme):
    """
    The same check with the metric Jacobi-preconditioned, where the scaling has to move.

    On a random matrix the preconditioner is a fixed diagonal; on a molecule it is not.
    ``diag(S)_PP = (sum_mu X_muP^2)^2`` is a function of the nuclear coordinates, so a
    gradient that treats ``E`` as a constant is wrong on the only input that matters -
    which is exactly how the scheme shipped, and why FINDINGS section 17 has no torque.
    """
    mol, mf, per_atom = system
    kwargs = dict(n_laplace=10, metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE,
                  metric_scheme=scheme)
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS, **kwargs)

    class Ref:
        mo_coeff = np.array(mf.mo_coeff)
        mo_energy = np.array(mf.mo_energy)

    R0 = mol.atom_coords()
    h = 1e-4

    def energy_at(coords):
        m = mol.copy(); m.set_geom_(coords, unit="Bohr"); m.build(False, False)
        return thc_mp2_gradient(m, Ref, FrozenGrid(m, per_atom), AUXBASIS,
                                mo_coeff=Ref.mo_coeff, mo_energy=Ref.mo_energy,
                                **kwargs).energy

    for ia in range(mol.natm):
        for k in range(3):
            dp, dm = R0.copy(), R0.copy()
            dp[ia, k] += h
            dm[ia, k] -= h
            num = (energy_at(dp) - energy_at(dm)) / (2.0 * h)
            assert abs(res.de[ia, k] - num) < 1e-7, f"atom {ia} component {k}"


def test_jacobi_torque_against_rotating_the_point_set(system):
    """
    The measurement section 17 was left unable to make.

    A preconditioned metric is the one thing that can replace the weights a transferable
    support cannot carry, and orientation is the quantity that decides whether the
    support is usable - so ``dE/dtheta`` under ``"_jacobi"`` is the number the section
    needs. It is meaningless unless it is a derivative, which is what this asserts.
    """
    from scipy.spatial.transform import Rotation

    mol, mf, per_atom = system
    kwargs = dict(n_laplace=10, metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE,
                  metric_scheme="damped_jacobi")
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS, **kwargs)

    class Ref:
        mo_coeff = np.array(mf.mo_coeff)
        mo_energy = np.array(mf.mo_energy)

    delta = 1e-4
    for ia in range(mol.natm):
        tau = res.torque[ia]
        norm = float(np.linalg.norm(tau))
        axis = tau / norm

        vals = []
        for sign in (+1, -1):
            rots = [np.eye(3)] * mol.natm
            rots[ia] = Rotation.from_rotvec(sign * delta * axis).as_matrix()
            vals.append(thc_mp2_gradient(mol, Ref, FrozenGrid(mol, per_atom, rots),
                                         AUXBASIS, mo_coeff=Ref.mo_coeff,
                                         mo_energy=Ref.mo_energy, **kwargs).energy)

        num = (vals[0] - vals[1]) / (2.0 * delta)
        assert abs(norm - num) < 1e-4 * abs(num), f"atom {ia}"


def test_gradient_is_translationally_invariant(system):
    """
    The forces sum to zero, exactly and without any finite difference.

    This is what catches a mis-signed or mis-mapped rigid-translation term: under a
    uniform shift the point-translation and AO-derivative halves of the collocation
    gradient are equal and opposite, so an error in either cannot cancel.
    """
    mol, mf, per_atom = system
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS, n_laplace=10,
                           metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE)

    assert np.linalg.norm(res.de.sum(axis=0)) < 1e-10 * np.linalg.norm(res.de)


def test_torque_against_rotating_the_point_set(system):
    """
    dE/dtheta against actually rotating each atom's grid about its own nucleus.

    Exact on both sides: rotating a point set leaves the molecule, the AOs and the SCF
    untouched, so no orbital response enters either the analytic or the numerical value.
    """
    from scipy.spatial.transform import Rotation

    mol, mf, per_atom = system
    res = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), AUXBASIS, n_laplace=10,
                           metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE)

    class Ref:
        mo_coeff = np.array(mf.mo_coeff)
        mo_energy = np.array(mf.mo_energy)

    delta = 1e-4
    for ia in range(mol.natm):
        tau = res.torque[ia]
        norm = float(np.linalg.norm(tau))
        axis = tau / norm

        vals = []
        for sign in (+1, -1):
            rots = [np.eye(3)] * mol.natm
            rots[ia] = Rotation.from_rotvec(sign * delta * axis).as_matrix()
            vals.append(thc_mp2_gradient(mol, Ref, FrozenGrid(mol, per_atom, rots),
                                         AUXBASIS, n_laplace=10,
                                         metric_ridge=TEST_RIDGE, aux_ridge=TEST_RIDGE,
                                         mo_coeff=Ref.mo_coeff,
                                         mo_energy=Ref.mo_energy).energy)

        num = (vals[0] - vals[1]) / (2.0 * delta)
        assert abs(norm - num) < 1e-4 * abs(num), f"atom {ia}"


def test_frozen_grid_translates_rigidly(system):
    """A displaced FrozenGrid must move its points with the nuclei and change nothing else."""
    mol, _, per_atom = system
    grid = FrozenGrid(mol, per_atom)
    coords0, weights0 = grid.build()

    shift = np.array([0.3, -0.2, 0.7])
    moved = grid.displaced(mol.atom_coords() + shift)
    coords1, weights1 = moved.build()

    assert np.allclose(coords1, coords0 + shift)
    assert np.allclose(weights1, weights0)
    assert np.array_equal(moved.atom_index, grid.atom_index)

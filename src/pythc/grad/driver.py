"""
The assembled gradient: one reverse pass from the correlation energy to the nuclei.

:func:`thc_mp2_gradient` runs the whole chain - forward through the LS-THC fit and the
Laplace MP2 expression, then backward through both and out into derivative integrals and
the collocation derivative. :func:`orientation_gradient` reuses the same machinery for
the rotational derivative, which needs nothing else.

**What the returned gradient is.** ``de`` is the derivative of the *correlation* energy
at a fixed SCF reference: MO coefficients and orbital energies are held at the values
passed in. That is the THC-specific content of the gradient and the part that had never
been written down - it contains every term the frozen grid, the collocation, the metric,
the ridge and the fitted ``Z`` contribute. It is **not** the total MP2 gradient on its
own: the orbital response and the Hartree-Fock gradient are separate and standard.
:func:`pythc.grad.total.total_mp2_gradient` assembles all three, and
:mod:`pythc.grad.response` supplies the response from the ``C_bar``, ``F_oo_bar`` and
``F_vv_bar`` returned here.

Keeping the split explicit is deliberate rather than a shortcut. The fixed-orbital
derivative is exactly finite-difference checkable on its own - hold ``C`` and ``e`` fixed
and move the nuclei - so the novel machinery could be verified to machine precision
without the response layer sitting in the way. It stays available separately because the
difference between the two is a quantity worth measuring: on water it is 18% of the
fixed-orbital correlation gradient's norm, and it is the whole of that gradient's
rotational-invariance error.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pyscf import gto

from pythc.grad.factorisation import ThcFactorisation
from pythc.grad.frozen import FrozenGrid
from pythc.grad.geometry import (collocation_gradient, orientation_torque, point_forces,
                                 three_center_gradient, two_center_gradient)
from pythc.grad.laplace_mp2 import energy_and_adjoints, laplace_factors

logger = logging.getLogger(__name__)


@dataclass
class ThcGradientResult:
    """Everything one reverse pass produces."""

    energy: float
    """The LS-THC MP2 correlation energy, by the same expression
    :func:`pythc.methods.mp2.mp2_energy_laplace` evaluates."""

    de: np.ndarray
    """``(n_atm, 3)`` fixed-orbital nuclear gradient of the correlation energy, Hartree
    per Bohr."""

    de_collocation: np.ndarray
    """The part of ``de`` from ``dX_ao/dR``: rigid point translation plus AO derivative.
    This is the frozen grid's own contribution."""

    de_three_center: np.ndarray
    """The part of ``de`` from the derivative ``(mu nu|M)`` integrals."""

    de_two_center: np.ndarray
    """The part of ``de`` from the derivative ``(M|N)`` auxiliary metric."""

    torque: np.ndarray
    """``(n_atm, 3)`` torque on each atom's frozen point set about its own nucleus. Zero
    would mean the attached orientation is a stationary point of the energy; it is not,
    and its size is what orientation dependence costs a gradient. See
    :func:`~pythc.grad.geometry.orientation_torque`."""

    C_bar: np.ndarray
    """``dE/dC``, ``(n_ao, n_mo)``, for the orbital-response layer."""

    eps_bar: np.ndarray
    """``dE/de``, ``(n_mo,)``, for the orbital-response layer."""

    F_oo_bar: np.ndarray = None
    """``dE/dF_oo``, ``(n_occ, n_occ)``. The matrix generalisation of ``eps_bar``'s
    occupied part, and what :mod:`pythc.grad.response` actually needs: the energy's
    dependence on ``e_i`` is really a dependence on the occupied Fock block, and off the
    diagonal the two differ. ``diag(F_oo_bar) == eps_bar[:n_occ]``."""

    F_vv_bar: np.ndarray = None
    """``dE/dF_vv``, ``(n_vir, n_vir)``. Likewise."""

    n_points: int = 0
    translation_only: np.ndarray = field(default=None)
    """``de_collocation`` with the AO-derivative term dropped, i.e. the pure rigid
    translation of the frozen points. Reported because it is the term that exists only
    by virtue of freezing the grid, and its size says how much of the gradient the
    freeze is responsible for."""


def _resolve_reference(mf, mo_coeff, mo_energy):
    coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
    energy = mf.mo_energy if mo_energy is None else mo_energy

    return np.asarray(coeff, dtype=float), np.asarray(energy, dtype=float)


def thc_mp2_gradient(mol: gto.Mole,
                     mf,
                     grid: FrozenGrid,
                     auxbasis: str,
                     n_laplace: int = 10,
                     metric_ridge: Optional[float] = None,
                     aux_ridge: Optional[float] = None,
                     ridge_scale: str = "trace",
                     mo_coeff: Optional[np.ndarray] = None,
                     mo_energy: Optional[np.ndarray] = None,
                     metric_scheme: str = "ridge") -> ThcGradientResult:
    """
    Fixed-orbital analytic nuclear gradient of the frozen-grid LS-THC MP2 energy.

    :param mol: The molecule, at the geometry the gradient is wanted at.
    :param mf: Converged SCF object supplying ``mo_coeff`` and ``mo_energy``.
    :param grid: The frozen grid, already attached to ``mol``.
    :param auxbasis: Auxiliary basis for the DF target integrals.
    :param n_laplace: Laplace quadrature points; must match the energy being compared to.
    :param metric_ridge: Ridge on the THC metric, or ``None`` for the truncated
        pseudoinverse. Passing ``None`` yields the derivative *at fixed retained
        subspace*, which is the right answer between eigenvalue crossings and no answer
        at all on one; see :func:`pythc.grad.linalg.pinv_adjoint`.
    :param aux_ridge: Ridge on the auxiliary Coulomb metric.
    :param ridge_scale: How the ridge strengths become absolute shifts.
    :param mo_coeff: Override for ``mf.mo_coeff``. The finite-difference check uses this
        to hold the orbitals at their reference-geometry values while the nuclei move.
    :param mo_energy: Override for ``mf.mo_energy``.
    :param metric_scheme: Which regularised inverse to use where a ridge is requested:
        ``"ridge"`` for ``(S + lambda I)^-1``, ``"damped"`` for the Tikhonov-filtered
        ``S (S^2 + mu^2 I)^-1``, which sends the numerically null directions to zero
        instead of to the largest gain in the operator, either with a ``"_jacobi"``
        suffix for the symmetrically preconditioned form. The preconditioned filter is
        invariant to the collocation weights, which for a frozen grid is the difference
        between a support that has to carry fitted weights and one that does not. See
        :func:`pythc.lib.damped_inv` and :func:`pythc.lib.jacobi_scaling`.
    """
    C, eps = _resolve_reference(mf, mo_coeff, mo_energy)
    n_occ = mol.nelectron // 2

    coords, weights = grid.build()
    atom_index = grid.atom_index

    fac = ThcFactorisation(mol, coords, weights, C, n_occ, auxbasis,
                           metric_ridge=metric_ridge, aux_ridge=aux_ridge,
                           ridge_scale=ridge_scale,
                           metric_scheme=metric_scheme).build()

    t, tau_o, tau_v = laplace_factors(eps, n_occ, n_laplace)
    (e, X_o_bar, X_v_bar, Z_bar, eps_o_bar, eps_v_bar,
     F_oo_bar, F_vv_bar) = energy_and_adjoints(
        fac.X_o, fac.X_v, fac.Z, tau_o, tau_v, t,
        want_fock_adjoints=True, eps_o=eps[:n_occ], eps_v=eps[n_occ:])

    adj = fac.backward(Z_bar, X_o_bar, X_v_bar)

    de_coll = collocation_gradient(mol, coords, fac.amp, atom_index, adj.X_ao_bar)
    de_3c = three_center_gradient(mol, fac.auxmol, adj.V_bar)
    de_2c = two_center_gradient(mol, fac.auxmol, adj.j2c_bar)
    torque = orientation_torque(mol, coords, fac.amp, atom_index, adj.X_ao_bar)

    # The translation half of the collocation term on its own - each point's force
    # gathered onto the nucleus it rides with, without the AO-centre counterterm.
    translation = np.zeros((mol.natm, 3))
    np.add.at(translation, atom_index, point_forces(mol, coords, fac.amp, adj.X_ao_bar))

    return ThcGradientResult(
        energy=float(e),
        de=de_coll + de_3c + de_2c,
        de_collocation=de_coll,
        de_three_center=de_3c,
        de_two_center=de_2c,
        torque=torque,
        C_bar=adj.C_bar,
        eps_bar=np.concatenate([eps_o_bar, eps_v_bar]),
        F_oo_bar=F_oo_bar,
        F_vv_bar=F_vv_bar,
        n_points=len(coords),
        translation_only=translation,
    )


def orientation_gradient(mol: gto.Mole,
                         mf,
                         grid: FrozenGrid,
                         auxbasis: str,
                         n_laplace: int = 10,
                         metric_ridge: Optional[float] = None,
                         aux_ridge: Optional[float] = None,
                         ridge_scale: str = "trace",
                         mo_coeff: Optional[np.ndarray] = None,
                         mo_energy: Optional[np.ndarray] = None,
                         metric_scheme: str = "ridge") -> np.ndarray:
    """
    The torque on each atom's frozen point set, ``(n_atm, 3)`` in Hartree per radian.

    Exact - no orbital response enters, because rotating a grid about its own nucleus
    leaves the SCF problem untouched. See :func:`~pythc.grad.geometry.orientation_torque`
    for why this rather than a rotation spread is the quantity that matters.
    """
    return thc_mp2_gradient(mol, mf, grid, auxbasis, n_laplace, metric_ridge,
                            aux_ridge, ridge_scale, mo_coeff, mo_energy,
                            metric_scheme).torque

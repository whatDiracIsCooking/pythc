"""
The orbital-response layer: what the fixed-orbital gradient deliberately leaves out.

:func:`~pythc.grad.driver.thc_mp2_gradient` differentiates the correlation energy at a
*fixed* SCF reference. That is the THC-specific content of the gradient, and it is exact,
but it is not the derivative of the energy an AIMD trajectory actually propagates on: the
MO coefficients and orbital energies move when the nuclei do. This module supplies the
missing piece,

    dE/dR = dE/dR|_(C, F)  +  sum_pq Lmo[q,p] U[q,p]  +  tr(F_bar dF/dR)

and there is no THC-specific theory in any of it - it is the same coupled-perturbed
Hartree-Fock machinery a DF-MP2 gradient needs, which is why the split was kept explicit
in the first place.

**The reformulation that makes this tractable.** Written with orbital energies in the
Laplace factors, the THC-MP2 energy is *not* invariant under a rotation among the
occupied orbitals, so its response would need the occupied-occupied and virtual-virtual
blocks of ``U`` - which a CPHF solver does not return, and which the canonical condition
pins down only through a second coupled equation of their own. Written with
``Theta_o = w^(1/4) exp(t F_oo)`` in place of ``diag(w^(1/4) exp(t e_i))`` it is the same
function at the canonical point but manifestly invariant, and those blocks then enter
only through the overlap derivative, which is a skeleton quantity. What is left needing
CPHF is the occupied-virtual block alone - exactly what
:func:`pyscf.hessian.rhf.solve_mo1` hands back. See
:func:`pythc.grad.laplace_mp2.loewner` for the adjoint of that reformulation.

**Why this one is not finite-difference checkable stage by stage.** The fixed-orbital
gradient could be checked against a displaced energy with the orbitals pinned, one adjoint
at a time. The response cannot: it only exists relative to a re-converged SCF, so the only
honest test is the total gradient against a finite difference of the total energy with the
SCF re-solved at every displaced geometry. That is what
``tests/test_thc_response.py`` does, and it is why that test is slow.

**Cost.** :func:`orbital_response_gradient` solves CPHF once per nuclear degree of
freedom, which is Hessian-level work: correct, and the reference the cheaper route is
checked against, but ``3 N`` solves rather than one. :func:`response_lagrangian` exposes
the intermediate a Z-vector implementation would contract instead, which is where that
work would start.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResponseResult:
    """The response contribution, and the diagnostics that say it is meaningful."""

    de: np.ndarray
    """``(n_atm, 3)`` orbital-response contribution to the correlation gradient,
    Hartree per Bohr. Add it to :attr:`~pythc.grad.driver.ThcGradientResult.de`."""

    asymmetry: float
    """How far the occupied-occupied and virtual-virtual blocks of the MO Lagrangian are
    from symmetric, relative to their own norm.

    This is the reformulation's own check on itself, and it is not a loose one. The
    invariant energy cannot depend on an antisymmetric rotation within either space, so
    those blocks *must* come out symmetric - if they do not, the Fock adjoint is wrong and
    the response is being built on a quantity that does not mean what it should. It is
    free to compute and it fails loudly, so it is always computed."""

    n_cphf: int
    """How many coupled-perturbed solutions were taken. ``3 n_atm`` for the explicit
    route, ``1`` for the Z-vector one."""


def _skeleton_overlap(mol, s1a: np.ndarray, ia: int) -> np.ndarray:
    """``(3, n_ao, n_ao)`` skeleton derivative of the AO overlap w.r.t. atom ``ia``."""
    p0, p1 = mol.aoslice_by_atom()[ia][2:4]
    out = np.zeros((3, mol.nao, mol.nao))
    out[:, p0:p1] += s1a[:, p0:p1]
    out[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)

    return out


def response_lagrangian(mf, C_bar: np.ndarray, F_oo_bar: np.ndarray,
                        F_vv_bar: np.ndarray, mo_coeff=None, mo_energy=None):
    """
    Assemble the MO-basis Lagrangian and the Fock adjoint the response contracts.

    :param mf: The converged SCF object. Used for the overlap, and for the two-electron
        response operator ``G`` - which is taken from ``mf`` rather than rebuilt so that a
        density-fitted reference is differentiated with its own integrals.
    :param C_bar: ``dE/dC`` at fixed orbital energies, from
        :attr:`~pythc.grad.driver.ThcGradientResult.C_bar`.
    :param F_oo_bar: ``dE/dF_oo``, from
        :func:`~pythc.grad.laplace_mp2.energy_and_adjoints` with ``want_fock_adjoints``.
    :param F_vv_bar: ``dE/dF_vv``, likewise.
    :return: ``(Lmo, F_bar, GF_mo, asymmetry)``.

    ``Lmo`` is ``dE/dC`` in the MO basis *at fixed Fock matrix* - which is not the same
    thing as ``C_bar``, because holding ``F`` fixed while moving ``C`` still moves
    ``F_oo = C_o^T F C_o``. The correction is ``2 e_p (dE/dF)[p,q]``, using
    ``F C = S C diag(e)``.
    """
    C = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff, dtype=float)
    eps = np.asarray(mf.mo_energy if mo_energy is None else mo_energy, dtype=float)
    n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    n_mo = C.shape[1]

    # dE/dF in the MO basis is block diagonal: the energy sees the Fock matrix only
    # through F_oo and F_vv.
    Fbar_mo = np.zeros((n_mo, n_mo))
    Fbar_mo[:n_occ, :n_occ] = F_oo_bar
    Fbar_mo[n_occ:, n_occ:] = F_vv_bar

    Lmo = C.T @ np.asarray(C_bar, dtype=float) + 2.0 * eps[:, None] * Fbar_mo

    # The invariance the whole reformulation rests on, checked rather than assumed.
    blocks = [Lmo[:n_occ, :n_occ], Lmo[n_occ:, n_occ:]]
    num = max(float(np.abs(b - b.T).max()) for b in blocks)
    den = max(float(np.abs(b).max()) for b in blocks)
    asymmetry = num / den if den > 0 else 0.0

    # F_bar is symmetric, so the RHF response operator G[F_bar] = J - K/2 applies to it
    # directly. Taken from mf so a density-fitted reference uses its own integrals.
    F_bar = C @ Fbar_mo @ C.T
    vj, vk = mf.get_jk(mf.mol, F_bar)
    GF = vj - 0.5 * vk

    return Lmo, F_bar, GF, asymmetry


def orbital_response_gradient(mol, mf, C_bar: np.ndarray, F_oo_bar: np.ndarray,
                              F_vv_bar: np.ndarray,
                              atmlst: Optional[list] = None) -> ResponseResult:
    """
    The orbital-response contribution to the correlation gradient.

    :param mol: The molecule, at the geometry the gradient is wanted at.
    :param mf: The converged SCF object the correlation energy was built on.
    :param C_bar: ``dE/dC`` at fixed orbital energies.
    :param F_oo_bar: ``dE/dF_oo``.
    :param F_vv_bar: ``dE/dF_vv``.
    :param atmlst: Atoms to compute, or ``None`` for all.

    Solves the coupled-perturbed equations explicitly, once per nuclear degree of freedom.
    That is the expensive route and deliberately the first one: it is the reference a
    Z-vector implementation has to reproduce, and it has no transposed-operator algebra in
    it to get wrong.
    """
    C = np.asarray(mf.mo_coeff, dtype=float)
    n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    C_occ = C[:, :n_occ]
    S = mf.get_ovlp(mol)

    Lmo, F_bar, GF, asymmetry = response_lagrangian(mf, C_bar, F_oo_bar, F_vv_bar)
    if asymmetry > 1e-6:
        logger.warning("MO Lagrangian is %.2e from symmetric in the oo/vv blocks; the "
                       "Fock adjoint is probably wrong", asymmetry)

    hess = mf.Hessian()
    h1ao = hess.make_h1(mf.mo_coeff, mf.mo_occ, atmlst=atmlst)
    mo1, _ = hess.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao,
                            atmlst=atmlst)

    s1a = -mol.intor("int1e_ipovlp", comp=3)
    atoms = range(mol.natm) if atmlst is None else atmlst
    de = np.zeros((mol.natm, 3))

    # C^T S, applied to the AO-basis mo1 PySCF returns, recovers U. PySCF's gauge puts
    # -S^x/2 in the occupied-occupied block, which is the symmetric part the constraint
    # fixes; its antisymmetric part is a gauge the invariant energy cannot see.
    CtS = C.T @ S

    for ia in atoms:
        Sx_ao = _skeleton_overlap(mol, s1a, ia)

        for x in range(3):
            Sx = C.T @ Sx_ao[x] @ C
            U_oi = CtS @ mo1[ia][x]                    # (n_mo, n_occ)

            # The full U. Only the occupied-virtual block needed a coupled solve; every
            # other block is fixed by U_pq + U_qp = -S^x_pq once invariance has removed
            # the antisymmetric parts within each space.
            U = np.zeros_like(Lmo)
            U[:, :n_occ] = U_oi
            U[:n_occ, n_occ:] = -Sx[:n_occ, n_occ:] - U_oi[n_occ:, :].T
            U[n_occ:, n_occ:] = -0.5 * Sx[n_occ:, n_occ:]

            # dP/dR from the same mo1, which already carries the right occupied gauge.
            dP = 2.0 * (mo1[ia][x] @ C_occ.T + C_occ @ mo1[ia][x].T)

            # Three routes: C moves; the skeleton Fock derivative at fixed density; and
            # the Fock matrix moving because the density did. The last is written
            # tr(dP G[F_bar]) rather than tr(F_bar G[dP]) - G is self-adjoint on
            # symmetric matrices, and this way G is applied once instead of per atom.
            de[ia, x] = (float(np.sum(Lmo * U))
                         + float(np.sum(F_bar * h1ao[ia][x]))
                         + float(np.sum(dP * GF)))

    n_cphf = 3 * (mol.natm if atmlst is None else len(atmlst))

    return ResponseResult(de=de, asymmetry=asymmetry, n_cphf=n_cphf)

"""
Turning the factorisation's adjoints into derivatives with respect to nuclear positions.

Three things in the pipeline depend on the geometry, and this module differentiates each
of them. The conventions below were fixed by finite-differencing PySCF's integrals rather
than read off documentation, and the checks live in ``tests/test_thc_gradient.py``:

* ``GTOval_ip_sph`` returns ``+grad_r phi``, so an AO centred on atom ``A`` satisfies
  ``d phi / dR_A = -grad_r phi``;
* ``int3c2e_ip1`` and ``int3c2e_ip2`` likewise differentiate with respect to the electron
  coordinate of the bra AO and of the auxiliary function, so each contributes with a
  minus sign for the functions sitting on the moving atom;
* ``int2c2e_ip1`` the same, for the auxiliary Coulomb metric.

**The collocation term is the one the whole programme is about.** ``X_ao[P,mu] =
w_P^(1/4) phi_mu(r_P)`` depends on ``R_A`` twice over, and the two contributions have
opposite sign and different support:

    dX_ao[P,mu] / dR_A = w_P^(1/4) grad phi_mu(r_P) * ( [P rides with A] - [mu sits on A] )

The second term is the ordinary AO-derivative any basis-set method has. The first exists
only because the grid is frozen and attached to nuclei - it is the rigid translation of
the target pipeline's second row. There is no third term, and that absence is the
proposal: a grid re-selected at each geometry would need one, and it would not be a
derivative at all.
"""
import logging

import numpy as np
from pyscf import df, gto

logger = logging.getLogger(__name__)


def basis_to_atom(mol: gto.Mole) -> np.ndarray:
    """``(n_basis,)`` array giving the atom each basis function is centred on."""
    index = np.empty(mol.nao_nr(), dtype=np.intp)
    for ia, (_, _, ao0, ao1) in enumerate(mol.aoslice_by_atom()):
        index[ao0:ao1] = ia

    return index


def point_forces(mol: gto.Mole, coords: np.ndarray, amp: np.ndarray,
                 X_ao_bar: np.ndarray) -> np.ndarray:
    """
    ``dE/dr_P``, the derivative with respect to each grid point's own position.

    This is the object both the translation term and the torque are built from: moving a
    point is the same operation whether a nucleus dragged it or a rotation did.

    :param amp: ``w_P^(1/4)``, the weight factor folded into the collocation matrix.
    :param X_ao_bar: ``dE/dX_ao``, shape ``(n_grid, n_ao)``.
    :return: ``(n_grid, 3)``.
    """
    ip = mol.eval_gto("GTOval_ip_sph", coords)      # (3, n_grid, n_ao), = +grad_r phi
    scaled = amp[:, None] * X_ao_bar

    return np.einsum("kgm,gm->gk", ip, scaled, optimize=True)


def collocation_gradient(mol: gto.Mole, coords: np.ndarray, amp: np.ndarray,
                         atom_index: np.ndarray, X_ao_bar: np.ndarray) -> np.ndarray:
    """
    Nuclear gradient of the collocation matrix, both terms.

    :param atom_index: ``(n_grid,)``, which atom each point rides with. This is what a
        frozen grid knows and a re-selected one does not.
    :return: ``(n_atm, 3)``.
    """
    ip = mol.eval_gto("GTOval_ip_sph", coords)
    contrib = ip * (amp[:, None] * X_ao_bar)[None, :, :]        # (3, n_grid, n_ao)

    de = np.zeros((mol.natm, 3))

    # Points ride with their nucleus: sum over the points belonging to each atom.
    per_point = contrib.sum(axis=2)                              # (3, n_grid)
    np.add.at(de, atom_index, per_point.T)

    # AOs sit on their nucleus, with the opposite sign.
    per_ao = contrib.sum(axis=1)                                 # (3, n_ao)
    ao_atom = basis_to_atom(mol)
    np.subtract.at(de, ao_atom, per_ao.T)

    return de


def orientation_torque(mol: gto.Mole, coords: np.ndarray, amp: np.ndarray,
                       atom_index: np.ndarray, X_ao_bar: np.ndarray) -> np.ndarray:
    """
    The torque on each atom's frozen point set, about its own nucleus.

    Spinning an atom's grid about its nucleus moves nothing else: the molecule, the AOs,
    the auxiliary basis and the SCF solution are all untouched, so this derivative is
    exact with no orbital-response term. With ``g_P = dE/dr_P`` and ``s_P`` the point's
    position relative to its nucleus, the energy's response to a rotation by angle
    ``theta`` about a unit axis ``n`` is ``n . tau_A`` with

        tau_A = sum_{P on A} s_P x g_P

    HANDOFF section 4(2) measured orientation dependence as an energy *spread* over
    random draws. A spread does not bound a derivative - a small-amplitude but rapidly
    varying function of orientation has a tiny spread and a large torque - and it is the
    torque, not the spread, that decides whether angular momentum is conserved in
    dynamics. This is that quantity.

    :return: ``(n_atm, 3)``, one torque vector per atom, in Hartree per radian.
    """
    g = point_forces(mol, coords, amp, X_ao_bar)
    rel = coords - mol.atom_coords()[atom_index]

    tau = np.zeros((mol.natm, 3))
    np.add.at(tau, atom_index, np.cross(rel, g))

    return tau


def three_center_gradient(mol: gto.Mole, auxmol: gto.Mole,
                          V_bar: np.ndarray) -> np.ndarray:
    """
    ``sum_{mu nu M} V_bar[mu,nu,M] d(mu nu|M)/dR``.

    The bra and ket AO indices both contribute through ``int3c2e_ip1``; the integral is
    symmetric in them, so their two contributions combine into one contraction against
    ``V_bar + V_bar^T``.
    """
    n_ao, n_aux = mol.nao_nr(), auxmol.nao_nr()
    ip1 = df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip1", aosym="s1",
                           comp=3).reshape(3, n_ao, n_ao, n_aux)
    ip2 = df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip2", aosym="s1",
                           comp=3).reshape(3, n_ao, n_ao, n_aux)

    V_sym = V_bar + V_bar.transpose(1, 0, 2)
    ao_atom = basis_to_atom(mol)
    aux_atom = basis_to_atom(auxmol)

    de = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        on_ao = ao_atom == ia
        on_aux = aux_atom == ia
        if np.any(on_ao):
            de[ia] -= np.einsum("kabq,abq->k", ip1[:, on_ao], V_sym[on_ao],
                                optimize=True)
        if np.any(on_aux):
            de[ia] -= np.einsum("kabq,abq->k", ip2[:, :, :, on_aux],
                                V_bar[:, :, on_aux], optimize=True)

    return de


def two_center_gradient(mol: gto.Mole, auxmol: gto.Mole,
                        j2c_bar: np.ndarray) -> np.ndarray:
    """``sum_{MN} j2c_bar[M,N] d(M|N)/dR``."""
    ip1 = auxmol.intor("int2c2e_ip1", comp=3)
    j_sym = j2c_bar + j2c_bar.T
    aux_atom = basis_to_atom(auxmol)

    de = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        on_aux = aux_atom == ia
        if np.any(on_aux):
            de[ia] -= np.einsum("kmn,mn->k", ip1[:, on_aux], j_sym[on_aux],
                                optimize=True)

    return de

"""
The Laplace THC-MP2 energy and its adjoints with respect to ``X``, ``Z`` and the orbital
energies.

This is the top of the reverse-mode chain. :func:`energy_and_adjoints` recomputes the
energy that :func:`pythc.methods.mp2.mp2_energy_laplace` computes - and is checked
against it to machine precision in the tests - while also returning

* ``X_o_bar``, ``X_v_bar``: ``dE/dX`` in the MO basis, which flows on into the metric and
  the collocation matrix;
* ``Z_bar``: ``dE/dZ``, which flows into the least-squares fit;
* ``eps_o_bar``, ``eps_v_bar``: ``dE/de``, which the orbital-response layer needs and
  which is free to compute here.

Conventions follow the energy code exactly. With ``t_v`` and ``w_v`` the Laplace nodes and
weights, ``tau_o[v,i] = w_v^(1/4) exp(+t_v e_i)`` and ``tau_v[v,a] = w_v^(1/4) exp(-t_v e_a)``,
and per node

    Ao = X_o diag(tau_o) X_o^T,     Av = X_v diag(tau_v) X_v^T
    J_v = tr(Ao o Av  Z  Ao o Av  Z)
    K_v = sum_PQRS Z_PQ Z_RS Ao[P,R] Av[P,S] Ao[Q,S] Av[Q,R]
    E   = sum_v (-2 J_v + K_v)

``J_v`` is the trace form the energy code evaluates directly. ``K_v`` is written here
through the same intermediate the energy code uses,

    Psi[P,R,i] = sum_Q Z[P,Q] X_o[Q,i] Av[Q,R]
    K_v        = sum_PR Ao[P,R] sum_i tau_o_i Psi[P,R,i] Psi[R,P,i]

which turns a formally quartic contraction over grid points into two GEMMs of cost
``O(n_P^2 n_occ)`` - the same trick, and the same reason, as
:func:`pythc.methods.mp2._calculate_mp2_K_optimized`. The price is an ``(n_P, n_P, n_occ)``
intermediate; see :func:`peak_memory_estimate` before running this on a large grid.
"""
import logging
from typing import Optional

import numpy as np
from pyscf.gw.gw_ac import _get_scaled_legendre_roots

logger = logging.getLogger(__name__)


def laplace_factors(mo_energy: np.ndarray, n_occ: int, n_laplace: int = 10):
    """
    The Laplace nodes and the ``tau`` factors the energy is built from.

    :param mo_energy: Orbital energies, in the order the THC ``X`` uses.
    :param n_occ: Number of occupied orbitals.
    :param n_laplace: Number of quadrature points.
    :return: ``(t, tau_o, tau_v)`` with ``t`` of shape ``(n_laplace,)`` and the ``tau``
        arrays of shape ``(n_laplace, n_occ)`` and ``(n_laplace, n_vir)``.
    """
    t, w = _get_scaled_legendre_roots(n_laplace)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)

    e_o = np.asarray(mo_energy, dtype=float)[:n_occ]
    e_v = np.asarray(mo_energy, dtype=float)[n_occ:]

    tau_o = np.power(w[:, None], 0.25) * np.exp(+t[:, None] * e_o[None, :])
    tau_v = np.power(w[:, None], 0.25) * np.exp(-t[:, None] * e_v[None, :])

    return t, tau_o, tau_v


def loewner(eps: np.ndarray, factors: np.ndarray, t: float) -> np.ndarray:
    """
    The Löwner (divided-difference) matrix of ``f(e) = w^(1/4) exp(t e)``.

    This is the Fréchet derivative of a matrix exponential evaluated at a *diagonal*
    argument, which is what the canonical basis makes the occupied and virtual Fock
    blocks. It is the one piece of machinery the orbital-response layer needs that the
    fixed-orbital gradient did not: the energy's dependence on ``e_i`` is really a
    dependence on the Fock matrix ``F_oo``, and off the diagonal those differ.

        L[i,j] = (f_i - f_j) / (e_i - e_j)   i != j,      L[i,i] = f'(e_i) = t f_i

    :param eps: Orbital energies of one block, ``(n,)``.
    :param factors: ``f(e_i)`` for the same block - a row of ``tau_o`` or ``tau_v``.
    :param t: The Laplace node, signed: ``+t`` for the occupied factors and ``-t`` for
        the virtual ones, matching ``tau_o = w^(1/4) exp(+t e)`` and
        ``tau_v = w^(1/4) exp(-t e)``.
    :return: The symmetric ``(n, n)`` divided-difference matrix.

    Neither closed form is usable on its own over the spread of a real virtual spectrum.
    The literal difference quotient loses all its digits as ``e_i -> e_j``, while the
    stable ``t f_j expm1(d)/d`` with ``d = t (e_i - e_j)`` overflows once ``d`` is large -
    which it is for a virtual block spanning tens of Hartree. So each is used where it is
    good: ``expm1`` inside ``|d| < 1``, where cancellation is the danger and no overflow
    is possible, and the plain quotient outside it, where the two values differ by enough
    that subtracting them is exact. The ``expm1`` form is symmetric under ``i <-> j``
    because ``f_i exp(-d) = f_j``.
    """
    diff = eps[:, None] - eps[None, :]
    d = t * diff
    small = np.abs(d) < 1.0

    # t f_j expm1(d)/d, continued to its limit t f_j at d = 0. Guarded so the large-|d|
    # entries, which this branch does not supply, never reach expm1 or the division.
    d_s = np.where(small, d, 1.0)
    ratio = np.where(np.abs(d_s) < 1e-12, 1.0, np.expm1(d_s) / np.where(d_s == 0, 1.0, d_s))
    near = t * factors[None, :] * ratio

    # (f_i - f_j) / (e_i - e_j), exact once the two are well separated.
    far = (factors[:, None] - factors[None, :]) / np.where(small, 1.0, diff)

    return np.where(small, near, far)


def peak_memory_estimate(n_grid: int, n_occ: int) -> float:
    """
    Rough peak working-set of :func:`energy_and_adjoints`, in bytes.

    The pass holds three ``(n_grid, n_grid, n_occ)`` intermediates at once (``Psi``, its
    adjoint, and the back-propagated ``Phi``) plus a handful of ``n_grid^2`` matrices.
    This is a reference implementation for grids of a few hundred to a couple of thousand
    points; it is not the production contraction order.
    """
    return 8.0 * (3.0 * n_grid * n_grid * n_occ + 8.0 * n_grid * n_grid)


def energy_and_adjoints(X_o: np.ndarray,
                        X_v: np.ndarray,
                        Z: np.ndarray,
                        tau_o: np.ndarray,
                        tau_v: np.ndarray,
                        t: np.ndarray,
                        want_orbital_adjoints: bool = True,
                        want_fock_adjoints: bool = False,
                        eps_o: Optional[np.ndarray] = None,
                        eps_v: Optional[np.ndarray] = None):
    """
    The Laplace THC-MP2 correlation energy together with its first derivatives.

    :param X_o: Occupied collocation block, ``(n_grid, n_occ)``.
    :param X_v: Virtual collocation block, ``(n_grid, n_vir)``.
    :param Z: THC kernel, ``(n_grid, n_grid)``. Assumed symmetric, as ``D^T D`` is.
    :param tau_o: ``(n_laplace, n_occ)`` occupied Laplace factors.
    :param tau_v: ``(n_laplace, n_vir)`` virtual Laplace factors.
    :param t: ``(n_laplace,)`` Laplace nodes, needed only for the orbital-energy adjoint.
    :param want_orbital_adjoints: Compute ``dE/de`` as well. Costs two extra einsums per
        node; switch it off when only the geometric gradient is wanted.
    :param want_fock_adjoints: Also compute ``dE/dF_oo`` and ``dE/dF_vv``, the *matrix*
        generalisation of ``dE/de`` that the orbital-response layer needs. Requires
        ``eps_o`` and ``eps_v``. Appends two arrays to the returned tuple.
    :param eps_o: Occupied orbital energies, ``(n_occ,)``. Required with
        ``want_fock_adjoints``.
    :param eps_v: Virtual orbital energies, ``(n_vir,)``. Likewise.
    :return: ``(energy, X_o_bar, X_v_bar, Z_bar, eps_o_bar, eps_v_bar)``, with
        ``(F_oo_bar, F_vv_bar)`` appended when ``want_fock_adjoints``. The orbital
        adjoints are ``None`` when not requested.

    **Why a Fock adjoint and not just an orbital-energy one.** Written with ``e_i`` in
    the Laplace factors, this energy is not invariant under a rotation among the occupied
    orbitals, so a nuclear gradient would need the occupied-occupied and
    virtual-virtual blocks of the orbital response - which a CPHF solver does not supply
    and which the canonical condition only determines through a further coupled
    equation. Written with ``Theta_o = w^(1/4) exp(t F_oo)`` in place of
    ``diag(w^(1/4) exp(t e_i))`` it is the *same function* at the canonical point but
    manifestly invariant, so those blocks enter only through the overlap derivative and
    the response reduces to the standard occupied-virtual one. ``F_oo_bar`` is the
    adjoint of that reformulation, and its diagonal is exactly ``eps_o_bar``.
    """
    n_grid, n_occ = X_o.shape
    n_vir = X_v.shape[1]
    n_laplace = tau_o.shape[0]

    if Z.shape != (n_grid, n_grid):
        raise ValueError(f"Z has shape {Z.shape}, expected ({n_grid}, {n_grid})")
    if tau_v.shape[0] != n_laplace:
        raise ValueError("tau_o and tau_v disagree on the number of Laplace points")

    logger.debug("THC-MP2 adjoints: n_grid=%d n_occ=%d n_vir=%d n_laplace=%d "
                 "(peak ~%.2f GB)", n_grid, n_occ, n_vir, n_laplace,
                 peak_memory_estimate(n_grid, n_occ) / 1024 ** 3)

    energy = 0.0
    X_o_bar = np.zeros_like(X_o)
    X_v_bar = np.zeros_like(X_v)
    Z_bar = np.zeros_like(Z)
    eps_o_bar = np.zeros(n_occ) if want_orbital_adjoints else None
    eps_v_bar = np.zeros(n_vir) if want_orbital_adjoints else None

    if want_fock_adjoints:
        if eps_o is None or eps_v is None:
            raise ValueError("want_fock_adjoints needs eps_o and eps_v")
        eps_o = np.asarray(eps_o, dtype=float)
        eps_v = np.asarray(eps_v, dtype=float)
        F_oo_bar = np.zeros((n_occ, n_occ))
        F_vv_bar = np.zeros((n_vir, n_vir))
    else:
        F_oo_bar = F_vv_bar = None

    # E = sum_v (c_J J_v + c_K K_v); carrying the coefficients into the adjoints keeps
    # every accumulation below a derivative of the total energy rather than of a piece.
    c_J, c_K = -2.0, 1.0

    for v in range(n_laplace):
        to, tv = tau_o[v], tau_v[v]

        Ao = (X_o * to) @ X_o.T
        Av = (X_v * tv) @ X_v.T

        Ao_bar = np.zeros_like(Ao)
        Av_bar = np.zeros_like(Av)
        tau_o_bar = np.zeros(n_occ)
        tau_v_bar = np.zeros(n_vir)

        # ---- J: E_hadamard = Ao o Av, J_v = tr(E Z E Z) --------------------------------
        Eh = Ao * Av
        EhZ = Eh @ Z
        energy += c_J * float(np.einsum("pq,qp->", EhZ, EhZ))

        # dJ/dEh = 2 Z Eh Z and dJ/dZ = 2 Eh Z Eh, both symmetric here.
        Eh_bar = (2.0 * c_J) * (Z @ EhZ)
        Z_bar += (2.0 * c_J) * (EhZ @ Eh)

        Ao_bar += Eh_bar * Av
        Av_bar += Eh_bar * Ao

        del Eh, EhZ, Eh_bar

        # ---- K, through Psi[P,R,i] = sum_Q Z[P,Q] X_o[Q,i] Av[Q,R] ---------------------
        # T[Q,R,i] = Av[Q,R] X_o[Q,i] is reused for the Z adjoint, so build it once.
        T = Av[:, :, None] * X_o[:, None, :]
        Psi = (Z @ T.reshape(n_grid, -1)).reshape(n_grid, n_grid, n_occ)

        Psi_T = Psi.transpose(1, 0, 2)            # Psi[R,P,i], a view
        G = np.einsum("pri,pri->pr", Psi * to, Psi_T, optimize=True)
        energy += c_K * float(np.einsum("pr,pr->", Ao, G))

        # Ao enters K twice: once explicitly and once through Psi's X_o. This is the
        # explicit occurrence; the other arrives via X_o_bar below.
        Ao_bar += c_K * G

        if want_orbital_adjoints:
            # dK/dtau_o at fixed Ao and Psi - the tau_o that sits in the sum over i.
            tau_o_bar += c_K * np.einsum("pr,pri,pri->i", Ao, Psi, Psi_T, optimize=True)

        if want_fock_adjoints:
            # The same object off the diagonal as well. Invariantly K is
            #   sum_ij Theta_o[i,j] sum_PR Ao[P,R] Psi[P,R,i] Psi[R,P,j],
            # and only the canonical basis collapses Theta_o to its diagonal. Symmetric
            # under i <-> j by relabelling P <-> R, since Ao is.
            Theta_o_bar = c_K * np.einsum("pr,pri,prj->ij", Ao, Psi, Psi_T, optimize=True)

        # dK/dPsi[a,b,i] = 2 tau_o_i Ao[a,b] Psi[b,a,i]  (Ao symmetric).
        Psi_bar = (2.0 * c_K) * (Ao[:, :, None] * Psi_T) * to

        Z_bar += Psi_bar.reshape(n_grid, -1) @ T.reshape(n_grid, -1).T
        del T, Psi, Psi_T, G

        Phi = (Z @ Psi_bar.reshape(n_grid, -1)).reshape(n_grid, n_grid, n_occ)
        del Psi_bar

        X_o_bar += np.einsum("qr,qri->qi", Av, Phi, optimize=True)
        Av_bar += np.einsum("qi,qri->qr", X_o, Phi, optimize=True)
        del Phi

        # ---- Ao, Av adjoints down to X and tau -----------------------------------------
        # Ao = X_o diag(tau_o) X_o^T, so dE/dX_o gets (Ao_bar + Ao_bar^T) X_o scaled by tau.
        X_o_bar += ((Ao_bar + Ao_bar.T) @ X_o) * to
        X_v_bar += ((Av_bar + Av_bar.T) @ X_v) * tv

        if want_orbital_adjoints:
            tau_o_bar += np.einsum("pr,pi,ri->i", Ao_bar, X_o, X_o, optimize=True)
            tau_v_bar += np.einsum("pr,pa,ra->a", Av_bar, X_v, X_v, optimize=True)

            # tau_o = w^(1/4) exp(+t e_i), tau_v = w^(1/4) exp(-t e_a).
            eps_o_bar += t[v] * to * tau_o_bar
            eps_v_bar += -t[v] * tv * tau_v_bar

        if want_fock_adjoints:
            # Ao = X_o Theta_o X_o^T: the same contraction tau_o_bar takes just above,
            # kept as a matrix instead of collapsed to its diagonal.
            Theta_o_bar += X_o.T @ Ao_bar @ X_o
            Theta_v_bar = X_v.T @ Av_bar @ X_v

            # Theta is a matrix function of the Fock block, so its adjoint is the Loewner
            # matrix times the Theta adjoint. The Loewner diagonal is t*to and -t*tv, so
            # diag(F_oo_bar) comes out equal to eps_o_bar - which the tests assert.
            F_oo_bar += loewner(eps_o, to, t[v]) * (0.5 * (Theta_o_bar + Theta_o_bar.T))
            F_vv_bar += loewner(eps_v, tv, -t[v]) * (0.5 * (Theta_v_bar + Theta_v_bar.T))

    if want_fock_adjoints:
        return (energy, X_o_bar, X_v_bar, Z_bar, eps_o_bar, eps_v_bar,
                F_oo_bar, F_vv_bar)

    return energy, X_o_bar, X_v_bar, Z_bar, eps_o_bar, eps_v_bar


def energy(X_o: np.ndarray, X_v: np.ndarray, Z: np.ndarray,
           tau_o: np.ndarray, tau_v: np.ndarray) -> float:
    """
    The energy alone, by the same expressions :func:`energy_and_adjoints` differentiates.

    Used by the tests to confirm that what is being differentiated really is what
    :func:`pythc.methods.mp2.mp2_energy_laplace` evaluates, and by the finite-difference
    harness, which needs a cheap re-evaluation rather than a second adjoint pass.
    """
    n_grid, n_occ = X_o.shape
    total = 0.0

    for v in range(tau_o.shape[0]):
        to, tv = tau_o[v], tau_v[v]
        Ao = (X_o * to) @ X_o.T
        Av = (X_v * tv) @ X_v.T

        EhZ = (Ao * Av) @ Z
        total += -2.0 * float(np.einsum("pq,qp->", EhZ, EhZ))

        T = Av[:, :, None] * X_o[:, None, :]
        Psi = (Z @ T.reshape(n_grid, -1)).reshape(n_grid, n_grid, n_occ)
        G = np.einsum("pri,pri->pr", Psi * to, Psi.transpose(1, 0, 2), optimize=True)
        total += float(np.einsum("pr,pr->", Ao, G))

    return total

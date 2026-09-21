"""
The LS-THC fit as a differentiable pipeline.

:class:`ThcFactorisation` rebuilds, stage by stage, exactly what
:meth:`pythc.thc.ls_ri_thc.LS_RI_THC.build` produces in ``ov`` mode - and the tests
assert that the ``X`` and ``Z`` it lands on agree with the production path to machine
precision, because a gradient of a slightly different energy would be worthless. The
difference is that every intermediate is kept, so :meth:`ThcFactorisation.backward` can
walk the chain in reverse.

The forward chain, in the order it is built:

    X_ao[P,mu]  = w_P^(1/4) phi_mu(r_P)              collocation on the frozen points
    X[P,p]      = sum_mu X_ao[P,mu] C[mu,p]          projected into the MO basis
    S           = (X_o X_o^T) o (X_v X_v^T)          the LS-THC metric
    S_inv       = (S + lambda I)^-1   (or pinv S)
    Jm          = (J + lambda_a I)^(-1/2)            auxiliary Coulomb metric
    W[M,P]      = sum_munu (mu nu|M) Xl[P,mu] Xr[P,nu]
    Y           = Jm W
    D           = Y S_inv
    Z           = D^T D

with ``Xl = X_ao P_occ`` and ``Xr = X_ao P_vir`` the occupied- and virtual-projected
collocation matrices that ``ov`` mode contracts the three-centre integrals against.

Reverse mode turns ``dE/dZ`` into four adjoints, which is everything the geometry needs:

* ``X_ao_bar`` - flows into the collocation derivative (AO gradient + rigid translation);
* ``V_bar``, ``j2c_bar`` - flow into derivative three- and two-centre integrals;
* ``C_bar``, and the ``eps_bar`` from the MP2 layer - the orbital-response inputs.

**Scale.** This keeps the full ``(n_ao, n_ao, n_aux)`` integral array and an
``(n_P, n_ao, n_aux)`` intermediate in core, where the production energy path chunks over
AO shells. That is deliberate for a reference implementation whose job is to be checkable,
and it is why the gradient is exercised on molecules up to about alanine rather than
beyond.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import df, gto

from pythc import lib
from pythc.grad.linalg import aux_coulomb_inv_adjoint, invert_metric_adjoint
from pythc.thc.ls_thc_funcs import (build_aux_coulomb_inv, build_auxmol, build_S,
                                    eval_basefuncs, invert_metric)

logger = logging.getLogger(__name__)


@dataclass
class FactorisationAdjoints:
    """What :meth:`ThcFactorisation.backward` hands to the geometry layer."""

    X_ao_bar: np.ndarray
    """``dE/dX_ao``, shape ``(n_grid, n_ao)``. Includes every route by which the
    collocation matrix reaches the energy: the MO-basis ``X`` used by the metric and the
    MP2 expression, and the projected ``Xl``/``Xr`` used by the integral contraction."""

    V_bar: np.ndarray
    """``dE/d(mu nu|M)``, shape ``(n_ao, n_ao, n_aux)``."""

    j2c_bar: np.ndarray
    """``dE/d(M|N)``, shape ``(n_aux, n_aux)``."""

    C_bar: np.ndarray
    """``dE/dC``, shape ``(n_ao, n_mo)``, at fixed orbital energies."""


class ThcFactorisation:
    """
    Forward and reverse passes over the ``ov``-mode LS-THC fit on a fixed point set.

    :param mol: The molecule.
    :param coords: Grid points, in Bohr, already placed at their nuclei.
    :param weights: Grid weights. Frozen per-element grids use ones; the blocked NNLS
        grids of ``rotate.blocked_fit_per_atom`` carry fitted weights. Either way they
        are constants of the geometry, so there is no ``dw/dR`` term.
    :param mo_coeff: MO coefficients, ``(n_ao, n_mo)``.
    :param n_occ: Number of doubly occupied orbitals.
    :param auxbasis: Auxiliary basis for the density fitting of the target ERIs.
    :param metric_ridge: Ridge strength for the THC metric, or ``None`` for the truncated
        pseudoinverse. Matches ``LS_RI_THC``'s argument of the same name.
    :param aux_ridge: The same for the auxiliary Coulomb metric.
    :param ridge_scale: How the strengths become absolute shifts.
    :param metric_scheme: ``"ridge"`` for ``(S + lambda I)^-1``, ``"damped"`` for the
        Tikhonov-filtered ``S (S^2 + mu^2 I)^-1``, either with a ``"_jacobi"`` suffix to
        precondition the metric by ``E = diag(1 / sqrt(diag S))`` around the filter.
        Ignored when the ridge is ``None``. See :func:`pythc.lib.damped_inv` and
        :func:`pythc.lib.jacobi_scaling`.
    """

    def __init__(self, mol: gto.Mole, coords: np.ndarray, weights: np.ndarray,
                 mo_coeff: np.ndarray, n_occ: int, auxbasis: str,
                 metric_ridge: Optional[float] = None,
                 aux_ridge: Optional[float] = None,
                 ridge_scale: str = "trace",
                 metric_scheme: str = "ridge"):
        self.mol = mol
        self.coords = np.asarray(coords, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.mo_coeff = np.asarray(mo_coeff, dtype=float)
        self.n_occ = int(n_occ)
        self.auxbasis = auxbasis
        self.metric_ridge = metric_ridge
        self.aux_ridge = aux_ridge
        self.ridge_scale = ridge_scale
        self.metric_scheme = metric_scheme

        self.auxmol = build_auxmol(mol, auxbasis)
        self.n_ao = int(mol.nao_nr())
        self.n_aux = int(self.auxmol.nao_nr())
        self.n_vir = self.mo_coeff.shape[1] - self.n_occ

        self._built = False

    # ---- forward -------------------------------------------------------------------

    def build(self) -> "ThcFactorisation":
        """Run the forward pass, keeping every intermediate the reverse pass needs."""
        C = self.mo_coeff
        C_occ, C_vir = C[:, :self.n_occ], C[:, self.n_occ:]

        self.amp = np.power(self.weights, 0.25)
        self.phi = eval_basefuncs(self.mol, coords=self.coords)
        self.X_ao = self.amp[:, None] * self.phi
        self.X = self.X_ao @ C
        self.X_o = self.X[:, :self.n_occ]
        self.X_v = self.X[:, self.n_occ:]

        self.G_o = self.X_o @ self.X_o.T
        self.G_v = self.X_v @ self.X_v.T
        self.S = build_S("ov", self.X, self.n_occ)
        self.S_inv = invert_metric(self.S.copy(), self.metric_ridge, self.ridge_scale,
                                   self.metric_scheme)

        self.j2c = self.auxmol.intor("int2c2e", aosym="s1")
        self.Jm = build_aux_coulomb_inv(self.auxmol, self.aux_ridge, self.ridge_scale,
                                        self.metric_scheme)

        self.P_occ = C_occ @ C_occ.T
        self.P_vir = C_vir @ C_vir.T
        self.Xl = self.X_ao @ self.P_occ
        self.Xr = self.X_ao @ self.P_vir

        self.V = df.incore.aux_e2(self.mol, self.auxmol, intor="int3c2e", aosym="s1")
        self.V = self.V.reshape(self.n_ao, self.n_ao, self.n_aux)

        self.W = lib.einsum("ijq,gi,gj->qg", self.V, self.Xl, self.Xr)
        self.Y = self.Jm @ self.W
        self.D = self.Y @ self.S_inv
        self.Z = self.D.T @ self.D

        self._built = True
        return self

    # ---- reverse -------------------------------------------------------------------

    def backward(self, Z_bar: np.ndarray,
                 X_o_bar: np.ndarray, X_v_bar: np.ndarray) -> FactorisationAdjoints:
        """
        Propagate the MP2 layer's adjoints back through the fit.

        :param Z_bar: ``dE/dZ``, ``(n_grid, n_grid)``.
        :param X_o_bar: ``dE/dX_o`` from the MP2 expression only - the metric's own
            dependence on ``X`` is added here.
        :param X_v_bar: likewise for the virtual block.
        """
        if not self._built:
            raise RuntimeError("call build() before backward()")

        n_grid = self.X.shape[0]
        X_o_bar = np.array(X_o_bar, dtype=float)
        X_v_bar = np.array(X_v_bar, dtype=float)

        # Z = D^T D. Z is symmetric by construction, so symmetrise its adjoint before
        # using it; an asymmetric part would be differentiating off the manifold.
        Z_bar = 0.5 * (np.asarray(Z_bar, dtype=float) + np.asarray(Z_bar, dtype=float).T)
        D_bar = 2.0 * (self.D @ Z_bar)

        # D = Y S_inv
        Y_bar = D_bar @ self.S_inv            # S_inv symmetric
        S_inv_bar = self.Y.T @ D_bar

        S_bar = invert_metric_adjoint(self.S, self.S_inv, S_inv_bar,
                                      self.metric_ridge, self.ridge_scale,
                                      self.metric_scheme)

        # Y = Jm W
        W_bar = self.Jm @ Y_bar               # Jm symmetric
        Jm_bar = Y_bar @ self.W.T
        j2c_bar = aux_coulomb_inv_adjoint(self.j2c, Jm_bar, self.aux_ridge,
                                          self.ridge_scale, self.metric_scheme)

        # S = (X_o X_o^T) o (X_v X_v^T)
        S_bar = 0.5 * (S_bar + S_bar.T)
        G_o_bar = S_bar * self.G_v
        G_v_bar = S_bar * self.G_o
        X_o_bar += 2.0 * (G_o_bar @ self.X_o)
        X_v_bar += 2.0 * (G_v_bar @ self.X_v)

        # W[M,P] = sum_munu V[mu,nu,M] Xl[P,mu] Xr[P,nu]
        V_bar = lib.einsum("qg,gi,gj->ijq", W_bar, self.Xl, self.Xr)
        Xl_bar = lib.einsum("qg,ijq,gj->gi", W_bar, self.V, self.Xr)
        Xr_bar = lib.einsum("qg,ijq,gi->gj", W_bar, self.V, self.Xl)

        # X = X_ao C, Xl = X_ao P_occ, Xr = X_ao P_vir
        X_bar = np.empty((n_grid, self.mo_coeff.shape[1]))
        X_bar[:, :self.n_occ] = X_o_bar
        X_bar[:, self.n_occ:] = X_v_bar

        X_ao_bar = X_bar @ self.mo_coeff.T + Xl_bar @ self.P_occ + Xr_bar @ self.P_vir

        # C enters directly through X and again through the two projectors.
        C_bar = self.X_ao.T @ X_bar
        P_occ_bar = self.X_ao.T @ Xl_bar
        P_vir_bar = self.X_ao.T @ Xr_bar
        C_bar[:, :self.n_occ] += (P_occ_bar + P_occ_bar.T) @ self.mo_coeff[:, :self.n_occ]
        C_bar[:, self.n_occ:] += (P_vir_bar + P_vir_bar.T) @ self.mo_coeff[:, self.n_occ:]

        return FactorisationAdjoints(X_ao_bar=X_ao_bar, V_bar=V_bar,
                                     j2c_bar=j2c_bar, C_bar=C_bar)

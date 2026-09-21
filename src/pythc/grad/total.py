"""
The total gradient, and the scanner that lets an integrator drive it.

:func:`~pythc.grad.driver.thc_mp2_gradient` gives the correlation gradient at a fixed SCF
reference and :mod:`pythc.grad.response` gives what the orbitals contribute when they
relax. This module adds the last two pieces needed before a trajectory can be propagated
on a frozen-grid THC surface at all:

* :func:`total_mp2_gradient` - the sum that is actually the derivative of a total energy,
  ``E_HF + E_corr``, with the Hartree-Fock part taken from PySCF's own analytic gradient;
* :class:`ThcMP2Gradients` and its scanner - the :mod:`pyscf.md` interface, so
  ``md.NVE(grad.as_scanner()).run()`` propagates Born-Oppenheimer dynamics on that
  surface with no further glue.

**The grid is the point.** A scanner holds one nucleus-relative point set per atom and
re-attaches it by rigid translation at every geometry the integrator asks about. Nothing
is re-selected, re-fitted or re-pruned along the trajectory, which is the property the
whole atom-centred programme exists to buy: no discrete function of geometry anywhere in
the pipeline, so the surface is differentiable and the gradient below is its derivative.

**What this does not make cheap.** The response layer solves the coupled-perturbed
equations once per nuclear degree of freedom, so a gradient costs Hessian-level work. It
is correct and it is checked against a finite difference of the fully relaxed energy; it
is not yet the one-solve Z-vector form that production dynamics would want. See
:mod:`pythc.grad.response`.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pyscf import gto, lib, scf

from pythc.grad.driver import thc_mp2_gradient
from pythc.grad.frozen import FrozenGrid
from pythc.grad.response import orbital_response_gradient

logger = logging.getLogger(__name__)


def total_mp2_gradient(mol: gto.Mole, mf, grid: FrozenGrid, auxbasis: str,
                       n_laplace: int = 10,
                       metric_ridge: Optional[float] = None,
                       aux_ridge: Optional[float] = None,
                       ridge_scale: str = "trace",
                       metric_scheme: str = "ridge",
                       with_response: bool = True):
    """
    ``(e_tot, de)`` for ``E_HF + E_corr`` on a frozen grid, with the orbitals relaxed.

    :param mol: The molecule at the geometry wanted.
    :param mf: A *converged* SCF object at that geometry.
    :param grid: The frozen grid, already attached to ``mol``.
    :param auxbasis: Auxiliary basis for the DF target integrals.
    :param with_response: Include the orbital-response term. ``False`` gives the
        fixed-orbital force, which is **not** the gradient of ``e_tot`` and must not be
        propagated on - it is here so a trajectory can measure what the response is worth.
    :return: ``(e_tot, de)`` with ``de`` of shape ``(n_atm, 3)`` in Hartree per Bohr.

    The Hartree-Fock part is PySCF's own analytic gradient, which needs no response of its
    own because the SCF energy is variational in the orbitals. Only the correlation energy
    does, which is why it is the only part this package had to differentiate.
    """
    res = thc_mp2_gradient(mol, mf, grid, auxbasis, n_laplace=n_laplace,
                           metric_ridge=metric_ridge, aux_ridge=aux_ridge,
                           ridge_scale=ridge_scale, metric_scheme=metric_scheme)

    de = np.asarray(mf.nuc_grad_method().kernel(), dtype=float) + res.de

    if with_response:
        resp = orbital_response_gradient(mol, mf, res.C_bar, res.F_oo_bar, res.F_vv_bar)
        if resp.asymmetry > 1e-6:
            logger.warning("orbital-response Lagrangian asymmetry %.2e", resp.asymmetry)
        de = de + resp.de

    return float(mf.e_tot) + float(res.energy), de


@dataclass
class ThcMP2Gradients:
    """
    A frozen-grid THC-MP2 gradient method, in the shape :mod:`pyscf.md` expects.

    :param per_atom: One ``(coords, weights)`` pair per atom, ``coords`` given relative to
        that atom's nucleus in Bohr - the offline per-element object, in the format
        :class:`~pythc.grad.frozen.FrozenGrid` takes. Held fixed for the whole trajectory.
    :param auxbasis: Auxiliary basis for the DF target integrals.
    :param n_laplace: Laplace quadrature points.
    :param metric_ridge: Regularisation of the THC metric. For gradient work prefer
        ``metric_scheme="damped"``; see FINDINGS section 12 for why the plain ridge's
        usable window is three to five decades narrower.
    :param with_response: Include the orbital response. Leave this on for dynamics.
    :param conv_tol: SCF convergence. Tight by default: the response is a difference of
        converged quantities and a loose SCF shows up in it directly.
    :param mol: The reference molecule. Only the scanner needs it, to know what it is
        moving before an integrator has handed it a geometry.
    """

    per_atom: list
    auxbasis: str
    mol: Optional[gto.Mole] = None
    n_laplace: int = 10
    metric_ridge: Optional[float] = None
    aux_ridge: Optional[float] = None
    ridge_scale: str = "trace"
    metric_scheme: str = "ridge"
    with_response: bool = True
    conv_tol: float = 1e-12
    base: object = field(default=None, repr=False)

    def scf(self, mol: gto.Mole):
        """A converged density-fitted RHF reference at ``mol``'s geometry."""
        mf = scf.RHF(mol).density_fit(auxbasis=self.auxbasis)
        mf.verbose = 0
        mf.conv_tol = self.conv_tol
        mf.kernel()

        return mf

    def kernel(self, mol: gto.Mole):
        """``(e_tot, de)`` at ``mol``'s current geometry, SCF and all."""
        mf = self.scf(mol)
        grid = FrozenGrid(mol, self.per_atom)

        return total_mp2_gradient(mol, mf, grid, self.auxbasis,
                                  n_laplace=self.n_laplace,
                                  metric_ridge=self.metric_ridge,
                                  aux_ridge=self.aux_ridge,
                                  ridge_scale=self.ridge_scale,
                                  metric_scheme=self.metric_scheme,
                                  with_response=self.with_response)

    def as_scanner(self, mol: Optional[gto.Mole] = None) -> "ThcMP2GradScanner":
        """
        The :mod:`pyscf.md` entry point. See :class:`ThcMP2GradScanner`.

        :param mol: The starting molecule, if it was not given to the constructor. An
            integrator reads ``scanner.mol`` before it ever calls the scanner, so one of
            the two has to supply it.
        """
        return ThcMP2GradScanner(self, mol if mol is not None else self.mol)


class ThcMP2GradScanner(lib.GradScanner):
    """
    A PySCF gradient scanner over the frozen-grid THC-MP2 surface.

    ``scanner(mol)`` re-converges the SCF and returns ``(e_tot, de)`` at whatever geometry
    ``mol`` currently holds, which is the whole protocol :mod:`pyscf.md`'s integrators
    drive. The frozen point sets ride along unchanged - the only thing that happens to the
    grid between steps is that it is translated onto the new nuclear positions.

    :class:`pyscf.lib.GradScanner`'s own ``__init__`` is bypassed: it expects the wrapped
    object to carry a PySCF method in ``.base`` that can itself be made into a scanner,
    and a THC fit is not one. Subclassing it anyway is what makes
    ``isinstance(x, lib.GradScanner)`` true, which is how ``pyscf.md`` accepts it.
    """

    def __init__(self, g: ThcMP2Gradients, mol: Optional[gto.Mole] = None):
        if mol is None:
            raise ValueError("a scanner needs a starting molecule: pass mol= to "
                             "ThcMP2Gradients or to as_scanner()")
        self.g = g
        self.base = g
        self.mol = mol
        self._converged = True

    @property
    def converged(self) -> bool:
        return self._converged

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.copy()
            mol.set_geom_(mol_or_geom, unit="Bohr")
            mol.build(False, False)

        self.mol = mol
        e_tot, de = self.g.kernel(mol)
        self._converged = True

        return e_tot, de

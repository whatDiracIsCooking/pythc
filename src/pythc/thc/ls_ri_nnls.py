import logging

import numpy as np
from pyscf import gto

from pythc.grid import GridProvider, NNLSGrid
from pythc.thc.ls_ri_thc import LS_RI_THC
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode

logger = logging.getLogger()


class LS_RI_NNLS(LS_RI_THC):
    """
    Least-squares THC on a grid whose quadrature weights have been refitted and pruned by
    non-negative least squares, following Hillers-Bendtsen, Lu, Martinez
    (2026 - DOI: 10.1021/acs.jctc.6c00664).

    The decomposition itself is ordinary LS-THC, exactly as in
    :class:`~pythc.thc.ls_ri_becke.LS_RI_Becke`; all that changes is the grid, which is
    reweighted by :class:`~pythc.grid.NNLSGrid` so it reproduces the AO overlap matrix
    under numerical integration. Most weights come out of that fit at exactly zero, so
    the grid is pruned as a side effect.

    Unlike the pivoted Cholesky pruning of
    :class:`~pythc.thc.ls_ri_cholesky.LS_RI_Cholesky`, which selects grid points but
    keeps their tabulated weights, refitting the weights means the resulting grid is not
    capped at the accuracy of the input grid.
    """

    def __init__(self,
                 mol: gto.Mole,
                 auxbasis: str,
                 grid: GridProvider = None,
                 mo_coeff: np.ndarray = None,
                 weight_threshold: float = 1e-4,
                 blocked: bool = False,
                 max_points: int = None):
        """
        :param grid: The *input* grid to reweight, not the grid used for the fit. Defaults
            to the level 0 Becke grid.
        :param weight_threshold: KKT tolerance of the NNLS fit; see :class:`~pythc.grid.NNLSGrid`.
        :param blocked: Fit each atomic sub-grid separately. Approximate; see
            :meth:`~pythc.grid.NNLSGrid._build_blocked`.
        :param max_points: Cap on the number of retained grid points.
        """
        super().__init__(mol=mol,
                         auxbasis=auxbasis,
                         grid=NNLSGrid(mol,
                                       parent=grid,
                                       weight_threshold=weight_threshold,
                                       blocked=blocked,
                                       max_points=max_points),
                         mo_coeff=mo_coeff)

    @classmethod
    def __str__(cls):
        return "ls_thc_nnls"

    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol) -> np.ndarray:
        # The grid arrives already pruned, so there is nothing left to select here: the
        # collocation matrix is built over every point the NNLS fit kept.
        grid, weights = self.grid.build()
        R = eval_basefuncs(self.mol, coords=grid)
        X = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R

        self.pruned_grid = grid

        return X

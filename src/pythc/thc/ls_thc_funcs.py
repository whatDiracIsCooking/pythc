import logging
from typing import Optional

import numba as nb
import numpy as np
import pyscf.lib as pyscflib
from pyscf import df, gto, ao2mo

from pythc import lib
from pythc.thc.checkpoints import METRIC_INVERSION
from pythc.thc.thc_base import Mode
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

def eval_basefuncs(mol: gto.Mole, coords: np.ndarray):
    return mol.eval_gto('GTOval_sph', coords=coords)

def get_reference_eri(mol: gto.Mole, nelec: int, mo_coeff: Optional[np.ndarray]) -> np.ndarray:
    if len(mo_coeff) > 0:
        eri = ao2mo.kernel(mol, mo_coeff, compact=False)
    else:
        eri = ao2mo.kernel(mol, np.eye(nelec), compact=False)

    return eri

@nb.njit(parallel=True, fastmath=True, cache=True)
def build_codensity_kernel_tril(X_block, n, out):
    n_grid = X_block.shape[0]

    for g in nb.prange(n_grid):
        k = 0
        for i in range(n):
            val_i = X_block[g, i]
            for j in range(i + 1):
                out[g, k] = val_i * X_block[g, j]
                k += 1


@nb.njit(parallel=True, fastmath=True, cache=True)
def build_codensity_kernel_rect(X_block, n_occ, n_vir, out):
    n_grid = X_block.shape[0]
    n_total = n_occ + n_vir

    for g in nb.prange(n_grid):
        k = 0
        for i in range(n_occ):
            val_i = X_block[g, i]
            for j in range(n_occ, n_total):
                out[g, k] = val_i * X_block[g, j]
                k += 1


@nb.jit(parallel=True, fastmath=True, cache=True)
def build_S_ov_block(X, n_occ, S):
    n_grid = X.shape[0]
    n_basis = X.shape[1]

    for ig in nb.prange(n_grid):
        dot_occ_diag = 0.0
        dot_vir_diag = 0.0

        for i in range(n_occ):
            dot_occ_diag += X[ig, i] * X[ig, i]

        for a in range(n_occ, n_basis):
            dot_vir_diag += X[ig, a] * X[ig, a]

        S[ig, ig] = dot_occ_diag * dot_vir_diag

        for jg in range(ig + 1, n_grid):
            dot_occ = 0.0
            dot_vir = 0.0

            for i in range(n_occ):
                dot_occ += X[ig, i] * X[jg, i]

            for a in range(n_occ, n_basis):
                dot_vir += X[ig, a] * X[jg, a]

            val = dot_occ * dot_vir

            S[ig, jg] = val
            S[jg, ig] = val


@nb.njit(parallel=True, fastmath=True, cache=True)
def build_S_full(X, S):
    n_grid = X.shape[0]
    n_basis = X.shape[1]

    for ig in nb.prange(n_grid):
        dot_prod_diag = 0.0
        for p in range(n_basis):
            dot_prod_diag += X[ig, p] * X[ig, p]
        S[ig, ig] = dot_prod_diag * dot_prod_diag

        for jg in range(ig + 1, n_grid):
            dot_prod = 0.0
            for q in range(n_basis):
                dot_prod += X[ig, q] * X[jg, q]

            val = dot_prod * dot_prod

            S[ig, jg] = val
            S[jg, ig] = val


def build_S(mode: Mode, X, n_occ):
    n_grid = X.shape[0]
    S = np.empty((n_grid, n_grid))
    if mode == 'ov':
        build_S_ov_block(X, n_occ, S)
    else:
        build_S_full(X, S)
    return S


def contract_codensity_full_eri(self, X, mo_coeff):
    logger.info("using full 4-index integrals for fitting matrix")
    Xs = lib.einsum("pn,pm->mnp", X, X).reshape(self.N ** 2, -1)
    eri = get_reference_eri(self.mol, self.N, mo_coeff)
    E = Xs.T @ eri @ Xs
    return E


def invert_metric(S: np.ndarray, ridge: Optional[float] = None,
                  ridge_scale: str = "trace",
                  scheme: str = "ridge") -> np.ndarray:
    """
    Invert the LS-THC metric ``S_PQ = (X X^T) o (X X^T)``.

    Two regularisations are available, and they differ in kind, not degree.

    The default truncated pseudoinverse discards every eigenvalue below a relative
    cutoff. That choice is *discrete*: the number of surviving eigenvalues is an integer
    function of the geometry, so as nuclei move an eigenvalue crosses the cutoff and the
    energy steps. Everything else in the LS-THC pipeline is smooth in the nuclear
    coordinates once the grid is frozen, which makes this truncation the last thing
    standing between the scheme and a differentiable potential energy surface.

    Passing ``ridge`` replaces it with ``(S + lambda I)^-1``, which damps the same
    ill-conditioned directions without ever dropping one, and is analytic in ``S``. It
    matters most for grids assembled as a union of per-atom point sets, whose
    near-duplicate points in bonding regions drive the smallest metric eigenvalue down
    to ~1e-20 and so leave a crowd of eigenvalues loitering near any cutoff.

    **Choosing lambda.** Ridge has a floor that truncation does not: where the
    pseudoinverse discards the numerically null directions, ridge inverts them at
    ``1/lambda``, so too small a lambda amplifies rounding noise instead of the signal.
    Measured on the grids in ``experiments/atom_centered_grids`` (cc-pVDZ, ``ov`` mode),
    a shift of ~1e-10 times the largest eigenvalue - ``ridge=1e-8`` at the default
    ``"trace"`` scaling - costs between 0.1 and 3 uHa against the truncation and is
    smooth on every system tried. Below ~1e-12 of the largest eigenvalue the energy is
    still right but its *derivative* is not, and below ~1e-15 the energy itself becomes
    noise. Smoothness, not accuracy, is the binding constraint from below.

    :param S: The metric to invert.
    :param ridge: Dimensionless ridge strength. ``None`` keeps the truncated
        pseudoinverse.
    :param ridge_scale: How ``ridge`` becomes an absolute shift; see
        :func:`pythc.lib.ridge_shift`.
    :return: The (regularised) inverse metric.
    """
    if ridge is None:
        return lib.pinv(S)

    if scheme.endswith(lib.JACOBI_SUFFIX):
        # Symmetric Jacobi preconditioning around the filter. The weights enter the
        # metric ONLY as S(w) = D S(1) D with D = diag(sqrt(w)) - a symmetric diagonal
        # scaling and nothing else - so a collocation weighting and a diagonal
        # preconditioner are the same kind of object. The pseudoinverse absorbs either
        # exactly; a ridge or damped filter, whose shift is absolute, absorbs neither.
        # This applies the scaling the metric itself suggests, at runtime, per molecule:
        #     Sj = E S E,  E = diag(1 / sqrt(diag S))   =>   S^-1 = E Sj^-1 E
        # diag(S)_PP = (sum_mu X_muP^2)^2 is smooth in the nuclear coordinates and
        # nowhere zero for a point carrying any amplitude, so unlike a fitted weight set
        # this costs no offline object and stays differentiable.
        # S_PP = (sum_mu X_muP^2)^2 is exactly zero for a point carrying no amplitude -
        # which a weight fit produces whenever it drops a point (the oracle fit of
        # levers.py zeroes 250 of 702). lib.jacobi_scaling leaves those rows unscaled;
        # the filter sends them to zero anyway, and dividing by zero turns the whole
        # metric into NaN. The adjoint reaches the same helper, so the two cannot drift.
        e = lib.jacobi_scaling(S)
        inner = invert_metric(e[:, None] * S * e[None, :], ridge, ridge_scale,
                              lib.strip_jacobi(scheme))
        return e[:, None] * inner * e[None, :]

    if scheme == "damped":
        return lib.damped_inv(S, ridge, scale=ridge_scale)
    if scheme == "ridge_eigh":
        return lib.ridge_inv_eigh(S, ridge, scale=ridge_scale)
    if scheme != "ridge":
        raise ValueError(f"unknown metric scheme {scheme!r}, expected 'ridge', "
                         "'ridge_eigh', 'damped', or any of those with a '_jacobi' "
                         "suffix")

    return lib.ridge_inv(S, ridge, scale=ridge_scale)


def build_aux_coulomb_inv(auxmol, ridge: Optional[float] = None,
                          ridge_scale: str = "trace",
                          scheme: str = "ridge"):
    """
    The auxiliary Coulomb metric ``J^{-1/2}``.

    ``ridge`` swaps the truncated pseudo inverse square root for the smooth
    ``(J + lambda I)^(-1/2)``; see :func:`invert_metric` for why that distinction is not
    cosmetic. This metric is far better conditioned than the THC one, so it is rarely
    the binding constraint - but it is geometry dependent too, and a pipeline claiming
    to be free of discrete decisions cannot leave one here.
    """
    logger.info("Computing 2-center Coulomb metric and J^{-1/2}...")
    j2c = auxmol.intor('int2c2e', aosym='s1')
    # A "_jacobi" scheme is about the THC metric's diagonal, which this metric does not
    # share; strip it rather than letting an unrecognised name fall through to the ridge
    # and silently un-damp a pipeline that asked to be damped.
    scheme = lib.strip_jacobi(scheme)
    if ridge is None:
        j2c_cholesky = lib.pseudo_inv_sqrt(j2c)
    elif scheme == "damped":
        j2c_cholesky = lib.damped_inv_sqrt(j2c, ridge, scale=ridge_scale)
    else:
        j2c_cholesky = lib.ridge_inv_sqrt(j2c, ridge, scale=ridge_scale)

    return j2c_cholesky

def build_auxmol(mol, auxbasis) -> gto.Mole:
    return df.addons.make_auxmol(mol, auxbasis=auxbasis)


def compute_ao_slices(mol, auxmol):
    n_ao = int(mol.nao_nr())
    n_aux = int(auxmol.nao_nr())

    max_mem_mb = lib.pyscf_max_memory()
    curr_mem_mb = lib.current_memory()
    live_avail_mb = max(100.0, max_mem_mb - curr_mem_mb)

    # 30% budget for V_chunk buffer
    v_budget_bytes = live_avail_mb * 0.30 * 1024.0 * 1024.0

    ao_loc = mol.ao_loc_nr()
    blocks = []
    start_shl = 0

    for shl in range(1, mol.nbas + 1):
        ao1 = ao_loc[shl]
        ao0 = ao_loc[start_shl]
        ni = int(ao1 - ao0)

        chunk_bytes = ni * n_ao * n_aux * 8

        if chunk_bytes >= v_budget_bytes:
            if shl - 1 == start_shl:
                blocks.append((start_shl, shl, ao_loc[start_shl], ao_loc[shl]))
                start_shl = shl
            else:
                blocks.append((start_shl, shl - 1, ao_loc[start_shl], ao_loc[shl - 1]))
                start_shl = shl - 1

    if start_shl < mol.nbas:
        blocks.append((start_shl, mol.nbas, ao_loc[start_shl], ao_loc[mol.nbas]))

    logger.info(f"Dynamic Blocking: V_chunk budget = {v_budget_bytes / 1e6:.0f} MB. Processing in {len(blocks)} chunks.")

    return blocks

def build_coulomb_matrix(mode: Mode, mol: gto.Mole, auxmol: gto.Mole, X, mo_coeff,
                         metric_ridge: Optional[float] = None,
                         aux_ridge: Optional[float] = None,
                         ridge_scale: str = "trace",
                         metric_scheme: str = "ridge"):
    active = ExperimentRun.get_active()
    n_occ = mol.nelectron // 2
    n_vir = mol.nao_nr() - n_occ

    S = build_S(mode, X, n_occ)
    S_inv = invert_metric(S, metric_ridge, ridge_scale, metric_scheme)
    if active: active.checkpoint(METRIC_INVERSION)

    j2c_inv = build_aux_coulomb_inv(auxmol, aux_ridge, ridge_scale, metric_scheme)
    E = contract_codensity_df_eri(mode, X, mo_coeff, mol, auxmol, j2c_inv, n_occ, n_vir)

    D = E @ S_inv

    return D

def build_coulomb_matrix_asym(mode: Mode, mol: gto.Mole, auxmol: gto.Mole, X_alpha, X_beta, mo_coeff,
                              metric_ridge: Optional[float] = None,
                              aux_ridge: Optional[float] = None,
                              ridge_scale: str = "trace"):
    active = ExperimentRun.get_active()

    S = mol.spin
    N = mol.nao_nr()
    nocc_alpha = (mol.nelectron + S) // 2
    nocc_beta = (mol.nelectron - S) // 2
    nvir_alpha = N - nocc_alpha
    nvir_beta = N - nocc_beta
    mo_coeff_alpha = mo_coeff[0]
    mo_coeff_beta = mo_coeff[1]


    S_aa = build_S(mode, X_alpha, nocc_alpha)
    S_aa_inv = invert_metric(S_aa, metric_ridge, ridge_scale)

    S_bb = build_S(mode, X_beta, nocc_beta)
    S_bb_inv = invert_metric(S_bb, metric_ridge, ridge_scale)
    if active: active.checkpoint(METRIC_INVERSION)

    j2c_inv = build_aux_coulomb_inv(auxmol, aux_ridge, ridge_scale)
    E_aa = contract_codensity_df_eri(mode, X_alpha, mo_coeff_alpha,
                                     mol, auxmol, j2c_inv,
                                     nocc_alpha, nvir_alpha)

    E_bb = contract_codensity_df_eri(mode, X_beta, mo_coeff_beta,
                                     mol, auxmol, j2c_inv,
                                     nocc_beta, nvir_beta)

    D_aa = E_aa @ S_aa_inv
    D_bb = E_bb @ S_bb_inv

    Z_aa = D_aa.T @ D_aa
    Z_bb = D_bb.T @ D_bb
    Z_ab = D_aa.T @ D_bb

    return Z_aa, Z_bb, Z_ab

def contract_codensity_sri_eri(mode: Mode, X: np.ndarray, mo_coeff: np.ndarray, auxmol, S_Lg, j2c_inv, grid, weights,):
    logger.info("building fakemol and 2c1e grid-auxiliary integrals")
    fakemol = gto.fakemol_for_charges(grid)
    int2c2e_gM = gto.intor_cross('int2c2e', fakemol, auxmol)
    B_gM = int2c2e_gM @ j2c_inv

    B_gM = B_gM * np.sqrt(weights)[:, np.newaxis]

    Y = S_Lg @ B_gM  # Shape: (num_rank, n_aux)

    return Y


def contract_codensity_df_eri(mode, X, mo_coeff, mol, auxmol, j2c_inv, n_occ, n_vir):
    logger.info("Using DIRECT density fitting with 2D Chunking and pure BLAS contraction")
    p, n = X.shape

    n_aux = auxmol.nao_nr()
    n_ao = mol.nao_nr()

    if mode == 'ao':
        X_left = X
        X_right = X
    else:
        C_occ = mo_coeff[:, :n_occ]
        C_vir = mo_coeff[:, n_occ:n_occ + n_vir]
        X_left = pyscflib.dot(X[:, :n_occ], C_occ.T)
        X_right = pyscflib.dot(X[:, n_occ:], C_vir.T)

    W = np.zeros((n_aux, p))

    for shl0, shl1, ao0, ao1 in compute_ao_slices(mol, auxmol):
        logger.info(f"Direct DF: Processing AO shells {shl0}-{shl1} / {mol.nbas} (AOs {ao0}-{ao1})")

        shls_slice = (shl0, shl1, 0, mol.nbas, 0, auxmol.nbas)

        V_chunk = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1', shls_slice=shls_slice)

        ni = ao1 - ao0
        nj = n_ao
        V_chunk = V_chunk.reshape(ni, nj, n_aux)

        W += lib.einsum('ijq,gi,gj->qg', V_chunk, X_left[:, ao0:ao1], X_right)

        del V_chunk

    logger.info("Applying inverse metric to final grid...")
    Y = pyscflib.dot(j2c_inv, W)

    return Y



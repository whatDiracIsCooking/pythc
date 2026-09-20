"""
Adjoints of the regularised matrix inversions the LS-THC fit is built on.

Two of the pipeline's steps invert a near-singular symmetric matrix: the THC metric
``S`` and the auxiliary Coulomb metric ``J``. :mod:`pythc.lib` offers each of them in a
truncated and a ridge flavour, and HANDOFF section 4(1) is about the difference. This
module supplies the derivative of both, and the difference shows up here as something
concrete rather than rhetorical:

* ``ridge`` shifts the whole spectrum and inverts everything, so the result is an
  analytic function of the matrix and the adjoint below is exact;
* the truncated pseudoinverse zeroes every eigenvalue under a cutoff, and *how many* it
  zeroes is an integer. Between crossings the map is still smooth and the adjoint below
  is its exact derivative; at a crossing there is no derivative at all, and nothing here
  can manufacture one. :func:`pinv_adjoint` therefore differentiates at fixed retained
  subspace - the only thing a gradient code can do - and the finite-difference check in
  ``experiments/atom_centered_grids/gradient.py`` is what exposes the resulting
  disagreement whenever a scan steps over a crossing.

All four routines reduce to :func:`spectral_adjoint`, the Daleckii-Krein formula for the
derivative of a function applied to the spectrum of a symmetric matrix.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def spectral_adjoint(eigvals: np.ndarray,
                     eigvecs: np.ndarray,
                     f_vals: np.ndarray,
                     f_prime: np.ndarray,
                     out_bar: np.ndarray,
                     degenerate_tol: float = 1e-12) -> np.ndarray:
    """
    Reverse-mode derivative of ``F(M) = U diag(f(lambda)) U^T`` for symmetric ``M``.

    The differential of a spectral function is ``dF = U [Phi o (U^T dM U)] U^T`` with the
    divided-difference matrix ``Phi_kl = (f_k - f_l) / (lambda_k - lambda_l)``, falling
    back to ``f'(lambda_k)`` on the diagonal and wherever two eigenvalues coincide
    (Daleckii-Krein). Transposing that gives the adjoint

        M_bar = U [ (U^T F_bar U) o Phi ] U^T

    which is symmetric, as it must be for a function of a symmetric argument.

    :param eigvals: Eigenvalues of ``M``, ascending, as :func:`numpy.linalg.eigh` returns.
    :param eigvecs: The matching eigenvectors in columns.
    :param f_vals: ``f`` evaluated on ``eigvals``.
    :param f_prime: ``f'`` evaluated on ``eigvals``, used on the diagonal and for
        degenerate pairs.
    :param out_bar: The adjoint of ``F(M)``. Symmetrised on the way in.
    :param degenerate_tol: Relative separation below which a pair of eigenvalues is
        treated as degenerate and the divided difference replaced by the derivative.
    :return: ``M_bar``, symmetric.
    """
    out_bar = 0.5 * (out_bar + out_bar.T)

    gap = eigvals[:, None] - eigvals[None, :]
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    close = np.abs(gap) <= degenerate_tol * scale

    # Dividing first and patching after would raise on the diagonal, where the gap is
    # exactly zero, so make the denominator safe before the division. The diagonal is
    # always "close", which is what puts f' there.
    safe_gap = np.where(close, 1.0, gap)
    phi = (f_vals[:, None] - f_vals[None, :]) / safe_gap
    phi = np.where(close, 0.5 * (f_prime[:, None] + f_prime[None, :]), phi)

    inner = eigvecs.T @ out_bar @ eigvecs
    M_bar = eigvecs @ (inner * phi) @ eigvecs.T

    return 0.5 * (M_bar + M_bar.T)


def _shift_adjoint(M: np.ndarray, A_bar: np.ndarray, lam: float,
                   scale: str, eigvals=None, eigvecs=None) -> np.ndarray:
    """
    Carry the adjoint of ``A = M + shift(M) I`` back onto ``M``.

    The shift is a function of ``M`` - that is the point of
    :func:`pythc.lib.ridge_shift` keeping ``lambda`` dimensionless - so it contributes a
    term of its own. Which term depends on the scaling, and this is where the choice of
    ``"trace"`` over ``"max_eig"`` earns itself: ``tr(M)/n`` is linear in ``M``, so its
    contribution is a clean multiple of the identity, while ``max(eig(M))`` brings in the
    top eigenvector and is undefined the moment that eigenvalue becomes degenerate.
    """
    n = M.shape[0]

    if scale == "trace":
        return A_bar + (lam / n) * float(np.trace(A_bar)) * np.eye(n)

    if scale == "max_eig":
        if eigvals is None:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (M + M.T))
        if len(eigvals) > 1 and abs(eigvals[-1] - eigvals[-2]) <= 1e-10 * max(abs(eigvals[-1]), 1.0):
            logger.warning("ridge_scale='max_eig' differentiated at a degenerate top "
                           "eigenvalue; the shift is not differentiable here")
        u = eigvecs[:, -1]
        return A_bar + lam * float(np.trace(A_bar)) * np.outer(u, u)

    if scale == "absolute":
        return A_bar

    raise ValueError(f"unknown ridge scale {scale!r}")


def ridge_inv_adjoint(M: np.ndarray, inv: np.ndarray, inv_bar: np.ndarray,
                      lam: float, scale: str = "trace") -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.ridge_inv`.

    With ``B = A^-1`` and ``A = M + shift I``, ``dB = -B dA B``, so the adjoint of ``A``
    is ``-B B_bar B``; :func:`_shift_adjoint` then carries it onto ``M``.

    :param M: The matrix that was inverted.
    :param inv: The result of the forward call, ``(M + shift I)^-1``.
    :param inv_bar: Adjoint of that result.
    :param lam: The dimensionless ridge strength used in the forward call.
    :param scale: The scaling used in the forward call.
    """
    inv_bar = 0.5 * (inv_bar + inv_bar.T)
    A_bar = -(inv @ inv_bar @ inv)

    return _shift_adjoint(M, A_bar, lam, scale)


def pinv_adjoint(M: np.ndarray, pinv_bar: np.ndarray,
                 epsilon: float = 1e-10) -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.pinv`, at fixed retained subspace.

    The truncated pseudoinverse applies ``g(lambda) = 1/lambda`` above the cutoff and
    ``0`` below it. Held at a fixed mask that is an ordinary spectral function, and
    :func:`spectral_adjoint` differentiates it exactly. What the fixed mask cannot
    represent is the cutoff being crossed, which is precisely the discontinuity HANDOFF
    section 4(1) measured; see this module's docstring.

    The cutoff itself is ``max(epsilon * lambda_max, 1e-12)``, which moves with ``M``.
    Since ``g`` depends on it only through the mask, and the mask is frozen, that
    dependence contributes nothing here - consistent with the convention above.
    """
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)

    thresh = max(epsilon * eigvals[-1], 1e-12)
    mask = eigvals > thresh

    f_vals = np.where(mask, 1.0 / np.where(mask, eigvals, 1.0), 0.0)
    f_prime = np.where(mask, -1.0 / np.where(mask, eigvals, 1.0) ** 2, 0.0)

    return spectral_adjoint(eigvals, eigvecs, f_vals, f_prime, pinv_bar)


def ridge_inv_sqrt_adjoint(M: np.ndarray, inv_sqrt_bar: np.ndarray,
                           lam: float, scale: str = "trace") -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.ridge_inv_sqrt`, i.e. of ``(M + shift I)^(-1/2)``.
    """
    from pythc.lib import ridge_shift

    M = 0.5 * (M + M.T)
    shift = ridge_shift(M, lam, scale)

    eigvals, eigvecs = np.linalg.eigh(M + shift * np.eye(M.shape[0]))
    clamped = np.maximum(eigvals, shift)

    f_vals = 1.0 / np.sqrt(clamped)
    f_prime = np.where(eigvals >= shift, -0.5 * clamped ** -1.5, 0.0)

    A_bar = spectral_adjoint(eigvals, eigvecs, f_vals, f_prime, inv_sqrt_bar)

    return _shift_adjoint(M, A_bar, lam, scale, eigvals, eigvecs)


def pseudo_inv_sqrt_adjoint(M: np.ndarray, pinv_sqrt_bar: np.ndarray,
                            epsilon: float = 1e-10) -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.pseudo_inv_sqrt`, at fixed retained subspace.

    That routine adds a *fixed absolute* ``1e-8`` before truncating, which is a ridge in
    all but name - so the shift contributes no term of its own and only the truncation
    mask has to be held fixed.
    """
    M = 0.5 * (M + M.T) + np.eye(M.shape[0]) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(M)

    mask = eigvals > epsilon * eigvals[-1]
    safe = np.where(mask, eigvals, 1.0)

    f_vals = np.where(mask, 1.0 / np.sqrt(safe), 0.0)
    f_prime = np.where(mask, -0.5 * safe ** -1.5, 0.0)

    return spectral_adjoint(eigvals, eigvecs, f_vals, f_prime, pinv_sqrt_bar)


def _mu_adjoint(M: np.ndarray, coeff: float, lam: float, scale: str,
                eigvals=None, eigvecs=None) -> np.ndarray:
    """
    Carry ``dE/dmu`` back onto ``M``, where ``mu = lam * reference(M)``.

    The damped inverse depends on ``mu`` through its filter rather than through an added
    shift, so it cannot reuse :func:`_shift_adjoint`: there the whole dependence is
    ``A = M + shift I`` and the chain rule runs through ``tr(A_bar)``, here it runs
    through a separately computed ``dE/dmu``. What the two share is the last step, and
    the same reason for preferring ``"trace"``: ``tr(M)/n`` is linear in ``M``, so
    ``dmu/dM`` is a clean multiple of the identity and analytic in the nuclear
    coordinates.

    :param coeff: ``dE/dmu``, already contracted.
    """
    n = M.shape[0]

    if scale == "trace":
        return (lam * coeff / n) * np.eye(n)

    if scale == "max_eig":
        if eigvals is None:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (M + M.T))
        if len(eigvals) > 1 and abs(eigvals[-1] - eigvals[-2]) <= 1e-10 * max(abs(eigvals[-1]), 1.0):
            logger.warning("ridge_scale='max_eig' differentiated at a degenerate top "
                           "eigenvalue; the damping is not differentiable here")
        u = eigvecs[:, -1]
        return (lam * coeff) * np.outer(u, u)

    if scale == "absolute":
        return np.zeros((n, n))

    raise ValueError(f"unknown ridge scale {scale!r}")


def damped_inv_adjoint(M: np.ndarray, inv_bar: np.ndarray, lam: float,
                       scale: str = "trace") -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.damped_inv`, i.e. of ``M (M^2 + mu^2 I)^-1``.

    The forward map is a spectral function ``f(sigma) = sigma / (sigma^2 + mu^2)`` whose
    parameter ``mu`` is itself a function of ``M``, so the adjoint is in two pieces:
    :func:`spectral_adjoint` at fixed ``mu``, plus the explicit ``mu`` dependence

        df/dmu = -2 sigma mu / (sigma^2 + mu^2)^2

    contracted against the adjoint on the diagonal of the eigenbasis and pushed back
    through ``dmu/dM`` by :func:`_mu_adjoint`. Dropping the second piece would leave a
    gradient that is wrong by a term proportional to ``lam`` - small, but not zero, and
    exactly the kind of omission a finite difference converges to a constant on rather
    than quadratically.

    :param M: The matrix that was inverted.
    :param inv_bar: Adjoint of the forward result.
    :param lam: The dimensionless strength used in the forward call.
    :param scale: The scaling used in the forward call.
    """
    from pythc.lib import ridge_shift

    M = 0.5 * (M + M.T)
    mu = ridge_shift(M, lam, scale)

    eigvals, eigvecs = np.linalg.eigh(M)
    denom = eigvals ** 2 + mu ** 2

    f_vals = eigvals / denom
    f_prime = (mu ** 2 - eigvals ** 2) / denom ** 2

    M_bar = spectral_adjoint(eigvals, eigvecs, f_vals, f_prime, inv_bar)

    # The explicit mu dependence. inv_bar is symmetrised inside spectral_adjoint; do the
    # same here so the two pieces see the same adjoint.
    sym_bar = 0.5 * (inv_bar + inv_bar.T)
    df_dmu = -2.0 * eigvals * mu / denom ** 2
    coeff = float(np.sum(df_dmu * np.einsum("ki,kl,li->i", eigvecs, sym_bar, eigvecs)))

    return M_bar + _mu_adjoint(M, coeff, lam, scale, eigvals, eigvecs)


def damped_inv_sqrt_adjoint(M: np.ndarray, inv_sqrt_bar: np.ndarray, lam: float,
                            scale: str = "trace") -> np.ndarray:
    """
    Adjoint of :func:`pythc.lib.damped_inv_sqrt`, i.e. of the spectral function
    ``sqrt(sigma / (sigma^2 + mu^2))``.

    Same two pieces as :func:`damped_inv_adjoint`. The clamp at ``sigma = 0`` in the
    forward call makes the map non-differentiable for a matrix with a negative
    eigenvalue, which a Coulomb metric does not have; ``f'`` is set to zero there rather
    than returning a NaN.
    """
    from pythc.lib import ridge_shift

    M = 0.5 * (M + M.T)
    mu = ridge_shift(M, lam, scale)

    eigvals, eigvecs = np.linalg.eigh(M)
    denom = eigvals ** 2 + mu ** 2
    pos = eigvals > 0.0

    g = np.where(pos, eigvals, 0.0) / denom
    f_vals = np.sqrt(g)

    safe = np.where(f_vals > 0.0, f_vals, 1.0)
    g_prime = np.where(pos, (mu ** 2 - eigvals ** 2) / denom ** 2, 0.0)
    f_prime = np.where(f_vals > 0.0, 0.5 * g_prime / safe, 0.0)

    M_bar = spectral_adjoint(eigvals, eigvecs, f_vals, f_prime, inv_sqrt_bar)

    sym_bar = 0.5 * (inv_sqrt_bar + inv_sqrt_bar.T)
    dg_dmu = np.where(pos, -2.0 * eigvals * mu / denom ** 2, 0.0)
    df_dmu = np.where(f_vals > 0.0, 0.5 * dg_dmu / safe, 0.0)
    coeff = float(np.sum(df_dmu * np.einsum("ki,kl,li->i", eigvecs, sym_bar, eigvecs)))

    return M_bar + _mu_adjoint(M, coeff, lam, scale, eigvals, eigvecs)


def invert_metric_adjoint(S: np.ndarray, S_inv: np.ndarray, S_inv_bar: np.ndarray,
                          ridge: Optional[float] = None,
                          ridge_scale: str = "trace",
                          scheme: str = "ridge") -> np.ndarray:
    """
    Adjoint of :func:`pythc.thc.ls_thc_funcs.invert_metric`, dispatching on ``ridge``
    exactly as the forward call does.
    """
    if ridge is None:
        return pinv_adjoint(S, S_inv_bar)

    if scheme == "damped":
        return damped_inv_adjoint(S, S_inv_bar, ridge, ridge_scale)

    return ridge_inv_adjoint(S, S_inv, S_inv_bar, ridge, ridge_scale)


def aux_coulomb_inv_adjoint(j2c: np.ndarray, j2c_inv_bar: np.ndarray,
                            ridge: Optional[float] = None,
                            ridge_scale: str = "trace",
                            scheme: str = "ridge") -> np.ndarray:
    """
    Adjoint of :func:`pythc.thc.ls_thc_funcs.build_aux_coulomb_inv`.
    """
    if ridge is None:
        return pseudo_inv_sqrt_adjoint(j2c, j2c_inv_bar)

    if scheme == "damped":
        return damped_inv_sqrt_adjoint(j2c, j2c_inv_bar, ridge, ridge_scale)

    return ridge_inv_sqrt_adjoint(j2c, j2c_inv_bar, ridge, ridge_scale)

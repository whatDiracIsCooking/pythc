import importlib.util
import json
import logging
import os
import subprocess

import cotengra as ctg
import numpy as np
import opt_einsum as oe
import psutil
import scipy.linalg
import pyscf.lib as pyscflib
import pytblis as pt
from pyscf import df

logger = logging.getLogger()

def cpu_count() -> int:
    return int(os.getenv("OMP_NUM_THREADS", os.cpu_count()))

def einsum(*args, **kwargs):
    """
    Drop-in replacement for numpy.einsum

    Can be configured via PYTHC_EINSUM environment variable to either to TBLIS ("tblis"), standard numpy ("numpy")
    or opt_einsum ("oe_numpy").
    """
    backend = os.environ.get("PYTHC_EINSUM", "opt_einsum")

    if backend == "tblis":
        pt.set_num_threads(cpu_count())
        return pt.einsum(*args, **kwargs, optimize='optimal')
    elif backend == "numpy":
        return np.einsum(*args, optimize=True, **kwargs)
    elif backend == "opt_einsum":
        return oe.contract(*args, **kwargs)
    elif backend == "cotengra":
        return contract_cotengra(*args)
    else:
        raise ValueError(f"Unknown einsum backend: {backend}")

rng = np.random.default_rng(42)

def pinv(M: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Pseudo invert a hermitian matrix.
    
    Builds a Moore-Penrose pseudo inversion using an eigenvalue decomposition. For safety, all values smaller than
    1e-15 are set to zero.

    :param M: Hermitian matrix
    :param epsilon: Percentage of the biggest eigenvalue to use as a cutoff
    :return: Pseudo inverted matrix
    """
    M[np.abs(M) < 1e-15] = 0.0

    if not M.flags['F_CONTIGUOUS']:
        M = np.asfortranarray(M)

    logger.info("pseudo inverting matrix of shape %s", M.shape)
    eig_vals, eig_vecs = np.linalg.eigh(M)

    max_eig = eig_vals[-1]
    thresh = max(epsilon * max_eig, 1e-12)
    mask = eig_vals > thresh

    inv_eigvals = np.zeros_like(eig_vals)
    inv_eigvals[mask] = 1.0 / eig_vals[mask]
    logger.info(f"selected {len(inv_eigvals[mask])} eigenvalues")

    S_inv = (eig_vecs * inv_eigvals) @ eig_vecs.T
    return S_inv


def ridge_shift(M: np.ndarray, lam: float, scale: str = "trace") -> float:
    """
    Convert a dimensionless ridge strength into the absolute shift to add to ``M``.

    Keeping ``lam`` dimensionless makes the regularised inverses below equivariant
    under ``M -> c M``, which matters because the LS-THC metric carries the units of
    whatever the collocation matrix was scaled by.

    :param M: Hermitian matrix the shift will be applied to.
    :param lam: Dimensionless regularisation strength.
    :param scale: ``"trace"`` uses the mean eigenvalue ``tr(M)/n``, which is *linear*
        in ``M`` and therefore an analytic function of nuclear coordinates - the
        property the whole ridge construction exists to preserve. ``"max_eig"`` uses
        the largest eigenvalue, matching the convention of :func:`pinv`'s ``epsilon``;
        it is easier to compare against but only piecewise smooth, since it has a kink
        wherever the top eigenvalue becomes degenerate. ``"absolute"`` takes ``lam`` at
        face value.
    :return: The absolute shift ``lambda``.
    """
    if scale == "trace":
        reference = float(np.trace(M)) / M.shape[0]
    elif scale == "max_eig":
        reference = float(np.linalg.eigvalsh(M)[-1])
    elif scale == "absolute":
        reference = 1.0
    else:
        raise ValueError(f"unknown ridge scale {scale!r}, expected "
                         "'trace', 'max_eig' or 'absolute'")

    return lam * reference


def ridge_inv(M: np.ndarray, lam: float = 1e-8, scale: str = "trace") -> np.ndarray:
    """
    Tikhonov-regularised inverse ``(M + lambda I)^-1`` of a Hermitian positive
    semi-definite matrix.

    A smooth alternative to :func:`pinv`, intended for the metric of a least-squares
    fit. Where ``pinv`` zeroes every eigenvalue below a cutoff, ridge damps the whole
    spectrum continuously. The difference is not an accuracy detail: truncation makes
    the effective rank, and therefore the result, a *discontinuous* function of ``M``,
    so as nuclei move an eigenvalue crosses the cutoff and the fitted quantity jumps.
    Ridge has no such crossing - for ``lambda > 0`` the result is an analytic function
    of ``M``, and so of the geometry. For a fit whose remaining steps are smooth, this
    is what decides whether the potential energy surface has steps in it.

    Applied to the metric of a normal-equation solve, this *is* the Tikhonov solution:
    minimising ``||b - A x||^2 + lambda ||x||^2`` gives ``x = (A^T A + lambda I)^-1 A^T b``,
    so replacing ``pinv(S)`` by ``ridge_inv(S, lam)`` in ``Z = S^-1 E S^-1`` solves a
    ridge-penalised version of the same fit rather than a differently truncated one.

    :param M: Hermitian, positive semi-definite matrix. Not modified.
    :param lam: Dimensionless regularisation strength; see :func:`ridge_shift`.
    :param scale: How ``lam`` becomes an absolute shift; see :func:`ridge_shift`.
    :return: The regularised inverse.
    """
    M = 0.5 * (M + M.T)
    shift = ridge_shift(M, lam, scale)
    logger.info("ridge inverting matrix of shape %s with shift %.3e (lam=%.1e, scale=%s)",
                M.shape, shift, lam, scale)

    A = M + shift * np.eye(M.shape[0])
    try:
        factor = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
        inv = scipy.linalg.cho_solve(factor, np.eye(A.shape[0]), check_finite=False)
    except scipy.linalg.LinAlgError:
        # Only reachable if M has an eigenvalue below -shift, i.e. it was not the
        # positive semi-definite matrix this function is for. Clamping puts the offender
        # back on the ridge floor; the resulting jump is confined to that branch.
        logger.warning("matrix is not positive semi-definite at ridge %.3e, "
                       "falling back to a clamped eigendecomposition", shift)
        eig_vals, eig_vecs = np.linalg.eigh(A)
        inv = (eig_vecs / np.maximum(eig_vals, shift)) @ eig_vecs.T

    return 0.5 * (inv + inv.T)


def ridge_inv_sqrt(M: np.ndarray, lam: float = 1e-8, scale: str = "trace") -> np.ndarray:
    """
    Builds ``(M + lambda I)^(-1/2)``.

    The ridge counterpart of :func:`pseudo_inv_sqrt`, and smooth in ``M`` for the same
    reason :func:`ridge_inv` is: no eigenvalue is ever dropped, so there is no cutoff
    for one to cross. Note that :func:`pseudo_inv_sqrt` already adds a *fixed absolute*
    ``1e-8`` shift before truncating, which is a ridge in all but name - but it then
    truncates on top of it, which is the part that reintroduces the discontinuity.

    :param M: Hermitian, positive semi-definite matrix. Not modified.
    :param lam: Dimensionless regularisation strength; see :func:`ridge_shift`.
    :param scale: How ``lam`` becomes an absolute shift; see :func:`ridge_shift`.
    :return: The regularised inverse square root.
    """
    M = 0.5 * (M + M.T)
    shift = ridge_shift(M, lam, scale)

    eig_vals, eig_vecs = np.linalg.eigh(M + shift * np.eye(M.shape[0]))
    # See ridge_inv: the clamp only bites for a matrix that was not positive
    # semi-definite to begin with.
    inv_sqrt_vals = 1.0 / np.sqrt(np.maximum(eig_vals, shift))

    return (eig_vecs * inv_sqrt_vals) @ eig_vecs.T


def ridge_inv_eigh(M: np.ndarray, lam: float = 1e-8, scale: str = "trace") -> np.ndarray:
    """
    ``(M + lambda I)^-1`` by eigendecomposition rather than by Cholesky.

    Mathematically identical to :func:`ridge_inv` and numerically not: a Cholesky
    factorisation of a matrix whose condition number runs to 1e11 and beyond loses
    digits that a backward-stable symmetric eigensolver keeps. That matters here only as
    a *control*. :func:`damped_inv` changes the filter **and** the algorithm at once, so
    without this function any improvement it shows could be either, and the question of
    whether the filter shape is what matters would stay open.

    Not intended for production use - :func:`ridge_inv` is cheaper and, where the metric
    is not pathological, indistinguishable.

    :param M: Hermitian, positive semi-definite matrix. Not modified.
    :param lam: Dimensionless regularisation strength; see :func:`ridge_shift`.
    :param scale: How ``lam`` becomes an absolute shift; see :func:`ridge_shift`.
    :return: The regularised inverse.
    """
    M = 0.5 * (M + M.T)
    shift = ridge_shift(M, lam, scale)

    eig_vals, eig_vecs = np.linalg.eigh(M)

    return (eig_vecs / (eig_vals + shift)) @ eig_vecs.T


JACOBI_SUFFIX = "_jacobi"


def strip_jacobi(scheme: str) -> str:
    """
    The filter a metric scheme names, with any Jacobi preconditioning taken off.

    The preconditioner wraps a filter rather than replacing one, so every consumer of a
    scheme string needs the same two answers - is it preconditioned, and what filter is
    underneath - and they must not disagree. They did: the forward inversion grew a
    ``"_jacobi"`` branch that the adjoint and the auxiliary Coulomb metric did not know
    about, and both silently applied a ridge instead.
    """
    return scheme[: -len(JACOBI_SUFFIX)] if scheme.endswith(JACOBI_SUFFIX) else scheme


def jacobi_scaling(M: np.ndarray) -> np.ndarray:
    """
    The symmetric Jacobi preconditioner of a Hermitian matrix, as a vector.

    ``e_P = 1 / sqrt(M_PP)`` turns ``M`` into ``E M E`` with unit diagonal, which is the
    scaling the LS-THC metric itself suggests: the collocation weights enter it only as
    ``S(w) = D S(1) D`` with ``D = diag(sqrt(w))``, so a weighting and a diagonal
    preconditioner are the same kind of object. The pseudoinverse absorbs either exactly;
    a ridge or damped filter, whose shift is absolute, absorbs neither - which is why
    applying this one at runtime can stand in for weights the offline object cannot carry.

    ``M_PP`` is exactly zero for a grid point carrying no amplitude, which a weight fit
    produces whenever it drops a point. Those rows are left unscaled rather than divided
    by zero; the filter sends them to zero anyway.

    Both the forward inversion and its adjoint go through this function, so the scaling
    they differentiate is the scaling that was applied.

    :param M: Hermitian matrix with a non-negative diagonal.
    :return: The scaling vector ``e``, one entry per row, strictly positive.
    """
    diag = np.diag(M)
    positive = diag > 0.0

    return np.where(positive, 1.0 / np.sqrt(np.where(positive, diag, 1.0)), 1.0)


def damped_inv(M: np.ndarray, lam: float = 1e-4, scale: str = "trace") -> np.ndarray:
    """
    Damped (Tikhonov-filtered) pseudoinverse ``M (M^2 + mu^2 I)^-1`` of a Hermitian
    positive semi-definite matrix, with ``mu = ridge_shift(M, lam, scale)``.

    This is the third option beside :func:`pinv` and :func:`ridge_inv`, and it exists
    because those two each get one half of what the LS-THC metric inversion needs.
    Writing the gain applied to an eigenvalue ``sigma``:

    =========================  ====================  =================  ==============
    scheme                     gain                  ``sigma >> mu``    ``sigma -> 0``
    =========================  ====================  =================  ==============
    :func:`pinv`               ``1/sigma`` or ``0``  ``1/sigma``        ``0``
    :func:`ridge_inv`          ``1/(sigma + mu)``    ``1/sigma``        ``1/mu``
    :func:`damped_inv`         ``sigma/(sigma^2 + mu^2)``  ``1/sigma``  ``sigma/mu^2``
    =========================  ====================  =================  ==============

    The truncation suppresses the numerically null directions but does so with a step,
    so the retained count is an integer function of the geometry and the energy jumps
    when one crosses. The ridge is analytic in ``M`` - no eigenvalue is ever dropped -
    but it hands the null directions the *largest* gain in the whole operator, ``1/mu``,
    which is the opposite of suppression. On a grid assembled as a union of per-atom
    point sets that crowd is large (a third of the metric's directions can sit below
    1e-14 of the top eigenvalue), and since ``Z = D^T D`` with ``D`` carrying ``S^-1``,
    their rounding noise returns amplified by ``1/mu^2``.

    The damped inverse is analytic in ``M`` like the ridge - it is a rational matrix
    function, with no cutoff for an eigenvalue to cross - while sending the null
    directions to zero like the truncation. Its peak gain, ``1/(2 mu)`` at
    ``sigma = mu``, is the same as the ridge's there, so ``lam`` means the same thing on
    both and the two are directly comparable.

    :param M: Hermitian, positive semi-definite matrix. Not modified.
    :param lam: Dimensionless regularisation strength; see :func:`ridge_shift`.
    :param scale: How ``lam`` becomes the absolute damping ``mu``; see
        :func:`ridge_shift`.
    :return: The damped inverse.
    """
    M = 0.5 * (M + M.T)
    mu = ridge_shift(M, lam, scale)

    logger.info("damped inverting matrix of shape %s with mu=%.3e (lam=%.1e, scale=%s)",
                M.shape, mu, lam, scale)

    eig_vals, eig_vecs = np.linalg.eigh(M)
    filt = eig_vals / (eig_vals ** 2 + mu ** 2)

    return (eig_vecs * filt) @ eig_vecs.T


def damped_inv_sqrt(M: np.ndarray, lam: float = 1e-4, scale: str = "trace") -> np.ndarray:
    """
    The damped counterpart of :func:`ridge_inv_sqrt`, applying
    ``sqrt(sigma / (sigma^2 + mu^2))`` to the spectrum.

    It is the square root of :func:`damped_inv` in the same sense that
    :func:`ridge_inv_sqrt` is the square root of :func:`ridge_inv`, and it is provided so
    the auxiliary Coulomb metric can be treated by the same scheme as the THC metric when
    that comparison is wanted. The auxiliary metric is far better conditioned, so this is
    rarely the binding choice.

    :param M: Hermitian, positive semi-definite matrix. Not modified.
    :param lam: Dimensionless regularisation strength; see :func:`ridge_shift`.
    :param scale: How ``lam`` becomes the absolute damping ``mu``.
    :return: The damped inverse square root.
    """
    M = 0.5 * (M + M.T)
    mu = ridge_shift(M, lam, scale)

    eig_vals, eig_vecs = np.linalg.eigh(M)
    filt = np.sqrt(np.maximum(eig_vals, 0.0) / (eig_vals ** 2 + mu ** 2))

    return (eig_vecs * filt) @ eig_vecs.T


def pseudo_inv_sqrt(M: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Builds the M^(-1/2)

    Builds square root of the pseudo inversion of a hermitian matrix M using an eigenvalue decomposition.
    :param M:  Hermitian matrix
    :param epsilon:
    :return: Pseudo inverse square root
    """
    M = M + np.eye(M.shape[0]) * 1e-8
    eig_vals, eig_vecs = np.linalg.eigh(M)
    thresh = epsilon * eig_vals[-1]
    mask = eig_vals > thresh

    inv_sqrt_vals = np.zeros_like(eig_vals)
    inv_sqrt_vals[mask] = 1.0 / np.sqrt(eig_vals[mask])

    return (eig_vecs * inv_sqrt_vals) @ eig_vecs.T

def current_memory() -> float:
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss # Resident Set Size
    mem_mb = mem_bytes / 1024 / 1024
    return mem_mb

def latest_tag() -> str:
    import git
    repo = git.Repo(search_parent_directories=True)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    latest_tag = tags[-1]

    return latest_tag.name

def get_l3_cache_size_bytes() ->int:
    result = subprocess.run(['lscpu', '-J', '--bytes'],
                            capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    for item in data.get('lscpu', []):
        field = item.get('field', '').lower().strip()
        # Accounts for "L3 cache:" or "L3:" depending on the lscpu version
        if 'l3 cache' in field or field == 'l3:':
            return int(item.get('data', 0).split(" ")[0])

    return 64 * 1024**2

def pyscf_max_memory():
    return pyscflib.param.MAX_MEMORY

def get_gpu_memory_mb() -> float:
    import cupy as cp

    device = cp.cuda.Device()
    _, total_bytes = device.mem_info
    total_bytes = total_bytes / (1024 ** 2)

    return total_bytes



def cotengra_target_size(safety=0.85, bytes_per_float=8) -> int:
    if has_cuda_gpu():
        remaining_mem_mb = get_gpu_memory_mb()
        logger.info(f"GPU has {remaining_mem_mb}MB memory left")
    else:
        # 1. Get raw system memory
        available_bytes = psutil.virtual_memory().available

        # 2. Subtract a flat reserve for MKL thread buffers and system overhead.
        # Adjust this depending on your thread count.
        # (e.g., 192 threads * ~15MB per thread = ~2.8GB)
        mkl_reserve_bytes = 4 * 1024 * 1024 * 1024
        available_bytes = max(0, available_bytes - mkl_reserve_bytes)
        remaining_mem_mb = available_bytes / 1024 / 1024
    # 3. Apply safety margin to what's left
    available_memory_bytes = remaining_mem_mb * 1024 * 1024
    usable_bytes = available_memory_bytes * safety

    # 4. Calculate total allowable elements in the footprint
    total_elements = int(usable_bytes / bytes_per_float)

    # 5. Divide by the concurrent tensor factor (safest is 4)
    # This ensures size(A) + size(B) + size(C) + overhead <= total_elements
    target_size = total_elements // 4
    logger.info(
        f"Allowed peak footprint: {usable_bytes / (1024**2):.2f} MB, "
        f"Target size per tensor: {(target_size * bytes_per_float) / (1024**2):.2f} MB"
    )
    return target_size


def get_optimizer(itemsize: int, max_repeats=1024) -> ctg.HyperOptimizer:
    opt = ctg.ReusableHyperOptimizer(
        minimize='flops', # STRICTLY prioritize flops
        slicing_opts={'target_size': cotengra_target_size(bytes_per_float=itemsize)},
        max_repeats=max_repeats, # Give it plenty of time to find the best sliced path
        progbar=False,
        parallel=False,
    )
    return opt

def contract_cotengra(expr: str, *inputs):
    logger.debug("building cotengra contraction")
    if has_cuda_gpu():
        import cupy as cp
        shapes = [cp.shape(arr) for arr in inputs]
    else:
        shapes = [np.shape(arr) for arr in inputs]
    itemsize = inputs[0].dtype.itemsize

    # 1. Build the tree
    tree = ctg.einsum_tree(expr, *shapes, optimize=get_optimizer(itemsize))

    # 2. Reorder the contraction paths to minimize peak concurrent memory
    tree.reorder_for_peak_size()
    # 3. Check the exact peak memory the tree requires
    peak_bytes = tree.peak_size() * itemsize
    logger.info(f"Tree requires a peak concurrent memory of {peak_bytes / (1024**3):.2f} GB")
    logger.info(f"carrying out contraction: {tree.contract_stats()}")
    # 4. Execute the contraction
    return tree.contract(arrays=inputs, progbar=False, implementation='cotengra')

def has_cuda_gpu() -> bool:
    use_cuda = os.getenv("PYTHC_USE_CUDA", "True") == "True"
    if not use_cuda:
        return False

    cupy_available = importlib.util.find_spec("cupy")
    if not cupy_available:
        return False
    import cupy as cp
    return cp.cuda.is_available()

def to_backend(*inputs, dtype=None):
    # Determine if we should return a single array or a tuple
    return_single = False

    # Flatten the inputs if a single tuple/list was passed
    if len(inputs) == 1:
        if isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]
        else:
            # A single, standalone array was passed
            return_single = True

    if dtype is None:
        type_env = os.getenv("PYTHC_CUDA_PRECISION", "float64").lower()
        if type_env in ["float64", "double", "f64"]:
            dtype = np.float64
        elif type_env in ["float32", "single", "f32"]:
            dtype = np.float32
        elif type_env in ["float16", "f16"]:
            dtype = np.float16
        else:
            dtype = np.float64

    # Process arrays
    if has_cuda_gpu():
        import cupy as cp
        logger.info(f"Copying arrays to GPU in type: {dtype}")
        out_arrays = [cp.asarray(i, dtype=dtype) for i in inputs]
    else:
        logger.debug(f"Keeping arrays on CPU in type: {dtype}")
        out_arrays = [np.asarray(i, dtype=dtype) for i in inputs]

    # Return a single array if requested, otherwise return a tuple
    if return_single:
        return out_arrays[0]
    return tuple(out_arrays)

def get_dfo(mol, auxbasis):
    dfo = df.DF(mol, auxbasis=auxbasis)
    dfo.build()

    return dfo


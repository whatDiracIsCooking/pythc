import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from pyscf import gto, dft
from pyscf.dft import gen_grid, treutler_prune

from pythc.configurable import Configurable
from pythc.decomp.nnls import GroupOperator, OverlapFitOperator, lawson_hanson
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()


class GridProvider(ABC,Configurable):
    @abstractmethod
    def build(self) -> tuple[np.ndarray, np.ndarray]:
        pass

class BeckeGrid(GridProvider):
    def __repr__(self):
        return f"BeckeGrid(level={self.level})"

    def __str__(self):
        return "becke"

    def __init__(self, mol: gto.Mole, level=0, prune: Callable[..., np.ndarray] = treutler_prune):
        self.mol = mol
        self.level = level
        self.prune = prune

    def build(self):
        grid = dft.gen_grid.Grids(self.mol)
        grid.level = self.level
        grid.prune = self.prune
        grid.build()

        active: ExperimentRun = ExperimentRun.get_active()
        if active:
            active.log_metric("grid_points", len(grid.coords))

        return grid.coords, grid.weights


def _eval_basefuncs(mol: gto.Mole, coords: np.ndarray) -> np.ndarray:
    """
    Collocation matrix of the AOs on ``coords``.

    Imported lazily: the THC classes depend on this module, so importing their helpers
    at module scope would point that dependency back the wrong way.
    """
    from pythc.thc.ls_thc_funcs import eval_basefuncs

    return eval_basefuncs(mol, coords)


def _rmsd_overlap(R: np.ndarray, weights: np.ndarray, S: np.ndarray) -> float:
    """
    Root-mean-square deviation per AO pair between the analytic overlap matrix and the
    one obtained by numerical integration on the grid.

    This is the diagnostic of eq 11 in Hillers-Bendtsen, Lu, Martinez (2026) and the
    quantity the NNLS reweighting minimizes.

    Note on conventions when comparing against that paper. Eq 11 reads
    ``sqrt(O_S) / n_AO`` with ``O_S = 1/2 ||S - S_numerical||^2`` from eq 9, which is
    what this function computes. The values plotted in their Figure 1A are a factor of
    sqrt(2) larger, i.e. consistent with ``||S - S_numerical||_F / n_AO``, as though the
    1/2 were not carried through. Scaling this function's output by sqrt(2) reproduces
    their alanine curve to within about 1% at thresholds of 1e-4, 1e-6 and 1e-8, so the
    factor is a reporting convention rather than a difference in the fit - the fit
    itself is unaffected, and the grid sizes agree without any such adjustment.
    """
    S_num = (R * weights[:, np.newaxis]).T @ R
    return float(np.sqrt(0.5 * np.sum((S_num - S) ** 2)) / S.shape[0])


def octahedral_orbits(coords: np.ndarray,
                      center: np.ndarray = None,
                      rtol: float = 1e-9) -> np.ndarray:
    """
    Label the points of an atom-centred grid by the ``(radial shell, angular orbit)``
    block they belong to, so that a fit can keep or drop whole orbits.

    A Becke grid is a product of a radial quadrature with Lebedev angular grids, and a
    Lebedev grid is a union of orbits of the octahedral group ``O_h`` - the 6 points
    ``(+-r,0,0)``, the 12 ``(+-a,+-a,0)``, the 8 ``(+-a,+-a,+-a)``, and orbits of 24 and
    48 - each carrying one common weight. A union of whole orbits with a common weight
    per orbit is exactly invariant under ``O_h``; a partial orbit is not invariant under
    anything. That is what makes this the natural block structure for pruning a grid
    that has to be attached to a nucleus at an arbitrary orientation.

    Elements of ``O_h`` permute the Cartesian axes and flip signs, so they leave the
    multiset ``{|x|, |y|, |z|}`` unchanged. Sorting those three numbers is therefore a
    *complete* invariant: two points get the same key exactly when one is the image of
    the other under some element of the group. The key carries the radius with it
    (``r^2`` is the sum of its squares), so one comparison separates the radial shell
    and the angular orbit at the same time, without having to know how PySCF ordered
    the points or which Lebedev order the pruning chose for each shell.

    :param coords: Grid points, shape ``(n, 3)``.
    :param center: Nucleus the grid is centred on. Defaults to the origin, which is the
        frame ``gen_atomic_grids`` returns its grids in.
    :param rtol: Relative tolerance on the key. Points of one Lebedev orbit are exact
        sign-and-permutation images of each other in floating point, so this only needs
        to absorb the rounding of the radial scaling.
    :return: Integer label per point, numbered from zero.
    """
    rel = np.asarray(coords, dtype=float)
    if center is not None:
        rel = rel - np.asarray(center, dtype=float)
    if rel.shape[0] == 0:
        return np.zeros(0, dtype=np.intp)

    key = np.sort(np.abs(rel), axis=1)
    radius = np.linalg.norm(rel, axis=1)

    # Lexicographic sort brings equal keys together; lexsort takes its primary key last.
    order = np.lexsort(key.T[::-1])
    new_group = np.ones(rel.shape[0], dtype=bool)
    if rel.shape[0] > 1:
        step = np.abs(np.diff(key[order], axis=0)).max(axis=1)
        scale = np.maximum(radius[order][1:], radius[order][:-1])
        new_group[1:] = step > rtol * np.maximum(scale, 1.0)

    labels = np.empty(rel.shape[0], dtype=np.intp)
    labels[order] = np.cumsum(new_group) - 1

    return labels


def _fit_block(op: OverlapFitOperator,
               b: np.ndarray,
               labels: np.ndarray,
               weight_threshold: float,
               max_passive: int) -> np.ndarray:
    """
    One NNLS solve, with the points optionally tied into orbits.

    Without ``labels`` this is the point-wise fit of the reference. With them the
    variables are the orbits rather than the points, so the KKT threshold is applied to
    the orbit gradient and the returned weights are constant on each retained orbit.
    """
    if labels is None:
        return lawson_hanson(op, b, weight_threshold=weight_threshold,
                             max_passive=max_passive)

    grouped = GroupOperator(op, labels)
    v = lawson_hanson(grouped, b, weight_threshold=weight_threshold,
                      max_passive=max_passive)

    return grouped.expand(v)


class NNLSGrid(GridProvider):
    """
    Reweights and prunes a parent quadrature grid by a non-negative least-squares fit to
    the AO overlap matrix, following Hillers-Bendtsen, Lu, Martinez
    (2026 - DOI: 10.1021/acs.jctc.6c00664).

    The weights of the parent grid are discarded and refitted so that numerical
    integration on the grid reproduces the analytic overlap matrix,

        S_{mu nu} = sum_P phi_mu(r_P) w_P phi_nu(r_P)

    subject to w >= 0. Because the overlap matrix is the integral of the generalized
    charge density that LS-THC factorizes, a grid that integrates it accurately is a
    good grid for LS-THC. The non-negativity constraint leaves most weights at exactly
    zero, so the fit also prunes the grid, and the surviving weights remain a valid
    quadrature rule usable for other integrals.

    This is a drop-in ``GridProvider``: any THC class in the package accepts it in place
    of a :class:`BeckeGrid`, since they all build their collocation matrix as
    ``X = w^(1/4) phi(r)`` from whatever ``(coords, weights)`` the grid returns.

    Compared to the pivoted Cholesky pruning of :class:`~pythc.thc.ls_ri_cholesky.LS_RI_Cholesky`,
    which selects points but keeps the tabulated weights, refitting the weights means
    the output grid is not capped at the accuracy of the input grid.

    Setting ``group_orbits`` changes what the fit selects: whole octahedral orbits of
    the parent grid rather than individual points. The resulting grid is larger, but it
    is invariant under the point group of the parent Lebedev shells, which is what makes
    it meaningful to freeze an atomic grid and attach it to a nucleus at an arbitrary
    orientation. See :func:`octahedral_orbits`.

    **Cost.** The active set admits one point per iteration and drops several along the
    way, so the fit scales steeply in the number of points it ends up keeping. Water in
    cc-pVDZ (24 AOs, 1648 input points) fits in about a second; alanine in cc-pVDZ
    (119 AOs, 7896 input points) takes ~20 s at a loose threshold and considerably
    longer at a tight one. Past roughly a hundred basis functions, use ``blocked=True``,
    which fits each atom separately and is what keeps the cost linear in system size.
    """

    def __repr__(self):
        return (f"NNLSGrid(parent={self.parent!r}, weight_threshold={self.weight_threshold:g}, "
                f"blocked={self.blocked}, group_orbits={self.group_orbits})")

    def __str__(self):
        return (f"nnls_{self.weight_threshold:g}"
                + ("_blocked" if self.blocked else "")
                + ("_orbits" if self.group_orbits else ""))

    def __init__(self,
                 mol: gto.Mole,
                 parent: GridProvider = None,
                 weight_threshold: float = 1e-4,
                 blocked: bool = False,
                 group_orbits: bool = False,
                 max_points: int = None,
                 screening: float = 1e-8):
        """
        :param mol: The molecule whose AOs define the overlap matrix being fitted.
        :param parent: Input grid to reweight. Defaults to the level 0 Becke grid, which
            is the input grid used throughout the reference.
        :param weight_threshold: KKT tolerance of the NNLS solve. This is the reference's
            central knob: looser thresholds terminate the active set earlier and leave a
            smaller grid. The default of 1e-4 is the value the reference settles on as a
            balance between accuracy and compactness.
        :param blocked: Solve one NNLS problem per atomic sub-grid instead of one global
            problem. See :meth:`_build_blocked`; this is an extension beyond the
            reference and is approximate.
        :param group_orbits: Select whole ``(radial shell, octahedral orbit)`` blocks of
            the parent grid instead of individual points, giving a retained grid that is
            invariant under the octahedral group about each nucleus. Requires an
            atom-partitioned parent grid, and is incompatible with ``max_points``.
        :param max_points: Cap on the number of retained grid points.
        :param screening: AO amplitude cut-off used to select which AO pairs enter each
            atomic block. Only used when ``blocked`` is set.
        """
        self.mol = mol
        self.parent = parent if parent is not None else BeckeGrid(mol)
        self.weight_threshold = weight_threshold
        self.blocked = blocked
        self.group_orbits = group_orbits
        self.max_points = max_points
        self.screening = screening

        if group_orbits and max_points is not None:
            # max_points caps points, but the solver's variables are now orbits of
            # varying size, so the cap could only be honoured approximately. Rather than
            # silently reinterpreting it, refuse the combination.
            raise ValueError("max_points caps individual points and cannot be applied to "
                             "an orbit-grouped fit, whose variables are whole orbits")

        self._cached = None

    def build(self):
        # THC builders call build() more than once in some code paths and the fit is the
        # expensive part, so hand back the same grid rather than refitting it.
        if self._cached is not None:
            return self._cached

        self._cached = self._build_blocked() if self.blocked else self._build_global()

        return self._cached

    def _atomic_subgrids(self):
        """
        The parent grid taken apart into its atomic sub-grids, with each atom's points
        labelled by the orbit they belong to.

        ``gen_partition`` returns the points of each atom in the order and count that
        ``gen_atomic_grids`` produced them, translated to the nucleus, so the orbit
        labels can be read off the element grid - which is already expressed about the
        origin - and applied to the sub-grid by index.
        """
        if not isinstance(self.parent, BeckeGrid):
            raise NotImplementedError(
                "an atom-partitioned parent grid is needed to take the fit apart over "
                f"atoms, but the parent grid is a {type(self.parent).__name__}")

        g = gen_grid.Grids(self.mol)
        atom_grids = g.gen_atomic_grids(self.mol, level=self.parent.level,
                                        prune=self.parent.prune)
        coords_per_atom, weights_per_atom = g.gen_partition(self.mol, atom_grids, concat=False)

        labels_per_atom = []
        for ia, coords_a in enumerate(coords_per_atom):
            element_grid = atom_grids[self.mol.atom_symbol(ia)][0]
            if len(element_grid) != len(coords_a):
                raise RuntimeError(
                    f"atom {ia} has {len(coords_a)} points but its element grid has "
                    f"{len(element_grid)}; the partition no longer preserves the "
                    "atomic grids point for point, so orbit labels cannot be assigned")
            labels_per_atom.append(octahedral_orbits(element_grid))

        return coords_per_atom, weights_per_atom, labels_per_atom

    def _build_global(self):
        """Single NNLS fit over the whole parent grid, as in the reference."""
        active: ExperimentRun = ExperimentRun.get_active()

        if self.group_orbits:
            # Orbits are defined about a nucleus, so the parent grid has to be taken
            # apart over atoms even though the fit that follows is a single global one.
            coords_per_atom, weights_per_atom, labels_per_atom = self._atomic_subgrids()
            coords = np.vstack(coords_per_atom)
            weights_in = np.concatenate(weights_per_atom)
            # Orbits of different atoms are different groups even when their keys agree.
            offsets = np.cumsum([0] + [int(l.max()) + 1 for l in labels_per_atom[:-1]])
            labels = np.concatenate([l + off for l, off in zip(labels_per_atom, offsets)])
        else:
            coords, weights_in = self.parent.build()
            labels = None

        R = _eval_basefuncs(self.mol, coords)
        S = self.mol.intor('int1e_ovlp_sph')

        rmsd_in = _rmsd_overlap(R, weights_in, S)
        logger.info(f"NNLS grid: reweighting {len(coords)} points, "
                    f"input grid reproduces S to {rmsd_in:.3e}")

        op = OverlapFitOperator(R)
        w = _fit_block(op, op.pack(S), labels,
                       weight_threshold=self.weight_threshold,
                       max_passive=self.max_points)

        keep = np.flatnonzero(w)
        coords_out, weights_out = coords[keep], w[keep]

        rmsd_out = _rmsd_overlap(R[keep], weights_out, S)
        logger.info(f"NNLS grid: kept {len(keep)} of {len(coords)} points "
                    f"({100.0 * (1.0 - len(keep) / len(coords)):.1f}% pruned), "
                    f"reproduces S to {rmsd_out:.3e}")

        if active:
            active.log_metric("grid_points_in", len(coords))
            active.log_metric("grid_points", len(keep))
            active.log_metric("rmsd_S_in", rmsd_in)
            active.log_metric("rmsd_S_out", rmsd_out)
            active.log_metric("sum_weights", float(weights_out.sum()))

        return coords_out, weights_out

    def _build_blocked(self):
        """
        One NNLS fit per atomic sub-grid, an extension beyond the reference.

        Becke grids are already partitioned over atoms by weight functions that sum to
        one, so the overlap matrix decomposes as a sum of atomic contributions,

            S = sum_A integral p_A(r) phi_mu(r) phi_nu(r) dr

        Fitting each atomic block against its own contribution and concatenating the
        results therefore yields a grid that still integrates the full S, at a cost that
        grows linearly rather than cubically in the number of atoms. Combined with AO
        screening over each block, this keeps the fit tractable for large molecules.

        Two approximations come with it. The per-atom target is obtained by numerical
        integration on the parent sub-grid rather than analytically, so unlike the global
        fit, accuracy here *is* capped by the input grid. And AO screening restricts each
        block to the orbitals with significant amplitude on it. Accuracy should therefore
        be verified against the global fit for a given system rather than assumed.

        With ``group_orbits`` the atomic blocks are the unit of transfer as well as of
        fitting: each atom's retained points are whole orbits about its own nucleus, so
        the sub-grid can be rotated about that nucleus by any element of the octahedral
        group without changing the grid at all.
        """
        active: ExperimentRun = ExperimentRun.get_active()

        coords_per_atom, weights_per_atom, labels_per_atom = self._atomic_subgrids()

        S = self.mol.intor('int1e_ovlp_sph')

        coords_out, weights_out = [], []
        n_in = 0

        # max_points caps the whole grid, so share the budget out over the atoms rather
        # than applying it to each block.
        max_per_atom = None if self.max_points is None else max(1, self.max_points // len(coords_per_atom))

        for ia, (coords_a, weights_a) in enumerate(zip(coords_per_atom, weights_per_atom)):
            n_in += len(coords_a)
            R_a = _eval_basefuncs(self.mol, coords_a)

            # Restrict the block to the AOs that actually have amplitude on it; the
            # remaining pairs contribute nothing to this atom's share of S.
            amplitude = np.abs(R_a).max(axis=0)
            kept_ao = np.flatnonzero(amplitude > self.screening * amplitude.max())

            R_a = R_a[:, kept_ao]
            # This atom's share of the overlap matrix, integrated on its own sub-grid.
            S_a = (R_a * weights_a[:, np.newaxis]).T @ R_a

            op = OverlapFitOperator(R_a)
            w_a = _fit_block(op, op.pack(S_a),
                             labels_per_atom[ia] if self.group_orbits else None,
                             weight_threshold=self.weight_threshold,
                             max_passive=max_per_atom)

            keep = np.flatnonzero(w_a)
            logger.info(f"NNLS grid: atom {ia} ({self.mol.atom_symbol(ia)}) "
                        f"{len(coords_a)} -> {len(keep)} points, {len(kept_ao)} of "
                        f"{self.mol.nao_nr()} AOs screened in")

            coords_out.append(coords_a[keep])
            weights_out.append(w_a[keep])

        coords_out = np.vstack(coords_out)
        weights_out = np.concatenate(weights_out)

        R = _eval_basefuncs(self.mol, coords_out)
        rmsd_out = _rmsd_overlap(R, weights_out, S)
        logger.info(f"NNLS grid: kept {len(coords_out)} of {n_in} points "
                    f"({100.0 * (1.0 - len(coords_out) / n_in):.1f}% pruned), "
                    f"reproduces S to {rmsd_out:.3e}")

        if active:
            active.log_metric("grid_points_in", n_in)
            active.log_metric("grid_points", len(coords_out))
            active.log_metric("rmsd_S_out", rmsd_out)
            active.log_metric("sum_weights", float(weights_out.sum()))

        return coords_out, weights_out

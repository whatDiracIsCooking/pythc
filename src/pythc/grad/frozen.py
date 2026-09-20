"""
A frozen atom-centred grid, and the bookkeeping a gradient needs from it.

The offline object this programme produces is a point set per element, stored relative to
the nucleus. At runtime a molecule's grid is the union of those sets translated onto its
nuclei - the one geometry-dependent operation in the whole scheme, and a rigid
translation at that.

:class:`FrozenGrid` is that object. It is a :class:`~pythc.grid.GridProvider`, so it drops
straight into any THC class in the package, and it additionally exposes the thing a
gradient cannot do without: **which atom each point belongs to**. Without that map there
is no way to say how a point moves when a nucleus does, and the translation term of
``dX/dR`` - half of the derivative, and the half that only exists because the grid is
frozen - cannot be written down.
"""
import numpy as np
from pyscf import gto

from pythc.grid import GridProvider


class FrozenGrid(GridProvider):
    """
    Per-atom point sets, attached to nuclei by rigid translation.

    :param mol: The molecule to attach to. The grid is rebuilt from ``mol``'s current
        coordinates on every :meth:`build`, so moving the nuclei moves the grid - which
        is exactly the intended behaviour and what makes finite-difference checks against
        this object meaningful.
    :param per_atom: One ``(coords, weights)`` pair per atom, with ``coords`` of shape
        ``(n_a, 3)`` given **relative to that atom's nucleus**, in Bohr. This is the
        format ``rotate.blocked_fit_per_atom`` and ``ghosts.per_atom_sets`` both return.
    :param rotations: Optional per-atom rotation matrices applied about each nucleus
        before translation, for probing orientation dependence. ``None`` means attach as
        fitted.
    """

    def __init__(self, mol: gto.Mole, per_atom, rotations=None):
        if len(per_atom) != mol.natm:
            raise ValueError(f"{len(per_atom)} point sets for {mol.natm} atoms")

        self.mol = mol
        self.rel = [np.asarray(c, dtype=float).reshape(-1, 3) for c, _ in per_atom]
        self.w = [np.asarray(w, dtype=float).ravel() for _, w in per_atom]
        self.rotations = None if rotations is None else [
            r.as_matrix() if hasattr(r, "as_matrix") else np.asarray(r, dtype=float)
            for r in rotations]

        for ia, (c, w) in enumerate(zip(self.rel, self.w)):
            if len(c) != len(w):
                raise ValueError(f"atom {ia}: {len(c)} points but {len(w)} weights")

    def __str__(self):
        return "frozen"

    def __repr__(self):
        return (f"FrozenGrid({self.mol.natm} atoms, {self.n_points} points"
                + (", rotated)" if self.rotations is not None else ")"))

    @property
    def n_points(self) -> int:
        return int(sum(len(w) for w in self.w))

    @property
    def atom_index(self) -> np.ndarray:
        """``(n_points,)`` array giving the atom each point rides with."""
        return np.concatenate([np.full(len(w), ia, dtype=np.intp)
                               for ia, w in enumerate(self.w)])

    def oriented(self) -> list:
        """Each atom's point set in its attached orientation, still nucleus-relative."""
        if self.rotations is None:
            return list(self.rel)

        return [c @ m.T for c, m in zip(self.rel, self.rotations)]

    def build(self):
        """``(coords, weights)`` at ``mol``'s current geometry."""
        coords = [c + self.mol.atom_coord(ia) for ia, c in enumerate(self.oriented())]

        return np.vstack(coords), np.concatenate(self.w)

    def displaced(self, coords_bohr: np.ndarray) -> "FrozenGrid":
        """
        The same frozen grid attached to a copy of ``mol`` at a new geometry.

        This is what a finite-difference check needs: the point sets must not be
        re-selected, re-fitted or re-oriented, only carried to the new nuclear positions.
        """
        moved = self.mol.copy()
        moved.set_geom_(np.asarray(coords_bohr, dtype=float), unit="Bohr")
        moved.build(False, False)

        per_atom = list(zip(self.rel, self.w))

        return FrozenGrid(moved, per_atom, self.rotations)

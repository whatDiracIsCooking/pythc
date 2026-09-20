"""
Anisotropy probe for frozen per-atom THC grids.

A transferable atomic grid is rigidly attached to its atom at whatever orientation the
offline fit happened to produce, which in a molecule is arbitrary relative to the bonds.
If the fitted point set is angularly anisotropic, that arbitrary orientation shows up in
the energy. Here we fit the blocked (per-atom) NNLS grid once, then spin each atom's
point set about its OWN nucleus by a random rotation - the molecule, the AOs and the
weights are untouched - and watch the MP2 energy move. The spread is the error a frozen
atomic grid would inherit purely from how it happens to be oriented.

The same rotations applied to a truly isotropic (whole-orbit) grid would leave the
energy invariant up to the quadrature's band limit. Pass ``--orbits`` to fit such a
grid: the NNLS variables become the octahedral orbits of each atomic sub-grid rather
than its points, so nothing is kept in pieces. See ``orbits.py``, which runs both and
puts the point-count cost against the invariance gained.
"""
import os, sys, json

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial.transform import Rotation

from pythc.decomp.nnls import GroupOperator, OverlapFitOperator, lawson_hanson
from pythc.grid import (BeckeGrid, GridProvider, _eval_basefuncs, _rmsd_overlap,
                        octahedral_orbits)
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS


def blocked_fit_per_atom(mol, threshold, screening=1e-8, group_orbits=False):
    """Per-atom NNLS fit, keeping each atom's surviving points separate.

    Mirrors NNLSGrid._build_blocked, but returns the points grouped by atom and
    expressed relative to their parent nucleus, which is what a frozen atomic grid
    would store.

    With ``group_orbits`` the variables are the octahedral orbits of each atom's
    sub-grid rather than its points, so the surviving set is a union of whole orbits
    carrying one weight each - the point set this experiment exists to test.
    """
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    coords_per_atom, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)

    out = []
    for ia, (coords_a, weights_a) in enumerate(zip(coords_per_atom, weights_per_atom)):
        R_a = _eval_basefuncs(mol, coords_a)
        amp = np.abs(R_a).max(axis=0)
        kept_ao = np.flatnonzero(amp > screening * amp.max())
        R_a = R_a[:, kept_ao]
        S_a = (R_a * weights_a[:, np.newaxis]).T @ R_a

        op = OverlapFitOperator(R_a)
        if group_orbits:
            grouped = GroupOperator(op, octahedral_orbits(atom_grids[mol.atom_symbol(ia)][0]))
            w_a = grouped.expand(lawson_hanson(grouped, op.pack(S_a),
                                               weight_threshold=threshold))
        else:
            w_a = lawson_hanson(op, op.pack(S_a), weight_threshold=threshold)
        keep = np.flatnonzero(w_a)
        # store relative to the nucleus: that is what travels with the atom
        out.append((coords_a[keep] - mol.atom_coord(ia), w_a[keep]))
    return out


class FixedGrid(GridProvider):
    """Hands back a grid that was built elsewhere."""
    def __init__(self, coords, weights):
        self.coords, self.weights = coords, weights

    def __str__(self):
        return "fixed"

    def build(self):
        return self.coords, self.weights


def assemble(mol, per_atom, rotations=None):
    """Place each atom's point set at its nucleus, optionally spun about that nucleus.

    ``rotations`` may be ``Rotation`` objects or plain 3x3 matrices; the octahedral
    elements ``orbits.py`` uses are easier to write down directly.
    """
    coords, weights = [], []
    for ia, (rel, w) in enumerate(per_atom):
        if rotations is None:
            r = rel
        else:
            m = rotations[ia]
            m = m.as_matrix() if hasattr(m, 'as_matrix') else np.asarray(m)
            r = rel @ m.T
        coords.append(r + mol.atom_coord(ia))
        weights.append(w)
    return np.vstack(coords), np.concatenate(weights)


def shell_structure(mol, per_atom, threshold, tol=1e-6):
    """How much of each radial shell survives: whole shells, or fragments?"""
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    report = []
    for ia, (rel, w) in enumerate(per_atom):
        parent = atom_grids[mol.atom_symbol(ia)][0]           # coords about the nucleus
        radii_parent = np.linalg.norm(parent, axis=1)
        shells = np.unique(np.round(radii_parent, 6))
        radii_kept = np.round(np.linalg.norm(rel, axis=1), 6)
        frac = []
        for s in shells:
            n_tot = int(np.sum(np.abs(radii_parent - s) < tol))
            n_kept = int(np.sum(np.abs(radii_kept - s) < tol))
            if n_tot:
                frac.append(n_kept / n_tot)
        frac = np.array(frac)
        report.append(dict(
            atom=mol.atom_symbol(ia), n_shells=len(frac),
            shells_empty=int(np.sum(frac == 0.0)),
            shells_full=int(np.sum(frac == 1.0)),
            shells_partial=int(np.sum((frac > 0.0) & (frac < 1.0))),
            mean_partial_fraction=float(frac[(frac > 0) & (frac < 1)].mean())
                                  if np.any((frac > 0) & (frac < 1)) else None))
    return report


def orbit_structure(mol, per_atom, tol=1e-6):
    """How much of each *orbit* survives - the level at which invariance is decided.

    A shell can be partial and the grid still octahedrally invariant, as long as every
    orbit it keeps is kept entire; a partial orbit is invariant under nothing. So this,
    not ``shell_structure``, is the diagnostic that says whether a frozen atomic grid
    has a well-defined orientation.
    """
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    report = []
    for ia, (rel, w) in enumerate(per_atom):
        parent = atom_grids[mol.atom_symbol(ia)][0]
        labels = octahedral_orbits(parent)

        # Which parent point is each retained point?
        distance = np.linalg.norm(rel[:, None, :] - parent[None, :, :], axis=2)
        onto = labels[distance.argmin(axis=1)]

        sizes = np.bincount(labels)
        kept, counts = np.unique(onto, return_counts=True)
        whole = int(np.sum(counts == sizes[kept]))
        report.append(dict(
            atom=mol.atom_symbol(ia), n_orbits=int(sizes.size),
            orbits_touched=int(kept.size), orbits_whole=whole,
            orbits_partial=int(kept.size - whole),
            mean_partial_fraction=float(np.mean((counts / sizes[kept])[counts < sizes[kept]]))
                                  if whole < kept.size else None))
    return report


def main(name, threshold, n_rot=6, seed=7, group_orbits=False):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]

    per_atom = blocked_fit_per_atom(mol, threshold, group_orbits=group_orbits)
    S = mol.intor('int1e_ovlp_sph')

    def energy(rots):
        coords, weights = assemble(mol, per_atom, rots)
        thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                          grid=FixedGrid(coords, weights))
        e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
        rmsd = _rmsd_overlap(_eval_basefuncs(mol, coords), weights, S)
        return e, rmsd

    e0, rmsd0 = energy(None)
    n_pts = sum(len(w) for _, w in per_atom)
    print(f"== {name}  thr={threshold:.0e}  {n_pts} points  "
          f"{'orbit-grouped' if group_orbits else 'point-wise'}  RI-MP2 ref {ref:.8f}",
          flush=True)
    print(f"  as fitted            E={e0:.12f}  err={1e6*(e0-ref):+9.2f} uHa  rmsd_S={rmsd0:.3e}",
          flush=True)

    rng = np.random.default_rng(seed)
    energies, rmsds = [], []
    for k in range(n_rot):
        rots = Rotation.random(mol.natm, random_state=int(rng.integers(1 << 30)))
        e, rmsd = energy(rots)
        energies.append(e); rmsds.append(rmsd)
        print(f"  random rotation {k}    E={e:.12f}  err={1e6*(e-ref):+9.2f} uHa  "
              f"rmsd_S={rmsd:.3e}  shift vs fitted={1e6*(e-e0):+9.2f} uHa", flush=True)

    energies = np.array(energies)
    print(f"  --> spread over orientations: {1e6*np.ptp(energies):.2f} uHa peak-to-peak, "
          f"std {1e6*energies.std():.2f} uHa", flush=True)
    print(f"  --> overlap RMSD: fitted {rmsd0:.3e} -> rotated mean {np.mean(rmsds):.3e}",
          flush=True)

    print("  orbit structure of the fitted atomic grids:", flush=True)
    for r in orbit_structure(mol, per_atom):
        print(f"    {r['atom']}: {r['orbits_touched']} of {r['n_orbits']} orbits touched "
              f"-> {r['orbits_whole']} whole, {r['orbits_partial']} partial"
              + (f" (mean kept fraction {r['mean_partial_fraction']:.2f})"
                 if r['mean_partial_fraction'] else ""), flush=True)

    print("  shell structure of the fitted atomic grids:", flush=True)
    for r in shell_structure(mol, per_atom, threshold):
        print(f"    {r['atom']}: {r['n_shells']} shells -> {r['shells_empty']} empty, "
              f"{r['shells_full']} full, {r['shells_partial']} partial"
              + (f" (mean kept fraction {r['mean_partial_fraction']:.2f})"
                 if r['mean_partial_fraction'] else ""), flush=True)

    return dict(molecule=name, threshold=threshold, group_orbits=group_orbits,
                n_points=n_pts, e_fitted=e0,
                e_rotated=[float(x) for x in energies], mp2_ref=float(ref),
                rmsd_fitted=rmsd0, rmsd_rotated=[float(x) for x in rmsds],
                shells=shell_structure(mol, per_atom, threshold),
                orbits=orbit_structure(mol, per_atom))


if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--orbits']
    group_orbits = '--orbits' in sys.argv
    name = argv[0] if len(argv) > 0 else 'water'
    thr = float(argv[1]) if len(argv) > 1 else 1e-4
    out = argv[2] if len(argv) > 2 else None
    res = main(name, thr, group_orbits=group_orbits)
    if out:
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=2)

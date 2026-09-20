"""
Anisotropy probe for frozen per-atom THC grids.

A transferable atomic grid is rigidly attached to its atom at whatever orientation the
offline fit happened to produce, which in a molecule is arbitrary relative to the bonds.
If the fitted point set is angularly anisotropic, that arbitrary orientation shows up in
the energy. Here we fit the blocked (per-atom) NNLS grid once, then spin each atom's
point set about its OWN nucleus by a random rotation - the molecule, the AOs and the
weights are untouched - and watch the MP2 energy move. The spread is the error a frozen
atomic grid would inherit purely from how it happens to be oriented.

The same rotations applied to a truly isotropic (whole-Lebedev-shell) grid would leave
the energy invariant up to the quadrature's band limit.
"""
import os, sys, json

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial.transform import Rotation

from pythc.decomp.nnls import OverlapFitOperator, lawson_hanson
from pythc.grid import BeckeGrid, GridProvider, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS


def blocked_fit_per_atom(mol, threshold, screening=1e-8):
    """Per-atom NNLS fit, keeping each atom's surviving points separate.

    Mirrors NNLSGrid._build_blocked, but returns the points grouped by atom and
    expressed relative to their parent nucleus, which is what a frozen atomic grid
    would store.
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
    """Place each atom's point set at its nucleus, optionally spun about that nucleus."""
    coords, weights = [], []
    for ia, (rel, w) in enumerate(per_atom):
        r = rel if rotations is None else rel @ rotations[ia].as_matrix().T
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


def main(name, threshold, n_rot=6, seed=7):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]

    per_atom = blocked_fit_per_atom(mol, threshold)
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
    print(f"== {name}  thr={threshold:.0e}  {n_pts} points  RI-MP2 ref {ref:.8f}", flush=True)
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

    print("  shell structure of the fitted atomic grids:", flush=True)
    for r in shell_structure(mol, per_atom, threshold):
        print(f"    {r['atom']}: {r['n_shells']} shells -> {r['shells_empty']} empty, "
              f"{r['shells_full']} full, {r['shells_partial']} partial"
              + (f" (mean kept fraction {r['mean_partial_fraction']:.2f})"
                 if r['mean_partial_fraction'] else ""), flush=True)

    return dict(molecule=name, threshold=threshold, n_points=n_pts, e_fitted=e0,
                e_rotated=[float(x) for x in energies], mp2_ref=float(ref),
                rmsd_fitted=rmsd0, rmsd_rotated=[float(x) for x in rmsds],
                shells=shell_structure(mol, per_atom, threshold))


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'water'
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    out = sys.argv[3] if len(sys.argv) > 3 else None
    res = main(name, thr)
    if out:
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=2)

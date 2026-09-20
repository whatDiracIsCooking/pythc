"""
What orbit-grouped NNLS pruning costs, and what it buys.

`rotate.py` established that a per-atom NNLS grid keeps only 8-20% of each Lebedev
shell, so a grid frozen per element has no well-defined orientation: spinning each
atom's point set about its own nucleus moves the MP2 energy by tens of uHa. This script
measures the proposed fix.

The fix is to make the NNLS variables the **octahedral orbits** of each atomic sub-grid
rather than its individual points (`pythc.grid.octahedral_orbits`,
`pythc.decomp.nnls.GroupOperator`). Orbits are kept or dropped entire and carry one
weight each, so the retained grid is a union of whole `O_h` orbits - exactly invariant
under the 24 proper rotations of the octahedral group, and no longer arbitrary in its
orientation to within the band limit of what survives.

Three numbers per fit, over a range of thresholds and for both modes:

* **points** - the tax. The extra points come straight off the 1.2-1.7x budget of
  FINDINGS section 1, so this has to be read at matched accuracy, not at matched
  threshold: the grouped fit's KKT gradient is a sum over up to 48 points, so the same
  numeric threshold is a much tighter one.
* **octahedral shift** - applying a random *element of the group* per atom. For a
  grouped grid this permutes points within their orbits and must leave the energy
  unchanged to round-off. It is the sharp, cheap check that the grouping is real.
* **general spread** - applying a random SO(3) rotation per atom, as `rotate.py` does.
  This is the quantity that has to fall, and it cannot fall to zero: a union of orbits
  integrates the sphere exactly only up to some degree, and anisotropy re-enters above
  it.

Usage:

    uv run python experiments/atom_centered_grids/orbits.py water --out orbits_water.json
"""
import argparse, itertools, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial.transform import Rotation

from pythc.grid import _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS
from rotate import FixedGrid, assemble, blocked_fit_per_atom, orbit_structure


def proper_octahedral_group():
    """The 24 rotation matrices of the octahedral group, as signed permutations."""
    elements = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            m[np.arange(3), perm] = signs
            if np.linalg.det(m) > 0:            # proper rotations only
                elements.append(m)
    assert len(elements) == 24
    return elements


OCTAHEDRAL = proper_octahedral_group()


def measure(mol, mf, ref, threshold, group_orbits, n_rot, rng):
    """Fit one grid and report its size, its accuracy and its orientation dependence."""
    S = mol.intor('int1e_ovlp_sph')

    t0 = time.time()
    per_atom = blocked_fit_per_atom(mol, threshold, group_orbits=group_orbits)
    fit_seconds = time.time() - t0

    def energy(rots):
        coords, weights = assemble(mol, per_atom, rots)
        thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                          grid=FixedGrid(coords, weights))
        e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
        return e, _rmsd_overlap(_eval_basefuncs(mol, coords), weights, S)

    e0, rmsd0 = energy(None)

    # One random element of the group per atom: a grouped grid is invariant under this
    # exactly, a point-wise one is not.
    octa = [OCTAHEDRAL[int(rng.integers(len(OCTAHEDRAL)))] for _ in range(mol.natm)]
    e_octa, _ = energy(octa)

    general = [energy([Rotation.random(random_state=int(rng.integers(1 << 30)))
                       for _ in range(mol.natm)])[0]
               for _ in range(n_rot)]
    general = np.array(general)

    structure = orbit_structure(mol, per_atom)
    n_kept = sum(len(w) for _, w in per_atom)
    whole = sum(r['orbits_whole'] for r in structure)
    touched = sum(r['orbits_touched'] for r in structure)

    return dict(
        threshold=threshold, group_orbits=group_orbits, n_points=n_kept,
        fit_seconds=fit_seconds, e_fitted=e0, err_uha=1e6 * (e0 - ref), rmsd_S=rmsd0,
        octahedral_shift_uha=1e6 * (e_octa - e0),
        general_ptp_uha=float(1e6 * np.ptp(general)),
        general_std_uha=float(1e6 * general.std()),
        e_general=[float(x) for x in general],
        orbits_touched=touched, orbits_whole=whole,
        orbits=structure)


def run(name, thresholds, orbit_thresholds, n_rot, seed, out_path):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]

    meta = dict(molecule=name, natm=mol.natm, nao=int(mol.nao_nr()), basis=BASIS,
                auxbasis=AUXBASIS, mp2_ri_reference=float(ref), n_rotations=n_rot,
                seed=seed)
    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, RI-MP2 ref {ref:.8f}",
          flush=True)
    print(f"{'mode':<12} {'thr':>8} {'points':>7} {'err/uHa':>10} {'octa':>10} "
          f"{'ptp/uHa':>9} {'std/uHa':>9} {'whole orbits':>14}", flush=True)

    rows = []
    for group_orbits, thrs in ((False, thresholds), (True, orbit_thresholds)):
        # Same rotations for every row, so the comparison is between grids and not
        # between draws.
        for thr in thrs:
            rng = np.random.default_rng(seed)
            try:
                row = measure(mol, mf, ref, thr, group_orbits, n_rot, rng)
            except Exception as exc:                          # keep the sweep going
                print(f"  FAILED thr={thr:.0e} orbits={group_orbits}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                rows.append(dict(threshold=thr, group_orbits=group_orbits,
                                 error=f"{type(exc).__name__}: {exc}"))
                continue
            rows.append(row)
            print(f"{'orbits' if group_orbits else 'point-wise':<12} {thr:>8.0e} "
                  f"{row['n_points']:>7d} {row['err_uha']:>+10.2f} "
                  f"{row['octahedral_shift_uha']:>+10.2f} {row['general_ptp_uha']:>9.2f} "
                  f"{row['general_std_uha']:>9.2f} "
                  f"{row['orbits_whole']:>6d}/{row['orbits_touched']:<7d}", flush=True)
            with open(out_path, 'w') as fh:
                json.dump(dict(meta=meta, rows=rows), fh, indent=2)

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('molecule', choices=sorted(MOLECULES))
    p.add_argument('--out', required=True)
    p.add_argument('--thresholds', default='1e-3,1e-4,1e-5',
                   help='weight thresholds for the point-wise fits')
    p.add_argument('--orbit-thresholds', default='1e-2,1e-3,1e-4',
                   help='weight thresholds for the orbit-grouped fits. The grouped '
                        'gradient sums over the orbit, so equal numbers are not '
                        'equally tight and the two ladders have to be compared at '
                        'matched accuracy instead')
    p.add_argument('--rotations', type=int, default=6)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    run(a.molecule,
        [float(x) for x in a.thresholds.split(',') if x],
        [float(x) for x in a.orbit_thresholds.split(',') if x],
        a.rotations, a.seed, a.out)

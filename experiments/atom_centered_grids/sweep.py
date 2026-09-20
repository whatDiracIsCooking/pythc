"""
Blocked (per-atom) vs global NNLS grid fitting: point count at matched accuracy.

Measures the one number the atom-centered THC proposal hinges on - how much bigger a
per-atom-fitted grid is than a molecularly-fitted one at the same MP2 error. Because the
blocked fit sees the *real* neighbours of each atom (not ghost approximations) and prunes
point-wise (no isotropy constraint), its inflation over the global fit is a strict LOWER
BOUND on what any frozen, transferable atom-centred scheme would pay.
"""
import json, os, sys, time, argparse

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.grid import BeckeGrid, NNLSGrid, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

AUXBASIS = 'cc-pvdz-ri'
BASIS = 'cc-pvdz'

WATER = """
H        0.087529        0.023820        0.930805
O        0.657172        0.599414        0.406256
H        0.792448        1.344387        1.004310
"""


def geom_from_smiles(smiles, seed=0xf00d):
    """3D geometry via RDKit ETKDG + MMFF, as an XYZ block for PySCF."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    AllChem.EmbedMolecule(m, ps)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    conf = m.GetConformer()
    return "\n".join(
        f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.8f} "
        f"{conf.GetAtomPosition(a.GetIdx()).y:.8f} {conf.GetAtomPosition(a.GetIdx()).z:.8f}"
        for a in m.GetAtoms())


MOLECULES = {
    'water':    lambda: WATER,
    'methanol': lambda: geom_from_smiles('CO'),
    'ethanol':  lambda: geom_from_smiles('CCO'),
    'alanine':  lambda: geom_from_smiles('C[C@@H](N)C(=O)O'),
}


def mp2_on_grid(mol, mf, grid_builder):
    """LS-THC MP2 correlation energy on a given grid, plus the grid's size and S-RMSD."""
    coords, weights = grid_builder.build()
    R = _eval_basefuncs(mol, coords)
    rmsd = _rmsd_overlap(R, weights, mol.intor('int1e_ovlp_sph'))

    thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, grid=grid_builder)
    eri = thc.build(mode='ov')
    e_corr = LaplaceRMP2(mol, mf, eri, n_laplace=10).kernel()
    return len(coords), rmsd, float(e_corr)


def run(name, thresholds_global, thresholds_blocked, out_path):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    mp2_ref = DFMP2(mf).kernel()[0]

    rows = []
    meta = dict(molecule=name, natm=mol.natm, nao=int(mol.nao_nr()),
                basis=BASIS, auxbasis=AUXBASIS, mp2_ri_reference=float(mp2_ref))
    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, RI-MP2 ref {mp2_ref:.8f}",
          flush=True)

    # Unpruned parent grid: the accuracy ceiling the fits are working against.
    t0 = time.time()
    n, rmsd, e = mp2_on_grid(mol, mf, BeckeGrid(mol))
    rows.append(dict(mode='becke', threshold=None, n_points=n, rmsd_S=rmsd,
                     e_corr=e, err_uha=(e - mp2_ref) * 1e6, fit_seconds=time.time() - t0))
    print(f"  becke      n={n:6d}  rmsd={rmsd:.3e}  err={rows[-1]['err_uha']:+10.2f} uHa",
          flush=True)

    for mode, thresholds in (('global', thresholds_global), ('blocked', thresholds_blocked)):
        for thr in thresholds:
            t0 = time.time()
            try:
                g = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=thr,
                             blocked=(mode == 'blocked'))
                n, rmsd, e = mp2_on_grid(mol, mf, g)
            except Exception as exc:                      # keep the sweep going
                print(f"  {mode:8s} thr={thr:.0e}  FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                rows.append(dict(mode=mode, threshold=thr, error=f"{type(exc).__name__}: {exc}"))
                continue
            dt = time.time() - t0
            rows.append(dict(mode=mode, threshold=thr, n_points=n, rmsd_S=rmsd,
                             e_corr=e, err_uha=(e - mp2_ref) * 1e6, fit_seconds=dt))
            print(f"  {mode:8s} thr={thr:.0e}  n={n:6d} ({n / mol.natm:6.1f}/atom)  "
                  f"rmsd={rmsd:.3e}  err={rows[-1]['err_uha']:+10.2f} uHa  {dt:7.1f}s",
                  flush=True)
            with open(out_path, 'w') as fh:
                json.dump(dict(meta=meta, rows=rows), fh, indent=2)

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('molecule', choices=sorted(MOLECULES))
    p.add_argument('--out', required=True)
    p.add_argument('--global-thresholds', default='1e-3,1e-4,1e-5,1e-6')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5,1e-6')
    a = p.parse_args()
    run(a.molecule,
        [float(x) for x in a.global_thresholds.split(',') if x],
        [float(x) for x in a.blocked_thresholds.split(',') if x],
        a.out)

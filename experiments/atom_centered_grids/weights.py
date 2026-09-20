"""
Do the fitted NNLS weights matter, or only the points they select?

LS-THC fits Z by least squares, so rescaling the collocation matrix by any positive
diagonal D is absorbed exactly: X -> D X is compensated by Z -> D^-1 Z D^-1, leaving the
fitted ERI unchanged. The quadrature weights should therefore influence LS-THC only
through the conditioning of the metric inversion, not through accuracy - which would mean
the NNLS fit's real product is its SUPPORT (which points survive), not the weight values.

Test: keep the NNLS-selected points, swap the weights for alternatives, compare MP2.
"""
import os, sys, json

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial import cKDTree

from pythc.grid import BeckeGrid, NNLSGrid, GridProvider, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_thc_funcs import build_S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS


class FixedGrid(GridProvider):
    def __init__(self, coords, weights):
        self.coords, self.weights = coords, weights

    def __str__(self):
        return "fixed"

    def build(self):
        return self.coords, self.weights


def main(name, threshold, blocked=False):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]
    nocc = mol.nelectron // 2
    S = mol.intor('int1e_ovlp_sph')

    coords, w_nnls = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=threshold,
                              blocked=blocked).build()

    # the parent grid's own tabulated weights at the same points
    pc, pw = BeckeGrid(mol).build()
    _, idx = cKDTree(pc).query(coords)
    w_becke = pw[idx]

    rng = np.random.default_rng(0)
    variants = {
        'NNLS fitted': w_nnls,
        'parent Becke': w_becke,
        'all ones': np.ones_like(w_nnls),
        'uniform (mean of NNLS)': np.full_like(w_nnls, w_nnls.mean()),
        'random log-uniform x1000': w_nnls.mean() * 10 ** rng.uniform(-1.5, 1.5, len(w_nnls)),
    }

    print(f"== {name}  {'blocked' if blocked else 'global'} thr={threshold:.0e}  "
          f"{len(coords)} points  RI-MP2 ref {ref:.8f}", flush=True)
    print(f"  {'weights':26s} {'E_corr':>15s} {'err/uHa':>9s} {'rmsd_S':>10s} "
          f"{'-log10 minEig':>14s}", flush=True)

    out = []
    for label, w in variants.items():
        thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                          grid=FixedGrid(coords, w))
        e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
        R = _eval_basefuncs(mol, coords)
        X = np.sqrt(np.sqrt(w))[:, None] * R
        ev = np.sort(np.abs(np.linalg.eigvalsh(build_S('ov', X @ mf.mo_coeff, nocc))))[::-1]
        cond = -np.log10(max(ev[-1] / ev[0], 1e-300))
        rmsd = _rmsd_overlap(R, w, S)
        out.append(dict(weights=label, e_corr=e, err_uha=1e6 * (e - ref),
                        rmsd_S=rmsd, cond_log10=cond))
        print(f"  {label:26s} {e:15.9f} {1e6 * (e - ref):+9.2f} {rmsd:10.2e} {cond:14.1f}",
              flush=True)
    return out


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'methanol'
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    blocked = len(sys.argv) > 3 and sys.argv[3] == 'blocked'
    res = main(name, thr, blocked)
    out = [a for a in sys.argv if a.endswith('.json')]
    if out:
        json.dump(res, open(out[0], 'w'), indent=2)

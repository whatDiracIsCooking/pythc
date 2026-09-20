"""
What a THC grid is actually being asked to do: span the co-density manifold.

LS-THC fits Z by least squares, so the fitted ERI depends on the SPAN of the co-density
vectors {phi_i(r_P) phi_a(r_P)}_P, not on where the points sit or how well they
integrate anything. Two grids spanning the same space give the same energy. That makes
the useful diagnostics the effective rank of the co-density matrix and the conditioning
of the LS-THC metric S_PQ = (X X^T) o (X X^T), not the overlap RMSD.

Reports, per grid: effective rank at a few tolerances, and the eigenvalue spread of the
metric that LS-THC has to pseudo-invert.
"""
import os, sys, json

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.grid import BeckeGrid, NNLSGrid, _eval_basefuncs, _rmsd_overlap
from pythc.thc.ls_thc_funcs import build_S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS


def diagnostics(mol, mf, coords, weights):
    R = _eval_basefuncs(mol, coords)
    X = np.sqrt(np.sqrt(weights))[:, None] * R
    Xmo = X @ mf.mo_coeff
    nocc = mol.nelectron // 2

    # occupied-virtual co-density vectors, one row per grid point
    M = (Xmo[:, :nocc, None] * Xmo[:, None, nocc:]).reshape(len(coords), -1)
    s = np.linalg.svd(M, compute_uv=False)
    s = s / s[0]

    # the metric LS-THC pseudo-inverts
    Smetric = build_S('ov', Xmo, nocc)
    ev = np.linalg.eigvalsh(Smetric)
    ev = np.sort(np.abs(ev))[::-1]
    ev = ev / ev[0]

    return dict(
        n_points=len(coords),
        rmsd_S=_rmsd_overlap(R, weights, mol.intor('int1e_ovlp_sph')),
        rank_1e6=int(np.sum(s > 1e-6)), rank_1e8=int(np.sum(s > 1e-8)),
        rank_1e10=int(np.sum(s > 1e-10)),
        metric_rank_1e8=int(np.sum(ev > 1e-8)),
        metric_min_eig=float(ev[-1]),
        metric_cond_log10=float(-np.log10(max(ev[-1], 1e-300))),
    )


def main(name, thresholds):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    nocc, nvir = mol.nelectron // 2, mol.nao_nr() - mol.nelectron // 2
    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, "
          f"nocc*nvir = {nocc * nvir} (upper bound on co-density rank)", flush=True)

    out = []
    hdr = (f"  {'grid':22s} {'n_pts':>6s} {'rank@1e-6':>10s} {'rank@1e-8':>10s} "
           f"{'metricRank':>11s} {'-log10 minEig':>14s} {'rmsd_S':>10s}")
    print(hdr, flush=True)

    def row(label, coords, weights):
        d = diagnostics(mol, mf, coords, weights)
        d['grid'] = label
        out.append(d)
        print(f"  {label:22s} {d['n_points']:6d} {d['rank_1e6']:10d} {d['rank_1e8']:10d} "
              f"{d['metric_rank_1e8']:11d} {d['metric_cond_log10']:14.1f} "
              f"{d['rmsd_S']:10.2e}", flush=True)

    c, w = BeckeGrid(mol).build()
    row('becke (unpruned)', c, w)

    for mode in ('global', 'blocked'):
        for thr in thresholds:
            c, w = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=thr,
                            blocked=(mode == 'blocked')).build()
            row(f'{mode} {thr:.0e}', c, w)
    return out


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'water'
    thrs = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '1e-3,1e-4,1e-5').split(',')]
    res = main(name, thrs)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as fh:
            json.dump(res, fh, indent=2)

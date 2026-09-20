"""
What does ridge regularisation of the LS-THC metric cost in accuracy?

The metric S_PQ = (X X^T) o (X X^T) is rank deficient, and the pipeline currently
handles that by truncating its eigenvalues below a relative cutoff. That truncation is
a *discrete* decision taken at runtime: the number of surviving eigenvalues is an
integer function of the nuclear coordinates. With the grid frozen and the weights gone
it is the last non-smooth step left in the whole scheme (see `scan.py` for what that
costs on a potential energy surface).

Ridge - (S + lambda I)^-1, the Tikhonov solution of the same least-squares fit - damps
the small eigenvalues instead of dropping them, and is analytic in S. The question this
script answers is what that swap costs in energy, and over how wide a range of lambda
the answer is "nothing much". A regulariser that only works in a narrow window would be
no improvement over a threshold.

Both knobs are reported on the same footing, as a shift relative to the largest
eigenvalue, so pinv's `epsilon` and ridge's `lambda` can be read against each other.
"""
import os, sys, json, argparse

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc import lib
from pythc.grid import BeckeGrid, NNLSGrid, _eval_basefuncs
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_thc_funcs import build_S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS

EPSILONS = [1e-8, 1e-10, 1e-12, 1e-14]
LAMBDAS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]


def metric_spectrum(mol, mf, coords, weights):
    """Eigenvalues of the metric LS-THC inverts, normalised to the largest."""
    X = np.sqrt(np.sqrt(weights))[:, None] * _eval_basefuncs(mol, coords)
    S = build_S('ov', X @ mf.mo_coeff, mol.nelectron // 2)
    ev = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]

    return S, ev / ev[0]


def main(name, threshold, blocked, out_path=None):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]

    grid = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=threshold, blocked=blocked)
    coords, weights = grid.build()
    S, ev = metric_spectrum(mol, mf, coords, weights)

    # pinv's default cutoff, and how crowded the spectrum is around it - the crowd is
    # what turns nuclear motion into eigenvalue crossings.
    near = int(np.sum((ev > 1e-11) & (ev < 1e-9)))
    print(f"== {name}  {'blocked' if blocked else 'global'} thr={threshold:.0e}  "
          f"{len(coords)} points  RI-MP2 ref {ref:.8f}", flush=True)
    print(f"   metric: min eig {ev[-1]:.1e} (relative), {int(np.sum(ev > 1e-10))} of "
          f"{len(ev)} above pinv's default cutoff, {near} within a decade of it",
          flush=True)
    print(f"   {'inversion':28s} {'E_corr':>15s} {'err/uHa':>9s} {'n_kept':>7s} "
          f"{'shift/maxeig':>13s}", flush=True)

    rows = []

    def run(label, n_kept, shift_rel, **kwargs):
        thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, grid=grid,
                          **kwargs)
        e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
        rows.append(dict(inversion=label, e_corr=e, err_uha=1e6 * (e - ref),
                         n_kept=n_kept, shift_rel=shift_rel, **kwargs))
        print(f"   {label:28s} {e:15.9f} {1e6 * (e - ref):+9.2f} "
              f"{'-' if n_kept is None else n_kept:>7} {shift_rel:13.1e}", flush=True)

    # Truncation, at the default cutoff and around it. pinv floors its threshold at an
    # absolute 1e-12, so the tightest epsilons here are not all distinct.
    for eps in EPSILONS:
        cut = max(eps * np.abs(np.linalg.eigvalsh(S)).max(), 1e-12)
        n_kept = int(np.sum(np.linalg.eigvalsh(S) > cut))
        label = f"pinv eps={eps:.0e}" + (" (default)" if eps == 1e-10 else "")
        run(label, n_kept, eps)

    # Ridge. `shift` is reported relative to the largest eigenvalue so it lands on the
    # same axis as epsilon; lambda itself is scaled by the mean eigenvalue, which is
    # what keeps it a smooth function of the geometry.
    for lam in LAMBDAS:
        shift_rel = lib.ridge_shift(S, lam, "trace") / np.abs(np.linalg.eigvalsh(S)).max()
        run(f"ridge lam={lam:.0e}", None, shift_rel, metric_ridge=lam, aux_ridge=lam)

    baseline = next(r['err_uha'] for r in rows if 'default' in r['inversion'])
    best = min((r for r in rows if r['inversion'].startswith('ridge')),
               key=lambda r: abs(r['err_uha']))
    print(f"   --> truncation at the default cutoff: {baseline:+.2f} uHa; "
          f"best ridge {best['inversion']}: {best['err_uha']:+.2f} uHa", flush=True)

    result = dict(molecule=name, threshold=threshold, blocked=blocked,
                  n_points=len(coords), mp2_ref=float(ref),
                  metric_min_eig=float(ev[-1]), n_near_cutoff=near, rows=rows)
    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('molecule', nargs='?', default='methanol', choices=sorted(MOLECULES))
    p.add_argument('threshold', nargs='?', type=float, default=1e-3)
    p.add_argument('--blocked', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()
    main(a.molecule, a.threshold, a.blocked, a.out)

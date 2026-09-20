"""
Does the metric truncation put a step in the potential energy surface, and does ridge
remove it?

Everything else in the frozen-grid scheme is smooth in the nuclear coordinates: the
point selection happens offline, the weights turn out not to matter (`weights.py`), the
collocation matrix X = phi(r_P) is an analytic function of the geometry, and the Z fit
is an ordinary least-squares solve. The one exception is that the metric is rank
deficient and its small eigenvalues are currently dropped below a relative cutoff. How
many survive is an integer, and integers do not vary smoothly - as nuclei move, an
eigenvalue crosses the cutoff and the energy jumps.

This scans a bond length with the grid FROZEN - the per-atom point sets are fitted once
at the reference geometry and thereafter only translated rigidly with their nuclei,
which is exactly the target pipeline - and watches for that jump. Reported per
inversion scheme:

* `E_thc - E_dfmp2`, the THC error, whose smooth part is small so a step stands out;
* the second difference of that curve, which reads a step off directly: for a smooth
  function it is O(h^2) and tiny, while a jump of size d contributes d;
* how many eigenvalues the truncation kept, which says whether a crossing happened.

A refitted-grid curve is included as the control: re-running the point selection at
every geometry is what freezing is meant to avoid, and its roughness is the scale
against which the truncation's own roughness should be read.
"""
import os, sys, json, argparse

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.grid import BeckeGrid, NNLSGrid, _eval_basefuncs
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_thc_funcs import build_S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS
from rotate import blocked_fit_per_atom, assemble, FixedGrid

BOHR = 0.52917721092


def parse_geometry(block):
    """The XYZ block the sweep hands out, as symbols and an (natm, 3) array in Angstrom."""
    symbols, coords = [], []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:]])

    return symbols, np.array(coords)


def default_bond(symbols, coords):
    """
    The bond to scan: a terminal hydrogen and the heaviest atom it is attached to.

    A terminal H needs no connectivity analysis to move - nothing hangs off it - so the
    scan stays a clean one-dimensional path through nuclear configuration space. The
    heaviest partner picks the most polar bond available, which perturbs the metric
    hardest for a given displacement.
    """
    charges = [gto.charge(s) for s in symbols]
    candidates = []
    for ih, s in enumerate(symbols):
        if charges[ih] != 1:
            continue
        distances = np.linalg.norm(coords - coords[ih], axis=1)
        distances[ih] = np.inf
        partner = int(np.argmin(distances))
        candidates.append((charges[partner], -ih, partner, ih))

    if not candidates:
        raise ValueError("no hydrogen to scan; pass --bond explicitly")

    _, _, heavy, light = max(candidates)

    return heavy, light


def build_mol(symbols, coords, anchor, mover, length):
    """The same molecule with the anchor-mover bond set to `length` Angstrom."""
    coords = coords.copy()
    axis = coords[mover] - coords[anchor]
    coords[mover] = coords[anchor] + axis / np.linalg.norm(axis) * length
    atom = [(s, tuple(c)) for s, c in zip(symbols, coords)]

    return gto.M(atom=atom, basis=BASIS, verbose=0)


def n_kept(mol, mf, coords, weights, epsilon=1e-10):
    """How many metric eigenvalues survive pinv's cutoff at this geometry."""
    X = np.sqrt(np.sqrt(weights))[:, None] * _eval_basefuncs(mol, coords)
    ev = np.linalg.eigvalsh(build_S('ov', X @ mf.mo_coeff, mol.nelectron // 2))

    return int(np.sum(ev > max(epsilon * ev.max(), 1e-12)))


def roughness(errors):
    """
    Second difference of the error curve, in uHa.

    On a uniform scan this is h^2 * f'' + O(h^4) for a smooth f - negligible at the step
    sizes used here - so anything large is a jump, and its size is the jump's size.
    """
    e = np.asarray(errors, float)
    d2 = e[:-2] - 2.0 * e[1:-1] + e[2:]

    return dict(max=float(np.max(np.abs(d2))), median=float(np.median(np.abs(d2))),
                d2=[float(x) for x in d2])


def main(name, threshold, bond=None, half_width=0.06, step=0.002, lambdas=(1e-8, 1e-10, 1e-12),
         refit=True, out_path=None):
    symbols, coords0 = parse_geometry(MOLECULES[name]())
    anchor, mover = bond if bond else default_bond(symbols, coords0)
    r0 = float(np.linalg.norm(coords0[mover] - coords0[anchor]))

    # Freeze the grid at the reference geometry: from here on the point sets only ride
    # along with their nuclei, which is the whole proposal.
    mol0 = build_mol(symbols, coords0, anchor, mover, r0)
    per_atom = blocked_fit_per_atom(mol0, threshold)
    n_points = sum(len(w) for _, w in per_atom)

    lengths = r0 + np.arange(-half_width, half_width + 0.5 * step, step)
    variants = [('pinv (default)', {})]
    variants += [(f'ridge {lam:.0e}', dict(metric_ridge=lam, aux_ridge=lam))
                 for lam in lambdas]

    print(f"== {name}  {symbols[anchor]}{anchor}-{symbols[mover]}{mover} bond, "
          f"{r0:.4f} +- {half_width:.3f} A in {len(lengths)} steps of {step:.4f} A",
          flush=True)
    print(f"   grid frozen at the reference geometry: blocked NNLS thr={threshold:.0e}, "
          f"{n_points} points", flush=True)

    curves = {label: [] for label, _ in variants}
    if refit:
        curves['refit + pinv'] = []
    kept, refs = [], []

    for r in lengths:
        mol = build_mol(symbols, coords0, anchor, mover, r)
        mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        mf.kernel()
        ref = DFMP2(mf).kernel()[0]
        refs.append(float(ref))

        c, w = assemble(mol, per_atom)
        kept.append(n_kept(mol, mf, c, w))

        for label, kwargs in variants:
            thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                              grid=FixedGrid(c, w), **kwargs)
            e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
            curves[label].append(1e6 * (e - ref))

        if refit:
            g = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=threshold, blocked=True)
            thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff, grid=g)
            e = float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())
            curves['refit + pinv'].append(1e6 * (e - ref))

        print(f"   r={r:.4f}  kept={kept[-1]:4d}  " +
              "  ".join(f"{k}={v[-1]:+8.2f}" for k, v in curves.items()), flush=True)

    changes = int(np.sum(np.diff(kept) != 0))
    print(f"\n   eigenvalues kept by the truncation: {min(kept)}-{max(kept)}, "
          f"changing at {changes} of {len(lengths) - 1} steps", flush=True)
    print(f"   {'curve':18s} {'range/uHa':>12s} {'max|d2|':>10s} {'median|d2|':>12s} "
          f"{'ratio':>8s}", flush=True)

    stats = {}
    for label, values in curves.items():
        r_ = roughness(values)
        stats[label] = r_
        ratio = r_['max'] / r_['median'] if r_['median'] > 0 else float('inf')
        print(f"   {label:18s} {np.ptp(values):12.2f} {r_['max']:10.4f} "
              f"{r_['median']:12.4f} {ratio:8.1f}", flush=True)

    result = dict(molecule=name, threshold=threshold, bond=[anchor, mover],
                  symbols=symbols, r0=r0, lengths=[float(x) for x in lengths],
                  n_points=n_points, n_kept=kept, mp2_ref=refs,
                  curves={k: [float(x) for x in v] for k, v in curves.items()},
                  roughness={k: {'max': v['max'], 'median': v['median']}
                             for k, v in stats.items()})
    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('molecule', nargs='?', default='methanol', choices=sorted(MOLECULES))
    p.add_argument('threshold', nargs='?', type=float, default=1e-3)
    p.add_argument('--bond', help='anchor,mover atom indices; default is a terminal X-H')
    p.add_argument('--half-width', type=float, default=0.06, help='scan half range in Angstrom')
    p.add_argument('--step', type=float, default=0.002, help='scan step in Angstrom')
    p.add_argument('--lambdas', default='1e-8,1e-10,1e-12')
    p.add_argument('--no-refit', action='store_true', help='skip the re-selected-grid control')
    p.add_argument('--out')
    a = p.parse_args()
    main(a.molecule, a.threshold,
         tuple(int(x) for x in a.bond.split(',')) if a.bond else None,
         a.half_width, a.step, [float(x) for x in a.lambdas.split(',') if x],
         not a.no_refit, a.out)

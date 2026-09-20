"""
Does the ghost ensemble matter?

This is the first half of step (4) of HANDOFF.md, and the cheapest open question in it.
Section 4(3) built a transferable grid and measured what it costs, but it did so with
*one* ghost ensemble - 12 icosahedral directions, partners H/C/O at three bond-length
multiples, one (partner, distance) combination cycled per direction - chosen once and
never varied. Nothing established that the answer is insensitive to that choice, and
until something does, the 1.6-2.7x headline is a measurement of one arbitrary ensemble
rather than of the scheme.

Two axes are varied, both already implemented in `ghosts.py` and neither previously run:

* **directions** - 12 icosahedral against 6 octahedral. Fewer directions is a coarser
  angular sample of what a neighbour can look like.
* **cycling** - one cycled (partner, scale) per direction against `--full-cross`, every
  combination in every direction. Full cross is `len(partners) * len(scales)` = 9x the
  environments, so it is the well-sampled limit the cheap ensemble is an approximation
  to.

The 2x2 is the point. If the cheap corner (octahedron, cycled: 7 environments) and the
expensive one (icosahedron, full cross: 109) reach the same accuracy at the same point
count, the ensemble does not matter, one cheap ensemble suffices, and the offline stage
of the scheme is finished. If they separate, the ensemble is a tuning knob and every
number in FINDINGS.md section 8 is conditional on it.

**How to read this, and the trap in it.** Point count at a fixed threshold is NOT
comparable across ensembles. NNLS retains at most `min(n_equations, n_variables)` points
and each environment contributes equations, so a bigger ensemble mechanically permits a
bigger support at the same KKT threshold - that is arithmetic, not quality. The same
applies to support overlap: HANDOFF.md section 4(3) already established that the ghost
fit reproduces only 24-31% of what `blocked` picks while landing within 1.1-1.8x of it
on points, and concluded "do not use support overlap as a quality metric; only point
count at matched energy means anything". So:

* stage A (`--calibrate`, no SCF, seconds) reports points per element and the Jaccard
  overlap of each ensemble's support against the baseline's. It is a *diagnostic* - it
  says whether the ensembles are picking different points, which they are entitled to
  do - and it exists to pick a threshold ladder, not to answer the question;
* stage B runs a threshold ladder per ensemble on a real molecule and reports the THC
  error against point count. **That curve is the answer.** Ensembles whose error-vs-
  points curves lie on top of each other are equivalent no matter how little their
  supports share; an ensemble that reaches a given error at fewer points is genuinely
  better.

Usage:

    # stage A: supports and overlaps, no SCF
    uv run python experiments/atom_centered_grids/ensemble.py --calibrate H,C,O

    # the ceiling each ensemble imposes, no SCF - read this before tuning a threshold
    uv run python experiments/atom_centered_grids/ensemble.py --saturate H,C,O

    # stage B: the decisive curve
    uv run python experiments/atom_centered_grids/ensemble.py methanol --out ens_methanol.json
    uv run python experiments/atom_centered_grids/analyse.py ens_methanol.json
"""
import argparse, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.grid import BeckeGrid, _eval_basefuncs, _rmsd_overlap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS
from rotate import blocked_fit_per_atom
from ghosts import (RIDGE, assemble_from_supports, element_grid, element_supports,
                    ghost_environments, thc_mp2)

# The 2x2. The first entry is the ensemble every number in FINDINGS.md section 8 was
# measured with, and the baseline every support overlap below is taken against.
ENSEMBLES = [
    ('icosa-cycled', dict(directions='icosahedron', full_cross=False)),
    ('icosa-full', dict(directions='icosahedron', full_cross=True)),
    ('octa-cycled', dict(directions='octahedron', full_cross=False)),
    ('octa-full', dict(directions='octahedron', full_cross=True)),
]

BASELINE = ENSEMBLES[0][0]


def jaccard(a, b):
    """|a & b| / |a | b| for two index arrays into the same parent grid."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return float('nan')
    return len(sa & sb) / len(sa | sb)


def covered(a, b):
    """Fraction of `a` that `b` also picks. Asymmetric, and the useful direction when
    the two supports are different sizes."""
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa) if sa else float('nan')


def saturate(elements, threshold=1e-10, out_path=None):
    """
    The largest support each ensemble can produce, at any threshold.

    Driving the KKT threshold to zero stops adding points long before the parent grid is
    exhausted: the Lawson-Hanson loop halts once the residual is orthogonal to every
    remaining column, so the support saturates at the *effective rank* of the stacked
    target. That number is a property of the ensemble alone, and it caps the accuracy
    any grid fitted from that ensemble can reach, however the threshold is tuned.

    This is the same failure mode as the free-atom fit of HANDOFF.md section 4(3) -
    a target that cannot support more points - but at a much higher ceiling and for a
    different reason. The free atom ran out of *equations*
    (`n_AO(n_AO+1)/2` = 15 for hydrogen). The ghost ensembles supply equations in the
    thousands; what they run out of is *independent* ones.
    """
    print(f"  {'element':>7s} {'parent':>7s}" +
          "".join(f" {n:>14s}" for n, _ in ENSEMBLES), flush=True)
    rows = []
    for symbol in elements:
        n_parent = len(element_grid(symbol))
        cells = []
        for name, kw in ENSEMBLES:
            idx = element_supports([symbol], threshold, quiet=True, **kw)[symbol]
            n_env = len(ghost_environments(symbol, **kw))
            cells.append(f" {len(idx):8d} ({n_env:3d})")
            rows.append(dict(element=symbol, ensemble=name, n_env=n_env,
                             n_saturated=int(len(idx)), n_parent=int(n_parent),
                             threshold=threshold))
        print(f"  {symbol:>7s} {n_parent:7d}" + "".join(cells), flush=True)
    print("  (support size at threshold -> 0, with the environment count in brackets)",
          flush=True)

    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(dict(meta=dict(stage='saturate', threshold=threshold,
                                     basis=BASIS), rows=rows), fh, indent=2)
    return rows


def calibrate(elements, thresholds, out_path=None):
    """
    Stage A: every ensemble's support for every element and threshold, no SCF.

    Reports point counts and, against the baseline ensemble, how much of the support is
    shared. Read the counts to pick a ladder for stage B; read the overlaps as a
    diagnostic only, for the reason in the module docstring.
    """
    rows = []
    for symbol in elements:
        n_parent = len(element_grid(symbol))
        print(f"\n== {symbol}: {n_parent} parent points", flush=True)
        print(f"   {'ensemble':<14s} {'n_env':>5s} " +
              " ".join(f"{f'{t:.0e}':>18s}" for t in thresholds), flush=True)

        base = {}
        for name, kw in ENSEMBLES:
            n_env = len(ghost_environments(symbol, **kw))
            cells, supports = [], {}
            for thr in thresholds:
                t0 = time.time()
                supports[thr] = element_supports([symbol], thr, quiet=True, **kw)[symbol]
                dt = time.time() - t0
                idx = supports[thr]
                if name == BASELINE:
                    cells.append(f"{len(idx):18d}")
                else:
                    cells.append(f"{len(idx):8d} (J={jaccard(idx, base[thr]):.2f})")
                rows.append(dict(element=symbol, ensemble=name, n_env=n_env,
                                 threshold=thr, n_points=int(len(idx)),
                                 n_parent=int(n_parent), fit_seconds=dt,
                                 support=[int(i) for i in idx]))
            if name == BASELINE:
                base = supports
            print(f"   {name:<14s} {n_env:5d} " + " ".join(cells), flush=True)

    # Overlap against the baseline, at each ensemble's own threshold, is the honest
    # comparison: matched threshold is not matched size, so also report the asymmetric
    # coverage, which does not punish an ensemble merely for selecting more points.
    print("\n== support vs baseline, matched threshold", flush=True)
    by_key = {(r['element'], r['ensemble'], r['threshold']): r for r in rows}
    for symbol in elements:
        for name, _ in ENSEMBLES[1:]:
            parts = []
            for thr in thresholds:
                a = np.array(by_key[(symbol, name, thr)]['support'])
                b = np.array(by_key[(symbol, BASELINE, thr)]['support'])
                parts.append(f"{thr:.0e}: {100 * covered(b, a):3.0f}% of base kept, "
                             f"{len(a) / max(len(b), 1):.2f}x size")
            print(f"   {symbol} {name:<14s} " + " | ".join(parts), flush=True)

    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(dict(meta=dict(stage='calibrate', elements=elements,
                                     thresholds=thresholds, basis=BASIS,
                                     ensembles=[n for n, _ in ENSEMBLES]),
                           rows=rows), fh, indent=2)
    return rows


def run(name, thresholds, out_path, blocked_thresholds):
    """
    Stage B: the error-against-point-count curve for each ensemble on a real molecule.

    The `becke` and `blocked` rows are computed once and shared by every ensemble - they
    do not depend on the ghost fit at all - so the only thing varying down the table is
    which ensemble selected the transferable grid.
    """
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]
    S = mol.intor('int1e_ovlp_sph')

    elements = sorted(set(mol.atom_symbol(ia) for ia in range(mol.natm)))
    rows = []
    meta = dict(molecule=name, natm=mol.natm, nao=int(mol.nao_nr()), basis=BASIS,
                auxbasis=AUXBASIS, mp2_ri_reference=float(ref), ridge=RIDGE,
                weights='ones', elements=elements, thresholds=thresholds,
                ensembles=[n for n, _ in ENSEMBLES], baseline=BASELINE)

    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, elements {elements}, "
          f"RI-MP2 ref {ref:.8f}", flush=True)

    def record(mode, threshold, coords, weights, dt, extra=None):
        e = thc_mp2(mol, mf, coords, weights)
        row = dict(mode=mode, threshold=threshold, n_points=len(coords),
                   rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                   e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=dt)
        row.update(extra or {})
        rows.append(row)
        print(f"  {mode:14s} thr={threshold if threshold is None else f'{threshold:.0e}'}"
              f"  n={len(coords):5d} ({len(coords) / mol.natm:5.1f}/atom)  "
              f"err={row['err_uha']:+9.2f} uHa", flush=True)
        with open(out_path, 'w') as fh:
            json.dump(dict(meta=meta, rows=rows), fh, indent=2)
        return row

    # Shared baselines: the unpruned parent grid (the accuracy floor) and the in-molecule
    # blocked fit (the lower bound on any transferable scheme).
    coords, weights = BeckeGrid(mol).build()
    t0 = time.time()
    record('becke', None, coords, weights, time.time() - t0)

    for thr in blocked_thresholds:
        t0 = time.time()
        per_atom = blocked_fit_per_atom(mol, thr)
        dt = time.time() - t0
        coords = np.vstack([rel + mol.atom_coord(ia)
                            for ia, (rel, _w) in enumerate(per_atom)])
        record('blocked', thr, coords, np.ones(len(coords)), dt)

    for ens_name, kw in ENSEMBLES:
        print(f"  -- ensemble {ens_name} "
              f"({len(ghost_environments(elements[0], **kw))} environments)", flush=True)
        for thr in thresholds:
            t0 = time.time()
            supports = element_supports(elements, thr, quiet=True, **kw)
            dt = time.time() - t0
            coords, weights = assemble_from_supports(mol, supports)
            record(ens_name, thr, coords, weights, dt,
                   extra=dict(ensemble=ens_name, n_env=len(ghost_environments(
                       elements[0], **kw)),
                       per_element={k: int(len(v)) for k, v in supports.items()},
                       support={k: [int(i) for i in v] for k, v in supports.items()}))

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('molecule', nargs='?', choices=sorted(MOLECULES))
    p.add_argument('--out')
    p.add_argument('--calibrate', help='comma-separated elements; prints points and '
                                       'support overlap per ensemble and exits, no SCF')
    p.add_argument('--saturate', help='comma-separated elements; prints the largest '
                                      'support each ensemble can reach at any '
                                      'threshold and exits, no SCF')
    p.add_argument('--thresholds', default='1e-3,3e-4,1e-4,3e-5,1e-5,1e-6',
                   help='KKT ladder, applied to every ensemble. Not on the blocked '
                        'scale, and not comparable between ensembles either - see the '
                        'module docstring')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5')
    a = p.parse_args()

    thresholds = [float(x) for x in a.thresholds.split(',') if x]

    if a.saturate:
        saturate(a.saturate.split(','), out_path=a.out)
    elif a.calibrate:
        calibrate(a.calibrate.split(','), thresholds, out_path=a.out)
    else:
        if not a.molecule or not a.out:
            p.error('a molecule and --out are required unless --calibrate is given')
        run(a.molecule, thresholds, a.out,
            [float(x) for x in a.blocked_thresholds.split(',') if x])

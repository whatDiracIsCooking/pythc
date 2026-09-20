"""
Transferability: does one element's point set serve environments it was not fitted in?

This is the second half of step (4) of HANDOFF.md, and what is left of the original
question once `ensemble.py` has answered the first half. Section 4(3) built a genuinely
offline, per-element grid and priced it; section 4(4) showed the *choice* of ghost
ensemble amortises. Neither ever used an element's point set anywhere but in the kind of
bonding its ghosts already described. Both measured on methanol and ethanol: one sp3
carbon bonded to one sp3 oxygen, over and over.

So nothing yet distinguishes "a transferable grid works" from "a grid fitted against
single bonds works on molecules made of single bonds". This script makes that
distinction. **One support per element is fitted once, offline, and then frozen** - the
same integer index list is handed to every molecule in the suite, and its fingerprint is
recorded in the output so that the claim is checkable rather than asserted.

What the suite is chosen to break
---------------------------------
The default ensemble places one ghost neighbour at ``(r_cov(A) + r_cov(B)) * s`` for
``s`` in ``(0.90, 1.05, 1.45)``, with partners H, C, O. Two kinds of environment fall
outside that by construction:

* **multiple bonds are shorter than the shortest ghost shell.** The tightest C-C ghost
  sits at 1.37 A and the tightest C-O at 1.28 A, while ethene's C=C is 1.34, acetylene's
  C#C is 1.20 and formaldehyde's C=O is 1.22. Those are not interpolations inside the
  training range, they are extrapolations below it - and the whole point of the ghost is
  to keep alive the tail points a bond needs, so a bond tighter than any ghost is the
  case where those points might not have been kept.
* **nitrogen is not in the partner list at all.** ``PARTNERS = ('H', 'C', 'O')``, so no
  carbon ghost environment has ever seen an N neighbour. HANDOFF.md 4(3) justified the
  omission on the grounds that N's basis is "all but indistinguishable from C's and O's
  at this level". That is an argument, not a measurement, and this is the measurement.

+----------------+--------------------------------------------------------------+
| molecule       | what it probes                                               |
+================+==============================================================+
| water          | O sp3, the most-fitted environment there is. Control.         |
| methanol       | O sp3 + C sp3. The molecule sections 8 and 9 were measured on.|
| ethane         | C sp3, C-C at 1.51 A - inside the ghost range. Control.        |
| ethene         | C sp2, C=C at 1.34 A - below the tightest ghost.               |
| acetylene      | C sp,  C#C at 1.20 A - far below it.                           |
| formaldehyde   | O sp2 + C sp2, C=O at 1.22 A - below the tightest C-O ghost.   |
| methylamine    | N sp3, C-N at 1.45 A - a partner the fit never saw.            |
| hcn            | C sp + N, C#N at 1.16 A - unseen partner AND far too short.    |
+----------------+--------------------------------------------------------------+

How to read the result
----------------------
Per molecule the script measures the same thing `ghosts.py` does - points needed to
reach a given MP2 accuracy, for the frozen grid and for that molecule's own in-molecule
`blocked` fit - and takes the ratio. That ratio is the ghost gap, and section 8 measured
it at 1.1-1.8x on molecules the ensemble describes well.

**Transferability is the spread of that ratio across the suite, not its value.** A
uniformly high ratio would mean per-element grids are expensive, which is already known
and already priced. A ratio that is flat on ethane and blows up on acetylene would mean
something quite different and much worse: that the offline object has to be refitted per
bonding environment, which is the one thing the scheme cannot afford, because "which
environment is this atom in" is a discrete function of geometry and putting one at
runtime undoes section 4(1) and the whole smooth-PES argument.

`--residual` is a no-SCF screen for the same thing: how well the frozen support can
represent each atom's REAL in-molecule overlap block, by unconstrained least squares
over those points only. It is an overlap-metric quantity, and HANDOFF.md section 5 is
explicit that `rmsd_S` is close to orthogonal to THC accuracy, so it is reported next to
the energies precisely so that whether it tracks them can be checked rather than
assumed.

Usage:

    # no SCF, seconds: what the frozen supports are, and the residual screen
    uv run python experiments/atom_centered_grids/transfer.py --residual

    # the measurement (long - one becke + blocked ladder + ghost ladder per molecule)
    uv run python experiments/atom_centered_grids/transfer.py --out transfer.json

    # nitrogen as a ghost partner, the control for the unseen-partner half
    uv run python experiments/atom_centered_grids/transfer.py --partners H,C,N,O \
        --molecules hcn,methylamine,methanol --out transfer_npartner.json

    uv run python experiments/atom_centered_grids/transfer.py --report transfer*.json
"""
import argparse, hashlib, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2

from pythc.decomp.nnls import OverlapFitOperator
from pythc.grid import BeckeGrid, _eval_basefuncs, _rmsd_overlap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import TARGETS, points_for
from ghosts import (COVALENT, PARTNERS, RIDGE, SCALES, assemble_from_supports,
                    element_grid, element_supports, thc_mp2)
from rotate import blocked_fit_per_atom
from sweep import AUXBASIS, BASIS, MOLECULES

# The suite, in the order the table in the docstring gives, with the environment each
# molecule exists to probe. `in_range` records whether the molecule's heavy-atom bonding
# lies inside the ghost shells - it is what the result is grouped by.
SUITE = [
    ('water',        'O sp3',             True),
    ('methanol',     'O sp3 + C sp3',     True),
    ('ethane',       'C sp3, C-C 1.51',   True),
    ('ethene',       'C sp2, C=C 1.34',   False),
    ('acetylene',    'C sp,  C#C 1.20',   False),
    ('formaldehyde', 'O sp2, C=O 1.22',   False),
    ('methylamine',  'N sp3, C-N unseen', False),
    ('hcn',          'C sp + N, C#N 1.16', False),
]


def fingerprint(idx):
    """A short, stable hash of one support, so that "the same point set was used in
    every molecule" is a checkable claim rather than a promise about control flow."""
    return hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


def ghost_shells(partners=PARTNERS, scales=SCALES):
    """The bond lengths the ghost ensemble actually trains on, per element pair.

    Printed alongside the suite because the whole design of this experiment is the
    comparison between these distances and the molecules' real ones, and getting it from
    the code rather than the docstring keeps it honest if `SCALES` ever changes.
    """
    out = {}
    for a in sorted(set(list(partners) + ['N'])):
        for b in partners:
            if a <= b:
                out[f"{a}-{b}"] = [(COVALENT[a] + COVALENT[b]) * s for s in scales]
    return out


def heavy_bonds(mol, cutoff=1.8):
    """Bonded pairs and their lengths in Angstrom, for the record in the output."""
    c = mol.atom_coords() * 0.52917721092
    out = []
    for i in range(mol.natm):
        for j in range(i + 1, mol.natm):
            d = float(np.linalg.norm(c[i] - c[j]))
            si, sj = mol.atom_symbol(i), mol.atom_symbol(j)
            if d < cutoff and not (si == 'H' and sj == 'H'):
                out.append(dict(pair=f"{si}-{sj}", r=round(d, 3)))
    return out


def frozen_supports(elements, thresholds, **env_kwargs):
    """
    The entire offline stage: one support per (element, threshold), fitted once.

    Returned as ``{threshold: {element: indices}}`` and then never recomputed. This is
    deliberately hoisted out of the per-molecule loop even though `element_supports` is
    molecule-independent anyway - the point of the experiment is that a single object
    is reused, so it is built once and handed round.
    """
    supports, fps = {}, {}
    for thr in thresholds:
        print(f"  fitting elements at thr={thr:.0e}:", flush=True)
        supports[thr] = element_supports(elements, thr, **env_kwargs)
        fps[thr] = {el: fingerprint(idx) for el, idx in supports[thr].items()}
    return supports, fps


def atom_overlap_targets(mol, screening=1e-8):
    """Each atom's Becke share of the overlap matrix, on its own sub-grid.

    The same target `blocked_fit_per_atom` selects against, returned un-fitted. The
    sub-grid is the element grid translated onto the nucleus - `ghosts.py` relies on
    that and checks it; here it is used to index the frozen support into the molecule.
    """
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    coords_per_atom, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)

    out = []
    for coords_a, weights_a in zip(coords_per_atom, weights_per_atom):
        R_a = _eval_basefuncs(mol, coords_a)
        amp = np.abs(R_a).max(axis=0)
        R_a = R_a[:, np.flatnonzero(amp > screening * amp.max())]
        S_a = (R_a * weights_a[:, np.newaxis]).T @ R_a
        out.append((R_a, S_a))
    return out


def environment_residual(mol, supports):
    """
    How well the frozen support represents each atom's REAL in-molecule overlap block.

    Unconstrained least squares over the support's points only - unconstrained because
    the runtime object this stands in for is the least-squares `Z` fit, not the
    non-negative selection that produced the support. The number reported is the
    relative residual ``||A v - b|| / ||b||``.

    The full parent grid is reported beside it and is always machine zero, which is not
    a bug and is worth stating once: the target is built as
    ``S_a = (R_a w_a)^T R_a``, so it is *by construction* an exact non-negative
    combination of the parent columns with coefficients ``w_a``. The parent therefore
    cannot normalise anything, and the raw residual of the support is the whole of the
    signal. Comparing one element's residual across molecules is apples to apples - it
    is literally the same point set and the same element each time, with only the
    environment changing.

    Caveat, and the reason this sits next to the energies instead of replacing them:
    this is an overlap-metric quantity, and HANDOFF.md section 5 records that `rmsd_S`
    degraded by 125x under rotation and ~2000x under all-ones weights with no loss of
    MP2 accuracy. Treat it as a hypothesis under test, not a screen to trust.
    """
    rows = []
    for ia, (R_a, S_a) in enumerate(atom_overlap_targets(mol)):
        symbol = mol.atom_symbol(ia)
        op = OverlapFitOperator(R_a)
        b = op.pack(S_a)
        nb = float(np.linalg.norm(b))

        def rel_residual(cols):
            A = np.stack([op.column(j) for j in cols], axis=1)
            v, *_ = np.linalg.lstsq(A, b, rcond=None)
            return float(np.linalg.norm(A @ v - b) / nb)

        idx = supports[symbol]
        r_sup = rel_residual(idx)
        r_par = rel_residual(np.arange(R_a.shape[0]))
        rows.append(dict(atom=ia, element=symbol, n_support=int(len(idx)),
                         n_parent=int(R_a.shape[0]), residual=r_sup,
                         residual_parent=r_par))
    return rows


def run_molecule(name, supports, fps, thresholds, blocked_thresholds, out_path,
                 rows, meta):
    """One molecule: the parent-grid floor, the blocked ladder, the frozen ladder."""
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]
    S = mol.intor('int1e_ovlp_sph')

    elements = sorted(set(mol.atom_symbol(ia) for ia in range(mol.natm)))
    print(f"\n== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, elements {elements}, "
          f"RI-MP2 ref {ref:.8f}", flush=True)
    print(f"   bonds: " + ", ".join(f"{b['pair']} {b['r']:.2f}"
                                    for b in heavy_bonds(mol)), flush=True)

    meta['molecules'][name] = dict(
        natm=mol.natm, nao=int(mol.nao_nr()), elements=elements,
        mp2_ri_reference=float(ref), bonds=heavy_bonds(mol))

    def record(mode, threshold, coords, weights, dt, extra=None):
        e = thc_mp2(mol, mf, coords, weights)
        row = dict(molecule=name, mode=mode, threshold=threshold,
                   n_points=len(coords), natm=mol.natm,
                   rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                   e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=dt)
        row.update(extra or {})
        rows.append(row)
        thr = 'parent' if threshold is None else f"{threshold:.0e}"
        print(f"  {mode:8s} thr={thr:>7s}  n={len(coords):5d} "
              f"({len(coords) / mol.natm:5.1f}/atom)  err={row['err_uha']:+9.2f} uHa",
              flush=True)
        with open(out_path, 'w') as fh:
            json.dump(dict(meta=meta, rows=rows), fh, indent=2)

    t0 = time.time()
    coords, weights = BeckeGrid(mol).build()
    record('becke', None, coords, weights, time.time() - t0)

    for thr in blocked_thresholds:
        t0 = time.time()
        per_atom = blocked_fit_per_atom(mol, thr)
        dt = time.time() - t0
        coords = np.vstack([rel + mol.atom_coord(ia)
                            for ia, (rel, _w) in enumerate(per_atom)])
        record('blocked', thr, coords, np.ones(len(coords)), dt)

    for thr in thresholds:
        # No fit here at all: the supports were built before any molecule was seen.
        coords, weights = assemble_from_supports(mol, supports[thr])
        record('ghost', thr, coords, weights, 0.0,
               extra=dict(per_element={k: int(len(supports[thr][k]))
                                       for k in elements},
                          fingerprint={k: fps[thr][k] for k in elements}))

    res = environment_residual(mol, supports[thresholds[len(thresholds) // 2]])
    meta['molecules'][name]['residual'] = res
    print("   residual of frozen support vs real block: " +
          ", ".join(f"{r['element']}{r['atom']} {r['residual']:.2e}" for r in res),
          flush=True)


def run(names, thresholds, blocked_thresholds, out_path, **env_kwargs):
    elements = sorted({sym for name in names
                       for sym in {gto.M(atom=MOLECULES[name](), basis=BASIS,
                                         verbose=0).atom_symbol(ia)
                                   for ia in range(gto.M(atom=MOLECULES[name](),
                                                         basis=BASIS,
                                                         verbose=0).natm)}})
    print(f"== offline stage: fitting {elements} once, for all of {names}", flush=True)
    supports, fps = frozen_supports(elements, thresholds, **env_kwargs)

    meta = dict(experiment='transfer', basis=BASIS, auxbasis=AUXBASIS, ridge=RIDGE,
                weights='ones', elements=elements, molecules={},
                thresholds=thresholds, blocked_thresholds=blocked_thresholds,
                ghost_shells=ghost_shells(env_kwargs.get('partners', PARTNERS),
                                          env_kwargs.get('scales', SCALES)),
                fingerprints={f"{t:.0e}": fps[t] for t in thresholds},
                suite=[dict(molecule=n, probes=p, in_ghost_range=r)
                       for n, p, r in SUITE if n in names],
                **{k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in env_kwargs.items()})

    rows = []
    for name in names:
        run_molecule(name, supports, fps, thresholds, blocked_thresholds,
                     out_path, rows, meta)

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


def saturated_at(pts, floor, target):
    """
    Is this curve's loosest grid already inside `target` of the floor?

    If it is, `points_for` returns that first point unchanged and the cell says only
    "the ladder started past the question" - the molecule needs FEWER points than any
    grid on the ladder, and how many fewer is not measured. Reading a ratio off two
    such cells compares two ladder start points, not two grids, which is exactly the
    artefact HANDOFF.md warns about for water: at 24 AOs every mode reaches the floor,
    so water's ratio is constant across every target and means nothing.

    Cells where either curve is saturated are marked `*` and kept out of the spread
    statistics.
    """
    ys = sorted((n, abs(e - floor)) for n, e in pts if abs(e - floor) > 1e-9)
    return bool(ys) and ys[0][1] <= target


def report(paths):
    """The ghost gap per molecule, and its spread - which is the actual answer."""
    for path in paths:
        d = json.load(open(path))
        meta, rows = d['meta'], d['rows']
        probes = {s['molecule']: s for s in meta['suite']}

        print(f"\n=== {os.path.basename(path)}: partners "
              f"{','.join(meta.get('partners', PARTNERS))}, scales "
              f"{','.join(str(s) for s in meta.get('scales', SCALES))}")
        print("    one frozen support per element, reused across every molecule below")
        for thr in meta['thresholds']:
            fp = meta['fingerprints'][f"{thr:.0e}"]
            print(f"      thr={thr:.0e}  " +
                  "  ".join(f"{k}:{v}" for k, v in sorted(fp.items())))

        names = [r['molecule'] for r in rows]
        names = sorted(set(names), key=names.index)

        print(f"\n  {'molecule':<14s} {'probes':<20s} {'nao':>4s} {'in-rng':>6s}" +
              "".join(f" {f'{t:.0f}uHa':>17s}" for t in TARGETS))
        gaps = {t: {} for t in TARGETS}
        for name in names:
            mrows = [r for r in rows if r['molecule'] == name]
            becke = next((r for r in mrows if r['mode'] == 'becke'), None)
            if becke is None:
                continue
            floor = becke['err_uha']
            cur = {m: sorted((r['n_points'], r['err_uha']) for r in mrows
                             if r['mode'] == m) for m in ('blocked', 'ghost')}
            cells = []
            for t in TARGETS:
                nb = points_for(cur['blocked'], floor, t)
                ng = points_for(cur['ghost'], floor, t)
                if nb and ng:
                    deg = (saturated_at(cur['blocked'], floor, t) or
                           saturated_at(cur['ghost'], floor, t))
                    if not deg:
                        gaps[t][name] = ng / nb
                    cells.append(f" {nb:6.0f}/{ng:5.0f} {ng / nb:4.2f}x"
                                 f"{'*' if deg else ' '}")
                else:
                    cells.append(f" {'n/a':>17s}")
            p = probes.get(name, {})
            print(f"  {name:<14s} {p.get('probes', ''):<20s} "
                  f"{meta['molecules'][name]['nao']:>4d} "
                  f"{str(p.get('in_ghost_range', '')):>6s}" + "".join(cells))
        print("   (blocked points / frozen points, and their ratio - the ghost gap)")
        print("   * the ladder's loosest grid is already inside the target, so the "
              "cell compares\n     two ladder start points rather than two grids - "
              "excluded from the spread below")

        print(f"\n  ghost gap spread across the suite (unsaturated cells only):")
        for t in TARGETS:
            g = gaps[t]
            if len(g) < 2:
                continue
            inr = [v for k, v in g.items() if probes.get(k, {}).get('in_ghost_range')]
            out = [v for k, v in g.items() if not probes.get(k, {}).get('in_ghost_range')]
            lo, hi = min(g.values()), max(g.values())
            extra = ""
            if inr and out:
                extra = (f"   in-range mean {np.mean(inr):.2f}x, "
                         f"out-of-range mean {np.mean(out):.2f}x")
            print(f"    {t:5.0f} uHa: {lo:.2f}-{hi:.2f}x over {len(g)} molecules"
                  f"  (worst: {max(g, key=g.get)}){extra}")

        # The residual screen, grouped by element so that the cross-environment
        # comparison the docstring describes can actually be read off.
        print(f"\n  frozen-support residual on the real in-molecule block "
              f"(relative, least squares over the support's points):")
        by_el = {}
        for name, m in meta['molecules'].items():
            for r in m.get('residual', []):
                by_el.setdefault(r['element'], []).append((name, r))
        for el in sorted(by_el):
            parts = " ".join(f"{n}/{r['element']}{r['atom']}:{r['residual']:.1e}"
                              for n, r in by_el[el])
            print(f"    {el}: {parts}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--out')
    p.add_argument('--report', nargs='*', help='read result files and print the table')
    p.add_argument('--residual', action='store_true',
                   help='no-SCF screen only: frozen supports and their residual on '
                        'each molecule\'s real per-atom blocks')
    p.add_argument('--molecules', default=','.join(n for n, _, _ in SUITE))
    p.add_argument('--thresholds', default='1e-3,3e-4,1e-4,3e-5,1e-5,1e-6')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5')
    p.add_argument('--partners', default=','.join(PARTNERS))
    p.add_argument('--scales', default=','.join(str(s) for s in SCALES))
    p.add_argument('--directions', default='icosahedron')
    p.add_argument('--full-cross', action='store_true')
    a = p.parse_args()

    if a.report is not None:
        report(a.report or ['transfer.json'])
        sys.exit()

    names = [n for n in a.molecules.split(',') if n]
    thresholds = [float(x) for x in a.thresholds.split(',') if x]
    env_kwargs = dict(directions=a.directions,
                      partners=tuple(a.partners.split(',')),
                      scales=tuple(float(s) for s in a.scales.split(',')),
                      full_cross=a.full_cross)

    if a.residual:
        print("ghost shells (Angstrom):")
        for pair, rs in sorted(ghost_shells(env_kwargs['partners'],
                                            env_kwargs['scales']).items()):
            print(f"  {pair:5s} " + "  ".join(f"{r:.2f}" for r in rs))
        elements = sorted({gto.M(atom=MOLECULES[n](), basis=BASIS,
                                 verbose=0).atom_symbol(ia)
                           for n in names
                           for ia in range(gto.M(atom=MOLECULES[n](), basis=BASIS,
                                                 verbose=0).natm)})
        thr = thresholds[len(thresholds) // 2]
        print(f"\nfrozen supports at thr={thr:.0e}:")
        sup = element_supports(elements, thr, **env_kwargs)
        for el, idx in sorted(sup.items()):
            print(f"    {el}: {len(idx):4d} points  fp={fingerprint(idx)}")
        for name in names:
            mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
            res = environment_residual(mol, sup)
            print(f"  {name:<14s} " +
                  " ".join(f"{r['element']}{r['atom']}:{r['residual']:.1e}"
                           for r in res))
        sys.exit()

    if not a.out:
        p.error('--out is required unless --residual or --report is given')
    run(names, thresholds, [float(x) for x in a.blocked_thresholds.split(',') if x],
        a.out, **env_kwargs)

"""
Four levers on the ghost fit, and whether any of them buys anything.

Section 8 found that a per-element NNLS fit cannot retain more points than its target
supplies *equations*, and that a free atom's overlap target supplies only
`n_AO (n_AO + 1) / 2` of them - 15 for hydrogen in cc-pVDZ, which is a hard cap no
threshold can lift. Ghost environments fix that by supplying thousands. But an NNLS
support still stops somewhere, and everything in this directory has been measured at one
point in the space of things that decide where:

* **the AO basis** - cc-pVDZ throughout, `sweep.py:23`, which sets the equation count
  per environment and so the cap section 8 identified;
* **the parent grid** - level 0 throughout, hardcoded in `ghosts.element_grid`, which
  sets the number of *candidate* points the solve chooses between;
* **the ghost direction set** - 12 icosahedral vertices, with 6 octahedral ones tried in
  `ensemble.py`; a tetrahedron, the smallest non-degenerate set and the one that matches
  sp3 bonding, has never been tried;
* **cage or not** - one ghost per environment throughout. `ghost_environments` argues
  against a cage on the grounds that it would squeeze the central atom's Becke share
  down to a lobe around the nucleus. That is an argument, not a measurement.

This script measures all four. It has two stages, and the reason they are separate is
the lesson this directory keeps re-learning: a cheap proxy is not the thing you care
about.

**Stage A (`--ceilings`) is SCF-free and takes seconds per element.** It drives the KKT
threshold to zero and reports where Lawson-Hanson stops - the support *ceiling*, which
is `min(effective rank of the stacked target, n candidate points)`. That is a genuine
structural quantity: section 9 shows an ensemble whose ceiling is below what a molecule
needs cannot reach that molecule's accuracy target at any threshold, which is exactly
how `octa-cycled` fails ethanol. So the ceiling is a **necessary-condition screen** -
run it before spending single points.

**It is not a quality metric, and stage A alone must not be read as one.** The same
section 9 has `octa-cycled` as the *best* ensemble on methanol at matched accuracy while
carrying one of the lowest ceilings. A bigger ceiling means a grid can be made bigger,
not that it is better per point. Stage B is what says whether a lever converts.

**Stage B (a molecule) runs the accuracy ladders**, one curve per lever setting, all
against one SCF, one DF-MP2 reference and one parent-grid floor, so that `analyse.py`
can interpolate them to matched accuracy and the ratios mean something. The in-molecule
`blocked` baseline stays at the committed footing (level 0, `w = 1`, `metric_ridge`)
whatever the transferable grid's parent level is: the question a parent-level row asks
is whether a richer candidate pool makes a better *frozen* support, not whether Becke
grids improve with level, which is not in doubt.

Usage:

    # the screen: all four axes, no SCF, seconds per cell
    uv run python levers.py --ceilings H,C,O --out data/levers_ceilings.json

    # does a ceiling convert into points per uHa?
    uv run python levers.py methanol --variants ghost,parent1,cage,tetra \\
        --out data/levers_methanol.json
    uv run python analyse.py data/levers_methanol.json

    # the basis lever needs its own floor, so it gets its own file
    uv run python levers.py methanol --basis cc-pvtz --variants ghost \\
        --out data/levers_methanol_tz.json

    uv run python levers.py --report data/levers_*.json
"""
import argparse, glob, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.grid import BeckeGrid, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import points_for
from ghosts import (DIRECTIONS, RIDGE, assemble_from_supports, element_grid,
                    element_supports, fit_element, ghost_environments)
from rotate import FixedGrid, blocked_fit_per_atom
from sweep import MOLECULES, BASIS

# The RI auxiliary basis that goes with each orbital basis. The offline object is per
# element AND per basis, so a basis row is a different object, not a tuning of this one.
AUXBASIS_FOR = {'cc-pvdz': 'cc-pvdz-ri', 'cc-pvtz': 'cc-pvtz-ri'}

# One named lever setting per curve. `ghost` is the committed configuration every number
# in sections 8-13 was measured with, and every other row is that one with a single axis
# moved - which is what makes the comparison attributable.
VARIANTS = {
    'ghost':     dict(),
    'parent1':   dict(level=1),
    'parent2':   dict(level=2),
    'cage':      dict(cage=True),
    'tetra':     dict(directions='tetrahedron'),
    'octa':      dict(directions='octahedron'),
    'icosafull': dict(full_cross=True),
}


def becke_share(element, basis=None, level=0, **env_kwargs):
    """
    How much of itself the central atom keeps once the ghosts take their partition bite.

    The per-environment target is the central atom's *Becke share* of that environment's
    overlap matrix, so this is the amplitude the fit is actually looking at, relative to
    the free atom's. `ghost_environments` predicts a cage drives it to a lobe around the
    nucleus; this is that prediction as a number.

    Reported as the mean over ghost environments of `sum(w_central) / sum(w_free)`.
    """
    from pyscf.dft import gen_grid, treutler_prune

    envs = ghost_environments(element, basis=basis, **env_kwargs)
    shares = []
    for spec, bas, _label in envs:
        mol = gto.M(atom=spec, basis=bas, spin=None, verbose=0)
        g = gen_grid.Grids(mol)
        atom_grids = g.gen_atomic_grids(mol, level=level, prune=treutler_prune)
        _coords, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)
        shares.append(float(weights_per_atom[0].sum()))
    return float(np.mean(shares[1:]) / shares[0]) if len(shares) > 1 else 1.0


def n_equations(element, basis=None, level=0, screening=1e-8, **env_kwargs):
    """Unique AO-pair equations the stacked target supplies, summed over environments.

    The *nominal* count. Section 8's free-atom cap is exactly this number, because one
    environment's equations are all independent. A stacked ghost target's are not, which
    is why its ceiling lands well below this - see `ensemble.saturate`.
    """
    from pyscf.dft import gen_grid, treutler_prune

    total = 0
    for spec, bas, _label in ghost_environments(element, basis=basis, **env_kwargs):
        mol = gto.M(atom=spec, basis=bas, spin=None, verbose=0)
        g = gen_grid.Grids(mol)
        atom_grids = g.gen_atomic_grids(mol, level=level, prune=treutler_prune)
        coords_per_atom, _w = g.gen_partition(mol, atom_grids, concat=False)
        R_a = _eval_basefuncs(mol, coords_per_atom[0])
        amp = np.abs(R_a).max(axis=0)
        n_ao = int((amp > screening * amp.max()).sum())
        total += n_ao * (n_ao + 1) // 2
    return total


def ceilings(elements, axes, threshold=1e-12, out_path=None):
    """
    Stage A: where each lever setting's support saturates, with no SCF anywhere.

    `axes` is a list of `(label, kwargs)`. Each is fitted at a threshold low enough that
    Lawson-Hanson stops on orthogonality rather than on the KKT test, so the number
    returned is the ceiling rather than a point on a ladder.
    """
    rows = []
    hdr = (f"  {'element':>7s} {'setting':>12s} {'nAO':>4s} {'envs':>5s} {'equations':>10s} "
           f"{'candidates':>11s} {'ceiling':>8s} {'used':>6s} {'becke_share':>12s} {'s':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    for element in elements:
        for label, kw in axes:
            basis = kw.get('basis') or BASIS
            level = kw.get('level', 0)
            env_kw = {k: v for k, v in kw.items() if k not in ('basis', 'level', 'free')}
            t0 = time.time()
            idx, n_parent, n_env = fit_element(element, threshold, free=kw.get('free', False),
                                               basis=kw.get('basis'), level=level, **env_kw)
            dt = time.time() - t0
            n_eq = (n_equations(element, basis=kw.get('basis'), level=level, **env_kw)
                    if not kw.get('free') else
                    n_equations(element, basis=kw.get('basis'), level=level, **env_kw))
            if kw.get('free'):
                # the free control fits environment 0 alone, so only its equations count
                mol = gto.M(atom=f"{element} 0 0 0", basis=basis, spin=None, verbose=0)
                n_eq = mol.nao * (mol.nao + 1) // 2
            share = (becke_share(element, basis=kw.get('basis'), level=level, **env_kw)
                     if not kw.get('free') else 1.0)
            nao = gto.M(atom=f"{element} 0 0 0", basis=basis, spin=None, verbose=0).nao

            row = dict(element=element, setting=label, basis=basis, level=level,
                       nao=int(nao), n_env=int(n_env), n_equations=int(n_eq),
                       n_candidates=int(n_parent), ceiling=int(len(idx)),
                       used=len(idx) / n_parent, becke_share=share, seconds=dt, **env_kw)
            rows.append(row)
            print(f"  {element:>7s} {label:>12s} {nao:>4d} {n_env:>5d} {n_eq:>10d} "
                  f"{n_parent:>11d} {len(idx):>8d} {100 * len(idx) / n_parent:>5.1f}% "
                  f"{share:>12.3f} {dt:>7.1f}", flush=True)

            if out_path:
                with open(out_path, 'w') as fh:
                    json.dump(dict(meta=dict(experiment='levers_ceilings',
                                             threshold=threshold, basis=BASIS), rows=rows),
                              fh, indent=2)
    return rows


def thc_mp2(mol, mf, coords, weights, auxbasis, ridge=RIDGE):
    """LS-THC MP2 correlation energy on a given point set. Mirrors `ghosts.thc_mp2`,
    with the auxiliary basis exposed because the basis lever moves it too."""
    thc = LS_RI_Becke(mol=mol, auxbasis=auxbasis, mo_coeff=mf.mo_coeff,
                      grid=FixedGrid(coords, weights), metric_ridge=ridge)
    return float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())


def run(name, variants, thresholds, blocked_thresholds, out_path, basis=BASIS):
    """
    Stage B: one accuracy curve per lever setting, all sharing one floor.

    Everything below is `w = 1` at `metric_ridge = RIDGE`, which is the footing sections
    8-10 used, so these curves are readable against those tables. That footing is NOT
    the one a gradient is legal at (section 12); a lever that wins here still has to be
    re-read under `--scheme damped` before it means anything for dynamics.
    """
    auxbasis = AUXBASIS_FOR[basis]
    mol = gto.M(atom=MOLECULES[name](), basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = 0
    mf.conv_tol = 1e-12          # the caveat in FINDINGS: sweep.py and ghosts.py do not
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]
    S = mol.intor('int1e_ovlp_sph')
    elements = sorted(set(mol.atom_symbol(ia) for ia in range(mol.natm)))

    rows = []
    meta = dict(experiment='levers', molecule=name, natm=mol.natm,
                nao=int(mol.nao_nr()), basis=basis, auxbasis=auxbasis,
                mp2_ri_reference=float(ref), ridge=RIDGE, weights='ones',
                elements=elements, variants=variants)
    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs in {basis}, "
          f"RI-MP2 ref {ref:.8f}", flush=True)

    def record(mode, threshold, coords, weights, dt, extra=None):
        e = thc_mp2(mol, mf, coords, weights, auxbasis)
        row = dict(mode=mode, threshold=threshold, n_points=len(coords),
                   rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                   e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=dt)
        row.update(extra or {})
        rows.append(row)
        print(f"  {mode:10s} thr={threshold if threshold is None else f'{threshold:.0e}'}"
              f"  n={len(coords):5d} ({len(coords) / mol.natm:5.1f}/atom)  "
              f"err={row['err_uha']:+9.2f} uHa", flush=True)
        with open(out_path, 'w') as fh:
            json.dump(dict(meta=meta, rows=rows), fh, indent=2)
        return row

    # The floor every curve is read against.
    coords, weights = BeckeGrid(mol).build()
    t0 = time.time()
    record('becke', None, coords, weights, time.time() - t0)

    # The in-molecule lower bound, weights discarded so the only difference from the
    # transferable grids is where the points came from. Level 0 throughout, on purpose.
    for thr in blocked_thresholds:
        t0 = time.time()
        per_atom = blocked_fit_per_atom(mol, thr)
        dt = time.time() - t0
        coords = np.vstack([rel + mol.atom_coord(ia) for ia, (rel, _w) in enumerate(per_atom)])
        record('blocked', thr, coords, np.ones(len(coords)), dt)

    # One curve per lever setting.
    for label in variants:
        kw = dict(VARIANTS[label])
        level = kw.pop('level', 0)
        print(f"  -- {label}: {kw or 'baseline'}, parent level {level}", flush=True)
        for thr in thresholds:
            t0 = time.time()
            supports = element_supports(elements, thr, quiet=True, basis=basis,
                                        level=level, **kw)
            dt = time.time() - t0
            coords, weights = assemble_from_supports(mol, supports, basis, level)
            record(label, thr, coords, weights, dt,
                   extra=dict(per_element={k: int(len(v)) for k, v in supports.items()},
                              level=level, lever=kw))

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


def report(paths):
    """Matched-accuracy point counts per lever setting, against the `ghost` baseline.

    `analyse.py` ratios against the leftmost mode a file holds, which for these files is
    `blocked`. That is the right denominator for "what does the freeze cost"; the
    question here is "what does the lever do", so the baseline is `ghost`.
    """
    for path in paths:
        d = json.load(open(path))
        meta, rows = d.get('meta', {}), d.get('rows', [])
        if meta.get('experiment') == 'levers_ceilings':
            print(f"\n=== {path}: support ceilings (no SCF)")
            base = {r['element']: r for r in rows if r['setting'] == 'ghost'}
            print(f"  {'element':>7s} {'setting':>12s} {'ceiling':>8s} {'vs ghost':>9s} "
                  f"{'equations':>10s} {'candidates':>11s} {'becke_share':>12s}")
            for r in rows:
                b = base.get(r['element'], {}).get('ceiling')
                rel = f"{r['ceiling'] / b:.2f}x" if b else '-'
                print(f"  {r['element']:>7s} {r['setting']:>12s} {r['ceiling']:>8d} "
                      f"{rel:>9s} {r['n_equations']:>10d} {r['n_candidates']:>11d} "
                      f"{r['becke_share']:>12.3f}")
            continue

        becke = next((r for r in rows if r.get('mode') == 'becke'), None)
        if becke is None:
            continue
        floor = becke['err_uha']
        modes = []
        for r in rows:
            if r.get('mode') not in (None, 'becke') and r['mode'] not in modes:
                modes.append(r['mode'])
        print(f"\n=== {path}: {meta.get('molecule')}, {meta.get('nao')} AOs in "
              f"{meta.get('basis')} (floor {floor:+.2f} uHa on {becke['n_points']} pts)")
        print(f"  {'target':>9s}" + "".join(f" {m:>14s}" for m in modes))
        for t in (50.0, 20.0, 10.0, 5.0):
            need = {m: points_for(sorted((r['n_points'], r['err_uha']) for r in rows
                                         if r.get('mode') == m), floor, t)
                    for m in modes}
            base = need.get('ghost')
            cells = []
            for m in modes:
                if need[m] is None:
                    cells.append(f" {'n/a':>14s}")
                elif base and m != 'ghost':
                    cells.append(f" {need[m]:8.0f} ({need[m] / base:4.2f}x)")
                else:
                    cells.append(f" {need[m]:14.0f}")
            print(f"  {t:7.0f}uHa" + "".join(cells))


def build_axes(elements_basis_levels, directions, cage, free):
    """The one-at-a-time sweep: the committed setting, plus each axis moved alone."""
    axes = [('ghost', dict())]
    if free:
        axes.append(('free', dict(free=True)))
    for b in elements_basis_levels['bases']:
        if b != BASIS:
            axes.append((f'basis:{b}', dict(basis=b)))
    for lv in elements_basis_levels['levels']:
        if lv != 0:
            axes.append((f'parent{lv}', dict(level=lv)))
    for d in directions:
        if d != 'icosahedron':
            axes.append((f'dir:{d[:5]}', dict(directions=d)))
    if cage:
        for d in directions:
            axes.append((f'cage:{d[:5]}', dict(directions=d, cage=True)))
    return axes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('molecule', nargs='?', choices=sorted(MOLECULES))
    p.add_argument('--out')
    p.add_argument('--ceilings', help='comma-separated elements: run stage A and exit')
    p.add_argument('--report', nargs='*', help='read stage A or B files back')
    p.add_argument('--bases', default='cc-pvdz,cc-pvtz')
    p.add_argument('--levels', default='0,1')
    p.add_argument('--directions', default=','.join(sorted(DIRECTIONS)))
    p.add_argument('--no-cage', action='store_true')
    p.add_argument('--no-free', action='store_true')
    p.add_argument('--basis', default=BASIS, choices=sorted(AUXBASIS_FOR))
    p.add_argument('--variants', default='ghost,parent1,cage,tetra')
    p.add_argument('--thresholds', default='1e-3,3e-4,1e-4,3e-5,1e-5,1e-6',
                   help='the ghost KKT ladder. A cage or a finer parent re-scales the '
                        'normalised stacked target, so a ladder that lands well on one '
                        'setting can land off the interesting range on another - widen '
                        'it rather than believing a curve that never reaches the target')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5')
    p.add_argument('--saturation-threshold', type=float, default=1e-12)
    a = p.parse_args()

    if a.report is not None:
        report(a.report or sorted(glob.glob('data/levers_*.json')))
    elif a.ceilings:
        axes = build_axes(dict(bases=a.bases.split(','),
                               levels=[int(x) for x in a.levels.split(',')]),
                          a.directions.split(','), not a.no_cage, not a.no_free)
        ceilings(a.ceilings.split(','), axes, a.saturation_threshold, a.out)
    else:
        if not a.molecule or not a.out:
            p.error('a molecule and --out are required unless --ceilings or --report')
        run(a.molecule, a.variants.split(','),
            [float(x) for x in a.thresholds.split(',') if x],
            [float(x) for x in a.blocked_thresholds.split(',') if x],
            a.out, basis=a.basis)

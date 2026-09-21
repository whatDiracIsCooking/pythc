"""
A richer purely-atomic target: fit the free atom's **ERIs** instead of its overlap.

Section 4(3) established that a free-atom fit fails, and *why* it fails. It is not that
an isolated atom has no amplitude where a bond would be - it is an equation count.
Lawson-Hanson can never retain more variables than the problem has equations, and the
overlap target supplies only ``n_AO (n_AO + 1) / 2`` of them: **15 for hydrogen in
cc-pVDZ**, at any threshold. Ghosts were introduced to lift that ceiling by stacking
several environments into one solve, and they work (1.1-1.8x over ``blocked``). But a
ghost ensemble is a fabricated sample of environments, and section 4(4) shows its choice
matters, carries a rank ceiling of its own, and gets *worse* if the variety is raised at
fixed environment count.

The Gaussian-basis literature contains both moves. ANO-RCC averages density matrices
over the neutral atom, the cation, the anion and the atom in a field - environments,
essentially. But cc-pVXZ uses no molecules at all: its correlating functions come from
the *atomic correlation energy*, and the reason that works where atomic HF energy would
not is that correlation energy is a far richer observable of the same free atom. So the
binding constraint need not be molecularity. It can be the information content of the
target.

This script takes that route. The exact identity

    (mu nu | lambda sigma) = integral dr phi_mu(r) phi_nu(r) V_{lambda sigma}(r)

makes the ERI a *linear* functional of the quadrature weights, so the same NNLS solver
fits it unchanged - against ``[n_AO (n_AO + 1) / 2]^2`` equations rather than
``n_AO (n_AO + 1) / 2``. Hydrogen goes from 15 equations to 225 and oxygen from 105 to
11025. If the ceiling was the whole of what broke the free-atom fit, this fixes it
without a training set, without an ensemble to choose, and without any transferability
question to argue about. See :class:`~pythc.decomp.nnls.ERIFitOperator`.

Five point sets per element are compared, all at ``w = 1`` under the same metric ridge
unless a row says otherwise:

* **blocked** - fitted in the molecule, per atom. The lower bound. Not transferable.
* **free** - the free atom's *overlap* fit. The section 4(3) strawman, ceiling-bound.
* **ghost** - the current proposal: overlap, stacked over 13 ghost environments.
* **eri** - the free atom's *ERI* fit. Strictly atomic, no neighbours of any kind.
* **eriw** - the same support carrying the weights that same solve produced. Section
  4(7) localised the whole remaining orientation gap to the weight footing and section
  4(8) asks for per-element weights fitted against an in-molecule objective; this is the
  cheapest thing in that direction, since an ERI target is a much better proxy for what
  LS-THC does with a grid than an overlap target is.

Usage:

    # the equation-count claim, directly: where each target's support saturates. No SCF.
    uv run python experiments/atom_centered_grids/atomic_eri.py --saturate H,C,N,O

    # points per element against threshold, plus where the points sit. No SCF, seconds.
    uv run python experiments/atom_centered_grids/atomic_eri.py --calibrate H,C,O

    # the measurement
    uv run python experiments/atom_centered_grids/atomic_eri.py methanol \
        --out data/eri_methanol.json
    uv run python experiments/atom_centered_grids/analyse.py data/eri_methanol.json
"""
import argparse, functools, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2

from pythc.decomp.nnls import ERIFitOperator, OverlapFitOperator, lawson_hanson
from pythc.grid import BeckeGrid, _eval_basefuncs, _rmsd_overlap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS
from rotate import blocked_fit_per_atom
from ghosts import (COVALENT, RIDGE, assemble_from_supports, element_grid,
                    element_supports as ghost_element_supports, per_atom_sets,
                    rotation_spread, thc_mp2)

# Relative singular-value cutoff for the exact range projection inside ERIFitOperator.
# The directions it removes are ones no weight vector on these points could have reached,
# so this is a size reduction and not an approximation; 1e-10 is comfortably inside the
# gap between the resolved and unresolved directions on every element tested (oxygen:
# 92 of 105 pair directions resolved on its 858-point level-0 grid, with the 93rd four
# decades down).
RANK_TOL = 1e-10


@functools.lru_cache(maxsize=None)
def atom_fit_data(symbol):
    """
    Everything a free-atom fit of one element needs, computed once per element.

    The grid is :func:`ghosts.element_grid` - the *same* level-0 atomic grid every other
    mode in this directory selects from, so a support fitted here is the same kind of
    object (a list of indices into a per-element parent) and the comparison is at matched
    candidate set.

    :return: ``(coords, weights, R, V, eri_quadrature, eri_exact)``. ``V`` is
        ``V_{mu nu}(r_P)``, the electrostatic potential of each AO co-density on the grid
        points, which is what turns the ERI into a linear function of the weights.
    """
    mol = gto.M(atom=f"{symbol} 0.0 0.0 0.0", basis=BASIS, spin=None, verbose=0)
    g = gen_grid.Grids(mol)
    coords, weights = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)[symbol][:2]

    parent = element_grid(symbol)
    if len(coords) != len(parent) or not np.allclose(coords, parent, atol=1e-12):
        raise RuntimeError(f"the {symbol} grid fitted here is not the element grid the "
                           f"support has to index into")

    R = _eval_basefuncs(mol, coords)
    V = mol.intor('int1e_grids', grids=coords)
    # The parent grid's own ERIs: what this candidate set can reproduce at all. Targeting
    # these rather than the exact integrals makes the fit a pure compression problem, the
    # way ghosts.py's overlap target is - the residual is reachable, and what the ladder
    # measures is the target's richness rather than the parent grid's quadrature error.
    eri_quad = np.einsum('p,pm,pn,pls->mnls', weights, R, R, V, optimize=True)
    return coords, weights, R, V, eri_quad, mol.intor('int2e')


@functools.lru_cache(maxsize=None)
def _solve_element(element, threshold, target, rank_tol, max_points, normalise):
    """The solve behind :func:`fit_element`, cached so that ``eri`` and ``eriw`` - which
    differ only in whether the weights are kept - cost one NNLS solve between them."""
    _, _, R, V, eri_quad, eri_exact = atom_fit_data(element)

    op = ERIFitOperator(R, V, rank_tol=rank_tol)
    eri = eri_quad if target == 'quadrature' else eri_exact
    b = op.pack(eri)
    # The unreachable part of the target, kept out of the fit but back in the residual so
    # that the number quoted is against the whole ERI array.
    dropped = op.dropped_norm_sq(eri)

    norm = float(np.linalg.norm(b))
    scale = 1.0 / norm if (normalise and norm > 0.0) else 1.0
    w = lawson_hanson(_ScaledOperator(op, scale), scale * b,
                      weight_threshold=threshold, max_passive=max_points)

    resid = float(np.linalg.norm(b - _apply(op, w)))
    rel = float(np.sqrt(resid ** 2 + dropped) / np.sqrt(norm ** 2 + dropped))

    idx = np.flatnonzero(w)
    idx.flags.writeable = False
    return idx, w[idx], op.shape[0], rel


def fit_element(element, threshold, target='quadrature', rank_tol=RANK_TOL,
                max_points=None, with_weights=False, normalise=True):
    """
    Select one element's transferable point set from its own ERIs, offline.

    No neighbours, no ghosts, no training molecules: the only input is the isolated
    atom. What makes that possible where section 4(3)'s free-atom overlap fit failed is
    the equation count, and nothing else about the setup differs.

    :param target: ``'quadrature'`` fits the ERIs the parent grid itself produces (a
        reachable target, and the analogue of ``ghosts.fit_element``'s ``S_a``);
        ``'exact'`` fits the true integrals, whose residual floor is the parent grid's
        own quadrature error - 1.5e-3 relative on hydrogen, 2.6e-4 on oxygen.
    :param with_weights: Keep the NNLS weights alongside the indices. They cost one float
        per point and differentiate as cleanly as the points do (``dw/dR = 0`` for both),
        and section 4(7) says the weight footing is where the remaining orientation gap
        lives.
    :return: ``(indices, n_parent, n_equations, rel_residual)`` - or
        ``((indices, weights), ...)`` when ``with_weights``.
    """
    idx, weights, n_eq, rel = _solve_element(element, threshold, target, rank_tol,
                                             max_points, normalise)
    return ((idx, weights) if with_weights else idx), len(element_grid(element)), n_eq, rel


class _ScaledOperator:
    """``c A`` for a scalar ``c``, so a fit can be put on a unit-norm target.

    The ghost fits normalise their stacked target (``stack_targets``), so their KKT
    thresholds are relative. Scaling here puts the ERI ladder on the same footing -
    which still does not make the two ladders numerically comparable, since the operators
    differ, but it does make a threshold mean the same *kind* of thing in both.
    """

    def __init__(self, op, scale):
        self.op, self.scale = op, float(scale)

    @property
    def shape(self):
        return self.op.shape

    def column(self, j):
        return self.scale * self.op.column(j)

    def gradient(self, r):
        return self.scale * self.op.gradient(r)


def _apply(op, w):
    """``A w`` for a sparse ``w``, without materialising A."""
    out = np.zeros(op.shape[0])
    for j in np.flatnonzero(w):
        out += w[j] * op.column(j)
    return out


def element_supports(elements, threshold, quiet=False, **kwargs):
    """Fit every element once. The whole offline stage, with no molecule in sight."""
    supports = {}
    for symbol in elements:
        t0 = time.time()
        support, n_parent, n_eq, rel = fit_element(symbol, threshold, **kwargs)
        supports[symbol] = support
        n_kept = len(support[0] if kwargs.get('with_weights') else support)
        if not quiet:
            print(f"    {symbol}: {n_kept:4d} of {n_parent} points from {n_eq} "
                  f"equations, residual {rel:.2e}  ({time.time() - t0:.1f}s)", flush=True)
    return supports


# --------------------------------------------------------------------------------------
# SCF-free probes. Both of these answer the question this script exists for without
# touching a molecule, and both run in seconds to minutes.
# --------------------------------------------------------------------------------------

def radial_profile(symbol, idx):
    """Where a support's points sit, relative to the element's covalent radius.

    The mechanism claim is that the ERI target keeps bonding-region points alive without
    a ghost there to hold them: ``V_{lambda sigma}`` falls off as ``1/r`` where a
    co-density falls off exponentially, so the target is still sensitive at a bond
    distance. That is checkable with no SCF at all - and it is the one diagnostic in this
    directory that looks at the *points* rather than the energy, which sections 4(3) and
    4(4) both warn against reading as a quality metric. It is here as evidence about the
    mechanism, not as a score.
    """
    r = np.linalg.norm(element_grid(symbol)[idx], axis=1) * 0.52917721092   # bohr -> A
    cov = COVALENT[symbol]
    return dict(n=len(idx), r_mean=float(np.mean(r)) if len(r) else 0.0,
                r_max=float(np.max(r)) if len(r) else 0.0,
                frac_beyond_cov=float(np.mean(r > cov)) if len(r) else 0.0,
                frac_beyond_2cov=float(np.mean(r > 2 * cov)) if len(r) else 0.0)


def saturate(elements, targets=('quadrature',), floor=1e-14, max_points=None):
    """
    Where each target's support saturates: the equation-count claim, measured directly.

    Driving the KKT threshold to zero retains every point the fit can use, so the number
    that comes back is the effective rank of the fitting matrix - the ceiling the scheme
    runs into however it is tuned. Section 4(3) measured this for the free-atom overlap
    fit and found it exactly at ``n_AO (n_AO + 1) / 2``; the whole question here is
    whether an ERI target lifts it.

    Both targets saturate - a nonzero residual does not stop Lawson-Hanson, since the
    stationarity condition is that the residual be orthogonal to the columns, not that it
    vanish - but only ``quadrature`` is swept by default. The ``exact`` target's residual
    floor is the parent grid's own ERI quadrature error (1.5e-3 relative on hydrogen,
    2.6e-4 on oxygen), and on hydrogen it saturates at 85 points against ``quadrature``'s
    97, so the two are close enough that the second is not worth an hour per heavy
    element. Pass it explicitly if that needs re-checking.
    """
    print(f"{'element':>8s} {'parent':>7s} {'nAO':>4s} {'ovl eqs':>8s} {'ovl sat':>8s}"
          + "".join(f" {'eri eqs':>8s} {f'eri {t[:4]}':>9s}" for t in targets), flush=True)
    out = []
    for symbol in elements:
        coords, weights, R, V, eri_quad, _ = atom_fit_data(symbol)
        n_ao = R.shape[1]
        n_pair = n_ao * (n_ao + 1) // 2

        op = OverlapFitOperator(R)
        b = op.pack((R * weights[:, None]).T @ R)
        w = lawson_hanson(op, b / np.linalg.norm(b), weight_threshold=floor,
                          max_passive=max_points)
        ovl_sat = int(np.count_nonzero(w))

        row = dict(element=symbol, n_parent=len(coords), n_ao=n_ao,
                   overlap_equations=n_pair, overlap_saturation=ovl_sat,
                   overlap_profile=radial_profile(symbol, np.flatnonzero(w)))
        cells = []
        for t in targets:
            idx, _, n_eq, rel = fit_element(symbol, floor, target=t,
                                            max_points=max_points)
            row[f'eri_{t}'] = dict(equations=n_eq, saturation=int(len(idx)),
                                   rel_residual=rel,
                                   profile=radial_profile(symbol, idx))
            cells.append(f" {n_eq:8d} {len(idx):9d}")
        out.append(row)
        print(f"{symbol:>8s} {len(coords):7d} {n_ao:4d} {n_pair:8d} {ovl_sat:8d}"
              + "".join(cells), flush=True)

    print("\n  where the points sit (Angstrom from the nucleus, against the covalent "
          "radius)", flush=True)
    print(f"  {'element':>8s} {'fit':>12s} {'n':>5s} {'r_mean':>8s} {'r_max':>8s} "
          f"{'>1 rcov':>8s} {'>2 rcov':>8s}", flush=True)
    for row in out:
        entries = [('overlap', row['overlap_profile'])]
        entries += [(f'eri/{t}', row[f'eri_{t}']['profile']) for t in targets]
        for label, p in entries:
            print(f"  {row['element']:>8s} {label:>12s} {p['n']:5d} {p['r_mean']:8.3f} "
                  f"{p['r_max']:8.3f} {100 * p['frac_beyond_cov']:7.0f}% "
                  f"{100 * p['frac_beyond_2cov']:7.0f}%", flush=True)
    return out


def calibrate(elements, thresholds, target='quadrature', with_ghost=False):
    """Points per element against threshold, with no SCF and no molecule in sight.

    The ERI fit's threshold is not on the scale of the overlap fit's or the stacked ghost
    fit's - different operator, different gradient - so a ladder picked by analogy lands
    in the wrong range. This is the cheap way to pick one, and ``--with-ghost`` puts the
    current proposal's counts next to it so the ladders can be aligned before any single
    point is spent.
    """
    head = (f"{'element':>8s} {'parent':>7s}"
            + "".join(f" {f'eri {t:.0e}':>10s}" for t in thresholds))
    print(head, flush=True)
    for symbol in elements:
        cells = []
        for thr in thresholds:
            idx, n_parent, _, rel = fit_element(symbol, thr, target=target)
            cells.append(f" {len(idx):4d}/{rel:.0e}")
        print(f"{symbol:>8s} {len(element_grid(symbol)):7d}" + "".join(cells), flush=True)

    if with_ghost:
        print("\n  the ghost ladder, for alignment (ghosts.py thresholds):", flush=True)
        for thr in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            sup = ghost_element_supports(elements, thr, quiet=True)
            print(f"    ghost thr={thr:.0e}: "
                  + ", ".join(f"{k} {len(v)}" for k, v in sup.items()), flush=True)


# --------------------------------------------------------------------------------------
# The measurement.
# --------------------------------------------------------------------------------------

def run(name, thresholds, out_path, blocked_thresholds, ghost_thresholds,
        target='quadrature', n_rot=0, seed=7, append=False,
        modes=('blocked', 'blockedw', 'ghost', 'eri', 'eriw')):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.kernel()
    ref = DFMP2(mf).kernel()[0]
    S = mol.intor('int1e_ovlp_sph')

    elements = sorted(set(mol.atom_symbol(ia) for ia in range(mol.natm)))
    # --append adds modes to a file an earlier invocation wrote, so that a control
    # thought of after the fact does not cost a re-run of everything around it. The
    # parent-grid row is kept from whichever invocation produced it first: it depends on
    # nothing this script varies.
    rows, have = [], set()
    if append and os.path.exists(out_path):
        with open(out_path) as fh:
            rows = json.load(fh)['rows']
        have = {r.get('mode') for r in rows}
        print(f"   appending to {out_path}, which already holds "
              f"{', '.join(sorted(m for m in have if m))}", flush=True)
    meta = dict(molecule=name, natm=mol.natm, nao=int(mol.nao_nr()), basis=BASIS,
                auxbasis=AUXBASIS, mp2_ri_reference=float(ref), ridge=RIDGE,
                elements=elements, n_rot=n_rot, eri_target=target, rank_tol=RANK_TOL,
                modes=list(modes))

    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, elements {elements}, "
          f"RI-MP2 ref {ref:.8f}", flush=True)
    print(f"   metric_ridge = {RIDGE:g}; every mode but 'eriw' runs at w = 1", flush=True)

    def record(mode, threshold, per_atom, dt, extra=None):
        coords = np.vstack([rel + mol.atom_coord(ia)
                            for ia, (rel, _w) in enumerate(per_atom)])
        weights = np.concatenate([w for _rel, w in per_atom])
        e = thc_mp2(mol, mf, coords, weights)
        row = dict(mode=mode, threshold=threshold, n_points=len(coords),
                   rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                   e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=dt)
        row.update(extra or {})
        rows.append(row)
        print(f"  {mode:8s} thr={threshold:.0e}  n={len(coords):5d} "
              f"({len(coords) / mol.natm:5.1f}/atom)  rmsd={row['rmsd_S']:.2e}  "
              f"err={row['err_uha']:+9.2f} uHa", flush=True)

        if n_rot:
            e_rot = rotation_spread(mol, mf, per_atom, n_rot, seed)
            row['e_rotated'] = e_rot
            row['ptp_uha'] = 1e6 * float(np.ptp(e_rot))
            print(f"    spun about each nucleus ({n_rot} draws): "
                  f"{row['ptp_uha']:.2f} uHa peak-to-peak, "
                  f"mean err {1e6 * (np.mean(e_rot) - ref):+.2f} uHa", flush=True)

        with open(out_path, 'w') as fh:
            json.dump(dict(meta=meta, rows=rows), fh, indent=2)
        return row

    # The accuracy ceiling: the unpruned parent grid, on the same footing as the rest.
    if 'becke' not in have:
        coords, weights = BeckeGrid(mol).build()
        t0 = time.time()
        e = thc_mp2(mol, mf, coords, weights)
        rows.append(dict(mode='becke', threshold=None, n_points=len(coords),
                         rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                         e_corr=e, err_uha=(e - ref) * 1e6,
                         fit_seconds=time.time() - t0))
        print(f"  becke             n={len(coords):5d}  "
              f"err={rows[-1]['err_uha']:+9.2f} uHa", flush=True)

    for mode in ('blocked', 'blockedw'):
        if mode not in modes:
            continue
        for thr in blocked_thresholds:
            t0 = time.time()
            fitted = blocked_fit_per_atom(mol, thr)
            dt = time.time() - t0
            # 'blocked' throws the NNLS weights away, which is the footing every
            # transferable support in sections 4(3) and 4(4) runs on; 'blockedw' keeps
            # them. Both are needed, because 'eriw' keeps its weights too and a ratio
            # read across that change of footing would not be a comparison - section
            # 4(7) measured the footing alone at 450x on the torque.
            record(mode, thr,
                   fitted if mode == 'blockedw'
                   else [(rel, np.ones(len(rel))) for rel, _w in fitted], dt)

    # The two overlap-target transferable grids, for the comparison this exists to make.
    for mode in ('free', 'ghost'):
        if mode not in modes:
            continue
        for thr in ghost_thresholds:
            print(f"  fitting {mode} grids at thr={thr:.0e}:", flush=True)
            t0 = time.time()
            supports = ghost_element_supports(elements, thr, free=(mode == 'free'))
            dt = time.time() - t0
            record(mode, thr, per_atom_sets(mol, supports), dt,
                   extra=dict(per_element={k: int(len(v)) for k, v in supports.items()}))

    # The ERI-target grids: strictly atomic, no neighbours of any kind.
    for mode in ('eri', 'eriw'):
        if mode not in modes:
            continue
        keep_w = mode == 'eriw'
        for thr in thresholds:
            print(f"  fitting {mode} grids at thr={thr:.0e}:", flush=True)
            t0 = time.time()
            supports = element_supports(elements, thr, target=target,
                                        with_weights=keep_w)
            dt = time.time() - t0
            if keep_w:
                per_atom = [(np.asarray(element_grid(mol.atom_symbol(ia))
                                        [supports[mol.atom_symbol(ia)][0]]),
                             np.asarray(supports[mol.atom_symbol(ia)][1]))
                            for ia in range(mol.natm)]
                counts = {k: int(len(v[0])) for k, v in supports.items()}
            else:
                per_atom = per_atom_sets(mol, supports)
                counts = {k: int(len(v)) for k, v in supports.items()}
            record(mode, thr, per_atom, dt, extra=dict(per_element=counts))

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('molecule', nargs='?', choices=sorted(MOLECULES))
    p.add_argument('--out')
    p.add_argument('--saturate', help='comma-separated elements; prints the support '
                                      'ceiling of each target and exits, no SCF')
    p.add_argument('--saturate-out', help='write the --saturate table to this JSON file')
    p.add_argument('--saturate-targets', default='quadrature',
                   help='comma-separated ERI targets to saturate; "exact" saturates '
                        'too but costs an hour per heavy element to say so')
    p.add_argument('--calibrate', help='comma-separated elements; prints points vs '
                                       'threshold and exits, no SCF')
    p.add_argument('--with-ghost', action='store_true',
                   help='--calibrate also prints the ghost ladder, for alignment')
    p.add_argument('--thresholds', default='1e-3,1e-4,1e-5,1e-6,1e-7,1e-8',
                   help='KKT ladder for the ERI fits. NOT on the scale of the overlap '
                        'or stacked-ghost ladders: different operator, different '
                        'gradient. Calibrate first.')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5')
    p.add_argument('--ghost-thresholds', default='1e-3,3e-4,1e-4,3e-5,1e-5')
    p.add_argument('--target', default='quadrature', choices=('quadrature', 'exact'))
    p.add_argument('--modes', default='blocked,blockedw,ghost,eri,eriw')
    p.add_argument('--append', action='store_true',
                   help='add these modes to the rows --out already holds, rather than '
                        'replacing them')
    p.add_argument('--rotate', type=int, default=0, metavar='N',
                   help="also spin each atom's frozen point set by N random rotations "
                        'about its own nucleus and report the spread')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--max-points', type=int, default=None,
                   help='cap on the support size per element, for --saturate')
    a = p.parse_args()

    thresholds = [float(x) for x in a.thresholds.split(',') if x]

    if a.saturate:
        table = saturate(a.saturate.split(','), max_points=a.max_points,
                         targets=tuple(a.saturate_targets.split(',')))
        if a.saturate_out:
            with open(a.saturate_out, 'w') as fh:
                json.dump(dict(meta=dict(basis=BASIS, rank_tol=RANK_TOL, grid_level=0,
                                         floor=1e-14,
                                         targets=a.saturate_targets.split(',')),
                               rows=table), fh, indent=2)
    elif a.calibrate:
        calibrate(a.calibrate.split(','), thresholds, target=a.target,
                  with_ghost=a.with_ghost)
    else:
        if not a.molecule or not a.out:
            p.error('a molecule and --out are required unless --saturate or '
                    '--calibrate is given')
        run(a.molecule, thresholds, a.out,
            [float(x) for x in a.blocked_thresholds.split(',') if x],
            [float(x) for x in a.ghost_thresholds.split(',') if x],
            target=a.target, n_rot=a.rotate, seed=a.seed, append=a.append,
            modes=tuple(a.modes.split(',')))

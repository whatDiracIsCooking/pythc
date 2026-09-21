"""
The ghost gap: how much accuracy does a grid fitted *offline, per element* give up?

This is step (3) of HANDOFF.md, and the one that can still kill the atom-centred
proposal. Everything measured so far has been a lower bound: `NNLSGrid(blocked=True)`
fits each atom separately, but it fits it *in the molecule*, seeing that atom's real
neighbours at their real positions. A transferable grid cannot. It has to be fitted once
per element against a guess at what a neighbour looks like, and then attached to every
atom of that element in every molecule. The gap between those two is what this script
measures.

Three point sets per element are compared, all selected by the same NNLS solver on the
same level-0 atomic grid, and all evaluated with the weights thrown away (`w = 1`, which
section 3 of HANDOFF.md established costs nothing) and the metric ridge-regularised:

* **blocked** - fitted in the molecule, per atom. The lower bound. Not transferable.
* **free** - fitted on the isolated atom, no neighbours at all. The strawman, and the
  thing the design discussion predicted would fail: an isolated atom has no amplitude
  where a bond would be, so NNLS should discard exactly the tail points a bond needs.
* **ghost** - fitted against ghost atoms: neighbour basis functions, no nuclear charge,
  stacked over several directions, distances and partner elements in ONE NNLS solve
  (`pythc.decomp.nnls.StackedOperator`). The proposal.

`free` is what makes the number mean something. Any transferable scheme pays *some*
penalty against `blocked`; the question is whether ghosts recover most of the gap that
fitting in vacuum opens up. If `ghost` lands next to `free`, the ghosts are not doing
their job and the idea needs a different offline target. If it lands next to `blocked`,
the remaining steps of HANDOFF.md section 4 are worth taking.

Because the three modes' KKT thresholds are NOT on the same scale - the stacked fit's
gradient is a sum over environments against a normalised target, see
:func:`~pythc.decomp.nnls.stack_targets` - they must be read at matched accuracy, never
at matched threshold. `analyse.py` does that interpolation.

Usage:

    # what a threshold ladder buys in points, no SCF, seconds: calibrate here first
    uv run python experiments/atom_centered_grids/ghosts.py --calibrate O,C,H

    # the measurement
    uv run python experiments/atom_centered_grids/ghosts.py water --out ghosts_water.json
    uv run python experiments/atom_centered_grids/analyse.py ghosts_water.json
"""
import argparse, functools, json, os, sys, time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial.transform import Rotation

from pythc.decomp.nnls import OverlapFitOperator, StackedOperator, lawson_hanson, stack_targets
from pythc.grid import BeckeGrid, _eval_basefuncs, _rmsd_overlap
from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_becke import LS_RI_Becke

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import MOLECULES, BASIS, AUXBASIS
from rotate import FixedGrid, assemble, blocked_fit_per_atom

# Cordero et al. (2008) covalent radii, Angstrom. Only used to turn a partner element
# into a plausible distance; nothing here is sensitive to the third decimal.
COVALENT = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66}

# Bond-length multipliers for the ghost shells: a short bond, a typical one, and a
# stretched / second-shell contact. Section 4(3) of HANDOFF.md says to stack several
# radii rather than tune one, and this is that.
SCALES = (0.90, 1.05, 1.45)

# Which neighbours an element is fitted against. H and the first-row heavies between
# them cover the environments in the test set; N is left out of the partner list because
# its basis is all but indistinguishable from C's and O's at this level, and each extra
# partner costs environments (and so fit time) linearly.
PARTNERS = ('H', 'C', 'O')

# Ridge strength for every energy in this script. w = 1 grids have no weight dynamic
# range to condition the metric, so HANDOFF.md section 4(1) says to pair them with
# ridge; 1e-8 at the default "trace" scaling is the value it settles on.
RIDGE = 1e-8


def icosahedron_directions():
    """
    The 12 vertices of a regular icosahedron, as unit vectors.

    Ghost directions want to be as close to isotropic as a small set can be: the
    selected support is the union over environments, so whatever the directions miss,
    the point set has no reason to cover. Twelve vertices integrate the sphere exactly
    through degree 5, against degree 3 for the 6 octahedral axes - and unlike the
    octahedron they are not aligned with the Lebedev orbits of the grid being pruned,
    which keeps the fit from preferentially seeing one orbit member.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = []
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            v += [[0.0, s1, s2 * phi], [s1, s2 * phi, 0.0], [s2 * phi, 0.0, s1]]
    v = np.array(v)
    return v / np.linalg.norm(v, axis=1)[:, None]


def octahedron_directions():
    """The 6 cartesian axis directions, for comparison against the icosahedron."""
    return np.vstack([np.eye(3), -np.eye(3)])


def tetrahedron_directions():
    """
    The 4 vertices of a regular tetrahedron.

    The smallest direction set that is not degenerate on the sphere, and the one that
    matches sp3 bonding geometry - which is the intuition for trying it, and which
    section 14 finds is not the property that matters. Integrates the sphere exactly
    through degree 2, against the octahedron's 3 and the icosahedron's 5.
    """
    v = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                  [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    return v / np.linalg.norm(v, axis=1)[:, None]


DIRECTIONS = {'icosahedron': icosahedron_directions,
              'octahedron': octahedron_directions,
              'tetrahedron': tetrahedron_directions}


def ghost_environments(element, directions='icosahedron', partners=PARTNERS,
                       scales=SCALES, full_cross=False, cage=False, basis=None):
    """
    The training environments for one element: an atom of that element at the origin,
    with a single ghost neighbour placed around it.

    One ghost per environment, not a cage. A cage of ghosts in every direction at once
    would squeeze the Becke partition of the central atom down to a small lobe around
    the nucleus - the opposite of the situation being trained for - whereas one
    neighbour at a time is what an atom in a molecule actually sees. Isotropy comes from
    the *union* over environments instead, which is the thing the stacked fit computes.

    By default each direction carries one ``(partner, scale)`` combination, cycled, so
    the environment count is the direction count. ``full_cross`` takes every combination
    instead: better sampled, ``len(partners) * len(scales)`` times more expensive, and a
    way to check that the cycling is not what the answer depends on.

    ``cage=True`` builds the arrangement the paragraph above argues against: every
    direction occupied at once, one environment per ``(partner, scale)``. It is here to
    be measured rather than argued about - see `levers.py` and FINDINGS section 14,
    which confirm the partition collapse the argument predicts (the central atom keeps
    0.2-1.2% of its free-atom Becke weight) and then find that the support ceiling goes
    *up* rather than down, so the argument's premise is right and its conclusion does
    not follow from rank alone.

    :param basis: AO basis for the central atom and for the ghost shells. Defaults to
        the module's ``BASIS``; the offline object is per element *and per basis*, so
        this is one of the levers section 14 measures.
    :return: List of ``(atom_spec, basis_dict, label)`` ready for :func:`gto.M`.
    """
    basis = basis or BASIS
    dirs = DIRECTIONS[directions]()
    combos = [(p, s) for p in partners for s in scales]

    envs = []
    # The free atom itself. Its own overlap block has to be integrated whatever the
    # neighbours are doing, and in the ghost environments it is damped by the partition
    # function, so without this the core is fitted only in a half-shadowed form.
    envs.append((f"{element} 0.0 0.0 0.0", {element: basis}, 'free'))

    if cage:
        # Every direction occupied simultaneously, one environment per (partner, scale).
        # The environment count is then len(partners) * len(scales) whatever the
        # direction set is, so a cage and a single-ghost ensemble are NOT matched in
        # environment count - which section 9 identifies as the thing that supplies
        # rank. Read the two against each other with that in mind.
        for partner, scale in combos:
            r = (COVALENT[element] + COVALENT[partner]) * scale
            ghost = f'ghost-{partner}'
            spec = f"{element} 0.0 0.0 0.0"
            for d in dirs:
                pos = r * d
                spec += f"; {ghost} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
            envs.append((spec, {element: basis, ghost: gto.basis.load(basis, partner)},
                         f"cage-{partner}@{r:.2f}"))
        return envs

    pairs = ([(d, c) for d in dirs for c in combos] if full_cross
             else [(d, combos[i % len(combos)]) for i, d in enumerate(dirs)])

    for d, (partner, scale) in pairs:
        r = (COVALENT[element] + COVALENT[partner]) * scale
        pos = r * d
        ghost = f'ghost-{partner}'
        spec = (f"{element} 0.0 0.0 0.0; "
                f"{ghost} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
        envs.append((spec, {element: basis, ghost: gto.basis.load(basis, partner)},
                     f"{partner}@{r:.2f}"))

    return envs


@functools.lru_cache(maxsize=None)
def element_grid(symbol, basis=None, level=0):
    """The atomic grid of one element, about its own nucleus.

    This is the candidate point set an offline fit selects from, and the object a
    transferable grid is a subset of. It depends on the element, the basis and the
    parent grid level and nothing else, so the same points reappear - to the last bit -
    around every atom of that element in every molecule, which is what makes an index
    list a portable grid. An index list is only portable *against the same
    (basis, level)*, which is why both travel with the support.

    ``level`` is the second lever of FINDINGS section 14: it sets the number of
    candidate columns the NNLS solve chooses between, which section 4(9) of HANDOFF.md
    expected to be non-binding and which section 14 measures as co-binding at level 0.
    """
    mol = gto.M(atom=f"{symbol} 0.0 0.0 0.0", basis=basis or BASIS, spin=None, verbose=0)
    g = gen_grid.Grids(mol)
    grid = g.gen_atomic_grids(mol, level=level, prune=treutler_prune)[symbol][0]
    grid.flags.writeable = False           # it is cached and handed out by reference
    return grid


def fit_element(element, threshold, screening=1e-8, max_points=None, free=False,
                normalise=True, with_weights=False, basis=None, level=0, **env_kwargs):
    """
    Select one element's transferable point set, offline.

    Each environment contributes the central atom's Becke share of the overlap matrix of
    that environment - exactly the per-atom target of ``NNLSGrid._build_blocked``, only
    with ghost neighbours standing in for real ones - and all of them are solved as one
    NNLS problem over the shared atomic grid. The weights are discarded: the return
    value is a support.

    :param free: Fit the isolated atom only, ignoring the ghost environments. This is
        the control, not a configuration worth using.
    :param with_weights: Return the surviving NNLS weights alongside the indices. Section
        3 discarded them because a positive diagonal rescaling of ``X`` is absorbed
        exactly by the ``Z`` fit - but that argument is about the *pseudoinverse*, which
        is scale-equivariant, and section 4(5) forces a ridge instead. A ridge is not
        scale-equivariant, so inside the gradient's ridge window the weights are doing
        conditioning work again and the offline object may need to carry them. They cost
        one float per point to store and nothing at all to differentiate: frozen weights
        have ``dw/dR = 0`` exactly as frozen points do.
    :param basis: AO basis to fit in. The support is an index list into
        ``element_grid(element, basis, level)`` and means nothing against another pair.
    :param level: Becke parent grid level the candidate points come from.
    :return: ``(indices, n_parent, n_env)``, indices into :func:`element_grid` - or
        ``((indices, weights), n_parent, n_env)`` when ``with_weights``.
    """
    envs = ghost_environments(element, basis=basis, **env_kwargs)
    if free:
        envs = envs[:1]

    parent = element_grid(element, basis, level)

    blocks, targets = [], []
    for spec, basis, label in envs:
        mol = gto.M(atom=spec, basis=basis, spin=None, verbose=0)
        g = gen_grid.Grids(mol)
        atom_grids = g.gen_atomic_grids(mol, level=level, prune=treutler_prune)
        coords_per_atom, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)

        coords_a, weights_a = coords_per_atom[0], weights_per_atom[0]
        if len(coords_a) != len(parent) or not np.allclose(coords_a, parent, atol=1e-12):
            raise RuntimeError(
                f"the {element} sub-grid of environment {label!r} is not the element "
                f"grid it has to be a subset of; a selection made here would not "
                f"transfer")

        R_a = _eval_basefuncs(mol, coords_a)
        amp = np.abs(R_a).max(axis=0)
        R_a = R_a[:, np.flatnonzero(amp > screening * amp.max())]
        S_a = (R_a * weights_a[:, np.newaxis]).T @ R_a

        op = OverlapFitOperator(R_a)
        blocks.append(op)
        targets.append(op.pack(S_a))

    b, scales = stack_targets(targets, normalise=normalise)
    w = lawson_hanson(StackedOperator(blocks, scales), b,
                      weight_threshold=threshold, max_passive=max_points)

    idx = np.flatnonzero(w)

    return ((idx, w[idx]) if with_weights else idx), len(parent), len(envs)


def element_supports(elements, threshold, free=False, quiet=False, **kwargs):
    """Fit every element once. This is the whole offline stage of the scheme.

    :param quiet: Suppress the per-element line. For callers that fit many ensembles in
        a loop and print their own table.
    :param kwargs: Passed to :func:`fit_element`; ``with_weights=True`` makes each value
        an ``(indices, weights)`` pair rather than a bare index array.
    """
    supports = {}
    for symbol in elements:
        t0 = time.time()
        support, n_parent, n_env = fit_element(symbol, threshold, free=free, **kwargs)
        supports[symbol] = support
        n_kept = len(support[0] if kwargs.get("with_weights") else support)
        if not quiet:
            print(f"    {symbol}: {n_kept:4d} of {n_parent} points from {n_env} "
                  f"environment{'s' if n_env != 1 else ''}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return supports


def assemble_from_supports(mol, supports, basis=None, level=0):
    """
    Build a molecule's grid by translating each element's point set onto its nuclei.

    This is the runtime half of the scheme, and the reason for all of it: the only
    geometry dependence is the ``+ atom_coord`` - a rigid translation, differentiable,
    with nothing selected, pruned or re-fitted as the nuclei move. Weights are 1 by
    section 3 of HANDOFF.md, so there is no ``dw/dR`` term either.
    """
    coords = []
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        coords.append(element_grid(symbol, basis, level)[supports[symbol]]
                      + mol.atom_coord(ia))
    coords = np.vstack(coords)
    return coords, np.ones(len(coords))


def per_atom_sets(mol, supports, basis=None, level=0):
    """A support dictionary in the per-atom form ``rotate.assemble`` expects."""
    return [(np.asarray(element_grid(mol.atom_symbol(ia), basis, level)
                        [supports[mol.atom_symbol(ia)]]),
             np.ones(len(supports[mol.atom_symbol(ia)])))
            for ia in range(mol.natm)]


def rotation_spread(mol, mf, per_atom, n_rot, seed):
    """
    What the arbitrary orientation of a frozen atomic grid costs, for THIS grid.

    Section 7 found orientation dependence to be a symptom of a rank-limited grid rather
    than a structural defect, and converging away with grid size - but it measured that
    on `blocked` grids, fitted against real neighbours, and flagged the ghost-fitted case
    as the one that might behave differently. A grid trained against ghosts in twelve
    chosen directions has a reason to be anisotropic that a real-neighbour fit does not.
    This is that check: spin each atom's point set about its own nucleus and watch.

    `blocked` goes through here too, and has to: §7 measured it with NNLS weights under
    the truncated pseudoinverse, and everything in §8 runs at `w = 1` under ridge. A
    spread read across that change of footing would not be a comparison.
    """
    rng = np.random.default_rng(seed)
    energies = []
    for _ in range(n_rot):
        rots = Rotation.random(mol.natm, random_state=int(rng.integers(1 << 30)))
        coords, weights = assemble(mol, per_atom, rots)
        energies.append(thc_mp2(mol, mf, coords, weights))
    return energies


def support_agreement(mol, supports, threshold_blocked):
    """
    How much of the in-molecule selection does the offline one contain?

    A cheap, SCF-free diagnostic of the same gap the energies measure: if the ghost fit
    were perfect it would be a superset of what the blocked fit picks for each atom.
    Reported per atom as the fraction of blocked's points that the transferable set
    also holds, and as the size ratio that comes with it.
    """
    per_atom = blocked_fit_per_atom(mol, threshold_blocked)
    report = []
    for ia, (rel, _w) in enumerate(per_atom):
        symbol = mol.atom_symbol(ia)
        parent = element_grid(symbol)
        # Which parent point is each blocked-selected point? They come from the same
        # grid, so this is an exact lookup, not a nearest-neighbour approximation.
        d = np.linalg.norm(rel[:, None, :] - parent[None, :, :], axis=2)
        blocked_idx = set(d.argmin(axis=1).tolist())
        offline_idx = set(supports[symbol].tolist())
        report.append(dict(
            atom=symbol, n_blocked=len(blocked_idx), n_offline=len(offline_idx),
            covered=len(blocked_idx & offline_idx) / max(len(blocked_idx), 1),
            size_ratio=len(offline_idx) / max(len(blocked_idx), 1)))
    return report


def thc_mp2(mol, mf, coords, weights, ridge=RIDGE):
    """LS-THC MP2 correlation energy on a given point set."""
    thc = LS_RI_Becke(mol=mol, auxbasis=AUXBASIS, mo_coeff=mf.mo_coeff,
                      grid=FixedGrid(coords, weights), metric_ridge=ridge)
    return float(LaplaceRMP2(mol, mf, thc.build(mode='ov'), n_laplace=10).kernel())


def run(name, thresholds, out_path, blocked_thresholds, n_rot=0, seed=7, **env_kwargs):
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
                weights='ones', elements=elements, n_rot=n_rot, **env_kwargs)

    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, elements {elements}, "
          f"RI-MP2 ref {ref:.8f}", flush=True)
    print(f"   all grids evaluated with w = 1 and metric_ridge = {RIDGE:g}", flush=True)

    def record(mode, threshold, coords, weights, dt, extra=None, per_atom=None):
        e = thc_mp2(mol, mf, coords, weights)
        row = dict(mode=mode, threshold=threshold, n_points=len(coords),
                   rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                   e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=dt)
        row.update(extra or {})
        rows.append(row)
        print(f"  {mode:8s} thr={threshold:.0e}  n={len(coords):5d} "
              f"({len(coords) / mol.natm:5.1f}/atom)  rmsd={row['rmsd_S']:.2e}  "
              f"err={row['err_uha']:+9.2f} uHa", flush=True)

        if n_rot and per_atom is not None:
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
    coords, weights = BeckeGrid(mol).build()
    t0 = time.time()
    e = thc_mp2(mol, mf, coords, weights, ridge=RIDGE)
    rows.append(dict(mode='becke', threshold=None, n_points=len(coords),
                     rmsd_S=_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S),
                     e_corr=e, err_uha=(e - ref) * 1e6, fit_seconds=time.time() - t0))
    print(f"  becke             n={len(coords):5d}  err={rows[-1]['err_uha']:+9.2f} uHa",
          flush=True)

    # The in-molecule lower bound, with its weights discarded so that the only
    # difference from the transferable grids is where the points came from.
    for thr in blocked_thresholds:
        t0 = time.time()
        per_atom = blocked_fit_per_atom(mol, thr)
        dt = time.time() - t0
        per_atom = [(rel, np.ones(len(rel))) for rel, _w in per_atom]
        coords = np.vstack([rel + mol.atom_coord(ia) for ia, (rel, _) in enumerate(per_atom)])
        record('blocked', thr, coords, np.ones(len(coords)), dt, per_atom=per_atom)

    # The two transferable grids. Both are fitted once per element and then only
    # translated, so the fit cost does not depend on the molecule at all.
    for mode, free in (('free', True), ('ghost', False)):
        for thr in thresholds:
            print(f"  fitting {mode} grids at thr={thr:.0e}:", flush=True)
            t0 = time.time()
            supports = element_supports(elements, thr, free=free, **env_kwargs)
            dt = time.time() - t0
            coords, weights = assemble_from_supports(mol, supports)
            record(mode, thr, coords, weights, dt,
                   extra=dict(per_element={k: int(len(v)) for k, v in supports.items()}),
                   per_atom=per_atom_sets(mol, supports))

            if mode == 'ghost' and blocked_thresholds:
                agreement = support_agreement(mol, supports, blocked_thresholds[0])
                rows[-1]['support_agreement'] = agreement
                print("    support vs blocked @ thr="
                      f"{blocked_thresholds[0]:.0e}: "
                      + ", ".join(f"{a['atom']} {100 * a['covered']:.0f}% "
                                  f"({a['size_ratio']:.2f}x)" for a in agreement),
                      flush=True)

    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, rows=rows), fh, indent=2)
    return rows


def calibrate(elements, thresholds, **env_kwargs):
    """
    Points per element against threshold, with no SCF and no molecule in sight.

    The stacked fit's threshold is not on the scale of the blocked fit's, so a ladder
    picked by analogy lands in the wrong range and wastes single points finding out.
    This is the cheap way to pick one - the same advice orbits.py gives for its own
    second ladder.
    """
    print(f"{'element':>8s}  {'parent':>7s}" +
          "".join(f" {'free':>6s} {f'{t:.0e}':>7s}" for t in thresholds), flush=True)
    for symbol in elements:
        n_parent = len(element_grid(symbol))
        cells = []
        for thr in thresholds:
            idx_free, _, _ = fit_element(symbol, thr, free=True, **env_kwargs)
            idx_ghost, _, _ = fit_element(symbol, thr, free=False, **env_kwargs)
            cells.append(f" {len(idx_free):6d} {len(idx_ghost):7d}")
        print(f"{symbol:>8s}  {n_parent:7d}" + "".join(cells), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('molecule', nargs='?', choices=sorted(MOLECULES))
    p.add_argument('--out')
    p.add_argument('--calibrate', help='comma-separated elements; prints points vs '
                                       'threshold and exits, no SCF')
    p.add_argument('--thresholds', default='1e-3,3e-4,1e-4,3e-5,1e-5,1e-6',
                   help='KKT ladder for the free and ghost fits. NOT on the same scale '
                        'as --blocked-thresholds: the stacked target is normalised')
    p.add_argument('--blocked-thresholds', default='1e-3,1e-4,1e-5')
    p.add_argument('--directions', default='icosahedron', choices=sorted(DIRECTIONS))
    p.add_argument('--partners', default=','.join(PARTNERS))
    p.add_argument('--scales', default=','.join(str(s) for s in SCALES))
    p.add_argument('--rotate', type=int, default=0, metavar='N',
                   help='also spin each atom\'s frozen point set by N random rotations '
                        'about its own nucleus and report the spread (N extra single '
                        'points per grid)')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--full-cross', action='store_true',
                   help='every (direction, partner, distance) rather than one cycled '
                        'combination per direction')
    a = p.parse_args()

    env_kwargs = dict(directions=a.directions,
                      partners=tuple(a.partners.split(',')),
                      scales=tuple(float(s) for s in a.scales.split(',')),
                      full_cross=a.full_cross)
    thresholds = [float(x) for x in a.thresholds.split(',') if x]

    if a.calibrate:
        calibrate(a.calibrate.split(','), thresholds, **env_kwargs)
    else:
        if not a.molecule or not a.out:
            p.error('a molecule and --out are required unless --calibrate is given')
        run(a.molecule, thresholds, a.out,
            [float(x) for x in a.blocked_thresholds.split(',') if x],
            n_rot=a.rotate, seed=a.seed, **env_kwargs)

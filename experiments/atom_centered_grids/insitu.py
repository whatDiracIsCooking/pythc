"""
Per-element weights fitted against real molecules - the mixed-footing control, 4(13).

Section 4(7) localises the entire orientation gap of the transferable scheme to the
*weight footing* rather than to the transfer: on methanol at matched support, `w = 1`
gives 9.02 uHa/rad of net torque and the in-molecule NNLS weights give 0.02, a factor of
450. `ghostw` - the weights the ghost fit itself produced, carried into the molecule -
gives 52.3 and is *worse than discarding them*, which 4(6) reads as those weights
conditioning the ghost metric rather than the in-molecule one.

That left step (8) open: fit a per-element weight set *for* the objective the runtime
actually faces. This script takes the in-molecule route to it. **4(9) got there first and
better** - the free atom's own ERIs are a rich enough target that no molecule and no ghost
is needed, and ``atomic_eri.py``'s `eriw` converges 1481x where this route manages ~3x
over `ghostw`. For a grid, use that.

What keeps this script alive is the one thing 4(9) cannot measure. Every pairing it
compares is self-consistent - a support and the weights the same fit produced - and it
warns in passing not to read an `eriw` number against a `w = 1` one. Breaking the pairing
is natural to build here and nowhere else: `molw` is a weight set fitted for one objective
and laid onto a support selected for another. It loses to `ghostw` **0 times out of 12**,
which turns that warning into a much stronger statement. See the mode table below.

The fit
-------

For element E the training blocks are the real atoms of E in a set of real molecules.
Block ``(M, A)`` presents the support points of E, translated onto atom A of molecule M,
against that atom's Becke share of M's overlap matrix,

    S_A = integral p_A(r) phi_mu(r) phi_nu(r) dr

evaluated on A's own full atomic sub-grid. That is the identical target
``NNLSGrid._build_blocked`` fits and the identical target ``ghosts.fit_element`` fits -
only the environment is a real neighbour at its real position instead of a ghost, and
the variables are tied across every atom of E in every training molecule instead of
being free per atom. One weight vector per element comes out, and because the blocks
are stacked into a single :class:`~pythc.decomp.nnls.StackedOperator` the solve is the
same Lawson-Hanson NNLS the rest of the package uses.

**This does not cost transferability.** The product is still ``element -> (indices,
weights)`` and carries no molecular index: the training molecules are consumed by the
offline fit exactly as a basis set's optimisation molecules are, and nothing about them
survives into the object. At runtime a support is still translated rigidly onto a
nucleus with nothing re-selected or re-fitted, so ``dw/dR = 0`` holds as it does for
`ghostw`, and the analytic gradient of ``pythc.grad`` is untouched. The line this
crosses was already crossed by ``ghosts.py``, whose twelve ghost environments are
neighbours too - fabricated ones. The question here is only whether a *sampled*
environment set beats a fabricated one.

The risk it does carry is overfitting, so the molecules are split: :data:`TRAIN` fits
and :data:`TEST` is never seen by any fit. A gain that does not survive on `TEST` is not
a gain.

Modes
-----

Support and weights are separable, and the point of the table is to separate them::

    mode       support                       weights
    ---------- ----------------------------- -----------------------------
    ghost      ghost fit                     1                 (section 8's proposal)
    ghostw     ghost fit                     ghost fit         (section 4(7))
    molw       ghost fit                     in-molecule       <- the mixed footing
    molfit     in-molecule, per element      in-molecule       (self-consistent pairing)
    molfit1    in-molecule, per element      1                 (is it support or weight?)
    blocked    in-molecule, per atom         in-molecule       (non-transferable bound)
    blocked1   in-molecule, per atom         1

`molw` against `ghost` and `ghostw` is the control - identical support, three weight
footings, only one of which was fitted with those points. `molfit` asks whether the ghost
*support* was ever the limiting factor, and `molfit1` splits its answer into the part the
support bought and the part the weights bought.

**The answer, written up in FINDINGS section 16.** `molw` loses: it beats `ghostw` 0
times out of 12 on held-out molecules, on energy and on rotation spread alike, despite
having much the better overlap residual. `molfit` wins, 8/10 on energy at a median of
~3x. So support and weights are one object - a weight set is worth nothing on a support
selected by a different fit, which is why 4(9)'s 1481x belongs to the ERI *fit* rather
than to weights that could be transplanted - and the weights themselves are worth 2-2.5x
at matched support rather than the 450x 4(7)'s single-rung torque pair implied.

Two things to know before re-running it. Every comparison has to be at **matched point
count**: the modes do not agree on size at a shared threshold, a first draft of section
16 read `molfit` at 401 points against `ghostw` at 344 and got a 26x that the next rung
inverted, and `matched_size_report` exists to stop that. And prefer the energy and spread
ladders to a torque quoted at one rung - `ghostw` on formaldehyde runs 163 uHa/rad at 344
points, 1.0 at 427 and 3.7 at 486, so a single-rung torque ratio means very little in
either direction.

Usage::

    # the fit alone, no SCF, seconds - check the supports and weight sums first
    uv run python experiments/atom_centered_grids/insitu.py --calibrate

    # energies and rotation spreads, train and held-out, over a threshold ladder
    uv run python experiments/atom_centered_grids/insitu.py --out data/insitu.json

    # the same modes as torque, in the harness that produced section 13's table
    uv run python experiments/atom_centered_grids/torque_ladder.py formaldehyde \\
        --modes ghost,ghostw,molw,molfit,molfit1 --ridges "" --pinv --draws 4
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghosts import RIDGE, element_grid, element_supports, thc_mp2
from rotate import assemble, blocked_fit_per_atom
from sweep import MOLECULES, BASIS, AUXBASIS

# The molecules the offline weight fit is allowed to see. Chosen to put every element of
# the test set into the bonding it will meet there - O-H and C-O in methanol, C-C and
# C-H in ethane, a lone water for the free O-H stretch - while staying small enough that
# the whole fit is seconds. They are training data, not part of the product.
TRAIN = ('water', 'methanol', 'ethane')

# Never fitted against. `ethanol` is an interpolation (same bonds, bigger), while
# `formaldehyde` and `propene` both carry a C=O / C=C double bond that is shorter than
# any bond in TRAIN and shorter than the shortest ghost shell, which is the transfer
# stress transfer.py was built to apply.
TEST = ('ethanol', 'formaldehyde', 'propene')

# AO amplitude cut-off for a block, matching ghosts.fit_element and
# NNLSGrid._build_blocked so the three targets differ only in their environment.
SCREENING = 1e-8

# The in-molecule weight fit runs on a *fixed* support, so its KKT threshold is not a
# compactness knob - it only decides how many support points NNLS is allowed to zero on
# the way to the least-squares solution. Tight, so `molw` stays at matched support.
WEIGHT_THRESHOLD = 1e-10

MODES = ("ghost", "ghostw", "molw", "molfit", "molfit1", "blocked", "blocked1")


@functools.lru_cache(maxsize=None)
def build_mol(name, basis=BASIS):
    """One training or test molecule. Cached: the RDKit embedding is not free."""
    return gto.M(atom=MOLECULES[name](), basis=basis, verbose=0)


def atom_blocks(element, names, support=None, screening=SCREENING, basis=BASIS):
    """
    The in-molecule training blocks for one element.

    One block per atom of ``element`` per molecule in ``names``. Each presents the
    element's candidate points - translated onto that atom - against the atom's Becke
    share of its molecule's overlap matrix, integrated on the atom's own full sub-grid.

    The target is built on the *full* atomic sub-grid while the variables are only the
    candidate points, which is the whole content of the fit: the retained points and
    their weights have to reproduce what all 858 of them integrate.

    :param support: Indices into :func:`~ghosts.element_grid` that the weights live on,
        or ``None`` to expose the entire element grid as variables (which turns the same
        solve into a selection, i.e. the `molfit` mode).
    :return: ``(blocks, targets, labels)`` for :func:`~pythc.decomp.nnls.stack_targets`.
    """
    parent = element_grid(element)
    support = None if support is None else np.asarray(support)

    blocks, targets, labels = [], [], []
    for name in names:
        mol = build_mol(name, basis)
        g = gen_grid.Grids(mol)
        atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
        coords_per_atom, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)

        for ia in range(mol.natm):
            if mol.atom_symbol(ia) != element:
                continue

            coords_a, becke_w = coords_per_atom[ia], weights_per_atom[ia]
            # The support is an index list into the element grid, so a block is only
            # meaningful if this atom's sub-grid *is* that grid, translated. It is the
            # same invariant ghosts.fit_element and NNLSGrid._atomic_subgrids check, and
            # a selection or a weight made against a different point set would silently
            # attach to the wrong points at runtime.
            if len(coords_a) != len(parent) or not np.allclose(
                    coords_a - mol.atom_coord(ia), parent, atol=1e-10):
                raise RuntimeError(
                    f"the {element} sub-grid of atom {ia} in {name!r} is not the element "
                    f"grid translated onto its nucleus; weights fitted here would not "
                    f"transfer")

            R_full = _eval_basefuncs(mol, coords_a)
            amp = np.abs(R_full).max(axis=0)
            R_full = R_full[:, np.flatnonzero(amp > screening * amp.max())]
            S_a = (R_full * becke_w[:, np.newaxis]).T @ R_full

            # pack() depends only on the AO count, which the restricted and full
            # collocation matrices share, so the block can pack its own target.
            op = OverlapFitOperator(R_full if support is None else R_full[support])
            blocks.append(op)
            targets.append(op.pack(S_a))
            labels.append(f"{name}:{ia}")

    return blocks, targets, labels


def fit_element_insitu(element, names=TRAIN, support=None,
                       threshold=WEIGHT_THRESHOLD, screening=SCREENING,
                       normalise=True, max_points=None, basis=BASIS):
    """
    One element's per-element weights, fitted against real atoms in real molecules.

    With ``support`` this is step (8): the point set is fixed and only the weights are
    solved for, so the result is comparable to `ghost` and `ghostw` point for point.
    Without it the same solve selects as well as weights, which is the `molfit` ceiling.

    ``normalise`` puts the blocks on an equal footing as relative residuals, matching
    ``ghosts.fit_element``. It rescales both the operator and the target by the same
    factor (see :func:`~pythc.decomp.nnls.stack_targets`), so it re-weights the blocks
    against each other without moving the solution off its physical footing: the weights
    that come out are still quadrature weights for the overlap integral, not numbers on
    an arbitrary scale.

    :return: ``(indices, weights)`` - indices into :func:`~ghosts.element_grid`, and the
        non-zero weights that go with them.
    """
    blocks, targets, labels = atom_blocks(element, names, support, screening, basis)
    if not blocks:
        raise ValueError(f"no atom of element {element!r} in {list(names)}, so there is "
                         f"nothing to fit its weights against")

    b, scales = stack_targets(targets, normalise=normalise)
    w = lawson_hanson(StackedOperator(blocks, scales), b,
                      weight_threshold=threshold, max_passive=max_points)

    keep = np.flatnonzero(w)
    # Variable j is support[j] when a support was given, and grid point j when it was
    # not; either way the caller gets element-grid indices.
    idx = keep if support is None else np.asarray(support)[keep]

    return idx, w[keep]


def insitu_supports(elements, names=TRAIN, supports=None, quiet=False, **kwargs):
    """Fit every element once. The whole offline stage of the in-molecule variant.

    :param supports: Per-element index arrays to hold fixed (the `molw` mode). ``None``
        lets the fit select as well, which is `molfit`.
    """
    out = {}
    for symbol in elements:
        t0 = time.time()
        support = None if supports is None else supports[symbol]
        idx, w = fit_element_insitu(symbol, names, support=support, **kwargs)
        out[symbol] = (idx, w)
        if not quiet:
            held = "" if support is None else f" of {len(support)} held"
            print(f"    {symbol}: {len(idx):4d}{held} points from {len(names)} training "
                  f"molecules, sum(w)={w.sum():8.3f}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return out


def per_atom_from_supports(mol, supports, ones=False):
    """Lay a per-element ``(indices, weights)`` table out per atom, nucleus-relative.

    This is the runtime half: an index list becomes points by lookup into the element
    grid and a rigid translation, with nothing re-selected as the nuclei move.
    """
    out = []
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        idx, w = supports[symbol]
        rel = np.asarray(element_grid(symbol)[np.asarray(idx)])
        out.append((rel, np.ones(len(rel)) if ones else np.asarray(w, dtype=float)))
    return out


def build_supports(elements, mode, threshold, train=TRAIN, cache=None, quiet=True):
    """
    The per-element ``(indices, weights)`` table for one transferable mode.

    Memoised on ``(mode, element, threshold)`` because every mode but `ghost` needs the
    ghost fit as its starting point and the evaluation loop asks for the same elements
    once per molecule.
    """
    cache = {} if cache is None else cache
    elements = sorted(elements)

    def ghost(els):
        missing = [e for e in els if ("ghost", e, threshold) not in cache]
        if missing:
            for e, sup in element_supports(missing, threshold, quiet=quiet,
                                           with_weights=True).items():
                cache[("ghost", e, threshold)] = sup
        return {e: cache[("ghost", e, threshold)] for e in els}

    if mode in ("ghost", "ghostw"):
        return ghost(elements)

    # molfit and molfit1 are the same fit read on two weight footings, so they share a
    # cache entry; only per_atom_for_mode's `ones` differs.
    key = "molfit" if mode in ("molfit", "molfit1") else mode

    missing = [e for e in elements if (key, e, threshold) not in cache]
    if missing:
        if key == "molw":
            # fit_element_insitu wants bare indices; the ghost fit hands back (idx, w).
            held = {e: sup[0] for e, sup in ghost(missing).items()}
            fitted = insitu_supports(missing, train, supports=held, quiet=quiet)
        elif key == "molfit":
            # No support held: the same stacked solve selects the points too, and the
            # threshold is a compactness knob again rather than a numerical floor.
            fitted = insitu_supports(missing, train, supports=None, quiet=quiet,
                                     threshold=threshold)
        else:
            raise ValueError(f"unknown transferable mode {mode!r}")
        cache.update({(key, e, threshold): s for e, s in fitted.items()})

    return {e: cache[(key, e, threshold)] for e in elements}


def per_atom_for_mode(mol, mode, threshold, train=TRAIN, cache=None):
    """The frozen per-atom point sets of one mode on one molecule.

    The weight footing is part of the mode. Section 4(5) forces a ridge, and a ridge is
    not scale-equivariant, so `ghost` and `ghostw` are genuinely different grids on the
    same points - which is exactly what makes the weight footing measurable at all.
    """
    if mode in ("blocked", "blocked1"):
        fitted = blocked_fit_per_atom(mol, threshold)
        if mode == "blocked":
            return fitted
        return [(rel, np.ones(len(rel))) for rel, _w in fitted]

    elements = {mol.atom_symbol(ia) for ia in range(mol.natm)}
    supports = build_supports(elements, mode, threshold, train, cache)

    return per_atom_from_supports(mol, supports, ones=mode in ("ghost", "molfit1"))


def stacked_residual(element, names, idx, w, screening=SCREENING, basis=BASIS):
    """
    Relative residual of a weight set against the in-molecule overlap objective.

    The quantity the fit minimises, reported per element without going near an SCF:
    ``||A w - b|| / ||b||`` over the normalised stack of every atom of ``element`` in
    ``names``. It is what separates a weight set that is merely different from one that
    is better, and - evaluated on :data:`TEST` - the cheapest overfitting check there is.
    """
    blocks, targets, _ = atom_blocks(element, names, idx, screening, basis)
    b, scales = stack_targets(targets, normalise=True)
    op = StackedOperator(blocks, scales)

    w = np.asarray(w, dtype=float)
    pred = np.zeros_like(b)
    for j in range(op.shape[1]):
        if w[j]:
            pred += w[j] * op.column(j)

    return float(np.linalg.norm(pred - b) / np.linalg.norm(b))


def rotation_spread(mol, mf, per_atom, n_rot, seed, ridge=RIDGE):
    """What the arbitrary orientation of a frozen atomic grid costs this grid.

    Each atom's point set is spun about its own nucleus by an independent random
    rotation. A transferable grid has no preferred orientation to attach at, so the
    spread over draws is the error bar the scheme carries whatever else it gets right.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rot):
        rots = Rotation.random(mol.natm, random_state=int(rng.integers(1 << 30)))
        coords, weights = assemble(mol, per_atom, rots)
        out.append(thc_mp2(mol, mf, coords, weights, ridge=ridge))
    return out


def evaluate(name, modes, thresholds, train, n_rot, seed, ridge, cache):
    """Every mode at every rung on one molecule: points, overlap residual, energy, spread.

    A ladder rather than one threshold because the modes do not agree on point count at
    a shared threshold - NNLS zeroes part of a held support wherever the non-negativity
    constraint binds, so `molw` is always a little smaller than the `ghost` support it
    started from - and section 4(3)'s rule is that these are read at matched accuracy or
    matched size, never at matched threshold.
    """
    mol = build_mol(name)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()
    ref = float(DFMP2(mf).kernel()[0])
    S = mol.intor('int1e_ovlp_sph')

    held = name not in train
    print(f"\n== {name} ({'HELD OUT' if held else 'in training set'}): {mol.natm} atoms, "
          f"{mol.nao_nr()} AOs, DF-MP2 ref {ref:.8f}", flush=True)
    print(f"   {'mode':<9s} {'thr':>8s} {'points':>7s} {'rmsd S':>10s} {'err/uHa':>10s} "
          f"{'spread/uHa':>11s} {'sum(w)':>10s}", flush=True)

    # The accuracy floor: the unpruned parent grid on the same footing as everything
    # else. No pruned grid can beat the grid it was pruned from, so an error is only
    # meaningful as a distance from this row - the ladders below run 20 to 600 uHa
    # against a floor of a few, so it does not reorder them, but a rung that lands
    # inside it (as `blocked` does on formaldehyde) has converged rather than won.
    # Note this file holds one such row per molecule, so it is not in the single-molecule
    # shape analyse.py expects; matched_size_report does the interpolation instead.
    coords, weights = BeckeGrid(mol).build()
    e = thc_mp2(mol, mf, coords, weights, ridge=ridge)
    rows = [dict(molecule=name, held_out=held, mode='becke', threshold=None,
                 n_points=len(coords), n_atoms=mol.natm, n_ao=int(mol.nao_nr()),
                 rmsd_S=float(_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S)),
                 e_corr=e, err_uha=1e6 * (e - ref), sum_weights=float(weights.sum()),
                 mp2_ri_reference=ref, seconds=0.0)]
    print(f"   {'becke':<9s} {'-':>8s} {rows[0]['n_points']:7d} "
          f"{rows[0]['rmsd_S']:10.2e} {rows[0]['err_uha']:+10.2f}", flush=True)

    for mode in modes:
        for threshold in thresholds:
            t0 = time.time()
            per_atom = per_atom_for_mode(mol, mode, threshold, train, cache)
            coords, weights = assemble(mol, per_atom)
            e = thc_mp2(mol, mf, coords, weights, ridge=ridge)

            row = dict(molecule=name, held_out=held, mode=mode, threshold=threshold,
                       n_points=len(coords), n_atoms=mol.natm, n_ao=int(mol.nao_nr()),
                       rmsd_S=float(_rmsd_overlap(_eval_basefuncs(mol, coords), weights, S)),
                       e_corr=e, err_uha=1e6 * (e - ref), sum_weights=float(weights.sum()),
                       mp2_ri_reference=ref, seconds=time.time() - t0)

            if n_rot:
                spun = rotation_spread(mol, mf, per_atom, n_rot, seed, ridge)
                row.update(draw_energies=spun, spread_uha=1e6 * float(np.ptp(spun)),
                           mean_rot_err_uha=1e6 * float(np.mean(spun) - ref))

            rows.append(row)
            print(f"   {mode:<9s} {threshold:8.0e} {row['n_points']:7d} "
                  f"{row['rmsd_S']:10.2e} {row['err_uha']:+10.2f} "
                  f"{row.get('spread_uha', np.nan):11.2f} {row['sum_weights']:10.3f}",
                  flush=True)

    return rows


def residual_table(modes, thresholds, train, test, cache):
    """Per-element in-molecule overlap residuals, on the training set and held out.

    SCF-free, and the honest overfitting check: `molw` minimises the TRAIN column by
    construction, so only the TEST column says whether it learned the objective or the
    training molecules. Held-out molecules contribute a block per atom of the element
    exactly as training ones do, so the two columns are the same quantity on different
    samples and can be read against each other directly.
    """
    def elements_of(names):
        return {build_mol(n).atom_symbol(ia) for n in names
                for ia in range(build_mol(n).natm)}

    elements = sorted(elements_of(tuple(train) + tuple(test)))

    print(f"\n== in-molecule overlap residual ||Aw - b|| / ||b||, per element", flush=True)
    print(f"   {'mode':<9s} {'thr':>8s} {'element':>8s} {'points':>7s} {'train':>12s} "
          f"{'held out':>12s}", flush=True)

    rows = []
    for mode in modes:
        if mode in ("blocked", "blocked1"):
            continue
        for threshold in thresholds:
            for element in elements:
                present = [n for n in test if element in elements_of([n])]
                idx, w = build_supports([element], mode, threshold, train, cache)[element]
                if mode in ("ghost", "molfit1"):
                    w = np.ones(len(idx))

                r_train = stacked_residual(element, train, idx, w)
                r_test = stacked_residual(element, present, idx, w) if present else np.nan
                rows.append(dict(mode=mode, element=element, threshold=threshold,
                                 residual_train=r_train, residual_test=r_test,
                                 n_points=len(idx)))
                print(f"   {mode:<9s} {threshold:8.0e} {element:>8s} {len(idx):7d} "
                      f"{r_train:12.4e} {r_test:12.4e}", flush=True)
    return rows


def matched_size_report(rows, modes, key="err_uha"):
    """Each mode's ``key`` interpolated onto a common point count, per molecule.

    The modes do not land on the same grid size at the same threshold, and a weight
    footing read off two different-sized grids is not a comparison. This interpolates
    each mode's ladder (log point count against the quantity) onto the point counts the
    *narrowest* mode covers, which is the widest window every mode can be quoted in.
    """
    molecules, out = [], []
    for r in rows:
        if r["molecule"] not in molecules:
            molecules.append(r["molecule"])

    for name in molecules:
        curves = {}
        for mode in modes:
            pts = sorted((r["n_points"], r.get(key)) for r in rows
                         if r["molecule"] == name and r["mode"] == mode
                         and r.get("threshold") is not None and r.get(key) is not None)
            if len(pts) >= 2:
                curves[mode] = (np.array([p[0] for p in pts], float),
                                np.array([p[1] for p in pts], float))
        if len(curves) < 2:
            continue

        lo = max(c[0][0] for c in curves.values())
        hi = min(c[0][-1] for c in curves.values())
        if not hi > lo:
            continue
        targets = np.unique(np.round(np.geomspace(lo, hi, 3)).astype(int))

        held = next(r["held_out"] for r in rows if r["molecule"] == name)
        print(f"\n   {name}{' (held out)' if held else ''}: {key} at matched point count",
              flush=True)
        print(f"   {'points':>8s} " + " ".join(f"{m:>10s}" for m in curves), flush=True)
        for t in targets:
            vals = {m: float(np.interp(np.log(t), np.log(x), y))
                    for m, (x, y) in curves.items()}
            out.append(dict(molecule=name, held_out=held, key=key, n_points=int(t), **vals))
            print(f"   {t:8d} " + " ".join(f"{vals[m]:10.2f}" for m in curves), flush=True)

    return out


def main(modes, thresholds, train, test, n_rot, seed, ridge, out_path):
    print(f"== in-molecule per-element weights, metric_ridge {ridge:g}, "
          f"thresholds {', '.join(f'{t:.0e}' for t in thresholds)}")
    print(f"   train {', '.join(train)}   |   held out {', '.join(test)}")
    print(f"   modes {', '.join(modes)}; {n_rot} random per-atom orientations per grid",
          flush=True)

    cache = {}
    rows, meta = [], dict(thresholds=list(thresholds), train=list(train),
                          test=list(test), modes=list(modes), n_rot=n_rot, seed=seed,
                          ridge=ridge, basis=BASIS, auxbasis=AUXBASIS,
                          screening=SCREENING, weight_threshold=WEIGHT_THRESHOLD)

    def dump(extra=None):
        if out_path:
            with open(out_path, 'w') as fh:
                json.dump(dict(meta=meta, rows=rows, residuals=residuals,
                               matched=extra or []), fh, indent=2)

    residuals = residual_table(modes, thresholds, train, test, cache)
    dump()

    for name in tuple(train) + tuple(test):
        rows.extend(evaluate(name, modes, thresholds, train, n_rot, seed, ridge, cache))
        dump()

    print(f"\n== matched-size comparison", flush=True)
    matched = matched_size_report(rows, modes, "err_uha")
    if n_rot:
        matched += matched_size_report(rows, modes, "spread_uha")
    dump(matched)

    if out_path:
        print(f"\nwrote {out_path}", flush=True)

    return dict(meta=meta, rows=rows, residuals=residuals, matched=matched)


def calibrate(elements, thresholds, train):
    """Supports and weight sums per element per threshold. No SCF, seconds."""
    print(f"   {'thr':>8s} {'element':>8s} {'ghost n':>8s} {'molw n':>8s} "
          f"{'molfit n':>9s} {'sum w ghost':>12s} {'sum w molw':>12s} "
          f"{'sum w molfit':>13s}", flush=True)
    cache = {}
    for thr in thresholds:
        for e in elements:
            g = build_supports([e], "ghost", thr, train, cache)[e]
            m = build_supports([e], "molw", thr, train, cache)[e]
            f = build_supports([e], "molfit", thr, train, cache)[e]
            print(f"   {thr:8.0e} {e:>8s} {len(g[0]):8d} {len(m[0]):8d} {len(f[0]):9d} "
                  f"{g[1].sum():12.3f} {m[1].sum():12.3f} {f[1].sum():13.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"comma separated, from {', '.join(MODES)}")
    p.add_argument("--threshold", default="1e-3,3e-4,1e-4",
                   help="comma separated KKT thresholds of the ghost / molfit "
                        "selection; the ladder the modes are compared along")
    p.add_argument("--train", default=",".join(TRAIN))
    p.add_argument("--test", default=",".join(TEST))
    p.add_argument("--n-rot", type=int, default=4,
                   help="random per-atom orientations; 0 for the as-fitted row only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ridge", type=float, default=RIDGE)
    p.add_argument("--calibrate", action="store_true",
                   help="print supports and weight sums per element, then stop")
    p.add_argument("--elements", default="H,C,O", help="only used by --calibrate")
    p.add_argument("--thresholds", default="1e-3,3e-4,1e-4",
                   help="only used by --calibrate")
    p.add_argument("--out")
    a = p.parse_args()

    train = tuple(n.strip() for n in a.train.split(",") if n.strip())
    test = tuple(n.strip() for n in a.test.split(",") if n.strip())

    if a.calibrate:
        calibrate([e.strip() for e in a.elements.split(",") if e.strip()],
                  [float(t) for t in a.thresholds.split(",") if t.strip()], train)
        raise SystemExit(0)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if any(m not in MODES for m in modes):
        raise SystemExit(f"modes must come from {', '.join(MODES)}")
    overlap = set(train) & set(test)
    if overlap:
        raise SystemExit(f"{', '.join(sorted(overlap))} is in both train and test, so "
                         f"the held-out column would not be held out")

    thresholds = sorted((float(t) for t in a.threshold.split(",") if t.strip()),
                        reverse=True)
    main(modes, thresholds, train, test, a.n_rot, a.seed, a.ridge, a.out)

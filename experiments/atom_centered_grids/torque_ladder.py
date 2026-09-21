"""
Does the torque converge away with grid size, the way the energy spread did?

This is the open question section 4(5) left behind, and HANDOFF.md section 5 calls it the
cheapest useful thing remaining. Section 4(2) measured orientation dependence as an energy
*spread* over random per-atom rotations, watched methanol go 218 -> 7.1 -> 0.01 uHa from
175 to 670 points, and concluded that orientation dependence is a symptom of a rank-limited
grid rather than a structural defect of rigid attachment. Section 4(5) then computed
``dE/dtheta`` exactly and found that the spread does not track it: water's ghost grid has
1.9x the spread of its blocked grid and **30x** the net torque. That left the deflation of
section 4(2) resting on a statistic that had just been shown not to measure the thing.

The net torque is the quantity with physical consequences. Rotating the molecule while the
point sets keep their lab orientation is the same as rotating everything - which the energy
is invariant under - and then counter-rotating each grid about its nucleus, so
``sum_A tau_A`` is exactly the energy's response to a rigid rotation under lab-fixed
attachment, i.e. the rate at which angular momentum leaks in dynamics. If it converges away
with grid size, lab-fixed attachment is fine and section 4(2)'s conclusion survives in the
form that matters. If it does not, the frame question is live again.

So: run the section 4(2) ladder in ``dE/dtheta`` instead of in spreads. Each grid is
evaluated as fitted and at several random per-atom orientations, and both statistics are
reported side by side on the *same* grids and the *same* draws, so the comparison between
them is internal.

Three things are controlled for, because each could fake a convergence:

* **The ridge.** Larger grids have more near-null metric directions, and section 4(5)
  showed ridge inverts those at ``1/lambda``. Every row is run at two ridges inside the
  gradient window, and a trend that only appears at one of them is a ridge artifact.
* **A lucky draw.** The as-fitted net torque is one orientation of one grid and its
  magnitude could be accidentally small. The rms over random draws is the robust statistic
  and is what the verdict should be read from.
* **The asymptote.** ``--parent`` adds the unpruned atomic grid as the top of the ladder.
  A complete atomic grid is very nearly rotationally invariant, so its torque is the floor
  the ladder should be heading for. If it is not small, the convergence is not real.

No finite differences: section 4(5) already checked this torque against one to six
significant figures, and ``gradient.py`` re-runs that check. This script only needs the
analytic value, so a row costs one gradient evaluation per draw.

Usage:

    uv run python experiments/atom_centered_grids/torque_ladder.py water --parent
    uv run python experiments/atom_centered_grids/torque_ladder.py methanol \
        --modes blocked,ghost --draws 4 --out torque_ladder_methanol.json
"""
import argparse
import json
import os
import sys
import time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.dft import gen_grid, treutler_prune
from pyscf.mp.dfmp2 import DFMP2
from scipy.spatial.transform import Rotation

from pythc.grad import FrozenGrid, thc_mp2_gradient
from pythc.grad.factorisation import ThcFactorisation
from pythc.lib import ridge_shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghosts import element_grid, element_supports, per_atom_sets
from insitu import TRAIN as INSITU_TRAIN, per_atom_for_mode
from rotate import blocked_fit_per_atom
from sweep import MOLECULES, BASIS, AUXBASIS

# The section 4(2) ladder, extended at both ends. blocked thresholds are read off
# ghosts_methanol.json (301 / 491 / 670 points) with a coarser rung added below; ghost
# thresholds are that file's own ladder.
DEFAULT_THRESHOLDS = {
    "blocked": (3e-3, 1e-3, 3e-4, 1e-4, 1e-5),
    "blocked1": (3e-3, 1e-3, 3e-4, 1e-4, 1e-5),
    "ghost": (1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    "ghostw": (1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    # The in-molecule weight modes of insitu.py (HANDOFF step 8). molw holds the ghost
    # support of the same rung, so its ladder has to be the ghost ladder; molfit selects
    # for itself and carries the same knob.
    "molw": (1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    "molfit": (1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    "molfit1": (1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
}

MODES = ("blocked", "blocked1", "ghost", "ghostw", "molw", "molfit", "molfit1",
         "parent", "parent1")

# Modes whose supports or weights come from insitu.py rather than from a ghost fit.
INSITU_MODES = ("molw", "molfit", "molfit1")


def parent_per_atom(mol, ones):
    """Every point of every atomic sub-grid, unpruned.

    The top of the ladder. A complete atomic grid has no selected support to be
    anisotropic, so whatever torque survives here is the floor set by the quadrature's
    band limit and the metric's conditioning rather than by the fit. ``ones`` picks the
    footing: the Becke partition weights (the genuine complete quadrature, and what
    ``ghosts.py``'s parent row uses) or ``w = 1`` (the footing every frozen support in
    this script runs on).
    """
    g = gen_grid.Grids(mol)
    atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
    coords_per_atom, weights_per_atom = g.gen_partition(mol, atom_grids, concat=False)
    return [(coords_a - mol.atom_coord(ia),
             np.ones(len(coords_a)) if ones else weights_a)
            for ia, (coords_a, weights_a)
            in enumerate(zip(coords_per_atom, weights_per_atom))]


def build_per_atom(mol, mode, threshold, support_cache):
    """The frozen per-atom point sets for one rung of one ladder.

    The weight footing is part of the mode, and has to be, because section 4(5) forces a
    ridge three decades larger than the one section 4(3) evaluated its ``w = 1`` grids
    at. Section 3's result - that a positive diagonal rescaling of ``X`` is absorbed
    exactly by the ``Z`` fit - holds for the *pseudoinverse*, which is scale-equivariant.
    A ridge is not: ``(D^2 S D^2 + lam I)^-1`` is not ``D^-2 (S + lam I)^-1 D^-2``. So
    inside the gradient's ridge window the weights stop being free, and ``blocked``
    (NNLS weights, the footing ``gradient.py`` and sections 1 and 7 use) and ``blocked1``
    (``w = 1``, the footing sections 8 to 10 use, and the only one a ghost support can
    have) are genuinely different grids. Both are run so the ghost comparison is at
    matched footing and the footing's own cost is visible.
    """
    if mode in ("parent", "parent1"):
        return parent_per_atom(mol, ones=mode.endswith("1"))
    if mode in ("blocked", "blocked1"):
        fitted = blocked_fit_per_atom(mol, threshold)
        if mode == "blocked":
            return fitted
        return [(rel, np.ones(len(rel))) for rel, _ in fitted]

    if mode in INSITU_MODES:
        # Section 4(7) left one square of the weight-footing table empty: a per-element
        # weight set fitted for the objective the runtime actually faces. insitu.py
        # fills it, and it is measured here so the torque is quoted in the harness that
        # produced section 13's table rather than a parallel one.
        return per_atom_for_mode(mol, mode, threshold, INSITU_TRAIN,
                                 support_cache.setdefault("_insitu", {}))

    keep_w = mode == "ghostw"
    elements = sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})
    key = [(e, threshold, keep_w) for e in elements]
    missing = [e for e, k in zip(elements, key) if k not in support_cache]
    if missing:
        for e, support in element_supports(missing, threshold, quiet=True,
                                           with_weights=keep_w).items():
            support_cache[(e, threshold, keep_w)] = support

    if not keep_w:
        return per_atom_sets(mol, {e: support_cache[k] for e, k in zip(elements, key)})

    # The same support, carrying the weights the same NNLS solve produced. Frozen
    # weights are as differentiable as frozen points, so this costs the scheme nothing
    # structurally; whether the weights a *ghost* fit produces are any use once the
    # support is transferred into a real molecule is what the row measures.
    return [(np.asarray(element_grid(mol.atom_symbol(ia))
                        [support_cache[(mol.atom_symbol(ia), threshold, True)][0]]),
             np.asarray(support_cache[(mol.atom_symbol(ia), threshold, True)][1]))
            for ia in range(mol.natm)]


def metric_diagnostics(mol, mf, per_atom, ridge, n_laplace):
    """How conditioned the LS-THC metric is, and how much of it the ridge is carrying.

    Section 3 found the NNLS weights nearly irrelevant to the *energy* and concluded the
    offline object is a bare support. Section 4(1) added that the weights had been doing
    conditioning work, so ``w = 1`` must be paired with a ridge. Put together with section
    4(5)'s ridge window, that predicts the torque should track the metric's conditioning
    rather than the point count - which is what these columns are for. ``below_ridge`` is
    the number of metric directions the ridge, not the metric, is inverting.
    """
    coords, weights = FrozenGrid(mol, per_atom).build()
    fac = ThcFactorisation(mol, coords, weights, np.asarray(mf.mo_coeff),
                           mol.nelectron // 2, AUXBASIS, metric_ridge=ridge,
                           aux_ridge=ridge).build()

    eig = np.linalg.eigvalsh(0.5 * (fac.S + fac.S.T))
    top = float(eig[-1])
    # ``ridge is None`` is the pinv control: nothing is added to the metric, and the
    # directions it drops are the ones below ``pinv``'s own relative cutoff.
    shift = (0.0 if ridge is None
             else float(ridge_shift(0.5 * (fac.S + fac.S.T), ridge, "trace")))

    # eig[0] runs slightly negative on a rank-deficient metric, so a condition number is
    # meaningless here; min_eig / max_eig says the same thing without dividing by it.
    return dict(max_eig=top, min_eig=float(eig[0]),
                min_over_max_eig=float(eig[0]) / top,
                shift=shift, shift_over_max_eig=shift / top,
                below_ridge=int(np.sum(eig < shift)), n_metric=int(len(eig)))


def measure(mol, mf, per_atom, ridge, n_laplace, draws, seed, scheme="ridge"):
    """One grid at one ridge: as fitted, then at ``draws`` random per-atom orientations.

    The energy and the torque come out of the same call, so the section 4(2) spread and
    the section 4(5) torque are computed on identical draws and the contrast between them
    carries no sampling difference.

    ``scheme`` picks the filter. Section 4(6) showed the ridge hands the metric's null
    directions the *largest* gain in the operator, `1/mu`, so at any `lambda` small enough
    to be accurate the gradient is mostly regulariser - and most of a torque measured
    there belongs to the regulariser too. ``"damped"`` sends those directions to zero
    instead, which is what makes a torque ladder measure the grid.
    """
    def run(rotations):
        return thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom, rotations), AUXBASIS,
                                n_laplace=n_laplace, metric_ridge=ridge, aux_ridge=ridge,
                                metric_scheme=scheme)

    res = run(None)
    net = float(np.linalg.norm(res.torque.sum(axis=0)))
    rms = float(np.sqrt(np.mean(np.sum(res.torque ** 2, axis=1))))
    grad = float(np.linalg.norm(res.de))

    row = dict(ridge=ridge, energy=float(res.energy), grad_norm=grad,
               net_torque=net, rms_atom_torque=rms, ratio=net / grad if grad else np.nan,
               translation_share=float(np.linalg.norm(res.translation_only)) / grad,
               collocation_share=float(np.linalg.norm(res.de_collocation)) / grad)

    if draws:
        rng = np.random.default_rng(seed)
        energies, nets = [], []
        for _ in range(draws):
            rots = Rotation.random(mol.natm, random_state=int(rng.integers(1 << 30)))
            r = run(rots)
            energies.append(float(r.energy))
            nets.append(float(np.linalg.norm(r.torque.sum(axis=0))))
        row.update(draw_energies=energies, draw_net_torques=nets,
                   spread_uha=1e6 * float(np.ptp(energies)),
                   rms_draw_net_torque=float(np.sqrt(np.mean(np.square(nets)))),
                   max_draw_net_torque=float(np.max(nets)))
    return row


def dump(path, name, mol, ref, n_laplace, draws, seed, scheme, rows):
    """Snapshot the ladder so far."""
    with open(path, "w") as fh:
        json.dump(dict(molecule=name, n_ao=int(mol.nao_nr()), natm=mol.natm, basis=BASIS,
                       auxbasis=AUXBASIS, mp2_ri_reference=ref, n_laplace=n_laplace,
                       draws=draws, seed=seed, scheme=scheme, rows=rows), fh, indent=2)


def main(name, modes, thresholds, ridges, n_laplace, draws, seed, with_parent,
         out_path, scheme="ridge", with_pinv=False):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()
    ref = float(DFMP2(mf).kernel()[0])

    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs, DF-MP2 ref {ref:.8f}")
    print(f"   modes {', '.join(modes)}; {draws} random per-atom orientations per grid; "
          f"{scheme} lambdas {', '.join(f'{r:.0e}' for r in ridges)}"
          + ("; plus a pinv control row per rung" if with_pinv else ""))
    print("   weight footing: blocked/ghostw/parent carry fitted weights, "
          "blocked1/ghost/parent1/molfit1 are w = 1, "
          "molw/molfit carry in-molecule per-element weights", flush=True)
    if any(m in INSITU_MODES for m in modes):
        print(f"   in-molecule weights trained on {', '.join(INSITU_TRAIN)}"
              + (f" - which INCLUDES {name}, so those rows are not held out"
                 if name in INSITU_TRAIN else f" - {name} is held out"), flush=True)

    rungs = [(m, t) for m in modes for t in thresholds.get(m, (None,))]
    if with_parent:
        rungs.extend([("parent", None), ("parent1", None)])

    # Section 4(6): both regularised schemes reduce to the truncated pseudoinverse as
    # lambda falls, so pinv is the torque they should converge on, and a ladder without
    # it cannot say whether a trend belongs to the grid or to the filter. window.py
    # carries this control for one grid per mode; here it runs on every rung.
    ladder = ([None] if with_pinv else []) + list(ridges)

    support_cache, rows = {}, []
    for mode, threshold in rungs:
        t0 = time.time()
        per_atom = build_per_atom(mol, mode, threshold, support_cache)
        n_points = sum(len(rel) for rel, _ in per_atom)
        label = f"{mode} {'parent' if threshold is None else f'{threshold:.0e}'}"

        print(f"\n-- {label}: {n_points} points  (build {time.time() - t0:.1f}s)",
              flush=True)
        print(f"   {'ridge':>7s} {'err/uHa':>10s} {'|grad|':>12s} "
              f"{'net tau':>11s} {'rms tau':>10s} {'tau/|grad|':>12s} "
              f"{'rms draw tau':>13s} {'spread/uHa':>11s} {'below ridge':>12s}")

        for ridge in ladder:
            t1 = time.time()
            row_scheme = "pinv" if ridge is None else scheme
            row = measure(mol, mf, per_atom, ridge, n_laplace, draws, seed, scheme)
            row.update(metric_diagnostics(mol, mf, per_atom, ridge, n_laplace))
            row.update(mode=mode, threshold=threshold, n_points=n_points,
                       scheme=row_scheme,
                       err_uha=1e6 * (row["energy"] - ref), seconds=time.time() - t1)
            rows.append(row)
            # Written per row, not at the end: a parent rung is an eigendecomposition of
            # a few-thousand-square metric and can be killed by the machine rather than
            # by itself, which on a write-at-the-end script loses every row before it.
            if out_path:
                dump(out_path, name, mol, ref, n_laplace, draws, seed, scheme, rows)

            print(f"   {'pinv' if ridge is None else f'{ridge:.0e}':>7s} "
                  f"{row['err_uha']:10.2f} {row['grad_norm']:12.4e} "
                  f"{1e6 * row['net_torque']:11.2f} {1e6 * row['rms_atom_torque']:10.2f} "
                  f"{row['ratio']:12.3e} "
                  f"{1e6 * row.get('rms_draw_net_torque', np.nan):13.2f} "
                  f"{row.get('spread_uha', np.nan):11.3f} "
                  f"{row['below_ridge']:6d}/{row['n_metric']:<5d}", flush=True)

    if out_path:
        dump(out_path, name, mol, ref, n_laplace, draws, seed, scheme, rows)
    return dict(molecule=name, n_ao=int(mol.nao_nr()), natm=mol.natm, basis=BASIS,
                auxbasis=AUXBASIS, mp2_ri_reference=ref, n_laplace=n_laplace,
                draws=draws, seed=seed, scheme=scheme, rows=rows)


def report(paths, ridge):
    """One ladder per mode, at one ridge, with the convergence factor across it.

    The verdict columns are ``net tau`` (the as-fitted attachment, which is what a
    trajectory would actually run on) and ``rms draw tau`` (the same statistic over
    random per-atom orientations, which is what the as-fitted number should be checked
    against in case its orientation is a lucky one). ``spread`` is section 4(2)'s
    statistic on the identical grids and draws, so the two can be read against each other
    without a sampling difference in the way.
    """
    for path in paths:
        d = json.load(open(path))
        label = "pinv" if ridge is None else f"{d['rows'][0].get('scheme', 'ridge')} {ridge:.0e}"
        print(f"\n=== {d['molecule']} ({d['n_ao']} AOs), {label}, "
              f"{d['draws']} draws")

        for mode in MODES:
            rows = [r for r in d["rows"]
                    if r["mode"] == mode and r["ridge"] == ridge]
            if not rows:
                continue
            rows.sort(key=lambda r: r["n_points"])
            print(f"\n  {mode}")
            print(f"    {'npts':>5s} {'err/uHa':>10s} {'net tau':>10s} "
                  f"{'rms draw tau':>13s} {'spread/uHa':>11s} {'tau/|grad|':>11s} "
                  f"{'null dirs':>10s}")
            for r in rows:
                print(f"    {r['n_points']:5d} {r['err_uha']:10.1f} "
                      f"{1e6 * r['net_torque']:10.1f} "
                      f"{1e6 * r.get('rms_draw_net_torque', float('nan')):13.1f} "
                      f"{r.get('spread_uha', float('nan')):11.1f} "
                      f"{r['ratio']:11.2e} "
                      f"{r['below_ridge'] / r['n_metric']:9.0%}")

            first, last = rows[0], rows[-1]
            for key, label in (("net_torque", "net tau"),
                               ("rms_draw_net_torque", "rms draw tau"),
                               ("spread_uha", "spread")):
                a, b = first.get(key), last.get(key)
                if a and b:
                    print(f"    {label:>14s}  {a / b:8.2f}x down over "
                          f"{first['n_points']} -> {last['n_points']} points")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--report", nargs="+",
                   help="read result JSONs and print the ladders; runs nothing")
    p.add_argument("--report-ridge", default="1e-4",
                   help="which rung to tabulate; 'pinv' selects the control row")
    p.add_argument("molecule", nargs="?", default="methanol", choices=sorted(MOLECULES))
    p.add_argument("--modes", default="blocked,blocked1,ghost",
                   help=f"comma separated, from {', '.join(MODES)}")
    p.add_argument("--thresholds", default=None,
                   help="comma separated; applied to every mode, overriding the defaults")
    p.add_argument("--ridges", default="1e-3,1e-4",
                   help="comma separated; must sit in section 4(5)'s 1e-2..1e-5 window")
    p.add_argument("--n-laplace", type=int, default=10)
    p.add_argument("--draws", type=int, default=4,
                   help="random per-atom orientations; 0 for the as-fitted row only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scheme", default="ridge", choices=["ridge", "damped"],
                   help="metric filter; section 4(6) says a torque must be quoted "
                        "with the regularisation that produced it")
    p.add_argument("--pinv", action="store_true",
                   help="add a pseudoinverse control row to every rung; without it a "
                        "trend cannot be attributed to the grid rather than the filter")
    p.add_argument("--parent", action="store_true",
                   help="add the unpruned atomic grid as the top of the ladder")
    p.add_argument("--out")
    a = p.parse_args()

    if a.report:
        report(a.report,
               None if a.report_ridge.strip() == "pinv" else float(a.report_ridge))
        raise SystemExit(0)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if any(m not in MODES for m in modes):
        raise SystemExit(f"modes must come from {', '.join(MODES)}")
    thresholds = dict(DEFAULT_THRESHOLDS)
    if a.thresholds:
        override = tuple(float(t) for t in a.thresholds.split(",") if t.strip())
        thresholds = {m: override for m in modes}

    main(a.molecule, modes, thresholds,
         [float(r) for r in a.ridges.split(",") if r.strip()],
         a.n_laplace, a.draws, a.seed, a.parent, a.out, a.scheme, a.pinv)

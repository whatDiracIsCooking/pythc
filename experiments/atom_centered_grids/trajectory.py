"""
What does the spurious torque do over a trajectory?

This is the measurement FINDINGS sections 11 and 12 both end by asking for, and the one
HANDOFF section 5 calls "the most informative single run left". Everything else in this
directory is a single point or a derivative at a single point. AIMD is the application the
whole smooth-PES argument is for, and nothing here had ever integrated one.

**What is at stake.** A per-element point set is attached to a nucleus by rigid
translation and keeps its *lab* orientation, so the energy of a molecule depends on how
that molecule is oriented relative to the lab, which a real energy does not. Section 11
made that concrete: ``tau_net = sum_A tau_A`` is exactly minus the energy's derivative
under a rigid rotation with the grids held lab-fixed, and therefore exactly the rate at
which nuclear angular momentum leaks::

    dL/dt = sum_A R_A x F_A = -sum_A R_A x dE/dR_A = +sum_A tau_A

Section 12 put that leak, on methanol's real transferable grid, at 3.1e-3 bohr/rad of the
gradient scale.

**Why one torque could not close it.** The frozen grid does not inject a non-conservative
force. ``E`` is a genuine function of all the coordinates, the three orientational ones
included, so the dynamics stays Hamiltonian and the total energy is still conserved. What
the freeze does is put the molecule in a weak *orientational potential* of its own
manufacture, and that has two possible consequences which differ by everything:

* the potential is bounded - its amplitude is section 7's rotation spread, hundreds of
  microhartree - so a tumbling molecule meets a torque that reverses every half turn and
  ``L`` oscillates inside a bound set by that amplitude over the angular velocity. A
  nuisance, not a defect; or
* the torque acts coherently, because the molecule barely rotates or because the
  orientational potential has a systematic sign across the orientations actually sampled,
  and ``L`` grows secularly. Fatal for long trajectories.

``dE/dtheta`` at one orientation cannot tell these apart. A trajectory can.

**How the leak is measured, and why not simply by propagating on the THC surface.**
``pythc.grad`` returns the correlation gradient at a *fixed* SCF reference; the orbital
response is not implemented. Propagating on ``E_HF + E_corr^THC`` with that force is
therefore energy-inconsistent, and the smoke test says what it costs: methanol drifts tens
of microhartree per step, which over a picosecond is a heated trajectory sampling
geometries the experiment did not ask for. So the default (``--propagate hf``) separates
the two jobs:

* the motion is driven by PySCF's analytic DF-RHF gradient, which is exact, consistent,
  and *exactly* rotationally invariant - so the reference trajectory conserves ``L`` to
  the integrator's floor and samples a proper thermal ensemble;
* the frozen grid's ``tau_net`` is evaluated at every geometry along it and integrated,
  giving ``dL = int tau dt`` - the angular momentum the corresponding frozen-grid
  trajectory would have leaked.

That is first order in the leak, and the approximation it makes is explicit and checked:
it neglects the feedback of the leaked ``L`` on the orientation. The run reports the
rotation angle that leaked ``L`` would have produced, so the reader can see whether the
orientation moved far enough for the neglected feedback to matter. ``--propagate full``
runs the fully coupled version for comparison over the short window where its energy
drift is still tolerable, and there the trajectory's own ``dL/dt`` is checked against the
analytic ``tau_net`` - two completely different routes to the same quantity, one a finite
difference of a dynamical variable and the other a reverse-mode derivative, with the
residual between them pricing the omitted orbital response.

**The initial condition is half the experiment.** With ``--project-rotation`` the net
angular momentum is removed from the starting velocities, so ``L(0) = 0`` exactly and
every subsequent quantum of it is spurious - nothing to subtract, no baseline to argue
about. That is also the coherent case, where a torque has its best chance of pointing the
same way for a long time. Without the projection the molecule carries thermal angular
momentum and tumbles, which is the realistic AIMD condition and the one where the torque
has a chance to average away. The two arms answer different halves of the question.

Everything runs on the damped filter at ``lambda = 1e-8``. Section 12 is what makes this
possible at all: under the ridge the gradient at that ``lambda`` is mostly noise, and
under ``pinv`` the surface has steps in it (section 6). The damped filter is the first
inversion in this programme that is at once smooth enough to integrate and quiet enough
to differentiate.

Usage:

    uv run python experiments/atom_centered_grids/trajectory.py methanol \
        --modes hf,blocked,ghost --steps 2000 --project-rotation \
        --out data/traj_methanol_fixed.json
    uv run python experiments/atom_centered_grids/trajectory.py methanol \
        --modes ghost --steps 200 --propagate full --out data/traj_coupled.json
    uv run python experiments/atom_centered_grids/trajectory.py --report data/traj_*.json
"""
import argparse
import json
import os
import sys
import time

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from pyscf.data import nist
from pyscf.dft import gen_grid, treutler_prune

from pythc.grad import FrozenGrid, thc_mp2_gradient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghosts import element_supports, per_atom_sets
from rotate import blocked_fit_per_atom
from sweep import MOLECULES, BASIS, AUXBASIS

AMU = nist.ATOMIC_MASS / nist.E_MASS       # amu -> electron masses
FS = 1e-15 / 2.4188843265857e-17           # femtoseconds -> atomic time units
KB = 3.166811563e-6                        # Hartree / Kelvin

MODES = ("hf", "blocked", "blocked1", "ghost", "parent")

# The grids sections 11 and 12 characterised on methanol, so the trajectory runs on an
# object whose torque at a point is already a published number rather than a new one.
DEFAULT_THRESHOLD = {"blocked": 1e-3, "blocked1": 1e-3, "ghost": 3e-4}


def build_per_atom(mol, mode, threshold):
    """The frozen point sets, fitted once at the starting geometry and never again.

    That "never again" is the whole proposal, so it is worth being explicit about where
    it is enforced: this runs once per mode before the first step, and the ``(rel, w)``
    pairs it returns are handed to a fresh :class:`FrozenGrid` at every subsequent
    geometry. Nothing downstream can re-select.
    """
    if mode == "hf":
        return None
    if mode == "parent":
        g = gen_grid.Grids(mol)
        atom_grids = g.gen_atomic_grids(mol, level=0, prune=treutler_prune)
        coords, weights = g.gen_partition(mol, atom_grids, concat=False)
        return [(c - mol.atom_coord(ia), w)
                for ia, (c, w) in enumerate(zip(coords, weights))]
    if mode in ("blocked", "blocked1"):
        fitted = blocked_fit_per_atom(mol, threshold)
        if mode == "blocked":
            return fitted
        return [(rel, np.ones(len(rel))) for rel, _ in fitted]

    elements = sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})
    return per_atom_sets(mol, element_supports(elements, threshold, quiet=True))


def inertia(mass, coords):
    """Inertia tensor about the centre of mass, and the COM-relative coordinates."""
    com = (mass[:, None] * coords).sum(0) / mass.sum()
    r = coords - com
    r2 = np.sum(mass[:, None] * r ** 2)
    return (r2 * np.eye(3) - np.einsum("a,ai,aj->ij", mass, r, r)), r


def angular_momentum(mass, coords, v):
    _, r = inertia(mass, coords)
    return np.einsum("a,ai->i", mass, np.cross(r, v))


def initial_velocities(mass, coords, temperature, seed, project_rotation):
    """Maxwell-Boltzmann, with net momentum - and optionally net ``L`` - removed.

    Removing the translation is bookkeeping. Removing the rotation is the experiment: it
    sets ``L(0) = 0`` exactly, so whatever angular momentum appears later is the frozen
    grid's doing and needs no reference to subtract.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=coords.shape) * np.sqrt(temperature * KB / mass)[:, None]
    v -= (mass[:, None] * v).sum(0) / mass.sum()

    if project_rotation:
        it, r = inertia(mass, coords)
        omega = np.linalg.solve(it, angular_momentum(mass, coords, v))
        v -= np.cross(omega, r)
        v -= (mass[:, None] * v).sum(0) / mass.sum()

    # Rescale on the surviving degrees of freedom, so both arms start with the same
    # vibrational energy and differ only in whether the molecule is also tumbling.
    dof = 3 * len(mass) - (6 if project_rotation else 3)
    kinetic = 0.5 * np.sum(mass[:, None] * v ** 2)
    return v * np.sqrt(0.5 * dof * KB * temperature / kinetic), dof


class Surface:
    """Energy, force and frozen-grid torque at a geometry.

    ``with_torque`` is how a long trajectory is afforded. The torque varies on the
    vibrational timescale - tens of femtoseconds - while the integrator needs half a
    femtosecond to follow an X-H stretch, so evaluating it at every step oversamples it
    by an order of magnitude. Sub-sampling it buys proportionally more physical time for
    the same cost, which is what decides whether the run reaches a rotational period.

    ``reference`` picks what generates the trajectory, and it is the whole of the DFT
    comparison. ``"rhf"`` is grid-free - density-fitted Hartree-Fock has no quadrature at
    all, which is why the reference trajectory conserves ``L`` to the integrator floor and
    why a leak measured along it belongs to the frozen grid. ``"rks"`` puts an ordinary
    Becke grid back in, generated in lab axes at every geometry exactly as the frozen
    grid is attached in lab axes - so whatever ``L`` an RKS trajectory fails to conserve
    is the same defect, in the method acTHC would have to be no worse than.

    ``orientation`` is the feedback. A leak is not a small correction that can be
    integrated at a fixed orientation: at a hundred microhartree per radian the molecule
    picks up enough angular momentum to turn through a radian in a few hundred
    femtoseconds, and the torque it meets there is not the torque it started in. Passing
    the accumulated spurious rotation ``Q`` here evaluates the torque where the molecule
    has actually got to, which is what decides between a leak that saturates once it has
    turned and one that does not.

    ``propagate`` decides which energy drives the motion. ``"hf"`` returns the DF-RHF
    force alone - exact, consistent and rotationally invariant - while still evaluating
    every frozen grid's torque at the same geometry, which is the separation the module
    docstring argues for. ``"full"`` adds one grid's fixed-orbital correlation force and
    accepts the energy inconsistency the missing orbital response brings with it.

    On the ``"hf"`` surface the motion does not depend on which grid is being measured,
    so several grids are carried at once and every one of them sees the *identical*
    sequence of geometries. That is not only cheaper than a trajectory apiece - it
    removes sampling from the comparison between them entirely, which matters because
    the whole question is a ratio between two grids' leaks.

    The SCF is seeded from the previous step's density, which is what makes a
    thousand-step run affordable. ``conv_tol`` is tight because section 11 found the
    gradient far more sensitive to a loose reference than the energy is.
    """

    def __init__(self, mol, grids, ridge, scheme, n_laplace, propagate, driver=None,
                 reference="rhf", xc="pbe", grid_level=0):
        self.orientation = None
        self.reference = reference
        self.xc = xc
        self.grid_level = grid_level
        self.mol = mol
        self.grids = grids
        self.ridge = ridge
        self.scheme = scheme
        self.n_laplace = n_laplace
        self.propagate = propagate
        self.driver = driver
        self.dm = None

    def __call__(self, coords, with_torque=True, orientation=None):
        self.mol.set_geom_(coords, unit="Bohr")
        self.mol.build(False, False)

        if self.reference == "rks":
            mf = scf.RKS(self.mol).density_fit(auxbasis=AUXBASIS)
            mf.xc = self.xc
            mf.grids.level = self.grid_level
        else:
            mf = scf.RHF(self.mol).density_fit(auxbasis=AUXBASIS)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        mf.kernel(dm0=self.dm)
        self.dm = mf.make_rdm1()

        energy = float(mf.e_tot)
        grad = mf.nuc_grad_method()
        if self.reference == "rks":
            # The Becke weight derivatives. Without them the DFT force is missing 12% of
            # its own norm on this grid, and the comparison would be measuring PySCF's
            # default rather than what DFT can do.
            grad.grid_response = True
        de = np.asarray(grad.kernel(), dtype=float)
        e_corr, torques = 0.0, {}

        if not with_torque:
            return energy, -de, e_corr, None

        # Rotating the molecule by Q against lab-fixed grids is the same configuration
        # as leaving it alone and turning every grid by Q^-1 about its own nucleus, which
        # FrozenGrid already supports exactly. That is how the leaked rotation is fed
        # back without touching the trajectory the RHF surface is generating.
        rotations = (None if orientation is None
                     else [orientation.T] * self.mol.natm)

        for mode, per_atom in self.grids.items():
            res = thc_mp2_gradient(self.mol, mf,
                                   FrozenGrid(self.mol, per_atom, rotations),
                                   AUXBASIS, n_laplace=self.n_laplace,
                                   metric_ridge=self.ridge, aux_ridge=self.ridge,
                                   metric_scheme=self.scheme)
            torques[mode] = res.torque.sum(axis=0)
            if self.propagate == "full" and mode == self.driver:
                e_corr = float(res.energy)
                energy += e_corr
                de = de + res.de

        return energy, -de, e_corr, torques


def rotate_by(omega, span, Q):
    """Advance a rotation matrix by angular velocity ``omega`` over ``span`` (Rodrigues)."""
    theta = float(np.linalg.norm(omega)) * span
    if theta < 1e-14:
        return Q
    k = omega / np.linalg.norm(omega)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K) @ Q


def run(name, modes, threshold, steps, dt_fs, temperature, seed, project_rotation,
        ridge, scheme, n_laplace, propagate, torque_every, feedback, out_path,
        reference="rhf", xc="pbe", grid_level=0):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mass = np.asarray(mol.atom_mass_list()) * AMU
    coords = mol.atom_coords().copy()

    t0 = time.time()
    grids = {m: build_per_atom(mol, m, threshold or DEFAULT_THRESHOLD.get(m))
             for m in modes if m != "hf"}
    n_points = {m: sum(len(r) for r, _ in g) for m, g in grids.items()}
    driver = next((m for m in modes if m != "hf"), None) if propagate == "full" else None

    print(f"== {name}: {mol.natm} atoms, {mol.nao_nr()} AOs; "
          + ", ".join(f"{m} {n} pts" for m, n in n_points.items())
          + (" (no grid)" if not grids else "")
          + f"  (fit {time.time() - t0:.1f}s)")
    print(f"   reference: {reference}"
          + (f"/{xc}, Becke grid level {grid_level}" if reference == "rks"
             else " (density fitted, no quadrature grid)"))
    print(f"   {steps} x {dt_fs} fs on the {propagate} surface"
          + (f" driven by {driver}" if driver else "")
          + f", T0 = {temperature} K, "
          f"{'L(0) = 0 by projection' if project_rotation else 'thermal rotation kept'}, "
          f"{scheme} lambda {ridge:.0e}"
          + (f", torque every {torque_every} steps" if torque_every > 1 else "")
          + (f", leaked rotation fed back on {feedback}" if feedback else ""))

    v, dof = initial_velocities(mass, coords, temperature, seed, project_rotation)
    surface = Surface(mol, grids, ridge, scheme, n_laplace, propagate, driver,
                      reference, xc, grid_level)
    dt = dt_fs * FS
    if propagate == "full":
        torque_every = 1

    # One grid is fed back on, because the spurious rotation is that grid's own and a
    # shared trajectory cannot carry two different ones. The others are still measured
    # along it, at the tracked grid's orientation.
    tracked = feedback if feedback in grids else None
    Q = np.eye(3)

    energy, force, e_corr, torques = surface(coords, orientation=Q if tracked else None)
    leak = {m: np.zeros(3) for m in grids}
    sampled, last_sample = torques, 0
    rows = []

    watched = driver or (list(grids)[-1] if grids else None)
    print(f"   {'step':>5s} {'t/fs':>8s} {'T/K':>7s} {'E_tot':>15s} {'dE/uHa':>9s} "
          f"{'|L|/hbar':>10s} " + (f"{'|dL| ' + watched:>16s} "
                                   f"{'|tau| ' + watched:>16s} " if watched else "")
          + f"{'s/step':>7s}", flush=True)

    for step in range(steps + 1):
        kinetic = 0.5 * float(np.sum(mass[:, None] * v ** 2))
        L = angular_momentum(mass, coords, v)
        rows.append(dict(step=step, time_fs=step * dt_fs, potential=energy,
                         e_corr=e_corr, kinetic=kinetic, total=energy + kinetic,
                         temperature=2 * kinetic / dof / KB, L=L.tolist(),
                         leak={m: l.tolist() for m, l in leak.items()},
                         torque=({m: t.tolist() for m, t in torques.items()}
                                 if torques is not None else None),
                         spun_deg=float(np.degrees(np.arccos(
                             np.clip(0.5 * (np.trace(Q) - 1), -1, 1)))),
                         coords=coords.tolist() if step % 100 == 0 else None))

        if step % 25 == 0 or step == steps:
            print(f"   {step:5d} {step * dt_fs:8.1f} {rows[-1]['temperature']:7.1f} "
                  f"{energy + kinetic:15.8f} "
                  f"{1e6 * (rows[-1]['total'] - rows[0]['total']):9.2f} "
                  f"{np.linalg.norm(L):10.3e} "
                  + (f"{np.linalg.norm(leak[watched]):16.3e} "
                     f"{1e6 * np.linalg.norm(sampled[watched]):16.2f} "
                     if watched else "")
                  + f"{(time.time() - t0) / max(step, 1):7.2f}", flush=True)
            if out_path:
                write(out_path, locals(), mol, rows)

        if step == steps:
            break

        v = v + 0.5 * dt * force / mass[:, None]
        coords = coords + dt * v
        energy, force, e_corr, torques = surface(
            coords, with_torque=(step + 1) % torque_every == 0 or step + 1 == steps,
            orientation=Q if tracked else None)
        v = v + 0.5 * dt * force / mass[:, None]

        # dL/dt = tau_net, integrated with the trapezoidal rule over whichever steps the
        # torque was sampled on; the leaked L then spins the molecule, and the next
        # torque is evaluated where that spin has put it.
        if torques is not None:
            span = (step + 1 - last_sample) * dt
            leak = {m: leak[m] + 0.5 * span * (sampled[m] + torques[m]) for m in grids}
            sampled, last_sample = torques, step + 1
            if tracked:
                it, _ = inertia(mass, coords)
                Q = rotate_by(np.linalg.solve(it, leak[tracked]), span, Q)

    if out_path:
        write(out_path, locals(), mol, rows)
    return rows


def write(path, scope, mol, rows):
    keys = ("name", "modes", "threshold", "n_points", "dt_fs", "temperature", "seed",
            "project_rotation", "ridge", "scheme", "propagate", "driver", "n_laplace",
            "torque_every", "feedback", "tracked", "reference", "xc", "grid_level")
    meta = {("molecule" if k == "name" else k): scope[k] for k in keys}
    meta.update(n_ao=int(mol.nao_nr()), natm=mol.natm, basis=BASIS, auxbasis=AUXBASIS,
                inertia=np.linalg.eigvalsh(inertia(
                    np.asarray(mol.atom_mass_list()) * AMU,
                    np.array(rows[0]["coords"]))[0]).tolist(), rows=rows)
    with open(path, "w") as fh:
        json.dump(meta, fh)


def analyse(d, mode):
    """The numbers the trajectory exists to produce, for one of the grids it carried.

    ``secular`` vs ``bounded`` is the verdict. A conservative orientational potential on a
    tumbling molecule gives a leak that oscillates inside a bound; a torque acting
    coherently gives one growing linearly in time. The straight-line fit's slope against
    the residual around it separates the two without having to eyeball a curve, and
    ``end / max`` says whether the run finished at its extreme (a ramp) or somewhere
    inside it (an oscillation).

    ``coherence`` is the same question asked of the torque rather than of its integral,
    and without reference to how long the run happened to be: the norm of the torque's
    time average over the rms of its instantaneous norm. 1 is a torque that never turns;
    0 is one that averages to nothing.

    ``spin_up`` and ``tilt`` are the decomposition that turns out to matter. A leak
    along ``L`` changes how fast the molecule spins, and energy conservation caps it: the
    work the orientational potential can do is bounded by its own amplitude, which is the
    section 7 rotation spread. A leak *perpendicular* to ``L`` does no work at leading
    order, so nothing bounds it - it tilts the rotation axis instead, and accumulates.
    The two are different failure modes and only the first would heat a trajectory, so a
    bare ``|leak|`` conflates the one energy conservation forbids with the one it allows.

    ``rotation_deg`` is the self-consistency check on the first-order treatment - the
    angle the leaked ``L`` would have turned the molecule through by the end. While that
    stays small the orientation barely moved, so the torque this trajectory sampled is
    the torque the fully coupled one would have sampled.
    """
    rows = d["rows"]
    t = np.array([r["time_fs"] for r in rows])
    bare = mode not in d["n_points"]
    # The torque is sampled on a subset of the steps; the leak is carried on all of them.
    samples = [r for r in rows if r["torque"] is not None]
    tau = (np.zeros((len(samples), 3)) if bare
           else np.array([r["torque"][mode] for r in samples]))
    tot = np.array([r["total"] for r in rows])
    L = np.array([r["L"] for r in rows])
    # On the hf surface the motion carries no leak, so the integrated torque is the
    # measurement; on the full surface the trajectory's own L is, and the two are checked
    # against each other below.
    coupled = d.get("propagate", "hf") == "full" and d.get("driver") == mode
    seen = ((L - L[0]) if (coupled or bare)
            else np.array([r["leak"][mode] for r in rows]))
    nrm = np.linalg.norm(seen, axis=1)

    out = dict(molecule=d["molecule"],
               mode=(d.get("reference", "rhf") if bare else mode),
               n_points=(0 if bare else d["n_points"][mode]),
               propagate="full" if coupled else "hf",
               project_rotation=d["project_rotation"],
               picoseconds=float(t[-1] / 1000), steps=len(rows) - 1,
               temperature=float(np.mean([r["temperature"] for r in rows])),
               energy_drift_uha=float(1e6 * (tot[-1] - tot[0])),
               measured_L=float(np.linalg.norm(L[-1])),
               final_leak=float(nrm[-1]), max_leak=float(nrm.max()),
               mean_torque_uha=float(1e6 * np.linalg.norm(tau.mean(axis=0))),
               rms_torque_uha=float(1e6 * np.sqrt(np.mean(np.sum(tau ** 2, axis=1)))))

    out["coherence"] = (out["mean_torque_uha"] / out["rms_torque_uha"]
                        if out["rms_torque_uha"] else float("nan"))

    # Split the leak along and across the angular momentum it is leaking into. With
    # L(0) = 0 there is no axis to split about and the whole leak is a spin-up.
    # Only meaningful when the molecule carries real rotation to decompose against; with
    # L(0) = 0 projected out there is no axis, and the whole leak is a spin-up.
    norm_L = np.linalg.norm(L, axis=1)
    if not d["project_rotation"] and norm_L[-1] > 10 * nrm[-1]:
        Lhat = L / np.maximum(norm_L, 1e-30)[:, None]
        along = np.einsum("ti,ti->t", seen, Lhat)
        across = np.linalg.norm(seen - along[:, None] * Lhat, axis=1)
        out["spin_up"] = float(along[-1])
        out["tilt"] = float(across[-1])
        out["tilt_deg"] = float(np.degrees(np.arctan2(across[-1], norm_L[-1])))
        out["L_scale"] = float(norm_L[-1])
    else:
        out["spin_up"] = float(nrm[-1])
        out["tilt"] = 0.0
        out["tilt_deg"] = 0.0
        out["L_scale"] = 0.0

    if len(rows) > 10:
        fit = np.polyfit(t / 1000, nrm, 1)
        out["secular_slope_per_ps"] = float(fit[0])
        out["residual_rms"] = float(np.std(nrm - np.polyval(fit, t / 1000)))
        out["end_over_max"] = float(nrm[-1] / nrm.max()) if nrm.max() else 0.0

    out["spun_deg"] = float(rows[-1].get("spun_deg", 0.0))
    out["feedback"] = d.get("feedback")

    if d.get("inertia"):
        # omega = I^-1 L on the smallest principal axis is the fastest the leak can spin
        # the molecule, so this is the worst case for the first-order treatment.
        out["rotation_deg"] = float(np.degrees(
            0.5 * (float(nrm[-1]) / min(d["inertia"])) * t[-1] * FS))

    out["torque_samples"] = len(samples)

    if coupled and len(rows) > 4:
        dt_au = (t[1] - t[0]) * FS
        dL = (L[2:] - L[:-2]) / (2 * dt_au)
        scale = np.linalg.norm(tau[1:-1])
        out["dL_dt_vs_tau"] = (float(np.linalg.norm(dL - tau[1:-1]) / scale)
                               if scale else float("nan"))
    return out


def report(paths):
    print(f"{'molecule':9s} {'mode':8s} {'prop':5s} {'pts':>5s} {'ps':>5s} "
          f"{'L(0)=0':>7s} {'T/K':>6s} {'<tau>':>9s} {'rms tau':>9s} {'coh':>6s} "
          f"{'leak end':>10s} {'spin up':>9s} {'tilt':>9s} {'tilt/deg':>9s} "
          f"{'per ps':>10s} {'spun/deg':>9s} {'E drift':>9s}")
    extra = []
    for path in paths:
        d = json.load(open(path))
        # With no frozen grid carried, there is no leak to integrate and the trajectory's
        # own failure to conserve L is the measurement - which is the whole point of the
        # rks comparison.
        for mode in (d["n_points"] or {"(reference)": 0}):
            a = analyse(d, mode)
            print(f"{a['molecule']:9s} {a['mode']:8s} {a['propagate']:5s} "
                  f"{a['n_points']:5d} {a['picoseconds']:5.2f} "
                  f"{str(a['project_rotation']):>7s} {a['temperature']:6.0f} "
                  f"{a['mean_torque_uha']:9.2f} {a['rms_torque_uha']:9.2f} "
                  f"{a['coherence']:6.2f} {a['final_leak']:10.3e} "
                  f"{a['spin_up']:9.3f} {a['tilt']:9.3f} {a['tilt_deg']:9.2f} "
                  f"{a.get('secular_slope_per_ps', float('nan')):10.3e} "
                  + (f"{a['spun_deg']:9.1f} " if a.get("feedback") == a["mode"]
                     else f"{a.get('rotation_deg', float('nan')):8.1f}* ")
                  + f"{a['energy_drift_uha']:9.1f}")
            if "dL_dt_vs_tau" in a:
                extra.append((a["molecule"], a["mode"], a["dL_dt_vs_tau"]))

    print("\nleak = int tau dt in hbar; torques in uHa/rad; E drift in uHa over the run.")
    print("coh = |<tau>| / rms|tau|: 1 is a torque that never turns, 0 one that averages")
    print("away. spin up / tilt split the leak along and across L: only the first does")
    print("work, so only the first is bounded by the orientational potential. tilt/deg is")
    print("the angle the rotation axis has been pushed through.")
    print("spun/deg is how far the leak actually turned the molecule; a starred value is")
    print("the first-order estimate for a grid that was not fed back on, and that grid's")
    print("leak is only meaningful while it stays small.")
    for molecule, mode, residual in extra:
        print(f"\n{molecule} / {mode}, coupled: the trajectory's own dL/dt against the "
              f"analytic\n  net torque is {residual:.2e} relative - that residual is the "
              f"omitted orbital response.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--report", nargs="+", help="read result JSONs; runs nothing")
    p.add_argument("molecule", nargs="?", default="methanol", choices=sorted(MOLECULES))
    p.add_argument("--modes", default="hf,blocked,ghost",
                   help=f"comma separated, from {', '.join(MODES)}")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the per-mode default NNLS threshold")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5, help="timestep in fs")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--project-rotation", action="store_true",
                   help="remove L from the initial velocities, so L(0) = 0 exactly")
    p.add_argument("--propagate", default="hf", choices=["hf", "full"],
                   help="which surface drives the motion; see the module docstring")
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--scheme", default="damped",
                   choices=["ridge", "damped", "ridge_jacobi", "damped_jacobi"])
    p.add_argument("--reference", default="rhf", choices=["rhf", "rks"],
                   help="rhf is grid-free and conserves L to the integrator floor; rks "
                        "puts an ordinary lab-fixed Becke grid back in, which is the "
                        "same defect the frozen grid has, in the method it has to beat")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--grid-level", type=int, default=0,
                   help="Becke grid level for --reference rks; 0 matches the parent grid "
                        "every THC fit in this directory is pruned from")
    p.add_argument("--n-laplace", type=int, default=10)
    p.add_argument("--feedback", default=None,
                   help="the grid whose leaked angular momentum is allowed to spin the "
                        "molecule, so its torque is evaluated where the leak has "
                        "actually put it. Without this the leak is integrated at a fixed "
                        "orientation and is only valid until the molecule has turned")
    p.add_argument("--torque-every", type=int, default=1,
                   help="evaluate the torque every N steps and integrate the leak over "
                        "those samples; the motion is still stepped at --dt. Forced to "
                        "1 by --propagate full, which needs the force at every step")
    p.add_argument("--out", help="one JSON carrying every mode's torque")
    a = p.parse_args()

    if a.report:
        report(a.report)
        raise SystemExit(0)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if any(m not in MODES for m in modes):
        raise SystemExit(f"modes must come from {', '.join(MODES)}")
    if a.propagate == "full" and len([m for m in modes if m != "hf"]) > 1:
        raise SystemExit("--propagate full drives the motion with one grid, so only one "
                         "non-hf mode can be carried; the hf surface carries any number")

    run(a.molecule, modes, a.threshold, a.steps, a.dt, a.temperature, a.seed,
        a.project_rotation, a.ridge, a.scheme, a.n_laplace, a.propagate,
        max(1, a.torque_every), a.feedback, a.out, a.reference, a.xc, a.grid_level)

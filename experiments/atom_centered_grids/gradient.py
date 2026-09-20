"""
Does the analytic gradient of a frozen-grid THC-MP2 energy agree with the curve?

This is section 5 of HANDOFF.md. Sections 4(1) to 4(4) established that the scheme has no
discrete step left at runtime - the points are selected offline, the weights do not
matter, the metric truncation was replaced by a ridge, and one frozen support per element
serves bonding it never saw. What none of that established is that anybody can actually
*compute* the derivative, or that it is the derivative of the curve `scan.py` measures.
`scan.py` can see a step in an energy curve; it cannot see a gradient that is quietly
missing a term.

Four things are measured here.

**1. Verification.** The analytic gradient against a central difference of the same
energy, orbitals held at their reference values in both so the comparison is of the THC
machinery alone and not of an orbital-response term neither side contains. A correct
implementation converges quadratically in the step size; anything that omitted a term -
the rigid point translation, say - would sit at a fixed relative error instead.

**2. The ridge floor, asked of a gradient rather than an energy.** Section 4(1) chose
`lambda = 1e-8` because the *energy* was smooth there, and warned that smoothness finds
ridge's floor before accuracy does. The same question put to the gradient has a sharper
answer, and it is the main result of this script: the derivative loses its meaning
several decades *above* the lambda the energy is happy at. The scan below is what says
where.

**3. Where the gradient comes from.** The collocation term splits into a rigid
translation of the frozen points and the ordinary AO derivative. The first exists only
because the grid is frozen and attached to nuclei, and it is the term a re-selected grid
would have no way to write down. Its size against the total says how much of the
gradient the freeze is responsible for.

**4. The torque.** Section 4(2) measured orientation dependence as an energy spread over
random rotations and concluded it converges away with grid size. A spread is not a
derivative: a small-amplitude, rapidly varying function of orientation has a tiny spread
and a large torque, and it is the torque that decides whether a frozen atomic grid
conserves angular momentum in dynamics. `dE/dtheta` is computed exactly here - rotating a
point set about its own nucleus leaves the molecule, the AOs and the SCF untouched, so no
response term enters - and printed beside the spread for the same grid.

A `pinv` row is included throughout as the control. The truncated pseudoinverse has no
derivative at an eigenvalue crossing, and the fixed-subspace derivative this package
computes for it is the right answer only between crossings; the finite difference is what
exposes the difference.
"""
import argparse
import json
import os
import sys

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf
from scipy.spatial.transform import Rotation

from pythc.grad import FrozenGrid, thc_mp2_gradient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghosts import element_supports, per_atom_sets
from rotate import blocked_fit_per_atom
from sweep import MOLECULES, BASIS, AUXBASIS


class FrozenReference:
    """The reference SCF solution, detached from any geometry.

    Both sides of the finite difference have to use the *same* orbitals, or the
    difference measures orbital relaxation - which this gradient deliberately does not
    contain - rather than the THC machinery.
    """

    def __init__(self, mf):
        self.mo_coeff = np.array(mf.mo_coeff)
        self.mo_energy = np.array(mf.mo_energy)


def build_grid(mol, threshold, mode):
    """The frozen point sets, either fitted in-molecule or transferred per element."""
    if mode == "blocked":
        return FrozenGrid(mol, blocked_fit_per_atom(mol, threshold))

    elements = sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})
    supports = element_supports(elements, threshold, quiet=True)

    return FrozenGrid(mol, per_atom_sets(mol, supports))


def energy_at(mol, ref, per_atom, coords_bohr, auxbasis, ridge, aux_ridge, n_laplace):
    """The frozen-grid THC-MP2 correlation energy at a displaced geometry."""
    moved = mol.copy()
    moved.set_geom_(coords_bohr, unit="Bohr")
    moved.build(False, False)

    return thc_mp2_gradient(moved, ref, FrozenGrid(moved, per_atom), auxbasis,
                            n_laplace=n_laplace, metric_ridge=ridge,
                            aux_ridge=aux_ridge, mo_coeff=ref.mo_coeff,
                            mo_energy=ref.mo_energy).energy


def fd_gradient(mol, ref, per_atom, auxbasis, ridge, aux_ridge, n_laplace, h,
                components=None):
    """Central-difference gradient over the requested components."""
    R0 = mol.atom_coords()
    out = np.full((mol.natm, 3), np.nan)

    for ia, k in (components if components is not None
                  else [(a, x) for a in range(mol.natm) for x in range(3)]):
        dp, dm = R0.copy(), R0.copy()
        dp[ia, k] += h
        dm[ia, k] -= h
        out[ia, k] = (energy_at(mol, ref, per_atom, dp, auxbasis, ridge, aux_ridge, n_laplace)
                      - energy_at(mol, ref, per_atom, dm, auxbasis, ridge, aux_ridge, n_laplace)) / (2.0 * h)

    return out


def verify(mol, mf, grid, auxbasis, ridges, steps, n_laplace, components):
    """Analytic against finite difference, across ridge strengths and step sizes."""
    ref = FrozenReference(mf)
    per_atom = list(zip(grid.rel, grid.w))
    rows = []

    print(f"\n{'ridge':>10s} {'|grad|':>12s} " +
          " ".join(f"{'max|d| h=' + f'{h:.0e}':>17s}" for h in steps) +
          f" {'E_corr':>15s}")

    for ridge in ridges:
        label = "pinv" if ridge is None else f"{ridge:.0e}"
        res = thc_mp2_gradient(mol, mf, grid, auxbasis, n_laplace=n_laplace,
                               metric_ridge=ridge, aux_ridge=ridge)
        norm = float(np.linalg.norm(res.de))

        errs = []
        for h in steps:
            num = fd_gradient(mol, ref, per_atom, auxbasis, ridge, ridge, n_laplace, h,
                              components)
            mask = ~np.isnan(num)
            errs.append(float(np.max(np.abs(res.de[mask] - num[mask]))))

        print(f"{label:>10s} {norm:12.4e} " +
              " ".join(f"{e:17.3e}" for e in errs) + f" {res.energy:15.9f}")

        rows.append(dict(ridge=ridge, grad_norm=norm, energy=res.energy,
                         steps=list(steps), max_abs_error=errs,
                         relative=[e / norm if norm else float("nan") for e in errs]))

    return rows


def invariances(mol, res):
    """
    Translational invariance, which is exact and needs no finite difference.

    Sliding the whole system changes nothing, so the forces sum to zero:

        sum_A dE/dR_A = 0

    This is a sharp test of the assembled gradient and in particular of the collocation
    term's two halves. The rigid translation moves every point with its nucleus and the
    AO derivative moves every basis function with its nucleus; under a uniform shift the
    two are equal and opposite across the whole molecule, so a missing, mis-signed or
    mis-mapped translation term cannot cancel and shows up here immediately.

    The rotational analogue - ``sum_A a_A x dE/dR_A + sum_A tau_A = 0`` - does **not**
    hold for this gradient, and that is not a defect. It assumes the orbitals rotate with
    the molecule; this gradient holds ``C`` fixed in the lab frame by construction, so
    the residual is exactly the orbital-response term ``C_bar . dC/dtheta`` that the
    fixed-orbital derivative omits. (Translation is unaffected because a uniform shift
    mixes no AOs, which is why the first identity survives and the second does not.) It
    becomes a test once the response layer is wired in; until then the torque is checked
    against a finite difference instead.
    """
    force_sum = res.de.sum(axis=0)
    scale = float(np.linalg.norm(res.de))

    print(f"  translational  |sum_A dE/dR_A| = {np.linalg.norm(force_sum):.3e}"
          f"   (relative to |grad| {scale:.3e}: {np.linalg.norm(force_sum) / scale:.2e})")

    return dict(translation=force_sum.tolist(),
                translation_norm=float(np.linalg.norm(force_sum)),
                translation_rel=float(np.linalg.norm(force_sum) / scale))


def decompose(mol, mf, grid, auxbasis, ridge, n_laplace):
    """Which stage of the pipeline the gradient comes from."""
    res = thc_mp2_gradient(mol, mf, grid, auxbasis, n_laplace=n_laplace,
                           metric_ridge=ridge, aux_ridge=ridge)

    ao_only = res.de_collocation - res.translation_only
    parts = [("rigid point translation", res.translation_only),
             ("AO derivative", ao_only),
             ("  (collocation total)", res.de_collocation),
             ("3-centre integrals", res.de_three_center),
             ("2-centre integrals", res.de_two_center),
             ("TOTAL", res.de)]

    print(f"\n  {'term':26s} {'norm / Ha bohr^-1':>20s} {'share of total':>16s}")
    total = float(np.linalg.norm(res.de))
    out = {}
    for name, part in parts:
        n = float(np.linalg.norm(part))
        out[name.strip()] = n
        print(f"  {name:26s} {n:20.6e} {n / total:15.2f}x")

    return out, res


def torque_check(mol, mf, grid, auxbasis, ridge, n_laplace, n_draw, seed, delta):
    """
    The analytic torque, a finite-difference check of it, and the energy spread beside it.

    The spread is what section 4(2) reported. It is computed here on the same grid so the
    two can be read against each other: a grid can have a negligible spread and a torque
    that is not negligible at all, and only one of those is what a gradient feels.
    """
    ref = FrozenReference(mf)
    per_atom = list(zip(grid.rel, grid.w))
    res = thc_mp2_gradient(mol, mf, grid, auxbasis, n_laplace=n_laplace,
                           metric_ridge=ridge, aux_ridge=ridge)

    def energy_with(rots):
        return thc_mp2_gradient(mol, ref, FrozenGrid(mol, per_atom, rots), auxbasis,
                                n_laplace=n_laplace, metric_ridge=ridge,
                                aux_ridge=ridge, mo_coeff=ref.mo_coeff,
                                mo_energy=ref.mo_energy).energy

    # Each atom is rotated about its own torque direction, so the finite difference sees
    # the full |tau| rather than whatever a fixed lab axis happens to project out - an
    # axis nearly orthogonal to the torque measures mostly noise. Several step sizes,
    # because the torque is small enough that a step has to clear the ridge's
    # conditioning floor, the same floor the nuclear finite difference runs into.
    deltas = [delta, delta * 10.0, delta * 100.0]
    print(f"\n  {'atom':>6s} {'analytic |tau|':>16s} " +
          " ".join(f"{'fd d=' + f'{d:.0e}':>16s}" for d in deltas))

    rows = []
    for ia in range(mol.natm):
        tau = res.torque[ia]
        norm = float(np.linalg.norm(tau))
        axis = tau / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

        nums = []
        for d in deltas:
            plus, minus = [np.eye(3)] * mol.natm, [np.eye(3)] * mol.natm
            plus[ia] = Rotation.from_rotvec(+d * axis).as_matrix()
            minus[ia] = Rotation.from_rotvec(-d * axis).as_matrix()
            nums.append((energy_with(plus) - energy_with(minus)) / (2.0 * d))

        print(f"  {mol.atom_symbol(ia) + str(ia):>6s} {norm:16.8e} " +
              " ".join(f"{n:16.8e}" for n in nums))
        rows.append(dict(atom=mol.atom_symbol(ia), torque=tau.tolist(),
                         analytic=norm, deltas=deltas, fd=nums))

    rng = np.random.default_rng(seed)
    energies = []
    for _ in range(n_draw):
        energies.append(energy_with(Rotation.random(mol.natm,
                                                    random_state=int(rng.integers(1 << 30)))))
    spread = 1e6 * float(np.ptp(energies)) if n_draw > 1 else float("nan")

    rms = float(np.sqrt(np.mean(np.sum(res.torque ** 2, axis=1))))
    net = float(np.linalg.norm(res.torque.sum(axis=0)))
    grad_norm = float(np.linalg.norm(res.de))

    # The net torque is the one with physical consequences. Rotating the molecule while
    # the point sets stay in their lab orientation is the same thing as rotating
    # everything - which the energy is invariant under - and then counter-rotating each
    # grid about its nucleus. So sum_A tau_A is exactly the energy's response to a rigid
    # rotation of the molecule under lab-fixed attachment, i.e. the spurious torque that
    # stops angular momentum being conserved. Section 4(2) posed the choice between a
    # lab frame and a molecule-dependent one and could only argue about it; this is the
    # number that was missing.
    print(f"\n  rms per-atom torque             {1e6 * rms:12.3f} uHa/rad")
    print(f"  net torque |sum_A tau_A|        {1e6 * net:12.3f} uHa/rad"
          "   <- angular momentum leak")
    print(f"  nuclear gradient norm           {1e6 * grad_norm:12.3f} uHa/bohr")
    print(f"  net torque / gradient norm      {net / grad_norm:12.3e} bohr/rad")
    if n_draw > 1:
        print(f"  energy spread over {n_draw:2d} draws     {spread:12.3f} uHa peak-to-peak"
              "   (what section 4(2) reported)")

    return dict(per_atom=rows, rms_torque=rms, net_torque=net,
                grad_norm=grad_norm, spread_uha=spread)


def main(name, threshold, mode, ridges, steps, n_laplace, h_rot, n_draw, seed,
         decompose_ridge, components, out_path):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()

    grid = build_grid(mol, threshold, mode)
    print(f"== {name}  {mode} grid, thr={threshold:.0e}, {grid.n_points} points, "
          f"{mol.nao_nr()} AOs, {mol.natm} atoms", flush=True)

    print("\n-- 1/2. analytic vs finite difference, and the ridge floor "
          "-------------------")
    rows = verify(mol, mf, grid, AUXBASIS, ridges, steps, n_laplace, components)

    print(f"\n-- 3. where the gradient comes from (ridge {decompose_ridge:.0e}) "
          "--------------------")
    parts, res = decompose(mol, mf, grid, AUXBASIS, decompose_ridge, n_laplace)

    print("\n-- exact invariance checks (no finite difference) --------------------")
    inv = invariances(mol, res)

    print(f"\n-- 4. torque on the frozen point sets (ridge {decompose_ridge:.0e}) "
          "-----------------")
    torque = torque_check(mol, mf, grid, AUXBASIS, decompose_ridge, n_laplace,
                          n_draw, seed, h_rot)

    result = dict(molecule=name, mode=mode, threshold=threshold,
                  n_points=grid.n_points, n_ao=int(mol.nao_nr()),
                  n_laplace=n_laplace, verification=rows,
                  decomposition=dict(ridge=decompose_ridge, norms=parts),
                  invariances=inv, torque=torque)

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("molecule", nargs="?", default="water", choices=sorted(MOLECULES))
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--mode", default="blocked", choices=["blocked", "ghost"],
                   help="in-molecule per-atom fit, or the transferable per-element one")
    p.add_argument("--ridges", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8",
                   help="comma separated; 'pinv' for the truncated pseudoinverse")
    p.add_argument("--steps", default="1e-3,1e-4",
                   help="finite-difference step sizes in Bohr")
    p.add_argument("--decompose-ridge", type=float, default=1e-4,
                   help="ridge for the decomposition and torque sections")
    p.add_argument("--n-laplace", type=int, default=10)
    p.add_argument("--h-rot", type=float, default=1e-4,
                   help="smallest rotation step in radians; x10 and x100 also run")
    p.add_argument("--draws", type=int, default=6,
                   help="random orientations for the energy spread, 0 to skip")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--atom", type=int, default=None,
                   help="restrict the finite difference to one atom, for speed")
    p.add_argument("--out")
    a = p.parse_args()

    ridges = [None if r.strip() == "pinv" else float(r)
              for r in a.ridges.split(",") if r.strip()]
    steps = [float(s) for s in a.steps.split(",") if s.strip()]
    comps = None if a.atom is None else [(a.atom, k) for k in range(3)]

    main(a.molecule, a.threshold, a.mode, ridges, steps, a.n_laplace, a.h_rot,
         a.draws, a.seed, a.decompose_ridge, comps, a.out)

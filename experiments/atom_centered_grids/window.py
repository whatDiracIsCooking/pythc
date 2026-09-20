"""
Where is the gradient's usable ridge window, what sets its floor, and does the damped
pseudoinverse move it?

FINDINGS section 11 measured a single uncomfortable fact: on water, the analytic
gradient is meaningless below `lambda = 1e-5`, three decades above the `1e-8` section 6
recommends for energies, so a gradient has to be bought with a hundred-fold worse energy
than an energy calculation would accept. It offered a diagnosis - ridge inverts the
metric's numerically null directions at `1/lambda`, and `Z = D^T D` carries `S^-1` twice
- but did not test it, and read the window off one system.

Three questions here, in the order they have to be answered.

**1. Is the diagnosis right?** If the floor is null-space amplification then it must
track the size of the null space, which `rank.py` can already count. The probe is a
*permutation*: relabelling an atom's grid points is an exact symmetry of the energy, the
gradient and the torque, so anything that changes under it is floating-point noise and
nothing else. That is a sharper instrument than section 11's three-processes comparison
and a deterministic one - it needs no threading nondeterminism to expose the floor, and
it says how much noise, not merely that there is some.

**2. Does the damped pseudoinverse move the floor?** `lib.damped_inv` applies
`sigma / (sigma^2 + mu^2)` instead of `1 / (sigma + mu)`. Both are analytic in `S` and
both peak at `1/(2 mu)`, so `lambda` means the same thing on each; they differ only on
the directions at issue, which the ridge amplifies and the damped filter suppresses. If
the diagnosis in (1) is right this should lower the floor by decades, and if it is wrong
it should do nothing at all.

**3. Does the window survive to a bigger molecule?** Section 11's window is water's. The
floor should rise with grid redundancy and the ceiling should fall with grid size - the
two walls close in - and if they meet before a useful system size then the scheme has no
usable lambda at all and the gradient is unavailable where it matters.

The ceiling is measured the same way throughout: the accuracy the regularisation costs,
against the same pipeline under the truncated pseudoinverse, which is the best either
scheme can do.
"""
import argparse
import json
import os
import sys

os.environ["PYTHC_USE_CUDA"] = "False"

import numpy as np
from pyscf import gto, scf

from pythc.grad import FrozenGrid, thc_mp2_gradient
from pythc.thc.ls_thc_funcs import build_S, eval_basefuncs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghosts import element_supports, per_atom_sets
from rotate import blocked_fit_per_atom
from sweep import MOLECULES, BASIS, AUXBASIS

NULL_CUTS = (1e-10, 1e-12, 1e-14, 1e-16)


def build_grid(mol, threshold, mode):
    """The frozen point sets, either fitted in-molecule or transferred per element."""
    if mode == "blocked":
        return FrozenGrid(mol, blocked_fit_per_atom(mol, threshold))

    elements = sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})

    return FrozenGrid(mol, per_atom_sets(mol, element_supports(elements, threshold,
                                                               quiet=True)))


def metric_spectrum(mol, mf, grid):
    """
    The LS-THC metric's spectrum, which is what the whole question is about.

    `S = (X_o X_o^T) o (X_v X_v^T)` is built exactly as the factorisation builds it, so
    the eigenvalue counts below are the ones the inversion actually sees.
    """
    coords, weights = grid.build()
    C = np.asarray(mf.mo_coeff)
    n_occ = mol.nelectron // 2

    X_ao = np.power(weights, 0.25)[:, None] * eval_basefuncs(mol, coords=coords)
    X = X_ao @ C
    S = build_S("ov", X, n_occ)

    eig = np.linalg.eigvalsh(0.5 * (S + S.T))
    top = float(eig[-1])
    rel = eig / top

    return dict(n_points=len(eig), max_eig=top, min_eig=float(eig[0]),
                mean_eig=float(np.mean(eig)), min_rel=float(rel[0]),
                n_null={f"{c:.0e}": int(np.sum(rel < c)) for c in NULL_CUTS},
                # The co-density rank is the most a grid of any size could resolve; a
                # grid past it is redundant by construction and its metric is singular
                # for reasons no lambda can fix.
                codensity_rank=int(n_occ * (C.shape[1] - n_occ)))


def permutations_of(grid, n_perm, seed):
    """
    Relabellings of each atom's point set: an exact symmetry, so a pure noise probe.

    The energy, the nuclear gradient and the per-atom torque are all invariant under
    permuting the points within an atom - the metric's rows and columns are permuted with
    them and every contraction is a sum. Whatever changes is floating-point noise, and
    how much of it gets through is exactly the amplification the regularisation applies
    to the directions where that noise lives.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        per_atom = []
        for coords, weights in zip(grid.rel, grid.w):
            p = rng.permutation(len(coords))
            per_atom.append((coords[p], weights[p]))
        out.append(per_atom)

    return out


def noise_floor(mol, mf, grid, auxbasis, ridge, scheme, n_laplace, perms):
    """
    Analytic gradient under the reference labelling and under permuted ones.

    Returns the worst relative disagreement in the gradient and in the energy. A
    well-conditioned inversion reproduces both to near machine precision; where the
    regularisation is amplifying the null space the gradient moves while the energy
    barely does, which is the whole asymmetry section 11 ran into.
    """
    kw = dict(n_laplace=n_laplace, metric_ridge=ridge, aux_ridge=ridge,
              metric_scheme=scheme)
    ref = thc_mp2_gradient(mol, mf, grid, auxbasis, **kw)
    norm = float(np.linalg.norm(ref.de))

    d_grad, d_energy, d_torque = 0.0, 0.0, 0.0
    for per_atom in perms:
        got = thc_mp2_gradient(mol, mf, FrozenGrid(mol, per_atom), auxbasis, **kw)
        d_grad = max(d_grad, float(np.max(np.abs(got.de - ref.de))) / norm)
        d_energy = max(d_energy, abs(got.energy - ref.energy))
        d_torque = max(d_torque, float(np.max(np.abs(got.torque - ref.torque))))

    net = float(np.linalg.norm(ref.torque.sum(axis=0)))

    return dict(ridge=ridge, scheme=scheme, energy=ref.energy, grad_norm=norm,
                net_torque=net, noise_grad=d_grad, noise_energy=d_energy,
                noise_torque=d_torque)


def fd_error(mol, mf, grid, auxbasis, ridge, scheme, n_laplace, h, atom):
    """Analytic against a central difference on one atom, orbitals fixed on both sides."""
    kw = dict(n_laplace=n_laplace, metric_ridge=ridge, aux_ridge=ridge,
              metric_scheme=scheme)
    C, eps = np.array(mf.mo_coeff), np.array(mf.mo_energy)
    res = thc_mp2_gradient(mol, mf, grid, auxbasis, **kw)
    per_atom = list(zip(grid.rel, grid.w))
    R0 = mol.atom_coords()

    worst = 0.0
    for k in range(3):
        vals = []
        for sign in (+1, -1):
            disp = R0.copy()
            disp[atom, k] += sign * h
            moved = mol.copy()
            moved.set_geom_(disp, unit="Bohr")
            moved.build(False, False)
            vals.append(thc_mp2_gradient(moved, mf, FrozenGrid(moved, per_atom), auxbasis,
                                         mo_coeff=C, mo_energy=eps, **kw).energy)
        worst = max(worst, abs(res.de[atom, k] - (vals[0] - vals[1]) / (2.0 * h)))

    return worst / float(np.linalg.norm(res.de))


def main(name, threshold, mode, ridges, schemes, n_laplace, n_perm, seed, fd_h, fd_atom,
         out_path):
    mol = gto.M(atom=MOLECULES[name](), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.kernel()

    grid = build_grid(mol, threshold, mode)
    spec = metric_spectrum(mol, mf, grid)

    print(f"== {name}  {mode} grid, thr={threshold:.0e}, {grid.n_points} points, "
          f"{mol.nao_nr()} AOs, {mol.natm} atoms", flush=True)
    print(f"   metric {spec['n_points']}x{spec['n_points']}, co-density rank "
          f"{spec['codensity_rank']}, min/max eig {spec['min_rel']:.2e}")
    print("   directions below a relative cutoff:  " +
          "  ".join(f"<{c}: {n}" for c, n in spec["n_null"].items()))

    # The truncation is the reference on both axes: it is the accuracy either
    # regularisation gives up against, and - since both schemes reduce to it as lambda
    # goes to zero - it is also the torque they should converge on. Without that second
    # control a torque measured at one lambda cannot be told apart from the
    # regulariser's own orientation dependence.
    perms = permutations_of(grid, n_perm, seed)
    pinv_row = noise_floor(mol, mf, grid, AUXBASIS, None, "ridge", n_laplace, perms)
    ref_energy = pinv_row["energy"]
    print(f"   pinv control: E_corr = {ref_energy:.9f}  |grad| = {pinv_row['grad_norm']:.4e}"
          f"  net torque = {1e6 * pinv_row['net_torque']:.3f} uHa/rad"
          f"  noise(grad) = {pinv_row['noise_grad']:.2e}")

    pinv_row.update(molecule=name, mode=mode, threshold=threshold,
                    n_points=grid.n_points, d_energy_vs_pinv=0.0,
                    shift_over_maxeig=float("nan"), scheme="pinv")
    rows = [pinv_row]

    for scheme in schemes:
        print(f"\n-- {scheme} " + "-" * 62)
        head = (f"{'lambda':>8s} {'shift/maxeig':>13s} {'E_corr':>14s} {'dE vs pinv':>12s} "
                f"{'|grad|':>11s} {'noise(grad)':>12s} {'noise(E)/uHa':>13s} "
                f"{'net torque':>11s}")
        if fd_h:
            head += f" {'fd err':>10s}"
        print(head)

        for ridge in ridges:
            row = noise_floor(mol, mf, grid, AUXBASIS, ridge, scheme, n_laplace, perms)
            row["molecule"], row["mode"], row["threshold"] = name, mode, threshold
            row["n_points"] = grid.n_points
            row["d_energy_vs_pinv"] = row["energy"] - ref_energy
            # Put lambda on pinv's axis. The ridge and the damped filter are both
            # scaled by the *mean* eigenvalue - the choice that keeps them analytic in
            # the nuclear coordinates - while pinv's epsilon is a fraction of the
            # largest, so the two ladders are not comparable as written.
            row["shift_over_maxeig"] = float(ridge * spec["mean_eig"] / spec["max_eig"])

            line = (f"{ridge:8.0e} {row['shift_over_maxeig']:13.2e} {row['energy']:14.9f} "
                    f"{1e6 * row['d_energy_vs_pinv']:11.2f}u {row['grad_norm']:11.4e} "
                    f"{row['noise_grad']:12.2e} {1e6 * row['noise_energy']:13.2e} "
                    f"{1e6 * row['net_torque']:11.3f}")

            if fd_h:
                row["fd_error"] = fd_error(mol, mf, grid, AUXBASIS, ridge, scheme,
                                           n_laplace, fd_h, fd_atom)
                line += f" {row['fd_error']:10.2e}"

            print(line, flush=True)
            rows.append(row)

    result = dict(molecule=name, mode=mode, threshold=threshold,
                  n_ao=int(mol.nao_nr()), n_laplace=n_laplace,
                  pinv_energy=ref_energy, spectrum=spec, n_perm=n_perm, rows=rows)

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("molecule", nargs="?", default="water", choices=sorted(MOLECULES))
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--mode", default="blocked", choices=["blocked", "ghost"])
    p.add_argument("--ridges", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10")
    p.add_argument("--schemes", default="ridge,damped")
    p.add_argument("--n-laplace", type=int, default=10)
    p.add_argument("--perms", type=int, default=3,
                   help="permutation draws for the noise probe")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--fd", type=float, default=None,
                   help="also run a one-atom finite difference at this step, in Bohr")
    p.add_argument("--fd-atom", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()

    main(a.molecule, a.threshold, a.mode,
         [float(r) for r in a.ridges.split(",") if r.strip()],
         [s.strip() for s in a.schemes.split(",") if s.strip()],
         a.n_laplace, a.perms, a.seed, a.fd, a.fd_atom, a.out)

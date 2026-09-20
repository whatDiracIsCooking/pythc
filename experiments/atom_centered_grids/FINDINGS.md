# Atom-centered THC grids: what the measurements say

## The proposal

Fit a THC grid once per element, offline, and translate it rigidly into any molecule -
the acCD move, applied to grid points instead of auxiliary basis functions. The prize is
unconditionally smooth potential energy surfaces: a frozen grid has no geometry
dependence, so `dw/dR = 0` and the only nuclear-derivative terms left are the ordinary
AO derivative and the rigid point-translation term. No active-set tracking, no
differentiation through a constrained optimisation.

The price is compactness. Per-atom grids cannot share bonding-region points between
neighbours, must provision for bond directions the atom may not have, and (see §4) may
need coarser, orbit-wise pruning to stay isotropic. Whether that price is 1.5x or 3x on
point count decides the whole programme, because `Z` is `n_P x n_P`.

## Why `blocked=True` answers the question cheaply

`NNLSGrid(blocked=True)` already fits each atomic sub-grid independently. Its inflation
over the global fit is a **strict lower bound** on what a frozen atom-centred scheme would
pay, because the blocked fit gets two advantages a frozen scheme never will:

* it sees each atom's **real** neighbours, not ghost approximations, so it is not
  over-provisioning for bond directions that turn out not to exist;
* it prunes **point-wise**, so it pays no isotropy tax.

It can therefore kill the idea cheaply, but cannot confirm it.

## Setup

cc-pVDZ / cc-pVDZ-RI, level-0 Becke parent grid, `ov` mode, 10 Laplace points, against a
DF-MP2 reference. Geometries from RDKit ETKDG + MMFF. "Converged" below means the MP2
error on the **full unpruned parent grid**, so quoted targets are deviations from that
floor and isolate grid error from the RI/Laplace error.

---

## 1. The size ratio is 1.2-1.7x, and improves with system size

Points needed to reach a given accuracy (global / blocked = ratio), interpolated log-log:

| target | water | methanol | ethanol | alanine |
| --- | --- | --- | --- | --- |
| 50 uHa | 91 / 156 = **1.71x** | 186 / 301 = **1.62x** | 350 / 477 = **1.36x** | 685 / 847 = **1.24x** |
| 20 uHa | 91 / 156 = **1.71x** | 193 / 301 = **1.56x** | 383 / 510 = **1.33x** | 743 / 1065 = **1.43x** |
| 10 uHa | 91 / 156 = **1.71x** | 199 / 301 = **1.51x** | 403 / 561 = **1.39x** | 819 / 1161 = **1.42x** |
| 5 uHa | 91 / 156 = **1.71x** | 205 / 301 = **1.47x** | 423 / 610 = **1.44x** | - |
| 2 uHa | 91 / 156 = **1.71x** | 212 / 308 = **1.45x** | 509 / 652 = **1.28x** | - |

Water saturates at the loosest grid tested, so its ratio is a single point, not a curve.
Alanine's tight rows are blank because the blocked fit at 1e-5 landed below the parent-grid
floor.

**1.2-1.7x on points is ~1.5-2.9x on the `n_P^2` parts** - not the 4-9x a 2-3x inflation
would have cost. The structural penalty for giving up molecular pruning is affordable,
and the trend with system size runs the right way.

Blocked is also far cheaper to fit, as its linear-vs-cubic scaling predicts: alanine at
1e-5 takes **39 s blocked against 164 s global**, and the gap widens with size.

## 2. LS-THC cares about the span of the co-density manifold, not about quadrature

`rank.py` on methanol (`n_occ * n_vir = 351`):

| grid | points | rank@1e-6 | metric rank | -log10 min eig |
| --- | --- | --- | --- | --- |
| global 1e-3 | 180 | 180 | 180 | 6.4 |
| global 1e-4 | 240 | 240 | 240 | 7.8 |
| global 1e-5 | 333 | 333 | 290 | 11.5 |
| blocked 1e-3 | 301 | 301 | 256 | 10.7 |
| blocked 3e-4 | 410 | 345 | 283 | 20.3 |
| blocked 1e-4 | 491 | 350 | 294 | 21.1 |
| blocked 1e-5 | 670 | 351 | 303 | 21.8 |

**Global NNLS is a perfect rank-revealing selector**: `rank == n_points` at every
threshold - every point it keeps adds exactly one independent co-density direction.
Blocked loses that above ~350 points, where independently-fitted atoms begin duplicating
each other's directions.

**The conditioning penalty is the real cost of unioning per-atom grids.** The metric's
smallest eigenvalue falls from ~1e-8..1e-11 (global) to ~1e-20..1e-22 (blocked). Ridge
regularisation of the `Z`-fit is therefore **mandatory** for an atom-centred scheme, not
an optional refinement - and it also removes the eigenvalue-truncation threshold, which is
itself a geometry-dependent discrete decision and so a second source of PES kinks.

## 3. The weights barely matter; the *selection* is the product

Because `Z` is fitted by least squares, rescaling the collocation matrix by any positive
diagonal `D` is absorbed exactly - `X -> DX` is compensated by `Z -> D^-2 Z D^-2`, since
the metric transforms as `S -> D^2 S D^2`. The fitted ERI is unchanged whenever the metric
is full-rank and the solve is exact. `weights.py` tests this by keeping the NNLS-selected
points and swapping the weights:

| system / grid | NNLS fitted | all ones | parent Becke | random x1000 |
| --- | --- | --- | --- | --- |
| methanol 1e-3, 180 pts | +120.69 | **+120.69** | +169.68 | +140.86 |
| ethanol 1e-3, 275 pts | +352.28 | **+352.28** | +667.93 | **+352.28** |
| methanol 1e-4, 240 pts | +2.65 | **+1.18** | +3.72 | +8.04 |
| ethanol 1e-4, 375 pts | +33.14 | +38.78 | +56.75 | +148.52 |

(MP2 error in uHa vs the DF-MP2 reference.)

Where the metric is full-rank, the energies are **bit-identical** across NNLS, all-ones
and uniform weights - exactly as the algebra predicts. At 240 points on methanol,
discarding the fitted weights entirely is *better* than keeping them (+1.18 vs +2.65 uHa).
The weights only start to matter once their dynamic range is wide enough to cause
numerical rank loss under the pseudoinverse's truncation: the parent Becke weights include
near-zero tail values that, through `w^(1/4)`, zero out whole rows of `X` (the methanol
1e-3 case is exactly singular, `-log10 min eig = 300`), and the same happens to weights
spanning three decades.

**This is the most consequential result for the programme.** It means the gradient
machinery the design discussion was built around is largely unnecessary. If the weights
are irrelevant, set `w = 1`: then `X = phi(r_P)` with no weight at all, `dX/dR` is the AO
derivative plus the rigid point-translation term, and there is no NNLS solve to
differentiate - no active set, no strict complementarity, no implicit function theorem, no
`theta^2` reparametrisation. The only thing that must be frozen is the **point selection**,
which a transferable scheme was going to freeze anyway.

A side note that is now mostly moot: this codebase builds `X = w^(1/4) phi(r)`
(`grid.py`, `ls_ri_becke.py`, `ls_ri_nnls.py`), not `X = w phi(r)`. So `dX/dw ~ w^(-3/4)`
diverges as `w -> 0`, and a `w = theta^2` reparametrisation would give `X = |theta|^(1/2) phi`,
still singular. The non-singular choice is to parametrise the **collocation amplitude**
directly, `X = t phi` with `w = t^4`: everything in the pipeline (including the metric
`(X X^T) o (X X^T)`) is then polynomial in `t`, and `t = 0` is a smooth point. Only needed
if a weight fit is kept at all.

## 4. Anisotropy is real, and the overlap RMSD cannot see it

`rotate.py` fits the blocked grid once, then spins each atom's point set about its **own**
nucleus - molecule, AOs and weights untouched - simulating a frozen grid attached at the
arbitrary orientation it would have in a real molecule.

| system | grid | regime | spread over 6 random orientations |
| --- | --- | --- | --- |
| water | 214 pts, 1e-4 | rank-saturated | **0.00 uHa** |
| methanol | 243 pts, 3e-3 | rank-limited | **39.0 uHa** peak-to-peak, std 15.3 |
| ethanol | 477 pts, 1e-3 | rank-limited | **26.6 uHa** peak-to-peak, std 9.0 |

Orientation is free once the grid oversamples the co-density manifold, and costs real
energy exactly in the compact, rank-limited regime the scheme wants to live in.

The shell structure explains it. Across methanol and ethanol, **not one angular shell
survives intact** - every retained shell keeps only 8-20% of its Lebedev points. NNLS
shreds the octahedral orbits that make the parent grid effectively isotropic. Since the
parent grid's `treutler_prune` already reduces Lebedev order per radial region in an
orbit-preserving way, the fix is to make NNLS prune **(radial shell, angular orbit) blocks**
rather than individual points - a group-sparsity variant. Its coarser granularity is a
further point-count tax not included in §1.

## 5. The overlap RMSD is close to orthogonal to THC accuracy

Two independent demonstrations:

* Rotating water's grid degraded `rmsd_S` from 2.1e-3 to 2.6e-1 - a factor of 125 - while
  leaving the MP2 energy unchanged **to 12 digits**.
* On methanol's 240-point grid, all-ones weights give `rmsd_S = 3.5` against the NNLS fit's
  `1.6e-3` - three orders of magnitude worse - and a *better* MP2 energy.

The overlap-fit objective is a convex, sparsity-inducing device for choosing a
well-conditioned support. It is not a measure of THC accuracy, and grids should not be
compared on it. What LS-THC needs is span and conditioning; §2's rank and minimum
eigenvalue are the diagnostics that track the energy.

---

## Verdict

The kill shot missed. The structural penalty for giving up molecular pruning is 1.2-1.7x,
not 3x, and it shrinks as molecules grow. More importantly, §3 removes most of the
difficulty from the gradient side of the proposal: freeze the point set, drop the weights,
and the nuclear derivative is elementary.

What remains to be measured, in order:

1. **The ghost gap.** §1 bounds a scheme that sees real neighbours. Build ghost-augmented
   per-element grids for H/C/N/O and measure how much worse than `blocked` they are. This
   is the only remaining question that can still kill the idea.
2. **The isotropy tax.** Implement group-NNLS over (radial shell, angular orbit) blocks and
   re-measure §1 and §4. Expect a larger grid and a smaller rotation spread.
3. **Ridge vs truncation.** Replace the pseudoinverse truncation with `(S + lambda I)^-1`
   and check the accuracy cost at fixed `lambda`. Needed for §2's conditioning, and removes
   a discrete decision from the PES.
4. **Transferability.** Fit the same element in several environments and compare the
   retained point sets. If they differ a lot, per-element grids need an environment
   ensemble rather than a single ghost geometry.

## Caveats

cc-pVDZ only; four small molecules; MMFF geometries; `ov` mode and MP2 only. The rank and
weight analyses are on methanol and ethanol. No gradients were computed - the smoothness
argument here is structural, not measured. Ghost-augmented per-element grids, the object
the proposal actually calls for, have not been built.

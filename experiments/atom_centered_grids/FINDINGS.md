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

## 6. The truncation really does put steps in the PES, and ridge removes them

Sections 1-5 measured grids. This one measures the *pipeline*, and settles the first
item on the next-directions list: with the grid frozen and the weights gone, the
eigenvalue truncation in the metric inversion is the last discrete decision left, and
`scan.py` shows it is not a theoretical worry.

The test scans a bond length with the point sets **frozen** - fitted once at the
reference geometry, thereafter only translated rigidly with their nuclei, which is
exactly the target pipeline - and reports the second difference of the THC error curve.
On a uniform scan that reads a jump off directly: for a smooth function it is
`O(h^2)` and negligible, while a step of size `d` contributes `d`.

| system, frozen grid | scheme | crossings | `|d2|` at a crossing | `|d2|` elsewhere |
| --- | --- | --- | --- | --- |
| methanol, 301 pts, 61 steps of 0.002 A | pinv (default) | 2 | **0.43, 0.46 uHa** | 0.006 |
| | ridge 1e-8 | - | 0.005, 0.001 | 0.002 |
| | ridge 1e-10 | - | 0.0003, 0.0001 | 0.002 |
| | *refit at every geometry* | - | 8.0, 3.7 | **4.0** |
| ethanol, 477 pts, 31 steps of 0.004 A | pinv (default) | 1 | **0.38 uHa** | 0.017 |
| | ridge 1e-8 | - | 0.001 | 0.012 |
| water, 118 pts, 61 steps of 0.002 A | pinv (default) | 0 | - | 0.0005 |

**The steps are real and they are exactly where the theory says.** Along methanol's
O-H scan the truncation retains 294-296 eigenvalues, changing at 2 of 60 steps, and the
two largest second differences in the whole curve sit precisely at those two steps -
80x the background. Ethanol has one crossing in 30 steps, with the largest second
difference on it, 22x the background. Under ridge the same points are indistinguishable
from their neighbours.

They are also *small*: about 0.5 uHa, on a curve whose THC error is 5 uHa (methanol) to
39 uHa (ethanol). The reason is section 3 all over again - the directions being
truncated carry almost no energy. Indeed **the truncation threshold makes no difference
to the energy at all**: from `epsilon = 1e-8` to `1e-14`, retaining anywhere from 412 to
477 of ethanol's 477 eigenvalues, the correlation energy is identical to nine decimals.
So the cutoff was never buying accuracy; it was numerical hygiene with a discontinuity
attached. That is the case for replacing rather than tuning it.

Magnitude is the wrong axis anyway. A 0.5 uHa step is a *discontinuity*: the derivative
at that geometry does not exist, an analytic gradient will disagree with finite
differences there, and an AIMD trajectory crossing it gains energy from nowhere.

**The frozen grid is what makes any of this visible.** The control row is the point:
re-running the NNLS selection at every geometry gives a second difference of ~4 uHa
*everywhere*, with no distinction between crossing and non-crossing points - the
selection noise is three orders of magnitude above the truncation's own steps and
swamps them completely. Freezing the grid removes that, and only then does the
truncation become the leading defect. The two steps are a package.

### Ridge has a floor, and smoothness finds it before accuracy does

Replacing `pinv(S)` with `(S + lambda I)^-1` costs very little accuracy over a wide
window (`lambda` below is dimensionless, scaled by the mean eigenvalue; `shift/maxeig`
puts it on the same axis as `pinv`'s `epsilon`):

| system / grid | truncation | ridge 1e-6 | ridge 1e-8 | ridge 1e-10 | ridge 1e-12 |
| --- | --- | --- | --- | --- | --- |
| water blocked 3e-3, 118 pts | -0.71 | +1.79 | -0.59 | -0.75 | -0.45 |
| methanol global 1e-4, 240 pts | +2.65 | +6.10 | +2.70 | +2.65 | +2.65 |
| methanol blocked 1e-3, 301 pts | +5.29 | +30.53 | +8.31 | +5.79 | +5.76 |
| ethanol blocked 1e-3, 477 pts | +38.64 | +80.81 | +40.84 | +38.37 | +37.80 |

(MP2 error in uHa vs the DF-MP2 reference. `shift/maxeig` at `lambda = 1e-8` is
3.3e-10 for water, 1.7e-10 for methanol and ethanol.)

At `lambda = 1e-8` the cost is 0.1-2.2 uHa; by `1e-10` it is under 0.5 uHa and the two
schemes have converged. `1e-6` is too strong and `1e-4` is useless (+170 to +450 uHa).

But ridge cannot be taken arbitrarily small, and this is the one genuinely new
constraint it introduces. Where the pseudoinverse *discards* the numerically null
directions, ridge inverts them at `1/lambda`. Water's blocked grid has 23 exactly null
directions (its co-density rank saturates at `n_occ * n_vir = 95` with 118 points), and
below `shift/maxeig ~ 1e-15` the fit is inverting rounding noise: the water energy error
goes from -0.45 uHa at `lambda = 1e-12` to several hundred uHa at `1e-14`, and is not
even reproducible between runs that differ only in SCF convergence tolerance.

The sharper version of the constraint comes from the scan, not the energy. On water,
where the truncation is inert (no crossings, `|d2| = 0.0005`), ridge is smooth at
`lambda = 1e-8` (`max|d2| = 0.003`) but **rough at `1e-10` (0.37) and destroyed at
`1e-12` (27.8)** - while the static energies at all three are within 0.3 uHa of each
other and of the truncation. So the static accuracy table above is not sufficient to
choose `lambda`: an energy that looks converged can sit on a curve whose derivative is
noise. **`lambda = 1e-8` at the default `"trace"` scaling is the value all three systems
agree on**: smooth everywhere, 0.1-2.2 uHa.

### What was implemented

`lib.ridge_inv` and `lib.ridge_inv_sqrt`, plus `metric_ridge` / `aux_ridge` keywords on
`LS_RI_THC`, which reach the metric inversion through
`ls_thc_funcs.invert_metric`. The default is unchanged - `None` keeps the truncated
pseudoinverse - so this is an opt-in path, not a silent change to every existing result.

One design note. `lambda` is dimensionless and scaled by `tr(S)/n`, the *mean*
eigenvalue, rather than by `max(eig(S))` as `pinv`'s `epsilon` is. That is not
cosmetic: the mean eigenvalue is linear in `S` and therefore an analytic function of the
nuclear coordinates, whereas the largest eigenvalue has a kink wherever it becomes
degenerate. Scaling the regulariser by a quantity with its own kink would reintroduce,
in miniature, the thing being removed. `"max_eig"` is available for comparing against
`epsilon` directly.

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
3. ~~**Ridge vs truncation.**~~ **Done - see §6.** The truncation does put ~0.5 uHa steps
   in the PES, exactly at the eigenvalue crossings; ridge at `lambda = 1e-8` removes them
   for 0.1-2.2 uHa. Ridge has a floor of its own, and smoothness finds it before accuracy
   does.
4. **Transferability.** Fit the same element in several environments and compare the
   retained point sets. If they differ a lot, per-element grids need an environment
   ensemble rather than a single ghost geometry.

## Caveats

cc-pVDZ only; four small molecules; MMFF geometries; `ov` mode and MP2 only. The rank and
weight analyses are on methanol and ethanol. Ghost-augmented per-element grids, the object
the proposal actually calls for, have not been built.

§6 measures smoothness by finite differences along one bond of one molecule per system;
no analytic gradient was computed, and nothing here checks that an implemented gradient
agrees with the curve. The scans move a terminal hydrogen only, so they probe the metric
less violently than a heavy-atom displacement would, and they are single 1D cuts - a
crossing rate of 2 per 60 steps is an estimate from a small sample. The `lambda = 1e-8`
recommendation rests on three systems in one basis, and the floor below which ridge
becomes noise depends on how null the null space really is, which is a property of the
grid and the basis.

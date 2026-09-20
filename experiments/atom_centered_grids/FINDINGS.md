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

It can therefore kill the idea cheaply, but cannot confirm it. **§8 supplies the
confirmation**: ghost-augmented per-element grids are now built and measured, and they
cost 1.1-1.8x over `blocked` on top of `blocked`'s 1.2-1.7x over `global`.

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
orbit-preserving way, the obvious fix is to make NNLS prune **(radial shell, angular
orbit) blocks** rather than individual points - a group-sparsity variant, at the cost of
a coarser granularity and so a further point-count tax not included in §1.

**That fix was built and measured in §7, and it is not the fix.** Both grids tabulated
here are rank-limited, and the spread falls to 0.01-1.1 uHa in *either* mode once the
grid is converged. Read this section as a measurement of compact grids, not as a defect
that survives.

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

## 7. Whole orbits buy exact octahedral invariance, and nothing else

`orbits.py`. §4 left the scheme without a well-defined orientation: no angular shell
survived a per-atom NNLS fit intact, and spinning each atom's point set about its own
nucleus moved the energy by tens of uHa. The proposed fix was to make the fit select
**(radial shell, octahedral orbit) blocks** rather than individual points. That is now
implemented - `grid.octahedral_orbits` labels the blocks, `decomp.nnls.GroupOperator`
ties their weights together, and `NNLSGrid(group_orbits=True)` uses them - and measured.

It works exactly as designed, and it does not do what it was wanted for.

Per-atom (`blocked`) fits against the DF-MP2 reference. "octa" is the energy shift when
each atom's grid is spun by a random element of the octahedral group; "ptp" is the
peak-to-peak spread over random `SO(3)` rotations (6 on methanol, 5 on ethanol).
Methanol, parent-grid floor `+2.75` uHa:

| mode | points | err/uHa | octa/uHa | ptp/uHa | whole orbits |
| --- | --- | --- | --- | --- | --- |
| point-wise | 175 | +627.76 | -81.93 | 218.22 | 0 / 82 |
| point-wise | 235 | +42.98 | -3.31 | 74.60 | 0 / 99 |
| point-wise | 271 | +18.56 | -8.93 | 15.49 | 0 / 109 |
| point-wise | 301 | +5.29 | -1.08 | 7.07 | 0 / 111 |
| point-wise | 356 | +3.33 | +0.48 | 3.00 | 0 / 118 |
| point-wise | 410 | +2.78 | +1.35 | 0.98 | 0 / 124 |
| point-wise | 491 | +2.76 | +0.10 | **0.17** | 0 / 134 |
| point-wise | 670 | +2.75 | -0.14 | **0.01** | 0 / 144 |
| orbits | 304 | +2007.61 | **+0.00** | 503.72 | 14 / 14 |
| orbits | 360 | +401.94 | **-0.01** | 371.53 | 17 / 17 |
| orbits | 408 | +78.83 | **-0.00** | 87.93 | 19 / 19 |
| orbits | 504 | +2.98 | **+0.00** | 0.53 | 23 / 23 |
| orbits | 512 | +2.91 | **-0.00** | 0.63 | 24 / 24 |
| orbits | 732 | +2.83 | **+0.01** | 0.05 | 34 / 34 |

Ethanol, floor `+5.96` uHa:

| mode | points | err/uHa | octa/uHa | ptp/uHa | whole orbits |
| --- | --- | --- | --- | --- | --- |
| point-wise | 375 | +136.95 | -32.59 | 96.46 | 0 / 156 |
| point-wise | 477 | +38.64 | -33.68 | 25.73 | 0 / 158 |
| point-wise | 602 | +11.97 | -0.28 | 5.64 | 0 / 172 |
| point-wise | 765 | +6.19 | -0.02 | **1.14** | 0 / 203 |
| orbits | 512 | +4942.73 | **-0.00** | 110.58 | 24 / 24 |
| orbits | 740 | +18.91 | **+0.00** | 40.80 | 34 / 34 |
| orbits | 772 | +13.36 | **-0.00** | 30.93 | 36 / 36 |
| orbits | 970 | +7.47 | **-0.00** | 1.15 | 47 / 47 |

### The octahedral invariance is exact

Every orbit-grouped fit is invariant to round-off under the 24 proper rotations of the
octahedral group, at every threshold on both molecules, while the point-wise fits shift
by up to 82 uHa under the same operation. That is not a numerical accident: the retained
set is a union of whole `O_h` orbits carrying one weight each, so a group element merely
permutes the points among themselves, and the LS-THC fit does not care how its points
are ordered. The `whole orbits` column reports it directly - **every orbit the grouped
fit touches, it keeps entire** (23/23, 47/47), against 0 of 82 to 0 of 203 point-wise.

The residual is 0.00-0.01 uHa rather than identically zero. The obvious suspect was §6's
mechanism in miniature - the permutation reorders the summations, an eigenvalue moves
across `pinv`'s cutoff - but that is not it: on methanol's 732-point grouped grid the
shift is `-0.0071` uHa under the truncation and `-0.0170` uHa under `metric_ridge = 1e-8`,
so regularising does not remove it. What is left is ordinary floating-point
non-associativity amplified by the metric's conditioning: a relative perturbation of
1e-16 through a metric whose smallest eigenvalue is ~1e-20 (§2) lands exactly at the
1e-8 Ha observed. The invariance is exact in the algebra and limited by the conditioning
in practice, which is a statement about §2, not about §6.

### It never reduces the general-rotation spread

`O_h` is a finite subgroup of `SO(3)`, and closing the discrete symmetry does nothing
for the continuous one. **At matched point count the point-wise grid is better on both
axes, everywhere it was measured:**

| points | point-wise err / ptp | orbit-grouped err / ptp |
| --- | --- | --- |
| methanol ~300 | +5.29 / 7.07 | +2007.61 / 503.72 |
| methanol ~500 | +2.76 / **0.17** | +2.98 / 0.53 |
| methanol ~700 | +2.75 / **0.01** | +2.83 / 0.05 |
| ethanol ~770 | +6.19 / **1.14** | +13.36 / 30.93 |

And at matched *accuracy* grouping buys nothing either: ethanol reaches a 1.14 uHa
spread on 765 points point-wise and a 1.15 uHa spread on 970 grouped points - the same
number for 1.27x the points.

The reason is visible in the point-wise column on its own. Methanol's spread falls
218 -> 74.6 -> 15.5 -> 7.07 -> 3.00 -> 0.98 -> 0.17 -> 0.01 uHa as the grid grows,
monotonically, with no structural change whatever. **Orientation dependence is a symptom
of a rank-limited grid, not an independent defect**: it converges away at the same rate
the energy error does, and a point-wise fit spends points on it more efficiently than a
grouped one, because an orbit of 24 or 48 points is a coarse thing to spend.

So §4's framing is the thing these measurements revise. Anisotropy is a real
rotational-invariance failure, and §4 measured it correctly - but on grids chosen in the
compact regime, and it is not what stands between the programme and a usable frozen
grid.

### The threshold is a coarser knob, and not monotone

Because the group gradient sums over up to 48 points, the grouped fit's KKT threshold is
far tighter at the same numeric value: methanol needs `1e-1` to `3e0` where the
point-wise fit needs `1e-5` to `1e-2`. The two ladders are comparable at matched
accuracy or matched size, never at matched threshold.

The grouped ladder is also coarse and not monotone in point count. Methanol goes from
408 points at +78.83 uHa straight to 504 at +2.98 with nothing in between; `4e-1` and
`3e-1` give the identical grid, while `2e-1` gives a *larger* one at 512 points. Orbits
differ in size, so admitting one is not a fixed increment, and the active set can settle
on a different subset when the threshold moves. There is no smooth size knob here.

### What this means for the programme

**Orbit grouping is not the fix for orientation dependence; converging the grid is.** The
justification for it - that it is what makes "attach the atomic grid rigidly" a
well-defined operation - does not survive contact with the numbers, and it should not
become the default.

What it is still good for is narrow but real. A *molecule-dependent local frame* (aligned
to bonds, say) can switch orientation discontinuously as the geometry changes; where the
switch is by an octahedral element, a grouped grid gives the identical energy and a
point-wise one kinks by up to 82 uHa. If a frame-based attachment scheme is ever built,
this removes that class of discontinuity for 1.2-1.6x the points. A fixed lab
orientation, which never switches frames, gets nothing from it.

It should therefore be decided alongside (3), the ghost gap, rather than before it. A
ghost-augmented per-element fit sees an artificial environment and may be anisotropic in
ways a real-neighbour `blocked` fit is not; that is the case where exact `O_h` invariance
could still earn its points, and it is untested.

## 8. The ghost gap: ghosts work, free atoms cannot

`ghosts.py`. This is the experiment the whole programme was pointed at. Every number
before this one bounds a scheme that still sees the molecule: `blocked` fits each atom
separately, but it fits it *in situ*, against that atom's real neighbours at their real
positions. A transferable grid never sees them. It is fitted once per element, offline,
against a guess at what a neighbour looks like, and then attached to every atom of that
element in every molecule. §1 priced giving up molecular pruning. This prices giving up
the molecule.

Three point sets per element, all selected by the same NNLS solver on the same level-0
atomic grid, all evaluated with `w = 1` (§3) and `metric_ridge = 1e-8` (§6) so that the
only difference between them is *where the points came from*:

* **blocked** - fitted in the molecule, per atom. The lower bound. Not transferable.
* **free** - fitted on the isolated atom. The strawman the design discussion predicted
  would fail.
* **ghost** - fitted against ghost atoms (neighbour basis functions, no nuclear charge)
  at 12 icosahedral directions, cycling through partners H/C/O at 0.90, 1.05 and 1.45
  times the covalent bond length, plus the free atom itself: 13 environments stacked
  into **one** NNLS solve over the shared atomic grid
  (`decomp.nnls.StackedOperator`). The proposal.

Methanol, parent-grid floor `+2.68` uHa; ethanol, floor `+5.96` uHa:

| mode | methanol pts | err/uHa | ethanol pts | err/uHa |
| --- | --- | --- | --- | --- |
| free | 105 | +7659.85 | 157 | +14904.79 |
| free | 141 | +1623.51 | 208 | +3436.43 |
| free | 170 | +386.35 | 255 | +1016.41 |
| free | 200 | +155.79 | 299 | +340.52 |
| ghost | 223 | +277.87 | 330 | +767.30 |
| ghost | 414 | +19.78 | 620 | +119.26 |
| ghost | 512 | +30.26 | 765 | +61.50 |
| ghost | 627 | **+8.85** | 927 | +16.80 |
| ghost | 702 | **+5.85** | 1043 | **+10.68** |
| ghost | 816 | **+5.67** | 1208 | +10.97 |
| blocked | 301 | +16.41 | 477 | +66.71 |
| blocked | 491 | +5.29 | 765 | +14.85 |
| blocked | 670 | +4.15 | 1090 | +9.14 |

`analyse.py`, interpolating each curve to a fixed distance above its floor:

| target | methanol blocked | methanol ghost | ethanol blocked | ethanol ghost |
| --- | --- | --- | --- | --- |
| 50 uHa | 301 | 326 (**1.08x**) | 500 | 775 (**1.55x**) |
| 20 uHa | 301 | 400 (**1.33x**) | 627 | 863 (**1.38x**) |
| 10 uHa | 330 | 587 (**1.78x**) | 743 | 938 (**1.26x**) |
| 5 uHa | 405 | 650 (**1.60x**) | 933 | 1035 (**1.11x**) |

`free` reaches no row of that table on either molecule: it never gets within 50 uHa of
the floor at all.

### The ghost gap is 1.1-1.8x, and it shrinks as the molecule grows

That is the headline, and it is the same shape as §1's result one level down. Ethanol's
ratios are *lower* than methanol's and fall as the target tightens - 1.55x at 50 uHa to
1.11x at 5 uHa - which is the opposite of what a scheme that is fundamentally missing
information would do. Whatever the ghost fit gets wrong is a fixed per-atom overhead,
not a defect that compounds.

Compounding it with §1 gives the number the programme actually has to pay. Against a
single *global* molecular fit, a frozen per-element grid costs

* methanol: 1.51x (blocked/global) x 1.78x (ghost/blocked) = **2.7x** at 10 uHa,
  1.47 x 1.60 = **2.4x** at 5 uHa;
* ethanol: 1.39 x 1.26 = **1.8x** at 10 uHa, 1.44 x 1.11 = **1.6x** at 5 uHa.

So **1.6-2.7x on points, 2.6-7.3x on the `n_P^2` parts**, improving with system size.
That lands back inside the design discussion's original "2-3x, i.e. 4-9x" estimate,
which §1 had deflated to 1.2-1.7x - and §1's deflation should not be quoted on its own
any more. It priced one of the two things a transferable grid gives up.

### Free atoms fail for a reason nobody anticipated: the target runs out of rank

The design discussion predicted the free-atom fit would fail *geometrically* - an
isolated atom has no amplitude where a bond would be, so NNLS discards exactly the tail
points a bond needs. It does fail, comprehensively: at 200 points methanol is still
156 uHa from its floor where `ghost` is at 8.85 uHa on 627.

But the binding mechanism is not placement, it is **information**. An NNLS solve can
retain at most `min(n_equations, n_variables)` points, and the isolated atom's target
supplies only `n_AO (n_AO + 1) / 2` equations - the unique entries of its own overlap
matrix. In cc-pVDZ that is **15 for hydrogen** and 105 for carbon or oxygen. Driving
the threshold to `1e-12` confirms it exactly:

| element | AOs | pair equations | free atom keeps | ghost keeps | grid |
| --- | --- | --- | --- | --- | --- |
| H | 5 | **15** | **15** (hard cap) | 132 | 392 |
| C | 14 | 105 | 92 | 155 | 858 |
| O | 14 | 105 | 92 | 199 | 858 |

Hydrogen sits exactly on its rank bound: **no threshold, no solver and no amount of
patience can give a free-atom fit more than 15 points per hydrogen in this basis.**
Methanol's free-atom grid therefore cannot exceed `4 x 15 + 92 + 92 = 244` points however
it is tuned. Built at exactly that ceiling it lands at **+69.79 uHa**, 67 uHa above the
floor, and there is nowhere further to go: 244 points is not a threshold choice, it is
the end of the ladder. Carbon and oxygen converge before their bound rather than on it,
but the bound is what makes the failure structural rather than a matter of tuning.

Ghosts fix this on both counts at once, and the rank one is the important one. Each
environment contributes its own AO-pair equations - `O` with a ghost carbon has 28 AOs
and 406 pairs, not 105 - and thirteen of them stacked give the solver roughly 3500
equations to work with instead of 105. The point set can then be as large as the grid
supports, and the extra points land where the ghosts put amplitude, which is where the
bonds are.

This is worth stating plainly because it changes what "fit an element offline" means.
The offline target is not a property of the element; it is a property of the element
*plus an environment*, and without an environment it is not merely biased, it is rank
deficient. A ghost is not a correction to the free-atom fit. It is what makes the fit
well posed.

### Orientation: worse per point, identical per uHa

§7 closed with one case it could not test - a grid trained against ghosts in twelve
chosen directions has a reason to be anisotropic that a real-neighbour `blocked` fit does
not, and that is exactly where exact `O_h` invariance might still have earned its points.
`ghosts.py --rotate` tests it, spinning each atom's frozen point set about its own
nucleus. All rows are methanol at `w = 1` and `metric_ridge = 1e-8`, so `blocked` is
re-measured here rather than read across from §7, which used NNLS weights under the
truncated pseudoinverse. Floor `+2.86` uHa, 4 random `SO(3)` draws per grid:

| mode | points | err/uHa | ptp/uHa |
| --- | --- | --- | --- |
| free | 244 (**its ceiling**) | +69.79 | 34.40 |
| ghost | 414 | +19.78 | 27.08 |
| ghost | 627 | +8.85 | 4.20 |
| ghost | 882 | +5.14 | **0.94** |
| blocked | 301 | +16.41 | 11.67 |
| blocked | 491 | +5.29 | **0.91** |
| blocked | 670 | +4.15 | 0.33 |

**At matched point count the ghost grid is 2-5x more orientation-dependent**: 627 points
spin by 4.20 uHa where 670 blocked points spin by 0.33. So the worry was well founded.

**At matched accuracy the difference vanishes entirely.** Ghost at 882 points sits at
+5.14 uHa with a 0.94 uHa spread; blocked at 491 points sits at +5.29 uHa with a 0.91 uHa
spread. Same energy, same spread, 1.80x the points - which is just §8's point-count tax,
paid once, with nothing extra owed on the orientation axis.

That is §7's thesis holding in the case §7 could not reach. Orientation dependence tracks
how rank-limited the grid is and nothing else; a ghost-fitted grid is further from
converged at a given size, so it is more anisotropic at a given size, and the fix is the
same fix - more points - rather than a structural one. **Orbit grouping does not get a
second chance here.** It would buy exact invariance under 24 of the rotations at 1.2-1.6x
the points (§7), where simply converging the grid buys the same spread under all of them
for point count that is already being spent.

The free-atom row is worth reading alongside. Driven to its hard rank ceiling of 244
points it still sits 67 uHa above the floor and spins by 34 uHa, which is the anisotropy
the design discussion predicted - an atom fitted with no neighbours really has no reason
to keep points where the bonds are. It is real. It is just not what stops the free-atom
scheme working, because the scheme runs out of equations first.

### The supports do not agree, and that is fine

The cheap SCF-free diagnostic in `ghosts.py` asks how much of the in-molecule `blocked`
selection the offline one reproduces. Not much: on ethanol at a comparable size the
ghost set contains 24-31% of what `blocked` picks for C and O and 47-64% for H. Yet it
matches `blocked`'s accuracy for 1.1-1.5x the points.

The two fits are not converging on the same answer and there is no reason they should.
§2 and §3 already established that what LS-THC wants is a point set that *spans the
co-density manifold*, not any particular set of quadrature nodes, and §5 that the
overlap RMSD - the thing both fits nominally minimise - is close to orthogonal to THC
accuracy. Support agreement is the same kind of diagnostic: it measures whether two
grids are the same grid, which is not the question. Only the point count at matched
energy is.

## 9. The ghost ensemble matters on small systems, and imposes a hard accuracy ceiling

`ensemble.py`. §8 fitted every element against one ghost ensemble - 12 icosahedral
directions, partners H/C/O at 0.90/1.05/1.45 times the covalent bond length, one
(partner, scale) combination cycled per direction, 13 environments - chosen once and
never varied. HANDOFF.md called varying it the cheapest open question in the programme.
It is. The answer is mixed, and the durable half of it is not the one the question
anticipated.

Two axes, both already implemented in `ghosts.py` and neither previously run: 12
icosahedral directions against 6 octahedral, and one cycled (partner, scale) per
direction against `--full-cross`, every combination in every direction:

| ensemble | directions | combinations | environments |
| --- | --- | --- | --- |
| `icosa-cycled` | 12 | cycled | 13 (this is §8's) |
| `icosa-full` | 12 | full cross | 109 |
| `octa-cycled` | 6 | cycled | 7 |
| `octa-full` | 6 | full cross | 55 |

Every grid at `w = 1` and `metric_ridge = 1e-8`. The `icosa-cycled` rows reproduce §8's
tables exactly on both molecules, so this is the same measurement with the ensemble
swapped underneath it.

```
methanol, floor +2.79 uHa    points: error above floor (uHa)
blocked          301:  13.6   491:   2.5   670:   1.4
icosa-cycled     223: 275.1   414:  17.0   512:  27.5   627: 6.1   702: 3.1   816: 2.9
icosa-full       318:  73.3   554:   2.6   726:   1.4   916: 0.6  1066: 0.4  1178: 0.5
octa-cycled      195: 422.2   269: 153.8   351:  80.1   408:13.2   454: 6.0   528: 4.0
octa-full        288: 133.5   414:  30.0   523:   7.5   690: 3.0   782: 1.5   816: 1.4

ethanol, floor +5.97 uHa
blocked          477:  60.7   765:   8.9  1090:   3.2
icosa-cycled     330: 761.3   620: 113.3   765:  55.5   927:10.8  1043: 4.7  1208: 5.0
icosa-full       473: 134.9   829:  18.9  1074:   4.2  1355: 1.7  1580: 1.0  1743: 1.0
octa-cycled      285:1209.6   400: 189.0   522: 164.0   606:29.6   673:13.8   783:16.0
octa-full        425: 640.5   619:  31.4   787:  23.2  1035: 3.8  1166: 2.5  1218: 3.3
```

`analyse.py`, interpolating each curve to a fixed distance above its floor, as a ratio
against the in-molecule `blocked` grid:

| target | methanol blocked | icosa-cycled | icosa-full | octa-cycled | octa-full |
| --- | --- | --- | --- | --- | --- |
| 50 uHa | 301 | 326 (1.08x) | 339 (1.13x) | 365 (1.21x) | 366 (1.22x) |
| 20 uHa | 301 | 399 (1.33x) | 395 (1.31x) | 394 (1.31x) | 443 (1.47x) |
| 10 uHa | 329 | 586 (**1.78x**) | 443 (1.35x) | 424 (**1.29x**) | 498 (1.51x) |
| 5 uHa | 402 | 647 (**1.61x**) | 497 (1.24x) | 486 (**1.21x**) | 591 (1.47x) |
| 2 uHa | 551 | n/a | 620 (1.12x) | n/a | 742 (1.35x) |

| target | ethanol blocked | icosa-cycled | icosa-full | octa-cycled | octa-full |
| --- | --- | --- | --- | --- | --- |
| 50 uHa | 500 | 774 (1.55x) | 628 (1.26x) | 579 (1.16x) | 584 (1.17x) |
| 20 uHa | 627 | 863 (1.38x) | 816 (1.30x) | 639 (1.02x) | 805 (1.28x) |
| 10 uHa | 743 | 938 (1.26x) | 925 (1.25x) | **n/a** | 895 (1.20x) |
| 5 uHa | 932 | 1034 (1.11x) | 1042 (1.12x) | **n/a** | 994 (1.07x) |

### The sensitivity is real on methanol and amortises away by ethanol

On methanol the spread across the 2x2 is 1.29x to 1.78x at 10 uHa and 1.21x to 1.61x at
5 uHa - a factor of ~1.35 between best and worst ensemble, the same magnitude as the
entire ghost gap §8 set out to measure. On ethanol, among the ensembles that reach the
target at all, it is 1.20x to 1.26x at 10 uHa and 1.07x to 1.12x at 5 uHa: a spread of
1.05x. **Ensemble sensitivity is a fixed per-atom overhead that amortises with system
size**, the same shape as §1's ratio, §8's ghost gap and §4's rotation spread.

This matters for how §8 should be read. `icosa-cycled` is the *worst* of the four on
methanol, so §8's methanol figures (1.78x at 10 uHa, 1.60x at 5 uHa) are an upper bound
and the achievable number there is 1.21-1.35x. On ethanol it is mid-pack, and §8's
ethanol figures stand as measured. The programme-level headline - 1.6-2.7x against a
global fit, improving with system size - survives, with its small-molecule end revised
down: compounding the best methanol ensemble with §1 gives 1.51 x 1.29 = **1.9x** at
10 uHa rather than 2.7x. Since the number that matters is the large-system limit, and
that end of the range is unaffected, this is a clarification rather than a correction.

### Ensembles run out of rank, and that caps the reachable accuracy

This is the durable result, and it is the one the question did not anticipate.
`octa-cycled` is the *best* ensemble on methanol at 5 and 10 uHa - and on ethanol it
cannot reach 10 uHa at any threshold. Driving the KKT threshold to zero shows why: the
support saturates far below the parent grid, at a size that is a property of the
ensemble alone (`ensemble.py --saturate`).

| element | parent | octa-cycled (7 env) | icosa-cycled (13) | octa-full (55) | icosa-full (109) |
| --- | --- | --- | --- | --- | --- |
| H | 392 | 80 | 131 | 130 | 204 |
| C | 858 | 110 | 155 | 160 | 246 |
| O | 858 | 140 | 201 | 219 | 318 |

Summing over a molecule's atoms gives a hard ceiling on its transferable grid, computable
with no SCF. `octa-cycled` allows methanol `110 + 4x80 + 140` = 570 points, which is just
enough - it reaches 5 uHa at 486. It allows ethanol `2x110 + 6x80 + 140` = 840, and
ethanol's `blocked` grid already needs 743 points for 10 uHa; a transferable grid needs
more than `blocked`, so 840 cannot get there. The measured curve tops out at 783 points
and 13.8 uHa above the floor, exactly as that arithmetic predicts.

This is §8's free-atom failure again, at a much higher ceiling and for a different
reason. The free atom ran out of **equations**: an isolated hydrogen's overlap target
supplies `n_AO(n_AO+1)/2` = 15 of them in cc-pVDZ. The ghost ensembles supply equations
in the thousands - 750 for `octa-cycled` on H, 36177 for `icosa-full` on C - far more
than the 392 or 858 parent points, so `min(n_equations, n_variables)` binds nowhere. What
they run out of is *independent* equations: Lawson-Hanson halts once the residual is
orthogonal to every remaining column, so the support saturates at the **effective rank of
the stacked target**. §8's lesson should be restated accordingly: a ghost ensemble has to
supply rank, and counting environments is not counting rank - `octa-full` has four times
the environments of `icosa-cycled` and essentially the same ceiling.

So the ensemble question is not "which corner of the 2x2 is best", which is worth ~5% by
ethanol. It is **"does this ensemble supply enough rank for the target system and
accuracy"**, which is a go/no-go and gets harder as the molecule grows. Check the
saturation table against the expected point count before fitting anything.

### The supports barely overlap between ensembles

The SCF-free diagnostic (`ensemble.py --calibrate`) puts the Jaccard overlap between each
ensemble's support and `icosa-cycled`'s at **0.01 to 0.48** across elements and
thresholds - lower, in places, than the 24-31% agreement §8 measured between the ghost
and `blocked` fits. Even `icosa-full` against `icosa-cycled`, which differ only in how
densely the same 12 directions are sampled, shares J = 0.24-0.48.

Consistent with §2, §5 and §8, this is not a defect: four largely disjoint point sets
land within 1.05x of each other on ethanol. It is one more piece of evidence that LS-THC
wants *a* spanning set rather than any particular nodes, and that support overlap remains
useless as a quality metric.

### The curves are not monotone

Adding points sometimes makes the error worse - `icosa-cycled` on methanol between 414
and 512 points (17.0 -> 27.5 uHa above floor), `octa-cycled` on ethanol between 673 and
783 (13.8 -> 16.0), and three other cases across the two molecules. A tighter KKT
threshold does not give a superset of the looser one's support: NNLS re-solves and can
drop points it previously kept, so nothing forces the error down. Re-running the
interpolation on a monotone envelope (best error at or below each size) moves methanol's
`icosa-cycled` at 10 uHa from 1.78x to 1.73x and changes nothing else, so this does not
explain the ensemble spread - but a threshold ladder is a coarse instrument, the same
warning §7 gave for the orbit-grouped fits.

---

## Verdict

**Every objection that could have killed this has now been measured, and none of them
did.** The proposed object exists: §8 builds genuine offline per-element point sets, fits
them against nothing but ghosts, translates them rigidly into methanol and ethanol, and
they land 1.1-1.8x off the in-molecule `blocked` grids at matched accuracy. Compounded
with §1, a frozen per-element grid costs **1.6-2.7x** the points of a single global
molecular fit, improving with system size - about `2.6-7.3x` on the `n_P^2` parts. That
is a real tax and it is affordable.

The two results that most changed the shape of the programme were both unanticipated.
§3: the fitted weights are almost irrelevant, so the offline object is a *point set*, the
whole active-set/reparametrisation apparatus disappears, and `X = phi(r_P)` has nothing
to differentiate. §8: the free-atom fit does not merely place points badly, it runs out
of **rank** - an isolated hydrogen's overlap matrix supplies 15 equations in cc-pVDZ and
so can never yield more than 15 points, at any threshold. A ghost is not a correction to
the free-atom fit; it is what makes the fit well posed.

The remaining structural objections went the other two ways. §6: the metric truncation's
PES steps were real, and ridge at `lambda = 1e-8` removes them. §7: the missing
orientation needed no structural fix at all - it converges away with grid size, and
whole-orbit pruning is not the way to buy it.

§9 adds a third of the same kind. The one arbitrary choice inside §8 - which ghost
ensemble to fit against - turns out to matter by 1.35x on methanol and 1.05x on ethanol,
so it amortises like everything else here; but it leaves a hard constraint behind, since
an ensemble's effective rank caps the accuracy its grids can ever reach. The cheapest
ensemble tested is the best on methanol and cannot reach 10 uHa on ethanol at all.

What remains to be measured, in order:

1. ~~**The ghost gap.**~~ **Done - see §8, and the answer is yes.** 1.1-1.8x over
   `blocked`, shrinking with system size; free-atom fits fail structurally.
2. **Transferability.** Still the leading question, with its first half now done. §9
   varied the ghost ensemble: the choice is worth ~1.35x on methanol and ~1.05x by
   ethanol, so it amortises, and §8's methanol ratios are upper bounds. What it left
   behind is a sharper constraint - each ensemble has a **rank ceiling** that caps the
   reachable accuracy regardless of threshold, and the cheapest ensemble fails ethanol
   outright on it. The untouched half is whether one element's point set serves
   environments it was not fitted in: O in water, methanol and formaldehyde; C in sp3,
   sp2 and sp. That must now be run on a deliberately chosen ensemble, since §9 showed
   the default is not a neutral one.
3. **A gradient.** Nothing in this directory computes one. Every smoothness claim here is
   a finite-difference statement about the energy curve; none of them tests an
   implementation. §8's grids are the first ones a gradient would actually be run on.
4. ~~**The isotropy tax.**~~ **Done - see §7 and §8.** Whole-orbit fitting makes each
   atomic grid exactly invariant under the octahedral group for 1.2-1.6x the points, and
   never reduces the spread under *general* rotations. §7 flagged one case as untested -
   ghost-fitted grids, which have a reason to be anisotropic that a real-neighbour fit
   does not - and §8 tested it: they *are* more orientation-dependent than `blocked` at
   matched size, but it still converges away with grid size rather than needing a
   structural fix. Keep orbit grouping opt-in.
5. ~~**Ridge vs truncation.**~~ **Done - see §6.** The truncation does put ~0.5 uHa steps
   in the PES, exactly at the eigenvalue crossings; ridge at `lambda = 1e-8` removes them
   for 0.1-2.2 uHa. Ridge has a floor of its own, and smoothness finds it before accuracy
   does.

## Caveats

cc-pVDZ only; four small molecules; MMFF geometries; `ov` mode and MP2 only. The rank and
weight analyses are on methanol and ethanol.

§8 rests on two molecules. Water was run and is not quoted for the ratio: at 24 AOs its
co-density manifold saturates at ~95 points, so every grid in the comparison is
rank-saturated and all three modes reach the floor - the ratios there are artefacts of
where each ladder happens to start, not measurements. The ghost ensemble (12 icosahedral
directions, partners H/C/O at 0.90/1.05/1.45 bond lengths, cycled one combination per
direction) was chosen once and never varied; `--full-cross` and `--directions octahedron`
exist to test that choice and were not run. N was never fitted, so the ghost grids cover
H/C/O only, and no molecule outside the fitting set was tried - which is the whole of
what transferability means and is why it is now the leading open question. The
orientation rows are 4 random draws per grid, a coarser statistic than §7's 5-6, and the
matched-accuracy comparison is matched to 0.15 uHa in energy but not in point count.

`blocked` appears in §8 with `w = 1` under ridge and in §1 and §7 with NNLS weights under
the truncated pseudoinverse, and the two do not give identical numbers - ethanol's
477-point blocked grid is +66.71 uHa here against a curve that reached 50 uHa at 477
points there. §3 predicts exactly this: the weights matter only where their dynamic range
was buying numerical rank. Ratios should be read within a table, not across them.

The parent-grid floor for methanol came out `+2.68` uHa in one run of `ghosts.py` and
`+2.86` uHa in another differing only in its threshold list. 0.18 uHa is immaterial
against the 5-50 uHa targets it anchors, but it is not nothing, and §6's ridge floor is
the obvious suspect on a 3288-point parent grid whose metric is maximally redundant.
`ridge.py` sets `mf.conv_tol = 1e-12` for related reasons; `sweep.py` and `ghosts.py` do
not. This was noticed, not diagnosed.

§7 covers two molecules over eight and four grid sizes, with 6 and 5 random rotations
each; a peak-to-peak over so few draws is a coarse statistic, and the matched-size rows
are matched only to within ~3% in point count. The octahedral-invariance result needs
none of that - it is exact by construction and merely confirmed here. The negative
result does rest on it, though the margins (3-27x in spread, at matched size, in the
same direction on both molecules) are wide relative to the noise. Nothing there tests a
ghost-fitted grid, which is the case where the conclusion might differ.

§6 measures smoothness by finite differences along one bond of one molecule per system;
no analytic gradient was computed, and nothing here checks that an implemented gradient
agrees with the curve. The scans move a terminal hydrogen only, so they probe the metric
less violently than a heavy-atom displacement would, and they are single 1D cuts - a
crossing rate of 2 per 60 steps is an estimate from a small sample. The `lambda = 1e-8`
recommendation rests on three systems in one basis, and the floor below which ridge
becomes noise depends on how null the null space really is, which is a property of the
grid and the basis.

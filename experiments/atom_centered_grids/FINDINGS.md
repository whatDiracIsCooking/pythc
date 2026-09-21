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

## 10. The frozen grid transfers: bonding it never saw costs nothing

`transfer.py`. §8 built the transferable object and §9 varied the ensemble it is fitted
against, but both measured it only on methanol and ethanol - one sp3 carbon bonded to one
sp3 oxygen, twice. Nothing in either distinguishes *"a transferable grid works"* from
*"a grid fitted against single bonds works on molecules made of single bonds"*. This is
that distinction, and it is the whole of what HANDOFF.md 4(4) had left.

One support per element is fitted **once**, frozen, and handed to every molecule below.
Each support's fingerprint is recorded in the output, and the four runs behind this
section produced **byte-identical supports** from separate processes (`C:8af372306cb1`,
`H:e7acfa9a8b2c`, `N:6c237bdd6424`, `O:2d1e3ff45e6c` at `3e-5`). The object really is one
point set, not one recipe run four times.

### What the suite is built to break

The ensemble places one ghost at `(r_cov(A) + r_cov(B)) * s` for `s` in 0.90, 1.05, 1.45,
with partners H, C, O. Two kinds of environment lie outside that by construction:

| ghost shell | tightest ghost | real bond in the suite | |
| --- | --- | --- | --- |
| C-C | 1.37 | ethane 1.51, propene 1.49 | inside |
| C-C | 1.37 | ethene / propene 1.34, acetylene 1.20 | **below** |
| C-O | 1.28 | methanol 1.42 | inside |
| C-O | 1.28 | formaldehyde 1.22 | **below** |
| C-N | *none at all* | methylamine 1.45, acetonitrile / hcn 1.16 | **never fitted** |

Multiple bonds are shorter than the shortest ghost, so they are extrapolations *below* the
training range rather than interpolations inside it - and keeping alive the tail points a
bond needs is precisely what the ghost is for. Nitrogen is not a partner at all; §8
justified that on the grounds that N's basis resembles C's and O's, which was an argument,
not a measurement.

### The answer: the ratio does not care

Points to reach a given accuracy, frozen grid against that molecule's own in-molecule
`blocked` fit. `sat` marks a cell whose ladder already began inside the target, so that it
compares two ladder start points rather than two grids - water is `sat` everywhere, as
HANDOFF.md predicted it would be at 24 AOs.

| molecule | probes | AOs | in range | 10 uHa | 5 uHa |
| --- | --- | --- | --- | --- | --- |
| water | O sp3 | 24 | yes | `sat` | `sat` |
| methanol | O sp3 + C sp3 | 48 | yes | 329/586 **1.78x** | 402/647 **1.61x** |
| ethane | C sp3, C-C 1.51 | 58 | yes | 401/484 1.20x | 451/543 1.20x |
| ethene | C sp2, C=C 1.34 | 48 | **no** | 315/482 1.53x | 352/549 1.56x |
| acetylene | C sp, C#C 1.20 | 38 | **no** | `sat` | 228/306 1.34x |
| formaldehyde | O sp2, C=O 1.22 | 38 | **no** | 233/399 1.72x | 267/447 1.68x |
| methylamine | N sp3, C-N 1.45 unseen | 53 | **no** | 409/501 1.22x | 503/553 1.10x |
| hcn | C sp + N, C#N 1.16 | 33 | **no** | 204/305 1.49x | 236/350 1.48x |
| acetonitrile | C sp + N, C#N 1.16 | 57 | **no** | 320/575 **1.80x** | 337/609 **1.81x** |
| propene | C sp2 + sp3, C=C 1.34 | 72 | **no** | 622/764 1.23x | 729/912 1.25x |

**In-range mean 1.49x, out-of-range mean 1.50x at 10 uHa; 1.41x against 1.46x at 5 uHa.**
The two groups are indistinguishable. The whole suite spans 1.20-1.80x at 10 uHa and
1.10-1.81x at 5 uHa, which is the band §8 already reported from methanol and ethanol
alone. **There is no transferability penalty to find.**

Three details sharpen that:

* **Methanol, at 1.78x, is all but the worst molecule in the suite** - and it is *in
  range*, and is the molecule §8 and §9 were measured on. Its curve here reproduces §9's
  `icosa-cycled` row at all six thresholds, so this is the same measurement with the
  molecule swapped rather than a re-tuned one. §8's headline is the top of the range, not
  a floor that unfamiliar bonding breaks through.
* **The most extreme extrapolation is among the better results.** Acetylene's C#C is
  1.20 A against a tightest C-C ghost of 1.37 - a 12% extrapolation below anything ever
  fitted - and it costs 1.34x, comfortably better than methanol.
* **The largest molecule is the best.** Propene, 72 AOs, carrying an in-range C-C at 1.49
  and an out-of-range C=C at 1.34 in the same molecule, is 1.23-1.25x. That is the
  expected direction: like §1's ratio, §8's ghost gap and §9's ensemble spread, this is a
  fixed per-atom overhead that amortises with system size.

What does *not* correlate with the ratio is how exotic the bonding is. Acetonitrile and
methanol bracket the suite at 1.80x and 1.78x, and one of them is an sp-carbon nitrile
that the ensemble never saw while the other is the molecule it was designed around.

### A partner element the fit never saw costs nothing - and supplying it makes things worse

Methylamine puts a C-N bond at 1.45 A in front of a carbon grid fitted only against H, C
and O, and comes out at 1.22x / 1.10x - **among the best in the suite**. Whatever a
nitrogen neighbour does to a carbon's co-density manifold, a grid fitted against H/C/O
spans it already.

Refitting every element with N added to the partner list tests the other direction. The
controls are what make it readable, because adding a fourth partner also changes the
sampling: both ensembles hold **13 environments**, but the baseline draws them from 10
distinct `(partner, distance)` combinations - three of the nine repeat across the twelve
icosahedral directions - and the N-partner one from 13. The equation supply is unchanged;
only the variety rises.

| molecule | N present | AOs | baseline (H,C,O) | with N as partner |
| --- | --- | --- | --- | --- |
| methanol | no (control) | 48 | 1.78x | 1.85x |
| propene | no (control) | 72 | **1.23x** | **1.48x** |
| methylamine | yes | 53 | 1.22x | 1.19x |
| acetonitrile | yes | 57 | **1.80x** | **2.33x** |
| hcn | yes | 33 | 1.49x | 1.05x |

Every properly resolved row gets worse or stays put. Both nitrogen-free controls degrade,
propene - the largest and least saturated molecule in the set - by a lot. Methylamine is
unchanged. **Acetonitrile, which contains the very C#N bond the added partner was supposed
to describe, degrades furthest of all**, from 1.80x to 2.33x.

The single apparent gain, hcn's 1.49x to 1.05x, is the one row that cannot be trusted: at
33 AOs it is the most rank-saturated molecule in the suite. Acetonitrile exists in this
table to settle exactly that, carrying the same C#N bond at 57 AOs with ladders widened
until neither curve is saturated, and it says the opposite. hcn's gain was saturation, not
nitrogen.

So the useful conclusion is not about nitrogen at all. It is that **partner variety at
fixed environment count dilutes**: the same 13 solves spread over 13 distinct neighbours
instead of 10 give a support that serves each one less well, and molecules that never
needed the extra neighbour pay for it anyway. That is a sharper form of §9's finding that
`octa-full` buys nothing with four times `icosa-cycled`'s environments. Put together:
**environments supply rank and more of them help; variety at fixed environment count is a
different knob, and turning it up costs.** Do not add a partner element to cover a
molecule - add environments.

### The overlap residual does not merely fail to track accuracy; it tracks it backwards

`transfer.py --residual` fits each atom's **real in-molecule** overlap block by
unconstrained least squares over the frozen support's points only, with no SCF. It was
included as a candidate cheap screen, and reported beside the energies so that whether it
tracks them could be checked rather than assumed. It does not:

| molecule | mean heavy-atom residual | ghost gap at 10 uHa |
| --- | --- | --- |
| hcn | 2.9e-2 | 1.49x |
| formaldehyde | 3.3e-2 | 1.72x |
| methanol | 3.3e-2 | 1.78x |
| ethene | 4.5e-2 | 1.53x |
| methylamine | 4.6e-2 | 1.22x |
| ethane | 5.2e-2 | 1.20x |

Pearson `r = -0.75`, Spearman `rho = -0.60`. The correlation is not weak, it is
**inverted**: the molecules whose real overlap blocks the frozen support reproduces worst
are the ones where it performs best. §5 found `rmsd_S` close to orthogonal to THC accuracy;
§8 and §9 found support overlap useless as a quality metric. This is the third independent
confirmation and the sharpest, because it holds the point set and the element fixed and
varies only the environment. **A cheap SCF-free transferability screen cannot be built on
the overlap target**, and the residual column survives in the output as a negative result
rather than a tool.

The mechanism is the one §2 and §3 give. What LS-THC needs is a point set spanning the
co-density manifold, with `Z` fitted by least squares against exact ERIs afterwards. How
well those same points support a *quadrature* of the overlap is a different question, and
its answer carries no information about the first.

### What this does not settle

The in-range group is thin - methanol and ethane are the only two molecules whose heavy
bonding lies inside the ghost shells and which are large enough to read, since water
saturates. §8's ethanol, fitted from the same ensemble, is 1.11-1.55x and sits in the same
band, which supports the comparison without being part of it.

The small molecules are the problem the suite was always going to have: the bonding that
most needs probing lives in small molecules, and small molecules saturate. Acetylene,
formaldehyde and hcn are 33-38 AOs, and 7 of their 30 cells are unreadable for that
reason. Propene and acetonitrile were added to answer the same questions at 57 and 72 AOs,
and they do, but the sp-carbon case rests on acetonitrile alone.

Acetonitrile is also the strongest instance of the non-monotonicity §9 flagged. Its error
runs -2.60, -3.30, -13.97, -12.61, -2.48 uHa across five adjacent thresholds at 506 to 704
points - a 15 uHa swing that crosses the 10 uHa target three times. The quoted
matched-accuracy point count is a first-crossing estimate on a curve that does not stay
crossed, so its two decimal places are not meaningful even though its magnitude is.

Everything here is one geometry per molecule, cc-pVDZ, `ov` mode, MP2, and one ghost
ensemble (plus its N-partner variant). Transferability across *basis sets* is untouched:
the offline object is a list of indices into an element's level-0 atomic grid, and nothing
here asks what happens to it in cc-pVTZ.

## 11. The gradient exists - and the ridge that the energy is happy at is far too small for it

`gradient.py`, with the machinery in `pythc.grad`. Section 6 replaced the metric
truncation with a ridge because the truncation was the last discrete step in the
pipeline, and argued from the resulting *energy* curve that the potential energy surface
was now smooth. Nothing checked that a derivative could actually be computed, or that it
was the derivative of that curve. This does both, and the answer to the first question is
yes while the answer to the second comes with a much tighter constraint on `lambda` than
section 6 imposed.

### What was implemented

A hand-written reverse-mode pass over the whole `ov`-mode pipeline, in `pythc/grad/`:
`laplace_mp2.py` differentiates the `-2J + K` Laplace expression with respect to `X`, `Z`
and the orbital energies; `linalg.py` supplies the adjoints of `ridge_inv`,
`ridge_inv_sqrt`, `pinv` and `pseudo_inv_sqrt`; `factorisation.py` carries `dE/dZ` back
through `Z = D^T D`, `D = J^(-1/2) W S^-1` and the metric to the collocation matrix;
`geometry.py` turns that into nuclear derivatives. `FrozenGrid` carries the one piece of
bookkeeping a frozen grid needs and a re-selected one cannot supply: **which atom each
point rides with**.

The collocation derivative is the whole proposal in one line,

    dX_ao[P,mu] / dR_A = w_P^(1/4) grad phi_mu(r_P) * ( [P rides with A] - [mu sits on A] )

- an AO derivative that any basis-set method has, and a rigid translation that exists
only because the grid is frozen. There is no third term. A grid re-selected per geometry
would need one, and it would not be a derivative.

What this computes is the derivative of the **correlation energy at a fixed SCF
reference**. That is the THC-specific content of the gradient; the orbital-response term
is structurally identical to DF-MP2's and is not implemented here. Keeping the split
explicit is what makes the check below exact on both sides.

### It is right

Water, 156-point blocked grid at `1e-3`, cc-pVDZ, orbitals held fixed on both sides:

| ridge | \|grad\| | max\|ana-fd\| h=1e-3 | h=1e-4 | E_corr |
| --- | --- | --- | --- | --- |
| 1e-2 | 4.9226e-02 | 1.88e-07 | **1.87e-09** | -0.196916566 |
| 1e-3 | 4.7635e-02 | 2.81e-07 | **2.92e-09** | -0.203668808 |
| 1e-4 | 4.5660e-02 | 6.00e-07 | **8.08e-09** | -0.204574645 |
| 1e-5 | 4.4539e-02 | 1.08e-06 | 7.30e-08 | -0.204716618 |
| 1e-6 | 4.3909e-02 | 1.16e-05 | 1.16e-05 | -0.204737174 |
| 1e-7 | 4.3688e-02 | 8.35e-04 | 8.35e-04 | -0.204739888 |
| 1e-8 | 1.6808e-01 | 1.07e-01 | 1.07e-01 | -0.204740292 |
| pinv | 4.3537e-02 | 1.68e-07 | 1.95e-06 | -0.204740339 |

The first three rows fall by exactly 100x when the step falls by 10x, which is what a
correct gradient does and what a gradient missing a term does not. Independently, and
with no finite difference anywhere, the forces sum to zero to **2.4e-15** of the gradient
norm - the sharpest available test of the collocation term, since under a uniform shift
its translation and AO halves are equal and opposite across the molecule and an error in
either cannot cancel. The energy itself reproduces `LS_RI_Becke` + `LaplaceRMP2` to 1.2e-11.

### The main result: the gradient's ridge floor is three decades above the energy's

Section 6 recommended `lambda = 1e-8` and warned that smoothness finds ridge's floor
before accuracy does. Put to a gradient, the same question has a much less comfortable
answer. Reading down the table: the finite difference stops converging at `1e-5`, is
losing digits by `1e-6`, is wrong in the third digit at `1e-7`, and at **`1e-8` - the
recommended value - the analytic and numerical gradients disagree by a factor of two**,
while the energies in the same rows agree to 3 nHa.

Running the *identical* calculation in three separate processes makes the point without
any finite difference at all. Same SCF to 1e-13, same 156 points:

| ridge | E_corr across runs | \|grad\| across runs | rms torque across runs |
| --- | --- | --- | --- |
| 1e-4 | agrees to 3e-13 | 4.5660213e-2, 4.5660215e-2, 4.5660210e-2 | agrees to 5 digits |
| 1e-6 | agrees to 3e-11 | 4.3921e-2, 4.3910e-2, 4.3918e-2 | 9.5e-6, 1.24e-5, 1.09e-5 |
| 1e-7 | agrees to 3e-10 | 4.368e-2, 4.466e-2, 4.421e-2 | 5.7e-4, 5.8e-4, 4.9e-4 |
| 1e-8 | agrees to 2e-9 | **1.34e-1, 2.10e-1, 1.02e-1** | **9.8e-2, 7.3e-2, 5.6e-2** |

At `lambda = 1e-8` the energy is reproducible to 0.002 uHa and the gradient varies by a
**factor of two** between runs of the same calculation. Nothing in the inputs changed;
what varies is the order floating-point reductions happen in, amplified by inverting 61
of water's 156 metric directions - the ones below 1e-14 of the largest eigenvalue - at
`1/lambda`.

So section 6's `lambda = 1e-8` is a recommendation for energies and must not be carried
over to gradient work. **The usable window here is `1e-2` to `1e-5`**, and it is bounded
on both sides: below it the null space is inverted into noise, above it the ridge costs
real accuracy (`1e-2` is 7.8 mHa from the converged energy, `1e-4` is 165 uHa, `1e-5` is
23 uHa). That is an uncomfortable place to have to sit, and it is the sharpest thing this
section has to say. Section 6 said "do not pick `lambda` from an accuracy table alone";
the correction is that its own smoothness table is not enough either, because a second
difference at 0.002 A steps over roughness that a derivative sees directly.

The `pinv` row is the control and behaves as section 6 predicts: water's blocked metric
has no eigenvalue near the cutoff at this geometry, so no crossing occurs, the
fixed-subspace derivative is the right object, and it agrees to 1.7e-7. That is not a
reprieve for the truncation - it is what "smooth between crossings" looks like when a
scan happens not to sit on one.

### Where the gradient comes from, and how little of it the freeze owns

At `lambda = 1e-4`, by norm:

| term | Ha/bohr | share |
| --- | --- | --- |
| rigid point translation | 6.54e-04 | 0.014x |
| AO derivative | 4.35e-03 | 0.095x |
| *(collocation total)* | 4.62e-03 | 0.10x |
| 3-centre integrals | 9.78e-02 | 2.14x |
| 2-centre integrals | 5.68e-02 | 1.24x |
| **total** | **4.57e-02** | 1.00x |

The gradient is dominated by the derivative integrals, which largely cancel against each
other as they do in any DF gradient. The grid contributes **10%**, and the rigid
translation - the term that exists *only* because the point set is frozen and attached -
is **1.4%** of the total. This is reassuring rather than disappointing: the LS-THC fit is
chosen to reproduce the DF ERIs, and the DF energy does not depend on the grid at all, so
the grid's contribution to the gradient is the fit residual's contribution. A frozen grid
is not injecting large spurious forces.

### The torque, which section 7 could not measure

Section 7 reported orientation dependence as an energy *spread* over random draws and
concluded it converges away with grid size. A spread is not a derivative. Rotating an
atom's point set about its own nucleus leaves the molecule, the AOs and the SCF untouched,
so `dE/dtheta` is exactly computable with no response term, and

    tau_A = sum_{P on A} s_P x (dE/dr_P)

is that derivative. Analytic against finite difference, water at `lambda = 1e-4`, each
atom rotated about its own torque axis:

| atom | analytic \|tau\| | fd d=1e-4 | fd d=1e-3 | fd d=1e-2 |
| --- | --- | --- | --- | --- |
| H0 | 1.53921135e-04 | 1.53921239e-04 | 1.53923600e-04 | 1.54407574e-04 |
| O1 | 3.91754368e-05 | 3.91763520e-05 | 3.91751850e-05 | 3.91678537e-05 |
| H2 | 1.80388533e-05 | 1.80373928e-05 | 1.80376728e-05 | 1.80113289e-05 |

Six significant figures. The quantity with physical consequences is the **net** torque:
rotating the molecule while the point sets keep their lab orientation is the same as
rotating everything - which the energy is invariant under - and then counter-rotating each
grid about its nucleus, so `sum_A tau_A` *is* the energy's response to a rigid rotation
under lab-fixed attachment, and therefore the rate at which angular momentum leaks.

    rms per-atom torque        92.3 uHa/rad
    net torque |sum_A tau_A|  187.8 uHa/rad
    nuclear gradient norm    45660  uHa/bohr
    ratio                     4.1e-3 bohr/rad
    energy spread, 6 draws    238   uHa peak-to-peak   (the section 7 statistic)

Section 7 posed the choice - lab frame gives spurious torques, a molecular frame is
itself geometry-dependent and can switch discontinuously - and could only argue about it.
The number is now 188 uHa/rad of net torque on water's blocked grid, 0.4% of the nuclear
gradient scale.

### The spread does not track the torque, and the transferable grid is where it shows

**[SUPERSEDED by section 13. This subsection generalises from one pair of water grids
under the ridge, and both halves of that are the worst available choice: water is the one
molecule whose torque is identically zero, and the ridge is the one filter that
manufactures a torque where none exists. Over a 20-rung ladder at `pinv`, `log|tau|`
tracks the rotation spread at Spearman +0.89 and the grid's own accuracy at +0.86, against
-0.54 for the point count. The spread is a serviceable proxy for the torque; what neither
tracks is the point count.]**

Running the same script on the actual proposed object - a ghost-fitted per-element
support, transferred onto water's nuclei, 208 points at threshold `3e-4` - reproduces
every verification result (quadratic convergence, 6.9e-8 -> 7.1e-10; forces summing to
zero at 4.8e-15; torque against finite difference to six figures), and then says
something the energy measurements did not:

| grid | points | energy spread, 6 draws | net torque | torque / \|grad\| |
| --- | --- | --- | --- | --- |
| blocked, 1e-3 | 156 | 238 uHa | 188 uHa/rad | 4.1e-03 bohr/rad |
| **ghost, 3e-4** | **208** | **452 uHa** | **5592 uHa/rad** | **1.35e-01 bohr/rad** |

The transferable grid has 33% more points and **1.9x** the energy spread - and **30x**
the net torque. Section 7 measured ghost grids at "2-5x more orientation-dependent than
`blocked` at matched point count, and identical at matched accuracy" and concluded that
orientation dependence is a symptom of rank limitation rather than a defect of the ghost
fit. Read through a spread that is defensible. Read through a derivative it is not: the
two grids differ by a factor of thirty in the quantity a dynamics run would actually
feel, and the spread gives no warning of it.

This is the concrete form of the general point. A spread is a peak-to-peak over draws of
a function of orientation; a torque is that function's slope at one orientation. A
small-amplitude, rapidly varying function has a small spread and a large slope, and a
ghost-fitted support - trained against twelve chosen directions and then attached at an
arbitrary orientation - is exactly the kind of thing that would be rougher in orientation
without being larger in amplitude. **Support-overlap and `rmsd_S` were already ruled out
as quality metrics (sections 5, 8, 10); the rotation spread now joins them for gradient
purposes.** Orientation quality has to be measured with `dE/dtheta`.

Whether 1.35e-1 bohr/rad is tolerable is a question about trajectory length rather than
about this table. But it is the first number attached to section 7's dichotomy, and it
points the other way from section 7's conclusion.

The collocation term also carries far more of the gradient on the transferable grid than
on the blocked one - rigid translation 9% against 1.4%, collocation total 25% against
10% - which is the same story in a second place: the frozen per-element support is a
rougher object in the geometric variables than an in-molecule fit, and the energy
comparisons of sections 8 and 10 do not see it because they are not derivatives.

### What this does not settle

One molecule, one geometry, two grids, cc-pVDZ, `ov` mode, MP2. Water is the molecule
sections 8 and 10 both decline to quote ratios for, because at 24 AOs everything
saturates - that does not affect a gradient check, which compares a derivative against
its own finite difference rather than against another grid, but it does mean the *sizes*
of the terms above should not be read as typical.

The orbital-response term is not implemented, so `de` is not the total MP2 gradient and
cannot be compared against `pyscf`'s. The rotational identity
`sum_A a_A x dE/dR_A + sum_A tau_A = 0` fails for exactly that reason - it assumes the
orbitals rotate with the molecule - and its residual (4.7e-4 relative) is a measure of
the omitted term rather than of an error. Wiring in the response would turn that identity
into a second free test, which is the main reason to do it.

The ridge window is read off one system. Section 6's `shift/maxeig` came out at
1.7e-10 to 3.3e-10 across three systems at `lambda = 1e-8`, so the window is probably
not wildly system-dependent, but three systems in one basis was already called a weak
test there and this is one system.

The blocked/ghost torque comparison is at neither matched point count nor matched
accuracy - 156 points at -0.204575 against 208 at -0.204006, both at `lambda = 1e-4`. The
30x ratio is far outside anything that mismatch could produce, and the 1.9x spread ratio
is measured on the same two grids so the *contrast* between them is internal, but a
matched-accuracy version of this table is the obvious next run and has not been done.

## 12. The gradient's ridge floor is the filter's shape, not the metric's rank - and most of section 11's torque was the regulariser

`window.py`, plus `scan.py --damped-lambdas` and `lib.damped_inv`. Section 11 left a
gradient that worked and an uncomfortable place to stand in: `lambda` had to be at least
`1e-5`, costing 23 uHa on water and more elsewhere, because below that the ridge inverts
the metric's numerically null directions at `1/lambda` and the gradient becomes noise. It
offered that mechanism as a diagnosis but did not test it, and read the window off one
system. All three gaps are closed here, and the answer to the first changes what to do
about the other two.

### A sharper instrument: permutation as a noise probe

Relabelling the points inside an atom's set is an **exact symmetry** of the energy, the
nuclear gradient and the per-atom torque - the metric's rows and columns are permuted with
them and every contraction is a sum over all of them. So anything that moves under a
permutation is floating-point noise and nothing else, and how much of it gets through is
exactly the amplification the inversion applies to the directions where that noise lives.

That is deterministic where section 11 had to run three processes and hope their
reductions landed in different orders, and it is quantitative where that comparison could
only say "these disagree". Every noise figure below is the worst relative change in the
gradient over a few permutations.

### The diagnosis is right, and it is `1/lambda^2`

Water's 156-point blocked grid, 61 of 156 directions below 1e-14 of the top eigenvalue:

| lambda | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 |
| --- | --- | --- | --- | --- | --- |
| noise in the gradient | 4.8e-08 | 5.2e-06 | 5.8e-04 | 4.8e-02 | 9.8e-01 |

That is a factor of ~80 per decade of `lambda`, i.e. `lambda^-1.9`. **Essentially
`1/lambda^2`, which is what "`Z = D^T D` carries `S^-1` twice" predicts and what no other
explanation would.** At `1e-8` the gradient is 98% noise while the energy in the same row
is right to 0.05 uHa - section 11's asymmetry, now with a mechanism attached rather than
a conjecture.

### It is the filter's shape, and `ridge_inv_eigh` is what proves it

`damped_inv` changes two things at once: the filter applied to the spectrum, and the
algorithm (`eigh` where `ridge_inv` uses a Cholesky factorisation). Those differ by 7.8e-8
at `lambda = 1e-8` on a matrix conditioned like this one, which is not obviously
negligible against the effect being chased. `ridge_inv_eigh` applies the ridge's own
filter spectrally, and is in the comparison for that reason alone:

| scheme | water 1e-6 | water 1e-8 | methanol 1e-6 | methanol 1e-8 |
| --- | --- | --- | --- | --- |
| ridge (Cholesky) | 5.8e-04 | 9.8e-01 | 3.3e-04 | 8.7e-02 |
| ridge (eigh) | 6.2e-04 | 6.7e-01 | 1.8e-04 | 7.3e-02 |
| **damped** | **1.5e-08** | **6.0e-07** | **3.1e-09** | **1.4e-07** |

The two ridges track each other to within a factor of two everywhere. **The algorithm is
not what matters; the filter shape is.** Writing the gain applied to an eigenvalue
`sigma`:

| scheme | gain | `sigma >> mu` | `sigma -> 0` |
| --- | --- | --- | --- |
| `pinv` | `1/sigma` or `0` | `1/sigma` | `0` |
| `ridge_inv` | `1/(sigma + mu)` | `1/sigma` | **`1/mu`** |
| `damped_inv` | `sigma/(sigma^2 + mu^2)` | `1/sigma` | `sigma/mu^2` |

The ridge hands the null directions the *largest* gain in the whole operator. That is the
opposite of suppression, and it is the entire problem. The damped filter is analytic in
`S` exactly as the ridge is - a rational matrix function, no cutoff for an eigenvalue to
cross - while sending those directions to zero the way the truncation does. Its peak gain
is `1/(2 mu)` at `sigma = mu`, the same as the ridge's, so `lambda` means the same thing
on both ladders.

### The window, on five grids

A `lambda` is *usable* if permutation noise in the gradient is below 1e-6 of `|grad|`;
among those, the best is the one closest to the truncation's energy. That pair is the
whole trade.

| grid | points | null dirs | ridge: best usable | damped: best usable | gain |
| --- | --- | --- | --- | --- | --- |
| water blocked | 156 | 61 | `1e-4`, **166 uHa** | `1e-8`, **0.00 uHa** | >1000x |
| water ghost | 208 | 113 | `1e-4`, **734 uHa** | `1e-9`, **0.00 uHa** | >1000x |
| methanol blocked | 301 | 0 | `1e-4`, **280 uHa** | `1e-8`, **1.77 uHa** | 158x |
| methanol ghost | 414 | 63 | `1e-4`, **1808 uHa** | `1e-8`, **4.32 uHa** | 419x |
| ethanol blocked | 477 | 0 | `1e-4`, **415 uHa** | `1e-9`, **0.21 uHa** | ~1400x |

Three things to read off it.

* **The ridge is stuck at `lambda = 1e-4` on every grid tested**, and what that costs
  grows with the system: 166 -> 280 -> 415 uHa across water, methanol and ethanol. The
  ceiling falls as the floor stays put, which is the closing-window worry, confirmed for
  the ridge.
* **The damped filter removes it.** Its usable `lambda` runs three to five decades lower
  and the penalty is microhartrees or less on every grid. The window does not close.
* **The null-direction count is not the whole story.** Methanol and ethanol blocked have
  *no* directions below 1e-12 and the ridge still fails at `1e-6`, because their smallest
  eigenvalues (2.0e-11, 1.3e-12 relative) are themselves below the shift at small
  `lambda`. "Numerically null" is relative to `mu`, not absolute - which is why the
  effect survives on grids that `rank.py` would call healthy.

### It stays smooth, which it had to

The damped filter approaches the truncation as `lambda` falls, and the truncation is what
put the steps in the PES that section 6 introduced the ridge to remove. A quiet gradient
bought by reintroducing them would be no use. Methanol's 60-step O-H scan, grid frozen:

| scheme | max\|d2\| | median\|d2\| | ratio | gradient usable? |
| --- | --- | --- | --- | --- |
| pinv | **0.511** | 0.006 | 82 | n/a - no derivative at a crossing |
| ridge 1e-8 | 0.017 | 0.002 | 8.0 | **no** - 81% noise |
| ridge 1e-4 | 0.006 | 0.005 | 1.4 | yes, at 280 uHa |
| damped 1e-8 | 0.024 | 0.005 | 5.0 | yes, at 1.8 uHa |
| **damped 1e-10** | **0.013** | 0.004 | **3.1** | **yes** |

Section 6 reproduced exactly - 2 crossings in 60 steps, the largest second differences
sitting on them, 82x the background. **Damped at `1e-10` is the smoothest curve in the
table, at a `lambda` where the ridge's gradient is pure noise.** The two requirements are
now satisfiable at the same time, which they were not before.

### Most of section 11's torque was the regulariser

This is the result that changes a conclusion rather than an implementation detail. Both
schemes reduce to the truncated pseudoinverse as `lambda` goes to zero, so **`pinv` is the
torque they should converge on** - a control section 11 did not have, because it could not
go to small `lambda` at all.

| grid | pinv | ridge at its best usable `lambda` | damped at its best |
| --- | --- | --- | --- |
| water blocked | **0.001** | 188 (section 11's number) | 0.018 |
| water ghost | **0.002** | **5592** (section 11's number) | 0.006 |
| methanol blocked | **39.0** | 234 | 45.7 |
| methanol ghost | **175.1** | 1596 | 109.5 |
| ethanol blocked | **103.1** | 202 | 103.4 |

in uHa/rad. The harness reproduces section 11's 188 and 5592 exactly at `lambda = 1e-4`,
so this is not a different measurement of the same thing - it is the same measurement,
continued to where the regulariser stops contributing.

* **Water's torque is zero, and section 11 should not have been read.** At 24 AOs water is
  rank-saturated - co-density rank 95 against grids of 156 and 208 points - so the LS-THC
  fit is exact, therefore grid-independent, therefore orientation-independent. Sections 8
  and 10 both decline to quote water for exactly this reason; section 11 quoted it for the
  torque, which is precisely the kind of quantity that vanishes at saturation. **The
  30x blocked-to-ghost ratio was two regularisation artefacts divided by each other.**
* **The effect is real on a molecule that does not saturate, and much smaller.** Methanol
  is 175 vs 39, a **4.5x** ghost-to-blocked ratio rather than 30x. The qualitative
  conclusion - a transferable per-element support is more orientation-dependent than an
  in-molecule fit - survives; the magnitude does not.
* **The number that matters for dynamics falls by a factor of 40.** Methanol's ghost grid
  at its best usable `lambda` has a net torque of 109.5 uHa/rad against a gradient norm of
  35636 uHa/bohr, i.e. **3.1e-3 bohr/rad**, against the **1.35e-1** section 11 reports
  from water's ghost grid. The angular-momentum leak is a few tenths of a percent of the
  gradient scale, not thirteen.
* **A torque is not a property of the grid alone.** It moves by two to three orders of
  magnitude with the inversion scheme and its strength. Section 11 added the rotation
  spread to the list of metrics that do not measure what they are used for; the torque
  belongs on a different list, of quantities that must be quoted **with the regularisation
  that produced them** or not at all.

### The accuracy tax was never the problem it looked like

Worth recording because it cuts against the framing of section 11 and of this section's
own opening. The ridge's 280 uHa penalty on methanol at `lambda = 1e-4` sounds
disqualifying beside a 5 uHa THC error. Along the scan it varies by **2.28 uHa, 0.8% of
itself** - a near-constant offset, which does nothing whatever to a trajectory. Converting
the curvature of each error curve into a force-constant bias on the O-H stretch
(`k = 1.76 Ha/A^2` at 3700 cm^-1):

| scheme | curvature bias / uHa A^-2 | shift in nu / cm^-1 |
| --- | --- | --- |
| pinv (the grid's own) | 1393 | 1.47 |
| ridge 1e-4 | 934 | 0.98 |
| damped 1e-8 | 1555 | 1.64 |
| damped 1e-10 | 546 | 0.58 |

**Every scheme is inside 1.7 cm^-1, and so is the frozen grid itself with no
regularisation at all.** The differences between the rows are at the edge of what a curve
of 1-3 uHa amplitude over 0.12 A resolves and should not be ranked. So the case against
the ridge rests entirely on the gradient noise, not on the energy it costs - and the
reason to prefer the damped filter is that it makes the derivative real, not that it makes
the energy better.

### What this does not settle

cc-pVDZ, `ov` mode, MP2, fixed-orbital gradient, one geometry per molecule, as everywhere
else here. The torque is one orientation per grid, not a sampling. `damped_inv` costs a
full eigendecomposition where `ridge_inv` costs a Cholesky - about 3x on that step, which
is not the pipeline's bottleneck but is not free either, and no attempt was made to reach
the same filter through a cheaper iteration.

The ghost grids' torque does not converge to `pinv` the way the blocked ones do (methanol
ghost: 175 at `pinv`, 109 at `1e-8`, 23 at `1e-9`, 7 at `1e-10`). That is not
inconsistency - `pinv` discards everything below 1e-10 of the top eigenvalue while the
damped filter at `1e-10` still resolves directions down to ~1e-12, so they are genuinely
different operators in the limit - but it does mean a ghost grid's torque has no single
well-defined value, only a value per inversion. The blocked grids, whose metrics have no
such crowd, agree to within noise (ethanol: 103.1 against 103.4).

Nothing here integrates a trajectory, which is still the measurement that would settle
what any of these torques do over time.

## 13. The ladder under the right filter, and a trajectory: the leak is real, bounded in the part that matters, and smaller than the grid it is pruned from

`torque_ladder.py --scheme damped --pinv`, `trajectory.py`. Section 12 ended by
naming two measurements. The first was a ladder: it had continued the torque to small
`lambda` on one grid per mode and could not say whether what was left converges with grid
size. The second was a trajectory - "nothing here integrates a trajectory, which is still
the measurement that would settle what any of these torques do over time". Both are done
here, and a third measurement that neither section asked for turns out to reframe both.

### The committed ladder measured the regulariser

The ladder in `data/torque_ladder_{methanol,water}.json` was run at `ridge` `1e-3` and
`1e-4`. Section 12's whole point is that a torque at that `lambda` is mostly the
regulariser, and the README says in terms that the ladder "must be run with
`--scheme damped`". It was not: `--scheme` entered `torque_ladder.py` in the merge that
*followed* those files, so the data predates the option. The reading taken from them -
that the in-molecule grid converges and the transferable one does not - is therefore a
statement about `(S + lambda I)^-1`, not about either grid.

`--pinv` is new here and is what makes the re-run interpretable. Both filters reduce to
the truncated pseudoinverse as `lambda` falls, so `pinv` is the torque they converge on;
`window.py` already carries that control for one grid per mode, and without it on a
ladder there is no way to attribute a trend down the rungs to the grid rather than to the
filter.

### Every mode converges

Methanol, `pinv`, net torque in uHa/rad:

| mode | 223-235 | 301 | 410-414 | 491-512 | 627 | 670-702 | factor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `blocked` (NNLS weights) | 357.4 | 39.0 | 0.90 | 0.12 | - | **0.02** | 17667x |
| `blocked1` (`w = 1`) | 357.4 | 121.3 | 63.8 | 3.15 | - | **9.02** | 39.6x |
| `ghost` (`w = 1`, transferable) | 968.6 | - | 175.1 | 792.8 | 79.4 | **27.2** | 35.6x |
| `ghostw` (ghost-fitted weights) | 698.2 | - | 42.5 | 143.1 | 1.4 | **52.3** | 13.4x |

**The torque converges away with grid size, in every mode, and section 12's open question
closes the other way from the ridge ladder's reading.** `blocked` does it cleanly and
monotonically over four orders of magnitude. The other three do it noisily and over one
to two.

Water does it before the ladder starts: 0.00 uHa/rad on all five rungs of all four modes
*and* on the unpruned 1642-point parent, against ridge values reaching 28993. At 24 AOs
its co-density manifold saturates, the fit is exact, and an exact fit cannot depend on the
grid. Sections 8 and 10 both decline to quote water; section 11 quoted it for the torque
and should not have.

### The gap is the weight footing, not the transfer

The ghost grid does not converge like an in-molecule fit. It converges like an in-molecule
fit **at its own weight footing**: 35.6x against `blocked1`'s 39.6x, where weighted
`blocked` manages 17667x. A transferable support can carry `w = 1` or the weights its
ghost fit produced and nothing else, so it inherits `blocked1`'s convergence, and that -
not the transfer - is what the orientation dependence costs.

At the top rung all four modes land within 2.7-3.6 uHa of the reference, so this is a
matched-accuracy comparison. The robust statistic (rms over random per-atom orientations)
reads 0.13, 9.28, 22.01, 9.22 uHa/rad in the table's order. The transferable object is
roughly a hundred times noisier in orientation than the weighted in-molecule fit and
within a factor of two of the same fit stripped of its weights.

This is the third time the weights have turned out to be the live variable after section 3
wrote them off, and the second time the correction has been in the same direction: a
positive diagonal rescaling is absorbed exactly by the *pseudoinverse*, and nothing in
this programme is evaluated at the pseudoinverse except these control rows.

### The rotation spread does track the torque

Section 11 put the spread on the list of metrics that do not measure what they are used
for, on the strength of one pair of water grids: 1.9x the spread, 30x the torque. Over all
20 `pinv` rungs here, `log|tau|` tracks

| against | Pearson | Spearman |
| --- | --- | --- |
| the rotation spread | +0.88 | **+0.89** |
| the grid's own `err_uHa` | +0.67 | **+0.86** |
| the point count | -0.55 | -0.54 |

**The torque tracks how good the grid is, not how big it is**, and the spread is a
perfectly serviceable proxy for it. Section 11's counterexample was two grids on the one
molecule whose torque is identically zero, under the one filter that manufactures one.
That also explains the non-monotonicity above: the ghost ladder's torque spikes at exactly
the rung where its *accuracy* spikes (512 points, 20.8 uHa), which is the coarse,
non-monotone threshold ladder section 9 already documented, not a defect of the torque.

### What a trajectory has to be propagated on, and why it is not the THC surface

`pythc.grad` returns the correlation gradient at a fixed SCF reference. That force is not
the gradient of `E_HF + E_corr^THC`, and the first attempt to propagate on it says what the
inconsistency costs: methanol on the 414-point ghost grid reaches **32.4 hbar** of angular
momentum in 0.1 ps with a 3826 uHa energy drift, while the grid's own torque integrates to
0.15 over the same window.

That is not the frozen grid, and the control proves it. The same coupled dynamics on a
670-point `blocked` grid - whose own torque along that run is **0.05 uHa/rad** rms against
the ghost grid's 242, some five thousand times smaller - gains **8.63 hbar in 25 fs**,
where the ghost grid gains 8.41. Both
correspond to about **8300 uHa/rad** of spurious torque. **The omitted orbital response
breaks rotational invariance roughly fifty times harder than the transferable grid does**,
and it does so identically whatever grid it is paired with. Section 11 prices the response
at 4.7e-4 relative in the rotational identity and calls it standard machinery left undone;
for dynamics it is not a refinement but a prerequisite, and it is the larger of the two
symmetry violations in the pipeline by a wide margin.

So the trajectory is driven by the DF-RHF gradient instead, and the frozen grid's
`tau_net` is evaluated along it and integrated: `dL = int tau dt`. Density-fitted
Hartree-Fock has no quadrature grid at all, so that surface is exactly rotationally
invariant as well as exactly consistent - methanol's 1.5 ps run conserves `L` to
**2.3e-5 hbar** from an initial zero - and a leak measured along it is the frozen grid's
and nothing else's. The leaked angular momentum is fed back as a real rotation of the
molecule against the lab-fixed grids (`FrozenGrid` already takes per-atom rotations, and
turning every grid by `Q^-1` about its nucleus is the same configuration as turning the
molecule by `Q`), so the torque is evaluated where the leak has actually put the molecule
rather than where it started.

### The leak, decomposed

The decomposition is the result. A leak **along** `L` changes how fast the molecule spins
and therefore does work, so the orientational potential's own amplitude - section 7's
rotation spread, tens of uHa - bounds it. A leak **across** `L` does no work at leading
order, so nothing bounds it a priori; it tilts the rotation axis instead. Only the first
could heat a trajectory, and a bare `|leak|` conflates them.

Methanol at 300 K carrying thermal rotation, `|L| = 18.24 hbar`, 2.5 ps - between one and
three turns about each principal axis, so every orientation the molecule visits has been
swept:

| grid | points | coherence | leak | along `L` | across `L` | axis tilt |
| --- | --- | --- | --- | --- | --- | --- |
| `blocked` | 301 | 0.04 | 0.316 | -0.191 | 0.252 | 0.79 deg |
| `ghost` | 414 | 0.05 | 1.487 | +0.357 | 1.444 | **4.53 deg** |

in hbar. **Coherence - the norm of the torque's time average over the rms of its
instantaneous norm - is 0.05.** The torque averages to five percent of itself over a
tumble, which is the quantitative form of the claim that a conservative orientational
potential cannot drive a molecule that turns through it. The component along `L` sits at
+0.36 hbar, a rotational energy of tens of uHa against that grid's 31 uHa rotation
spread, which is where energy conservation says it has to be.

What is left should not be quoted as bounded. Across the run the leak goes
0 -> 1.16 (0.9 ps) -> 1.02 (1.5 ps) -> 1.49 (2.5 ps): it rises to a plateau and then
wanders on it, and the straight-line slope of 0.42 hbar/ps is dominated by the initial
rise rather than by anything secular. **The failure mode is slow artificial reorientation
of the rotation axis - 4.5 degrees in 2.5 ps - not heating and not spin-up.**

Water is the null control and behaves like one: rank-saturated, torque 0.00 on every rung
of the ladder, and a leak of **0.0027 hbar over 2 ps**, 0.03 degrees of tilt, against
methanol's 1.487. The leak tracks the torque, which is the one thing a trajectory had to
demonstrate before any number it produced could be attributed to the freeze.

The companion arm, `--project-rotation`, removes `L` from the initial velocities so that
`L(0) = 0` exactly and there is nothing to subtract. It is the worst case by construction:
a molecule that is not turning never lets the torque reverse. The ghost grid leaks
1.787 hbar over 1.5 ps there and spins the molecule through 116 degrees. It should be read
as an upper bound rather than as a trajectory - it is the one arm whose rotational kinetic
energy has no legitimate source, since the RHF trajectory generating its geometries does
not pay for it, and by the end that energy is about twice what the orientational potential
can supply.

### Against the quadrature it is pruned from

None of the above says whether the defect is acTHC's. A Becke grid is Lebedev shells
generated in **lab axes** and translated onto nuclei; rotate the molecule and the points
move relative to it. That is exactly what a frozen per-element support does, and it is
what every atom-centred-grid method does, DFT included. The trajectories above do not show
it only because their reference is density-fitted Hartree-Fock, which has no grid.

`trajectory.py --reference rks` puts an ordinary grid back in. Same molecule, same initial
condition, same level-0 Becke grid every THC fit in this directory is pruned from,
`grid_response=True` so the Becke weight derivatives are in the force - they are 12% of
its norm, and without them this would measure PySCF's default rather than what DFT can do
- and the energy conserved to tens of uHa:

`|L - L0|` in hbar, methanol, same initial condition throughout. **Quoted at matched
times, which matters more than it sounds: these surfaces do not leak at the same rate, so
a number read off one window says nothing against a number read off another.**

| surface | grid | points | 100 fs | 200 fs | 300 fs | 1000 fs |
| --- | --- | --- | --- | --- | --- | --- |
| RKS/PBE | level-0 Becke | 4656 | 0.638 | 2.276 | 2.380 | **9.250** |
| frozen `ghost` THC | pruned from level 0 | 414 | 0.144 | 0.427 | 0.686 | **1.188** |
| frozen `blocked` THC | in-molecule, weighted | 301 | 0.030 | 0.052 | 0.065 | **0.106** |
| RKS/PBE | level-3 Becke | 67432 | 0.004 | 0.020 | 0.028 | - |
| DF-RHF | none | - | \~0 | \~0 | \~0 | 2.3e-5 |

The level-0 row is not an integrator artefact: its energy drifts 21.7 uHa over the whole
picosecond while `|L|` swings 18.24 -> 10.07 -> 15.61, so the force is consistent with the
energy and the angular momentum is going into the grid.

Three things follow.

**Against the quadrature it is actually built from, the frozen support wins, by about
eightfold at a picosecond.** The frozen per-element support is not merely no worse than
its parent grid - it is quieter than it. The ladder said so first and more directly: the
complete 3284-point parent grid has a torque of 0.24 uHa/rad and the *pruned* 670-point
weighted grid has 0.02. Pruning is a benefit, not a cost, because the NNLS reweighting
makes the pruned grid a better quadrature than its parent - the property the README
already advertises for the overlap matrix, which turns out to buy orientation quality too.

**Against a production DFT grid it loses, by a factor of about 25.** At 300 fs level 3 is
0.028 against the transferable support's 0.686. An earlier reading of this table put the
gap at two to three *decades* by comparing level 3 at 38 fs against the support at 2.5 ps
- a factor of 66 in the window, on surfaces that do not leak at the same rate, so the
comparison was void. A factor of 25 on 163x fewer points is what compression costs, not a
defect of the freeze.

**And the in-molecule grid is close to production-DFT quality.** `blocked` at 301 points
gives 0.065 against level 3's 0.028 at 300 fs - **within a factor of 2.3, on 224 times
fewer points** - and it beats the level-0 parent it was pruned from by 37x. `blocked` and
`ghost` differ only in the weight footing and the transfer, and the ladder says the
footing is nearly all of it. So the factor of 25 separating the transferable support from
production DFT is not a property of frozen grids, of pruning, or of translating a support
between molecules. **It is the weights**, and closing it is an offline fitting problem
rather than a structural one - which is what makes section 4(8) of HANDOFF the most
valuable unrun experiment in the programme.

### What this does not settle

cc-pVDZ, `ov` mode, MP2, fixed-orbital, methanol and water, one initial condition per arm,
gas phase, 1.5-2.5 ps. The saturation argument is physics and should generalise, but the
*size* of the plateau on a larger molecule with smaller rotational constants - where
tumbling is slower and the torque has longer to act coherently - is untested, and that is
the obvious next run.

The DFT comparison is against acTHC's own parent grid, which is the right control for "is
this defect acTHC's fault" and the wrong one for "is this defect tolerable". Level 0 is
very coarse by production standards.

The leak is integrated along a trajectory the frozen grid does not itself generate. The
feedback closes the loop that matters - the torque is evaluated at the orientation the leak
has produced - but the internal coordinates are those of the RHF trajectory, so the
treatment is first order in the leak. The tumbling arm turns the molecule through 133
degrees of spurious rotation by the end, which is not small, and a fully coupled run would
need the orbital response first.

`trajectory.py` carries several grids along one trajectory so that the comparison between
them has no sampling difference in it, but only the grid named by `--feedback` gets its
own orientation; the others are measured at that grid's. On the tumbling arm that is a
few degrees of difference and the `blocked` row should be read with it in mind.

## 14. A richer purely-atomic target: the free atom's ERIs lift the ceiling that forced ghosts - and carry weights that transfer

`atomic_eri.py`, `pythc.decomp.nnls.ERIFitOperator`. Section 8 killed the free-atom fit
and named the mechanism: **an equation count**. Lawson-Hanson can never retain more
variables than the problem has equations, an isolated atom's overlap target supplies
`n_AO (n_AO + 1) / 2` of them, and for hydrogen in cc-pVDZ that is 15 - confirmed exactly,
at threshold `1e-12`, in §8. Ghosts exist to lift it: stacking 13 environments into one
solve takes the solver from 105 equations to ~3500, and §8 is explicit that "a ghost is
not a correction to the free-atom fit; it is what makes the fit well posed."

That diagnosis has a consequence §8 did not draw. If a *count* is what breaks the free
atom, anything that supplies more equations about the same free atom would serve, and a
ghost ensemble is not the only source. §9 sharpened the case from the other side: the
ensemble has to be chosen, the choice is worth 1.35x on methanol, and it carries a rank
ceiling of its own that `octa-cycled` fails ethanol on. §10 added that raising the
*variety* of environments at fixed count actively costs - a fourth partner element takes
propene from 1.23x to 1.48x. All of that is overhead of having a training set at all.

The Gaussian-basis literature contains both moves, and the mixture is the lesson. ANO-RCC
averages density matrices over the neutral atom, the cation, the anion and the atom in an
electric field - environments, essentially, with the field term there precisely because
free-atom natural orbitals are too contracted to use in a molecule. But cc-pVXZ uses no
molecules anywhere: its `sp` exponents come from atomic HF and its correlating functions
from the *atomic correlation energy*, and that works because correlation energy is a far
richer observable of the same free atom than the HF energy is. So the binding constraint
need not be molecularity. It can be the information content of the target - and the grid
analogue of "atomic HF energy" is exactly the free-atom overlap matrix whose ceiling §8
measured.

### The target, and why it is still an NNLS problem

The two-electron integrals are a *linear* functional of the quadrature weights. From

    (mu nu | lambda sigma) = integral dr phi_mu(r) phi_nu(r) V_{lambda sigma}(r),
    V_{lambda sigma}(r)   = integral dr' phi_lambda(r') phi_sigma(r') / |r - r'| ,

a quadrature reproduces every ERI exactly when

    (mu nu | lambda sigma) = sum_P w_P phi_mu(r_P) phi_nu(r_P) V_{lambda sigma}(r_P),

which is linear in `w`, so the same Lawson-Hanson solver fits it unmodified.
`V_{lambda sigma}` on the grid points is PySCF's `int1e_grids`, a one-electron integral
costing `O(n_grid n_AO^2)`. What changes is the row count:
`[n_AO (n_AO + 1) / 2]^2` instead of `n_AO (n_AO + 1) / 2`, i.e. `O(n_AO^4)` against
`O(n_AO^2)`. Hydrogen goes from 15 equations to 225 and oxygen from 105 to 11025.

Nothing else moves. The same level-0 per-element parent grid, the same solver, the same
`w = 1` and `metric_ridge = 1e-8` evaluation as §8, so the offline object is the same kind
of thing - a list of indices into an element's atomic grid - and the comparison isolates
the target.

`ERIFitOperator` never materialises the fitting matrix: column `P` is the outer product of
the packed co-density and the packed potential at `r_P`, and the gradient contracts the
residual against both without the pair-pair index appearing. It also carries an **exact**
rank reduction. Every column lies in `range(Rho^T) (x) range(Vp^T)`, so projecting both
indices onto orthonormal bases of those ranges cannot move the minimiser - what it
discards is a constant of the fit - and it takes oxygen's rows from 11025 to 8464. The
13 directions it removes are not rounding: they are co-density pairs the 858-point level-0
parent does not resolve at all, and the same 13 turn up independently as the gap between
carbon's overlap equation count (105) and its overlap saturation (92).

### The ceiling is lifted five- to six-fold

`atomic_eri.py --saturate`, no SCF, no molecule. Driving the KKT threshold to `1e-14`
retains every point the fit can use, so what comes back is the effective rank of the
fitting matrix - the ceiling the scheme meets however it is tuned.

| element | parent | n_AO | overlap equations | overlap saturation | ERI equations | ERI saturation | ERI residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H | 392 | 5 | 15 | **15** | 225 | **97** | 3.1e-08 |
| C | 858 | 14 | 105 | **92** | 8464 | **463** | 9.4e-07 |
| N | 858 | 14 | 105 | **92** | 8464 | **467** | 4.8e-07 |
| O | 858 | 14 | 105 | **92** | 8464 | **423** | 1.3e-06 |

Hydrogen's overlap fit stops at exactly its equation count, reproducing §8 to the point.
The heavies stop *below* theirs, at 92 of 105, and that 92 is the parent grid's own
resolving power rather than the target's - so on heavy elements the overlap fit was
already limited by two things at once. Either way the ERI target clears the old ceiling by
**6.5x (H), 5.0x (C), 5.1x (N), 4.6x (O)**, and none of those is an equation-count bound
any more: 463 retained out of 8464 equations is not a ceiling being hit, it is a fit
converging - to a relative residual of `1e-6` to `1e-8` on the whole four-index array.

**The mechanism claim also checks out.** The worry the ghosts were built to answer is that
an isolated atom has no amplitude where a bond would be, so the fit throws the tail away.
A co-density decays exponentially, but `V_{lambda sigma}` decays as `1/r`, so an ERI
target is still sensitive out where a neighbour would sit. Measured (Angstrom from the
nucleus, saturated supports):

| element | fit | r_mean | r_max | beyond 1 r_cov | beyond 2 r_cov |
| --- | --- | --- | --- | --- | --- |
| H | overlap | 0.681 | 1.452 | 93% | 40% |
| H | ERI | 0.711 | **2.232** | 78% | 36% |
| C | overlap | 0.828 | 2.224 | 50% | 16% |
| C | ERI | **0.979** | **2.987** | 56% | **26%** |
| O | overlap | 0.593 | 1.819 | 41% | 13% |
| O | ERI | **0.673** | 1.819 | 49% | 18% |

The ERI support reaches further out on every element and puts a larger fraction of its
points beyond twice the covalent radius. This is evidence about the *mechanism*, not a
quality score - §5, §8 and §10 all warn against reading supports - and it is here because
it is the one prediction the target's `1/r` tail makes that can be checked without an SCF.

### What it costs, at matched accuracy and matched footing

`atomic_eri.py methanol|ethanol`, read through `analyse.py`. Five modes, all on the same
level-0 parent and the same `metric_ridge = 1e-8`: `blocked` and `blockedw` are the
in-molecule per-atom fits with the NNLS weights discarded and kept respectively, `ghost`
is §8's transferable object, and `eri` / `eriw` are the free-atom ERI support with weights
discarded and kept.

Point count at matched MP2 error, as a ratio to the in-molecule fit **on the same weight
footing** - which matters, because §13 measured the footing alone at 450x on the torque
and a ratio read across it would not be a comparison:

| | 50 uHa | 20 uHa | 10 uHa | 5 uHa |
| --- | --- | --- | --- | --- |
| methanol `ghost` / `blocked` (both `w = 1`) | 1.08x | 1.33x | 1.78x | 1.61x |
| methanol `eri` / `blocked` (both `w = 1`) | **0.89x** | **1.08x** | **1.10x** | n/a |
| methanol `eriw` / `blockedw` (both weighted) | **0.80x** | **0.93x** | **1.03x** | **1.07x** |
| ethanol `ghost` / `blocked` (both `w = 1`) | 1.55x | 1.38x | 1.26x | 1.11x |
| ethanol `eri` / `blocked` (both `w = 1`) | **1.04x** | **1.16x** | n/a | n/a |
| ethanol `eriw` / `blockedw` (both weighted) | **1.04x** | **1.11x** | **1.17x** | **1.22x** |

**A free-atom fit with no neighbour of any kind costs 0.89-1.16x the in-molecule lower
bound**, where thirteen fabricated environments cost 1.11-1.78x. The object §8 called a
strawman, refitted against a richer observable of the same isolated atom, is better than
the ghost ensemble everywhere the two are both readable - and on methanol it is *cheaper
than the in-molecule fit it is measured against*, which is not a contradiction: `blocked`
is a lower bound on the *support quality* a transferable scheme can select, not on the
point count, and the two fits allocate their budget differently.

**That reallocation is where the saving is.** At matched total size the ERI fit gives
hydrogen far fewer points than the overlap fit does and the heavies more - methanol at
~700 points: `ghost` is C 125 / H 108 / O 145, `eri` at 565 is C 182 / H 45 / O 203. A
ghost fit sees hydrogen's target inflated by a heavy partner's basis functions and spends
accordingly; an ERI target says plainly that a hydrogen contributes little to the
two-electron integrals. Methanol has four hydrogens, so that is ~250 points.

### The torque: an ERI-fitted weight set is the first transferable one that helps

This is the result §13 asked for and did not expect from here. §13 localised the entire
remaining orientation gap to the **weight footing**: over a `pinv` ladder on methanol,
`ghost` converges like `blocked1` (35.6x against 39.6x) and not like weighted `blocked`
(17667x), and `ghostw` - the ghost fit's own weights, transferred - is *worse* than
`w = 1`, 52.3 uHa/rad against 9.0, because those weights condition the ghost metric rather
than the in-molecule one. Its conclusion was that "what a transferable support cannot do
is carry in-molecule weights", and §4(8) of HANDOFF.md made fitting a per-element weight
set for the in-molecule objective the highest-leverage thing left.

`torque_ladder.py methanol --modes eri,eriw --scheme damped --ridges 1e-8,1e-10 --pinv
--draws 4`, at the `pinv` control, against the committed ladder on the same molecule, the
same seed and the same four draws:

| mode | footing | transferable | net tau, top rung | rms draw tau | spread/uHa | net tau converged |
| --- | --- | --- | --- | --- | --- | --- |
| `blocked` | NNLS weights | no | 0.0 (670 pts) | 0.1 | 0.0 | 17667x |
| `blocked1` | `w = 1` | no | 9.0 (670 pts) | 9.3 | 0.4 | 39.6x |
| `ghost` | `w = 1` | yes | 27.2 (702 pts) | 22.0 | 0.4 | 35.6x |
| `ghostw` | ghost weights | yes | 52.3 (702 pts) | 9.2 | 0.7 | 13.4x |
| `eri` | `w = 1` | yes | 13.4 (565 pts) | 12.6 | 3.4 | 35.4x |
| **`eriw`** | **ERI weights** | **yes** | **0.19 (565 pts)** | **0.17** | **0.053** | **1481x** |

`eriw` converges like weighted `blocked` - 1481x on the net torque and 2384x on the rms
over draws, against `blocked`'s 17667x and 2606x - and not like anything else in the
table, all of which sit between 13x and 40x. At the top rung it is **275x quieter than
`ghostw` on 137 fewer points**, and 0.19 uHa/rad is *below* the 0.24 uHa/rad §13 measured
for the complete unpruned 3284-point parent grid.

So the sentence §13 left standing - a transferable support cannot carry in-molecule
weights - was true of *ghost* weights and is not true in general. Weights fitted against
the free atom's own two-electron integrals transfer into a molecule and land on the
in-molecule curve. The reason is the one §12 gave for why the weights matter at all: they
do conditioning work on a metric the ridge is not equivariant to, and an ERI target is a
much better proxy for what LS-THC asks of a grid than an overlap target is. `ghostw`'s
weights were fitted to condition a *ghost* metric; `eriw`'s are fitted to reproduce
integrals of exactly the kind the `Z` fit has to reproduce.

The accuracy column says the same thing more cheaply: at 487 and 565 points `eriw` sits at
+2.8 uHa against the parent grid's +2.74, i.e. **within 0.05 uHa of the complete
3288-point grid on a sixth of the points**, with no molecule anywhere in its fit.

### What this does and does not settle

It does not retire ghosts by itself - it is two molecules, one basis, one parent-grid
level. But it removes the reason ghosts were introduced. §8's argument for them was that
the free-atom fit is ill-posed; on an ERI target it is well posed, and the object that
comes out is smaller, quieter under rotation, and closer to the in-molecule bound than the
ghost-fitted one. If that holds up, the ensemble-choice question of §9, the rank-ceiling
constraint it left behind, the partner-variety trap of §10 and the whole notion of a
training set go with it.

Three things are worth not over-reading:

* **`eri` at `w = 1` is only as good as `ghost`, not better, on the orientation
  statistics** - 35.4x convergence against 35.6x, and its spread at the top rung (3.4 uHa)
  is the worst in the table. The support alone is not what buys the torque; the weights
  are. This is the third independent confirmation of §13's reading.
* **The ladders are not monotone**, as §9 warned. Methanol `eri` goes +9.44 uHa at 487
  points and +9.86 at 565, and ethanol `eri` stalls near +24 uHa from 717 points onward,
  which is why its 10 and 5 uHa cells are `n/a`. That is a threshold ladder re-solving
  rather than extending, not a ceiling: saturation says carbon can reach 463 points where
  the `1e-6` rung uses 182.
* **`eriw` is a weight set, and weights are a footing.** Every ratio above is quoted
  against a control on the same footing for exactly that reason, and the `eriw` rows must
  never be compared to `ghost`, `blocked1` or §8's numbers, all of which are `w = 1`.


## 15. The orbital response, and the first AIMD to run on this surface

`pythc.grad.response` and `pythc.grad.total`, tested in `tests/test_thc_response.py`.
Section 13 promoted the orbital response from "standard machinery left undone" to the one
thing blocking the application, on the grounds that the fixed-orbital force is not the
gradient of the propagated energy and breaks rotational invariance at ~8300 uHa/rad -
fifty times the transferable grid's own torque. That is now built, and the two claims it
was blocking are now measured rather than argued.

### The reformulation that made it a one-solve problem, not a two

The obstacle was not the CPHF solve, which is standard and which PySCF supplies. It was
that the Laplace factors carry orbital *energies*: written with `exp(t e_i)`, the THC-MP2
energy is **not invariant** under a rotation among the occupied orbitals, so its response
needs the occupied-occupied and virtual-virtual blocks of `U` - which no CPHF solver
returns, and which the canonical condition pins down only through a second coupled
equation whose unknowns feed back into the first.

Writing `Theta_o = w^(1/4) exp(t F_oo)` in place of `diag(w^(1/4) exp(t e_i))` removes
the problem rather than solving it. It is the *same function* at the canonical point -
verified to 1e-12 - but manifestly invariant, so those blocks enter only through the
overlap derivative, which is a skeleton quantity. What is left needing a coupled solve is
the occupied-virtual block alone, which is exactly what a CPHF solver hands back.

The adjoint of that rewriting is a Loewner divided-difference matrix,
`L[i,j] = (f_i - f_j)/(e_i - e_j)` with `L[i,i] = t f_i`, and the new quantity is
`dE/dF_oo = L o dE/dTheta_o` rather than the old `dE/de`. Its **diagonal is exactly the
old `eps_bar`** (agreeing to 5e-17), which is a free and exact check that the rewriting
reduces correctly; the off-diagonal is new, and finite-differences against a genuine
Fock-block perturbation with the expected quadratic convergence.

The invariance then checks *itself*: if the Fock adjoint is right, the occupied-occupied
and virtual-virtual blocks of the MO Lagrangian cannot have an antisymmetric part. They
come out symmetric to **6.4e-15**. That diagnostic is computed on every call, because it
is free and because it fails loudly.

### It is right

Water, 158-point blocked grid at `1e-3`, cc-pVDZ, ridge `1e-2`, 6 Laplace points, against
a central difference of the correlation energy with the **SCF re-converged at every
displaced geometry**:

| step | err, fixed-orbital | err, with response | ratio |
| --- | --- | --- | --- |
| `h = 1e-2` | 2.8e-3 - 5.6e-3 | 2.9e-7 | |
| `h = 1e-3` | 2.8e-3 - 5.6e-3 | **2.2e-9** | 129x |
| `h = 1e-4` | 2.8e-3 - 5.6e-3 | 5.4e-9 | (FD floor) |

The relaxed error falls by 129x and 190x per decade on the two components continued
furthest, then flattens where the finite difference hits its own noise floor at an SCF
converged to `1e-14` - which is what a correct gradient does and what a gradient missing
a term does not. The fixed-orbital error does not move at all, because it is not a
step-size artefact: it is the missing response, and on this system it is **18% of the
fixed-orbital correlation gradient's norm**.

### The rotational identity closes, and that is the sharper test

Section 11's identity - for a lab-fixed atom-centred grid the angular-momentum leak is
exactly the grid's own orientation torque,

    dL/dt = sum_A R_A x F_A = -sum_A R_A x dE/dR_A = +sum_A tau_A

- needs no finite difference, so it is the sharpest instrument available. Density-fitted
Hartree-Fock has no quadrature grid and satisfies it to 0.000 uHa/rad on its own, so any
residual belongs to the correlation gradient:

| gradient | \|dL/dt\| | residual vs the grid's own torque | relative |
| --- | --- | --- | --- |
| fixed-orbital | 413.7 uHa/rad | **195.6** uHa/rad | 9.1e-3 |
| with response | 347.9 uHa/rad | **8.6e-5** uHa/rad | **6.1e-9** |

The residual falls by a factor of 2.3 million and the identity closes to machine
precision. Section 11 could only check it to 4.7e-4 relative and called the gap the
omitted response; it was. Note that the 347.9 uHa/rad the relaxed gradient lands on is
the grid's own torque *at ridge 1e-2* and is a regulariser artefact in the sense section
12 insists on - water is rank-saturated and its true torque is zero. What is meaningful
here is the residual, not the absolute.

### And the trajectory runs

NVE on the frozen-grid THC-MP2 surface through `pyscf.md`, water, `dt = 20` a.u., the
grid frozen for the whole run and only translated onto the nuclei at each step:

| force | drift per step | total, 12 steps |
| --- | --- | --- |
| fixed-orbital | 21.68 uHa | 207.4 uHa |
| with response | **0.247 uHa** | **-1.4 uHa** |

**Eighty-eight times better, and the fixed-orbital number reproduces section 13's "tens of
microhartree per step" exactly.** Energy conservation is the property that decides whether
a force is the gradient of the energy being propagated, and it is the one a trajectory
can test that a single derivative cannot. Section 13 had to drive its trajectories with
the DF-RHF gradient and integrate the frozen grid's torque along them, because propagating
on the THC surface was not possible; it now is.

### What this does and does not settle

It settles the prerequisite. The gradient is the derivative of the energy it propagates,
to machine precision by two independent routes - a relaxed finite difference and an exact
rotational identity - and a trajectory on it conserves energy. Nothing in the acTHC
programme is now blocked on unbuilt machinery.

It does not make it cheap. `orbital_response_gradient` solves the coupled-perturbed
equations **once per nuclear degree of freedom**, which is Hessian-level work: `3 N`
solves where a Z-vector formulation needs one. That is deliberate - it has no transposed
operator algebra in it to get wrong, so it is the reference the cheap route must
reproduce - but it is not what production dynamics would use. `response_lagrangian`
already returns the intermediate a Z-vector implementation contracts, and the change is
local to one function.

Everything here is water, cc-pVDZ, `ov` mode, one grid, ridge `1e-2` rather than the
damped filter section 12 mandates for gradient work, and twelve steps of a trajectory
started from rest. The response is basis- and method-general and the rotational identity
is exact rather than statistical, so none of those restrictions is load-bearing for the
*correctness* claim; all of them are for any claim about cost or about what a picosecond
would do. In particular, **nothing here re-runs section 13's leak measurements on the
propagated surface**, which is now possible for the first time and is the obvious next
experiment: the 4.5 degrees of axis tilt in 2.5 ps was measured along an RHF trajectory
with the leak fed back, not along a trajectory the THC force actually drove.

## 16. Support and weights cannot be chosen separately - the mixed-footing control, and what the weight footing is really worth

This was run as the *in-molecule* route to HANDOFF step (8) - fit a per-element weight
set against the objective the runtime actually faces - in parallel with §14, which
reached the same step from the other side and answered it better. §14 settles (8): the
free atom's own ERIs supply a rich enough target that no molecule and no ghost is needed,
and `eriw` converges 1481x where `ghostw` manages 13.4x.

What this section adds is the **decomposition**. §14 compares self-consistent pairings -
a support and the weights the same fit produced - and warns in passing not to read an
`eriw` number against a `w = 1` one. It does not measure what happens if you break the
pairing. `insitu.py` does, because the in-molecule route makes the mixed case natural to
build: weights fitted for one objective, laid onto a support selected for another. That
control turns out to matter more than the route it came from, and it is the reason §14's
warning is the weak form of a stronger statement.

For element E the training blocks are the real atoms of E in real molecules. Block
`(M, A)` presents E's candidate points, translated onto atom A of molecule M, against
that atom's Becke share `S_A` of M's overlap matrix, integrated on A's own full sub-grid
- the identical target `NNLSGrid._build_blocked` and `ghosts.fit_element` both fit, with
a real neighbour at its real position instead of a ghost. All blocks stack into one
`StackedOperator` and one Lawson-Hanson solve, so the variables are tied across every
atom of E in every training molecule. The product is still `element -> (indices,
weights)` with no molecular index in it; the training molecules are consumed by the
offline fit exactly as a basis set's optimisation molecules are, and `dw/dR = 0` holds as
it does for `ghostw`, so nothing in `pythc.grad` changes.

Trained on water, methanol and ethane; held out on ethanol, formaldehyde and propene,
the last two carrying a C=O / C=C bond shorter than any bond in training and shorter than
the shortest ghost shell. Support and weights separate, so five modes:

| mode | support | weights |
| --- | --- | --- |
| `ghost` | ghost fit | 1 |
| `ghostw` | ghost fit | ghost fit |
| `molw` | ghost fit | **in-molecule** |
| `molfit` | **in-molecule, per element** | **in-molecule** |
| `molfit1` | **in-molecule, per element** | 1 |

### The fit reaches its own objective, and the gain transfers

Relative residual `||Aw - b|| / ||b||` against the in-molecule overlap objective at
threshold 1e-4, held-out column over ethanol / formaldehyde / propene:

| mode | C train | C held | H train | H held | O train | O held |
| --- | --- | --- | --- | --- | --- | --- |
| `ghost` (`w = 1`) | 6.55 | 6.02 | 4.25 | 1.76 | 9.71 | 8.89 |
| `ghostw` | 1.30 | 1.48 | 1.19 | 0.75 | 0.53 | 0.41 |
| `molw` | 0.19 | 0.36 | 0.30 | 0.34 | 0.22 | 0.84 |
| `molfit` | 0.17 | 0.32 | 0.28 | 0.34 | 0.22 | 0.37 |

4x on carbon against `ghostw`, 2x on hydrogen, with training and held-out columns close
enough that the fit is learning the objective rather than the molecules. So the offline
fit does what it was asked to.

### `molw` never wins

Everything below is at **matched point count** - each mode's ladder interpolated onto
shared sizes, errors taken as distance from that molecule's own unpruned parent grid -
because the modes do not agree on size at a shared threshold. Median ratio against
`ghostw` over the three rungs of each held-out molecule; below 1 means the mode beats
`ghostw`:

| molecule | metric | `ghost` | `molw` | `molfit` | `molfit1` |
| --- | --- | --- | --- | --- | --- |
| ethanol | err | 1.14 | 1.77 | **0.21** | 0.25 |
| formaldehyde | err | 1.19 | 1.72 | **0.32** | 0.38 |
| propene | err | 1.21 | 1.52 | **0.82** | 0.93 |
| ethanol | spread | 1.37 | 1.23 | 1.46 | 1.48 |
| formaldehyde | spread | 0.99 | 2.76 | **0.46** | 0.53 |
| propene | spread | 1.10 | 1.72 | **0.31** | 0.32 |

Counted rung by rung over the held-out molecules, `molw` beats `ghostw` **0 times out of
12** on energy and **0 out of 12** on rotation spread. It is not close and it is not
noise: better weights laid onto the ghost support are worse than the ghost's own weights
on every held-out molecule at every size measured. Refitting the weights while keeping
someone else's support does not work, whatever the weights are fitted against.

`molfit` wins 8/10 on energy and 7/10 on spread, at a median of ~3x on energy. `molfit1`
tracks it closely, which is the next result.

### Support and weights are one object

`molw` is the diagnostic. It carries the best-fitted weights in the table and the second
worst grid, and the only thing distinguishing it from `molfit` is that its points were
chosen by a different fit than its weights. A weight set is worth something only on the
support it was selected with; mixing footings is worse than either self-consistent
pairing.

That sharpens two earlier readings. 4(6)'s "those weights condition the ghost metric
rather than the in-molecule one" was the right mechanism one step too narrow - it is not
that ghost *weights* are wrong, it is that a ghost *fit* is wrong, points and weights
together. And §14's "do not read an `eriw` number against a `w = 1` number" is the
cautious form: the footings are not merely incommensurate, a weight set carried onto the
wrong support is *worse than nothing*. §14's own result is consistent with this and could
not have shown it, since every pairing it measures is self-consistent - which is also why
its 1481x is a property of the ERI *fit*, not of ERI weights that could be transplanted
onto a ghost or in-molecule support.

### What the weights are actually worth: ~2-2.5x, growing with grid size

`molfit` and `molfit1` share a support exactly, so this needs no interpolation at all.
Ratio of `w = 1` to weighted, so above 1 means the weights help:

| molecule | points | err gain | spread gain |
| --- | --- | --- | --- |
| ethanol (held) | 424 / 723 / 957 | 1.06 / 2.40 / 2.36 | 1.01 / 1.06 / 1.51 |
| formaldehyde (held) | 190 / 307 / 401 | 1.07 / 2.31 / 1.59 | 1.14 / 0.89 / 3.33 |
| propene (held) | 426 / 744 / 984 | 1.04 / 1.71 / 2.06 | 0.88 / 1.56 / 2.23 |
| methanol | 282 / 475 / 629 | 1.17 / 2.65 / 8.21 | 1.09 / 2.09 / 3.11 |
| ethane | 376 / 664 / 884 | 1.05 / 2.34 / 2.52 | 1.00 / 1.38 / 1.82 |

So 4(7)'s localisation of the gap to the weight footing survives in direction but not in
magnitude. The weights are worth about **2-2.5x on the energy's distance from the floor
and 1.5-3x on the rotation spread**, they are worth essentially *nothing* at the coarsest
rung, and the gain grows with grid size. They are not the 450x that 4(7)'s single-rung
`blocked` / `blocked1` pair (0.02 against 9.02 uHa/rad) suggested. Water is excluded from
the table above as §8 and §13 both exclude it - at 24 AOs its co-density manifold
saturates and every absolute error is sub-uHa, so its 7-17x ratios are division by noise.

### The torque agrees in direction and is too noisy to rank with

Run in the harness that produced §13 - the `ghost`, `blocked` and `blocked1` rows
reproduce that table to the second decimal (175.09 against 175.1, 792.80 against 792.8,
3.14 against 3.15), so these are the same numbers and not a parallel measurement. Net
torque, `pinv`, uHa/rad, on held-out formaldehyde:

| mode | points | err/uHa | net tau | rms draw tau | spread/uHa |
| --- | --- | --- | --- | --- | --- |
| `ghost` | 344 / 427 / 486 | 2.73 / -4.83 / -6.00 | 239.22 / 58.63 / 12.06 | 209.11 / 58.43 / 7.57 | 7.79 / 3.46 / 0.44 |
| `ghostw` | 344 / 427 / 486 | -2.13 / -6.40 / -6.25 | 163.37 / 1.00 / 3.74 | 34.20 / 0.56 / 8.99 | 2.67 / 0.06 / 1.51 |
| `molw` | 281 | 8.21 | 179.75 | 503.56 | 9.30 |
| `molfit1` | 307 / 401 | -7.47 / -8.08 | 71.37 / 63.87 | 179.33 / 86.36 | 4.87 / 6.83 |
| `molfit` | 307 / 401 | -5.19 / -6.42 | 3.61 / 6.19 | 90.54 / 5.40 | 2.02 / 0.12 |
| `blocked` | 287 / 328 | -6.42 / -6.40 | 0.85 / 0.09 | 6.47 / 0.82 | 0.20 / 0.03 |

`molfit` at ~300 points is 15-23x quieter than either ghost footing at ~276, which agrees
with the energy and spread tables. But `ghostw` runs 163.37 at 344 points, **1.00** at
427 and back up to 3.74 at 486, and its rms-over-draws does the same (34.20, 0.56, 8.99).
A ladder that moves by two orders of magnitude between adjacent rungs and then reverses
cannot support a ratio quoted at one rung, in either direction - a first draft of this
section read `molfit`'s 6.19 at 401 points against `ghostw`'s 163.37 at 344 and concluded
26x, which the 427-point rung then inverted. The torque is quoted here because §13 quotes
it and the direction agrees; the ranking above rests on the energy and spread tables,
which have 6 molecules x 3 rungs behind them and are interpolated to matched size.

That noise is a caveat on §13's own table as much as on this one: its headline ratios
(0.02 / 9.02 / 27.2 / 52.3) are single-rung reads of the same statistic.

### The overlap residual cannot see any of this

`molw` has the best in-molecule overlap residual of any transferable mode on the training
set (0.19-0.30, within noise of `molfit`'s 0.17-0.28) and an S-RMSD of 5.0e-2 against
`molfit`'s 4.6e-2 - and it loses to `ghostw` 0 times out of 12 in the other direction.
`molfit1`, meanwhile, has a *terrible* residual (8-36, worse than `ghost` in places,
because a support selected by a weighted fit and then stripped of its weights is not a
quadrature rule at all) and lands within a factor of 2 of `molfit` on energy. Two grids
matched on the fitted objective to 10% differ by 1.5-2.8x in the quantities that matter,
and in the opposite order.

This is §5 and §3's decoupling in its sharpest form yet: the overlap matrix is a *cheap
proxy that has visibly run out*. It was enough to select points spanning the co-density
manifold and it is enough to weight them once the points are right, but it cannot rank
two grids that both fit it, and the conclusion it does support here is the wrong way
round. The remedy this pointed at - a target closer to what is computed, the atom's own
ERIs - is exactly what §14 built independently and concurrently, and §14's numbers are
the better evidence for it. Read the two together: §14 shows the richer target works,
this section shows why the overlap target could not have told you.

### Cost

Seconds per element, independent of the molecule the grid is used in, exactly as the
ghost fit is. `molfit` is larger than `ghost` at a given threshold (629 against 512 points
on methanol at 1e-4), which is why every comparison here is at matched size; `insitu.py`
interpolates each ladder for that reason and `data/insitu.json` carries the result.

Against §14 this route is strictly worse and should not be preferred: it needs training
molecules, it lands at ~3x over `ghostw` where `eriw` lands at 1481x over the ladder, and
its product is no more transferable. The `molw` control is what earns it a section.

## 17. Four levers on the ghost fit, the weight footing priced, and a preconditioner that replaces the weights in the energy

**Read §14 first, because it moves this section's subject out from under it.** This work
was done in parallel with §14 and §16 and asks what sets where a *ghost* fit's support
stops. §14 then showed the free atom's own ERIs lift the equation ceiling that forced
ghosts in the first place, so the ensemble these levers tune may not be the object worth
tuning. Three things survive that:

* the levers are properties of **any** NNLS support fit against a stacked target, not of
  ghosts specifically - the parent-grid and basis rows in particular transfer directly to
  §14's ERI target, whose ceiling is set the same way;
* the oracle below is the **ceiling** form of §16's `molw` control, run with more freedom
  than §16 gives it and agreeing with it;
* the Jacobi result is about the **metric inversion** and is independent of the target,
  the support and the weights alike.

Everything in §§8-13 was measured at one point in the space of things that decide where a
ghost fit's support stops. Four axes were never varied:

* the **AO basis** - cc-pVDZ throughout (`sweep.py:23`), which sets the equation count;
* the **parent grid** - level 0 throughout (hardcoded in `ghosts.element_grid`), which
  sets the number of candidate points the solve chooses between;
* the **direction set** - 12 icosahedral vertices, with 6 octahedral ones tried in §9; a
  tetrahedron, the smallest non-degenerate set and the one that matches sp3 bonding, had
  never been tried;
* **cage or not** - one ghost per environment throughout, on the strength of an argument
  in `ghost_environments` that a cage would squeeze the central atom's Becke share down
  to a lobe around the nucleus. An argument, never a measurement.

`levers.py` runs all four in two stages, and they are separate because of the lesson this
directory keeps re-learning about cheap proxies. Stage A drives the KKT threshold to zero
and reports where Lawson-Hanson stops - the support **ceiling**, SCF-free, seconds per
cell. Stage B runs accuracy ladders against one SCF, one DF-MP2 reference and one floor.

**A ceiling is a necessary condition, not a quality.** §9 has `octa-cycled` carrying one
of the lowest ceilings and the *best* methanol accuracy, while failing ethanol precisely
because its ceiling is too low. Stage A says what a setting *can* reach; only stage B
says what it costs to get there. Both tables are below and they do not rank the levers
the same way, which is the point.

### Stage A: where each setting saturates

Threshold `1e-12`, no SCF. `ghost` is the committed configuration of §§8-13.

| element | setting | equations | candidates | ceiling | vs `ghost` | Becke share |
| --- | --- | --- | --- | --- | --- | --- |
| H | `free` | 15 | 392 | **15** | 0.11x | 1.000 |
| H | `ghost` | 1485 | 392 | 132 | - | 0.575 |
| H | `basis:cc-pVTZ` | 8481 | 392 | **205** | 1.55x | 0.575 |
| H | `parent1` | 1485 | **2040** | **187** | 1.42x | 0.551 |
| H | `dir:octahedron` | 750 | 392 | 80 | 0.61x | 0.577 |
| H | `dir:tetrahedron` | 370 | 392 | 47 | 0.36x | 0.562 |
| H | `cage:icosahedron` | 96756 | 392 | **172** | 1.30x | **0.003** |
| C | `free` | 105 | 858 | 92 | 0.59x | 1.000 |
| C | `ghost` | 3681 | 858 | 155 | - | 0.572 |
| C | `basis:cc-pVTZ` | 17385 | 858 | **362** | 2.34x | 0.572 |
| C | `parent1` | 3681 | **4412** | **216** | 1.39x | 0.554 |
| C | `dir:tetrahedron` | 1081 | 858 | 99 | 0.64x | 0.563 |
| C | `cage:icosahedron` | 108348 | 858 | **336** | 2.17x | **0.002** |
| O | `ghost` | 3681 | 858 | 199 | - | 0.581 |
| O | `basis:cc-pVTZ` | 17385 | 858 | **390** | 1.96x | 0.581 |
| O | `parent1` | 3681 | **4412** | **251** | 1.26x | 0.560 |
| O | `dir:tetrahedron` | 1081 | 858 | 103 | 0.52x | 0.570 |
| O | `cage:icosahedron` | 108348 | 858 | **314** | 1.58x | **0.003** |

Three things are already visible.

**The ceiling is not the nominal equation count, and the gap is large.** `cage:icosahedron`
supplies 108348 equations to `ghost`'s 3681 - a factor of 29 - and lifts the ceiling by
2.2x. Nominal equations are not what binds a stacked ghost fit; *independent* ones are,
exactly as `ensemble.saturate` says. Only the free-atom cap of §8 is exact, because one
environment's equations are all independent of each other.

**At level 0 in cc-pVDZ both constraints are active.** `parent1` moves the ceiling by
1.26-1.42x at **unchanged** equation count - it only adds candidate columns. **This
contradicts the premise HANDOFF section 4(9) is written on**, which reasons that since
NNLS cannot retain more points than the target supplies equations, a finer parent offers
"the same support size, chosen better". The support size is not the same; it moves. The
level-0 parent has been co-limiting all along, and 4(9) should be re-written before it is
run rather than after.

**The cage's partition really does collapse, and it does not matter.** The central atom
keeps 0.2-0.3% of its free-atom Becke weight under an icosahedral cage against 57% with
one neighbour, so `ghost_environments` called the mechanism correctly. What does not
follow is the conclusion: the ceiling goes **up**. `stack_targets` normalises each
environment's target to unit norm, which is what rescues a nearly-vanished fragment, and
what is left is a target that sees every direction at once.

### Stage B: what the levers cost per microhartree

Methanol, 48 AOs, floor `+2.69` uHa on the 3288-point parent grid, `w = 1`,
`metric_ridge = 1e-8` - the footing of §§8-10, so these rows are readable against those
tables. The `becke`, `blocked` and `ghost` rows reproduce §8 exactly, point for point and
microhartree for microhartree, which is what makes the new columns comparable.

| points / err (uHa) | curve |
| --- | --- |
| `blocked` | 301/+16.41, 491/+5.29, 670/+4.15 |
| `ghost` | 223/+277.87, 414/+19.78, 512/+30.26, 627/+8.85, 702/+5.85, 816/+5.67 |
| `parent1` | 245/+93.48, 455/+15.34, 593/+7.12, 745/+4.38, 912/+3.91, 1111/+3.44 |
| `cage` | 314/+44.34, 529/**+4.84**, 795/+3.33, 923/+3.07, 1053/+2.96, 1112/+2.96 |
| `tetra` | 191/+407.07, 210/+441.29, 278/+30.36, 316/+20.81, 363/+13.79, 375/+15.32 |

Interpolated to matched accuracy, against the in-molecule `blocked` lower bound:

| target | `blocked` | `ghost` | `parent1` | `cage` | `tetra` |
| --- | --- | --- | --- | --- | --- |
| 50 uHa | 301 | 326 (1.08x) | 295 (0.98x) | 314 (1.04x) | **262 (0.87x)** |
| 20 uHa | 301 | 400 (1.33x) | 394 (1.31x) | 357 (1.19x) | 307 (1.02x) |
| 10 uHa | 330 | 587 (1.78x) | 483 (1.46x) | **404 (1.22x)** | n/a |
| 5 uHa | 405 | 650 (1.60x) | 575 (1.42x) | **456 (1.13x)** | n/a |
| 2 uHa | 565 | n/a | 716 (1.27x) | **542 (0.96x)** | n/a |

**The cage is the best lever measured in this basis, and it nearly closes the ghost
gap.** 1.78x becomes **1.22x** at 10 uHa and 1.60x becomes **1.13x** at 5 uHa. (In
cc-pVTZ it does the opposite - see the basis lever below, which is why "in this basis"
is not a hedge.) At 2 uHa the transferable grid
is *smaller* than the in-molecule fit it is supposed to be paying a penalty against, and
`ghost` cannot reach that target at all while `cage` reaches the parent-grid floor
(+2.96 uHa against +2.69) and stops improving because there is nothing left to improve.

Compounding with §1's `blocked`/`global` ratios on the same molecule, a frozen
per-element grid against a single global molecular fit costs

| target | with `ghost` (§8) | with `cage` |
| --- | --- | --- |
| 10 uHa | 1.51 x 1.78 = **2.7x** | 1.51 x 1.22 = **1.85x** |
| 5 uHa | 1.47 x 1.60 = **2.4x** | 1.47 x 1.13 = **1.65x** |
| 2 uHa | n/a | 1.45 x 0.96 = **1.39x** |

i.e. **1.9-3.4x on the `n_P^2` parts where §8 reported 5.8-7.3x**, on the one molecule
measured. The headline tax of this programme is roughly halved by an option that was
turned off on the strength of a docstring.

**The tetrahedron reproduces §9's rank ceiling on a new axis.** It is the *cheapest*
setting at loose accuracy - 262 points at 50 uHa, better than `blocked` itself - and it
cannot reach 10 uHa at any threshold, flattening at +13.79 uHa and then getting worse. Its
ceilings (47/99/103 for H/C/O) cap methanol at 390 points, and the curve stops at 375. So
a small direction set is not a bad ensemble; it is a *short* one, and §9's advice - sum
the per-element saturation sizes before spending single points - is what catches it. Its
non-monotone start (191/+407 then 210/+441) is the coarse-ladder behaviour §9 documents.

**The parent-grid lever helps, and is not the lever.** 1.78x to 1.46x at 10 uHa on 5.1x
the candidate points. Worth having, much cheaper to obtain than the cage's equations, and
nowhere near the cage. Its real interest is elsewhere: §13 shows the level-0 parent is a
poor grid to be rotationally invariant on, and nothing here measures a torque.

### The basis lever, and the cage does not survive it

The basis gets its own file and its own floor, because the parent grid is a different
object in a different basis: the same 3288-point level-0 Becke grid is **+7.32 uHa** in
cc-pVTZ against +2.69 in cc-pVDZ. Nothing here ratios across the two.

Methanol, 116 AOs, cc-pVTZ / cc-pVTZ-RI, error **above that floor**:

| | curve |
| --- | --- |
| `blocked` | 767/+105.6, 1053/+32.0, 1293/+19.3 |
| `ghost` | 523/+594.4, 778/+74.8, 1063/+8.4, 1202/**+3.9**, 1351/+4.4 |
| `cage` | 699/+513.1, 918/+66.0, 1150/+27.9, 1366/+17.4, 1504/**+6.0** |

| target | `blocked` | `ghost` | `cage` |
| --- | --- | --- | --- |
| 50 uHa | 935 | 824 | 987 (**1.20x** `ghost`) |
| 20 uHa | 1274 | 940 | 1298 (**1.38x**) |
| 10 uHa | n/a | 1038 | 1436 (**1.38x**) |

**In cc-pVTZ the cage is 1.2-1.4x worse than the single-ghost ensemble it beats by
1.5x in cc-pVDZ.** That is the reversal, and it is the reason the stage-B headline above
must be read as a statement about one molecule in one basis rather than about caging.

Two things stop this being a clean refutation, and both point the same way - the ladders
are off-scale in this basis, as §10 warns they go:

* **`cage`'s curve is still falling steeply at its tightest rung** (27.9 -> 17.4 -> 6.0)
  while `ghost`'s has flattened and gone non-monotone (8.4 -> 3.9 -> 4.4). A curve that
  is still descending has not reached the point count its ratio is being read at.
* **[CORRECTED by the extended ladder - see below.** This read "`blocked`'s ladder never
  converges at all. Its tightest rung is 19.3 uHa above the floor and still falling,
  which makes the apparent result that `ghost` *beats* the in-molecule fit here an
  artefact of an unconverged denominator and not a measurement." Extended two decades it
  does not fall. It plateaus.**]**

### The `blocked` plateau was the weights, and the lower bound holds after all

`levers.py --blocked-diagnostic`. The extended ladder above grows `blocked`'s support
1293 -> 1746 points for no accuracy at all, ~19 uHa above the parent-grid floor
throughout, while `ghost` passes it and reaches 2.5. Read at face value that breaks the
premise §1 and §8 rest on - that the in-molecule fit is a *strict lower bound* on any
frozen atom-centred scheme. **It does not survive the diagnostic, and what it was is the
weights.**

One set of `blocked` supports, fitted once and re-weighted, evaluated four ways at the
same `lambda = 1e-8` on both filters, so that everything that moves between arms is the
footing rather than the grid. Methanol, cc-pVTZ, uHa **above the +7.18 floor**:

| arm | 1053 pts | 1293 | 1531 | 1746 |
| --- | --- | --- | --- | --- |
| `w = 1`, ridge | +32.15 | +19.43 | +18.68 | +19.06 |
| `w = 1`, damped | +8.60 | +5.65 | +6.72 | +5.99 |
| NNLS weights, ridge | +0.36 | -0.08 | -0.04 | **+0.00** |
| NNLS weights, damped | +0.29 | -0.16 | -0.07 | **-0.07** |

**With its own weights the in-molecule fit sits on the parent-grid floor from 1053 points
upward.** Not near it - on it, to within 0.4 uHa at the loosest rung and 0.1 at the rest.
The plateau is gone, and so is everything that was read off it:

* **RETRACTED: "in cc-pVTZ the transferable grid beats the in-molecule fit."** It does
  not. `blocked` is at the floor at 1053 points; `ghost` needs 1485 to reach 2.5 uHa
  above it and never reaches it on the ladder run. §1 and §8's lower-bound premise holds
  in cc-pVTZ, and holds *harder* than in cc-pVDZ.
* **The filter is worth a factor of three and the weights are worth everything.** Ridge
  to damped takes the `w = 1` plateau from 19 to 6 uHa - §12's result reproducing in a
  second basis, on a grid five times larger than anything it was measured on. Restoring
  the weights takes it from 19 to zero. Both matter; they are not the same size.

**And that is the result, because a transferable support cannot carry those weights.**
§13 localised the orientation gap to exactly this - `ghost` converges like `blocked1`
(`w = 1`) and not like weighted `blocked`, 35.6x against 17667x - and this prices the
same footing in *energy*, in a basis where it is no longer a rounding error:

| | cost of discarding the weights, in-molecule support |
| --- | --- |
| cc-pVDZ, 670 pts | ~1.5 uHa |
| cc-pVTZ, 1746 pts | **~19 uHa** |

§3 concluded the weights barely matter, on the strength of bit-identical energies under
the *pseudoinverse*. §12 walked that back for conditioning. §13 walked it back for
orientation. This walks it back for the energy, and adds the part that matters for where
the programme goes: **the penalty grows with basis size.** So the ghost gap is not merely
unimproved in cc-pVTZ - measured against a `blocked` that is allowed its own weights, it
is far worse than the 1.1-1.8x §8 reports in cc-pVDZ, and HANDOFF 4(8) - fit per-element
weights *for* the in-molecule objective - stops being the highest-leverage idea on the
list and becomes a prerequisite for using this scheme in a production basis.

**One caveat this puts on §8, §10 and this section alike.** Every ghost-gap number in this
directory is measured against a `blocked` row that is itself at `w = 1`, deliberately, so
that the only difference between the modes is where the points came from. That is the
right control for isolating the *selection*, and this diagnostic shows it is a materially
handicapped baseline - increasingly so with basis size. The published gaps therefore
flatter the transferable scheme relative to what an in-molecule fit can actually do. They
are not wrong; they are answering a narrower question than their phrasing suggests.

So cc-pVTZ establishes two things, one narrow and one not. The narrow one: **the cage's
cc-pVDZ win does not generalise across basis sets as measured**, and that ranking is
still provisional, because `cage`'s ladder is the one arm never driven to convergence -
`levers_methanol_tz_wide.json` was killed by the machine on its second cage rung, a
cc-pVTZ cage environment carrying ~360 AOs across 9 environments being the heaviest solve
in this directory. The broad one is the weight footing above, which is not about the cage
at all.

What does carry across is stage A's mechanism. cc-pVTZ raises the per-element ceilings
1.55-2.34x and the supports are correspondingly larger at every threshold - `ghost` spans
523-1351 points here against 223-816 on the same molecule in cc-pVDZ. The basis lever
moves what it was predicted to move. It just does not follow that a lever which buys
points buys accuracy, which is the same lesson `tetra` teaches from the other end.

### The ceiling on an in-molecule weight fit, and it lands below the floor

`levers.py --oracle`. §16 runs `molw` - a ghost support carrying weights fitted per
*element* against the in-molecule overlap objective - and finds it never wins. This is
the same control with the constraint removed: weights fitted **per atom, per molecule**,
against that atom's own Becke share of the real molecular overlap. That is strictly more
freedom than HANDOFF 4(8) or §16's `molw` can have, so whatever it reaches is a
**ceiling** on the whole idea rather than a proposal.

Methanol, cc-pVDZ, `pinv`, 4 random per-atom orientations. The first four rows reproduce
§13's ladder to the digit - 0.02 / 9.02 / 27.2 / 52.3 uHa/rad - which is what licenses
reading the fifth:

| footing | support | weights | n | err/uHa | net tau | rms over draws | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `blocked` | in-molecule | in-molecule | 670 | +2.75 | **0.02** | 0.12 | 0.01 |
| `blocked1` | in-molecule | 1 | 670 | +3.65 | 9.02 | 9.28 | 0.40 |
| `ghost` | ghost | 1 | 702 | +3.15 | 27.25 | 22.03 | 0.41 |
| `ghostw` | ghost | ghost | 702 | +2.83 | 52.27 | 9.22 | 0.68 |
| `ghost_oracle` | ghost | **per-atom in-molecule** | 702 | +5.54 | **155.67** | 87.37 | 16.79 |

**The oracle is the worst of the five on every column**, 5.7x above simply setting
`w = 1` on the same support. Repeated at `damped 1e-10` it is 6.92 against `ghost`'s 0.30,
23x, so both filters agree in direction and this is not the regulariser §12 warns about.

So §16's "`molw` never wins" is not an artefact of fitting per element rather than per
atom, and not an artefact of its training set: **the ceiling is below the floor**. Fitting
weights against the in-molecule *overlap* objective is worse than not fitting them, with
any amount of freedom.

The reason is the one this file has now recorded four times. The overlap target carries no
information about THC accuracy - §5's `rmsd_S`, §8's support agreement, §10's residual,
which is *anti*-correlated - so an oracle for that objective is an oracle for the wrong
thing. §14 is the constructive form of the same point: it replaces the target rather than
optimising against it, and its weights transfer where these do not. **What is refuted is
4(8) against the overlap objective, which is how both routes to it were built. 4(8)
against the torque itself (4(11)) or against the LS-THC residual is untouched**, and §14's
`eriw` is evidence the second of those works.

### The weights are a diagonal preconditioner, and a runtime one replaces them in the energy

`levers.py --blocked-diagnostic`, `metric_scheme="*_jacobi"`. The collocation is
`X = w^(1/4) R` and the metric is `S_PQ = (X X^T) o (X X^T)`, so

    S(w) = D S(1) D,   D = diag(sqrt(w))

exactly. **The weights' entire effect on the fit is a symmetric diagonal scaling of the
metric.** That is why §3 found them irrelevant - a positive diagonal rescaling is absorbed
identically by the *pseudoinverse* - and why they are not: `ridge` and `damped` both apply
an absolute shift `lambda tr(S)/n`, which is not scale-equivariant.

If a diagonal scaling is all they are, the metric can supply one itself. `E = diag(1 /
sqrt(diag S))` gives `S^-1 = E (E S E)^-1 E`, `diag(S)_PP = (sum_mu X_muP^2)^2` is smooth
in the nuclear coordinates, and nothing about it is fitted or transferred. Methanol
cc-pVTZ, uHa above the +7.18 parent-grid floor, one set of supports throughout:

| arm | 1053 pts | 1293 | 1531 | 1746 |
| --- | --- | --- | --- | --- |
| `w = 1`, ridge | +32.15 | +19.43 | +18.68 | +19.06 |
| `w = 1`, damped | +8.60 | +5.65 | +6.72 | +5.99 |
| NNLS weights, ridge | +0.36 | -0.08 | -0.04 | +0.00 |
| **`w = 1`, ridge_jacobi** | **-0.10** | **-0.03** | **-0.05** | **-0.05** |
| **`w = 1`, damped_jacobi** | **-0.17** | **-0.05** | **-0.04** | **-0.04** |
| NNLS weights, damped_jacobi | -0.17 | -0.05 | -0.04 | -0.04 |

`w = 1` with a runtime preconditioner sits **on** the floor, indistinguishable from the
NNLS weights, under both filters, and does not hurt where weights are already present. The
19 uHa that discarding the weights costs in cc-pVTZ is recovered in full by a scaling read
off `S` in closed form - no offline object, no training molecules, nothing transferred.

**This was an energy result and nothing more, for a reason that was a defect rather than
a caveat - and §18 has since removed it.** `invert_metric_adjoint` had no `_jacobi`
branch. The forward scheme was added without one, the dispatch fell through to the *ridge*
adjoint, and the reverse pass therefore differentiated a different operator than the
forward pass applied: `E` depends on `S`, so `d(E filter(E S E) E)/dS` carries a term
through `E` that nothing computed. Energies right, gradients silently wrong - measured,
net torques of ~4e5 uHa/rad against ~1e-1 for the same grids under `damped`. It then
raised `NotImplementedError` naming the missing term, and an unknown scheme raises rather
than falling through. **§18 derives the term, verifies it three ways and runs the torque
ladder this section could not**, and what comes back is larger than the gap it was meant
to close: the preconditioned metric inverse is *exactly* invariant to the collocation
weights, so the weight footing §13 and §16 put the whole remaining orientation gap on
ceases to exist.

One implementation note that is not cosmetic: `diag(S)_PP` is exactly zero for a point
carrying no amplitude, which is what a weight fit produces whenever it drops a point - the
oracle above zeroes 250 of 702. Those rows are left unscaled; `1/sqrt(0)` turns the whole
metric to NaN.


### What this does not settle

**The two stages disagree and stage B is the one that counts.** `basis:cc-pVTZ` has the
highest ceilings in stage A and `cage` the second highest, but stage A cannot see that
`tetra`'s low ceiling comes with the best points-per-uHa at 50 uHa. Do not rank levers on
stage A.

**`cage` and `ghost` are not matched in environment count** - 10 against 13, since a cage
carries one environment per `(partner, scale)` rather than one per direction. §9 and §10
both identify environment count as what supplies rank, so part of the cage's win may be
bookkeeping rather than geometry. A cage ensemble widened to 13 environments, or a
single-ghost ensemble narrowed to 10, would separate the two. Nothing here does.

**Everything is at `w = 1` and `metric_ridge = 1e-8`,** which §12 disqualifies for a
gradient. A lever that wins on energy has to be re-read under `--scheme damped` before it
means anything for the application, and none of these settings has been through
`window.py`, `torque_ladder.py` or `trajectory.py`. **In particular no torque has been
measured for the cage** - §18 measures one for the preconditioner and for nothing else
here, and the four levers remain energy-only.

One molecule, one geometry, one ensemble family **in cc-pVDZ - and the one basis it was
carried to reverses its ranking**, on ladders that are not converged in either direction.
Until `levers_methanol_tz_wide.json` lands, the cage is a cc-pVDZ result and the honest
summary of the programme's point tax is still §8's, with §14's figure as the best case
rather than the number. The cage was run with the icosahedral,
octahedral and tetrahedral direction sets in stage A but only the icosahedral one in
stage B, and `cage:tetrahedron` is the interesting cell there - it reaches `ghost`'s
ceiling on H exactly (132) and 1.4-1.9x of it on C and O with an order of magnitude
fewer equations than `cage:icosahedron`, so it may be most of the win for a fraction of
the fit cost. The cage's fit is also the slowest thing in stage A by a wide margin
(76-80 s per heavy element at saturation against 1-2 s), which is irrelevant to an
offline object run once per element but is what makes a stage-A sweep over cages
expensive.

## 18. The preconditioner has a derivative, and it erases the weight footing the whole programme was stuck on

**Read §17 last, and this section against it.** §17 found that `E = diag(1 / sqrt(diag S))`
applied at runtime recovers, in the energy, the ~19 uHa that discarding the NNLS weights
costs in cc-pVTZ - and then had to stop, because `invert_metric_adjoint` had no `_jacobi`
branch. The forward scheme had been added without one, the dispatch fell through to the
*ridge* adjoint, and the reverse pass differentiated an operator the forward pass never
applied. The scheme has since been raising `NotImplementedError`, which left the one lever
that replaces the weights usable for single points and for nothing else.

The adjoint now exists (`pythc.grad.linalg.jacobi_inv_adjoint`), and with it the
measurement §17 could not make. The result is larger than the gap it was meant to close.

### The derivative

Write `Mj = E S E`, `G = f(Mj)` and `B = E G E`. `E` is a function of `S`, so it appears in
two places and contributes twice:

    G_bar   = E B_bar E,        Mj_bar = adjoint of the inner filter at Mj
    e_bar   = 2 (Mj_bar o S) e  +  2 (B_bar o G) e
    S_bar   = E Mj_bar E        +  diag(e_bar o (-e^3 / 2))

The last term runs through `de_P / dS_PP = -e_P^3 / 2`, and is zero on the rows
`lib.jacobi_scaling` leaves unscaled - `S_PP = (sum_mu X_muP^2)^2` is exactly zero for a
point carrying no amplitude, which is what a weight fit produces whenever it drops one.

Three checks, of which the second needs no finite difference. A central difference on a
matrix whose *diagonal* spans four decades agrees to 1e-8 relative, for `ridge`,
`ridge_eigh` and `damped` inner filters under all three ridge scalings. An exact inverse
cannot see a similarity transform - `E (E S E)^-1 E = S^-1` for any invertible diagonal -
so at `lambda = 0` the three paths must cancel to `-S^-1 S_bar S^-1`, and they do to 1e-12.
And the assembled nuclear gradient finite-differences under `ridge_jacobi` and
`damped_jacobi` on water, with the torque agreeing with actually rotating each point set
about its own nucleus. Both ways of getting it wrong are pinned as wrong against the same
difference: the historic fall-through, and the near miss of taking the inner filter's term
at `Mj` while holding `E` fixed.

### The ladder, against a matched control

`torque_ladder.py methanol --modes blocked,blocked1,ghost --ridges 1e-8 --pinv --draws 4`,
run twice and differing in `--scheme` alone - same supports, same draws, same seed, same
machine. The control reproduces §13 (`blocked1` 39.62x against 39.6x, `ghost` 35.56x
against 35.6x over the same rungs), which is what makes the arm readable.

Net torque, uHa/rad, at `1e-8`:

| mode | 235/223 | 301/414 | 410/512 | 491/627 | 670/702 | converges |
| --- | --- | --- | --- | --- | --- | --- |
| `blocked` damped | 341.70 | 45.72 | 1.04 | 0.11 | 0.02 | 14065x |
| `blocked` **damped_jacobi** | 357.33 | 57.05 | 0.21 | 0.06 | 0.02 | 17203x |
| `blocked1` damped | 316.61 | 75.69 | 6.17 | 3.26 | 1.32 | 240x |
| `blocked1` **damped_jacobi** | 357.33 | 57.08 | 0.22 | 0.03 | **0.01** | **69121x** |
| `ghost` damped | 814.31 | 109.49 | 630.20 | 21.37 | 15.15 | 54x |
| `ghost` **damped_jacobi** | 731.91 | 2.97 | 22.08 | **0.09** | **0.21** | **3511x** |

The energies move the same way and further: every preconditioned row from 410 points up
sits at **+2.75 uHa**, the parent-grid floor `blocked` reaches, where `ghost` under
`damped` is at +4.22 and +3.18 on its last two rungs and `blocked1` at +3.08. The
rotation spread falls with it - `ghost` at 702 points goes 0.40 to 0.028 uHa.

### What the numbers are saying: the weights are gone, exactly

`blocked` and `blocked1` are the same support at two weight footings, and §13's whole
account of the remaining orientation gap rests on the distance between them. Under the
preconditioner that distance is not reduced. It is **zero**:

| points | `blocked` energy | `blocked1` energy | `blocked` tau | `blocked1` tau |
| --- | --- | --- | --- | --- |
| 235 | -0.34335895057407 | -0.34335895055798 | 357.334 | 357.334 |
| 410 | -0.34339917009482 | -0.34339916940064 | 0.2147 | 0.2162 |
| 670 | -0.34339918873177 | -0.34339918912093 | 0.0208 | 0.0052 |

Ten digits of agreement in the energy is not a coincidence, and the algebra says it had to
happen. §17 established `S(w)_PQ = sqrt(w_P w_Q) S(1)_PQ`. Then
`diag(S(w))_PP = w_P S(1)_PP`, so

    e(w)_P = 1 / sqrt(w_P S(1)_PP) = e(1)_P / sqrt(w_P)
    =>  E(w) S(w) E(w) = E(1) S(1) E(1)      exactly, for any positive w.

**The Jacobi-preconditioned metric inverse is invariant to the collocation weights.** Not
approximately, not up to the filter: the weights cancel before the filter is ever applied.
Where §3 found the weights irrelevant under the *pseudoinverse* and §12, §16 and §17 found
them worth 2-2.5x and then ~19 uHa once a regulariser with an absolute shift entered, the
preconditioner restores the pseudoinverse's invariance to a filter that is analytic in `S`.

That is why `ghost` moves so much further than `blocked` does. §13 localised the entire
remaining gap between a transferable support and an in-molecule one to the weight footing;
§16 then showed the footing cannot be fixed by transplanting a weight set, because support
and weights are one object. Both are statements about a quantity that no longer exists.
A transferable support does not need to carry weights, and cannot be penalised for not
carrying them, because the metric supplies the scaling itself - in closed form, from the
geometry it is already at, with nothing fitted, nothing stored and nothing transferred.

At 627 points the transferable grid's net torque is **0.09 uHa/rad**, below the **0.24**
§13 measures for the complete 3284-point parent grid, and its energy is on the floor.

### What this does not settle

**One molecule, one basis, one seed.** Methanol, cc-pVDZ, `ov`, four draws at `damped
1e-8`. §17's own energy result is cc-pVTZ, and the two have not been run together.

**The gain appears only once the grid resolves.** On the smallest rungs the preconditioner
does nothing useful and can be slightly worse - `blocked` goes 341.7 to 357.3 at 235
points, and `ghost`'s first rung is 814 to 732 while its energy is still 193 uHa off. It
is not a rescue for a rank-starved grid; it starts paying at 410 points here.

**The ladder is still not monotone.** `ghost` reads 2.97 at 414 points, 22.08 at 512 and
0.09 at 627. That is §9's threshold-ladder artefact, which §16 warns against taking any
ratio from at a single rung. Read the column, not a cell.

**Nothing here is a trajectory.** §13's leak was measured by integrating a torque along a
DF-RHF trajectory, and §15 made propagation on the THC surface possible. The obvious next
run - `trajectory.py --scheme damped_jacobi`, which the script now accepts - has not been
made, and a 72x smaller torque is a prediction about rotational diffusion rather than a
measurement of it.

**The invariance is exact for the *metric*, and the metric is not the whole pipeline.**
`X = w^(1/4) R` still enters `W` and `D`; what cancels is the weights' effect on
`S^-1`, which §17 shows is where essentially all of it was. The ten-digit energy agreement
above is the evidence, not the algebra alone, and it is one molecule's worth.

**The auxiliary Coulomb metric is not preconditioned** and does not need to be, but note
that until this section it was silently *un-damped* whenever a `_jacobi` scheme was
requested: `build_aux_coulomb_inv` did not recognise the suffix and fell through to the
ridge, on both the forward and the reverse side. §17's energy table was produced with that
behaviour in place. It is a ridge on a well-conditioned metric rather than a wrong
derivative, so nothing there is invalidated, but the numbers are not exactly reproducible
against today's code.

## Verdict

The proposed object exists: §8 builds genuine offline per-element point sets, fits
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

**§10 closes the last go/no-go question, and it closes it cleanly.** One frozen support
per element, handed unchanged to ten molecules spanning sp3, sp2 and sp carbon, sp3 and
sp2 oxygen, nitrogen in two bonding modes, and four heavy-atom bonds shorter than any
ghost the fit ever saw: in-range molecules average **1.49x** at 10 uHa and out-of-range
molecules average **1.50x**. The whole suite spans 1.20-1.80x, which is the band §8
reported from methanol and ethanol alone. Bonding the ensemble never saw costs nothing
measurable, an unseen *partner element* costs nothing either, and the largest molecule in
the study - propene at 72 AOs, carrying an unseen C=C - is the best of the set at 1.23x.
The transferable grid is transferable.

That leaves the programme with **no unmeasured objection and no unbuilt component except
the gradient.**

**§12 revises §11 on both of the things §11 brought back.** The ridge floor is not a
property of the metric's rank but of the ridge's *filter shape*: `(S + lambda I)^-1` hands
the numerically null directions the largest gain in the operator, `1/lambda`, and `Z =
D^T D` carries `S^-1` twice, so the gradient noise grows as `1/lambda^2` - measured, on
water, as `lambda^-1.9`. Replacing it with the damped filter `sigma/(sigma^2 + mu^2)`,
which is analytic in `S` like the ridge and suppresses those directions like the
truncation, drops the usable `lambda` by three to five decades and the accuracy it costs
from 166-1808 uHa to under 5 on all five grids tested, while leaving the PES smoother than
the ridge did. And §11's torques were mostly the regulariser: continued to small `lambda`
with a `pinv` control, water's 188 and 5592 uHa/rad go to 0.001 and 0.002 - water is
rank-saturated, so its torque was always going to be zero - and the ghost-to-blocked ratio
on a molecule that does not saturate is **4.5x, not 30x**. The angular-momentum leak on
the real object is **3.1e-3 bohr/rad**, not 1.35e-1.

**§13 closes the orientation question and moves it off the critical path.** The ladder
§12 asked for, run under the filter §12 mandated and against a `pinv` control, says the
torque converges away with grid size in every mode - the committed ladder that said
otherwise was taken at `ridge 1e-4`, three decades outside where a torque means anything.
What separates the transferable grid from the in-molecule one is the **weight footing**,
not the transfer: `ghost` converges like `blocked1` and not like weighted `blocked`. The
trajectory then says what the residue costs. Over 2.5 ps of thermal tumbling the torque is
**95% incoherent**; the component along `L` that could spin the molecule up is pinned by
energy conservation at tens of uHa, the size of the orientational potential itself; what
accumulates is **4.5 degrees of rotation-axis tilt**, which is artificial rotational
diffusion rather than heating. And the control that reframes the whole question:
**RKS/PBE on the level-0 Becke grid these fits are pruned from loses angular momentum
about eight times faster at a picosecond, and 4.4x faster at 100 fs.** Lab-fixed
atom-centred quadrature is not an acTHC defect - it is what every atom-centred grid
does, and the pruned, reweighted support is quieter at it than its own parent. The one thing §13 does make urgent is the
**orbital response**: on the fixed-orbital surface the force is not the gradient of the
propagated energy, and that alone breaks rotational invariance at ~8300 uHa/rad, fifty
times the frozen grid's own torque and identical on a grid with five thousand times less.

**§14 removes the reason ghosts exist, and answers the question §13 left as the
highest-leverage one.** §8's case for a training set was that the free-atom fit is
ill-posed - an equation count, 15 for hydrogen in cc-pVDZ. Fitting the free atom's own
*two-electron integrals* instead of its overlap matrix is `O(n_AO^4)` equations rather
than `O(n_AO^2)`, still linear in the weights, still the same solver, and it lifts the
support ceiling **4.6-6.5x** across H, C, N and O. The support that comes out - no ghosts,
no partners, no training molecules, nothing but the isolated atom - costs **0.89-1.16x**
the in-molecule `blocked` grid at matched accuracy and matched weight footing, where
thirteen fabricated ghost environments cost 1.11-1.78x. And the weights that same solve
produces are the first transferable ones that *help*: `eriw` converges on the torque
ladder at **1481x**, alongside weighted `blocked`'s 17667x and against 13-40x for
`blocked1`, `ghost`, `ghostw` and `eri`, reaching **0.19 uHa/rad on 565 points** - below
the 0.24 §13 measured for the complete 3284-point parent grid. §13's sentence that "what a
transferable support cannot do is carry in-molecule weights" was true of *ghost* weights
and is not true in general.

**§15 removes the prerequisite, and the application runs.** §13's one urgent item was
the orbital response, on the grounds that the fixed-orbital force is not the gradient of
the propagated energy and breaks rotational invariance fifty times harder than the
transferable grid does. What made it more than plumbing was not the coupled-perturbed
solve but the Laplace factors: carrying orbital *energies*, they make the energy
non-invariant under a rotation among the occupied orbitals, so the response would have
needed blocks of `U` that no CPHF solver returns. Written with `Theta_o = w^(1/4)
exp(t F_oo)` the energy is the same at the canonical point and manifestly invariant, and
the standard occupied-virtual response is all that is left. The result verifies by two
independent routes that share no machinery - a finite difference of the fully relaxed
energy, converging quadratically, and section 11's rotational identity, which now closes
to **6.1e-9** relative where the fixed-orbital force sits at 9.1e-3 - and an NVE
trajectory on the THC surface conserves energy to **0.247 uHa per step against the
fixed-orbital force's 21.68**. The programme has no unbuilt component left.

**§18 closes §17's defect and, with it, the question §13 and §16 left as the programme's
last structural one.** The preconditioner §17 found - `E = diag(1 / sqrt(diag S))`, read
off the metric at runtime - had no adjoint, so it was usable for single points and not
for the application. It has one now, verified against a finite difference, against the
identity that an exact inverse cannot see a similarity transform, and end to end on an
assembled nuclear gradient and torque. The ladder it unlocks says more than that the
scheme is differentiable. On methanol at matched supports and matched draws, `ghost` -
the transferable object, at `w = 1` - goes from **15.15 to 0.21 uHa/rad** at 702 points
and from 21.37 to **0.09** at 627, below the 0.24 §13 measures for the complete
3284-point parent grid, while its energy drops onto the same +2.75 uHa floor the
in-molecule fit reaches. The reason is exact rather than empirical: `diag(S(w))_PP =
w_P diag(S(1))_PP`, so `E(w) S(w) E(w) = E(1) S(1) E(1)` for any positive weights, and
the preconditioned metric inverse is invariant to the weight footing that §13 localised
the entire transferable-support gap to and §16 showed could not be fixed by transplanting
a weight set. `blocked` and `blocked1` - the same support at two footings - agree to ten
digits in the energy under it.

What remains to be measured, in order:

1. ~~**The ghost gap.**~~ **Done - see §8, and the answer is yes.** 1.1-1.8x over
   `blocked`, shrinking with system size; free-atom fits fail structurally.
2. ~~**Transferability.**~~ **Done - see §9 and §10, and the answer is yes.** §9 did
   the first half: the ghost ensemble is worth ~1.35x on methanol and ~1.05x by ethanol,
   so the choice amortises, and it leaves a **rank ceiling** behind that caps reachable
   accuracy regardless of threshold. §10 did the second and larger half - whether one
   element's point set serves environments it was not fitted in - across ten molecules,
   and found in-range and out-of-range indistinguishable (1.49x against 1.50x at 10 uHa).
   It also settles the sub-question about a partner element the fit never saw: nitrogen
   costs nothing, and *adding* it to the partner list makes every properly resolved
   molecule worse, including the nitrile it was meant to help. Environment count supplies
   rank; partner variety at fixed environment count only dilutes.
3. ~~**A gradient.**~~ **Done - see §11, and §12 for what it cost to make it usable.**
   It exists, verifies to machine precision, and after §12 it no longer has to be bought
   with a hundred-fold worse energy: the damped pseudoinverse puts the usable `lambda`
   three to five decades below the ridge's, at microhartree accuracy, on a surface that
   is smoother than the ridge's. What remains unbuilt is the orbital response, which is
   standard DF-MP2 machinery.
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
6. ~~**Does the torque §12 is left holding converge away with grid size?**~~ **Done -
   see §13, and the answer is yes, in every mode.** Methanol at `pinv`, over 223-235 to
   670-702 points: `blocked` 357 -> 0.02 uHa/rad, `blocked1` 357 -> 9.0, `ghost`
   969 -> 27, `ghostw` 698 -> 52. Water is 0.00 everywhere including its unpruned parent.
   The ladder that said otherwise was taken at `ridge` `1e-4`, which §12 disqualifies -
   `--scheme` did not exist when that data was produced. §13 also settles what the gap
   between the transferable grid and the in-molecule one is made of: the **weight
   footing**, since `ghost` converges like `blocked1` (35.6x against 39.6x) and not like
   weighted `blocked` (17667x). §14 qualifies the size of that: at matched support the
   weights are worth 2-2.5x, not the 450x this pair implies, and a torque quoted at one
   rung is not a reliable statistic to take a ratio from.
7. ~~**What does the torque do over a trajectory?**~~ **Done - see §13.** It reorients
   rather than heats. Over 2.5 ps of thermal tumbling the torque is 95% incoherent, the
   component along `L` that could spin the molecule up is held at +0.36 hbar by energy
   conservation, and what accumulates is 4.5 degrees of rotation-axis tilt. Water, which
   saturates, leaks 0.0027 hbar. And the comparison nobody had made: **plain RKS/PBE on
   the same level-0 Becke grid the THC fits are pruned from loses angular momentum about
   eight times faster at a picosecond**, so the frozen support is quieter in
   orientation than its own parent quadrature.
8. ~~**Fit per-element weights against an objective LS-THC cares about.**~~ **Done -
   see §14, and the answer is that it works.** §13 made this the highest-leverage
   unrun experiment, on the grounds that the whole remaining orientation gap is the
   weight footing and that `ghostw` - the only transferable weight set tried - was
   *worse* than `w = 1`. Weights fitted against the free atom's own ERIs are not: they
   converge 1481x over the ladder against `ghostw`'s 13.4x, and land at 0.19 uHa/rad on
   565 points against `ghostw`'s 52.3 on 702. The mechanism §12 proposed for why weights
   matter at all - conditioning work on a metric the regulariser is not equivariant to -
   predicts exactly this: an ERI target is a far better proxy for what the `Z` fit has to
   reproduce than an overlap target is.
9. ~~**Implement the orbital response.**~~ **Done - see §15, and the trajectory runs.**
   §13 promoted this from standard machinery left undone to the thing blocking the
   application. The obstacle turned out not to be the CPHF solve but the Laplace factors'
   dependence on orbital *energies*, which makes the energy non-invariant under
   occupied-occupied rotation and so demands response blocks no solver returns; writing
   `Theta_o = w^(1/4) exp(t F_oo)` instead is the same function at the canonical point
   and manifestly invariant, and reduces the problem to the standard occupied-virtual
   response. The relaxed gradient finite-differences quadratically against a re-converged
   SCF, the rotational identity closes to **6.1e-9** relative against the fixed-orbital
   force's 9.1e-3, and NVE on the THC surface drifts **0.247 uHa/step against 21.68**.

10. **Make the response cheap, and re-run §13's leak on the propagated surface.** Two
    things §15 leaves. The response solves the coupled-perturbed equations once per
    nuclear degree of freedom - `3 N` solves where a Z-vector formulation needs one, which
    is the difference between a reference implementation and production dynamics;
    `response_lagrangian` already returns the intermediate it would contract. And every
    leak number in §13 was measured along a *DF-RHF* trajectory with the frozen grid's
    torque integrated along it, because propagating on the THC surface was not possible.
    It now is, so the 4.5 degrees of axis tilt in 2.5 ps can be measured on the surface it
    was always meant to describe rather than inferred first-order from another one.

11. ~~**Can a support and its weights be chosen separately?**~~ **Answered, no - see §16.**
    The control §14's warning implies but does not run: a per-element weight set fitted
    against real molecules, laid onto the ghost support it was *not* selected with,
    beats `ghostw` 0 times out of 12 on held-out molecules. Refit support and weights
    together and the same machinery wins 8/10. So the footings are not merely
    incommensurate - a weight set carried onto the wrong support is worse than none, and
    §14's 1481x belongs to the ERI *fit* rather than to weights that could be
    transplanted. §16 also prices the footing itself at 2-2.5x at matched support, well
    short of the 450x §13's single-rung `blocked`/`blocked1` pair implied.



12. ~~**Give the Jacobi preconditioner a derivative, and find out what it does to
    orientation.**~~ **Done - see §18.** §17's preconditioner recovers in the energy
    what discarding the NNLS weights costs, and had no adjoint, so nothing could be
    asked of it beyond a single point. With one, the transferable support's torque falls
    72x at 702 points and 237x at 627, its energy lands on the parent-grid floor, and the
    weight footing turns out to cancel out of the preconditioned metric inverse exactly.
    What is left is to run it on a second molecule and a second basis, and to put it on a
    trajectory - `trajectory.py --scheme damped_jacobi` now accepts it and nothing has
    run it.

## Caveats

§15 is water only, one grid, twelve steps from rest, and ridge `1e-2` rather than the
damped filter §12 mandates for gradient work. The correctness claims do not lean on any
of that - the rotational identity is exact rather than statistical, and the response is
basis- and method-general - but every claim about cost or about what a picosecond would do
does. The response costs `3 N` coupled-perturbed solves, not one.

§16 trains on three molecules and tests on three, all H/C/O in cc-pVDZ. Nitrogen is
absent, as it is from the ghost partner list, so nothing here says whether one
per-element weight set serves an element across *rows* of the periodic table rather than
across bonding within one. Three training molecules is also few enough that the held-out
gain (2-3x) could be a property of this particular split; the honest claim is that the
gain exists and transfers to unseen bonding, not that its size is converged.

The rotation spreads behind §16 are peak-to-peak over 4 draws, which is a coarse
statistic - §7 used 5 and 6 and flagged the same thing. The margins it carries the
conclusion on (0 wins out of 12 for `molw`, 7-8 out of 10 for `molfit`) are counts rather
than magnitudes for that reason.

`molfit1` in §16 is handicapped by construction, and the "what the weights are worth"
table should be read with it in mind. Its support was selected by a *weighted* fit and
then stripped of its weights; a support chosen for `w = 1` from the start would be a
different and probably better point set. So 2-2.5x is an upper bound on what the weights
buy over a genuinely `w = 1`-optimal transferable grid, and the comparison that avoids
this entirely - `molfit` against `ghost`, both self-consistent - is the 8/10 and 7/10
counts rather than the ratio table.

The `molw` ladder sits at smaller point counts than `ghost` at every threshold, because
NNLS zeroes part of a held support wherever non-negativity binds. The matched-size
interpolation only quotes inside the window both ladders cover, so this does not bias the
0/12, but it does mean `molw` was never measured at the sizes `ghost`'s top rungs reach.

§18 is one molecule, one basis, one seed and four draws, at `damped 1e-8` - while §17's
energy result, which it is the sequel to, is cc-pVTZ; the two have never been run
together. Its gain appears only once the grid resolves: on the smallest rungs the
preconditioner does nothing useful and is slightly worse. And it changes no trajectory
number in §13, because no trajectory has been run on the preconditioned surface.

cc-pVDZ only; MMFF geometries; `ov` mode and MP2 only. The rank and weight analyses are
on methanol and ethanol; §10 widens the molecule set to twelve but widens neither the
basis, the method, nor the one-geometry-per-molecule sampling.

§8 rests on two molecules. Water was run and is not quoted for the ratio: at 24 AOs its
co-density manifold saturates at ~95 points, so every grid in the comparison is
rank-saturated and all three modes reach the floor - the ratios there are artefacts of
where each ladder happens to start, not measurements. The ghost ensemble (12 icosahedral
directions, partners H/C/O at 0.90/1.05/1.45 bond lengths, cycled one combination per
direction) was chosen once and never varied; `--full-cross` and `--directions octahedron`
exist to test that choice and were not run. N was never fitted, so the ghost grids cover
H/C/O only, and no molecule outside the fitting set was tried - which is the whole of
what transferability means and is why it was, at the time, the leading open question -
**§10 has since done both**, fitting N and running ten molecules outside the fitting set.
The orientation rows are 4 random draws per grid, a coarser statistic than §7's 5-6, and
the matched-accuracy comparison is matched to 0.15 uHa in energy but not in point count.

§14 rests on two molecules (methanol and ethanol), one basis, and one parent-grid level,
and on the `quadrature` ERI target - the integrals the parent grid itself produces, chosen
so the residual is reachable and the comparison isolates the target's richness rather than
the parent's quadrature error. The `exact` target is implemented and was swept on hydrogen
only, where it saturates at 85 points against `quadrature`'s 97; the two are close enough
that sweeping it on the heavy elements was not worth the hour each it costs, so nothing
here says whether fitting the *true* integrals selects a better support than fitting the
ones the parent grid can reach. The ladders are not monotone - methanol `eri` worsens from +9.44 to
+9.86 uHa between 487 and 565 points, and ethanol `eri` stalls near +24 uHa - which is the
threshold-ladder artefact §9 already reported and is why two cells of the ratio table read
`n/a`. The torque ladder is methanol only, four draws, one seed. Nothing has been fitted
against a ghost *and* an ERI target together, which is the obvious next control, and
nothing tests whether an ERI-fitted support means anything in a different basis.

§10 carries its own caveats, recorded at the end of that section: a thin in-range group
(methanol and ethane alone, since water saturates), seven unreadable cells among the
33-38 AO molecules, an sp-carbon case resting on acetonitrile alone, and an acetonitrile
error curve that crosses the 10 uHa target three times, so that its matched-accuracy point
count is a first-crossing estimate rather than a converged one.

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

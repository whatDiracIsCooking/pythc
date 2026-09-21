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

## 14. Four levers on the ghost fit: the cage was rejected on an argument, and the argument is wrong

`levers.py`. Section 8 found that an NNLS support cannot exceed the number of equations
its target supplies, and that a free atom supplies `n_AO (n_AO + 1) / 2` of them - 15 per
hydrogen in cc-pVDZ, a cap no threshold can lift. Ghosts fix it. But a ghost fit's
support stops somewhere too, and everything in sections 8-13 was measured at one point in
the space of things that decide where. Four axes were never varied:

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
* **`blocked`'s ladder never converges at all.** Its tightest rung is 19.3 uHa above the
  floor and still falling, which makes the apparent result that `ghost` *beats* the
  in-molecule fit here (0.74-0.88x) an artefact of an unconverged denominator and not a
  measurement. The cc-pVDZ default `--blocked-thresholds` is simply the wrong ladder in
  cc-pVTZ.

So what cc-pVTZ establishes is narrow and negative: **the cage's cc-pVDZ win does not
generalise across basis sets as measured**, and the ladders need extending (`1e-6`,
`1e-7`) before either the cage's ranking or the ghost gap in a larger basis means
anything. That run is `levers_methanol_tz_wide.json`.

What does carry across is stage A's mechanism. cc-pVTZ raises the per-element ceilings
1.55-2.34x and the supports are correspondingly larger at every threshold - `ghost` spans
523-1351 points here against 223-816 on the same molecule in cc-pVDZ. The basis lever
moves what it was predicted to move. It just does not follow that a lever which buys
points buys accuracy, which is the same lesson `tetra` teaches from the other end.

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
measured for the cage**, and §13 puts the whole remaining gap to production DFT on the
weight footing and orientation rather than on the energy - so the lever that halves the
point-count tax is not yet known to help, or not to hurt, the quantity that actually
blocks AIMD.

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
   weighted `blocked` (17667x).
7. ~~**What does the torque do over a trajectory?**~~ **Done - see §13.** It reorients
   rather than heats. Over 2.5 ps of thermal tumbling the torque is 95% incoherent, the
   component along `L` that could spin the molecule up is held at +0.36 hbar by energy
   conservation, and what accumulates is 4.5 degrees of rotation-axis tilt. Water, which
   saturates, leaks 0.0027 hbar. And the comparison nobody had made: **plain RKS/PBE on
   the same level-0 Becke grid the THC fits are pruned from loses angular momentum about
   eight times faster at a picosecond**, so the frozen support is quieter in
   orientation than its own parent quadrature.
8. **Implement the orbital response.** §13 promotes this from the standard machinery §11
   left undone to a prerequisite. On the fixed-orbital surface the force is not the
   gradient of the propagated energy, and the resulting violation of rotational invariance
   is **~8300 uHa/rad** - about fifty times the transferable grid's own torque, and
   identical on a grid with five thousand times less torque of its own. No AIMD is
   possible on the present gradient whatever the grid does.


## Caveats

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

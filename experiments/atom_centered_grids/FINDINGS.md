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

---

## Verdict

The kill shot missed. The structural penalty for giving up molecular pruning is 1.2-1.7x,
not 3x, and it shrinks as molecules grow. More importantly, §3 removes most of the
difficulty from the gradient side of the proposal: freeze the point set, drop the weights,
and the nuclear derivative is elementary. §6 and §7 have since disposed of the two
remaining structural objections, in opposite ways: the metric truncation's steps were
real and are now fixable with ridge, while the missing orientation turned out not to
need a structural fix at all - it converges away with grid size.

What remains to be measured, in order:

1. **The ghost gap.** §1 bounds a scheme that sees real neighbours. Build ghost-augmented
   per-element grids for H/C/N/O and measure how much worse than `blocked` they are. This
   is the only remaining question that can still kill the idea.
2. ~~**The isotropy tax.**~~ **Done - see §7, and the answer is no.** Whole-orbit fitting
   makes each atomic grid exactly invariant under the octahedral group, for 1.2-1.6x the
   points. It never reduces the spread under *general* rotations: at matched point count
   a point-wise grid is better on both accuracy and spread. §4's anisotropy converges
   away with grid size in either mode. Keep it opt-in; revisit only for (1), where a
   ghost-fitted grid may be anisotropic in a way a real-neighbour fit is not.
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

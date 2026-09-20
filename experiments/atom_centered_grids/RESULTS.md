# Results

cc-pVDZ / cc-pVDZ-RI, level-0 Becke parent grid, `ov` mode, 10 Laplace points, DF-MP2 reference.
"Converged" means the MP2 error on the full unpruned parent grid; targets below are
deviations *from that floor*, so they isolate grid error from the RI/Laplace error.

## 1. The size ratio: 1.2-1.7x, improving with system size

Points needed to reach a given accuracy, interpolated log-log.

| target | water | methanol | ethanol | alanine |
| --- | --- | --- | --- | --- |
| 50 uHa | 91 / 156 = **1.71x** | 186 / 301 = **1.62x** | 350 / 477 = **1.36x** | 685 / 847 = **1.24x** |
| 20 uHa | 91 / 156 = **1.71x** | 193 / 301 = **1.56x** | 383 / 510 = **1.33x** | 743 / 1065 = **1.43x** |
| 10 uHa | 91 / 156 = **1.71x** | 199 / 301 = **1.51x** | 403 / 561 = **1.39x** | 819 / 1161 = **1.42x** |
| 5 uHa | 91 / 156 = **1.71x** | 205 / 301 = **1.47x** | 423 / 610 = **1.44x** | - |
| 2 uHa | 91 / 156 = **1.71x** | 212 / 308 = **1.45x** | 509 / 652 = **1.28x** | - |

(global / blocked = ratio. Water saturates at the loosest grid tested, so its ratio is
a single point, not a curve.)

**1.2-1.7x on points is ~1.5-2.9x on the `n_P^2` parts** - not the 4-9x that a 2-3x
inflation would have cost. This is well inside the range where the scheme is worth
building, and the trend with system size runs the right way.

Remember this is a **lower bound**: the blocked fit sees each atom's real neighbours and
prunes point-wise. A frozen, ghost-fitted, orbit-wise-pruned grid pays this *plus* the
ghost-approximation and isotropy costs.

Blocked is also much cheaper to fit - alanine at 1e-5: **39 s blocked vs 164 s global**,
and the gap widens with size, as the linear-vs-cubic scaling predicts.

## 2. Why: LS-THC cares about span, not quadrature

`rank.py` on methanol (`n_occ*n_vir = 351`), co-density rank and the conditioning of the
metric LS-THC pseudo-inverts:

| grid | points | rank@1e-6 | metric rank | -log10 min eig |
| --- | --- | --- | --- | --- |
| global 1e-3 | 180 | 180 | 180 | 6.4 |
| global 1e-4 | 240 | 240 | 240 | 7.8 |
| global 1e-5 | 333 | 333 | 290 | 11.5 |
| blocked 1e-3 | 301 | 301 | 256 | 10.7 |
| blocked 3e-4 | 410 | 345 | 283 | 20.3 |
| blocked 1e-4 | 491 | 350 | 294 | 21.1 |
| blocked 1e-5 | 670 | 351 | 303 | 21.8 |

Two things fall out.

**Global NNLS is a perfect rank-revealing selector**: `rank == n_points` at every threshold.
Every point it keeps adds exactly one independent co-density direction. Blocked loses that
above ~350 points, where independently-fitted atoms start duplicating each other's
directions.

**The conditioning penalty is severe and is the real cost of unioning per-atom grids.**
The metric's smallest eigenvalue drops from ~1e-8..1e-11 (global) to ~1e-20..1e-22
(blocked). Ridge regularisation of the Z-fit is therefore **mandatory** for an
atom-centred scheme, not an optional refinement.

## 3. Anisotropy is real, and the overlap RMSD does not see it

`rotate.py` fits the blocked grid once, then spins each atom's point set about its **own**
nucleus - molecule, AOs and weights untouched - simulating a frozen grid attached at
arbitrary orientation.

| system | grid | regime | spread over 6 random orientations |
| --- | --- | --- | --- |
| water | 214 pts, 1e-4 | rank-saturated | **0.00 uHa** |
| methanol | 243 pts, 3e-3 | rank-limited | **39.0 uHa** peak-to-peak, std 15.3 |
| ethanol | 477 pts, 1e-3 | rank-limited | **26.6 uHa** peak-to-peak, std 9.0 |

Orientation is free once the grid oversamples the co-density manifold, and costs real
energy exactly where the scheme wants to live - in the compact, rank-limited regime.

The shell structure says why. Across methanol and ethanol, **not one angular shell survives
intact**: every retained shell keeps only 8-20% of its Lebedev points. NNLS shreds the
octahedral orbits that make the parent grid effectively isotropic, so orbit-wise
(group-sparsity) pruning is needed - and its coarser granularity is a further point-count
tax on top of section 1.

**The overlap RMSD is a weak proxy for THC accuracy.** Rotating water's grid degraded
`rmsd_S` from 2.1e-3 to 2.6e-1 - a factor of 125 - while leaving the MP2 energy unchanged
to 12 digits. Matching grids on the NNLS fit target therefore over-constrains relative to
what LS-THC actually needs, which is span and conditioning.

## Verdict

The kill shot missed: the structural penalty for giving up molecular pruning is 1.2-1.7x,
not 3x, and it shrinks as molecules grow. The remaining risks are conditioning (fixable,
cheaply, with ridge) and anisotropy (fixable with orbit-wise pruning, at a cost that has
not been measured yet). Building ghost-augmented per-element grids is now worth doing.

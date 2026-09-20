# Handoff: atom-centered THC grids

Read this first, then [`FINDINGS.md`](FINDINGS.md) for the measurements. This file records
what the idea is, what was tried, **which parts of the prior design reasoning the
measurements overturned**, and what to do next. The point of §3 is that a fresh session
should not re-derive machinery that turned out to be unnecessary.

## 1. The idea

**In one sentence:** run NNLS per atom (with ghost atoms to keep bonding-region points
alive) to select a point set per element, **discard the fitted weights and keep only the
points**, build a molecule's grid as the union of its atoms' point sets, and - given
section 4(1), now done - get analytic nuclear gradients and a smooth PES for free.

Fit a THC grid once per element, offline, and translate it rigidly into any molecule - the
acCD move (atomic Cholesky decomposition: do the pivoted selection once per element
offline, emit ordinary atom-centred basis functions, never re-select at runtime), applied
to grid points instead of auxiliary basis functions.

The prize is smooth potential energy surfaces. THC point selection is discrete
(QRCP/pivoted-Cholesky/NNLS), so it is not differentiable; the standard workaround is to
freeze the selection at the reference geometry and differentiate the rest. A grid frozen
*per element* needs no such assumption at all - nothing is re-selected as nuclei move.

The price is compactness: per-atom grids cannot share bonding-region points between
neighbours and must cover bond directions the atom may not have. (They were also expected
to need coarser orbit-wise pruning to stay isotropic; section 4(2) measured that and it
turned out not to be necessary.) Since `Z` is `n_P x n_P`, the point-count penalty is what
decides the programme. **It is now measured end to end: 1.6-2.7x the points of a global
molecular fit, i.e. 2.6-7.3x on the `n_P^2` parts, improving with system size** - see
section 4(3), which is done.

## 2. What was attempted

Eight experiments, all in this directory, all on cc-pVDZ / cc-pVDZ-RI, level-0 Becke
parent grid, `ov` mode, 10 Laplace points, against DF-MP2. Geometries from RDKit ETKDG +
MMFF.

| script | question | headline |
| --- | --- | --- |
| `sweep.py` + `analyse.py` | how many more points does per-atom fitting need? | **1.2-1.7x**, improving with size |
| `rank.py` | why - and what does it cost in conditioning? | global is a perfect rank-revealing selector; blocked metric min-eig 1e-20 vs 1e-8 |
| `rotate.py` | does an arbitrarily-oriented frozen grid cost energy? | 0 uHa when rank-saturated, **27-39 uHa** when rank-limited; **no Lebedev shell survives intact** |
| `weights.py` | do the fitted weights matter, or only the selection? | **only the selection**; all-ones weights are bit-identical or better |
| `ridge.py` | what does ridge cost against the truncated pseudoinverse? | **0.1-2.2 uHa at `lambda = 1e-8`**; the truncation threshold itself changes nothing |
| `scan.py` | does the truncation actually put steps in the PES? | **yes, ~0.5 uHa, exactly at the eigenvalue crossings**; ridge removes them |
| `orbits.py` | does fitting whole orbits fix the orientation dependence? | **no.** Exact under the octahedral group, but never better than point-wise at matched size |
| `ghosts.py` | what does an offline, ghost-fitted per-element grid cost? | **1.1-1.8x over `blocked`**, shrinking with size; free-atom fits fail structurally |

The key methodological move: `NNLSGrid(blocked=True)` already fits each atomic sub-grid
independently, so the penalty for giving up molecular pruning is measurable today. Because
the blocked fit sees each atom's **real** neighbours and prunes **point-wise**, its
inflation is a strict **lower bound** on a frozen atom-centred scheme. It could have killed
the idea cheaply; it did not. `ghosts.py` then built the frozen scheme itself and measured
how far above that bound it actually lands.

## 3. Prior design reasoning: what survived, what did not

This programme began as a long design conversation before anything was measured. Its
conclusions are listed here with their current status, so they are not re-derived.

| claim from the design discussion | status |
| --- | --- |
| Freeze the discrete point selection, differentiate the smooth remainder (AO derivative + rigid point translation) | **Stands.** This is what Song & Martínez actually do. |
| LS-THC needs no Becke partition function, since `Z` is fitted rather than quadratured | **Confirmed, and then some** - see the weights result below. |
| NNLS weights need active-set freezing, QP sensitivity theory, strict complementarity | **Superseded.** The weights barely affect the energy at all; there is nothing to freeze. They were doing *conditioning* work, though, so `w = 1` must be paired with ridge - see section 4(1). |
| Reparametrise `w = theta^2` to remove the inequality constraint | **Moot**, and it was the wrong power: this codebase builds `X = w^(1/4) phi`, so `theta^2` still leaves `X = |theta|^(1/2) phi`, singular at 0. If a weight fit is ever kept, parametrise the *collocation amplitude* `X = t phi` with `w = t^4`; the whole pipeline is then polynomial in `t`. |
| Add `sum_P w_P = 1`; equality constraints differentiate cleanly | **Moot** for the same reason. |
| Isolated-atom NNLS will discard exactly the tail points a bond needs; fix with ghost atoms | **Measured. Right conclusion, wrong mechanism, and the mechanism matters.** Ghosts are indeed necessary and they work (1.1-1.8x over `blocked`). But the free-atom fit does not fail by misplacing points - it fails because its target supplies only `n_AO(n_AO+1)/2` equations, so NNLS cannot retain more than **15 points per hydrogen in cc-pVDZ** at any threshold. The predicted anisotropy is real too (34 uHa rotation spread at the ceiling), it just is not what stops the scheme. See section 4(3) and FINDINGS section 8. |
| Octahedral ghosts suffice because p orbitals are octahedral | **Rejected in the discussion itself, correctly** - points are sampling locations, not functions, and do not superpose. |
| Frozen per-atom grids risk orientation-dependent energies | **Confirmed, then deflated, and the deflation now holds for the real object.** No angular shell survives intact, and rank-limited grids shift 27-39 uHa under random per-atom rotation, but the shift converges away with grid size (methanol 218 -> 7.1 -> 0.01 uHa from 175 to 670 points). Ghost-fitted grids are 2-5x more orientation-dependent than `blocked` **at matched point count** - and identical **at matched accuracy** (882 ghost points: +5.14 uHa, 0.94 uHa spread; 491 blocked points: +5.29 uHa, 0.91 uHa spread). Still a symptom of rank limitation, not a separate defect. |
| Fix isotropy with orbit-wise (group-sparsity) pruning over (radial shell, Lebedev orbit) blocks | **Implemented, measured, and rejected as a fix.** It delivers exactly what it promised - whole orbits, exact octahedral invariance - and that turns out not to be the useful property. At matched point count a point-wise grid is better on both accuracy and rotation spread. See section 4(2) and FINDINGS section 7. |
| Grid inflation will be 2-3x, i.e. 4-9x on the `n_P^2` parts | **Roughly right after all.** The 1.2-1.7x of section 4 priced only *blocked vs global* - one of the two things a transferable grid gives up. Adding the ghost gap gives **1.6-2.7x, i.e. 2.6-7.3x**, improving with system size. Do not quote the 1.2-1.7x figure as the cost of the scheme; it is the cost of half of it. |
| Unioned per-atom grids will wreck metric conditioning; use ridge, not eigenvalue truncation | **Confirmed, and now implemented and measured** - see section 4(1). Min eigenvalue 1e-20 vs 1e-8; the truncation does put ~0.5 uHa steps in the PES at the eigenvalue crossings, and ridge removes them. One correction: ridge was expected to be a pure improvement, but it has a **floor** the truncation does not - too small a `lambda` inverts the null space's rounding noise. |
| The overlap-metric target inherits a decade of validation from the 2013 DVR paper | **Substantially undermined.** `rmsd_S` is close to orthogonal to THC accuracy: it degraded 125x under rotation and ~2000x under all-ones weights with no loss (sometimes a gain) in MP2 accuracy. |
| PyTHC is the fastest route to the decisive experiment | **Correct** - `blocked=True` made it cheaper still. |

### The one that changes the most

Because `Z` is fitted by least squares, rescaling `X` by a positive diagonal is absorbed
exactly (`S -> D^2 S D^2`, `Z -> D^-2 Z D^-2`). Measured: wherever the metric is full-rank,
NNLS / all-ones / uniform weights give **bit-identical** energies. Weights only matter when
their dynamic range costs numerical rank under the pseudoinverse truncation.

So set `w = 1`. Then `X = phi(r_P)`, there is no weight to differentiate, and the entire
active-set / reparametrisation apparatus disappears. Only the point **selection** needs
freezing - which a transferable scheme was going to freeze anyway. NNLS remains valuable,
but as a *point selector*, not a weight fitter.

## 4. Next directions, in order

**(1), (2) and (3) are all done.** (1) was a real prerequisite: the metric truncation was
the last remaining source of PES non-smoothness, and it is now measured and fixed. (2) was
believed to be a prerequisite too - the thing that makes "attach the atomic grid rigidly"
well-defined - and it is not; orientation dependence is a grid-size problem, not a
structural one. (3) was the one that could still have killed the idea, and it did not:
ghost-fitted per-element grids exist, they work, and they cost 1.1-1.8x over `blocked`.

**Nothing left can kill the idea cheaply.** (4) transferability and (5) a gradient are now
the work, and they are ordinary development rather than go/no-go experiments.

The target pipeline, for orientation:

| stage | when | geometry dependence |
| --- | --- | --- |
| select points per element (NNLS + ghosts) | **offline, once** | none |
| attach point sets to nuclei, take the union | per geometry | rigid translation only |
| build `X = phi(r_P)` - no weights | per geometry | AO derivative + point translation |
| fit `Z` by least squares against exact ERIs | per geometry | smooth, RI-like chain rule |

Nothing discrete happens at runtime. That is the whole point, and (1) has now made it true
of the `Z` fit as well. **`ghosts.py` implements every row of that table** - the offline
selection, the rigid attachment, `w = 1`, ridge - and measures what the pipeline costs
end to end. What is missing is no longer a mechanism or a measurement but the nuclear
derivative itself, which is (5).

**(1) Ridge instead of truncation. DONE** - `lib.ridge_inv`, `lib.ridge_inv_sqrt`, and
`metric_ridge` / `aux_ridge` on `LS_RI_THC`, reaching the inversion through
`ls_thc_funcs.invert_metric`. Measured in `ridge.py` and `scan.py`, written up in
FINDINGS.md section 6. The default is unchanged, so nothing that came before is
invalidated; ridge is opt-in.

What it settled, and what it did not:

* **The steps were real.** With the grid frozen, scanning a bond, the truncation retains
  294-296 eigenvalues on methanol and changes at 2 of 60 steps - and the two largest
  second differences in the whole error curve sit exactly on those two steps, 80x the
  background. Ethanol: one crossing in 30 steps, 22x. Under ridge the same points are
  indistinguishable from their neighbours.
* **They were small** - about 0.5 uHa - and that is the mildly deflationary part. The
  truncated directions carry almost no energy: changing `epsilon` from 1e-8 to 1e-14,
  retaining 412 to 477 of ethanol's 477 eigenvalues, leaves the correlation energy
  identical to nine decimals. The cutoff was never buying accuracy. That is an argument
  for replacing it rather than tuning it, but it is not a large error being fixed.
* **The frozen grid is what makes it the leading defect.** Re-selecting the grid at every
  geometry gives ~4 uHa of second difference *everywhere*, three orders of magnitude
  above the truncation's steps. (1) only matters because the freeze comes first; do not
  quote the 0.5 uHa number without that context.
* **Ridge has a floor the truncation does not**, which was not anticipated. Where the
  pseudoinverse discards the numerically null directions, ridge inverts them at
  `1/lambda`. Water's blocked grid has 23 exactly null directions, and below
  `shift/maxeig ~ 1e-15` the energy becomes noise - several hundred uHa, not reproducible
  between runs differing only in SCF convergence tolerance.
* **Smoothness finds that floor before accuracy does.** On water the static energies at
  `lambda` = 1e-8, 1e-10 and 1e-12 agree within 0.3 uHa, but the scan's `max|d2|` goes
  0.003 -> 0.37 -> 27.8. An energy that looks converged can sit on a curve whose
  derivative is noise, so **do not pick `lambda` from an accuracy table alone.**
* **Use `lambda = 1e-8`** at the default `"trace"` scaling. Smooth on all three systems
  tested, costing 0.1-2.2 uHa. `1e-6` is too strong (+3 to +42 uHa), `1e-4` useless.

One design decision worth not re-litigating: `lambda` is scaled by `tr(S)/n`, the mean
eigenvalue, not by `max(eig(S))` as `pinv`'s `epsilon` is. The mean is linear in `S` and
so analytic in the nuclear coordinates; the largest eigenvalue has a kink wherever it
becomes degenerate, which would reintroduce in miniature the thing being removed.
`"max_eig"` scaling exists for comparing against `epsilon` directly.

Left undone here: nothing checks that an *analytic* gradient agrees with these curves,
because there is no analytic gradient yet - see (5). The scans move a terminal hydrogen
only, so a heavy-atom displacement may cross more often than 2 in 60 steps. And the
aux Coulomb metric has the same truncation; `aux_ridge` handles it, but it was never the
binding constraint and was not studied separately.

**(2) Group-NNLS over (radial shell, angular orbit) blocks.** The reason this matters is
**rotational invariance, not accuracy**. Attaching an atomic grid rigidly requires choosing
an orientation, and there are only two options, both bad for an anisotropic point set:

* *fixed lab orientation* - rotating the molecule changes the energy, so the PES is not
  rotationally invariant and the gradient carries spurious torques; AIMD will not conserve
  angular momentum;
* *molecule-dependent local frame* (aligned to bonds, say) - the frame definition is itself
  geometry-dependent and can switch discontinuously, so the kinks come back.

The way out is for the grid to be close enough to isotropic that the choice does not
matter. Keeping whole Lebedev orbits makes the point set exactly invariant under the
octahedral group, so orientation stops mattering to within the quadrature's band limit.
Measured, **no shell survives intact** (8-20% of each kept), which is why the rank-limited
rotation spread is 27-39 uHa. So this is what makes the rigid-attachment step well-defined,
not a cosmetic improvement.

Needs: keep the shell/orbit structure from `gen_atomic_grids` (currently flattened in
`grid.py`), and a group-sparsity variant of the Lawson-Hanson loop in `decomp/nnls.py`.
Re-run `sweep.py` and `rotate.py` afterwards - expect a larger grid and a smaller rotation
spread, and quantify that trade, since the extra points come straight off the 1.2-1.7x
budget in FINDINGS.md section 1.

**(3) The ghost gap. DONE** - `ghosts.py`, written up in FINDINGS.md section 8. The answer
is **1.1-1.8x over `blocked`** at matched accuracy, shrinking with system size (methanol
1.08-1.78x, ethanol 1.11-1.55x), which compounds with section 1 to **1.6-2.7x over a
global molecular fit**. The idea survives.

What it settled, and what it did not:

* **The object is real now.** `ghosts.py` builds an actual transferable grid: one point
  set per element, fitted offline against ghost neighbours, indexed into the element's
  level-0 atomic grid, translated onto nuclei at runtime with `w = 1` and nothing
  re-selected. Everything before this measured a proxy.
* **Free-atom fits fail structurally, not by a little.** This is the result to carry
  forward. NNLS retains at most `min(n_equations, n_variables)` points and an isolated
  atom's overlap target has only `n_AO(n_AO+1)/2` equations - **15 for hydrogen in
  cc-pVDZ**, confirmed exactly at threshold `1e-12`. Methanol's free-atom grid therefore
  cannot exceed 244 points however it is tuned, and at that ceiling it is still 67 uHa
  above the floor. **A ghost is not a correction to the free-atom fit; it is what makes
  the fit well posed.** Whatever replaces this ghost ensemble must keep supplying
  equations, not just amplitude.
* **The stacking is the mechanism.** 13 environments (the free atom plus 12 icosahedral
  directions cycling H/C/O partners at 0.90/1.05/1.45 bond lengths) go into ONE NNLS
  solve over the shared atomic grid, via `decomp.nnls.StackedOperator`. That is what
  takes the solver from 105 equations to ~3500. Per-environment targets are normalised
  to unit norm (`stack_targets`) so a heavy partner does not dominate a light one.
* **The thresholds are not on the blocked scale.** The stacked target is normalised, so
  the KKT ladder is different: use `1e-3 .. 1e-6` for ghost fits against `1e-3 .. 1e-5`
  blocked. `ghosts.py --calibrate H,C,O` prints points per element with no SCF in
  seconds; do that before spending single points.
* **Orientation is settled for the real object too** - see the section 4(2) entry and
  FINDINGS section 8. Ghost grids are 2-5x more anisotropic than `blocked` at matched
  *size* and indistinguishable at matched *accuracy*. Orbit grouping does not get a
  second chance.
* **Support agreement is low and it does not matter.** The ghost fit reproduces only
  24-31% of what `blocked` picks for C and O. Expected, given sections 2, 3 and 5: what
  LS-THC wants is a point set that spans the co-density manifold, not any particular
  nodes. Do not use support overlap as a quality metric; only point count at matched
  energy means anything.

Left undone here: the ghost ensemble was chosen once and never varied. `--full-cross`
(every direction x partner x distance) and `--directions octahedron` exist and were not
run, so nothing establishes that the answer is insensitive to the choice - which is the
first thing (4) should check. N was never fitted; the grids cover H/C/O. Water was run
and is not quoted, because at 24 AOs every grid in the comparison is rank-saturated and
all three modes reach the floor. And no molecule outside the fitting set was tried, which
is the whole of what (4) means.

**(4) Transferability - now the leading question.** Two halves, and (3) left both open.
First: does the *ensemble* matter? Re-run `ghosts.py` with `--full-cross` and
`--directions octahedron` and see whether the supports or the point counts move. If they
do not, one cheap ensemble suffices and the offline stage is finished. Second: does one
element's point set serve environments it was not fitted in? Fit O once and use it in
water, methanol and formaldehyde; fit C once and use it in an sp3, an sp2 and an sp
environment. The low support agreement with `blocked` (24-31%) cuts both ways here - it
may mean the choice of points hardly matters, or it may mean the answer is unstable, and
those predict opposite outcomes for this experiment. Also worth adding N, which was
skipped, and checking whether a partner element the fit never saw costs anything.

**(5) Actually compute a gradient.** Still nothing here computes one, and this is now the
other half of the remaining work. `scan.py` measures the *curve* rather than arguing about
it structurally, so the smoothness claim is no longer purely theoretical - but a
finite-difference-vs-analytic check would test the implementation, which the scan cannot.
`scan.py` is the natural harness: it already walks a frozen grid along a bond at fixed
ridge, which is the reference curve such a check needs. What (3) adds is that the grids to
run it on now exist: `ghosts.py`'s per-element supports are the first point sets in this
directory that are genuinely frozen with respect to geometry, so a gradient computed on
one has no re-selection term to omit and no excuse for disagreeing with the curve.

## 5. Open questions

* Does the 1.2-1.7x ratio hold in larger basis sets (cc-pVTZ) and for HF exchange rather
  than MP2 correlation? Everything here is cc-pVDZ / `ov` / MP2.
* The ratio improves from water to alanine. Does it keep improving, or turn around?
* If weights are irrelevant, is NNLS still the best *selector*? Pivoted Cholesky and QRCP
  select points directly and are already implemented (`ls_ri_cholesky.py`, `ls_ri_qrcp.py`).
  A like-for-like selector comparison at matched point count has not been done here.
  Section 4(3) sharpens this: NNLS's point ceiling is its equation count, which is a
  property of the *target*, not of the selector. A selector working directly on the
  co-density would not have that ceiling, and might not need ghosts at all - or might
  need them for a completely different reason.
* Does the ghost ensemble matter? 12 icosahedral directions, partners H/C/O at three
  bond-length multiples, one combination cycled per direction, chosen once and never
  varied. `--full-cross` and `--directions octahedron` are implemented and unrun. This is
  the cheapest open question in the list and it gates 4(4).
* How far can the offline object be pushed before it stops being a point set? Right now
  it is a list of indices into the element's level-0 atomic grid. Nothing forces that -
  the points could come from a finer parent grid, or from several parent grids unioned -
  but indexing into a fixed element grid is what makes the object trivially portable, and
  giving that up should be a deliberate choice.
* Does `rmsd_S` ever track accuracy, or should grid comparisons move to rank + min
  eigenvalue wholesale?
* Does the parent-grid floor move between identical runs? Methanol's came out `+2.68` uHa
  in one `ghosts.py` run and `+2.86` in another differing only in its threshold list.
  0.18 uHa does not threaten any conclusion here, but section 4(1) says ridge inverts
  near-null directions at `1/lambda`, and a 3288-point parent grid is the most redundant
  metric in the set. `ridge.py` sets `mf.conv_tol = 1e-12`; `sweep.py` and `ghosts.py` do
  not. Noticed, not diagnosed - and it is exactly the kind of thing that matters more for
  a gradient (5) than for an energy.
* Should `metric_ridge` become the default rather than an opt-in? It costs a few uHa and
  removes a discontinuity, which is the right trade for gradient work and the wrong one
  for reproducing published single-point energies. (The argument that 4(2)'s larger, more
  redundant grids would make the near-null crowd worse no longer applies, since those
  grids are not the recommended path - but the orbit-grouped grids do exist and were not
  run through `rank.py`, so what they do to the metric is unmeasured.)
* How small does the rotation spread have to be? Section 4(2) reports it falling to 0.01
  uHa on methanol's 670-point grid, which looks like enough - but nothing here converts a
  spread into a torque, and a spurious torque that is negligible for a single point energy
  may still spoil angular-momentum conservation over an AIMD trajectory. The quantity that
  matters is `dE/dtheta` at the attachment orientation, not the peak-to-peak over draws.
* The orbit-grouped fit's octahedral invariance is exact in the algebra but lands at
  0.00-0.01 uHa in practice, and ridge does *not* remove it (methanol's 732-point grid:
  -0.0071 uHa truncated, -0.0170 uHa at `metric_ridge = 1e-8`). That is round-off
  amplified by the metric's ~1e-20 smallest eigenvalue, so it is a conditioning floor on
  how exact *any* symmetry of a unioned per-atom grid can be. Whether it also floors the
  gradient at a comparable relative size is unmeasured, and matters more than the energy
  does.
* Is a *fixed* `lambda` right, or should it track the grid? `shift/maxeig` at
  `lambda = 1e-8` came out at 1.7e-10 to 3.3e-10 across three systems - close enough that
  fixed looks defensible, but three systems in one basis is not a strong test. A much
  larger molecule, or cc-pVTZ, would say.

## 6. Literature

Verified by search during this session:

* **Kokkila Schumacher, Hohenstein, Parrish, Wang, Martínez**, *THC-MP2: Grid Optimization
  and Reaction Energies*, JCTC **11**, 3042-3052 (2015), DOI 10.1021/acs.jctc.5b00272.
  Generates grids for first-row atoms at **<100 points/atom** with negligible energy error,
  cc-pVDZ and cc-pVTZ. **This is prior art for per-atom, per-basis THC grids** - read the SI
  first. For calibration, the per-atom densities measured here bracket that figure (water
  global ~41/atom, alanine blocked ~100/atom), and section 4(3)'s ghost-fitted grids land
  at **100-147 points/element** where they reach 5-10 uHa. Same order, consistently on the
  expensive side of it, which is the single most useful external check available on
  whether this ensemble is any good - and a reason to read that SI before spending effort
  on (4).
* **Song & Martínez**, *Analytical gradients for tensor hyper-contracted MP2 and SOS-MP2 on
  GPUs*, JCP **147**, 161723 (2017). Quartic/cubic gradient scaling; AIMD energy
  conservation demonstrates the gradients are consistent with the THC PES.

Cited in this repo already: Hohenstein/Parrish/Martínez (10.1063/1.4768233), Matthews
(10.1021/acs.jctc.9b01205), Lee/Lin/Head-Gordon (10.1021/acs.jctc.9b00820),
Hillers-Bendtsen/Lu/Martínez NNLS (10.1021/acs.jctc.6c00664), PyTHC (arXiv 2608.17885).

Surfaced by the same searches, not yet read, plausibly relevant:

* *Atomic orbital-based SOS-MP2 with THC. II. Local tensor hypercontraction*, JCP **146**,
  034104 (2017) - locality in THC, closest existing relative of this idea.
* *A critical analysis of least-squares THC applied to MP3*, JCP **154**, 134102 (2021).
* *Tensor Hypercontraction Error Correction Using Regression*, arXiv 2602.23567.
* *Interpolative separable density fitting on adaptive real space grids*, arXiv 2510.20826.

Claims from the design discussion that were **not** verified and should be treated as
unconfirmed: a 2021 Song/Martínez/Neaton automatic-differentiation/diagrammatic THC gradient
paper; arXiv id 2604.03899 for the NNLS paper (this repo cites the DOI, not that id); the
FHI-aims dimer-based basis construction analogy.

## 7. Reproducing

```shell
uv sync
uv run python experiments/atom_centered_grids/sweep.py alanine --out alanine.json
uv run python experiments/atom_centered_grids/analyse.py alanine.json
uv run python experiments/atom_centered_grids/rank.py methanol 1e-3,1e-4,1e-5
uv run python experiments/atom_centered_grids/rotate.py ethanol 1e-3
uv run python experiments/atom_centered_grids/weights.py methanol 1e-4
uv run python experiments/atom_centered_grids/ridge.py methanol 1e-3 --blocked --out ridge_methanol.json
uv run python experiments/atom_centered_grids/scan.py methanol 1e-3 --out scan_methanol.json
uv run python experiments/atom_centered_grids/rotate.py ethanol 1e-3 --orbits
uv run python experiments/atom_centered_grids/orbits.py methanol --out orbits_methanol.json
uv run python experiments/atom_centered_grids/ghosts.py --calibrate H,C,O
uv run python experiments/atom_centered_grids/ghosts.py methanol --out ghosts_methanol.json
uv run python experiments/atom_centered_grids/ghosts.py methanol --out ghosts_rot_methanol.json \
    --thresholds 3e-4,3e-5,1e-12 --blocked-thresholds 1e-3,1e-4,1e-5 --rotate 4
```

Runtimes on 4 cores: water seconds, methanol ~3 min, ethanol ~10 min, alanine ~40 min
(dominated by the unpruned-parent-grid baseline and the tight global fits). `ridge.py` is
seconds to a couple of minutes. `scan.py` rebuilds the SCF and the fit at every geometry,
so it runs (number of steps) x (number of inversion schemes) single-point THC calculations:
about 4 min for methanol's 61-step scan, 25 min for ethanol's 31-step one. Cut
`--half-width` or raise `--step` to trade resolution for time, but note that the crossings
are what the scan is looking for and a coarse scan can step over one.

`orbits.py` runs one fit plus `2 + n_rot` single points per row, so it costs about what a
`rotate.py` run costs per threshold: minutes on methanol, tens of minutes on ethanol. Its
two threshold ladders are on different scales - the orbit-grouped KKT gradient sums over
the whole orbit - so the defaults are `1e-3..1e-5` point-wise against `1e-2..1e-4`
grouped, and both need widening per molecule. Calibrate with `blocked_fit_per_atom` alone,
which needs no SCF, before spending single points on a threshold that lands off the
interesting range.

`ghosts.py` fits each element in seconds - the offline stage is genuinely cheap, and
independent of the molecule - so its cost is all in the single points: 16 of them on the
default ladder, about 12 min on methanol and an hour on ethanol, plus `--rotate N` extra
per grid. Run `--calibrate` first; its threshold ladder is not on the `blocked` scale and
a ladder guessed by analogy lands off the interesting range. Water is not worth running
for the ratio - at 24 AOs every grid saturates the co-density rank and all three modes
reach the floor.

Raw output from the runs behind `FINDINGS.md` is in [`data/`](data).

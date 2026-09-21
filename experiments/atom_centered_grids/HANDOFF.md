# Handoff: atom-centered THC grids

Read this first, then [`FINDINGS.md`](FINDINGS.md) for the measurements. This file records
what the idea is, what was tried, **which parts of the prior design reasoning the
measurements overturned**, and what to do next. The point of §3 is that a fresh session
should not re-derive machinery that turned out to be unnecessary.

§4 and §5 below are ordered by what is *left to learn*. If the question is instead whether
the scheme is ready to be claimed as a method, read [`VIABILITY.md`](VIABILITY.md): it
reorders the same open items by what a demonstration needs, and notes that the
configuration 4(15) recommends has never had a point count, a cost, or a trajectory
measured for it.

## 1. The idea

**In one sentence:** run NNLS per atom (with ghost atoms to keep bonding-region points
alive) to select a point set per element, **keep the points and the weights the same fit
produced**, build a molecule's grid as the union of its atoms' point sets, and - given
sections 4(1) and 4(5), both now done - get analytic nuclear gradients and a smooth PES
for free. The gradient is no longer a promise: it is implemented in `pythc.grad` and
agrees with a finite difference to machine precision. The ridge window that 4(5) had
to live inside turned out to be a property of the ridge rather than of the metric, and
4(6) removes it: with the damped pseudoinverse the gradient is clean at microhartree
accuracy on a surface smoother than the ridge's. The one sentence used to say *discard*
the weights, on section 3's pseudoinverse result; 4(6), 4(9) and 4(13) walked that back. Frozen
weights cost the gradient nothing (`dw/dR = 0` exactly as for frozen points) and are worth
2-2.5x under the filter the gradient actually needs, so the offline object carries them -
but only the weights that the fit which chose the points produced. See 4(9) and 4(13).

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
section 4(3), which is done. Section 4(4) then showed the 2.7x end of that range is an
artefact of an unlucky ghost ensemble on the smallest molecule; with the best ensemble
tested methanol is 1.9x. The large-system end, which is the one that matters, is
unaffected. **Section 4(4) is now finished in full: one frozen point set per element,
handed unchanged to ten molecules spanning sp3/sp2/sp carbon, sp3/sp2 oxygen, nitrogen,
and bonds shorter than any the fit ever saw, costs the same 1.2-1.8x everywhere - 1.49x
in-range against 1.50x out-of-range. The transferable grid transfers.**

## 2. What was attempted

Fifteen experiments, all in this directory, all on cc-pVDZ / cc-pVDZ-RI, level-0 Becke
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
| `ensemble.py` | does the *choice* of ghost ensemble change that? | **1.35x on methanol, 1.05x on ethanol** - it amortises; but each ensemble has a hard **rank ceiling** |
| `transfer.py` | does one element's grid serve bonding it was never fitted in? | **yes.** 1.49x in-range vs **1.50x** out-of-range; an unseen partner element costs nothing, and adding one makes things worse |
| `gradient.py` | can the gradient be computed, and does it match the curve? | **yes, to machine precision** - but only for `lambda` in `1e-2..1e-5`; at 4(1)'s `1e-8` it varies 2x between runs. Net torque 188 uHa/rad blocked, **5592 ghost** |
| `window.py` | what sets that floor, and is the torque a property of the grid? | **the filter's shape, not the metric's rank.** The damped inverse buys 3-5 decades of `lambda` and <5 uHa. The torques were mostly the regulariser: **water's are 0.001/0.002 at `pinv`**, and the ghost:blocked ratio is 4.5x, not 30x |
| `torque_ladder.py` | does the torque 4(6) is left holding converge away with grid size? | **yes, in every mode.** methanol at `pinv`: blocked 357 -> 0.02 uHa/rad, ghost 969 -> 27. The committed ridge data said otherwise and was taken three decades outside where a torque means anything |
| `trajectory.py` | what does the torque do over a trajectory, and is it worse than the grid it is pruned from? | **it reorients, it does not heat** - 95% incoherent, 4.5 deg of axis tilt in 2.5 ps. RKS/PBE on the same level-0 parent is **8x worse** at a picosecond; level 3 is ~25x better on 163x the points |
| `atomic_eri.py` | does a richer *purely atomic* target lift the ceiling that forced ghosts? | **yes, 4.6-6.5x.** The free atom's own ERIs are `O(n_AO^4)` equations against the overlap's `O(n_AO^2)`. The support costs **0.89-1.16x** `blocked` at matched footing against ghosts' 1.11-1.78x, and its weights converge on the torque ladder at **1481x** - like weighted `blocked`, unlike anything else transferable |
| `levers.py` | what sets where a ghost fit's support stops - basis, parent level, directions, cage? | **the cage, which was rejected on an argument.** Ghost gap 1.78x -> **1.22x** at 10 uHa; a finer parent moves the ceiling at *unchanged* equations, contradicting 4(9)'s premise; a tetrahedron is cheapest at 50 uHa and cannot reach 10 |

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
| Freeze the discrete point selection, differentiate the smooth remainder (AO derivative + rigid point translation) | **Stands, and is now implemented** - `pythc.grad`, section 4(5). Those are exactly the two terms and there is no third; the gradient verifies to machine precision. The one correction is that the smooth remainder is only *numerically* smooth above `lambda ~ 1e-5`, well short of 4(1)'s `1e-8`. |
| LS-THC needs no Becke partition function, since `Z` is fitted rather than quadratured | **Confirmed, and then some** - see the weights result below. |
| NNLS weights need active-set freezing, QP sensitivity theory, strict complementarity | **Superseded, but only the freezing apparatus - and section 4(6) has partly walked the rest back.** There is still nothing to *freeze*: a frozen weight has `dw/dR = 0` like a frozen point, so keeping the weights costs the gradient nothing. What is no longer true is that they do not matter. Section 3's exact-absorption argument is about the **pseudoinverse**; 4(5) forces a ridge, which is not scale-equivariant, and inside its window the same 156-point water support gives 165 uHa and 188 uHa/rad with NNLS weights against 3304 uHa and 7145 uHa/rad at `w = 1`. **Keep the weights in the offline object.** 4(13) then sharpened this twice: the effect is 2-2.5x at matched support rather than the 450x 4(7)'s single-rung torque pair implied, and the weights have to be the ones that *selected* the points - a weight set laid onto a support some other fit chose loses 0 times out of 12. 4(9) supplies the weights actually worth keeping. See FINDINGS sections 12, 14 and 16. |
| Reparametrise `w = theta^2` to remove the inequality constraint | **Moot**, and it was the wrong power: this codebase builds `X = w^(1/4) phi`, so `theta^2` still leaves `X = |theta|^(1/2) phi`, singular at 0. If a weight fit is ever kept, parametrise the *collocation amplitude* `X = t phi` with `w = t^4`; the whole pipeline is then polynomial in `t`. |
| Add `sum_P w_P = 1`; equality constraints differentiate cleanly | **Moot**, for the same reason, and section 16 adds the algebra for why normalising cannot substitute for fitting. A positive *diagonal* rescale of `w` sends `X -> D^(1/4) X`, `S -> D^(1/2) S D^(1/2)` and `Z -> D^(-1/2) Z D^(-1/2)`, leaving `X^T Z X` exactly unchanged - and `ridge_shift` keeps `lam` dimensionless so a global scalar is absorbed by the regularised inverses too. Only the *relative* distribution within a support can matter, and only a fit can set it. |
| Isolated-atom NNLS will discard exactly the tail points a bond needs; fix with ghost atoms | **Measured, and then overturned by 4(9) - but read the mechanism first, because it is what overturned it.** The free-atom fit does not fail by misplacing points; it fails because its *target* supplies only `n_AO(n_AO+1)/2` equations, so NNLS cannot retain more than **15 points per hydrogen in cc-pVDZ** at any threshold. Ghosts lift that by stacking environments, and they work (1.1-1.8x over `blocked`). But a count is not the only thing a richer target can supply: the free atom's own ERIs give `O(n_AO^4)` equations, lift the ceiling 4.6-6.5x, and produce a *better* support than the ghosts do (0.89-1.16x at matched footing). **Ghosts were the right fix for the right diagnosis and are no longer the only one.** The predicted anisotropy is real too (34 uHa rotation spread at the ceiling), it just is not what stops the scheme. See section 4(3), 4(9), and FINDINGS sections 8 and 14. |
| Octahedral ghosts suffice because p orbitals are octahedral | **Rejected in the discussion itself, correctly** - points are sampling locations, not functions, and do not superpose. |
| Frozen per-atom grids risk orientation-dependent energies | **Confirmed, deflated, re-confirmed, and now deflated again by 4(7) - and this time with the mechanism.** The torque converges with grid size in every mode; what separates the transferable grid from the in-molecule one is the **weight footing**, not the transfer or the pruning. The complete 3284-point parent grid sits at 0.24 uHa/rad and the *pruned* 670-point weighted grid at 0.02, so pruning is not a cost at all - it is a benefit. Over a trajectory the leak reorients rather than heats. Read the rest of this cell as the record of how the question looked before 4(7). **[Historic below.]** Everything in this cell is about energy *spreads*, and FINDINGS section 12 shows the spread converges away on the transferable grid while `dE/dtheta` does not. Read the rest of this cell as a statement about spreads only. No angular shell survives intact, and rank-limited grids shift 27-39 uHa under random per-atom rotation, but the shift converges away with grid size (methanol 218 -> 7.1 -> 0.01 uHa from 175 to 670 points). Ghost-fitted grids are 2-5x more orientation-dependent than `blocked` **at matched point count** - and identical **at matched accuracy** (882 ghost points: +5.14 uHa, 0.94 uHa spread; 491 blocked points: +5.29 uHa, 0.91 uHa spread). Still a symptom of rank limitation, not a separate defect. |
| Fix isotropy with orbit-wise (group-sparsity) pruning over (radial shell, Lebedev orbit) blocks | **Implemented, measured, and rejected as a fix.** It delivers exactly what it promised - whole orbits, exact octahedral invariance - and that turns out not to be the useful property. At matched point count a point-wise grid is better on both accuracy and rotation spread. See section 4(2) and FINDINGS section 7. |
| One element's grid will need refitting per bonding environment | **Rejected. Section 4(4), second half.** In-range and out-of-range molecules are indistinguishable, and the worst two in a ten-molecule suite are an sp-nitrile and methanol - the molecule the ensemble was designed around. |
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

**(1), (2), (3) and (4) are all done.** (1) was a real prerequisite: the metric
truncation was the last remaining source of PES non-smoothness, and it is now measured
and fixed. (2) was
believed to be a prerequisite too - the thing that makes "attach the atomic grid rigidly"
well-defined - and it is not; orientation dependence is a grid-size problem, not a
structural one. (3) was the one that could still have killed the idea, and it did not:
ghost-fitted per-element grids exist, they work, and they cost 1.1-1.8x over `blocked`.

(4) is now done too, in both halves, and it was the last question that could still have
sent the programme back to the drawing board: if a per-element grid had needed refitting
per bonding environment, "which environment is this atom in" is a discrete function of
geometry, putting a discrete step back at runtime and undoing (1) and the whole smooth-PES
argument. It does not. See section 4(4) and FINDINGS section 10.

**(5) is now done as well, and it was not quite the ordinary development this file
expected.** The gradient itself went in cleanly and verifies to machine precision, but it
brought back two results that no energy measurement could have produced: the ridge
strength 4(1) recommends is three decades too small for a gradient to mean anything, and
the orientation dependence 4(2) deflated has a torque attached to it that is now measured
rather than argued about. See 4(5) and FINDINGS section 11.

**4(9) is the one item here that was not on this list**, and it is worth saying why it
turned up: it comes from taking 4(3)'s *diagnosis* literally rather than its remedy. 4(3)
found that the free-atom fit fails on an equation count and concluded that ghosts are what
make it well posed. They are - but so is any richer observable of the same isolated atom,
and the atom's own two-electron integrals are one. The result is that the training set the
whole of 4(3), 4(4) and FINDINGS sections 8-10 is about may not be needed at all.

**Nothing is left that can kill the idea.** 4(6) removed the one thing 4(5) left looking
awkward - the `lambda` window - and cut the orientation problem down with it. **4(7) has
now closed the orientation question outright and reordered what is left.** The torque
converges with grid size in every mode, a trajectory says the residue reorients rather
than heats, and - the result nobody asked for - plain DFT on the level-0 Becke grid these
fits are pruned from loses angular momentum *eight times faster at a picosecond* than
the frozen support does. Lab-fixed atom-centred quadrature is not an acTHC defect; it is what the whole
family does, and the pruned reweighted support is better at it than its own parent.

**The prerequisite 4(7) left is now built, and the application runs.** It was the
orbital-response layer: on the fixed-orbital surface the force is not the gradient of the
propagated energy, which alone broke rotational invariance at ~8300 uHa/rad, fifty times
the transferable grid's own torque and identically so on a grid with five thousand times
less. What made it more than plumbing was not the CPHF solve but the Laplace factors -
carrying orbital *energies*, they make the energy non-invariant under occupied-occupied
rotation, so the response needs blocks of `U` no solver returns. Writing
`Theta_o = w^(1/4) exp(t F_oo)` instead is the same function at the canonical point,
manifestly invariant, and leaves only the standard occupied-virtual response. The
rotational identity now closes to **6.1e-9** relative against 9.1e-3, and NVE on the THC
surface drifts **0.247 uHa/step against 21.68**. See FINDINGS section 15; what is left
there is making it cost one coupled-perturbed solve rather than `3 N`. The lever was the
**weights**, and 4(9) has now pulled it: weights fitted against the free atom's own ERIs transfer, converge on the torque
ladder like the in-molecule weighted fit, and come with a support that beats the ghost
one. So what is left is the prerequisite, the finer parent grid of (10), and the open
questions in section 5 - where the selector question, which 4(9) speaks to directly,
now has top billing.

**4(15) is the newest entry and it changes what the list is about.** 4(14) found a
runtime diagonal preconditioner that recovers, in the energy, what discarding the NNLS
weights costs - and could not use it, because the forward scheme had no adjoint. It has
one now, and the ladder it unlocks says the weight footing that 4(7), 4(8), 4(9) and
4(13) are all about **cancels exactly** out of the preconditioned metric inverse. The
transferable support reaches 0.09 uHa/rad and the parent-grid energy floor carrying no
weights at all. That does not retire 4(9) - an ERI-fitted support is still the best
support here, and the two have never been compared on one ladder - but it does mean the
*weights* half of this list is a question about a quantity the pipeline need not have.

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
end to end. The nuclear derivative that was missing is (5), and with FINDINGS section 15's
response layer on top of it the table is now something a trajectory is actually propagated
on: `pythc.grad.total.ThcMP2Gradients.as_scanner()` hands the whole pipeline to
`pyscf.md`, with the per-element point sets held fixed for the entire run.

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

Left undone here, and **all of it since done in 4(4)**:

* the ghost ensemble was chosen once and never varied. `--full-cross` and `--directions
  octahedron` have now been run (`ensemble.py`), and the choice is worth 1.35x on
  methanol and 1.05x on ethanol - so the methanol ratios below are upper bounds, and the
  ethanol ones stand;
* N was never fitted, so the grids covered H/C/O only. `transfer.py` fits it, and it
  behaves exactly like C and O (44 to 159 points across the same ladder);
* no molecule outside the fitting set was tried, which was the whole of what (4) meant.
  Ten of them have now been, and the ratio does not move - see 4(4).

The one item that stands: water was run and is not quoted, because at 24 AOs every grid
in the comparison is rank-saturated and all three modes reach the floor. 4(4) reproduces
that and generalises it - anything below about 40 AOs saturates, which is why its
`transfer.py` report marks such cells rather than ratioing them.

**(4) Transferability. DONE, both halves.** Two halves. The first - does the *ensemble*
matter? - is answered in `ensemble.py` and FINDINGS section 9, and the answer is "less
than feared, but it leaves a constraint behind". The second - does one element's point set
serve environments it was not fitted in? - is answered in `transfer.py` and FINDINGS
section 10, and the answer is yes.

* **The sensitivity amortises.** Across the 2x2 of {12 icosahedral, 6 octahedral} x
  {cycled, full cross}, the spread at matched accuracy is 1.29-1.78x on methanol and
  1.20-1.26x on ethanol. Like §1's ratio, §8's ghost gap and §4's rotation spread, it is
  a fixed per-atom overhead that shrinks with system size.
* **Section 4(3)'s ensemble is the worst of four on methanol and mid-pack on ethanol.**
  So the 1.78x / 1.60x methanol figures in FINDINGS section 8 are upper bounds - the
  achievable number there is 1.21-1.35x, compounding with section 1 to **1.9x** against a
  global fit rather than 2.7x. The ethanol figures stand as measured, and since the
  large-system end is what matters, the programme headline is unchanged.
* **The durable result is a rank ceiling, and it was not what the question was looking
  for.** Driving the threshold to zero saturates each element's support at the effective
  rank of its stacked target - a property of the ensemble alone. Summing per-element
  saturation sizes gives a hard cap on a molecule's grid, computable with no SCF
  (`ensemble.py --saturate`). `octa-cycled` is the *best* ensemble on methanol and
  **cannot reach 10 uHa on ethanol at any threshold**, because its ceiling (840 points)
  is below what ethanol needs. So the question to ask of an ensemble is not "is it the
  best" but "does it supply enough rank", and that gets harder as molecules grow.
* **Do not read the supports.** Jaccard overlap between ensembles is 0.01-0.48 - lower in
  places than the ghost-vs-blocked agreement - while the energies land within 1.05x on
  ethanol. Section 4(3) already said support overlap is not a quality metric; this is the
  second independent confirmation.
* **The error curves are not monotone in point count.** A tighter KKT threshold re-solves
  rather than extending the support, so adding points can make the error worse (five
  cases across the two molecules). A threshold ladder is a coarse instrument.

**The second half is DONE** - `transfer.py`, FINDINGS section 10 - and the answer is that
one element's point set serves environments it was not fitted in at no measurable cost.

One support per element is fitted once, frozen, fingerprinted, and handed unchanged to ten
molecules: O in water, methanol and formaldehyde; C in sp3 (ethane, methanol, propene), sp2
(ethene, propene, formaldehyde) and sp (acetylene, hcn, acetonitrile); N in two bonding
modes. Four of the heavy-atom bonds are *shorter than the tightest ghost shell* - the
closest C-C ghost is 1.37 A against acetylene's 1.20, and the closest C-O is 1.28 against
formaldehyde's 1.22 - so they are extrapolations below the training range, not
interpolations inside it.

* **In-range 1.49x, out-of-range 1.50x at 10 uHa** (1.41x vs 1.46x at 5 uHa). The suite
  spans 1.20-1.80x, which is the band section 4(3) already reported from methanol and
  ethanol alone. There is no transferability penalty to find, and the low support
  agreement with `blocked` turns out to have meant the first thing, not the second: the
  choice of points hardly matters.
* **The worst molecules are not the exotic ones.** Acetonitrile (1.80x) and methanol
  (1.78x) bracket the suite, and methanol is the molecule the ensemble was designed
  around. Acetylene, a 12% extrapolation below anything ever fitted, is 1.34x. Propene at
  72 AOs is the best of the set at 1.23x, so this overhead amortises with size like every
  other one here.
* **An unseen partner element costs nothing.** Methylamine puts a C-N bond in front of a
  carbon grid fitted only against H/C/O and is 1.22x / 1.10x, among the best in the suite.
* **Adding the partner makes it worse, which is the surprise.** Refitting with N in the
  partner list degrades both nitrogen-free controls (propene 1.23x -> 1.48x, methanol
  1.78x -> 1.85x) and degrades acetonitrile - which contains the very C#N bond the partner
  was meant to describe - furthest of all, 1.80x -> 2.33x. Both ensembles hold 13
  environments; the baseline draws them from 10 distinct (partner, distance) combinations
  and the N-partner one from 13. So the equation supply is unchanged and only the variety
  rises. **Environments supply rank and more of them help (section 4(4) first half);
  variety at fixed environment count is a different knob and turning it up costs.** Do not
  add a partner element to cover a molecule - add environments.
* **The one apparent counter-example is a saturation artefact, and was caught.** hcn
  appears to gain from the added partner (1.49x -> 1.05x), but at 33 AOs it is the most
  rank-saturated molecule in the suite. Acetonitrile carries the same C#N bond at 57 AOs
  with ladders widened until neither curve is saturated, and says the opposite. Do not
  read the 33-38 AO molecules on their own.

Left undone here: the in-range group is thin - methanol and ethane are the only in-range
molecules large enough to read, since water saturates (section 4(3) said as much). Seven
of the 33-38 AO cells are unreadable for the same reason, so the sp-carbon case rests on
acetonitrile alone, and acetonitrile's own error curve crosses the 10 uHa target three
times, which makes its point count a first-crossing estimate. And transferability across
*basis sets* is untouched: the offline object is a list of indices into an element's
level-0 atomic grid, and nothing asks what becomes of it in cc-pVTZ.

**(5) Actually compute a gradient. DONE** - `pythc.grad`, driven by `gradient.py`, written
up in FINDINGS section 11. The gradient exists, it agrees with a finite difference to
machine precision, and it brought back two things nobody had asked it for.

What was built: a hand-written reverse-mode pass over the whole `ov` pipeline
(`laplace_mp2.py` -> `linalg.py` -> `factorisation.py` -> `geometry.py`), plus
`FrozenGrid`, which carries the one piece of bookkeeping a frozen grid has and a
re-selected one cannot - which atom each point rides with. The collocation derivative is
the target pipeline's second row in one line: an AO derivative, a rigid translation, and
**no third term**.

What it settled, and what it did not:

* **It is correct.** Quadratic convergence in the finite-difference step (1.88e-7 ->
  1.87e-9 as `h` goes 1e-3 -> 1e-4), and the forces sum to zero to 2.4e-15 of the
  gradient norm with no finite difference involved - which is the sharp test of the
  collocation term, since under a uniform shift its two halves are equal and opposite and
  an error in either cannot cancel. The energy reproduces `LS_RI_Becke` + `LaplaceRMP2`
  to 1.2e-11.
* **[SUPERSEDED by 4(6).** The floor below is real for the *ridge* and was correctly
  measured, but it is a property of that filter rather than of the metric: the damped
  pseudoinverse puts the usable window three to five decades lower at microhartree cost.
  Do not carry the `1e-2..1e-5` recommendation forward without reading 4(6).**]**
  **Section 4(1)'s `lambda = 1e-8` is far too small for a gradient, and this is the
  result to carry forward.** At that value the energy is reproducible to 0.002 uHa and
  the gradient varies by a **factor of two** between runs of the identical calculation,
  because 61 of water's 156 metric directions sit below 1e-14 of the top eigenvalue and
  ridge inverts them at `1/lambda`. The usable window is **`1e-2` to `1e-5`**, bounded
  below by that noise and above by real accuracy loss (23 uHa at `1e-5`, 165 uHa at
  `1e-4`). 4(1) said not to pick `lambda` from an accuracy table alone; the correction is
  that its own smoothness table is not enough either - a second difference at 0.002 A
  steps over roughness a derivative sees directly.
* **The frozen grid owns very little of the gradient.** At `lambda = 1e-4` the collocation
  term is 10% of the total and the rigid translation - the part that exists *only* because
  the grid is frozen - is 1.4%. The rest is derivative integrals. That is the reassuring
  direction: a frozen grid is not injecting large spurious forces, which is what you would
  expect given that `Z` is fitted to reproduce DF ERIs that do not depend on the grid at all.
* **[SUPERSEDED by 4(6).** These numbers are measured at `lambda = 1e-4` and are
  dominated by the regulariser, not the grid. Against a `pinv` control the same two
  water grids give **0.001 and 0.002 uHa/rad** - water is rank-saturated, so its torque
  was always going to be zero, which is exactly why 4(3) and 4(4) refuse to quote it.
  The ghost:blocked ratio on a molecule that does not saturate is **4.5x, not 30x**.**]**
  **The torque is now a number, and it does not say what the spread said.** Section 4(2)
  measured orientation dependence as an energy spread and could only argue about the
  lab-frame-versus-molecular-frame choice. `dE/dtheta` is exactly computable - rotating a
  point set about its own nucleus leaves the molecule, the AOs and the SCF untouched, so
  no response term enters - and agrees with a finite difference to six significant
  figures. The net torque `sum_A tau_A` is the one that matters: it is exactly the
  energy's response to rigidly rotating the molecule under lab-fixed attachment, i.e. the
  rate angular momentum leaks. On water: blocked (156 points) **188 uHa/rad**, ghost-fitted
  per-element (208 points) **5592 uHa/rad**. The transferable grid has 1.9x the energy
  spread of the blocked one and **30x** the torque.
* **So the rotation spread joins `rmsd_S` and support overlap on the list of metrics that
  do not measure what they are used for.** 4(2) deflated orientation dependence on the
  strength of spreads converging away with grid size. Spreads are peak-to-peak over draws;
  torques are slopes at one orientation, and a rough-but-small-amplitude function has a
  tiny spread and a large slope. Do not assess a frozen grid's orientation quality with a
  spread. The collocation term tells the same story twice over: on the ghost grid the
  rigid translation is 9% of the gradient against blocked's 1.4%.
* **What it is not.** This is the derivative of the correlation energy at a *fixed SCF
  reference*. That is the THC-specific content - everything the frozen grid, collocation,
  metric and ridge touch - and it is deliberately separated so both sides of the
  finite-difference check are exact. The orbital-response term is standard DF-MP2
  machinery and is not implemented; `C_bar` and `eps_bar` are returned ready for it. Until
  it is, `de` is not the total MP2 gradient and cannot be compared against `pyscf`'s.

Left undone: the orbital response, which would also turn the rotational identity
`sum_A a_A x dE/dR_A + sum_A tau_A = 0` into a second free test (it currently fails by
4.7e-4 relative, which *is* the omitted term). Everything above is water, one geometry,
cc-pVDZ; the ridge window in particular is read off a single system.

**(6) Fix the ridge window. DONE** - `lib.damped_inv`, driven by `window.py`, written up
in FINDINGS section 12. 4(5) left the gradient working but standing somewhere
uncomfortable: usable only for `lambda >= 1e-5`, costing 23 uHa on water and up to 1808
elsewhere. That is now fixed, and fixing it overturned 4(5)'s other result as well.

* **The floor was the filter's shape, not the metric's rank.** `(S + lambda I)^-1` gives
  an eigenvalue `sigma` the gain `1/(sigma + mu)`, which *rises* to `1/mu` as `sigma`
  falls - the null directions get the largest gain in the whole operator, and `Z = D^T D`
  carries `S^-1` twice. Measured, the gradient's permutation noise grows as `lambda^-1.9`:
  essentially `1/lambda^2`, which is what that mechanism predicts and what nothing else
  would.
* **The fix is one spectral function.** `damped_inv` applies `sigma/(sigma^2 + mu^2)`:
  analytic in `S` like the ridge, with no cutoff for an eigenvalue to cross, but sending
  the null directions to zero like the truncation. Same peak gain `1/(2 mu)`, so `lambda`
  means the same thing on both ladders. Usable `lambda` drops three to five decades and
  the accuracy it costs goes from **166-1808 uHa to under 5** across five grids. Select it
  with `metric_scheme="damped"` on `ThcFactorisation`, `thc_mp2_gradient` or `LS_RI_THC`;
  the default is unchanged.
* **`ridge_inv_eigh` is why that conclusion is allowed.** `damped_inv` changes the filter
  *and* the algorithm (`eigh` against `ridge_inv`'s Cholesky), and those differ by 7.8e-8
  at `lambda = 1e-8` on a metric like this. The spectral ridge tracks the Cholesky ridge
  to within a factor of two at every `lambda` on both molecules. Do not re-litigate this
  as a linear-algebra quality question; it is the filter.
* **The permutation probe is the instrument to reuse.** Relabelling an atom's points is an
  exact symmetry of the energy, the gradient and the torque, so anything that moves under
  it is noise. It is deterministic, needs no multi-process comparison, and says *how much*
  noise rather than only that there is some.
* **It stays smooth, which was the thing that could have gone wrong.** The damped filter
  approaches the truncation as `lambda` falls, and the truncation is what 4(1) removed the
  steps with. `scan.py --damped-lambdas` says damped at `1e-10` is the *smoothest* curve
  in the table - `max|d2| = 0.013` against `pinv`'s 0.511 - at a `lambda` where the
  ridge's gradient is 81% noise.
* **4(5)'s torques were mostly the regulariser, and this is the conclusion to carry
  forward.** Both schemes reduce to `pinv` as `lambda` falls, so `pinv` is the control
  4(5) could not reach. Water's 188 and 5592 uHa/rad become **0.001 and 0.002**. Water is
  rank-saturated - co-density rank 95 against 156 and 208 points - so its torque was
  always going to be zero, and 4(3) and 4(4) both say in terms not to quote water. On
  methanol, which does not saturate, the ghost:blocked ratio is **4.5x rather than 30x**,
  and the leak on the real transferable object is **3.1e-3 bohr/rad** against 4(5)'s
  1.35e-1. **A torque is not a property of a grid; quote it with the regularisation that
  produced it.**
* **The accuracy tax was never the problem.** The ridge's 280 uHa penalty on methanol
  varies by 2.28 uHa along a bond scan - a constant offset, invisible to a trajectory -
  and every scheme, including the bare frozen grid, biases the O-H force constant by under
  1.7 cm^-1. The case against the ridge is the gradient noise and nothing else.

Left undone: `damped_inv` costs a full eigendecomposition where `ridge_inv` costs a
Cholesky, ~3x on that step, and no cheaper iteration to the same filter was tried - the
obvious one being to solve the `Z` fit as a least-squares problem directly rather than
forming `S^-1` twice, which was the second escape 4(5) listed and would reach this filter
by construction. The ghost grids' torque does not converge to `pinv` the way the blocked
ones do, because `pinv` and a small-`mu` damped filter are genuinely different operators
in the limit, so a ghost grid's torque has a value per inversion rather than a value.
Everything is cc-pVDZ, `ov`, MP2, fixed-orbital, one geometry and one orientation per
grid. And nothing integrates a trajectory.

**(7) The ladder and the trajectory. DONE** - `torque_ladder.py --scheme damped --pinv`
and the new `trajectory.py`, written up in FINDINGS section 13. Three results, and the
third is the one that matters most.

* **The torque converges with grid size, in every mode.** Methanol at `pinv`, over
  223-235 to 670-702 points: `blocked` 357 -> 0.02 uHa/rad, `blocked1` 357 -> 9.0,
  `ghost` 969 -> 27, `ghostw` 698 -> 52. Water is 0.00 everywhere. **The committed ladder
  that said otherwise was taken at `ridge` `1e-4`** - `--scheme` entered the script in the
  merge that followed that data - which is three decades outside where 4(6) says a torque
  means anything. `--pinv` is new and is what makes a ladder interpretable at all: both
  filters reduce to the pseudoinverse as `lambda` falls, so without that control a trend
  down the rungs cannot be attributed to the grid rather than to the filter.
* **The gap is the weight footing, not the transfer and not the pruning.** `ghost`
  converges like `blocked1` (35.6x against 39.6x), not like weighted `blocked` (17667x).
  And the complete 3284-point parent grid sits at **0.24 uHa/rad** while the *pruned*
  670-point weighted grid sits at **0.02** - twelve times quieter on a fifth of the
  points. Pruning is not a cost; it is a benefit. What a transferable support cannot do is
  carry in-molecule weights.
* **4(5)'s "the spread does not track the torque" does not survive a ladder.** Over 20
  `pinv` rungs, `log|tau|` tracks the rotation spread at Spearman +0.89 and the grid's own
  accuracy at +0.86, against -0.54 for the point count. That claim rested on two water
  grids under the ridge - the one molecule whose torque is zero, and the one filter that
  manufactures one. The torque tracks how *good* a grid is, not how big.
* **Over a trajectory the leak reorients rather than heats.** 2.5 ps of thermal tumbling
  on methanol: the torque is 95% incoherent, the component along `L` is pinned at
  +0.36 hbar against 18.24 by energy conservation (it is the only component that does
  work, and the orientational potential's 31 uHa amplitude bounds it), and what
  accumulates is 4.5 degrees of rotation-axis tilt.
* **And it is quieter than the quadrature it is pruned from.** `trajectory.py
  --reference rks` on methanol, same initial condition, `|L - L0|` in hbar at **matched
  times** - which matters, because these surfaces do not leak at the same rate:

  | | points | 100 fs | 300 fs | 1000 fs |
  | --- | --- | --- | --- | --- |
  | RKS/PBE level-0 Becke | 4656 | 0.638 | 2.380 | 9.250 |
  | frozen `ghost` | 414 | 0.144 | 0.686 | 1.188 |
  | frozen `blocked` | 301 | 0.030 | 0.065 | 0.106 |
  | RKS/PBE level-3 Becke | 67432 | 0.004 | 0.028 | - |

  acTHC beats the grid it is pruned from by 8x at a picosecond and loses to a production
  grid by ~25x on 163x fewer points. **The in-molecule `blocked` grid is within 2.3x of
  level-3 DFT on 224x fewer points**, which puts the whole remaining gap on the weights
  and makes (8) below the most valuable unrun experiment here. **[(8) is now done, via
  (9): the weight set that works is fitted against the free atom's own ERIs.]**

Left undone: the trajectory is driven by DF-RHF, which is grid-free and therefore exactly
rotationally invariant, with the frozen grid's torque integrated along it and the leaked
rotation fed back; the internal coordinates are still the RHF trajectory's, so the
treatment is first order in the leak. A fully coupled run needs the orbital response
first. One molecule that does not saturate, one initial condition per arm, gas phase.

**(8) Fit per-element weights against an objective LS-THC cares about. DONE, by 4(9)
below, and the answer is that it works.** 4(7) localised the entire orientation gap to the
weight footing: same support, `w = 1` gives 9.02 uHa/rad and NNLS weights give 0.02.
`ghostw` - the ghost fit's own weights, transferred - gives 52.3 and is *worse* than
`w = 1`, consistent with 4(6)'s note that those weights condition the ghost metric rather
than the in-molecule one. The framing here was "fit a per-element weight set *for* the
in-molecule objective", which sounded like it needed molecules. It did not. Weights fitted
against the **free atom's own two-electron integrals** transfer into a molecule and land
on the in-molecule curve: `eriw` converges 1481x over the torque ladder against weighted
`blocked`'s 17667x, where every `w = 1` mode and `ghostw` sit at 13-40x, and it reaches
0.19 uHa/rad on 565 points - below the complete 3284-point parent grid's 0.24. Frozen
weights differentiate exactly as cleanly as frozen points (`dw/dR = 0` either way), so all
of that is free in the gradient. See 4(9) and FINDINGS section 14.

Two things 4(14) adds, both about the route 4(9) did *not* take. First, the cost of
getting this wrong grows with basis: discarding the weights costs ~1.5 uHa on 670 cc-pVDZ
points and **~19 uHa on 1746 cc-pVTZ points**, so a transferable support that cannot carry
weights is in more trouble in a production basis than cc-pVDZ suggests. Second, the
in-molecule *overlap* route is not merely the weaker one - it is **refuted at its
ceiling**. Weights fitted per atom, per molecule against that objective, which is strictly
more freedom than 4(13)'s `molw` or than one frozen vector per element, give **155.67
uHa/rad against plain `w = 1`'s 27.25** on the same support. 4(13) found `molw` beats
`ghostw` 0 times out of 12; 4(14) says the ceiling of that route is below the floor. What
is refuted is the overlap objective, which is how both in-molecule routes were built -
not 4(9)'s, and not this entry against the torque itself (4(12)) or the LS-THC residual.

And 4(14) prices what the weights are doing mechanically: `X = w^(1/4) R` and
`S = (X X^T) o (X X^T)` give `S(w) = D S(1) D` exactly, so their entire effect is a
symmetric diagonal scaling of the metric - absorbed by the *pseudoinverse* (4(3)'s
result) and not by an absolute shift. A runtime Jacobi scaling recovers the whole 19 uHa
in the energy with no offline object at all. It had no adjoint, which is what kept it an
energy result; **4(15) derives one, and the ladder it unlocks says this entry's whole
subject is removable rather than solvable.** `diag(S(w))_PP = w_P diag(S(1))_PP`, so
`E(w) S(w) E(w) = E(1) S(1) E(1)` exactly and the preconditioned metric inverse does not
see the weights at all - `blocked` and `blocked1` agree to ten digits under it, and
`ghost` reaches 0.09 uHa/rad and the parent-grid energy floor carrying no weights of any
kind. They have since been run against each other, and 4(9)'s `eriw` - the best
*fitted* transferable weight set there is - collapses onto its own unweighted row under
the preconditioner and is beaten by it at matched support. What survives of 4(9) is its
support, which nothing here touches.

**(9) A richer purely-atomic target: fit the atom's ERIs. DONE** - `atomic_eri.py` and
`pythc.decomp.nnls.ERIFitOperator`, written up in FINDINGS section 14. This is the one
item on this list that was not on it: it comes out of taking 4(3)'s diagnosis literally.

* **The ceiling was an equation count, so supply more equations about the same atom.**
  `(mu nu|lambda sigma) = int dr phi_mu phi_nu V_{lambda sigma}(r)` makes the ERI a
  *linear* functional of the quadrature weights, so the same Lawson-Hanson solver fits it
  unmodified - against `[n_AO(n_AO+1)/2]^2` equations instead of `n_AO(n_AO+1)/2`. The
  potentials are PySCF's `int1e_grids`. Nothing else changes: same per-element level-0
  parent, same solver, same `w = 1` / `metric_ridge = 1e-8` evaluation, so the offline
  object is still a list of indices into an element's atomic grid.
* **The ceiling lifts 4.6-6.5x** (`--saturate`, no SCF): H 15 -> 97, C 92 -> 463,
  N 92 -> 467, O 92 -> 423. Hydrogen's overlap fit stops at exactly its equation count,
  reproducing 4(3); the heavies stop at 92 of 105, which is the *parent grid's* resolving
  power and not the target's - the same 92 the operator's exact rank reduction reports.
* **It beats the ghosts.** At matched accuracy and matched weight footing the free-atom
  ERI support costs **0.89-1.16x** the in-molecule `blocked` grid on methanol and ethanol,
  where 13 ghost environments cost 1.11-1.78x. Most of the saving is a reallocation: an
  ERI target gives hydrogen far fewer points than an overlap target does (methanol at
  ~700 points, `ghost` is C 125/H 108/O 145 and `eri` at 565 is C 182/H 45/O 203).
* **The weights are the real prize** - see (8). Do not read an `eriw` number against a
  `w = 1` number; every ratio in section 14 is quoted against a control on the same
  footing for that reason.
* **Thresholds are on their own scale again.** Use `--calibrate` before spending single
  points; `1e-4 .. 1e-6` lands on the ghost ladder's point counts. The ladders are not
  monotone, as 4(4) warned.

Left undone: two molecules, one basis, one parent level, `ov`/MP2, and the `quadrature`
target only. The `exact` target - the true integrals rather than the ones the parent grid
reproduces - is implemented and was swept on hydrogen alone, where it saturates at 85
points against `quadrature`'s 97; whether it selects a *better* support on the heavies is
untested, and it costs about an hour each to find out.
Nothing has been fitted against ghosts *and* an ERI target together, which is the obvious
next control and would say whether environments still add anything once the target is
rich. The torque ladder is methanol, four draws, one seed. And 4(4)'s ten-molecule
transferability suite has not been re-run on ERI supports, which is what would turn
"better on two molecules" into the same statement 4(4) makes about ghosts.

**(10) Try a finer parent grid. PARTLY DONE - `levers.py`, FINDINGS section 17 - and the
premise this entry was written on is false.** The reasoning below is that NNLS cannot
retain more points than its target supplies equations, so a finer parent buys "the same
support size, chosen better". Measured, a level-1 parent moves the *ceiling* by
**1.26-1.42x at unchanged equation count** (H 132 -> 187, C 155 -> 216, O 199 -> 251):
the level-0 parent has been co-limiting all along and support size is not held fixed. On
methanol it is worth 1.78x -> 1.46x at 10 uHa - real, cheap, and much smaller than the
cage of 4(14). **What is still not done is the part this entry was actually for:** nothing
measures a torque or a rotation spread on a finer-parent support, and 4(7) shows the
level-0 parent is a poor grid to be rotationally invariant on, so the orientation question
that motivated it is untouched. `ghosts.element_grid` now takes `level`, and the same
question is open for 4(9)'s ERI target, whose ceiling the heavies already hit at the
*parent grid's* resolving power rather than the target's - 92 of 105 - which is exactly
the constraint a finer parent lifts.

**[Historic premise, kept because it is the thing that turned out false:]** The
offline object is a list of indices into an element's level-0 atomic grid, and nothing
forces level 0. 4(3) established that NNLS cannot retain more points than its target
supplies equations - `n_AO (n_AO + 1) / 2`, a property of the *basis*, not of the parent.
So a finer parent offers a richer candidate set against the same number of equations: the
same support size, chosen better. If that holds, orientation quality is buyable at no cost
in `n_P` at all.

**(11) The orbital response.** Unchanged in content from 4(5) and still standard DF-MP2
machinery, but 4(7) moves it to the front: it is the larger of the two symmetry violations
in the pipeline by fifty-fold, it is what makes a fully coupled trajectory possible, and it
turns the rotational identity into a second free test of the gradient.

**(12) Make `dE/dtheta` an objective of the offline selection.** 4(6) suggested it and
nobody has run it. The torque needs no SCF beyond the reference, so it is nearly free to
evaluate inside a selection loop. 4(7) weakens the case slightly - the torque converges on
its own, and (8) and (9) both look like larger levers - but it targets the defect directly
and remains the only idea here that would make a support *designed* to be rotationally
quiet rather than incidentally so.

**(13) Can a support and its weights be chosen separately? DONE, and the answer is no.**
`insitu.py`, FINDINGS section 16. Run as the *in-molecule* route to (8), in parallel with
4(9) and superseded by it: per-element weights fitted against real atoms in real molecules
(train water/methanol/ethane, held out ethanol/formaldehyde/propene), stacked into one
NNLS solve per element, so the product is still `element -> (indices, weights)` with no
molecular index and `dw/dR = 0` as before. On its own terms it is the weaker route - ~3x
over `ghostw` where 4(9)'s `eriw` converges 1481x, and it needs training molecules 4(9)
does not. **Use 4(9). What this earns its place for is the control.**

4(9) compares self-consistent pairings only - a support with the weights the same fit
produced - and warns not to read an `eriw` number against a `w = 1` one. The in-molecule
route makes the *mixed* case natural to build, and it is decisive: weights fitted for one
objective, laid onto a support selected for another (`molw`), beat `ghostw` **0 times out
of 12** on held-out molecules, on energy and rotation spread alike, despite carrying much
the better overlap residual. Refit support and weights together (`molfit`) and the same
machinery wins 8/10. So 4(9)'s warning is the weak form: a weight set carried onto the
wrong support is *worse than none*, and 4(9)'s 1481x is a property of the ERI **fit**
rather than of ERI weights that could be transplanted onto some other support. Support and
weights are one object.

Two numbers worth carrying forward. The weight footing is worth **2-2.5x at matched
support** (`molfit` against `molfit1`, identical points, no interpolation), not the 450x
4(7)'s single-rung `blocked`/`blocked1` pair implied. And the overlap objective can no
longer rank the candidates - `molw` and `molfit` are matched on it to 10% and differ by
1.5-2.8x in what matters, in the opposite order - which is the same conclusion 4(9)
reaches from the equation-count side and independent evidence for it.

Left undone: three training molecules and three held out, H/C/O only, cc-pVDZ, no
nitrogen; spreads are peak-to-peak over four draws. The obvious control this route makes
cheap and nobody has run is `molw`'s mirror - **ERI-fitted weights on an in-molecule
support, and in-molecule weights on the ERI support** - which would say whether "one
object" is about the objective or only about the fit that produced the pair.

**(14) Four levers on the ghost fit, and a preconditioner that replaces its weights in
the energy. DONE on energy, NOT STARTED on orientation** - `levers.py`, FINDINGS
section 17. **Read 4(9) first:** it lifts the equation ceiling that forced ghosts at
all, so the ensemble these levers tune may not be the object worth tuning. What
survives is that the levers are properties of any NNLS support fit - the parent-grid
and basis rows transfer straight to the ERI target, whose ceiling is set the same way
- plus two results that are not about ghosts at all, below. Everything in 4(3) through 4(7) was measured at one
point in the space of things that set where a ghost fit's support stops: cc-pVDZ, a
level-0 parent, 12 icosahedral directions, one ghost per environment. All four are now
varied, as a ceiling (SCF-free, seconds) and as points per microhartree on methanol.

* **The cage is the result, and it inverts a decision this directory made on an
  argument.** `ghost_environments` refuses to put ghosts in every direction at once on
  the grounds that the central atom's Becke share would collapse. It does collapse -
  **0.2-0.3% of the free-atom weight against 57%** - and the ceiling rises anyway (2.2x
  on carbon), because `stack_targets` normalises each environment's target and what
  survives sees every direction at once. On methanol the ghost gap is **1.78x -> 1.22x**
  at 10 uHa and **1.60x -> 1.13x** at 5 uHa; at 2 uHa the transferable grid is *smaller*
  than the in-molecule fit (0.96x) and reaches the parent-grid floor, which `ghost`
  never does. Against a global fit that is **1.85x rather than 2.7x**, halving the
  headline `n_P^2` tax of the programme to 3.4x **in cc-pVDZ, and only there** - see the
  basis entry below.
* **Stage A and stage B do not rank the levers the same way, and stage B is the one that
  counts.** `basis:cc-pVTZ` has the highest ceilings and `tetra` the lowest, yet `tetra`
  is the *cheapest* setting measured at 50 uHa (0.87x, better than `blocked` itself) and
  cannot reach 10 uHa at any threshold. A ceiling is a necessary condition. It is still
  worth running first, because it is free and it is what catches `tetra` before the
  single points do - which is exactly the use section 9 already prescribes.
* **4(10)'s premise is false** - see that entry. A finer parent moves the ceiling at
  unchanged equation count.
* **Nothing here has been near a gradient** except the preconditioner, which 4(15) has
  since taken all the way. All of it is `w = 1` at `metric_ridge = 1e-8`, which 4(6)
  disqualifies for gradient work, and no *lever* has been through `window.py`,
  `torque_ladder.py` or `trajectory.py`. **The cage's torque is unmeasured**, and 4(7)
  puts the whole remaining gap to production DFT on orientation and the weight footing
  rather than on the energy. So the lever that halves the point tax is not yet known to
  help the quantity that actually blocks AIMD - and it could plausibly hurt it, since a
  cage-fitted support is trained on a fragment of the atom.
  **Run `torque_ladder.py --scheme damped --pinv` on a cage support before believing
  section 17 is good news for the application** rather than only for the energy.
* **The cage does not survive a change of basis, which is the single most important
  thing in section 17 after the cage itself.** Carried to cc-pVTZ on the same molecule it
  is **1.2-1.4x worse** than the single-ghost ensemble it beats by 1.5x in cc-pVDZ.
  `cage`'s ladder is still falling steeply at its tightest rung while `ghost`'s has
  converged, so that ranking is provisional. **Quote section 8's tax, not section 17's**,
  and treat the cage as a cc-pVDZ best case.
* **The cc-pVTZ `blocked` plateau was the weights, and it is the most consequential
  thing in section 17.** The extended ladder showed `blocked` flat at ~19 uHa above the
  floor while its support grew 1293 -> 1746 points, with `ghost` passing it - which would
  have broken §1 and §8's strict-lower-bound premise. `levers.py --blocked-diagnostic`
  ran the 2x2 that separates grid from footing on one set of supports, and the premise
  survives: **with its own NNLS weights `blocked` sits ON the parent-grid floor from 1053
  points up** (+0.36, -0.08, -0.04, +0.00 uHa). The filter is worth a factor of three
  (`w = 1` ridge 19 -> damped 6, 4(6) reproducing in a second basis on a grid five times
  larger than it was measured on); the weights are worth all of it.
* **What that costs the programme is the point - and 4(15) is why it no longer does.**
  A transferable support can carry `w = 1` or its ghost-fit weights and nothing else -
  4(7) already localised the *orientation* gap to exactly this - so the penalty this
  prices is one acTHC pays and an in-molecule fit does not. And it grows with basis:
  discarding the weights costs ~1.5 uHa on 670 cc-pVDZ points and **~19 uHa on 1746
  cc-pVTZ points**. That made (8) a prerequisite for using the scheme in cc-pVTZ at all.
  **4(15) removes the premise**: `E(w) S(w) E(w) = E(1) S(1) E(1)` exactly, so under the
  preconditioner there is no weight footing to be penalised for not carrying, in any
  basis. What remains of this bullet is that the *measurement* of the penalty was right,
  and that every ratio in this directory was taken under it.
* **A caveat it puts on 4(3), 4(4) and section 17 together:** every ghost-gap number in
  this directory is measured against a `blocked` row that is itself at `w = 1`. That is
  the right control for isolating the selection and a handicapped baseline for everything
  else, increasingly so with basis size. The published gaps flatter the transferable
  scheme.
* `levers_methanol_tz_wide.json` was killed by the machine during its second `cage` rung
  - a cc-pVTZ cage environment carries ~360 AOs across 9 environments and is the heaviest
  solve here. Re-run the cage rungs chunked, so a kill costs one rung.
* Left undone: one molecule per basis, and `cage` was run in stage B with the icosahedral
  direction set only. `cage:tetrahedron` hits `ghost`'s hydrogen ceiling exactly on an order of
  magnitude fewer equations, so it may be most of the win at a fraction of the fit cost.
  `cage` and `ghost` are also not matched in environment count (10 against 13), which
  section 9 identifies as the thing that supplies rank, so part of the win may be
  bookkeeping rather than geometry.


**(15) Give the runtime preconditioner a derivative, and read the torque under it. DONE,
and it is the largest single result since 4(9)** - `pythc.grad.linalg.jacobi_inv_adjoint`,
`torque_ladder.py --scheme damped_jacobi`, FINDINGS section 18.

4(14) found that `E = diag(1 / sqrt(diag S))`, applied at runtime and fitted from nothing,
recovers in the *energy* the ~19 uHa that discarding the NNLS weights costs in cc-pVTZ -
and then stopped, because the forward scheme had shipped without an adjoint. The dispatch
fell through to the ridge's, so the reverse pass differentiated an operator the forward
pass never applied: energies right, gradients silently wrong, ~4e5 uHa/rad of torque
against ~1e-1 for the same grids under `damped`. Since then it has raised rather than
answered.

`E` appears twice - in `Mj = E S E` and in the result `E G E` - so the adjoint carries
`2 (Mj_bar o S) e + 2 (B_bar o G) e` on top of the inner filter's term, closing through
`de_P / dS_PP = -e_P^3 / 2` on the rows that are scaled at all. Verified against a central
difference on a matrix whose diagonal spans four decades; against an identity that needs no
finite difference, since an exact inverse cannot see a similarity transform and the three
paths must cancel at `lambda = 0` (they do, to 1e-12); and end to end, on an assembled
nuclear gradient and on a torque checked by actually rotating each point set.

**What the ladder then said is not what this entry was written to find out.** Methanol,
matched supports, matched draws, `--scheme` the only difference, against a control that
reproduces FINDINGS section 13 to the second decimal:

| mode | 235/223 pts | 410/512 | 491/627 | 670/702 | converges |
| --- | --- | --- | --- | --- | --- |
| `blocked1` damped | 316.6 | 6.2 | 3.3 | 1.3 | 240x |
| `blocked1` damped_jacobi | 357.3 | 0.2 | 0.03 | **0.01** | **69121x** |
| `ghost` damped | 814.3 | 630.2 | 21.4 | 15.2 | 54x |
| `ghost` damped_jacobi | 731.9 | 22.1 | **0.09** | **0.21** | **3511x** |

The transferable grid is at 0.09 uHa/rad at 627 points, *below* the 0.24 section 13
measures for the complete 3284-point parent grid, and its energy is on the +2.75 uHa
parent-grid floor from 410 points up. But the explanation is the result. `blocked` and
`blocked1` are one support at two weight footings, and under the preconditioner they agree
to **ten digits in the energy**, because

    diag(S(w))_PP = w_P diag(S(1))_PP   =>   E(w) S(w) E(w) = E(1) S(1) E(1)

exactly, for any positive `w`. **The preconditioned metric inverse is invariant to the
collocation weights.** 4(7) localised the whole remaining gap between a transferable
support and an in-molecule one to the weight footing; 4(13) showed the footing cannot be
repaired by transplanting a weight set, because support and weights are one object. Both
are statements about a quantity that the preconditioner removes.

The sharper version of that test came from 4(9)'s supports rather than from
`blocked`/`blocked1`, where one support at two footings is true by construction: `eri` and
`eriw` - a support selected by an ERI fit, carrying the weights that same fit produced -
also collapse onto one row, agreeing to nine or ten digits at every rung. And the
preconditioner beats them at matched support: 565 points reads 0.35 uHa/rad with `eriw`'s
weights and **0.06** with none. 4(9)'s support survives intact; its weights are redundant.

Three things follow that nothing here has done:

* **Every ghost-gap ratio in this directory is now measurable on a fair footing for the
  first time.** 4(14) flags that they are all quoted against a `blocked` row held at
  `w = 1`, which handicaps the in-molecule baseline and flatters the transferable scheme,
  increasingly so with basis size. Under the preconditioner there is only one footing, so
  `sweep.py` / `analyse.py` re-run under `damped_jacobi` would give the first ghost gap
  that is not a footing artefact in either direction - and `ghost` reaching the floor at
  410-627 points rather than 702 says the number will move.
* **No trajectory has been run on it.** `trajectory.py --scheme damped_jacobi` now
  accepts the scheme. Section 13's 4.5 degrees of axis tilt per 2.5 ps was integrated
  from a torque 72x larger than the one this section measures, on a surface that was not
  even the THC one; 4(11) has since made propagating on the real surface possible.
* **One molecule, one basis, one seed**, and the gain does not appear until the grid
  resolves - on the smallest rungs the preconditioner does nothing useful and is
  occasionally slightly worse. Section 17's own energy result is cc-pVTZ and has never
  been run together with this.

A defect found on the way, recorded because it touches every number in section 17:
`build_aux_coulomb_inv` did not recognise the `_jacobi` suffix either, so a pipeline
asking for `damped_jacobi` silently got a *ridge* on the auxiliary Coulomb metric, on both
the forward and the reverse side. That metric is well conditioned and the two sides agreed
with each other, so nothing in 4(14) is invalidated - but its table was produced with a
filter it did not ask for, and is not bit-reproducible against today's code.


## 5. Open questions

* Does the 1.2-1.7x ratio hold in larger basis sets (cc-pVTZ) and for HF exchange rather
  than MP2 correlation? Everything here is cc-pVDZ / `ov` / MP2.
* The ratio improves from water to alanine. Does it keep improving, or turn around?
* ~~Can the free-atom fit be made well posed without a training set?~~ **Answered -
  `atomic_eri.py`, FINDINGS section 14.** Yes: fit the atom's own ERIs rather than its
  overlap matrix. `O(n_AO^4)` equations against `O(n_AO^2)`, still linear in the weights,
  same solver, ceiling lifted 4.6-6.5x, and the support costs 0.89-1.16x `blocked` at
  matched footing against ghosts' 1.11-1.78x. What is open now is whether that survives
  the 4(4) treatment - ten molecules, unseen bonding, an unseen partner element - and
  whether a ghost-*and*-ERI fit beats either alone.
* If weights are irrelevant, is NNLS still the best *selector*? Pivoted Cholesky and QRCP
  select points directly and are already implemented (`ls_ri_cholesky.py`, `ls_ri_qrcp.py`).
  A like-for-like selector comparison at matched point count has not been done here.
  Section 4(3) sharpens this: NNLS's point ceiling is its equation count, which is a
  property of the *target*, not of the selector. A selector working directly on the
  co-density would not have that ceiling, and might not need ghosts at all - or might
  need them for a completely different reason. **Section 4(9) took the other branch of
  exactly that sentence** - it kept NNLS and changed the target - and found that a target
  with enough equations does not need ghosts. The selector comparison is still unrun, and
  is now the more interesting half: against an ERI target, what does a direct co-density
  selector buy that NNLS does not?
* ~~Does the ghost ensemble matter?~~ **Answered - `ensemble.py`, FINDINGS section 9.**
  1.35x on methanol, 1.05x on ethanol, so it amortises; but each ensemble has a rank
  ceiling that caps reachable accuracy at any threshold, and the cheapest one tested
  fails ethanol on it. What remains open is whether the *rank* of a ghost ensemble can be
  raised cheaply. **Section 14 answers this: yes, by caging.** An icosahedral cage
  supplies 29x the nominal equations of the single-ghost ensemble and lifts the ceiling
  1.3-2.2x per element, which is the cheapest rank available - it costs environments
  nothing (10 against 13) and buys the accuracy in FINDINGS section 14. What is still
  open is the count-versus-geometry split, since the two ensembles are not matched in
  environment count. Section 4(4)'s second half narrows this usefully: raising the *variety*
  at fixed environment count is not merely neutral but actively harmful (a fourth partner
  element costs 1.23x -> 1.48x on propene), so the knob to turn is the number of
  environments, not the number of distinct neighbours among them. Nothing yet says how
  many environments buy how much rank.
* ~~Does one element's point set serve environments it was not fitted in?~~ **Answered -
  `transfer.py`, FINDINGS section 10.** Yes: 1.49x in-range against 1.50x out-of-range
  over ten molecules, with bonds up to 12% shorter than any ghost the fit saw. What is
  still open is the same question across *basis sets* rather than across molecules - the
  offline object is a list of indices into an element's level-0 atomic grid, and whether
  a cc-pVDZ-fitted support means anything in cc-pVTZ is untested.
* Does the overlap residual of a frozen support on a real in-molecule block predict
  anything? **No, and it is worth not re-deriving:** it is *anti*-correlated with the
  energy (Pearson -0.75, Spearman -0.60 over the section 4(4) suite). That is the third
  independent confirmation, after section 5's `rmsd_S` result and section 4(3)'s support
  agreement, that overlap-metric quantities carry no information about THC accuracy. A
  cheap SCF-free screen for grid quality cannot be built on the overlap target.
* How far can the offline object be pushed before it stops being a point set? Right now
  it is a list of indices into the element's level-0 atomic grid. Nothing forces that -
  the points could come from a finer parent grid, or from several parent grids unioned -
  but indexing into a fixed element grid is what makes the object trivially portable, and
  giving that up should be a deliberate choice.
* Does `rmsd_S` ever track accuracy, or should grid comparisons move to rank + min
  eigenvalue wholesale? Three independent results now say no (section 5, section 4(3)'s
  support agreement, section 4(4)'s residual), and the third is *anti*-correlated, so the
  answer is looking like "move wholesale".
* Does the parent-grid floor move between identical runs? Methanol's came out `+2.68` uHa
  in one `ghosts.py` run and `+2.86` in another differing only in its threshold list.
  0.18 uHa does not threaten any conclusion here, but section 4(1) says ridge inverts
  near-null directions at `1/lambda`, and a 3288-point parent grid is the most redundant
  metric in the set. `ridge.py` sets `mf.conv_tol = 1e-12`; `sweep.py` and `ghosts.py` do
  not. Noticed, not diagnosed - and it is exactly the kind of thing that matters more for
  a gradient (5) than for an energy.
* ~~**What is the right `lambda` for gradient work?**~~ **Answered - `window.py`,
  FINDINGS section 12.** Not by tuning `lambda` but by changing the filter. The third
  escape listed below - a better-conditioned treatment of the near-null directions - is
  the one that worked, in the cheapest available form: keep the overlap target, keep the
  grid, and replace `(S + lambda I)^-1` with `S (S^2 + mu^2 I)^-1`. Usable `lambda` drops
  from `1e-4` to `1e-8..1e-10` and the cost from 166-1808 uHa to under 5, on a PES
  smoother than the ridge's. The first escape (projecting the null space out) is now
  moot; the second (solving the fit as least squares directly) would be a cheaper route
  to the same filter and is still worth doing. **Superseded question below, kept for the
  record:**
* **What is the right `lambda` for gradient work, and is `1e-5` really the best available?**
  Section 4(5) brackets it at `1e-2..1e-5` on water: below that the null space is inverted
  into noise and the gradient moves by a factor of two between runs; above it the ridge
  costs 23 uHa at `1e-5` and 165 uHa at `1e-4`. So a gradient currently has to be bought
  with a hundred-fold worse energy than an energy calculation would accept, which is not a
  comfortable place to sit and may not be necessary. The obvious escapes were not tried:
  project the numerically null directions out of the metric before inverting rather than
  damping them; solve the `Z` fit as a least-squares problem directly instead of forming
  `S^-1` twice in the adjoint; or use a better-conditioned target than the overlap metric
  (which connects to the selector question above). One of these probably widens the window.
* Should `metric_ridge` become the default rather than an opt-in? It costs a few uHa and
  removes a discontinuity, which is the right trade for gradient work and the wrong one
  for reproducing published single-point energies. Section 4(5) adds that if the answer is
  yes for gradient work, the default value cannot be 4(1)'s `1e-8`. (The argument that 4(2)'s larger, more
  redundant grids would make the near-null crowd worse no longer applies, since those
  grids are not the recommended path - but the orbit-grouped grids do exist and were not
  run through `rank.py`, so what they do to the metric is unmeasured.)
* ~~How small does the rotation spread have to be?~~ **Wrong question - section 4(5).**
  The quantity that matters is `dE/dtheta`, it is now computable, and it does not track
  the spread: water's ghost grid has 1.9x the spread of its blocked grid and 30x the net
  torque. Section 4(2)'s deflation of orientation dependence rests on spreads and does
  not survive being differentiated. ~~What is still open is the part that was always the
  real question: does the torque converge away with grid size the way the spread does?~~
  **Answered - `torque_ladder.py --scheme damped --pinv`, FINDINGS section 13, and the
  answer is yes, in every mode.** Methanol at `pinv`: `blocked` 357 -> 0.02 uHa/rad over
  235 -> 670 points, `blocked1` 357 -> 9.0, `ghost` 969 -> 27, `ghostw` 698 -> 52. Water
  is 0.00 on every rung of every mode and on its unpruned parent.
  **[SUPERSEDED: the paragraph this replaced read "the answer is no" and quoted
  459 -> 16 against 4410 -> 2937. That ladder was taken at `ridge` `1e-3` and `1e-4`,
  because `--scheme` did not exist when the data was produced - it entered with the merge
  that followed. Section 12 says in terms that a torque there is mostly the regulariser,
  so those numbers measured the filter. The frame question is not live again.]**
  What the re-run leaves open is not the frame but the **weights**: `ghost` converges like
  `blocked1` (35.6x against 39.6x) and not like weighted `blocked` (17667x), and the
  complete parent grid at 0.24 uHa/rad is *above* the pruned weighted grid at 0.02. So
  pruning is free and the transfer is nearly free; the whole gap is that a transferable
  support cannot carry in-molecule weights. See the new questions at the end of this
  section.
* ~~What is the torque's effect over an actual trajectory?~~ **Answered - `trajectory.py`,
  FINDINGS section 13.** It reorients rather than heats. Over 2.5 ps of thermal tumbling
  on methanol the torque is **95% incoherent**; the component along `L` that could spin
  the molecule up is held at +0.36 hbar against a thermal 18.24 by energy conservation,
  since it is the only component that does work and the orientational potential's own
  amplitude (31 uHa) bounds it; what accumulates is the perpendicular component, **4.5
  degrees of rotation-axis tilt in 2.5 ps**. Water, which saturates, leaks 0.0027 hbar.
  The failure mode is slow artificial rotational diffusion, not drift in energy or in
  rotational speed. **What is still open is the requirement**: 4.5 deg / 2.5 ps is only
  tolerable relative to an observable, and nothing here computes one. In condensed phase
  real collisional decorrelation runs on ~1 ps and would bury it; gas-phase rovibrational
  structure would not. **[Historic, and wrong in its premise:** the paragraph below
  reasoned from water's 5592 uHa/rad, which section 12 showed was the ridge - water is
  rank-saturated and its torque is 0.00.**]**
* What is the torque's effect over an actual trajectory? 5592 uHa/rad net on water's
  208-point transferable grid is 13% of the nuclear gradient norm, which is not obviously
  negligible, but a torque is not an error bar - its consequence is secular drift in
  angular momentum, and nothing here integrates it. AIMD is the application the whole
  smooth-PES argument is for, so this is the measurement that decides whether the
  programme's main use case is actually served. **Section 12 raises the stakes on this
  rather than settling it**: the torque is now known not to shrink with grid size on the
  transferable grid, so it cannot be outrun and a trajectory is the only thing that says
  whether the magnitude matters. This is now the most informative single run left.
* **Do the ghost-gap ratios survive being re-taken on one footing?** 4(14) records that
  every ratio in this directory is quoted against a `blocked` row held at `w = 1`, which
  handicaps the in-molecule baseline and flatters the transferable scheme, worse with
  basis size. 4(15) removes the asymmetry rather than correcting for it - under the
  preconditioner there is one footing and `blocked` and `blocked1` are the same number -
  so `sweep.py` and `analyse.py` re-run under `--scheme damped_jacobi` would give the
  first ghost gap that is not a footing artefact in either direction. `ghost` reaching
  the parent-grid floor at 410-627 points under it, where `damped` needs 702 and does not
  quite get there, says the number will move. Nothing has been re-run.
* **What does the preconditioner do on a trajectory?** 4(15) measures a torque 72x
  smaller than the one 4(7)'s 4.5 degrees of axis tilt per 2.5 ps was integrated from,
  and 4(11) has made propagating on the real THC surface possible, so the two open
  trajectory questions collapse into one run: `trajectory.py --scheme damped_jacobi`,
  which the script now accepts and nobody has used.
* ~~Does the preconditioner beat 4(9)'s ERI-fitted weights, or compose with them?~~
  **Answered - FINDINGS section 18.** Neither: it absorbs them. `eri` and `eriw` collapse
  onto one row under the preconditioner, agreeing to nine or ten digits in the energy at
  every rung, on a support 4(9) selected and weighted for its own objective - which is a
  sharper test of the invariance than `blocked`/`blocked1`, where one support at two
  footings is true by construction. And it beats them: at 565 points `eriw` reads 0.35
  uHa/rad weighted against **0.06** preconditioned and unweighted, on the parent-grid
  energy floor where `eri` under `damped` is still 1.33 uHa above it at 852 points. 4(9)
  therefore splits - its **support** is untouched and still the best transferable one
  here, its **weights** are redundant rather than wrong.
* **Does the ten-digit collapse hold in cc-pVTZ?** 4(14)'s energy table is cc-pVTZ and
  4(15)'s ladder is cc-pVDZ, and the weight penalty the preconditioner replaces is the
  thing that grows with basis. The algebra is basis-independent; the accuracy it buys at
  a given point count is not.
* **Does any point-count ratio in 4(3) and 4(4) survive at a gradient-legal ridge?**
  Section 12 says probably not as quoted. Those sections evaluated `w = 1` grids at
  `RIDGE = 1e-8`; 4(5) forbids that for a gradient, and at `lambda = 1e-4` the same
  methanol grids come in at 1153-3119 uHa rather than 4-6 uHa. The matched-accuracy
  interpolation `analyse.py` performs is what produced "1.49x at 10 uHa", and 10 uHa is
  not reachable in the gradient's ridge window by any grid measured so far. Re-running
  the 4(4) suite at `lambda = 1e-4` would say what the scheme actually costs for the use
  case it is for. Nothing has been re-run.
* Is there a cheap way to reduce the torque without paying 4(2)'s orbit-grouping tax? The
  torque is `sum_P s_P x (dE/dr_P)`, computable with no SCF beyond the reference, so it
  could be used as an *objective* in the offline fit rather than only as a diagnostic -
  select a support that is stationary in orientation. Nothing here tries that.
* The orbit-grouped fit's octahedral invariance is exact in the algebra but lands at
  0.00-0.01 uHa in practice, and ridge does *not* remove it (methanol's 732-point grid:
  -0.0071 uHa truncated, -0.0170 uHa at `metric_ridge = 1e-8`). That is round-off
  amplified by the metric's ~1e-20 smallest eigenvalue, so it is a conditioning floor on
  how exact *any* symmetry of a unioned per-atom grid can be. Whether it also floors the
  gradient at a comparable relative size is unmeasured, and matters more than the energy
  does.
* **What is the requirement?** 4(7) measures the leak - 4.5 degrees of rotation-axis
  tilt in 2.5 ps on methanol's transferable grid - but "tolerable" is only meaningful
  against an observable, and nothing here computes one. In condensed phase, real
  collisional rotational decorrelation runs on ~1 ps and would bury it; gas-phase
  rovibrational structure computed from a dipole autocorrelation would not. Until some
  observable is put next to it, the number is a measurement without a threshold.
* **Does the leak stay bounded on a molecule with smaller rotational constants?** The
  saturation argument is physics - a conservative orientational potential cannot drive a
  molecule that turns through it, and 4(7) measures the torque as 95% incoherent over a
  tumble - so it should generalise. But a heavier molecule tumbles more slowly, which
  gives the torque longer to act coherently before it reverses, and the *size* of the
  plateau is what would change. Methanol at 48 AOs is the smallest molecule in this
  directory that does not saturate; ethanol or propene would say.
* **Does the energy converge before the torque does, and what does that cost?** On the
  ghost ladder, 414 points gives 3.11 uHa and 175 uHa/rad while 702 gives 3.15 uHa and
  27 uHa/rad - the same energy, 6.5x less torque, 1.7x more points. If a grid has to be
  sized for its torque rather than for its accuracy, that is ~1.7x on `n_P` and ~3x on the
  `n_P^2` work *on top of* the 1.6-2.7x 4(3) and 4(4) price the freeze at, and none of the
  compression figures in this directory include it. 4(9) changes this picture and has not
  been re-read against it: `eriw` reaches 0.19 uHa/rad at 565 points, *below* its own
  accuracy plateau's point count rather than above it, so on that mode the torque is not
  what sizes the grid. Whether that holds beyond methanol is unmeasured.
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
uv run python experiments/atom_centered_grids/ensemble.py --saturate H,C,O
uv run python experiments/atom_centered_grids/ensemble.py --calibrate H,C,O
uv run python experiments/atom_centered_grids/ensemble.py methanol --out ens_methanol.json
uv run python experiments/atom_centered_grids/ghosts.py methanol --out ghosts_rot_methanol.json \
    --thresholds 3e-4,3e-5,1e-12 --blocked-thresholds 1e-3,1e-4,1e-5 --rotate 4
uv run python experiments/atom_centered_grids/transfer.py --residual
uv run python experiments/atom_centered_grids/transfer.py --out transfer.json
uv run python experiments/atom_centered_grids/transfer.py --partners H,C,N,O \
    --molecules hcn,methylamine,methanol --out transfer_npartner.json
uv run python experiments/atom_centered_grids/transfer.py --molecules propene,acetonitrile \
    --out transfer_large.json
uv run python experiments/atom_centered_grids/transfer.py --molecules acetonitrile \
    --thresholds 1e-3,3e-4,2e-4,1e-4,6e-5,3e-5,2e-5,1e-5 \
    --blocked-thresholds 3e-2,1e-2,3e-3,1e-3,1e-4 --out transfer_acn.json
uv run python experiments/atom_centered_grids/transfer.py --report transfer*.json
uv run python experiments/atom_centered_grids/gradient.py water --out gradient_water.json
uv run python experiments/atom_centered_grids/gradient.py water --mode ghost \
    --threshold 3e-4 --out gradient_water_ghost.json
uv run python experiments/atom_centered_grids/window.py water --mode blocked \
    --threshold 1e-3 --schemes ridge,ridge_eigh,damped --out data/window_water_blocked.json
uv run python experiments/atom_centered_grids/window.py methanol --mode ghost \
    --threshold 3e-4 --perms 2 --out data/window_methanol_ghost.json
uv run python experiments/atom_centered_grids/window.py --report data/window_*.json
uv run python experiments/atom_centered_grids/scan.py methanol 1e-3 --no-refit \
    --lambdas 1e-8 --damped-lambdas 1e-8,1e-10 --out data/scan_methanol_damped.json
uv run python experiments/atom_centered_grids/torque_ladder.py methanol \
    --scheme damped --ridges 1e-8,1e-10 --pinv --draws 4 \
    --modes blocked,blocked1,ghost,ghostw --out data/torque_ladder_methanol_damped.json
uv run python experiments/atom_centered_grids/torque_ladder.py water --parent \
    --scheme damped --ridges 1e-8,1e-10 --pinv --out data/torque_ladder_water_damped.json
uv run python experiments/atom_centered_grids/torque_ladder.py methanol \
    --modes parent,parent1 --scheme damped --ridges 1e-8 --pinv --draws 2 \
    --out data/torque_ladder_methanol_parent.json
uv run python experiments/atom_centered_grids/torque_ladder.py \
    --report data/torque_ladder_methanol_damped.json --report-ridge pinv
uv run python experiments/atom_centered_grids/trajectory.py methanol \
    --modes hf,blocked,ghost --steps 5000 --torque-every 4 --feedback ghost \
    --out data/traj_methanol_tumbling.json
uv run python experiments/atom_centered_grids/trajectory.py methanol \
    --modes hf,blocked,ghost --steps 3000 --torque-every 2 --feedback ghost \
    --project-rotation --out data/traj_methanol_fixed.json
uv run python experiments/atom_centered_grids/trajectory.py methanol --modes ghost \
    --steps 200 --propagate full --project-rotation --out data/traj_methanol_coupled.json
uv run python experiments/atom_centered_grids/trajectory.py methanol --modes hf \
    --reference rks --xc pbe --grid-level 0 --steps 2000 --out data/traj_methanol_rks.json
uv run python experiments/atom_centered_grids/insitu.py --calibrate
uv run python experiments/atom_centered_grids/insitu.py --out data/insitu.json
uv run python experiments/atom_centered_grids/torque_ladder.py formaldehyde \
    --modes ghost,ghostw,molw,molfit,molfit1,blocked,blocked1 \
    --thresholds 1e-3,3e-4,1e-4 --ridges "" --pinv --draws 4 \
    --out data/torque_insitu_formaldehyde.json
uv run python experiments/atom_centered_grids/torque_ladder.py formaldehyde \
    --modes ghost,ghostw --thresholds 3e-5,1e-5 --ridges "" --pinv --draws 4 \
    --out data/torque_insitu_formaldehyde_tight.json
uv run python experiments/atom_centered_grids/trajectory.py --report data/traj_*.json
uv run pytest tests/test_thc_gradient.py
```

`torque_ladder.py` is minutes per molecule with the pruned modes and about half an hour
for methanol's 3284-point parent rungs. **Always pass `--pinv`**: both regularised schemes
reduce to the pseudoinverse as `lambda` falls, and without that control row a trend down
the rungs cannot be told from the filter's own orientation dependence - which is exactly
the trap the first ladder fell into.

`trajectory.py` costs one SCF, one reference gradient and one THC gradient per carried
grid per sampled step. Three things to know before running it. `--torque-every` is what
makes a picosecond affordable: the torque varies on the vibrational timescale while the
integrator needs half a femtosecond, so sampling every 2 steps reproduces the leak to 0.7%
and every 4 to 4%. Several grids ride one trajectory, so the comparison between them
carries no sampling difference - but only the `--feedback` grid gets its own orientation.
And `--propagate full` is a diagnostic, not a production mode: the fixed-orbital force is
not the gradient of the energy it propagates, and the trajectory gains angular momentum
fifty times faster than any grid in this directory leaks it.

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

`ensemble.py` runs the same fits across four ghost ensembles. `--saturate` is the one to
run first: it needs no SCF, takes a couple of minutes, and its per-element numbers summed
over a molecule give a hard ceiling on the grid that ensemble can produce. If that
ceiling is below what `blocked` needs for the target accuracy, no threshold will save it
and there is no point running the single points. `--calibrate` is the threshold ladder on
top of that. Stage B costs four ensembles' worth of single points - about 15 min on
methanol, an hour on ethanol.

`transfer.py` fits every element once up front and then only translates, so its cost is
all in the single points - one parent grid, a blocked ladder and a frozen ladder per
molecule, about 10 of them. The default eight-molecule suite is roughly two hours on 4
cores. `--residual` needs no SCF and runs in seconds; run it first to see the ghost shells
printed against nothing, then the suite. Two ladder traps, both hit in the runs behind
FINDINGS section 10: the default `--blocked-thresholds` starts too tight for molecules
with a large parent-grid floor (acetonitrile's loosest blocked grid is already 1.4 uHa
from its floor, which makes every matched-accuracy cell degenerate - widen to `3e-2,1e-2`),
and molecules below ~40 AOs saturate, so their rows are unreadable however the ladder is
set. The report marks degenerate cells `*` and leaves them out of the statistics; if a
molecule is all `*`, widen the blocked ladder rather than believing the ratio.

`gradient.py` costs one analytic gradient plus `2 * 3 * natm` frozen-orbital energies per
ridge per step size, so it is quadratic in nothing and just slow in proportion to the
ridge ladder: water's default ladder is a few minutes, and `--atom N` restricts the finite
difference to one atom when only the verification is wanted. The analytic gradient alone
is one forward pass and one reverse pass - comparable to a single THC-MP2 energy, plus an
`(n_ao, n_ao, n_aux)` integral array and an `(n_P, n_P, n_occ)` intermediate held in core,
which is what keeps `pythc.grad` a reference implementation rather than a production one
and caps it near alanine. Two traps, both hit while writing it: a finite-difference step
that moves the metric's null directions by more than the ridge shift measures
nonlinearity rather than a derivative, so `--steps` and `--ridges` are not independent
knobs; and the torque must be finite-differenced about each atom's *own* torque axis,
since a fixed lab axis nearly orthogonal to it returns noise.

`window.py` runs `(1 + n_schemes * n_lambdas) * (1 + n_perms)` analytic gradients and no
finite difference, so it is minutes on water, ~2 min on methanol and ~10 min on ethanol at
`--perms 2`. Two things to know before running it. The `pinv` control row is not optional:
both regularised schemes converge to it as `lambda` falls, and without it a torque
measured at one `lambda` cannot be distinguished from the regulariser's own orientation
dependence - which is exactly the trap 4(5) fell into. And a molecule below ~40 AOs
saturates the co-density rank, which drives the true torque to zero, so water is a
verification system here and not a measurement; read methanol or ethanol for the numbers.

`ghosts.py` fits each element in seconds - the offline stage is genuinely cheap, and
independent of the molecule - so its cost is all in the single points: 16 of them on the
default ladder, about 12 min on methanol and an hour on ethanol, plus `--rotate N` extra
per grid. Run `--calibrate` first; its threshold ladder is not on the `blocked` scale and
a ladder guessed by analogy lands off the interesting range. Water is not worth running
for the ratio - at 24 AOs every grid saturates the co-density rank and all three modes
reach the floor.

`insitu.py` is the cheapest measurement in this directory: the fits are seconds per
element like `ghosts.py`'s, and the full seven-mode, three-rung ladder over six molecules
is about 20 minutes. Two warnings, both learned the hard way in 4(13). Read every
comparison at **matched point count** - the modes do not agree on size at a shared
threshold, `molw` is always smaller than the `ghost` support it started from because
non-negativity binds, and `matched_size_report` exists for this. And do not take a ratio
from a torque quoted at one rung: `ghostw` on formaldehyde runs 163 uHa/rad at 344 points,
1.00 at 427 and 3.74 at 486, so a single-rung pair can be made to say almost anything.
Prefer the energy and spread ladders, which carry six molecules behind them. Water is not
worth running here either, for the usual reason.

Raw output from the runs behind `FINDINGS.md` is in [`data/`](data).

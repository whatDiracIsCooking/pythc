# What acTHC has to pass to be called viable

Read [`HANDOFF.md`](HANDOFF.md) for what the idea is and [`FINDINGS.md`](FINDINGS.md) for
the measurements. This file is narrower: it assumes both, and asks the one question they do
not answer - **which experiments would turn "no objection survives" into "this works"**.

The distinction matters because the programme has reached a state that is easy to
misread. Eighteen sections have each removed an objection, and §18 removed the last
structural one. That is *not* the same as having demonstrated the scheme, and the gap is
not more experiments of the kind already run.

## 1. Why the question is live

Three results in a row made acTHC a candidate rather than a research direction:

* **§14** removed the training set. Fitting a free atom's own two-electron integrals gives
  `O(n_AO^4)` equations instead of the overlap's `O(n_AO^2)`, lifts the support ceiling
  4.6-6.5x, and selects a support costing **0.89-1.16x** the in-molecule `blocked` grid -
  better than thirteen fabricated ghost environments manage. The offline object needs
  nothing but an isolated atom.
* **§15** built the orbital response, so there is no unimplemented component. The
  rotational identity closes to **6.1e-9** and NVE on the THC surface drifts **0.247
  uHa/step** against the fixed-orbital force's 21.68.
* **§18** removed the weight footing, which §13 had localised the entire
  transferable-support gap to and §16 had shown could not be repaired by transplanting a
  weight set. Under `E = diag(1/sqrt(diag S))`, `diag(S(w))_PP = w_P diag(S(1))_PP` gives
  `E(w) S(w) E(w) = E(1) S(1) E(1)` exactly, so the preconditioned metric inverse cannot
  see the weights. The transferable support at `w = 1` reaches **0.09 uHa/rad** - below the
  0.24 §13 measures for the *complete* 3284-point parent grid - and lands on the
  parent-grid energy floor.

So the object is buildable, differentiable, transferable, and no longer penalised for
being unable to carry in-molecule weights.

## 2. Why that is not yet a demonstration

**No single configuration in this directory has been measured on more than one axis.**
Every headline number was taken under whichever metric filter was current when its section
was written, and the filters are not interchangeable - §12 and §13 both turn on exactly
that point (`a torque is not a property of a grid; quote it with the regularisation that
produced it`).

| headline | § | filter it was measured under | baseline's weights |
| --- | --- | --- | --- |
| point tax 1.2-1.7x (global vs blocked) | 1 | truncated `pinv` | NNLS |
| ghost gap 1.1-1.8x | 8 | `ridge 1e-8` | `w = 1` |
| transferability 1.49x in / 1.50x out, ten molecules | 10 | `ridge 1e-8` | `w = 1` |
| ERI support 0.89-1.16x | 14 | `ridge 1e-8` | `w = 1` |
| cage 1.22x; cc-pVTZ reversal | 17 | `ridge 1e-8` | `w = 1` |
| gradient legality, usable `lambda` | 12 | `damped 1e-8..1e-10` | - |
| torque converges with grid size | 13 | `pinv` | both |
| trajectory: 4.5 deg tilt / 2.5 ps | 13 | `damped` | `w = 1` |
| weight footing erased, 0.09 uHa/rad | 18 | `damped_jacobi 1e-8` | none |

Read down the filter column. The configuration §18 recommends - **preconditioned filter,
no weights, ERI-fitted support** - has **never had a single point-count ratio measured for
it**, and the configuration every compression ratio was taken under (`ridge 1e-8`) is one
§12 disqualifies for gradient work. §14 flags the asymmetry from the other side: every
ghost-gap ratio here is quoted against a `blocked` row held at `w = 1`, which handicaps the
in-molecule baseline and **flatters the transferable scheme**, worse with basis size.

**What *is* held fixed, checked against every result file rather than the write-ups:** the
AO basis. 46 of the 49 data files that record one are `cc-pVDZ` / `cc-pVDZ-RI`; the eight
scripts that record no basis (`rank`, `ridge`, `scan`, `rotate`, `weights`, `window`,
`gradient`, `orbits`) all `from sweep import BASIS`, which is `cc-pVDZ`. **The only
exceptions are §17's three `levers.py` runs**, which are cc-pVTZ. So the directory is
single-basis apart from §17 - and that is precisely the problem T9 records, since §17
(cc-pVTZ) and §18 (cc-pVDZ) are the two most recent results and have never been run
together. Both arms of the trajectory comparison are cc-pVDZ, so the DFT head-to-head in
T8 is basis-matched; what it is *not* matched in is method, since it puts MP2 correlation
on a THC grid against RKS/PBE on a Becke grid. That is the right comparison for `how fast
does a lab-fixed atom-centred grid leak angular momentum` and the wrong one for anything
else.

Three further things are simply absent:

* **No cost measurement of any kind.** There is no wall-clock or crossover study anywhere
  in `FINDINGS.md`. The `n_P` tax is a *proxy* for cost, and `pythc.grad` is explicitly a
  reference implementation that holds `(n_ao, n_ao, n_aux)` and `(n_P, n_P, n_occ)` in core
  and caps near alanine. Compression ratios are not a speedup.
* **No head-to-head against the DFT anyone would actually run.** §13 measures 4.5 degrees
  of rotation-axis tilt per 2.5 ps and `HANDOFF.md` says of it, three times, that it is
  `a measurement without a threshold`. The threshold is not missing because nobody picked
  a tolerance - it is missing because the comparison stops at level-0 Becke, the grid
  acTHC prunes from, which acTHC beats 8x. Against level-3 it loses 25x. See T8.
* **No trajectory on the real surface.** Every leak number in §13 was integrated along a
  *DF-RHF* trajectory with the frozen grid's torque carried along it, from a torque **72x
  larger** than the one §18 measures. `trajectory.py` still calls the fixed-orbital
  `thc_mp2_gradient` - its module docstring says `response is not implemented` - so §15's
  relaxed gradient has never driven a trajectory longer than the twelve water steps in
  `tests/test_thc_response.py`.

## 3. The tests

Ordered so that each tier is worthless until the one above it is done. Costs are scaled
from the runtimes recorded at the end of `HANDOFF.md` (4 cores).

### Tier 0 - fix the configuration. Nothing below is quotable until this is done.

**T1. `eri` against `cage` on one torque ladder under the preconditioner.**
The two best supports here have never been compared on a single ladder - `HANDOFF.md`
4(15) says so in terms - and `cage`'s torque is unmeasured, which 4(14) flags as the
reason not to believe §17 is good news for the application: a cage-fitted support is
trained on a fragment of the atom, so it could plausibly be *worse* in orientation while
being cheaper in points.

*Code work:* `torque_ladder.py` has `eri`/`eriw` modes but **no `cage` mode**; the cage is
a `cage=True` kwarg to `levers.element_supports`, reachable only from `levers.py`. Add it
to `MODES` and `DEFAULT_THRESHOLDS`.

```shell
uv run python experiments/atom_centered_grids/torque_ladder.py methanol \
    --modes blocked,blocked1,eri,cage,ghost --scheme damped_jacobi \
    --ridges 1e-8,1e-10 --pinv --draws 4 --out data/viab_support_choice.json
```

*Pass:* one support is better on both energy and `|tau|` at matched point count, or the two
are indistinguishable and the cheaper fit wins. *Cost:* hours on methanol.
*Falsifies nothing on its own* - it chooses the object the rest of the plan measures.

**T2. Plumb the scheme through the cost scripts.**
`sweep.py`, `ghosts.py`, `transfer.py`, `atomic_eri.py` and `insitu.py` have no `--scheme`
flag at all: `ghosts.thc_mp2` passes `metric_ridge=RIDGE` and takes the `"ridge"` default,
and the other three import `RIDGE` from it. `levers.py` plumbs `--schemes` only inside
`--blocked-diagnostic`; its stage-B `record` calls `thc_mp2` without a scheme.

*Code work:* one keyword in `ghosts.thc_mp2`, a `--scheme` flag in each driver, and
`mf.conv_tol = 1e-12` in `sweep.py` and `ghosts.py` - which `levers.py` already sets and
which is the open suspect for the parent-grid floor moving 0.18 uHa between runs. Cheap,
and it is the gate on Tier 1.

### Tier 1 - re-take the cost on the footing the scheme will actually run at.

**T3. The global-vs-blocked ratio under `damped_jacobi`.** §1's 1.2-1.7x is at the
truncated pseudoinverse with NNLS weights, which no production configuration uses.

```shell
uv run python experiments/atom_centered_grids/sweep.py alanine \
    --scheme damped_jacobi --out data/viab_sweep_alanine.json
uv run python experiments/atom_centered_grids/analyse.py data/viab_sweep_alanine.json
```

**T4. The transferable tax under `damped_jacobi`, over the ten-molecule suite.** This is
the number the programme is quoted on, and it is the one most likely to move: §18 notes
that `ghost` reaches the parent-grid floor at 410-627 points under the preconditioner where
`damped` needs 702 and does not get there, and §14 warns the published gaps flatter the
scheme. Under the preconditioner there is only **one** footing, so this is the first
ghost-gap number that is not a footing artefact in either direction.

```shell
uv run python experiments/atom_centered_grids/transfer.py --scheme damped_jacobi \
    --out data/viab_transfer.json
uv run python experiments/atom_centered_grids/transfer.py --report data/viab_transfer.json
```

*Pass:* the compounded tax stays inside the **1.6-2.7x** band already published (2.6-7.3x
on the `n_P^2` parts). *Falsify:* above ~3x, the headline compression claim was a footing
artefact and every ratio in this directory needs restating. Widen the blocked ladder to
`3e-2,1e-2` and ignore the 33-38 AO molecules, both per §10's own traps. *Cost:* ~2 hours.

**T5. Confirm the accuracy targets are reachable at a gradient-legal filter.** An open
question in `HANDOFF.md` §5 that nothing has touched: §8-§10's matched-accuracy
interpolation is done at 10 uHa, and `HANDOFF.md` records that at `ridge 1e-4` the same
methanol grids come in at 1153-3119 uHa rather than 4-6. §12's damped filter is supposed to
have removed that (under 5 uHa across five grids), but it has never been checked *on the
ladders the ratios are interpolated from*. T3 and T4 settle this as a side effect - if
10 uHa is not reachable, the matched-accuracy cells are degenerate and the ratios are not
defined.

### Tier 2 - the application. This is the viability test.

**T6. AIMD on the relaxed THC surface.** The single most informative run left. §13's leak
was inferred first-order from a DF-RHF trajectory at a torque 72x larger than §18's;
§15 made the real thing possible and nothing has run it.

*Code work:* wire `pythc.grad.total.ThcMP2Gradients.as_scanner()` into
`trajectory.py`'s `Surface`, replacing the `thc_mp2_gradient` + reference-gradient sum that
`--propagate full` currently uses. The scanner exists and returns relaxed `(e_tot, de)`;
the torque path stays as it is. Until this is done `--propagate full` remains what its
docstring calls it - a diagnostic that gains angular momentum fifty times faster than any
grid here leaks it.

```shell
uv run python experiments/atom_centered_grids/trajectory.py methanol \
    --modes <T1 winner> --scheme damped_jacobi --propagate full \
    --steps 5000 --torque-every 2 --out data/viab_traj.json
```

*Report three things, all against controls already in §13's table at matched times.* The
angular-momentum row is not a tolerance - it is a head-to-head against DFT, for the reason
T8 sets out.

| quantity | reference already measured | what to report |
| --- | --- | --- |
| `\|L - L0\|` at matched times | RKS/PBE level-0 Becke: 0.638 / 2.380 / 9.250 hbar on 4656 points | the same three times, on the same initial condition |
| | RKS/PBE level-3 Becke: 0.004 / 0.028 hbar on 67432 points | ditto - this is the baseline that matters, see T8 |
| NVE drift per step | §15: 0.247 uHa/step (water, 12 steps, `ridge 1e-2`) | drift on the same integrator on the grid-free DF-RHF surface, which isolates the grid from the integrator |
| axis tilt over 2.5 ps | §13: 4.5 deg, from a 72x larger torque | ditto |

*Falsify:* the leak does not improve on §13 despite the 72x smaller torque - which would
mean the torque is not what sets the leak and the integrator or the response is. *Cost:*
the response is `3 N` coupled-perturbed solves per step, so this is the expensive run
here; start at 500 steps to establish the rate before committing to a picosecond.

**T7. A second molecule, and one with smaller rotational constants.** §18 is one molecule,
one basis, one seed. The saturation argument in §13 is physics - a conservative
orientational potential cannot spin up a molecule that turns through it - but a heavier
molecule tumbles more slowly, giving the torque longer to act coherently, and the *size* of
the plateau is what changes. Methanol at 48 AOs is the smallest molecule here that does not
rank-saturate; ethanol or propene would say. Run T1 and T6 on one of them.

### Tier 3 - the requirement. The bar is DFT, not a tolerance - and both sides have a grid-level knob.

**T8. Find acTHC parameters whose rotational diffusion is no worse than production DFT's.**

`HANDOFF.md` asks three separate times what the requirement is and calls 4.5 deg / 2.5 ps
`a measurement without a threshold`. The temptation is to invent an absolute tolerance -
"the spectrum must be clean to within line widths" - and that is the wrong test, because
**DFT does not pass it either**. §13 already made exactly this point and it is the frame to
carry forward: lab-fixed atom-centred quadrature is not an acTHC defect, it is what the
whole family does, and the pruned support is quieter at it than its own parent grid.

So state the requirement comparatively, and existentially:

> **There exist acTHC parameters for which rotational diffusion is comparable to DFT's** -
> and preferably, **for which it is less bad than DFT's**.

That is a decidable claim with no invented threshold in it, because the baseline is a
competing method's measured number rather than a tolerance nobody has set. It is also
weaker and cheaper to establish than an absolute standard would be: the deliverable is a
**witness** - one configuration (support type, threshold, `n_P`, parent level, filter,
`lambda`) - not a property of every configuration.

**Where §13 leaves the claim today, which is the honest starting point:** acTHC sits
*between* the two DFT baselines. Against RKS/PBE on level-0 Becke - the grid the fits are
pruned from, 4656 points - the frozen supports win by **4.4x at 100 fs and ~8x at a
picosecond**. Against level-3 Becke, 67432 points, they lose by **~25x** on 163x fewer
points. So "less bad than DFT" is already true of the DFT you would not run, and not yet
true of the DFT you would. **Which baseline is named decides whether the claim is
interesting**, and only level 3 (or whatever the production setting is) is.

**Levels 1 and 2 have never been run, and the crossing has to be in there.** §13 samples
the DFT knob at exactly two points, at opposite ends:

| Becke level | points (methanol, cc-pVDZ) | vs level 0 | `\|L - L0\|` measured |
| --- | --- | --- | --- |
| 0 | 4,656 | 1.0x | 0.638 / 2.380 / 9.250 hbar |
| **1** | **20,248** | **4.3x** | **never run** |
| **2** | **43,904** | **9.4x** | **never run** |
| 3 | 67,432 | 14.5x | 0.004 / 0.028 hbar |

acTHC on ~700 points is 8x better than the top row and 25x worse than the bottom one, so
it crosses DFT *somewhere in the two rows nobody has measured*, and locating that crossing
is the whole content of the claim: "as rotationally clean as a level-2 Becke grid on 60x
fewer points" is a result, and "between level 0 and level 3" is not. **Run these two
first.** They need no code at all - an existing, working path (`--reference rks
--grid-level 1,2`), one SCF and one gradient per step, no THC, no frozen grid, no `3 N`
coupled-perturbed solves - and they sharpen the target T6 is aiming at before T6 is paid
for.

Measured on 4 cores, methanol/cc-pVDZ, PBE with `grid_response=True`, the per-step cost is
**nearly flat in grid level** - 1.5-1.6 s at levels 1, 2 and 3 alike, since a 48-AO
molecule's SCF is not grid-bound at these sizes. So each 2000-step arm is **under an hour**
and the refinement is close to free. Note the existing level-3 row was run to only 600
steps, which is why its 1 ps cell is a dash; match level 0's 2000 so all three of §13's
time points exist.

The reason to expect the gap to have moved: every §13 trajectory number was integrated
from a torque **72x larger** than the one §18 measures under the preconditioner, on a
surface that was not even the THC one. Nothing has re-measured the leak since.

*How to run it.* Both methods have a knob - DFT buys less rotational diffusion with grid
level, acTHC with `n_P` and, per T8b, with its own parent level - so this is a comparison
of curves, not of two points. A
trajectory ladder is unaffordable (T6 costs `3 N` coupled-perturbed solves per step), so
screen first and propagate only the candidates:

1. **Fill in the DFT ladder at levels 1 and 2** - the table above, and the cheapest run
   in this file. Needs no code and no THC:

   ```shell
   for lv in 1 2; do
     uv run python experiments/atom_centered_grids/trajectory.py methanol --modes hf \
         --reference rks --xc pbe --grid-level $lv --steps 2000 \
         --out data/viab_traj_rks_l$lv.json
   done
   ```

2. **Screen the acTHC candidates on the torque**, which needs no SCF beyond the reference
   and is already laddered. §13 reports `log|tau|` tracking the rotation spread at Spearman
   **+0.89** and grid accuracy at **+0.86** over 20 rungs, so it orders candidates cheaply.
   Treat it as a screen and not as proof - §13's own warning is that a torque quoted at one
   rung is not a statistic, and §16's `ghostw` swings 163 -> 1.00 -> 3.74 uHa/rad across
   three consecutive rungs.
3. **Propagate the best two or three rungs** on one initial condition, reporting
   `|L - L0|` at matched times against the four-level DFT ladder. `trajectory.py` already
   carries several grids along one trajectory, so the acTHC arms cost nothing extra in
   sampling. Add level 4-5 only if level 3 is beaten.

*Pass, in increasing strength:* (a) some acTHC configuration beats level-0 DFT - **already
true in §13**, and the re-run should widen it; (b) some configuration **matches level 1 or
level 2** at 30-60x fewer points, which is a publishable statement and the outcome to
expect; (c) some configuration is **within a small factor of level 3**, which makes the
claim a cost argument and hands off to T13; (d) some configuration **beats level 3
outright**, which ends the orientation question permanently. *Falsify:* no rung reaches (b)
even as `n_P` grows - i.e. the leak plateaus above the cheapest DFT anyone would run and
cannot be bought down - which would mean the frozen grid has a floor the quadrature it
replaces does not.

**T8b. The other level knob: prune from a level-1 or level-2 parent and measure the
torque.** There are *two* grid-level ladders in this comparison, and the second one is
acTHC's own. Every support in this directory is selected from a **level-0** atomic parent,
and 4(10) is marked `PARTLY DONE` for exactly this reason: `levers.py` measured a level-1
parent on the **ceiling and the energy** (ceilings H 132 -> 187, C 155 -> 216, O 199 -> 251
at *unchanged* equation count; methanol 1.78x -> 1.46x at 10 uHa) and **nothing has ever
measured a torque or a rotation spread on a finer-parent support**. Level 2 has not been
tried at all.

This is the one lever whose original motivation was orientation, and it is the obvious
suspect for the remaining gap: §13's own control shows the level-0 Becke grid is a **poor
grid to be rotationally invariant on** - it is the 4656-point row that acTHC beats 8x and
level 3 beats 25x - so a support pruned from it may be inheriting its parent's anisotropy.
If the crossing T8 is looking for sits at level 1 or 2 on the DFT side, the cheapest way to
move acTHC past it may be to prune from a level 1-2 parent rather than to add points at
level 0. 4(10) already shows the ceiling and the energy both improve, so the point tax of
doing so is bounded and known.

*Code work:* `ghosts.element_grid`, `element_supports` and `assemble_from_supports` all
take `level` already, but **`torque_ladder.py` hardcodes `level=0`** (line 137) and so does
**`trajectory.py`** (line 128), so neither instrument can see this lever today. Thread
`--parent-level` through both. Small, and it unlocks the one 4(10) item that is still open.

*Pass:* a finer-parent support at matched `n_P` has a lower torque, which would convert
4(10) from an energy lever into an orientation lever and feed straight back into T1's
choice of support. *Falsify:* the torque is flat in parent level, which closes 4(10)
entirely and says the anisotropy is the support's own rather than inherited.

*Optional, and no longer load-bearing:* a **dipole autocorrelation spectrum** on both
surfaces. Under the comparative framing this is no longer needed to set the bar - `|L - L0|`
against DFT is the bar - but it answers the separate question of whether any of this is
physically visible, and it is the thing to run if (b) holds and someone asks whether the
residual factor matters. In condensed phase real collisional decorrelation runs on ~1 ps
and would bury the whole effect; gas-phase rotational structure would not.

### Tier 4 - generality. These are the claims that are currently one molecule, one basis.

**T9. cc-pVTZ.** The largest single risk, and it cuts two ways. §17's energy result is
cc-pVTZ and §18's collapse is cc-pVDZ, and *the two have never been run together*. The
algebra behind the collapse is basis-independent; the accuracy it buys at a given point
count is not. Worse, **the cage does not survive the basis change** - carried to cc-pVTZ it
is 1.2-1.4x *worse* than the single-ghost ensemble it beats by 1.5x in cc-pVDZ - so if T1
picks `cage`, T9 may unpick it. Note §17's own warning to chunk the cc-pVTZ cage rungs:
`levers_methanol_tz_wide.json` was killed by the machine mid-rung.

**T10. Does the offline object survive a change of basis at all?** Open in §5 and untouched.
The product is a list of indices into an element's level-0 atomic grid; nothing asks what a
cc-pVDZ-fitted support means in cc-pVTZ. Note this is a convenience question, not a
structural one - acCD refits per basis offline, and so may acTHC - but the answer decides
whether the offline stage is run once per element or once per (element, basis).

**T11. A method other than `ov`/MP2.** Everything here is MP2 correlation in `ov` mode.
HF exchange is the open question in §5 and the cheapest generalisation to test.

**T12. Does the ratio keep improving with size?** §1 has it improving from water to
alanine, which is the whole amortisation argument, and alanine is the largest molecule in
the directory. One system substantially larger would say whether it keeps improving or
turns around. This is also where the memory ceiling in `pythc.grad` binds, which makes it
partly a T13 question.

### Tier 5 - the claim nobody has tested.

**T13. Measure the cost.** acTHC's case is a *compression* ratio, and a compression ratio
is not a speedup. Needed: wall-clock and memory for the THC-MP2 energy and the relaxed
gradient against conventional DF-MP2 on the same systems, over a size series, with the
crossover point identified. Two specific costs to price, both flagged in the directory and
neither measured:

* `damped_inv` costs a **full eigendecomposition** of `S` where `ridge_inv` costs a
  Cholesky - ~3x on that step - and the whole selling point of THC is that the expensive
  parts are `n_P^2`. An `O(n_P^3)` `eigh` per geometry may dominate. §12 lists the escape
  (solve the `Z` fit as least squares directly, reaching the same filter by construction)
  and nobody has taken it.
* The response is `3 N` coupled-perturbed solves where a **Z-vector** formulation needs
  one. `response_lagrangian` already returns the intermediate it would contract. This is
  the difference between a reference implementation and production dynamics, and it caps
  T6's run length.

**T14. State the result against the prior art.** Kokkila Schumacher *et al.* (JCTC **11**,
3042, 2015) generate per-atom, per-basis THC grids for first-row atoms at **<100
points/atom** in cc-pVDZ *and* cc-pVTZ with negligible error. The supports here land at
**100-147 points/element** where they reach 5-10 uHa - the same order, consistently on the
expensive side. `HANDOFF.md` calls reading that SI the single most useful external check
available and it has not been done. A viability claim that does not place itself against
that number is not a viability claim.

## 4. What would falsify viability

Stated up front, so the tests are not read as a formality:

1. **T4 above ~3x.** The compression headline was a footing artefact.
2. **T6 no better than §13.** A 72x smaller torque that does not produce a smaller leak
   means the leak is not the torque, and the whole §12-§18 arc was optimising the wrong
   quantity.
3. **T8 finds no witness.** If no acTHC configuration gets within a small factor of
   production-level DFT's rotational diffusion, and the leak plateaus as `n_P` grows rather
   than converging, then the frozen grid has an orientation floor that the quadrature it
   replaces does not - and the smooth-PES argument buys a surface nobody can run dynamics
   on. Note this is a *weaker* bar than it sounds and still the right one: acTHC does not
   have to be rotationally clean, only cleaner than what it replaces at comparable cost.
4. **T9 reverses T1.** If the best support in cc-pVDZ is not the best in cc-pVTZ - which
   §17 already observes for the cage - then "fit once per element offline" has a basis
   dependence the acCD analogy does not, and every ratio needs re-taking per basis.
5. **T13 finds no crossover below the memory ceiling.** A scheme that compresses `n_P` but
   is not faster than DF-MP2 anywhere reachable is a result, not a method.

## 5. What not to re-run

Recorded so the plan is not padded with settled questions. All of these are closed, several
of them twice:

* **Orbit-grouped / group-NNLS pruning** (§7, 4(2)). Delivers exact octahedral invariance
  and is worse than point-wise at matched size. Keep it opt-in; do not give it a third
  chance.
* **Whether ghosts are needed** (§14). No - a rich enough purely atomic target is well
  posed on its own. The `--calibrate H` run reproduces the mechanism in seconds: the
  free-atom overlap fit stops at exactly 15 points in cc-pVDZ, its equation count, while
  the ghost fit reaches 124.
* **Whether a weight set can be transplanted onto another support** (§16). No, 0 times out
  of 12 - and §18 makes the question moot, since the preconditioner absorbs the footing
  entirely.
* **Support overlap, `rmsd_S`, and the rotation energy spread as quality metrics.** Three
  independent results say the overlap family carries no information about THC accuracy (the
  third is *anti*-correlated, Pearson -0.75), and §13 shows the spread is the wrong
  instrument for orientation - use `dE/dtheta`.
* **Whether one element's support serves unseen bonding** (§10). 1.49x in-range against
  1.50x out-of-range over ten molecules, including four bonds shorter than any the fit saw.
  T4 re-takes this on a fair footing; it does not reopen the question.
* **Adding partner elements to the ghost ensemble** (§10). Actively harmful at fixed
  environment count. Add environments, not variety.
* **Water, for any ratio.** At 24 AOs everything rank-saturates. It is a verification
  system, not a measurement.
* **Any torque quoted without its regulariser, or from a single rung.** §13's ladder trap
  and §16's `ghostw` (163 -> 1.00 -> 3.74 uHa/rad across three consecutive rungs) are both
  cautionary.

## 6. Code work the plan needs, in one place

| for | change | size |
| --- | --- | --- |
| T1 | `cage` mode in `torque_ladder.py` (`MODES`, `DEFAULT_THRESHOLDS`, `cage=True` to `element_supports`) | small |
| T2 | `--scheme` through `sweep.py`, `ghosts.py`, `transfer.py`, `atomic_eri.py`, `insitu.py`; `metric_scheme` keyword on `ghosts.thc_mp2` | small |
| T2 | `mf.conv_tol = 1e-12` in `sweep.py` and `ghosts.py` | trivial |
| T6 | `ThcMP2Gradients.as_scanner()` into `trajectory.py`'s `Surface`, for `--propagate full` | moderate |
| T8 | none - `trajectory.py` already takes `--reference rks --grid-level`, and carries several grids along one trajectory | none |
| T8b | `--parent-level` through `torque_ladder.py` (hardcoded `level=0`, line 137) and `trajectory.py` (line 128); `ghosts.element_grid` / `element_supports` / `assemble_from_supports` already take it | small |
| T8 (optional) | dipole ACF + FFT on a finished trajectory, only if the witness lands within a factor rather than beating level 3 | moderate |
| T13 | Z-vector contraction of `response_lagrangian`'s intermediate; least-squares `Z` fit in place of forming `S^-1` twice | substantial |

One bookkeeping item that touches every number in §17: `build_aux_coulomb_inv` did not
recognise the `_jacobi` suffix, so §17's table was produced with a filter it did not ask
for and **is not bit-reproducible against today's code**. Nothing in it is invalidated -
that metric is well conditioned and both sides agreed - but any table carried into a
viability claim should be regenerated.

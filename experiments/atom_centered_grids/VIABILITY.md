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

Three further things are simply absent:

* **No cost measurement of any kind.** There is no wall-clock or crossover study anywhere
  in `FINDINGS.md`. The `n_P` tax is a *proxy* for cost, and `pythc.grad` is explicitly a
  reference implementation that holds `(n_ao, n_ao, n_aux)` and `(n_P, n_P, n_occ)` in core
  and caps near alanine. Compression ratios are not a speedup.
* **No observable.** §13 measures 4.5 degrees of rotation-axis tilt per 2.5 ps and
  `HANDOFF.md` says of it, three times, that it is `a measurement without a threshold`.
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

*Report three things, all against controls already in §13's table at matched times:*

| quantity | reference already measured | pass |
| --- | --- | --- |
| `\|L - L0\|` at 1 ps | RKS/PBE level-0 Becke: 9.250 hbar on 4656 points | below its own parent quadrature, as §13's 1.188 already is |
| | RKS/PBE level-3 Becke: 0.028 hbar at 300 fs | within ~2.5x, which `blocked` already achieves on 224x fewer points |
| NVE drift per step | §15: 0.247 uHa/step (water, 12 steps, `ridge 1e-2`) | accumulated drift over the run below `kT` at 300 K |
| axis tilt over 2.5 ps | §13: 4.5 deg, from a 72x larger torque | scales down, i.e. well under 1 deg |

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

### Tier 3 - the requirement. Without this, T6 produces a number with nothing to compare it to.

**T8. Put an observable next to the leak.** `HANDOFF.md` raises this three separate times
and nothing computes one. The natural choice, because it is exactly what artificial
rotational diffusion would corrupt, is a **gas-phase rovibrational or IR spectrum from a
dipole autocorrelation function** on the acTHC surface, against the same trajectory on the
grid-free DF-RHF/relaxed-MP2 surface.

*Pass:* line positions and widths agree to within the spectral resolution of the run
length. *Why it decides the programme:* in condensed phase real collisional decorrelation
runs on ~1 ps and would bury 4.5 deg/2.5 ps entirely; gas-phase rotational structure would
not. If the spectrum is clean, the orientation question is closed against an observable
rather than against a tolerance nobody has set. *Code work:* a dipole ACF and FFT on top of
T6's trajectory - modest, and reusable.

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
3. **T9 reverses T1.** If the best support in cc-pVDZ is not the best in cc-pVTZ - which
   §17 already observes for the cage - then "fit once per element offline" has a basis
   dependence the acCD analogy does not, and every ratio needs re-taking per basis.
4. **T13 finds no crossover below the memory ceiling.** A scheme that compresses `n_P` but
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
| T8 | dipole ACF + FFT on a finished trajectory | moderate |
| T13 | Z-vector contraction of `response_lagrangian`'s intermediate; least-squares `Z` fit in place of forming `S^-1` twice | substantial |

One bookkeeping item that touches every number in §17: `build_aux_coulomb_inv` did not
recognise the `_jacobi` suffix, so §17's table was produced with a filter it did not ask
for and **is not bit-reproducible against today's code**. Nothing in it is invalidated -
that metric is well conditioned and both sides agreed - but any table carried into a
viability claim should be regenerated.

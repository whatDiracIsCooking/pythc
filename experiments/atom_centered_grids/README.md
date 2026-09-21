# Are per-atom THC grids affordable?

**Picking this up fresh? Start with [`HANDOFF.md`](HANDOFF.md)** - it records what was
tried, which parts of the prior design reasoning the measurements overturned, and the
ordered list of next steps.

The atom-centered proposal is to fit a THC grid once per element, offline, and translate it
rigidly into any molecule. That buys unconditionally smooth potential energy surfaces —
the fitted weights carry no geometry dependence at all, so `dw/dR = 0` and the only
nuclear-derivative terms left are the ordinary AO derivative and the rigid point-translation
term. What it costs is compactness, and that cost is what these scripts measure.

`NNLSGrid(blocked=True)` already fits each atomic sub-grid independently, so the penalty for
giving up molecular pruning can be measured today, with no new grid machinery. Because the
blocked fit sees each atom's **real** neighbours (not ghost approximations) and prunes
point-wise (no isotropy constraint), its inflation over the global fit is a **strict lower
bound** on what any frozen, transferable atom-centred scheme would pay. It can therefore
kill the idea cheaply, but it cannot confirm it.

`ghosts.py` supplies the confirmation: it builds the transferable object itself - one
point set per element, fitted offline against ghost neighbours, translated rigidly into a
molecule and never re-selected - and prices it against that lower bound.

## Scripts

| script | what it measures |
| --- | --- |
| `sweep.py` | points and LS-THC MP2 error vs NNLS threshold, blocked and global |
| `analyse.py` | interpolates point count at matched MP2 accuracy; prints the ratio |
| `rank.py` | co-density rank and LS-THC metric conditioning per grid |
| `rotate.py` | energy shift when each atom's point set is spun about its own nucleus |
| `orbits.py` | what selecting whole octahedral orbits costs in points and buys in orientation independence |
| `weights.py` | whether the fitted weights matter, or only the points they select |
| `ridge.py` | what ridge regularisation of the metric costs in accuracy, against truncation |
| `scan.py` | whether the metric truncation puts steps in the PES, and whether ridge removes them |
| `ghosts.py` | what an offline, per-element grid fitted against ghost neighbours costs against `blocked` |
| `ensemble.py` | whether the *choice* of ghost ensemble changes that cost, and the accuracy ceiling each ensemble imposes |
| `transfer.py` | whether one element's frozen point set serves bonding it was never fitted in - sp2/sp carbon, sp2 oxygen, nitrogen, and bonds shorter than any ghost |
| `gradient.py` | whether the analytic nuclear gradient agrees with the curve, where the ridge floor is for a derivative, and how much torque a frozen point set exerts |
| `window.py` | what sets that floor, whether a different filter moves it, and what the torque is once the regulariser is taken out of it |
| `torque_ladder.py` | whether the torque `window.py` is left holding converges away with grid size, the way the energy spread did |
| `trajectory.py` | what that torque does over a trajectory, and whether it is worse than the quadrature the grid is pruned from |
| `atomic_eri.py` | whether a **richer purely-atomic target** - the free atom's own ERIs rather than its overlap matrix - lifts the equation-count ceiling that forced ghosts, and what the support it selects is worth |
| `insitu.py` | whether fitting against *real* neighbours beats fitting against ghosts, and - by pairing a support with weights fitted for a different objective - whether a support and its weights can be chosen separately at all |
| `levers.py` | the four things that set where a ghost fit's support stops - AO basis, parent grid level, ghost direction set, cage or not - as a ceiling (no SCF) and as points per microhartree |

All use cc-pVDZ / cc-pVDZ-RI on a level-0 Becke parent grid, `ov` mode, 10 Laplace points,
against a DF-MP2 reference. Geometries come from RDKit ETKDG + MMFF.

```shell
uv run python sweep.py ethanol --out ethanol.json
uv run python analyse.py ethanol.json
uv run python rank.py ethanol 1e-3,1e-4,1e-5
uv run python rotate.py ethanol 1e-3            # add --orbits for a whole-orbit fit
uv run python orbits.py methanol --out orbits_methanol.json
uv run python weights.py ethanol 1e-4
uv run python ridge.py methanol 1e-3 --blocked
uv run python scan.py methanol 1e-3
uv run python ghosts.py --calibrate H,C,O          # threshold ladder, no SCF, seconds
uv run python ghosts.py methanol --out ghosts_methanol.json
uv run python ensemble.py --saturate H,C,O         # each ensemble's ceiling, no SCF
uv run python ensemble.py methanol --out ensemble_methanol.json
uv run python transfer.py --residual                # ghost shells + no-SCF screen
uv run python transfer.py --out transfer.json       # the ten-molecule suite
uv run python transfer.py --report transfer*.json
uv run python gradient.py water                     # analytic vs FD, ridge floor, torque
uv run python gradient.py water --mode ghost --threshold 3e-4
uv run python window.py methanol --mode ghost --threshold 3e-4   # the lambda window
uv run python window.py --report data/window_*.json              # the trade, per grid
uv run python scan.py methanol 1e-3 --damped-lambdas 1e-8,1e-10  # still smooth?
uv run python torque_ladder.py methanol --scheme damped --ridges 1e-8,1e-10 --draws 4 \
    --out torque_ladder_methanol_damped.json                     # the torque, vs grid size
uv run python torque_ladder.py water --scheme damped --parent    # with the complete-grid floor
uv run python torque_ladder.py --report data/torque_ladder_*.json --report-ridge pinv
uv run python torque_ladder.py methanol --modes parent,parent1 --scheme damped \
    --ridges 1e-8 --pinv --draws 2          # the complete-grid floor
uv run python trajectory.py methanol --modes hf,blocked,ghost --steps 5000 \
    --torque-every 4 --feedback ghost --out data/traj_methanol_tumbling.json
uv run python trajectory.py methanol --modes hf --reference rks --xc pbe \
    --grid-level 0 --steps 2000 --out data/traj_methanol_rks.json   # the DFT control
uv run python trajectory.py --report data/traj_*.json
uv run python atomic_eri.py --saturate H,C,N,O            # the ceiling, per target, no SCF
uv run python atomic_eri.py --calibrate H,C,O --with-ghost   # the ladder, no SCF
uv run python atomic_eri.py methanol --out data/eri_methanol.json
uv run python analyse.py data/eri_methanol.json
uv run python torque_ladder.py methanol --modes eri,eriw,ghost --scheme damped \
    --ridges 1e-8 --pinv --draws 4 --out data/torque_ladder_methanol_eri.json
uv run python insitu.py --calibrate                  # supports + weight sums, no SCF
uv run python insitu.py --out data/insitu.json       # train/held-out, energy and spread
uv run python torque_ladder.py formaldehyde --modes ghost,ghostw,molw,molfit,molfit1 \
    --ridges "" --pinv --draws 4                     # the mixed-footing control, as torque
uv run python levers.py --ceilings H,C,O --out data/levers_ceilings.json  # no SCF
uv run python levers.py methanol --variants ghost,parent1,cage,tetra \
    --out data/levers_methanol.json                                  # does a ceiling convert?
uv run python levers.py methanol --basis cc-pvtz --variants ghost,cage \
    --out data/levers_methanol_tz.json               # the basis lever, with its own floor
uv run python levers.py --report data/levers_*.json
uv run python torque_ladder.py methanol --modes blocked,blocked1,ghost --ridges 1e-8 \
    --pinv --draws 4 --scheme damped_jacobi   # the preconditioner, in dE/dtheta
uv run python trajectory.py methanol --modes ghost --scheme damped_jacobi   # unrun
```

## Results

See [`FINDINGS.md`](FINDINGS.md). Twenty-two headlines:

* The per-atom penalty is **1.2-1.7x** on point count, shrinking with system size - well
  inside the range where the scheme is worth building.
* The fitted weights turn out to be **almost irrelevant** to LS-THC accuracy; the NNLS
  fit's real product is its support. Where the metric is full-rank, all-ones weights give
  bit-identical energies. That removes most of the gradient difficulty from the proposal,
  since `X = phi(r_P)` has no weight to differentiate. *(This is the pseudoinverse result.
  §12 and §16 qualify it: under the ridge or damped filter the weights are worth 2-2.5x at
  matched support, and they cost the gradient nothing either way, so the offline object
  should carry them.)*
* With the grid frozen, the metric's **eigenvalue truncation is the last discrete step**
  left, and it puts measurable ~0.5 uHa jumps in the energy exactly where an eigenvalue
  crosses the cutoff. Ridge regularisation removes them for 0.1-2.2 uHa, and is now
  available as `metric_ridge` on `LS_RI_THC`.
* Fitting **whole octahedral orbits** (`NNLSGrid(group_orbits=True)`) makes each atomic
  grid *exactly* invariant under the 24 rotations of the octahedral group, for 1.2-1.6x
  the points - but it **never reduces the spread under general rotations**. At matched
  point count a point-wise grid is better on both accuracy and spread. The anisotropy of
  §4 converges away with grid size in either mode, so it is a symptom of a rank-limited
  grid rather than a defect needing a structural fix.
* **Ghost-fitted per-element grids work.** Fitted offline against ghost neighbours and
  then only translated, they cost **1.1-1.8x** the points of the in-molecule `blocked`
  fit at matched accuracy, shrinking with system size - so **1.6-2.7x** against a single
  global molecular fit. Fitting the *free* atom instead does not work and cannot be made
  to: an isolated atom's overlap matrix supplies only `n_AO (n_AO + 1) / 2` equations, so
  NNLS can never retain more than 15 points per hydrogen in cc-pVDZ, at any threshold.
* **Which ghost ensemble is used matters by ~1.35x on methanol and ~1.05x on ethanol**,
  so the sensitivity amortises with system size and §8's methanol ratios are upper
  bounds. What does not amortise is that each ensemble has a **rank ceiling**: the
  support saturates at the effective rank of its stacked target, which caps the accuracy
  its grids can ever reach at any threshold. The cheapest ensemble tested is the best on
  methanol and cannot reach 10 uHa on ethanol at all. Sum the per-element saturation
  sizes (`ensemble.py --saturate`, no SCF) against the expected point count before
  fitting.
* **The frozen grid transfers.** One point set per element, fitted once and handed
  unchanged to ten molecules spanning sp3/sp2/sp carbon, sp3/sp2 oxygen and nitrogen -
  including four heavy-atom bonds *shorter than the tightest ghost the fit ever saw* -
  costs **1.49x** on molecules inside the training range and **1.50x** on molecules
  outside it. Indistinguishable, and the whole suite lands in the 1.1-1.8x band already
  reported from methanol and ethanol. A partner element the fit never saw (N) costs
  nothing, and *adding* it to the ghost partner list makes every properly resolved
  molecule worse - environment count supplies rank, but variety at fixed environment
  count only dilutes. With this the programme has no unmeasured objection left.
* **The ridge floor was the filter's shape, not the metric's rank, and it is now gone.**
  `(S + lambda I)^-1` gives the numerically null directions the *largest* gain in the
  operator, `1/lambda`, and `Z = D^T D` carries `S^-1` twice - so gradient noise grows as
  `1/lambda^2` (measured: `lambda^-1.9`). The damped filter `sigma/(sigma^2 + mu^2)` is
  analytic in `S` like the ridge and suppresses those directions like the truncation; it
  drops the usable `lambda` by three to five decades and the accuracy it costs from
  **166-1808 uHa to under 5** across five grids, on a PES **smoother** than the ridge's.
  A `ridge_inv_eigh` control rules out the algorithm: it is the filter.
* **Most of §11's torque was the regulariser.** Continued to small `lambda` against a
  `pinv` control, water's 188 and 5592 uHa/rad become **0.001 and 0.002** - water is
  rank-saturated, so its torque was always zero, and §8 and §10 already refuse to quote
  it. On methanol, which does not saturate, the ghost-to-blocked ratio is **4.5x, not
  30x**, and the angular-momentum leak on the real transferable object is **3.1e-3
  bohr/rad against the 1.35e-1** §11 reported. A torque must be quoted with the
  regularisation that produced it.
* **The accuracy tax was never the problem.** The ridge's 280 uHa penalty on methanol
  varies by 2.28 uHa along a bond scan - a constant offset, which does nothing to a
  trajectory. Every scheme, and the bare frozen grid itself, biases the O-H force constant
  by under **1.7 cm^-1**. The case against the ridge is the gradient noise alone.
* **The torque converges away with grid size - in every mode.** The ladder that said
  otherwise was taken at `ridge 1e-4`, three decades outside where §12 says a torque means
  anything; `--scheme` did not exist when that data was produced. Re-run damped against a
  `--pinv` control, methanol goes `blocked` **357 -> 0.02** uHa/rad over 235 -> 670 points
  and `ghost` **969 -> 27** over 223 -> 702. Water is 0.00 on every rung of every mode and
  on its unpruned parent.
* **The orientation gap is the weight footing - not the transfer, and not the pruning.**
  `ghost` converges like `blocked1` (35.6x against 39.6x) and not like weighted `blocked`
  (17667x). The complete 3284-point parent grid sits at **0.24** uHa/rad and the *pruned*
  670-point weighted grid at **0.02**, so pruning is a benefit rather than a cost. What a
  transferable support cannot do is carry in-molecule weights - which made fitting
  per-element weights for that objective the highest-leverage thing left. §14 ran it, and
  §16 shows why the pairing rather than the objective is what carries it.
* **Over a trajectory the leak reorients rather than heats, and it is smaller than the
  grid it is pruned from.** 2.5 ps of thermal tumbling on methanol: the torque is 95%
  incoherent, the spin-up component is pinned by energy conservation at the size of the
  orientational potential itself, and what accumulates is **4.5 degrees of rotation-axis
  tilt**. Meanwhile, read at matched times, RKS/PBE on the same level-0 Becke grid loses
  **9.25 hbar in 1 ps** against the frozen support's **1.19** - lab-fixed atom-centred
  quadrature is what the whole family does, and the pruned reweighted support is eight
  times better at it than its own parent. A level-3 grid is ~25x better still, on 163x the
  points; the in-molecule `blocked` grid is within **2.3x of level-3 DFT on 224x fewer
  points**, which localises the remaining gap to the weights.
* **The orbital response is built, and AIMD runs on the THC surface.** It was the
  programme's last unbuilt component. The obstacle was not the CPHF solve but the Laplace
  factors: carrying orbital *energies*, they make the energy non-invariant under a
  rotation among the occupied orbitals, so the response needs blocks of `U` that no
  solver returns. Written `Theta_o = w^(1/4) exp(t F_oo)` the energy is the same at the
  canonical point and manifestly invariant, leaving only the standard occupied-virtual
  response. Section 11's rotational identity now closes to **6.1e-9** relative where the
  fixed-orbital force sits at 9.1e-3, and NVE through `pyscf.md` conserves energy to
  **0.247 uHa/step against 21.68** - the fixed-orbital number being exactly the "tens of
  microhartree per step" section 13 reported. What is left is making it cost one
  coupled-perturbed solve rather than `3 N`.
* **The ghosts turn out not to be necessary: fit the atom's own ERIs instead.** §4(3)
  killed the free-atom fit on an *equation count* - an isolated atom's overlap target
  supplies `n_AO(n_AO+1)/2` equations, 15 for hydrogen in cc-pVDZ, and NNLS can never
  retain more points than that. Ghosts lift the count by stacking environments. So does a
  richer observable of the same isolated atom: `(mu nu|lambda sigma) = int dr phi_mu
  phi_nu V_{lambda sigma}` makes the ERI *linear* in the quadrature weights, so the same
  solver fits it unchanged against `O(n_AO^4)` equations instead of `O(n_AO^2)`. The
  ceiling lifts **4.6-6.5x** (H 15 -> 97, C 92 -> 463, N 92 -> 467, O 92 -> 423) and the
  support - no ghosts, no partners, no training molecules - costs **0.89-1.16x** the
  in-molecule `blocked` grid at matched accuracy and matched weight footing, where
  thirteen ghost environments cost 1.11-1.78x.
* **And the weights that same fit produces are the first transferable ones that help.**
  §13 put the whole remaining orientation gap on the weight footing and found `ghostw`
  *worse* than `w = 1`. `eriw` converges on the `pinv` torque ladder at **1481x**, next to
  weighted `blocked`'s 17667x and against 13-40x for `blocked1`, `ghost`, `ghostw` and
  `eri` alike, reaching **0.19 uHa/rad on 565 points** - below the 0.24 the complete
  3284-point parent grid sits at - while landing within 0.05 uHa of that parent grid's
  energy. A transferable support *can* carry usable weights; they just have to be fitted
  against something LS-THC cares about.
* **Support and weights cannot be chosen separately.** §14 fits both against the atom's
  ERIs and wins; this is the control that says the *pairing* is why. Per-element weights
  fitted against real atoms in real molecules - trained on three, held out on three - are
  laid onto the ghost support they were not selected with (`molw`), and beat `ghostw`
  **0 times out of 12** on held-out molecules, on energy and rotation spread alike,
  despite carrying much the better overlap residual. Refit the support *and* the weights
  together (`molfit`) and it wins **8/10** on energy at a median of ~3x. So §14's
  instruction not to read an `eriw` number against a `w = 1` one is the weak form: a
  weight set is worth nothing at all on a support selected by a different fit. Two
  methodology notes come with it - read these at **matched point count**, and take no
  ratio from a torque quoted at one rung, since `ghostw` on formaldehyde runs 163 uHa/rad
  at 344 points, 1.0 at 427 and 3.7 at 486.
* **The orbital response is now a prerequisite, not a refinement.** On the fixed-orbital
  surface the force is not the gradient of the propagated energy, and that alone breaks
  rotational invariance at **~8300 uHa/rad** - fifty times the transferable grid's own
  torque, and identical on a grid with five thousand times less. No AIMD runs on the
  present gradient whatever the grid does.
* **The ghost cage was rejected on an argument, and the argument is wrong - the point
  tax is roughly halved.** `ghost_environments` declines to put ghosts in every direction
  at once because that would squeeze the central atom's Becke share to a lobe around the
  nucleus. It does - the caged atom keeps **0.2-0.3%** of its free-atom weight against
  57% with one neighbour - and the support ceiling goes **up** anyway, 2.2x on carbon,
  because `stack_targets` normalises the fragment and what is left sees every direction
  at once. On methanol the ghost gap goes **1.78x -> 1.22x** at 10 uHa and
  **1.60x -> 1.13x** at 5 uHa, and at 2 uHa the transferable grid is *smaller* than the
  in-molecule fit (0.96x) while `ghost` cannot reach that target at all. Compounded
  against a global molecular fit that is **1.85x rather than 2.7x**, i.e. 3.4x rather
  than 7.3x on the `n_P^2` parts. **All of that is cc-pVDZ, and cc-pVTZ reverses it** -
  there the cage is 1.2-1.4x *worse* than the single-ghost ensemble. Treat the halved tax
  as a best case, not a result.
* **The bigger cc-pVTZ finding is not about the cage: discarding the NNLS weights costs
  ~19 uHa there against ~1.5 in cc-pVDZ.** A `blocked` ladder that looked like it had
  hit an accuracy ceiling at ~19 uHa above the floor - and that `ghost` appeared to beat,
  which would have broken the strict-lower-bound premise of sections 1 and 8 - turns out
  to sit exactly **on** the floor once its own weights are restored (+0.36 to +0.00 uHa
  from 1053 points up). The filter is worth a factor of three, the weights are worth all
  of it, and a transferable support can carry neither. **The weight penalty grows with
  basis size**, so the ghost gap is worse in a production basis, not better, and fitting
  per-element weights for the in-molecule objective is now a prerequisite rather than an
  improvement. Two more levers behave as §9 predicts: a finer parent
  grid moves the ceiling 1.26-1.42x *at unchanged equation count* - which contradicts the
  premise HANDOFF 4(9) is written on - and a tetrahedral direction set is the cheapest
  thing measured at 50 uHa and cannot reach 10 uHa at any threshold, which is §9's rank
  ceiling on a new axis. **No torque has been measured for any of them.**
* **The weight footing cancels out of a preconditioned metric - exactly - and with it
  the last structural objection to a transferable support.** §17's runtime scaling
  `E = diag(1/sqrt(diag S))` recovers in the energy what discarding the NNLS weights
  costs, and had no adjoint, so it could not be asked what it does to a gradient. It has
  one now, verified against a finite difference, against the identity that an exact
  inverse cannot see a similarity transform, and end to end on an assembled gradient and
  torque. On methanol at matched supports and matched draws, `ghost` - the transferable
  object, at `w = 1` - goes **15.15 -> 0.21 uHa/rad** at 702 points and **21.37 -> 0.09**
  at 627, below the 0.24 the complete 3284-point parent grid sits at, while its energy
  lands on the same +2.75 uHa floor the in-molecule fit reaches. The reason is algebra
  rather than luck: `diag(S(w))_PP = w_P diag(S(1))_PP`, so `E(w) S(w) E(w) = E(1) S(1)
  E(1)` for any positive weights, and `blocked` and `blocked1` - one support at two
  footings - agree to **ten digits** under it. §13 put the whole remaining transferable
  gap on the weight footing and §16 showed it could not be repaired by transplanting a
  weight set; both are about a quantity the preconditioner removes. One molecule, one
  basis, one seed, and no trajectory has been run on it.
* **The gradient exists and verifies to machine precision** (`pythc.grad`, driven by
  `gradient.py`): quadratic convergence against a finite difference, and forces that sum
  to zero to 2.4e-15 with no finite difference involved. Two things came back with it
  that no energy measurement could have produced. First, **§6's `lambda = 1e-8` is three
  decades too small for a gradient** - at that value the energy is reproducible to
  0.002 uHa while the gradient varies by a *factor of two* between runs of the identical
  calculation; the usable window is `1e-2` to `1e-5`. Second, **the rotation spread does
  not track the torque**: `dE/dtheta` is now exactly computable, and water's transferable
  ghost grid has 1.9x the energy spread of its blocked grid but **30x** the net torque
  (5592 against 188 uHa/rad). §7's deflation of orientation dependence rests on spreads
  and does not survive being differentiated.
Raw sweep output is under [`data/`](data).

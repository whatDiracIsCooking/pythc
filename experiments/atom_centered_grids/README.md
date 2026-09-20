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
```

## Results

See [`FINDINGS.md`](FINDINGS.md). Fourteen headlines:

* The per-atom penalty is **1.2-1.7x** on point count, shrinking with system size - well
  inside the range where the scheme is worth building.
* The fitted weights turn out to be **almost irrelevant** to LS-THC accuracy; the NNLS
  fit's real product is its support. Where the metric is full-rank, all-ones weights give
  bit-identical energies. That removes most of the gradient difficulty from the proposal,
  since `X = phi(r_P)` has no weight to differentiate.
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
  transferable support cannot do is carry in-molecule weights - which makes fitting
  per-element weights for that objective the highest-leverage thing left.
* **Over a trajectory the leak reorients rather than heats, and it is smaller than the
  grid it is pruned from.** 2.5 ps of thermal tumbling on methanol: the torque is 95%
  incoherent, the spin-up component is pinned by energy conservation at the size of the
  orientational potential itself, and what accumulates is **4.5 degrees of rotation-axis
  tilt**. Meanwhile RKS/PBE on the same level-0 Becke grid loses **9.25 hbar in 1 ps**
  against the frozen support's **1.49 in 2.5 ps** - lab-fixed atom-centred quadrature is
  what the whole family does, and the pruned reweighted support is six times better at it
  than its own parent. A level-3 grid is 2-3 decades better still, on 163x the points.
* **The orbital response is now a prerequisite, not a refinement.** On the fixed-orbital
  surface the force is not the gradient of the propagated energy, and that alone breaks
  rotational invariance at **~8300 uHa/rad** - fifty times the transferable grid's own
  torque, and identical on a grid with five thousand times less. No AIMD runs on the
  present gradient whatever the grid does.
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

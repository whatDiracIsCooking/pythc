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
```

## Results

See [`FINDINGS.md`](FINDINGS.md). Six headlines:

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

Raw sweep output is under [`data/`](data).

# Handoff: atom-centered THC grids

Read this first, then [`FINDINGS.md`](FINDINGS.md) for the measurements. This file records
what the idea is, what was tried, **which parts of the prior design reasoning the
measurements overturned**, and what to do next. The point of §3 is that a fresh session
should not re-derive machinery that turned out to be unnecessary.

## 1. The idea

Fit a THC grid once per element, offline, and translate it rigidly into any molecule - the
acCD move (atomic Cholesky decomposition: do the pivoted selection once per element
offline, emit ordinary atom-centred basis functions, never re-select at runtime), applied
to grid points instead of auxiliary basis functions.

The prize is smooth potential energy surfaces. THC point selection is discrete
(QRCP/pivoted-Cholesky/NNLS), so it is not differentiable; the standard workaround is to
freeze the selection at the reference geometry and differentiate the rest. A grid frozen
*per element* needs no such assumption at all - nothing is re-selected as nuclei move.

The price is compactness: per-atom grids cannot share bonding-region points between
neighbours, must cover bond directions the atom may not have, and may need coarser
orbit-wise pruning to stay isotropic. Since `Z` is `n_P x n_P`, the point-count penalty is
what decides the programme.

## 2. What was attempted

Four experiments, all in this directory, all on cc-pVDZ / cc-pVDZ-RI, level-0 Becke parent
grid, `ov` mode, 10 Laplace points, against DF-MP2. Geometries from RDKit ETKDG + MMFF.

| script | question | headline |
| --- | --- | --- |
| `sweep.py` + `analyse.py` | how many more points does per-atom fitting need? | **1.2-1.7x**, improving with size |
| `rank.py` | why - and what does it cost in conditioning? | global is a perfect rank-revealing selector; blocked metric min-eig 1e-20 vs 1e-8 |
| `rotate.py` | does an arbitrarily-oriented frozen grid cost energy? | 0 uHa when rank-saturated, **27-39 uHa** when rank-limited; **no Lebedev shell survives intact** |
| `weights.py` | do the fitted weights matter, or only the selection? | **only the selection**; all-ones weights are bit-identical or better |

The key methodological move: `NNLSGrid(blocked=True)` already fits each atomic sub-grid
independently, so the penalty for giving up molecular pruning is measurable today. Because
the blocked fit sees each atom's **real** neighbours and prunes **point-wise**, its
inflation is a strict **lower bound** on a frozen atom-centred scheme. It could have killed
the idea cheaply; it did not.

## 3. Prior design reasoning: what survived, what did not

This programme began as a long design conversation before anything was measured. Its
conclusions are listed here with their current status, so they are not re-derived.

| claim from the design discussion | status |
| --- | --- |
| Freeze the discrete point selection, differentiate the smooth remainder (AO derivative + rigid point translation) | **Stands.** This is what Song & Martínez actually do. |
| LS-THC needs no Becke partition function, since `Z` is fitted rather than quadratured | **Confirmed, and then some** - see the weights result below. |
| NNLS weights need active-set freezing, QP sensitivity theory, strict complementarity | **Superseded.** The weights barely affect the energy at all; there is nothing to freeze. |
| Reparametrise `w = theta^2` to remove the inequality constraint | **Moot**, and it was the wrong power: this codebase builds `X = w^(1/4) phi`, so `theta^2` still leaves `X = |theta|^(1/2) phi`, singular at 0. If a weight fit is ever kept, parametrise the *collocation amplitude* `X = t phi` with `w = t^4`; the whole pipeline is then polynomial in `t`. |
| Add `sum_P w_P = 1`; equality constraints differentiate cleanly | **Moot** for the same reason. |
| Isolated-atom NNLS will discard exactly the tail points a bond needs; fix with ghost atoms | **Still the central untested idea.** Highest-value next experiment. |
| Octahedral ghosts suffice because p orbitals are octahedral | **Rejected in the discussion itself, correctly** - points are sampling locations, not functions, and do not superpose. |
| Frozen per-atom grids risk orientation-dependent energies | **Confirmed by measurement**: no angular shell survives intact (8-20% of each kept), and rank-limited grids shift 27-39 uHa under random per-atom rotation. |
| Fix isotropy with orbit-wise (group-sparsity) pruning over (radial shell, Lebedev orbit) blocks | **Endorsed, unimplemented.** The parent grid is already orbit-structured via `treutler_prune`, so the structure exists to exploit. |
| Grid inflation will be 2-3x, i.e. 4-9x on the `n_P^2` parts | **Too pessimistic.** Measured 1.2-1.7x, i.e. ~1.5-2.9x. |
| Unioned per-atom grids will wreck metric conditioning; use ridge, not eigenvalue truncation | **Confirmed** (min eigenvalue 1e-20 vs 1e-8). Ridge is now mandatory, not optional. |
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

**(1) Ridge instead of truncation.** Smallest, most independent, plausibly upstreamable -
it is arguably a defect in the current pseudoinverse path regardless of whether
atom-centred grids happen. Replace the truncated pseudoinverse with `(S + lambda I)^-1` at
fixed `lambda`, measure the accuracy cost, and check it removes the eigenvalue-crossing
discontinuity. Touches `lib.pseudo_inv_sqrt` / the metric inversion in
`thc/ls_thc_funcs.py`.

**(2) Group-NNLS over (radial shell, angular orbit) blocks.** Do this *before* (3), so the
ghost experiment measures the object you would actually ship rather than a point-wise-pruned
stand-in. Needs: keep the shell/orbit structure from `gen_atomic_grids` (currently flattened
in `grid.py`), and a group-sparsity variant of the Lawson-Hanson loop in `decomp/nnls.py`.
Re-run `sweep.py` and `rotate.py` afterwards - expect a larger grid and a smaller rotation
spread, and quantify that trade.

**(3) The ghost gap - the only thing that can still kill the idea.** Build ghost-augmented
per-element point sets for H/C/N/O and measure how much worse than `blocked` they are.
Simpler than originally conceived: since weights do not matter, the offline object is just a
**point set** per element. Stack several ghost radii (and ideally several partner elements)
into one fit rather than agonising over octahedron vs icosahedron.

**(4) Transferability.** Fit the same element in several environments (O in water, methanol,
formaldehyde) and compare supports. Decides whether one ghost geometry suffices or an
environment ensemble is needed.

**(5) Actually compute a gradient.** Nothing here measured one; the smoothness argument is
structural. A finite-difference-vs-analytic check on a frozen grid, or simply scanning a
bond length and looking for kinks, would be worth more than another accuracy sweep.

## 5. Open questions

* Does the 1.2-1.7x ratio hold in larger basis sets (cc-pVTZ) and for HF exchange rather
  than MP2 correlation? Everything here is cc-pVDZ / `ov` / MP2.
* The ratio improves from water to alanine. Does it keep improving, or turn around?
* If weights are irrelevant, is NNLS still the best *selector*? Pivoted Cholesky and QRCP
  select points directly and are already implemented (`ls_ri_cholesky.py`, `ls_ri_qrcp.py`).
  A like-for-like selector comparison at matched point count has not been done here.
* Does `rmsd_S` ever track accuracy, or should grid comparisons move to rank + min
  eigenvalue wholesale?

## 6. Literature

Verified by search during this session:

* **Kokkila Schumacher, Hohenstein, Parrish, Wang, Martínez**, *THC-MP2: Grid Optimization
  and Reaction Energies*, JCTC **11**, 3042-3052 (2015), DOI 10.1021/acs.jctc.5b00272.
  Generates grids for first-row atoms at **<100 points/atom** with negligible energy error,
  cc-pVDZ and cc-pVTZ. **This is prior art for per-atom, per-basis THC grids** - read the SI
  first. For calibration, the per-atom densities measured here bracket that figure (water
  global ~41/atom, alanine blocked ~100/atom).
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
```

Runtimes on 4 cores: water seconds, methanol ~3 min, ethanol ~10 min, alanine ~40 min
(dominated by the unpruned-parent-grid baseline and the tight global fits). Raw output from
the runs behind `FINDINGS.md` is in [`data/`](data).

# Are per-atom THC grids affordable?

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

## Scripts

| script | what it measures |
| --- | --- |
| `sweep.py` | points and LS-THC MP2 error vs NNLS threshold, blocked and global |
| `analyse.py` | interpolates point count at matched MP2 accuracy; prints the ratio |
| `rank.py` | co-density rank and LS-THC metric conditioning per grid |
| `rotate.py` | energy shift when each atom's point set is spun about its own nucleus |
| `weights.py` | whether the fitted weights matter, or only the points they select |

All use cc-pVDZ / cc-pVDZ-RI on a level-0 Becke parent grid, `ov` mode, 10 Laplace points,
against a DF-MP2 reference. Geometries come from RDKit ETKDG + MMFF.

```shell
uv run python sweep.py ethanol --out ethanol.json
uv run python analyse.py ethanol.json
uv run python rank.py ethanol 1e-3,1e-4,1e-5
uv run python rotate.py ethanol 1e-3
uv run python weights.py ethanol 1e-4
```

## Results

See [`FINDINGS.md`](FINDINGS.md). Two headlines:

* The per-atom penalty is **1.2-1.7x** on point count, shrinking with system size - well
  inside the range where the scheme is worth building.
* The fitted weights turn out to be **almost irrelevant** to LS-THC accuracy; the NNLS
  fit's real product is its support. Where the metric is full-rank, all-ones weights give
  bit-identical energies. That removes most of the gradient difficulty from the proposal,
  since `X = phi(r_P)` has no weight to differentiate.

Raw sweep output is under [`data/`](data).

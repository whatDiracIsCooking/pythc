# PyTHC

![logo](pythc_logo.svg)



Python-based implementations of Tensor Hypercontraction (THC).
Preprint paper available at: https://arxiv.org/abs/2608.17885.
The full code can be found at https://github.com/QuantumCorrelators/pythc.

## Description

This project implements Tensor Hypercontraction for use with Electron Repulsion Integrals (ERIs). It is built in pure Python on top of:

* [PySCF](https://github.com/pyscf/pyscf)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Cotengra](https://cotengra.readthedocs.io/en/latest/)

## Installation

The project is not yet published to PyPI, so you will need to install it directly from Git.

### Installing into a virtual environment

On Debian-based distributions, ensure that the following packages are installed and up to date: `python3`, `python3-pip`, `python3-venv`, `gcc`, `gfortran`, `libopenblas-dev`. (Tested on Ubuntu 24.04/26.04).

To create a new virtual environment, run:

```shell
python3 -m venv .venv
```

Activate the environment with:

```shell
source .venv/bin/activate
```

Now install the project using:

```shell
pip install pythc
```

### Installing with Conda / Mamba

To install the library in a new environment using Conda or Mamba:

```shell
mamba create -n my_env python=3.12
mamba activate my_env
pip install pythc 
```

### Installing into an existing uv project 

This is the easiest way to install the project. You can create a new uv project using `uv init my-project-name`.

If you already have a project set up using the [uv](https://docs.astral.sh/uv/) package manager, running:

```shell
uv add pythc 
```

will add the package to your `pyproject.toml`.


## Developer Instructions

If you want to contribute to `PyTHC`, you can clone the repository and install the dependencies either via mamba
```shell
mamba create -n pythc python=3.12
mamba activate pythc
git clone git@github.com:QuantumCorrelators/pythc.git
cd pythc
pip install . 
```

or via uv
```shell
git clone git@github.com:QuantumCorrelators/pythc.git
cd pythc
uv sync
```
given uv is installed on your system.


## Basic Usage

Detailed examples can be found in [`src/pythc/examples`](src/pythc/examples).

All THC calculations require a PySCF `Molecule` object as well as a completed mean-field calculation:

```python
from pyscf import gto, scf

mol = gto.M('path/to/your/mol', basis='cc-pvdz')
mf = scf.RHF(mol)
mf.kernel()
```

From those objects, we can build the THC representation by importing the appropriate THC class and grid builder:

```python
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.grid import BeckeGrid

grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, grid=grid_builder, auxbasis='cc-pvdz-ri')
thc_eri = thc_builder.build(mode='ao')
```
Supported modes are `ao` for the AO-ERI $(\mu \nu|\lambda \sigma)$ and `ov` for the occupied-virtual block if the MO-ERI $(i j|a b)$.
The THC representation **must** be built in `mode='ao'` for the HF-SCF calculation.

### Grid reweighting and pruning

The accuracy and cost of LS-THC are largely set by the quadrature grid. `NNLSGrid` refits the grid weights
so that numerical integration on the grid reproduces the AO overlap matrix, subject to $w_P \ge 0$
([Hillers-Bendtsen, Lu, Martínez 2026](https://doi.org/10.1021/acs.jctc.6c00664)). The non-negativity
constraint leaves most weights at exactly zero, so the fit prunes the grid as a side effect, while the
surviving weights remain a valid quadrature rule.

It is an ordinary grid builder, so it drops into any THC class in place of `BeckeGrid`:

```python
from pythc.grid import BeckeGrid, NNLSGrid

grid_builder = NNLSGrid(mol, parent=BeckeGrid(mol), weight_threshold=1e-4)
thc_builder = LS_RI_Becke(mol=mol, grid=grid_builder, auxbasis='cc-pvdz-ri')
```

`weight_threshold` trades accuracy against compactness: looser values terminate the fit earlier and
retain fewer points. For water in cc-pVDZ, the default of `1e-4` keeps 124 of the 1648 points of the
level 0 Becke grid, and `1e-6` keeps 186 while reproducing the overlap matrix *better* than the full
input grid — something a pure point-selection scheme such as the pivoted Cholesky pruning in
`LS_RI_Cholesky` cannot do, since it keeps the tabulated weights.

`LS_RI_NNLS` is the same thing packaged as a THC class, taking the *input* grid as its `grid` argument:

```python
from pythc.thc.ls_ri_nnls import LS_RI_NNLS

thc_builder = LS_RI_NNLS(mol=mol, auxbasis='cc-pvdz-ri', weight_threshold=1e-4)
```

The cost of the fit grows steeply with the number of points it retains: water in cc-pVDZ takes about a
second, alanine in cc-pVDZ (119 basis functions) takes ~20 s at a loose threshold and considerably
longer at a tight one. Past roughly a hundred basis functions, pass `blocked=True` to fit each atomic
sub-grid separately, which keeps the cost linear in the number of atoms (alanine at `1e-4`: 1289 points
in 28 s, giving an MP2 correlation energy within 9 µHa of the RI reference). Blocking is an extension
beyond the reference and is approximate — verify it against the global fit for your system.

Once the THC ERI is built, we can retrieve the $X_\mu^P$ and $Z^{P Q}$ matrices using:

```python
X, Z = thc_eri.get_X_Z()
```

or construct the full 4-dimensional quantity:

$$
(\mu \nu | \lambda \sigma) = \sum_{P Q} X_\mu^P X_\nu^P Z^{P Q} X_\lambda^Q X_\sigma^Q
$$

```python
full_eri = thc_eri.get_full()
```

The matrices can also be saved to HDF5 format:

```python
thc_eri.save('path/to/place/eri.hdf5')
```

Alternatively, you can load an existing ERI from disk:

```python
from pythc.thc.thc_base import ThcEri

thc_eri = ThcEri.from_file("path/to/place/eri.hdf5")
```

With this THC ERI interface, you can implement improved quantum chemistry algorithms. This project currently implements:

1. The HF-SCF algorithm 
2. Møller-Plesset Perturbation Theory of 2nd order
3. Random Phase Approximation (RPA)

The HF-SCF implementations inherit from PySCF's `RHF`/`UHF` classes. We build in `ao` mode so the THC does not need to be rebuilt between SCF iterations:

```python
from pythc.grid import BeckeGrid
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.hf import THC_RHF

auxbasis = 'cc-pvdz-ri'
grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, grid=grid_builder, auxbasis=auxbasis)
thc_eri = thc_builder.build(mode='ao')

mf = THC_RHF(mol,
             thc_eri,
             auxbasis,
             verbose=4,
             thc_threshold=0.1,
             min_exact_cycles=1)
mf.kernel()
```

The usage of THC during SCF can be configured using `thc_threshold` and `min_exact_cycles`. The `thc_threshold` parameter specifies the energy difference threshold (in Hartree) below which the solver switches from exact RI/DF cycles to THC, provided at least `min_exact_cycles` iterations have passed. If the parameter `thc_only=True` is provided, the SCF will be converged exclusively using the THC-ERI ([details](src/pythc/examples/scf_with_ls_snri_cholesky.py)).

To calculate a Laplace-transformed MP2 energy contribution with 10 integration points, build the THC in `ov` mode:

```python
from pythc.grid import BeckeGrid
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.mp2 import LaplaceRMP2

grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, mo_coeff=mf.mo_coeff, grid=grid_builder,
                          auxbasis='cc-pvdz-ri')
thc_eri = thc_builder.build(mode='ov')

mp2_e_thc = LaplaceRMP2(mol, mf, thc_eri, n_laplace=10).kernel()
```

The THC representation **must** be built in `mode='ov'` for the MP2 calculation.


### Analytic nuclear gradients and AIMD on a frozen grid

`pythc.grad` differentiates the Laplace THC-MP2 energy analytically with respect to the
nuclei. It requires a **frozen** grid — one point set per atom, stored relative to that
atom's nucleus and attached by rigid translation — because a grid re-selected at every
geometry is a discrete function of geometry and has no derivative:

```python
from pythc.grad import FrozenGrid, thc_mp2_gradient

grid = FrozenGrid(mol, per_atom)     # per_atom: [(coords_rel_to_nucleus, weights), ...]
res = thc_mp2_gradient(mol, mf, grid, auxbasis='cc-pvdz-ri', metric_ridge=1e-4)
res.de          # (n_atm, 3) fixed-orbital correlation gradient, Hartree/Bohr
res.torque      # (n_atm, 3) torque on each atom's point set about its own nucleus
```

`thc_mp2_gradient` returns the correlation gradient at a **fixed** SCF reference. For the
derivative of a total energy — which is what dynamics needs — add the Hartree-Fock
gradient and the orbital response:

```python
from pythc.grad.total import total_mp2_gradient

e_tot, de = total_mp2_gradient(mol, mf, grid, auxbasis='cc-pvdz-ri', metric_ridge=1e-4)
```

The whole pipeline is a PySCF gradient scanner, so `pyscf.md` drives it directly. The
per-element point sets stay fixed for the entire trajectory; the only thing that happens
to the grid between steps is that it is translated onto the new nuclear positions:

```python
from pyscf import md
from pythc.grad.total import ThcMP2Gradients

grad = ThcMP2Gradients(per_atom=per_atom, auxbasis='cc-pvdz-ri', metric_ridge=1e-4)
md.NVE(grad.as_scanner(mol), dt=20, steps=100).run()
```

Two caveats worth knowing before using this. The orbital response currently solves the
coupled-perturbed equations once per nuclear degree of freedom, so a gradient costs
Hessian-level work rather than the single Z-vector solve a production implementation
would use. And the regularisation window for a *gradient* is much narrower than for an
energy — prefer `metric_scheme='damped'`, and see
[`experiments/atom_centered_grids/FINDINGS.md`](experiments/atom_centered_grids/FINDINGS.md)
sections 11–12 for why a ridge that the energy is happy at can be three decades too small
for its derivative.

On a frozen grid, prefer `metric_scheme='damped_jacobi'`. The suffix preconditions the
metric by `E = diag(1/sqrt(diag S))` around the filter, which costs nothing and is
differentiated exactly. Because the collocation weights enter the metric only as
`S(w) = D S(1) D`, that scaling absorbs them: the preconditioned inverse is invariant to
the weights, so a grid carrying none is not penalised for it. Measured on methanol, it
drops a transferable support's net torque from 15.2 to 0.2 µHa/rad at 702 points and puts
its energy on the same floor an in-molecule fit reaches — sections 17–18.

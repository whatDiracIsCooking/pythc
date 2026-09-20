"""
Analytic nuclear derivatives of the frozen-grid LS-THC MP2 correlation energy.

This is section 5 of ``experiments/atom_centered_grids/HANDOFF.md``. The proposal there
is that a THC grid frozen *per element* and translated rigidly onto nuclei leaves nothing
discrete at runtime, so the potential energy surface is differentiable and the gradient
has no re-selection term to omit. Everything in this package exists to compute that
gradient and to check the claim rather than argue it.

The energy being differentiated is exactly the one ``pythc`` already computes:

* a point set frozen at fit time, attached to nuclei by rigid translation;
* ``X = w^(1/4) phi(r_P)`` evaluated at those points and projected into the MO basis;
* ``S = (X_o X_o^T) o (X_v X_v^T)`` inverted by ridge (or, for comparison, by the
  truncated pseudoinverse);
* ``Z = D^T D`` with ``D = J^(-1/2) W S^(-1)`` fitted against DF ERIs;
* the Laplace-transformed MP2 correlation energy ``-2 J + K`` built from ``X`` and ``Z``.

The implementation is a hand-written reverse-mode pass over that pipeline. Each stage
exposes its adjoint separately so it can be finite-difference tested on its own, which is
how every formula here was checked - see ``tests/test_thc_gradient.py``.

**What is and is not included.** :func:`~pythc.grad.driver.thc_mp2_gradient` returns the
derivative of the *correlation* energy at a fixed SCF reference: the MO coefficients and
orbital energies are held at their reference-geometry values. That is the THC-specific
content of the gradient - every term the frozen grid, the collocation, the metric and the
ridge touch - and it is what had not been written down before.

The orbital-response term (``dC/dR``, ``de/dR``) is **not implemented**, so this is not
the total MP2 gradient and should not be compared against one. That term is structurally
identical to DF-MP2's and needs no THC-specific theory: solve the coupled-perturbed
equations once and contract the result with the ``C_bar`` and ``eps_bar`` that
:class:`~pythc.grad.driver.ThcGradientResult` already returns for the purpose. Keeping
the split explicit is deliberate rather than a shortcut - the fixed-orbital derivative is
exactly finite-difference checkable on its own, so the novel machinery could be verified
to machine precision without the response layer's approximations in the way.

**The orientation derivative needs no response at all.** Spinning an atom's frozen point
set about its own nucleus leaves the molecule, the AOs and the SCF solution untouched -
only the grid moves. So ``dE/dtheta`` is exactly a fixed-orbital derivative, and
:func:`~pythc.grad.driver.orientation_gradient` computes it without approximation. That
matters because HANDOFF section 4(2) measured orientation dependence as an energy spread
over random draws, and a spread does not bound a derivative: a small-amplitude but
rapidly varying function of orientation has a tiny spread and a large torque. The torque
is the quantity that decides whether a frozen atomic grid conserves angular momentum in
dynamics, and it is what this package makes measurable.
"""

__all__ = [
    "FrozenGrid",
    "ThcGradientResult",
    "orientation_gradient",
    "thc_mp2_gradient",
]


def __getattr__(name):
    # Deferred so that importing a single stage (for a unit test, say) does not drag in
    # the whole pipeline, and so that pythc.grad stays importable without pyscf's
    # integral machinery being touched.
    if name == "FrozenGrid":
        from pythc.grad.frozen import FrozenGrid

        return FrozenGrid
    if name in ("ThcGradientResult", "orientation_gradient", "thc_mp2_gradient"):
        import pythc.grad.driver as driver

        return getattr(driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

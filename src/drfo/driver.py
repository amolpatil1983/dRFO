"""Top-level public API: `find_ts()` composes the Schlegel-paper-specific
preparation (`hessian.schlegel_prep.build_schlegel_ts_preparation` --
bond-order interpolation guess, constrained pre-relaxation, Delta-b-guided
initial Hessian) with the standalone dRFO search core
(`optimize.search.run_drfo_search`) into a single convenience call.

Both halves are independently usable: a caller with their own TS guess (from
docking, ML, or manual construction) can call `run_drfo_search` directly and
skip this module's guess-building machinery entirely; a caller who wants the
paper's guess-building but a different search loop can call
`build_schlegel_ts_preparation` directly.
"""
from __future__ import annotations

from .calculators.base import Calculator, CalculatorError, SCFConvergenceError
from .geometry import Geometry
from .hessian.schlegel_prep import build_schlegel_ts_preparation
from .optimize.search import Status, TSResult, run_drfo_search
from .optimize.trust import ConvergenceCriteria

__all__ = ["find_ts", "TSResult", "Status"]


def find_ts(
    reactant: Geometry, product: Geometry, *,
    calculator: Calculator,
    cheap_calculator: Calculator | None = None,
    bond_scale: float = 1.4,
    overlap_thresh: float = 0.5,
    max_relax_steps: int = 100,
    max_ts_steps: int = 100,
    convergence: ConvergenceCriteria | None = None,
    keep_trajectory: bool = False,
    imaginary_mode_tol: float = 1e-4,
    hessian_refresh_interval: int | None = None,
    breaking_bonds: list[tuple[int, int]] | None = None,
    forming_bonds: list[tuple[int, int]] | None = None,
) -> TSResult:
    """Locate the transition state connecting `reactant` and `product`
    using the Connectivity Transition State (CTS) method: bond-order
    interpolation guess -> constrained pre-relaxation -> Delta-b-guided
    initial Hessian (`hessian.schlegel_prep.build_schlegel_ts_preparation`)
    -> internal-coordinate dRFO search to convergence
    (`optimize.search.run_drfo_search`).

    Any calculator failure during preparation (guess-building or
    pre-relaxation) is caught here and reported as an early `TSResult`
    with the appropriate `status` rather than raised, matching how a
    failure during the search itself is reported -- callers get a
    consistent `TSResult`-shaped answer regardless of which phase failed.

    `hessian_refresh_interval` (None = off, matching the paper's default
    use of pure Hessian updating): every N accepted TS-search steps,
    replace the Bofill-updated Hessian with a freshly computed exact one
    (a numerical Hessian from the real calculator -- for GFN2-xTB this is
    cheap, ~2x a gradient call, not the ~2N+1 separate evaluations a naive
    finite-difference count would suggest, since xtb's own numerical
    Hessian routine is well optimized internally) instead of continuing to
    update it from secant pairs.

    Tested on 1,3-pentadiene (13 atoms/39 DOF) expecting this to help --
    it did not. Neither periodic refresh (every 50 steps) nor refreshing
    every single step changed the convergence rate; refreshing every step
    was strictly worse (same per-step progress, ~2x wall-clock cost).
    Root cause: the internal-coordinate Hessian transform (B+^T H_cart B+)
    amplifies a couple of B's own near-singular (but not redundant) singular
    values by ~200-260x regardless of whether H_cart is exact or
    Bofill-approximate, so a fresher H_cart doesn't avoid the pathology --
    it just gets amplified the same way every time instead of persisting
    unchanged from step 0. The actual fix would be a more robust internal-
    coordinate generation scheme (e.g. natural/delocalized internal
    coordinates) for floppy molecules, not fresher curvature information.
    Kept as a toggle since other systems where staleness genuinely is the
    bottleneck could still benefit; see optimize/rfo_core.py's
    floor_eigenvalues for the eigenvalue-cap mitigation that IS effective.

    `breaking_bonds`/`forming_bonds`: optional externally-supplied bond
    classification passed straight through to
    `hessian.schlegel_prep.build_schlegel_ts_preparation`, bypassing its
    default covalent-radius diffing -- see that function's docstring.
    """
    try:
        prep = build_schlegel_ts_preparation(
            reactant, product, calculator=calculator, cheap_calculator=cheap_calculator,
            bond_scale=bond_scale, max_relax_steps=max_relax_steps, convergence=convergence,
            keep_trajectory=keep_trajectory,
            breaking_bonds=breaking_bonds, forming_bonds=forming_bonds,
        )
    except CalculatorError as exc:
        # Already logged inside build_schlegel_ts_preparation -- just
        # classify it into the matching TSResult.status here.
        status: Status = "scf_convergence_failure" if isinstance(exc, SCFConvergenceError) else "calculator_failure"
        return TSResult(
            geometry=reactant, energy=float("nan"), gradient_rms=float("nan"),
            hessian=None, n_imaginary=None, converged=False, status=status,
            n_steps=0,
        )

    result = run_drfo_search(
        prep.geometry, prep.coord_system, prep.initial_hessian, prep.guide_vector_provider,
        calculator, overlap_thresh=overlap_thresh, max_ts_steps=max_ts_steps,
        convergence=convergence, keep_trajectory=keep_trajectory,
        imaginary_mode_tol=imaginary_mode_tol, hessian_refresh_interval=hessian_refresh_interval,
    )
    result.breaking_bonds = prep.breaking_bonds
    result.forming_bonds = prep.forming_bonds
    if keep_trajectory:
        result.trajectory = prep.trajectory + result.trajectory
    return result

"""Standalone dRFO transition-state search core: given an already-built
starting geometry, internal coordinate system, initial Hessian, and a
guide-vector provider (the callable that tells the dRFO step which
direction defines the transition-vector subspace each step -- see
`rfo_core.classify_eigenvectors`), run the divided-RFO search to
convergence.

Deliberately knows nothing about *how* those inputs were constructed. The
paper's own recipe -- bond-order interpolation guess, constrained
pre-relaxation, and a Delta-b-derived guide vector -- lives in
`hessian.schlegel_prep.build_schlegel_ts_preparation`, which produces
exactly the inputs this module needs. Any other guess source (a docking
pose, an ML-predicted TS, a hand-built structure) or a different
transition-vector heuristic can be substituted by calling `run_drfo_search`
directly with its own inputs, without touching this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from ..calculators.base import Calculator, CalculatorError, SCFConvergenceError
from ..geometry import Geometry
from ..hessian.updates import bofill_update
from ..internal.coordinates import InternalCoordinateSystem
from ..internal.transform import b_pinv
from .drfo import DRFOStepper, cartesian_gradient_to_internal
from .trust import ConvergenceCriteria

logger = logging.getLogger(__name__)

Status = Literal[
    "converged", "max_iter", "calculator_failure", "scf_convergence_failure",
    "backtransform_failure", "no_progress",
]


def log_calculator_error(context: str, exc: CalculatorError) -> Status:
    """Log a calculator failure and classify it: SCF/SCC non-convergence is
    logged at INFO (a real, sometimes-expected outcome for geometries far
    from equilibrium -- not evidence the optimizer itself is broken) with
    its own distinct status, while every other calculator failure (crashed
    subprocess, malformed input, missing binary, ...) is logged as a
    genuine error."""
    if isinstance(exc, SCFConvergenceError):
        logger.info(
            "%s: xtb's SCF/SCC iterator did not converge at this geometry -- "
            "this is a reference-method electronic-structure limitation, not "
            "necessarily an optimizer bug (see SCFConvergenceError). %s",
            context, exc,
        )
        return "scf_convergence_failure"
    logger.exception("%s", context)
    return "calculator_failure"


@dataclass
class TSResult:
    geometry: Geometry
    energy: float
    gradient_rms: float
    hessian: np.ndarray | None
    n_imaginary: int | None
    converged: bool
    status: Status
    n_steps: int
    breaking_bonds: list[tuple[int, int]] = field(default_factory=list)
    forming_bonds: list[tuple[int, int]] = field(default_factory=list)
    trajectory: list[Geometry] = field(default_factory=list)
    used_best_available: bool = False


def run_drfo_search(
    guess: Geometry,
    ics: InternalCoordinateSystem,
    initial_hessian: np.ndarray,
    guide_vector_provider: Callable[[np.ndarray], np.ndarray],
    calculator: Calculator,
    *,
    overlap_thresh: float = 0.5,
    max_ts_steps: int = 100,
    convergence: ConvergenceCriteria | None = None,
    keep_trajectory: bool = False,
    imaginary_mode_tol: float = 1e-4,
    hessian_refresh_interval: int | None = None,
) -> TSResult:
    """Run the divided-RFO transition-state search starting from `guess`,
    whose Hessian `initial_hessian` is expressed in the internal-coordinate
    space defined by `ics`. `guide_vector_provider(B) -> s_NR` is called at
    every step with the current B-matrix and must return the vector used to
    classify Hessian eigenvectors into TS-space vs. minimization-space
    (`rfo_core.classify_eigenvectors`); the paper's own choice -- projecting
    a fixed Delta-b vector through B B+ each step -- is built for you by
    `hessian.schlegel_prep.build_schlegel_ts_preparation`, but any other
    strategy can be substituted here.

    `hessian_refresh_interval` (None = off, matching the paper's default use
    of pure Hessian updating): every N accepted TS-search steps, replace the
    Bofill-updated Hessian with a freshly computed exact one (a numerical
    Hessian from the calculator) instead of continuing to update it from
    secant pairs. Tested on 1,3-pentadiene (13 atoms/39 DOF) expecting this
    to help -- it did not; see driver.find_ts's former docstring / git
    history for the root-cause analysis (the B+^T H_cart B+ transform
    amplifies some of B's near-singular values regardless of whether H_cart
    is exact or Bofill-approximate). Kept as a toggle for systems where
    staleness genuinely is the bottleneck.
    """
    crit = convergence or ConvergenceCriteria()
    stepper = DRFOStepper(ics, guide_vector_provider, calculator, overlap_thresh=overlap_thresh, crit=crit)

    geom = guess
    H_int = initial_hessian
    converged = False
    status: Status = "max_iter"
    n_steps = 0
    final_gradient_rms = float("nan")
    trajectory: list[Geometry] = []

    # Track the best (lowest-gradient) geometry seen, not just wherever the
    # search happens to stop: the dRFO search can get very close to a
    # genuine TS and then overshoot past it into a worse stationary region
    # (observed on SiH2+H2->SiH4: gradient RMS reached 7.6e-4 at step 450,
    # close to the 3.2e-4 threshold, then rose to 2.0e-3 by step 550 with a
    # second imaginary mode appearing). When the search doesn't cleanly
    # converge, "closest approach to a stationary point" is a far more
    # useful answer than "wherever the last accepted step happened to
    # land," which is otherwise an arbitrary point on the trajectory.
    best_geom = guess
    best_gradient_rms = float("inf")

    try:
        for step_idx in range(max_ts_steps):
            outcome = stepper.step(geom, H_int)
            n_steps = step_idx + 1
            final_gradient_rms = float(np.sqrt(np.mean(outcome.g_int**2)))

            if not outcome.accepted:
                status = "backtransform_failure" if not outcome.backtransform_ok else "no_progress"
                break

            B_new = ics.B(outcome.geometry)
            g_int_new = cartesian_gradient_to_internal(
                B_new, calculator.gradient(outcome.geometry).reshape(-1))
            g_new_rms = float(np.sqrt(np.mean(g_int_new**2)))
            if g_new_rms < best_gradient_rms:
                best_gradient_rms = g_new_rms
                best_geom = outcome.geometry

            if hessian_refresh_interval and n_steps % hessian_refresh_interval == 0:
                H_cart_exact = calculator.hessian(outcome.geometry)
                Bp_new = b_pinv(B_new)
                H_int = Bp_new.T @ H_cart_exact @ Bp_new
            else:
                H_int = bofill_update(H_int, outcome.dq, g_int_new - outcome.g_int)
            geom = outcome.geometry

            if keep_trajectory:
                trajectory.append(geom.copy())

            if outcome.converged:
                converged = True
                status = "converged"
                final_gradient_rms = g_new_rms
                break
    except CalculatorError as exc:
        status = log_calculator_error("calculator failure during TS search", exc)

    used_best_available = False
    if not converged and status in ("max_iter", "backtransform_failure", "no_progress"):
        if best_gradient_rms < final_gradient_rms or (
            np.isnan(final_gradient_rms) and np.isfinite(best_gradient_rms)
        ):
            geom = best_geom
            final_gradient_rms = best_gradient_rms
            used_best_available = True

    hessian = None
    n_imaginary = None
    energy = float("nan")
    try:
        energy = calculator.energy(geom)
        if status in ("converged", "max_iter"):
            hessian = calculator.hessian(geom)
            eigvals = np.linalg.eigvalsh(hessian)
            n_imaginary = int(np.sum(eigvals < -imaginary_mode_tol))
    except CalculatorError as exc:
        final_status = log_calculator_error("calculator failure during final characterization", exc)
        if status == "converged":
            status = final_status

    return TSResult(
        geometry=geom, energy=energy, gradient_rms=final_gradient_rms,
        hessian=hessian, n_imaginary=n_imaginary, converged=converged, status=status,
        n_steps=n_steps, trajectory=trajectory, used_best_available=used_best_available,
    )

"""M1 validation-only stepper: plain Cartesian-space pRFO/dRFO, bypassing the
internal-coordinate machinery entirely. This exists solely to validate the
augmented-Hessian shift-matrix math (`rfo_core.py`) and the xtb subprocess
plumbing (`calculators/xtb.py`) end-to-end on a real PES before the heavier
internal-coordinate stepper (`drfo.py`, M4) is built. Not part of the public
API; superseded by the internal-coordinate `DRFOStepper` once M4 lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calculators.base import Calculator
from ..geometry import Geometry
from .rfo_core import drfo_step
from .trust import ConvergenceCriteria, TrustRadiusController, check_converged


@dataclass
class ToyTSResult:
    geometry: Geometry
    energy: float
    converged: bool
    n_steps: int
    trajectory: list[Geometry] = field(default_factory=list)


def bfgs_update_cartesian(H: np.ndarray, dx: np.ndarray, dg: np.ndarray,
                           eps: float = 1e-8) -> np.ndarray:
    denom = float(dx @ dg)
    if denom <= eps * np.linalg.norm(dx) * np.linalg.norm(dg):
        return H
    Hdx = H @ dx
    return H + np.outer(dg, dg) / denom - np.outer(Hdx, Hdx) / float(dx @ Hdx)


def find_ts_cartesian_toy(
    guess: Geometry, calc: Calculator, guide_vector: np.ndarray, *,
    max_steps: int = 100, overlap_thresh: float = 0.5,
    convergence: ConvergenceCriteria | None = None,
    keep_trajectory: bool = False,
) -> ToyTSResult:
    """A minimal Cartesian pRFO/dRFO transition-state search, used only to
    validate `rfo_core.drfo_step` and the xtb calculator on a real PES.

    `guide_vector` plays the role Δb plays in the full pipeline: a fixed
    (3N,) direction used to pick which Hessian eigenvector is the
    transition mode. For a toy 3-atom case this can be hand-specified
    (e.g. the reactant->product Cartesian displacement) since the full
    bonding-derived Δb machinery is not yet built (see M4).
    """
    crit = convergence or ConvergenceCriteria()
    trust = TrustRadiusController(initial=0.3, upper=0.5)

    geom = guess.copy()
    n = geom.natoms
    dim = 3 * n

    H = calc.hessian(geom).copy()
    g = calc.gradient(geom).reshape(dim)
    energy = calc.energy(geom)

    trajectory = [geom.copy()] if keep_trajectory else []
    converged = False
    step_count = 0

    for step_count in range(1, max_steps + 1):
        dq = drfo_step(H, g, guide_vector, overlap_thresh=overlap_thresh)
        dq = trust.scale(dq)

        new_geom = geom.copy()
        new_geom.coords = geom.coords + dq.reshape(n, 3)
        new_g = calc.gradient(new_geom).reshape(dim)
        new_energy = calc.energy(new_geom)

        good_step = new_energy < energy + 1e-6 or np.linalg.norm(new_g) < np.linalg.norm(g)
        trust.report(good_step)

        H = bfgs_update_cartesian(H, dq, new_g - g)

        geom, g, energy = new_geom, new_g, new_energy
        if keep_trajectory:
            trajectory.append(geom.copy())

        if check_converged(g, dq, crit):
            converged = True
            break

    return ToyTSResult(geometry=geom, energy=energy, converged=converged,
                        n_steps=step_count, trajectory=trajectory)

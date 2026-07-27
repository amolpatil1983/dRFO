"""Constrained relaxation: minimize energy while a chosen subset of
redundant internal coordinates is held fixed by projecting them out of the
gradient (rather than literally clamping them), matching how the paper's
"relax orthogonal coordinates" step is normally implemented in redundant-
coordinate optimizers. Shared by the guess-builder's interpolation/
relaxation steps and by `driver.find_ts`'s pre-relaxation phase.

A hand-rolled, explicitly trust-region-capped BFGS loop is used rather than
scipy's raw `minimize(method="BFGS")`. Earlier versions of this module used
scipy directly; on a small molecule (3-4 atoms) that happened to behave,
but on a larger one (13-atom 1,3-pentadiene) the unconstrained first step
from an identity-Hessian BFGS start sent atoms flying to >1000 Angstrom
separation -- scipy reported this as "converged" (gradients vanish at that
range) while every subsequent real calculation on that geometry failed.
Catching the calculator failure hid the crash but not the underlying
divergence. An explicit trust radius, with steps rejected outright (not
just penalized) when energy doesn't behave as the quadratic model predicts,
fixes this at the root instead of papering over its symptoms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calculators.base import Calculator, CalculatorError
from ..geometry import Geometry
from ..hessian.updates import bfgs_update
from ..internal.coordinates import InternalCoordinateSystem
from .trust import TrustRadiusController


@dataclass
class RelaxResult:
    geometry: Geometry
    energy: float
    converged: bool
    n_steps: int
    hessian: np.ndarray | None = None  # Cartesian, BFGS-updated (not exact)


def _projector(B_active: np.ndarray, dim: int, *, rcond: float = 1e-10) -> np.ndarray:
    if B_active.shape[0] == 0:
        return np.eye(dim)
    G = B_active @ B_active.T
    G_inv = np.linalg.pinv(G, rcond=rcond)
    return np.eye(dim) - B_active.T @ G_inv @ B_active


def constrained_relax(
    geom: Geometry, ics: InternalCoordinateSystem, frozen_indices: set[int],
    calc: Calculator, *, max_steps: int = 100, rms_grad_threshold: float = 3.2e-4,
    trust_radius: float = 0.3, max_retries: int = 6, return_hessian: bool = False,
) -> RelaxResult:
    """Minimize `calc`'s energy starting from `geom`, holding the internal
    coordinates named by `frozen_indices` fixed (via gradient projection,
    recomputed at every step since bond directions rotate as the geometry
    changes). Every step is capped by an explicit trust radius and rejected
    outright (not merely penalized) if it doesn't decrease energy within a
    small tolerance -- this is what prevents the divergence described in
    the module docstring."""
    n = geom.natoms
    dim = 3 * n
    frozen = sorted(frozen_indices)

    def projected_gradient(g: Geometry) -> tuple[float, np.ndarray]:
        result = calc.compute(g, gradient=True)
        B_full = ics.B(g)
        B_active = B_full[frozen] if frozen else np.zeros((0, dim))
        P = _projector(B_active, dim)
        return result.energy, P @ result.gradient.reshape(-1)

    current = geom.copy()
    H = np.eye(dim)
    trust = TrustRadiusController(initial=trust_radius, upper=max(trust_radius, 0.3))

    energy, g_proj = projected_gradient(current)
    converged = np.sqrt(np.mean(g_proj**2)) < rms_grad_threshold
    n_steps = 0

    for _ in range(max_steps):
        if converged:
            break
        n_steps += 1

        accepted = False
        candidate_geom = current
        candidate_energy = energy
        candidate_g_proj = g_proj
        step_taken = np.zeros(dim)

        for _retry in range(max_retries + 1):
            step_raw = -np.linalg.solve(H + 1e-8 * np.eye(dim), g_proj)
            norm = np.linalg.norm(step_raw)
            step = step_raw if norm <= trust.radius else step_raw * (trust.radius / norm)

            trial_geom = current.copy()
            trial_geom.coords = current.coords + step.reshape(n, 3)
            try:
                trial_energy, trial_g_proj = projected_gradient(trial_geom)
            except CalculatorError:
                trust.report(False)
                continue

            if trial_energy <= energy + 1e-6:
                accepted = True
                candidate_geom, candidate_energy, candidate_g_proj = trial_geom, trial_energy, trial_g_proj
                step_taken = step
                break
            trust.report(False)

        if not accepted:
            break  # cannot make further progress within the trust-radius floor

        trust.report(True)
        H = bfgs_update(H, step_taken, candidate_g_proj - g_proj)
        current, energy, g_proj = candidate_geom, candidate_energy, candidate_g_proj
        converged = np.sqrt(np.mean(g_proj**2)) < rms_grad_threshold

    return RelaxResult(
        geometry=current, energy=float(energy), converged=bool(converged), n_steps=n_steps,
        hessian=H if return_hessian else None,
    )

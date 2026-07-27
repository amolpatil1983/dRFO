"""The internal-coordinate dRFO stepper: ties together B-matrix rebuild,
Cartesian->internal gradient transform, the dRFO shift-matrix solve
(`rfo_core.drfo_step`), trust-region step control, and the iterative
back-transform to Cartesians. Hessian updates are deliberately performed
*outside* this class (by the caller, e.g. `driver.find_ts`) so the same
stepper works whether the Hessian is being updated with BFGS or Bofill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..calculators.base import Calculator
from ..geometry import Geometry
from ..internal.coordinates import InternalCoordinateSystem
from ..internal.transform import b_pinv, internal_to_cartesian_step
from .rfo_core import drfo_step, floor_eigenvalues
from .trust import ConvergenceCriteria, TrustRadiusController, check_converged


@dataclass
class StepOutcome:
    geometry: Geometry
    energy: float
    g_int: np.ndarray
    dq: np.ndarray
    s_NR: np.ndarray
    converged: bool
    step_norm: float
    backtransform_ok: bool
    accepted: bool


def cartesian_gradient_to_internal(B: np.ndarray, g_cart: np.ndarray) -> np.ndarray:
    """g_q = (B+)^T g_x."""
    Bp = b_pinv(B)
    return Bp.T @ g_cart


def _step_is_acceptable(
    predicted_dE: float, actual_dE: float, *,
    flat_tol: float = 1e-7, ratio_lo: float = 0.15, ratio_hi: float = 6.0,
) -> bool:
    """Trust-region ratio test: compares the quadratic model's predicted
    energy change (g.dq + 0.5 dq^T H dq) against the energy change actually
    observed after taking the step. This is what catches a step that is
    numerically well-formed (the back-transform converges to *some*
    geometry) but physically nonsensical -- e.g. a badly-scaled dRFO step
    whose predicted-vs-actual energy change disagree wildly, which a
    "did the back-transform converge?" check alone cannot detect.

    A saddle-point search can predict *either* sign of energy change
    (uphill along the TS mode, downhill elsewhere), so unlike a pure
    minimizer's trust region, this only checks that the actual change has
    the same sign as predicted and a comparable magnitude -- not that
    energy decreased.
    """
    if abs(predicted_dE) < flat_tol:
        # Model predicts an (near-)stationary step; accept only if the
        # actual energy also barely moved.
        return abs(actual_dE) < 10 * flat_tol
    ratio = actual_dE / predicted_dE
    if ratio <= 0:
        return False  # wrong sign entirely -- model badly disagrees with reality
    return ratio_lo <= ratio <= ratio_hi


class DRFOStepper:
    def __init__(
        self, ics: InternalCoordinateSystem, s_NR_provider: Callable[[np.ndarray], np.ndarray],
        calc: Calculator, *, overlap_thresh: float = 0.5,
        trust: TrustRadiusController | None = None,
        crit: ConvergenceCriteria | None = None,
        max_backtransform_retries: int = 15,
        min_abs_eigval: float = 0.02,
        max_abs_eigval: float | None = 10.0,
    ) -> None:
        self.ics = ics
        self.s_NR_provider = s_NR_provider
        self.calc = calc
        self.overlap_thresh = overlap_thresh
        self.trust = trust or TrustRadiusController()
        self.crit = crit or ConvergenceCriteria()
        self.max_backtransform_retries = max_backtransform_retries
        self.min_abs_eigval = min_abs_eigval
        self.max_abs_eigval = max_abs_eigval

    def step(self, geom: Geometry, H_int: np.ndarray) -> StepOutcome:
        B = self.ics.B(geom)
        result_old = self.calc.compute(geom, gradient=True)
        g_cart = result_old.gradient.reshape(-1)
        energy_old = result_old.energy
        g_int = cartesian_gradient_to_internal(B, g_cart)
        s_NR = self.s_NR_provider(B)

        # Floor (and cap) H_int's eigenvalue magnitudes once, up front, and
        # reuse the SAME adjusted Hessian both for the RFO step and for the
        # predicted-energy-change quadratic model below -- using the
        # unadjusted H_int for the latter would score the step against
        # curvature information that didn't actually produce it, causing
        # spurious rejections. See floor_eigenvalues' docstring for why both
        # a floor and a cap are needed.
        H_floored = (
            floor_eigenvalues(H_int, self.min_abs_eigval, max_abs_eigval=self.max_abs_eigval)
            if self.min_abs_eigval > 0 or self.max_abs_eigval is not None else H_int
        )

        dq_raw = drfo_step(H_floored, g_int, s_NR, overlap_thresh=self.overlap_thresh)
        # Project onto B's row space (same B B+ trick used for s_NR):
        # dq_raw comes out of an unconstrained solve in the abstract
        # redundant-coordinate vector space, which has no way to know that
        # only a <=3N-6 dimensional subspace of that space is actually
        # achievable by any Cartesian displacement. An unprojected dq_raw
        # can carry a component in the unreachable complement -- and
        # crucially, that component does NOT shrink away under trust-
        # radius scaling (a structurally-unreachable direction is
        # unreachable at any magnitude), so the back-transform's Newton
        # residual plateaus immediately and never converges, however small
        # the requested step. Observed directly: a target step scaled down
        # to 0.001 in magnitude still left a fixed ~5e-4 residual after the
        # first iteration, with zero further correction possible.
        Bp_full = b_pinv(B)
        dq_raw = B @ (Bp_full @ dq_raw)
        raw_norm = np.linalg.norm(dq_raw)

        trust_radius = self.trust.radius
        accepted = False
        backtransform_ok = False
        new_geom = geom
        energy_new = energy_old
        dq = dq_raw

        for _ in range(self.max_backtransform_retries + 1):
            dq_scaled = dq_raw if raw_norm <= trust_radius else dq_raw * (trust_radius / raw_norm)
            candidate_geom, backtransform_ok = internal_to_cartesian_step(geom, self.ics, dq_scaled)
            if backtransform_ok:
                candidate_energy = self.calc.energy(candidate_geom)
                predicted_dE = float(g_int @ dq_scaled + 0.5 * dq_scaled @ H_floored @ dq_scaled)
                actual_dE = candidate_energy - energy_old
                if _step_is_acceptable(predicted_dE, actual_dE):
                    accepted = True
                    new_geom, energy_new, dq = candidate_geom, candidate_energy, dq_scaled
                    break
            trust_radius *= 0.5

        self.trust.report(accepted)
        if accepted:
            converged = check_converged(g_int, dq, self.crit)
        else:
            converged = False
            dq = np.zeros_like(dq_raw)

        return StepOutcome(
            geometry=new_geom, energy=energy_new, g_int=g_int, dq=dq, s_NR=s_NR,
            converged=converged, step_norm=float(np.linalg.norm(dq)),
            backtransform_ok=backtransform_ok, accepted=accepted,
        )

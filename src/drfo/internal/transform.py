"""B+ pseudoinverse and the iterative Pulay-Fogarasi internal<->Cartesian
back-transformation shared by the guess-builder, the constrained minimizer,
and the dRFO stepper.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Geometry
from .coordinates import InternalCoordinateSystem, Torsion


def b_pinv(B: np.ndarray, *, rcond: float = 1e-10) -> np.ndarray:
    """B+ = G^-1 B^T where G = B B^T (Pulay/Fogarasi), falling back to an
    SVD-based pseudoinverse of G when it is near-singular (e.g. too few
    internal coordinates for the DOF count, or coincident coordinates)."""
    G = B @ B.T
    G_inv = np.linalg.pinv(G, rcond=rcond)
    return B.T @ G_inv


def _wrap_torsion_residuals(ics: InternalCoordinateSystem, residual: np.ndarray) -> np.ndarray:
    """Wrap torsion-coordinate residuals into (-pi, pi] so a target near
    +pi is not chased the "long way around" through 0."""
    out = residual.copy()
    for idx, c in enumerate(ics.coords):
        if isinstance(c, Torsion):
            out[idx] = (out[idx] + np.pi) % (2 * np.pi) - np.pi
    return out


def cartesian_to_internal(geom: Geometry, ics: InternalCoordinateSystem) -> np.ndarray:
    return ics.values(geom)


def internal_to_cartesian_step(
    geom: Geometry, ics: InternalCoordinateSystem, dq: np.ndarray, *,
    max_iter: int = 25, tol: float = 1e-4, max_dx_norm: float = 1.0,
) -> tuple[Geometry, bool]:
    """Iteratively find the Cartesian displacement whose internal-coordinate
    change matches `dq` (a redundant coordinate system generally has no
    exact solution for an arbitrary dq, so this converges to a
    least-squares-consistent geometry via repeated B+ correction):

        x_1 = x_0 + B+ dq
        repeat: q_actual = ics.values(x_k)
                residual = dq - (q_actual - q_0)  [torsion components wrapped]
                x_{k+1} = x_k + B+(x_k) residual
        until ||residual|| < tol or max_iter reached.

    Each Newton correction `dx` is capped at `max_dx_norm` (scaled down,
    preserving direction, if it would exceed that) and the loop bails out
    early -- reporting not converged, rather than continuing to amplify --
    if the residual grows instead of shrinking between iterations. Without
    this, a momentarily ill-conditioned B+ (e.g. on a larger, more flexible
    molecule where B briefly loses rank along some direction mid-iteration)
    can produce an uncapped correction that sends the geometry to physically
    absurd coordinates (observed: >1000 Angstrom atom separations on a
    13-atom test case) instead of just failing to converge cleanly.

    Returns (new_geometry, converged).
    """
    q0 = ics.values(geom)
    target = q0 + dq

    x = geom.copy()
    converged = False
    prev_residual_norm = float("inf")
    for _ in range(max_iter):
        B = ics.B(x)
        Bp = b_pinv(B)
        q_actual = ics.values(x)
        residual = target - q_actual
        residual = _wrap_torsion_residuals(ics, residual)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tol:
            converged = True
            break
        if residual_norm > prev_residual_norm:
            break  # diverging -- stop before amplifying further
        prev_residual_norm = residual_norm

        dx = Bp @ residual
        dx_norm = np.linalg.norm(dx)
        if dx_norm > max_dx_norm:
            dx = dx * (max_dx_norm / dx_norm)
        x = x.copy()
        x.coords = x.coords + dx.reshape(x.natoms, 3)
    else:
        B = ics.B(x)
        q_actual = ics.values(x)
        residual = _wrap_torsion_residuals(ics, target - q_actual)
        converged = bool(np.linalg.norm(residual) < tol)

    return x, converged

"""Hessian update formulas operating in internal-coordinate space. BFGS is
used during the constrained pre-relaxation phase (always positive
definite -- appropriate for a minimization); Bofill's combined SR1/PSB
update (Bofill, J. Comput. Chem. 1994, 15, 1) is used during the
transition-state search since it can represent negative curvature, unlike
BFGS.
"""
from __future__ import annotations

import numpy as np


def bfgs_update(H: np.ndarray, dx: np.ndarray, dg: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Standard BFGS update. Skips the update (returns H unchanged) if the
    curvature condition dx.dg > 0 fails, which BFGS requires to remain
    positive definite."""
    dx_dg = float(dx @ dg)
    if dx_dg <= eps * np.linalg.norm(dx) * np.linalg.norm(dg):
        return H
    Hdx = H @ dx
    dx_Hdx = float(dx @ Hdx)
    if dx_Hdx <= eps:
        return H
    return H + np.outer(dg, dg) / dx_dg - np.outer(Hdx, Hdx) / dx_Hdx


def bofill_update(H: np.ndarray, dx: np.ndarray, dg: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Bofill's combined SR1/PSB update:

        E = dg - H@dx
        phi = (E.dx)^2 / ((E.E)(dx.dx))
        H_new = phi * SR1(H,dx,dg) + (1-phi) * PSB(H,dx,dg)

    where
        SR1(H,dx,dg) = H + outer(E,E) / (E.dx)
        PSB(H,dx,dg) = H + (outer(E,dx)+outer(dx,E))/(dx.dx)
                         - (dx.E)*outer(dx,dx)/(dx.dx)^2

    Unlike BFGS, this can produce and maintain negative eigenvalues, which
    is required once the optimizer is searching for a transition state
    rather than minimizing.
    """
    dx_dx = float(dx @ dx)
    if dx_dx <= eps:
        return H
    E = dg - H @ dx
    E_dx = float(E @ dx)
    E_E = float(E @ E)

    psb = H + (np.outer(E, dx) + np.outer(dx, E)) / dx_dx - E_dx * np.outer(dx, dx) / dx_dx**2

    if abs(E_dx) <= eps or E_E <= eps:
        # SR1 term is ill-conditioned (E nearly zero, or E nearly
        # orthogonal to dx); fall back to the well-conditioned PSB term.
        return psb

    sr1 = H + np.outer(E, E) / E_dx
    phi = (E_dx**2) / (E_E * dx_dx)
    phi = float(np.clip(phi, 0.0, 1.0))
    return phi * sr1 + (1.0 - phi) * psb

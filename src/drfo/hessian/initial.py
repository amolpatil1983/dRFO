"""Construct the initial transition-state Hessian from the (positive
definite) BFGS-updated Hessian obtained during constrained pre-relaxation,
by flipping and scaling its curvature along the projected Delta-b
direction (paper eq. 12)."""
from __future__ import annotations

import numpy as np


def build_initial_ts_hessian(
    H_tilde: np.ndarray, s_NR: np.ndarray, *, flip_scale: float = 1.5,
) -> np.ndarray:
    """H0 = H_tilde - flip_scale * (s^T H_tilde s)/(s^T s) * outer(s,s)/(s^T s)."""
    s_dot_s = float(s_NR @ s_NR)
    if s_dot_s < 1e-12:
        raise ValueError("s_NR is (numerically) zero; cannot construct initial TS Hessian")
    rayleigh = float(s_NR @ H_tilde @ s_NR) / s_dot_s
    return H_tilde - flip_scale * rayleigh * np.outer(s_NR, s_NR) / s_dot_s

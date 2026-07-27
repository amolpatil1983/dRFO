"""Shared eigenvector classification and augmented-Hessian shift-matrix math
for pRFO/dRFO transition-state stepping (Birkholz & Schlegel, JCC 2015, eqs. 6, 13).

`drfo_step` is the single function used for both plain partitioned RFO (pRFO)
and the paper's divided RFO (dRFO): pRFO is simply the outcome when
`classify_eigenvectors` happens to place exactly one eigenvector in the
"transition vector" space, not a separate code path.

Everything here is coordinate-system agnostic: it operates on whatever
Hessian `H` and gradient `g` it is given (Cartesian or internal-coordinate),
and a "guide" vector `s` (e.g. Δb, or a reactant/product tangent) used only
to decide which eigenvector(s) to climb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class EigenClassification:
    ts_indices: list[int]
    min_indices: list[int]
    eigvals: np.ndarray
    eigvecs: np.ndarray  # columns are eigenvectors, matches np.linalg.eigh convention


def classify_eigenvectors(
    H: np.ndarray, s: np.ndarray, *, overlap_thresh: float = 0.5,
) -> EigenClassification:
    """Split eigenvectors of H into a "transition vector" space (overlap
    with `s` exceeds `overlap_thresh`) and a "minimization" space (the
    rest). `s` need not be normalized or exactly in the row space of H;
    only its direction matters.

    Falls back to putting the single most negative eigenvalue's index in
    ts_indices (with the caller expected to have guarded against a truly
    degenerate `s` upstream) if no eigenvector clears the threshold.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-12:
        raise ValueError("guide vector s is (numerically) zero; cannot classify eigenvectors")
    s_hat = s / s_norm

    overlaps = eigvecs.T @ s_hat  # (n,)
    ts_indices = [i for i in range(len(eigvals)) if abs(overlaps[i]) > overlap_thresh]
    if not ts_indices:
        ts_indices = [int(np.argmin(eigvals))]
    min_indices = [i for i in range(len(eigvals)) if i not in ts_indices]

    return EigenClassification(ts_indices=ts_indices, min_indices=min_indices,
                                eigvals=eigvals, eigvecs=eigvecs)


def augmented_hessian_eigs(
    H_sub: np.ndarray, g_sub: np.ndarray, *, which: Literal["min2", "max"],
) -> float:
    """Diagonalize the augmented Hessian [[H_sub, g_sub], [g_sub^T, 0]]
    built in a subspace and return either:
      - 'max':  the most negative eigenvalue (used for the minimization
                space's shift factor, lambda2 in the paper).
      - 'min2': the second-most-negative eigenvalue (used for the TS
                space's shift factor, lambda1, and for the full-space
                correction lambda3; reduces to the single-mode pRFO
                lambda when H_sub is 1x1).
    """
    k = H_sub.shape[0]
    H_aug = np.zeros((k + 1, k + 1))
    H_aug[:k, :k] = H_sub
    H_aug[:k, k] = g_sub
    H_aug[k, :k] = g_sub
    eigvals = np.linalg.eigvalsh(H_aug)  # ascending order
    if which == "max":
        return float(eigvals[0])
    if which == "min2":
        if len(eigvals) < 2:
            return float(eigvals[0])
        return float(eigvals[1])
    raise ValueError(f"unknown which={which!r}")


def rfo_shift_matrix(cls: EigenClassification, g: np.ndarray) -> np.ndarray:
    """Assemble the dRFO shift matrix S (paper eq. 13):

        S = -(lambda1+lambda3) * sum_{i in TS}  v_i v_i^T
            -(lambda2+lambda3) * sum_{i in min} v_i v_i^T

    lambda1: second-most-negative eigenvalue of the augmented Hessian built
             in the TS-space subspace (reduces to standard pRFO's lambda
             when the TS space has one vector).
    lambda2: most-negative eigenvalue of the augmented Hessian built in the
             minimization-space subspace.
    lambda3: second-most-negative eigenvalue of the augmented Hessian built
             in the *full* space (the dRFO stabilization term).
    """
    V = cls.eigvecs
    ts, mn = cls.ts_indices, cls.min_indices
    g_ts = V[:, ts].T @ g
    g_mn = V[:, mn].T @ g
    H_ts = np.diag(cls.eigvals[ts])
    H_mn = np.diag(cls.eigvals[mn])
    H_full = np.diag(cls.eigvals)
    g_full = V.T @ g

    lam1 = augmented_hessian_eigs(H_ts, g_ts, which="min2")
    lam2 = augmented_hessian_eigs(H_mn, g_mn, which="max")
    lam3 = augmented_hessian_eigs(H_full, g_full, which="min2")

    n = V.shape[0]
    S = np.zeros((n, n))
    for i in ts:
        v = V[:, i]
        S -= (lam1 + lam3) * np.outer(v, v)
    for i in mn:
        v = V[:, i]
        S -= (lam2 + lam3) * np.outer(v, v)
    return S


def floor_eigenvalues(
    H: np.ndarray, min_abs_eigval: float, *, max_abs_eigval: float | None = None,
) -> np.ndarray:
    """Reconstruct H with every eigenvalue's magnitude floored to at least
    `min_abs_eigval` (sign preserved; exact zeros treated as positive), and
    optionally also capped at `max_abs_eigval` from above.

    A redundant internal-coordinate Hessian routinely carries near-zero
    eigenvalues -- from genuine coordinate redundancy, or from a
    B-matrix row that was zeroed out (see internal/bmatrix.py's runtime
    torsion-singularity guard). Passed through unmodified, the RFO solve
    divides by these near-zero curvatures and produces a step dominated by
    a large, poorly-determined component along that direction; trust-region
    scaling then has to shrink the *entire* step vector to tame that one
    component, strangling progress along the well-conditioned directions
    too. Flooring only the offending eigenvalues (not scaling the whole
    step) fixes this at its source rather than symptomatically.

    The upper cap guards a different, empirically-observed pathology: the
    Cartesian-to-internal-coordinate Hessian transform (B+^T H_cart B+)
    divides by B's singular values, and a B-matrix singular value that is
    small but *not* near the redundancy threshold (a genuinely soft, valid
    coordinate direction -- e.g. a wide bend angle in a large flexible
    molecule) still gets divided by twice (once in each B+ factor),
    amplifying whatever approximate/BFGS-noise curvature the Cartesian
    Hessian has in that direction by two to three orders of magnitude.
    Observed directly on 1,3-pentadiene: a Cartesian Hessian with a sane
    0.02-1.3 eigenvalue range produced an internal-coordinate Hessian with
    eigenvalues of 205 and 260 sitting next to an otherwise-normal 0.06-5.2
    spectrum -- an RFO step's component along that spurious direction is
    suppressed by dividing by ~200-260 rather than a physically-reasonable
    ~1-5, which starves the *entire* step (all components share one linear
    solve) even though only one or two coordinates were actually
    ill-conditioned.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    sign = np.where(eigvals < 0, -1.0, 1.0)
    adjusted = np.where(np.abs(eigvals) < min_abs_eigval, sign * min_abs_eigval, eigvals)
    if max_abs_eigval is not None:
        adjusted = np.where(np.abs(adjusted) > max_abs_eigval, sign * max_abs_eigval, adjusted)
    return eigvecs @ np.diag(adjusted) @ eigvecs.T


def drfo_step(
    H: np.ndarray, g: np.ndarray, s: np.ndarray, *,
    overlap_thresh: float = 0.5, ridge: float = 1e-10,
    min_abs_eigval: float = 0.0, max_abs_eigval: float | None = None,
) -> np.ndarray:
    """Compute a dRFO step dq = -(H+S)^-1 g.

    This single function serves both pRFO (n_TS == 1, the common case when
    the Hessian already has the right curvature) and general dRFO (n_TS > 1)
    — which one happens is a consequence of `classify_eigenvectors`, not a
    caller-selected mode.

    `min_abs_eigval` (0 = off, matching prior behavior) floors H's
    eigenvalue magnitudes before stepping, and `max_abs_eigval` (None = off)
    caps them from above; see `floor_eigenvalues`.
    """
    if min_abs_eigval > 0 or max_abs_eigval is not None:
        H = floor_eigenvalues(H, min_abs_eigval, max_abs_eigval=max_abs_eigval)
    cls = classify_eigenvectors(H, s, overlap_thresh=overlap_thresh)
    S = rfo_shift_matrix(cls, g)
    shifted = H + S
    shifted = shifted + ridge * np.eye(shifted.shape[0])
    dq = -np.linalg.solve(shifted, g)
    return dq

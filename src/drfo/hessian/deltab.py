"""The Delta-b transition-vector guess: a trivial, geometry-independent
redundant-internal-coordinate vector built purely from which bonds are
breaking/forming, and its projection into the locally non-redundant space
via the B B+ projector.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import pinv

from ..internal.coordinates import InternalCoordinateSystem, Stretch


def delta_b_vector(
    ics: InternalCoordinateSystem, breaking: list[tuple[int, int]], forming: list[tuple[int, int]],
) -> np.ndarray:
    """+1 at each forming-bond stretch coordinate, -1 at each breaking-bond
    stretch coordinate, 0 elsewhere."""
    forming_set = set(forming)
    breaking_set = set(breaking)
    s = np.zeros(len(ics.coords))
    for idx, c in enumerate(ics.coords):
        if not isinstance(c, Stretch):
            continue
        if (c.i, c.j) in forming_set:
            s[idx] = 1.0
        elif (c.i, c.j) in breaking_set:
            s[idx] = -1.0
    return s


def project_transition_vector(B: np.ndarray, s: np.ndarray, *, rtol: float = 1e-10) -> np.ndarray:
    """s_NR = B @ B+ @ s: project a redundant-internal-coordinate vector
    onto the locally non-redundant row space of B."""
    B_pinv = pinv(B, rtol=rtol)
    return B @ (B_pinv @ s)

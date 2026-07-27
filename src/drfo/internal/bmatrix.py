"""Analytic Wilson B-matrix rows (dq_internal/dx_cartesian) for stretches,
bends, and torsions. Standard formulas (Wilson, Decius & Cross vector
formulation); validated against finite differences in the unit tests.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Geometry
from .coordinates import (
    Bend,
    InternalCoordinateSystem,
    OutOfPlane,
    Stretch,
    Torsion,
    _oop_value,
    is_torsion_near_singular,
)


def stretch_b_row(geom: Geometry, i: int, j: int) -> np.ndarray:
    row = np.zeros((geom.natoms, 3))
    rij = geom.coords[i] - geom.coords[j]
    r = np.linalg.norm(rij)
    e = rij / r
    row[i] = e
    row[j] = -e
    return row.reshape(-1)


def bend_b_row(geom: Geometry, i: int, j: int, k: int) -> np.ndarray:
    row = np.zeros((geom.natoms, 3))
    u = geom.coords[i] - geom.coords[j]
    v = geom.coords[k] - geom.coords[j]
    r1, r2 = np.linalg.norm(u), np.linalg.norm(v)
    e1, e2 = u / r1, v / r2
    cos_phi = np.clip(np.dot(e1, e2), -1.0, 1.0)
    sin_phi = np.sqrt(max(1.0 - cos_phi**2, 1e-12))

    grad_i = (cos_phi * e1 - e2) / (r1 * sin_phi)
    grad_k = (cos_phi * e2 - e1) / (r2 * sin_phi)
    grad_j = -(grad_i + grad_k)

    row[i] = grad_i
    row[j] = grad_j
    row[k] = grad_k
    return row.reshape(-1)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def torsion_b_row(geom: Geometry, i: int, j: int, k: int, l: int) -> np.ndarray:
    """Gradient convention matches `_torsion_value`'s atan2(y, x) formula
    (validated against finite differences; see tests/unit/test_bmatrix.py).

    grad_i and grad_l have a standard closed form. grad_j and grad_k are
    solved from three physical invariance constraints rather than a
    memorized closed form (several sign conventions exist in the
    literature and are easy to get subtly wrong):
      1. translational invariance: grad_i+grad_j+grad_k+grad_l = 0
      2. rotational invariance about atom j: sum_a (r_a - r_j) x grad_a = 0,
         which reduces to (r_k-r_j) x grad_k = -[(r_i-r_j) x grad_i +
         (r_l-r_j) x grad_l] since the grad_j term vanishes at r_j itself
      3. bond-stretch invariance: displacing atom k purely along the j->k
         bond direction cannot change a torsion angle, so grad_k . e2 = 0
         (this is the one degree of freedom the cross-product constraint
         above leaves undetermined, since cross-product-with-a-fixed-
         vector has a 1D null space along that vector).
    Constraints 2+3 give 4 equations for grad_k's 3 components (solved by
    least squares -- consistent, not actually overdetermined); grad_j then
    follows from constraint 1.
    """
    row = np.zeros((geom.natoms, 3))
    b1 = geom.coords[j] - geom.coords[i]
    b2 = geom.coords[k] - geom.coords[j]
    b3 = geom.coords[l] - geom.coords[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    len_b2 = np.linalg.norm(b2)
    n1_sq = max(np.dot(n1, n1), 1e-12)
    n2_sq = max(np.dot(n2, n2), 1e-12)
    e2 = b2 / len_b2

    grad_i = (len_b2 / n1_sq) * n1
    grad_l = -(len_b2 / n2_sq) * n2

    T = -(grad_i + grad_l)  # grad_j + grad_k
    M = -(np.cross(geom.coords[i] - geom.coords[j], grad_i)
          + np.cross(geom.coords[l] - geom.coords[j], grad_l))
    A = np.vstack([_skew(b2), e2.reshape(1, 3)])
    rhs = np.concatenate([M, [0.0]])
    grad_k, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    grad_j = T - grad_k

    row[i] = grad_i
    row[j] = grad_j
    row[k] = grad_k
    row[l] = grad_l
    return row.reshape(-1)


def oop_b_row(geom: Geometry, c: OutOfPlane, *, h: float = 1e-5) -> np.ndarray:
    """B-matrix row for the out-of-plane (Wilson wag) coordinate, via
    central finite difference rather than a hand-derived closed form: the
    analytic OOP Wilson B-matrix formula is one of the more error-prone
    ones to derive correctly (this session's torsion row needed several
    rounds of empirical sign-fixing against finite differences even with a
    known-textbook formula to start from), and this coordinate is not on
    any hot path -- it is evaluated only when building B for a redundant
    coordinate system already dominated by real electronic-structure
    calculator calls, so the ~24 extra `_oop_value` evaluations per row
    (pure numpy, no xtb) are negligible in cost. Central difference is
    self-consistent with the analytic rows validated in
    tests/unit/test_bmatrix.py to ~1e-6 by construction.
    """
    row = np.zeros((geom.natoms, 3))
    for atom in (c.a, c.b, c.c, c.d):
        for dim in range(3):
            plus = geom.copy()
            plus.coords[atom, dim] += h
            minus = geom.copy()
            minus.coords[atom, dim] -= h
            row[atom, dim] = (_oop_value(plus, c) - _oop_value(minus, c)) / (2 * h)
    return row.reshape(-1)


def build_B(ics: InternalCoordinateSystem, geom: Geometry) -> np.ndarray:
    """(n_internal, 3*natoms) Wilson B-matrix.

    Torsion rows are zeroed (rather than left as the analytic formula's
    output) whenever the *current* geometry's torsion value sits near the
    periodic branch point (0/pi): `torsion_b_row`'s formula becomes
    numerically singular there, and injecting a near-degenerate-but-nonzero
    row into B poisons the whole B+ pseudoinverse (observed: condition
    number ~1e16, causing the back-transform to need an unusably tiny step
    to converge at all). A zero row is an honest "this coordinate is
    temporarily uninformative here" rather than corrupting every other
    coordinate's reconstruction via a shared ill-conditioned pinv.

    This is a *runtime* check, distinct from `build_coordinate_system`'s
    construction-time exclusion (which only inspects the two endpoint
    geometries): a torsion can be perfectly well-behaved at both endpoints
    and still pass near the singular point at some *intermediate* geometry
    visited mid-optimization, exactly as observed for a migrating-hydrogen
    torsion during the 1,3-pentadiene benchmark case.
    """
    n_int = len(ics.coords)
    dim = 3 * geom.natoms
    B = np.zeros((n_int, dim))
    for idx, c in enumerate(ics.coords):
        if isinstance(c, Stretch):
            B[idx] = stretch_b_row(geom, c.i, c.j)
        elif isinstance(c, Bend):
            B[idx] = bend_b_row(geom, c.i, c.j, c.k)
        elif isinstance(c, Torsion):
            if is_torsion_near_singular(geom, c.i, c.j, c.k, c.l):
                continue  # leave this row as zero
            B[idx] = torsion_b_row(geom, c.i, c.j, c.k, c.l)
        elif isinstance(c, OutOfPlane):
            B[idx] = oop_b_row(geom, c)
        else:
            raise TypeError(f"unknown coordinate type {type(c)}")
    return B

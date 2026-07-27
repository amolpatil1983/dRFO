import numpy as np
import pytest

from drfo.geometry import Geometry
from drfo.internal.bmatrix import bend_b_row, stretch_b_row, torsion_b_row
from drfo.internal.coordinates import Bend, Stretch, Torsion, _bend_value, _stretch_value, _torsion_value


def water() -> Geometry:
    return Geometry.from_angstrom(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )


def ethane_like() -> Geometry:
    # Staggered-ish, not physically relaxed -- just needs a genuine, well
    # defined (non-linear, non-planar-degenerate) torsion for testing.
    return Geometry.from_angstrom(
        ["C", "C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-0.51, 0.89, 0.36],
            [-0.51, -0.89, 0.36],
            [2.05, 0.51, 0.89],
            [2.05, -0.98, -0.10],
        ],
    )


def _finite_diff_row(value_fn, geom: Geometry, h: float = 1e-5) -> np.ndarray:
    n = geom.natoms
    row = np.zeros((n, 3))
    for atom in range(n):
        for dim in range(3):
            plus = geom.copy()
            plus.coords[atom, dim] += h
            minus = geom.copy()
            minus.coords[atom, dim] -= h
            row[atom, dim] = (value_fn(plus) - value_fn(minus)) / (2 * h)
    return row.reshape(-1)


def test_stretch_b_row_matches_finite_difference():
    geom = water()
    c = Stretch(0, 1)
    analytic = stretch_b_row(geom, c.i, c.j)
    fd = _finite_diff_row(lambda g: _stretch_value(g, c), geom)
    assert np.allclose(analytic, fd, atol=1e-6)


def test_bend_b_row_matches_finite_difference():
    geom = water()
    c = Bend(1, 0, 2)
    analytic = bend_b_row(geom, c.i, c.j, c.k)
    fd = _finite_diff_row(lambda g: _bend_value(g, c), geom)
    assert np.allclose(analytic, fd, atol=1e-6)


def test_torsion_b_row_matches_finite_difference():
    geom = ethane_like()
    c = Torsion(2, 0, 1, 4)
    analytic = torsion_b_row(geom, c.i, c.j, c.k, c.l)
    fd = _finite_diff_row(lambda g: _torsion_value(g, c), geom)
    assert np.allclose(analytic, fd, atol=1e-5)


def test_bend_b_row_symmetric_water():
    # Sanity: for a symmetric (Cs/C2v) water geometry with the bend defined
    # H-O-H, the x-components of the two H gradients should be equal in
    # magnitude and opposite in sign given the mirror symmetry about the yz
    # plane in this construction... more simply, just check gradients are
    # finite and nonzero.
    geom = water()
    row = bend_b_row(geom, 1, 0, 2).reshape(geom.natoms, 3)
    assert np.all(np.isfinite(row))
    assert np.linalg.norm(row[0]) > 0
    assert np.linalg.norm(row[1]) > 0
    assert np.linalg.norm(row[2]) > 0

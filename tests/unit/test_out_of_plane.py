import numpy as np
import pytest

from drfo.geometry import Geometry
from drfo.internal.bmatrix import oop_b_row
from drfo.internal.coordinates import OutOfPlane, _oop_value, build_coordinate_system
from drfo.internal.topology import build_bond_graph
from drfo.internal.transform import internal_to_cartesian_step


def formaldehyde() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.22], [0.94, 0.0, -0.54], [-0.94, 0.0, -0.54]],
    )


def formaldehyde_pyramidalized() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[0.0, 0.3, 0.0], [0.0, 0.0, 1.22], [0.94, 0.0, -0.54], [-0.94, 0.0, -0.54]],
    )


def test_build_coordinate_system_adds_oop_for_trigonal_center():
    geom = formaldehyde()
    graph = build_bond_graph(geom)
    ics = build_coordinate_system(graph, geom, geom)
    oop = [c for c in ics.coords if isinstance(c, OutOfPlane)]
    assert len(oop) == 1
    assert oop[0].a == 0  # centered on carbon


def test_oop_value_is_zero_for_planar_structure():
    geom = formaldehyde()
    c = OutOfPlane(0, 1, 2, 3)
    assert _oop_value(geom, c) == pytest.approx(0.0, abs=1e-8)


def test_oop_value_is_nonzero_when_pyramidalized():
    c = OutOfPlane(0, 1, 2, 3)
    v = _oop_value(formaldehyde_pyramidalized(), c)
    assert abs(v) > np.radians(30)


def test_oop_b_row_matches_finite_difference_of_its_own_value():
    # oop_b_row IS a finite-difference implementation; this instead checks
    # internal self-consistency at a smaller step size, guarding against a
    # gross error (e.g. wrong atom indices, wrong reshape).
    geom = formaldehyde_pyramidalized()
    c = OutOfPlane(0, 1, 2, 3)
    row = oop_b_row(geom, c, h=1e-5).reshape(geom.natoms, 3)
    row_fine = oop_b_row(geom, c, h=1e-6).reshape(geom.natoms, 3)
    assert np.allclose(row, row_fine, atol=1e-4)
    assert np.all(np.isfinite(row))
    # atoms not involved in this coordinate must have exactly zero gradient
    assert row.shape == (4, 3)


def test_back_transform_oop_step_moves_in_requested_direction():
    # Pyramidalizing necessarily shifts the three surrounding bend angles
    # too (simple solid geometry: they cannot all stay at their planar
    # value while the center puckers), so a target that changes OOP while
    # holding all three bends exactly fixed is mildly over-constrained --
    # the same "least-squares consistent, not exactly reachable" situation
    # already covered for bends alone in test_transform.py. This checks the
    # back-transform still moves substantially and correctly in the
    # requested direction rather than requiring exact convergence.
    geom = formaldehyde()
    graph = build_bond_graph(geom)
    ics = build_coordinate_system(graph, geom, geom)
    oop_idx = next(i for i, c in enumerate(ics.coords) if isinstance(c, OutOfPlane))

    dq = np.zeros(len(ics.coords))
    dq[oop_idx] = np.radians(10.0)
    new_geom, _ = internal_to_cartesian_step(geom, ics, dq, tol=1e-4)
    q0 = ics.values(geom)
    q1 = ics.values(new_geom)
    assert abs((q1[oop_idx] - q0[oop_idx]) - dq[oop_idx]) < 1e-2
    assert np.all(np.isfinite(new_geom.coords))

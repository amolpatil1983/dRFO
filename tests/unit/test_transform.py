import numpy as np

from drfo.geometry import Geometry
from drfo.internal.coordinates import Bend, InternalCoordinateSystem, Stretch, Torsion
from drfo.internal.transform import internal_to_cartesian_step


def water() -> Geometry:
    return Geometry.from_angstrom(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )


def formaldehyde() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.22], [0.94, 0.0, -0.54], [-0.94, 0.0, -0.54]],
    )


def water_ics() -> InternalCoordinateSystem:
    return InternalCoordinateSystem(
        coords=[Stretch(0, 1), Stretch(0, 2), Bend(1, 0, 2)], natoms=3,
    )


def formaldehyde_ics() -> InternalCoordinateSystem:
    return InternalCoordinateSystem(
        coords=[
            Stretch(0, 1), Stretch(0, 2), Stretch(0, 3),
            Bend(1, 0, 2), Bend(1, 0, 3), Bend(2, 0, 3),
        ],
        natoms=4,
    )


def test_back_transform_small_stretch_step_water():
    geom = water()
    ics = water_ics()
    q0 = ics.values(geom)
    dq = np.array([0.05, -0.03, 0.0])  # bohr, bohr, radians

    new_geom, converged = internal_to_cartesian_step(geom, ics, dq)
    assert converged
    q1 = ics.values(new_geom)
    assert np.allclose(q1 - q0, dq, atol=1e-4)


def test_back_transform_bend_step_water():
    geom = water()
    ics = water_ics()
    q0 = ics.values(geom)
    dq = np.array([0.0, 0.0, np.radians(5.0)])

    new_geom, converged = internal_to_cartesian_step(geom, ics, dq)
    assert converged
    q1 = ics.values(new_geom)
    assert np.allclose(q1 - q0, dq, atol=1e-4)


def test_back_transform_combined_step_formaldehyde():
    # Two simultaneous, mutually-independent bond stretches from the same
    # central atom (unlike the three H-C-H/H-C-O/H-C-O bend angles, which
    # are NOT independent for a planar center -- they must sum to 360 deg,
    # so an arbitrary combination of bend deltas is a genuinely redundant,
    # generally-inconsistent target; that case is covered separately below).
    geom = formaldehyde()
    ics = formaldehyde_ics()
    q0 = ics.values(geom)
    dq = np.zeros(6)
    dq[0] = 0.08   # stretch C-O
    dq[1] = -0.04  # stretch C-H (independent of C-O)

    new_geom, converged = internal_to_cartesian_step(geom, ics, dq)
    assert converged
    q1 = ics.values(new_geom)
    assert np.allclose(q1 - q0, dq, atol=1e-4)


def test_back_transform_zero_step_is_identity():
    geom = water()
    ics = water_ics()
    new_geom, converged = internal_to_cartesian_step(geom, ics, np.zeros(3))
    assert converged
    assert np.allclose(new_geom.coords, geom.coords, atol=1e-8)


def test_back_transform_redundant_inconsistent_target_stays_bounded():
    # Formaldehyde's three H-C-H/H-C-O/H-C-O bend angles around the planar
    # C center are NOT mutually independent (they must sum to 360 deg), so
    # asking to change just one of them while holding the other two fixed
    # is a genuinely redundant, inconsistent target. The iterative
    # back-transform must not diverge/NaN on an inconsistent target -- it
    # should settle at a bounded, least-squares-consistent geometry instead
    # of reaching the (unreachable) exact target, and `converged` must
    # honestly report False.
    geom = formaldehyde()
    ics = formaldehyde_ics()
    dq = np.zeros(6)
    dq[3] = np.radians(-3.0)  # change only one of three coupled bends

    new_geom, converged = internal_to_cartesian_step(geom, ics, dq)
    assert not converged  # target is inconsistent; must not be silently claimed solved
    assert np.all(np.isfinite(new_geom.coords))
    # displacement should still be modest, not a blow-up/divergence
    assert np.linalg.norm(new_geom.coords - geom.coords) < 1.0

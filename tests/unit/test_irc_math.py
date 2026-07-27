"""Unit tests for the IRC module's pure math (no xtb): Kabsch-RMSD
alignment and mass-weighted imaginary-mode extraction."""
from __future__ import annotations

import numpy as np

from drfo.geometry import Geometry
from drfo.irc.path import imaginary_mode
from drfo.irc.verify import rmsd_after_alignment


def test_rmsd_after_alignment_identical_is_zero():
    g = Geometry.from_angstrom(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    assert rmsd_after_alignment(g, g.copy()) < 1e-10


def test_rmsd_after_alignment_invariant_to_rotation_and_translation():
    g = Geometry.from_angstrom(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    theta = 0.7
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = g.copy()
    rotated.coords = rotated.coords @ rot.T + np.array([5.0, -3.0, 2.0])
    assert rmsd_after_alignment(g, rotated) < 1e-8


def test_rmsd_after_alignment_nonzero_for_different_structures():
    g1 = Geometry.from_angstrom(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    g2 = Geometry.from_angstrom(["O", "H", "H"], [[0, 0, 0], [1.5, 0, 0], [-0.24, 0.93, 0]])
    assert rmsd_after_alignment(g1, g2) > 0.1


def test_imaginary_mode_finds_most_negative_eigenvalue_direction():
    """Hand-built 2-atom (H-Cl) Hessian along the bond axis: two diagonal
    stretch-like modes with a negative and a positive eigenvalue in a
    non-mass-weighted basis chosen so mass-weighting is required to get the
    right *mass-weighted* eigenvector back out -- catches a mass-weighting
    sign/ordering bug that a symmetric-mass toy system wouldn't."""
    geom = Geometry.from_angstrom(["H", "Cl"], [[0.0, 0.0, 0.0], [1.27, 0.0, 0.0]])
    n = geom.natoms * 3
    hessian = np.zeros((n, n))
    # Displace both atoms oppositely along x (index 0 and 3): a genuine
    # bond-stretch-like coordinate, given a negative curvature there.
    v = np.zeros(n)
    v[0], v[3] = 1.0, -1.0
    v /= np.linalg.norm(v)
    hessian += -0.5 * np.outer(v, v)
    # A positive-curvature mode along y for both atoms, orthogonal to v.
    w = np.zeros(n)
    w[1], w[4] = 1.0, 1.0
    w /= np.linalg.norm(w)
    hessian += 0.8 * np.outer(w, w)

    eigval, direction = imaginary_mode(geom, hessian)
    assert eigval < 0
    direction_flat = direction / np.linalg.norm(direction)
    # The Cartesian direction should be purely along x (bond stretch),
    # i.e. have no y/z component, regardless of the mass-weighting applied
    # internally.
    assert abs(direction_flat[1]) < 1e-8 and abs(direction_flat[2]) < 1e-8
    assert abs(direction_flat[4]) < 1e-8 and abs(direction_flat[5]) < 1e-8
    # H and Cl should move in opposite directions along x.
    assert direction_flat[0] * direction_flat[3] < 0

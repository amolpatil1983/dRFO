"""Redundant internal coordinate primitives and the coordinate-system
container. Stretches, bends, torsions, and an out-of-plane (Wilson wag)
coordinate for trigonal (3-connected) centers, which describes
pyramidalization at a planar center -- the one degree of freedom stretches
and bends cannot see and that plain torsions become numerically singular
trying to approximate (see `_TORSION_PLANARITY_EPS`). A dedicated LinearBend
coordinate for near-linear angles remains a documented follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np

from ..geometry import Geometry
from .topology import BondGraph


class Stretch(NamedTuple):
    i: int
    j: int


class Bend(NamedTuple):
    i: int
    j: int
    k: int  # angle vertex is j


class Torsion(NamedTuple):
    i: int
    j: int
    k: int
    l: int  # dihedral about the j-k bond


class OutOfPlane(NamedTuple):
    a: int  # center (trigonal atom)
    b: int  # plane atom
    c: int  # plane atom
    d: int  # wagging atom, displaced out of the a-b-c plane


InternalCoord = Union[Stretch, Bend, Torsion, OutOfPlane]

# Angles closer to 0/pi than this are considered numerically unsafe for the
# plain Bend/Torsion Wilson formulas (near-linear angle handling -- e.g. a
# dedicated LinearBend coordinate -- is deferred; this guard only prevents
# silent NaN propagation before that lands).
_ANGLE_SINGULARITY_EPS = 1e-3

# Torsion (dihedral) values closer to 0/pi than this are excluded: the
# atan2-based torsion formula's B-matrix row becomes severely
# ill-conditioned near the periodic branch point, which is exactly where
# planar/symmetric molecules (e.g. formaldehyde) sit -- a dedicated
# OutOfPlane coordinate (deferred, like LinearBend) would describe such
# geometries properly; this guard avoids building a torsion coordinate
# that would make the Wilson B-matrix numerically singular.
_TORSION_PLANARITY_EPS = 0.1


def _stretch_value(geom: Geometry, c: Stretch) -> float:
    return float(np.linalg.norm(geom.coords[c.i] - geom.coords[c.j]))


def _bend_value(geom: Geometry, c: Bend) -> float:
    u = geom.coords[c.i] - geom.coords[c.j]
    v = geom.coords[c.k] - geom.coords[c.j]
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _torsion_value(geom: Geometry, c: Torsion) -> float:
    b1 = geom.coords[c.j] - geom.coords[c.i]
    b2 = geom.coords[c.k] - geom.coords[c.j]
    b3 = geom.coords[c.l] - geom.coords[c.k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def _oop_value(geom: Geometry, c: OutOfPlane) -> float:
    """Wilson out-of-plane angle: the angle by which atom d is displaced
    out of the plane spanned by center a and its two other neighbors b, c.
    Zero for a perfectly planar (e.g. trigonal sp2) center; nonzero as it
    pyramidalizes."""
    a, b, cc, d = c.a, c.b, c.c, c.d
    rab = geom.coords[b] - geom.coords[a]
    rac = geom.coords[cc] - geom.coords[a]
    rad = geom.coords[d] - geom.coords[a]
    eab, eac, ead = rab / np.linalg.norm(rab), rac / np.linalg.norm(rac), rad / np.linalg.norm(rad)

    cos_bac = np.clip(np.dot(eab, eac), -1.0, 1.0)
    sin_bac = np.sqrt(max(1.0 - cos_bac**2, 1e-12))

    cross = np.cross(eab, eac)
    sin_gamma = np.clip(np.dot(cross, ead) / sin_bac, -1.0, 1.0)
    return float(np.arcsin(sin_gamma))


def is_angle_near_singular(geom: Geometry, i: int, j: int, k: int) -> bool:
    u = geom.coords[i] - geom.coords[j]
    v = geom.coords[k] - geom.coords[j]
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle < _ANGLE_SINGULARITY_EPS or angle > np.pi - _ANGLE_SINGULARITY_EPS


def is_torsion_near_singular(geom: Geometry, i: int, j: int, k: int, l: int) -> bool:
    phi = abs(_torsion_value(geom, Torsion(i, j, k, l)))
    return phi < _TORSION_PLANARITY_EPS or phi > np.pi - _TORSION_PLANARITY_EPS


@dataclass
class InternalCoordinateSystem:
    coords: list[InternalCoord]
    natoms: int

    def __len__(self) -> int:
        return len(self.coords)

    def values(self, geom: Geometry) -> np.ndarray:
        out = np.empty(len(self.coords))
        for idx, c in enumerate(self.coords):
            if isinstance(c, Stretch):
                out[idx] = _stretch_value(geom, c)
            elif isinstance(c, Bend):
                out[idx] = _bend_value(geom, c)
            elif isinstance(c, Torsion):
                out[idx] = _torsion_value(geom, c)
            elif isinstance(c, OutOfPlane):
                out[idx] = _oop_value(geom, c)
            else:
                raise TypeError(f"unknown coordinate type {type(c)}")
        return out

    def index_of(self, coord: InternalCoord) -> int:
        return self.coords.index(coord)

    def stretch_index(self, i: int, j: int) -> int | None:
        a, b = (i, j) if i < j else (j, i)
        for idx, c in enumerate(self.coords):
            if isinstance(c, Stretch) and c.i == a and c.j == b:
                return idx
        return None

    def B(self, geom: Geometry) -> np.ndarray:
        from .bmatrix import build_B  # local import to avoid a cycle
        return build_B(self, geom)


def build_coordinate_system(
    graph: BondGraph, geom_reactant: Geometry, geom_product: Geometry,
) -> InternalCoordinateSystem:
    """Stretch for every bond in the (already-merged) graph; Bend for every
    pair of stretches sharing an atom; Torsion for every pair of bends
    sharing a central bond. A Bend/Torsion is only included if it is not
    numerically near-singular (near-linear angle) in *either* endpoint
    geometry, matching the paper's requirement that the coordinate set be
    valid for both the reactant and the product.
    """
    natoms = geom_reactant.natoms
    bonds = sorted(graph.bonds)
    coords: list[InternalCoord] = [Stretch(i, j) for i, j in bonds]

    # bonds incident to each atom, for building bends/torsions
    incident: dict[int, list[tuple[int, int]]] = {}
    for i, j in bonds:
        incident.setdefault(i, []).append((i, j))
        incident.setdefault(j, []).append((j, i))

    bend_set: set[tuple[int, int, int]] = set()
    for vertex, edges in incident.items():
        others = [nb for (_, nb) in edges]
        for a in range(len(others)):
            for b in range(a + 1, len(others)):
                i, k = others[a], others[b]
                if i == k:
                    continue
                key = (i, vertex, k) if i < k else (k, vertex, i)
                if key in bend_set:
                    continue
                if is_angle_near_singular(geom_reactant, i, vertex, k):
                    continue
                if is_angle_near_singular(geom_product, i, vertex, k):
                    continue
                bend_set.add(key)
                coords.append(Bend(i, vertex, k))

    # out-of-plane: one Wilson wag coordinate per trigonal (exactly
    # 3-connected) center, describing the pyramidalization DOF that
    # stretches/bends cannot see and that a torsion approximates only
    # numerically-unstably right at the planar point (see
    # _TORSION_PLANARITY_EPS). Guarded the same way as a Bend: skipped if
    # the b-a-c reference angle is near-singular in either endpoint.
    for vertex, edges in incident.items():
        neighbors = sorted(nb for (_, nb) in edges)
        if len(neighbors) != 3:
            continue
        b, cc, d = neighbors
        if is_angle_near_singular(geom_reactant, b, vertex, cc):
            continue
        if is_angle_near_singular(geom_product, b, vertex, cc):
            continue
        coords.append(OutOfPlane(vertex, b, cc, d))

    # torsions: pairs of bends sharing a central bond (j,k), i.e. Bend(i,j,k)
    # and Bend(j,k,l) style adjacency -- build from bonds directly.
    torsion_set: set[tuple[int, int, int, int]] = set()
    for j, k in bonds:
        i_candidates = [nb for (_, nb) in incident.get(j, []) if nb != k]
        l_candidates = [nb for (_, nb) in incident.get(k, []) if nb != j]
        for i in i_candidates:
            for l in l_candidates:
                if i == l:
                    continue
                if is_angle_near_singular(geom_reactant, i, j, k):
                    continue
                if is_angle_near_singular(geom_reactant, j, k, l):
                    continue
                if is_angle_near_singular(geom_product, i, j, k):
                    continue
                if is_angle_near_singular(geom_product, j, k, l):
                    continue
                if is_torsion_near_singular(geom_reactant, i, j, k, l):
                    continue
                if is_torsion_near_singular(geom_product, i, j, k, l):
                    continue
                key = (i, j, k, l)
                rev = (l, k, j, i)
                if key in torsion_set or rev in torsion_set:
                    continue
                torsion_set.add(key)
                coords.append(Torsion(i, j, k, l))

    return InternalCoordinateSystem(coords=coords, natoms=natoms)

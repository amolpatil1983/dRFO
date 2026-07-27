"""Bonding-graph construction: the basis for building a redundant internal
coordinate set. Bonds are detected purely by distance against a scaled sum
of covalent radii -- no valence/hybridization logic, matching the paper's
own "standard approach" description.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..elements import covalent_radius
from ..geometry import Geometry


@dataclass
class BondGraph:
    bonds: set[tuple[int, int]] = field(default_factory=set)  # (i, j), i < j
    virtual_bonds: set[tuple[int, int]] = field(default_factory=set)  # subset of bonds

    def fragments(self, natoms: int) -> list[frozenset[int]]:
        """Connected components under `bonds` (real + virtual)."""
        parent = list(range(natoms))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j in self.bonds:
            union(i, j)

        groups: dict[int, set[int]] = {}
        for atom in range(natoms):
            groups.setdefault(find(atom), set()).add(atom)
        return [frozenset(g) for g in groups.values()]


def build_bond_graph(geom: Geometry, scale: float = 1.4) -> BondGraph:
    """Two atoms are bonded if their distance is less than
    `scale * (cov_radius[i] + cov_radius[j])`."""
    n = geom.natoms
    bonds: set[tuple[int, int]] = set()
    for i in range(n):
        ri = covalent_radius(geom.symbols[i])
        for j in range(i + 1, n):
            rj = covalent_radius(geom.symbols[j])
            dist = float(np.linalg.norm(geom.coords[i] - geom.coords[j]))
            if dist < scale * (ri + rj):
                bonds.add((i, j))
    return BondGraph(bonds=bonds)


def merge_bond_graphs(g1: BondGraph, g2: BondGraph) -> BondGraph:
    return BondGraph(bonds=g1.bonds | g2.bonds, virtual_bonds=g1.virtual_bonds | g2.virtual_bonds)


def add_virtual_interfragment_bonds(geom: Geometry, graph: BondGraph) -> BondGraph:
    """For every pair of disconnected fragments, add the shortest-distance
    atom pair between them as an extra "virtual" bond/stretch coordinate,
    so multi-molecular systems (e.g. H2 + CO) get a coordinate connecting
    all fragments. Repeats until only one fragment remains (handles more
    than two fragments)."""
    bonds = set(graph.bonds)
    virtual = set(graph.virtual_bonds)
    n = geom.natoms

    while True:
        frags = BondGraph(bonds=bonds).fragments(n)
        if len(frags) <= 1:
            break
        frag_list = sorted(frags, key=lambda f: min(f))
        a, b = frag_list[0], frag_list[1]
        best: tuple[float, tuple[int, int]] | None = None
        for i in a:
            for j in b:
                dist = float(np.linalg.norm(geom.coords[i] - geom.coords[j]))
                if best is None or dist < best[0]:
                    best = (dist, (min(i, j), max(i, j)))
        assert best is not None
        bonds.add(best[1])
        virtual.add(best[1])

    return BondGraph(bonds=bonds, virtual_bonds=virtual)


def diff_bonds(
    reactant_graph: BondGraph, product_graph: BondGraph,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Returns (breaking, forming): bonds present only in the reactant vs.
    only in the product."""
    breaking = reactant_graph.bonds - product_graph.bonds
    forming = product_graph.bonds - reactant_graph.bonds
    return breaking, forming

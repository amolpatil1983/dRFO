import numpy as np

from drfo.geometry import Geometry
from drfo.internal.topology import (
    add_virtual_interfragment_bonds,
    build_bond_graph,
    diff_bonds,
    merge_bond_graphs,
)


def water() -> Geometry:
    return Geometry.from_angstrom(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )


def h2_plus_co() -> Geometry:
    # Two separate fragments: CO and H2, well separated. Atom ordering
    # (C, O, H, H) deliberately matches h2co()'s so index i always refers
    # to the same physical atom across reactant/product -- required for
    # diff_bonds to be meaningful (comparing bond graphs across structures
    # with inconsistent atom ordering is not physically meaningful).
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[5.0, 0.0, 0.0], [5.0, 0.0, 1.13], [0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )


def h2co() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.22], [0.94, 0.0, -0.54], [-0.94, 0.0, -0.54]],
    )


def test_build_bond_graph_water_has_two_OH_bonds():
    graph = build_bond_graph(water())
    assert graph.bonds == {(0, 1), (0, 2)}


def test_build_bond_graph_no_spurious_HH_bond_in_water():
    # H...H distance in water (~1.5 A) should not be misidentified as bonded.
    graph = build_bond_graph(water())
    assert (1, 2) not in graph.bonds


def test_fragments_single_molecule():
    graph = build_bond_graph(water())
    frags = graph.fragments(3)
    assert len(frags) == 1
    assert frags[0] == frozenset({0, 1, 2})


def test_fragments_two_separate_molecules():
    geom = h2_plus_co()
    graph = build_bond_graph(geom)
    frags = graph.fragments(geom.natoms)
    assert len(frags) == 2
    assert frozenset({0, 1}) in frags  # CO
    assert frozenset({2, 3}) in frags  # H2


def test_add_virtual_interfragment_bonds_connects_fragments():
    geom = h2_plus_co()
    graph = build_bond_graph(geom)
    assert len(graph.fragments(geom.natoms)) == 2

    merged = add_virtual_interfragment_bonds(geom, graph)
    assert len(merged.fragments(geom.natoms)) == 1
    assert len(merged.virtual_bonds) == 1
    # the virtual bond should be the closest atom pair between fragments:
    # CO at x=5 (atoms 0,1), H2 at x=0 (atoms 2,3) -> closest pair is
    # atom 0 (C, x=5,z=0) to atom 2 (H, x=0,z=0), distance 5.0 exactly.
    vbond = next(iter(merged.virtual_bonds))
    assert vbond == (0, 2)


def test_diff_bonds_identifies_breaking_and_forming():
    # H2CO (reactant-like) vs H2+CO (product-like), consistent atom
    # ordering (C=0,O=1,H=2,H=3) in both: the two C-H bonds break, and the
    # H-H bond forms; C-O persists in both so it's absent from both sets.
    reactant = build_bond_graph(h2co())
    product = build_bond_graph(h2_plus_co())

    breaking, forming = diff_bonds(reactant, product)
    assert breaking == {(0, 2), (0, 3)}  # C-H, C-H
    assert forming == {(2, 3)}  # H-H


def test_merge_bond_graphs_is_union():
    reactant = build_bond_graph(h2co())
    product = build_bond_graph(h2_plus_co())
    merged = merge_bond_graphs(reactant, product)
    assert merged.bonds == reactant.bonds | product.bonds

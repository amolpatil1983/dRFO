import numpy as np

from drfo.geometry import Geometry
from drfo.internal.coordinates import Bend, Stretch, Torsion, build_coordinate_system
from drfo.internal.topology import (
    add_virtual_interfragment_bonds,
    build_bond_graph,
    merge_bond_graphs,
)


def h2co() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.22], [0.94, 0.0, -0.54], [-0.94, 0.0, -0.54]],
    )


def h2_plus_co() -> Geometry:
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [[5.0, 0.0, 0.0], [5.0, 0.0, 1.13], [0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )


def test_build_coordinate_system_h2co_reaction_set():
    reactant, product = h2co(), h2_plus_co()
    g_r = build_bond_graph(reactant)
    g_p = build_bond_graph(product)
    merged = merge_bond_graphs(g_r, g_p)
    merged = add_virtual_interfragment_bonds(product, merged)

    ics = build_coordinate_system(merged, reactant, product)
    stretches = {(c.i, c.j) for c in ics.coords if isinstance(c, Stretch)}

    # union bonds: C-O (persists), C-H, C-H (reactant only), H-H (product
    # only, from build_bond_graph on the product), plus whatever virtual
    # bond add_virtual_interfragment_bonds contributed on the *product*
    # topology (won't appear here since it's built against the reactant
    # geometry not being fragmented -- merged already connects everything
    # via the real bonds' union, so no virtual bond should be needed after
    # the union itself already connects all atoms).
    assert (0, 1) in stretches  # C-O
    assert (0, 2) in stretches  # C-H
    assert (0, 3) in stretches  # C-H
    assert (2, 3) in stretches  # H-H (product)

    # Bends should include the reactant's H-C-H and H-C-O angles (all
    # well-defined, non-linear at both endpoints).
    bends = {(c.i, c.j, c.k) for c in ics.coords if isinstance(c, Bend)}
    assert any(b[1] == 0 for b in bends)  # some bend centered on C


def test_build_coordinate_system_values_match_geometry():
    reactant, product = h2co(), h2_plus_co()
    merged = merge_bond_graphs(build_bond_graph(reactant), build_bond_graph(product))
    ics = build_coordinate_system(merged, reactant, product)

    vals = ics.values(reactant)
    assert len(vals) == len(ics.coords)
    assert np.all(np.isfinite(vals))
    for c, v in zip(ics.coords, vals):
        if isinstance(c, Stretch):
            assert v > 0
        if isinstance(c, Bend):
            assert 0 < v < np.pi

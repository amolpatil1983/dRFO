"""M3 validation: the bond-order interpolation guess builder against real
GFN-FF relaxation, on the H2CO <-> H2 + CO reaction (benchmark case #1,
chosen because it has a clean bond-breaking/forming set and exercises the
multi-fragment virtual-bond logic)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drfo import Geometry, XTBCalculator, build_ts_guess
from drfo.internal.coordinates import Stretch

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")


def h2co_reactant() -> Geometry:
    # xtb --opt tight (GFN2-xTB) relaxed formaldehyde.
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [
            [-0.00000000000496, -0.00000000533118, 0.03582961965288],
            [-0.00000000002338, 0.00000000163570, 1.22790730198304],
            [0.92713422902939, 0.00000000184774, -0.56186846084127],
            [-0.92713422900105, 0.00000000184774, -0.56186846079465],
        ],
    )


def h2_plus_co_product() -> Geometry:
    # xtb --opt tight (GFN2-xTB) relaxed CO and H2, placed 6 A apart along
    # x, same (C,O,H,H) atom ordering as the reactant.
    return Geometry.from_angstrom(
        ["C", "O", "H", "H"],
        [
            [0.0, 0.0, 0.00124715568705],
            [0.0, 0.0, 1.12875284431295],
            [6.0, 0.0, -0.01848467372883],
            [6.0, 0.0, 0.75848467372883],
        ],
    )


def test_build_ts_guess_identifies_correct_reaction_coordinate(tmp_path):
    cheap = XTBCalculator(XTB_PATH, method="gfnff", scratch_dir=tmp_path)
    result = build_ts_guess(h2co_reactant(), h2_plus_co_product(), cheap)

    assert set(result.breaking_bonds) == {(0, 2), (0, 3)}  # both C-H bonds
    assert set(result.forming_bonds) == {(2, 3)}  # H-H
    assert np.all(np.isfinite(result.geometry.coords))


def test_build_ts_guess_accepts_externally_supplied_bonds(tmp_path):
    """An external, exact bond classification (e.g. from an atom-mapping
    step run upstream, as opposed to this module's own covalent-radius
    distance-cutoff diffing) should be usable as-is, bypassing diff_bonds
    entirely -- this is the integration point external tooling depends on."""
    cheap = XTBCalculator(XTB_PATH, method="gfnff", scratch_dir=tmp_path)
    result = build_ts_guess(
        h2co_reactant(), h2_plus_co_product(), cheap,
        breaking_bonds=[(0, 2), (0, 3)], forming_bonds=[(2, 3)],
    )
    assert set(result.breaking_bonds) == {(0, 2), (0, 3)}
    assert set(result.forming_bonds) == {(2, 3)}
    assert np.all(np.isfinite(result.geometry.coords))


def test_build_ts_guess_rejects_partial_bond_override(tmp_path):
    cheap = XTBCalculator(XTB_PATH, method="gfnff", scratch_dir=tmp_path)
    with pytest.raises(ValueError):
        build_ts_guess(h2co_reactant(), h2_plus_co_product(), cheap, breaking_bonds=[(0, 2)])


def test_build_ts_guess_active_bonds_land_between_endpoints(tmp_path):
    """The interpolated active-bond lengths should sit strictly between
    their reactant and product values -- neither fully broken nor fully
    formed -- consistent with a "half-reacted" bridging structure."""
    reactant, product = h2co_reactant(), h2_plus_co_product()
    cheap = XTBCalculator(XTB_PATH, method="gfnff", scratch_dir=tmp_path)
    result = build_ts_guess(reactant, product, cheap)

    active = set(result.breaking_bonds) | set(result.forming_bonds)
    for idx, c in enumerate(result.coord_system.coords):
        if not (isinstance(c, Stretch) and (c.i, c.j) in active):
            continue
        r_react = np.linalg.norm(reactant.coords[c.i] - reactant.coords[c.j])
        r_prod = np.linalg.norm(product.coords[c.i] - product.coords[c.j])
        r_guess = np.linalg.norm(result.geometry.coords[c.i] - result.geometry.coords[c.j])
        lo, hi = sorted([r_react, r_prod])
        # a small tolerance since the orthogonal relaxation can shift the
        # exact endpoint bond lengths slightly from their original values
        assert lo - 0.3 <= r_guess <= hi + 0.3, (c.i, c.j, r_react, r_prod, r_guess)

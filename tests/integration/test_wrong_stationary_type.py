"""Regression test: run_drfo_search must not report `status == "converged"`
for a gradient-converged point that isn't actually a first-order saddle
(n_imaginary != 1) -- surfaced by a real case (electrocyclic ring closure,
e.g. butadiene -> cyclobutene) where a bond-length-only guide vector has no
rotational component to represent the true reaction coordinate, so the
search just slides downhill to a minimum and gradient-converges there.

Reproduced here standalone (no Schlegel guess-building, no dependency on
that specific reaction converging a particular way) by handing
run_drfo_search a guide vector that carries no real transition-vector
information, starting near HCN's own minimum -- cheap (3 atoms) and
deterministic, isolating the status-classification logic itself from any
particular reaction's guess-building behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drfo import XTBCalculator
from drfo.internal.coordinates import build_coordinate_system
from drfo.internal.topology import add_virtual_interfragment_bonds, build_bond_graph
from drfo.internal.transform import b_pinv
from drfo.io.xyz import read_xyz
from drfo.optimize.search import run_drfo_search

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")

STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "structures"


def test_gradient_converged_minimum_is_not_reported_as_converged(tmp_path):
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)

    graph = add_virtual_interfragment_bonds(hcn, build_bond_graph(hcn))
    ics = build_coordinate_system(graph, hcn, hcn)
    B = ics.B(hcn)
    H_int = b_pinv(B).T @ calc.hessian(hcn) @ b_pinv(B)

    # Start slightly off HCN's own minimum, with a guide vector carrying no
    # real transition-vector information -- the search has nothing to guide
    # it toward a saddle, so it just relaxes straight back to the minimum.
    perturbed = hcn.copy()
    perturbed.coords = perturbed.coords + 0.05

    def bogus_guide_vector(B_current: np.ndarray) -> np.ndarray:
        v = np.zeros(B_current.shape[0])
        v[0] = 1.0
        return v

    result = run_drfo_search(perturbed, ics, H_int, bogus_guide_vector, calc, max_ts_steps=60)

    assert result.n_imaginary == 0
    assert result.status == "wrong_stationary_type"
    assert result.converged is False

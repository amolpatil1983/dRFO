"""Direct calculator-layer smoke tests against the real xtb binary (both
GFN2-xTB and GFN-FF), covering the file-format-parsing assumptions
documented in `drfo/calculators/xtb.py`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drfo import Geometry
from drfo.calculators import CalculatorError, XTBCalculator

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")


def _water() -> Geometry:
    return Geometry.from_angstrom(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )


@pytest.mark.parametrize("method", ["gfn2", "gfnff"])
def test_energy_gradient_hessian(tmp_path, method):
    calc = XTBCalculator(XTB_PATH, method=method, scratch_dir=tmp_path)
    water = _water()

    e = calc.energy(water)
    assert np.isfinite(e)

    g = calc.gradient(water)
    assert g.shape == (3, 3)
    assert np.all(np.isfinite(g))

    h = calc.hessian(water)
    assert h.shape == (9, 9)
    assert np.allclose(h, h.T)


def test_combined_gradient_and_hessian_call(tmp_path):
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = calc.compute(_water(), gradient=True, hessian=True)
    assert result.gradient is not None
    assert result.hessian is not None
    assert result.extra is not None
    assert "frequencies_rcm" in result.extra


def test_charge_spin_mismatch_raises_value_error(tmp_path):
    calc = XTBCalculator(XTB_PATH, method="gfn2", charge=0, scratch_dir=tmp_path)
    charged = Geometry.from_angstrom(_water().symbols, _water().coords_angstrom(), charge=1)
    with pytest.raises(ValueError):
        calc.energy(charged)


def test_degenerate_geometry_raises_calculator_error(tmp_path):
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    bad = Geometry(["O", "H", "H"], np.zeros((3, 3)))
    with pytest.raises(CalculatorError):
        calc.energy(bad)


def test_missing_binary_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        XTBCalculator("/nonexistent/xtb")

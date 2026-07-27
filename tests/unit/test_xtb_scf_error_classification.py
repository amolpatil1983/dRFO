"""Verifies the SCF-non-convergence detection in XTBCalculator._run without
depending on being able to reliably reproduce a genuine xtb SCF failure
(that class of failure is inherently geometry/method-sensitive and fragile
to pin down as a fixture). A mocked subprocess.run reproducing xtb's actual
output text is deterministic and fast.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from drfo import Geometry, XTBCalculator
from drfo.calculators import CalculatorError, SCFConvergenceError

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")

# The exact phrasing xtb 6.6.1 emits on SCC non-convergence (confirmed
# empirically): "-1- scf: Self consistent charge iterator did not converge".
_SCF_FAILURE_STDOUT = """
########################################################################
[ERROR] Program stopped due to fatal error
-3- Single point calculation terminated
-2- xtb_calculator_singlepoint: Electronic structure method terminated
-1- scf: Self consistent charge iterator did not converge
########################################################################
"""

_OTHER_FAILURE_STDOUT = "some unrelated fatal error banner"


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _water() -> Geometry:
    return Geometry.from_angstrom(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )


def test_scf_non_convergence_raises_specific_subclass(tmp_path):
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    with patch(
        "drfo.calculators.xtb.subprocess.run",
        return_value=_FakeCompletedProcess(128, _SCF_FAILURE_STDOUT, "abnormal termination of xtb"),
    ):
        with pytest.raises(SCFConvergenceError) as excinfo:
            calc.energy(_water())
    # SCFConvergenceError must remain catchable as a plain CalculatorError
    # for callers that don't need the distinction.
    assert isinstance(excinfo.value, CalculatorError)


def test_other_failures_raise_plain_calculator_error_not_scf_subclass(tmp_path):
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    with patch(
        "drfo.calculators.xtb.subprocess.run",
        return_value=_FakeCompletedProcess(1, _OTHER_FAILURE_STDOUT, "abnormal termination of xtb"),
    ):
        with pytest.raises(CalculatorError) as excinfo:
            calc.energy(_water())
    assert not isinstance(excinfo.value, SCFConvergenceError)

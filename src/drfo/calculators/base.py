from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..geometry import Geometry


class CalculatorError(RuntimeError):
    """Raised when a backend calculator fails (non-convergence, abnormal termination, ...)."""


class SCFConvergenceError(CalculatorError):
    """Raised specifically when the electronic-structure SCF/SCC iterator
    fails to converge at a given geometry. This is a distinct subclass
    (rather than a plain CalculatorError) so callers -- and anyone reading
    logs -- can tell "the reference method genuinely struggled with this
    geometry's electronic structure" apart from other failure modes
    (crashed subprocess, malformed input, degenerate/overlapping atoms,
    missing binary, ...). It is a real, sometimes-expected outcome when
    exploring geometries far from equilibrium (e.g. along an interpolated
    reaction path near a near-degenerate/multi-reference-character point),
    not necessarily evidence of a bug in the optimizer driving it there."""


@dataclass(frozen=True)
class CalcResult:
    energy: float  # Hartree
    gradient: np.ndarray | None = None  # (N, 3) Hartree/bohr
    hessian: np.ndarray | None = None  # (3N, 3N) Hartree/bohr^2
    extra: dict | None = None  # backend-specific bonus info (e.g. xtb's own frequency list)


class Calculator(abc.ABC):
    """Backend-agnostic energy/gradient/Hessian source.

    Subclasses implement a single `compute()` entry point so a backend can
    batch an energy+gradient call into one underlying invocation when that is
    cheaper (e.g. one xtb subprocess call rather than two).
    """

    @abc.abstractmethod
    def compute(self, geom: Geometry, *, gradient: bool = False,
                hessian: bool = False) -> CalcResult:
        raise NotImplementedError

    def energy(self, geom: Geometry) -> float:
        return self.compute(geom).energy

    def gradient(self, geom: Geometry) -> np.ndarray:
        result = self.compute(geom, gradient=True)
        assert result.gradient is not None
        return result.gradient

    def hessian(self, geom: Geometry) -> np.ndarray:
        result = self.compute(geom, hessian=True)
        assert result.hessian is not None
        return result.hessian

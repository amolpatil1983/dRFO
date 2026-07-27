from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceCriteria:
    rms_grad: float = 3.2e-4  # hartree/bohr, matches the paper's constrained-relax threshold
    max_grad: float = 4.5e-4
    rms_step: float = 1.8e-3
    max_step: float = 2.5e-3


def check_converged(g: np.ndarray, dq: np.ndarray, crit: ConvergenceCriteria) -> bool:
    rms_g = float(np.sqrt(np.mean(g**2)))
    max_g = float(np.max(np.abs(g)))
    rms_s = float(np.sqrt(np.mean(dq**2)))
    max_s = float(np.max(np.abs(dq)))
    return (rms_g < crit.rms_grad and max_g < crit.max_grad
            and rms_s < crit.rms_step and max_s < crit.max_step)


class TrustRadiusController:
    """Simple trust-radius controller: grow on consecutive good steps (energy
    went down / gradient norm decreased), shrink on a bad step."""

    def __init__(self, initial: float = 0.1, lower: float = 1e-3, upper: float = 0.3,
                 grow: float = 1.2, shrink: float = 0.5):
        self.radius = initial
        self.lower = lower
        self.upper = upper
        self.grow = grow
        self.shrink = shrink
        self._good_streak = 0

    def scale(self, dq: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(dq)
        if norm > self.radius:
            return dq * (self.radius / norm)
        return dq

    def report(self, good: bool) -> None:
        if good:
            self._good_streak += 1
            if self._good_streak >= 2:
                self.radius = min(self.upper, self.radius * self.grow)
                self._good_streak = 0
        else:
            self._good_streak = 0
            self.radius = max(self.lower, self.radius * self.shrink)

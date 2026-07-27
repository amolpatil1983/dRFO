"""Core Geometry data structure. All internal math uses atomic units (bohr, Hartree)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BOHR_PER_ANGSTROM = 1.8897259886
AMU_PER_ELECTRON_MASS = 1822.888486209  # atomic mass unit, in atomic units (m_e = 1)


@dataclass
class Geometry:
    symbols: list[str]
    coords: np.ndarray  # (N, 3) bohr
    charge: int = 0
    spin: int = 0  # number of unpaired electrons (xtb's --uhf)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(len(self.symbols), 3)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    def copy(self) -> "Geometry":
        return Geometry(list(self.symbols), self.coords.copy(), self.charge, self.spin)

    @classmethod
    def from_angstrom(cls, symbols: list[str], coords_angstrom: np.ndarray,
                       charge: int = 0, spin: int = 0) -> "Geometry":
        coords = np.asarray(coords_angstrom, dtype=float) * BOHR_PER_ANGSTROM
        return cls(list(symbols), coords, charge, spin)

    def coords_angstrom(self) -> np.ndarray:
        return self.coords / BOHR_PER_ANGSTROM

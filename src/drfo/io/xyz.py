from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import BOHR_PER_ANGSTROM, Geometry


def write_xyz(path: str | Path, geom: Geometry, comment: str = "") -> None:
    coords_ang = geom.coords_angstrom()
    lines = [str(geom.natoms), comment]
    for sym, (x, y, z) in zip(geom.symbols, coords_ang):
        lines.append(f"{sym:<3s} {x:20.12f} {y:20.12f} {z:20.12f}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_xyz(path: str | Path, charge: int = 0, spin: int = 0) -> Geometry:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords = np.zeros((n, 3))
    for i in range(n):
        parts = lines[2 + i].split()
        symbols.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return Geometry.from_angstrom(symbols, coords, charge=charge, spin=spin)

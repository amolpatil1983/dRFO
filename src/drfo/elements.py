"""Element reference data: covalent radii and single-bond reference lengths.

Covalent radii are the single-bond values from:
    P. Pyykko, M. Atsumi, Chem. Eur. J. 2009, 15, 186 / and the widely used
    Cordero et al. covalent radii table (B. Cordero et al., Dalton Trans. 2008, 2832),
    a standard open dataset redistributed across quantum-chemistry software.
Values below are in Angstrom in the source table; converted to bohr at import time.
"""
from __future__ import annotations

from .geometry import BOHR_PER_ANGSTROM

# Cordero et al. 2008 single-bond covalent radii, Angstrom. Subset covering
# main-group elements relevant to organic/main-group reaction chemistry
# (the paper explicitly excludes transition-metal reactions).
_COVALENT_RADII_ANGSTROM: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76,
    "Br": 1.20, "I": 1.39,
}

COVALENT_RADII: dict[str, float] = {
    sym: r * BOHR_PER_ANGSTROM for sym, r in _COVALENT_RADII_ANGSTROM.items()
}

# Element-pair single-bond reference lengths (r0) that override the covalent-radii-sum
# default when a more accurate literature single-bond value is known, for the pairs
# expected in the benchmark reaction set. Angstrom in source, converted to bohr.
_REFERENCE_BOND_LENGTHS_ANGSTROM: dict[tuple[str, str], float] = {
    ("C", "H"): 1.09, ("H", "C"): 1.09,
    ("C", "C"): 1.54,
    ("C", "N"): 1.47, ("N", "C"): 1.47,
    ("C", "O"): 1.43, ("O", "C"): 1.43,
    ("C", "F"): 1.35, ("F", "C"): 1.35,
    ("C", "Cl"): 1.77, ("Cl", "C"): 1.77,
    ("H", "H"): 0.74,
    ("H", "N"): 1.01, ("N", "H"): 1.01,
    ("H", "O"): 0.96, ("O", "H"): 0.96,
    ("H", "F"): 0.92, ("F", "H"): 0.92,
    ("N", "N"): 1.45,
    ("N", "O"): 1.40, ("O", "N"): 1.40,
    ("O", "O"): 1.48,
    ("Si", "H"): 1.48, ("H", "Si"): 1.48,
    ("Si", "Si"): 2.33,
    ("S", "O"): 1.57, ("O", "S"): 1.57,
    ("S", "S"): 2.05,
}

_REFERENCE_BOND_LENGTHS: dict[tuple[str, str], float] = {
    pair: r * BOHR_PER_ANGSTROM for pair, r in _REFERENCE_BOND_LENGTHS_ANGSTROM.items()
}


def covalent_radius(symbol: str) -> float:
    """Covalent radius in bohr. Raises KeyError for unsupported elements."""
    return COVALENT_RADII[symbol]


def reference_bond_length(symbol_a: str, symbol_b: str) -> float:
    """Reference single-bond length r0 (bohr) for a pair of elements.

    Uses a tabulated literature value when available, otherwise falls back
    to the sum of covalent radii.
    """
    pair = (symbol_a, symbol_b)
    if pair in _REFERENCE_BOND_LENGTHS:
        return _REFERENCE_BOND_LENGTHS[pair]
    return covalent_radius(symbol_a) + covalent_radius(symbol_b)

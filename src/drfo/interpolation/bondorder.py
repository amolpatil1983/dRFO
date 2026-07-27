"""Bond-order/length interpolation model (paper eq. 3): decays to exactly
zero at twice the reference bond length (b=2 default), unlike the Pauling
bond order (eq. 1-2) which decays slowly to infinity -- better suited for
interpolating bonds that are partially broken/formed at a transition state.
"""
from __future__ import annotations


def bond_order(r: float, r0: float, b: float = 2.0) -> float:
    """o~(r) = max(0, (b*(r0/r) - 1) / (b - 1))."""
    return max(0.0, (b * (r0 / r) - 1.0) / (b - 1.0))


def bond_length_from_order(order: float, r0: float, b: float = 2.0) -> float:
    """r(o~) = b / ((b-1)*o~ + 1) * r0, for o~ >= 0."""
    if order < 0:
        raise ValueError("bond order must be non-negative")
    return (b / ((b - 1.0) * order + 1.0)) * r0

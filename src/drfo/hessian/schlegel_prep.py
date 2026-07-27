"""Birkholz & Schlegel (JCC 2015) CTS-method-specific preparation: bond-order
interpolation guess building, constrained pre-relaxation on the active
bonds, the Delta-b transition-vector guess, and the eq. 12 Rayleigh-flip
initial Hessian. `build_schlegel_ts_preparation` bundles everything
`optimize.search.run_drfo_search` needs to start a dRFO search from a
reactant/product pair.

Deliberately factored out from the search itself (`optimize/search.py`) so
a caller with their own TS guess (from docking, ML, or manual construction)
or their own transition-vector heuristic can skip this module entirely and
call `run_drfo_search` directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..calculators.base import Calculator, CalculatorError
from ..calculators.xtb import XTBCalculator
from ..geometry import Geometry
from ..internal.coordinates import Bend, InternalCoordinateSystem, Stretch
from ..internal.transform import b_pinv
from ..interpolation.guess import build_ts_guess
from ..optimize.minimizer import constrained_relax
from ..optimize.search import log_calculator_error
from ..optimize.trust import ConvergenceCriteria
from .deltab import delta_b_vector, project_transition_vector
from .initial import build_initial_ts_hessian


@dataclass
class TSPreparation:
    geometry: Geometry
    coord_system: InternalCoordinateSystem
    initial_hessian: np.ndarray
    guide_vector_provider: Callable[[np.ndarray], np.ndarray]
    breaking_bonds: list[tuple[int, int]]
    forming_bonds: list[tuple[int, int]]
    relax_converged: bool
    trajectory: list[Geometry] = field(default_factory=list)


def _active_stretch_indices(ics: InternalCoordinateSystem, active_bonds: set[tuple[int, int]]) -> set[int]:
    return {idx for idx, c in enumerate(ics.coords)
            if isinstance(c, Stretch) and (c.i, c.j) in active_bonds}


def _frozen_indices_for_relaxation(
    ics: InternalCoordinateSystem, active_bonds: set[tuple[int, int]],
) -> set[int]:
    """Freeze the breaking/forming bond stretches plus any angle between a
    pair of those bonds that share an atom, per the paper's constrained
    pre-relaxation phase."""
    frozen = _active_stretch_indices(ics, active_bonds)
    for idx, c in enumerate(ics.coords):
        if not isinstance(c, Bend):
            continue
        b1 = (c.i, c.j) if c.i < c.j else (c.j, c.i)
        b2 = (c.j, c.k) if c.j < c.k else (c.k, c.j)
        if b1 in active_bonds and b2 in active_bonds:
            frozen.add(idx)
    return frozen


def build_schlegel_ts_preparation(
    reactant: Geometry, product: Geometry, *,
    calculator: Calculator,
    cheap_calculator: Calculator | None = None,
    bond_scale: float = 1.4,
    max_relax_steps: int = 100,
    convergence: ConvergenceCriteria | None = None,
    keep_trajectory: bool = False,
    breaking_bonds: list[tuple[int, int]] | None = None,
    forming_bonds: list[tuple[int, int]] | None = None,
) -> TSPreparation:
    """Build everything `optimize.search.run_drfo_search` needs from a
    reactant/product pair, per the paper's CTS recipe: bond-order
    interpolation guess -> constrained pre-relaxation on the real
    calculator (frozen = active bonds + angles sharing an atom with them)
    -> Delta-b transition vector -> eq. 12 flip/scale initial Hessian.

    `breaking_bonds`/`forming_bonds` (each `(i, j)`, `i < j`, 0-based):
    optional externally-supplied bond classification, passed straight
    through to `build_ts_guess` to bypass its default covalent-radius
    distance-cutoff diffing -- use this when a more reliable classification
    is already available (e.g. from an exact atom-mapping step run
    upstream). Both must be given together, or neither.

    Raises `CalculatorError` (already logged) on any calculator failure
    during guess-building or pre-relaxation -- this function either returns
    a complete `TSPreparation` or raises, it never returns a partial
    result. Callers that want a `TSResult`-shaped early-return on failure
    (as `driver.find_ts` does) should catch `CalculatorError` themselves.
    """
    if reactant.symbols != product.symbols:
        raise ValueError("reactant and product must have matching atom ordering/identity")
    if reactant.charge != product.charge or reactant.spin != product.spin:
        raise ValueError("reactant and product must have matching charge/spin")

    crit = convergence or ConvergenceCriteria()
    if cheap_calculator is None:
        if not isinstance(calculator, XTBCalculator):
            raise ValueError("cheap_calculator must be supplied explicitly when calculator "
                              "is not an XTBCalculator (no default cheap potential available)")
        cheap_calculator = XTBCalculator(
            calculator.xtb_path, method="gfnff", charge=calculator.charge,
            spin=calculator.spin, scratch_dir=calculator.scratch_dir,
        )

    try:
        guess = build_ts_guess(
            reactant, product, cheap_calculator, bond_scale=bond_scale,
            breaking_bonds=breaking_bonds, forming_bonds=forming_bonds,
        )
    except CalculatorError as exc:
        log_calculator_error("cheap-calculator failure during guess building", exc)
        raise

    ics = guess.coord_system
    breaking = set(guess.breaking_bonds)
    forming = set(guess.forming_bonds)
    active_bonds = breaking | forming
    frozen_indices = _frozen_indices_for_relaxation(ics, active_bonds)
    delta_b_raw = delta_b_vector(ics, guess.breaking_bonds, guess.forming_bonds)

    trajectory: list[Geometry] = [guess.geometry.copy()] if keep_trajectory else []

    try:
        relax_result = constrained_relax(
            guess.geometry, ics, frozen_indices, calculator,
            max_steps=max_relax_steps, rms_grad_threshold=crit.rms_grad,
            return_hessian=True,
        )
    except CalculatorError as exc:
        log_calculator_error("calculator failure during constrained pre-relaxation", exc)
        raise
    relaxed_geom, H_cart_tilde = relax_result.geometry, relax_result.hessian

    if keep_trajectory:
        trajectory.append(relaxed_geom.copy())

    B = ics.B(relaxed_geom)
    s_NR = project_transition_vector(B, delta_b_raw)
    Bp = b_pinv(B)
    H_int_tilde = Bp.T @ H_cart_tilde @ Bp
    H_int = build_initial_ts_hessian(H_int_tilde, s_NR)

    def guide_vector_provider(B_current: np.ndarray) -> np.ndarray:
        return project_transition_vector(B_current, delta_b_raw)

    return TSPreparation(
        geometry=relaxed_geom, coord_system=ics, initial_hessian=H_int,
        guide_vector_provider=guide_vector_provider,
        breaking_bonds=sorted(breaking), forming_bonds=sorted(forming),
        relax_converged=relax_result.converged, trajectory=trajectory,
    )

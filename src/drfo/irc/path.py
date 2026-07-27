"""Intrinsic Reaction Coordinate (IRC) verification: given a converged
transition state (a stationary point with exactly one imaginary Hessian
mode), follow the steepest-descent path downhill from it in both directions
-- along +/- the imaginary mode's eigenvector -- to confirm it actually
connects two genuine minima, and that those minima are (chemically) the
intended reactant and product rather than some other stationary point the
TS search happened to land near.

Deliberately standalone: takes only a geometry, an (optional) Hessian, and
a `Calculator` -- no dependency on how the TS was found (drfo's own
divided-RFO search, or any other TS optimizer/guess source). This mirrors
`optimize.search.run_drfo_search`'s own independence from the Schlegel-
paper-specific guess-building in `hessian.schlegel_prep`.

Method: a short run of classical mass-weighted steepest descent (the
definition of an IRC, Fukui 1970/1981, with a backtracking line search at
each step -- halve the step and retry if it doesn't lower the energy)
starting right at the TS, to commit to the correct basin -- followed by a
hand-off to the same trust-region BFGS minimizer used elsewhere in this
codebase (`optimize.minimizer.constrained_relax`, unconstrained here) to
finish the descent to an actual minimum quickly and robustly.

This mirrors standard practice in production IRC codes (a handful of true
IRC steps away from the saddle, then a normal geometry optimization to
converge the endpoint) rather than reinventing it: plain steepest descent
famously zig-zags and converges at a crawl once mass-weighting makes the
surface anisotropic (light H atoms vs. heavy neighbors) -- confirmed
directly on HCN<->HNC, where 150 steepest-descent steps barely moved the
energy -- while the commitment to the right basin only needs a handful of
initial steepest-descent steps, not full convergence by that method alone.
Not intended to produce a publication-quality minimum energy path; this
module's purpose is verification (does this TS connect to the expected
minima at all?), where reaching the right minimum matters, not the exact
shape of the path getting there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calculators.base import Calculator
from ..elements import atomic_mass_amu
from ..geometry import AMU_PER_ELECTRON_MASS, Geometry
from ..internal.coordinates import build_coordinate_system
from ..internal.topology import add_virtual_interfragment_bonds, build_bond_graph
from ..optimize.minimizer import constrained_relax


def _mass_weights(geom: Geometry) -> np.ndarray:
    """(3N,) array of 1/sqrt(mass) per Cartesian DOF, mass in atomic units
    (m_e = 1), repeated 3x per atom."""
    masses_au = np.array([atomic_mass_amu(s) for s in geom.symbols]) * AMU_PER_ELECTRON_MASS
    return np.repeat(1.0 / np.sqrt(masses_au), 3)


def imaginary_mode(geom: Geometry, hessian_cart: np.ndarray) -> tuple[float, np.ndarray]:
    """Mass-weight `hessian_cart` (Hartree/bohr^2) and return the
    (eigenvalue, Cartesian displacement direction) of its most negative
    mode -- the reaction coordinate at a first-order saddle point.

    The returned direction is normalized in mass-weighted space and
    converted back to Cartesian (i.e. `direction * inv_sqrt_mass` from the
    eigenvector), ready to scale by a Cartesian step size directly.
    """
    inv_sqrt_m = _mass_weights(geom)
    H_mw = hessian_cart * np.outer(inv_sqrt_m, inv_sqrt_m)
    eigvals, eigvecs = np.linalg.eigh(H_mw)
    idx = int(np.argmin(eigvals))
    direction_mw = eigvecs[:, idx]
    direction_cart = direction_mw * inv_sqrt_m
    direction_cart /= np.linalg.norm(direction_cart)
    return float(eigvals[idx]), direction_cart


def _step_geometry(geom: Geometry, displacement_bohr: np.ndarray) -> Geometry:
    new = geom.copy()
    new.coords = new.coords + displacement_bohr.reshape(geom.natoms, 3)
    return new


@dataclass
class IRCBranchResult:
    path: list[Geometry]
    converged: bool
    n_steps: int

    @property
    def endpoint(self) -> Geometry:
        return self.path[-1]


def _relax_to_minimum(
    geom: Geometry, calculator: Calculator, *, max_steps: int, rms_grad_threshold: float,
) -> tuple[Geometry, bool]:
    """Full (unconstrained) internal-coordinate relaxation to the nearest
    minimum, reusing `optimize.minimizer.constrained_relax` with an empty
    frozen set -- the same trust-region BFGS minimizer already validated
    elsewhere in this codebase for exactly this kind of small-molecule
    relaxation, just without freezing any coordinates."""
    graph = build_bond_graph(geom)
    graph = add_virtual_interfragment_bonds(geom, graph)
    ics = build_coordinate_system(graph, geom, geom)
    result = constrained_relax(
        geom, ics, frozen_indices=set(), calc=calculator,
        max_steps=max_steps, rms_grad_threshold=rms_grad_threshold,
    )
    return result.geometry, result.converged


def follow_downhill(
    start: Geometry, calculator: Calculator, *,
    step_size_bohr: float = 0.15,
    n_path_steps: int = 6,
    max_step_halvings: int = 8,
    finish_max_steps: int = 100,
    rms_grad_threshold: float = 3.2e-4,
    keep_path: bool = True,
) -> IRCBranchResult:
    """Take up to `n_path_steps` of mass-weighted steepest descent from
    `start` (assumed already slightly displaced off a stationary point --
    see `run_irc`) to commit to the correct basin, then hand off to
    `_relax_to_minimum` to finish converging quickly.

    Each steepest-descent step takes a fixed mass-weighted arc-length
    (`step_size_bohr`) along the current steepest-descent direction,
    halving the step (up to `max_step_halvings` times) if the candidate
    point's energy doesn't decrease -- a standard backtracking line search,
    since a fixed-length step in a steepest-descent direction is not
    guaranteed downhill once curvature bends the true path away from that
    straight line. Stops early (before `n_path_steps`) if the gradient is
    already small -- a weak initial imaginary mode can land close enough to
    a minimum that no path-following is needed at all.
    """
    inv_sqrt_m = _mass_weights(start)
    geom = start
    energy = calculator.energy(geom)
    path = [geom] if keep_path else []
    n_steps = 0

    for step_idx in range(n_path_steps):
        g = calculator.gradient(geom).reshape(-1)
        g_rms = float(np.sqrt(np.mean(g**2)))
        if g_rms < rms_grad_threshold:
            break

        g_mw = g * inv_sqrt_m
        direction_mw = -g_mw / np.linalg.norm(g_mw)
        current_step = step_size_bohr
        accepted = False
        for _ in range(max_step_halvings + 1):
            displacement = (current_step * direction_mw) * inv_sqrt_m
            candidate = _step_geometry(geom, displacement)
            candidate_energy = calculator.energy(candidate)
            if candidate_energy < energy:
                accepted = True
                break
            current_step *= 0.5
        if not accepted:
            # Stuck: no downhill step found even after repeated halving --
            # stop path-following early and let the minimizer take over.
            break

        geom, energy = candidate, candidate_energy
        n_steps = step_idx + 1
        if keep_path:
            path.append(geom)

    geom, converged = _relax_to_minimum(
        geom, calculator, max_steps=finish_max_steps, rms_grad_threshold=rms_grad_threshold,
    )
    if keep_path:
        path.append(geom)
    else:
        path = [geom]
    return IRCBranchResult(path=path, converged=converged, n_steps=n_steps)


@dataclass
class IRCResult:
    forward: IRCBranchResult
    reverse: IRCBranchResult
    imaginary_eigenvalue: float

    @property
    def both_converged(self) -> bool:
        return self.forward.converged and self.reverse.converged


def run_irc(
    ts_geometry: Geometry, calculator: Calculator, *,
    hessian: np.ndarray | None = None,
    initial_displacement_bohr: float = 0.3,
    step_size_bohr: float = 0.15,
    n_path_steps: int = 6,
    finish_max_steps: int = 100,
    rms_grad_threshold: float = 3.2e-4,
    keep_path: bool = True,
) -> IRCResult:
    """Follow the mass-weighted steepest-descent path downhill from
    `ts_geometry` in both directions along its imaginary Hessian mode, to
    verify it connects two genuine minima.

    `hessian` (Cartesian, Hartree/bohr^2): if not supplied, computed via
    `calculator.hessian(ts_geometry)`. Since the TS mode's imaginary
    eigenvalue defines the two initial displacement directions, this
    should be a Hessian genuinely evaluated at (or very near) `ts_geometry`
    -- not a stale Bofill-updated one from partway through a search.

    Raises `CalculatorError` (uncaught) on any calculator failure -- unlike
    `optimize.search.run_drfo_search`, this is a verification step run
    once on an already-converged TS, so a failure here should surface
    directly rather than being absorbed into a status enum.
    """
    if hessian is None:
        hessian = calculator.hessian(ts_geometry)

    eigenvalue, direction = imaginary_mode(ts_geometry, hessian)
    displacement = initial_displacement_bohr * direction

    forward_start = _step_geometry(ts_geometry, displacement)
    reverse_start = _step_geometry(ts_geometry, -displacement)

    forward = follow_downhill(
        forward_start, calculator, step_size_bohr=step_size_bohr, n_path_steps=n_path_steps,
        finish_max_steps=finish_max_steps, rms_grad_threshold=rms_grad_threshold, keep_path=keep_path,
    )
    reverse = follow_downhill(
        reverse_start, calculator, step_size_bohr=step_size_bohr, n_path_steps=n_path_steps,
        finish_max_steps=finish_max_steps, rms_grad_threshold=rms_grad_threshold, keep_path=keep_path,
    )
    return IRCResult(forward=forward, reverse=reverse, imaginary_eigenvalue=eigenvalue)

"""Bond-order interpolation with relaxation: builds an initial transition-
state guess geometry from reactant and product structures, following the
paper's six-step algorithm (Birkholz & Schlegel, JCC 2015, "Bond order
interpolation with relaxation").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..calculators.base import Calculator
from ..elements import reference_bond_length
from ..geometry import Geometry
from ..internal.coordinates import InternalCoordinateSystem, Stretch, build_coordinate_system
from ..internal.topology import (
    BondGraph,
    add_virtual_interfragment_bonds,
    build_bond_graph,
    diff_bonds,
    merge_bond_graphs,
)
from ..internal.transform import internal_to_cartesian_step
from ..optimize.minimizer import constrained_relax
from .bondorder import bond_length_from_order, bond_order

_MIN_GOAL_ORDER = 0.05  # floor to avoid r(order) -> infinity as order -> 0

# Small, deterministic symmetry-breaking perturbation applied before the
# interpolation walk. Without this, a perfectly symmetric reactant/product
# pair (e.g. a linear triatomic like HCN<->HNC) can leave the entire guess-
# building pipeline stuck exactly on the symmetric configuration: every
# gradient/projection involved is exactly zero in the symmetry-breaking
# direction, so nothing ever nudges it off that subspace, and interpolating
# bond lengths while stuck collinear can force other atoms into an
# unphysical collision. The paper's own Discussion section explicitly
# describes this failure mode ("the gradient in the direction of the atomic
# motion required to break symmetry is zero, it can be difficult for an
# optimizer to step away from the symmetric geometry"). A tiny, reproducible
# (fixed-seed) perturbation is the standard mitigation.
_SYMMETRY_BREAK_MAGNITUDE_BOHR = 0.02


def _break_symmetry(geom: Geometry, *, seed: int = 0) -> Geometry:
    rng = np.random.default_rng(seed)
    perturbed = geom.copy()
    perturbed.coords = perturbed.coords + rng.uniform(
        -_SYMMETRY_BREAK_MAGNITUDE_BOHR, _SYMMETRY_BREAK_MAGNITUDE_BOHR, size=perturbed.coords.shape,
    )
    return perturbed


@dataclass
class GuessResult:
    geometry: Geometry
    source: Literal["reactant_path", "product_path"]
    cheap_energy: float
    breaking_bonds: list[tuple[int, int]]
    forming_bonds: list[tuple[int, int]]
    coord_system: InternalCoordinateSystem


def _active_stretch_indices(
    ics: InternalCoordinateSystem, active_bonds: set[tuple[int, int]],
) -> set[int]:
    indices = set()
    for idx, c in enumerate(ics.coords):
        if isinstance(c, Stretch) and (c.i, c.j) in active_bonds:
            indices.add(idx)
    return indices


def _clamp_long_bonds(
    geom: Geometry, ics: InternalCoordinateSystem, active_bonds: set[tuple[int, int]],
    clamp_factor: float,
) -> Geometry:
    dq = np.zeros(len(ics))
    changed = False
    for idx, c in enumerate(ics.coords):
        if not isinstance(c, Stretch) or (c.i, c.j) not in active_bonds:
            continue
        r0 = reference_bond_length(geom.symbols[c.i], geom.symbols[c.j])
        current = float(np.linalg.norm(geom.coords[c.i] - geom.coords[c.j]))
        cap = clamp_factor * r0
        if current > cap:
            dq[idx] = cap - current
            changed = True
    if not changed:
        return geom
    new_geom, _ = internal_to_cartesian_step(geom, ics, dq)
    return new_geom


def _walk_to_goal_lengths(
    geom: Geometry, ics: InternalCoordinateSystem, active_bonds: set[tuple[int, int]],
    goal_length: dict[tuple[int, int], float], cheap_calc: Calculator, *,
    max_internal_step_bohr: float, active_indices: set[int],
    tol: float = 1e-3, max_outer_iter: int = 20, relax_max_steps: int = 40,
) -> Geometry:
    current = geom
    for _ in range(max_outer_iter):
        dq = np.zeros(len(ics))
        vals = ics.values(current)
        any_remaining = False
        for idx, c in enumerate(ics.coords):
            if not isinstance(c, Stretch) or (c.i, c.j) not in active_bonds:
                continue
            target = goal_length[(c.i, c.j)]
            delta = target - vals[idx]
            if abs(delta) > tol:
                any_remaining = True
            dq[idx] = float(np.clip(delta, -max_internal_step_bohr, max_internal_step_bohr))
        if not any_remaining:
            break
        current, _ = internal_to_cartesian_step(current, ics, dq)
        relax = constrained_relax(current, ics, active_indices, cheap_calc,
                                   max_steps=relax_max_steps)
        current = relax.geometry
    return current


def build_reaction_coordinate_system(
    reactant: Geometry, product: Geometry, *,
    bond_scale: float = 1.4,
    breaking_bonds: list[tuple[int, int]] | None = None,
    forming_bonds: list[tuple[int, int]] | None = None,
) -> tuple[InternalCoordinateSystem, set[tuple[int, int]], set[tuple[int, int]]]:
    """Build the redundant internal coordinate system spanning `reactant`
    and `product`, plus the breaking/forming bond classification it was
    built from. Factored out of `build_ts_guess` so any guess-building
    strategy -- not just this module's own bond-order interpolation --
    can get the same coordinate system/active-bond classification without
    duplicating this logic (e.g. an externally-built guess geometry passed
    to `hessian.schlegel_prep.build_schlegel_ts_preparation`).

    By default, breaking/forming bonds are derived from a covalent-radius
    distance cutoff (`bond_scale`) on each endpoint independently, diffed
    against each other -- a heuristic that can misclassify bonds near the
    cutoff or across a genuinely long breaking/forming bond. If a more
    reliable classification is already available (e.g. from an exact
    graph-isomorphism atom mapping computed upstream, as opposed to pure
    geometry), pass it directly via `breaking_bonds`/`forming_bonds` (each
    `(i, j)` with `i < j`, 0-based, matching `reactant`/`product`'s shared
    atom ordering) to bypass the distance-cutoff diffing entirely -- both
    must be given together, or neither.
    """
    if reactant.symbols != product.symbols:
        raise ValueError("reactant and product must have matching atom ordering/identity")
    if reactant.charge != product.charge or reactant.spin != product.spin:
        raise ValueError("reactant and product must have matching charge/spin")
    if (breaking_bonds is None) != (forming_bonds is None):
        raise ValueError("breaking_bonds and forming_bonds must be supplied together, or neither")

    graph_r = build_bond_graph(reactant, scale=bond_scale)
    graph_p = build_bond_graph(product, scale=bond_scale)
    merged = merge_bond_graphs(graph_r, graph_p)

    if breaking_bonds is None:
        breaking, forming = diff_bonds(graph_r, graph_p)
    else:
        breaking, forming = set(breaking_bonds), set(forming_bonds)
        # Externally-supplied active bonds need a stretch coordinate too,
        # even if the covalent-radius cutoff didn't happen to detect them
        # (e.g. a long breaking bond) -- fold them into the merged graph
        # before building the coordinate system.
        merged = BondGraph(bonds=merged.bonds | breaking | forming, virtual_bonds=merged.virtual_bonds)

    merged = add_virtual_interfragment_bonds(reactant, merged)

    active_bonds = breaking | forming
    if not active_bonds:
        raise ValueError("reactant and product have identical bonding; no reaction coordinate")

    ics = build_coordinate_system(merged, reactant, product)
    return ics, breaking, forming


def build_ts_guess(
    reactant: Geometry, product: Geometry, cheap_calc: Calculator, *,
    bond_scale: float = 1.4, max_internal_step_bohr: float = 0.5,
    clamp_factor: float = 2.0, b_param: float = 2.0,
    breaking_bonds: list[tuple[int, int]] | None = None,
    forming_bonds: list[tuple[int, int]] | None = None,
) -> GuessResult:
    """Build a TS guess from `reactant`/`product` via bond-order
    interpolation with relaxation -- see `build_reaction_coordinate_system`
    for the `breaking_bonds`/`forming_bonds` override semantics shared
    with it.
    """
    ics, breaking, forming = build_reaction_coordinate_system(
        reactant, product, bond_scale=bond_scale,
        breaking_bonds=breaking_bonds, forming_bonds=forming_bonds,
    )
    active_bonds = breaking | forming
    active_indices = _active_stretch_indices(ics, active_bonds)

    goal_length: dict[tuple[int, int], float] = {}
    for (i, j) in active_bonds:
        r0 = reference_bond_length(reactant.symbols[i], reactant.symbols[j])
        r_r = float(np.linalg.norm(reactant.coords[i] - reactant.coords[j]))
        r_p = float(np.linalg.norm(product.coords[i] - product.coords[j]))
        order_r = bond_order(r_r, r0, b_param)
        order_p = bond_order(r_p, r0, b_param)
        goal_order = max(_MIN_GOAL_ORDER, 0.5 * (order_r + order_p))
        goal_length[(i, j)] = bond_length_from_order(goal_order, r0, b_param)

    candidates: list[GuessResult] = []
    for seed, (source, endpoint) in enumerate((("reactant_path", reactant), ("product_path", product))):
        cleaned = _clamp_long_bonds(endpoint, ics, active_bonds, clamp_factor)
        cleaned = _break_symmetry(cleaned, seed=seed)
        relax0 = constrained_relax(cleaned, ics, active_indices, cheap_calc, max_steps=60)
        cleaned = relax0.geometry

        final = _walk_to_goal_lengths(
            cleaned, ics, active_bonds, goal_length, cheap_calc,
            max_internal_step_bohr=max_internal_step_bohr, active_indices=active_indices,
        )
        energy = cheap_calc.energy(final)
        candidates.append(GuessResult(
            geometry=final, source=source, cheap_energy=energy,
            breaking_bonds=sorted(breaking), forming_bonds=sorted(forming),
            coord_system=ics,
        ))

    return min(candidates, key=lambda r: r.cheap_energy)

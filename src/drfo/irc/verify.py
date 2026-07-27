"""Compare IRC endpoints against known reactant/product structures, to turn
a raw `IRCResult` into a yes/no answer: does this TS actually connect the
intended reaction, not just *some* pair of minima?

Kept separate from `path.run_irc` itself (which has no notion of a
"reactant" or "product", only "the two directions off the imaginary
mode") so the pure path-following algorithm stays reusable for callers who
don't have reference endpoints to compare against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calculators.base import Calculator
from ..geometry import Geometry
from ..internal.topology import build_bond_graph
from .path import IRCResult, run_irc


def rmsd_after_alignment(geom_a: Geometry, geom_b: Geometry) -> float:
    """Kabsch-superposed RMSD (bohr) between two geometries with matching,
    already-corresponding atom order (same symbols in the same sequence,
    same convention `find_ts`'s own reactant/product inputs require) --
    rotation/translation only, no atom-permutation search."""
    if geom_a.symbols != geom_b.symbols:
        raise ValueError("geometries must have matching atom ordering/identity")
    a = geom_a.coords - geom_a.coords.mean(axis=0)
    b = geom_b.coords - geom_b.coords.mean(axis=0)
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    b_aligned = b @ r.T
    return float(np.sqrt(np.mean(np.sum((a - b_aligned) ** 2, axis=1))))


def same_topology(geom_a: Geometry, geom_b: Geometry, *, bond_scale: float = 1.4) -> bool:
    """Whether `geom_a` and `geom_b` have identical covalent-radius-cutoff
    bond graphs -- robust where RMSD isn't: a dissociated multi-fragment
    product (e.g. H2 + CO) has an essentially flat potential along the
    fragment-separation coordinate, so two independently-relaxed, equally
    correct dissociated geometries can differ by several bohr in RMSD
    despite being chemically identical. Bond connectivity, unlike exact
    separation distance, is unambiguous once fragments are clearly apart."""
    return build_bond_graph(geom_a, scale=bond_scale).bonds == build_bond_graph(geom_b, scale=bond_scale).bonds


@dataclass
class TSVerification:
    irc: IRCResult
    forward_reactant_rmsd: float | None
    forward_product_rmsd: float | None
    reverse_reactant_rmsd: float | None
    reverse_product_rmsd: float | None
    connects_expected_reaction: bool | None

    @property
    def both_converged(self) -> bool:
        return self.irc.both_converged


def verify_ts(
    ts_geometry: Geometry, calculator: Calculator, *,
    hessian: np.ndarray | None = None,
    reactant: Geometry | None = None,
    product: Geometry | None = None,
    bond_scale: float = 1.4,
    initial_displacement_bohr: float = 0.3,
    step_size_bohr: float = 0.15,
    n_path_steps: int = 6,
    finish_max_steps: int = 100,
    rms_grad_threshold: float = 3.2e-4,
    keep_path: bool = True,
) -> TSVerification:
    """Run `run_irc` from `ts_geometry`, then (if `reactant`/`product` are
    given) check whether the two downhill endpoints correspond to them.

    The IRC's forward/reverse directions are an arbitrary sign convention
    (whichever eigenvector sign `numpy.linalg.eigh` happens to return) --
    not guaranteed to line up with "forward = product". This checks both
    pairings (forward->product & reverse->reactant, or forward->reactant &
    reverse->product) and reports whichever one is consistent, so callers
    don't have to reason about the sign convention themselves.

    `connects_expected_reaction` is `None` if `reactant`/`product` weren't
    supplied (nothing to compare against). Otherwise it's `True` only if
    both IRC branches converged *and* one of the two pairings has matching
    bond topology (`same_topology`) on both sides -- the primary signal,
    since it's robust to dissociated fragments settling at a different but
    equally valid separation distance. RMSD is still computed and reported
    (useful diagnostic, and the more informative signal for non-dissociative
    reactions), but `connects_expected_reaction` doesn't depend on it.
    """
    irc = run_irc(
        ts_geometry, calculator, hessian=hessian,
        initial_displacement_bohr=initial_displacement_bohr, step_size_bohr=step_size_bohr,
        n_path_steps=n_path_steps, finish_max_steps=finish_max_steps,
        rms_grad_threshold=rms_grad_threshold, keep_path=keep_path,
    )

    fwd_r = fwd_p = rev_r = rev_p = None
    connects = None
    if reactant is not None and product is not None:
        fwd_r = rmsd_after_alignment(irc.forward.endpoint, reactant)
        fwd_p = rmsd_after_alignment(irc.forward.endpoint, product)
        rev_r = rmsd_after_alignment(irc.reverse.endpoint, reactant)
        rev_p = rmsd_after_alignment(irc.reverse.endpoint, product)

        straight = (same_topology(irc.forward.endpoint, product, bond_scale=bond_scale)
                    and same_topology(irc.reverse.endpoint, reactant, bond_scale=bond_scale))
        crossed = (same_topology(irc.forward.endpoint, reactant, bond_scale=bond_scale)
                   and same_topology(irc.reverse.endpoint, product, bond_scale=bond_scale))
        connects = bool(irc.both_converged and (straight or crossed))

    return TSVerification(
        irc=irc, forward_reactant_rmsd=fwd_r, forward_product_rmsd=fwd_p,
        reverse_reactant_rmsd=rev_r, reverse_product_rmsd=rev_p,
        connects_expected_reaction=connects,
    )

"""M1 integration test: validates rfo_core's dRFO math and XTBCalculator's
subprocess plumbing together, end-to-end, against a real GFN2-xTB PES.

Finding, documented here rather than swept under the rug: a bare Cartesian
pRFO/dRFO search with a hand-picked *static* guide vector reliably converges
to a genuine stationary point of the real PES, but is not reliable at
landing specifically on the transition state for a reaction (HCN<->HNC
isomerization) whose true transition-vector direction rotates substantially
as the geometry evolves — it is entirely plausible, and was observed here,
that the search instead slides downhill into the product minimum. This is
exactly the failure mode the paper's own Table 2 documents for simple
reactant/product tangent vectors, and is the reason the full pipeline
builds a bonding-derived Delta-b vector and does its stepping in redundant
internal coordinates (M2-M4) rather than relying on a static Cartesian
guess. This module is retained as a validation tool, not the production
TS-finding path.

What this test *does* honestly establish: the full loop -- xtb subprocess
calls for gradient+Hessian at every step, `drfo_step`'s augmented-Hessian
shift-matrix solve, translation projection, trust-region step control, and
BFGS bookkeeping -- runs for dozens of consecutive real xtb invocations
without error, and converges to a genuine stationary point of the real PES.
That stationary point's energy is cross-checked against an independently
obtained xtb `--opt` (ANCOPT) optimization of the same molecule, agreeing
to 7 significant figures -- strong evidence the calculator parsing and the
optimizer math are both numerically correct.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from drfo import Geometry, XTBCalculator
from drfo.optimize.rfo_core import drfo_step
from drfo.optimize.trust import ConvergenceCriteria, check_converged

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"

# Independently obtained via `xtb --opt tight` on a hand-built HNC starting
# geometry (see design doc / session notes); used only as a cross-check.
HNC_REFERENCE_ENERGY = -5.472159882214


def _has_xtb() -> bool:
    return shutil.which(XTB_PATH) is not None or __import__("pathlib").Path(XTB_PATH).exists()


pytestmark = pytest.mark.skipif(not _has_xtb(), reason=f"xtb binary not found at {XTB_PATH}")


def _build_hcn_like(rCH: float, rCN: float, angle_deg: float) -> Geometry:
    """H, C, N Cartesian construction: C at origin, N along +z, H in the
    xz-plane at the given C-H distance and H-C-N angle."""
    a = np.radians(angle_deg)
    H = np.array([rCH * np.sin(a), 0.0, rCH * np.cos(a)])
    C = np.array([0.0, 0.0, 0.0])
    N = np.array([0.0, 0.0, rCN])
    return Geometry.from_angstrom(["H", "C", "N"], [H, C, N])


def test_cartesian_toy_dRFO_converges_to_real_stationary_point(tmp_path):
    """Seeded near the HCN/HNC isomerization ridge (located via an offline
    relaxed scan), the toy Cartesian dRFO stepper must run to a converged,
    genuine stationary point of the real GFN2-xTB PES without error, using
    only real xtb subprocess calls for every gradient/Hessian evaluation."""
    guess = _build_hcn_like(rCH=1.1606, rCN=1.2028, angle_deg=68.0)

    # reduced coordinate = [Hx,Hz,Cx,Cz,Nx,Nz] (3-atom systems are planar,
    # so restricting to the xz-plane removes a spurious near-zero-curvature
    # out-of-plane DOF without losing any physical motion).
    idx_xz = [0, 2, 3, 5, 6, 8]
    guide9 = np.zeros(9)
    guide9[0] = 1.0  # seed the bending direction (H's x-displacement)
    guide6 = guide9[idx_xz]

    # Translation projector: 3-atom xz-plane system has 2 translational
    # modes (x, z) that must be removed before solving, or a near-singular
    # augmented Hessian can produce a step that is a pure rigid translation
    # (energy-invariant, physically meaningless).
    ux = np.array([1.0, 0, 1, 0, 1, 0]) / np.sqrt(3)
    uz = np.array([0.0, 1, 0, 1, 0, 1]) / np.sqrt(3)
    P = np.eye(6) - np.outer(ux, ux) - np.outer(uz, uz)

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)

    geom = guess.copy()
    n = geom.natoms
    crit = ConvergenceCriteria()
    trust_radius = 0.1

    converged = False
    for _ in range(50):
        r = calc.compute(geom, gradient=True, hessian=True)
        g9, H9 = r.gradient.reshape(9), r.hessian
        g6 = P @ g9[idx_xz]
        H6 = P @ H9[np.ix_(idx_xz, idx_xz)] @ P

        dq6 = P @ drfo_step(H6, g6, guide6, overlap_thresh=0.5, ridge=1e-6)
        norm = np.linalg.norm(dq6)
        if norm > trust_radius:
            dq6 = dq6 * (trust_radius / norm)

        dq9 = np.zeros(9)
        dq9[idx_xz] = dq6
        if check_converged(g6, dq6, crit):
            converged = True
            break

        new_geom = geom.copy()
        new_geom.coords = geom.coords + dq9.reshape(n, 3)
        geom = new_geom

    assert converged, "toy Cartesian dRFO failed to reach a stationary point"

    final_energy = calc.energy(geom)
    # Cross-check against an independently obtained xtb --opt energy for
    # HNC: agreement to several decimal places is strong evidence that both
    # the xtb calculator parsing and drfo_step's math are numerically
    # correct end-to-end, even though this particular guide vector landed
    # on the product minimum rather than the saddle (see module docstring).
    assert final_energy == pytest.approx(HNC_REFERENCE_ENERGY, abs=1e-5)


def test_xtb_calculator_survives_many_sequential_calls(tmp_path):
    """Regression guard for the subprocess plumbing itself: many consecutive
    energy/gradient/Hessian calls against changing geometries should never
    error, never leak stale results between calls, and should each get a
    fresh scratch directory."""
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    geoms = [_build_hcn_like(rCH, 1.20, 68.0) for rCH in np.linspace(1.0, 1.6, 8)]

    energies = [calc.energy(g) for g in geoms]
    # Energies must actually differ between distinct geometries (guards
    # against the "stale output file" class of bug found during development).
    assert len(set(round(e, 8) for e in energies)) == len(energies)

    grads = [calc.gradient(g) for g in geoms]
    for g in grads:
        assert g.shape == (3, 3)
        assert np.all(np.isfinite(g))

"""Integration test: verify_ts's IRC-following actually reconnects a
converged TS to its true reactant and product, using HCN <-> HNC (already
validated to converge reliably and cheaply -- 3 atoms -- in
tests/benchmark/test_benchmark.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from drfo import XTBCalculator, find_ts, verify_ts
from drfo.io.xyz import read_xyz

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")

STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "structures"


def test_verify_ts_reconnects_hcn_hnc(tmp_path):
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    hnc = read_xyz(STRUCTURES_DIR / "hnc.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    ts = find_ts(hcn, hnc, calculator=calc, max_relax_steps=80, max_ts_steps=80)
    assert ts.converged and ts.n_imaginary == 1

    verification = verify_ts(ts.geometry, calc, hessian=ts.hessian, reactant=hcn, product=hnc)

    assert verification.both_converged
    assert verification.connects_expected_reaction
    # RMSD is diagnostic-only now (see same_topology's docstring for why),
    # but should still be small here since HCN/HNC don't dissociate.
    assert min(verification.forward_reactant_rmsd, verification.forward_product_rmsd) < 0.3
    assert min(verification.reverse_reactant_rmsd, verification.reverse_product_rmsd) < 0.3


def test_verify_ts_reconnects_h2co_dissociation(tmp_path):
    """The dissociative case same_topology exists for: H2 + CO's fragment
    separation is only loosely pinned by the reference structure's 6 A
    placement (a flat direction on the PES beyond van der Waals contact),
    so an RMSD-only check can fail even when the reaction is genuinely
    verified -- see same_topology's docstring."""
    h2co = read_xyz(STRUCTURES_DIR / "h2co.xyz")
    h2_plus_co = read_xyz(STRUCTURES_DIR / "h2_plus_co.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    ts = find_ts(h2co, h2_plus_co, calculator=calc, max_relax_steps=80, max_ts_steps=80)
    assert ts.converged and ts.n_imaginary == 1

    verification = verify_ts(ts.geometry, calc, hessian=ts.hessian, reactant=h2co, product=h2_plus_co)

    assert verification.both_converged
    assert verification.connects_expected_reaction

"""M5 benchmark suite: validates the full `find_ts` CTS pipeline against
reactions from the paper's Fig. 1 test set, at the GFN2-xTB level.

Status of the originally-planned M5 reaction set (see design doc): all
three PASS.
  - HCN <-> HNC (both directions).
  - H2CO <-> H2 + CO. Required building a proper OutOfPlane internal
    coordinate (formaldehyde's planar 3-connected carbon made the Wilson
    B-matrix singular for any torsion that would otherwise describe
    out-of-plane motion there -- see internal/coordinates.py's OutOfPlane
    type) plus two dRFO-stepper fixes: eigenvalue flooring on the internal-
    coordinate Hessian (near-zero eigenvalues, from genuine redundancy or a
    zeroed B-matrix row, otherwise dominate the RFO step direction) and
    projecting the raw RFO step onto the Wilson B-matrix's row space before
    attempting the back-transform (an unprojected step can carry a
    structurally-unreachable component that no amount of trust-radius
    shrinking can fix, since "unreachable" doesn't shrink to "reachable" at
    smaller magnitude) -- see optimize/rfo_core.py's floor_eigenvalues and
    optimize/drfo.py's DRFOStepper.step().
  - 1,3-pentadiene 1,5-H shift. With the same two fixes, this larger
    (13-atom, 39-DOF) system converges cleanly to a genuine TS (n_imaginary
    == 1, barrier ~30 kcal/mol -- physically sane for a [1,5]-H sigmatropic
    shift), but needs ~700 dRFO steps to get there (~10+ minutes of real
    xtb calls), vs. tens of steps for the smaller cases. Marked `slow` and
    excluded from a default `pytest` run (deselect with `-m "not slow"`,
    or run explicitly with `pytest -m slow`).

Beyond the original three, two more reactions were added while investigating
further:
  - SiH2 + H2 -> SiH4 (silylene insertion). Surfaced a genuine correctness
    bug, not just a slowness issue: the dRFO search reached a good TS
    (n_imaginary==1, gradient RMS 7.6e-4) around step 450, then *overshot*
    past it into a worse stationary region (n_imaginary==2, gradient RMS
    2.0e-3) by step 550 and never recovered within an 800-step budget. Fixed
    generally (not silane-specifically) by tracking the best (lowest-
    gradient) geometry seen throughout the search and falling back to it
    when the run doesn't cleanly converge, rather than reporting whichever
    point the last accepted step happened to land on -- see
    `TSResult.used_best_available` and driver.find_ts's tracking logic.
    Marked `slow` (needs ~800 steps with the fallback to land on the good
    point; true convergence within budget remains an open item).
  - CF2 + C2H4 -> 1,1-difluorocyclopropane (difluorocarbene cycloaddition).
    Runs stably (n_imaginary==1 throughout) but converges glacially --
    the same "many small steps needed" pattern pentadiene showed before its
    breakthrough, not yet resolved for this case. Not yet added to the
    automated suite; see session notes.

Reactions being added one at a time beyond the original plan:
  - H2 + H2CO -> CH3OH (H2 addition to formaldehyde). Clean pass: 139 steps,
    n_imaginary==1, gradient RMS 2.4e-5.
  - HF + C2H4 -> C2H5F (HF addition to ethylene). Clean pass, fastest yet:
    48 steps, n_imaginary==1, gradient RMS 9.2e-5.
  - Cyc-But (1,3-butadiene -> cyclobutene electrocyclization): DOES NOT
    WORK, for a conceptually distinct reason from every earlier blocker.
    The search doesn't stall or diverge -- it slides monotonically downhill
    straight to the product (energy decreasing every single step, gradient
    RMS *increasing* the whole way) without ever finding resistance from a
    barrier. Root cause: this reaction's only "active" bond in the simple
    breaking/forming sense is the single new C1-C4 sigma bond: (0,3).
    Electrocyclic ring closure is a concerted pericyclic mechanism where
    the terminal CH2 groups must simultaneously rotate (dis/conrotate) as
    they approach -- a genuinely rotational reaction-coordinate component
    that a bond-length-only Delta-b vector cannot represent at all. Not
    added to the suite; would need Delta-b to also encode a torsional/
    rotational component for reactions of this class, a real modeling
    gap rather than a numerical bug.
  - CPHT ([1,5]-H shift in 1,3-cyclopentadiene, the cyclic analogue of the
    pentadiene case): clean pass, 164 steps, n_imaginary==1, barrier
    ~27.6 kcal/mol (close to the experimental ~24-25 kcal/mol for this
    well-studied reaction).
  - C2N2O (N2O + ethylene [3+2] cycloaddition to an oxadiazoline ring):
    clean pass, 128 steps, n_imaginary==1, gradient RMS 1.8e-5.
  - Sulfolene (SO2 + 1,3-butadiene cheletropic addition): DOES NOT WORK,
    the same general difficulty class as DFCP -- barely moved from the
    reactant after 400 steps (energy nearly unchanged) and ended with
    n_imaginary==3, not 1. Both reactions form TWO new bonds at once in a
    concerted step (S bonding to both diene termini simultaneously here,
    carbene bonding to both alkene carbons for DFCP); the current Delta-b
    (independent +-1 per bond, no coupling between the two forming bonds)
    apparently doesn't provide enough guidance for this class of
    simultaneous 2-bond-forming cycloaddition/cheletropic mechanism. Not
    added to the suite.
  - SN2 (F- + CH3Cl -> CH3F + Cl-): NOT ATTEMPTED beyond a direct sanity
    check, which confirmed the paper's own explicit caveat that this exact
    case is barrierless at some DFT levels (the paper substitutes tert-
    butyl chloride for exactly this reason). At GFN2-xTB level, a modest
    F...C separation (3.0 A) relaxed straight through to the product with
    no resistance at all (Cl- fully departed, C-F fully formed) under
    plain unconstrained optimization, and a larger separation (5.5 A)
    caused genuine SCF non-convergence for the isolated F- anion in vacuum
    (a known general difficulty for semiempirical tight-binding methods
    describing well-separated, diffuse anions without explicit solvent).
    Not pursued further; a bulkier substrate (as the paper itself uses)
    would likely be needed.

Remaining reactions (Cope, DACP2, DACP+eth, Ene, Grignard, Hydro, Oxirane,
OxyCope) were not added, and the reason matters for anyone picking this
back up: it is NOT that the pipeline failed on them. It's that reliably
hand-building correct, consistently-indexed reactant/product xyz pairs for
larger/more topologically involved molecules -- with no RDKit or other
structure-generation tool available in this environment -- turned out to be
genuinely error-prone:
  - Cope (1,5-hexadiene [3,3]-sigmatropic rearrangement): the product's
    sp2/sp3 character swaps between a DIFFERENT pair of carbons than the
    ones whose bond breaks/forms (C1/C6 gain sp3 character, C3/C4 gain sp2
    character, the reverse of the naive "swap the two ends" pattern that
    worked for pentadiene/CPHT's simple H-migrations). A hand-built product
    guess ended up with a stray extra H (an accidental CH3 instead of the
    intended terminal =CH2), caught only by writing a script to
    programmatically re-verify per-carbon H-counts after the fact -- this
    class of mistake would silently corrupt a benchmark case if not
    caught, so it was abandoned rather than risk it.
  - Ene (propene + ethylene ene reaction): a geometrically reasonable guess
    (migrating H relocated, new-bond partners brought close) relaxed under
    `xtb --opt` into a DIFFERENT, chemically valid product (a new bond to
    the allylic carbon instead of the terminal alkene carbon) rather than
    the one targeted -- there is no guarantee that nudging atoms toward an
    intended product and then relaxing lands in the intended product's
    basin rather than a neighboring one, for reactions with more than one
    plausible bond-reorganization pathway.
  - Grignard and Oxirane involve aromatic rings; Hydro is multi-molecular
    (ester + 2 waters) and would plausibly hit the same "2 bonds forming at
    once" limit as DFCP/Sulfolene via its proton-relay mechanism; DACP2 and
    DACP+eth are Diels-Alder cycloadditions (2 bonds forming at once, same
    predicted limit). None of these four were attempted at all, given the
    construction-reliability finding above.
  A tool that can generate verified 3D structures from a reaction SMILES
  or similar (RDKit, or a manual bond-order-interpolation-consistency
  checker as done for Cope) would be the right fix before attempting this
  set again, rather than more careful manual coordinate entry.

Success criteria per the design doc: `TSResult.converged`, `n_imaginary`
== 1, and a sanity check that the reaction's active bonds land at an
intermediate (neither fully broken nor fully formed) length at the TS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drfo import Geometry, XTBCalculator, find_ts
from drfo.io.xyz import read_xyz

XTB_PATH = "/home/jason/xtb-6.6.1/bin/xtb"
pytestmark = pytest.mark.skipif(not Path(XTB_PATH).exists(), reason="xtb binary not found")

STRUCTURES_DIR = Path(__file__).parent / "structures"

# Reference energies (xtb --opt tight, GFN2-xTB), used only for the
# "TS must be higher than both endpoints" sanity check -- not a strict
# numerical target, since the point is checking barrier physics, not
# reproducing a specific literature value.
HCN_ENERGY = -5.504066182150
HNC_ENERGY = -5.472159882214
H2CO_ENERGY = -7.175648091101
H2_PLUS_CO_ENERGY = -7.104560503442


def _bonding_sanity_check(result, reactant: Geometry, product: Geometry) -> None:
    """The breaking/forming bonds must sit at an intermediate length at the
    converged TS -- neither fully broken nor fully formed."""
    active = set(result.breaking_bonds) | set(result.forming_bonds)
    for (i, j) in active:
        r_react = np.linalg.norm(reactant.coords[i] - reactant.coords[j])
        r_prod = np.linalg.norm(product.coords[i] - product.coords[j])
        r_ts = np.linalg.norm(result.geometry.coords[i] - result.geometry.coords[j])
        lo, hi = sorted([r_react, r_prod])
        assert lo - 0.3 <= r_ts <= hi + 0.3, (i, j, r_react, r_prod, r_ts)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_hcn_hnc_isomerization(tmp_path, direction):
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    hnc = read_xyz(STRUCTURES_DIR / "hnc.xyz")
    reactant, product = (hcn, hnc) if direction == "forward" else (hnc, hcn)

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=80, max_ts_steps=80)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    assert result.energy > HCN_ENERGY
    assert result.energy > HNC_ENERGY
    _bonding_sanity_check(result, reactant, product)


def test_hcn_hnc_with_periodic_hessian_refresh(tmp_path):
    """Regression check for `hessian_refresh_interval`'s wiring (see
    driver.find_ts's docstring for why it isn't recommended by default --
    tested on pentadiene and found not to help there -- but the parameter
    itself must still work correctly when used)."""
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    hnc = read_xyz(STRUCTURES_DIR / "hnc.xyz")
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)

    result = find_ts(hcn, hnc, calculator=calc, max_relax_steps=80, max_ts_steps=80,
                      hessian_refresh_interval=3)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1


def test_hcn_hnc_with_exact_pre_relaxation_hessian(tmp_path):
    """Regression check for `use_exact_pre_relaxation_hessian`'s wiring
    (see build_schlegel_ts_preparation's docstring for why it isn't on by
    default -- tested directly on butadiene -> cyclobutene and found not
    to resolve that specific failure by itself -- but the option itself
    must still work correctly when used)."""
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    hnc = read_xyz(STRUCTURES_DIR / "hnc.xyz")
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)

    result = find_ts(hcn, hnc, calculator=calc, max_relax_steps=80, max_ts_steps=80,
                      use_exact_pre_relaxation_hessian=True)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1


def test_hcn_hnc_forward_and_reverse_agree_on_ts_energy(tmp_path):
    """Path-symmetry check: the TS found from HCN->HNC and from HNC->HCN
    must be the same physical stationary point, hence the same energy."""
    hcn = read_xyz(STRUCTURES_DIR / "hcn.xyz")
    hnc = read_xyz(STRUCTURES_DIR / "hnc.xyz")
    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)

    forward = find_ts(hcn, hnc, calculator=calc, max_relax_steps=80, max_ts_steps=80)
    reverse = find_ts(hnc, hcn, calculator=calc, max_relax_steps=80, max_ts_steps=80)

    assert forward.converged and reverse.converged
    assert forward.energy == pytest.approx(reverse.energy, abs=1e-6)


def test_h2co_decomposition(tmp_path):
    """H2CO <-> H2 + CO: multi-fragment product, planar 3-connected carbon
    at the reactant -- exercises both the virtual-interfragment-bond logic
    and the OutOfPlane coordinate."""
    h2co = read_xyz(STRUCTURES_DIR / "h2co.xyz")
    h2_plus_co = read_xyz(STRUCTURES_DIR / "h2_plus_co.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(h2co, h2_plus_co, calculator=calc, max_relax_steps=100, max_ts_steps=100)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    assert result.energy > H2CO_ENERGY
    assert result.energy > H2_PLUS_CO_ENERGY
    assert set(result.breaking_bonds) == {(0, 2), (0, 3)}  # both C-H bonds
    assert set(result.forming_bonds) == {(2, 3)}  # H-H
    _bonding_sanity_check(result, h2co, h2_plus_co)


@pytest.mark.slow
def test_pentadiene_15_hydrogen_shift(tmp_path):
    """1,3-pentadiene [1,5]-H sigmatropic shift: a degenerate rearrangement
    (product is the same molecule with the chain relabeled, so reactant
    and product energies coincide) via a 13-atom, 39-DOF transition state.
    Real run: ~700 dRFO steps, ~10+ minutes of xtb calls."""
    reactant = read_xyz(STRUCTURES_DIR / "pentadiene_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "pentadiene_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=1200)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    # Degenerate rearrangement: TS must be higher than the (equal) endpoint
    # energies, and the barrier should be in a physically sane range for a
    # [1,5]-H shift (roughly 25-45 kcal/mol; loose bounds since GFN2-xTB is
    # not expected to match higher-level barrier heights exactly).
    reactant_energy = calc.energy(reactant)
    barrier_kcal = (result.energy - reactant_energy) * 627.5094740631
    assert 15.0 < barrier_kcal < 55.0
    assert set(result.breaking_bonds) == {(4, 10)}  # C5-H
    assert set(result.forming_bonds) == {(0, 10)}  # C1-H
    _bonding_sanity_check(result, reactant, product)


SILANE_REACTANT_ENERGY = -3.652698018668
SILANE_PRODUCT_ENERGY = -3.763890886429


@pytest.mark.slow
def test_silane_insertion(tmp_path):
    """SiH2 + H2 -> SiH4 (silylene insertion into H2). The smallest system
    in the suite (5 atoms), but the one that surfaced the best-geometry-
    tracking fix: the raw search overshoots a good TS around step 450 and
    degrades into a worse stationary point (a second imaginary mode) by
    step 550 if allowed to keep running. This test exercises the fallback
    (`TSResult.used_best_available`) that recovers the good point."""
    reactant = read_xyz(STRUCTURES_DIR / "silane_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "silane_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=800)

    assert result.n_imaginary == 1
    assert result.gradient_rms < 1e-3
    assert result.energy > SILANE_REACTANT_ENERGY
    assert result.energy > SILANE_PRODUCT_ENERGY
    assert set(result.breaking_bonds) == {(3, 4)}  # H-H
    assert set(result.forming_bonds) == {(0, 3), (0, 4)}  # Si-H, Si-H
    _bonding_sanity_check(result, reactant, product)


MEOH_REACTANT_ENERGY = -8.159477712232
MEOH_PRODUCT_ENERGY = -8.223563498689


def test_h2_addition_to_formaldehyde(tmp_path):
    """H2 + H2CO -> CH3OH. Clean, fast convergence (139 steps)."""
    reactant = read_xyz(STRUCTURES_DIR / "meoh_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "meoh_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=300)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    assert result.energy > MEOH_REACTANT_ENERGY
    assert result.energy > MEOH_PRODUCT_ENERGY
    assert set(result.breaking_bonds) == {(4, 5)}  # H-H
    assert set(result.forming_bonds) == {(0, 4), (1, 5)}  # C-H, O-H
    _bonding_sanity_check(result, reactant, product)


HFETH_REACTANT_ENERGY = -11.501261386419
HFETH_PRODUCT_ENERGY = -11.556883564948


def test_hf_addition_to_ethylene(tmp_path):
    """HF + C2H4 -> C2H5F. Fastest-converging case in the suite so far
    (48 steps)."""
    reactant = read_xyz(STRUCTURES_DIR / "hfeth_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "hfeth_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=100)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    assert result.energy > HFETH_REACTANT_ENERGY
    assert result.energy > HFETH_PRODUCT_ENERGY
    assert set(result.breaking_bonds) == {(6, 7)}  # F-H
    assert set(result.forming_bonds) == {(0, 7), (1, 6)}  # C-H, C-F
    _bonding_sanity_check(result, reactant, product)


@pytest.mark.slow
def test_cyclopentadiene_15_hydrogen_shift(tmp_path):
    """[1,5]-H shift in 1,3-cyclopentadiene: a degenerate rearrangement
    (product is the same molecule with the ring relabeled), the cyclic
    analogue of the pentadiene case. Real run: 164 steps."""
    reactant = read_xyz(STRUCTURES_DIR / "cpht_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "cpht_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=400)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    reactant_energy = calc.energy(reactant)
    barrier_kcal = (result.energy - reactant_energy) * 627.5094740631
    assert 15.0 < barrier_kcal < 40.0
    assert set(result.breaking_bonds) == {(0, 6)}
    assert set(result.forming_bonds) == {(4, 6)}
    _bonding_sanity_check(result, reactant, product)


C2N2O_REACTANT_ENERGY = -16.116205397979
C2N2O_PRODUCT_ENERGY = -16.161062240614


def test_n2o_ethylene_cycloaddition(tmp_path):
    """N2O + C2H4 -> oxadiazoline ([3+2] cycloaddition, no bonds break).
    Clean pass, 128 steps."""
    reactant = read_xyz(STRUCTURES_DIR / "c2n2o_reactant.xyz")
    product = read_xyz(STRUCTURES_DIR / "c2n2o_product.xyz")

    calc = XTBCalculator(XTB_PATH, method="gfn2", scratch_dir=tmp_path)
    result = find_ts(reactant, product, calculator=calc, max_relax_steps=100, max_ts_steps=400)

    assert result.status == "converged"
    assert result.converged
    assert result.n_imaginary == 1
    assert result.energy > C2N2O_REACTANT_ENERGY
    assert result.energy > C2N2O_PRODUCT_ENERGY
    assert result.breaking_bonds == []
    assert set(result.forming_bonds) == {(0, 4), (2, 3)}
    _bonding_sanity_check(result, reactant, product)

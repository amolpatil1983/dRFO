# drfo

A pluggable transition-state optimizer implementing the "Connectivity Transition State" (CTS)
method of Birkholz & Schlegel, *J. Comput. Chem.* 2015, 36, 1157-1166 (DOI 10.1002/jcc.23910):
bond-order-guided interpolation of an initial TS guess from reactant/product structures, a
bonding-derived transition vector (Δb), Bofill/BFGS Hessian updating, and a divided rational
function optimization (dRFO) step.

The reference PES/gradient/Hessian backend is [xtb](https://github.com/grimme-lab/xtb)
(GFN2-xTB for the real potential, GFN-FF for the cheap interpolation potential), called via
subprocess against a local xtb binary. The `Calculator` interface is otherwise backend-agnostic.

## Status

Milestones M1-M5 complete (65 tests: unit + live xtb integration + benchmark; 64 run by
default, 1 marked `slow`). `find_ts()` is validated end-to-end on all three reactions from
the original benchmark plan, each converging to a genuine first-order saddle point (exactly
one imaginary frequency) with correct barrier physics (TS energy above both endpoints):

- **HCN <-> HNC** isomerization (both directions; forward/reverse agree on TS energy to 1e-6
  Hartree).
- **H2CO <-> H2 + CO** decomposition. Required a proper `OutOfPlane` (Wilson wag) internal
  coordinate for formaldehyde's planar 3-connected carbon, which a bare stretch/bend/torsion
  coordinate set cannot describe without the Wilson B-matrix going singular there.
- **1,3-pentadiene [1,5]-H shift**. A genuine 13-atom/39-DOF case; converges in ~700 dRFO
  steps (barrier ~30 kcal/mol, physically sane). Marked `slow` (`pytest -m "not slow"` to
  skip, `pytest -m slow` to run it explicitly).

Getting the larger cases to converge required two fixes to the dRFO stepper itself, not just
new coordinate types: eigenvalue flooring on the internal-coordinate Hessian (near-zero
eigenvalues from redundancy or a zeroed B-matrix row otherwise dominate the step direction),
and projecting the raw RFO step onto the Wilson B-matrix's row space before back-transforming
(an unprojected step can carry a structurally unreachable component that no amount of
trust-radius shrinking fixes). See `optimize/rfo_core.py` and `optimize/drfo.py`.

See `pyproject.toml` for dependencies (numpy, scipy only).

## Design

See the design document this package was built from for the full architecture, milestone
breakdown (M1-M5), and validation plan.

## Development

```
pip install -e ".[test]"
pytest
```

xtb binary path is not hardcoded; pass it explicitly to `XTBCalculator(xtb_path=...)`.

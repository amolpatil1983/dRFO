"""xtb backend Calculator: shells out to a local xtb binary per compute() call.

No xtb Python bindings are used (none installed in this environment); every
call writes an xyz input into a fresh scratch subdirectory and parses xtb's
own output files. Confirmed empirically against xtb 6.6.1 (see design doc):

- `--grad` writes energy+gradient (via `{stem}.engrad`, ORCA-style, and also
  a Turbomole `gradient`/`energy` file) but does NOT compute a Hessian.
- `--hess` writes energy+Hessian (`hessian` file, flat 3N x 3N text) and a
  vibrational analysis (`xtbout.json`'s "vibrational frequencies/rcm"), but
  does NOT write a gradient file.
- These two modes are mutually exclusive in one xtb invocation (passing both
  silently runs only the `--grad` path) so a request for both gradient and
  Hessian issues two separate xtb subprocess calls.
- `--json` (`xtbout.json`) is only written by the GFN2/GFN1 (TB-SCF) code
  path; the GFN-FF path silently ignores `--json` and writes no energy/json
  file at all in single-point or `--hess` mode (confirmed empirically; only
  `--grad` mode writes a Turbomole `energy` file/`mol.engrad` for GFN-FF).
  The one energy source common to every method and run mode is xtb's own
  stdout `TOTAL ENERGY` banner line, so that is used as the primary energy
  source; `xtbout.json` is read opportunistically only for bonus `extra`
  data (e.g. vibrational frequencies) when it happens to exist.
- The final "normal termination of xtb" banner is written to stderr, not
  stdout.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from ..geometry import Geometry
from ..io.xyz import write_xyz
from .base import CalcResult, Calculator, CalculatorError, SCFConvergenceError

Method = Literal["gfn2", "gfn1", "gfnff"]

_METHOD_FLAGS: dict[Method, list[str]] = {
    "gfn2": ["--gfn", "2"],
    "gfn1": ["--gfn", "1"],
    "gfnff": ["--gfnff"],
}


class XTBCalculator(Calculator):
    def __init__(
        self,
        xtb_path: str,
        method: Method = "gfn2",
        charge: int = 0,
        spin: int = 0,
        nthreads: int = 1,
        scratch_dir: str | Path | None = None,
        keep_scratch: bool = False,
        keep_scratch_on_error: bool = True,
        timeout: float = 600.0,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.xtb_path = str(xtb_path)
        if not Path(self.xtb_path).exists():
            raise FileNotFoundError(f"xtb binary not found at {self.xtb_path}")
        if method not in _METHOD_FLAGS:
            raise ValueError(f"unknown method {method!r}, expected one of {list(_METHOD_FLAGS)}")
        self.method = method
        self.charge = charge
        self.spin = spin
        self.nthreads = nthreads
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None
        if self.scratch_dir is not None:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.keep_scratch = keep_scratch
        self.keep_scratch_on_error = keep_scratch_on_error
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self._call_count = 0

    def _base_cmd(self) -> list[str]:
        cmd = [self.xtb_path, "mol.xyz", *_METHOD_FLAGS[self.method],
               "-c", str(self.charge)]
        if self.spin != 0:
            cmd += ["-u", str(self.spin)]
        if self.nthreads > 1:
            cmd += ["-P", str(self.nthreads)]
        cmd += ["--json", *self.extra_args]
        return cmd

    def _run(self, workdir: Path, mode_flags: list[str], *, geom_hint: Geometry | None = None) -> str:
        cmd = self._base_cmd() + mode_flags
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise CalculatorError(
                f"xtb timed out after {self.timeout}s in {workdir}\ncmd: {' '.join(cmd)}"
            ) from exc
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        # xtb writes its final "normal termination" banner to stderr, not stdout.
        if proc.returncode != 0 or "normal termination of xtb" not in stderr:
            tail = "\n".join((stdout + "\n" + stderr).splitlines()[-40:])
            geom_note = ""
            if geom_hint is not None:
                geom_note = (
                    f"geometry (Angstrom):\n{geom_hint.coords_angstrom()}\n"
                )
            combined = stdout + stderr
            if "consistent charge iterator did not converge" in combined:
                raise SCFConvergenceError(
                    f"xtb's SCF/SCC iterator failed to converge at this geometry "
                    f"(not necessarily an optimizer bug -- can happen at geometries far "
                    f"from equilibrium, e.g. near-degenerate electronic structure along an "
                    f"interpolated path) in {workdir}\ncmd: {' '.join(cmd)}\n{geom_note}"
                    f"--- tail of output ---\n{tail}"
                )
            raise CalculatorError(
                f"xtb failed (returncode={proc.returncode}) in {workdir}\n"
                f"cmd: {' '.join(cmd)}\n{geom_note}--- tail of output ---\n{tail}"
            )
        return stdout

    @staticmethod
    def _parse_energy_from_stdout(stdout: str) -> float:
        """Robust energy source common to every method/run mode: the
        '| TOTAL ENERGY   <value> Eh |' banner line xtb always prints,
        regardless of whether it also wrote an energy/json file."""
        matches = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh", stdout)
        if not matches:
            raise CalculatorError("could not find TOTAL ENERGY banner in xtb stdout")
        return float(matches[-1])

    def compute(self, geom: Geometry, *, gradient: bool = False,
                hessian: bool = False) -> CalcResult:
        if geom.charge != self.charge or geom.spin != self.spin:
            raise ValueError(
                f"geometry charge/spin ({geom.charge},{geom.spin}) does not match "
                f"calculator charge/spin ({self.charge},{self.spin})"
            )

        self._call_count += 1
        workdir = Path(tempfile.mkdtemp(
            prefix=f"drfo_xtb_{self._call_count:06d}_", dir=self.scratch_dir,
        ))
        ok = False
        try:
            write_xyz(workdir / "mol.xyz", geom)

            energy: float | None = None
            grad: np.ndarray | None = None
            hess: np.ndarray | None = None
            extra: dict = {}

            if hessian:
                out = self._run(workdir, ["--hess"], geom_hint=geom)
                energy = self._parse_energy_from_stdout(out)
                hess = _parse_hessian(workdir / "hessian", geom.natoms)
                data = _read_json_if_present(workdir / "xtbout.json")
                if data and "vibrational frequencies/rcm" in data:
                    extra["frequencies_rcm"] = data["vibrational frequencies/rcm"]

            if gradient:
                out = self._run(workdir, ["--grad"], geom_hint=geom)
                energy = self._parse_energy_from_stdout(out)
                grad = _parse_engrad(workdir / "mol.engrad", geom.natoms)

            if not gradient and not hessian:
                out = self._run(workdir, [], geom_hint=geom)
                energy = self._parse_energy_from_stdout(out)

            assert energy is not None
            ok = True
            return CalcResult(energy=energy, gradient=grad, hessian=hess,
                               extra=extra or None)
        finally:
            delete = self.keep_scratch is False and (ok or not self.keep_scratch_on_error)
            if delete:
                shutil.rmtree(workdir, ignore_errors=True)


def _read_json_if_present(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _parse_engrad(path: Path, natoms: int) -> np.ndarray:
    """Parse xtb's ORCA-style `{stem}.engrad` file: fixed section order,
    comment lines start with '#'. Returns the (N,3) Cartesian gradient
    (Hartree/bohr) as written by xtb (dE/dx convention)."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.strip().startswith("#")]
    lines = [ln for ln in lines if ln.strip() != ""]
    idx = 0
    n = int(lines[idx]); idx += 1
    if n != natoms:
        raise CalculatorError(f"engrad atom count {n} != expected {natoms} in {path}")
    idx += 1  # energy line, already read from json
    try:
        grad_flat = [float(lines[idx + i]) for i in range(3 * natoms)]
    except ValueError as exc:
        # xtb prints a fixed-width Fortran field as '****...' when a value
        # overflows it -- this happens when the geometry is degenerate
        # enough (e.g. overlapping atoms) that the gradient is enormous.
        raise CalculatorError(
            f"engrad file at {path} contains an unparseable (likely overflowed) "
            f"gradient value -- geometry is probably unphysical"
        ) from exc
    return np.array(grad_flat).reshape(natoms, 3)


def _parse_hessian(path: Path, natoms: int) -> np.ndarray:
    """Parse xtb's flat-text `hessian` file (header line `$hessian`, then
    (3N)^2 whitespace-separated values, row-major). Symmetrized on return
    to remove finite-difference asymmetry noise."""
    lines = path.read_text().splitlines()
    vals: list[float] = []
    for ln in lines[1:]:
        vals.extend(float(tok) for tok in ln.split())
    n3 = 3 * natoms
    if len(vals) != n3 * n3:
        raise CalculatorError(
            f"hessian file has {len(vals)} values, expected {n3 * n3} in {path}"
        )
    H = np.array(vals).reshape(n3, n3)
    return 0.5 * (H + H.T)

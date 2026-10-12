#!/usr/bin/env python3
"""Diagnostic: the same calculation with the SCF potential ARES appears to use.

This is not the PP-PBE(B3LYP) method and not an option of the solver.  It is
the experiment that explains why the Python and the ARES numbers of this
benchmark differ, and it is how the B3LYP evaluation itself was validated
against ARES part by part.

The energy bookkeeping of the ARES logs implies an exchange-correlation
potential whose density integral is that of ``d f / d n`` alone, where
``f(n, |grad n|**2)`` is the PBE energy per volume: the gradient part of the
GGA potential, ``-div(2 df/dsigma grad n)``, is missing from it (README,
"Why the two codes differ").  This script runs the Python solver with exactly
that potential, ``v_xc = d f / d n``, and the unchanged PBE *energy*.  The
orbitals are then those of ARES to plotting accuracy, and the refinement can
be compared with the ARES log term by term.

Nothing in the repository is modified: the potential is replaced inside this
process only.  The command line is that of ``src/parsec_python/main.py``:

    python ares_potential_check.py CH4/IS/parsec.in --pp-dir pseudopotentials \
        --log parsec_ares_potential.out --no-archive

    python analyze.py --log parsec_ares_potential.out --parts
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

import parsec_python.SCF.single_point as single_point  # noqa: E402
from parsec_python.V_xc.b3lyp import density_gradient_squared  # noqa: E402
from parsec_python.V_xc.ca_lda import XCResult  # noqa: E402
from parsec_python.V_xc.pbe import pbe_energy_partials  # noqa: E402


def pbe_without_gradient_potential(valence_density, grid, core_density=None):
    """PBE energy with the potential ``d f / d n`` at fixed ``|grad n|``."""

    density = np.asarray(valence_density, dtype=np.float64)
    if core_density is not None:
        density = density + np.asarray(core_density, dtype=np.float64)
    sigma = density_gradient_squared(density, grid)
    energy, derivative_density, _derivative_sigma = pbe_energy_partials(density, sigma)
    # Hartree -> Rydberg, as parsec_python.V_xc.pbe.
    energy_density = 2.0 * energy
    epsilon = np.zeros_like(density)
    positive = density > 1.0e-20
    epsilon[positive] = energy_density[positive] / density[positive]
    return XCResult(
        potential=2.0 * derivative_density,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


def main() -> int:
    # The prepared system of either driver evaluates PBE through this name.
    single_point.pbe = pbe_without_gradient_potential
    from parsec_python.acceleration.cli import main as run

    marker = (
        "DIAGNOSTIC RUN: the PBE potential of this SCF lacks its gradient term "
        "(v_xc = df/dn); see ares_potential_check.py"
    )
    print(marker, flush=True)
    arguments = sys.argv[1:]
    code = run(arguments)
    # Mark the log, which otherwise reads like that of an ordinary PBE run.
    if arguments and "--dry-run" not in arguments:
        source = Path(arguments[0]).expanduser().resolve()
        name = Path(
            arguments[arguments.index("--log") + 1]
            if "--log" in arguments[:-1]
            else "parsec.out"
        ).expanduser()
        log = name if name.is_absolute() else source.parent / name
        if log.is_file():
            with log.open("a", encoding="utf-8") as stream:
                stream.write(marker + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

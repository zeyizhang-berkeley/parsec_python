#!/usr/bin/env python3
"""Compare the Python PBE core-hole calculations with the supplied ARES runs.

The comparable quantity is the reference-corrected all-electron energy.  The
ordinary and C-1s-hole pseudopotentials have different ionic charges, so their
raw pseudopotential total energies do not share an energy zero.
"""

from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARES = HERE.parent / "0_ARES_CH4_CF4" / "test_CH4_CF4" / "pbe"
MOLECULES = ("CH4", "CF4")
STATES = ("IS", "FS_1s")

PYTHON_ENERGY = re.compile(
    r"Reference-corrected all-electron total\s*:\s*([-+0-9.Ee]+)\s*\[eV\]"
)
ARES_ENERGY = re.compile(r"\*AE energy \(test\)->\s*([-+0-9.Ee]+)")


def _last_match(path: Path, pattern: re.Pattern[str]) -> float | None:
    try:
        matches = pattern.findall(path.read_text(errors="replace"))
    except OSError:
        return None
    return float(matches[-1]) if matches else None


def _load_python() -> dict[str, dict[str, float | None]]:
    return {
        molecule: {
            state: _last_match(HERE / molecule / state / "parsec.out", PYTHON_ENERGY)
            for state in STATES
        }
        for molecule in MOLECULES
    }


def _load_ares() -> dict[str, dict[str, float | None]]:
    return {
        molecule: {
            state: _last_match(ARES / molecule / state / "ares.log", ARES_ENERGY)
            for state in STATES
        }
        for molecule in MOLECULES
    }


def _binding_energy(values: dict[str, float | None]) -> float | None:
    initial, final = values["IS"], values["FS_1s"]
    if initial is None or final is None:
        return None
    return final - initial


def _number(value: float | None, width: int = 15) -> str:
    return f"{value:{width}.8f}" if value is not None else f"{'unfinished':>{width}s}"


def main() -> None:
    python = _load_python()
    ares = _load_ares()

    print("Reference-corrected all-electron energies (eV)")
    print(f"{'system':<10s} {'Python':>15s} {'ARES':>15s} {'Python-ARES':>15s}")
    print("-" * 58)
    for molecule in MOLECULES:
        for state in STATES:
            py_value, ref_value = python[molecule][state], ares[molecule][state]
            delta = None if py_value is None or ref_value is None else py_value - ref_value
            print(
                f"{molecule + ' ' + state:<10s} {_number(py_value)} "
                f"{_number(ref_value)} {_number(delta)}"
            )

    py_be = {molecule: _binding_energy(python[molecule]) for molecule in MOLECULES}
    ref_be = {molecule: _binding_energy(ares[molecule]) for molecule in MOLECULES}

    print("\nC 1s Delta-SCF binding energies (eV)")
    print(f"{'molecule':<10s} {'Python':>15s} {'ARES':>15s} {'Python-ARES':>15s}")
    print("-" * 58)
    for molecule in MOLECULES:
        delta = (
            None
            if py_be[molecule] is None or ref_be[molecule] is None
            else py_be[molecule] - ref_be[molecule]
        )
        print(
            f"{molecule:<10s} {_number(py_be[molecule])} "
            f"{_number(ref_be[molecule])} {_number(delta)}"
        )

    py_shift = (
        None
        if py_be["CH4"] is None or py_be["CF4"] is None
        else py_be["CF4"] - py_be["CH4"]
    )
    ref_shift = (
        None
        if ref_be["CH4"] is None or ref_be["CF4"] is None
        else ref_be["CF4"] - ref_be["CH4"]
    )
    shift_delta = (
        None if py_shift is None or ref_shift is None else py_shift - ref_shift
    )
    print("\nCF4-CH4 C 1s chemical shift (eV)")
    print(f"Python       {_number(py_shift)}")
    print(f"ARES         {_number(ref_shift)}")
    print(f"Python-ARES {_number(shift_delta)}")


if __name__ == "__main__":
    main()

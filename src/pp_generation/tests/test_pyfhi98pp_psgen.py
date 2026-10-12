"""Tests of the pseudopotential construction of the pure-Python FHI98PP port.

Everything after the all-electron atom in ``fhipp.x``: default radii and
reference energies, Troullier-Martins and Hamann construction, nonlinear
core correction, unscreening, and the files assembled by the ``psgen``
script (``.dat``, ``.cpi``).

The references under ``data/pyfhi98pp/pp_*`` were written by the original
Fortran program (``fhipp.x`` rev 1998 through the ``psgen`` script;
gfortran 13.3 -O2, glibc 2.39, OpenBLAS 0.3.26), so the tests run without
the Fortran:

* ``pp_<case>.ini``   input as the program read it (``fort.22``)
* ``pp_<case>.dat``   the ``.dat`` file without its ``#info`` line
* ``pp_<case>.cpi``   the pseudopotential table (three cases)
* ``pp_dgesv_calls.json``  calls of LAPACK ``dgesv`` logged from the
  running Fortran program (inputs and outputs as hexadecimal floats)

Comparison rules.  With the math library of the reference build the port
reproduces every file bit for bit.  Other math libraries differ in the last
bit of ``exp``/``log``/``pow`` now and then; the checks therefore allow one
unit in the last printed digit of the ``.dat`` file, one self-consistency
iteration more or less, and 1e-8 relative (plus 1e-10 absolute) on the
14-digit numbers of the ``.cpi`` table and the 17-digit pseudocore lines.
Set ``PYFHI98PP_EXACT=1`` to demand identical text of the main part of the
``.dat`` files instead.  The linear solver is pure arithmetic and is always
compared bit for bit.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from pp_generation.pyfhi98pp import FortranStop, parse_ini
from pp_generation.pyfhi98pp.defrtm import defrtm
from pp_generation.pyfhi98pp.fortran_list import list_record, unit6_lines
from pp_generation.pyfhi98pp.lapack import _fma_exact, dgesv
from pp_generation.pyfhi98pp.psgen import CpiData, fhipp, format_cpi, format_dat, psgen

DATA = Path(__file__).parent / "data" / "pyfhi98pp"
EXACT = os.environ.get("PYFHI98PP_EXACT") == "1"
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([EeDd][-+]?\d+)?$")
_CORE_START = " Improved Polynomial Pseudocore -- dnlcc7"


def _read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


def _split(lines: list[str]) -> tuple[list[str], list[str]]:
    """Unit 23 part and the appended pseudocore part (unit 24) of a ``.dat``."""
    if _CORE_START in lines:
        k = lines.index(_CORE_START)
        return lines[:k], lines[k:]
    return lines, []


def _run(case: str, **kwargs):
    return psgen(_read(f"pp_{case}.ini"), out="pp", write=False, **kwargs)


@unittest.skipUnless(DATA.is_dir(), "pyfhi98pp reference data are not available")
class ReferenceCase(unittest.TestCase):
    """Helpers to compare with the files written by the Fortran program."""

    def assert_dat(self, text: str, ref: str) -> None:
        """A ``.dat`` text (without ``#info`` line) against the Fortran one."""
        mine, mine_core = _split(text.splitlines())
        want, want_core = _split(ref.splitlines())
        self.assertEqual(len(mine), len(want))
        if EXACT:
            self.assertEqual(mine, want)
        for a, b in zip(mine, want):
            if a == b:
                continue
            ta, tb = a.split(), b.split()
            self.assertEqual(len(ta), len(tb), msg=f"{a!r} / {b!r}")
            for k, (x, y) in enumerate(zip(ta, tb)):
                if x == y:
                    continue
                self.assertTrue(_NUMBER.match(x) and _NUMBER.match(y), msg=f"{a!r} / {b!r}")
                if "number of iterations" in b and k == 3:
                    self.assertLessEqual(abs(int(x) - int(y)), 1, msg=b)
                elif "E" in y.upper():
                    self.assertLessEqual(abs(float(x) - float(y)), 1e-5 * abs(float(y)), msg=b)
                else:
                    decimals = len(y.partition(".")[2])
                    self.assertLessEqual(abs(float(x) - float(y)), 1.5 * 10.0 ** (-decimals),
                                         msg=f"{a!r} / {b!r}")
        # list-directed records of dnlcc7 (17 significant digits)
        self.assertEqual(len(mine_core), len(want_core))
        for a, b in zip(mine_core, want_core):
            if a == b:
                continue
            ta, tb = a.split(), b.split()
            self.assertEqual(len(ta), len(tb), msg=f"{a!r} / {b!r}")
            self.assertEqual(len(a), len(b), msg=f"{a!r} / {b!r}")
            for x, y in zip(ta, tb):
                if x != y:
                    self.assertLessEqual(abs(float(x) - float(y)), 1e-8 * abs(float(y)) + 1e-10,
                                         msg=f"{a!r} / {b!r}")

    def assert_cpi(self, text: str, ref: str) -> None:
        """A ``.cpi`` table against the Fortran one."""
        mine, want = text.splitlines(), ref.splitlines()
        self.assertEqual(len(mine), len(want))
        self.assertEqual(mine[:11], want[:11])          # charge, channels, unused parameters
        for a, b in zip(mine[11:], want[11:]):
            if a == b:
                continue
            ta, tb = a.split(), b.split()
            self.assertEqual(len(ta), len(tb), msg=f"{a!r} / {b!r}")
            self.assertEqual(len(a), len(b), msg=f"{a!r} / {b!r}")
            for x, y in zip(ta, tb):
                if x != y:
                    self.assertLessEqual(abs(float(x) - float(y)), 1e-8 * abs(float(y)) + 1e-10,
                                         msg=f"{a!r} / {b!r}")

    def check(self, case: str, cpi: bool = False, **kwargs):
        res = _run(case, **kwargs)
        self.assert_dat(res.dat_text.partition("\n")[2], _read(f"pp_{case}.dat"))
        if cpi:
            self.assert_cpi(res.cpi_text, _read(f"pp_{case}.cpi"))
        return res


# ---------------------------------------------------------------------------
# linear solver, list-directed output, default radii
# ---------------------------------------------------------------------------

class LinearSolverTests(unittest.TestCase):
    def test_fma_is_correctly_rounded(self) -> None:
        rng = random.Random(3)
        for _ in range(3000):
            a = rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40)
            b = rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40)
            if rng.random() < 0.5:      # cancellation: the case a rounded product gets wrong
                c = -a * b * (1 + rng.uniform(-1, 1) * 2.0 ** rng.randint(-60, -1))
            else:
                c = rng.uniform(-1, 1) * 2.0 ** rng.randint(-90, 90)
            self.assertEqual(_fma_exact(a, b, c), float(Fraction(a) * Fraction(b) + Fraction(c)))
        self.assertEqual(_fma_exact(0.0, 5.0, -3.0), -3.0)
        self.assertEqual(_fma_exact(2.0, 0.5, -1.0), 0.0)
        self.assertEqual(_fma_exact(1e308, 10.0, 0.0), float("inf"))
        self.assertTrue(np.isnan(_fma_exact(float("nan"), 1.0, 1.0)))
        tiny = _fma_exact(2.0 ** -600, 2.0 ** -470, 2.0 ** -1074)
        self.assertEqual(tiny, float(Fraction(2) ** -1070 + Fraction(2) ** -1074))

    @unittest.skipUnless((DATA / "pp_dgesv_calls.json").is_file(), "no logged dgesv calls")
    def test_dgesv_reproduces_the_reference_library(self) -> None:
        """LU factors, pivots and solution, bit for bit, for calls logged from
        fhipp.x (including second factorizations of an already factored matrix)."""
        calls = json.loads(_read("pp_dgesv_calls.json"))
        self.assertGreater(len(calls), 30)
        self.assertTrue(any(c.get("factored_input") for c in calls))
        for call in calls:
            n = call["n"]
            a = [[float.fromhex(x) for x in col] for col in call["a"]]
            b = [float.fromhex(x) for x in call["b"]]
            ipiv, info = dgesv(n, a, b)
            self.assertEqual(info, call["info"])
            self.assertEqual(ipiv, call["ipiv"])
            self.assertEqual([[x.hex() for x in col] for col in a], call["lu"])
            self.assertEqual([x.hex() for x in b], call["x"])

    def test_dgesv_singular_and_pivoting(self) -> None:
        a = [[0.0, 2.0], [1.0, 1.0]]            # columns; needs a row interchange
        b = [3.0, 5.0]
        ipiv, info = dgesv(2, a, b)
        self.assertEqual((ipiv, info), ([2, 2], 0))
        self.assertEqual(b, [1.0, 3.0])
        a = [[1.0, 2.0], [2.0, 4.0]]
        b = [1.0, 1.0]
        ipiv, info = dgesv(2, a, b)
        self.assertEqual(info, 2)               # exactly singular: U(2,2) = 0
        self.assertEqual(b, [1.0, 1.0])         # right-hand side untouched, as in LAPACK


class ListDirectedOutputTests(unittest.TestCase):
    def test_records_match_gfortran(self) -> None:
        # strings written by gfortran 13 for the same items
        self.assertEqual(list_record("a0 ", 0.19673093313945220),
                         " a0   0.19673093313945220     ")
        self.assertEqual(list_record("a1 ", 0.0), " a1    0.0000000000000000     ")
        self.assertEqual(list_record("a6 ", -4.8603766903140867e-3),
                         " a6   -4.8603766903140867E-003")
        self.assertEqual(list_record("Pseudocore matching at i", 288, " r ", 1.3745903929284453),
                         " Pseudocore matching at i         288  r    1.3745903929284453     ")
        self.assertEqual(list_record(4.4281263708392847e-3, 4.4281263708385588e-3, "density"),
                         "   4.4281263708392847E-003   4.4281263708385588E-003 density")
        self.assertEqual(list_record(0.30170456901528397, 0.30170456901528420, "2nd derivative"),
                         "  0.30170456901528397       0.30170456901528420      2nd derivative")
        self.assertEqual(list_record("& ncpp -- warning: valency mismatch?", 3.0, 2.0),
                         " & ncpp -- warning: valency mismatch?   3.0000000000000000        "
                         "2.0000000000000000     ")
        self.assertEqual(list_record("&  iteration j k       ", 1, 21),
                         " &  iteration j k                  1          21")
        self.assertEqual(list_record(float("nan"), "x"), "                       NaN x")
        self.assertEqual(list_record("a6 ", -0.0), " a6   -0.0000000000000000     ")
        self.assertEqual(list_record("r", 76.654855215310803), " r   76.654855215310803     ")

    def test_stage1_messages_get_the_fortran_layout(self) -> None:
        self.assertEqual(unit6_lines("& dftseq - stop: convergence error"),
                         [" & dftseq - stop: convergence error"])
        self.assertEqual(unit6_lines("&   nint n l                 201 3 2"),
                         [" &   nint n l                         201           3           2",
                          " &   eigenvalue log to file err.dftseq_e",
                          " &   potential  log to file err.dftseq_v"])
        self.assertEqual(unit6_lines(" & already formatted"), [" & already formatted"])


class DefaultRadiusTests(unittest.TestCase):
    def test_table(self) -> None:
        self.assertEqual(defrtm(3, 0, 14), 0.01 * 173)
        self.assertEqual(defrtm(3, 1, 14), 0.01 * 190)
        self.assertEqual(defrtm(3, 2, 14), 0.01 * 203)
        self.assertEqual(defrtm(4, 3, 14), 0.01 * 203)      # l = 3 falls back to the d entry
        self.assertEqual(defrtm(2, 1, 6), 0.01 * 150)       # only s tabulated for C
        self.assertEqual(defrtm(4, 0, 19), 0.01 * 330)
        self.assertEqual(defrtm(3, 0, 19), 0.01 * 180)      # semicore K: n = 3
        self.assertEqual(defrtm(5, 3, 58), 0.01 * 320)      # the only f entry (Ce)

    def test_slips_of_the_source_are_kept(self) -> None:
        self.assertEqual(defrtm(3, 1, 12), 0.01 * 250)      # r(2,12) assigned twice
        self.assertEqual(defrtm(3, 2, 12), 0.01 * 250)      # r(3,12) never assigned
        self.assertEqual(defrtm(3, 0, 17), 0.01 * 160)      # r(1,17) assigned twice
        self.assertEqual(defrtm(3, 1, 17), 0.01 * 160)      # r(2,17) never assigned
        self.assertEqual(defrtm(4, 2, 42), 0.01 * 244)      # Tc block assigns r(3,42)
        self.assertEqual(defrtm(4, 2, 43), 0.01 * 265)      # ... so Tc d falls back to p
        self.assertEqual(defrtm(4, 2, 49), 0.01 * 234)      # Sn block assigns r(3,49) for n = 4
        self.assertEqual(defrtm(4, 2, 50), 0.01 * 275)

    def test_untabulated_elements(self) -> None:
        for z in (57, 59, 71, 81, 82, 92, 103, 120):       # no entry / beyond the Fortran table
            for l in range(4):
                self.assertEqual(defrtm(7, l, z), 0.0)
        with self.assertRaises(FortranStop):
            defrtm(1, 1, 14)
        with self.assertRaises(FortranStop):
            defrtm(1, 0, 0)


# ---------------------------------------------------------------------------
# complete generations
# ---------------------------------------------------------------------------

class PseudopotentialTests(ReferenceCase):
    def test_troullier_martins_and_result_object(self) -> None:
        """Si, PBE, Troullier-Martins, lmax = 2: .dat and the result object."""
        res = self.check("si_tm_x6")
        st = res.state
        self.assertEqual(st.stage, "pseudopotential")
        self.assertEqual((res.lmax, res.zion, res.rcore), (2, 4.0, 0.0))
        self.assertEqual(st.i_pp_type, [2, 2, 2])
        self.assertEqual(st.np_, [3, 3, 3])
        self.assertEqual(st.rc, [0.01 * 173, 0.01 * 190, 0.01 * 203])    # defaults of defrtm
        # energies in full precision; the .dat prints five decimals
        dat = res.dat_text
        self.assertIn(f"{res.etot_ae:16.5f}", dat)
        self.assertIn(f"{'total energy':>30}  {res.etot_ps:16.5f}", dat)
        self.assertAlmostEqual(res.etot_ae, -289.83668, delta=1e-5)
        self.assertAlmostEqual(res.etot_ps, -3.74151, delta=1e-5)
        # valence eigenvalues and the default reference energy
        self.assertEqual([(n, l, f) for n, l, f, _ in res.valence_states], [(3, 0, 2.0), (3, 1, 2.0)])
        self.assertEqual(res.emax, max(res.valence_eigenvalues))
        self.assertAlmostEqual(res.emax_ev, -4.0812386, delta=2e-7)
        # the rows of the "pseudo atom" table
        rows = res.channels
        self.assertEqual([c.l for c in rows], [0, 1, 2])
        self.assertEqual([c.scheme for c in rows], ["t", "t", "t"])
        self.assertEqual([c.f for c in rows], [2.0, 2.0, 0.0])
        for c in rows:
            line = (f"{c.l:3d}  {c.scheme}  {c.rc:12.7f}{c.rmatch:12.7f}{c.e_ae_ev:12.7f}"
                    f"{c.e_ps_ev:12.7f}{c.norm:12.7f}{c.slope:12.7f}")
            self.assertIn(line, dat)
            self.assertEqual(c.rc, float(res.mesh[c.nrc - 1]))        # a mesh point ...
            self.assertLessEqual(c.rc, c.rc_input)                    # ... at or below the request
            self.assertEqual(c.rmatch, float(res.mesh[c.mch - 1]))
            self.assertAlmostEqual(c.norm, 1.0, delta=1e-5)
        self.assertAlmostEqual(rows[2].e_ae, res.emax, delta=1e-12)    # unoccupied d: emax
        self.assertEqual(res.messages, [])
        # ncpp result
        nc = st.ncpp
        self.assertEqual(nc.ups.shape, (3, st.mmax))
        self.assertEqual(nc.vps.shape, (3, st.mmax))
        self.assertIsNone(nc.core)
        self.assertAlmostEqual(nc.tc, 4.0, delta=1e-6)
        self.assertAlmostEqual(nc.etot, nc.ekin + 0.5 * nc.eeel + nc.evps + nc.ex + nc.ec, delta=1e-12)
        # ionic potentials approach -zion/r
        self.assertAlmostEqual(float(nc.vps[0, -1] * st.r[-1]), -4.0, delta=1e-6)

    def test_cpi_table_and_arrays(self) -> None:
        """C with a 1s core hole: the .cpi file and its data as arrays."""
        res = self.check("c_1s_x6", cpi=True)
        cpi = res.cpi
        self.assertIsInstance(cpi, CpiData)
        self.assertEqual(cpi.valence_charge, 5.0)
        self.assertEqual(len(cpi.wavefunctions), 2)
        self.assertIsNone(cpi.core_density)
        self.assertEqual(cpi.mesh_increment, res.state.r[1] / res.state.r[0])
        lines = res.cpi_text.splitlines()
        mmax = res.state.mmax
        self.assertEqual(lines[0].split(), ["0.50000000000000E+01", "2"])
        self.assertEqual(len(lines), 11 + 2 * (mmax + 1))
        for l in range(2):
            block = lines[11 + l * (mmax + 1):11 + (l + 1) * (mmax + 1)]
            self.assertEqual(block[0].split(), [str(mmax), "0.10247000000000E+01"])
            table = np.array([[float(x) for x in row.split()] for row in block[1:]])
            np.testing.assert_array_equal(table[:, 0], np.arange(1, mmax + 1))
            np.testing.assert_allclose(table[:, 1], cpi.radius, rtol=1e-13)
            np.testing.assert_allclose(table[:, 2], cpi.wavefunctions[l], rtol=1e-13, atol=1e-99)
            np.testing.assert_allclose(table[:, 3], cpi.potentials[l], rtol=1e-13)
        self.assertEqual(sorted(cpi.as_kwargs()), [
            "core_density", "core_first_derivative", "core_second_derivative", "mesh_increment",
            "potentials", "radius", "valence_charge", "wavefunctions"])
        self.assertEqual(format_cpi(res.state.ncpp), res.cpi_text)

    def test_cpi_arrays_feed_the_converter(self) -> None:
        try:
            from pp_generation.converters import CpiTable
        except Exception:  # pragma: no cover - converters are a separate package
            self.skipTest("pp_generation.converters is not available")
        res = _run("b_tm_nlcc_x6")
        table = CpiTable.from_arrays(**res.cpi.as_kwargs())
        self.assertEqual(table.lmax, 0)
        self.assertEqual(table.mesh_size, res.state.mmax)
        self.assertIsNotNone(table.core)

    def test_nonlinear_core_correction(self) -> None:
        """B, PBE, Troullier-Martins with pseudocore: .dat (with the 17-digit
        records of dnlcc7), .cpi with the appended model core density."""
        res = self.check("b_tm_nlcc_x6", cpi=True)
        nc = res.state.ncpp
        mmax = res.state.mmax
        self.assertEqual(res.cpi_text.count("\n"), 11 + (mmax + 1) + mmax)
        # the core radius is moved to the mesh point at or below the input value
        self.assertEqual(res.state.rnlc, 1.4)
        self.assertEqual(res.rcore, float(res.mesh[res.mesh <= 1.4][-1]))
        self.assertIn(f"{res.rcore:8.2E}".replace("1.37E+00", "0.14E+01"), res.dat_text)
        dc, dcp, dcpp = nc.core
        inside = res.mesh < res.rcore
        self.assertTrue(np.all(np.diff(dc[inside]) < 0.0))            # monotonically decaying
        np.testing.assert_array_equal(dc[~inside][1:], res.state.dc[~inside][1:])   # full core outside
        # zero slope and curvature at the origin (the polynomial starts with x**3)
        self.assertLess(abs(dcp[0]), 1e-4 * dc[0])
        self.assertLess(abs(dcpp[0]), 0.1 * dc[0])
        self.assertEqual(nc.fort24[0], _CORE_START)
        self.assertTrue(nc.fort24[-1].startswith("p "))
        self.assertNotIn("\np ", res.dat_text)                         # fgrep -v "p " of the script
        # lmax = 0 leaves the 2p electron out: the program warns about it
        self.assertEqual(res.messages, [
            " & ncpp -- warning: valency mismatch?   3.0000000000000000        2.0000000000000000     "])
        self.assertEqual(sorted(res.units), [
            "fort.19", "fort.22", "fort.23", "fort.24", "fort.25", "fort.27", "fort.28", "fort.37",
            "fort.38", "fort.39", "fort.40", "fort.45"])

    def test_hamann_with_core_correction(self) -> None:
        """Li, LDA, Hamann scheme with pseudocore."""
        res = self.check("li_ham_nlcc_x8", cpi=True)
        (row,) = res.channels
        self.assertEqual((row.scheme, row.i_pp_type), ("h", 1))
        self.assertGreater(row.rmatch, 2.0 * row.rc)          # matching radius of the Hamann scheme

    def test_hamann_nonrelativistic(self) -> None:
        res = self.check("si_ham_nr_x6", nonrelativistic=True)
        self.assertEqual(res.state.irl, 2)
        self.assertEqual([c.scheme for c in res.channels], ["h", "h", "h"])

    def test_schemes_per_channel_from_the_command_line(self) -> None:
        """psgen -xc 6 -ts h -tp t -td h -rc 1.3 on the Troullier-Martins template."""
        args = json.loads(_read("pp_si_mix.args.json"))
        self.assertEqual(args, ["-xc", "6", "-ts", "h", "-tp", "t", "-td", "h", "-rc", "1.3"])
        res = psgen(_read("pp_si_mix.orig.ini"), out="pp", write=False, xctyp="6", rcore="1.3",
                    channel_types={0: "h", 1: "t", 2: "h"})
        self.assertEqual(res.ini_text, _read("pp_si_mix.ini"))        # edits of the script
        self.assert_dat(res.dat_text.partition("\n")[2], _read("pp_si_mix.dat"))
        self.assertEqual([c.scheme for c in res.channels], ["h", "t", "h"])

    def test_two_valence_states_of_one_l(self) -> None:
        """Ti with 3s 3p 3d 4s valence: the 4s state is an "excited state" row."""
        res = self.check("ti_semicore_x6")
        s_row = res.channels[0]
        self.assertEqual(s_row.n, 3)
        (extra,) = s_row.excited
        self.assertEqual((extra.n, extra.f), (4, 2.0))
        self.assertIn(f"  0  troullier-martins   {'':26}{extra.e_ae_ev:12.7f}{extra.e_ps_ev:12.7f}"
                      "   (excited state!)", res.dat_text)
        self.assertEqual(res.state.ncpp.cval, 12.0)

    def test_frozen_core(self) -> None:
        if not (DATA / "si_x6_sr.fc").is_file():
            self.skipTest("core density of the all-electron tests is not available")
        res = self.check("si_fc_ion", core_file=DATA / "si_x6_sr.fc")
        self.assertFalse(res.state.tcore)
        self.assertIsNone(res.fc_text)                 # no .fc in frozen-core mode
        self.assertIsNotNone(res.cpi_text)

    def test_radii_and_reference_energies_from_the_input(self) -> None:
        res = self.check("si_opt_et")
        rows = res.channels
        self.assertEqual([c.rc_input for c in rows], [1.6, 1.8, 2.0])
        self.assertAlmostEqual(rows[2].e_ae_ev, 5.0, delta=1e-12)      # 2 2.0 5.0 t
        self.assertEqual(res.dat_text.splitlines()[-3:], [
            "@ 0   1.60  0.00E+00  t  : lt rct et s_pp_type",
            "@ 1   1.80  0.00E+00  t",
            "@ 2   2.00  0.18E+00  t"])

    def test_functionals_with_core_correction(self) -> None:
        """PW91, relativistic LDA and BLYP with a pseudocore (psgen -rc 1.3)."""
        for case in ("si_nlcc_x4", "si_nlcc_x7", "si_nlcc_x9"):
            with self.subTest(case=case):
                self.check(case)

    def test_lmax_4_hamann(self) -> None:
        res = self.check("si_lmax4_ham")
        self.assertEqual([c.l for c in res.channels], [0, 1, 2, 3, 4])
        self.assertEqual(len(res.cpi.potentials), 5)
        self.assertIn("fort.44", res.units)
        self.assertIn("fort.49", res.units)

    def test_failure_inside_a_channel(self) -> None:
        """K with a 1s core hole: the Fortran stops in tromar while it builds l = 0."""
        inp = parse_ini(_read("pp_k_1s_x6.ini"))
        with self.assertRaises(FortranStop) as ctx:
            fhipp(inp)
        exc = ctx.exception
        self.assertEqual(str(exc), "tromar - stop: convergence error cc0 (norm)")
        self.assertEqual(exc.channel, 0)
        self.assertEqual(exc.messages[0], " & tromar - stop: convergence error cc0 (norm)")
        self.assertEqual(len(exc.messages), 4)
        self.assertTrue(exc.messages[3].startswith(" &  iteration j k       "))
        # unit 23 as the Fortran left it: it ends with the head of the table
        state = exc.state
        self.assertIsNone(state.ncpp)
        self.assert_dat(format_dat(state).partition("\n")[2], _read("pp_k_1s_x6.dat"))
        self.assertIsNotNone(state.fort37)              # .aep and .fc were already written
        self.assertIsNotNone(state.fort19)

    def test_cutoff_radius_too_small(self) -> None:
        ini = _read("pp_si_tm_x6.ini") + "1  0.2  0.0  t\n"
        with self.assertRaises(FortranStop) as ctx:
            psgen(ini, write=False)
        exc = ctx.exception
        self.assertEqual(str(exc), "fhipp - stop: cutoff radius too small l           1")
        self.assertEqual(exc.channel, 1)
        self.assertEqual(exc.messages, [" & fhipp - stop: cutoff radius too small l           1"])

    def test_untabulated_default_radius(self) -> None:
        """La has no entry in defrtm: the computed Troullier-Martins default of a
        channel with nodes is zero and the program stops (as the Fortran does)."""
        states = ((1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (3, 2, 10), (4, 0, 2),
                  (4, 1, 6), (4, 2, 10), (5, 0, 2), (5, 1, 6), (5, 2, 1), (6, 0, 2))
        ini = "\n".join(["57.0  11  2  6  0.0"] + [f" {n} {l} {f}.0" for n, l, f in states]
                        + ["3  t", ""])
        with self.assertRaises(FortranStop) as ctx:
            psgen(ini, write=False)
        exc = ctx.exception
        self.assertEqual(exc.channel, 0)
        self.assertEqual(exc.messages, [
            " & fhipp - warning: no tabulated default core radius l           %d" % l for l in range(4)
        ] + [" & fhipp - stop: cutoff radius too small l           0"])
        self.assertEqual(exc.state.rc[0], 0.0)

    def test_bad_pseudopotential_records(self) -> None:
        head = "".join(_read("pp_si_tm_x6.ini").splitlines(keepends=True)[:6])
        for tail, message in (
                ("", "fhipp - verify input data file fort.22"),
                ("2  x\n", "fhipp - stop: bad pseudopotential typex"),
                ("5  t\n", "fhipp - lmax out of range (0 ... 4)"),
                ("2  t\n-1 1.7 0.0 t\n", "fhipp - optional data: l out of range (0 ... 4)")):
            with self.subTest(tail=tail):
                with self.assertRaises(FortranStop) as ctx:
                    psgen(head + tail, write=False)
                self.assertEqual(str(ctx.exception), message)
                self.assertIsNone(ctx.exception.channel)          # not a failure of one channel
                self.assertIsNotNone(ctx.exception.state.fort37)
        # a stop of the all-electron stage carries neither a state nor a channel
        with self.assertRaises(FortranStop) as ctx:
            psgen("14.0  3  2  6  0.0\n 1 0 2.0\n", write=False)
        self.assertEqual(str(ctx.exception), "fhipp - verify input data file fort.22")
        self.assertIsNone(ctx.exception.state)
        self.assertIsNone(ctx.exception.channel)

    def test_driver_writes_the_files_of_the_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = psgen(DATA / "pp_li_ham_nlcc_x8.ini", out="li", workdir=tmp, keep_scratch=True)
            self.assertEqual(sorted(res.files), sorted([
                "li.dat", "li.cpi", "li.fc", "li.aep", "li.ae_wfct", "fort.19", "fort.20", "fort.22",
                "fort.23", "fort.24", "fort.25", "fort.27", "fort.28", "fort.37", "fort.38", "fort.39",
                "fort.40", "fort.45"]))
            self.assertEqual((Path(tmp) / "li.cpi").read_text(), res.cpi_text)
            self.assertEqual((Path(tmp) / "li.dat").read_text(), res.dat_text)
            self.assertTrue(res.dat_text.startswith("#info "))
            self.assert_cpi(res.cpi_text, _read("pp_li_ham_nlcc_x8.cpi"))
            # the .cpi ends with the model core density, which is unit 27
            fort27 = (Path(tmp) / "fort.27").read_text()
            self.assertTrue(res.cpi_text.endswith(fort27))
            self.assertEqual(res.aep_text, (Path(tmp) / "fort.37").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            res = psgen(DATA / "pp_li_ham_nlcc_x8.ini", out="li", workdir=tmp, only_atom=True,
                        wavefunctions=False)
            self.assertEqual(sorted(res.files), ["li.dat"])
            self.assertIsNone(res.cpi_text)
            self.assertEqual(res.channels, [])
            self.assertIsNone(res.etot_ps)


if __name__ == "__main__":
    unittest.main()

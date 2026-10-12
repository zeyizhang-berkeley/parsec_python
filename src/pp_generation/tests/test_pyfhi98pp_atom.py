"""Tests of the pure-Python FHI98PP all-electron core (``pyfhi98pp``).

The reference files under ``data/pyfhi98pp`` were produced by the original
Fortran program (``fhipp.x`` rev 1998 driven by the ``psgen`` script;
gfortran 13.3 -O2, glibc 2.39), so the tests run without the Fortran.

Comparison rules.  With the math library of the reference build the port
reproduces the Fortran bit for bit; other math libraries differ in the last
bit of ``exp``/``log``/``pow`` for about one argument in a thousand.  The
checks therefore allow one unit in the last printed digit and one
self-consistency iteration more or less (a state whose energy shift sits
within round-off of the 1e-12 threshold).  Set ``PYFHI98PP_EXACT=1`` to
demand identical text of the ``.dat`` files instead.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pp_generation.pyfhi98pp import (
    FortranStop, all_electron, anderson, apply_psgen_options, atomini, dftseq,
    dmelm, fhipp, fmom, labelmap, libm, logmesh, parse_ini, read_fort20, spp,
    vestat, vexcor,
)
from pp_generation.pyfhi98pp.labels import labels
from pp_generation.pyfhi98pp.psgen import psgen
from pp_generation.pyfhi98pp.fortran_io import (
    FortranEndOfFile, FortranReadError, ListReader, fmt_a, fmt_e, fmt_es,
    fmt_f, fmt_i,
)

DATA = Path(__file__).parent / "data" / "pyfhi98pp"
EXACT = os.environ.get("PYFHI98PP_EXACT") == "1"
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([EeDd][-+]?\d+)?$")

SI_INI = """14.00  3  2  8  0.00 : z  nc  nv iexc rnlc
    1  0   2.00      : n  l   f
    2  0   2.00
    2  1   6.00
    3  0   2.00
    3  1   2.00
2  t                 : lmax  s_pp_def
"""


def _read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


def _table(text: str, skip: int = 1) -> np.ndarray:
    return np.array([[float(x) for x in line.split()] for line in text.splitlines()[skip:]])


def _ae_part(lines: list[str]) -> list[str]:
    """Unit 23 up to ``fhipp - all-electron atom done`` (the stored ``.dat``
    references of this module end there; the pseudopotential part that
    follows is covered by ``test_pyfhi98pp_psgen``)."""
    marker = " fhipp - all-electron atom done"
    return lines[:lines.index(marker) + 1] if marker in lines else lines


@unittest.skipUnless(DATA.is_dir(), "pyfhi98pp reference data are not available")
class ReferenceCase(unittest.TestCase):
    """Helpers to compare with the files written by the Fortran program."""

    def assert_dat(self, mine: list[str], ref: list[str]) -> None:
        """All-electron part of ``fort.23`` against the Fortran text."""
        self.assertEqual(len(mine), len(ref))
        if EXACT:
            self.assertEqual(mine, ref)
            return
        for a, b in zip(mine, ref):
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

    def assert_aep(self, mine: str, ref: str) -> None:
        """``fort.37``: effective potential, 14 digits."""
        self.assertEqual(mine.splitlines()[0], ref.splitlines()[0])
        a, b = _table(mine), _table(ref)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        qa, qb = a[:, 0] * a[:, 1], b[:, 0] * b[:, 1]
        # r*v is the screened charge; outside the density of a neutral atom
        # it is a rounding residue (1e-13) already in the Fortran
        self.assertLess(np.max(np.abs(qa - qb)), 1e-9)
        sel = np.abs(qb) >= 1e-3
        self.assertLess(np.max(np.abs(a[sel, 1] - b[sel, 1]) / np.abs(b[sel, 1])), 1e-9)

    def assert_fc(self, mine: str, ref: str) -> None:
        """``fort.19``: core density and its derivatives, 14 digits."""
        ha, hb = mine.splitlines()[0].split(), ref.splitlines()[0].split()
        self.assertEqual(ha[0], hb[0])
        self.assertLessEqual(abs(float(ha[1]) - float(hb[1])), 1e-9 * abs(float(hb[1])) + 1e-300)
        if int(hb[0]) == 0:
            self.assertEqual(mine, ref)
            return
        a, b = _table(mine), _table(ref)
        self.assertEqual(a.shape, b.shape)
        self.assertLess(np.max(np.abs(a - b) / np.abs(b)), 1e-8)

    def assert_wfc(self, mine: str, ref: str) -> None:
        """``fort.38``: wave functions, 6 digits."""
        la, lb = mine.splitlines(), ref.splitlines()
        self.assertEqual(len(la), len(lb))
        for a, b in zip(la, lb):
            if a == b:
                continue
            self.assertFalse(b.startswith("#"), msg=f"{a!r} / {b!r}")
            xa, xb = [float(x) for x in a.split()], [float(x) for x in b.split()]
            for x, y in zip(xa, xb):
                self.assertLessEqual(abs(x - y), 2e-6 * abs(y) + 1e-300, msg=f"{a!r} / {b!r}")


# ---------------------------------------------------------------------------
# formatted output and list-directed input
# ---------------------------------------------------------------------------

class FortranIoTests(unittest.TestCase):
    def test_edit_descriptors_match_gfortran(self) -> None:
        # expected strings were printed by gfortran 13.3
        self.assertEqual(fmt_e(0.44642857142857e-03, 20, 14), "0.44642857142857E-03")
        self.assertEqual(fmt_e(-0.11771841748411e+07, 20, 14), "-.11771841748411E+07")
        self.assertEqual(fmt_e(0.0, 20, 14), "0.00000000000000E+00")
        self.assertEqual(fmt_e(0.446429e-03, 12, 6), "0.446429E-03")
        self.assertEqual(fmt_e(-27006.732007851, 12, 6), "-.270067E+05")
        self.assertEqual(fmt_e(1.0e-100, 12, 6), "0.100000E-99")
        self.assertEqual(fmt_e(-1.0e-101, 12, 6), "-.100000-100")
        self.assertEqual(fmt_e(0.0, 8, 2), "0.00E+00")
        self.assertEqual(fmt_f(14.0, 5, 2), "14.00")
        self.assertEqual(fmt_f(-0.5, 5, 2), "-0.50")
        self.assertEqual(fmt_f(123.0, 5, 2), "*****")
        self.assertEqual(fmt_f(-1778.47617, 18, 4), "        -1778.4762")
        self.assertEqual(fmt_f(1.0247, 12, 6), "    1.024700")
        self.assertEqual(fmt_f(-0.000001, 16, 5), "        -0.00000")
        self.assertEqual(fmt_es(0.0, 9, 1), "  0.0E+00")
        self.assertEqual(fmt_es(8.2e-4, 9, 1), "  8.2E-04")
        self.assertEqual(fmt_i(495, 4), " 495")
        self.assertEqual(fmt_i(12345, 4), "****")
        self.assertEqual(fmt_a("chemical symbol", 30), " " * 15 + "chemical symbol")
        self.assertEqual(fmt_a("LDA CA Perdew/Wang 1991 + X relativistic", 30),
                         "LDA CA Perdew/Wang 1991 + X re")
        self.assertEqual(fmt_f(float("nan"), 16, 5).strip(), "NaN")
        self.assertEqual(fmt_e(float("inf"), 20, 14).strip(), "Infinity")

    def test_list_directed_read(self) -> None:
        reader = ListReader("14.0 3 2\n6 0.d0 : rest\n1,0,2.0D0 ignored\n\n 3*7 /\n.t. f\n'a b' x\n")
        self.assertEqual(reader.read("riiir"), [14.0, 3, 2, 6, 0.0])   # continues on the next record
        self.assertEqual(reader.read("iir"), [1, 0, 2.0])
        self.assertEqual(reader.read("iiii"), [7, 7, 7, None])         # repeat count, slash
        self.assertEqual(reader.read("ll"), [True, False])
        self.assertEqual(reader.read("aa"), ["a b", "x"])
        with self.assertRaises(FortranEndOfFile):
            reader.read("i")
        with self.assertRaises(FortranReadError):
            ListReader("14.0 3.0 2").read("rii")                      # 3.0 is not an integer
        with self.assertRaises(FortranReadError):
            ListReader("1 2 : 3").read("iii")

    def test_read_fort20(self) -> None:
        self.assertEqual(read_fort20(".t. .f. : TDOPSP TNRL\n"), (True, False))
        self.assertEqual(read_fort20(".f. .t.\n"), (False, True))
        self.assertEqual(read_fort20(None), (True, False))
        self.assertEqual(read_fort20("\n"), (True, False))             # end= branch: defaults
        self.assertEqual(read_fort20(".f. garbage\n"), (False, False))  # err= branch after one item


class InputTests(unittest.TestCase):
    def test_parse_template(self) -> None:
        inp = parse_ini(SI_INI + "1  0.00 10.00  t     : lt  rct  et  s_pp_type\n2 1.9 0.0 h\n\n")
        self.assertEqual((inp.z, inp.nc, inp.nv, inp.iexc, inp.rnlc), (14.0, 3, 2, 8, 0.0))
        self.assertEqual(inp.n, (1, 2, 2, 3, 3))
        self.assertEqual(inp.l, (0, 0, 1, 0, 1))
        self.assertEqual(inp.f, (2.0, 2.0, 6.0, 2.0, 2.0))
        self.assertEqual(inp.sf, 14.0)
        self.assertEqual((inp.ltmx, inp.spptype), (2, "t"))
        self.assertEqual([(o.lt, o.rct, o.et, o.spptype) for o in inp.options],
                         [(1, 0.0, 10.0, "t"), (2, 1.9, 0.0, "h")])
        self.assertFalse(inp.pp_header_error or inp.options_error)

    def test_normalizations_and_stops(self) -> None:
        inp = parse_ini("-1.0 0 -1 0 -0.5\n-1 0 -1.0\n")
        self.assertEqual((inp.z, inp.nc, inp.nv, inp.iexc, inp.rnlc, inp.f), (1.0, 0, 1, 8, 0.5, (1.0,)))
        self.assertTrue(inp.pp_header_error)          # only an error once pseudopotentials are made
        self.assertEqual(parse_ini("1.0 0 1 13 0\n1 0 1.0\n").iexc, 8)
        for text, message in (
            ("14.0 3 2 : 6\n", "fhipp - verify input data file fort.22"),
            ("14.0 1 1 6 0.0\n1 0 2.0\n", "fhipp - verify input data file fort.22"),
            ("14.0 0 1 6 0.0\n2 2 2.0\n", "fhipp - data: l .le. n"),
            ("14.0 0 1 6 0.0\n6 5 2.0\n", "fhipp - data: l out of range (0 ... 4)"),
        ):
            with self.assertRaises(FortranStop) as ctx:
                parse_ini(text)
            self.assertEqual(str(ctx.exception), message)

    def test_optional_records(self) -> None:
        # a short record is completed from the next one, as in the Fortran
        inp = parse_ini(SI_INI + "0 2.5 0.0\n1 2.5 0.0 t\n2 2.5 0.0 h\n")
        self.assertEqual([(o.lt, o.spptype) for o in inp.options], [(0, "1"), (2, "h")])
        inp = parse_ini(SI_INI + "0 2.5 0.0 t\nx 1 2 t\n1 2.5 0.0 t\n")
        self.assertEqual(len(inp.options), 1)
        self.assertTrue(inp.options_error)

    def test_psgen_options(self) -> None:
        # expected text: fort.22 written by the csh script for
        #   psgen -T h -rp 2.5 -ts q -rd 12.5 (79-Au:tm.ini) and -xc 6 -rc 1.3
        text = "79.00 1 1 8 0.00 : z nc nv iexc rnlc\n 1 0 2.00\n 6 0 1.00\n2  t  : lmax\n1  0.00 10.00  t\n"
        out = apply_psgen_options(text, default_type="h", channel_radii={1: 2.5, 2: 12.5},
                                  channel_types={0: "q"})
        self.assertEqual(out.splitlines()[3:], ["2 h", "1 2.5 10.00 t", "0 0.0 0.0 h", "2 0.0 0.0 h"])
        out = apply_psgen_options(SI_INI, xctyp=6, rcore="1.3")
        self.assertEqual(out.splitlines()[0], "14.00 3 2 6 1.3 : z nc nv")
        self.assertEqual(out.splitlines()[1:], SI_INI.splitlines()[1:])


# ---------------------------------------------------------------------------
# labels, mesh, radial routines
# ---------------------------------------------------------------------------

class LabelMeshTests(unittest.TestCase):
    def test_labels(self) -> None:
        self.assertEqual(labelmap(14, 6), ("Si", "GGA Perdew/Burke/Ernzerhof".ljust(40)))
        self.assertEqual(labelmap(1, 7)[0], "H ")
        self.assertEqual(labelmap(1, 7)[1], "LDA CA Perdew/Wang 1991 + X relativistic")
        self.assertEqual(labelmap(90, 12), ("no", "unknown".ljust(40)))
        self.assertEqual(labels(3, 1, 1), ("3", "p", "+"))
        self.assertEqual(labels(7, 5, 3), (".", ".", "."))
        self.assertEqual(spp(1).strip(), "hamann")
        self.assertEqual(spp(2), "troullier-martins".ljust(20))

    def test_logmesh(self) -> None:
        # mesh points are pure products: identical to the Fortran on any machine
        for z, mmax, first, last in (
            (1.0, 387, "0x1.999999999999ap-8", "0x1.33cee627dffb4p+6"),
            (14.0, 495, "0x1.d41d41d41d41ep-12", "0x1.329e925d957ffp+6"),
            (79.0, 567, "0x1.4bd3edda68fe1p-14", "0x1.3ad28816d556bp+6"),
        ):
            m, amesh, al, r = logmesh(z)
            self.assertEqual((m, amesh), (mmax, 1.0247))
            self.assertEqual(len(r), mmax)
            self.assertEqual((float(r[0]).hex(), float(r[-1]).hex()), (first, last))
            self.assertAlmostEqual(al, math.log(1.0247), delta=1e-17)
            self.assertEqual(m % 2, 1)


class RadialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mmax, _, self.al, self.r = logmesh(1.0)
        self.wa = np.zeros(self.mmax)
        self.wb = np.ones(self.mmax)

    def test_hydrogen_eigenvalues(self) -> None:
        v = -1.0 / self.r
        for n, l, tol in ((1, 0, 2e-9), (2, 0, 2e-8), (2, 1, 2e-8), (3, 2, 2e-8)):
            res = dftseq(2, 1.0, self.mmax, self.r, n, l, 1.0, v, self.wa, self.wb, e=-0.6 / n ** 2)
            self.assertAlmostEqual(res.e, -0.5 / n ** 2, delta=tol)
            self.assertEqual(res.u.shape, (self.mmax,))
            self.assertTrue(np.all(res.u[res.nin:] == 0.0))
            # number of nodes inside the matching point
            inner = res.u[3:res.mch]
            self.assertEqual(int(np.sum(inner[1:] * inner[:-1] <= 0.0)), n - l - 1)
            # normalization int u**2 dr = 1
            self.assertAlmostEqual(fmom(-2, self.mmax, self.al, 1.0, self.r, res.u ** 2), 1.0, delta=1e-5)
        # scalar-relativistic 1s: -1/2 - alpha**2/8
        res = dftseq(1, 1.0, self.mmax, self.r, 1, 0, 1.0, v, self.wa, self.wb, e=-0.6)
        self.assertAlmostEqual(res.e, -0.5 - 0.125 / 137.0359896 ** 2, delta=2e-9)

    def test_dftseq_fixed_energy_modes(self) -> None:
        v = -1.0 / self.r
        bound = dftseq(2, 1.0, self.mmax, self.r, 2, 1, 1.0, v, self.wa, self.wb, e=-0.1)
        mch = bound.mch
        # mode 6: non-relativistic integration at fixed energy out to mch
        fixed = dftseq(6, 1.0, self.mmax, self.r, 2, 1, 1.0, v, self.wa, self.wb, mch=mch, e=bound.e)
        self.assertEqual((fixed.nin, fixed.mch, fixed.e), (mch, mch, bound.e))
        uld = fixed.up[mch - 1] / fixed.u[mch - 1]
        # mode 4: find the energy that reproduces this logarithmic derivative
        back = dftseq(4, 0.0, self.mmax, self.r, 2, 1, 1.0, v, self.wa, self.wb, mch=mch, uld=uld,
                      e=bound.e + 0.01)
        self.assertAlmostEqual(back.e, bound.e, delta=1e-10)
        # mode 8 returns e = 0 instead of stopping when there is no such bound state
        messages: list[str] = []
        none = dftseq(8, 0.0, self.mmax, self.r, 3, 0, 1.0, 1e-3 * v, self.wa, self.wb, e=-0.1,
                      warn=messages.append)
        self.assertEqual(none.e, 0.0)
        self.assertTrue(messages and "no bound state found" in messages[0])

    def test_mode8_without_bound_state_keeps_array_state(self) -> None:
        r, mmax = self.r, self.mmax
        bound = dftseq(8, 1.0, mmax, r, 1, 0, 1.0, -1.0 / r, self.wa, self.wb, e=-0.4)
        self.assertAlmostEqual(bound.e, -0.5, delta=2e-9)
        # repulsive potential: no classical turning point on the first pass.  The
        # Fortran returns e = 0 and leaves nin, u, up, upp as they were on entry
        # apart from the first four points, which it clears
        none = dftseq(8, 0.0, mmax, r, 1, 0, 1.0, 1.0 / r, self.wa, self.wb, e=-0.1,
                      nin=bound.nin, u=bound.u, up=bound.up, upp=bound.upp)
        self.assertEqual((none.e, none.nin), (0.0, bound.nin))
        for new, old in ((none.u, bound.u), (none.up, bound.up), (none.upp, bound.upp)):
            np.testing.assert_array_equal(new[:4], 0.0)
            np.testing.assert_array_equal(new[4:], old[4:])
        with self.assertRaises(FortranStop):         # mode 2 stops instead
            dftseq(2, 0.0, mmax, r, 1, 0, 1.0, 1.0 / r, self.wa, self.wb, e=-0.1)

    def test_math_library_seam(self) -> None:
        # every exp/log/pow goes through pyfhi98pp.libm and can be substituted
        calls = {"exp": 0, "pow": 0, "log": 0}
        saved = (libm.sexp, libm.spow, libm.spow_series, libm.slog)

        def count(name, func):
            def wrapper(*args):
                calls[name] += 1
                return func(*args)
            return wrapper
        libm.sexp = count("exp", saved[0])
        libm.spow = count("pow", saved[1])
        libm.spow_series = count("pow", saved[2])
        libm.slog = count("log", saved[3])
        try:
            res = dftseq(2, 1.0, self.mmax, self.r, 1, 0, 1.0, -1.0 / self.r, self.wa, self.wb, e=-0.6)
        finally:
            libm.sexp, libm.spow, libm.spow_series, libm.slog = saved
        self.assertEqual(calls["log"], 1)
        self.assertEqual(calls["exp"], 5 * res.nint)       # five start values of each inward pass
        self.assertEqual(calls["pow"], 1 + res.nint)       # series start + one per normalization

    def test_poisson_and_quadratures(self) -> None:
        r, mmax, al = self.r, self.mmax, self.al
        rho = 4.0 * np.exp(-2.0 * r)                       # hydrogen 1s density (times 4 pi)
        vh, eeel = vestat(mmax, 1.0, r, rho, True)
        exact = 1.0 / r - (1.0 + 1.0 / r) * np.exp(-2.0 * r)
        self.assertLess(np.max(np.abs(vh - exact)), 1e-7)
        self.assertAlmostEqual(eeel, 0.625, delta=1e-6)
        self.assertIsNone(vestat(mmax, 1.0, r, rho)[1])
        self.assertAlmostEqual(fmom(0, mmax, al, 1.0, r, rho), 1.0, delta=1e-6)
        self.assertAlmostEqual(fmom(-1, mmax, al, 1.0, r, rho), 1.0, delta=1e-6)
        self.assertAlmostEqual(fmom(2, mmax, al, 2.0, r, rho), 1.5, delta=1e-6)
        self.assertAlmostEqual(dmelm(mmax, al, r, -1.0 / r, rho), -1.0, delta=1e-6)

    def test_anderson(self) -> None:
        r = self.r
        vi, vo = -1.0 / r, -2.0 / r
        vn, vi1, vo1 = anderson(0.5, 1, self.mmax, r, np.zeros_like(r), vo, np.zeros_like(r), vi)
        np.testing.assert_array_equal(vn, 0.5 * vi + 0.5 * vo)
        np.testing.assert_array_equal(vi1, vi)
        np.testing.assert_array_equal(vo1, vo)
        # a linear problem vo = a*vi + b is solved by one Anderson step
        f = lambda x: 0.3 * x - 0.7 / r                    # noqa: E731
        v2, _, _ = anderson(0.5, 2, self.mmax, r, vo1=f(vi), vo=f(vn), vi1=vi, vi=vn)
        np.testing.assert_allclose(f(v2), v2, rtol=1e-12)

    def test_atomini(self) -> None:
        mmax, _, _, r = logmesh(14.0)
        vi, e = atomini(5, mmax, 14.0, (1, 2, 2, 3, 3), (2.0, 2.0, 6.0, 2.0, 2.0), r)
        self.assertAlmostEqual(vi[0] * r[0], -14.0, delta=0.05)     # -z/r at the origin
        self.assertAlmostEqual(vi[-1] * r[-1], -1.0, delta=1e-12)   # -1/r far outside
        np.testing.assert_allclose(e, [-84.5, -15.125, -3.125, -0.5, -0.0555555555555555], rtol=1e-13)


# ---------------------------------------------------------------------------
# exchange-correlation
# ---------------------------------------------------------------------------

@unittest.skipUnless((DATA / "xc_si_pbe.npz").is_file(), "pyfhi98pp reference data are not available")
class XcTests(unittest.TestCase):
    """``vexcor`` against the Fortran on the density of the Si atom."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = np.load(DATA / "xc_si_pbe.npz")

    def test_all_functionals(self) -> None:
        ref = self.ref
        r, rho, rhop, rhopp = ref["r"], ref["rho"], ref["rhop"], ref["rhopp"]
        mmax = len(r)
        for iexc in range(1, 12):
            with self.subTest(iexc=iexc):
                xc = vexcor(iexc, mmax, r, rho, rhop, rhopp, True)
                np.testing.assert_allclose(xc.vpxc, ref["vpxc"][iexc - 1], rtol=1e-10, atol=0.0)
                evxc, ex, ec = ref["energies"][iexc - 1]
                self.assertAlmostEqual(xc.evxc, evxc, delta=1e-10 * abs(evxc))
                self.assertAlmostEqual(xc.ex, ex, delta=1e-10 * abs(ex))
                if iexc == 2:
                    # Hedin-Lundqvist: the Fortran integrates an unassigned
                    # array for ec; dex carries the whole xc energy density
                    self.assertEqual(xc.ec, 0.0)
                    continue
                np.testing.assert_allclose(xc.dexc, ref["dexc"][iexc - 1], rtol=1e-10, atol=0.0)
                self.assertAlmostEqual(xc.ec, ec, delta=1e-10 * abs(ec))
                if iexc in (7, 11):
                    self.assertEqual(xc.ec, 0.0)       # iexc 7: dec(i)=ec quirk of vexcor.f

    def test_flag_and_relativistic_exchange(self) -> None:
        ref = self.ref
        args = (len(ref["r"]), ref["r"], ref["rho"], ref["rhop"], ref["rhopp"])
        xc = vexcor(8, *args)
        self.assertIsNone(xc.ex)
        self.assertIsNone(xc.dexc)
        np.testing.assert_array_equal(xc.vpxc, vexcor(8, *args, True).vpxc)
        # iexc 7 passes the incoming ec through as correlation energy density
        xc7 = vexcor(7, *args, True, ec_in=-0.25)
        np.testing.assert_array_equal(xc7.dec, np.full(args[0], -0.25))
        with self.assertRaises(FortranStop):
            vexcor(12, *args)

    def test_zero_density_tail(self) -> None:
        mmax, _, _, r = logmesh(1.0)
        rho = np.where(r < 20.0, 4.0 * np.exp(-2.0 * r), 0.0)
        rhop, rhopp = -2.0 * rho, 4.0 * rho
        for iexc in range(1, 12):
            xc = vexcor(iexc, mmax, r, rho, rhop, rhopp, True)
            self.assertTrue(np.all(np.isfinite(xc.vpxc)), msg=f"iexc {iexc}")
            self.assertTrue(np.all(xc.vpxc[r >= 20.0] == 0.0), msg=f"iexc {iexc}")


# ---------------------------------------------------------------------------
# all-electron atoms
# ---------------------------------------------------------------------------

class AllElectronTests(ReferenceCase):
    FULL = (("si_x6_sr", False), ("h_x8_sr", False), ("c_1s1_x6_nr", True))
    DAT_ONLY = (("au_x8_sr", False), ("cl_neg_x8_sr", False), ("si_x4_nr", True), ("c_x9_sr", False),
                ("si_x7_sr", False), ("si_x3_sr", False), ("si_x1_sr", False), ("si_x11_sr", False),
                ("c_x10_sr", False), ("fe_x5_sr", False))

    def test_psgen_full_runs(self) -> None:
        """PBE Si, LDA H and the C 1s core hole: .dat, .aep, .fc, wave functions."""
        for case, tnrl in self.FULL:
            with self.subTest(case=case):
                st = fhipp(parse_ini(_read(f"{case}.ini")), tdopsp=True, tnrl=tnrl)
                self.assert_dat(_ae_part(st.fort23), _read(f"{case}.dat").splitlines()[1:])
                self.assert_aep(st.fort37, _read(f"{case}.aep"))
                self.assert_fc(st.fort19, _read(f"{case}.fc"))
                self.assert_wfc(st.fort38, _read(f"{case}.wfc"))

    def test_all_electron_only(self) -> None:
        """Other functionals, a heavy atom, an anion (psgen -ao: .dat only)."""
        for case, tnrl in self.DAT_ONLY:
            with self.subTest(case=case):
                st = fhipp(parse_ini(_read(f"{case}.ini")), tdopsp=False, tnrl=tnrl)
                self.assert_dat(st.fort23, _read(f"{case}.dat").splitlines()[1:])
                self.assertIsNone(st.fort37)
                self.assertIsNone(st.fort19)
                self.assertLessEqual(st.it, 100)

    def test_result_object(self) -> None:
        st = fhipp(parse_ini(_read("si_x6_sr.ini")))
        nst = st.nc + st.nv
        self.assertEqual((st.mmax, st.irl, st.igr, st.tcore, nst), (495, 1, 1, True, 5))
        self.assertEqual(st.uu.shape, (nst, st.mmax))
        for name in ("r", "vi", "rho", "rhop", "rhopp", "dc", "dcp", "dcpp", "ves", "vxc"):
            self.assertEqual(getattr(st, name).shape, (st.mmax,), msg=name)
        self.assertEqual(len(st.ninu), nst)
        for i in range(nst):
            # u is normalized on the points 1..ninu(i) ...
            norm = fmom(-2, st.ninu[i] | 1, st.al, 1.0, st.r, st.uu[i] ** 2)
            self.assertAlmostEqual(norm, 1.0, delta=1e-4)
        # ... but as in the Fortran (dcpv(mx,1,nin,u,uu(1,i))) the points beyond
        # ninu(i) keep the values of earlier iterations with a larger nin
        self.assertTrue(any(np.any(st.uu[i, st.ninu[i]:] != 0.0) for i in range(nst)))
        self.assertAlmostEqual(st.tc, 14.0, delta=1e-5)
        self.assertAlmostEqual(st.etot, st.ekin + 0.5 * st.eeel + st.ecou + st.ex + st.ec, delta=1e-9)
        self.assertEqual(st.de, 0.0)                    # converged: eigenvalues did not change
        self.assertTrue(np.all(st.rpk[3:] > 1.0))       # outermost peaks of 3s, 3p (bohr)
        self.assertEqual(st.unit6, [])
        self.assertIn(f"{st.etot:16.5f}", st.fort23_text)

    def test_frozen_core_and_initial_potential(self) -> None:
        core = _read("si_x6_sr.fc")
        st = fhipp(parse_ini(_read("si_fc_x6_sr.ini")), tdopsp=True, fort18=core)
        self.assertFalse(st.tcore)
        self.assertIsNone(st.fort19)                    # no core density is written in this mode
        self.assert_dat(_ae_part(st.fort23), _read("si_fc_x6_sr.dat").splitlines()[1:])
        self.assertIn("              frozen-core mode", st.fort23)
        self.assertTrue(st.fort38.startswith("# 3p  5  3  1  0"))
        self.assertEqual(st.fort38.count("#"), 2)       # valence states only

        aep = _read("si_x6_sr.aep")
        st = fhipp(parse_ini(_read("si_ip_x6_sr.ini")), tdopsp=False, fort36=aep)
        self.assertEqual(st.unit6, [" & fhipp - initial potential read from file fort.36"])
        self.assert_dat(st.fort23, _read("si_ip_x6_sr.dat").splitlines()[1:])
        self.assertLessEqual(st.it, 4)
        # a potential on another mesh is silently ignored, as in the Fortran
        st = all_electron(parse_ini(_read("h_x8_sr.ini")), fort36=aep)
        self.assertEqual(st.unit6, [])

    def test_negative_ion_and_non_convergence(self) -> None:
        st = fhipp(parse_ini(_read("cl_neg_x8_sr.ini")), tdopsp=False)
        self.assertEqual(st.fort23[0], " & fhipp - warning: negative ion")
        self.assertIn("                  total charge  -0.50", st.fort23)
        st = all_electron(parse_ini(_read("h_x8_sr.ini")), itmx=5)
        self.assertEqual(st.it, 6)
        self.assertEqual(st.unit6, [" & fhipp - warning: potential not converged"])
        # line written by the Fortran built with itmx = 5
        self.assertIn("          number of iterations                 6   convergence  1.2E-02", st.fort23)

    def test_psgen_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = psgen(DATA / "h_x8_sr.ini", out="h", workdir=tmp, keep_scratch=True)
            # (the pseudopotential stage adds h.cpi and further unit files)
            self.assertLessEqual(
                {"h.dat", "h.fc", "h.aep", "h.ae_wfct", "fort.19", "fort.20", "fort.22", "fort.23",
                 "fort.37", "fort.38"}, set(res.files))
            dat = (Path(tmp) / "h.dat").read_text().splitlines()
            self.assertTrue(dat[0].startswith("#info "))
            self.assert_dat(_ae_part(dat[1:]), _read("h_x8_sr.dat").splitlines()[1:])
            self.assertEqual((Path(tmp) / "fort.20").read_text(), ".t. .f. : TDOPSP TNRL\n")
            self.assert_aep((Path(tmp) / "h.aep").read_text(), _read("h_x8_sr.aep"))
            self.assertEqual((Path(tmp) / "h.fc").read_text(), _read("h_x8_sr.fc"))
            self.assert_wfc((Path(tmp) / "h.ae_wfct").read_text(), _read("h_x8_sr.wfc"))
        with tempfile.TemporaryDirectory() as tmp:
            # psgen -ao -r -xc 6 : no .fc/.aep, input edited like the script does
            res = psgen(_read("c_1s1_x6_nr.ini"), out="c", workdir=tmp, only_atom=True,
                        nonrelativistic=True, xctyp=6, wavefunctions=False)
            self.assertEqual(sorted(res.files), ["c.dat"])
            self.assertEqual(res.ini_text.splitlines()[0], "6.00 1 2 6 0.00 : z nc nv")
            self.assert_dat(res.state.fort23, _read("c_1s1_x6_nr.dat").splitlines()[1:])
            self.assertIn("         non-relativistic mode", res.state.fort23)


if __name__ == "__main__":
    unittest.main()

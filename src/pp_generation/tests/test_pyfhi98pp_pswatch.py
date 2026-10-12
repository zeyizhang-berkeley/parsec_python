"""Tests of the pure-Python FHI98PP test program (``pyfhi98pp.pswatch``).

The reference files ``data/pyfhi98pp/pswatch_*`` were produced by the
original Fortran program (``pslp.x`` rev 1998 driven by the ``pswatch``
script, gfortran 13.3 -O2, glibc 2.39, OpenBLAS 0.3.26), so the tests run
without the Fortran.  The pseudopotentials they test (``.cpi``) come from
the Fortran ``psgen`` as well.

Comparison rules.  With the math library of the reference build the port
reproduces every report of the Fortran reference set character by
character.  Other math libraries differ in the last bit of ``exp``, ``log``,
``pow``, ``sin`` and ``cos`` for about one argument in a thousand, so the
checks allow one unit in the last printed digit and one self-consistency
iteration more or less; the verdicts of the ghost-state analysis must agree
exactly.  Set ``PYFHI98PP_EXACT=1`` to demand identical report text.
"""

from __future__ import annotations

import gzip
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pp_generation.pyfhi98pp import pswatch as pw
from pp_generation.pyfhi98pp.dspevx import dspevx_lowest, dsptrd_lower, dstebz_index
from pp_generation.pyfhi98pp.errors import FortranStop
from pp_generation.pyfhi98pp.kinkon import FortranCrash, bessj, dfbt
from pp_generation.pyfhi98pp.klbyii import LaguerreBasis, NMX, bas0, plgr0
from pp_generation.pyfhi98pp.ppcheck import derlkb, energy_mesh, rcovalent
from pp_generation.pyfhi98pp.radial import dmelm, fmom

DATA = Path(__file__).parent / "data" / "pyfhi98pp"
EXACT = os.environ.get("PYFHI98PP_EXACT") == "1"
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([EeDd][-+]?\d+)?$")


def _read(name: str) -> str:
    path = DATA / name
    if name.endswith(".gz"):
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    return path.read_text(encoding="utf-8")


def _report(name: str) -> list[str]:
    """Lines of a stored ``.test`` file without the ``#info`` line."""
    lines = _read(f"pswatch_{name}.test").split("\n")[1:]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


@unittest.skipUnless((DATA / "pswatch_si.ini").is_file(),
                     "pswatch reference data are not available")
class ReferenceCase(unittest.TestCase):
    """Helpers to compare with the reports written by the Fortran program."""

    _cache: dict = {}

    @classmethod
    def inputs(cls, key: str) -> tuple[str, str, str]:
        if key not in cls._cache:
            aep = "si_x6_sr.aep" if key == "si" else f"pswatch_{key}.aep.gz"
            cls._cache[key] = (_read(f"pswatch_{key}.ini"), _read(f"pswatch_{key}.cpi.gz"),
                               _read(aep))
        return cls._cache[key]

    def assert_report(self, mine: list[str], ref: list[str]) -> None:
        if EXACT:
            self.assertEqual(mine, ref)
            return
        self.assertEqual(len(mine), len(ref))
        for a, b in zip(mine, ref):
            if a == b:
                continue
            # verdicts, notes and messages must be identical
            self.assertFalse(b.lstrip().startswith(("*", "&", "note", "---")), msg=f"{a!r} / {b!r}")
            ta, tb = a.split(), b.split()
            self.assertEqual(len(ta), len(tb), msg=f"{a!r} / {b!r}")
            for k, (x, y) in enumerate(zip(ta, tb)):
                if x == y:
                    continue
                self.assertTrue(_NUMBER.match(x) and _NUMBER.match(y), msg=f"{a!r} / {b!r}")
                if "number of iterations" in b and k == 3:
                    self.assertLessEqual(abs(int(x) - int(y)), 1, msg=b)
                elif b.startswith("ck") and k == 3:
                    # cutoff in Ry: one step of the k mesh
                    self.assertLessEqual(abs(int(x) - int(y)), 2, msg=b)
                elif "E" in y.upper():
                    mant, _, expo = y.upper().partition("E")
                    unit = 10.0 ** (int(expo) - len(mant.partition(".")[2]))
                    self.assertLessEqual(abs(float(x) - float(y)), 1.5 * unit, msg=f"{a!r} / {b!r}")
                else:
                    decimals = len(y.partition(".")[2])
                    self.assertLessEqual(abs(float(x) - float(y)), 1.5 * 10.0 ** (-decimals),
                                         msg=f"{a!r} / {b!r}")

    def run_case(self, key: str, **options) -> pw.PswatchResult:
        ini, cpi, aep = self.inputs(key)
        if options.pop("no_aep", False):
            aep = "\n"
        return pw.pswatch(ini, cpi, aep, prefix="pp", **options)


# ---------------------------------------------------------------------------
# complete reports
# ---------------------------------------------------------------------------

class ReportTests(ReferenceCase):
    def test_silicon_local_s_with_curves(self) -> None:
        res = self.run_case("si", lloc=0)
        self.assert_report(res.lines, _report("si_l0"))
        self.assertTrue(res.report.startswith("#info pyfhi98pp.pswatch pp "))
        self.assertEqual(res.stage, "tested")
        # pseudo-atom
        atom = res.atom
        self.assertEqual((atom.symz, atom.iexc, atom.mmax, atom.nv), ("Si", 6, 495, 2))
        self.assertEqual((atom.n, atom.l, atom.f), ([1, 2], [0, 1], [2.0, 2.0]))
        self.assertTrue(atom.converged)
        self.assertAlmostEqual(atom.etot, -3.74151, places=5)
        self.assertAlmostEqual(atom.ehartree, 2.93357, places=5)
        np.testing.assert_allclose(atom.e_eV, [-10.8130, -4.0812], atol=6e-5)
        # ghost analysis
        self.assertEqual(res.verdicts, {1: "no ghost", 2: "undetermined"})
        p, d = res.channels
        self.assertEqual(p.message, "* no ghost (ekb < 0, eref < eloc0)")
        self.assertAlmostEqual(p.ekb_eV, -65.5703, places=4)
        self.assertAlmostEqual(p.ckb, -0.1129, places=4)
        self.assertAlmostEqual(p.eloc0_eV, -3.6048, places=4)
        self.assertEqual(p.eloc1, 0.0)
        self.assertAlmostEqual(p.eref_eV, -4.0812, places=4)
        self.assertAlmostEqual(p.margin, (-3.6048 + 4.0812) / 27.2116, places=5)
        self.assertEqual(d.message, "* undetermined (ekb < 0, eref = eloc0 = 0)")
        self.assertEqual(d.notes, ("eref", "eloc0", "eloc1"))
        self.assertTrue(d.searched_reference)
        self.assertFalse(res.ghost_free)
        np.testing.assert_allclose(p.semilocal, [-4.0812, 0.0, 0.0], atol=6e-5)
        np.testing.assert_allclose(p.nonlocal_, [-4.0812, 0.0, 0.0], atol=6e-5)
        # kinetic-energy convergence
        self.assertEqual([(k.l, k.n) for k in res.kinetic], [(0, 1), (1, 2)])
        self.assertEqual([row.cutoff_Ry for row in res.kinetic[0].rows], [6, 12, 33, 55])
        self.assertTrue(all(k.converged for k in res.kinetic))
        self.assertAlmostEqual(res.kinetic[1].x_ekin, 0.383683, places=6)
        # logarithmic derivatives against the unit files fort.5x/6x/7x
        self.assertAlmostEqual(res.r_diag, 2.7758, places=4)
        ref = np.load(DATA / "pswatch_si_l0_curves.npz")
        self.assertEqual([c.l for c in res.logder], [0, 1, 2])
        for c in res.logder:
            pairs = {5: (c.e_ae, c.d_ae), 6: (c.e_sl, c.d_sl), 7: (c.e_nl, c.d_nl)}
            for unit, (e, dl) in pairs.items():
                name = f"fort{unit}{c.l}"
                if e is None:
                    self.assertNotIn(name, ref.files)
                    continue
                tab = ref[name]
                self.assertEqual(len(e), len(tab), msg=name)
                # the unit files have seven significant digits
                np.testing.assert_allclose(e, tab[:, 0], rtol=6e-7, atol=1e-9, err_msg=name)
                smooth = np.abs(tab[:, 1]) < 20.0
                np.testing.assert_allclose(dl[smooth], tab[smooth, 1], rtol=2e-6, atol=1e-9,
                                           err_msg=name)
        self.assertIsNone(res.logder[0].e_nl)             # local channel: no projector
        # wave functions
        self.assertEqual(res.fort38.splitlines()[0], _read("pswatch_si_l0.fort38").splitlines()[0])
        a = np.array([[float(x) for x in s.split()] for s in res.fort38.splitlines()
                      if not s.startswith("#")])
        b = np.array([[float(x) for x in s.split()] for s in _read("pswatch_si_l0.fort38").splitlines()
                      if not s.startswith("#")])
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-12)

    def test_silicon_other_local_channels(self) -> None:
        for lloc in (1, 2):
            res = self.run_case("si", lloc=lloc, compute_logder=False)
            self.assert_report(res.lines, _report(f"si_l{lloc}"))
        # default local channel: the highest one of the file
        self.assertEqual(res.control.l_loc, 2)
        res = self.run_case("si", compute_logder=False)
        self.assert_report(res.lines, _report("si_l2"))
        self.assertEqual(res.verdicts, {0: "no ghost", 1: "no ghost"})
        self.assertTrue(res.ghost_free)

    def test_script_options(self) -> None:
        cases = {
            "rl": dict(lloc=0, rdiag="1.75"),
            "L12": dict(lloc=2, lrange=(1, 2)),
            "kb": dict(lloc=0, check_ghost=False),
            "d": dict(lloc=0, log_derivative=False),
            "r": dict(lloc=0, nonrelativistic=True),
            "noaep": dict(lloc=0, no_aep=True),
        }
        for tag, options in cases.items():
            with self.subTest(option=tag):
                res = self.run_case("si", compute_logder=False, **options)
                self.assert_report(res.lines, _report(f"si_{tag}"))
        self.assertIn(" & pslp - warning: bad/missing full potential file fort.37", res.lines)

    def test_range_beyond_lmax(self) -> None:
        res = self.run_case("si", lloc=0, lrange=(0, 3))
        self.assert_report(res.lines, _report("si_L03"))
        ref = np.load(DATA / "pswatch_si_l0_curves.npz")
        f = res.logder[3]
        self.assertEqual(f.l, 3)
        self.assertIsNone(f.e_nl)
        for e, dl, name in ((f.e_ae, f.d_ae, "L03_fort53"), (f.e_sl, f.d_sl, "L03_fort63")):
            tab = ref[name]
            self.assertEqual(len(e), len(tab))
            smooth = np.abs(tab[:, 1]) < 20.0
            np.testing.assert_allclose(e, tab[:, 0], rtol=6e-7, atol=1e-9)
            np.testing.assert_allclose(dl[smooth], tab[smooth, 1], rtol=2e-6, atol=1e-9)

    def test_only_atom(self) -> None:
        res = self.run_case("si", only_atom=True)
        self.assert_report(res.lines, _report("si_ao"))
        self.assertEqual((res.stage, res.stop), ("pseudoatom", "pslp - pseudoatom done"))
        self.assertEqual(res.channels, [])
        # -kb -d stops at the same place
        res = self.run_case("si", lloc=0, check_ghost=False, log_derivative=False)
        self.assertEqual(res.stop, "pslp - pseudoatom done")

    def test_cpo_recalculated_wavefunctions(self) -> None:
        res = self.run_case("si", lloc=0, cpo=True, compute_logder=False)
        self.assert_report(res.lines, _report("si_cpo"))
        self.assertIn(" --- calculated wavefunctions used for kb potentials ---", res.lines)
        ref = _read("pswatch_si_cpo.cpi.gz").splitlines()
        mine = res.cpo.splitlines()
        self.assertEqual(len(mine), len(ref))
        self.assertEqual(mine[:12], ref[:12])
        for start in (12, 12 + 496, 12 + 2 * 496):
            a = np.array([[float(x) for x in s.split()] for s in mine[start:start + 495]])
            b = np.array([[float(x) for x in s.split()] for s in ref[start:start + 495]])
            np.testing.assert_array_equal(a[:, :2], b[:, :2])
            np.testing.assert_array_equal(a[:, 3], b[:, 3])
            np.testing.assert_allclose(a[:, 2], b[:, 2], rtol=1e-8, atol=1e-13)
        with tempfile.TemporaryDirectory() as tmp:
            ini, cpi, aep = self.inputs("si")
            out = pw.pswatch(ini, cpi, aep, prefix="x/si", lloc=0, cpo="new.cpi", workdir=tmp,
                             write=True, compute_logder=False)
            self.assertEqual(sorted(out.files), ["new.cpi", "si.test"])
            self.assertEqual((Path(tmp) / "new.cpi").read_text(), out.cpo)
            self.assertEqual((Path(tmp) / "si.test").read_text(), out.report)

    def test_nonlinear_core_correction(self) -> None:
        res = self.run_case("li", lloc=1, compute_logder=False)
        self.assert_report(res.lines, _report("li_l1"))
        self.assertTrue(res.atom.tnlc)
        self.assertIn("     nonlinear core-valence XC", res.lines)
        self.assertIsNotNone(res.atom.cexc)
        self.assertGreater(res.atom.tc_core, 0.0)

    def test_two_states_per_angular_momentum(self) -> None:
        res = self.run_case("na_semicore", lloc=0, compute_logder=False)
        self.assert_report(res.lines, _report("na_semicore_l0"))
        self.assertEqual(res.atom.n, [1, 2, 2, 3])
        self.assertEqual(res.atom.l, [0, 1, 0, 1])

    def test_hamann_potential(self) -> None:
        res = self.run_case("o_ham", lloc=1, compute_logder=False)
        self.assert_report(res.lines, _report("o_ham_l1"))


# ---------------------------------------------------------------------------
# states without bound solution: stack leftovers and the crash of pslp.x
# ---------------------------------------------------------------------------

class UnboundStateTests(ReferenceCase):
    def test_no_bound_state_rows(self) -> None:
        # the cx row is computed from what ppcheck's last dftseq call left behind
        res = self.run_case("o_lmax0", lloc=0)
        self.assert_report(res.lines, _report("o_lmax0"))
        self.assertEqual(float(res.atom.e[0]), 0.0)
        self.assertFalse(res.kinetic[0].valid)
        self.assertIn("ck   0     1.0E+00      0 < 0.000000   0.000000E+00", res.lines)
        # same result when the curves are not asked for
        res2 = self.run_case("o_lmax0", lloc=0, compute_logder=False)
        self.assertEqual(res2.lines, res.lines)
        self.assertEqual(res2.logder, [])

    def test_no_bound_state_without_log_derivatives(self) -> None:
        res = self.run_case("o_lmax0", lloc=0, log_derivative=False)
        self.assert_report(res.lines, _report("o_lmax0_d"))

    def test_crash_of_the_fortran_program(self) -> None:
        ini, cpi, aep = self.inputs("sc_lmax1")
        with self.assertRaises(FortranCrash) as ctx:
            pw.pswatch(ini, cpi, aep, lloc=0, compute_logder=False)
        self.assertIsInstance(ctx.exception, FortranStop)
        self.assertIn("segmentation fault", str(ctx.exception))
        ref = _report("sc_lmax1_crash")          # from a run without -q
        self.assert_report(ctx.exception.messages, ref)
        # with -q nothing of this report reaches the file
        self.assertEqual(pw.flushed_output(ctx.exception.messages), [])
        # the ghost analysis itself is not affected
        scan = pw.ghost_scan(ini, cpi)
        self.assertEqual(sorted(scan), [0, 1])
        self.assertEqual(scan[0].verdicts, {1: "undetermined"})

    def test_flushed_output(self) -> None:
        lines = ["x" * 99] * 100                  # 100 bytes per record
        self.assertEqual(len(pw.flushed_output(lines)), 82)        # 2 * 41 records >= 4096
        self.assertEqual(pw.flushed_output(lines[:40]), [])
        self.assertEqual(len(pw.flushed_output(lines[:41])), 41)


# ---------------------------------------------------------------------------
# fast ghost test
# ---------------------------------------------------------------------------

class GhostScanTests(ReferenceCase):
    def test_scan_agrees_with_full_reports(self) -> None:
        ini, cpi, aep = self.inputs("si")
        scan = pw.ghost_scan(ini, cpi)
        self.assertEqual(sorted(scan), [0, 1, 2])
        for lloc, ghost in scan.items():
            full = pw.pswatch(ini, cpi, aep, lloc=lloc, compute_logder=False, kinetic=False)
            self.assertEqual(len(ghost.channels), 2)
            for a, b in zip(ghost.channels, full.channels):
                self.assertEqual((a.l, a.verdict, a.message), (b.l, b.verdict, b.message))
                self.assertEqual((a.ekb, a.ckb, a.eloc0, a.eloc1, a.eref),
                                 (b.ekb, b.ckb, b.eloc0, b.eloc1, b.eref))
            self.assertEqual(ghost.atom.evps_loc, full.atom.evps_loc)
            self.assertEqual(ghost.atom.lines, full.atom.lines)
            self.assertIsNone(ghost.channels[0].semilocal)
        self.assertEqual({k: v.ghost_free for k, v in scan.items()}, {0: False, 1: False, 2: True})
        # the verdict lines of the reports
        for lloc in (0, 1, 2):
            ref = [s.strip() for s in _report(f"si_l{lloc}") if s.startswith(" * ")]
            self.assertEqual([ch.message for ch in scan[lloc].channels], ref)

    def test_single_channel_and_spectra(self) -> None:
        ini, cpi, _ = self.inputs("si")
        one = pw.ghost_check(ini, cpi, 0, spectra=True)
        self.assertEqual(one.l_loc, 0)
        self.assertEqual(one.verdicts, {1: "no ghost", 2: "undetermined"})
        np.testing.assert_allclose(one.channels[0].semilocal, [-4.0812, 0.0, 0.0], atol=6e-5)
        self.assertEqual(pw.ghost_check(ini, cpi).l_loc, 2)
        with self.assertRaises(ValueError):
            pw.ghost_check(ini, cpi, 3)

    def test_paths_as_input(self) -> None:
        ini, cpi, aep = self.inputs("li")
        with tempfile.TemporaryDirectory() as tmp:
            for ext, text in (("ini", ini), ("cpi", cpi), ("aep", aep)):
                (Path(tmp) / f"li.{ext}").write_text(text)
            res = pw.pswatch(Path(tmp) / "li.ini", prefix="li", workdir=tmp, lloc=1,
                             compute_logder=False)
            self.assert_report(res.lines, _report("li_l1"))


# ---------------------------------------------------------------------------
# input errors
# ---------------------------------------------------------------------------

class InputTests(ReferenceCase):
    def test_bad_pseudopotential_file(self) -> None:
        ini, cpi, aep = self.inputs("si")
        lines = cpi.splitlines()
        with self.assertRaises(FortranStop) as ctx:
            pw.pswatch(ini, "\n".join(lines[:11 + 496 + 200]) + "\n", aep, lloc=0)
        self.assertEqual(str(ctx.exception), "pslp - stop: bad pseudopotential file fort.31")
        self.assertEqual(ctx.exception.messages, [
            " & pslp - stop: bad pseudopotential file fort.31",
            " &        current l,mmax,i            1         495         200"])
        broken = list(lines)
        broken[11 + 496 + 1 + 100] = "  101 0.52437e-02 xxx 0.1"
        with self.assertRaises(FortranStop) as ctx:
            pw.pswatch(ini, "\n".join(broken) + "\n", aep, lloc=1)
        self.assertEqual(ctx.exception.messages[1],
                         " &        current l,mmax,i            1         495         101")

    def test_missing_pseudocore(self) -> None:
        ini, cpi, aep = self.inputs("si")
        with self.assertRaises(FortranStop) as ctx:
            pw.pswatch(ini.replace("6  0.00 :", "6  1.00 :", 1), cpi, aep, lloc=0)
        self.assertEqual(ctx.exception.messages, [
            " & pslp - stop: bad pseudopotential file fort.31",
            " &        error reading pseudo core density"])

    def test_pseudocore_without_flag(self) -> None:
        ini, cpi, aep = self.inputs("li")
        res = pw.pswatch(ini.replace("6  1.80 :", "6  0.00 :", 1), cpi, aep, lloc=1,
                         compute_logder=False)
        self.assertEqual(res.lines[0], " & pslp - pseudocore present (fort.31)")
        self.assertTrue(res.atom.tnlc)

    def test_fortran_exponent_without_letter(self) -> None:
        ini, cpi, aep = self.inputs("si")
        lines = cpi.splitlines()
        self.assertIn("E-03", lines[12])
        lines[12 + 400] = lines[12 + 400].replace("E-", "E+", 0)      # unchanged record
        fields = lines[12].split()
        fields[2] = "0.1068540314297-103"                             # 0.1068540314297e-103
        lines[12] = " ".join(fields)
        data, _ = pw.read_cpi("\n".join(lines) + "\n", 3)
        self.assertEqual(data.uu[0, 0], 0.1068540314297e-103)

    def test_local_channel_out_of_range(self) -> None:
        ini, cpi, aep = self.inputs("si")
        with self.assertRaises(ValueError):
            pw.pswatch(ini, cpi, aep, lloc=3)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class BuildingBlockTests(unittest.TestCase):
    def test_bisection_eigenvalues(self) -> None:
        rng = np.random.default_rng(7)
        n = 30
        a = rng.standard_normal((n, n))
        a = a + a.T + np.diag(np.linspace(-3.0, 40.0, n))
        ap = [float(a[i, j]) for j in range(n) for i in range(j, n)]
        exact = np.linalg.eigvalsh(a)
        m, w, info = dspevx_lowest(n, ap, 1, 4, 1.e-6)
        self.assertEqual((m, info), (4, 0))
        # bisection stops at an interval of width abstol
        np.testing.assert_allclose(w, exact[:4], atol=1.0e-6)
        m, w, info = dspevx_lowest(n, ap, 1, 4, 1.e-13)
        np.testing.assert_allclose(w, exact[:4], atol=1.0e-11)
        # tridiagonal form has the same spectrum
        d, e = dsptrd_lower(n, list(ap))
        t = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        np.testing.assert_allclose(np.linalg.eigvalsh(t), exact, atol=1.0e-11)
        m, w, info = dstebz_index(n, 3, 5, 1.e-12, d, e)
        np.testing.assert_allclose(w, exact[2:5], atol=1.0e-10)
        # a matrix with NaN: dspevx returns info = 4 and no eigenvalue
        ap[5] = float("nan")
        self.assertEqual(dspevx_lowest(n, ap, 1, 4, 1.e-6), (0, [], 4))

    def test_laguerre_basis(self) -> None:
        # orthonormality of the basis functions on a fine mesh
        r = np.array([1.0e-4 * 1.0123 ** i for i in range(1101)])
        basis = LaguerreBasis(len(r), r)
        g = basis.g0
        for n1, n2 in ((0, 0), (3, 3), (10, 10), (0, 1), (4, 9)):
            val = dmelm(len(r), basis.al, r, g[n1] * g[n2] / r ** 2, np.ones(len(r)))
            self.assertAlmostEqual(val, 1.0 if n1 == n2 else 0.0, places=5)
        self.assertEqual(g.shape, (NMX, len(r)))
        # hydrogen: -1/(2 n**2) with the Coulomb potential as "semilocal" part
        from pp_generation.pyfhi98pp.klbyii import klbyii
        enl, he, info = klbyii(0, len(r), r, np.zeros(len(r)), -1.0 / r, 0, 0.0, np.zeros(len(r)),
                               np.zeros(20), basis)
        self.assertEqual(info, 0)
        # the basis is made for the compact ground state ("a crude method
        # for any excited state", pslp.f)
        self.assertAlmostEqual(he[0], -0.5, places=5)
        self.assertAlmostEqual(he[1], -0.125, places=2)
        self.assertTrue(he[0] < he[1] < he[2] < 0.0)
        np.testing.assert_allclose(enl[:3], he[:3])
        self.assertAlmostEqual(float(plgr0(0, 2, 0.3)), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(float(bas0(0, 0.0)), 0.0)

    def test_bessel_functions(self) -> None:
        x = np.array([0.0, 1.0e-12, 0.3, 2.5, 7.0, 40.0])
        np.testing.assert_allclose(bessj(0, x), np.sin(x))
        np.testing.assert_allclose(bessj(1, x[2:]), np.sin(x[2:]) / x[2:] - np.cos(x[2:]))
        with np.errstate(all="ignore"):
            xj3 = ((15.0 / x ** 3 - 6.0 / x) * np.sin(x) - (15.0 / x ** 2 - 1.0) * np.cos(x))
        np.testing.assert_allclose(bessj(3, x[2:]), xj3[2:], rtol=1e-8, atol=1e-12)
        self.assertEqual(bessj(2, x)[0], 0.0)
        # below 1e-10 bessel.f uses x**(l+1)/(2l-1)!! (the leading term is
        # x**(l+1)/(2l+1)!!); kept as it is
        self.assertAlmostEqual(bessj(2, x)[1], 1.0e-36 / 3.0, delta=1e-45)

    def test_fourier_bessel_transform(self) -> None:
        # 1s hydrogen function: its transform is known in closed form
        r = np.array([1.0e-4 * 1.0247 ** i for i in range(501)])
        kmx, fk, gk, cnorm, back = dfbt(len(r), r, 2.0 * np.exp(-r), 0, 169.0, 0.05, inverse=True)
        self.assertEqual(kmx, 261)
        self.assertAlmostEqual(cnorm, 2.0 / math.pi)
        self.assertAlmostEqual(fk[-1], 13.0)
        exact = cnorm * 4.0 * fk / (1.0 + fk ** 2) ** 2
        low = fk <= 2.0                           # the mesh resolves these wavelengths
        np.testing.assert_allclose(gk[low], exact[low], rtol=1e-4, atol=1e-7)
        self.assertEqual(gk[0], 0.0)
        self.assertEqual(back.shape, r.shape)

    def test_energy_mesh_and_radii(self) -> None:
        ecc = energy_mesh()
        self.assertEqual(ecc[0], -2.0)
        self.assertLessEqual(ecc[-1], 2.0)
        self.assertAlmostEqual(ecc[1] - ecc[0], 0.03, places=7)
        self.assertNotEqual(ecc[1] - ecc[0], 0.03)            # single precision step
        self.assertAlmostEqual(float(np.min(np.diff(ecc))), 0.003, places=7)
        self.assertEqual(rcovalent(14), float(np.float32(2.10)))
        self.assertEqual(rcovalent(100), 3.0)
        self.assertLess(rcovalent(101), 1e-300)               # beyond the table
        self.assertEqual(rcovalent(103), 0.0)
        with self.assertRaises(FortranStop):
            rcovalent(120)

    def test_derlkb_scalar_and_vector(self) -> None:
        r = np.array([1.0e-3 * 1.0247 ** i for i in range(401)])
        v = -1.0 / r * (1.0 - np.exp(-2.0 * r))
        vkb = r ** 2 * np.exp(-r * r)
        e = np.array([-1.0, -0.3, 0.2])
        vec = derlkb(1, e, 301, 0.7, len(r), r, vkb, v)
        for k in range(3):
            self.assertEqual(derlkb(1, float(e[k]), 301, 0.7, len(r), r, vkb, v), vec[k])
        with self.assertRaises(FortranStop):
            derlkb(1, e, 300, 0.7, len(r), r, vkb, v)         # even number of points

    def test_moments_used_by_kinkon(self) -> None:
        r = np.array([1.0e-4 * 1.0247 ** i for i in range(501)])
        al = math.log(1.0247)
        self.assertAlmostEqual(fmom(0, len(r), al, 1.0, r, 4.0 * np.exp(-2.0 * r)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

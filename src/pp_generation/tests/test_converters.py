"""Tests of the in-process converters (``pp_generation.converters``).

Run with::

    PYTHONPATH=src python -m unittest pp_generation.tests.test_converters -v

The references in ``data/converters`` come from the Fortran chain, not from
the code under test: ``fhi2upf_qe.x`` (Quantum ESPRESSO 7.5 upflib) wrote the
UPF files and the ``src/tools/upf_to_parsec.py`` of the commit before the
converters were moved wrote the POTRE files, under glibc.  ``expected.json``
holds the conversion arguments, SHA-256 digests and, for POTRE, per-section
sums: the POTRE resampling calls the C library's ``log``, ``expm1`` and
``pow``, whose last bit differs between platforms, so off glibc a POTRE file
agrees with the reference to about 1e-12 instead of byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pp_generation.converters import (
    DEFAULT_GENERATED,
    FHI2UPF_GENERATED,
    XC_LABELS,
    ConversionError,
    CpiTable,
    convert_outputs,
    cpi_to_upf,
    format_cpi,
    format_fhi,
    format_upf,
    infer_parsec_xc,
    parse_cpi,
    read_cpi,
    upf_to_potre,
    valence_channels,
    write_cpi,
    write_fhi,
    write_upf,
    xc_labels,
)
from pp_generation.converters._fortran import (
    correctly_rounded_log,
    format_e,
    format_es,
    format_list_directed,
    format_r2c,
    parse_real,
)
from pp_generation.converters.potre import parse_float
from pp_generation.errors import ConfigurationError
from pp_generation.models import OutputFormat

try:
    from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
except ImportError:  # the converters need neither numpy nor the solver
    read_parsec_pseudopotential = None


PACKAGE = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data" / "converters"
SHIM = PACKAGE.parent / "tools" / "upf_to_parsec.py"
# A library generated with the Fortran chain (FHI98PP, fhi2upf, upf_to_parsec), one
# folder per element with FHI/, UPF/, PARSEC/ and diagnostics/*.raw.UPF.  The library
# shipped in ``NCPPs`` is built with the Python port and has no such reference files.
LIBRARY = Path(os.environ.get("PP_GENERATION_FORTRAN_LIBRARY", PACKAGE / "NCPPs"))
CORE_HOLE = PACKAGE.parents[1] / "examples" / "0_CH4_CF4" / "fhi98pp_generated" / "C-1s"
EXPECTED = (
    json.loads((DATA / "expected.json").read_text(encoding="utf-8"))
    if (DATA / "expected.json").is_file() else None
)
needs_fixtures = unittest.skipUnless(
    EXPECTED is not None, f"reference files are missing from {DATA}"
)


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def potre_fingerprint(text: str) -> dict:
    """Text records of a POTRE file and three sums of every numeric block.

    A record with a word in it is kept verbatim; the numbers between two such
    records form one block, summarised as its length, sum, absolute sum and
    position-weighted sum.
    """

    records: list[str] = []
    blocks: list[list[float]] = []
    values: list[float] | None = None
    for line in text.splitlines():
        try:
            row = [parse_float(token) for token in line.split()]
        except ValueError:
            records.append(line)
            values = None
            continue
        if values is None:
            values = []
            blocks.append(values)
        values.extend(row)
    return {
        "records": records,
        "blocks": [
            [
                len(block),
                math.fsum(block),
                math.fsum(abs(value) for value in block),
                math.fsum(index * value for index, value in enumerate(block, start=1)),
            ]
            for block in blocks
        ],
    }


def potre_tokens(text: str) -> list[float | str]:
    tokens: list[float | str] = []
    for token in text.split():
        try:
            tokens.append(parse_float(token))
        except ValueError:
            tokens.append(token)
    return tokens


class PotreComparison(unittest.TestCase):
    """Assertions for POTRE text that may carry platform rounding."""

    def assertPotreMatches(self, text: str, expected: dict) -> None:
        if sha256(text) == expected["potre_sha256"]:
            return
        actual = potre_fingerprint(text)
        reference = expected["potre_fingerprint"]
        self.assertEqual(actual["records"], reference["records"])
        self.assertEqual(len(actual["blocks"]), len(reference["blocks"]))
        for mine, theirs in zip(actual["blocks"], reference["blocks"]):
            self.assertEqual(mine[0], theirs[0])
            tolerance = 1.0e-10 * theirs[2] + 1.0e-30
            self.assertLessEqual(abs(mine[1] - theirs[1]), tolerance)
            self.assertLessEqual(abs(mine[2] - theirs[2]), tolerance)
            self.assertLessEqual(abs(mine[3] - theirs[3]), tolerance * mine[0])

    def assertPotreClose(self, text: str, reference: str) -> None:
        if text == reference:
            return
        mine, theirs = potre_tokens(text), potre_tokens(reference)
        self.assertEqual(len(mine), len(theirs))
        for left, right in zip(mine, theirs):
            if isinstance(left, str) or isinstance(right, str):
                self.assertEqual(left, right)
            elif left != right:
                self.assertLessEqual(
                    abs(left - right), 1.0e-9 * max(abs(left), abs(right)) + 1.0e-25
                )


def small_table(core: bool = True, channels: int = 3, mesh: int = 21) -> CpiTable:
    """A toy table built with exact arithmetic only; it is not a potential."""

    def power(base: float, exponent: int) -> float:
        result = 1.0
        for _ in range(exponent):
            result = result * base
        return result

    radius = [0.01]
    for _ in range(mesh - 1):
        radius.append(radius[-1] * 1.5)
    wavefunctions = [
        [power(r, ell + 1) / power(1.0 + r * r, ell + 2) for r in radius]
        for ell in range(channels)
    ]
    potentials = [
        [-3.0 / r + (ell - 1.5) / power(1.0 + r, 8) for r in radius]
        for ell in range(channels)
    ]
    extra = {}
    if core:
        extra = dict(
            core_density=[2.0 / power(1.0 + r * r, 4) for r in radius],
            core_first_derivative=[-16.0 * r / power(1.0 + r * r, 5) for r in radius],
            core_second_derivative=[-16.0 / power(1.0 + r * r, 5) for r in radius],
        )
    return CpiTable.from_arrays(
        3.0, radius, wavefunctions, potentials, mesh_increment=1.5, **extra
    )


def upf_arguments(case: dict) -> dict:
    return dict(
        element=case["element"],
        functional=case["functional"],
        local_channel=case["local_channel"],
        labels=case["labels"],
        occupations=case["occupations"],
        lmax=case["lmax"],
    )


class FortranEditingTests(unittest.TestCase):
    def test_e_descriptor(self) -> None:
        self.assertEqual(format_e(0.56818181818182e-3, 20, 14), "0.56818181818182E-03")
        self.assertEqual(format_e(-0.39456428640058, 20, 14), "-.39456428640058E+00")
        self.assertEqual(format_e(1.0, 20, 14), "0.10000000000000E+01")
        self.assertEqual(format_e(0.0, 20, 14), "0.00000000000000E+00")
        self.assertEqual(format_e(-0.0, 20, 14), "-.00000000000000E+00")
        # No exponent letter once the exponent has three digits.
        self.assertEqual(format_e(0.10299042144031e-143, 20, 14), "0.10299042144031-143")
        self.assertEqual(format_e(-1.5e120, 20, 14), "-.15000000000000+121")
        with self.assertRaises(ConversionError):
            format_e(float("nan"), 20, 14)

    def test_es_descriptor(self) -> None:
        self.assertEqual(format_es(5.6818181818182e-4, 24, 15), "   5.681818181818200E-04")
        self.assertEqual(format_es(-1.47895226272142, 24, 15), "  -1.478952262721420E+00")
        self.assertEqual(format_es(0.0, 24, 15), "   0.000000000000000E+00")
        self.assertEqual(format_es(-0.0, 24, 15), "  -0.000000000000000E+00")
        self.assertEqual(format_es(1.234e-104, 24, 15), "   1.234000000000000-104")

    def test_list_directed_double(self) -> None:
        self.assertEqual(format_r2c(1.0), "1.0000000000000000")
        self.assertEqual(format_r2c(0.0), "0.0000000000000000")
        self.assertEqual(format_r2c(11.0), "11.000000000000000")
        self.assertEqual(format_r2c(76.437707995235996), "76.437707995235996")
        self.assertEqual(format_r2c(-5.0751738152338239), "-5.0751738152338239")
        self.assertEqual(format_r2c(2.4399886823535104e-2), "2.4399886823535104E-002")
        self.assertEqual(format_r2c(0.5), "0.50000000000000000")
        self.assertEqual(format_r2c(0.1), "0.10000000000000001")
        self.assertEqual(format_r2c(-2.4399886823535104e-2), "-2.4399886823535104E-002")
        self.assertEqual(
            format_list_directed(8.8026280773888690), "  8.8026280773888690     "
        )
        self.assertEqual(len(format_list_directed(-3.0e-5)), 25)

    def test_reals_are_read_as_fortran_reads_them(self) -> None:
        self.assertEqual(parse_real("0.5D+01"), 5.0)
        self.assertEqual(parse_real(".699054148690-131"), 0.699054148690e-131)
        self.assertEqual(parse_real("0.10299042144031E-144"), 0.10299042144031e-144)
        self.assertEqual(parse_float("1.234000000000000-104"), 1.234e-104)
        with self.assertRaises(ValueError):
            parse_real("1.0E")

    def test_logarithm_is_correctly_rounded(self) -> None:
        self.assertEqual(correctly_rounded_log(1.0), 0.0)
        self.assertEqual(correctly_rounded_log(1.0247), float.fromhex("0x1.8fc48afbb94c3p-6"))
        self.assertEqual(correctly_rounded_log(134531.66), float.fromhex("0x1.79e7df8d9e50ep+3"))
        with self.assertRaises(ConversionError):
            correctly_rounded_log(0.0)


class CpiTests(unittest.TestCase):
    def test_layout_is_psgen_s(self) -> None:
        lines = format_cpi(small_table()).splitlines()
        self.assertEqual(lines[0], "0.30000000000000E+01   3")
        self.assertEqual(lines[1], "  0.0000    0.0000    0.0000   0.0000")
        self.assertEqual(lines[2:11], ["  0.0000    .00e+00   .00e+00"] * 9)
        self.assertEqual(lines[11], "21  0.15000000000000E+01")
        self.assertEqual(
            lines[12],
            "   1 0.10000000000000E-01 0.99980002999600E-02 -.30138522483372E+03",
        )
        self.assertEqual(len(lines), 11 + 3 * 22 + 21)
        self.assertEqual(
            lines[-21],
            "0.10000000000000E-01 0.19992001999600E+01 -.15992002399440E+00 "
            "-.15992002399440E+02",
        )

    def test_text_round_trip_is_exact(self) -> None:
        for core in (True, False):
            text = format_cpi(small_table(core=core))
            table = parse_cpi(text)
            self.assertEqual(table.has_core_correction, core)
            self.assertEqual(table.lmax, 2)
            self.assertEqual(table.mesh_size, 21)
            self.assertEqual(table.valence_charge, 3.0)
            self.assertEqual(table.mesh_increment, 1.5)
            self.assertIsNone(table.abinit_header)
            self.assertEqual(format_cpi(table), text)
            # Trailing blank lines are not a core block.
            self.assertEqual(parse_cpi(text + "\n\n"), table)

    def test_files_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.cpi"
            write_cpi(path, small_table())
            self.assertNotIn(b"\r", path.read_bytes())
            self.assertEqual(format_cpi(read_cpi(path)), format_cpi(small_table()))
            with self.assertRaises(ConversionError):
                read_cpi(Path(directory) / "absent.cpi")

    def test_abinit_header_follows_the_library_files(self) -> None:
        # Header of 11-Na.GGA.fhi of the FHI98PP library for abinit, with the
        # mesh size of the toy table.
        library = (
            "sodium, fhi98PP : Trouiller-Martins-type, GGA Perdew/Burke/Ernzerhof (1996), l= 2 local\n"
            " 11.000  3.000    021003              zatom,zion,pspdat\n"
            "   6        11    2   2     21     0       pspcod,pspxc,lmax,lloc,mmax,r2well\n"
            "  6.043  1.000  0.000                 rchrg, fchrg, qchrg\n"
            "5--- These two lines are available for giving more information, later\n"
            "6\n"
            "7-Here follows the cpi file from the fhi98pp code-\n"
        )
        table = small_table()
        text = format_fhi(
            table, zatom=11.0, pspxc=11, lloc=2, title=library.splitlines()[0],
            pspdat=21003, rchrg=6.043,
        )
        self.assertEqual(text, library + format_cpi(table))
        header = parse_cpi(text).abinit_header
        self.assertIsNotNone(header)
        self.assertEqual(
            (header.zatom, header.zion, header.pspdat, header.pspcod, header.pspxc),
            (11.0, 3.0, 21003, 6, 11),
        )
        self.assertEqual((header.lmax, header.lloc, header.mmax), (2, 2, 21))
        self.assertEqual((header.rchrg, header.fchrg, header.qchrg), (6.043, 1.0, 0.0))
        self.assertTrue(header.title.startswith("sodium, fhi98PP"))
        # The body is the same table.
        self.assertEqual(format_cpi(parse_cpi(text)), format_cpi(table))
        # zatom >= 100 widens its field, as in 103-Lr.GGA.fhi.
        wide = format_fhi(table, zatom=103.0, pspxc=11, lloc=0, title="lawrencium")
        self.assertEqual(
            wide.splitlines()[1], " 103.000  3.000    000000              zatom,zion,pspdat"
        )

    def test_abinit_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.fhi"
            write_fhi(path, small_table(core=False), zatom=5.0, pspxc=2, lloc=1, title="toy")
            header = read_cpi(path).abinit_header
            self.assertEqual((header.rchrg, header.fchrg, header.pspxc, header.lloc), (0.0, 0.0, 2, 1))
            write_fhi(path, small_table(), zatom=5.0, pspxc=7, lloc=0, title="toy")
            header = read_cpi(path).abinit_header
            self.assertEqual(header.fchrg, 1.0)
            # First mesh radius from which 4*pi*rho_core stays below 1e-8.
            table = small_table()
            index = next(
                i for i, r in enumerate(table.radius) if abs(r - header.rchrg) < 5.0e-4
            )
            self.assertLess(table.core.density[index], 1.0e-8)
            self.assertGreaterEqual(table.core.density[index - 1], 1.0e-8)
        with self.assertRaises(ConversionError):
            format_fhi(small_table(), zatom=5.0, pspxc=11, lloc=3, title="toy")
        with self.assertRaises(ConversionError):
            format_fhi(small_table(), zatom=5.0, pspxc=11, lloc=0, title="3.0 3 looks like a table")

    def test_malformed_tables_are_refused(self) -> None:
        text = format_cpi(small_table())
        lines = text.splitlines()
        with self.assertRaises(ConversionError):
            parse_cpi("\n".join(lines[:40]))  # ends inside a channel
        with self.assertRaises(ConversionError):
            parse_cpi("\n".join(lines[:-3]))  # incomplete core block
        shifted = lines[:]
        shifted[-1] = shifted[-1].replace("0.33252567300797E+02", "0.33352567300797E+02")
        self.assertNotEqual(shifted, lines)
        with self.assertRaises(ConversionError):
            parse_cpi("\n".join(shifted))  # core block on another mesh
        other_mesh = lines[:]
        other_mesh[11 + 22] = "21  0.15100000000000E+01"
        with self.assertRaises(ConversionError):
            parse_cpi("\n".join(other_mesh))
        with self.assertRaises(ConversionError):
            parse_cpi("not a table\n")
        abinit = format_fhi(small_table(), zatom=5.0, pspxc=11, lloc=0, title="toy")
        with self.assertRaises(ConversionError):
            parse_cpi(abinit.replace("   6        11", "   8        11"))  # pspcod
        with self.assertRaises(ConversionError):
            parse_cpi(abinit.replace("  5.000  3.000", "  5.000  4.000"))  # zion

    def test_table_validation(self) -> None:
        radius = [0.1, 0.2, 0.4]
        with self.assertRaises(ConversionError):
            CpiTable.from_arrays(1.0, radius, [[0.0] * 3], [[0.0] * 2])
        with self.assertRaises(ConversionError):
            CpiTable.from_arrays(1.0, [0.1, 0.1, 0.4], [[0.0] * 3], [[0.0] * 3])
        with self.assertRaises(ConversionError):
            CpiTable.from_arrays(0.0, radius, [[0.0] * 3], [[0.0] * 3])
        with self.assertRaises(ConversionError):
            CpiTable.from_arrays(1.0, radius, [[0.0] * 3], [[float("nan")] * 3])
        with self.assertRaises(ConversionError):
            CpiTable.from_arrays(1.0, radius, [[0.0] * 3], [[0.0] * 3], core_density=[0.0] * 3)
        table = CpiTable.from_arrays(1.0, radius, [[0.0] * 3], [[0.0] * 3])
        self.assertEqual(table.mesh_increment, 2.0)
        self.assertEqual(table.channels[0].potential, (0.0, 0.0, 0.0))


class UpfTests(unittest.TestCase):
    def convert(self, table: CpiTable, **overrides):
        arguments = dict(
            element="B", functional="PBE", local_channel=2,
            labels=("2s", "2p", "2d"), occupations=(2.0, 1.0, 0.0),
        )
        arguments.update(overrides)
        return cpi_to_upf(table, **arguments)

    def test_document_structure(self) -> None:
        upf = self.convert(
            small_table(), total_psenergy_ry=-5.25, total_aeenergy_ry=-49.5
        )
        self.assertEqual((upf.l_max, upf.l_local, len(upf.projectors)), (1, 2, 2))
        self.assertEqual(upf.zmesh, 5.0)
        self.assertEqual(upf.generated, DEFAULT_GENERATED)
        text = format_upf(upf)
        root = ET.fromstring(text)
        header = root.find("PP_HEADER").attrib
        self.assertEqual(root.get("version"), "2.0.1")
        self.assertEqual(header["pseudo_type"], "SL")
        self.assertEqual(header["relativistic"], "scalar")
        self.assertEqual(header["core_correction"], "true")
        self.assertEqual(header["functional"], "PBE")
        self.assertEqual(header["z_valence"], "3.0000000000000000")
        self.assertEqual(header["total_psenergy"], "-5.2500000000")
        self.assertEqual(header["total_aeenergy"], "-49.5000000000")
        self.assertEqual((header["l_max"], header["l_local"]), ("1", "2"))
        self.assertEqual((header["number_of_wfc"], header["number_of_proj"]), ("3", "2"))
        self.assertEqual(header["mesh_size"], "21")
        # annotate_upf_energies puts the nonstandard attribute on its own line.
        self.assertIn('number_of_proj="2"\ntotal_aeenergy="-49.5000000000"/>', text)
        self.assertEqual(
            [child.tag for child in root.find("PP_SEMILOCAL")], ["PP_VNL.0", "PP_VNL.1"]
        )
        chi = list(root.find("PP_PSWFC"))
        self.assertEqual([item.get("label") for item in chi], ["2s", "2p", "2d"])
        # As in QE, n is written only when it exceeds l.
        self.assertEqual([item.get("n") for item in chi], ["2", "2", None])
        self.assertEqual(len(root.find("PP_NLCC").text.split()), 21)
        self.assertEqual(root.find("PP_MESH").get("zmesh"), "5.0000000000000000")
        # Potentials are doubled to Rydberg, the core density divided by 4 pi.
        table = small_table().as_written()
        self.assertEqual(upf.local_potential[0], 2.0 * table.channels[2].potential[0])
        self.assertAlmostEqual(
            upf.core_density[0] * 4.0 * math.pi, table.core.density[0], places=14
        )
        self.assertEqual(
            upf.valence_density[3],
            2.0 * (table.channels[0].wavefunction[3] * table.channels[0].wavefunction[3])
            + 1.0 * (table.channels[1].wavefunction[3] * table.channels[1].wavefunction[3]),
        )

    def test_local_channel_below_lmax_and_all_local(self) -> None:
        upf = self.convert(small_table(), local_channel=0)
        self.assertEqual((upf.l_max, len(upf.projectors)), (2, 2))
        self.assertEqual([item.angular_momentum for item in upf.projectors], [1, 2])
        upf = self.convert(
            small_table(channels=1), local_channel=0, labels=("2s",), occupations=(2.0,)
        )
        self.assertEqual((upf.l_max, len(upf.projectors)), (-1, 0))
        text = format_upf(upf)
        self.assertIn("  <PP_SEMILOCAL>\n  </PP_SEMILOCAL>\n", text)
        self.assertIn('    <PP_DIJ columns="0" rows="0">\n    </PP_DIJ>\n', text)
        # A lower lmax drops the highest channels.
        upf = self.convert(
            small_table(), lmax=1, local_channel=1, labels=("2s", "2p"),
            occupations=(2.0, 1.0),
        )
        self.assertEqual((upf.l_max, len(upf.wavefunctions), len(upf.projectors)), (0, 2, 1))

    def test_raw_form_and_provenance(self) -> None:
        table = small_table(core=False)
        raw = format_upf(self.convert(table, generated=FHI2UPF_GENERATED))
        self.assertIn('total_psenergy="0.0000000000000000"', raw)
        self.assertNotIn("total_aeenergy", raw)
        self.assertNotIn("PP_NLCC", raw)
        self.assertIn('core_correction="false"', raw)
        ours = format_upf(self.convert(table))
        differing = [
            (left, right)
            for left, right in zip(raw.splitlines(), ours.splitlines())
            if left != right
        ]
        self.assertEqual(len(raw.splitlines()), len(ours.splitlines()))
        self.assertEqual(len(differing), 2)  # PP_INFO line and PP_HEADER
        self.assertTrue(all("fhi2upf.x v.5.0.2" in left for left, _ in differing))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.UPF"
            write_upf(path, self.convert(table))
            self.assertEqual(path.read_bytes(), ours.encode("ascii"))

    def test_table_is_converted_as_its_file_holds_it(self) -> None:
        # Full-precision arrays and the 14-digit file written from them must
        # not give two different UPF files.
        table = small_table()
        written = parse_cpi(format_cpi(table))
        self.assertNotEqual(table.channels[1].potential, written.channels[1].potential)
        self.assertEqual(table.as_written(), written)
        self.assertEqual(written.as_written(), written)
        self.assertEqual(format_upf(self.convert(table)), format_upf(self.convert(written)))
        self.assertEqual(self.convert(table).radius, written.radius)

    def test_bad_arguments_are_refused(self) -> None:
        table = small_table()
        for overrides in (
            dict(element="Xx"),
            dict(local_channel=3),
            dict(local_channel=-1),
            dict(lmax=3),
            dict(labels=("2s", "2p")),
            dict(labels=("2s", "2d", "2p")),
            dict(labels=("2s", "2p", "3dx")),
            dict(occupations=(2.0, 1.0, -1.0)),
            dict(occupations=(2.0, 1.0, float("inf"))),
            dict(functional='PBE" x="'),
            dict(functional=" "),
            dict(relativistic="full"),
        ):
            with self.assertRaises(ConversionError, msg=str(overrides)):
                self.convert(table, **overrides)
        # Free provenance text cannot break the document.
        text = format_upf(self.convert(table, generated='a <b> & "c"', relativistic="no"))
        header = ET.fromstring(text).find("PP_HEADER").attrib
        self.assertEqual(header["generated"], "a *b* @ 'c'")
        self.assertEqual(header["relativistic"], "no")
        self.assertIn("Non-Relativistic Calculation", text)

    @needs_fixtures
    def test_synthetic_tables_match_fhi2upf(self) -> None:
        """Files written by the real ``fhi2upf_qe.x`` for stress tables.

        Together the cases cover even and odd meshes and projector extents,
        mesh increments other than FHI98PP's, potential differences at the
        1e-6 Ry projector threshold, three-digit exponents, negative zeros,
        a core correction, a truncated lmax and every position of the local
        channel.
        """
        self.assertGreaterEqual(len(EXPECTED["synthetic"]), 4)
        for case in EXPECTED["synthetic"]:
            with self.subTest(case=case["name"]):
                table = read_cpi(DATA / f"{case['name']}.cpi")
                reference = (DATA / f"{case['name']}.raw.UPF").read_text(encoding="ascii")
                text = format_upf(
                    cpi_to_upf(table, generated=FHI2UPF_GENERATED, **upf_arguments(case))
                )
                self.assertEqual(text.splitlines(), reference.splitlines())
                self.assertEqual(text, reference)

    @needs_fixtures
    def test_generated_potentials_match_fhi2upf(self) -> None:
        for case in EXPECTED["potentials"]:
            with self.subTest(case=case["name"]):
                path = DATA / case["cpi"]
                table = read_cpi(path)
                # FHI98PP's own file is written back byte for byte.
                self.assertEqual(format_cpi(table).encode("ascii"), path.read_bytes())
                self.assertEqual(sha256(path.read_bytes()), case["cpi_sha256"])
                arguments = upf_arguments(case)
                raw = format_upf(cpi_to_upf(table, generated=FHI2UPF_GENERATED, **arguments))
                self.assertEqual(sha256(raw), case["raw_upf_sha256"])
                annotated = format_upf(
                    cpi_to_upf(
                        table, generated=FHI2UPF_GENERATED,
                        total_psenergy_ry=2.0 * case["pseudo_energy_hartree"],
                        total_aeenergy_ry=2.0 * case["ae_energy_hartree"],
                        **arguments,
                    )
                )
                self.assertEqual(sha256(annotated), case["upf_sha256"])


class PotreTests(PotreComparison):
    def test_functional_labels(self) -> None:
        self.assertEqual(infer_parsec_xc("PBE"), "pb")
        for label in ("PZ", "SLA-PZ", "CA"):
            self.assertEqual(infer_parsec_xc(label), "ca")
        for label in ("PW", "SLA-PW", "sla_pw", "PW92"):
            self.assertEqual(infer_parsec_xc(label), "pl")
        for label in ("LDA", "PW91", "BLYP", ""):
            with self.assertRaises(ConversionError):
                infer_parsec_xc(label)

    @needs_fixtures
    def test_generated_potentials_match_the_reference(self) -> None:
        for case in EXPECTED["potentials"]:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as tmp:
                written = convert_outputs(
                    DATA / case["cpi"], tmp, element=case["element"], xc=case["xc"],
                    local_channel=case["local_channel"], valence=case["valence"],
                    prefix=case["prefix"], lmax=case["lmax"],
                    ae_energy_hartree=case["ae_energy_hartree"],
                    pseudo_energy_hartree=case["pseudo_energy_hartree"],
                    ionized_reference=case["ionized_reference"],
                    generated=FHI2UPF_GENERATED,
                )
                self.assertEqual(sha256(written["upf"].read_bytes()), case["upf_sha256"])
                self.assertEqual(sha256(written["fhi"].read_bytes()), case["cpi_sha256"])
                potre = written["parsec"].read_bytes()
                self.assertNotIn(b"\r", potre)
                self.assertPotreMatches(potre.decode("ascii"), case)
                # The provenance line of the UPF does not reach the POTRE file.
                again = convert_outputs(
                    DATA / case["cpi"], Path(tmp) / "ours", element=case["element"],
                    xc=case["xc"], local_channel=case["local_channel"],
                    valence=case["valence"], prefix=case["prefix"], lmax=case["lmax"],
                    ionized_reference=case["ionized_reference"], formats=("parsec",),
                )
                self.assertEqual(again["parsec"].read_bytes(), potre)
                self.assertEqual(list(again), ["parsec"])

    @needs_fixtures
    def test_ionized_reference_needs_the_flag(self) -> None:
        case = next(item for item in EXPECTED["potentials"] if item["ionized_reference"])
        with tempfile.TemporaryDirectory() as tmp:
            arguments = dict(
                element=case["element"], xc=case["xc"],
                local_channel=case["local_channel"], valence=case["valence"],
                prefix=case["prefix"], lmax=case["lmax"],
            )
            with self.assertRaises(ConversionError) as caught:
                convert_outputs(DATA / case["cpi"], tmp, **arguments)
            self.assertIn("z_valence", str(caught.exception))
            # Nothing is written when a conversion is refused.
            self.assertEqual(list(Path(tmp).iterdir()), [])
            written = convert_outputs(
                DATA / case["cpi"], tmp, ionized_reference=True, **arguments
            )
            self.assertEqual(sorted(written), ["fhi", "parsec", "upf"])

    @needs_fixtures
    def test_command_line_tool_is_the_same_converter(self) -> None:
        case = EXPECTED["potentials"][0]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            written = convert_outputs(
                DATA / case["cpi"], directory, element=case["element"], xc=case["xc"],
                local_channel=case["local_channel"], valence=case["valence"],
                prefix=case["prefix"], lmax=case["lmax"],
            )
            output = directory / "tool_POTRE.DAT"
            command = [sys.executable, str(SHIM), str(written["upf"]), str(output)]
            done = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn(f"wrote {output}", done.stdout)
            self.assertIn("Correlation_Type: pb", done.stdout)
            self.assertEqual(output.read_bytes(), written["parsec"].read_bytes())
            # An existing output is kept unless --force is given.
            done = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn("use --force", done.stderr)
            done = subprocess.run(
                command[:2] + ["--force", "--xc-code", "ca"] + command[2:],
                capture_output=True, text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertTrue(output.read_text().startswith(f" {case['element']:<2} ca nrl"))
            # In-process entry point.
            data = upf_to_potre(written["upf"], directory / "direct_POTRE.DAT")
            self.assertEqual(data.xc_code, "pb")
            self.assertEqual(
                (directory / "direct_POTRE.DAT").read_bytes(), written["parsec"].read_bytes()
            )
            with self.assertRaises(FileExistsError):
                upf_to_potre(written["upf"], directory / "direct_POTRE.DAT", overwrite=False)

    def test_command_line_tool_needs_only_the_standard_library(self) -> None:
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); import upf_to_parsec; "
            "assert callable(upf_to_parsec.main) and callable(upf_to_parsec.parse_upf); "
            "print('numpy' in sys.modules, 'pp_generation' in sys.modules)"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe, str(SHIM.parent)], capture_output=True, text=True
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["False", "False"])
        done = subprocess.run(
            [sys.executable, str(SHIM), "--help"], capture_output=True, text=True
        )
        self.assertEqual(done.returncode, 0)
        for option in ("--xc-code", "--allow-ionized-reference", "--force", "--grid-refinement"):
            self.assertIn(option, done.stdout)

    @needs_fixtures
    @unittest.skipIf(read_parsec_pseudopotential is None, "the solver's reader is unavailable")
    def test_solver_reads_the_result(self) -> None:
        for case in EXPECTED["potentials"]:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as tmp:
                written = convert_outputs(
                    DATA / case["cpi"], tmp, element=case["element"], xc=case["xc"],
                    local_channel=case["local_channel"], valence=case["valence"],
                    prefix=case["prefix"], lmax=case["lmax"],
                    ionized_reference=case["ionized_reference"],
                )
                table = read_cpi(DATA / case["cpi"])
                potential = read_parsec_pseudopotential(written["parsec"])
                self.assertEqual(potential.symbol, case["element"])
                self.assertEqual(potential.correlation, XC_LABELS[case["xc"]].parsec_code)
                self.assertEqual(potential.ionic_charge, table.valence_charge)
                self.assertEqual(potential.number_of_channels, case["lmax"] + 1)
                self.assertEqual(
                    potential.has_nonlinear_core_correction, table.has_core_correction
                )
                self.assertEqual(float(potential.core_density.max()) > 0.0, table.has_core_correction)
                potential.validate_local_channel(case["local_channel"])
                self.assertAlmostEqual(
                    sum(potential.channel_occupations.values()),
                    sum(occupation for _, _, occupation in case["valence"]),
                )
                for angular_momentum in range(case["lmax"] + 1):
                    if angular_momentum != case["local_channel"]:
                        projector, sign = potential.radial_projector(
                            angular_momentum, case["local_channel"]
                        )
                        self.assertIn(sign, (-1.0, 1.0))
                        self.assertTrue(all(math.isfinite(value) for value in projector))


class FacadeTests(unittest.TestCase):
    def test_functional_labels_of_every_format(self) -> None:
        self.assertEqual(sorted(XC_LABELS), ["ca", "pbe", "pw92"])
        pbe, ca, pw92 = xc_labels("pbe"), xc_labels("ca"), xc_labels("pw92")
        self.assertEqual(
            (pbe.fhi98pp_iexc, pbe.upf_functional, pbe.parsec_code, pbe.abinit_pspxc),
            (6, "PBE", "pb", 11),
        )
        self.assertEqual(
            (ca.fhi98pp_iexc, ca.upf_functional, ca.parsec_code, ca.abinit_pspxc),
            (3, "PZ", "ca", 2),
        )
        self.assertEqual(
            (pw92.fhi98pp_iexc, pw92.upf_functional, pw92.parsec_code, pw92.abinit_pspxc),
            (8, "PW", "pl", 7),
        )
        for labels in XC_LABELS.values():
            self.assertEqual(infer_parsec_xc(labels.upf_functional), labels.parsec_code)
        for unknown in ("lda", "PBE", "pw91", "", None):
            with self.assertRaises(ConfigurationError):
                xc_labels(unknown)
        # The generator's input writer must mean the same functionals.
        from pp_generation.fhi_input import XC_CODES

        for name, code in XC_CODES.items():
            self.assertEqual(xc_labels(name).fhi98pp_iexc, code, name)

    def test_valence_channels(self) -> None:
        self.assertEqual(
            valence_channels([(3, 0, 1.0)], 2), (("3s", "3p", "3d"), (1.0, 0.0, 0.0))
        )
        self.assertEqual(
            valence_channels([(3, 2, 6.0), (4, 0, 2.0)], 2),
            (("4s", "4p", "3d"), (2.0, 0.0, 6.0)),
        )
        self.assertEqual(valence_channels([(2, 0, 2.0), (2, 1, 0.5)], 0), (("2s",), (2.0,)))
        with self.assertRaises(ConfigurationError):
            valence_channels([], 1)

    def test_requests_are_checked_before_anything_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "out"
            arguments = dict(
                element="B", local_channel=2, valence=[(2, 0, 2.0), (2, 1, 1.0)],
                prefix="b_toy",
            )
            with self.assertRaises(ConfigurationError):
                convert_outputs(small_table(), directory, xc="lda", **arguments)
            with self.assertRaises(ConfigurationError):
                convert_outputs(
                    small_table(), directory, xc="pbe", formats=("psp8",), **arguments
                )
            self.assertFalse(directory.exists())

    def test_formats_and_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            written = convert_outputs(
                small_table(), directory, element="B", xc="ca", local_channel=1,
                valence=[(2, 0, 2.0), (2, 1, 1.0)], prefix="b_ca_tm",
                formats=(OutputFormat.FHI, "abinit", OutputFormat.UPF, "upf"),
                ae_energy_hartree=-24.5, pseudo_energy_hartree=-2.625,
                species_label="B-1s",
            )
            self.assertEqual(
                {kind: path.name for kind, path in written.items()},
                {
                    "fhi": "B_FHIPP.DAT", "fhi_species": "B-1s_FHIPP.DAT",
                    "abinit": "b_ca_tm.fhi", "upf": "b_ca_tm.UPF",
                },
            )
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["B-1s_FHIPP.DAT", "B_FHIPP.DAT", "b_ca_tm.UPF", "b_ca_tm.fhi"],
            )
            self.assertEqual(written["fhi"].read_text(), format_cpi(small_table()))
            self.assertEqual(
                written["fhi_species"].read_bytes(), written["fhi"].read_bytes()
            )
            header = ET.parse(written["upf"]).getroot().find("PP_HEADER").attrib
            self.assertEqual(header["functional"], "PZ")
            self.assertEqual(header["l_local"], "1")
            # Hartree in, Rydberg in the file.
            self.assertEqual(header["total_psenergy"], "-5.2500000000")
            self.assertEqual(header["total_aeenergy"], "-49.0000000000")
            self.assertEqual(header["generated"], DEFAULT_GENERATED)
            abinit = read_cpi(written["abinit"])
            self.assertEqual(
                (abinit.abinit_header.zatom, abinit.abinit_header.pspxc, abinit.abinit_header.lloc),
                (5.0, 2, 1),
            )
            self.assertEqual(format_cpi(abinit), format_cpi(small_table()))
            for xc, functional, pspxc in (("pbe", "PBE", 11), ("pw92", "PW", 7)):
                written = convert_outputs(
                    small_table(), directory / xc, element="B", xc=xc, local_channel=2,
                    valence=[(2, 0, 2.0), (2, 1, 1.0)], prefix="b",
                    formats=("upf", "abinit"),
                )
                header = ET.parse(written["upf"]).getroot().find("PP_HEADER").attrib
                self.assertEqual(header["functional"], functional)
                self.assertEqual(read_cpi(written["abinit"]).abinit_header.pspxc, pspxc)

    @needs_fixtures
    def test_lda_labels_reach_the_potre_file(self) -> None:
        # Only the labels are under test: the table itself is a PBE potential.
        case = next(
            item for item in EXPECTED["potentials"]
            if not item["ionized_reference"] and item["lmax"] > 0
        )
        for xc, line in (("ca", "ca nrl"), ("pw92", "pl nrl"), ("pbe", "pb nrl")):
            with self.subTest(xc=xc), tempfile.TemporaryDirectory() as tmp:
                written = convert_outputs(
                    DATA / case["cpi"], tmp, element=case["element"], xc=xc,
                    local_channel=case["local_channel"], valence=case["valence"],
                    prefix="relabelled", formats=("upf", "parsec"),
                    species_label=f"{case['element']}-x",
                )
                first = written["parsec"].read_text().splitlines()[0]
                self.assertEqual(first[1:3].strip(), case["element"])
                self.assertEqual(first[4:10], line)
                self.assertEqual(
                    written["parsec_species"].name, f"{case['element']}-x_POTRE.DAT"
                )
                self.assertEqual(
                    written["parsec_species"].read_bytes(), written["parsec"].read_bytes()
                )
                self.assertIn(
                    "converted from relabelled.UPF (UPF 2.0.1",
                    written["parsec"].read_text().splitlines()[1],
                )


@unittest.skipUnless(
    any(LIBRARY.glob("*/diagnostics/*.cpi")),
    "the generated NCPPs library (not part of a clone) is unavailable",
)
class LibraryReproductionTests(PotreComparison):
    """Every element of the PBE library, when its generation files are here."""

    def cases(self):
        from pp_generation.fhi_input import parse
        from pp_generation.parsers import energies

        for directory in sorted(LIBRARY.iterdir()):
            reports = sorted((directory / "diagnostics").glob("*.report.json"))
            if not reports:
                continue
            report = json.loads(reports[0].read_text())
            stem = directory / "diagnostics" / report["prefix"]
            yield dict(
                element=directory.name, prefix=report["prefix"],
                cpi=stem.with_suffix(".cpi"), raw=Path(f"{stem}.raw.UPF"),
                upf=directory / "UPF" / f"{report['prefix']}.UPF",
                potre=directory / "PARSEC" / f"{directory.name}_POTRE.DAT",
                parsed=parse(stem.with_suffix(".ini").read_text()),
                energies=energies(stem.with_suffix(".dat")),
                local_channel=report["selected_local_channel"], species_label=None,
            )
        stem = CORE_HOLE / "C-1s_generated"
        if stem.with_suffix(".cpi").is_file():
            report = json.loads(Path(f"{stem}.report.json").read_text())
            yield dict(
                element="C", prefix=stem.name, cpi=stem.with_suffix(".cpi"),
                raw=Path(f"{stem}.raw.UPF"), upf=stem.with_suffix(".UPF"),
                potre=CORE_HOLE / "C-1s_POTRE.DAT",
                parsed=parse(stem.with_suffix(".ini").read_text()),
                energies=energies(stem.with_suffix(".dat")),
                local_channel=report["selected_local_channel"], species_label="C-1s",
            )

    def test_library_is_reproduced(self) -> None:
        count = 0
        for case in self.cases():
            with self.subTest(case=case["prefix"]), tempfile.TemporaryDirectory() as tmp:
                parsed = case["parsed"]
                table = read_cpi(case["cpi"])
                self.assertEqual(format_cpi(table).encode("ascii"), case["cpi"].read_bytes())
                labels, occupations = valence_channels(parsed.valence, parsed.lmax)
                raw = format_upf(
                    cpi_to_upf(
                        table, element=case["element"], functional="PBE",
                        local_channel=case["local_channel"], labels=labels,
                        occupations=occupations, lmax=parsed.lmax,
                        generated=FHI2UPF_GENERATED,
                    )
                )
                self.assertEqual(raw.encode("ascii"), case["raw"].read_bytes())
                ae_energy, pseudo_energy = case["energies"]
                written = convert_outputs(
                    table, tmp, element=case["element"], xc="pbe",
                    local_channel=case["local_channel"], valence=parsed.valence,
                    prefix=case["prefix"], ae_energy_hartree=ae_energy,
                    pseudo_energy_hartree=pseudo_energy,
                    ionized_reference=(
                        abs(parsed.reference_electrons - parsed.ionic_charge) > 1.0e-8
                    ),
                    species_label=case["species_label"], generated=FHI2UPF_GENERATED,
                )
                self.assertEqual(written["upf"].read_bytes(), case["upf"].read_bytes())
                kind = "parsec_species" if case["species_label"] else "parsec"
                self.assertPotreClose(
                    written[kind].read_text(), case["potre"].read_text()
                )
                count += 1
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()

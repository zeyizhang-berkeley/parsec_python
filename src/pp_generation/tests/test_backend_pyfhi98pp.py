"""Behaviour of the built-in backend: template rules, core holes matched to the
ground state, files written.  Runs without any external program."""

import json
import tempfile
import unittest
from pathlib import Path

from pp_generation import CoreHole, GenerationRequest, generate
from pp_generation.errors import ConfigurationError
from pp_generation.fhi_input import parse
from pp_generation.models import BackendName, OutputFormat

try:
    from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
except ImportError:  # the generator does not need the solver
    read_parsec_pseudopotential = None


def run(tmp: str, element: str, **controls):
    controls.setdefault("formats", ())
    return generate(GenerationRequest(
        element=element, output_dir=Path(tmp), backend=BackendName.PYFHI98PP, **controls,
    ))


def radii(result) -> dict[int, float]:
    return {x.angular_momentum: x.cutoff_radius for x in result.channels}


class GroundStateTests(unittest.TestCase):
    def test_carbon_pbe_matches_the_fortran_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp, "C", xc="pbe")
        self.assertEqual((result.lmax, result.selected_local_channel), (2, 2))
        self.assertTrue(result.ghost_free)
        # FHI98PP (Fortran) prints -37.76478 and -5.36415 Ha for this input.
        self.assertAlmostEqual(result.ae_energy_hartree, -37.76478, places=5)
        self.assertAlmostEqual(result.pseudo_energy_hartree, -5.36415, places=5)
        selected = next(x for x in result.local_channel_results if x.local_channel == 2)
        self.assertAlmostEqual(selected.minimum_margin_hartree, 0.0804, places=4)

    def test_reviewed_override_makes_hydrogen_purely_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp, "H", xc="ca")
        self.assertEqual(result.lmax, 0)
        self.assertEqual(result.selected_local_channel, 0)
        self.assertEqual(result.local_channel_results[0].channels, ())

    def test_lmax_is_reduced_until_a_ghost_free_form_exists(self) -> None:
        # Ga: the template lmax 2 has no ghost-free local channel, lmax 1 has.
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp, "Ga", xc="pbe")
        self.assertEqual((result.lmax, result.selected_local_channel), (1, 1))
        self.assertTrue(result.ghost_free)

    def test_ca_is_the_perdew_zunger_functional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp, "Si", xc="ca")
            self.assertEqual(parse(result.input_path.read_text()).xc_code, 3)
        # FHI98PP (Fortran) with iexc 3 prints -288.82125 Ha.
        self.assertAlmostEqual(result.ae_energy_hartree, -288.82125, places=5)

    def test_unknown_format_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigurationError):
                run(tmp, "Si", formats=(OutputFormat.PSP8,))


class CoreHoleTests(unittest.TestCase):
    def test_hole_takes_lmax_radii_and_local_channel_of_the_ground_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ground = run(tmp + "/g", "Si", xc="pbe")
            hole = run(tmp + "/h", "Si", xc="pbe", core_hole=CoreHole("2p"))
        self.assertTrue(hole.matched_to_ground_state)
        self.assertEqual(hole.lmax, ground.lmax)
        self.assertEqual(hole.selected_local_channel, ground.selected_local_channel)
        self.assertEqual(radii(hole), radii(ground))
        self.assertEqual(hole.ionic_charge, ground.ionic_charge + 1.0)
        # FHI98PP (Fortran): Si with 2p^5 gives -285.79258 Ha.
        self.assertAlmostEqual(hole.ae_energy_hartree, -285.79258, places=5)

    def test_template_with_an_empty_valence_state(self) -> None:
        # Ca lists the empty 3d as a valence state.  An explicit radius record would
        # reset the reference energy of that channel and the construction fails; the
        # hole must be generated from the same records as the ground state.
        with tempfile.TemporaryDirectory() as tmp:
            ground = run(tmp + "/g", "Ca", xc="pbe")
            hole = run(tmp + "/h", "Ca", xc="pbe", core_hole=CoreHole("1s"))
            records = parse(hole.input_path.read_text()).channel_lines
        self.assertTrue(hole.matched_to_ground_state)
        self.assertEqual(radii(hole), radii(ground))
        self.assertEqual([ell for ell, *_ in records], [1])       # only the template's p record
        d_channel = next(x for x in hole.channels if x.angular_momentum == 2)
        # the d channel is built at the 3d eigenvalue of the core-hole atom, not at emax
        self.assertLess(d_channel.ae_eigenvalue_ev, -5.0)

    def test_reference_energies_follow_the_valence_levels(self) -> None:
        # Zr: p channel at +15 eV in the template; unusable for the ion without the shift.
        with tempfile.TemporaryDirectory() as tmp:
            hole = run(tmp, "Zr", xc="pbe", core_hole=CoreHole("1s"))
        self.assertTrue(hole.ghost_free)
        self.assertLess(hole.reference_shift_ev, -5.0)
        self.assertTrue(any("shifted" in text for text in hole.warnings))

    def test_reviewed_radius_is_used_for_ground_state_and_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ground = run(tmp + "/g", "K", xc="pbe")
            hole = run(tmp + "/h", "K", xc="pbe", core_hole=CoreHole("1s"))
        self.assertAlmostEqual(radii(ground)[0], 3.1732, places=4)
        self.assertEqual(radii(hole), radii(ground))
        self.assertTrue(hole.matched_to_ground_state)


class OutputTests(unittest.TestCase):
    def test_files_and_report(self) -> None:
        formats = (OutputFormat.FHI, OutputFormat.UPF, OutputFormat.PARSEC, OutputFormat.ABINIT)
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp, "C", xc="pbe", core_hole=CoreHole("1s"), formats=formats)
            names = sorted(p.name for p in Path(tmp).iterdir())
            report = json.loads(result.artifacts["report"].read_text())
            self.assertNotIn(b"\r", result.artifacts["dat"].read_bytes())
            if read_parsec_pseudopotential is not None:
                potential = read_parsec_pseudopotential(result.artifacts["parsec_species"])
                self.assertEqual(potential.symbol, "C")
                self.assertEqual(potential.ionic_charge, 5.0)
                self.assertEqual(sorted(potential.channel_potentials), [0, 1, 2])
        for name in ("C-1s_POTRE.DAT", "C-1s_FHIPP.DAT", "c_1s_hole_1_pbe_tm.UPF",
                     "c_1s_hole_1_pbe_tm.fhi", "c_1s_hole_1_pbe_tm.test",
                     "c_1s_hole_1_pbe_tm.report.json"):
            self.assertIn(name, names)
        self.assertEqual(report["core_hole"], {"shell": "1s", "removed_electrons": 1.0})
        self.assertEqual(report["ionic_charge"], 5.0)
        self.assertAlmostEqual(report["atomic_energy_correction_rydberg"], -35.46519, places=4)
        self.assertEqual(report["selected_local_channel"], 2)
        self.assertTrue(report["ghost_free"])


if __name__ == "__main__":
    unittest.main()

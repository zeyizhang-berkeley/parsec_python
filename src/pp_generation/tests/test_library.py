"""The library builder: layout, manifests, and what happens when a member fails."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pp_generation.generator as generator
from pp_generation.errors import BackendError, GhostStateError
from pp_generation.library import MANIFEST_COLUMNS, build_family, build_library


class FamilyTests(unittest.TestCase):
    def test_carbon_family_layout_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = build_library(root, ["C"], ["pbe"], progress=lambda text: None)
            ground = root / "NCPPs" / "PBE" / "C"
            hole = root / "CoreHole_NCPPs" / "PBE" / "C" / "C-1s"
            self.assertTrue((ground / "PARSEC" / "C_POTRE.DAT").is_file())
            self.assertTrue((ground / "FHI" / "C_FHIPP.DAT").is_file())
            self.assertTrue((hole / "PARSEC" / "C-1s_POTRE.DAT").is_file())
            self.assertFalse((hole / "PARSEC" / "C_POTRE.DAT").exists())
            self.assertEqual(len(list((hole / "UPF").glob("*.UPF"))), 1)
            snippet = (hole / "PARSEC" / "parsec_atom_type.in").read_text()
            self.assertIn("Atom_Type: C-1s", snippet)
            self.assertIn("Element_Symbol: C", snippet)
            self.assertIn("Local_Component: d", snippet)
            report = json.loads(next((hole / "diagnostics").glob("*.report.json")).read_text())
            for item in report["artifacts"].values():       # every recorded file exists
                self.assertTrue((hole / "diagnostics" / item["path"]).is_file(), item["path"])
            with open(root / "CoreHole_NCPPs" / "manifest.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([x.state for x in entries], ["ground", "1s"])
        self.assertEqual(tuple(rows[0]), MANIFEST_COLUMNS)
        row = rows[0]
        self.assertEqual((row["element"], row["state"], row["status"]), ("C", "1s", "ok"))
        self.assertEqual(row["matched_to_ground_state"], "True")
        self.assertEqual(row["local_channel"], "d")
        self.assertEqual(float(row["ionic_charge"]), 5.0)
        self.assertAlmostEqual(float(row["atomic_energy_correction_Ry"]), -35.46519, places=4)
        # same radii as the ground state
        self.assertEqual(entries[0].radii, entries[1].radii)

    def test_a_hole_without_ghost_free_form_does_not_discard_the_family(self) -> None:
        real = generator.generate

        def fake(request):
            if request.core_hole is not None and request.core_hole.shell == "2p":
                raise GhostStateError("no ghost-free Kleinman-Bylander form")
            return real(request)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(generator, "generate", fake):
            entries = build_family("Si", "pbe", Path(tmp))
            kept = (Path(tmp) / "NCPPs" / "PBE" / "Si" / "PARSEC" / "Si_POTRE.DAT").is_file()
            gone = not (Path(tmp) / "CoreHole_NCPPs" / "PBE" / "Si" / "Si-2p").exists()
        status = {x.state: x.status for x in entries}
        self.assertEqual(status, {"ground": "ok", "1s": "ok", "2s": "ok", "2p": "failed"})
        self.assertTrue(kept and gone)
        self.assertIn("no ghost-free form", next(x.note for x in entries if x.state == "2p"))

    def test_a_hole_that_cannot_be_constructed_moves_a_radius_for_the_whole_family(self) -> None:
        real = generator.generate
        calls = []

        def fake(request):
            # the 1s hole "has no solution" at the template radii, as K and Y have
            if (request.core_hole is not None and request.core_hole.shell == "1s"
                    and not request.channel_radii):
                error = BackendError("FHI98PP stopped: tromar - stop: convergence error cc0 (norm)")
                error.channel = 0
                raise error
            calls.append(request)
            return real(request)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(generator, "generate", fake):
            entries = build_family("C", "pbe", Path(tmp))
        self.assertEqual({x.state: x.status for x in entries}, {"ground": "ok", "1s": "ok"})
        ground, hole = entries
        self.assertEqual(ground.radii, hole.radii)                    # still matched
        self.assertGreater(ground.radii[0], 1.4982)                   # s moved up one mesh point
        self.assertAlmostEqual(ground.radii[1], 1.4982, places=4)     # p untouched
        self.assertIn("s radius moved +1 mesh point", ground.note)
        self.assertTrue(hole.matched_to_ground_state)


if __name__ == "__main__":
    unittest.main()

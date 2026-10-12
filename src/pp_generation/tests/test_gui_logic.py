"""The parts of the graphical front end that do not need a display."""

import unittest
from pathlib import Path

from pp_generation.gui import PERIODIC_ROWS, FormState, build_request, read_cpi_channels
from pp_generation.models import BackendName, ConstructionScheme, OutputFormat
from pp_generation.templates import SYMBOLS, available_elements, element_core_shells


class FormTests(unittest.TestCase):
    def test_ground_state_request(self) -> None:
        request = build_request(FormState(element="Si", xc="ca", output_dir="out"))
        self.assertEqual(request.element, "Si")
        self.assertEqual(request.xc, "ca")
        self.assertIsNone(request.core_hole)
        self.assertIs(request.backend, BackendName.PYFHI98PP)
        self.assertIs(request.scheme, ConstructionScheme.TROULLIER_MARTINS)
        self.assertEqual(request.output_dir, Path("out") / "ca" / "Si")
        self.assertIsNone(request.local_channel)
        self.assertIsNone(request.lmax)
        self.assertIsNone(request.channel_radii)

    def test_core_hole_request(self) -> None:
        state = FormState(
            element="Si", core_hole_shell="2p", hole_charge="0.5", lmax="2", local_channel="d",
            radii={0: "2.5", 1: "", 2: "2.6"}, core_radius="0",
            formats=(OutputFormat.PARSEC,),
        )
        request = build_request(state)
        self.assertEqual(request.core_hole.shell, "2p")
        self.assertEqual(request.core_hole.removed_electrons, 0.5)
        self.assertEqual(request.lmax, 2)
        self.assertEqual(request.local_channel, 2)
        self.assertEqual(request.channel_radii, {0: 2.5, 2: 2.6})
        self.assertEqual(request.core_radius, 0.0)
        self.assertEqual(request.output_dir.name, "Si-2p")

    def test_bad_entries_are_reported(self) -> None:
        with self.assertRaises(ValueError):
            build_request(FormState(radii={0: "two"}))
        with self.assertRaises(ValueError):
            build_request(FormState(formats=()))
        with self.assertRaises(ValueError):
            build_request(FormState(core_hole_shell="1s", hole_charge="x"))


class TableTests(unittest.TestCase):
    def test_every_template_element_has_a_button(self) -> None:
        shown = {symbol for row in PERIODIC_ROWS for symbol in row.split() if symbol != "."}
        self.assertEqual(shown, set(SYMBOLS))
        self.assertTrue(set(available_elements()) <= shown)
        self.assertTrue(all(len(row.split()) == 18 for row in PERIODIC_ROWS))

    def test_core_shells_of_templates(self) -> None:
        self.assertEqual(element_core_shells("H"), [])
        self.assertEqual(element_core_shells("Si"), ["1s", "2s", "2p"])


class CpiReaderTests(unittest.TestCase):
    def test_channels(self) -> None:
        rows = ["4.0  2"] + ["0 0 0 0"] * 10
        for _ in range(2):
            rows.append("3 1.0247")
            rows += [f"{i + 1} {0.1 * (i + 1)} {0.5 * (i + 1)} {-1.0 * (i + 1)}" for i in range(3)]
        path = Path(self.id().split(".")[-1] + ".cpi")
        try:
            path.write_text("\n".join(rows) + "\n")
            channels = read_cpi_channels(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(len(channels), 2)
        radius, wave, potential = channels[1]
        self.assertEqual(radius, [0.1, 0.2, 0.30000000000000004])
        self.assertEqual(wave, [0.5, 1.0, 1.5])
        self.assertEqual(potential, [-1.0, -2.0, -3.0])


if __name__ == "__main__":
    unittest.main()

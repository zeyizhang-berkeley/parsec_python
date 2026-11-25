"""A hybrid functional under the launch settings of ``scripts/run_multi_gpu_scf.sh`` (``hybrid_launch_filter``).

The script spreads the filter of a sector over the GPUs of its rank by default (``PARSEC_CUPY_DISTRIBUTED_FILTER=1``)
and records the value it was given (``PARSEC_LAUNCH_DISTRIBUTED_FILTER``, empty where its default stands).  The
full-SCF runner runs a hybrid with 0 where the 1 is the script's own, with a NOTE; a 1 given to the script or set
without it stays (the driver refuses it with a hybrid), and no other functional is touched.  CPU only; the lines of
the script are run by bash where it is there.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from parsec_python.acceleration.benchmarks.mpi_full_scf import LAUNCH_DISTRIBUTED_FILTER, hybrid_launch_filter

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "run_multi_gpu_scf.sh"


class HybridLaunchFilterTests(unittest.TestCase):
    def test_the_default_of_the_script_gives_way_for_a_hybrid(self) -> None:
        environment = {"PARSEC_CUPY_DISTRIBUTED_FILTER": "1", LAUNCH_DISTRIBUTED_FILTER: ""}
        note = hybrid_launch_filter(True, environment)
        self.assertEqual(environment["PARSEC_CUPY_DISTRIBUTED_FILTER"], "0")
        self.assertIn("PARSEC_CUPY_DISTRIBUTED_FILTER=0", note)

    def test_everything_else_is_left_alone(self) -> None:
        for hybrid, environment in (
            (False, {"PARSEC_CUPY_DISTRIBUTED_FILTER": "1", LAUNCH_DISTRIBUTED_FILTER: ""}),  # not a hybrid
            (True, {"PARSEC_CUPY_DISTRIBUTED_FILTER": "1", LAUNCH_DISTRIBUTED_FILTER: "1"}),  # given to the script
            (True, {"PARSEC_CUPY_DISTRIBUTED_FILTER": "1"}),  # set without the script
            (True, {"PARSEC_CUPY_DISTRIBUTED_FILTER": "0", LAUNCH_DISTRIBUTED_FILTER: "0"}),
            (True, {LAUNCH_DISTRIBUTED_FILTER: ""}),
        ):
            before = dict(environment)
            self.assertIsNone(hybrid_launch_filter(hybrid, environment), (hybrid, before))
            self.assertEqual(environment, before)

    def test_the_script_records_what_it_was_given(self) -> None:
        lines = _SCRIPT.read_text().splitlines()
        exports = [line for line in lines if re.match(r'^export [A-Z0-9_]+="\$\{[A-Z0-9_]+:-[^}]*\}"$', line)]
        names = [line.split("=", 1)[0].split()[1] for line in exports]
        # the record first, in the form the launch settings are read in (one export of a default per line)
        self.assertLess(names.index(LAUNCH_DISTRIBUTED_FILTER), names.index("PARSEC_CUPY_DISTRIBUTED_FILTER"))
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        program = "\n".join(exports) + '\nprintf "%s|%s" "$PARSEC_LAUNCH_DISTRIBUTED_FILTER" "$PARSEC_CUPY_DISTRIBUTED_FILTER"'
        for given, expected in ((None, "|1"), ("1", "1|1"), ("0", "0|0")):
            environment = {key: value for key, value in os.environ.items()
                           if key not in ("PARSEC_CUPY_DISTRIBUTED_FILTER", LAUNCH_DISTRIBUTED_FILTER)}
            if given is not None:
                environment["PARSEC_CUPY_DISTRIBUTED_FILTER"] = given
            result = subprocess.run([bash, "-c", program], env=environment, capture_output=True, text=True,
                                    timeout=60)
            self.assertEqual(result.stdout.strip(), expected, (given, result.stderr))


if __name__ == "__main__":
    unittest.main()

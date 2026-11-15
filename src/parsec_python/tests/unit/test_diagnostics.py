"""Debug output must preserve science and keep large-array work opt-in."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python import diagnostics as debug


class DiagnosticsTests(unittest.TestCase):
    def test_off_does_not_serialize_or_create_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'PARSEC_DEBUG': 'off', 'PARSEC_DEBUG_DIR': directory}), redirect_stdout(StringIO()) as out:
            debug.emit('unused', array=np.ones((100, 100)))
            self.assertEqual(out.getvalue(), '')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rank_selection_strict_json_and_file_agree(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'PARSEC_DEBUG': 'summary', 'PARSEC_DEBUG_DIR': directory,
                'PARSEC_DEBUG_RANKS': 'leaders', 'PMI_RANK': '16'}), redirect_stdout(StringIO()) as out:
            debug.emit('not_a_leader')
            debug.emit('sector_owner', leader=True, missing=float('nan'), scalar=np.float64(2.))
            line = out.getvalue().strip().removeprefix('PARSEC_DEBUG ')
            record = json.loads(line)
            self.assertEqual(record['rank'], 16)
            self.assertIsNone(record['missing'])
            self.assertEqual(record['scalar'], 2.)
            self.assertEqual(json.loads((Path(directory) / 'rank-00016.jsonl').read_text()), record)

    def test_real_scf_is_bitwise_unchanged_with_audit_output(self):
        from parsec_python.tests.unit.test_strict_scf_eigensolver import StrictSCFEigensolverTests
        from parsec_python.SCF import prepare_single_point, run_scf

        results = []
        for mode in ['off', 'audit']:
            with patch.dict(os.environ, {'PARSEC_DEBUG': mode, 'PARSEC_DEBUG_DIR': '',
                                        'PARSEC_DEBUG_RANKS': 'leaders', 'PMI_RANK': '0'}), redirect_stdout(StringIO()) as out:
                result = run_scf(prepare_single_point(StrictSCFEigensolverTests()._problem()))
                results.append(result)
                lines = [json.loads(x[len('PARSEC_DEBUG '):]) for x in out.getvalue().splitlines()
                         if x.startswith('PARSEC_DEBUG ')]
                self.assertEqual(bool(lines), mode == 'audit')
                if mode == 'audit':
                    fields = [x for x in lines if x['event'] == 'scf_fields']
                    self.assertEqual(len(fields), result.iterations)
                    self.assertAlmostEqual(fields[-1]['density']['integral'], 1., delta=1e-9)
        for name in ['density', 'eigenvalues', 'next_effective_potential']:
            np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))
        self.assertEqual(results[0].energies.total, results[1].energies.total)


if __name__ == '__main__':
    unittest.main()

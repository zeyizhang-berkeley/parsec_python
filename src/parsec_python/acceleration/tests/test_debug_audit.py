"""Numerical residual audits against independent dense real/complex references."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.Eigensolvers.debug_audit import column_audit, shared_ritz_audit
from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout


class DebugAuditTests(unittest.TestCase):
    def test_complex_ritz_audit_preserves_vectors_values_and_residual_policy(self):
        from parsec_python.acceleration.Eigensolvers.chebff_columns import (
            ColumnGroup, ColumnLayout as ComplexLayout, _orthonormal_ritz, _Stats)
        rng = np.random.default_rng(12)
        a = rng.normal(size=(79, 79)) + 1j * rng.normal(size=(79, 79))
        a = (a + a.conj().T) / 2
        initial = rng.normal(size=(79, 13)) + 1j * rng.normal(size=(79, 13))
        operator = SimpleNamespace(apply=lambda block: a @ block)
        results = []
        for mode in ('off', 'audit'):
            group = ColumnGroup(SimpleNamespace(rank=0, size=1,
                Reduce=lambda send, receive, root=0: np.copyto(receive, send)), np)
            with patch.dict(os.environ, PARSEC_DEBUG=mode, PARSEC_DEBUG_DIR='',
                            PARSEC_DEBUG_BUFFER_BYTES='1024'), redirect_stdout(StringIO()) as log:
                result = _orthonormal_ritz(np, operator, [initial.copy()], group,
                    ComplexLayout(13, 6, 1), 79, 11, False, 6, _Stats(None))
            results.append(result)
            self.assertIsNone(result[2])
            if mode == 'audit':
                records = [json.loads(line.removeprefix('PARSEC_DEBUG '))
                           for line in log.getvalue().splitlines()]
                report = next(x for x in records if x['event'] == 'complex_eigenpair_audit')
                expected = np.linalg.norm(a @ result[0][:, :11] -
                                          result[0][:, :11] * result[1][:11], axis=0)
                self.assertAlmostEqual(report['maximum_residual_ry'], expected.max(), places=12)
                self.assertFalse(report['residuals_used_by_solver'])
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_bounded_tiles_match_dense_residuals_without_changing_basis(self):
        rng = np.random.default_rng(4)
        for complex_ in [False, True]:
            a = rng.normal(size=(37, 37))
            q = rng.normal(size=(37, 11))
            if complex_:
                a = a + 1j * rng.normal(size=a.shape)
                q = q + 1j * rng.normal(size=q.shape)
            a = (a + a.conj().T) / 2
            q, _ = np.linalg.qr(q)
            q = np.asfortranarray(q)
            original = q.copy()
            eigenvalues = np.linspace(-2, 2, 19)
            indices = np.array([0, 2, 3, 5, 6, 7, 9, 12, 14, 17, 18])
            widths = []

            def apply(x, out):
                widths.append(x.shape[1])
                out[...] = a @ x

            report = column_audit(SimpleNamespace(apply_into=apply), q, indices, eigenvalues, np, 1024)
            expected = np.linalg.norm(a @ q - q * eigenvalues[indices], axis=0)
            self.assertAlmostEqual(report['maximum'], expected.max(), places=12)
            self.assertAlmostEqual(report['squared_sum'], expected @ expected, places=10)
            self.assertLess(report['norm_squared_error'], 2e-15)
            self.assertTrue(report['finite'])
            self.assertLessEqual(max(widths), max(1, 1024 // (3 * 37 * q.dtype.itemsize)))
            np.testing.assert_array_equal(q, original)

    def test_disabled_audit_never_touches_basis_or_backend(self):
        with patch.dict(os.environ, PARSEC_DEBUG='off'):
            self.assertIsNone(shared_ritz_audit(None, None, None, phase='disabled'))

    def test_disjoint_interleaved_columns_cover_every_state_once(self):
        rng = np.random.default_rng(9)
        q, _ = np.linalg.qr(rng.normal(size=(51, 9)))
        eigenvalues = np.arange(9.)
        layout = ColumnLayout((((0, 2), (6, 8)), ((2, 6), (8, 9))))
        blocks = []
        for rank in range(2):
            columns = [j for start, _, count in layout.spans(rank) for j in range(start, start + count)]
            blocks.append(q[:, columns])
        operator = SimpleNamespace(apply_into=lambda x, out: np.copyto(out, np.arange(51.)[:, None] * x))
        group = SimpleNamespace(map=lambda f: [f(i, i) for i in range(2)], _operator=lambda d: operator)
        basis = SimpleNamespace(blocks=blocks, layout=layout, shape=q.shape)
        with patch.dict(os.environ, PARSEC_DEBUG='audit', PARSEC_DEBUG_DIR=''), \
                patch('parsec_python.acceleration.backends.cupy.require_cupy', return_value=(np, None)), \
                redirect_stdout(StringIO()):
            report = shared_ritz_audit(group, basis, eigenvalues, phase='test')
        self.assertEqual(report['checked_states'], 9)
        expected = np.linalg.norm(np.arange(51.)[:, None] * q - q * eigenvalues, axis=0)
        self.assertAlmostEqual(report['maximum_residual_ry'], expected.max(), places=12)


if __name__ == '__main__':
    unittest.main()

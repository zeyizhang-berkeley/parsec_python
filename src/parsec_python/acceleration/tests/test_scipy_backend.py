from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.V_ion import NonlocalProjectorOperator

from parsec_python.acceleration.backends.scipy import ScipyHamiltonianBackend


class ScipyHamiltonianBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        diagonal = np.linspace(2.0, 3.0, 9)
        off_diagonal = np.full(8, -0.25)
        self.kinetic = sp.diags(
            (off_diagonal, diagonal, off_diagonal),
            offsets=(-1, 0, 1),
            format="csr",
        )
        projectors = sp.csc_matrix(
            np.column_stack(
                (
                    np.linspace(-0.3, 0.4, 9),
                    np.linspace(0.5, -0.2, 9),
                )
            )
        )
        self.nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.asarray((1.0, -1.0)),
            labels=((0, 0, 0), (0, 1, 0)),
        )
        self.local = np.linspace(-1.0, 0.2, 9)
        self.reference = KohnShamHamiltonian(
            self.kinetic,
            self.local,
            self.nonlocal_operator,
        )
        self.backend = ScipyHamiltonianBackend(
            self.kinetic,
            self.nonlocal_operator,
        )
        self.accelerated = self.backend.bind(self.local)

    def test_vector_and_block_actions_match_reference(self) -> None:
        rng = np.random.default_rng(14)
        for vectors in (rng.normal(size=9), rng.normal(size=(9, 4))):
            np.testing.assert_allclose(
                self.accelerated.apply(vectors),
                self.reference.apply(vectors),
                rtol=2.0e-15,
                atol=2.0e-15,
            )

    def test_each_component_and_sparse_materialization_match(self) -> None:
        vectors = np.arange(27, dtype=float).reshape(9, 3) / 19.0
        np.testing.assert_allclose(
            self.accelerated.apply_kinetic(vectors),
            self.reference.apply_kinetic(vectors),
        )
        np.testing.assert_allclose(
            self.accelerated.apply_local(vectors),
            self.reference.apply_local(vectors),
        )
        np.testing.assert_allclose(
            self.accelerated.apply_nonlocal(vectors),
            self.reference.apply_nonlocal(vectors),
        )
        np.testing.assert_allclose(
            self.accelerated.as_sparse().toarray(),
            self.reference.as_sparse().toarray(),
        )

    def test_statistics_and_explicit_component_profile(self) -> None:
        vectors = np.ones((9, 2))
        self.accelerated.apply(vectors)
        timings = self.backend.profile_components(vectors)
        self.assertEqual(self.backend.statistics.applications, 1)
        self.assertEqual(self.backend.statistics.vectors_applied, 2)
        self.assertGreaterEqual(self.backend.statistics.apply_seconds, 0.0)
        self.assertEqual(
            set(timings),
            {
                "finite_difference_apply",
                "local_potential_apply",
                "nonlocal_potential_apply",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))


if __name__ == "__main__":
    unittest.main()

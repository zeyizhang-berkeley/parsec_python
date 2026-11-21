"""Units of the irreducible representations of the point group in complete GPU calculations (PARSEC_ORBITAL_IRREPS).

A Td Ge5 cluster (D2d sectors: A1 of Td with E projected out, A2 and E of B1, T1, T2), the Si8 cube with its point
group about the origin (Td) and with its translations (Fd-3m modulo the cube, 192 operations, units of ``once`` and
``split``): the SCF energies, the occupied levels and the forces of the units equal those of the weighted sectors of
before (``off``), also on a coarse grid with a high first-filter degree, whose rounding takes the states off the
range of the projectors.  The projector of a unit applied on the device equals the host tables; states that round
off the range, and orthonormalized bases with columns drawn for dependent ones, are projected back; a first solve
that the units cannot take (CHEBDAV) is refused before the SCF.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom


def _available() -> bool:
    return cupy_available() and native_available()


def _ge5(directory: Path, *, radius: float = 9.0, states: int = 16, spacing: float | None = None, **eigensolver):
    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.force_reference import load_cases, write_case

    problem = parse_parsec_input(write_case(load_cases()["ge2h2_nlcc_tiny"], directory)).problem
    # moved a little outward along the bonds: Td stays and the atoms have forces
    bond = 5.0 / np.sqrt(3.0)
    atoms = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
        Atom("Ge", (bond * x, bond * y, bond * z)) for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    )
    grid = replace(problem.grid, radius=radius)
    if spacing is not None:
        grid = replace(grid, spacing=spacing)
    return replace(
        problem,
        atoms=atoms,
        recenter_geometry=False,
        grid=grid,
        scf=replace(problem.scf, number_of_states=states, convergence_criterion=1.0e-8, max_iterations=100),
        eigensolver=replace(problem.eigensolver, method="chebff", tolerance=1.0e-10, **eigensolver),
    )


def _pair_projection(rows: int):
    """The orthogonal projector onto vectors with equal entries at 2i and 2i + 1 (a unit of half the rows)."""

    from parsec_python.acceleration.Eigensolvers.irrep_projection import IrrepProjection

    partner = np.arange(rows) ^ 1
    return IrrepProjection(np.stack([np.arange(rows), partner]), np.full((2, rows), 0.5), representation=3), partner


def _filter_calls():
    """Patches that count the filter calls of every operator (the filtered blocks a unit must project)."""

    from collections import Counter

    from parsec_python.acceleration.Eigensolvers import chebyshev

    calls = Counter()
    chebff_filter, subspace_filter = chebyshev._chebff_filter, chebyshev._subspace_filter

    def counted_chebff(operator, *args, **kwargs):
        calls[id(operator)] += 1
        return chebff_filter(operator, *args, **kwargs)

    def counted_subspace(operator, *args, **kwargs):
        calls[id(operator)] += 1
        return subspace_filter(operator, *args, **kwargs)

    patches = (
        mock.patch.object(chebyshev, "_chebff_filter", counted_chebff),
        mock.patch.object(chebyshev, "_subspace_filter", counted_subspace),
    )
    return calls, patches


def _run(problem, environment, *, forces=True, backend=None):
    from parsec_python.acceleration import driver

    options = {} if backend is None else {"backend": backend}
    with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = driver.prepare_single_point(problem, **options)
        system.materialize_final_wavefunctions = False
        result = driver.run_scf(system)
        computed = driver.compute_forces(system, result, symmetrize=False) if forces else None
    sectors = getattr(system.backend, "symmetry_eigensolver", None)
    return result, computed, sectors, dict(result.backend.details)


@unittest.skipUnless(_available(), "CUDA runtime or native extension is unavailable")
class DeviceProjectionTests(unittest.TestCase):
    def test_the_device_projection_is_that_of_the_tables(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.irrep_projection import IrrepProjection, check_unit_basis

        rng = np.random.default_rng(3)
        rows, terms = 1000, 3
        indices = rng.integers(0, rows, size=(terms, rows))
        indices[0] = np.arange(rows)
        scales = rng.standard_normal((terms, rows))
        projection = IrrepProjection(indices, scales, representation=7)
        block = rng.standard_normal((rows, 37))
        expected = projection.apply_host(block)
        for order in ("C", "F"):
            device = cp.asarray(block, order=order)
            out = projection.apply(device)
            self.assertIs(out, device)
            np.testing.assert_allclose(cp.asnumpy(device), expected, rtol=0, atol=1e-12, err_msg=order)
        # a column view and one vector
        device = cp.asarray(block, order="F")
        projection.apply(device[:, 5:9])
        np.testing.assert_allclose(cp.asnumpy(device[:, 5:9]), expected[:, 5:9], atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(device[:, :5]), block[:, :5], atol=0)
        vector = cp.asarray(block[:, 0])
        projection.apply(vector)
        np.testing.assert_allclose(cp.asnumpy(vector), expected[:, 0], atol=1e-12)
        # chunks of the temporary
        with mock.patch("parsec_python.acceleration.Eigensolvers.irrep_projection._CHUNK_BYTES", 8 * rows * 4):
            device = cp.asarray(block, order="C")
            projection.apply(device)
            np.testing.assert_allclose(cp.asnumpy(device), expected, atol=1e-12)
        # the check of a basis: an orthogonal projector's range passes, anything else is refused
        orthogonal = np.linalg.qr(rng.standard_normal((rows, 4)))[0]
        exact = IrrepProjection(np.tile(np.arange(rows), (1, 1)), np.ones((1, rows)))
        check_unit_basis(exact, cp.asarray(orthogonal), 4)
        half = IrrepProjection(np.tile(np.arange(rows), (1, 1)), 0.5 * np.ones((1, rows)))
        with self.assertRaises(RuntimeError):
            check_unit_basis(half, cp.asarray(orthogonal), 4)
        # only the projections of filtered blocks count as applications
        self.assertEqual((projection.applications, exact.applications, half.applications), (5, 0, 0))

    def test_states_that_round_off_the_range_are_projected_back(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.irrep_projection import (
            UNIT_BASIS_TOLERANCE,
            check_unit_basis,
            unit_basis_share,
        )

        rows, count = 1000, 6
        projection, partner = _pair_projection(rows)
        rng = np.random.default_rng(5)
        inside = rng.standard_normal((rows, count))
        inside = np.linalg.qr(0.5 * (inside + inside[partner]))[0]
        outside = rng.standard_normal((rows, count))
        outside = 0.5 * (outside - outside[partner])
        outside /= np.linalg.norm(outside, axis=0)
        for scale in (0.0, 1.0e-9):
            # within the tolerance: nothing changes
            vectors = cp.asarray(inside + scale * outside, order="F")
            share = check_unit_basis(projection, vectors, count)
            self.assertLessEqual(share, UNIT_BASIS_TOLERANCE)
            np.testing.assert_array_equal(cp.asnumpy(vectors), inside + scale * outside)
        self.assertEqual(projection.repairs, 0)
        # the rounding of an ill-conditioned block (shares of 2e-7 to 9e-6 seen on A100): projected back in place
        # and orthonormalized, close to the states inside
        for scale, order in ((3.0e-6, "F"), (2.0e-4, "C")):
            vectors = cp.asarray(inside + scale * outside, order=order)
            share = check_unit_basis(projection, vectors, count)
            self.assertGreater(share, UNIT_BASIS_TOLERANCE)
            repaired = cp.asnumpy(vectors)
            np.testing.assert_allclose(repaired[partner], repaired, rtol=0, atol=1e-14)
            np.testing.assert_allclose(repaired.T @ repaired, np.eye(count), rtol=0, atol=1e-13)
            np.testing.assert_allclose(repaired, inside, rtol=0, atol=10 * scale * scale + 1e-14)
            self.assertLessEqual(unit_basis_share(projection, vectors), 1e-13)
        self.assertEqual((projection.repairs, projection.applications), (2, 0))
        self.assertGreater(projection.largest_share, 1e-4)
        # beyond the limit (a column that no projection reached): refused
        vectors = cp.asarray(inside + 0.05 * outside)
        with self.assertRaisesRegex(RuntimeError, "left the range"):
            check_unit_basis(projection, vectors, count)

    def test_orthonormalized_bases_of_a_unit_stay_in_its_range(self) -> None:
        import cupy as cp
        from types import SimpleNamespace

        from parsec_python.acceleration.Eigensolvers.irrep_projection import orthonormal_in_unit, unit_basis_share
        from parsec_python.acceleration.Eigensolvers.orthogonalize import orthonormalize_complete_subspace

        rows, count = 1000, 8
        projection, partner = _pair_projection(rows)
        unit = SimpleNamespace(irrep_projection=projection)
        rng = np.random.default_rng(9)
        block = rng.standard_normal((rows, count))
        block = 0.5 * (block + block[partner])
        # a dependent column, which selective Gram-Schmidt and a Householder QR both complete by its rounding, and a
        # zero one, for which Gram-Schmidt draws a random column: vectors of the whole sector
        block[:, 3] = block[:, 1] - 2.0 * block[:, 0]
        block[:, 6] = 0.0
        for policy in ("mgs", "qr"):
            with self.subTest(policy=policy), mock.patch.dict(
                os.environ, {"PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION": policy}
            ):
                result = orthonormalize_complete_subspace(cp.asarray(block), rng=np.random.default_rng(1))
                if policy == "mgs":
                    self.assertGreaterEqual(result.random_replacements, 1)
                self.assertGreater(unit_basis_share(projection, result.basis), 1e-3)
                repairs = projection.repairs
                basis = cp.asnumpy(orthonormal_in_unit(unit, result, np.random.default_rng(2)))
                self.assertGreater(projection.repairs, repairs)
                np.testing.assert_allclose(basis[partner], basis, rtol=0, atol=1e-13)
                np.testing.assert_allclose(basis.T @ basis, np.eye(count), rtol=0, atol=1e-12)
                # the span of the independent columns is kept
                independent = np.linalg.qr(block[:, [0, 1, 2, 4, 5, 7]])[0]
                np.testing.assert_allclose(
                    np.linalg.svd(independent.T @ basis, compute_uv=False)[:6], np.ones(6), rtol=0, atol=1e-10
                )
        # a basis in the range, and an operator without a unit, are returned as they are
        clean = orthonormalize_complete_subspace(cp.asarray(np.linalg.qr(block[:, :3])[0]), rng=np.random.default_rng(1))
        repairs = projection.repairs
        self.assertIs(orthonormal_in_unit(unit, clean, None), clean.basis)
        self.assertEqual(projection.repairs, repairs)
        drawn = orthonormalize_complete_subspace(cp.asarray(block), rng=np.random.default_rng(1))
        self.assertIs(orthonormal_in_unit(SimpleNamespace(), drawn, None), drawn.basis)
        self.assertEqual(projection.applications, 0)


@unittest.skipUnless(_available(), "CUDA runtime or native extension is unavailable")
class UnitSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.ge5 = _ge5(Path(directory.name))

    def assert_same(self, left, right, label, *, energy=2.0e-8, level=1.0e-7, force=2.0e-6) -> None:
        (result, forces, _, _), (reference, reference_forces, _, _) = left, right
        self.assertTrue(result.converged and reference.converged, label)
        # The sum of the levels follows the potential of the last iteration, which two converged SCF paths reach to
        # their criterion only: it is compared to the tolerance of the levels.
        for field, delta in (("total", energy), ("exchange_correlation", energy), ("hartree", energy),
                             ("eigenvalue", level)):
            self.assertAlmostEqual(
                getattr(result.energies, field), getattr(reference.energies, field), delta=delta, msg=f"{label} {field}"
            )
        occupied = int(np.count_nonzero(np.asarray(reference.occupations) > 1.0e-6))
        np.testing.assert_allclose(
            np.sort(result.eigenvalues)[:occupied], np.sort(reference.eigenvalues)[:occupied], rtol=0, atol=level,
            err_msg=label,
        )
        if forces is not None and reference_forces is not None:
            np.testing.assert_allclose(forces.total, reference_forces.total, rtol=0, atol=force, err_msg=label)

    def test_a_td_cluster(self) -> None:
        before = _run(self.ge5, {"PARSEC_ORBITAL_IRREPS": "off"})
        self.assertGreater(float(np.abs(before[1].total).max()), 1.0e-3)
        reduction = before[2].decomposition.reduction
        self.assertEqual(sorted(int(w) for w in reduction.sector_weights), [1, 1, 3, 3])
        self.assertIsNone(reduction.sector_projected)
        for mode in ("once", "split"):
            with self.subTest(mode=mode):
                calls, patches = _filter_calls()
                with patches[0], patches[1]:
                    # the rows of the sectors of H (the units on partner rows project nothing: their own tests)
                    units = _run(self.ge5, {"PARSEC_ORBITAL_IRREPS": mode, "PARSEC_ORBITAL_IRREP_ROWS": "sector"})
                decomposition = units[2].decomposition
                self.assertEqual(sorted(int(w) for w in decomposition.sector_weights), [1, 1, 2, 3, 3])
                self.assertEqual(len(decomposition.projected_sectors), 3)
                self.assertIn("5 units of the irreducible representations", units[3]["symmetry_detection"])
                # every projected unit projected each block that a filter returned, and nothing else counts
                for index in decomposition.projected_sectors:
                    operator = units[2]._operators[index]
                    self.assertGreater(calls[id(operator)], 1)
                    self.assertEqual(operator.irrep_projection.applications, calls[id(operator)])
                self.assert_same(units, before, f"Ge5 {mode}")

    def test_a_coarse_grid_with_a_high_first_filter_degree(self) -> None:
        # Spacing 0.7 bohr, first-filter degree 90 in four cycles (review of 2026-11-20): the orthonormalizations
        # and Ritz steps of the units leave up to 9e-6 of their states outside the range of the projectors, which
        # stopped this run before; they are projected back.
        from parsec_python.acceleration.Eigensolvers.irrep_projection import UNIT_BASIS_TOLERANCE

        once = {"PARSEC_ORBITAL_IRREPS": "once", "PARSEC_ORBITAL_IRREP_ROWS": "sector"}
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(radius=9.5, states=24, spacing=0.7, first_filter_cycles=4)
            problem = _ge5(Path(directory), first_filter_degree=90, **settings)
            before = _run(problem, {"PARSEC_ORBITAL_IRREPS": "off"})
            units = _run(problem, once)
            # degree 30 leaves nothing outside the range (1e-15)
            reference = _run(_ge5(Path(directory), first_filter_degree=30, **settings), once)

        def projections(run):
            return [run[2]._operators[index].irrep_projection for index in run[2].decomposition.projected_sectors]

        self.assertGreater(max(projection.largest_share for projection in projections(units)), UNIT_BASIS_TOLERANCE)
        self.assertGreater(sum(projection.repairs for projection in projections(units)), 0)
        self.assertEqual(sum(projection.repairs for projection in projections(reference)), 0)
        # The total energy, the occupied levels and the forces of off (A100: 9e-12 Ry, 4e-9 Ry, 2e-10 Ry/bohr).
        # The energy terms of off and of the units differ by up to 5e-8 Ry on this grid at degree 30 as well (the
        # Hartree energy, which the SCF criterion leaves free to first order); those of the reference units agree.
        self.assertTrue(units[0].converged and before[0].converged)
        self.assertAlmostEqual(units[0].energies.total, before[0].energies.total, delta=1.0e-9)
        occupied = int(np.count_nonzero(np.asarray(before[0].occupations) > 1.0e-6))
        np.testing.assert_allclose(np.sort(units[0].eigenvalues)[:occupied], np.sort(before[0].eigenvalues)[:occupied],
                                   rtol=0, atol=5.0e-8)
        np.testing.assert_allclose(units[1].total, before[1].total, rtol=0, atol=1.0e-8)
        self.assert_same(units, reference, "Ge5 coarse once, degree 90 against 30")

    def test_chebdav_is_refused_before_the_scf(self) -> None:
        from parsec_python.acceleration import driver

        problem = replace(self.ge5, eigensolver=replace(self.ge5.eigensolver, method="chebdav"))
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_IRREPS": "once"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "Eigensolver: chebff"):
                driver.prepare_single_point(problem)

    def test_silicon_cells(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = silicon("pbe")
        problem = replace(problem, eigensolver=replace(problem.eigensolver, method="chebff"))
        for translations, modes in (("off", ("once",)), ("auto", ("once", "split"))):
            environment = {"PARSEC_PERIODIC_TRANSLATIONS": translations}
            before = _run(problem, dict(environment, PARSEC_ORBITAL_IRREPS="off"), forces=False, backend="cupy")
            for mode in modes:
                with self.subTest(translations=translations, mode=mode):
                    units = _run(problem, dict(environment, PARSEC_ORBITAL_IRREPS=mode), forces=False, backend="cupy")
                    # The fcc translations with Td about the atom (96 operations, sectors of D2d x fcc): every
                    # representation of the group fills a sector of its own once, and split projects them.
                    projected = units[2].decomposition.projected_sectors
                    if (translations, mode) == ("auto", "once"):
                        self.assertEqual(projected, (), f"{translations} {mode}")
                    else:
                        self.assertTrue(projected, f"{translations} {mode}")
                    self.assert_same(units, before, f"Si8 {translations} {mode}")


if __name__ == "__main__":
    unittest.main()

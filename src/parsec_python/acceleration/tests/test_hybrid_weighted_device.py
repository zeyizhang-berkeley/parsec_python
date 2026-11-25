"""The exact exchange of the weighted sectors of a point group (``PARSEC_ORBITAL_GROUP_HYBRID``).

A model Hamiltonian (Laplacian and Gaussian wells) of a structure of a non-abelian point group is solved in the
weighted sectors of a subgroup ``H`` (:class:`parsec_python.acceleration.Symmetry.PointGroupReduction`).  The
exchange operator of their states, the coset images of the states that hold one partner of a level and the solved
states of the others (``Hamiltonian/exchange_device.WeightedExchange``), must be that of the full-grid states of the
same levels to round-off: ``W`` of every solved state, ``E_x`` and the ACE, for Td over D2d (grid shifts 1/2 and 0),
Oh over D4h and N2's D4h over a subgroup of 8, the Coulomb and the screened kernel, whole and smeared occupations,
through the parity classes and the full grid; the abelian sectors of the same structure give the same ``E_x`` with
more pair solves.  The host tests check the forms of the sectors and their density matrix without a device.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available


def _structures():
    from parsec_python.models import Atom

    a = 1.2
    td = [Atom("C", (0.0, 0.0, 0.0))] + [
        Atom("H", tuple(a * np.array(signs, dtype=float)))
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    ]
    oh = [Atom("S", (0.0, 0.0, 0.0))] + [Atom("F", tuple(1.5 * v)) for v in np.vstack([np.eye(3), -np.eye(3)])]
    n2 = [Atom("N", (0.0, 0.0, 1.0)), Atom("N", (0.0, 0.0, -1.0))]
    return {
        "Td": (td, (3.0, 1.0, 1.0, 1.0, 1.0)),
        "Oh": (oh, (3.0,) + (1.0,) * 6),
        "N2": (n2, (2.0, 2.0)),
    }


def _model(name, shift, *, spacing=0.5, radius=4.0):
    import scipy.sparse as sp

    from parsec_python.Grid import build_cluster_grid
    from parsec_python.Laplacian import build_negative_laplacian
    from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction
    from parsec_python.models import GridSettings

    atoms, depths = _structures()[name]
    grid = build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=6, shift=(shift,) * 3))
    points = np.asarray(grid.physical_coordinates(grid.integer_coordinates))
    potential = np.zeros(points.shape[0])
    for atom, depth in zip(atoms, depths):
        potential -= depth * np.exp(-np.sum((points - np.asarray(atom.position)) ** 2, axis=1) / 0.8)
    hamiltonian = (0.5 * build_negative_laplacian(grid) + sp.diags(potential)).tocsr()
    abelian = SignedPermutationReduction.detect(grid, atoms)
    weighted = detect_point_group_reduction(grid, atoms, abelian)
    return grid, hamiltonian, abelian, weighted


def _cut(values, start):
    """An energy in a gap of ``values`` (sorted) above the ``start``-th: the states below it are whole levels."""

    gaps = np.flatnonzero(np.diff(values) > 1.0e-6)
    above = gaps[gaps + 1 >= start][0]
    return 0.5 * (values[above] + values[above + 1])


def _sector_states(grid, hamiltonian, reduction, top):
    """``(decomposition, [(representation, values, vectors)])``: the states of every sector up to the energy ``top``."""

    from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition

    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
    found = []
    for representation, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
        values, vectors = np.linalg.eigh(matrix.toarray())
        keep = np.flatnonzero(values < top)
        if keep.size:
            found.append((representation, values[keep], vectors[:, keep]))
    return decomposition, found


def _expansion(decomposition, representation):
    from parsec_python.acceleration.Hamiltonian.exchange_device import SectorExpansion

    orbits = decomposition.sector_orbit_indices(representation)
    scales = 1.0 / np.sqrt(decomposition.reduction.multiplicities[orbits])
    return SectorExpansion(
        decomposition.reduction.full_to_wedge, decomposition.phases[representation], orbits, scales,
        decomposition.wedge_size,
    )


def _occupation(values, cut, smearing):
    values = np.asarray(values, dtype=np.float64)
    if smearing is None:
        return (values < cut).astype(np.float64)
    return 1.0 / (1.0 + np.exp((values - cut) / smearing))


class ExchangeFormTests(unittest.TestCase):
    """The forms of the sectors and the density matrix they give (host)."""

    def test_forms_of_the_sectors(self) -> None:
        from parsec_python.acceleration.Symmetry import PointGroupReduction
        from parsec_python.acceleration.Symmetry.exchange_partners import exchange_partners

        expected = {
            "Td": ([1, 1, 3, 3], 3),
            "Oh": ([1, 1, 3, 3, 1, 1, 3, 3], 3),
            "N2": ([1, 1, 2, 1, 1, 2], 2),
        }
        for name, (weights, cosets) in expected.items():
            with self.subTest(name=name):
                grid, _hamiltonian, _abelian, reduction = _model(name, 0.5)
                self.assertIsInstance(reduction, PointGroupReduction)
                partners = exchange_partners(reduction)
                self.assertEqual(partners.weights.astype(int).tolist(), weights)
                self.assertEqual(partners.images.tolist(), [w > 1 for w in weights])
                self.assertEqual(partners.cosets, cosets)
                # a row map of every coset representative other than the identity, a permutation of the grid
                for coset in range(1, cosets):
                    mapping = partners.row_map(coset)
                    self.assertEqual(sorted(mapping.tolist()), list(range(grid.size)))
                with self.assertRaises(ValueError):
                    partners.row_map(0)

    def test_density_matrix_of_the_weighted_states(self) -> None:
        from parsec_python.acceleration.Symmetry.exchange_partners import exchange_partners

        for name, shift in (("Td", 0.5), ("Td", 0.0), ("N2", 0.5)):
            with self.subTest(name=name, shift=shift):
                grid, hamiltonian, _abelian, reduction = _model(name, shift, radius=3.5)
                values, vectors = np.linalg.eigh(hamiltonian.toarray())
                cut = _cut(values, 12)
                full = vectors[:, values < cut] @ vectors[:, values < cut].T
                partners = exchange_partners(reduction)
                maps = [np.arange(grid.size)] + [partners.row_map(c) for c in range(1, partners.cosets)]
                decomposition, found = _sector_states(grid, hamiltonian, reduction, cut)
                mixed = np.zeros_like(full)
                images = np.zeros_like(full)
                for representation, _values, sector_vectors in found:
                    weight = partners.weights[representation]
                    for column in range(sector_vectors.shape[1]):
                        state = decomposition.expand_vector(sector_vectors[:, column], representation)
                        term = sum(np.outer(state[m], state[m]) for m in maps) * (weight / partners.cosets)
                        images += term
                        mixed += term if partners.images[representation] else weight * np.outer(state, state)
                self.assertLess(float(np.abs(mixed - full).max()), 1e-11)
                self.assertLess(float(np.abs(images - full).max()), 1e-11)

    def test_the_switch(self) -> None:
        from parsec_python.acceleration.Hamiltonian.exchange_device import weighted_hybrid_enabled
        from parsec_python.acceleration.Symmetry.weighted_inputs import weighted_input_settings

        with patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_HYBRID": "auto"}):
            self.assertTrue(weighted_hybrid_enabled())
        with patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_HYBRID": "off"}):
            self.assertFalse(weighted_hybrid_enabled())
        with patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_HYBRID": "on"}):
            with self.assertRaises(ValueError):
                weighted_input_settings()


class WeightedGuardTests(unittest.TestCase):
    """The partly filled degenerate levels of an unpolarized result of weighted sectors (host)."""

    @staticmethod
    def _result(values, occupations, sectors, repeats, spin=None):
        from types import SimpleNamespace

        repeats = np.asarray(repeats)
        listed_values = np.repeat(np.asarray(values, dtype=np.float64), repeats)
        listed = np.repeat(np.asarray(occupations, dtype=np.float64), repeats)
        solved = np.asarray(occupations, dtype=np.float64) * repeats
        return SimpleNamespace(
            sector_states=(np.asarray(sectors), solved, repeats), eigenvalues=listed_values, occupations=listed, spin=spin
        )

    def test_partly_filled_partner_levels(self) -> None:
        from parsec_python.acceleration.SCF.hybrid import PARTLY_FILLED_LEVEL_REASON, partly_filled_partner_levels

        # a1 filled and a t level of weight 3: closed shell, CH4+ (5 electrons in the t level), its tail of 1e-3
        self.assertIsNone(partly_filled_partner_levels(self._result([-1.0, -0.5], [1.0, 1.0], [1, 3], [1, 3])))
        report = partly_filled_partner_levels(self._result([-1.0, -0.5], [1.0, 5.0 / 6.0], [1, 3], [1, 3]))
        self.assertEqual(report.reason, PARTLY_FILLED_LEVEL_REASON)
        self.assertAlmostEqual(report.largest, 1.0 / 6.0, places=12)
        self.assertIn("5.0000 electrons", report.lines[0])
        self.assertIsNone(partly_filled_partner_levels(self._result([-1.0, -0.5], [1.0, 0.999], [1, 3], [1, 3])))
        # an e level of two sectors of weight 1, half filled; a lone level of one sector half filled does not count
        self.assertIsNotNone(
            partly_filled_partner_levels(self._result([-1.0, -0.4, -0.4], [1.0, 0.5, 0.5], [1, 1, 2], [1, 1, 1]))
        )
        self.assertIsNone(partly_filled_partner_levels(self._result([-1.0, -0.4], [1.0, 0.5], [1, 2], [1, 1])))
        # two channels (the guard of the spin sectors) and a result without weighted sectors
        self.assertIsNone(
            partly_filled_partner_levels(self._result([-1.0, -0.5], [1.0, 0.5], [1, 3], [1, 3], spin=(1, 2)))
        )
        from types import SimpleNamespace

        self.assertIsNone(partly_filled_partner_levels(SimpleNamespace(eigenvalues=[0.0], occupations=[0.5])))

    def test_the_engine_raises_and_the_note(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.acceleration.SCF.hybrid import DeviceFockEngine, WeightedHybridAbandoned
        from parsec_python.models import SpinSectorFallbackRecord

        engine = object.__new__(DeviceFockEngine)
        engine.fixed_occupations = False
        partly = self._result([-1.0, -0.5], [1.0, 5.0 / 6.0], [1, 3], [1, 3])
        engine.weighted_guard(self._result([-1.0, -0.5], [1.0, 1.0], [1, 3], [1, 3]), "the start")
        with self.assertRaises(WeightedHybridAbandoned) as caught:
            engine.weighted_guard(partly, "the semilocal start of the Fock loop")
        self.assertEqual(caught.exception.stage, "the semilocal start of the Fock loop")
        # occup.in: its equal occupations of the partners are the input's
        engine.fixed_occupations = True
        engine.weighted_guard(partly, "the start")
        report = caught.exception.report
        lines = spin_sector_fallback_lines(
            SpinSectorFallbackRecord(
                stage=caught.exception.stage, largest_fractional_occupation=report.largest,
                tolerance=report.tolerance, multiplets=report.lines, reason=report.reason, measure=report.measure,
                route="weighted_hybrid",
            )
        )
        text = "\n".join(lines)
        self.assertIn("NOTE: weighted symmetry sectors: a partly filled degenerate level", text)
        self.assertIn("PARSEC_ORBITAL_GROUP_HYBRID=off", text)
        self.assertIn(report.lines[0], text)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class WeightedExchangeDeviceTests(unittest.TestCase):
    """``W``, ``E_x`` and the ACE of weighted sector orbitals against the full grid and the abelian sectors."""

    def _channels(self, cp, name, shift, *, smearing=None, empty=4, devices=(0,), spin_weight=2.0):
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            ChannelOrbitals,
            OrbitalSector,
            WeightedExchange,
        )

        grid, hamiltonian, abelian, reduction = _model(name, shift)
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        cut = _cut(values, 10)
        # every state the occupations reach (the tail of the smearing) and a few empty ones, whole levels
        reached = int(np.flatnonzero(_occupation(values, cut, smearing) > 1.0e-12)[-1]) + 1
        top = _cut(values, max(reached, int(np.searchsorted(values, cut))) + empty)

        def sector_channel(reduction, weighted):
            decomposition, found = _sector_states(grid, hamiltonian, reduction, top)
            weighting = WeightedExchange(reduction) if weighted else None
            sectors, occupations, start = [], [], 0
            for index, (representation, sector_values, sector_vectors) in enumerate(found):
                count = sector_values.size
                device = int(devices[index % len(devices)])
                with cp.cuda.Device(device):
                    basis = cp.ascontiguousarray(cp.asarray(sector_vectors))
                sectors.append(
                    OrbitalSector(
                        representation, device, basis, np.arange(count), np.arange(start, start + count),
                        _expansion(decomposition, representation),
                    )
                )
                weight = 1.0 if weighting is None else weighting.sector_weight(representation)
                occupations.append(_occupation(sector_values, cut, smearing) * weight)
                start += count
            occupations = np.concatenate(occupations)
            if weighting is None:
                return ChannelOrbitals(tuple(sectors), occupations, spin_weight, int(grid.size),
                                       float(grid.volume_element)), decomposition, found
            return (
                weighting.channel(tuple(sectors), occupations, spin_weight, int(grid.size), float(grid.volume_element)),
                decomposition,
                found,
            )

        weighted = sector_channel(reduction, True)
        plain = sector_channel(abelian, False)
        # the full-grid reference: the occupied full-grid states and, as targets, the weighted states on the full grid
        occupied = np.flatnonzero(_occupation(values, cut, smearing) > 1.0e-12)
        targets = []
        decomposition, found = weighted[1], weighted[2]
        for representation, _values, sector_vectors in found:
            for column in range(sector_vectors.shape[1]):
                targets.append(decomposition.expand_vector(sector_vectors[:, column], representation))
        columns = np.column_stack([vectors[:, occupied]] + [np.column_stack(targets)])
        full_occupations = np.concatenate([_occupation(values[occupied], cut, smearing), np.zeros(len(targets))])
        with cp.cuda.Device(int(devices[0])):
            block = cp.asfortranarray(cp.asarray(columns))
        full = ChannelOrbitals(
            (OrbitalSector(None, int(devices[0]), block, np.arange(columns.shape[1]), np.arange(columns.shape[1]), None),),
            full_occupations, spin_weight, int(grid.size), float(grid.volume_element),
        )
        return grid, weighted[0], plain[0], full, occupied.size

    def check(self, name, shift, terms, *, smearing=None, devices=(0,), tolerance=1e-11, spin_weight=2.0):
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
            compress_device_exchange,
            parity_context,
        )

        grid, weighted, plain, full, occupied = self._channels(
            cp, name, shift, smearing=smearing, devices=devices, spin_weight=spin_weight
        )
        self.assertTrue(weighted.images)
        solvers = DevicePairSolvers(grid, terms, 6, 4)
        reference = apply_device_exchange(full, solvers, (int(devices[0]),), columns="all")
        with cp.cuda.Device(int(devices[0])):
            reference_w = cp.asnumpy(reference.applied[0])[:, occupied:]
        scale = float(np.abs(reference_w).max())
        abelian = apply_device_exchange(plain, solvers, devices, columns="all")
        results = {}
        for setting in ("auto", "off"):
            with patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_PARITY": setting}):
                if setting == "auto":
                    self.assertIsNotNone(parity_context(weighted, solvers))
                first = apply_device_exchange(weighted, solvers, devices, columns="occupied")
                application = apply_device_exchange(weighted, solvers, devices, columns="all", known=first)
            column = 0
            for index, sector in enumerate(weighted.sectors):
                with cp.cuda.Device(sector.device):
                    unfolded = cp.asnumpy(sector.expansion.unfold(cp, cp.asfortranarray(application.applied[index])))
                for local in range(sector.local.size):
                    error = float(np.abs(unfolded[:, local] - reference_w[:, column]).max())
                    self.assertLess(error, tolerance * scale, (name, shift, setting, index, local))
                    column += 1
            self.assertAlmostEqual(application.energy, reference.energy, delta=tolerance * abs(reference.energy))
            self.assertAlmostEqual(application.energy, abelian.energy, delta=tolerance * abs(reference.energy))
            results[setting] = application
        # fewer pair solves than the abelian sectors, the same on both paths
        self.assertEqual(results["auto"].pair_solves, results["off"].pair_solves)
        self.assertLess(results["auto"].pair_solves, abelian.pair_solves)
        # the ACE of every weighted sector gives E_x of its own orbitals back, to the asymmetry of M = -Psi^T W (the
        # discretized kernel is symmetric only to that), and the band energy of the ACE of the abelian sectors
        operator = compress_device_exchange(results["auto"], fraction=0.25)
        band = operator.band_energy(weighted, scaled=False)
        bound = max(1.0e-10, 10.0 * operator.asymmetry) * abs(operator.energy)
        self.assertAlmostEqual(band, 2.0 * operator.energy, delta=bound)
        abelian_operator = compress_device_exchange(abelian, fraction=0.25)
        self.assertAlmostEqual(
            abelian_operator.band_energy(plain, scaled=False), band,
            delta=bound + 10.0 * abelian_operator.asymmetry * abs(operator.energy),
        )
        return results["auto"], abelian_operator

    def test_td_over_d2d(self) -> None:
        coulomb = (("coulomb", 1.0, None),)
        for shift in (0.5, 0.0):
            with self.subTest(shift=shift):
                self.check("Td", shift, coulomb)

    def test_oh_and_d4h(self) -> None:
        coulomb = (("coulomb", 1.0, None),)
        for name in ("Oh", "N2"):
            with self.subTest(name=name):
                self.check(name, 0.5, coulomb)

    def test_screened_kernel_smeared_occupations_and_one_channel(self) -> None:
        self.check("Td", 0.5, (("erfc", 1.0, 0.11),), smearing=0.05, tolerance=1e-10)
        self.check("N2", 0.5, (("coulomb", 0.19, None), ("erf", 0.46, 0.33)), smearing=0.05, tolerance=1e-10,
                   spin_weight=1.0)

    def test_every_device(self) -> None:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 2:
            self.skipTest("one device")
        self.check("Td", 0.5, (("coulomb", 1.0, None),), devices=tuple(range(count)))

    def test_natural_channel_of_the_window_keeps_the_weights(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.SCF.fock_device import (
            WeightedOccupiedBlock,
            natural_channels,
            occupied_blocks,
        )

        _grid, weighted, _plain, _full, _occupied = self._channels(cp, "Td", 0.5)
        blocks = occupied_blocks([weighted])[0]
        for sector in weighted.sectors:
            block = blocks.get(sector.representation)
            if block is None:
                continue
            self.assertIsInstance(block, WeightedOccupiedBlock)
            self.assertEqual(block.weight, weighted.weighting.sector_weight(sector.representation))
            self.assertLessEqual(float(np.max(block.occupations)), 1.0)
        natural = [{key: (block.vectors, block.occupations) for key, block in blocks.items()}]
        (channel,) = natural_channels([weighted], natural)
        self.assertIs(channel.weighting, weighted.weighting)
        np.testing.assert_allclose(channel.partner_occupations()[channel.occupied_states()], 1.0)
        self.assertEqual(len(channel.images), len(weighted.images))
        self.assertAlmostEqual(float(channel.occupations.sum()), float(weighted.occupations.sum()), places=12)


if __name__ == "__main__":
    unittest.main()

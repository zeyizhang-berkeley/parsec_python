"""The exchange of a periodic cell's sector orbitals in the classes of their half-cell translations
(``acceleration/Hamiltonian/exchange_translations.py``, ``PARSEC_CUPY_EXCHANGE_TRANSLATIONS``).

On the CPU: the switch, the class solve of the reduced box against the FFT of the whole box (every subgroup ``T`` of
the half translations, every character), the translations and characters found on the sectors of diamond, body-centred
and simple cells, and the coset form of a sector function and its fold.  On a GPU: ``W`` and ``E_x`` of random sector
orbitals against the pooled full-grid block (``off``), one build and the occupied build completed, on one device and
on several: one, two and four classes per pair, boxes halved along every axis and along some; coset blocks that do
not fit the device.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Hamiltonian import exchange_translations as translations
from parsec_python.acceleration.Hamiltonian.exchange_device import ChannelOrbitals, OrbitalSector, SectorExpansion
from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction, periodic_abelian_reduction
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model
from parsec_python.acceleration.tests.test_periodic_symmetry_detection import _diamond


def _even_kernel(shape, seed=5) -> np.ndarray:
    """A random real kernel on the box with ``K(-G) = K(G)``."""

    values = np.random.default_rng(seed).uniform(0.5, 2.0, size=shape)
    flipped = values
    for axis in range(3):
        flipped = np.roll(np.flip(flipped, axis=axis), 1, axis=axis)
    return values + flipped


def _repeated(base, repeats, side=6.0):
    """The atoms of ``base`` (fractions of a cubic cell of ``side`` bohr) in a supercell of ``repeats``."""

    from parsec_python.models import Atom

    cells = np.array(list(product(*(range(int(count)) for count in repeats))), dtype=float)
    sides = side * np.asarray(repeats, dtype=float)
    positions = np.mod((cells[:, None, :] + np.asarray(base, dtype=float)[None]).reshape(-1, 3) * side, sides)
    return [Atom("C", tuple(position)) for position in positions], sides


def _bcc(repeats):
    """A body-centred cubic cell: the body centre is its one half-cell translation (four classes per pair)."""

    return _repeated(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)), repeats)


def _simple(repeats):
    """A simple tetragonal cell of a pair of atoms along z, no centring: only the repeats give half translations."""

    return _repeated(((0.0, 0.0, 0.0), (0.0, 0.0, 0.3)), repeats)


def _cell(repeats, points, environment=None, kind="diamond"):
    """A cell (diamond by default, ``bcc`` or ``simple``), its periodic grid and the decomposition of its abelian
    group with the half-cell translations."""

    atoms, sides = {"diamond": _diamond, "bcc": _bcc, "simple": _simple}[kind](repeats)
    grid, _ = _periodic_model(sides, points, 0.0, atoms, {"C": 2.0})
    with mock.patch.dict(os.environ, environment or {}):
        reduction = periodic_abelian_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
    return grid, reduction, ReflectionRepresentationDecomposition.build(grid, reduction)


def _expansion(decomposition, representation) -> SectorExpansion:
    orbits = decomposition.sector_orbit_indices(representation)
    scales = 1.0 / np.sqrt(decomposition.reduction.multiplicities[orbits])
    return SectorExpansion(
        decomposition.reduction.full_to_wedge, decomposition.phases[representation], orbits, scales,
        decomposition.wedge_size,
    )


def _host_channel(grid, decomposition) -> ChannelOrbitals:
    """Every sector with one (absent) column: what the context reads (the expansions)."""

    sectors = tuple(
        OrbitalSector(representation, 0, None, np.arange(1), np.array([representation]),
                      _expansion(decomposition, representation))
        for representation in range(decomposition.representation_count)
    )
    return ChannelOrbitals(sectors, np.ones(len(sectors)), 2.0, int(grid.size), float(grid.volume_element))


def _solver(shape, batch_size=32):
    shape = tuple(int(value) for value in shape)
    return SimpleNamespace(
        box=SimpleNamespace(identity=True, fft_shape=shape, box_shape=shape),
        solve_translation_classes=lambda *args: None,
        batch_size=batch_size,
    )


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {translations.SETTING: value}):
                self.assertEqual(translations.exchange_translation_setting(), expected)
        for value in ("on", "0", "classes"):
            with mock.patch.dict(os.environ, {translations.SETTING: value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_EXCHANGE_TRANSLATIONS must be auto or off"):
                    translations.exchange_translation_setting()

    def test_a_wrong_value_is_refused_at_setup(self) -> None:
        # device_exchange_for reads it beside the switches of the parity classes and of the pooled block
        from pathlib import Path
        import shutil
        import tempfile

        from parsec_python.acceleration.SCF import hybrid as device_hybrid
        from parsec_python.acceleration.backends import cupy as cupy_backend
        from parsec_python.acceleration.tests.test_gamma_hybrid_route import _translation, _write_silicon

        folder = Path(tempfile.mkdtemp(prefix="parsec_translation_switch_"))
        self.addCleanup(shutil.rmtree, folder, True)
        problem = _translation(_write_silicon(folder)).problem
        reference = SimpleNamespace(input=problem, grid=SimpleNamespace(volume_element=0.125))
        sectors = SimpleNamespace(device_ids=(0,))
        with mock.patch.object(cupy_backend, "require_cupy", return_value=(None, None)):
            with mock.patch.dict(os.environ, {translations.SETTING: "on"}):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_EXCHANGE_TRANSLATIONS must be auto or off"):
                    device_hybrid.device_exchange_for(reference, None, sectors, selected_backend="cupy")
            for value in ("", "auto", "off"):
                with mock.patch.dict(os.environ, {translations.SETTING: value}):
                    exchange = device_hybrid.device_exchange_for(reference, None, sectors, selected_backend="cupy")
                    self.assertEqual(exchange.devices, (0,))


class DescriptionTests(unittest.TestCase):
    def test_the_solver_names_the_translation_classes(self) -> None:
        from parsec_python.acceleration.tests.test_gamma_hybrid_route import _fallback_note, _periodic_solvers

        solvers = _periodic_solvers()
        plain = solvers.description
        solvers.note_translations("7 half-cell translations, the 2x2x2 box of each class")
        note = (
            "; the pair densities of the symmetry-sector orbitals in the classes of their 7 half-cell translations, "
            "the 2x2x2 box of each class"
        )
        unfit = "; the coset blocks of the half-cell translations of the sector orbitals did not fit the device in "
        self.assertEqual(solvers.describe(Counter({"translations": 3})), plain + note)
        # the fallback to the tiles stays beside the tiles it names
        self.assertEqual(
            solvers.describe(Counter({"translations": 2, "pool-fallback": 1, "translations-unfit": 1})),
            plain + _fallback_note(1, 3) + note + " (in 2 of 3 builds)" + unfit + "1 of 3 builds",
        )
        # coset blocks that did not fit, the builds on the pooled block or on the tiles
        self.assertEqual(
            solvers.describe(Counter({"pool": 2, "translations-unfit": 2})), plain + unfit + "2 of 2 builds"
        )
        self.assertEqual(
            solvers.describe(Counter({"pool-fallback": 2, "translations-unfit": 2})),
            plain + _fallback_note(2, 2) + unfit + "2 of 2 builds",
        )
        # without such builds the text of before
        self.assertEqual(solvers.describe(Counter({"pool": 2, "full": 2})), plain)
        self.assertEqual(solvers.describe(Counter({"pool": 1, "pool-fallback": 1})), plain + _fallback_note(1, 2))

    def test_the_label_counts_the_translations_without_the_identity(self) -> None:
        layout = translations.translation_layout((8, 8, 8), 0b111)
        for elements, expected in (
            ((0, 1, 2, 3, 4, 5, 6, 7), "7 half-cell translations, the 4x4x4 box of each class"),
            ((0, 3, 5, 6), "3 half-cell translations, the 4x4x4 box of each class"),
            ((0, 7), "half-cell translation, the 4x4x4 box of each class"),
        ):
            group = translations._coset_group(elements, 0b111)
            context = translations.TranslationContext(layout, group, (), (), 32)
            self.assertEqual(context.order, len(elements))
            self.assertEqual(context.label, expected)


class ClassSolveTests(unittest.TestCase):
    """The class solve on the reduced box is the FFT solve of the whole box, on the coset form."""

    def _check(self, shape, elements) -> None:
        halved = 0
        for mask in elements:
            halved |= mask
        layout = translations.translation_layout(shape, halved)
        group = translations._coset_group(tuple(elements), halved)
        self.assertEqual(len(group.cosets) * len(elements), 1 << len(translations._axes(halved)))
        context = translations.TranslationContext(layout, group, (), (), 32)
        kernel = _even_kernel(shape)
        points = layout.images[list(group.cosets)].reshape(-1)
        # S is a fundamental domain of T: its translates cover the grid once
        covered = np.concatenate([translations._shift(shape, mask)[points] for mask in elements])
        np.testing.assert_array_equal(np.sort(covered), np.arange(int(np.prod(shape))))
        rng = np.random.default_rng(len(elements) * 100 + halved)
        characters = set()
        for signs in product((1, -1), repeat=3):
            character = {}
            for mask in elements:
                character[mask] = int(np.prod([signs[d] for d in translations._axes(mask)])) if mask else 1
            characters.add(tuple(sorted(character.items())))
        for character in map(dict, characters):
            noise = rng.standard_normal((int(np.prod(shape)), 2))
            densities = sum(character[mask] * noise[translations._shift(shape, mask)] for mask in elements)
            densities /= len(elements)
            full = np.stack(
                [np.real(np.fft.ifftn(kernel * np.fft.fftn(densities[:, b].reshape(shape)))).reshape(-1)
                 for b in range(2)],
                axis=1,
            )
            classes = context.classes(character)
            self.assertEqual(len(classes), len(group.cosets))
            found = translations.reference_class_potentials(densities[points], layout, group, classes, kernel)
            scale = float(np.abs(full).max())
            np.testing.assert_allclose(found, full[points], rtol=0.0, atol=1.0e-12 * scale)

    def test_every_subgroup_of_the_half_translations(self) -> None:
        for shape, elements in (
            ((8, 6, 10), (0, 1, 2, 3, 4, 5, 6, 7)),  # a supercell of even repeats: one class, eighth box
            ((8, 6, 10), (0, 3, 5, 6)),  # the face-centring translations of a cube: two classes
            ((8, 6, 10), (0, 7)),  # the body centre: four classes
            ((8, 7, 5), (0, 4)),  # one axis halved (the others odd)
            ((8, 6, 5), (0, 2, 4, 6)),  # two axes halved
        ):
            with self.subTest(shape=shape, elements=elements):
                self._check(shape, elements)

    def test_the_twist_and_the_kernel_of_a_class(self) -> None:
        layout = translations.translation_layout((8, 6, 5), 0b110)
        self.assertEqual(layout.reduced_shape, (4, 3, 5))
        self.assertEqual(layout.label, "4x3x5")
        np.testing.assert_array_equal(layout.images[0][:6], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(layout.images[0b100][:2], [4 * 30, 4 * 30 + 1])
        self.assertTrue(np.all(layout.images[0b001] == -1))
        self.assertEqual(layout.kernel_slices(0b100), (slice(1, None, 2), slice(0, None, 2), slice(None)))
        twist = layout.twist(0b110).reshape(4, 3, 5)
        self.assertAlmostEqual(complex(twist[1, 2, 3]), complex(np.exp(-2j * np.pi * (1 / 8 + 2 / 6))))
        with self.assertRaisesRegex(ValueError, "even number of points"):
            translations.translation_layout((8, 6, 5), 0b001)


class SectorTranslationTests(unittest.TestCase):
    """The translations and characters found on the sectors of cells, and the coset form of their functions."""

    def test_the_translations_of_cells(self) -> None:
        for kind, repeats, points, expected, cosets in (
            ("diamond", (2, 1, 1), 16, (0, 3, 4, 7), 2),  # along x and the face centring in y, z
            ("diamond", (2, 2, 2), 16, tuple(range(8)), 1),  # every half of the cell
            ("diamond", (1, 1, 1), 8, (0, 3, 5, 6), 2),  # the face centring of the cube
            ("diamond", (2, 2, 3), 16, (0, 2, 4, 6), 1),  # x and y halved, z not (three cubes)
            ("bcc", (1, 1, 1), 16, (0, 7), 4),  # the body centre: four cosets on the eighth box
            ("simple", (2, 1, 1), 16, (0, 4), 1),  # x alone
            ("simple", (2, 2, 1), 12, (0, 2, 4, 6), 1),  # x and y
        ):
            with self.subTest(kind=kind, repeats=repeats):
                grid, reduction, decomposition = _cell(repeats, points, kind=kind)
                shape = tuple(int(value) for value in grid.points_per_axis)
                # the pure translations of the group (identity matrix), as masks
                pure = {
                    sum(1 << (2 - d) for d in range(3) if int(step[d]) == shape[d] // 2)
                    for operation, step in zip(reduction.operations, reduction.translations)
                    if np.array_equal(np.asarray(operation), np.eye(3))
                }
                channel = _host_channel(grid, decomposition)
                elements, characters = translations._translation_group(channel, shape)
                self.assertEqual(elements, expected)
                self.assertEqual(set(elements), pure)
                context = translations.translation_context(channel, _solver(shape))
                self.assertIsNotNone(context)
                self.assertEqual(context.coset_rows * len(elements), grid.size)
                self.assertEqual(len(context.group.cosets), cosets)
                rng = np.random.default_rng(1)
                seen = set()
                for index, sector in enumerate(channel.sectors):
                    expansion = sector.expansion
                    coefficients = rng.standard_normal(expansion.rows)
                    function = expansion.point_weights * coefficients[expansion.point_rows]
                    for mask in elements:
                        shifted = function[translations._shift(shape, mask)]
                        np.testing.assert_allclose(shifted, characters[index][mask] * function, atol=1.0e-14)
                    seen.add(tuple(characters[index][mask] for mask in elements))
                    # the coset form and the fold back into the sector rows
                    coset = context.sectors[index]
                    points_of_s = context.layout.images[list(context.group.cosets)].reshape(-1)
                    gathered = coefficients[coset.rows] * coset.weights
                    np.testing.assert_allclose(gathered, function[points_of_s], atol=1.0e-14)
                    folded = gathered[coset.fold_entries] * coset.fold_weights
                    np.testing.assert_allclose(folded, coefficients, rtol=1.0e-13, atol=1.0e-14)
                # every character of T is met
                self.assertEqual(len(seen), len(elements))

    def test_where_the_pooled_block_stays(self) -> None:
        grid, _reduction, decomposition = _cell((2, 1, 1), 16)
        shape = tuple(int(value) for value in grid.points_per_axis)
        channel = _host_channel(grid, decomposition)
        self.assertIsNotNone(translations.translation_context(channel, _solver(shape)))
        with mock.patch.dict(os.environ, {translations.SETTING: "off"}):
            self.assertIsNone(translations.translation_context(channel, _solver(shape)))
        # a solver without class solves (a cluster's Kronecker solver), a padded box (a slab), orbitals of one sector
        self.assertIsNone(translations.translation_context(channel, SimpleNamespace(batch_size=32)))
        padded = _solver(shape)
        padded.box.fft_shape = (shape[0], shape[1], 2 * shape[2])
        self.assertIsNone(translations.translation_context(channel, padded))
        one = ChannelOrbitals(channel.sectors[:1], np.ones(1), 2.0, channel.full_size, channel.volume_element)
        self.assertIsNone(translations.translation_context(one, _solver(shape)))
        # the commuting involutions about the origin alone: no translation in the sectors
        grid, _reduction, decomposition = _cell((2, 1, 1), 16, {"PARSEC_PERIODIC_ABELIAN_TRANSLATIONS": "off"})
        self.assertIsNone(translations.translation_context(_host_channel(grid, decomposition), _solver(shape)))


def _device_channel(grid, decomposition, *, states_per_sector, occupied_fraction, devices, seed=3):
    """Random smooth sector orbitals of every sector on ``devices`` (round robin), some occupied."""

    import cupy as cp

    rng = np.random.default_rng(seed)
    coordinates = np.asarray(grid.coordinates)
    centre = coordinates.mean(axis=0)
    envelope = 1.0 + 0.5 * np.cos(np.sum(coordinates - centre, axis=1))
    sectors, occupations, state = [], [], 0
    for representation in range(decomposition.representation_count):
        expansion = _expansion(decomposition, representation)
        columns = states_per_sector[representation % len(states_per_sector)]
        if not columns:
            continue
        full = rng.standard_normal((grid.size, columns)) * envelope[:, None]
        host = np.column_stack(
            [decomposition.reduce_vector(full[:, column], representation) for column in range(columns)]
        )
        host /= np.linalg.norm(host, axis=0)[None, :]
        taken = int(round(occupied_fraction * columns))
        occupations += list(np.linspace(1.0, 0.7, taken)) + [0.0] * (columns - taken)
        device = int(devices[representation % len(devices)])
        with cp.cuda.Device(device):
            vectors = cp.ascontiguousarray(cp.asarray(host))
        sectors.append(OrbitalSector(representation, device, vectors, np.arange(columns),
                                     np.arange(state, state + columns), expansion))
        state += columns
    return ChannelOrbitals(tuple(sectors), np.asarray(occupations), 2.0, int(grid.size), float(grid.volume_element))


def _device_solver(grid, batch_size, *, screened=False):
    from parsec_python.Hartree.exchange_kernel import PeriodicExchangeKernel
    from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
        DevicePeriodicBox,
        DevicePeriodicPairSolvers,
    )

    shape = tuple(int(value) for value in grid.points_per_axis)
    lattice = np.diag(np.asarray(grid.step_vectors, dtype=float).diagonal() * np.asarray(shape, dtype=float))
    terms = dict(alpha=0.0, beta=1.0, omega=0.106) if screened else dict(alpha=1.0, beta=0.0, omega=None)
    kernel = PeriodicExchangeKernel(lattice, shape, (1, 1, 1), periodic_axes=(0, 1, 2), **terms)
    return DevicePeriodicPairSolvers(kernel, DevicePeriodicBox(shape, shape), batch_size)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceTranslationTests(unittest.TestCase):
    """``W`` and ``E_x`` of the translation classes against the pooled full-grid block on a GPU."""

    def _builds(self, orbitals, solvers, devices, setting):
        from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange

        import cupy as cp

        with mock.patch.dict(os.environ, {translations.SETTING: setting, "PARSEC_CUPY_EXCHANGE_POOL": "auto"}):
            whole = apply_device_exchange(orbitals, solvers, devices, columns="all")
            occupied = apply_device_exchange(orbitals, solvers, devices, columns="occupied")
            kept = occupied.pooled
            completed = apply_device_exchange(orbitals, solvers, devices, columns="all", known=occupied)
        host = lambda application: [cp.asnumpy(part) for part in application.applied]  # noqa: E731
        return whole, host(whole), kept, completed, host(completed), occupied

    def _compare(self, repeats, points, *, devices=(0,), screened=False, batch_size=4, states=(5, 3, 0, 4),
                 kind="diamond"):
        grid, _reduction, decomposition = _cell(repeats, points, kind=kind)
        orbitals = _device_channel(grid, decomposition, states_per_sector=states, occupied_fraction=0.6,
                                   devices=devices)
        solvers = _device_solver(grid, batch_size, screened=screened)
        solvers.take_paths()
        whole, reference, kept_off, _completed, completed_off, _ = self._builds(orbitals, solvers, devices, "off")
        self.assertEqual(solvers.take_paths(), Counter({"pool": 3, "full": 3}))
        found, values, kept, completed, completed_values, occupied = self._builds(orbitals, solvers, devices, "auto")
        self.assertEqual(solvers.take_paths(), Counter({"translations": 3}))
        self.assertIsInstance(kept.orbitals, translations.TranslationChannel)
        # consumed by the completion: nothing of the occupied build is left
        self.assertIsNone(occupied.pooled)
        self.assertTrue(all(part is None for part in kept.applied))
        self.assertTrue(all(block.vectors is None for block in kept.orbitals.sectors))
        self.assertIsNone(completed.pooled)
        self.assertIsNone(found.pooled)
        scale = max(float(np.abs(part).max()) for part in reference)
        for index, sector in enumerate(orbitals.sectors):
            self.assertEqual(values[index].shape, (sector.rows, sector.local.size))
            np.testing.assert_allclose(values[index], reference[index], rtol=0.0, atol=1.0e-12 * scale)
            np.testing.assert_allclose(completed_values[index], completed_off[index], rtol=0.0, atol=1.0e-12 * scale)
            self.assertEqual(int(found.applied[index].device.id), sector.device)
        self.assertAlmostEqual(found.energy, whole.energy, delta=1.0e-12 * abs(whole.energy))
        self.assertEqual(found.pair_solves, whole.pair_solves)
        self.assertEqual(completed.pair_solves, whole.pair_solves)
        self.assertEqual(found.columns, whole.columns)
        text = solvers.describe(Counter({"translations": 2}))
        self.assertIn(f"in the classes of their {kept.orbitals.context.label}", text)
        return kept.orbitals.context

    def test_a_supercell_of_even_repeats(self) -> None:
        # |T| = 8: one class per pair, the eighth of the box
        context = self._compare((2, 2, 2), 16)
        self.assertEqual(context.order, 8)

    def test_the_face_centring_translations(self) -> None:
        # |T| = 4 with two cosets: two classes per pair; the screened kernel of HSE
        self._compare((2, 1, 1), 16, screened=True)
        self._compare((1, 1, 1), 12, batch_size=3, states=(2, 0, 3))

    def test_the_body_centre(self) -> None:
        # |T| = 2 with four cosets: four classes per pair on the eighth of the box
        context = self._compare((1, 1, 1), 16, kind="bcc")
        self.assertEqual((context.order, len(context.group.cosets)), (2, 4))
        self.assertEqual(context.layout.reduced_shape, (8, 8, 8))

    def test_boxes_halved_along_some_axes(self) -> None:
        # the twist and the kernel of the class on the halved axes only: x and y of three diamond cubes along z,
        # x alone of a simple cell (one class per pair each)
        context = self._compare((2, 2, 3), 16, states=(3, 0, 2, 4))
        self.assertEqual(context.layout.reduced_shape, (8, 8, 24))
        context = self._compare((2, 1, 1), 16, kind="simple", screened=True)
        self.assertEqual(context.layout.reduced_shape, (8, 8, 8))

    def test_several_devices(self) -> None:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 2:
            self.skipTest("one device")
        devices = tuple(range(min(count, 4)))
        # every tile on several devices (no small build), the sectors on all of them
        with mock.patch.object(translations, "SMALL_BUILD_POINTS", 0.0):
            self._compare((2, 1, 1), 16, devices=devices, states=(6, 5, 4, 3))
            self._compare((1, 1, 1), 16, devices=devices, kind="bcc")
            self._compare((2, 2, 1), 12, devices=devices, kind="simple")

    def test_coset_blocks_that_do_not_fit_take_the_pooled_block_or_the_tiles(self) -> None:
        from parsec_python.acceleration.Hamiltonian import exchange_device

        grid, _reduction, decomposition = _cell((2, 1, 1), 16)
        orbitals = _device_channel(grid, decomposition, states_per_sector=(3, 2), occupied_fraction=0.6, devices=(0,))
        solvers = _device_solver(grid, 4)
        context = translations.translation_context(orbitals, solvers)
        self.assertIsNotNone(context)
        need = translations.translation_exchange_bytes(orbitals, context, solvers, (0,))
        self.assertEqual(set(need), {0})
        self.assertGreater(need[0], 2 * 8 * context.coset_rows * orbitals.state_count)
        real = exchange_device._device_memory
        calls = []

        def memory(device):
            # the translation classes do not fit; the pooled block (asked next) does
            calls.append(device)
            if len(calls) == 1:
                return need[0] - 1, need[0] * 10
            return real(device)

        solvers.take_paths()
        with mock.patch.object(exchange_device, "_device_memory", side_effect=memory):
            exchange_device.apply_device_exchange(orbitals, solvers, (0,), columns="all")
        paths = solvers.take_paths()
        self.assertEqual(paths, Counter({"pool": 1, "full": 1, "translations-unfit": 1}))
        self.assertTrue(solvers.describe(paths).endswith(
            "; the coset blocks of the half-cell translations of the sector orbitals did not fit the device in "
            "1 of 1 builds"
        ))
        # neither fits: the tiles of the sectors, the same W
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_EXCHANGE_POOL": "auto"}):
            reference = exchange_device.apply_device_exchange(orbitals, solvers, (0,), columns="all")
            solvers.take_paths()
            with mock.patch.object(exchange_device, "_device_memory", return_value=(0, 1)):
                tiles = exchange_device.apply_device_exchange(orbitals, solvers, (0,), columns="all")
        self.assertEqual(solvers.take_paths(), Counter({"pool-fallback": 1, "full": 1, "translations-unfit": 1}))
        import cupy as cp

        for found, expected in zip(tiles.applied, reference.applied):
            scale = float(cp.abs(expected).max())
            np.testing.assert_allclose(cp.asnumpy(found), cp.asnumpy(expected), rtol=0.0, atol=1.0e-12 * scale)


if __name__ == "__main__":
    unittest.main()

"""The update of ``W`` by coset row and column in the translation classes (``PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE``).

The kernel by rows (``off``) runs every pair of a batch in one thread per coset row; the kernel by columns (``auto``)
runs, in one thread per coset row and column of ``W``, the pairs that write that column in their order
(:func:`.exchange_translations.column_events`).  The same operations in the same order give the same bits.

On the CPU: the events of every tile kind and batch, played as scalar sums in both orders, give the same values to
the last bit.  On a GPU: one update of random class potentials with both kernels, and the builds of random sector
orbitals, bit for bit.
"""

from __future__ import annotations

from collections import Counter
import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Hamiltonian import exchange_translations as translations
from parsec_python.acceleration.Hamiltonian.exchange_device import _tile_pairs
from parsec_python.acceleration.backends.cupy import cupy_available


def _by_rows(left, right, update, v, first, second, first_weights, second_weights, first_acc, second_acc):
    """One coset row of the kernel by rows (``translation_pair_update``), on Python floats."""

    total, current = 0.0, -1
    for k in range(len(left)):
        i, j = int(left[k]), int(right[k])
        if i != current:
            if current >= 0:
                first_acc[current] = first_acc[current] - total
            total, current = 0.0, i
        total = total + second_weights[j] * second[j] * v[k]
        if update[k]:
            second_acc[j] = second_acc[j] - first_weights[i] * first[i] * v[k]
    if current >= 0:
        first_acc[current] = first_acc[current] - total


def _by_columns(offsets, events, roles, split, aliased, left, right, v, first, second, first_weights,
                second_weights, first_acc, second_acc):
    """One coset row of the kernel by columns (``translation_pair_update_columns``), column by column."""

    for column in range(len(offsets) - 1):
        begin, end = int(offsets[column]), int(offsets[column + 1])
        if begin == end:
            continue
        target, index = (first_acc, column) if aliased or column < split else (second_acc, column - split)
        value, total, running, last = target[index], 0.0, False, -2
        for e in range(begin, end):
            k, role = int(events[e]), int(roles[e])
            if running and not ((role & 1) and k == last + 1):
                value, running = value - total, False
            if role & 1:
                if not running:
                    total, running = 0.0, True
                j = int(right[k])
                total = total + second_weights[j] * second[j] * v[k]
                last = k
            if role & 2:
                i = int(left[k])
                value = value - first_weights[i] * first[i] * v[k]
        if running:
            value = value - total
        target[index] = value


class ColumnEventTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("AUTO", "auto"), (" off ", "off")):
            with mock.patch.dict(os.environ, {translations.COLUMN_UPDATE_SETTING: value}):
                self.assertEqual(translations.exchange_column_update_setting(), expected)
        for value in ("on", "rows", "1"):
            with mock.patch.dict(os.environ, {translations.COLUMN_UPDATE_SETTING: value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE must be auto or off"):
                    translations.exchange_column_update_setting()

    def test_the_events_play_the_sums_of_the_rows_in_the_same_order(self) -> None:
        rng = np.random.default_rng(11)
        for kind, first_count, second_count in (
            ("diagonal", 7, 7), ("diagonal", 1, 1), ("pair", 5, 4), ("pair", 3, 6), ("empty", 4, 5),
        ):
            left_all, right_all, update_all = _tile_pairs(kind, first_count, second_count)
            # batches of every size: runs of one i cut by the end of a batch, a batch of one pair
            for step in (1, 2, 3, 5, int(left_all.size)):
                first = [float(value) for value in rng.standard_normal(first_count)]
                second = first if kind == "diagonal" else [float(value) for value in rng.standard_normal(second_count)]
                first_weights = [float(value) for value in rng.uniform(0.0, 2.0, first_count)]
                second_weights = first_weights if kind == "diagonal" else [
                    float(value) for value in rng.uniform(0.0, 2.0, second_count)
                ]
                start_first = [float(value) for value in rng.standard_normal(first_count)]
                start_second = [float(value) for value in rng.standard_normal(second_count)]
                rows_first, rows_second = list(start_first), list(start_second)
                columns_first, columns_second = list(start_first), list(start_second)
                if kind == "diagonal":
                    rows_second, columns_second = rows_first, columns_first
                aliased = kind != "pair"
                for begin in range(0, int(left_all.size), step):
                    finish = min(begin + step, int(left_all.size))
                    left, right, update = left_all[begin:finish], right_all[begin:finish], update_all[begin:finish]
                    v = [float(value) for value in rng.standard_normal(left.size)]
                    _by_rows(left, right, update, v, first, second, first_weights, second_weights, rows_first,
                             rows_second if kind != "empty" else rows_first)
                    offsets, events, roles = translations.column_events(left, right, update, first_count,
                                                                        aliased=aliased)
                    self.assertEqual(int(offsets[-1]), events.size)
                    self.assertTrue(np.all(np.diff(offsets) >= 0))
                    _by_columns(offsets, events, roles, first_count, aliased, left, right, v, first, second,
                                first_weights, second_weights, columns_first,
                                columns_second if kind != "empty" else columns_first)
                with self.subTest(kind=kind, step=step):
                    self.assertEqual(columns_first, rows_first)
                    self.assertEqual(columns_second, rows_second)

    def test_the_classes_that_take_the_update_by_columns(self) -> None:
        # one class (a supercell of even repeats) and two (the face-centring translations) take it; four (a
        # body-centred cell, where reading the class potentials twice was slower) keep the update by rows
        layout = translations.TranslationLayout((2, 2, 2), 0, (1, 1, 1), np.zeros((8, 1), dtype=np.int64))
        for cosets, expected in ((1, True), (2, True), (4, False)):
            group = translations.ParityGroup(tuple(range(8 // cosets)), tuple(range(cosets)))
            context = translations.TranslationContext(layout, group, (), (), 64)
            with mock.patch.dict(os.environ, {translations.COLUMN_UPDATE_SETTING: "auto"}):
                self.assertIs(context.updates_by_columns(), expected, cosets)
            with mock.patch.dict(os.environ, {translations.COLUMN_UPDATE_SETTING: "off"}):
                self.assertFalse(context.updates_by_columns())

    def test_a_pair_of_one_column_is_one_event(self) -> None:
        # i == j of one accumulator with an update (not a tile of the builds): both roles in one event, sum first
        offsets, events, roles = translations.column_events(
            np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0]), 2, aliased=True
        )
        self.assertEqual(offsets.tolist(), [0, 2, 4])
        self.assertEqual(events.tolist(), [0, 1, 1, 2])
        self.assertEqual(roles.tolist(), [3, 1, 2, 1])


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceColumnUpdateTests(unittest.TestCase):
    def _context(self, cosets: int, size: int):
        layout = translations.TranslationLayout((2, 2, 2), 0, (1, 1, size), np.zeros((8, size), dtype=np.int64))
        group = translations.ParityGroup(tuple(range(8 // cosets)), tuple(range(cosets)))
        return translations.TranslationContext(layout, group, (), (), 64)

    def test_one_update_by_columns_is_the_update_by_rows(self) -> None:
        import cupy as cp

        rng = np.random.default_rng(5)
        for kind, first_count, second_count, cosets, classes, step in (
            ("diagonal", 9, 9, 1, 1, 7),
            ("pair", 6, 5, 2, 2, 8),
            ("pair", 4, 7, 4, 4, 28),
            ("empty", 5, 3, 1, 1, 4),
        ):
            size = 300
            context = self._context(cosets, size)
            rows = cosets * size
            pairs = _tile_pairs(kind, first_count, second_count)
            first = cp.asfortranarray(cp.asarray(rng.standard_normal((rows, first_count))))
            second = first if kind == "diagonal" else cp.asfortranarray(
                cp.asarray(rng.standard_normal((rows, second_count)))
            )
            first_weights = cp.asarray(rng.uniform(0.0, 2.0, first_count))
            second_weights = first_weights if kind == "diagonal" else cp.asarray(rng.uniform(0.0, 2.0, second_count))
            table = cp.asarray(rng.choice([-1.0, 1.0], size=(classes, cosets)))
            twists = cp.asarray(np.exp(1j * rng.uniform(0.0, 6.3, (classes, size))))
            start_first = rng.standard_normal((rows, first_count))
            start_second = rng.standard_normal((rows, second_count))
            results = {}
            for setting in ("off", "auto"):
                first_acc = cp.asfortranarray(cp.asarray(start_first))
                second_acc = (
                    first_acc if kind == "diagonal" else
                    cp.asfortranarray(cp.asarray(start_second)) if kind == "pair" else None
                )
                draws = np.random.default_rng(17)
                left_all, right_all, update_all = (cp.asarray(values) for values in pairs)
                for begin in range(0, int(left_all.size), step):
                    finish = min(begin + step, int(left_all.size))
                    count = finish - begin
                    halves = translations.batch_halves(count)
                    boxes = cp.asarray(
                        draws.standard_normal((classes, halves, size)) + 1j * draws.standard_normal((classes, halves, size))
                    )
                    events = None
                    if setting == "auto":
                        aliased = second_acc is None or second_acc is first_acc
                        events = translations._device_events(
                            cp, (kind, first_count, second_count), pairs, begin, finish, first_count, aliased
                        )
                    translations.translation_update(
                        cp, context, boxes, twists, table, first, second, left_all[begin:finish],
                        right_all[begin:finish], first_weights, second_weights, update_all[begin:finish],
                        first_acc, second_acc, events=events,
                    )
                results[setting] = (cp.asnumpy(first_acc), None if second_acc is None else cp.asnumpy(second_acc))
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(results["auto"][0], results["off"][0])
                if results["off"][1] is not None:
                    np.testing.assert_array_equal(results["auto"][1], results["off"][1])
                self.assertFalse(np.array_equal(results["off"][0], start_first))

    def test_the_builds_are_those_of_the_update_by_rows(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange
        from parsec_python.acceleration.tests.test_exchange_translations import _cell, _device_channel, _device_solver

        # one class per pair (diamond 2^3), two (the face-centring translations of the cube) and four (bcc, by
        # columns only where the limit of the classes is raised)
        for repeats, points, kind, states in (
            ((2, 2, 2), 16, "diamond", (5, 3, 0, 4)), ((1, 1, 1), 12, "diamond", (2, 0, 3)),
            ((1, 1, 1), 16, "bcc", (4, 6)),
        ):
            grid, _reduction, decomposition = _cell(repeats, points, kind=kind)
            orbitals = _device_channel(grid, decomposition, states_per_sector=states, occupied_fraction=0.6,
                                       devices=(0,))
            solvers = _device_solver(grid, 4)
            found = {}
            for setting in ("off", "auto"):
                variables = {translations.COLUMN_UPDATE_SETTING: setting, translations.SETTING: "auto"}
                with mock.patch.dict(os.environ, variables), mock.patch.object(
                    translations, "COLUMN_UPDATE_CLASSES", 8
                ), mock.patch.object(translations, "_device_events", wraps=translations._device_events) as events:
                    solvers.take_paths()
                    occupied = apply_device_exchange(orbitals, solvers, (0,), columns="occupied")
                    energy = occupied.energy
                    completed = apply_device_exchange(orbitals, solvers, (0,), columns="all", known=occupied)
                    self.assertEqual(solvers.take_paths(), Counter({"translations": 2}))
                    self.assertEqual(events.called, setting == "auto")
                found[setting] = (energy, [cp.asnumpy(part) for part in completed.applied])
            with self.subTest(kind=kind):
                self.assertEqual(found["auto"][0], found["off"][0])
                for auto_part, off_part in zip(found["auto"][1], found["off"][1]):
                    np.testing.assert_array_equal(auto_part, off_part)


if __name__ == "__main__":
    unittest.main()

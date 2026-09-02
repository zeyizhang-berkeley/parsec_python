"""Sequence parity for the skip-ahead LAPACK random generator."""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom
from parsec_python.Eigensolvers.lapack_random import (
    LapackRandom as ReadableLapackRandom,
)


class VectorizedLapackRandomTests(unittest.TestCase):
    def test_threshold_lengths_are_bit_exact(self) -> None:
        for count in (0, 1, 7, 16_383, 16_384, 16_385, 100_000):
            with self.subTest(count=count):
                expected = ReadableLapackRandom()
                actual = LapackRandom()
                np.testing.assert_array_equal(
                    actual.uniform_0_1(count), expected.uniform_0_1(count)
                )
                self.assertEqual(actual.seed, expected.seed)

    def test_multiple_calls_and_column_major_shape_preserve_stream(self) -> None:
        expected = ReadableLapackRandom()
        actual = LapackRandom()
        for shape in ((37, 5), (20_003,), (11, 3)):
            np.testing.assert_array_equal(
                actual.uniform_minus_1_1(shape, column_major=True),
                expected.uniform_minus_1_1(shape, column_major=True),
            )
            self.assertEqual(actual.seed, expected.seed)


if __name__ == "__main__":
    unittest.main()

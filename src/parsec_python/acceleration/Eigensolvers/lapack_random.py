"""Bit-exact, vectorized host generation of PARSEC's DLARNV stream.

The readable implementation advances the 48-bit linear congruential
generator one value at a time.  That is ideal as an explanation and very
costly for an ``N x states`` GPU trial basis.  If ``L`` consecutive states
are initialized once, every lane can jump to its next value independently:

``s[k + L] = a**L * s[k] mod 2**48``.

NumPy then advances 2,048 lanes per operation.  The values and final LAPACK
seed remain bit-for-bit identical; this changes only how independent chunks
of the already-defined sequence are evaluated on the host.
"""

from __future__ import annotations

import numpy as np

from parsec_python.Eigensolvers.lapack_random import (
    LapackRandom as _ReadableLapackRandom,
    PARSEC_RANDOM_ARRAY_SEED,
)


_BASE = 4096
_MODULUS = 1 << 48
_MASK = np.uint64(_MODULUS - 1)
_MULTIPLIER = 33_952_834_046_453
# A lane is initialized by a dependent scalar LCG step, whereas all later
# jumps are vectorized.  Sweeps from 512 through 32,768 on the 65k-row sector
# sizes used by the solver put the crossover near 2,048: more lanes spend
# unnecessary time in Python, fewer lanes issue too many small NumPy passes.
# Lane count does not enter the random-number definition, so this is a pure
# execution-tiling choice and the stream/final seed remain bit exact.
_LANES = 2_048


def _seed_to_integer(seed: tuple[int, int, int, int]) -> int:
    state = 0
    for digit in seed:
        state = _BASE * state + int(digit)
    return state


def _integer_to_seed(state: int) -> tuple[int, int, int, int]:
    digits = [0, 0, 0, 0]
    for index in range(3, -1, -1):
        state, digits[index] = divmod(state, _BASE)
    return tuple(digits)  # type: ignore[return-value]


class LapackRandom(_ReadableLapackRandom):
    """Drop-in accelerated version of the readable stateful generator."""

    def uniform_0_1(self, count: int) -> np.ndarray:
        """Return the exact LAPACK stream using vectorized skip-ahead lanes."""

        count = int(count)
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return np.empty(0, dtype=np.float64)

        state = _seed_to_integer(self.seed)
        lane_count = min(count, _LANES)
        states = np.empty(lane_count, dtype=np.uint64)
        # Only this short prefix is dependent scalar work.  Subsequent blocks
        # are independent applications of the exact L-step transition.
        for index in range(lane_count):
            state = (_MULTIPLIER * state) % _MODULUS
            states[index] = state

        values = np.empty(count, dtype=np.float64)
        scale = 1.0 / _MODULUS
        values[:lane_count] = states * scale
        offset = lane_count
        if offset < count:
            jump = np.uint64(pow(_MULTIPLIER, lane_count, _MODULUS))
            while offset < count:
                # Unsigned overflow discards high bits; masking then gives
                # multiplication modulo 2**48 exactly.
                states = np.bitwise_and(states * jump, _MASK)
                take = min(lane_count, count - offset)
                values[offset : offset + take] = states[:take] * scale
                state = int(states[take - 1])
                offset += take

        self.seed = _integer_to_seed(state)
        return values

__all__ = ["LapackRandom", "PARSEC_RANDOM_ARRAY_SEED"]

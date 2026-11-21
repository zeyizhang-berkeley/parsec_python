"""Two-byte coefficient codes of the sector stencils (``PARSEC_SECTOR_PALETTE_CODES``, 2026-11-19).

A packed stencil names each coefficient by a code into an exact FP64 palette.  The codes were one byte, a palette of
256 values: the sector stencils of an even repeat of a cubic cell on ``n = 2 mod 4`` points per edge (orbits of
several sizes; Si 2x2x2 at Grid_Spacing 0.6 needs 295 values at the default order 12) or of a cell whose grid steps
differ at a high order need more, and such a cell stepped down the palette ladder to D2 or the full grid.  ``auto``,
the default, codes the stencil of such a sector in two bytes (``uint16``, up to 65536 values) and keeps the group;
every other stencil keeps its one-byte codes and its kernels.  ``on`` codes every sector stencil in two bytes (the
same coefficients, palette and order: a measurement setting) and ``off`` keeps one byte and the ladder.

The host tests compare the arrays of every builder with the reduced full-grid matrix, the packing into tiles, the
operator cache, the sources of the kernels and the cluster route.  The device tests (A100) compare the kernels of
two-byte codes with those of one-byte codes to the bit and a sector of 295 values with its CSR.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.acceleration.Symmetry import sector_stencil
from parsec_python.acceleration.Symmetry.operator_cache import load_or_build_reduced_operators
from parsec_python.acceleration.Symmetry.sector_stencil import (
    PALETTE_CAPACITY,
    PaletteOverflow,
    build_sector_stencils,
    sector_palette_sizes,
)
from parsec_python.acceleration.backends import cupy_mixed_precision
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.cupy_launch import kernel_parameters, launch_arguments
from parsec_python.acceleration.backends.cupy_stencil_major import (
    _CUDA_SOURCE,
    NARROW_CODES,
    WIDE_CODES,
    WIDE_PALETTE_CAPACITY,
    StencilMajorHostMetadata,
    build_stencil_major_metadata,
    code_source,
    sector_code_dtype,
    sector_palette_capacity,
    sector_palette_codes_setting,
)
from parsec_python.acceleration.backends.implicit_stencil import (
    PackedTileHostMetadata,
    _pack_affine_tiles_reference,
    implicit_source,
    pack_affine_tiles,
    regular_tile_share,
    stencil_source,
    unpack_affine_tiles,
)
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.backends.stencil_pipeline import pipelined_source
from parsec_python.acceleration.tests.test_sector_palette import (
    _SILICON_SIDE,
    _decomposition,
    _diamond,
    _grid,
    _groups,
    silicon_supercell,
)

_SWITCH = "PARSEC_SECTOR_PALETTE_CODES"


def _wide(metadata):
    """``metadata`` with its codes in two bytes: the same values, palette and order."""

    return StencilMajorHostMetadata(
        shape=metadata.shape,
        neighbors=metadata.neighbors,
        coefficient_codes=metadata.coefficient_codes.astype(np.uint16),
        coefficient_palette=metadata.coefficient_palette,
    )


def _assert_same_arrays(test, actual, expected, label=""):
    test.assertEqual(actual.shape, expected.shape, label)
    test.assertEqual(actual.coefficient_codes.dtype, expected.coefficient_codes.dtype, label)
    np.testing.assert_array_equal(actual.neighbors, expected.neighbors, err_msg=label)
    np.testing.assert_array_equal(actual.coefficient_codes, expected.coefficient_codes, err_msg=label)
    np.testing.assert_array_equal(
        actual.coefficient_palette.view(np.uint64), expected.coefficient_palette.view(np.uint64), err_msg=label
    )


class _Silicon:
    """Si 2x2x2 at Grid_Spacing 0.6, order 12 (34^3 points): the weighted sectors of its point group, 129 to 295
    values per sector, and the commuting group with translations."""

    _cached = None

    @classmethod
    def get(cls):
        if cls._cached is None:
            grid = _grid((2 * _SILICON_SIDE,) * 3, spacing=0.6, order=12)
            weighted, abelian, origin = _groups(grid, _diamond(_SILICON_SIDE, 2, "Si"))
            cls._cached = SimpleNamespace(
                grid=grid,
                weighted=_decomposition(grid, weighted),
                abelian=_decomposition(grid, abelian),
                reductions=(weighted, abelian, origin),
            )
        return cls._cached


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for raw, expected in (
            (None, "auto"), ("", "auto"), ("auto", "auto"), (" AUTO ", "auto"), ("on", "on"), ("1", "on"),
            ("true", "on"), ("yes", "on"), ("off", "off"), ("0", "off"), ("false", "off"), ("no", "off"),
        ):
            environment = {} if raw is None else {_SWITCH: raw}
            with mock.patch.dict(os.environ, environment):
                if raw is None:
                    os.environ.pop(_SWITCH, None)
                self.assertEqual(sector_palette_codes_setting(), expected, raw)
        for raw in ("16", "wide", "two"):
            with mock.patch.dict(os.environ, {_SWITCH: raw}), self.assertRaises(ValueError) as caught:
                sector_palette_codes_setting()
            self.assertIn(_SWITCH, str(caught.exception))

    def test_capacity_and_dtype(self) -> None:
        self.assertEqual(sector_palette_capacity("off"), PALETTE_CAPACITY)
        self.assertEqual(sector_palette_capacity("auto"), WIDE_PALETTE_CAPACITY)
        self.assertEqual(sector_palette_capacity("on"), WIDE_PALETTE_CAPACITY)
        for size, setting, expected in (
            (256, "auto", np.uint8), (257, "auto", np.uint16), (65536, "auto", np.uint16), (1, "on", np.uint16),
            (256, "on", np.uint16), (256, "off", np.uint8),
        ):
            self.assertEqual(sector_code_dtype(size, setting), np.dtype(expected), (size, setting))
        for size, setting in ((257, "off"), (65537, "auto"), (65537, "on")):
            with self.assertRaises(PaletteOverflow):
                sector_code_dtype(size, setting)



class MetadataTests(unittest.TestCase):
    def test_two_byte_codes_hold_a_palette_beyond_256(self) -> None:
        rows, palette = 600, np.linspace(-3.0, 3.0, 300)
        neighbors = np.vstack([np.arange(rows), (np.arange(rows) + 1) % rows]).astype(np.int32)
        codes = np.vstack([np.arange(rows) % 300, (np.arange(rows) * 7) % 300]).astype(np.uint16)
        metadata = StencilMajorHostMetadata((rows, rows), neighbors, codes, palette)
        self.assertEqual(metadata.coefficient_codes.dtype, np.uint16)
        reference = sp.csr_matrix(
            (palette[codes].T.ravel(), neighbors.T.ravel(), np.arange(0, 2 * rows + 1, 2)), shape=(rows, rows)
        )
        np.testing.assert_array_equal(metadata.to_csr().toarray(), reference.toarray())
        with self.assertRaises(ValueError):
            StencilMajorHostMetadata((rows, rows), neighbors, codes.astype(np.uint8), palette)
        bad = codes.copy()
        bad[1, 5] = 300
        with self.assertRaises(ValueError):
            StencilMajorHostMetadata((rows, rows), neighbors, bad, palette)
        # codes of any other dtype are converted to one byte, as before
        converted = StencilMajorHostMetadata((rows, rows), neighbors, (codes % 200).astype(np.int64), palette[:200])
        self.assertEqual(converted.coefficient_codes.dtype, np.uint8)

    def test_the_csr_route_codes_as_the_setting_says(self) -> None:
        values = np.linspace(-1.0, 1.0, 400)
        rows = values.size
        matrix = sp.csr_matrix((values, np.arange(rows), np.arange(rows + 1)), shape=(rows, rows))
        with self.assertRaises(PaletteOverflow):
            build_stencil_major_metadata(matrix)
        with self.assertRaises(PaletteOverflow):
            build_stencil_major_metadata(matrix, codes="off")
        wide = build_stencil_major_metadata(matrix, codes="auto")
        self.assertEqual(wide.coefficient_codes.dtype, np.uint16)
        np.testing.assert_array_equal(wide.to_csr().toarray(), matrix.toarray())
        small = matrix[:100, :100]
        self.assertEqual(build_stencil_major_metadata(small, codes="auto").coefficient_codes.dtype, np.uint8)
        on = build_stencil_major_metadata(small, codes="on")
        _assert_same_arrays(self, on, _wide(build_stencil_major_metadata(small)))


class SourceTests(unittest.TestCase):
    def test_the_sources_of_two_byte_codes_change_one_parameter_type(self) -> None:
        self.assertIs(code_source(_CUDA_SOURCE, np.uint8), _CUDA_SOURCE)
        wide = code_source(_CUDA_SOURCE, np.uint16)
        self.assertEqual(wide.count(WIDE_CODES), 2)
        self.assertNotIn(NARROW_CODES, wide)
        self.assertEqual(wide.replace(WIDE_CODES, NARROW_CODES), _CUDA_SOURCE)
        with self.assertRaises(ValueError):
            code_source(_CUDA_SOURCE, np.int32)
        for tile in (0, 16, 32):
            narrow = stencil_source(tile, 37, "pipelined")
            two = stencil_source(tile, 37, "pipelined", np.uint16)
            self.assertIn("const unsigned short* __restrict__ meta_c", two)
            self.assertNotIn("unsigned char", two)
            self.assertEqual(
                two.replace(WIDE_CODES, NARROW_CODES).replace(
                    "const unsigned short* __restrict__ meta_c", "const unsigned char* __restrict__ meta_c"
                ),
                narrow,
            )
            self.assertEqual(stencil_source(tile, 37, "former", np.uint16).replace(WIDE_CODES, NARROW_CODES),
                             stencil_source(tile, 37, "former"))
        self.assertEqual(implicit_source(16, np.uint16).count(WIDE_CODES), 2)
        self.assertEqual(pipelined_source(_CUDA_SOURCE, 25, 0), stencil_source(0, 25, "pipelined"))

    def test_the_fp32_recurrence_declares_two_byte_codes(self) -> None:
        name = cupy_mixed_precision._KERNEL_NAMES[0]
        narrow = dict((parameter, kind) for kind, parameter in kernel_parameters(cupy_mixed_precision._SOURCE, name))
        wide = dict(
            (parameter, kind) for kind, parameter in kernel_parameters(cupy_mixed_precision._source(np.uint16), name)
        )
        self.assertEqual(narrow["coefficient_codes"], "unsigned char *")
        self.assertEqual(wide["coefficient_codes"], "unsigned short *")
        self.assertEqual(cupy_mixed_precision._source(np.uint8), cupy_mixed_precision._SOURCE)
        parameters = kernel_parameters(cupy_mixed_precision._source(np.uint16), name)
        values = {
            parameter: (np.zeros(1, {"unsigned short *": np.uint16, "unsigned char *": np.uint8, "int *": np.int32,
                                     "float *": np.float32, "double *": np.float64}[kind])
                        if kind.endswith("*") else 0)
            for kind, parameter in parameters
        }
        self.assertEqual(len(launch_arguments(parameters, values)), len(parameters))
        values["coefficient_codes"] = np.zeros(1, np.uint8)
        with self.assertRaises(Exception):
            launch_arguments(parameters, values)


class BuilderTests(unittest.TestCase):
    """The sector stencils of Si 2x2x2 at Grid_Spacing 0.6 from the grid and from the reduced full-grid matrix."""

    def test_sectors_beyond_256_values_code_them_in_two_bytes(self) -> None:
        case = _Silicon.get()
        sizes = sector_palette_sizes(case.grid, case.weighted, wrap=True)
        self.assertEqual(max(sizes), 295)
        built = build_sector_stencils(case.grid, case.weighted, implementation="numpy", wrap=True)
        matrix = build_negative_laplacian(case.grid)
        from parsec_python.acceleration.tests.test_sector_stencil import _reduced_like_the_native_route

        wide_sectors = 0
        for index, (size, stencil) in enumerate(zip(sizes, built, strict=True)):
            self.assertEqual(int(stencil.coefficient_palette.size), size)
            self.assertEqual(stencil.coefficient_codes.dtype, np.uint16 if size > PALETTE_CAPACITY else np.uint8)
            wide_sectors += size > PALETTE_CAPACITY
            if size > PALETTE_CAPACITY or index == 0:
                reduced = _reduced_like_the_native_route(matrix, case.weighted, index)
                _assert_same_arrays(self, stencil, build_stencil_major_metadata(reduced, codes="auto"), f"sector {index}")
                np.testing.assert_array_equal(stencil.to_csr().toarray(), reduced.toarray())
        self.assertEqual(wide_sectors, 3)

    def test_on_codes_every_sector_in_two_bytes_and_off_stops(self) -> None:
        case = _Silicon.get()
        auto = build_sector_stencils(case.grid, case.weighted, implementation="numpy", wrap=True)
        with mock.patch.dict(os.environ, {_SWITCH: "on"}):
            on = build_sector_stencils(case.grid, case.weighted, implementation="numpy", wrap=True)
        for index, (left, right) in enumerate(zip(on, auto, strict=True)):
            self.assertEqual(left.coefficient_codes.dtype, np.uint16)
            _assert_same_arrays(self, left, _wide(right), f"sector {index}")
        with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaises(PaletteOverflow) as caught:
            build_sector_stencils(case.grid, case.weighted, implementation="numpy", wrap=True)
        self.assertIn(f"more than {PALETTE_CAPACITY} coefficients", str(caught.exception))

    def test_the_check_keeps_the_group_and_names_the_count(self) -> None:
        from parsec_python.acceleration import driver

        case = _Silicon.get()
        weighted, abelian, origin = case.reductions
        for reduction in (weighted, abelian):
            checked, refusal = driver._sector_palette_check(
                case.grid, reduction, lambda orbital: (_decomposition(case.grid, orbital), None)
            )
            self.assertIsNone(refusal)
            self.assertIs(checked[0], reduction)
            self.assertEqual(checked[2], 295)
        checked, refusal = driver._sector_palette_check(
            case.grid, origin, lambda orbital: (_decomposition(case.grid, orbital), None)
        )
        self.assertIsNone(refusal)
        self.assertEqual(checked[2], 0)
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            checked, refusal = driver._sector_palette_check(
                case.grid, weighted, lambda orbital: (_decomposition(case.grid, orbital), None)
            )
        self.assertIsNone(checked)
        self.assertIn("PARSEC_SECTOR_PALETTE_CODES=off", refusal)

    def test_the_bundle_of_both_routes_and_its_cache(self) -> None:
        # the commuting group with translations (295 values; a disk entry, unlike the weighted sectors): the bundle
        # from the grid equals the one through the matrix, and the cache keeps the codes as they are
        from parsec_python.acceleration.tests.test_sector_stencil import _nonlocal

        case = _Silicon.get()
        matrix = build_negative_laplacian(case.grid)
        operator = _nonlocal(case.grid)
        with mock.patch.dict(os.environ, {"PARSEC_ACCELERATED_RESIDENT": "0"}), tempfile.TemporaryDirectory() as folder:
            direct = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=Path(folder))
            self.assertEqual(direct.cache_info.stencil_builder, "direct-numpy-wrapped")
            self.assertEqual(direct.cache_info.status, "miss-written")
            dtypes = [stencil.coefficient_codes.dtype for stencil in direct.stencil_metadata]
            self.assertIn(np.dtype(np.uint16), dtypes)
            self.assertIn(np.dtype(np.uint8), dtypes)
            # the route through the matrix with the native reduction (the order of the builder's sums); SciPy's
            # reduction adds the entries of one sector column in another order
            from parsec_python.acceleration.tests.test_periodic_unequal_steps import _native_reduction

            native = _native_reduction()
            with mock.patch.dict(
                os.environ, {"PARSEC_SECTOR_STENCIL": "csr", "PARSEC_NATIVE_SECTOR_ASSEMBLY": "1" if native else "0"}
            ):
                through_matrix = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=None)
            self.assertEqual(through_matrix.cache_info.stencil_builder, "csr")
            for left, right in zip(direct.stencil_metadata, through_matrix.stencil_metadata, strict=True):
                if native:
                    _assert_same_arrays(self, left, right)
                else:
                    np.testing.assert_array_equal(left.neighbors, right.neighbors)
                    np.testing.assert_allclose(left.to_csr().toarray(), right.to_csr().toarray(), rtol=0, atol=1e-12)
                    self.assertEqual(
                        right.coefficient_codes.dtype,
                        np.uint16 if right.coefficient_palette.size > PALETTE_CAPACITY else np.uint8,
                    )
            loaded = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=Path(folder))
            self.assertEqual(loaded.cache_info.status, "hit")
            for left, right in zip(loaded.stencil_metadata, direct.stencil_metadata, strict=True):
                _assert_same_arrays(self, left, right)
            # off: the entry of two-byte codes is not read; the rebuild stops as before
            with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaises(PaletteOverflow):
                load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=Path(folder))
            # on: an entry of its own, every stencil in two bytes
            with mock.patch.dict(os.environ, {_SWITCH: "on"}):
                on = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=Path(folder))
            self.assertNotEqual(on.cache_info.key, direct.cache_info.key)
            self.assertEqual(on.cache_info.status, "miss-written")
            for left, right in zip(on.stencil_metadata, direct.stencil_metadata, strict=True):
                _assert_same_arrays(self, left, _wide(right) if right.coefficient_codes.dtype == np.uint8 else right)

    def test_a_resident_bundle_of_two_byte_codes_is_not_taken_under_off(self) -> None:
        from parsec_python.acceleration.Symmetry import operator_cache
        from parsec_python.acceleration.tests.test_sector_stencil import _nonlocal

        case = _Silicon.get()
        matrix = build_negative_laplacian(case.grid)
        operator = _nonlocal(case.grid)
        with mock.patch.dict(os.environ, {"PARSEC_ACCELERATED_RESIDENT": "1"}), \
                mock.patch.dict(operator_cache._MEMORY_BUNDLES, clear=True):
            first = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=None)
            again = load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=None)
            self.assertEqual(again.cache_info.status, "memory-hit")
            self.assertIs(again.stencil_metadata, first.stencil_metadata)
            with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaises(PaletteOverflow):
                load_or_build_reduced_operators(case.abelian, matrix, operator, cache_directory=None)

    def test_tiles_of_two_byte_codes(self) -> None:
        case = _Silicon.get()
        built = build_sector_stencils(case.grid, case.weighted, (3,), implementation="numpy", wrap=True)[3]
        self.assertEqual(built.coefficient_codes.dtype, np.uint16)
        self.assertEqual(int(built.coefficient_palette.size), 295)
        slots, rows = built.neighbors.shape
        for tile in (16, 32):
            packed_n, packed_c, statistics = pack_affine_tiles(built, tile)
            reference = _pack_affine_tiles_reference(built, tile)
            np.testing.assert_array_equal(packed_n, reference[0])
            np.testing.assert_array_equal(packed_c, reference[1])
            self.assertEqual(packed_c.dtype, np.uint16)
            self.assertEqual(statistics["packed_bytes"], 6 * packed_n.size)
            self.assertEqual(statistics["original_bytes"], 6 * slots * rows)
            neighbors, codes = unpack_affine_tiles(packed_n, packed_c, rows, slots, tile)
            active = built.neighbors >= 0
            np.testing.assert_array_equal(neighbors, built.neighbors)
            np.testing.assert_array_equal(codes[active], built.coefficient_codes[active])
            host = PackedTileHostMetadata(built.shape, slots, packed_n, packed_c, built.coefficient_palette,
                                          dict(statistics))
            self.assertEqual(host.tile, tile)
            with self.assertRaises(ValueError):
                PackedTileHostMetadata(built.shape, slots, packed_n, packed_c.astype(np.uint8),
                                       built.coefficient_palette, dict(statistics))
            # the share of regular tiles: that of the packing itself (every tile of this sector is looked at)
            self.assertEqual(
                regular_tile_share(built, tile), statistics["regular_rows"] / statistics["rows"]
            )


class ClusterRouteTests(unittest.TestCase):
    """A cluster's sector of more than 256 values: the native kernel codes in one byte and refuses it, NumPy's builder
    builds it in two (none measured: at most 229 values, Td at order 20).  Stand-ins for the kernel."""

    def _case(self):
        from parsec_python.acceleration.tests.test_sector_stencil import _case

        return _case("d2 cluster, points on the axes")

    def test_numpy_builds_what_the_kernel_refuses(self) -> None:
        grid, decomposition = self._case()
        expected = build_sector_stencils(grid, decomposition, implementation="numpy")

        def refuse(native, grid, decomposition, index):
            raise ValueError("finite-difference operator has more than 256 coefficients")

        def narrow(native, grid, decomposition, index):
            return expected[index]

        with mock.patch.object(sector_stencil, "native_kernel_available", return_value=True), \
                mock.patch("parsec_python.acceleration.backends.native._load_native", return_value=object()):
            with mock.patch.object(sector_stencil, "_native_sector_stencil", refuse):
                built = build_sector_stencils(grid, decomposition, implementation="native")
                for left, right in zip(built, expected, strict=True):
                    _assert_same_arrays(self, left, right)
                with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaises(ValueError):
                    build_sector_stencils(grid, decomposition, implementation="native")
            with mock.patch.object(sector_stencil, "_native_sector_stencil", narrow), \
                    mock.patch.dict(os.environ, {_SWITCH: "on"}):
                for left, right in zip(build_sector_stencils(grid, decomposition, implementation="native"), expected,
                                       strict=True):
                    _assert_same_arrays(self, left, _wide(right))


# ----------------------------------------------------------------------------------------------------------------
# Device tests (A100)
# ----------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class TwoByteKernelTests(unittest.TestCase):
    """The kernels of two-byte codes against those of one-byte codes on the same stencil: every word the same."""

    def setUp(self) -> None:
        from parsec_python.acceleration.backends.cupy import require_cupy

        self.cp, _ = require_cupy()
        self.verified = self.cp.cuda.Device().compute_capability == "80"

    def check(self, found, label) -> None:
        if self.verified:
            self.assertEqual(found["differing_words"], 0, f"{label}: {found['differing']}")
        else:
            self.assertEqual(found["differing_not_as_two_nans"], 0, label)

    def test_every_launch_has_the_bits_of_one_byte_codes(self) -> None:
        from parsec_python.acceleration.backends.cupy_stencil_major import CuPyStencilMajorFiniteDifference
        from parsec_python.acceleration.tests import test_stencil_pipeline as pipeline

        cp = self.cp
        for name in ("irregular", "box8", "holes", "box12"):
            metadata = pipeline.CASES[name]()
            wide = _wide(metadata)
            rows = int(metadata.shape[0])
            for setting in ("former", "pipelined"):
                for tile in (0, 16):
                    with pipeline._switch(setting):
                        narrow_operator = CuPyStencilMajorFiniteDifference(cp, metadata=metadata, implicit_tile=tile)
                        wide_operator = CuPyStencilMajorFiniteDifference(cp, metadata=wide, implicit_tile=tile)
                    self.assertEqual(narrow_operator.code_bytes, 1)
                    self.assertEqual(wide_operator.code_bytes, 2)
                    self.assertIn("uint16", wide_operator.storage_mode)
                    self.assertEqual(wide_operator.fast_path_slots, narrow_operator.fast_path_slots)
                    for index, kind in enumerate(("finite", "wide", "specials", "twonan")):
                        operands = pipeline.Operands(cp, rows, kind, 97 * index + tile + len(name))
                        with self.subTest(case=name, setting=setting, tile=tile, kind=kind):
                            self.check(pipeline.compare_recurrence_kernels(cp, narrow_operator, wide_operator, operands),
                                       f"{name} {setting} tile {tile} {kind} recurrence")
                            self.check(pipeline.compare_operator_methods(cp, narrow_operator, wide_operator, operands),
                                       f"{name} {setting} tile {tile} {kind} methods")
                        del operands

    def test_a_sector_of_295_values_against_its_csr(self) -> None:
        from parsec_python.acceleration.backends.cupy_stencil_major import CuPyStencilMajorFiniteDifference
        from parsec_python.acceleration.Eigensolvers.distributed_filter import replica_stencil
        from parsec_python.acceleration.tests import test_stencil_pipeline as pipeline

        cp = self.cp
        case = _Silicon.get()
        stencil = build_sector_stencils(case.grid, case.weighted, (3,), implementation="numpy", wrap=True)[3]
        self.assertEqual(stencil.coefficient_codes.dtype, np.uint16)
        matrix = stencil.to_csr()
        rows = int(matrix.shape[0])
        generator = np.random.default_rng(5)
        block = generator.standard_normal((rows, 14))
        expected = matrix @ block
        outputs = {}
        for setting in ("former", "pipelined"):
            for tile in (0, 16):
                with pipeline._switch(setting):
                    operator = CuPyStencilMajorFiniteDifference(cp, metadata=stencil, implicit_tile=tile)
                result = cp.asnumpy(operator.apply(cp.asarray(block, order="F")))
                scale = np.abs(matrix).max() * np.abs(block).max()
                self.assertLess(np.abs(result - expected).max(), 1.0e-13 * scale, (setting, tile))
                outputs[(setting, tile)] = result
                # a device that shares the sector builds it from the owner's arrays, codes as they are
                owner = SimpleNamespace(compact_finite_difference=operator, shape=stencil.shape,
                                        effective_potential=cp.zeros(rows))
                metadata, replica_tile = replica_stencil(owner)
                self.assertEqual(replica_tile, tile)
                self.assertEqual(np.dtype(metadata.coefficient_codes.dtype), np.uint16)
                with pipeline._switch(setting):
                    replica = CuPyStencilMajorFiniteDifference(cp, metadata=metadata, implicit_tile=tile)
                np.testing.assert_array_equal(cp.asnumpy(replica.apply(cp.asarray(block, order="F"))), result)
        for key, value in outputs.items():
            np.testing.assert_array_equal(value.view(np.uint64), outputs[("former", 0)].view(np.uint64), str(key))

    def test_the_fp32_recurrence_of_two_byte_codes(self) -> None:
        from parsec_python.acceleration.backends.cupy_mixed_precision import CuPyMixedPrecisionRecurrence
        from parsec_python.acceleration.tests import test_stencil_pipeline as pipeline

        cp = self.cp
        metadata = pipeline.CASES["box8"]()
        rows = int(metadata.shape[0])
        generator = np.random.default_rng(11)
        projectors = sp.random(rows, 3, density=0.01, random_state=3, format="csr", dtype=np.float64)
        projectors.indices = projectors.indices.astype(np.int32)
        projectors.indptr = projectors.indptr.astype(np.int32)
        potential = generator.uniform(-0.5, 0.5, rows)
        current = cp.asarray(generator.standard_normal((rows, 8)).astype(np.float32), order="F")
        previous = cp.asarray(generator.standard_normal((rows, 8)).astype(np.float32), order="F")
        results = []
        for host in (metadata, _wide(metadata)):
            stencil = SimpleNamespace(
                shape=host.shape, slot_count=int(host.neighbors.shape[0]), neighbors=cp.asarray(host.neighbors),
                coefficient_codes=cp.asarray(host.coefficient_codes),
                coefficient_palette=cp.asarray(host.coefficient_palette),
            )
            recurrence = CuPyMixedPrecisionRecurrence(cp, stencil, projectors, np.array([1.0, -1.0, 0.5]), potential)
            results.append(cp.asnumpy(recurrence(current, previous=previous, center=0.1, scale=0.7, sigma=0.3,
                                                 sigma_next=1.1)))
        np.testing.assert_array_equal(results[0].view(np.uint32), results[1].view(np.uint32))


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class TwoByteSCFTests(unittest.TestCase):
    def _run(self, problem, environment):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
        return result, dict(result.backend.details)

    def test_a_value_the_switch_does_not_know_stops_the_preparation(self) -> None:
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, {_SWITCH: "wide"}), self.assertRaises(ValueError) as caught:
            driver.prepare_single_point(silicon_supercell(1, 0.64), backend="cupy")
        self.assertIn(_SWITCH, str(caught.exception))

    def test_two_byte_codes_everywhere_give_the_same_bits(self) -> None:
        # Si 2x2x2 at Grid_Spacing 0.64 (32 points per edge: one orbit size, every sector within one byte):
        # PARSEC_SECTOR_PALETTE_CODES=on codes every sector stencil in two bytes, with the same results to the bit
        problem = silicon_supercell(2, 0.64)
        auto, auto_details = self._run(problem, {})
        on, on_details = self._run(problem, {_SWITCH: "on"})
        self.assertTrue(auto_details["orbital_sector_coefficient_codes"].startswith("one byte"))
        self.assertTrue(on_details["orbital_sector_coefficient_codes"].startswith("two bytes"))
        self.assertNotIn("two bytes", auto_details["symmetry_detection"])
        self.assertEqual(auto.iterations, on.iterations)
        for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
            self.assertEqual(getattr(auto.energies, field), getattr(on.energies, field), field)


if __name__ == "__main__":
    unittest.main()

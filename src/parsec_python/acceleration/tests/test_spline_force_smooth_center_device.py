"""The device sums of the local forces with the slopes of the smooth behaviour next to an atom
(``PARSEC_SPLINE_FORCE_CENTER=auto``; the readable rule in :mod:`parsec_python.tests.unit.test_spline_force_smooth_center`).

Every device route forms the slope ``s(c) d/c`` within ``c = 2 rs(2)`` of an atom as its readable twin does, and keeps
the forces of ``tip`` elsewhere.  Checked on a GPU, atoms on grid points (1e-11 bohr off them), within ``c`` of them
(carbon FHI 5e-4 bohr, ``c`` = 2.08e-3; Ge POTRE 1e-6 bohr, ``c`` = 1.56e-6; carbon POTRE 1e-5 bohr, ``c`` = 2.56e-5)
and farther, Ge (POTRE with a core density; linear and spline tables), carbon POTRE and carbon FHI tables:

* clusters: :func:`parsec_python.acceleration.Forces.local_device.cupy_local_forces` against
  :func:`parsec_python.Forces.local.local_forces` (also ``Old_Interpolation_Format``);
* the Fortran-derived routines of a cell: the CUDA kernel of linear species
  (:func:`parsec_python.acceleration.Forces.periodic_device.device_periodic_local_forces`) and the spline ball sums
  with their device density terms (:mod:`parsec_python.acceleration.Forces.periodic_spline`) against the readable
  :func:`parsec_python.Forces.periodic.periodic_local_forces`;
* the default routines: the batched (fused kernel) and per-atom device sums of
  :func:`parsec_python.acceleration.Forces.periodic_local.fast_periodic_local_forces` against the host;
* ``tip`` keeps the device sums of 2026-11-21 (their difference to ``auto`` lies in the atoms next to a grid point).
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces import periodic as fortran_sums
from parsec_python.Forces.local import local_forces
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import GridSettings
from parsec_python.V_ion import load_pseudopotentials, superpose_atomic_density

from ..backends.cupy import cupy_available

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_ROOT = Path(__file__).resolve().parents[4]
_GE = _DATA / "Ge_POTRE.DAT"
_C_POTRE = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_C_FHI = _ROOT / "examples" / "3d_hybrid_functionals" / "C_FHIPP.DAT"
_ENV = "PARSEC_SPLINE_FORCE_CENTER"
_DIRECTION = np.array([1.0, 0.7, 0.4]) / np.linalg.norm([1.0, 0.7, 0.4])
_CUBE = 6.3 * np.eye(3)


def _load(specifications):
    # the carbon FHI table as the file gives it (PARSEC_FHI_ORIGIN=off: first sample 1.04e-3 bohr), whose padded first
    # interval the rule reaches; under auto, the default, the reader continues it to 1e-5 bohr
    with mock.patch.dict(os.environ, {"PARSEC_FHI_ORIGIN": "off"}):
        return load_pseudopotentials(specifications, xc_functional="pbe", check_correlation=False)


def _near(grid, guess, offset):
    coordinates = np.asarray(grid.coordinates)
    return coordinates[int(np.argmin(np.linalg.norm(coordinates - np.asarray(guess), axis=1)))] + offset * _DIRECTION


def _fields(grid, scale=1.0):
    coordinates = np.asarray(grid.coordinates)
    density = 0.2 * np.exp(-0.05 * np.einsum("ij,ij->i", coordinates, coordinates) / scale) + 0.01
    residual = 1.0e-3 * np.sin(coordinates[:, 0]) * np.cos(coordinates[:, 1])
    xc = -0.5 * np.cbrt(density) + 0.02 * np.sin(coordinates[:, 2])
    return density, residual, xc


def _compare(test, first, second, atol, label):
    for name in ("ionic", "self_consistency", "core_correction"):
        np.testing.assert_allclose(getattr(first, name), getattr(second, name), rtol=0.0, atol=atol,
                                   err_msg=f"{name}: {label}")


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class ClusterDeviceTests(unittest.TestCase):
    def test_device_sums_equal_the_readable_rule(self) -> None:
        from parsec_python.acceleration.Forces.local_device import cupy_local_forces

        grid = build_cluster_grid(GridSettings(spacing=0.35, radius=5.0, expansion_order=8))
        for spline, direct in ((True, False), (False, False), (False, True)):
            specifications = {
                "Ge": replace(SpeciesPotential(_GE, 1, use_spline=spline), old_interpolation=direct),
                "C": SpeciesPotential(
                    _C_FHI, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
                    electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=1.6,
                ),
            }
            potentials = _load(specifications)
            atoms = (
                Atom("C", _near(grid, (0.3, -0.2, 0.1), 5.0e-4).tolist()),
                Atom("Ge", _near(grid, (-1.9, 1.4, 0.9), 1.0e-11).tolist()),
                Atom("Ge", _near(grid, (1.6, 1.9, -1.2), 1.0e-6).tolist()),
                Atom("C", (1.33, -2.07, 1.81)),
            )
            arguments = (grid, atoms, potentials, specifications, *_fields(grid))
            results = {}
            for setting in ("auto", "tip"):
                with mock.patch.dict(os.environ, {_ENV: setting}):
                    results[setting] = (local_forces(*arguments), cupy_local_forces(*arguments))
                _compare(self, results[setting][1], results[setting][0], 2.0e-12, f"{setting} spline={spline} direct={direct}")
            # the C FHI atom next to a grid point takes another slope there, and with the spline so do the potential and
            # the core density of the Ge atom 1e-6 bohr from one (within c = 2 rs(2) = 1.56e-6 bohr: 5e-8 Ry/bohr; the
            # linear slope tables of pseudo.f90 vanish over the first samples of that table, under either value); the
            # last atom is far from every point
            self.assertGreater(float(np.max(np.abs(results["auto"][0].ionic[0] - results["tip"][0].ionic[0]))), 1.0e-6)
            if spline:
                for term in ("ionic", "core_correction"):
                    change = getattr(results["auto"][1], term)[2] - getattr(results["tip"][1], term)[2]
                    self.assertGreater(float(np.max(np.abs(change))), 1.0e-9)
            np.testing.assert_array_equal(results["auto"][1].ionic[3], results["tip"][1].ionic[3])


def _cell(specification, symbol, fortran: bool, others=None, offset: float = 5.0e-4):
    """Three atoms: ``offset`` bohr from a grid point, 1e-11 bohr from one and far from every one; ``others`` (symbol,
    specification) gives the second and third atoms another species."""

    grid = build_periodic_grid(
        PeriodicCell(lattice_vectors=_CUBE), PeriodicGridSettings(spacing=0.45, expansion_order=6, fortran_routines=fortran)
    )
    species = {symbol: specification}
    second = symbol
    if others is not None:
        second = others[0]
        species[second] = others[1]
    potentials = _load(species)
    atoms = (
        Atom(symbol, _near(grid, (1.0, 1.2, 1.4), offset).tolist()),
        Atom(second, _near(grid, (4.1, 3.7, 2.2), 1.0e-11).tolist()),
        Atom(second, (2.61, 4.83, 5.07)),
    )
    specification = species
    density = superpose_atomic_density(grid, atoms, potentials, species, _CUBE)
    coordinates = np.asarray(grid.coordinates)
    residual = 1.0e-3 * np.sin(2.0 * np.pi * coordinates[:, 0] / 6.3)
    xc = -0.9 * np.cbrt(density) + 0.01 * np.cos(2.0 * np.pi * coordinates[:, 1] / 6.3)
    return grid, atoms, potentials, species, density, residual, xc


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class FortranRoutineDeviceTests(unittest.TestCase):
    @staticmethod
    def _readable(case):
        grid, atoms, potentials, specifications, density, residual, xc = case
        previous = fortran_sums._local_force_provider
        fortran_sums.set_local_force_provider(None)
        try:
            return fortran_sums.periodic_local_forces(
                grid, atoms, potentials, specifications, _CUBE, density, residual, xc
            )
        finally:
            fortran_sums.set_local_force_provider(previous)

    def test_the_kernel_of_linear_species(self) -> None:
        from parsec_python.acceleration.Forces.periodic_device import device_periodic_local_forces

        fhi = SpeciesPotential(
            _C_FHI, 2, pseudopotential_format="fhipp", number_of_channels=3, electrons_per_orbital=(2.0, 2.0, 0.0),
            core_cutoff_radius=1.6,
        )
        case = _cell(fhi, "C", True, others=("Ge", SpeciesPotential(_GE, 1)))
        grid, atoms, potentials, specifications, density, residual, xc = case
        results = {}
        for setting in ("auto", "tip"):
            with mock.patch.dict(os.environ, {_ENV: setting}):
                device = device_periodic_local_forces(grid, atoms, potentials, specifications, _CUBE, density, residual, xc)
                self.assertIsNotNone(device)
                results[setting] = (self._readable(case), device)
            # the readable sum reaches farther (the converged Ewald sum): 1e-7 as in test_periodic_device_forces
            _compare(self, results[setting][1], results[setting][0], 1.0e-7, setting)
        self.assertGreater(float(np.max(np.abs(results["auto"][1].ionic[0] - results["tip"][1].ionic[0]))), 1.0e-6)
        # the Ge atom 1e-11 bohr from a grid point: its core slope there is s(c) d/c instead of zero
        self.assertGreater(
            float(np.max(np.abs(results["auto"][1].core_correction[1] - results["tip"][1].core_correction[1]))), 0.0
        )
        np.testing.assert_array_equal(results["auto"][1].ionic[2], results["tip"][1].ionic[2])

    def test_the_spline_ball_sums_with_device_density_terms(self) -> None:
        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

        for specification, symbol in (
            (SpeciesPotential(_GE, 1, use_spline=True), "Ge"),
            (
                SpeciesPotential(
                    _C_FHI, 2, use_spline=True, pseudopotential_format="fhipp", number_of_channels=3,
                    electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=1.6,
                ),
                "C",
            ),
        ):
            case = _cell(specification, symbol, True)
            grid, atoms, potentials, specifications, density, residual, xc = case
            for setting in ("auto", "tip"):
                with self.subTest(symbol=symbol, setting=setting), mock.patch.dict(os.environ, {_ENV: setting}):
                    fast = spline_periodic_local_forces(grid, atoms, potentials, specifications, _CUBE, density, residual, xc)
                    self.assertIsNotNone(fast)
                    _compare(self, fast, self._readable(case), 2.0e-6, setting)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DefaultRoutineDeviceTests(unittest.TestCase):
    def test_batched_and_per_atom_device_sums_equal_the_host(self) -> None:
        import cupy as cp
        from cupyx.scipy.special import erf as device_erf

        from parsec_python.acceleration.Forces.periodic_local import fast_periodic_local_forces

        for spline in (True, False):
            # the carbon atom 1e-5 bohr from a grid point: within its first sample (1.28e-5 bohr, skipped under tip)
            # and within c = 2 rs(2)
            case = _cell(
                SpeciesPotential(_C_POTRE, 2, use_spline=spline), "C", False,
                others=("Ge", SpeciesPotential(_GE, 1, use_spline=spline)), offset=1.0e-5,
            )
            grid, atoms, potentials, specifications, density, _residual, xc = case
            arguments = (grid, atoms, potentials, specifications, _CUBE, density, xc)
            results = {}
            for setting in ("auto", "tip"):
                with mock.patch.dict(os.environ, {_ENV: setting}):
                    host = fast_periodic_local_forces(*arguments)
                    for balls in ("auto", "off"):
                        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_BALLS": balls}):
                            device = fast_periodic_local_forces(*arguments, xp=cp, erf=device_erf)
                        with self.subTest(spline=spline, setting=setting, balls=balls):
                            _compare(self, device, host, 1.0e-10, f"{setting} {balls}")
                results[setting] = host
            self.assertGreater(float(np.max(np.abs(results["auto"].ionic[0] - results["tip"].ionic[0]))), 1.0e-7)
            np.testing.assert_array_equal(results["auto"].ionic[2], results["tip"].ionic[2])


if __name__ == "__main__":
    unittest.main()

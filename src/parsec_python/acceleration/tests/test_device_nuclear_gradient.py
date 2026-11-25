"""PARSEC_GGA_NUCLEAR_GRADIENT on the device: the device GGA evaluators with the correction of the density gradient
near the nuclei against the host functionals in its scope (``parsec_python.V_xc.nuclear_gradient``), on the full grid
of a cluster, on the orbits of its reflection groups, and on the orbits of the point group of a cell (the correction
is a vector field that the operations carry, as the flux is); unpolarized and spin (both spin forms)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.models import Atom, GridSettings
from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
from parsec_python.V_xc import nuclear_gradient as ng
from parsec_python.V_xc.selection import evaluate_unpolarized_xc
from parsec_python.acceleration.backends.cupy import cupy_available

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
#: both sides evaluate the same float64 operations (test_device_xc.FIELD_TOLERANCE)
FIELD_TOLERANCE = 1.0e-13


def _relative(reference, value) -> float:
    reference = np.asarray(getattr(reference, "values", reference))
    value = np.asarray(getattr(value, "values", value))
    return float(np.max(np.abs(reference - value)) / max(float(np.max(np.abs(reference))), 1e-300))


def _system(grid, atoms, lattice=None):
    cell = None if lattice is None else SimpleNamespace(lattice_vectors=lattice, dimensions=3)
    return SimpleNamespace(
        grid=grid,
        atoms=tuple(atoms),
        pseudopotentials={"Au": read_parsec_pseudopotential(_AU_POTRE)},
        input=SimpleNamespace(periodic_cell=cell),
    )


def _density(grid, atoms, side=None):
    density = np.full(grid.size, 1.0e-4)
    for atom in atoms:
        difference = grid.coordinates - np.asarray(atom.position)
        if side is not None:
            difference -= np.round(difference / side) * side
        r2 = np.sum(difference * difference, axis=1)
        # a d-like shell with a hole at the nucleus
        density += 0.6 * r2 * r2 * np.exp(-1.6 * r2) + 0.05 * np.exp(-0.4 * r2)
    return density


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {ng.NUCLEAR_GRADIENT_ENV: "on"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _compare(self, system, reducer, density, label) -> None:
        from parsec_python.acceleration.V_xc import device_xc

        grid = system.grid
        correction = ng.system_correction(system)
        source = lambda: correction  # noqa: E731
        wedge = density if reducer is None else reducer.from_full(density)
        full = density if reducer is None else reducer.to_full(wedge)
        values = (lambda field: field) if reducer is None else reducer.wedge_values
        for functional in ("pbe", "pw91"):
            with ng.applied(system):
                reference = evaluate_unpolarized_xc(functional, full, grid, None)
            without = evaluate_unpolarized_xc(functional, full, grid, None)
            self.assertGreater(abs(reference.total_energy - without.total_energy), 1e-7, label)
            value = device_xc.DeviceXCEvaluator(functional, grid, reducer=reducer, nuclear_gradient=source)(wedge)
            self.assertLess(_relative(values(reference.potential), value.potential), FIELD_TOLERANCE, f"{functional} {label}")
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=FIELD_TOLERANCE * abs(reference.total_energy),
                msg=f"{functional} {label}",
            )
        # spin: pbe_spin (evaluate_spin_xc) and the s_ab form of the periodic spin drivers
        up_full, down_full = 0.6 * full, 0.4 * full
        up = up_full if reducer is None else reducer.from_full(up_full)
        down = down_full if reducer is None else reducer.from_full(down_full)
        from parsec_python.V_xc.pbe_spin import pbe_spin

        with ng.applied(system):
            reference = pbe_spin(up_full, down_full, grid)
        evaluator = device_xc.DeviceSpinXCEvaluator("pbe", grid, reducer=reducer, nuclear_gradient=source)
        value = evaluator(up, down)
        for field in ("potential_up", "potential_down"):
            self.assertLess(
                _relative(values(getattr(reference, field)), getattr(value, field)), FIELD_TOLERANCE, f"spin {label}"
            )
        self.assertAlmostEqual(
            value.total_energy, reference.total_energy, delta=FIELD_TOLERANCE * abs(reference.total_energy), msg=label
        )
        if reducer is None:
            from parsec_python.V_xc.pbe import pbe_spin_polarized

            with ng.applied(system):
                reference = pbe_spin_polarized(up_full, down_full, grid)
            evaluator.form = "ab"
            value = evaluator(up, down)
            self.assertLess(_relative(reference.potential_up, value.potential_up), 1e-12, f"ab {label}")
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=1e-12 * abs(reference.total_energy), msg=label
            )

    def test_cluster_full_grid(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=5.5, expansion_order=12))
        atoms = (Atom("Au", (0.11, -0.05, 0.07)), Atom("Au", (2.5, 1.9, -1.3)))
        self._compare(_system(grid, atoms), None, _density(grid, atoms), "cluster")

    def test_cluster_reflection_orbits(self) -> None:
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction

        for shift in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
            grid = build_cluster_grid(GridSettings(spacing=0.4, radius=5.5, expansion_order=8, shift=shift))
            atoms = (Atom("Au", (2.4, 0.0, 0.0)), Atom("Au", (-2.4, 0.0, 0.0)), Atom("Au", (0.0, 1.9, 0.0)),
                     Atom("Au", (0.0, -1.9, 0.0)))
            reduction = AxisReflectionReduction.detect(grid, atoms)
            self.assertEqual(reduction.group_order, 8)
            self._compare(_system(grid, atoms), SymmetrySCFReducer(reduction), _density(grid, atoms), f"D2h {shift}")

    def test_cell_point_group_orbits(self) -> None:
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        side = 7.6
        fcc = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
        atoms = [Atom("Au", tuple(side * np.array(b))) for b in fcc]
        for shift in (0.0, 0.5):
            grid = build_periodic_grid(
                PeriodicCell(np.eye(3) * side),
                PeriodicGridSettings(spacing=side / 20, expansion_order=12, shift=(shift,) * 3),
            )
            system = _system(grid, atoms, np.eye(3) * side)
            density = _density(grid, atoms, side)
            reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
            self.assertGreater(reduction.group_order, 8)
            for sub in (reduction, SignedPermutationReduction.detect(grid, atoms)):
                self._compare(system, SymmetrySCFReducer(sub), density, f"fcc order {sub.group_order} shift {shift}")
            self._compare(system, None, density, f"fcc full grid shift {shift}")


if __name__ == "__main__":
    unittest.main()

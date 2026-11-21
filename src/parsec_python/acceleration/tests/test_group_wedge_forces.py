"""Local forces of a cluster in weighted sectors summed over one point per orbit of the full group.

``PARSEC_LOCAL_FORCE_SUMS=orbits`` (Forces/local_device.py, :func:`group_wedge`): the orbits of the full group ``G``
of a :class:`PointGroupReduction` from the coset images of the orbits of its subgroup ``H`` (CPU), and the forces of
those sums against the sums over the whole grid and over the wedge of ``H`` (GPU).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings


def _tetrahedral(centre: str, ligand: str, a: float):
    return [Atom(centre, (0.0, 0.0, 0.0))] + [
        Atom(ligand, tuple(a * np.array(signs, dtype=float)))
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    ]


def _reduction(grid, atoms):
    from parsec_python.acceleration.Symmetry import (
        PointGroupReduction,
        SignedPermutationReduction,
        detect_point_group_reduction,
    )

    abelian = SignedPermutationReduction.detect(grid, atoms)
    reduction = detect_point_group_reduction(grid, atoms, abelian)
    assert isinstance(reduction, PointGroupReduction), type(reduction)
    return reduction


class GroupWedgeTests(unittest.TestCase):
    def test_the_orbits_of_the_full_group(self) -> None:
        from parsec_python.acceleration.Forces.local_device import group_wedge
        from parsec_python.acceleration.Symmetry.axis_reflection import _grid_row_mapping_operation

        for shift in (0.0, 0.5):
            with self.subTest(shift=shift):
                grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.6, expansion_order=4, shift=(shift,) * 3))
                reduction = _reduction(grid, _tetrahedral("C", "H", 1.2))
                self.assertEqual(np.asarray(reduction.full_operations).shape[0], 24)
                members, sizes = group_wedge(reduction)
                # the orbits of G by brute force: the smallest row reached by the operations of G
                maps = np.stack([
                    _grid_row_mapping_operation(grid, np.asarray(operation, dtype=np.int8), 1.0e-10)
                    for operation in reduction.full_operations
                ])
                smallest = maps.min(axis=0)
                labels, counts = np.unique(smallest, return_counts=True)
                self.assertEqual(members.size, labels.size)
                self.assertEqual(float(sizes.sum()), float(grid.size))
                rows = np.asarray(reduction.representative_rows)[members]
                # each member stands for its own orbit of G, with that orbit's size
                np.testing.assert_array_equal(np.sort(smallest[rows]), labels)
                order = np.argsort(smallest[rows])
                np.testing.assert_array_equal(sizes[order], counts.astype(float))

    def test_other_reductions_keep_their_wedge(self) -> None:
        from parsec_python.acceleration.Forces.local_device import group_wedge
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, expansion_order=4))
        atoms = [Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.9, 0.0, 0.6)), Atom("H", (-0.9, 0.0, 0.6))]
        self.assertIsNone(group_wedge(SignedPermutationReduction.detect(grid, atoms)))

    def test_the_switch(self) -> None:
        from parsec_python.acceleration.Forces.local_device import local_force_sums_setting

        for value, expected in (("", "wedge"), ("orbits", "orbits"), ("WEDGE", "wedge"), ("full", "full")):
            with mock.patch.dict(os.environ, {"PARSEC_LOCAL_FORCE_SUMS": value}):
                self.assertEqual(local_force_sums_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_LOCAL_FORCE_SUMS": "group"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_LOCAL_FORCE_SUMS"):
                local_force_sums_setting()


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # pragma: no cover - no CuPy
        return False


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class GroupWedgeForceTests(unittest.TestCase):
    def test_the_sums_of_the_full_group_equal_those_over_the_grid(self) -> None:
        from parsec_python.V_ion import load_pseudopotentials
        from parsec_python.acceleration.Forces.local_device import cupy_local_forces, cupy_wedge_local_forces
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.models import SpeciesPotential
        from parsec_python.tests.force_reference import DATA

        specifications = {
            "H": SpeciesPotential(DATA / "water_H_POTRE.DAT", 0),
            "Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 1, use_spline=True),
        }
        potentials = load_pseudopotentials(specifications)
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=6.0, expansion_order=6))
        # Td: Ge at the centre, H on the diagonals, and a Ge4 tetrahedron farther out (two orbits of 4)
        atoms = _tetrahedral("Ge", "H", 1.6) + [
            Atom("Ge", tuple(-2.6 * np.array(signs, dtype=float)))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        reduction = _reduction(grid, atoms)
        coordinates = grid.coordinates
        rng = np.random.default_rng(5)
        raw = (
            0.2 * np.exp(-0.1 * np.einsum("ij,ij->i", coordinates, coordinates)) * (1.0 + 0.3 * rng.random(grid.size)),
            0.01 * rng.normal(size=grid.size),
            -0.5 * rng.random(grid.size),
        )
        # fields invariant under G: averaged over the orbits of H, then over the cosets
        wedge = [
            reduction.symmetrize_values(
                np.bincount(reduction.full_to_wedge, weights=field) / reduction.multiplicities
            )
            for field in raw
        ]
        fields = [values[reduction.full_to_wedge] for values in wedge]
        full = cupy_local_forces(grid, atoms, potentials, specifications, *fields)
        self.assertGreater(np.abs(full.ionic).max(), 1e-3)
        subgroup = cupy_wedge_local_forces(grid, atoms, potentials, specifications, *fields, reduction)
        for given in (fields, [SymmetryScalarField(reduction, values) for values in wedge]):
            group = cupy_wedge_local_forces(
                grid, atoms, potentials, specifications, *given, reduction, device_ids=(0, 0), full_group=True
            )
            for name in ("ionic", "self_consistency", "core_correction"):
                np.testing.assert_allclose(getattr(group, name), getattr(full, name), rtol=0, atol=5e-12, err_msg=name)
                np.testing.assert_allclose(
                    getattr(group, name), getattr(subgroup, name), rtol=0, atol=5e-12, err_msg=name
                )


if __name__ == "__main__":
    unittest.main()

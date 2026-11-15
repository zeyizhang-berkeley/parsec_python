"""Actual complex k-point/spin SCF hooks against an independent ARPACK solve."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import parsec_python
from parsec_python.models import (Atom, EigensolverSettings, PeriodicCell,
    PeriodicGridSettings, SCFSettings, SinglePointInput, SpeciesPotential)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from parsec_python.SCF.single_point import SCFWarmStart
from parsec_python.SCF.restart_refinement import RestartRefinementSettings

potential = Path(parsec_python.__file__).parent / 'tests/data/H_POTRE.DAT'
kpoints = np.array([[.13, .07, .0], [-.13, -.07, .0]])
reports = []
for spin in (False, True):
    problem = SinglePointInput(
        atoms=[Atom('H', [2., 2., 2.])],
        pseudopotentials={'H': SpeciesPotential(potential, 0)},
        grid=PeriodicGridSettings(spacing=1., expansion_order=2),
        periodic_cell=PeriodicCell(4. * np.eye(3)),
        scf=SCFSettings(max_iterations=1, number_of_states=3, spin_polarized=spin),
        eigensolver=EigensolverSettings(method='chebff', first_filter_cycles=1,
            first_filter_degree=10, tolerance=1e-9))
    reference = replace(problem, eigensolver=EigensolverSettings(
        method='arpack', subspace_buffer=0, matvec_block_size=1, tolerance=1e-11))
    outputs = {}
    for rule in ('arpack', 'inherit', 'smart'):
        source = reference if rule == 'arpack' else problem
        policy = None if rule == 'arpack' else RestartRefinementSettings(1e-9, lower_bound_rule=rule)
        system = prepare_periodic_single_point(source)
        v = system.ionic_potential.copy()
        warm = SCFWarmStart(input_potential=(v.copy(), v.copy()) if spin else v,
                            hartree_potential=-v, restart_refinement=policy)
        solve = run_scf_kpoints_spin_polarized if spin else run_scf_kpoints
        outputs[rule] = solve(system, kpoints, np.array([.5, .5]), warm_start=warm)
    old = outputs['arpack']
    for rule in ('inherit', 'smart'):
        new = outputs[rule]
        if spin:
            level_error = max(float(np.max(np.abs(getattr(new, n) - getattr(old, n))))
                              for n in ('eigenvalues_up', 'eigenvalues_down'))
            density_error = max(float(np.max(np.abs(getattr(new, n) - getattr(old, n))))
                                for n in ('density_up', 'density_down'))
            density = new.density_up + new.density_down
        else:
            level_error = float(np.max(np.abs(new.eigenvalues - old.eigenvalues)))
            density_error = float(np.max(np.abs(new.density - old.density)))
            density = new.density
        charge = float(np.sum(density) * system.grid.volume_element)
        assert level_error < 1e-7 and density_error < 2e-7 and abs(charge - 1.) < 1e-9
        reports.append(dict(spin_polarized=spin, lower_bound_rule=rule,
            level_error_ry=level_error, density_max_error=density_error, electrons=charge))
output = dict(passed=True, scope='CPU physical complex-k-point check, not large GPU validation', comparisons=reports)
Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))

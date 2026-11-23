# Forces, structural relaxation and molecular dynamics

This document describes what moves the atoms of an isolated system or a periodic cell in
`parsec_python`: the atomic forces, the three minimization schemes of PARSEC
(`Minimization: bfgs | simple | manual`) and its molecular dynamics. It says
how to run them, which Fortran routine each part follows, where the package
deliberately differs, and what was compared with Fortran PARSEC.

Units are PARSEC's throughout: bohr, Rydberg, Ry/bohr.

## Scope

- Isolated systems (`Boundary_Conditions: cluster`), sphere or box domain,
  CA/PZ LDA or PBE, without or with spin polarization, with or without
  nonlinear core correction, projectors through `l = 3`.
- Self-consistent hybrid functionals of isolated systems (PBE0, HSE06,
  B3LYP, HF, ...), without or with spin polarization: forces, relaxation and
  molecular dynamics on both command lines (`main.py` on the GPUs of one
  node) and on several nodes (the MPI runner of
  [acceleration/MULTI_GPU.md](acceleration/MULTI_GPU.md#hybrid-functionals-on-several-nodes));
  of periodic cells (bulk at the Gamma point, also with spin, and with
  k-points; slabs) on the readable solver and on the GPUs of one node; see
  [Hybrid functionals](#hybrid-functionals) below.
- Periodic cells (`Boundary_Conditions: bulk`), without spin-orbit coupling, at the Gamma point or with a Monkhorst-Pack
  k-point sampling (collinear spin polarization at the Gamma point only): forces, `Minimization: bfgs | simple | manual`
  and molecular dynamics. See "Periodic cells" below; the spin-unpolarized Gamma-point cell also runs on the accelerated
  path, and a cell with k-points on the GPUs of the k-point solver (see "On GPUs"). A spin-polarized one with k-points
  does not have them on these routines; such an input runs on the Fortran-derived routines by itself, with a warning.
- Self-consistent spin-orbit coupling (`SO_from_scratch`, alone or with `Spin_Polarization`) and non-collinear magnetism
  (`Non_Collinear_magnetism`, with or without spin-orbit), isolated or periodic at the Gamma point or with k-points:
  forces, relaxation and molecular dynamics. See "Spin-orbit coupling and non-collinear magnetism" below. The
  perturbative spin-orbit correction (`SO_PSP` without `SO_from_scratch`) has none.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - Periodic (bulk) cells with a fixed cell: forces, relaxation and molecular
    dynamics at Gamma and with k-points (also with collinear spin, with an
    irreducible k-point wedge and with `Double_Grid_Order > 1`), on the
    orthorhombic and on Fortran's general grid; see
    [Periodic cells](#periodic-cells).
  - Wires and slabs (`Boundary_Conditions: wire` / `slab`): forces (Gamma,
    collinear spin, k-points with any weights, spin with k-points), relaxation
    and molecular dynamics; see
    [Wires and slabs](#wires-and-slabs). Spin-orbit coupling, non-collinear
    magnetism and `Double_Grid_Order > 1` are refused for wires and slabs.
  - Spin-orbit coupling (`SO_from_scratch`/`SCF_SO`, also with collinear spin)
    and non-collinear magnetism, isolated or periodic (bulk); see
    [Spin-orbit coupling and non-collinear magnetism](#spin-orbit-coupling-and-non-collinear-magnetism).
    Refused: forces with the perturbative spin-orbit correction (`SO_PSP`
    without `SO_from_scratch`).
- A spin-polarized calculation uses the same routines: `forcloc.f90` reads
  the total density and the averages over the two channels of the potential
  difference and of the exchange-correlation potential, and `forcnloc.f90`
  sums over the states of both channels with the weight `4/nspin`.
- Not translated from `forcloc.f90` and `forcnloc.f90`: point charges,
  LDA+U and van der Waals corrections.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - The spin-orbit term of `forcnloc.f90` (`forcso`) is in
    `Forces/spin_orbit.py` (section below); the point-charge, LDA+U and TS
    terms are those of `Forces/total.py::correction_forces` and
    `Forces/ion_ion.py`.
- Every execution path computes the same numbers: the readable NumPy path
  (`reference_main.py`), the accelerated single-node path (`main.py`, CPU or
  GPUs) and the multi-node GPU runner
  (`acceleration/benchmarks/mpi_full_scf.py`).

## Running a calculation

### Input keywords

| Keyword | Default | Meaning |
|---|---|---|
| `Skip_Force` | absent | `false` adds the forces to a single point. A relaxation or dynamics run computes them regardless. A single point without the line computes none, so existing inputs keep their cost. |
| `Minimization` | `none` | `bfgs` (limited-memory BFGS), `simple` (steepest descent), `manual` (structures from `manual.dat`). |
| `Movement_Num` | 500 | Largest number of movements `M`: at most `M + 1` structures are computed. |
| `Force_Min` | 0.01 Ry/bohr | The run ends when the largest force on a movable atom is below it. |
| `Max_Step` | 0.05 bohr | `simple`: largest displacement of a coordinate per movement. `bfgs`: when given, every coordinate stays within it of its initial value for the whole run; absent, BFGS is unbounded. |
| `Min_Step` | 1e-4 bohr | `simple`: displacement of a coordinate whose force equals `Force_Min`. |
| `BFGS_number_corr` | 7 | Corrections the limited-memory matrix keeps. |
| `Relax_Restart` | `false` | Read `relax_restart.dat` beside the input: its coordinates replace those of the input and its BFGS block continues the optimizer. |
| `Write_Relax_Restart` | `true` here | PARSEC 1.5: write `relax_restart.dat` after every movement. 1.5 writes it only with `true` (its default is `false`); this package writes it unless the input says `false`. |
| `Calculate_Force` | not given | PARSEC 1.5: `true` gives the forces of a single point (as `Skip_Force: false`), `false` none. Without either keyword a single point computes no forces here (PARSEC computes them in every run). `false` with a relaxation or molecular dynamics is refused. `Use_Old_Forces` (1.5's serial force routines, the same forces) is accepted and changes nothing. |
| `Recenter_Atoms_In_Aperiodic_Directions` | `true` | PARSEC 1.5: `false` keeps the coordinates of the input (no re-centring of a cluster or, along z, of a slab; also in a molecular dynamics restart). Since 1.5 a slab is re-centred in a `Restart_Run` too. |
| `Move_Flag` (per `Atom_Type`) | 0 | 0 moves every atom of the type, `n > 0` its first `n` atoms, a negative value reads a fourth column of `Atom_Coord` (1 moves, 0 fixes). |
| `Molecular_Dynamics` | `false` | Switches molecular dynamics on; it excludes `Minimization`. |
| `Cooling_Method`, `Tinit`, `T_Final`, `T_Step`, `Time_Step`, `Friction_Coefficient`, `Step_num`, `Restart_mode`, `rng_seed` (PARSEC 1.5: `Mol_RNG_Seed`), `test_rng` | PARSEC's | Controls of `moldyn.f90`; see "Molecular dynamics" below. |
| `Initial_Structure` | `input` | Not a PARSEC keyword. `mlff` relaxes the input structure with a machine-learned force field (UMA, MACE, `MLFF_Model`) before the first SCF loop; see [MLFF/README.md](MLFF/README.md). |
| `Potential_Extrapolation` | `none` | Not a PARSEC keyword. How the SCF loop of a moved structure starts: `none` as PARSEC, `atomic` with the superposed atomic densities moved along with the atoms, `second_order` with the density change also extrapolated along the trajectory; see "Start of the next SCF loop" below. |
| `Relaxation_SCF_Upscale` | 1 | Not a PARSEC keyword. `bfgs` and `simple` only: the SCF loop of a structure may stop at up to this many times `Convergence_Criterion` while the forces are large; 1 keeps the criterion fixed as PARSEC does. See "SCF criterion of a relaxation". |

### Command lines

```powershell
python src\parsec_python\main.py calculation\parsec.in
python src\parsec_python\main.py calculation\parsec.in --restart-from previous\relax_restart.dat
python src\parsec_python\reference_main.py calculation\parsec.in
```

`--restart-from FILE` starts from the structure of another run without
editing the input: a `relax_restart.dat` written by this package or by
Fortran PARSEC (its BFGS block continues the optimizer at the coordinates of
the file), or an XYZ file of any program, whose last frame gives the
coordinates in Angstrom; they are recentred like those of an input and the
optimizer starts afresh. The file must list the atoms in the order of the
input, type by type; another order is refused. On several GPU nodes the launch script of the
repository takes the same input:

```bash
scripts/run_multi_gpu_scf.sh calculation/parsec.in output_directory
```

The structure of a restart (`Relax_Restart`, `--restart-from`, the
`Restart_mode` of molecular dynamics) also decides the symmetry of a periodic
cell, as in PARSEC 1.5, which reads `relax_restart.dat` and `mdinit.dat` in
`initial.f90` before `symmetries.f90` and `irrbz.f90` look for the group: the
input reader finds the space group, the irreducible k-point wedge and its
weights, the density average and, on the Fortran-derived routines, the
fractional translations of the grid again from the restarted atoms
(`Input/restarted_symmetry.py`, on the routines of the input), and
`parsec.out` says so in a NOTE where they differ from the input's. A restart
whose atoms keep the group of the input runs exactly as before. Where the
reader refuses the input with the restarted atoms (for example a `Move_Flag`
that the wedge of their group cannot follow) or the default routines do not
compute it, the run stops with the reason. Before 2026-11-20 a cell kept the
group of the input's atoms: the 8-atom silicon cube with a 2x2x2 mesh and one
atom moved 0.1 bohr came out 8.9 mRy per cell too low (13.0 mRy for the
64-atom cube), on either set of routines, and its forces stopped (an
operation without an image of an atom). Clusters were not affected: their
point group is found from the atoms they solve, as are the sectors of a cell
at the Gamma point and of its k-points. `Restart_Run` keeps the atoms of the
input (`parsec.dat` stores none).

The exit status is 0 for a finished run, 3 when an SCF loop did not converge
and 4 when the structure is not relaxed (movements used up, or the optimizer
stopped above `Force_Min`).  As in PARSEC 1.5, an SCF loop that does not
converge ends a relaxation or molecular dynamics run: the forces of that
structure are computed and reported, the atoms are not moved from it, and the
log says " WARNING !!! atom movement is terminated" (the previous Fortran tree,
and this package before, moved on).

Two inputs of Fortran PARSEC are kept as runnable examples with their Fortran
trajectories: `examples/0d_water_geoopt` (BFGS) and
`examples/0d_SiNa4_relaxation` (BFGS with an old-format pseudopotential and
occupations from `occup.in`).

### Files

A relaxation writes beside the input (the multi-node runner: into its output
directory):

| File | Content |
|---|---|
| `parsec.out` | Every SCF loop, the force tables of every structure, the lines of the mover, and at the end a summary with the wall time of the parts of the ionic steps. |
| `atom.cor` | PARSEC's file: every structure with its energy per atom and largest force. |
| `relax_restart.dat` | PARSEC's file, rewritten after every movement: the next structure and, for BFGS, the complete optimizer state. It is exchanged with Fortran PARSEC in both directions. |
| `iterate.dat` | The iteration table of L-BFGS-B, as PARSEC writes it. |
| `relaxation.xyz` | Extended XYZ: every structure in Angstrom with its energy (eV and Ry) and forces (eV/Angstrom). |
| `parsec_python_results.npz` | Unless `--no-archive`: the result of the last structure and, under `relaxation_*`, positions, energies and forces of all of them. |

With `PARSEC_STANDARD_ORIENTATION=on` (opt-in; acceleration README, "Standard
orientation of a cluster") a cluster whose axes are not along the coordinate
axes is calculated in the frame of its largest group of signed permutations.
`parsec.out`, `atom.cor`, `relaxation.xyz` and the forces and final positions
of the MPI runner's JSON report then give the coordinates and forces in the
frame of the input (as PARSEC places it), so that they compare with a run
without the switch; `relax_restart.dat` and the archive stay in the
calculation frame, which parsec.out prints (`x_calc = Q^T (x - d)`).
`relax_restart.dat` records that frame after all that PARSEC reads: a
`Relax_Restart` from it reports in the frame of the run that wrote it (also
with `Restart_Run`, which does not recentre) and needs the switch on, and a
file written without the switch is continued in the frame of the input. In a
relaxation with `Move_Flag` the search takes only frames whose group the
relaxation can use (no operation of the species exchanges a fixed atom with a
moving one).

Molecular dynamics writes PARSEC's `md_nrg.dat`, `md_mech.dat`, `mdinit.dat`,
`atom.cor` and `movie.xyz`, in the formats of PARSEC 1.5: integer fields `i6`
for the steps and `i7` for the atom count, and the velocities of
`md_mech.dat` (and so of an `mdinit.dat` cut from it) are `v(t)` of the step,
where the previous Fortran tree wrote its estimate of the next step's velocity.

### From Python

```python
from parsec_python import ionic_backend, prepare_single_point, run_relaxation
from parsec_python.Input import parse_parsec_input

translation = parse_parsec_input("calculation/parsec.in")
system = prepare_single_point(translation.problem)
outcome = run_relaxation(
    system,
    translation.relaxation,
    ionic_backend(),
    movable=translation.movable,
)
print(outcome.records[-1].energy, outcome.records[-1].maximum_force)
```

These are the accelerated workflow functions, as elsewhere at the top level
of the package; `prepare_reference_single_point` with
`reference_ionic_backend()` run the readable path. `compute_forces(system,
result)` and `compute_reference_forces(system, result)` give the forces of a
converged single point. A system prepared with symmetry holds for structures
of its point group: moving the atoms off it stops with an error, so a
relaxation whose movable flags break the symmetry is prepared with
`symmetry_fixed_atoms=symmetry_fixed_atoms(translation)` (the fixed atoms a
species of their own for the detection: the subgroup the relaxation keeps),
as the command line does under `PARSEC_MOVE_FLAG_SUBGROUP=auto`, or with
`symmetry="off"` (under `off`).

## Forces

The order is that of `parsec.F90`.

1. **Ion-ion** (`forceion.f90`, `Forces/ion_ion.py`):
   `F_a = sum_b 2 Z_a Z_b (R_a - R_b) / |R_a - R_b|^3`.
2. **Local pseudopotential** (`forcloc.f90`, `Forces/local.py`): three grid
   sums with `d = r - R_a`,
   - ionic: `integral rho(r) V_local'(|d|) d/|d|`;
   - SCF correction: `integral (V_out - V_in)(r) rho_atom'(|d|) d/|d|`, which
     removes the first-order error of an SCF loop that stopped at its
     criterion;
   - core correction: `integral V_xc(r) rho_core'(|d|) d/|d|` for species with
     a nonlinear core correction.
3. **Nonlocal projectors** (`forcnloc.f90`, `Forces/nonlocal_force.py`):
   `F_a = 4 sum_n f_n sum_(q on a) s_q <beta_q|psi_n> <grad beta_q|psi_n>`.
4. **Symmetrization** (`symm_forces.f90`, `Forces/symmetry.py`) over the
   cubic point-group operations (signed permutations of the axes) that map
   the atoms onto themselves, then removal of the net force.

The radial derivatives are the tables of `pseudo.f90`
(`Pseudopotential/radial_derivatives.py`): an 18th-order central difference
on the logarithmic grid, the local potential tabulated as `r^2 dV/dr` with
the tail `2Z/r^2`, and the projector derivative held constant near the
origin. Grid points closer to an atom than its first radial sample are left
out, as in PARSEC, under `PARSEC_SPLINE_FORCE_CENTER=tip`. The default `auto`
(since 2026-11-22) gives every grid point closer than `c = 2 rs(2)` (twice the
first radial sample) to its atom the slope of the smooth behaviour of the
local potential and of the core density, `s(c) d/c` with `s` the slope the
routine takes at `c` (a deliberate departure; the energy is unchanged): below
its first sample a table has no data, the padded spline has a slope of its own
at `r = 0+` (0.42 Ry/bohr for carbon FHI, first sample 1.04e-3 bohr), and a
step of 1e-3 bohr or more averages that interval out and follows
`V(0) + a r^2`, whose slope `forcloc.f90`'s skip misses. Tables sampled down
to the origin (POTRE, first samples 4e-7 to 8e-5 bohr) keep their own slope
beyond `2 rs(2)`, and tables whose first sample lies at or beyond 3e-3 bohr
(H and He FHI, whose padded interval the steps resolve) keep `tip`. A CH4
cluster with the carbon FHI table, carbon 1e-4, 5e-4 and 1e-3 bohr from a grid
point, off the slope of the SCF energy at steps of 1e-3 / 2e-3 bohr: `tip`
1.2e-4/4.3e-5, 1.7e-4/2.1e-4 and 3.5e-4/4.3e-4 Ry/bohr, `auto` 8.3e-5/6.3e-6,
7.5e-5/3.3e-5 and 1.4e-4/6.6e-5 (A100). What remains is nonlocal; the section
on cells below gives the measurements.

Two properties of PARSEC's tables limit how closely a force follows the
slope of the energy over finite steps. Without `Cubic_Spline` (PARSEC 1.5's
default) the energy interpolates `r V_local` and the projectors linearly
between the radial samples (`ionpot.f90`, `nonloc.F90`), while the forces
take the smooth derivative tables above: the energy has a kink wherever a
grid point crosses a radial sample. At h 0.3 bohr its slope over intervals
of 1e-4 bohr jumps at such positions by up to 1e-3 Ry/bohr (H) and 4e-2
(O), more for an atom on a symmetry line, which moves several equivalent
points at once. A100, one atom of a dimer moved by 1e-3 bohr, tight SCF:
the slope differs from the force by 5.6e-4 Ry/bohr (H2 at h 0.3), 3.3e-2
(O2) and 9.7e-2 (Cr2), and on a Si8 cell at h 0.49 by 6.5e-4; the exact
derivative of the interpolated energy misses those slopes by as much (H2
4.6e-4, O2 2.8e-2, local term at fixed density), because the kinks are
closer together than the steps, and tighter SCF and eigensolver tolerances
change nothing. With `Cubic_Spline: .true.` the same cases agree within
2.2e-5 (H2; 5.6e-6 extrapolated from the two steps), 2.2e-5 (O2: the O
projector is zero at the edge of its support but its spline is not flat
there, a kink of that size) and 3.6e-6 (Si8). The second property is that
edge: the support of the projectors ends at `rcore`, the first radial sample
beyond the largest channel cutoff, where most tables converted from UPF
files still hold a value, so that the energy jumps when a grid point
crosses it (Cr2 with `Cubic_Spline` 6.7e-4 Ry/bohr, the diamond cell of
eight atoms at h 0.37 with the carbon table of the diamond examples
7.4e-5). `PARSEC_PROJECTOR_TAIL=auto` (opt-in) sets those tails to zero, as
the UPF files have them: Cr2 3.2e-6, the diamond cell 8.9e-6 (5e-7
extrapolated). A check of forces against differences of the energy
therefore needs `Cubic_Spline` species and, with converted tables, that
switch or steps over which no grid point crosses a support edge. On the
Fortran-derived routines it also needs one grid for all its inputs: their
grid rule depends on the structure, and an Si8 cell with one atom moved by
1e-3 bohr along x or z ran on Fortran's general grid (3.3e-4 Ry above the
unmoved cell), along y on the grid of the unmoved cell, so that a
difference between separate inputs mixed two grids; an ionic loop keeps
the grid of its first structure. A
spin-polarized run adds no force term of its own here (the core-correction
term takes the mean of the two exchange-correlation potentials, as
`forcloc.f90`), but a structure moved off a symmetric start may leave the
state of that start: Fe2 (FM, 3.82 bohr) moved by 1e-3 bohr breaks its
inversion symmetry and lands 0.038 Ry lower, so differences against the
symmetric run are not its force; started in that broken-symmetry state
(unequal starting moments) it agrees within the step truncation, as do
Fe2 AFM and non-magnetic Fe2 (Richardson values 2.7e-6, 3.2e-6 and 3.0e-6
Ry/bohr, `Cubic_Spline`).

**Linear tables.** A species without `Cubic_Spline` (PARSEC's default)
samples its local potential (`r V`), its core density and its projectors by
linear interpolation between the radial samples, 1 to 2.5 % of the radius
apart. Its energy therefore changes slope wherever a grid point crosses a
sample, and is rough on the scale of a relaxation step, while the slope
tables are the derivative of the smooth table. So the forces are close to
those of the same tables with `Cubic_Spline` but are not the slope of the
energy the run evaluates. Full SCF on A100 (criterion 1e-8 Ry, `E - TS`,
central differences along unit vectors over all coordinates, steps of 1e-3
and 2e-3 bohr): the forces miss the slope by 2e-3 to 1e-2 Ry/bohr on
Si28H36 (`examples/0d_Si28H36`), CH4 with FHI tables, GeH4, NH4 at h = 0.4
bohr and a silicon cell of 216 atoms, by 4e-2 to 9e-2 on H2O and Au2, by 1e-3
on MgO and Si cells of 64 atoms with a 3x3x3 mesh (1e-2 on MgO8 with an atom
elsewhere), by 3e-2 to 6e-2 on Ge slabs and 0.11 on a Ge64 cell (the core
correction, default routines), and by up to 0.45 Ry/bohr on
an Ag4 cluster at h = 0.45 bohr, whose slopes at the two steps differ by a
factor of five. With `Cubic_Spline: .true.` the same inputs give the slope
within 1e-6 to 3e-4 Ry/bohr (2e-5 on the Ge slabs), and the forces of the two settings differ by
9e-6 to 3e-3 Ry/bohr: it is the energy of the linear tables that is rough.
It is offset as well, by an amount that depends on the species, the table
format and the positions: the spline energy differs from it by -5e-4 to
+2.5e-4 Ry per atom for the C, Si, N, O and H POTRE tables of the clusters and
the cells measured, by -1e-5 to +1.4e-4 for MgO (the Mg table carries a core
correction; the value depends on the positions), by +9.4e-4 to +1.3e-3 for Ge
with its core correction (cells of 8 and 64 atoms, slabs), by -1.7e-4 to
-5.3e-4 for carbon FHI tables and -2.1e-3 for silicon FHI tables (the FHI
grids are twice as coarse), and by +9e-3 to +1.2e-2 for Au and Ag. The spline
is not free everywhere. First runs with forces on one A100 each, the linear
and the spline run started together (spline / linear wall time): a core
correction superposed with the spline costs its preparation (the core density
of a Ge8 slab 0.011 -> 0.92 s, of Ge64 cells 0.002 -> 0.7-0.9 s, of MgO64 0.002 ->
0.59 s, with the block superposition of `PARSEC_SPLINE_SETUP`) and, on slabs
and the Fortran-derived routines, a slower force stage (Ge8 slab 0.21 -> 1.52 s,
Ge64 0.32 -> 2.10 s): Ge8 slab 1.89 (6.3 -> 12.0 s), Ge64 on the Fortran-derived
routines 1.45 (7.1 -> 10.3 s), Ge64 on the default routines 1.06, Ge2 slab
1.03, MgO64 1.04; a silicon cell of 216 atoms 1.30 (8.1 -> 10.6 s: three more SCF
iterations, and its forces take the Fortran-derived routines, the parser's
choice for `Cubic_Spline`); the other 14 cases 0.89 to 1.10. PARSEC 1.5 shows the same mismatch
(its own grid, 8 MPI ranks, energies printed to 1e-8 Ry: the breathing mode of
GeH4 1.3e-2 and 1.2e-2 Ry/bohr, of NH4 2.6e-3 and 3.0e-3; with `Cubic_Spline`
4e-6 and 2e-6, 4e-5 and 1.6e-4). `PARSEC_LINEAR_TABLE_SLOPES=interpolant` differentiates
the linear interpolants instead, the exact derivative of the computed energy
(within 2e-6 Ry/bohr of central differences at 1e-7 bohr,
`tests/unit/test_linear_table_slopes.py`), except within 3e-3 bohr of a grid
point, where `PARSEC_PROJECTOR_CENTER=auto` takes the projector gradient at
that distance (up to 2.6e-2 Ry/bohr off on an Ag table there; exact under
`tip`); it sums the local forces of a cluster on the host (molecular dynamics
about twice as long). It is no better defined over
realistic steps (2e-4 to 4e-2 Ry/bohr off the slopes above, 0.35 on Ag4),
strays from the spline forces by up to 0.75 Ry/bohr, and leaves the optimizer
a noisy gradient: `examples/0d_water_geoopt` needs 23 structures instead of 7
and the hand-built benzene of `examples/0d_mlff_initial_structure` does not
converge in its 60 moves (14 with the tables, 13 with `Cubic_Spline`). It
stays opt-in. In microcanonical molecular dynamics the spline halves to
quarters the scatter of the conserved energy (C60, 21 steps of 20 a.u.: 2.2e-3
-> 9.5e-4 Ry; Si28H36, 16 steps: 7.3e-4 -> 3.5e-4 Ry). For forces,
relaxations and molecular dynamics, give every species `Cubic_Spline: .true.`.

**CA correlation.** The Perdew-Zunger fit of `Correlation_Type: ca` (the
constants of PARSEC 1.5's `exc.f90`) has two branches that differ by 6.413e-5
Ry per electron at r_s = 1, so the energy steps by 6.413e-5 x 3/(4 pi) x h^3
Ry whenever the density of a grid point crosses 3/(4 pi) e/bohr^3, and a
symmetric structure moves symmetric points across together. C60 at h = 0.4
bohr (carbon POTRE, `Cubic_Spline`), moved along its forces: 24 points cross
within 2.5e-4 bohr on one side, a predicted step of 2.352e-5 Ry against 2.347e-5
observed, which a central difference at 1e-3 bohr turns into 1.2e-2 Ry/bohr;
the forces are the slope on either side. PBE has no branch change: the same
C60 gives the slope within 1e-6 Ry/bohr at steps of 2.5e-4 to 2e-3 bohr. Not
changed, so that CA energies stay those of PARSEC 1.5.

`parsec.out` prints the tables PARSEC prints: the ion-ion, local, SCF
correction, core correction and nonlocal forces, the total before and after
symmetrization, the net force that was removed and the torque.

### Hybrid functionals

With a hybrid `Correlation_Type` ([README.md](README.md#self-consistent-hybrid-functionals))
the forces keep the form above, with the orbitals and the density of the
converged generalized Kohn-Sham state: with norm-conserving
pseudopotentials the exchange operator has no explicit dependence on the
ions, and the only term that sees the core density is the semilocal part,
whose potential (with the reduced semilocal exchange) the core-correction
force reads. Finite differences of the total energy (H2, H2O and LiH, whose
Li has a core correction, `h = 0.4` bohr, displacements of 0.01 bohr) give
the same analytic-minus-numeric difference for PBE0 as for PBE, 1.2e-3 to
3.4e-3 Ry/bohr (the grid error of that spacing), equal within 1e-4 Ry/bohr.
In a relaxation or in molecular dynamics the Fock loop of a moved structure
builds its first exchange operator from the orbitals of the structure
before, so only the first structure runs the PBE start.

On several nodes (`scripts/run_multi_gpu_scf.sh`) the ranks keep the orbitals
of their sectors between the structures as well, and the nonlocal force terms
come from the sectors of every rank as for a semilocal functional; a
structure without symmetry (molecular dynamics, `manual` moves) has its sector
on one GPU of the root and the pair densities of its exchange on the GPUs of
all ranks. Measured on A100 nodes: two BFGS moves of C185H124 (PBE0, forces
of every structure) on 2 nodes against one node, energies within 6.7e-10 Ry,
forces within 1.2e-11 Ry/bohr, the same SCF iterations; three steps of
molecular dynamics of H2O (PBE0, one sector) on 2 nodes, the trajectory of one
node to the last bit.

Periodic cells ([README.md](README.md#periodic-cells)) take the same forces:
at the Gamma point from the real orbitals of the Fock loop (also with
collinear spin), with k-points from the Bloch factors of every k-point the
loop solved (the time-reversal representatives of the mesh, whose weights
count their partners); on Andrew Grieder's routines the nonlocal force is
accumulated from them after the loop
(`SCF/hybrid_periodic.bloch_nonlocal_forces`), on the Fortran-derived ones
`Forces/periodic.py` reads them. A moved structure starts its Fock loop
from the potentials, the eigensubspaces of every k-point and the Bloch
orbitals of the structure before. There is no stress.
An unpolarized hybrid at the Gamma point on the accelerated command line
keeps the real sectors of its abelian group for its forces and its
`bfgs`/`simple` relaxations where its SCF takes them (a basis of at least
2^24 entries, `PARSEC_PERIODIC_SECTOR_FORCES=auto`), whose forces and moved
positions are averaged over that group as for a semilocal functional. A
smaller cell keeps the full grid; since 2026-11-21 its forces and moved
positions are averaged there over the operations of the cell on the default
routines too, as the Fortran-derived routines average theirs
(`PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto`, the same SCF and the same time;
before, nothing averaged them on the default routines, and a BFGS relaxation
of a rocksalt NaCl cube left its C4v group after one move; `sectors` takes the
sectors instead, 1.1-1.6 times slower on 8-atom cells). Measured on A100
against the full grid (PBE0, BFGS with 2 moves, 1e-6 Ry SCF): diamond 512
atoms in the classes of its half-cell translations within 6.7e-6, 1.03e-5 and
3.7e-6 Ry/bohr at its three structures and 7.5e-6 bohr, 100.7 s against
1099.1 s on one node of 4 A100. Single-point forces of the routes agree within
the SCF noise; relaxation trajectories agree only as far as the optimizer does
not amplify it (two BFGS moves of the NaCl cube carry force differences of
1e-8 Ry/bohr into positions up to 1e-3 bohr apart; the switch table of
`acceleration/README.md`).
With a k-point mesh, by default
(`PARSEC_HYBRID_KPOINT_FORCE_WEDGE=auto`, since 21 November 2026; `off`
keeps the time-reversal pairs) the forces and `bfgs`/`simple`
relaxations of a bulk hybrid with a Monkhorst-Pack mesh run on the
irreducible wedge of `PARSEC_HYBRID_KPOINT_WEDGE` (the operations that map
the grid, its finite-difference operators and the mesh onto themselves and
keep movable and fixed atoms apart): the nonlocal forces of the wedge's
k-points are completed over its group and the forces averaged over it, as
for semilocal k-point forces (`PARSEC_KPOINT_FORCE_WEDGE`), so the structure
keeps the group; under `auto` the first structure's semilocal start decides
(an insulator takes the wedge for its Fock loop) and the later structures
of a relaxation keep that route (their Fock loops start from the wedge's
orbitals, `SCF/hybrid_periodic.py`, `BlochExchange.adopt_carried`; the
forces of the default routines take the wedge's system,
`hybrid_force_system`). A later structure that turns metallic keeps the
wedge too; `parsec.out` then says in a NOTE that its Fock loop ended with a
partly filled state (`BlochExchange.after_loop`), and
`PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off` keeps the pairs. On the
Fortran-derived routines `PARSEC_KPOINT_FORCE_WEDGE` is not read (as for
semilocal forces there); `PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off` works on
both. Molecular dynamics and `manual` movements keep the time-reversal
pairs.

### Meta-GGA functionals

With a meta-GGA ([README.md](README.md#meta-gga-functionals)) the forces keep
the form above as well: `tau` is a functional of the orbitals alone, and the
orbital term `-div(vtau grad)` (with the first-order term of `W = de/dg` in
the orbital form of the density gradient) has no explicit dependence on the
ions. The only term that sees the frozen core density is the core
correction. The core density enters the density `n` and the density
gradient, never `tau` (as in Quantum ESPRESSO and PARSEC 1.5 with libxc), so
the force reads `dE_xc/dn_core`:

- density form: the exchange-correlation potential `vrho + D^T (2 vsigma D n)`
  of the total density, as for a GGA;
- orbital form: `vrho + D^T W` with `W` formed from the gradient of the valence
  orbitals plus that of the core density
  (`MetaGGARecord.core_potential`, half the core density per channel with
  spin), since the potential of that form holds `vrho` alone.

The SCF test of `tests/unit/test_meta_gga_scf.py` compares this force with
the derivative of the meta-GGA energy with respect to the position of the
core density (LiH, both forms): 2e-4 relative, as for PBE. The SCF
correction force uses `V_out - V_in` as for every functional; the residual
of `vtau` (and `W`) is not part of it. In a relaxation or in molecular
dynamics the next structure starts from the mixed `vtau` (and `W`) of the
structure before, so only the first one runs the PBE start.

Periodic cells (bulk at Gamma and with k-points, slabs and wires) take the same terms: the periodic and slab
forces read `dE_xc/dn_core` through `Forces.periodic.meta_gga_core_potential` (the core potential of the orbital form,
the exchange-correlation potential otherwise), and the k-point loops carry `vtau` and `W` to the next structure
(Andrew Grieder's backend through `SCFWarmStart`, the Fortran-derived one through `meta_gga_start` of
`run_scf_kpoints`). Central differences of the total energy of a Si cell (fcc, two atoms, one displaced by
(0.1, 0.05, -0.08) bohr, Andrew Grieder's routines, step 0.01 bohr, SCF to 1e-9 Ry) against the forces of
`compute_periodic_forces`, largest component difference in Ry/bohr (forces ~0.03-0.06):

| functional | Gamma, h 0.4 | 2 k-points, h 0.4 | Gamma, h 0.3 |
|---|---|---|---|
| PBE | 3.5e-4 | 2.7e-4 | 1.7e-4 |
| smooth test meta-GGA (`tests/unit/test_meta_gga.py`) | 3.4e-4 | - | - |
| SCAN, density form | 1.8e-3 | 2.5e-3 | 3.8e-4 |
| SCAN, orbital form | - | 2.6e-3 | - |

The meta-GGA terms of the forces are complete: the smooth meta-GGA matches PBE, and SCAN's larger difference at
h 0.4 falls by a factor of five at h 0.3 (PBE's by two), the grid sensitivity of SCAN rather than a missing term. On
the Fortran-derived routines (`Use_Fortran_Routines: .true.`, the same cell with the FHI98PP potential of
`examples/3d_meta_gga`, two k-points, h 0.3, `Forces.compute_forces`) the differences are 2.1e-3 (PBE) and 2.6e-3
Ry/bohr (SCAN): the meta-GGA adds nothing to what those routines show for PBE.

## Periodic cells

A periodic cell has forces, relaxation and molecular dynamics through the periodic backend
(`Relaxation.periodic_backend`) of the readable solver. `main.py` hands such an input to `parsec_python.cli`
(`NOTE: ... running the reference parsec_python.cli solver instead`) when it has k-points or spin; a spin-unpolarized
Gamma-point cell stays on the accelerated path, and so do the forces and relaxations (`bfgs`, `simple`) of a
spin-polarized bulk cell at the Gamma point in the sectors of its group on the default routines
(`PARSEC_SPIN_SECTOR_FORCES=auto`, `acceleration/SCF/gamma_spin.py`), and its molecular dynamics, manual movements and
relaxations with fixed atoms on the full grid of the accelerated driver (one device, or one sector of the identity
shared by the GPUs of the process where one would not hold both channels, `PARSEC_SPIN_FALLBACK_SHARED`; the
readable solver where neither does); the Fortran-derived routines (the default of
the forces of a spin-polarized PBE cell) keep the readable solver. There the ionic loop of a spin-polarized cell at
the Gamma point is the 1x1x1 k-point run of the device k-point solver on the Fortran-derived routines (since
2026-11-21, `PARSEC_SPIN_IONIC_KPOINT`: V- in 2^3 diamond with PBE, forces 174.8 -> 12.3 s, a `bfgs` relaxation of
three moves 469.9 -> 29.3 s, three molecular dynamics steps 526.3 -> 31.6 s on an A100 node, the same results within
1.7e-6 Ry/bohr; its `parsec.dat` holds the Bloch orbitals of k = 0, made real since 2026-11-22 as the file of a run
without k-points, `PARSEC_GAMMA_PARSEC_DAT`) and the
Gamma-point loop with the eigensolvers of the host on the default routines.

Forces (`Forces.compute_periodic_forces`; the Fortran routines are `forpbc.f90`, `force_loc_pb.f90`, `ewald_sum.f90` and
`forcnloc.f90`):

1. **Ion-ion**: the exact derivative of the Ewald energy (`Forces/ion_ion.py: ewald_ion_ion_forces`): the real-space sum
   over the atom pairs within the cutoff and the reciprocal sum over the structure factors of a box of reciprocal
   vectors (`V_ion.ewald.ewald_ion_ion_terms`, 4 s for 4,096 atoms on a laptop).
2. **Local pseudopotential** (`Forces/periodic_local.py`): `integral rho(r) grad V_a(r - R_a)` over the images of the local
   potential as it is built (the short-ranged `V_a(d) + 2 Z erf(eta d)/d` and the reciprocal series), and
   `integral V_xc(r) rho_core'(d) d/|d|` over the images of the core density. There is no SCF-residual term, as in
   Fortran's periodic path.
3. **Nonlocal projectors** (`Forces/nonlocal_force.py`): the same formula with the Bloch projectors of the k-point,
   `F_a = 4 sum_k w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`. A k-point run keeps no eigenvectors,
   so `run_scf_kpoints(..., compute_forces=True)` accumulates the term from them at the end of the SCF loop
   (`result.nonlocal_forces`). The row-distributed MPI k-point solver does not support it; the MPI k-point pools do
   (since 2026-11-17: each rank adds the terms of its columns, summed over the ranks).
4. The **net force is removed**, as Fortran does for a crystal, and no torque is reported; the force report prints
   `(Ewald forces included)` as Fortran's. The `(net) force` line is the sum over the atoms before the removal. Each term
   is the derivative of the energy on the grid, so that sum is the derivative of the energy under a rigid shift of all
   atoms: the egg-box effect of the grid, zero by symmetry in a perfect crystal, plus, for atoms that sit on grid points,
   whatever the gradient there makes of the projectors next to them (`PARSEC_PROJECTOR_CENTER`). Under `auto` that
   gradient is the slope of the energy over the steps of a relaxation, and the net force is the slope of the SCF energy
   under a rigid shift: on a 64-atom diamond cell with eight atoms moved by 0.05 bohr (56 atoms on grid points, carbon
   POTRE) -0.267992 Ry/bohr per component against the central difference at 1e-3 bohr -0.267979 (1.4e-5 for the sum
   over 64 atoms); a 512-atom cell of the same kind prints -0.262. Under `tip`, the gradient `nonloc.F90` forms at the point, the 64-atom cell prints -0.118619, which
   equals the central difference within 1e-8 only for steps inside the first radial sample (5e-6 bohr) and is -0.268 at
   1e-3 bohr; the 512-atom cell printed -0.110. Under `off` it printed 1.78, from the nonlocal gradients of the atoms
   within round-off of a grid point. At the Gamma point and on the whole mesh (or its time-reversal pairs) the
   forces are **not symmetrized**. A Monkhorst-Pack mesh (with or without a `Monkhorst_Pack_Shift` block, which since
   2026-11-18 is PARSEC 1.5's Gamma-centred mesh, `PARSEC_MP_UNSHIFTED_MESH`) of the forces of a single point or of a
   `bfgs`/`simple` relaxation runs its irreducible wedge since 2026-11-18 (`PARSEC_KPOINT_FORCE_WEDGE`, `auto` by
   default; also for non-magnetic spin-orbit coupling since 2026-11-19, `PARSEC_SOC_KPOINT_WEDGE`), as a single point without forces and Fortran's routines do: the operations of the space group that map the
   grid of `Grid_Spacing` onto itself reduce the mesh and symmetrize the density, the nonlocal force of the wedge is
   completed over them (its points stand for their stars) and the total averaged over them (`symm_forces.f90`), so the
   structure keeps the group. On the default routines these are since 2026-11-21 only the operations that also keep
   the grid's finite-difference operators (`PARSEC_SCALAR_KPOINT_WEDGE=auto`; all of them on an orthogonal grid), and
   the k-points are the shortest of their images, a point of the zone boundary with all its shortest images at equal
   shares (`PARSEC_KPOINT_FOLD=shortest`): the wedge then gives the k-point sum of the whole mesh without symmetry on
   the same grid (hcp Au2 with an atom moved along c, 0.3 bohr, order 6, 4x4x2: 19 of 57 points, the energy within
   9e-11 Ry and the force along c within 8e-8 Ry/bohr). The total is averaged over every operation that maps the grid
   and the mesh onto themselves, a larger group there (`PARSEC_KPOINT_FORCE_GROUP=crystal`,
   `SinglePointInput.force_space_group`; 6 operations against the 2 of that wedge): the group of the converged forces,
   onto which the average projects. An axis that turns an oblique grid's stencil (the three-fold axis of hcp) then keeps
   its atoms with zero in-plane forces, which the same-grid run without symmetry has at the order of the discretization
   (4.8e-5 Ry/bohr at order 6 and 0.3 bohr, 3.4e-6 Ry/bohr at order 12, 5.5e-7 Ry/bohr at 0.2 bohr and order 12), and
   three `bfgs` movements of that cell kept the 12 operations of its structure (`PARSEC_KPOINT_FORCE_GROUP=wedge`,
   the average over the wedge's operations alone, moved the atoms off the axis by 1.2e-4 bohr and left 4; a structure
   whose operations all turn the stencil, such as a hexagonal cell with C3 alone, runs the whole mesh and averages its
   forces over them);
   `PARSEC_SCALAR_KPOINT_WEDGE=grid` takes every operation that maps the grid for the k-point sum too, as before, whose
   wedge lies at the order of the discretization from the whole mesh (3.4e-5 Ry/bohr along c there). A relaxation
   whose `Move_Flag` fixes atoms that an operation exchanges with movable ones takes the wedge of the operations that
   map the fixed atoms onto fixed atoms and the grid's finite-difference operators onto themselves (on an oblique grid
   a subgroup: hcp Au2 4 of 12), which every structure keeps (`PARSEC_MOVE_FLAG_SPACE_GROUP=auto`, since 2026-11-21;
   on the Fortran-derived routines as well, which refused such a relaxation before), its forces averaged over that
   subgroup. Molecular dynamics and `manual` movements (PARSEC uses no symmetry for them), a `Move_Flag` whose fixed
   atoms no operation but the identity keeps (or any breaking one under `PARSEC_MOVE_FLAG_SPACE_GROUP=off`),
   `Ignore_Symmetry` and a structure without operations keep the whole mesh, run as
   its pairs `k`, `-k` (`PARSEC_KPOINT_TIME_REVERSAL`; on an oblique cell of the default routines with every shortest
   image of a point of the zone boundary, so that each pair is exactly `k`, `-k`: a primitive fcc Au2 cell without
   symmetry, 6x6x6, gave the forces of its whole mesh within 5e-9 Ry/bohr); `PARSEC_KPOINT_FORCE_WEDGE=off` keeps the
   whole mesh everywhere.
   Since 2026-11-19 the local ionic and core-correction forces of the wedge are summed for one atom per orbit of its
   group and turned onto the others (`PARSEC_PERIODIC_LOCAL_ORBITS`, also for a Gamma cell in abelian sectors), and on
   a GPU the ball sums of an orthorhombic cell are formed for many atoms per call (`PARSEC_PERIODIC_LOCAL_BALLS`; the
   same terms in another order, 2.5e-15 Ry/bohr): diamond 6^3 with the 3x3x3 mesh, one relaxation step on 16 A100
   49.9 -> 42.3 s, its forces 13.6 -> 0.49 s.

The terms are exact derivatives of the energies the SCF evaluates (`Forces` tests: finite differences of each energy and of the
SCF total energy, 1e-5 Ry/bohr or better), with the exceptions of the tables described under Forces: for a species without `Cubic_Spline` the projectors and the core density of the default routines (at the Gamma point
and on k-point meshes) use PARSEC's derivative tables (`PARSEC_LINEAR_TABLE_SLOPES=tabulated`, the local potential of those routines is always the spline), while the
energy interpolates the tables linearly and is rough on the scale of a relaxation step (see the paragraph on linear
tables in the forces of an isolated system): a silicon cell of 216 atoms at the Gamma point misses the slope of its
energy by 6e-3 Ry/bohr along all atoms, with forces within 9e-6 Ry/bohr of those of `Cubic_Spline`, which gives the
slope within 2e-7. The Fortran-derived routines take the slopes of the interpolants (`PARSEC_PERIODIC_FORCE_SLOPES=interpolant`), the
exact derivative of that rough energy: on an 8-atom silicon cell with a 2x2x2 mesh they miss the slope by 2e-4 to 6e-4
Ry/bohr (it differs by 5e-4 between the two steps) and the forces of `Cubic_Spline` by 1.3e-3, and a BFGS relaxation
of that cell stops at the noise of those forces (17 structures to 1e-3 Ry/bohr against 3 with `Cubic_Spline`).
`PARSEC_PERIODIC_FORCE_SLOPES=tables` (opt-in) takes the tables of `pseudo.f90` there as well; it relaxes POTRE cells
as `Cubic_Spline` does, but does not follow the energy of linear FHI tables and is slower (acceleration/README.md). The support of the projectors of a table converted from a UPF file ends with a jump of
the energy (`PARSEC_PROJECTOR_TAIL`).

Atoms leave the cell as they move. The coordinates of the optimizer and the integrator are kept continuous (they are never
wrapped), and every image sum is built for the atoms folded into the cell (`PeriodicPreparedSinglePointSystem.cell_atoms`).
Fortran's `back_to_cell` wraps the positions in the state; the physics is the same, the positions in `atom.cor` and
`movie.xyz` are the unwrapped ones.

A move rebuilds the local potential, the projectors, the densities and the Ewald energy for the new positions
(`SCF.update_periodic_geometry`) and the SCF loop continues from the last mixed potential. The Gamma-point loop also carries
its eigensubspace; the k-point loops start their eigenproblems cold (`Potential_Extrapolation` applies to both). The image
sums stop where the tables end: the local potential tables are `-2Z/r` to 1e-14 Ry beyond 2 to 7 bohr and the atomic
densities are below 1e-16 beyond 12 (core) to 35 (valence) bohr, so a move of a small cell takes a fraction of a second
instead of the minute the 118-bohr tables needed.

`Cubic_Spline` does not change the periodic local potential, as Fortran's: there the potential is a sum of form factors
that integrate the table to high order, so it is the accurate potential whatever the flag says. The periodic local
potential and its force therefore always use the cubic spline of the table (`V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`
set to `False` gives linear interpolation). Linear interpolation of the logarithmic table (1 % steps) shifted the Ge3
total energy by +0.6 to +0.9 mRy against Fortran at every grid from 24 points per side up; with the spline the difference
is 0.003 to 0.09 mRy, the remainder being Fortran's own finite-difference estimate of the `alpha` energy (+0.064 mRy) and,
on a coarse grid, aliasing. The projectors and the core density follow `Cubic_Spline`.

**Grid of an input file.** An input file gets the Fortran grid: half a step off the origin along vectors orthogonal to the
others (zero with `Ignore_Symmetry` and for non-orthogonal vectors) and the largest even number of points per vector whose
step is not below `Grid_Spacing` (`adjust_step`; steps may differ between the axes of a rectangular cell). The
`PeriodicGridSettings` of the API keep `shift=0` and `points_rule="round"`.

**A caveat for hard pseudopotentials.** On a coarse grid the energy of a hard pseudopotential (the hydrogen of
`examples/0d_AuH` at 0.6 bohr) changes by Ry with the offset of the grid, and the energy of a rigid translation by tens of
mRy at 0.3 bohr. The forces are consistent with that energy (net force removed, as in Fortran, but then a relaxation can
slide a molecule over the grid) and Fortran's reciprocal-space potential has much less of it.

Validated against Fortran PARSEC (`parsec_tests/pbc_forces`; three Ge atoms with a nonlinear core correction, forces up to
2.5 Ry/bohr, the same grid in both codes). Largest difference of the total force after the net force is removed, and of the
nonlocal term, in Ry/bohr, with the total-energy difference (Python minus Fortran) in Ry:

| cell | k-points | points per side | total force | nonlocal | energy |
|---|---|---|---:|---:|---:|
| cubic, 9.6 bohr | Gamma | 24 / 32 / 40 | 2.0e-3 / 6.1e-4 / 4.1e-4 | 3e-5 / 6e-6 / 3e-6 | -4.5e-5 / +8.5e-5 / +7.2e-5 |
| cubic, 9.6 bohr | 2, generic | 24 | 2.0e-3 | 3.4e-5 | -9.3e-6 |
| fcc primitive, 10.2 bohr | Gamma | 20 / 26 / 32 / 40 | 3.3e-4 / 2.7e-4 / 2.3e-4 / 1.5e-4 | 1.5e-5 / 4e-6 / 9e-7 / 9e-7 | +2.2e-4 / +1.9e-4 / +2.0e-4 / +2.0e-4 |
| fcc primitive, 10.2 bohr | 2, generic | 20 | 2.4e-4 | 8.7e-6 | +2.1e-4 |

The k-point forces are as close to Fortran as the Gamma ones at the same grid, and the oblique cell is as close as the cubic
one. The energy offset is Fortran's: its `alpha` energy is a finite difference in q (spacing 0.015) and is low by 6.4e-5 Ry
for the 885 bohr^3 cubic cell, by 2.1e-4 Ry for the 265 bohr^3 fcc cell (it scales as 1/volume), which is the whole offset
at 32 points and above. The residual force difference falls with the grid: Fortran's local and core forces are reciprocal
sums over form factors (they equal ion-ion plus the numerical derivative of that functional to 2e-5 on PARSEC.py's density),
PARSEC.py's are the exact derivatives of the grid-sampled energy. The relaxation and dynamics loops are tested on a
hydrogen pair at the Gamma point and with two k-points (`tests/unit/test_periodic_ionic.py`); the energy of a short NVE run
is conserved to 1e-3 eV/atom.

**Spin polarization (Gamma point).** A spin-polarized periodic cell that asks for forces, a relaxation or molecular
dynamics runs the loop of the isolated spin-polarized path (`run_spin_scf`, which `run_scf` selects; the periodic system
supplies its starting spin densities, spin-polarized XC and the `alpha` energy). The forces are the unpolarized ones on the
total density, the average of the two channels' exchange-correlation potential and the orbitals of both channels at one
electron per state; there is no term for the SCF residual. A spin-polarized input with k-points, spin-orbit coupling or
non-collinear magnetism has no forces. Checked three ways (`tests/unit/test_periodic_spin_forces.py`): a non-magnetic spin run
has the energy and forces of the unpolarized run (2e-6); each term is the derivative of the spin energy functional at the
converged orbitals (ion-ion 1e-12, local ionic 1e-7, nonlocal 1e-8 Ry/bohr with the cubic spline; the core term as in the
unpolarized case); and against Fortran with the same fixed occupations (`occup.in`, 6 up and 5 down states, three Ge atoms
with one electron removed): total force 2.0e-3 / 6.3e-4 at 24 / 32 points (2.0e-3 / 6.1e-4 unpolarized), nonlocal
3.2e-5 / 5.2e-6. Two Fortran conventions show in such a comparison. Its `alpha` energy is multiplied by the total ionic
charge, not by the number of electrons, so a charged cell differs by a constant (0.191 Ry per electron of charge for this
cell; forces are not affected). And `rho_har.F90` shifts the exchange-correlation potential of each spin channel to zero
average in a periodic cell, which removes the average exchange splitting between the channels: with Fermi smearing, where
both channels share one Fermi level, Fortran settles at a smaller moment (0.69 against 0.94 Bohr magnetons for that cell at
1000 K) than the LSDA solution that PARSEC.py keeps; with fixed occupations the two agree.

## Spin-orbit coupling and non-collinear magnetism

`SO_from_scratch` and `Non_Collinear_magnetism` solve for two-component spinor states, each holding at most one electron.
Their forces are the Hellmann-Feynman forces of the energy the SCF loop minimized, in the pieces of the spin-polarized
case:

- **Ion-ion, local ionic, core correction, SCF residual**: the total density, the exchange-correlation potential averaged
  over the two spin directions (`V_avg` of the local-rotation form, whose `B_xc.sigma` part does not depend on the
  positions), and the last `V_out - V_in` of the averaged potential. The result records these (`xc_potential`,
  `input_effective_potential`, `output_effective_potential`).
- **Kleinman-Bylander projectors**: they act alike on both spin components, so the force is that of the spin-polarized
  case on the `2N` components at the weight `f_n/2` (`Forces/spin_orbit.py`, `spinor_nonlocal_forces`).
- **Spin-orbit projectors**: the spin-orbit term is a bilinear form in the overlaps `<P_j|u_sigma>` of the spinor with the 16
  projector functions of an atom (`V_ion` and `V_so` for the harmonics of `l = 1, 2`, complex spherical harmonics, KB sign
  per `l`); `apply_spin_orbit` multiplies the projectors with coefficients linear in these overlaps (`lzsz` on the same
  spin, the `lsxy` ladders on the other). A projector depends on the atom only through `r - R_a`, so
  `F_a = 2 sum_n f_n Re sum_(sigma,j) conj(<grad P_j|u_n,sigma>) c_(sigma,j)`. The gradient takes the derivative of the
  radial functions the Hamiltonian samples, so the force is the derivative of the energy that is minimized, not of
  another table (Fortran's `dv_ion`/`dv_so` tables): for a species with `Cubic_Spline` PARSEC's cubic spline of `vionr`
  and `vsor` and its derivative, as `nonloc.F90` evaluates them under `uspline` (`PARSEC_SOC_PROJECTOR_SPLINE=auto`, since
  2026-11-22), otherwise the linear interpolation of the tables and its slope. The energy of linear tables has a kink
  wherever a grid point crosses a radial knot: their forces are exact derivatives over steps of 1e-6 bohr but miss the
  slope of the energy over the steps of a relaxation, 1.6e-5 and 2.5e-5 Ry/bohr at 1e-3 and 2e-3 bohr in the spin-orbit
  Ge8 cell, where the spline is within 1.5e-6 and 6.1e-6 (the h^2 truncation of the difference; Richardson 1.4e-8).
  `PARSEC_SOC_PROJECTOR_SPLINE=off` keeps the linear tables for every species, as before. Within 3e-3 bohr of an atom the
  gradient is taken at that distance on both sets of routines (`PARSEC_PROJECTOR_CENTER=auto`, the rule of the scalar
  projectors below).

Isolated runs use `compute_spinor_forces`; a periodic run uses `compute_periodic_forces` on a result whose nonlocal force
the k-point job accumulated from the orbitals of every k-point in a last pass (`compute_forces=True`, as for the scalar
k-point driver), with the Bloch phases of the projectors and their gradients and the atoms folded into the cell, each
k-point at its own weight. On the default routines the Monkhorst-Pack mesh of plain spin-orbit coupling (no spin
polarization, no non-collinear magnetism) runs its irreducible wedge since 2026-11-19 (`PARSEC_SOC_KPOINT_WEDGE`), as for
scalar orbitals but by the operations that also keep the grid's finite-difference operators (all of them on an
orthogonal grid, a subgroup on an oblique one): the nonlocal force of the wedge is completed over its group and the
forces averaged over it; molecular dynamics, `manual` movements and the other runs that keep the whole mesh run its
time-reversal pairs (the Kramers partner of a spinor at `-k` has the same density and forces). Bcc Au2 with an atom
moved along (1, 1, 1), 4x4x4 on one A100 node: the forces of the 13-point wedge within 6e-6 Ry/bohr of the 64 points
(22.5 against 77.6 s), two relaxation steps 62.2 against 251.6 s with every energy within 1.2e-6 Ry (the SCF criterion
is 1e-6 Ry). hcp Au2 with an atom moved along c (4x4x2): 2 operations kept, the forces of the 14-point wedge within
1e-7 Ry/bohr of the 20 time-reversal pairs. A relaxation whose `Move_Flag` fixes atoms that an operation exchanges
with movable ones takes the wedge of the subgroup that maps the fixed atoms onto fixed atoms since 2026-11-22
(`PARSEC_SPINOR_MOVE_FLAG_SUBGROUP=auto`, as for scalar orbitals; `off`: the time-reversal pairs). Collinear spin and
non-collinear magnetism keep the whole mesh. Moved
atoms keep the direction of their initial moment (`Initial_NCL_Moment`). The SCF loop of a moved structure continues from the
previous one, for plain spin-orbit coupling, collinear spin and non-collinear magnetism alike (isolated or periodic, Gamma
point or k-points): the last mixed potential with the new ionic potential (for spin the pair `V_up`, `V_down`; for
non-collinear magnetism `V` and the exchange-correlation field `B_xc`, which carries the moments to the moved atoms, so
`Initial_NCL_Moment` is not asked for again), the Hartree potential as the start of the Poisson solve, and the eigensubspace
of every k-point (kept on the host between structures; on the GPUs it goes back to the device when the k-point is solved),
so the first iteration filters the old orbitals instead of solving from random vectors. `Potential_Extrapolation` applies
to plain spin-orbit coupling only (the atomic density guess is a total density, as for the scalar spin-polarized loop). On
Au2 in a 9 bohr cell at 0.45 bohr spacing (Gamma point, CPU) a molecular dynamics step took 7 to 9 SCF iterations instead of
21 to 31 (plain spin-orbit 9 against 31, collinear spin 7 against 24, non-collinear 7 against 23), the wall time of a
relaxation 111 s instead of 195 s (12 structures, the same path). The gain shrinks as the steps grow: with steps of a few
tenths of a bohr (a relaxation on a 0.7 bohr grid, whose forces are several Ry/bohr) it vanishes. The direction of a
non-collinear moment is a soft mode, so a continued and a fresh loop agree on its size and to a few hundredths of a
Bohr magneton on its components. The forces are not symmetrized over the point group of
the atoms for a spin-polarized or non-collinear run, whose magnetization need not respect it; the net force is removed.

Fortran's `forcnloc.f90` has a spin-orbit term (`forcso`, `soforc.F90`) but warns that it "may be inaccurate", and its
`flsxy` pairs the same harmonic index on both sides where the Hamiltonian pairs `lm` with `lm +- 1`; with the divisor
`nspin/mxwd` of `forcnloc.f90` for a spinor it is not a reference. These forces are validated against the energy of this
package instead (`tests/unit/test_spinor_forces.py`, Au dimer with a relativistic pseudopotential, cubic spline):

| Check | Result |
|---|---|
| Each term against a finite difference of its energy at the converged spinors (non-collinear, spin-orbit, one unpaired electron) | ion-ion 1e-8, local ionic 1e-6, nonlocal plus spin-orbit 7e-5 Ry/bohr |
| Spin-orbit term alone, smooth orbitals, step 1e-4 | 1e-5 Ry/bohr with the piecewise-linear radial tables, whose kinks are the residue (measured before the spline of `PARSEC_SOC_PROJECTOR_SPLINE=auto`) |
| Total force against the derivative of the converged SCF energy (Richardson, steps 0.01 / 0.005 bohr), isolated, non-collinear with spin-orbit | -9.5720 against -9.5725, 4.9827 against 4.9817 Ry/bohr |
| The same, collinear spin with spin-orbit (isolated and 2x1x1 k-points), unpolarized spin-orbit (Gamma), non-collinear (2x1x1 k-points, periodic) | 5e-5 to 3e-3 Ry/bohr on forces of 0.7 to 10 Ry/bohr |
| Without spin-orbit, non-collinear against the collinear spin-polarized driver (whose forces were compared with Fortran) | energy 1e-6 Ry, forces 2e-4 Ry/bohr |

The forces on this dimer are large (10 Ry/bohr, with a net force of the same size before it is removed) because a
0.7 bohr grid is far too coarse for Au; they show that the forces follow the energy of the grid, which is what the checks
test, not that the grid is converged.

Not covered: the perturbative correction (`SO_PSP` with an ordinary spin-polarized SCF), which shifts the eigenvalues of that
SCF to first order and reports no total energy that includes the shift, so a relaxation would have no energy to follow
(`SO_from_scratch` has forces); `Energy_Refinement`; the row-distributed MPI k-point solver (it raises, as for the scalar k-point forces); the
accelerated path, which hands every spin-orbit and non-collinear input with forces to the reference solver (the pooled
k-point solver of that solver runs on the GPUs for periodic and non-collinear inputs and forms the overlaps of the force on the
device, which agrees with the CPU to 1e-5 Ry/bohr on four A100 GPUs: `parsec_tests/pbc_forces/gpu_spinor` and `gpu_k`).

## Structural relaxation

`Relaxation/domove.py` is one call of `domove.f90`; `Relaxation/ionic_loop.py`
is the loop of `parsec.F90` around it.

- **`bfgs`**: `Relaxation/lbfgsb.py` is a routine-by-routine translation of
  the L-BFGS-B 3.0 that PARSEC ships (`lbfgsb.f`, `linpack.f`), driven by
  reverse communication with PARSEC's settings (`factr = pgtol = 1e-15`, so
  that the force criterion and not the optimizer ends the run). Its state has
  the layout of the Fortran arrays, which is what makes `relax_restart.dat`
  exchangeable. Its first step moves along the force by `1/|g|`, so the
  second structure often lies higher in energy than the first; the line
  search then corrects the step. This is PARSEC's behaviour: on four recorded
  unbounded trajectories (`nbd = 0`) the points it asks for equal those of the
  compiled `lbfgsb.f` bit for bit. While the energy keeps falling each further
  trial of a line search extrapolates, so in a small cell near equilibrium the
  path can leave the basin of the input structure. Only `Max_Step` bounds the
  whole run, every coordinate staying within it of its start; the environment
  variable `PARSEC_BFGS_FIRST_STEP` (off by default, a length in bohr) limits
  only the largest atomic move of the first trial. See the switch table in
  `acceleration/README.md`.
- **`simple`**: each coordinate moves along its force component by a length
  that grows from `Min_Step` at `Force_Min` and saturates at `Max_Step`.
- **`manual`**: the structures of `manual.dat` are computed one after another
  without recentring and without symmetry, as in PARSEC; the accelerated
  command line solves them in the sectors of the operations every structure
  keeps (`PARSEC_MOVING_SECTORS`, "Molecular dynamics" below), with the same
  energies and forces.

After a movement everything that depends on the positions is rebuilt on the
unchanged grid (`isolat`, `nonloc`, `corecd`, `ionpot`, `forceion` of
PARSEC): local ionic potential, projectors, core density, ion-ion energy and
the atomic part of the Hartree boundary. The next SCF loop starts from the
previous wavefunctions (the first diagonalization is a subspace-filter step)
and from the previous mixed potential with the ionic part exchanged
(`V_new - V_ion,old + V_ion,new`); the mixer history is restarted. The domain
stays that of the input structure: an atom that leaves it stops the run.

### Start of the next SCF loop

`Potential_Extrapolation` changes where the SCF loop of a moved structure
starts, not where it ends (`SCF/single_point.py`):

- `none`, the default: PARSEC's start above.
- `atomic` (Quantum ESPRESSO's `pot_extrapolation = 'atomic'`): the density
  `rho_out - rho_atoms(R_old) + rho_atoms(R_new)`, the last output density
  with the superposed atomic densities moved along with their nuclei, gives
  the Hartree and exchange-correlation potentials of the first input
  potential, which replace those of the last output potential. The first
  iteration filters with the degree of the first solve (`Chebdav_Degree`),
  so that the saved orbitals catch up with the moved potential before the
  mixer sees their density.
- `second_order` (QE's `'second_order'`): from the second move on, the change
  `d = rho_out - rho_atoms` is extrapolated as well,
  `rho_atoms(R_new) + 2 d(R_old) - d(R_older)`.

The eigensolver always starts from the subspace of the last structure. A
first-order extrapolation of that subspace along an MD trajectory (QE's
`wfc_extrapolation = 'first_order'`: `2 X(t) - X(t-1) R`, with `R` the
polar factor of the overlap of the two bases) was tried and left out: it
saves no SCF iterations, since the density decides when the loop ends, and
only lets the first iteration after a move filter at the later degree. On
C35H36 (`second_order`) that gave 3.2 iterations per step against 3.1 with
the stronger first filter; on C795H300 (four A100) 4.0 against 3.6
iterations, 10.96 against 10.66 s per step and 18 GB more on each GPU,
and C1653H508 no longer fitted.

Without superposed atomic densities as the first density (an imported or
predicted one) and with spin polarization the loop starts as with `none`.
The extrapolation alone saved nothing on nanodiamonds; with the stronger
first filter it saves a quarter to a third of the SCF iterations of a later
structure (numbers under "On A100 GPUs"). Converged tightly, both starts give
Fortran's trajectory; at a loose SCF criterion a trajectory changes within
the SCF noise of its forces, so a relaxation can end one or two structures
earlier or later.

### SCF criterion of a relaxation

With `Relaxation_SCF_Upscale: U` the SCF loop of a structure stops at
`Convergence_Criterion * clamp(F / Force_Min, 1, U)`, `F` the largest force
on a movable atom of the structure before, and that of the input structure
at `U` times the criterion (`adaptive_scf_criterion` in
`Relaxation/ionic_loop.py`; Quantum ESPRESSO's `upscale` scales its
criterion the same way from the other end). A structure whose forces meet
`Force_Min` at a looser criterion is converged again at
`Convergence_Criterion`, continuing from its own loop, before the mover sees
it, so the run ends on forces of the input's accuracy. The log names the
criterion of every structure.

The forces of a looser loop are less accurate. C35H36 at its input
structure (largest force 0.037 Ry/bohr), from superposed atomic densities:

| Criterion | SCF iterations | Energy error | Largest force error |
|---|---:|---:|---:|
| 2e-4 Ry | 13 | 9e-7 Ry | 9e-5 Ry/bohr |
| 5e-4 Ry | 11 | 5e-6 Ry | 5e-5 Ry/bohr |
| 1e-3 Ry | 9 | 5e-5 Ry | 6e-4 Ry/bohr |
| 2e-3 Ry | 7 | 4e-4 Ry | 1.1e-3 Ry/bohr |

The criterion is `U` times looser only where the forces are `U` times
`Force_Min` or more, so the error stays near a hundredth of the force. The
BFGS relaxation of C35H36 (`Force_Min` 0.01 Ry/bohr, `atomic`) took 9
structures and 64 SCF iterations at the fixed criterion, 8 structures and
43 iterations with `U = 10`; the two final structures differ by at most
0.015 bohr, within what `Force_Min` allows. On 16 A100 (inputs of
`gpu_benchmark/`, `atomic`), with `U = 10` against 1:

| Cluster | Structures | SCF per later structure | Program |
|---|---:|---:|---:|
| C795H300 | 7 and 7 | 1.51 s against 1.77 s | 19.2 s against 21.0 s |
| C1653H508 | 6 and 6 | 3.95 s against 4.24 s | 38.0 s against 40.4 s |

The final energies differ by 4e-5 Ry and 2.5e-3 Ry (about 1e-6 Ry per
atom): both runs meet `Force_Min` at slightly different structures.

Symmetry: the point-group operations of the input structure that respect the
movable flags are found once. Forces are averaged over them and so are the
positions the mover returns, which keeps a symmetric structure symmetric to
round-off. When `Move_Flag` fixes an atom that an operation exchanges with a
movable one, the run keeps the subgroup of the operations that map the fixed
atoms onto fixed atoms and says so (`PARSEC_MOVE_FLAG_SUBGROUP=auto`, since
2026-11-19; the accelerated driver solves the sectors of that subgroup, found
with the fixed atoms as a species of their own, and a periodic cell at the Gamma
point does the same whenever atoms are fixed); under `off`, or where only the
identity respects the flags, the run uses no symmetry, as before.
`PARSEC_SYMMETRIZE_POSITIONS=on` averages the input positions over the group
found within `PARSEC_SYMMETRY_ATOM_TOLERANCE` first, so that a rounded input
gets its group exactly (`Forces/position_symmetry.py`; a bulk cell is then
shifted as a whole so that the operations whose translations map its grid
onto itself to within the tolerance do so exactly, as the sectors need).
The test of the whole group of the atoms also decides whether a relaxation
with point charges or `Foldblock_Side_Length 2` keeps the sectors of the group
the charges or the fold-block domain leave intact on the accelerated path
(`PARSEC_POINT_CHARGE_SYMMETRY`, `PARSEC_FOLDBLOCK_SYMMETRY`,
`acceleration/README.md`): where an operation exchanges a fixed atom with a
movable one the run takes the full grid, as without those switches, whatever
`PARSEC_MOVE_FLAG_SUBGROUP` says (the subgroup is not combined with the kept
group of a constraint). Such a relaxation keeps its forces unsymmetrized, as
PARSEC does without a group: the sector density holds the group to round-off.

## Molecular dynamics

`MolecularDynamics/` translates `moldyn.f90`: Beeman steps without friction
(microcanonical) or Langevin dynamics with `Friction_Coefficient`,
the cooling schedules `stair`, `linear` and `log`. The mean force is
subtracted before the integrator and no symmetry is used, as in PARSEC.
`Restart_mode` continues from `mdinit.dat`.

Symmetric trajectories in sectors (`PARSEC_MOVING_SECTORS=auto`, the default
since 2026-11-21; `acceleration/SCF/moving_sectors.py`). Positions, forces and
a state of the integrator that keep a group keep it in exact arithmetic, so the
accelerated command line solves such a trajectory in the sectors of that group,
as it solves a relaxation. The runs it takes: the microcanonical integrator
restarted from an `mdinit.dat` whose positions, last displacement and
accelerations keep a group, or started with one movable atom (its Maxwell
velocity is removed with the centre of mass), and manual movements (the
operations every structure of `manual.dat` keeps with the permutation of the
input, also `Movement_Num: 0`). A fresh run of several movable atoms draws
random velocities and a Langevin run a random force, which leave every group
after the first structure: they keep the full grid, bit for bit as before. A
cluster keeps the signed permutations about the origin that keep its atoms
(those `Move_Flag` fixes as a species of their own) and the state or the
structures, which the detection takes as pseudo-atoms where they keep less
than the atoms; a Gamma cell the group its detection finds, above the floor of
the sectors of an SCF (`PARSEC_PERIODIC_SECTOR_FLOOR`) and where the group of
its sectors has 4 operations or more (the 2 sectors of a mirror cost more than
they save: silicon 3^3 with one movable atom on a bond 30.1 -> 31.4 s, 5.2 ->
11.9 GiB; 4 sectors, the atom on an axis: 23.6 -> 16.6 s), checked against the
state and the structures after its preparation. At every structure the forces
are averaged over the group of the sectors; after every step the new
positions, the orbits of their atoms and the state are checked against it to
1e-6 bohr (the displacement and the accelerations times `Time_Step^2`) and the
state is averaged over it. A structure that leaves the group is prepared again,
in the sectors of the group it keeps (a cluster) or on the full grid, from the
potential of the structure before, with a NOTE in `parsec.out`. Two spin
channels take the sectors too since 2026-11-22 (`PARSEC_MOVING_SPIN_SECTORS=auto`,
both channels with the spin guard of every loop, which leaves the sectors where
it fires as in a single point: weighted sectors for the abelian ones or the full
grid, abelian ones for the full grid, kept for the later structures), but for a
start whose labels carry `Initial_Spin_Polarization` of both signs, which keeps
the full grid with a NOTE: the sectors of its labels can hold a symmetric state
that the full grid leaves (Fe2, Co2: 10-28 mRy higher); `on` takes those too,
`off` keeps every spin run on the full grid.
Spinors, `Ignore_Symmetry`, the other reasons PARSEC switches the
symmetry off for, `Restart_From_Wfndat`, MPI, wires, slabs and k-points keep
the full grid; so does the MPI full-SCF runner. `off` restores the full grid of
before. Results: acceleration/README.md, switch table.

Random numbers decide the initial velocities and the Langevin force, so a
trajectory can be compared with Fortran PARSEC only where both draw the same
ones:

- `test_rng: true` selects PARSEC's own deterministic generator, started from
  `rng_seed`; the numbers are those of the Fortran program.  A restart draws
  `2 (iframe - 1)` numbers first, as PARSEC 1.5 does, so that the generator
  continues about where the earlier run left it.
- `Mol_RNG_Seed` is PARSEC 1.5's name of `rng_seed` (1.5 no longer reads the
  old name); both are read here with the same meaning, an input with both is
  refused.
- `rng_seed: n` with `n > 0` alone makes PARSEC seed the generator of its
  compiler. This package then draws what a PARSEC built by gfortran 8 or
  later draws, so the two give the same trajectory;
  `PARSEC_MD_RANDOM_NUMBERS=numpy` takes NumPy's generator with that seed
  instead.
- Without either, PARSEC seeds from the clock and this package from fresh
  entropy: two runs differ, in either code.

Behaviour of the Fortran that looks unintended is reproduced so that
trajectories can be compared, and is listed in the module documentation of
`MolecularDynamics/moldyn.py`: `Time_Step` is taken in atomic units whatever
unit the input names, masses use 1836 electron masses per atomic mass unit,
and kinetic and potential energies are reported per different atom counts.

## Periodic cells

`Forces/periodic.py` evaluates the forces of a bulk cell with the structure of
Fortran's `forpbc.f90`/`force_loc_pb.f90` (local and core-correction terms),
`ewald_sum.f90` (ion-ion) and `forcnloc.f90` (projectors with their lattice
replicas and, for k-points, the Bloch phases `right`), then symmetrizes them
and removes the net force. Each term is the exact derivative of the energy this
package computes, which differs from Fortran's in its local potential (a
real-space Ewald split instead of the G-space `v_first`/`pot_local`):

- local: `dV sum_r rho(r) [sum_T f'(|d_T|) d_T/|d_T| + 2Z (4 pi/V) sum_G w_G G sin(G.(r - R))]`
  with `f = V_loc + 2Z erf(eta r)/r` and the translations and G vectors of
  `ewald_local_ionic_potential`; ion-ion: the derivative of
  `ewald_ion_ion_energy` over the same atom pairs within the real-space cutoff
  and the same reciprocal vectors (`V_ion.ewald.ewald_ion_ion_terms` evaluates
  both; both sums keep every image a pair or a grid point reaches, for any cell
  shape, so an atom that crosses a cell face and is translated back changes no
  energy);
- core correction: `dV sum_r V_xc(r) rho_c'(|d|) d/|d|` with the potential itself.
  PARSEC's `potshift` (`Zero_Average_Vxc`, the default of a 3-D cell without
  spin) removes the average `c` of `V_xc` from the potential of the loop; with
  that potential the term gains `c dQ_c/dR`, the derivative of the grid sum of
  the core density (5 to 7 % of the force of a Ge atom 0.01 to 0.1 bohr off
  its site). The average is added back (`PARSEC_CORE_FORCE_XC=auto`; `shifted`:
  the former forces). PARSEC 1.5 forms the term in G-space, where the average
  drops out;
- radial functions are differentiated as the energy samples them (slope of the
  linear interpolant of `r V`, of the densities and of the projectors, or the
  spline derivative). `PARSEC_PERIODIC_FORCE_SLOPES=tabulated` takes for the
  projectors Fortran's `dvw` table instead, which reproduces Fortran's nonlocal
  forces but is not the derivative of the sampled projector (the two differ by
  up to 1e-2 Ry/bohr on Si cells at h = 0.6 bohr; the forces of an isolated
  system use the table);
- next to an atom the projector *values* and *gradients* follow two rules
  (`PARSEC_PROJECTOR_CENTER=auto`). Within the first radial sample `rs(2)` a
  projector is `w(rs(2)) Y_l(d/rs(2))`, `nonloc.F90`'s
  `rnorm = max(|d|, rs(2))`. Its gradient deliberately departs from PARSEC
  1.5: within 3e-3 bohr of the atom it is evaluated at that distance, on
  every route (both sets of routines of a cell, clusters, wires, slabs, the
  double grid, spin-orbit, LDA+U; not the `tabulated` slopes, which
  reproduce Fortran). A table whose first sample lies at or beyond 3e-3 bohr
  (no table of the package with projectors; coarse test tables) keeps the gradient
  of `tip`: the polynomial `w(rs(2)) Y(d/rs(2))` of the energy then covers
  every realistic step. The reason is the tables: they are
  not `c r^l` next to the origin (POTRE: flat out to 7e-6 to 2.7e-4 bohr; FHI:
  the first eight samples, up to 2.5e-3 bohr, 25 to 48 % above `c r` for a p
  channel), so the value `nonloc.F90` forms at the point, `w(rs(2)) dY/du /
  rs(2)`, is the derivative only for displacements within `rs(2)`: 19 times
  the slope of a p projector on a POTRE table and 1.48 times on an FHI table.
  With the gradient at 3e-3 bohr the force of an atom on a grid point is the
  slope of the SCF energy over the steps of a relaxation: 64 carbon atoms,
  POTRE, both sets of routines, central difference at 1e-3 and 2e-3 bohr
  within 5.5e-8 and 2.1e-7 Ry/bohr (`tip`: 4.1e-3 off); with a 2x2x2 mesh
  1.8e-8 and 6.9e-8 (1.3e-3); C FHI at 2e-3 bohr 1.8e-7 (1.1e-4);
  `Si_disp_k222` 2.3e-9 and 9.2e-9 (7.9e-6); Ge8 1.7e-8 and 5.4e-8 (4.2e-4).
  The two sets of routines agree within 1e-9 Ry/bohr, and the energies are
  those of `tip` to round-off. A central difference with a step inside the
  part of the table that is not `c r^l` (below 1.5e-4 bohr on carbon POTRE,
  1.3e-3 on carbon FHI) follows that part instead, and differs from the force
  by up to 6e-4 Ry/bohr. `PARSEC_PROJECTOR_CENTER=tip` keeps the gradient at
  the point (the first form of `auto`, `nonloc.F90`), `off` the rule of before
  it, under which the harmonics took the true direction down to 1e-6 bohr and
  a grid point within round-off of an atom got `w(rs(2))/1e-6`. The LDA+U
  d tables follow `c r^2` from their first sample on within 1e-5, so for them
  the rule changes nothing measurable (it keeps every route on one rule). The
  local term had an error of the same family within the first radial sample:
  the padded spline of the local potential has a slope of its own at the atom
  (0.42 Ry/bohr for carbon FHI, 3.6e-3 carbon POTRE, 0.035 silicon FHI, below
  1e-4 for Ge and Ag POTRE) where the potential has none, and the
  Fortran-derived routines took that slope within `rs(2)`, while the default
  routines took none there (`forcloc.f90`'s skip), where the potential has
  `2 a r`: an atom 1e-6 to 5e-4 bohr from a grid point had a local force up to
  9e-4 Ry/bohr (carbon FHI) off the slope of the energy over steps beyond
  `rs(2)`. Since 2026-11-22 (`PARSEC_SPLINE_FORCE_CENTER=auto`) every route
  takes the slope of the smooth behaviour within `c = 2 rs(2)` of an atom,
  `s(c) d/c` with `s` the route's own slope at `c` (both routine sets of a
  cell, clusters, wires, slabs, readable, fast and device sums; `tip` keeps the
  former forces), so tables sampled down to the origin keep their own slope
  beyond `2 rs(2)` (2.6e-5 bohr for carbon POTRE). At fixed density (CPU; C, O,
  N, B, Si, Mg FHI and C, Ge, Au POTRE tables, clusters and the Fortran-derived
  routines, an atom 1e-11 to 3e-3 bohr from a grid point) the local term then
  follows the central differences of the local energy at steps of 1e-3 and
  2e-3 bohr: summed worst error 5.69e-3 Ry/bohr against 9.59e-2 under `tip`.
  The radius `max(2 rs(2), 1.5e-3 bohr)` of the first form of the rule
  (5.50e-3) departs from the own slope of the tables sampled to the origin and
  made the carbon POTRE cube worse in the SCF (5e-4 bohr from a grid point
  1.57e-5 / 2.65e-6 Ry/bohr off the slope of the energy at steps of 1e-3 / 2e-3
  bohr, 5.25e-6 / 1.31e-5 with its own slope). On A100, at steps of 1e-3 / 2e-3
  bohr: carbon FHI cube 1.1e-3/8.9e-4 -> 2.0e-4/2.3e-5 Ry/bohr 1e-4 bohr from
  the point, 7.8e-4/8.6e-4 -> 2.1e-4/1.2e-4 at 5e-4, 2.5e-4/1.4e-4 ->
  3.5e-4/2.4e-4 at 1e-3 (worse, just inside `rs(2)`); silicon FHI
  3.2e-5/3.1e-5 -> 3.6e-6/3.1e-6 at 1e-4 but 4.9e-6/2.6e-6 -> 1.2e-5/1.0e-5 at
  5e-4; oxygen FHI (MgO) 2.1e-3/2.0e-3 -> 5.2e-5/4.6e-5 at 1e-4, 1.5e-3/1.6e-3
  -> 2.9e-4/2.4e-4 at 5e-4 (the row of `PARSEC_SPLINE_FORCE_CENTER` in
  `acceleration/README.md` lists every case). What remains next to a grid point
  is not the local slope, and is open. The gap that both steps show lies in the
  projector values near the origin (nonlocal): with the local potential
  replaced by its smooth form within `c` in the energy itself, the carbon FHI
  cube 5e-4 bohr from a grid point keeps 1.16e-4 / 1.20e-4 Ry/bohr (constant in
  the step, linear in the distance), and the step-1e-3 feature of a carbon wire
  with an atom on a grid point is unchanged by that smoothing but moves with the
  projector values of `PARSEC_PROJECTOR_CENTER=off`; for the carbon POTRE table
  the projector values replaced by their smooth form within 3e-3 bohr took the
  gap 1e-4 bohr from the point from 7.6e-5 to 2.3e-6 Ry/bohr. The orbitals
  respond to such features at first order, so no force of the structure has the
  slope of the energy over steps that leave the point. The difference between
  the two steps next to the point is the padded interval of the local table,
  which a step of 1e-3 bohr resolves (8.6e-5 Ry/bohr at 5e-4 bohr on the carbon
  FHI cube, 3.8e-6 with that smoothing). The silicon FHI cube is worse than
  under `tip` because `tip`'s local error (1.1e-5 at fixed density) had
  cancelled a residual of about -1.1e-5 outside the local term; the oxygen FHI
  residual is not analysed. For carbon, steps of 1e-3 and 2e-3 bohr also differ
  by up to 2.5e-5 Ry/bohr, since `V'(r)/r` of the tables changes from -813 to
  -796 Ry/bohr^2 between 1e-3 and 2e-3 bohr;
- the self-consistency correction of `forcloc.f90` is added (Fortran's periodic
  routine has none; it vanishes at convergence); LDA+U (J = 0) and TS-vdW forces
  at Gamma as for clusters (since 2026-11-21 the LDA+U forces and relaxations of a
  cell whose SCF takes the sectors keep them, `PARSEC_LDAU_SECTOR_FORCES=auto`: the
  LDA+U term of every atom from the sector bases, averaged over the full group in
  weighted sectors, the total and the moves averaged over the group). The
  LDA+U energy follows `upot.F90`: the d projectors are cut hard at the radius
  that holds 95 % of the norm, where `R_d` is still 11 % of its peak for Ag, and
  `R_d` is interpolated linearly. So the energy steps when a grid point crosses
  that radius (Ag4 with U = 4 eV, atom 0 moved along x: -4.0e-4, +2.0e-4 and
  +4.0e-4 Ry at the predicted positions), and the force has kinks of about 7e-4
  Ry/bohr where a point crosses a knot of the table (rms 2.4e-4 Ry/bohr about a
  quadratic fit over 2e-3 bohr, against 1.9e-6 without U). Between those events
  the forces are the derivative of the energy (frozen orbitals within 5e-10
  Ry/bohr), but a central difference across a cut is not: 5e-2 Ry/bohr at a step
  of 2e-3 bohr is one crossing. The cut and the linear interpolation follow
  `upot.F90` on purpose, so that LDA+U energies stay those of PARSEC 1.5; the
  steps and the kinks are a property of that convention. A finer grid makes
  each step smaller (a step is the term of the one grid point that crosses the
  cut), and a check of the forces against differences of the energy needs
  steps that cross no cut;
- with fractional occupations the forces are the derivative of the Mermin free
  energy `E - TS`, not of the printed total energy (as in Fortran).

Symmetrization (`symm_forces.f90`): where the problem carries Fortran's space
group (a Monkhorst-Pack mesh with `Monkhorst_Pack_Shift`, or since 2026-11-18
without it, `Symmetrize_Charge_Density`, or a weighted `Kpoint_List` with
forces) the forces are averaged over its operations,
non-symmorphic ones included (`T(r) = U (r - A t)`: the image of an atom is
found modulo lattice vectors as `symm_forces` finds it, the force turns with
`U`), and a relaxation averages its positions over the same operations with
their translations. Otherwise the point operations about the origin that map
the lattice, the grid, the atoms (modulo lattice vectors) and the k-points onto
themselves are used, as before. Molecular dynamics and manual movements use
none, as in PARSEC (`parsec.F90` sets no symmetry for them: on a
Monkhorst-Pack mesh their wedge is that of time reversal alone); no torque
is reported. On the default routines a `Kpoint_List` of scalar or
collinear-spin orbitals with forces takes, since 2026-11-22
(`PARSEC_KPOINT_LIST_FORCE_GROUP=auto`), the operations of the space group
that map the grid and the weighted list (its classes `{k, -k}` modulo the
reciprocal lattice) onto themselves, as the same points given as a mesh do:
the forces of a moved hcp Au2 cell given as the list of its 4x4x2 mesh equal
those of the mesh within 1.6e-10 Ry/bohr (4.2e-5 in plane without the
average, `off`).

An irreducible k-point wedge gives the nonlocal forces of its own points
only; summed over the zone they are `(1/N) sum_g U_g F(g^-1 a)` of the wedge
forces, which `Forces.periodic.reduced_kpoint_operations` applies to the
k-dependent terms whatever the `symmetrize` flag (Fortran gets the same sum from
`symm_forces`). A relaxation whose `Move_Flag` the group does not respect
would leave the group the wedge stands for, but not the subgroup of the
operations that map the fixed atoms onto fixed atoms: since 2026-11-21
(`PARSEC_MOVE_FLAG_SPACE_GROUP=auto`) the problem carries that subgroup, found
from the atoms with the fixed ones as a species of their own (the group of
the atoms holds one translation per rotation, which in a supercell need not
keep a fixed atom) and kept to the operations that also map the
finite-difference operators of the grid onto themselves (on an oblique grid
the others change the energy at the order of the discretization), and the
mesh, the density average and the forces take it; the grid stays that of the whole group, and a
subgroup whose translations it does not take is refused as before. Since
2026-11-22 the spinors of non-magnetic spin-orbit coupling on a mesh take the
subgroup too (`PARSEC_SPINOR_MOVE_FLAG_SUBGROUP=auto`; `off`: refused as
before); non-collinear magnetism and spin-polarized spin-orbit coupling stay
refused. A weighted
`Kpoint_List` (a wedge of the input's group) is still refused with such flags,
and so is molecular dynamics with one; `off` refuses every such relaxation, as
before. A Gamma-point relaxation with `Symmetrize_Charge_Density` and such
flags averages the density over the subgroup; where none is kept (spin-orbit
coupling, non-collinear magnetism, or a subgroup whose translations the grid
does not take) it averages nothing, since the moved structures leave the
group (`off`: over the whole group, as before).

Collinear spin with k-points runs `run_scf_kpoints_spin_polarized` keeping its
orbitals and potentials; the forces sum the nonlocal term over both channels
(one electron per orbital) and the local terms take the total density and the
channel averages, as `forpbc.f90`. With `Double_Grid_Order > 1` the projector
gradients are those of the fine points with their images and the Bloch phase
of the coarse point (`nonloc.F90`).

Relaxation and dynamics use `Relaxation.periodic_backend`: `run_scf` at Gamma
(`run_spin_scf` with collinear spin, started from the periodic spin densities),
`run_scf_kpoints` keeping its orbitals with k-points (with collinear spin
`run_scf_kpoints_spin_polarized`, started from the two channel potentials of
the structure before), and
`SCF.pbc.update_periodic_geometry` after every move (local potential,
projectors, LDA+U projectors, densities, Ewald energy). Atoms that leave the
cell are translated back (`back_to_cell`); the optimizer and the integrator keep
the unwrapped coordinates. The next SCF loop starts from the last mixed
potential with the ionic part replaced (and at Gamma from the saved orbitals).

On a GPU the accelerated command line (`main.py`) runs the forces,
relaxation and dynamics of a bulk cell on Fortran's orthorhombic grid at
Gamma without spin itself, and hands the other periodic inputs (k-points,
collinear spin, LDA+U, TS, wires and slabs, the general grid, ...) to the
reference solver, whose k-point eigenproblems run on the device workers.
For every bulk cell on the orthorhombic grid whose forces are asked for or
whose atoms move, on either route, the ionic fields and their forces are
built on the device (`acceleration/V_ion/periodic_device.py`,
`acceleration/Forces/periodic_device.py`):

- the readable builders sum the local potential and the atomic densities
  over every lattice image out to the end of the radial tables (118 bohr for
  the POTRE files: about 6000 images of each atom of an 8-atom silicon cube,
  70 s for each of the two fields at h 0.45 bohr on the host of an A100 node,
  before the first SCF loop and after every move); the device sums stop where
  the terms vanish: the
  short-ranged piece of the Ewald split `V_a + 2 Z erf(eta r)/r` (with the
  `eta` of the FFT potential of `periodic_setup.py`) where the table of
  `r V_a` is the Coulomb tail and `erfc(eta r)` is below the Ewald tolerance,
  an atomic density where it is below `1e-16` of its peak. The fields equal
  the readable ones to round-off (potential 1e-13 Ry, densities 5e-16);
- the local, core-correction and self-consistency forces are the exact
  derivatives of those fields (short range in one CUDA kernel, the
  reciprocal part from one FFT of the density), so the forces remain the
  derivative of the energy computed;
- at Gamma the nonlocal term is contracted with the orbitals where the
  eigensolver keeps them, with the projector gradients of the lattice images
  (`acceleration/Forces/nonlocal_device.py`); the moved atoms replace the
  projectors of the device Hamiltonian in place, and the next SCF loop
  continues from the saved subspace, as for a cluster.

The G-space local potential keeps the readable fields and forces.
`Cubic_Spline` species keep the fields of the host preparation
(`periodic_setup.py`); their local forces are the derivatives of those fields
(`acceleration/Forces/periodic_spline.py`, since 2026-11-07: the short-ranged
terms on the host over the balls of the potential, the reciprocal part from
one FFT, the self-consistency and core terms in the CUDA kernel above). Grid
points within `2 rs(2)` of an atom take the slopes of the
smooth behaviour of the potential and of the core density
(`PARSEC_SPLINE_FORCE_CENTER=auto`, since 2026-11-22, above); under `tip` the
points within round-off of an atom (`min(rs(2), 1e-6 bohr)`) are left out of
the spline slopes (the default of 2026-11-21): the padded spline has a
slope of its own at `r = 0+` (3.6e-3 Ry/bohr for the carbon POTRE table, 0.42
and 0.73 for the carbon and oxygen FHI tables), so the energy has a cone at
the atom, and an atom within round-off of a grid point (half the atoms of the
`make_diamond.py` cubes, 4e-12 to 2e-11 bohr from one) took the slope of the
cone along the direction of the round-off, up to 1.5e-5 Ry/bohr (POTRE) and
2e-3 Ry/bohr (FHI) that changed when the atom moved by 1e-11 bohr; under `tip`
such a point gives the symmetric value, zero, as a central difference of the
energy does, and every other point keeps the derivative of the energy at the
point, cone included.
`PARSEC_SPLINE_FORCE_CENTER=fortran` leaves out every point within the first
radial sample, the rule of `forcloc.f90` and of the other local forces here,
which is not the derivative of the energy between 1e-6 bohr and `rs(2)`
(up to 1e-3 bohr for FHI tables).
The readable routine they replace summed every grid point of every image within
`erfcinv(1e-26)/eta`: 34 s against 3.5 s for the local forces of the 8-atom cube
at h 0.25 on an A100 node, 892 s against 5.2 s for the 64-atom cube at h 0.30.
On Fortran's general (oblique) grid, whose potential takes the same split by
default (`PARSEC_PERIODIC_OBLIQUE_SPLIT`: `Cubic_Spline` species since
2026-11-07, linear POTRE and FHIPP tables since 2026-11-22), the same routine
differentiates it over the lattice balls (the density terms in a lattice
kernel on the device; within 2.2e-8 Ry/bohr of fourth-order differences of
the energy of the potential for a distorted fcc Si16 cell at h 0.30, 4.5e-7 of
the readable forces for a distorted Si2 cell). A plain single point (no forces, no moving atoms) keeps
the readable fields bit for bit on both command lines. The projector
gradients of the forces find their grid points with the grid-ball query of
`periodic_setup.py` (the same matrices). Si8 cube (h 0.64 bohr, laptop RTX
5070): forces equal to the reference solver within 1.3e-12 Ry/bohr, a BFGS
run of four structures within 1.2e-10 bohr (7 s against 272 s). One later
ionic step on an A100 (h 0.45 bohr, Gamma): 149 s before, 0.65 s now for the
8-atom cube; 1473 s and 5.3 s for the 64-atom cube; with 8 k-points the
8-atom cube goes from 151 s to 9.2 s, nearly all of it the k-point SCF
loop.

Validation against Fortran PARSEC (Si2 fcc and Si8 cube at Gamma, Si2 with 8
k-points, spin-polarized Na3, BFGS relaxation and Verlet dynamics of Si2): with
Fortran's G-space local potential emulated and Fortran's projector slopes,
the total forces agree to 2.5e-6 Ry/bohr (Si cells; 6.5e-6 for spin-polarized
Na3) and the relaxation and dynamics trajectories to 1e-6 bohr; the report
The report of the parity round has the numbers. The same comparison
for the space group and the wedge (fcc Si2 with eight operations, half of them
with the translation (1/4, 1/4, 1/4), and an hcp cell, both on reduced
Monkhorst-Pack grids), collinear spin with k-points and the double grid gives
1.3e-7 to 1.8e-6 Ry/bohr, and symmetric BFGS relaxations take Fortran's
structures; numbers in the report of the parity round.

## Spin-orbit coupling and non-collinear magnetism

`Forces/spin_orbit.py` gives the forces of the spinor drivers
(`run_self_consistent_soc`, its collinear-spin form, `run_self_consistent_noncollinear`
and the k-point forms of all three): the ion-ion and local terms with the
total density and the spin-independent part of the potentials, the
Kleinman-Bylander term on both spinor components (one electron per spinor),
and the spin-orbit term, the derivative of the `lzsz`/`lsxy` energy
`sum_n f_n <psi_n|H_SO|psi_n>` with the gradients of the `v_ion`/`v_so`
projectors (Fortran's `forcso` of `soforc.F90`). Every term is the derivative
of the energy the drivers compute (finite differences: 1e-9 Ry/bohr). The
Zeeman terms of collinear spin and of the non-collinear field depend on the
atoms through the density only.

Fortran's spin-orbit term is not the derivative of its energy:
`pseudo.f90` tabulates `dvr_ion` as a value (`(l+1) vlj(l) + l vlj(l+2)`) and
`dvr_so` without the factor `sqrt(h^3)/(2l+1)` of `vsor`, `forcso` multiplies
both overlaps of each product by the KB sign (which then cancels) and
`flsxy` pairs the same `m_l` in both spin components (the Hamiltonian's
`lsxy` shifts it). On AuH (`examples/0d_AuH`) the local and Kleinman-Bylander
tables equal Fortran's, the spin-orbit term on Au is +0.0053 Ry/bohr here and
-0.0580 in Fortran, which a line-by-line port of `soforc.F90` with those
tables reproduces (-0.0577). `forcnloc.f90` sums the Kleinman-Bylander term
over the up components only (times four): equal to both components for
Kramers pairs, not with a magnetization.

Relaxation and dynamics use `Relaxation.spinor_backend`. As in `parsec.F90`
every structure after the first starts from the last mixed potential of the
structure before with its ionic part replaced (`vnew - vion_old + vion_new`
for every spin component; a non-collinear loop keeps its exchange-correlation
field, the direction of the magnetization), and the isolated spin-orbit loops
(with and without collinear spin) also filter the spinors of the structure
before, as PARSEC filters its saved wavefunctions; the k-point and
non-collinear loops keep no subspace between loops and start their first
eigenproblem from random vectors. The first structure of a `Restart_Run` from
a spin-orbit `parsec.dat` starts from the potentials of the file as the single
point does. On AuH (manual movements of 0.03-0.06 bohr) the second and third
structures take 12 and 15 iterations (16 from the atomic densities; PARSEC 11
and 13); after a large BFGS step (0.7 bohr) the warm start costs more
iterations than the atomic densities, in PARSEC as here. The TS-vdW term of a
converged spinor loop is added as for a scalar calculation. The forces are
symmetrized as for a scalar calculation (PARSEC
does so with spin-orbit coupling and non-collinear magnetism too; a magnetic
structure whose moments break the atoms' group needs `Ignore_Symmetry`). The
perturbative correction (`SO_PSP` without `SO_from_scratch`) has spinor
eigenvalues but no spin-orbit total energy here, and PARSEC flags its forces
as inaccurate: forces, relaxation and dynamics are refused with it. With
`Double_Grid_Order > 1` the spin-orbit term uses the gradients of the
double-grid spin-orbit projectors (the fine-point sum of `nonloc.F90`, each
fine sample with the slope of its radial interpolant). With `Polarizability`
and `Skip_Force: false` the forces are those of the last of the seven field
loops (field along -z), as `parsec.F90` computes them after `polar.f90`
(the accelerated driver always solves that loop, since 2026-11-19 in the
sectors of the operations of the atoms that keep the z axis:
`PARSEC_POLARIZABILITY_SYMMETRY`, `acceleration/README.md`).
Refused: forces with LDA+U and k-points (scalar or spinor; PARSEC's LDA+U
forces are zero).

## Wires and slabs

`Forces/wire_slab.py` follows `forcloc_wire.f90`/`forcloc_slab.f90` (the
short-range local term, the force on the Gaussian charges in the Hartree field,
the core and self-consistency terms), `forceion_wire.f90`/`ewald_slab.f90`
(ion-ion, with the charged sheets of a slab) and `forcnloc.f90` (projectors
with their images along the periodic directions, Bloch phases for k-points).
With collinear spin the local terms read the total density and the channel
averages, the nonlocal term sums both channels; with spin and k-points it uses
the orbitals the spin k-point loop keeps (`keep_wavefunctions`). LDA+U forces
(J = 0) are added as for clusters (Fortran's are zero). The two terms of the
Gaussian compensation charge (its density in the Hartree field and the slope of
`2 Z erf(r/sigma)/r`) are evaluated at every grid point
(`PARSEC_WIRE_SLAB_GAUSSIAN=auto`), a departure from `forcloc.f90`, which skips
every term of a point within the first radial sample `rs(2)` although the
Gaussian density is largest at the atom (`off`: the former forces). The terms
of the radial tables take the slopes of `PARSEC_SPLINE_FORCE_CENTER` (since
2026-11-22 the smooth slopes within `2 rs(2)`, which reach the points within
`rs(2)`; the skip left 4.2e-4 / 2.6e-4 Ry/bohr at steps of 1e-3 / 2e-3 bohr
on a carbon FHI wire 1e-3 bohr off a point, 2.8e-4 / 1.0e-4 now). Two further departures from PARSEC 1.5 make the
forces the derivative of the energy (since 2026-11-22): the potential `V_G`
of the Gaussian charges is solved to a relative residual of 1e-12
(`PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE=auto`) instead of the Hartree tolerance of
the run, whose error enters the energy at first order and moves with the atoms
(`-dV rho . V_G` 6.8e-6 Ry from the converged value on an hBN slab: the 7.5e-5
Ry/bohr left in F_z of B, 4.7e-7 to 1.0e-5 now at 0 to 2e-3 bohr from a grid
point; energies move by that error, +8.5e-6 Ry there); and the force of the
Gaussian charges takes the analytic slope of the Gaussian,
`dV sum V_H dn_g/dR` (`PARSEC_WIRE_SLAB_GAUSSIAN_FORCE=auto`), instead of the
finite-difference gradient of `V_H`, which does not resolve the Gaussian on
coarse grids (2.5e-4 Ry/bohr at h 0.4 for hydrogen; an hBN slab at h 0.4 meets
the slope of the energy within 6e-8 instead of 7.9e-5).

Symmetrization (`symm_forces.f90`): over the space group the input reader
finds for the wire or slab (`symmetries.f90` with `alatt` = `latt_vec` and
`4 rmax` along the confined directions), non-symmorphic operations included;
operations whose translation moves the cylinder or slab off itself are dropped
(PARSEC applies them), and so are those that do not map the charged sheets of
a slab onto sheets of the same charge (`Grid/space_group.py`,
`sheet_preserving_operations`, since 2026-11-21; PARSEC searches from the
atoms alone, so with bandstruc, spin-orbit or an NSCF run z -> -z averaged
the density and the forces of an asymmetric capacitor and erased its
dipole; A100, a square carbon slab with bandstruc and sheets of +0.05 e at
z = -4 bohr and -0.05 e at z = +4 bohr, 3 x 3 x 1: D4h 16 -> 8 operations,
3 of 5 k-points, the energy and the dipole of the whole mesh within 4e-11 Ry
and 2e-10 e bohr, where the 16 operations had been 6.7e-3 Ry off with no
dipole). A slab has no group in PARSEC unless bandstruc,
spin-orbit or an NSCF run turns its symmetry on; here a slab with a
Monkhorst-Pack mesh of scalar or collinear-spin orbitals and a native LDA or
GGA also takes the operations of its group that keep its grid and its mesh
and are signed permutations of the lattice directions
(`PARSEC_SLAB_MESH_SYMMETRY=auto`, since 2026-11-19; `full` also the three-
and six-fold rotations of a hexagonal slab: the mesh reduced to their wedge,
the forces averaged over them; a `Move_Flag` that breaks the group keeps the
former route, charged sheets that are not symmetric about z = 0 drop the
operations that reverse z); molecular dynamics, manual movements and
Polarizability have none. The averaged forces also complete a k-point set
reduced to the irreducible wedge; as for a bulk cell, the movements such a
wedge cannot follow are refused: molecular dynamics and manual movements with
an unequally weighted `Kpoint_List` when the structure has a group (PARSEC
switches the group off for them; a slab without bands, spin-orbit coupling or
NSCF has none, its list is a plain quadrature), and a relaxation with a
reduced set whose `Move_Flag` a group operation does not respect.

Relaxation and dynamics use `Relaxation.wire_slab_backend` (the periodic
backend with `SCF.wire_slab.update_wire_slab_geometry`): after every move
`back_to_cell` (periodic fractional coordinates into `[-1/2, 1/2]`, the
confined ones kept), `isolat` (an atom outside the cylinder or slab stops the
run, error 291), then the local potential, projectors, LDA+U projectors,
densities and ion-ion terms through the same function as a fresh preparation
(bit for bit). With collinear spin and k-points the spin k-point loop starts
every structure from the two channel potentials of the structure before
(`vnew - vion_old + vion_new`, as the bulk cell).

Validation against Fortran PARSEC (the report of the parity round):
BFGS of a Na4 wire (with and without its D2h group), of an H2 slab (Gamma and a
2x2 shifted grid) and of a three-H slab with collinear spin, of polyacetylene
with 4 shifted k-points; Verlet dynamics of the Na4 wire; manual movements of
`tests/chain_Na`: positions agree to 1e-5 bohr, energies and forces to the
printed digits (LDA; PBE cases differ by the known EASYPBE potential).

## On GPUs and several nodes

Nothing leaves the devices that a single point keeps there.

- **Moved atoms**: the projector matrices of every symmetry sector are
  replaced in place on the device operators, with their replicas on the other
  devices of a shared sector basis. Stencils, potential buffers, recorded
  filter graphs and the sector subspaces stay.
  The local ionic potential and the atomic densities of a moved cluster are
  summed on the GPU (the default `PARSEC_IONIC_BACKEND=auto` since
  2026-11-19), at one point per orbit of the full point group where the moved
  atoms are its images to 1e-10 bohr (`PARSEC_IONIC_MOVED_SUMS=orbits`;
  `moved_ionic_sum_grid` of the backend says what ran).
- **Warm start**: each sector solver keeps its subspace and marks it as the
  start of a new SCF loop.
- **Forces**: the nonlocal term is contracted where the orbitals are, sector
  by sector and device by device, and only one small array per projector is
  downloaded. The local term runs in a CUDA kernel on the devices of the
  process that holds the density (in a run on several nodes, the four GPUs of
  the root node), each of which sums a part of the atoms. Where symmetry
  sectors are solved the sums run over the symmetry wedge of the grid and the
  group operations complete them, which costs as many times less as the group
  has operations and changes the forces by round-off only;
  `PARSEC_LOCAL_FORCE_SUMS=full` sums over the whole grid as before. Weighted
  sectors (a Td or Oh cluster solved in the sectors of D2d or D4h) can sum over
  one point per orbit of the full group instead of the wedge of the solved
  subgroup, with the fields averaged over the full group, which their
  symmetrized total does in any case (`PARSEC_LOCAL_FORCE_SUMS=orbits`,
  opt-in since 2026-11-19: three times fewer points for Td in D2d, the same
  forces within 4e-14 Ry/bohr, but no faster, since the average of the fields
  on the host costs what the points save: on one node of 4 A100 the local
  forces of the nanodiamonds k06 and k08 took 0.070 and 0.118 s against 0.058
  and 0.090 s on the wedge, warm; the default `wedge` keeps the wedge of the
  subgroup).
- **Projector gradients**: once the atoms of a structure are in place, a host
  thread samples the gradients of its projectors and reduces them to the
  sectors of the process while the devices solve its SCF loop; the forces
  then take them (`PARSEC_CUPY_FORCE_PREFETCH=on`, the default; `off` builds
  them when the forces are asked for). Nothing reaches a device before the
  forces, and the forces are the same to the last bit.
- **Several nodes**: the rank of each node rebuilds the projectors of its own
  sectors (command `geometry`), restarts its solvers (`new_scf_loop`) and
  contracts its part of the nonlocal force (`forces`); the root sums them.
- **Projectors of one atom per orbit** (a point group with weighted sectors, Td or
  Oh, and since 2026-11-18 the abelian sectors of clusters and Gamma cells,
  `PARSEC_SECTOR_PROJECTOR_ORBITS=auto`; with LDA+U since 2026-11-21, `PARSEC_LDAU_PROJECTOR_ORBITS`): the sector operators hold the projectors of one atom of each orbit of the
  group of the sectors, scaled by the square root of the orbit size, and since
  2026-11-18 the projector gradients of the forces are compressed by the same rule
  (`compress_gradients_for_sectors`): the terms belong to the representatives,
  and the average over the full group of the reduction
  (`acceleration/Forces/sector_symmetry.py`, translations of a cell included)
  gives every atom of an orbit its term and the partners of the weighted levels
  theirs, whatever symmetrization is asked for. Before, a structure whose atoms
  with projectors formed such orbits stopped in the contraction (a Td cluster
  with forces in its sectors, a regression of the compression of 2026-11-16);
  Ge5 in Td against the full grid: within 2e-6 Ry/bohr
  (`test_forces_geometry.TetrahedralSectorForceTests`). Since 2026-11-19 the
  two channels of a spin-polarized run take the weighted sectors too
  (`PARSEC_ORBITAL_GROUP_SPIN`): each channel contracts the states its sectors
  solved with their occupations times their weights
  (`SpinChannelResult.sector_states`), and the same average completes them
  (GeH4 and its anion in Td, O2 in D4h: the forces of the full grid within
  2e-5 and 1e-7 Ry/bohr, `test_weighted_inputs.py`).
- **Periodic cells in sectors** (a bulk cell at the Gamma point on the default
  routines, solved in the sectors of its point group, `PARSEC_PERIODIC_SYMMETRY`):
  - *Forces*: the nonlocal term as above, the Ewald and local terms from the
    density, which the SCF has averaged over the full group G of the reduction,
    and the total averaged over G, translations included (`tau = -R^T t h`, atom
    images modulo the lattice). The completion is exact also for half-cell
    translations of supercells and the non-symmorphic operations of Fd-3m:
    displaced Si 2 x 2 x 2 cells with G of 96 and 192 against the full grid within
    7e-10 Ry/bohr (`test_periodic_sector_forces`, which also proves the identity
    on the host to round-off for any H-invariant density matrix of sector states).
  - *Projector gradients of the representatives*
    (`PARSEC_SECTOR_GRADIENT_ATOMS=representatives`, since 2026-11-18): where the
    sectors hold the projectors of one atom per orbit, the gradients are built for
    those atoms only (`atom_indices` of the gradient builders); the forces are the
    same bit for bit. Diamond 6^3 at Gamma: 63 of 1,728 atoms, gradient phase
    5.4 -> 0.23 s.
  - *Local terms of the orbits* (`PARSEC_PERIODIC_SECTOR_LOCAL_FORCES=orbits`,
    since 2026-11-18): the local ionic and core-correction forces are equivariant
    on the G-invariant density (`F_ga = R_g F_a`), so they are summed for one atom
    per G-orbit and turned onto the orbit; up to eight other atoms are summed in
    the same call to check the equivariance, and
    `phase_seconds['local_ewald_asymmetry']` (Ry/bohr, not seconds) reports how far
    the average over G moved the Ewald and local terms (round-off, ~1e-11, on an
    invariant density; a large value means the density was not G-invariant: use
    `all`). Diamond 20^3 (64,000 atoms, 446 orbits) on one A100: 503.6 -> 4.3 s,
    the same forces within 1.6e-11 Ry/bohr.
  - *Reciprocal series* (`PARSEC_PERIODIC_FORCE_SERIES=stream`, every periodic
    local force): half of the reciprocal box from the real transform of the
    density, weighted and contracted slab by slab, with a host route when the
    device is short of memory; at 360^3 0.70 instead of 3.71 GiB and 3 x faster,
    the same sums to ~1e-13.
  - *Moves*: the atoms of a cell in sectors move (`driver.update_geometry`,
    `run_relaxation`): `check_sector_atoms`
    (`acceleration/Forces/sector_symmetry.py`) requires the moved atoms to keep G
    and the orbits of H to 1e-6 bohr (a move off the group raises `ValueError`
    naming the operation; a group that `PARSEC_SYMMETRY_ATOM_TOLERANCE` took for
    atoms off it by more than that is refused before the SCF of a run with
    forces, since 2026-11-19, `sector_force_tolerance_refusal`, of clusters too),
    every route then shares one tail: device operator,
    sector operators (compressed as at the preparation), the rebound reference,
    the prefetch of the gradients. An orbital-only system of an MPI sector rank
    takes the projectors of the moved atoms alone (`SCF/pbc.py`,
    `update_periodic_geometry`). One sector of the identity
    (`PARSEC_CUPY_SINGLE_SECTOR=on`) now takes the moved projectors too; before
    2026-11-18 it kept those of the first structure (a wrong SCF after a move).
    A relaxation in sectors averages forces and moved positions over G
    (`with_periodic_operations` gives G of the sectors): displaced Si8 and Si 2 x 2 x
    2 (G of 192 with half-cell translations), three BFGS moves against the full
    grid within 1e-8 bohr and 1e-10 Ry.
  - *Command line* (`PARSEC_PERIODIC_SECTOR_FORCES=auto`, since 2026-11-18):
    single-point forces (`Skip_Force: .false.`) and relaxations by `bfgs` or
    `simple` (with `Relax_Restart`, `Potential_Extrapolation`,
    `Relaxation_SCF_Upscale`) keep the sectors, and `--symmetry on` is accepted for
    them; they keep them below the size floor of an SCF alone too
    (`PARSEC_PERIODIC_SECTOR_FLOOR`, since 2026-11-19: the full grid would not
    average forces and moves over the group), where an unpolarized cell of a few atoms
    solves fewer, larger sectors of a smaller subgroup of the same G since 2026-11-21
    (`PARSEC_SECTOR_OVERHEAD`, the forces and moves still averaged over G); `parsec.out` says `Forces of the periodic cell in the sectors of its
    point group: averaged over |G| operations; projectors of n of N atoms ...`.
    A `Move_Flag` fixing atoms keeps the sectors of the subgroup that maps the
    fixed atoms onto fixed atoms (`PARSEC_MOVE_FLAG_SUBGROUP=auto`), except in
    the relaxation of a spin-polarized cell (the full grid of both channels,
    `PARSEC_SPIN_SECTOR_FORCES`); on the Fortran-derived routines since
    2026-11-21 under `PARSEC_MOVING_SECTORS=auto` where the cell's basis reaches
    the floor of the sectors and their group has 4 operations or more, in one
    process (their forces are averaged over the operations of their own group
    and of the sectors that respect the flags; else, and under `off`, their full
    grid, as before; `on`: the sectors at any size). Molecular dynamics and manual
    movements keep the sectors of the group their structures keep under
    `PARSEC_MOVING_SECTORS=auto` ("Molecular dynamics" above); under `off`
    they keep the full grid. A `Move_Flag` fixing atoms under
    `PARSEC_MOVE_FLAG_SUBGROUP=off`, point charges, `Polarizability`, nscf,
    `Restart_From_Wfndat`, k-points beyond Gamma and the TRIM supercell keep the
    full grid (the Fortran-derived routines only under
    `PARSEC_FORTRAN_SECTOR_FORCES=off`, below), and so do bands under `PARSEC_PERIODIC_SECTOR_BANDS=off` (under
    `auto`, the default since 2026-11-18, a band structure keeps the sectors of
    its SCF); `Ignore_Symmetry` too unless `--symmetry on` is given; `off` puts
    every forces, relaxation or MD run of a cell on the full grid, bit for bit as
    before. A cell in the sectors of an Abelian group (every level solved; since
    2026-11-18 its projectors and their gradients are those of one atom per orbit,
    the force of every other atom its representative's turned by the operation
    that carries it there) has its forces and moved positions averaged over that
    group (`sector_symmetry.abelian_cell_operations`), so that the atoms it fixes
    stay on it. Steepest descent moves the components of every nonzero force by about
    `Min_Step`: on the full grid the round-off forces of the atoms that the group
    fixes move them (~1e-4 bohr in the tests), in the sectors they stay.
  - *Several nodes* (`benchmarks/mpi_full_scf.py`, `scripts/run_multi_gpu_scf.sh`):
    the same rule; the ranks compare the shape and a digest of the projector
    labels before they sum their force rows (a mismatch stops every rank), every
    rank moves the projectors of its sectors (command `geometry`), molecular
    dynamics, manual movements, fixed atoms and `off` run as one sector of the
    identity. Displaced Si8 and Si 2 x 2 x 2, two BFGS moves on 1 x 1, 1 x 4
    (`--rank-per-gpu`) and 2 x 1 ranks: the trajectory of `main.py` within 1e-13
    bohr. Inputs whose species use `Cubic_Spline` (all of `make_diamond.py`) put the
    forces of a Gamma cell on the Fortran-derived routines by default, which keep
    the sectors too since 2026-11-19 (below); `Use_Fortran_Routines: .false.` gives
    them the default routines.
  - Diamond on 16 A100 (one rank per node, perfect crystals, one relaxation step =
    SCF + forces): 6^3 9.6 s, 10^3 14.6 s, 20^3 (64,000 atoms) 164.7 s (SCF 118.5 s,
    forces 14.8 s), 22^3 (85,184 atoms; `PARSEC_CUPY_SECTOR_MARGIN=fixed` and
    `PARSEC_CUPY_FORCE_PREFETCH=off`) 304.1 s at 79.1 GiB on the fullest GPU.
  - *The Fortran-derived routines* (`PARSEC_FORTRAN_SECTOR_FORCES=auto`, since
    2026-11-19): a Gamma cell whose forces or relaxation land on those routines
    (`Use_Fortran_Routines: .true.`, or the parser's choice: `Cubic_Spline` forces,
    `Pseudopotential_Format: fhipp`, a core correction, `Restart_Run`, another
    mixer or eigensolver, a double grid, `Dynamic_Diag_Tol`) keeps the sectors of its
    point group, on the command line and in the runner (which reads no `parsec.dat`:
    it leaves a `Restart_Run`, and spin-polarized cells, to `main.py`). The terms are those of the
    routines (`Forces/periodic_device.compute_device_periodic_forces`): the Ewald
    sum, the local ionic, self-consistency and core-correction terms of their
    tables on the fields the sectors expand to the full grid (one atom per orbit of
    the group of the sectors, turned onto the others), the nonlocal term contracted
    in the sectors with the gradients of their projectors with lattice images and
    completed over G; the total and the moved positions are averaged over the group
    that the operations of the routines and of the sectors generate
    (`sector_symmetry.fortran_force_operations`). Displaced Si8 (Td and C2v),
    `Use_Fortran_Routines`, `Cubic_Spline`, `Restart_Run` (also with a relaxation),
    Broyden mixing, a double grid, MgO with the FHI pseudopotentials and the core
    correction of Mg, two BFGS moves: the full grid within 2e-6 Ry/bohr and 1e-6
    bohr. `off` keeps them on the full grid, bit for bit as before.
- **Spin-orbit coupling and non-collinear magnetism**: the accelerated entry point runs these inputs, when they ask for
  forces, with the reference solver and says so. Its pooled k-point solver (periodic and non-collinear inputs) hands the
  orbitals of a k-point to the force pass on the device that solved it: the overlaps with the projectors, the spin-orbit
  projectors and their gradients are sparse products there and only the `(atoms, 3)` result comes back
  (`Forces/spin_orbit.py` is written against the array module of the orbitals; it is tested on CPU arrays through a
  module that offers only CuPy's operations, not yet on a GPU).
- **Periodic cells (Gamma point, no spin)**: forces, relaxation and molecular dynamics run on the accelerated path.
  Moving the atoms replaces the projectors on the device operator and rebuilds the ionic potential, the densities and the
  Ewald terms from the folded positions (`update_periodic_geometry`); the Hartree solve of a periodic cell is the readable
  one. The nonlocal force is contracted on the device that holds the orbitals with the gradients of the periodic
  projectors, as for an isolated structure. The local ionic and core-correction forces
  (`acceleration/Forces/periodic_local.py`) replace the scan of the whole grid for every image and the grid-by-G phase matrix
  of the readable routine with a ball around each image of each atom and one FFT of the density, read at the reciprocal
  vectors of the Ewald series, with the Gaussian width chosen so that the series ends below the Nyquist frequency of the grid
  (the way the fast ionic potential is made). The sums are written against an array module: NumPy on the host, CuPy
  (with `cupyx.scipy.special.erf`) when the calculation ran on a GPU, which then forms the radial slopes, products and sums
  and the FFT; the balls themselves are generated on the host from index ranges. The forces are those of the readable routine
  to 1e-7 Ry/bohr (the splitting of the Ewald sum; 1e-13 for a table cut at 14 bohr). Not on this path: k-points (next
  item) and spin. Several nodes run the forces of a Gamma cell in the MPI runner since 2026-11-18 (above), and the Ewald
  ion-ion force of a cell of 1,024 atoms or more runs on the device since 2026-11-17 (`PARSEC_DEVICE_EWALD`); smaller
  cells keep it on the host.
- **Periodic cells with k-points**: the accelerated command line hands such an input to the reference driver, whose k-point
  solver puts one worker thread on each CUDA device (the output names the device under "Execution resources"). Its
  eigenproblems, the density of each k-point and now the forces run there: the nonlocal force of a k-point is contracted on
  the device that holds its orbitals (`acceleration/Forces/kpoint_nonlocal_device.py`: the overlaps with the complex Bloch
  projectors and their gradients are sparse products on the device, a block of orbitals at a time, and only the `(atoms, 3)`
  array returns; a k-point whose subspace was spilled to the host contracts there), and the local ionic and core-correction
  forces are the ball sums and one FFT of the density of the Gamma-point path, with CuPy when the k-point plan uses a GPU
  (the reference `periodic_local_forces` hands its sums to them once the acceleration package is imported). Not done: the
  row-distributed MPI k-point solver, which has no nonlocal force; and the Ritz subspaces of the k-points, which restart
  from scratch after an ionic move (the Gamma-point loop carries its subspace over), so a relaxation step costs a first solve.

The multi-node runner divides the symmetry sectors of the structure among its
ranks, one per node. A structure without point-group symmetry, a relaxation
whose movable flags break the symmetry, a `manual` run and molecular
dynamics have one sector, the identity. Launched with one rank, the runner
solves it on the GPUs of one node, which share its basis column by column
(`PARSEC_CUPY_SINGLE_SECTOR=auto`, the default; `on` gives `main.py` the one
sector of the identity too, whose basis the GPUs of the process share after a
CHEBFF first solve under `PARSEC_CUPY_SECTOR_PLACEMENT=size`, the default since
2026-11-19, and the first GPU holds under `cyclic`; `off` never). The numbers are those of the full grid; each
GPU holds a quarter of the orbitals. `--serial-control` keeps the former
route, on which the first GPU holds them all.

On several nodes such a run starts one rank per GPU
(`scripts/run_multi_gpu_scf.sh --nodes N --rank-per-gpu`), or one rank per
node with `Eigensolver: chebff`, each rank spreading its share over the GPUs of
its node (`acceleration/Eigensolvers/mpi_nested_basis.py`,
`PARSEC_MPI_NESTED_SHARING`; one rank per node without it, after a CHEBDAV
first solve or under `PARSEC_MPI_NESTED_SHARING=off`, is refused, since
2026-11-19 also where the detection finds no group: each rank would hold its
share on the first GPU of its node, `PARSEC_SINGLE_SECTOR_NODE_LAYOUT=allow`
runs it so as before), and the ranks
share the basis of the one sector (`acceleration/Eigensolvers/mpi_shared_basis.py`):
each holds all grid rows of its columns with a replica of the operator and
filters them alone; the Ritz step moves the filtered columns and `H` times
them to row blocks (`Alltoallv` of GPU arrays), sums the Gram matrices,
lets the root solve the small problem and broadcast the coefficients, and
moves the rotated rows back. Densities and nonlocal forces are summed over
the ranks; the root alone runs the Hartree solve, the mixer and the mover.
The trial basis is the single-device DLARNV stream, so the runs give the
trajectory of one node. Molecular dynamics with `second_order` on A100,
per later structure (`md_nrg.dat` equal to that of one node to the last
printed digit in every case, 3.6 SCF iterations per structure):

| Cluster | One node, one rank | 4 ranks, one node | 8 ranks, two nodes | 16 ranks, four nodes |
|---|---:|---:|---:|---:|
| C795H300 | 10.66 s, 14.2 GB | 13.07 s | 9.65 s | 6.77 s, 15.6 GB |
| C1653H508 | 40.47 s, 39.8 GB | | | 23.02 s, 39.0 GB |

On one node the shared basis of a single rank is faster: its devices copy
their blocks to each other directly. Across nodes the exchanges carry about
17 GB/s per rank, near what the network interface of a GPU carries, and take
half of the eigensolver's time on 16 ranks (C795H300: per Ritz step 0.37 s to
rows, 0.19 s back, against 0.32 s of filtering). A filter pipelined with
those exchanges was measured slower and left out (see the module).

## Differences from Fortran PARSEC

| Subject | Fortran PARSEC | This package | Why |
|---|---|---|---|
| Forces of a single point | Always computed | Only with `Skip_Force: false` | Existing inputs keep their cost. |
| Periodic forces | Reciprocal-space form factors (`force_loc_pb.f90`) | Real-space sums over images, with an FFT version for the potential | The same terms; the grid-sampled potential has egg-box effects the form-factor one has not (see "Periodic cells"). |
| Periodic forces and symmetry | Symmetrized with the space group | Not symmetrized | No space-group operations are found. |
| Forces of spin-orbit and non-collinear runs | `forcso`, with a warning that they "may be inaccurate" | The derivative of the energy of the SCF loop, pieces of `forcnloc.f90` and `soforc.F90` rewritten | The spin-flip term of `flsxy` pairs different harmonics than the Hamiltonian; see "Spin-orbit coupling and non-collinear magnetism". |
| Moved atoms with spin-orbit or non-collinear spinors | SCF restarted from the previous wavefunctions | The same: potential (with the field and moments of a non-collinear run) and eigensubspaces | The atomic density extrapolation is for plain spin-orbit coupling only. |
| Positions of a periodic run | Wrapped into the cell (`back_to_cell`) | Continuous; the builders fold them | The optimizer needs continuous coordinates. |
| SCF-correction force with `Cubic_Spline` | Uses the derivative table together with the spline coefficients of the density | Interpolates the atomic valence density linearly in that term | The combination in `forcloc.f90` mixes two tables; the term vanishes at SCF convergence. |
| `atom.cor` of a restarted run | "Step # 0" shows the coordinates of the input file | Shows the restart structure, which is the one computed | The file then describes the run. |
| Movable flags that break the symmetry | Symmetrizes regardless | Keeps the subgroup that maps the fixed atoms onto fixed atoms (`PARSEC_MOVE_FLAG_SUBGROUP=auto`; `off`: runs without symmetry) | The relaxation would leave the group it started in, but not that subgroup. |
| Molecular dynamics and manual movements | No symmetry | The accelerated command line solves a trajectory that keeps a group in its sectors, checked at every structure (`PARSEC_MOVING_SECTORS=auto`, from 2^24 grid points x states for a cluster, the floor of the sectors and 4 operations for a cell; `off`: no symmetry) | The same energies and forces in less time; the full grid where the velocities are random, a structure keeps no group or the run is small. |
| Positions after a movement | As the optimizer returns them | Averaged over the symmetry operations | Removes round-off drift; no change at print precision. |
| Energy that is not finite | L-BFGS-B reports a converged minimum | The run stops with an error | The Fortran outcome is wrong. |
| Local and core slopes within `2 rs(2)` of an atom | `forcloc.f90` skips the points within `rs(2)` | The slope of the smooth behaviour of the table, `s(c) d/c` (`PARSEC_SPLINE_FORCE_CENTER=auto`; `tip`: the former forces) | The energy over realistic steps averages the padded first interval of the table out; carbon FHI atom 5e-4 bohr from a grid point 7.8e-4 / 8.6e-4 -> 2.1e-4 / 1.2e-4 Ry/bohr off the slope of the SCF energy at steps of 1e-3 / 2e-3 bohr (the rest is nonlocal). |
| Potential of the Gaussian charges of a wire or slab | Solved to the Hartree tolerance of the run | Solved to a relative residual of 1e-12 (`PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE=auto`; `off`: as Fortran) | Its error enters the energy at first order and moves with the atoms: 7.5e-5 Ry/bohr in F_z of B in an hBN slab. |
| Force of the Gaussian charges of a wire or slab | Finite-difference gradient of `V_H` (`gradmvs`) | Analytic slope of the Gaussian (`PARSEC_WIRE_SLAB_GAUSSIAN_FORCE=auto`; `gradient`: as Fortran) | The derivative of the energy; the gradient misses it by 2.5e-4 Ry/bohr at h 0.4 for hydrogen. |
| Gaussian terms of the local forces of a wire or slab within `rs(2)` of an atom | Skipped with every other term of the point (`forcloc.f90`) | Evaluated at every point (`PARSEC_WIRE_SLAB_GAUSSIAN=auto`; `off`: the skip) | They are analytic and largest at the atom: 2.1e-3 Ry/bohr on an hBN slab with an atom on a grid point. |
| Hartree boundary of larger clusters | Multipoles at `Solver_Lpole` | Higher order and atomic tail by default | See "Physical scope" in the package README; `Hartree_Boundary_Tolerance: off` restores PARSEC's boundary for a comparison. |
| `Solver_Lpole` | Read and printed, then order 9 is used whatever it says (runs with 2, 4 and 9 give identical results) | The given order is the lowest one used | Not specific to moving atoms; give 9 for a comparison. |
| PBE | PARSEC's discretization | A discrete-variational one | Energies of the two codes differ by about 1e-4 Ry, so PBE forces have no Fortran reference; they are checked by finite differences. |
| Restart of molecular dynamics | gfortran builds cannot read their own `mdinit.dat` | Reads it | A format statement in `initial.f90`. |

One difference is not a choice. Where the optimizer stalls, that is where a
line search tries steps whose energy differences are below the SCF noise, the
two codes take different decisions from numbers that agree to all printed
digits, and their trajectories part. Builds of the Fortran with different
BLAS libraries differ among themselves in the same way.

## Validation

Every comparison with Fortran PARSEC uses the same input for both codes and
adds `Hartree_Boundary_Tolerance: off` on the Python side, so that both use
PARSEC's Hartree boundary. The Fortran is PARSEC built with gfortran 13.3 and
run on 4 to 12 MPI ranks.

### Forces at a fixed structure

Both SCF loops converged to 1e-8 Ry. Every force table PARSEC prints (ion-ion,
local, SCF correction, core correction, nonlocal, total) agrees within its six
printed decimals, and the total energies within 3e-8 Ry.

| Structure | What it exercises |
|---|---|
| H2O, asymmetric | LDA, s projector |
| H2O with `Cubic_Spline` | spline interpolation of potential and projectors |
| Ge2H2 | nonlinear core correction, s and d projectors |
| Ag2H2 | p and d projectors, relativistic pseudopotential file |
| SiNa4, also displaced and as a cation | old Martins format, occupations from `occup.in` |

`tests/unit/test_forces_parsec_reference.py`,
`tests/unit/test_sina4_parsec_reference.py` and
`acceleration/tests/test_forces_geometry.py` hold these comparisons. PBE forces
are compared with finite differences of the energy instead.

The domain of a cluster bounds how closely its forces follow the slope of its
SCF energy. An Au2 cluster in a sphere of radius 8 bohr (h = 0.7 bohr,
`Solver_Lpole: 4`, raised to 6 with the atomic tail, forces near 10 Ry/bohr)
stays 1.3 to 4.6e-5 Ry/bohr from that slope with an atom on a grid point
(1.6e-4 after Richardson extrapolation with the atom off the grid), while each
force term is the derivative of its energy at fixed density and orbitals
within 1.2e-7. The residual comes from the Hartree multipole boundary of the
small domain: a radius of 11 bohr brings it to about 6e-7. Two mechanisms
contribute: the Poisson operator with multipole boundary values is not
symmetric (the density of the outer layer enters moments that the boundary
does not give back to it), so the potential it gives is not the derivative of
the Hartree energy it gives, which the forces assume; and the atomic tail of
the boundary moves with the atoms with no force term of its own.

`PARSEC_HARTREE_VARIATIONAL` (off by default; `Hartree/variational.py`)
removes both. With `on` a cluster with a multipole boundary takes the
derivative of its Hartree energy as its Hartree potential and adds the force
of the tail: with `u = A^-1 rho` (zero boundary values, one more CG per
solve),

`V' = (V_H + 8 pi u - M^T A_IB^T u)/2 + (u.t)/(2 sum rho)`,

where `M` maps the density to its multipole boundary values, `A_IB` is the
block of the Laplacian that couples to the exterior points and `t` the tail
rows of the right-hand side. The energy functional is the same, `(h^3/2)
rho.V' = (h^3/2) rho.V_H` at every density, so the usual bookkeeping of the
total energy stands and the self-consistent density makes it stationary; the
forces then add `(h^3/2) d/dR_a [y.C_L(R)]` with `y = A_IB^T u` and the tail
values `C_L` (table "Hartree boundary force components" of `parsec.out`,
`forces_hartree_boundary` of the archive). The results keep `V_H` of the
Poisson solve as their Hartree potential (`hartree_potential_ry` of the
archive, the `v_hartree` cube of `Output_Grid_Data`, the `pot.dat` of a
frozen-density embedding): `V'` is not the electrostatic potential of the
density and differs from it by up to 5e-2 Ry near the sphere. The effective
potentials of a result are those the SCF ran with.

What this buys is consistency, not accuracy. The forces become the exact
slope of the energy of the multipole boundary. On A100 (the accelerated
command line, `Cubic_Spline`, SCF to 1e-10 Ry, central differences at steps of
1e-3 and 2e-3 bohr) force and slope then differ by the truncation error of the
central difference: Au2 in the sphere of 8 bohr with the atom on a grid point
by 1.1e-7 to 9.9e-7 Ry/bohr at 1e-3 bohr (4e-8 after Richardson
extrapolation), with the atom off the grid by 1.3e-6 after extrapolation
(1.6e-4 with the Poisson potential); Ge2H2 in a sphere of 6.5 bohr by 7e-7
(8.4e-6), Si2H6 in 7 bohr by 1.7e-7 (5.1e-6), Ag2H2 in 6.5 bohr by 5.1e-7 for
the component whose error came from the boundary (1.4e-5), the breathing mode
of a Ge5 tetrahedron in 9 bohr (h = 0.5 bohr) by 1.6e-8 after extrapolation
(6.6e-5 at 1e-3 bohr). A finite-difference check of the forces, and the
stopping test of a relaxation (`Force_Min`) against the energy it lowers, then
see the same function. That function is itself an approximation of the one of
the boundary values summed directly (`Full_Hartree: true`): in the same
spheres the slopes of the two energies differ by 1.8e-4 (Au2, 8 bohr) and
2.2e-4 Ry/bohr (Au2, 8.3 bohr), and the forces of the Poisson potential lie
closer to the forces of the direct boundary than those of `V'` on every
cluster compared: 3.1e-5 against 1.6e-4 Ry/bohr (Au2, 8 bohr), 1.7e-5 against
1.9e-4 (Au2, 8.3 bohr), 3.2e-5 against 5.2e-5 (Ag2H2), 4.9e-6 against 2.0e-5
(Ge2H2), 3.3e-6 against 8.6e-6 (Si2H6), 1.3e-7 against 3.8e-7 (Na4, 9 bohr).
Accurate forces of a small cluster come from a larger sphere, or from
`Full_Hartree` where it is affordable (Au2 in 8 bohr: 7.8 s against 3.1 s;
C185H124 in the sphere of the default rule did not reach its first SCF step in
29 minutes on four A100); the switch gives forces consistent with the energy
of the sphere as it is. The potential that would make the forces the slope of
the direct energy is the transpose of the direct Coulomb sums of the boundary
values in place of that of the multipole series; it is not built.

`auto` takes `V'` where the forces are computed (a relaxation, molecular
dynamics, manual movements without `Skip_Force: true`, `Calculate_Force`) and
the first Poisson solve of a structure estimates a force change of 3e-7
Ry/bohr or more (the self-consistency force term of `forcloc.f90` with `V' -
V_H` as the residual, plus the force of the tail; on the clusters measured
0.07 to 50 times the change); below it, and with `Hartree_Boundary_Tolerance:
off` (PARSEC's boundary values), the run keeps the Poisson solution bit for
bit. A run that keeps it decides again at the first solve of a structure in
which an atom has moved outward by more than one grid spacing since its last
decision; a run that takes `V'` keeps it. `hartree_variational_solves` of the
backend report gives the estimates, the decisions and the solves of `u`. A
restart (`Restart_Run`) from a run that kept the Poisson solution starts from
the fixed point of the other potential and takes a few more SCF iterations
under `auto` or `on` (an AuH spin-orbit restart with forces 8 instead of at
most 6).

What the opt-in costs on A100 (the accelerated command line, first runs, wall
times `off` -> `auto` -> `on`): one more CG per Hartree solve and the
solid-harmonic sums, with the sums of the tail over atoms and exterior points
on the device of a GPU route. C795H300 (forces at one structure, four GPUs)
14.2 -> 15.2 -> 14.8 s; C185H124 in the sphere of its example with the default
boundary 9.7 -> 11.0 -> 11.0 s, three structures of its relaxation 14.0 ->
17.6 -> 17.1 s; Au2 in the sphere of 8 bohr (one GPU) 3.4 -> 3.7 -> 3.6 s. A
run that `auto` leaves with the Poisson solution pays its decisions, 0.4 to
1.0 s each: H2O relaxed through seven structures (two decisions) 7.8 -> 8.1 s,
benzene 3.7 -> 4.2 s.

Two properties of a cluster calculation decide what a finite-difference check
of its forces compares, both as in PARSEC 1.5.

The first is the re-centring of the input (`initial.f90`, `Recenter_Atoms_In_Aperiodic_Directions`):
a cluster whose bounding box has its midpoint more than 0.1 bohr from the
origin is translated so that the midpoint is the origin, and `parsec.out`
says so (`WARNING: atom coordinates were shifted so that the cluster is
centered at the origin`, the shift vector and the new coordinates). Moving one
input coordinate of an atom that bounds the box moves the midpoint by half the
step and with it every atom, so the slope of the energy against that input
coordinate is `sum_b F_b . dR_b/dx`: for an atom that alone bounds its axis,
`F_a - F_net/2` with the forces before the net force is removed. The printed
forces have the net force removed (`F_a - F_net/N`; the archive holds
`forces_total` so and `forces_net`). An Au3 cluster (h = 0.42 bohr, a sphere of
9.5 bohr, `Cubic_Spline`, SCF to 1e-10 Ry, the atom of the largest y moved
along y): the slope of `E - TS` at 1e-3 and 2e-3 bohr is `F_a - F_net/2` to
8e-8 and 2.8e-7 Ry/bohr (1.6e-8 after Richardson extrapolation), while the
force before the net force is removed misses it by 7.9e-3 and the printed one
by 2.6e-3. A check compares like with like on a centred input, with
`Recenter_Atoms_In_Aperiodic_Directions: false`, or with the sum above; a
relaxation re-centres only its first structure, so its steps follow the
forces.

The second is the correlation of `Correlation_Type: CA`, PARSEC's default.
Its two Perdew-Zunger forms meet at `r_s = 1`, the density `3/(4 pi)`
e/bohr^3, where PARSEC's coefficients leave a step of 6.41e-5 Ry in the
correlation energy per electron. Whenever the density of a grid point crosses
that value the total energy steps by `(3/(4 pi)) 6.41e-5 h^3` Ry: 6.6e-7,
1.13e-6, 1.40e-6 and 1.91e-6 Ry at h = 0.35, 0.42, 0.45 and 0.5 bohr, and a
multiple of it where points of one symmetry orbit cross together (24 points of
C60 at h = 0.4 bohr: 2.35e-5 Ry). The forces are the slope of the energy
between the steps: on A100 the energy of Au2, Au3, Au4, Ag4, a spin-polarized
Au3 and a cell of fcc Au over displacements of +-2e-3 bohr equals the integral
of the forces within 2e-9 Ry apart from these steps, which do not change with
the SCF criterion, the eigensolver, the Hartree tolerance or the sphere. A
central difference whose two points straddle a step misses the force by the
step over twice the displacement: Au2 at h = 0.35 by 3.3e-4 and 1.7e-4
Ry/bohr at 1e-3 and 2e-3 bohr, Au3 by 5.7e-4 and 2.8e-4 (h = 0.42) or 4.8e-4
at 2e-3 (h = 0.5), Ag4 at h = 0.45 by 6.9e-4 and 3.1e-4, the fcc Au cell (two
points at once) by 6.0e-4 at 2e-3. `PARSEC_CA_CONTINUITY=on` (opt-in;
`V_xc/ca_lda.py`) joins the two forms with continuous energy and slope at
`r_s = 1` (the coefficients of libxc's `LDA_C_PZ_MOD`); the same checks then
agree to the truncation of the central difference, 1e-7 to 9e-6 Ry/bohr at
1e-3 bohr and 5e-9 to 8e-7 after Richardson extrapolation, with an atom off
and on a grid point, spin-polarized, and on the default and the
Fortran-derived routines of the cell. Two cases keep a few 1e-6 after
extrapolation for reasons of their own: Ag4 in a sphere of 10 bohr 3.7e-6
(its multipole boundary: 3.1e-7 under `PARSEC_HARTREE_VARIATIONAL=on`), and
displacements that carry the atom across a grid point (starting 5e-4 bohr from
it: Au4 1.2e-6, Ag4 6.1e-6). It lowers the energies of these Au and Ag systems
by 2.9e-4 to 3.5e-4 Ry per atom against PARSEC 1.5 and moves their forces by
up to 1.8e-5 Ry/bohr, which is why it is not the default.

### Trajectories

"Identical" means: the same number of structures, and every energy per atom
and largest force of `atom.cor` equal in all printed digits.

| Case | Structures | Result |
|---|---:|---|
| PARSEC example `0d_water_geoopt`, as shipped (SCF criterion 2e-4 Ry) | 7 | energies within 3e-5 eV/atom, forces within 8e-5 Ry/bohr |
| the same with SCF criterion 1e-8 Ry | 7 | energies identical, forces within 1e-5 Ry/bohr, coordinates within 1e-6 bohr |
| PARSEC test `tests/relaxation` (SiNa4), as shipped | 4 | identical, coordinates within 3e-6 bohr |
| the same with SCF criterion 1e-8 Ry | 4 | identical, coordinates identical |
| water, BFGS, symmetric and without symmetry | 8 and 17 | identical, `iterate.dat` identical |
| water, steepest descent | 7 | identical |
| water, `manual.dat` | 3 | identical |
| water, stopped by `Movement_Num` and continued with `Relax_Restart` | 3 + 14 | identical |
| the same, Fortran continuing the file written by Python | 14 | identical to Fortran continuing its own |
| nanodiamond C35H36, `Cubic_Spline`, `Read_VCD`, SCF criterion 1e-7 Ry | 9 | energies identical, forces within 1e-5 Ry/bohr, coordinates within 2.1e-5 bohr |

Molecular dynamics, both SCF loops converged to 1e-8 Ry: `md_nrg.dat` is the
same text as the Fortran file in every case below, and positions, velocities
and accelerations of `md_mech.dat` agree within one unit of their last
printed digit (the previous Fortran tree; since the formats of PARSEC 1.5 the
integer fields are wider and the velocities are `v(t)`).  Against PARSEC 1.5
itself (H2, Langevin with `test_rng` and `Mol_RNG_Seed`, 3 steps and a restart
of 2, both codes with their own SCF): `md_nrg.dat` and `movie.xyz` are the same
text (but for the last digit of one temperature), `md_mech.dat` agrees to its
eighth digit (the forces of the two codes differ at 1e-7), and a restart
reproduces 1.5 only with its fast-forward of the generator
(`tests/unit/test_moldyn.py::Parsec15ExecutableTests` replays 1.5's printed
forces).

| Case | Steps |
|---|---:|
| water, microcanonical | 8 |
| water, Langevin at 300 K, `test_rng` | 6 |
| water, Langevin with `stair`, `linear` and `log` cooling | 5 to 6 |
| water, Langevin with `rng_seed` and the generator of gfortran | 6 |
| water, microcanonical with a fixed atom (`Move_Flag`) | 5 |
| water, restarted at step 5 from an `mdinit.dat` written by Fortran | 3 |
| PARSEC example `0d_silver_cluster_md` (Ag8) with `test_rng` | 5 |

The silver example as shipped seeds from the clock, so only its first
structure can be compared with the reference output of the example: the
energies agree in all printed digits but the last. At the SCF criterion of
the example (2e-4 Ry) the two codes differ after five steps by 8e-4 eV/atom
in the kinetic energy, which is how far each is from its own tightly
converged trajectory. One structure of Ag8 takes 31 s with Fortran PARSEC on
4 ranks and 7 s with this package on a laptop GPU.

Two water cases, bounded BFGS (`Max_Step` given) and fixed atoms
(`Move_Flag`), agree for 20 and 31 structures; there the line search works on
energy differences at round-off and the two optimizers decide differently, as
described under
"Differences from Fortran PARSEC".

The readable path and the accelerated path give the same trajectories. The
nanodiamond relaxation takes 325 s with Fortran PARSEC on 12 ranks and 60 s
with this package on a laptop GPU (RTX 5070, 8 GB).

### On A100 GPUs, one to four nodes

Perlmutter, four A100 80 GB per node, double precision, no cache.

Equality of the execution layouts:

| System | Compared | Result |
|---|---|---|
| water, BFGS | one-rank serial control, 1, 2 and 4 ranks against the Fortran trajectory | identical |
| SiNa4, BFGS | the same four layouts against the Fortran trajectory | identical |
| C35H36, forces | serial control against 1, 2 and 4 ranks, with sector bases shared by 2 and 4 GPUs | largest difference 9e-14 Ry/bohr |
| C87H76, BFGS | serial control, 1, 2 and 4 ranks | identical, 6 structures |
| C795H300, forces | 4, 8 and 16 GPUs | largest difference 7e-13 Ry/bohr |
| C795H300, BFGS | 4, 8 and 16 GPUs | identical, 7 structures |
| water and Ag8, molecular dynamics | one node against the Fortran `md_nrg.dat` | same text |

Relaxation of C795H300 (1095 atoms, 3480 valence electrons, 0.2 Angstrom
grid, CA-LDA) from its GFN2-xTB structure, with the solver controls of the
measured single points (`Eigensolver: chebff`, `FF_MaxIter: 4`,
`Chebdav_Degree: 30`, `Chebyshev_Degree: 15`):

| Run | GPUs | Structures | Program time | Later structure: SCF, forces, rebuilding |
|---|---:|---:|---:|---|
| `Force_Min` 0.01 Ry/bohr, SCF criterion 2e-4 Ry | 4 | 7 | 47 s | 4.5 s, 0.6 s, 0.26 s |
| the same | 8 | 7 | 35 s | 3.0 s, 0.6 s, 0.26 s |
| the same | 16 | 7 | 30 s | 2.3 s, 0.5 s, 0.26 s |
| `Force_Min` 0.001 Ry/bohr, SCF criterion 1e-6 Ry | 16 | 17 | 85 s | 3.6 s, 0.5 s, 0.27 s |

The first structure costs about 6 s on 16 GPUs, as a single point does.
The energy falls from -9437.8200 to -9438.0419 Ry at the first criterion and
to -9438.0569 Ry at the second. A run stopped after three movements and
continued from its `relax_restart.dat` reaches the criterion in the same
number of structures; its energies differ by up to 5e-5 Ry from the uninterrupted
run, because the continued run starts its SCF loop without wavefunctions and
the SCF criterion is 2e-4 Ry.

For the next size, C2455H636 (3091 atoms) on 16 GPUs, a later structure costs
11.3 s of SCF, 1.7 s of forces and 0.8 s of rebuilding. Of the 1.7 s, 1.0 s
are the projector gradients, which every node builds on its host, 0.25 s the
local sums and 0.2 s the ion-ion forces. For C795H300 the three parts take
0.35, 0.07 and 0.04 s. The local sums grow with the number of atoms times the
number of grid points: for C2455H636 they took about 2.8 s over the whole grid
on one GPU and 1.8 s on the four GPUs of the root node, and take 0.25 s on
those four GPUs over the wedge of the D2 group.

### Starts of the SCF loop and molecular dynamics on A100 GPUs

The nanodiamonds of `examples/0d_nanodiamonds`. Fortran PARSEC is the same
version built on Perlmutter's CPU nodes (GNU compilers, 128 MPI ranks per
node); both codes use the inputs of `fortran_comparison/` with the SCF
criterion lowered to 1e-7 Ry. BFGS:

| Cluster | Structures | `none` | `atomic` | `second_order` |
|---|---:|---|---|---|
| C35H36 | 9 | coordinates within 1.6e-5 bohr | within 5.9e-5 bohr | |
| C87H76 | 6 | within 3e-6 bohr | within 1.4e-5 bohr | |
| C185H124 | 7 | within 3.6e-5 bohr | within 1.4e-5 bohr | within 2.1e-5 bohr |

Energies per atom and largest forces agree within one unit of their last
printed digit, with the same number of structures. Molecular dynamics at
300 K (5 steps of 20 a.u., `test_rng`): `md_nrg.dat` of C35H36 and C87H76
(the latter also with `second_order`) agrees with the Fortran file within
one unit of the last digit. Fortran PARSEC needed the 3500 s of its job for
five rows of C185H124 on one node; those five agree, and this package took
128 s on four GPUs.

Relaxations with the inputs of `gpu_benchmark/` (`Force_Min` 0.01 Ry/bohr,
SCF criterion 2e-4 Ry) on 16 GPUs. A later structure of C795H300:

| Start | Gradients | SCF | Forces |
|---|---|---:|---:|
| `none` | built for the forces | 2.20 s | 0.52 s |
| `atomic` | built for the forces | 1.71 s | 0.53 s |
| `atomic` | built beside the SCF loop (default) | 1.73 s | 0.14 s |

Program times, gradients beside the SCF loop:

| Cluster | Structures | `none` | `atomic` |
|---|---:|---:|---:|
| C1213H412 | 4 | 27.3 s | 25.7 s |
| C1653H508 | 6 | 54.2 s | 46.0 s |

At this SCF criterion the forces carry noise of about 1e-3 Ry/bohr, so the
two starts can end a relaxation at different structures: with the inputs
of `fortran_comparison/` C795H300 took 7 structures with `atomic` and 5 with
`none`. `second_order` saves nothing in BFGS, whose steps do not follow a
smooth path (C795H300 on 4 GPUs: 4.13 s of SCF per later structure against
3.44 s with `atomic`).

Molecular dynamics (5 steps) on the four GPUs of one node, per later
structure and largest memory of one GPU; C185H124 to C795H300 with the
first-solve controls of the GPU benchmark, the larger clusters with the
inputs of `gpu_benchmark/`:

| Cluster | `--serial-control` | Sector of the identity, `none` | `atomic` | `second_order` |
|---|---|---|---:|---:|
| C185H124 | 4.33 s, 17.1 GB | 2.99 s, 4.2 GB | 2.73 s | 2.02 s |
| C459H204 | 17.6 s, 69.4 GB | 9.8 s, 9.1 GB | 8.1 s | 5.9 s |
| C795H300 | out of memory (inputs of `fortran_comparison/`) | 18.1 s, 14.2 GB | 14.2 s | 10.7 s |
| C1213H412 | | 41.9 s, 27.0 GB | 32.7 s | 25.5 s |
| C1653H508 | | 69.0 s, 39.8 GB | 54.2 s | 40.5 s |

A later structure takes 6 to 7 SCF iterations with `none`, 5 with `atomic`
and 3 to 4 with `second_order`.

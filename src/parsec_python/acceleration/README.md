# Acceleration implementation

The [GPU setup and memory profile](GPU_SETUP_OPTIMIZATION.md) documents the
Perlmutter-tested ionic preparation, bounded native CSR construction, and
column-major filter output, including the launch settings of those measurements.
The [streaming Hartree and spatial-ordering study](GPU_SFC_OPTIMIZATION.md)
adds Ritz workspace reuse, GPU boundary construction, measured memory
limits, and the spatial-ordering variants that were not retained.

`parsec_python.acceleration` is the internal performance layer of the canonical
`parsec_python` package. It reuses the readable modules' input translation,
physical models, pseudopotentials, real-space conventions, Hartree and XC
treatment, SCF policy, convergence test, energy expression, and result
objects. The public workflow and command-line entry points select this layer
by default; explicit `*_reference` API aliases and `reference_main.py` retain
the readable SciPy path for audits.

Backend changes must preserve the selected physical model and float64 result
accuracy. Component arrays, SCF behavior, and energies are tested against the
readable implementation and PARSEC reference cases.

## Installation and command line

Install the reference NumPy/SciPy dependencies first:

```powershell
python -m pip install -r src\parsec_python\requirements.txt
```

SciPy then works immediately. To add the C++17/OpenMP backend, build its
separate extension from the repository root:

```powershell
python -m pip install -v src\parsec_python\acceleration\native
```

CuPy is optional; install the CuPy wheel matching the machine's CUDA runtime.
The package intentionally does not pin a CUDA-specific wheel.

Installing both optional runtimes enables the fastest default composition:
the native extension constructs the finite-difference operator and solves the
Hartree system, while CuPy keeps the Hamiltonian and eigensolver work on the
GPU. Exact commuting Cartesian signed permutations are detected automatically
for scalar fields and Hartree. The CuPy eigensolver uses all real
one-dimensional representations of the proved diagonal-reflection subgroup,
as PARSEC does, or, where the full point group of signed permutations is
larger and that solves less (a tetrahedral or cubic cluster), the
one-dimensional representations of a larger subgroup with weights (below,
"Weighted sectors of a larger subgroup"). The operations act about the origin
of the cluster as PARSEC places it; `PARSEC_STANDARD_ORIENTATION=on` first
turns a molecule given in another orientation into the frame of its largest
group (below, "Standard orientation of a cluster"). Fixed-point orbits are admitted by
their exact stabilizer characters, so representation dimensions may differ. All operations,
dimensions, device assignments, and fallback decisions are reported.

The canonical launcher accepts the same supported `parsec.in` and adjacent
`*_POTRE.DAT` files on every backend:

```powershell
python src\parsec_python\main.py INPUT
python src\parsec_python\main.py INPUT --device gpu
python src\parsec_python\main.py INPUT --device cpu --threads 16
python src\parsec_python\main.py INPUT --device gpu --gpus 2
python src\parsec_python\main.py INPUT --backend auto
python src\parsec_python\main.py INPUT --backend scipy
python src\parsec_python\main.py INPUT --backend native
python src\parsec_python\main.py INPUT --backend cupy
python src\parsec_python\main.py INPUT --symmetry off
python src\parsec_python\main.py INPUT --resident
```

On Windows, this source-tree launcher automatically re-runs itself with the
validated project-local `.venv312` interpreter when that environment exists.
This prevents a bare `python main.py parsec.in` from loading the legacy
`.venv` and its obsolete native extension. Set
`PARSEC_ACCELERATED_USE_ACTIVE_PYTHON=1` only when intentionally testing a
different fully configured environment.

The complete selection/profile form is:

```text
python src\parsec_python\main.py INPUT \
  --device {auto,gpu,cpu} [--gpus N|all | --gpu-ids LIST] [--threads N|all|auto] \
  --backend {auto,scipy,native,cupy} \
  --symmetry {auto,on,off} \
  [--symmetry-cache DIRECTORY | --no-symmetry-cache] \
  [--profile-operator]
```

`--device` says where a calculation runs: `auto` (the default) takes the GPU
if CuPy finds a usable one and the CPU otherwise, `gpu` stops where none is
usable, and `cpu` neither imports CuPy nor creates a CUDA context. Under
`--backend auto` a GPU run is the hybrid described below and a CPU run takes
the native backend, or SciPy without the extension; `scipy` and `native`
are CPU backends and `cupy` a GPU backend, and a backend that contradicts the
device is refused. `--gpus N` takes the first `N` GPUs the process sees,
`--gpu-ids` names them, and both ask for a GPU run; the default is every
visible GPU. `--threads` names the OpenMP team of the native kernels (`all`
for every CPU the process may use; the default keeps four of them free). The
options are carried by `PARSEC_DEVICE`, `PARSEC_CUPY_DEVICES` and
`OMP_NUM_THREADS`, which a job script can set instead; an option outranks its
variable, and the launcher sets them for its calculation only. The CPUs
counted are those of the affinity of the process, or fewer where
`SLURM_CPUS_PER_TASK`, `PBS_NP`, `NCPUS`, `NSLOTS` or `LSB_DJOB_NUMPROC` says
so, and in that last case the default thread count is taken from the variable
instead of the count of the extension. `parsec.out` reports the device, the
GPUs and the threads that ran under "Execution resources", and `--dry-run`
prints the plan. The module is `acceleration/resources.py`.

The symmetry mode defaults to `auto`: detect exact supported operations,
apply every reduction supported by the selected backend, and safely retain
the full grid if no nontrivial group or orbital representation is usable.
`on` requires nontrivial usable symmetry and reports an error instead of
falling back; `off` skips detection and forces the full-grid comparison path.
A PARSEC input with `Ignore_Symmetry=true` selects `off` unless an explicit
command-line mode overrides it.

No symmetry cache is read or written unless a directory is named. A first
calculation of a structure is fastest without one: the symmetry maps take a
fraction of a second, and a process builds the packed stencil of every sector
it owns from the grid (below). `--symmetry-cache DIRECTORY` names a
persistent cache; `--no-symmetry-cache` spells the default out, and the two
together are an error. The rule is the same for `main.py`,
`python -m parsec_python`, the resident worker and
`benchmarks/mpi_full_scf.py`; `prepare_single_point` takes the directory as
`symmetry_cache_directory` and has no cache without it. Earlier versions
used `.parsec_cache/symmetry` beside the input unless `--no-symmetry-cache`
was given. No run reads that directory any more unless it is named, and none
deletes it; `--symmetry-cache .parsec_cache/symmetry`, given from the input
directory, is the former behaviour with the same keys, files and loading.
The report says which it was: `symmetry_cache_directory` is `disabled` or
the directory (`not used` in a run with symmetry off that was given one),
also in the short text report, and `timing.json` of the MPI runner has the
key as `null` or the directory. The key speaks of the persistent cache only.
The resident worker keeps reference systems and operator bundles in memory
from one request to the next whatever these flags say, so its later requests
for a structure are no first calculations although they report `disabled`;
the detailed report of a GPU run (`Output_Level: 2`) then has
`reference_static_cache = hit` and `orbital_operator_cache = memory-hit`.

A cache pays only for repeated calculations of the same structure and grid,
and what a hit spares depends on the route of the sector stencils (below).
On the former full-grid route (`PARSEC_SECTOR_STENCIL=csr`) it spares the
full-grid matrix. On the default route it exchanges builds for reads: of the
symmetry maps, the character phases, the geometry of a native Hartree
boundary, and the stencils and reduced projector factors, which a process
without a cache builds for the sectors it owns and with one reads for all
sectors. An MPI rank that owns one sector of four gains nothing by it
(figures below). A single process that owns all sectors can gain at most a
part of its preparation: for 10,456 electrons (C2455H636) on 4 GPUs, with
the GPU Hartree boundary, these builds took 1.6 s of the 6.7 s before the
first SCF step of a 69 s calculation.
Complete SCF runs on A100 nodes with the version before the sector stencils
were built from the grid, 23,768 electrons (C5659H1132) on 16 GPUs: 136.0 s
with a host high-water mark of 13.1 GiB per rank without a cache, 148.0 s
and 17.6 GiB for the run that filled one, 124.5 s and 5.4 GiB for the run
that read it; 3,480 electrons (C795H300) on 16 GPUs took 10.9, 13.4 and
8.8 s. With the stencils from the grid and the fast symmetry maps the run
without a cache takes 114.1 s and 5.4 GiB: less time than the former run
that read its cache, in the same memory. With that version, in one
allocation, the same cluster on 16 GPUs took 114.7 s without a
cache, 119.4 s for the run that filled a named one (2.2 GB on disk) and
117.6 s for the run that read it, with 7.2, 12.0 and 10.3 s before the first
SCF step; C795H300 on 16 GPUs took 8.3, 9.2 and 8.4 s; C2455H636 on 4 GPUs,
one process that owns all four sectors, took 68.5, 69.9 and 67.7 s. A rank
with one sector of four now loses time to a cache even when it hits, and a
process with all sectors gained 0.8 s in that one pair of runs. A geometry
that changes from one
calculation to the next (a relaxation, dynamics) never hits, and every miss
writes a new bundle.

In a named directory, symmetry geometry, native multipole/boundary geometry,
character phases, and representation operators are content-addressed. The
SHA-256 key covers
the complete labeled atomic geometry, exact active grid, detector tolerances,
canonical finite-difference/projector buffers, projector signs,
grid-to-wedge map, and character phases. A grid, geometry, pseudopotential,
or symmetry change therefore builds a new entry. Loaded orbit metadata is
validated before use; a missing, old, or damaged entry is rebuilt by the
exact detector.
A key only addresses its cache entry: without a cache directory (and, for the
reduced operators, without the resident bundle cache) it is not hashed and the
report lists it as `not computed (cache disabled)`.
The cache stores GPU-ready stencil-major coefficient codes and palettes rather
than duplicate reduced CSR matrices. When representation sectors have the
same neighbor topology, one host archive array and one CUDA int32 neighbor
allocation are shared across all sectors; only representation-dependent
codes, palettes, and KB factors remain separate. Native multipole geometry is
exported to its own exact-key archive, so later Python processes reconstruct
the C++ boundary builder without repeating its angular and missing-neighbor
setup.

The default symmetry/GPU path does not build the full-grid finite-difference
matrix, with or without a cache. A process builds the packed stencil of every
sector it owns from the grid lookup table, the orbit and phase maps and the
stencil coefficients (`Symmetry/sector_stencil.py`), visiting only the
representative rows of that sector. With a cache directory it builds or loads
the stencils of all sectors instead, since a cache entry is the complete
bundle. For an MPI rank that owns one sector of four the cache then no longer
pays: on a workstation, with four sectors of 1.1 million rows at order 12,
loading the entry took 0.84 s and 0.54 GiB, and building and writing it
2.75 s and 1.66 GiB for a file of 0.81 GiB, against 0.72 s and 0.22 GiB to
build the one stencil without a cache. The measured series ran without a
cache, which is the default now. The arrays are those of the former route
(full-grid CSR, `reduce_sector_csr`, audit, packing) element for element:
same neighbor order, coefficient codes and palette, repeated columns added in
the same order. That order is the one of `PARSEC_NATIVE_SECTOR_ASSEMBLY=1`,
which `scripts/run_multi_gpu_scf.sh` exports and every measured run had.
The variable is 0 when unset, and the SciPy reduction it then selects adds
three or more repeated columns of a row in another order. Against a former
run started without the variable the results are therefore not bit for bit
the same: entries of a sector near the symmetry axes and planes can differ in
the last bits (up to 41 entries per sector and at most 7e-15 on test grids of
24,000 points of orders 6 to 12, half-shifted and with points on the planes;
none on the grid of the 3,480-electron cluster C795H300), and the total energies of
small test molecules moved by 1e-13 to 1e-11 Ry. `PARSEC_SECTOR_STENCIL=csr`
restores the former route with either setting of the variable. A symmetry
cache does not tell the two orders apart: its key covers the inputs of the
stencils, not the route that summed them, so a run that reads an entry
(`orbital_operator_stencil_builder` is then `cached`) takes the numbers of the
run that wrote it. Compare a cached run bit for bit only with the run that
wrote its entry, or write the entry with the variable set to 1. The
C++/OpenMP kernel `build_sector_stencil` builds the arrays when the installed
extension has it; an extension built before the kernel existed is served by
a NumPy builder with the same result, several times slower and with a higher
host peak (see `GPU_SETUP_OPTIMIZATION.md`). Three settings of
`PARSEC_SECTOR_STENCIL` select the route; any other value is an error:

| `PARSEC_SECTOR_STENCIL` | Sector stencils |
|---|---|
| `direct` (default) | from the grid; the native kernel if present, NumPy otherwise |
| `numpy` | from the grid, NumPy only |
| `csr` | the former route: the full-grid CSR, reduced and packed |

A periodic cell (since 2026-11-18) holds the matrix itself, the stencil wrapped
on every axis (`Laplacian/finite_difference.py` names its grid on it); its
sector stencils are built from that grid with a lookup padded by the rows of the
opposite faces (NumPy, the sector rows alone: the native kernel leaves the
neighbours across a face out), the same arrays as the reduction of the matrix
bit for bit, unless the matrix was changed (its shape, entry count and a few
rows are checked against the stencil) or `PARSEC_SECTOR_STENCIL=csr`. The
reduction read every row and column of the matrix once per sector: 5.5 s of the
setup of the 64,000 atoms of a 20^3 diamond cell, against 0.013 s for the
sectors of the 16^3-point silicon cube.

The accelerated preparation of such a box does not build the matrix at all
(`PARSEC_PERIODIC_DEFER_LAPLACIAN`, `auto` by default since 2026-11-18;
`acceleration/Laplacian/deferred.py`, `DeferredPeriodicNegativeLaplacian`): a
descriptor of the wrapped stencil stands for it, from which the sector stencils
and the row the FFT Hartree solve takes its symbol from are made; a reader of
the matrix itself (`descriptor @ x`, its sparse attributes, the host backends,
the conjugate-gradient Hartree solve) builds the same matrix once. A 20^3
diamond cell on 16 A100 never builds it: 5.8 s and 13 GiB of host memory per
rank less (the high-water mark 21.3-24.0 -> 8.1-10.7 GiB). `off` builds it at once, as before.

The descriptor that stands for the matrix carries a SHA-256 provenance key
over every discrete grid and stencil input. It is hashed only where a cache
looks it up, and the reduced bundle of either route is stored under the same
key in the same format. Only CuPy execution takes the descriptor; the native
and SciPy backends read the matrix and build it as before. A symmetry
fallback, an explicit full-grid run, the pure-CuPy Poisson solver, a Hartree
solve without the orbital reduction, or a request for the full operator
(`system.reference.hamiltonian`, the component profile) materializes the same
validated C++ CSR. The report records whether this happened
(`finite_difference_full_grid_materialization`) and what packed the stencils
(`orbital_operator_stencil_builder`). In a resident process the matrix is
built by the first calculation that reads it and shared by the later ones,
which report `reused_from_resident_reference`. The largest native Hartree
coefficient table is stored as a memory-mapped NPY sidecar, eliminating one
full archive copy while retaining the same native-owned complex128 buffers.

### Resident fast-start mode

An opt-in resident worker removes repeated Python imports, CUDA context
creation, kernel-module loading, and allocator initialization across separate
calculations:

```powershell
python src\parsec_python\main.py --resident-start
python src\parsec_python\main.py INPUT --resident --no-archive
python src\parsec_python\main.py --resident-status
python src\parsec_python\main.py --resident-stop
```

Starting explicitly is optional because the first `--resident` calculation
starts the worker automatically. Requests run sequentially. Every request
reparses `parsec.in`, validates pseudopotentials and caches, and constructs
fresh physical, eigensolver, mixer, and SCF state. Only process/runtime
artifacts are retained, so this changes startup cost rather than the DFT
trajectory. The ordinary command without `--resident` remains a standalone
process. Stop the worker when its GPU resources should be released.

Canonical runs use these default filenames beside the input:

- `parsec.out`
- `parsec_python_results.npz`

Explicit `--log` and `--output` controls select other paths. Use
`--profile-repeats N` to control the average reported by
`--profile-operator`.

## Backends

| Backend | Execution strategy | Optional dependency | Intended use |
|---|---|---|---|
| `scipy` | Canonical float64 CSR/CSC operators, allocation-reduced fused host action, and fast recurrence multipoles | NumPy and SciPy | Portable accelerated baseline and numerical reference for other backends |
| `native` | Cached C++17 finite-difference construction plus OpenMP-capable fused Hamiltonian action and Poisson CG | Compiled `parsec_accelerated_native` extension | Multicore CPU acceleration without changing SCF physics |
| `cupy` | Float64 device-resident Hamiltonian, CHEBFF/CHEBDAV/SUBSPACE state, an opt-in FP32 later filter, and shared-CSR Poisson CG | CuPy build matching the installed CUDA runtime and a usable GPU | GPU eigensolver and Hartree linear algebra |
| `auto` | Accuracy-preserving hybrid when possible: native finite-difference construction and Hartree boundary, plus a device-resident CuPy Hamiltonian and eigensolver and the Hartree CG on the same GPU | Native extension and CuPy for the complete hybrid; missing components fall back with recorded provenance | Fastest default execution while retaining a reproducible component-by-component fallback trail |

`auto` selects components independently. When both optional runtimes are
available, finite-difference construction uses the native C++ kernels, while
Hamiltonian applications, CHEBFF/CHEBDAV/SUBSPACE and, since 2026-11-07, the
conjugate gradients of the Hartree solve, with the boundary of a full grid,
use CuPy (`PARSEC_HARTREE_LINEAR_BACKEND`, `Hartree/GPU_HARTREE.md`; `native`
keeps both on the host). Where the orbitals are solved in symmetry sectors
the boundary of the wedge is built on the same GPU too and every Hartree
solve stays there, in the device-resident chain (since 2026-11-07,
`PARSEC_CUPY_RESIDENT_HARTREE`; `0` keeps the native orbit table of the
wedge on the host and the host interface). If only one accelerator is available, `auto` uses the compatible
combination and records each choice. Explicit `scipy`, `native`, and `cupy`
remain clean end-to-end comparison modes rather than hybrid aliases.

### Spin-polarized calculations

`Spin_Polarization: true` runs on every backend. The SCF loop is the one of
`SCF/spin_polarized.py`; a backend supplies the Hamiltonian application, the
Hartree solver and, on a GPU, the eigensolver and the density of each
channel.

- With the CuPy backend and a point group, both channels are solved in the
  symmetry sectors by default (`Spin_Symmetry_Sectors: auto`, for every
  functional; `../README.md`, "Spin channels in the symmetry sectors"): each
  channel has its own subspace in every sector of
  `CuPySymmetrySCFEigensolver` (`for_spin_channel`), and the channel
  densities, the Hartree potential (on the symmetry wedge), the spin
  functional and the mixer keep one value per symmetry orbit
  (`SCF/spin_sectors.py`). The symmetry guard of that module sends a run
  whose start Hamiltonian has a partly filled multiplet, whose channel
  density breaks the point group of the atoms or whose occupations end
  fractional back to the full grid, with a `NOTE`: that of one GPU, or,
  where both channels would not fit there (planned and said in a `NOTE` at
  preparation), one sector of the identity whose basis the GPUs of the
  process or the ranks of the runner share (`PARSEC_SPIN_FALLBACK_SHARED`);
  a run for which neither fits stops with an error before that full grid
  is prepared (under `off` the full grid of one GPU is tried anyway, as
  before). The `NOTE` names the layout taken; the forces and any later
  structure are computed there. Since 2026-11-19 a point group larger than
  its abelian subgroup gives both channels the weighted sectors of a
  subgroup (`PARSEC_ORBITAL_GROUP_SPIN`; on several MPI ranks too since
  2026-11-21, `PARSEC_ORBITAL_GROUP_MPI`; "Weighted sectors of a larger
  subgroup"), and a run the guard stops there goes first to the abelian
  sectors, then, where their guard stops it too, to that full grid; since
  2026-11-21 straight to the full grid where the abelian sectors would stop
  the same levels (`PARSEC_WEIGHTED_GUARD_SHORTCUT`: clusters, cells whose
  operations take no translation, and since 2026-11-22 cells whose abelian
  sectors take their translations, `PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS`).
- An antiferromagnetic or broken-symmetry start names two labels of one
  species with opposite `Initial_Spin_Polarization` (`Cr_a` at `+p`, `Cr_b`
  at `-p`); the sectors and the guard then keep the operations that map
  every label onto itself. With `PARSEC_SPIN_FLIP_SYMMETRY=auto` (since
  2026-11-19, default `off`) an operation of the atoms and the grid that
  exchanges the partner labels (a signed permutation about the origin of a
  cluster, `r -> R r + t` with a pure translation first on a bulk cell) and
  maps the up start density onto the down one makes the down channel the
  image of the up channel at every step: the loop solves the up channel
  alone (in its sectors where it has them) and takes the down levels,
  occupations, density (one gather of the orbit values), orbitals (formed
  where they are read), sector labels and nonlocal forces from it
  (`SCF/spin_flip.py`). Hybrids, meta-GGAs, LDA+U,
  `Symmetrize_Charge_Density`, an odd electron count and `occup.in` with
  unequal channels keep both channels; so does an operation that does not
  map the local ionic potential onto itself to the rounding (point charges
  of `Add_Point_Charges`: the detection takes the next operation that keeps
  them, if any). The spin loop checks every loop it runs again: one whose
  atoms lost the operation, whose ionic potential it breaks (the six field
  loops of `Polarizability`) or whose warm start holds a down potential that
  is not the image of the up potential (`Restart_Run` of a state that breaks
  the flip) solves both channels. `parsec.out` says which route ran at every
  `Output_Level` (a `NOTE` in the backend block, the detail
  `spin_flip_symmetry`) and names each loop that solves both channels and
  why (a `NOTE` before its iterations). It works in the sectors
  of clusters and of spin cells at the Gamma point, on the full grid and in
  the full-SCF runner (whose ranks then divide the up channel's sectors
  alone), and halves the eigensolve of a step: SCF 1.2-1.9x faster on the
  A100 cases of the switch table, energies within 3e-11 Ry and forces within
  1e-6 Ry/bohr of both channels solved. It keeps the state in the magnetic
  group, which PARSEC 1.5 does not impose: two solved channels from the same
  start reach a state that breaks the flip only through the noise of their
  solves, which grows where a level and its image in the other channel are
  partly filled. So a flip loop is left where its up channel ends with a
  fractional Fermi-Dirac occupation (`min(f, 1 - f)` from 0.05) and where a
  test of the spin sectors fails in it (test 1 reads the exactly equal levels
  of the two channels as a partly filled multiplet of the start Hamiltonian
  that `off` does not see: O2 from `+0.5 / -0.5`): `parsec.out` says so, the
  rest of the run solves both channels (later loops, ionic steps, the
  systems the guard prepares again) and the loop runs again from its start
  in the sectors it started in, the loop of `off` (since 2026-11-21). It
  stays `off` by default: a flip loop can also settle with integer
  occupations in another state than two solved channels reach, which no
  test sees (Ni2 AFM, 0.77 eV higher; see the switch table).
- With `Spin_Symmetry_Sectors: off`, a CPU backend, `--symmetry off` or no
  point group, the two channels and the Hartree potential are solved on the
  full grid; the run reports `orbital_symmetry = full grid (spin-polarized:
  ...)` where symmetry would otherwise apply.
- On a GPU the channels share one Hamiltonian allocation, whose local
  potential is bound before each solve, and each keeps its own buffered
  subspace on the device (`CuPySCFEigensolver.for_spin_channel`). The device
  holds two subspaces; the orbitals of the result stay those two blocks and
  are joined only on the host, where they are asked for.
- The exchange-correlation potentials are evaluated by the NumPy functionals
  of the readable package on the host (`xc_implementation` says so); the C++
  CA evaluator is unpolarized. The CuPy backend evaluates PW92, LB94,
  Casida-Salahub, PBE and PW91 on the device instead (PBE since 2026-10-30;
  `PARSEC_CUPY_XC=off` keeps the host; Exchange-correlation on the device,
  below).
- The MPI SCF (`mpi_context`, the multi-node runner without
  `--serial-control`) divides the symmetry sectors of both channels among
  its ranks (every command names the channel it is for, a rank keeps a
  subspace of each channel in each sector it owns), for single points and,
  under `PARSEC_SPIN_SECTOR_FORCES=auto`, for the forces and relaxations
  that keep their sectors (the nonlocal terms of both channels summed over
  the ranks, the `spin_forces` command); channels that stay on the full grid
  (`Spin_Symmetry_Sectors: off`, no point group, LDA+U) and molecular
  dynamics of spin-polarized runs are refused there. A run that the symmetry
  guard sends to the full grid is finished in the process of the root, or,
  where it would not fit one GPU, on all ranks as one sector of the identity
  that they share (`PARSEC_SPIN_FALLBACK_SHARED`; `MULTI_GPU.md`, "Hybrid
  functionals on several nodes").
- An eigensolver hook that keeps a subspace of its own and has no
  `for_spin_channel` is refused by the SCF loop, as are symmetry-reduced
  scalar-field hooks without the two-channel interface of
  `SCF/spin_sectors.py`.

Forces, relaxation, molecular dynamics and the B3LYP refinement take a
spin-polarized result on every backend.

### Self-consistent spin-orbit coupling of clusters

`SO_from_scratch` with `SO_PSP` species and without `Spin_Polarization`, on
an isolated system with CuPy, runs the loop of
`../SCF/self_consistent_soc.py` with the spinor operator, the CHEBFF and
SUBSPACE filters and the density on one GPU
(`SCF/self_consistent_soc.py`, `Eigensolvers/chebff_spinor.py`). PARSEC 1.5
drops to the identity with spin-orbit coupling, and so does this loop by
default: it filters `States_Num` plus the buffer complex spinors of
`2 N_grid` rows.

`PARSEC_SOC_SECTORS` (since 2026-11-19; default `auto` since 2026-11-21,
`off` before) solves the sectors of the double group instead
(`SCF/soc_sectors.py`); `auto` takes the `kramers` sector (below) where the
whole-grid spinor basis (`2 N_grid` rows times the states and the buffer)
reaches the floor `soc_sectors.MINIMUM_ENTRIES`, or under `--symmetry on`,
and keeps the whole grid below it with a `NOTE`, since a first run with an
empty kernel cache compiles the kernels of the sector (the floor is 2^25 =
33,554,432 entries; the first runs that set it are in the third table
below); under the opt-in `PARSEC_SOC_AUTO_FOURFOLD=auto` from 2^26 entries
on its `fourfold` split (below, since 2026-11-22). A two-fold rotation or
mirror `g` of the atoms on the grid (`C2z` or `sigma_z` where the group has
one, else along `x` or `y`) acts on spinors as `-i sigma_a` times its
spatial map, with the eigenvalues `+/- i`; time reversal commutes with it
and swaps the two, so its `+i` sector holds exactly one state of every
Kramers pair, on `N_grid` rows (`kramers`). With a second operation along
`x` or `y` (`C2v`, `D2`, `D2h` and their subgroups that have one) the
antiunitary `T h` squares to `+1` on that sector, whose operator is then
real symmetric: the eigensolver runs in real arithmetic. `on` also splits
the sector by inversion parity where the cluster has a centre of inversion
(two problems of about `N_grid / 2` rows; the pairs of each are set by its
share of the rows and regrown when it may miss one). These are all the
double-group sectors an abelian point group gives: every spinor irrep of
`D2` and `D2h` is two-dimensional. Every reduced state stands for a Kramers
pair (its eigenvalue twice in the occupations, its density with the
occupations of both); Hartree, exchange-correlation, mixing and energies
stay on the whole grid, and the result holds the `States_Num` whole-grid
spinors (each reduced state and its partner `T psi`). The kinetic term, the
local potential and the Kleinman-Bylander projectors stay block diagonal in
the real symmetry functions of the scalar group; only the spin-orbit
projectors couple the blocks. A cluster without a two-fold rotation or
mirror (`C1`, `Ci`), `--symmetry off`, and `Ignore_Symmetry` unless a mode
on the command line overrides it (as in every other run), keep the whole
grid with a `NOTE`. `--symmetry on` asks for the sectors (under `auto` at
every size): it is refused before the SCF with `PARSEC_SOC_SECTORS=off`
and for a cluster without such an operation. `parsec.out` of a spin-orbit run that allows symmetry prints
PARSEC 1.5's block of the point group (whose Abelian subgroup is the
identity for spinors, as in the Fortran), not the `Ignore_Symmetry` lines,
and its `orbital_symmetry` line names the sector the loop solves, or the
whole grid and why. Measured on one A100 (`chebff`, `CA`, 300 K, every run
a fresh process with the CuPy kernel cache warm; `kramers` against `off`;
the Au20 h = 0.3 bohr, Au13 and Au38 rows and the first Au20 row are of the
final code, the others of the code before its faster final unfold, which
saved at most 0.1 s on them):

| Cluster (group on the grid) | States, spinor rows | SCF [s] | GPU peak [GiB] | Iterations | Energy difference [Ry] |
|---|---|---|---|---|---|
| Au20 Td (`D2`), h = 0.4 bohr | 236, 536,192 | 57.2 -> 11.3 | 15.3 -> 4.8 | 17, 17 | 2.9e-6 |
| the same at `Convergence_Criterion` 1e-6 | 236, 536,192 | 81.3 -> 22.1 | 15.3 -> 5.3 | 23, 25 | 0 (to 1e-8) |
| Au20 Td (`D2`), h = 0.3 bohr | 236, 1,270,720 | 160.7 -> 29.6 | 35.5 -> 10.5 | 20, 20 | -1.5e-5 |
| Au38 truncated octahedron (`Oh`, `D2h` on the grid) | 460, 441,184 | 111.5 -> 18.8 | 18.6 -> 5.3 | 21, 18 | 2.4e-6 |
| Au13 cuboctahedron (`Oh`, `D2h` on the grid) | 160, 226,208 | 15.6 -> 3.4 | 5.7 -> 1.8 | 16, 14 | 3.2e-6 |
| Au4 Td (`D2`) | 52, 310,096 | 9.3 -> 3.1 | 4.2 -> 1.5 | 19, 17 | 4.0e-6 |
| Au2 (`D2h`), `Convergence_Criterion` 1e-6 | 30, 412,272 | 10.9 -> 4.3 | 3.7 -> 1.4 | 27, 31 | 0 (to 1e-8) |
| Au2 tilted in the xz plane (`C2h`, complex sector) | 30, 412,272 | 8.1 -> 3.6 | 3.7 -> 1.6 | 19, 20 | 2.2e-7 |
| AuH, `examples/0d_AuH/reduced_self_consistent_soc` (`C2v`) | 20, 14,416 | 0.91 -> 0.76 | 0.58 -> 0.50 | 7, 7 | 4.1e-6 |
| the same at `Convergence_Criterion` 1e-6 | 20, 14,416 | 1.23 -> 1.23 | 0.58 -> 0.50 | 15, 15 | 0 (to 1e-8) |

The energy differences are those of the SCF tolerance (`1e-4`, AuH
`5e-3`); at `1e-6` the two routes print the same energy. The parity split
of `on` is slower than `kramers` on every centrosymmetric cluster measured
except the largest, and slower than `off` on a single atom: each parity
sector runs its own Lanczos bound and Rayleigh-Ritz problem with a buffer
of at least two states, and a sector that may miss a pair is regrown with a
second first solve. It paid only on Au38, where the filter time fell from
10.5 to 6.5 s. Clusters without inversion take the `kramers` route under
`on`. The same A100 runs (SCF seconds; energies of `on` within the SCF
tolerance in every case, at `Convergence_Criterion` 1e-6 for the atom and
Au6 the same to 1e-8 Ry):

| Centrosymmetric cluster (group on the grid) | `off` | `kramers` | `on` |
|---|---|---|---|
| Au atom (`D2h`), 1e-6 | 2.25 | 2.02 | 3.86 |
| Au2 (`D2h`) | 8.5 to 10.9 | 3.6 to 4.3 | 4.8 to 6.1 |
| Au2 tilted in the xz plane (`C2h`) | 8.1 | 3.5 | 4.4 |
| Au6 octahedron (`D2h`), 1e-6 | 13.4 | 3.6 | 4.5 |
| Au13 cuboctahedron (`D2h`) | 15.6 | 3.4 | 3.6 |
| Au38 truncated octahedron (`D2h`) | 111.5 | 18.6 | 13.4 |

`on` is therefore offered for large centrosymmetric clusters only; between
Au13 and Au38 the break-even has not been measured.

`fourfold` (since 2026-11-22, opt-in; also under `auto` from 2^26
whole-grid spinor entries on with `PARSEC_SOC_AUTO_FOURFOLD=auto`, below)
splits the real sector of C2z by a
four-fold operation about z, C4z or the rotoinversion S4z, where the atoms
and the grid keep one (`soc_sectors._four_fold_splits`): it squares to C2z
and turns spinors as the rotation by pi/2 about z, so its eigenvalues in the
sector are `+/- exp(i pi/4)`; the real-making `T h` inverts it, so its
matrix in the real columns of the sector is a permutation of order two with
phases, and the columns of each eigenspace are real combinations of one or
two of them: two real sectors of about `N_grid / 2` rows, solved as the
parity sectors are. Hartree, exchange-correlation and the rest stay as in
`kramers`. The atom, a square and a tetrahedron give the spectrum of the
whole grid (CPU tests, `test_soc_fourfold.py`). One A100, SCF seconds,
`kramers` -> `fourfold` (both runs of each with a warm kernel cache; energies
within the SCF tolerance, the same to 1e-8 Ry for Au20 at 1e-6): Au20 (Td,
S4z) at h = 0.4 bohr 11.1 -> 8.6, peak 4.8 -> 3.3 GiB; at h = 0.3 bohr
29.7 -> 21.5, 10.5 -> 7.1 GiB; Au38 (Oh, C4z) 18.7 -> 14.6, 5.3 -> 4.2
GiB; Au13 (Oh) 3.4 -> 3.4-3.6; Au4 (Td) 3.1 -> 4.3 and Au2 (C4z along the
bond) 3.6 -> 4.8, slower, as the parity split is on small clusters (each
sector its own Lanczos bound and Rayleigh-Ritz problem). The relativistic
germanium and silver tables of the test data (`tests/data/Ge_POTRE.DAT`, with
its core correction, and `Ag_POTRE.DAT`) carry spin-orbit channels too and
behave alike. With a warm kernel cache every cluster measured up to 3.8e7
whole-grid spinor entries was slower with `fourfold`, on the three species
(one A100, SCF seconds `kramers` -> `fourfold`, the same iterations unless
given): Ge5 (C4v, h = 0.4 bohr, 5.8e6 entries) 4.8 -> 5.5 (50 -> 31
iterations), Ge8 (D4h, 7.6e6) 2.7 -> 4.4, Ge6 (Oh) 5.2 -> 6.2 (55 -> 35), Ge4
(Td, S4z) 2.3 -> 4.6 (21 -> 25), Ag6 (Oh, 2.2e7) 3.7 -> 5.5, Ag13 (3.8e7) 4.3
-> 4.6 (22 -> 20), with the gold clusters above; with an empty cache the same
except Ag6 (process wall 25.8 -> 23.9 s on its first run, 6.8 -> 8.6 s warm).
Larger clusters were mostly faster, but not all: `fourfold` costs more per
SCF iteration where the cluster has few states (each sector its own Lanczos
bound and Rayleigh-Ritz problem: 29 % more per iteration on an Au2 of 7.2e7
entries, 12 % on Au6), and part of its gain comes from SCF loops that took
fewer iterations, which the route does not guarantee. Process wall seconds,
`kramers` -> `fourfold`, first run with an empty kernel cache (warm in
brackets): the Au4 tetrahedron squeezed to D2d (S4z, 56 states, h = 0.26
bohr) in a sphere of 13.5 bohr (7.27e7 entries) 30.5 -> 30.8 (14.9 -> 15.5, 24
iterations both) and of 15.5 bohr (1.10e8) 36.6 -> 38.9 (21.0 -> 23.4, 23 ->
30 iterations), slower; Au6 (Oh,
7.27e7) 28.5 -> 27.4 (12.7 -> 11.7, 22 -> 17 iterations). It therefore stays
an opt-in: `PARSEC_SOC_AUTO_FOURFOLD=auto` (since 2026-11-22; `off`, the
default, keeps `kramers`) lets `auto` take `fourfold` from
`soc_sectors.FOURFOLD_MINIMUM_ENTRIES` (2^26 = 67,108,864) entries on, where
it lowers the GPU peak by 18 to 31 % (Au4 at 1.10e8 entries 7935 -> 5861 MiB)
and was faster on most clusters measured, in the table below.
First runs on one A100 (each route with its
own empty CuPy kernel cache, a fresh process; process wall seconds `kramers`
-> `fourfold`, the warm second run in brackets, criterion `1e-4`; atom-centred
Td fragments of diamond Ge, h = 0.4 bohr):

| Cluster (entries) | First run [s] | Warm [s] | Iterations | GPU peak [MiB] |
|---|---|---|---|---|
| Ge17 (4.19e7, below the floor) | 22.9 -> 21.5 | 9.7 -> 8.6 | 26 -> 15 | 3111 -> 2329 |
| Au13 at h = 0.33 bohr (6.68e7, below) | 25.6 -> 24.6 | 9.7 -> 9.2 | 17, 17 | 2827 -> 2405 |
| Ag13 at h = 0.30 bohr (8.90e7) | 27.3 -> 26.2 | 11.8 -> 10.5 | 17, 17 | 3551 -> 2923 |
| Ge29 (8.86e7) | 28.0 -> 25.3 | 15.0 -> 11.9 | 23 -> 18 | 4021 -> 3191 |
| Au20 (1.30e8) | 27.6 -> 24.9 | 15.0 -> 12.1 | 17, 17 | 4877 -> 3349 |
| Ge35 (1.38e8) | 41.7 -> 27.6 | 28.6 -> 14.5 | 45 -> 16 | 5189 -> 4199 |
| Ag38 (2.06e8) | 28.8 -> 25.8 | 18.9 -> 15.8 | 15 -> 16 | 5263 -> 4119 |

The criterion `1e-4` leaves these SCF loops apart by up to 4.6e-4 Ry on the
Ge fragments (their dangling bonds converge slowly, and the two routes take
different numbers of iterations); at `1e-7` the whole grid, `kramers` and
`fourfold` give the same energy within 2.1e-7 Ry (Ge17, Ge29, Ge35, Au13,
Ag13: `fourfold` within 1e-8 of the whole grid, Ge35 in 28 iterations against
84 for the whole grid and `kramers`), and at `1e-8` (Ge5, Ge8) and `1e-6`
(Ag6, Ag13) `fourfold` and `kramers` print the same energy. A cluster without
such an operation, or whose sector is complex, takes `kramers`. With an empty CuPy
kernel cache (the first run on a machine) both routes
compile their kernels in the first solve, `kramers` its real-arithmetic
ones as well (3 to 6 s more on an A100): AuH took 6.8 -> 9.8 s, Au2 14.7
-> 15.9 s. So `auto`, the default since 2026-11-21 (the energies are those
of the whole grid within the SCF tolerance), takes the sector only from a
size of the spinor basis on.

First runs set the floor of `auto`: one A100, every run a fresh process
with an empty CuPy kernel cache (a first run on a machine compiles the
kernels of each route; the sector compiles real-arithmetic kernels besides),
`chebff`, `CA`, 300 K, criterion `1e-4` Ry (AuH `5e-3`), `off` -> `kramers`.
Below 2^25 entries the sector was slower on such a run, from it
faster; `auto` keeps the whole grid below 2^25 (H-Au-Au-H, faster already,
stays below it):

| Cluster | Grid points, states | Spinor basis entries | Process wall [s] | SCF [s] | GPU peak [GiB] | Energy difference [Ry] | `auto` |
|---|---|---|---|---|---|---|---|
| AuH (the example) | 7,208, 20 | 3.7e5 | 10.5 -> 13.2 | 7.0 -> 10.1 | 0.58 -> 0.50 | -4.1e-6 | whole grid |
| Au2, h = 0.4 bohr | 65,752, 30 | 4.7e6 | 12.7 -> 17.7 | 8.1 -> 13.8 | 1.5 -> 0.8 | 1.4e-7 | whole grid |
| Au2, h = 0.3 bohr | 206,136, 30 | 1.48e7 | 19.0 -> 20.5 | 14.7 -> 16.0 | 3.7 -> 1.5 | -5.0e-8 | whole grid |
| Au4 Td | 155,048, 52 | 1.80e7 | 19.8 -> 22.0 | 15.4 -> 17.6 | 4.2 -> 1.7 | 4.0e-6 | whole grid |
| H-Au-Au-H (`D2h`) | 268,096, 32 | 2.04e7 | 23.0 -> 21.2 | 16.7 -> 15.4 | 4.8 -> 2.0 | 6.5e-6 | whole grid |
| Au6 octahedron | 206,136, 76 | 3.38e7 | 27.1 -> 25.5 | 22.2 -> 18.9 | 7.3 -> 2.6 | -3.5e-6 | `kramers` |
| Au13 cuboctahedron | 113,104, 160 | 3.76e7 | 28.6 -> 22.8 | 24.3 -> 18.2 | 5.7 -> 1.8 | 3.2e-6 | `kramers` |
| Au13, h = 0.33 bohr | 201,144, 160 | 6.68e7 | 47.0 -> 26.7 | 40.2 -> 21.6 | 9.8 -> 2.8 | 2.9e-5 | `kramers` |
| Au20 Td | 268,096, 236 | 1.30e8 | 74.6 -> 29.1 | 68.9 -> 23.3 | 15.3 -> 4.8 | 2.9e-6 | `kramers` |

Gold is the one species with spin-orbit pseudopotentials in the repository,
so the floor rests on gold clusters of several groups and sizes (and AuH
and H-Au-Au-H); it is a size of the spinor basis, not of a species.

### Hybrid functionals

The Fock loop of a hybrid functional (`../README.md`, "Self-consistent
hybrid functionals") runs on the GPUs of one process through `SCF/hybrid.py`
(`DeviceExchange`, `Hamiltonian/exchange_device.py`), and on several nodes in
the MPI runner through `SCF/hybrid_mpi.py`: the ranks divide the symmetry
sectors, every sector basis on one GPU, and the pair densities of the
exchange are solved on every GPU of every rank (`MULTI_GPU.md`, "Hybrid
functionals on several nodes"). `PARSEC_MPI_EXCHANGE_TRANSPORT` (`auto`,
`cuda`, `host`) says how its orbitals travel between the ranks,
`PARSEC_MPI_EXCHANGE_CHUNK_BYTES` (256 MiB) bounds one message, and
`PARSEC_MPI_EXCHANGE_OVERLAP` (`1`, the default; `0` broadcasts first) lets
the devices start on their tiles while the orbitals of the other ranks are
broadcast.

### LDA+U and the TS correction

An isolated system with `ldaplusu_u` (and `ldaplusu_j = 0`), unpolarized or
spin-polarized, runs on the accelerated driver with the physics of
`Hamiltonian/hubbard.py` (`upot.F90`, `plusu.f90z`, `calcden.f90z`):

- The term `sum_a |phi_a,m> D_mm' <phi_a,m'|` is a separable operator like
  the KB term, whose device kernels apply `B diag(s) B.T`. A symmetric 5x5
  `D` is a combination of fifteen fixed rank-one terms, `e_i e_i^T` and
  `(e_i + e_j)(e_i + e_j)^T`, so `Hamiltonian/hubbard_device.py` appends the
  columns `phi_a,i` and `phi_a,i + phi_a,j` of every LDA+U atom to the KB
  factor once per structure, with coefficient zero, and every SCF iteration
  writes their fifteen coefficients per atom into the arrays of the signs of
  every device operator (the full-grid operator, the sector operators, their
  replicas on other devices, an opt-in FP32 shadow). No kernel changed; the
  fused stencil scatter, the filter graphs and the sector reduction
  `U_G^T B` take the term with the KB columns. `H_U` commutes with the
  point group (the occupation matrices of the orbitals of all sectors follow
  the d harmonics of the atoms), so the sector blocks are exact. A channel of
  a spin-polarized run binds its `D` with its potential.
- The occupation matrices `N` of every iteration and the LDA+U forces are
  formed where the orbitals are: the full-grid basis on its device, the basis
  of every sector on its device or spilled to the host, a basis that several
  devices share, a basis that holds the Ritz hand-over; only `(atoms, 5, 5)`
  arrays come back (the reduced projectors give the overlaps of the
  full-grid orbitals). The KB force kernels read only their own columns.
- The native and SciPy backends add the host term to their bound
  Hamiltonian (`HostHubbardHamiltonian`).
- A bulk cell at the Gamma point takes the same route, with the images of
  the cell in its projectors, also for its forces, relaxations and dynamics:
  `Forces/periodic_device.py` takes the LDA+U and TS terms from
  `Forces/compute.py::correction_forces` (the LDA+U term from the orbitals on
  the device with the projector gradients of the cell images, the TS forces
  of the device sums). Since 2026-11-19 the SCF of such a cell can take the
  sectors of its point group under `PARSEC_GAMMA_LDAU_SECTORS` and
  `PARSEC_GAMMA_TS_SECTORS` (LDA+U `auto` by default since 2026-11-21, from
  2^26 grid points x states on; TS `auto` by default since 2026-11-21, from 2^27
  for an unpolarized cell; "TS, LDA+U, meta-GGA and libxc in the Gamma sectors");
  the forces and relaxations of an LDA+U cell keep those sectors
  (`PARSEC_LDAU_SECTOR_FORCES=auto`), those of TS the full grid of the
  Fortran-derived routines, which these inputs run on.
- Since 2026-11-21 the sectors of an LDA+U run carry the KB and LDA+U
  columns of one atom per orbit (`PARSEC_LDAU_PROJECTOR_ORBITS`): an
  operation `r -> R r + t` that carries atom `a` onto `b` gives
  `P_g B_a = B_b D(R)` with the orthogonal matrix `D(R)` of the d harmonics,
  the density matrix of the sector states commutes with `P_g`, so
  `N_b = D N_a D^T`, the potential matrices follow alike and every atom of an
  orbit gives a sector the same block: the columns of the representative,
  scaled by the square root of the orbit size, with its coefficients
  (`HubbardOrbits`), and the occupation matrices of the representatives
  turned onto their orbits. The weighted sectors of a subgroup of the point
  group take LDA+U as well (`PARSEC_LDAU_WEIGHTED_SECTORS`): the occupation
  matrices of the weighted states are averaged over the full group, as the
  density is (`HubbardGroupAverage`). A spin-polarized Gamma cell stays on the
  1x1x1 k-point driver by default; the opt-in `PARSEC_LDAU_SPIN_SECTORS=auto`
  (`on`: a cluster too) takes it into the spin sectors above the floor, each
  channel writing its own potential matrices into the shared sector operators
  before its solve, and the opt-in `PARSEC_LDAU_SPIN_FULL_GRID` takes a single
  point below the floor to the full grid of one device; both are faster, but
  either can end in another magnetic state than the k-point driver. The images of the LDA+U atoms under the operations of the
  sectors are matched within `PARSEC_SYMMETRY_ATOM_TOLERANCE` (1e-4 bohr at
  least), the tolerance of the detection; a structure that keeps the group
  only to more than 1e-6 bohr (accepted within a looser tolerance) keeps the
  columns of every atom, which the representative of an orbit would give its
  offset (Ag108 with an atom 5e-4 bohr off: 7.7e-5 against 1.3e-3 Ry). The LDA+U routes of before 2026-11-21 take
  all of `PARSEC_LDAU_PROJECTOR_ORBITS=off`, `PARSEC_LDAU_WEIGHTED_SECTORS=off`,
  `PARSEC_LDAU_SPIN_SECTORS=off` (the default) and, for a cell, `PARSEC_GAMMA_LDAU_SECTORS=off`
  (each switch alone restores only its own part). The LDA+U projectors and
  their gradients find the support of
  each atom from the grid points a k-d tree finds near its images (the same
  entries bit for bit; 42.5 -> 0.3 s for the 256 atoms of an fcc Ag cell on
  64^3 points, which every LDA+U run of that size paid in its setup).
- As in `upot.F90` the d projectors are cut hard at the radius that holds
  95 % of the norm (`R_d` there still 11 % of its peak for Ag) and `R_d` is
  interpolated linearly, so the LDA+U energy steps when a grid point crosses
  the cut (Ag4, U = 4 eV: -4.0e-4, +2.0e-4, +4.0e-4 Ry) and its forces kink
  by about 7e-4 Ry/bohr where a point crosses a knot of the table. The
  forces are the derivative of the energy between those events; a central
  difference across a cut is not a test of them. The cut and the linear
  interpolation follow `upot.F90` on purpose, so that LDA+U energies stay
  those of PARSEC 1.5; the steps and the kinks are a property of that
  convention. A finer grid makes each step smaller (a step is the term of
  the one grid point that crosses the cut) (`FORCES_AND_RELAXATION.md`).
  The opt-in `PARSEC_LDAU_SMOOTH_PROJECTORS` (switch table) replaces both by
  a cubic spline of `R_d` that a taper centred on the cut takes to zero, so
  that the energy is smooth and the forces follow its slope over finite steps.
- Refused here (the command line hands them to the reference solver):
  `ldaplusu_j != 0` (PARSEC's `D` is then not symmetric and has no such
  form), the MPI SCF of several ranks.
- The k-point (and periodic spin) drivers of the reference solver run LDA+U
  on the k-point device workers: `Hamiltonian/device_bloch.py` applies
  `P_k D P_k^H` beside the KB term of its Bloch operators (the Bloch
  projectors cached per k-point, `D` uploaded with the potential), and
  `SCF/kpoint_hubbard.py` forms `w_k N_k` from the device orbitals of each
  k-point (the support rows of each atom).  Before, LDA+U kept these loops on
  the CPU workers.  The band structures (`SCF/bands.py`) and the Gamma-point
  spin runs of a wire or slab (the 1x1x1 k-point driver) take the same
  workers.

`Van_Der_Waals` forms the Hirshfeld volumes of the converged density and the
pair sum of `efvdw.f90` on one device (`Energy/dispersion_device.py`): for a
cluster, and for a bulk cell with its Hirshfeld cells and lattice shells (the
accelerated Gamma path, and every periodic driver of the reference solver
whose eigenproblems run on a CUDA device: `Energy.dispersion.set_device_dispersion`,
installed with the k-point workers). The correction adds no potential, as in
PARSEC. A cluster on the reference solver keeps the host sums.
`benchmarks/check_corrections.py` compares an input on the two solvers,
with the device memory a thread samples.
`acceleration/tests/test_spin_polarized.py` compares complete calculations
on the three backends with Fortran PARSEC.

### Fields, point charges, symmetrized densities, domains and restarts of clusters

These cluster inputs of PARSEC run on the accelerated SCF loop (also with
collinear spin); the CLI hands them to the reference solver only where a
periodic cell or spin-orbit coupling comes with them, by name:

- `Polarizability`: the seven field loops of `polar.f90`
  (`SCF/polarizability.py`) through the reference loop driver, which
  replaces the ionic potential of the reference system by `V_ion + 2 F d.r`
  for each field; the accelerated system keeps its Hamiltonian, eigensolver
  and Hartree solver, and each loop after the first starts from the input
  potential of the last iteration and the saved subspace on its devices,
  as PARSEC does. `polar.dat`, the dipole lines and the forces of the last
  loop are those of the reference CLI. PARSEC uses no point group here;
  since 2026-11-19 (`PARSEC_POLARIZABILITY_SYMMETRY=auto` where the points
  of the full grid times the states reach 1.2e7, `on` for any size;
  `SCF/symmetry_off_inputs.py`) the zero-field loop solves in the sectors of
  the group `G` of the atoms, the loops along an axis in those of the
  stabilizer `G_a = {R : R e_a = e_a}` (the system is prepared again for each
  axis, with a pseudo-atom on the axis in the detection; the first loop on a
  new group starts from the potential of the zero field with a first
  filter, the loops of one axis follow each other as in PARSEC), and a loop
  whose field an operation of `G` (that maps the grid onto itself) carries
  from a loop solved later is not solved: its energy is that loop's and its
  density the image, `rho_d(R r) = rho_d'(r)`, from which its dipole
  follows (`Polarizability run i of 7 ... the image of run j ... not
  solved` in `parsec.out`). The last loop (`-z`), whose forces and state the
  run reports, is always solved. A Td or Oh cluster solves the zero field and
  `-z`, a D2h molecule the zero field, `-x`, `-y` and `-z`. Two spin
  channels: the guard of `Spin_Symmetry_Sectors` tests the densities against
  the operations of the atoms that also keep the field axis (the group of
  the Hamiltonian of the loop); a loop that leaves the sectors all the same
  solves on a full grid that carries its field, and the loops after it stay
  there. `off`: no point group, the full grid on one GPU, or the one sector of
  `PARSEC_CUPY_SINGLE_SECTOR=on`, whose basis the GPUs of `main.py` share
  after a CHEBFF first solve (`PARSEC_CUPY_SECTOR_PLACEMENT=size`, the
  default; under `cyclic` it stays on the first GPU; the MPI runner refuses
  `Polarizability`), as before.
- `Add_Point_Charges`: the potential and the ion-charge energy are in the
  reference preparation that the accelerated one calls; the forces of the
  charges on the nuclei are added to the accelerated forces
  (`Forces/compute.py`). PARSEC finds the group of the atoms alone (a defect
  of the Fortran); since 2026-11-19 (`PARSEC_POINT_CHARGE_SYMMETRY=auto`) the
  detection matches the charges as atoms of a label of their value, and the
  run solves in the sectors of the group that carries the charges onto
  charges of their value (none where no operation does; the guard of a
  spin-polarized run tests that group too); `off`: no point group, as
  before.
- What every kept group of these inputs shares
  (`symmetry_off_inputs.kept_group_reasons`, since
  2026-11-19): none is kept under `--symmetry off`, for spinors
  (`SO_from_scratch`, non-collinear spin: their drivers take no sectors),
  for two spin channels on the full grid (`Spin_Symmetry_Sectors: off`, or
  a restart that keeps them there), and for a relaxation whose `Move_Flag`
  fixes an atom that an operation of the atoms exchanges with a movable one
  (the test of the cluster route, `Relaxation/run.symmetry_decision`, on the
  group of the atoms, which contains every group kept here; before the
  review such a relaxation stopped after its first forces with "the atoms
  do not have the symmetry of the sectors"). Those runs are the full grid
  of `off`. A run that keeps a group prints the symmetry block PARSEC 1.5
  prints for its input (`ignoresym= F` and no group for an applied field,
  the group of the atoms alone for point charges, the suppressed block of
  the fold blocks, which set `Ignore_Symmetry` in PARSEC), not the
  `Ignore_Symmetry` block of the full grid of `off`, and a NOTE names the
  group kept and the value of its switch.
- FDET (`Potential_Field`) stays on the full grid: its frozen sample field
  is a grid field without the point group of the tip atoms, which no
  operation of the atoms alone is known to keep.
- `Foldblock_Side_Length 2`: PARSEC 1.5 switches the symmetry off and so
  takes the grid without shift; since 2026-11-19
  (`PARSEC_FOLDBLOCK_SYMMETRY=auto`, also in the MPI runner) that grid stays
  and the detection keeps the operations that map its domain of whole
  blocks onto itself: the permutations of the axes the atoms allow (the
  blocks `[2m, 2m + 1]` are no images of each other under a sign change
  without the half-step shift), a mirror `x <-> y` of a Td cluster.
  `parsec.out` keeps PARSEC 1.5's warning that the symmetry is switched off,
  with a clause on the group kept, and leaves out the port's warning on
  `Ignore_Symmetry`, which `off` prints as before. `off`: no point group, as
  before.
- `Symmetrize_Charge_Density`: where the loop keeps one value per orbit of
  the symmetry sectors (or, for weighted sectors, of their full group,
  `PARSEC_SECTOR_FIELD_GROUP=full`), the average over the full point group
  of the atoms is formed on those orbits (`SCF/density_symmetry.py`): the operations
  of the group that the sectors lack (the threefold axes of a tetrahedral
  cluster over its `D_2` sectors) map each representative row to an orbit,
  and the average at that row is the reference's, term for term. The two
  channels of a spin-polarized cluster are averaged alike, each on the
  orbits (`PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto`, since 2026-11-19; they
  went to the full grid before), and so is the density of a bulk cell at the
  Gamma point over the parser's space group, whose operations need not be
  those of the sectors: the build checks on a sample of rows that the
  average stays constant on their orbits.
- `Save_Intermediate_Charge_Density`: the fields of every iteration go to
  `parsec.dat` on the host, the orbitals of the sectors or devices as one
  full-grid matrix where the file holds wavefunctions
  (`Output/iteration_fields.py`); in every SCF loop, relaxation, dynamics and
  polarizability included.
- `Cluster_Domain_Shape: ellipsoid` and `cylinder`: the slab-wise grid
  builder decides the points of these shapes with the reference test
  (`Grid/cluster.py`), and the direct Coulomb sum of the Hartree boundary
  values runs on the GPU (`Hartree/cupy_direct_boundary.py`, one thread per
  exterior point, compensated FP64 sums; `PARSEC_HARTREE_DIRECT_DEVICE`
  `auto`, the default, for these two shapes; `off` takes the host sum of a
  box instead). A box keeps its host sum by default, which now also takes
  the orbit values of a run with symmetry sectors (such a run stopped in the
  boundary before); `PARSEC_HARTREE_DIRECT_DEVICE=on` sums it on the GPU
  too.
- `Restart_From_Wfndat`: the stored subspace of `wfn.dat` is uploaded into
  the persistent CuPy eigensolver of the full grid (`SCF/restart.py`); the
  file holds states of one representation. Opt-in since 2026-11-19
  (`PARSEC_WFNDAT_RESTART_SECTORS=auto`; the default `off` keeps the route
  of before): a cluster whose stored density keeps the group of the atoms on
  the grid (test 2 of the spin guard, 0.01 electrons per channel) solves in
  its sectors from the stored potential, with a first filter instead of the
  stored orbitals, which saves memory but not time; a density that breaks
  the group prepares the run again on the full grid seeded with the stored
  orbitals.
  `PARSEC_CUPY_SINGLE_SECTOR=on` with it is handed to the reference solver.
- A spin-polarized `Restart_Run` (`parsec.dat`) of a cluster solved both
  channels on the full grid (a stored state may break the group); since
  2026-11-19 (`PARSEC_SPIN_RESTART_SECTORS=auto` where the points of the
  full grid times the states reach 2^25, `on` for any size) it keeps the
  sectors where both stored channel densities keep the group the run was
  prepared in (the atoms on the grid, with the point charges where
  `PARSEC_POINT_CHARGE_SYMMETRY=auto` keeps their group; the same test), and
  is prepared again as `off` prepares it (`Spin_Symmetry_Sectors: off`)
  where they break it, before its first iteration; the guard checks the end
  of the run as for any spin run in sectors. A smaller restart, faster on
  the full grid (O2, benzene, naphthalene), is the run of `off`.

The MPI runner refuses point charges, `Symmetrize_Charge_Density` and
`Polarizability`.

### SciPy

The SciPy backend retains the same sparse finite-difference matrix and
Kleinman--Bylander projector factors as `parsec_python`. For each block
`Q`, it evaluates

```text
H Q = A Q + V_eff[:, None] Q + B diag(sign(D)) (B.T Q)
```

and accumulates the local and nonlocal contributions into the CSR product
instead of allocating three full grid-by-block result arrays and two addition
outputs. Only the length-`N_grid` local potential changes between SCF
iterations.

### Native C++/OpenMP

The optional native extension canonicalizes and copies the finite-difference
CSR buffers and sparse projector factors once. Its cached fused Hamiltonian
then updates only the local diagonal field between iterations. For spherical
Hartree problems it also caches active-point angular geometry and every
missing exterior stencil neighbor. OpenMP may parallelize the compiled block
action, multipole/RHS construction, and Hartree CG loops. When
`OMP_NUM_THREADS` is unset, extension import detects the logical processors
available to the process and reserves four: for example, 32 detected gives a
28-thread native default. Set `OMP_NUM_THREADS` before launching Python to
override that policy explicitly. Repeated grid-vector kernels choose a useful
team no larger than that maximum, currently one worker per 8,192 points. This
avoids waking more threads than a symmetry wedge can feed while allowing
larger domains to scale to the configured maximum. The resident GPU worker
also defaults tiny (order at most 64) host OpenBLAS eigensolves to one thread;
an explicit `OPENBLAS_NUM_THREADS` remains authoritative.

The same extension can construct the active-domain finite-difference CSR
matrix with native compressed-grid lookup loops. It must reproduce the
reference row ordering, centered coefficients, and zero exterior orbital
boundary exactly. A missing or unloadable extension is an availability issue,
not permission to change algorithms.

The extension also caches active-grid coordinates for one-time ionic setup.
Local `r*V(r)`/spline interpolation, valence and NLCC density sampling, and
KB support/radial/real-harmonic loops run with OpenMP. POTRE parsing, KB
denominator construction, projector labels/signs, and sparse CSC assembly
remain visible in Python. A cached native CA/PZ-LDA evaluator handles the
repeated exchange-correlation grid loop while preserving the reference
float64 branch formulas. Under symmetry it evaluates one physical value per
orbit and applies integer orbit multiplicities to the energy quadrature.

Three pseudopotential set-up options of PARSEC run on the accelerated path
(`V_ion/double_grid.py`, `V_ion/setup_options.py`):

- `Double_Grid_Order > 1` (Ono-Hirose double grid, `nonloc.F90`): the
  projector builder of the native and CuPy preparations averages each
  projector over the `(2N-1)**3` fine points of a coarse point with NumPy,
  on the support box of the atom (the grid lookup) and on up to four host
  threads for 32 atoms or more. The matrix, and the projector gradients of
  the forces, are those of the readable builders bit for bit; the device
  Hamiltonians, symmetry sectors and force contractions take them as they
  take the ordinary projectors. C795H300 (2.71 million points, 1095 atoms,
  order 2) on an A100 node (CPUs shared with other jobs): 0.95 s for the
  projectors and 1.1 s for the gradients, against 107 s and 105 s for the
  readable builders (one thread: 2.5 s and 3.1 s).
- `Old_Interpolation_Format` (`ionpot.f90:107-112`, `forcloc.f90:201-210`):
  the local potential of a species without cubic spline interpolates `V`
  itself. The native builder adds the native density kernel of the table
  (linear inside `rs(ns-1)`, zero beyond) and the local-potential kernel of a
  zero table (zero inside, `-2Z/r` beyond); the CuPy builder and the device
  local forces compile a variant of their kernels with a per-species flag,
  only for calculations that set the option.
- A periodic cell at Gamma takes the readable periodic preparation, which
  builds the double-grid projectors and applies the Fourier filters
  (`Alpha_Filter`, `Beta1_Filter`, `Core_Filter`). A cluster's filters are
  ignored by the parser, as in PARSEC.

The double grid together with spin-orbit species (`SO_PSP`) runs on the
reference solver. The prepared-system cache key hashes the interpolation
and filter options only where a species sets them.

### CuPy

The CuPy path uses float64 for physical operators, initial eigensolver
acceptance, Rayleigh--Ritz, densities, potentials, convergence, and energies. The finite-difference operator,
projector factors and cached transpose, local potential, Chebyshev blocks, and
saved eigensolver state remain on the device across Hamiltonian applications
and SCF iterations. Synchronization is restricted to coarse transfer and
solver timing boundaries.

Every filter runs in FP64 by default. `PARSEC_CUPY_MIXED_FILTER` is an opt-in
to an FP32 later-SCF Chebyshev recurrence: `off`, the default, builds none;
`on` builds one for every operator; `auto` builds one for operators of at
least 100,000 real-space rows (`PARSEC_CUPY_MIXED_FILTER_MIN_ROWS`) and uses
it in passes with `N_grid * N_working_states^2 >= 100000000`
(`PARSEC_CUPY_MIXED_FILTER_MIN_WORK`). The recurrence uses FP32 stencil
coefficients, projector factors, vectors, and local-potential shadows. Its
output is converted back to FP64 before the generalized Ritz projection and
every SCF/energy operation, and initial CHEBDAV/CHEBFF remains entirely FP64.
Shared symmetry-sector potentials update both the FP64 field and every enabled
FP32 shadow before a solve.

The FP32 path has not been validated above 80 states: its complete-SCF check
is Si28H36, where every printed SCF energy matched the FP64 run (see
`ACCELERATION_AUDIT.md`). Compare a complete SCF with an FP64 run before
relying on it for anything larger. On an A100, one FP32 recurrence step of a
nanodiamond sector ran 1.29 to 1.31 times faster than the FP64 step and agreed
with it to 1e-7 relative. It does not reach every route: a basis shared among
devices is always filtered in FP64, `PARSEC_CUPY_DISTRIBUTED_FILTER=1` rejects
it, it refuses a stencil packed into implicit tiles, and it writes the
filtered basis into a second `N x m` array where the FP64 filter of a
single-array sector works in place.

`orbital_sector_later_filter_precision` in the backend report of a result
says what the later passes ran in: `float64` if none of them, in any sector of
any rank, took the FP32 recurrence, and otherwise how many of how many passes
did and in which sectors. Every sector solve carries that with its record, so
the root rank of an MPI run reports the sectors of the other ranks too. The
setup report, written before the SCF, can only say what was prepared: `float64`
if no sector of the process holds an FP32 recurrence, `... prepared for
sectors ...` otherwise. Where either of the two is not `float64`, the text
log adds a line `Later filter precision as run` at the end. The entry of a
calculation without symmetry sectors, `gpu_later_subspace_filter_precision`,
is still the statement of the preparation.

The stencil of a symmetry sector is stored as affine tiles of 16 rows
(`GPU_COMPACT_OPTIMIZATION.md`) where the filter work of the run in that
sector repays the packing of the tiles on the host. That is
`PARSEC_CUPY_IMPLICIT_TILE=auto`, the default, on every route: a sector that
one device filters and one whose basis or filter several devices share (8
and 16 GPUs), where the owner packs the tiles and the other devices are
given its packed arrays. The rule reads what every calculation has: the
columns of the sector (its share of `States_Num` with the subspace buffer),
the Hamiltonian applications that the input fixes per column (`FF_MaxIter`
times `Chebdav_Degree` of the first solve and eight later steps at the
lowest degree their filter falls to, 200 for four cycles of degree 30), the
devices of the process, which wait together while its host packs, and the
share of the rows that lie in regular tiles, taken from 1,024 tiles of the
stencil itself. A sector gets tiles where, per row,

```text
28 ps x (regular share - 0.50) x columns x applications  >=  70 ns x devices
```

the seconds a device saves per row, column and application against the
seconds the host takes to pack a row, both measured on A100 nodes. The rows
cancel. With four devices and 200 applications a sector whose rows are
regular to two thirds gets tiles from about 300 columns: 3,480 electrons
(442 columns a sector) do on 4, 8 and 16 GPUs, where their filter took 7%
fewer seconds. On 8 GPUs that was 0.19 s a sector against 0.18 s more
before the SCF, next to no gain; the packing of shared sectors took more
there than the 70 ns a row of the rule, and the larger sectors gained
(the table of two defaults below, `GPU_COMPACT_OPTIMIZATION.md`). A sector
with few states gets none at any number of rows,
and neither does one of many states whose rows lie to half or more in
tiles that are not regular, as those of a small sector do. The kernels do
the arithmetic of the slot-major stencil in the same order, and complete
calculations gave the same energies to the last bit, on one device per
sector and with a sector shared by two and by four.
Before the rule `auto` went by rows: 650,000 for a sector that one device
filters (`PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS`), over the square root of the
sectors that its device filters one after another (459,620 rows with two,
325,000 with four or more), and 1,100,000 for a sector that several devices
share (`PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`). Either variable, when it
is set, is again the row limit of its route, and the two at `650000` and
`1100000` give every sector the choice it had.
`0` keeps the slot-major stencil everywhere, as before, and a tile size packs
every stencil. The constants of the rule were measured on A100 nodes; `0` is
the setting for a device whose tile kernels do not gain, where the host
packing is paid for nothing. `auto` keeps the slot-major stencil for a sector
that is to hold the FP32 recurrence, and outside symmetry sectors. Tiles that
`auto` chose and that cannot be built leave the sector on the slot-major
stencil; a tile size named in the environment is built or is an error, and
the other devices of a group build the tiles of their owner or the
calculation stops.
`orbital_sector_finite_difference_storage` in the backend report names the
layout that was built.

Large generalized Rayleigh--Ritz overlap matrices form only the lower triangle
of `X.T @ X`, from FP64 GEMMs on trailing columns (at most
`PARSEC_CUPY_RITZ_GRAM_SLABS` column slabs, 8 by default; slabs of 128
columns for a basis that several devices share, below). cuBLAS DSYRK does
as few operations, but ran at 8 to 9 Tflop/s on an A100 where these products
reach 13 to 17. Set `PARSEC_CUPY_RITZ_SYRK=on` for DSYRK or `off` for one
full GEMM. Optional
asynchronous stage timing is enabled with `PARSEC_CUPY_STAGE_TIMING=1`; it
reports first-CHEBDAV and later-SUBSPACE spectral-bound, filtering,
orthogonalization, projection, residual/locking, cleanup, Ritz-Hamiltonian,
overlap, and rotation subtotals without synchronizing every Hamiltonian
application.

The projection `X.T @ (H X)` of that solve is read in its lower triangle only
as well. Where `H X` is held as a second array beside the basis, and on every
device of a sector whose basis is spread over several devices, it is
multiplied a slab of columns at a time by the basis columns from the slab's
first one onwards: at most `PARSEC_CUPY_RITZ_GRAM_SLABS` slabs, 8 by default,
which is 56% of the single full FP64 GEMM used before. Set
`PARSEC_CUPY_RITZ_GRAM_SLABS=1` to restore that full product.

The slabs of both Gram matrices, and the rounds of a shared basis below, are
cut at whole multiples of 64 columns (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, 64 by
default). For a tall product `A.T @ B` cuBLAS takes on an A100 the time of a
`B` whose columns are rounded up to the next multiple of 64. Timed there one
product at a time, 1,032,628 rows by 3,704 columns of `A`, a product took
0.44 to 0.48 ms per column of the rounded width at each of 15 widths of `B`
from 124 to 640 columns: 12.0, 13.9, 14.5 and 15.4 Tflop/s at 136, 272, 464
and 616 columns against 16.8 to 17.3 at 192, 256, 320, 384, 512 and 640. Over
24 first runs (3,480 to 39,368 electrons on 1 to 16 GPUs) the Gram stage ran
at 10.5 to 15.6 Tflop/s as its products were cut, a spread of 10.7%, and at
15.6 to 17.4 with a spread of 3.4% once every width is counted as rounded up.
The columns of `A` cost what they are on an A100: a step in them fits those
runs worse at every size tried (4.9% at 64 columns, 20% at 512).
So a slab of 64 columns and more is cut down to the multiple below it, and a
narrower one stays what it is:

- the overlap, an eighth of the columns per slab: 192 instead of 231 columns
  for a sector of 14,680 electrons (ten slabs instead of eight), 64 instead
  of 84 for 5,264 electrons and 576 instead of 616 for 39,368;
- the projection of the one-array route, the columns of the streaming budget
  per slab: 192 instead of 206 for 14,680 electrons, 128 instead of 130, 164
  and 186 for 5,264, 10,456 and 19,392, and 64 instead of 112 for 7,120;
- the projection of a shared basis, whose product is as wide as a round: see
  the one-block step below.

The narrower last slab of a matrix is multiplied by its own columns only, the
smallest product. A slab only narrows: the bytes of a budget stay the most
that a slab takes, the rotation tile of the one-array route keeps its rows,
and the fit rule of a shared basis counts the slabs of the budget as before,
so the same bases are shared. Both matrices are summed from other pieces:
the results differ by round-off from those of the former widths.
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` selects those, bit for bit, on every
route, and the serial control of the runner names it. (A shared basis takes
`PARSEC_CUPY_RITZ_GRAM_SLABS=8` beside it since: its overlap is cut by a
width of its own, below.)
The rule ran on A100 nodes, the default against
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` in one allocation: the last table of
switches, below, has those runs. Before them its widths ran there for two cases,
set by hand with the switches that existed (first runs, one allocation, the
same code on both sides): 14,680 electrons on 4 GPUs with slabs of 192
columns (`PARSEC_CUPY_STREAMING_RITZ_BYTES=4001043456`) and ten overlap slabs
of 185 (`PARSEC_CUPY_RITZ_GRAM_SLABS=10`) took 118.5 instead of 122.2 s of
SCF, the projection stage 66.7 instead of 81.7 thread-seconds, at 42,861
instead of 43,141 MiB on the fullest device and with the total energy 2e-8 Ry
apart; 5,264 electrons on 1 GPU with slabs of 128 columns and eleven of 61
took 54.5 instead of 56.4 s, the stage 3.3 instead of 5.2 s, 9e-10 Ry apart.
That budget also shortened the rotation tile, which the rule leaves alone:
the peak of the rule is that of a multiple of 1 (43,097 MiB on both sides for
14,680 electrons on 4 GPUs, where the rule itself took 118.4 instead of 121.6
s of SCF and 64.2 instead of 81.2 thread-seconds in the projection stage).
Another device can cost otherwise. On a workstation GPU (RTX 5070, CUDA
12.9) a product took the same time at every width of `B` between two
multiples of 64, to 0.1 ms in 104 to 520, but the columns of `A` cost in
coarse steps there as well, so that more slabs do not always gain: the two
Gram matrices of one array, timed alone with slabs cut by 64 against 1, took
1.87 instead of 2.36 s at 400,000 rows by 665 columns, 6.44 instead of 6.68
s at 250,000 by 1,842, 2.51 instead of 2.28 s at 300,000 by 897 (slabs of 64
for 112 and 113) and the same at 442 columns. `1` is the setting for a
device on which the stage does not gain. A complete SCF there (864
electrons, 280,790 rows per sector, slabs of 30 columns by a budget) gave
with `1` the density, eigenvalues, energies and potentials of the code
before the rule to the last bit, and so did the default, whose slabs no
sector that small reaches; a multiple of 8 named for it moved the total
energy by 1e-10 Ry in the same nine steps.

On that GPU the number of columns of `B` also selects the kernel. cuBLAS
12.9.1 runs a tall `A.T @ B` (CuPy 14.0.1 passes `B` as the transposed
operand of one DGEMM) in one of two ways, by whether `B` has an odd or an
even number of columns, whatever `A` has. Timed one product at a time on the
idle device at 1,065,944 rows:

| Columns of `B` | Time of a product |
|---|---|
| odd | 6.2 to 13.9 ms up to 63 columns of `B` by 31 of `A`, 24.8 ms by 64 |
| even | 221.7 ms (208 ns per row) for every size from 2 by 2 to 66 by 64 |
| one | 0.1 to 1.7 ms |

The time of the even kernel follows the rows alone (4.2 ms at 20,000 rows,
55.2 at 266,486, 441 at 2,131,888) until the operands are wide, and then
grows in whole multiples of that: 110, 221 and 331 ms at 266,486 rows for
64, 128 and 256 columns of `B` by 665 of `A`, where 63, 127 and 255 columns
took 68, 147 and 276 ms. The call returns at once and the device is busy
that long. cuSOLVER runs into the same kernel inside its Householder QR: 80
ms for 23 columns at 1,065,944 rows, 285 for 22, 292 for 24, 771 for 38 and
2.1 s for 54.

A basis of 30 columns, the 24 states and 6 buffer states of a small
molecule, forms each of its two Gram matrices from seven slabs of 4 columns
and one of 2: sixteen even products, 3.54 of the 3.68 s that an SCF step
took, where the 38 columns of 32 states, in slabs of 5 and 3, took 0.32 s
for the whole step. A product `A.T @ B` of two tall arrays whose `B` has an
even number of columns below 64 is therefore formed in two pieces, from the
columns of `B` but the last and from its last column
(`Eigensolvers/tall_product.py`): in the slabs and full products of the Ritz
step on one device, in the Gram matrices and projections of the
orthogonalization and in the projection of CHEBDAV, whose block is six
columns wide by default. From 64 columns on a product is handed over whole,
as before: the slabs of a large basis are whole multiples of 64 by the rule
above, and neither an odd width nor a single column beside each slab has
been timed on A100 nodes. The products that the devices of a shared basis
form for its projection round by round are handed over whole at every
width as well; they run on those nodes only. The two pieces are other sums
than the one product, so results differ by round-off;
`PARSEC_CUPY_TALL_PRODUCT_SPLIT=0` hands every product over whole and gives
the former bits. The serial control of the runner does not name it.

Measured with CF4 on 1,065,944 grid points (`Eigensolver: chebff`,
`Matvec_Blocksize: 16`, `Subspace_Buffer_Size: 6`, FP64, idle RTX 5070,
first runs; seconds of "Diagonalization time" in each of SCF steps 3 to 6,
and of step 1):

| `States_Num` | Columns | Slabs | Step before | Step now | First step before | First step now |
|---|---|---|---|---|---|---|
| 17 | 23 | 3 and 2 | 0.65 | 0.20 | 0.84 | 0.86 |
| 24 | 30 | 4 and 2 | 3.68 | 0.25 | 1.89 | 1.44 |
| 31 | 37 | 5 and 2 | 0.74 | 0.30 | 2.20 | 2.24 |
| 32 | 38 | 5 and 3 | 0.32 | 0.32 | 2.98 | 2.59 |
| 40 | 46 | 6 and 4 | 3.79 | 0.39 | 3.26 | 2.86 |
| 48 | 54 | 7 and 5 | 0.40 | 0.40 | 3.93 | 3.53 |

The energies of all six steps (eight decimals) are the same in every case,
and so are the eigenvalues but for one unit in the tenth decimal of seven of
them and of one residual, in four of the six runs. With
`PARSEC_CUPY_TALL_PRODUCT_SPLIT=0` every printed number and the times are
those of the code before. The largest part of what is left of the first
step is the QR of its two cycles: 0.69 of 1.51 s for 24 states and 1.60 of
2.61 s for 32, with the stage timing on. The first solve by CHEBDAV of the
24 states took 5.4 instead of 34.5 s with blocks of 6 columns and 6.7
instead of 25.7 s with blocks of 14; with blocks of 15 it took 12.9 and
13.1 s.

Two switches that exist shorten a basis of fewer than 64 columns further on
that GPU, with the same total energies: `PARSEC_CUPY_RITZ_GRAM_SLABS=1`
forms each Gram matrix from one product instead of eight (0.17 instead of
0.25 s per step for 24 states, 0.31 instead of 0.40 for 48), and
`PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ=1` spares the first solve its QR where
the audits pass (with both, a first step of 0.91 instead of 1.44 s for 24
states and of 1.79 instead of 3.53 s for 48). Neither default is changed:
the slab count is set for large bases on A100 nodes, and the generalized
first solve is an opt-in.

The overlap of a basis that the devices of a sector share (below) is cut by
a width instead of a count. Whether a basis is shared is decided per sector
and not by the number of GPUs: a process that drives more devices than it
solves sectors gives each sector a group of them, and the basis of a sector
with a group is shared where the rule for that allows it
(`distributed_state.shared_basis_devices`). With the four sectors of the
measured series that is so on 8 and 16 GPUs only; a system with fewer
sectors than the devices of its process, one sector on 2 or 4 GPUs or two
on 4, shares its bases there as well and takes this cut. The slabs are 128
columns wide
(`PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS`), and one slab, the full
product, where the basis has no more. A slab of `w` of `m` columns is
multiplied by the columns from its own first one onwards, so the slabs form
about `(1 + w / m) / 2` of the full product: what lies above the diagonal
inside a slab is formed for nothing, and an eighth of the columns per slab
widens with the basis. Eight slabs of 4,928 states are 576 columns wide and
form 0.557 of the full product, slabs of 128 form 0.513. Of the whole
multiples of 64 the narrowest ran slower. In first runs on A100 80 GB nodes
the overlap of the row blocks, by the products of its slabs and the seconds
it took, ran at 12.8 to 16.0 Tflop/s per device in slabs of 64 columns
(5,264 and 7,120 electrons on 8 and 16 GPUs), at 17.0 to 17.6 in slabs of
128, 192 and 256 (10,456, 14,680 and 19,392 electrons) and at 15.8 to 16.9
in slabs of 320, 448 and 576 (23,768, 29,576 and 39,368): 128 columns are
the narrowest that ran at the rate of the best. With the count named so
that the slabs are 128 columns wide, beside the eight slabs in one
allocation (first runs), the overlap of a sector took 26.79 instead of 31.46
s for 39,368 electrons on 16 GPUs (27.81 s in slabs of 256), 11.04 instead
of 12.18 s for 29,576 and 6.57 instead of 7.14 s for 23,768, and on 8 GPUs
22.53 instead of 25.83 s for 29,576 and 13.59 instead of 14.97 s for 23,768,
at 16.7 to 17.3 Tflop/s; the total energies moved by 2e-8 to 1.6e-7 Ry,
the lower triangle being summed from other pieces. Those are the slabs that
the rule cuts. A sector of 1,017 to 1,528 columns keeps its slabs and its
bits, since an eighth of those is cut to 128 as well (10,456 electrons).
Where the rule changes the cut otherwise it has not run on an A100: from
1,529 to 2,552 columns slabs of 192 and 256 become 128, for 3 to 5% fewer
operations (14,680 and 19,392 electrons), and below 1,017 columns slabs of
64 and fewer become 128 and form 7 to 13% more (3,480 to 7,120 electrons,
where the overlap of a sector takes 0.04 to 0.47 s of a run).
`PARSEC_CUPY_RITZ_GRAM_SLABS=8` gives the former slabs of every step, bit
for bit: a count that is named is the cut of every route, and the width is
then not read. `overlap_columns` of a shared sector in `timing.json` names
the slabs that ran. A sector on one device keeps the count. Its overlap is
the same product over all its rows, but its records time it together with
the projection, so no run says what another width gives there, and on a
workstation GPU more slabs did not always gain (above). The projection of
the step with three blocks keeps the count as well.

The buffer of a sector on one device (a basis that no group of devices
shares, as every sector of the measured series has on 1, 2 and 4 GPUs),
which is the slab of its projection
and then the tile of its rotation, is an eighth of the basis, at least 1 GiB
and at most 2 GiB (`PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES`; 4 GiB before). A
basis of up to 16 GiB keeps the buffer it had; one of 16 to 32 GiB took its
eighth and takes 2 GiB, and a larger one took 4. First runs on A100 80 GB
nodes with the budget set to 2 GiB, against the default of 4 in one
allocation: 14,680 electrons on 2 GPUs peaked at 78,109 instead of 80,157
MiB on the fullest device in 231.4 against 231.6 s, and on 4 GPUs at 41,049
instead of 43,141 MiB in 123.9 against 124.5 s; 19,392 electrons on 4 GPUs at
58,155 instead of 60,251 MiB in 180.7 against 180.5 s. Their slabs are 64
columns wide where they were 192 and 128, their tiles half as high, and the
total energies moved by 5e-9 to 5e-8 Ry. With 3 GiB the slabs of 19,392
electrons stay 128 columns under a lower tile: 59,227 MiB, 180.4 s and the
total energy to the last bit. The sector of 10,456 electrons (18.8 GiB) is
cut that way by the limit, 128 columns under a tile of 2.00 instead of 2.36
GiB, which has not run. Slabs of 64 columns are one whole multiple. A
narrower slab pays for 64 columns in every product and none has been timed
alone, so a sector of more than 4,194,304 rows, of which 2 GiB hold fewer
than 64 columns, keeps slabs of 64 columns in 2 to 4 GiB, and from 8,388,608
rows the 4 GiB and the narrower slab that it had. That guard is derived from
the slabs of 64 columns and not measured. It acts only for a sector of more
than 4,194,304 rows whose basis lies on one device and holds more than 16
GiB, and on A100 80 GB nodes no case of the measured series has one: its
only sectors of that many rows, those of 39,368 electrons (5,660,200 rows,
207.8 GiB each), are shared by four devices. The limit acts where
`PARSEC_CUPY_STREAMING_RITZ_BYTES` is not set: a budget that is set is the
buffer as before. No step of a shared
basis reads it; their slabs start from the 4 GiB as they did (below).
`4294967296` gives the former buffer for a basis of any size, bit for bit,
and the serial control of the runner names it; the budget at that value does
not, since a basis below 32 GiB took an eighth of itself. The rule that
decides where the states of the sectors are kept counts the buffer as the
step takes it, and its share of a device has come down with that count
(`MULTI_GPU.md`).

The tile of that rotation takes the bytes of the budget, so the buffer is
the budget also where the slab is cut down to a whole multiple: 2.00 GiB for
slabs of 64 of the 103 columns that 2 GiB hold. A tile only as tall as the
slab that was cut would make the buffer smaller, but the rows would be
rotated in tiles of another height, and cuBLAS does not form the product of
a tile by the same sums at every height: on a workstation GPU (RTX 5070,
CUDA 12.9) one random basis, rotated with tiles of 12 to 17 heights up to
all of its rows, came out with other bits at some heights for every one of
eight shapes from 257 x 17 to 400,000 x 442 and 120,000 x 1,314, by 8e-13
at most where the entries are of the order of 30. No such tile is offered.

The steps of that solve, and the placement of the sector bases and of the
GPU Hartree objects, take by default the routes that measured fastest or
smallest on A100 80 GB nodes (nanodiamond sectors of 19 to 84 GiB). Each has
a setting that selects the former one:

| Environment variable | Default | Former route |
|---|---|---|
| `PARSEC_CUPY_RITZ_SYRK` | `auto`: overlap from GEMMs on trailing columns | `on`: DSYRK |
| `PARSEC_CUPY_RITZ_DENSE_BACKEND` | `device`: the small generalized problem is solved by cuSOLVER where the Gram matrices were formed | `host`: download, host LAPACK, upload |
| `PARSEC_CUPY_RITZ_CONDITION_HELPER` | `auto`: beside that solve on the owner of a shared basis, another device of the sector estimates the condition number of the overlap, never the device that holds the Hartree objects. `any` asks that device too where the sector has no other and the sharing rule counts the room for the estimate; it was the default for a while (below); ranks that share a basis over MPI (`Eigensolvers/mpi_shared_basis.py`): rank 1 estimates it for the root (same bits) | `off`: the owner estimates it first |
| `PARSEC_MPI_SHARED_RITZ` | `auto`: a basis that MPI ranks share takes the Ritz step with each rank's row block in the memory of its columns (`block`: one share of the basis per rank and the buffers of a round, the exchanges of the rounds; cells and clusters) | `rounds`: one row block beside the columns, `H X` formed and projected a round at a time (two shares; the former default of a cell); `rows`: the two row blocks (three shares; the former default of a cluster) |
| `PARSEC_KPOINT_RITZ` | `auto`: the pools take `block` (each rank's columns column-major in an allocation with room for its rows; filter, moves and Rayleigh-Ritz in place, one share of the basis, five moves of it per pass, residuals from `H` on the rotated columns) where `rows` would need more than half of a device or plan worse (fewer pools, subspaces on the host, no fit), else `rows` | `rows`: row-major columns, the filter into a second block, three shares in a Rayleigh-Ritz pass, four moves (the former route, bit for bit); `block` always |
| `PARSEC_HARTREE_DIAGONAL_BOUNDARY` | `auto`: a Hartree problem whose group permutes the axes (the weighted sectors of a tetrahedral or cubic cluster) builds its GPU boundary on the wedge of the diagonal members of the group, D2 for D2d (see "Weighted sectors of a larger subgroup") | `off`: the kernels under the identity on the full grid |
| `PARSEC_HARTREE_BOUNDARY_ROWS` | `wedge` (since 2026-11-19): that boundary evaluates its values at the exterior points of the stencils of the rows of the Hartree wedge only and assembles the right-hand side there (Td: a sixth of the exterior points of D2; the moments are summed on the diagonal wedge either way; the same right-hand side to round-off, test) | `diagonal`: the values at the exterior points of every row of the diagonal wedge, the right-hand side gathered from its rows, bit for bit as before |
| `PARSEC_CUPY_SECTOR_MARGIN` | `auto`: the margin of the first counts of weighted sectors (10 %) is cut where the devices cannot hold it, to the largest share whose counted states after one growth take at most `PARSEC_CUPY_SECTOR_MARGIN_BUDGET` (0.82) of a device; a sector so cut that misses states grows by its estimated deficit, at most 2 % a step, from a new start | `fixed`: the full margin and the growth by half of before everywhere; a run whose devices hold the full margin is the same either way |
| `PARSEC_WEIGHTED_FIRST_SHARE` | `auto` (the default since 2026-11-21): `orbitals` in one process where the basis of the full grid (grid points times the states and the buffer) reaches `sector_shares.MINIMUM_ENTRIES` (2^27), `even` below it and on the ranks of an MPI run (`sector_shares.weighted_first_share_route`; every rank takes the same value from its MPI context). One node of 4 A100, one process, first runs with empty kernel caches (the second run in brackets), `even` -> `orbitals`, SCF seconds: nanodiamonds k06 16.3 -> 11.0 (10.2 -> 5.9), fullest GPU 3.2 -> 1.9 GiB; k08 39.0 -> 19.5 (32.3 -> 14.0), 8.5 -> 4.5 GiB; k10 135.9 -> 56.4, 32.2 -> 16.8 GiB (with the counts of the shares the placement gives every sector the device pool of the four GPUs, `PARSEC_CUPY_SECTOR_PLACEMENT=size`; with the even counts one GPU each); the spin-polarized negative vacancy of diamond 4^3 (3.8e8 entries) 21.5 -> 20.9 and 5^3 (1.5e9) 25.1 -> 13.6 (9.7 -> 6.6), 1.9 -> 1.2 GiB; energies within 1.1e-5 Ry (the criterion 2e-4 Ry). Checked on other species and on one GPU, SCF seconds: aluminium cuboctahedra on 4 A100, Al147 15.8 -> 14.8 (23 -> 20 iterations, dE 2e-8 Ry) and Al309 19.9 -> 19.3 (dE 1e-6); on one A100 the 4^3 and 5^3 vacancies 9.6 -> 9.2 and 14.4 -> 13.8, k06 14.8 -> 14.4, k08 37.6 -> 37.4 (dE 2.2e-6 to 1.9e-5). Below the floor: the vacancy in 3^3 (6.9e7) took 9.6 -> 13.4 s on its first run (twice; 4.7 -> 3.2 s on the second), and SF6, CF4, Na55, Si86, Si113, CH4 and H2 on one A100 the same times, so `auto` keeps the even share there | `even` (the default before 2026-11-21): every weighted sector starts with the even share of the states and a margin of 10 % (`Eigensolvers/symmetry.first_sector_count`); `orbitals` (2026-11-19, every run): each starts with the share of the valence orbitals of the atoms it holds (the characters of their channels at the operations that fix the atoms, `Symmetry/sector_shares.py`) and a margin of 3 %; the same count in every sector (bulk diamond and silicon cells) keeps the even share. The shares match those of the lowest levels (nanodiamonds k02 to k14 to the third digit, bare Si29/Si47/Si71 to one level). Measured on A100, first run / second run: k06, k08 and k10 on one node of 4 GPUs, first solve 9.1 / 13.2 -> 6.6 / 5.4 s, 26.9 / 21.5 -> 20.1 / 21.4 s, 107.5 / 104.9 -> 91.3 / 90.9 s (whole run of k10 349 / 331 -> 322 / 314 s), the same steps, energies within the benchmark criterion (1.1e-5 Ry); SF6, CF4, Na55 and the Si clusters on one GPU the same times (steps -2 to +1; H2 PBE 36 -> 23, CH4 36 -> 30). Not taken by `auto` in MPI runs: in the MPI runner on two nodes k10 took 14.1 instead of 12.8 s and 9.1 instead of 8.1 GiB, the smaller counts of two sectors falling below the work at which a basis is divided over the devices of its group (`sector_bases_divisible`), which changed the layout of the sectors |
| `PARSEC_ORBITAL_GROUP` | `auto`: where the full point group of signed permutations is larger than its abelian subgroup and that solves less, the orbitals use the one-dimensional sectors of the subgroup that `Symmetry/orbital_groups.py` chooses, with weights (Td: D2d, sectors of about a quarter of the D2 basis; see "Weighted sectors of a larger subgroup"); spin-orbit coupling, hybrids and meta-GGAs keep the abelian group, and since 2026-11-19 `Fix_Number_Eigenvalues`, libxc, `occup.in` and collinear spin take the weighted sectors under the four switches below (LDA+U since 2026-11-21 under `PARSEC_LDAU_WEIGHTED_SECTORS`) | `abelian`: the reflection or commuting signed-permutation group of before (bit for bit with `PARSEC_SECTOR_PROJECTOR_ORBITS=weighted`, which keeps the projectors of every atom in abelian sectors; a periodic cell also needs `PARSEC_PERIODIC_ABELIAN_TRANSLATIONS=off`, else its abelian group takes translations) |
| `PARSEC_WEIGHTED_STATE_COUNT` | `auto` (since 2026-11-22): the states that weighted sectors return to the SCF hold `States_Num` states counted with their weights, and where the window ends inside a level (a state of weight 3 where one state is left, a level that several solved states hold, also where the weights add up exactly but the next solved state belongs to the level) that level shares the states left by weight (`Eigensolvers/symmetry.py::CuPySymmetrySCFEigensolver._weighted_selection`). Its states are the solved states within `max(Diag_Tolerance, 1e-4) max(1, |e|)` Ry of the last one (at most `1e-3 max(1, |e|)`), selected or not, that can be its partners: states of sectors of its weight (one sector, or two that a symmetry of the cell beyond the group of the sectors exchanges, as the 12-fold levels of bcc Na 4x4x4 in two sectors of weight 6), of a sector whose partner mask shares a bit with its own or where neither sector has a mask (the 24-fold Fermi level of bcc Na 4x4x4 over sectors of weights 3 and 6: `States_Num` 66, 70 and 74 at 3000 K converge in 9 steps, where the full grid, `off` and a split by weight did not in 150), and any state within 1e-8 Ry (`weighted_level_width`, `CuPySymmetrySCFEigensolver._level_partners`; a distinct level of weight 1 1.75e-5 Ry above a t level of weight 3 in an Na15 cluster of Oh at 3000 K stays apart, which keeps the energy of the full grid within 7e-8 Ry instead of 7.3e-6 Ry): the filters leave the two states of one level up to 4.3e-5 Ry apart at the default tolerance and 8.7e-5 Ry after a CHEBFF first solve, and the later SCF steps filter once per step whatever `Diag_Tolerance` is (CHEBFF at `Diag_Tolerance` 1e-8 left the pair 1.7e-8 Ry apart; a width of 1e-8 Ry, the first version of this switch, then split the level 3 + 1, whose density follows the rotation of the two states inside it, and the SCF did not converge in 150 steps). The Fermi level, the occupations, the density and the energy count `States_Num` states, as the full grid does, and the density of that level stays symmetric where every partner of it joins (a level over sectors of different weights, one with a partner mask that the other lacks, left more than 1e-8 Ry apart by the eigensolver, is shared in part). The lists of the result hold `States_Num` states, as the full grid's: a shared level lists the whole states of its shares in order (shares 1.5 + 1.5 list three states, two of the first), each with the occupation of the shares under it, so that the listed occupations add up to the electrons (`SCF/single_point.py::_state_repeats`); a solved state can then list no state (a share of 0.5 or less after the others of its level); the levels of the k-points of the TRIM supercell route take every partner of the states of such a level, each with its part of the occupation of its solved state (`s f / w` for a share `s` of a state of weight `w` at the Fermi occupation `f`, as the density spreads it, so that the levels of the k-points add up to the electrons; `CuPySymmetryOrbitals.whole_levels`, `SCF/supercell_levels.py`: the cube of Si8 2x2x2 keeps its levels, Fermi level and non-self-consistent levels of the k-point route), and skip solved states that list nothing, and the orbitals of the result keep the occupations of the solved states for LDA+U. Runs with `occup.in` and the spin channels keep the whole level. It matters where the states at the end of the window carry electrons (Fermi smearing in a metal, a small window): conventional bcc Na 2x2x2 (`States_Num` 17, CHEBDAV, SCF to 1e-10 Ry, A100), weighted sectors against the full grid 8.7e-9 -> 2.4e-11 Ry at 300 K, 2.6e-4 -> 2.0e-7 Ry at 1000 K, 3.8e-3 -> 1.1e-5 Ry at 3000 K (the abelian sectors, which take some partners of that level as the full grid does, 2e-13, 8e-8 and 4e-6 Ry from it; with `States_Num` 40 every route within 3e-11 Ry at 300 and 3000 K), bcc K 2x2x2 at 1000 K 3.8e-4 -> 4.6e-7 Ry; the same energy at 19 SCF steps (a share of the last state alone, without its partner state, did not converge to 1e-10 Ry: the density of the pair then follows the rotation of the two states inside the level). At the default tolerances (`Diag_Tolerance` 1e-4, criteria 1e-5 to 1e-7, CHEBDAV and CHEBFF) that cell converges in 7 to 11 steps (the full grid in 18 to 32, a width of 1e-8 Ry not in 150) to the energies of 1e-10 Ry; with `States_Num` 14, 16 and 18 in 9 steps, 1.4e-7 to 2.4e-7 Ry from the full grid at 1000 K and 6e-6 to 1.4e-5 Ry at 3000 K; `States_Num` 16 (13 states and the first state of the 6-fold level) at 1e-10 Ry in 19 and 17 steps, 2.2e-7 Ry from the full grid at 1000 K (before: not converged in 150 steps). bcc Na 4x4x4 (128 atoms, `States_Num` 96, 3000 K, criterion 1e-6): 9 steps (a width of 1e-8 Ry 107, the full grid and `off` not converged in 150). At 1e-10 Ry a width of 1e-4 Ry gave the energies of 1e-8 Ry bit for bit (`States_Num` 13 to 19). fcc Al, bcc Li and fcc Ni at 1000 K and Si at 300 K end between whole levels: the same energies under both values (within 9e-11 Ry of the full grid). `parsec.dat` of a run with such a level holds every partner of its solved states with that occupation (also where the shares are whole numbers below the weights of their sectors, as 2 + 2 of a level of two states of weight 3; `acceleration/SCF/single_point.whole_partner_states`, `Output/parsec_dat.result_channels`), so that its orbitals give the density of the file to round-off, while `parsec.out` lists `States_Num` states | `off`: every partner of the last state counts, as before (19 states in that Na cell) |
| `PARSEC_ORBITAL_GROUP_FIXED_COUNT` | `auto` (since 2026-11-19): `Fix_Number_Eigenvalues` takes the weighted sectors of the orbital group (`Symmetry/weighted_inputs.py`): the first solve brackets the lowest `States_Num` levels counted with their weights and every sector keeps the levels it holds among them (`_fix_sector_states`, as before by weight), on one process and on the MPI ranks. Nanodiamond k06 (C795H300, Td, `States_Num` 1920, a restart of the converged run; one node of 4 A100, first runs): SCF 62.5 -> 11.8 s, the run 79.6 -> 28.7 s, GPU peak 12.5 -> 3.2 GiB, the energy within 8e-7 Ry (criterion 2e-4 Ry); the MPI runner on 2 nodes (one rank with 4 A100 each; from the atomic densities, as the runner of that date read no restart file) SCF 17.5 -> 3.5 s, 4.4 -> 1.4 GiB, the energy within 9e-7 Ry. | `off`: the abelian group, as before |
| `PARSEC_FIXED_STATES_BUFFER` | `auto` (default, since 2026-11-21): `Fix_Number_Eigenvalues` in the symmetry sectors of the accelerated driver (`CuPySymmetrySCFEigensolver`, clusters and Gamma cells) keeps the eigensolver's working buffer (`Subspace_Buffer_Size`) and the state counts each sector finds for itself (its regular trims and the spectral bracket), as a run without the keyword, and a NOTE of `parsec.out` says so; the run still solves `States_Num` states and the restart file must hold that many levels (PARSEC error 404). The full grid, the reference solver and the k-point drivers keep PARSEC's `nn = nec` (`States_Num` vectors, no buffer) under both values (`models.fixed_states_sector_buffer`). A restart never reads the stored orbitals (`restart_run.F90` resets the computed states to zero, here as in PARSEC), so its first solve starts anew; in the sectors the counts of that solve (CHEBFF: unconverged), frozen without a buffer, lose occupied levels where few states are empty. Nanodiamond k06 (C795H300, `States_Num` 1746 = 1740 occupied + 6, a restart of the converged run without stored orbitals; one node of 4 A100, first runs): weighted sectors -9435.71624047 Ry in 39 iterations (off) against -9437.81997051 Ry in 13 (auto: the restart without the keyword bit for bit; from the atoms -9437.81996698), SCF 18.0 -> 11.0 s; abelian sectors (`PARSEC_ORBITAL_GROUP_FIXED_COUNT=off`) -9432.97117125 in 27 against -9437.81997069 in 16, SCF 53.3 -> 36.0 s, peak 11.2 -> 8.6 GiB. Si28H36 (`States_Num` 80 = 74 + 6): within 2.5e-6 Ry either way (criterion 1e-4 Ry), the SCF 1.48 -> 1.70 s with CHEBDAV (1 against 6 iterations) and 2.06 -> 2.92 s with CHEBFF (14 against 25, those of the restart without the keyword); on the full grid (`--symmetry off`) the two values give the output of before line for line (CHEBDAV -263.51152278 Ry in 6 iterations, CHEBFF -263.51152284 in 21; one A100). | `off`: no buffer and the sector counts of the first solve frozen (`models.fixed_states_keep_parsec_counts`, `CuPySymmetrySCFEigensolver._fix_sector_states`), no NOTE, as before |
| `PARSEC_STATES_CLOSE_MULTIPLET` | `off` (default): `States_Num` as given; a loop that ends without convergence while its highest computed state holds an occupation says so (`Output/state_truncation.py`, under every value). Opt-in `auto` (since 2026-11-22; `SCF/state_extension.py`): a loop on the full grid (`SCF.run_scf` and the spin loop of `run_spin_scf`: the reference solver, the full grid of the accelerated driver) or in one sector of the identity (the full grid of the MPI runner, whose solver grows its count) that stalls (its residual above the criterion and fallen by less than a factor 2 over three iterations, counted from the third iteration after the start or after its last extension) while its highest returned state holds more than 3 x `Convergence_Criterion` takes the levels of the eigensolver's buffer down to the first whose Fermi occupation is below that, from the next iteration on (the mixer begins a new history; again while it keeps stalling, at most eight times; a NOTE each time). A loop that continues it (an ionic step, a hybrid's inner loops) keeps the raised count and adds none of its own. A cut degenerate multiplet holds the residual at about 0.06 times the occupation of its states. A100, first runs, criterion 1e-8 Ry unless named: an 8-atom silicon cube (PBE, h 0.5 bohr, 300 K) with 20 states 52 -> 32 iterations on the full grid (SCF 1.24 -> 0.93 s) and no convergence in 100 -> 32 on the reference solver (20 -> 23 states, -63.079188 Ry as with 24 states), with 21 states 53 -> 31 and 58 -> 31; the same cube (h 0.45 bohr, criterion 1e-6) at 450 and 600 K, which off does not converge (residual 2e-6 and 2e-5), 17 and 21 iterations; a diamond C8 cube at 3000 K with 18 states not converged -> 49 iterations (18 -> 23 -> 26 states, -90.10108489 Ry; 24 states not converged -> 40, the same energy), and the MPI runner on 1 and 2 ranks the same states and energy in 39. Where off converges without a plateau it changes nothing (bit for bit): the C8 cube at 2e-4 and 1e-5 Ry and at 1500 K, the silicon cubes at 2e-4 Ry, displaced silicon cubes, a PBE0 run (37 iterations both), 24 silicon states. Why it is not the default: displacing one C8 atom by 0.5-3.5 mbohr leaves a plateau of about seven iterations (5e-6 Ry) that off passes in 29-35 iterations with 18 states; `auto` raises the count there (24 states, 33 iterations, SCF 0.80 against 0.62-0.72 s), which moves the energy by the truncation it removes (1.4e-4 Ry), and a longer stall window (eight iterations) raised it for some of these displacements and not for others, a jump of the same size between neighbouring structures: the jumps are bounded by the truncation of the structure that is not extended, not by the threshold. | `off`: `States_Num` as given, bit for bit as before (the symmetry sectors of a point group, the k-point loops, `run_scf_spin_polarized`, `occup.in` and `Fermi_Temp` 0 keep it under both values) |
| `PARSEC_ORBITAL_GROUP_LIBXC` | `auto` (since 2026-11-19): a libxc LDA or GGA (`LibXC_Functional`; not a libxc hybrid or meta-GGA) takes the weighted sectors: libxc's CUDA build evaluates the orbit fields of the subgroup with the operation table of the code's own GGA stencil (`V_xc/device_libxc.py`), a host build the fields expanded to the full grid and reduced again. One process (libxc is refused on several ranks). k06 with libxc's LDA (`lda_x lda_c_pz`, the CUDA build): SCF 31.0 -> 10.6 s, the run 46.3 -> 26.6 s, GPU peak 9.1 -> 3.6 GiB, the energy of the code's CA in the weighted sectors to the printed digits. | `off`: the abelian group, as before |
| `PARSEC_ORBITAL_GROUP_OCCUPATIONS` | `auto` (since 2026-11-19): `occup.in` takes the weighted sectors in one process: the n-th entry of the file goes to the n-th level counted with its partners, so a state of weight `w` takes `w` consecutive entries, which must be equal (`Occupations.partner_occupations`; zero past `States_Num`), and so must the entries of the degenerate partners of a level that several solved states hold (the e of Td in the sectors A1 and B1 of D2d, weight 1 each; levels within `PARTNER_LEVEL_WIDTH` = 1e-4 Ry whose sectors hold such a representation, `PointGroupReduction.sector_partner_irreps`), which the density averaged over the full group would give the mean of theirs. Where they are not (a multiplet the file fills partner by partner, or one that the order of the levels of a later step splits; checked at every SCF step and in both spin channels) the run starts again from the atoms in the abelian sectors with a `NOTE` (`driver._SpinSectorFallback`, record route `weighted`). Several MPI ranks take them too since 2026-11-21 (`PARSEC_ORBITAL_GROUP_MPI`; before, they kept the abelian group). k06 with an aufbau `occup.in`: SCF 29.8 -> 11.0 s, GPU peak 8.0 -> 3.4 GiB; k08 (C1653H508): 93.2 -> 32.7 s, 26.8 -> 8.1 GiB; the energies within the SCF criterion (5e-6 and 4e-5 Ry at 2e-4 Ry). | `off`: the abelian group, as before |
| `PARSEC_ORBITAL_GROUP_SPIN` | `auto` (since 2026-11-19): collinear spin in the symmetry sectors (`Spin_Symmetry_Sectors`) of a cluster or of a bulk cell at the Gamma point (`PARSEC_GAMMA_SPIN_SECTORS`) takes the weighted sectors in one process: the weights in the common Fermi level, the occupations, the density (averaged over the full group per channel), the band energy and the nonlocal forces of each channel (`SpinChannelResult.sector_states`), every partner in the lists of the result. The guard counts a state of weight `w` as `w` partner levels in distinct sectors (tests 1 and 3 of `SCF/spin_sectors.py`; test 2 cannot see the operations outside the subgroup in a density averaged over the group), and a run it stops starts again in the abelian sectors, whose own guard may send it on to the full grid (two `NOTE`s), or, since 2026-11-21, goes straight to the full grid where the abelian sectors would stop the same levels (`PARSEC_WEIGHTED_GUARD_SHORTCUT`). Hybrids and meta-GGAs keep the abelian group (LDA+U under `PARSEC_LDAU_WEIGHTED_SECTORS`); several MPI ranks take the weighted sectors too since 2026-11-21 (`PARSEC_ORBITAL_GROUP_MPI`), and there is no size floor (`PARSEC_ORBITAL_GROUP_SPIN_FLOOR`). One node of 4 A100, first runs, SCF: k06 neutral 64.5 -> 25.3 s, GPU peak 11.2 -> 4.1 GiB; k08 192.5 -> 66.0 s, 35.9 -> 13.4 GiB; V- in diamond 5^3 at the Gamma point (999 atoms, S = 3/2) 23.9 -> 9.8 s, 4.5 -> 1.8 GiB (3^3: SCF 4.0 -> 4.8 s, the run 8.0 -> 7.7 s); the energies within the SCF criterion (4e-6, 7e-6 and 8e-7 Ry). The O2 example (`examples/0d_hybrid_functionals/O2/pbe`, D4h, 1.41 M points, `occup.in`): the same energy, 0.26 against 0.37 s per SCF iteration, but 38 against 29 iterations (both routes hover at the noise floor of its `Diag_Tolerance` until a step falls below the criterion), the run 18.9 against 16.6 s. A state the guard stops paid the weighted loop first until 2026-11-21: k06+ at 500 K (fractional e and t2 levels at the Fermi level) took 40 s more before its full grid (since then the shortcut below: 33.5 s less than `off`). | `off`: the abelian group, as before (the sector tests of `test_spin_sectors.py` and `test_gamma_spin_sectors.py` run with it) |
| `PARSEC_ORBITAL_GROUP_MPI` | `auto` (since 2026-11-21): spin polarization and `occup.in` take the weighted sectors of `PARSEC_ORBITAL_GROUP_SPIN` and `PARSEC_ORBITAL_GROUP_OCCUPATIONS` on several MPI ranks too (the full-SCF runner), as in one process: every rank prepares the same plan, and a run that leaves them (a spin guard that the abelian sectors may pass, unequal partners of `occup.in`) is prepared again in the abelian sectors on every rank (the root sends the input with the reason, `MPISectorContext.reprepare`, every rank configures the sectors of the abelian plan and its worker loop serves the new solver; `driver._SpinSectorFallback.replace_with_abelian`), with the `NOTE` of one process; a guard stop that the abelian sectors would repeat goes straight to the full grid of the plan of `PARSEC_SPIN_FALLBACK_SHARED` (one device of the root where both channels fit there, as for the Mn atom on 2 nodes; else the full grid that the ranks share, as for k06+). Before, several ranks kept the abelian group without saying so. Two nodes, one rank with 4 A100 each, first runs, against `off`: nanodiamond k06 spin-polarized SCF 23.4 -> 13.3 s, the run 25.2 -> 15.2 s, GPU peak 3.8 -> 1.5 GiB; k08 spin SCF 67.0 -> 29.6 s, the run 69.8 -> 32.8 s, 10.5 -> 3.7 GiB; k08 with an aufbau `occup.in` SCF 26.1 -> 8.2 s, the run 28.9 -> 11.0 s; V- in diamond 4^3 (511 atoms, S = 3/2) SCF 2.8 -> 1.8 s, the run 5.2 -> 3.9 s; Si185H124 (the nanodiamond k04 with Si for C) spin SCF 11.0 -> 7.3 s, the run 12.8 -> 8.7 s, 5.2 -> 2.5 GiB; Cr2 (PBE, D4h) SCF 3.4 -> 2.9 s; the energies within the SCF criterion (4e-6, 7e-6, 4e-5, 1e-5 and 7e-6 Ry at 2e-4 Ry, Cr2 6e-8 at 1e-5 Ry), each route's energy that of one node to the printed digits where measured there; the forces of k04 spin-polarized and breathed by 1 % (`spin_forces` of the ranks, criterion 1e-6 Ry): the energy of `off` to the printed digits and its forces within 1e-6 Ry/bohr (largest 0.039), SCF 13.7 -> 10.2 s. k06 with an `occup.in` that fills its top t2 level 1, 1, 0.5 leaves them at its first occupations and runs the 20 iterations of `off` bit for bit (the run 2.8 s longer); k06+ at 500 K goes from them to the full grid that the ranks share (the shortcut below), the energy of one node. | `off`: the abelian group on several ranks, as before, now with a `NOTE` at every output level |
| `PARSEC_WEIGHTED_GUARD_SHORTCUT` | `auto` (since 2026-11-21): a spin run whose guard stops the weighted sectors (test 1, a partly filled multiplet of the start Hamiltonian, or test 3, a fractional multiplet at the end of the SCF) goes straight to the full grid where the sectors of the abelian group of the atoms would stop the same levels: a reported multiplet holds a state of a sector whose representations of the point group all have partners in two or more abelian sectors (`PointGroupReduction.sector_abelian_split`, `orbital_groups.abelian_split_sectors`; the tests of `SCF/spin_sectors.py` run again on the labels of `abelian_partner_labels`): the t1 and t2 of Td in D2, the t1u, t2g of Oh in D2h, not the e of Td (both partners in A of D2, where the abelian sectors may hold a state the weighted ones cannot). It applies to clusters and to cells whose operations take no translation (a defect at the origin of a symmorphic cell), whose abelian route keeps the group about the origin (`periodic_abelian_reduction` finds no larger group of commuting involutions); a cell with translations takes them in its abelian sectors and keeps the way through them under `PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS=off` (since 2026-11-22 `auto` gives it the shortcut too: the next row). A test-1 stop gives the result of the abelian route bit for bit: both routes start from the same Hamiltonian, and the run of the abelian sectors it skips is the one `PARSEC_ORBITAL_GROUP_SPIN=off` pays before its own full grid (the full grid from the atomic densities). A test-3 stop is predicted from the final occupations of the weighted run, relabelled for the abelian sectors: it gives the full-grid result, the reference without symmetry, which is the abelian route's wherever that route stops at its end too (so in every test-3 stop measured: the nanodiamond cations k06+ and k05+, Si185H124+ and the neutral vacancy of a 3^3 diamond cell); an abelian SCF that broke the degeneracy and ended with integer occupations would have kept its own sector result under `off`. One node of 4 A100: nanodiamond k06+ at 500 K (test 3: fractional e and t2 at the Fermi level; the full grid shared by the four GPUs) 223.0 -> 157.8 s for the run (`off` of `PARSEC_ORBITAL_GROUP_SPIN`: 191.3 s), the energy -9437.47439445 Ry in all three; the Mn atom (PBE, d shell partly filled at the start) the same energy; the Fe atom 31.6 -> 28.2 s; the neutral vacancy at the origin of a 2^3 diamond cell (63 atoms, a1^2 t2^2 at the start) 7.11 -> 5.05 s (`off` of `PARSEC_ORBITAL_GROUP_SPIN`: 5.58 s), -717.56894663 Ry in all three; test 3, the run with shortcut `off` -> `auto` (`off` of `PARSEC_ORBITAL_GROUP_SPIN`): C459H204+ (k05+, 500 K) 179.6 -> 135.5 s (151.0 s), Si185H124+ 224.2 -> 206.6 s (212.4 s), the neutral vacancy of a 3^3 cell (215 atoms) 21.7 -> 16.4 s (15.1 s: its weighted loop is 1.2 s slower than the abelian one, the small-cell stencils of `PARSEC_ORBITAL_GROUP_SPIN`); test 1: Si35H36+ 14.8 -> 12.8 s (12.7 s), Si87H76+ 38.9 -> 36.0 s (36.0 s), C87H76+ 29.2 -> 26.5 s (26.0 s), C185H124+ 125.1 -> 119.1 s (121.8 s); in each case the three energies agree to the printed digits; on 2 nodes (the full-SCF runner) k06+ and the vacancies of the 2^3 and 3^3 cells (a test-1 and a test-3 stop of a cell, 4.3 and 13.4 s) to the energy of one node to the printed digits, and under `off` the 2^3 vacancy and the Mn atom (their abelian sectors prepared again on every rank) too. The NOTE of the weighted sectors says so. A stop of unequal partners of `occup.in` and a multiplet the abelian sectors may hold keep the way through the abelian sectors. | `off`: always through the abelian sectors, as before |
| `PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS` | `auto` (since 2026-11-22; read and checked in every preparation, `Symmetry/weighted_inputs.py`): the shortcut of `PARSEC_WEIGHTED_GUARD_SHORTCUT` also for a cell whose abelian route takes its commuting involutions with their translations (a pristine supercell, a defect off the origin: `periodic_abelian_reduction`). The flags of the weighted sectors are formed against that very group where a guard report asks for them (`PointGroupReduction.sector_abelian_split_source`, `axis_reflection.CellAbelianSplit`: the operations of the cell in the order of `_space_group_operations`, which `CellGroup` repeats, so that `largest_commuting_involutions` of their table finds the group the abelian route builds, or the group about the origin where that one is no larger; 1-11 ms for diamond 1^3, 2^3 and 3^3 and a vacancy off the origin, whose elements are those of the route's group in every case, `tests/test_weighted_mpi_route.py`). A stop of either test is a prediction for such a cell: the abelian sectors solve the start Hamiltonian again and may pass test 1 (the neutral vacancy on a B site of a diamond 3^3 cell, and of a 2^3 cell: their abelian sectors passed test 1 and stopped at test 3, the same energies, -2451.19991682 and -717.57912667 Ry, in 26 against 16 and 29 against 18 iterations in all); where they stop at test 1 too, the result is theirs bit for bit (both full grids start from the atomic densities). An abelian route that passed both tests would keep its sector result under `off`, where the shortcut gives the full-grid reference. The palette ladder of the abelian preparation can step below that group (stencils beyond the two-byte codes), where the prediction names a group the run would not take. One A100, first runs, `PARSEC_ORBITAL_GROUP_SPIN` cells with a starting polarization of 0.3, the run's total wall `off` -> `auto` (the full grid without symmetry, `PARSEC_PERIODIC_SYMMETRY=off`): test 1, fcc Al 3^3 (108 atoms) 38.6 -> 31.0 s (27.4 s), Al 2^3 (32 atoms) 26.4 -> 19.2 s (14.3 s), Si 2^3 (64 atoms) 27.5 -> 21.7 s (19.0 s); test 3, fcc Fe (4 atoms, the conventional cube) 29.6 -> 18.2 s, 88 -> 64 iterations in all (13.9 s, 47); the three energies of each case agree to the printed digits (-447.96892088, -131.50091212, -509.9733457, -157.86520855 Ry); the full-SCF runner on 2 nodes (one rank per node, `PARSEC_CUPY_DISTRIBUTED_FILTER=0`): Al 2^3 11.2 -> 7.4 s of the run, -131.50091212 Ry in both, the energy of one process. | `off`: such a cell goes through its abelian sectors, as before |
| `PARSEC_ORBITAL_GROUP_SPIN_FLOOR` | `auto` (since 2026-11-21): no floor (`SPIN_FLOOR_ENTRIES` = 0): the weighted spin sectors are taken at every size at which the run takes sectors (an integer `PARSEC_PERIODIC_SPIN_SECTOR_FLOOR` keeps smaller cells on the full grid). Measured on one node of 4 A100 against `PARSEC_ORBITAL_GROUP_SPIN=off` (fresh processes, two runs each for the small cases; wall seconds of the process, `weighted / abelian`; basis = grid points x states of a channel): N atom (PBE NCPP, 3.7e6) 9.4 / 9.7; Cr2 (1.3e7) 9.0 / 10.5; O2 (`examples/0d_hybrid_functionals/O2/pbe`, 1.6e7) 15.7 / 16.3; V- in diamond 2^3 (6.3e6) 6.2 / 7.4, 3^3 (6.9e7) 10.3 / 9.4, 4^3 (3.9e8) 13.9 / 17.7; nanodiamonds k02 (5.2e7) 9.2 / 9.3, k03 (2.0e8) 10.7 / 10.8, k04 (6.7e8) 17.6 / 17.5, k05 (2.4e9) 43.4 / 71.2; their Si analogues (Si for C, bonds scaled; 0.23 A) Si35H36 (6.3e7) 8.9 / 9.6, Si87H76 (2.5e8) 10.7 / 12.7, Si185H124 (9.7e8) 15.4 / 20.9; with empty kernel caches (first runs of a new installation) N 22.3 / 22.1, Cr2 21.8 / 22.8, O2 28.4 / 28.4, V- 2^3 16.4 / 17.4, 3^3 19.8 / 30.5, k02 21.7 / 22.8, Si35H36 21.2 / 21.8. Energies within the SCF criterion (largest 1.5e-5 Ry at 2e-4 Ry), the GPU peak lower in every case (k05 5.5 -> 3.5 GiB). The one slower case, V- 3^3 with warm caches (SCF 3.8 -> 4.8 s), applies its weighted sector stencils 2.6 times slower per vector than its abelian ones, which no size rule tells from its neighbours (2^3 and 4^3 are faster). | an integer: a spin run whose grid points times states of a channel are fewer keeps the abelian group, with a `NOTE` at every output level (on every rank alike); `off`: no floor |
| `PARSEC_SECTOR_FIELD_GROUP` | `full` (since 2026-11-19): the density, the potentials, the mixer, the residual norms, the energy integrals and, for a cluster, the Hartree CG and its boundary keep one value per orbit of the full point group `G` of weighted sectors instead of the solved subgroup `H` (`Symmetry/field_orbits.py`; Td: about 3 times fewer values, diamond supercells with 192 operations 4.4 to 5.4); the sector eigensolver and the MPI ranks receive the potential per orbit of `H`; `scf_scalar_field_orbits` in the report (see "Weighted sectors of a larger subgroup"). Measured on A100, first runs: k14 (39,368 electrons) on 16 A100 in the MPI runner, SCF 44.97 -> 40.65 s (Hartree 4.91 -> 2.10, mixing and energies 1.00 -> 0.33, density 0.83 -> 0.61 s; fullest GPU 18.89 -> 18.86 GB), energy within 6e-7 Ry (1e-9 Ry for k10 on 8 A100, 12.92 -> 11.35 s); one node with the launch settings k10 26.37 -> 25.09 s, k12 63.89 -> 60.30 s; diamond 6^3 and 10^3 at Gamma (6^3: 7,308 values instead of 39,366) the same energy bits, 10^3 on one node 9.54 -> 9.33 s | `subgroup`: one value per orbit of `H`, as before. The former bits need `PARSEC_SECTOR_INPUT_GROUP_AVERAGE=off` as well (its `auto` averages the full-grid inputs over `G` by round-off), and where they apply `PARSEC_IONIC_BACKEND=native`, `PARSEC_IONIC_SUM_GROUP=first`, `PARSEC_IONIC_MOVED_SUMS=full` and `PARSEC_HARTREE_BOUNDARY_ROWS=diagonal` |
| `PARSEC_SECTOR_INPUT_GROUP_AVERAGE` | `auto` (since 2026-11-19): with `PARSEC_SECTOR_FIELD_GROUP=subgroup`, the full-grid inputs of weighted sectors (local ionic potential, SAD/ML/`Initial_Density`/`Restart_Run` start density, core density, full-grid evaluators) are averaged over the cosets of `G` after the orbits of `H` (with `full` the orbits of `G` average them already). A structure Td within a loose `PARSEC_SYMMETRY_ATOM_TOLERANCE` then has an energy error of second order in its distance from the group: GeH4 with its hydrogens 1.7e-2 / 8.4e-3 bohr off, 6.4e-4 / 1.6e-4 Ry against the exact group, 9.0e-3 / 4.4e-3 Ry before (A100); exact structures change by round-off | `off`: the orbit average of `H` alone, as before |
| `PARSEC_ORBITAL_IRREPS` | `off` (2026-11-19): the weighted sectors of `PARSEC_ORBITAL_GROUP` find a level of a representation of the full group in every solved sector its restriction holds (the E levels of Td in A1 and B1 of D2d), as before. Kept the default after the first runs of 2026-11-22 (`once` on the partner rows, see "Units of the irreducible representations"): faster only where one GPU holds the sectors of a large cluster (nanodiamonds of 1,095 to 4,273 atoms and Si795H300, 9 to 23 %), slower on the small clusters of one GPU (k02, k03, k05: 13 to 19 %), neutral or slower on several GPUs or ranks (k10 on two nodes +4 %, k08 on eight ranks of one GPU +9 %), and `split` slower on the cubes of one node | `once`: every irreducible representation solved in one sector, a sector that keeps only some of its representations split into units of one weight each and projected onto them after every Chebyshev filter (Td: five units, 1.2x fewer rows x states, 1.5x less Rayleigh-Ritz work, the same largest unit; see "Units of the irreducible representations"). First runs on A100, eigensolves of the SCF, every unit on the rows of its sector (`PARSEC_ORBITAL_IRREP_ROWS=sector`; the partner rows of that row, its default since 2026-11-22, change these figures): nanodiamond k10 on one GPU 58.8 -> 49.3 s, on four GPUs of one rank 16.4 -> 16.3 / 16.5 s (two runs: neutral), on two ranks of four GPUs 9.4 -> 9.8 s (+4 %, two runs each), k14 on 16 GPUs 36.1 -> 36.1 s, k18 on 16 GPUs 207.1 -> 207.1 s (70.0 GiB at the fullest GPU both); k08 on one GPU 13.3 -> 12.7 s at a peak 4.7 % higher (12.1 -> 12.7 GiB, the A1 unit grew once); slower on eight ranks of one GPU each (k08: 3.5 -> 4.5 s, five units on eight ranks balance worse than four sectors). `split`: also one unit per representation of one weight (the 192 operations of an even diamond cube: 20 units instead of 10 sectors, 1.67x less Rayleigh-Ritz work; 64,000 atoms on 16 A100: SCF 208.7 -> 170.0 / 175.9 s in two runs, peak +1.4 %, 46.0 -> 46.6 GiB at the fullest GPU), but slower than `off` on the cubes of one node: whole first runs on the four GPUs of one process, `off` -> `split` (sector rows / partner rows), diamond of 1,728 atoms 28.2 -> 30.1 / 30.4 s, of 4,096 atoms 24.1 -> 37.8 / 34.7 s, of 8,000 atoms 32.7 -> 55.8 / 47.0 s, Si of 512 atoms 19.8 -> 22.3 / 21.8 s, MgO of 1,000 atoms 32.4 -> 40.1 / 38.5 s (more SCF steps where a unit grows mid-SCF, 19 -> 24 / 22). States that rounding takes off the range of a unit's projector (a high first-filter degree on a coarse grid) are projected back; beyond 1e-3 of a combination of them the run stops. CHEBDAV first solves (the input default) project every block they filter and append (since 2026-11-22; refused before); ARPACK and DIAGLA first solves are refused before the SCF. With the switches of 2026-11-19 (merge rules): the first count of a unit is its share of the states even under `PARSEC_WEIGHTED_FIRST_SHARE=orbitals`, which estimates the sectors without units; one process with several GPUs keeps the placement by size of `PARSEC_CUPY_SECTOR_PLACEMENT=size` (its costs count the states of the units), and places the units by cost under `cyclic`; `occup.in` (`PARSEC_ORBITAL_GROUP_OCCUPATIONS`) checks the partners of a unit that holds a level in several states (m(lambda, Gamma) > 1), and none of a representation solved in one state per level |
| `PARSEC_ORBITAL_IRREP_ROWS` | `partner` (default since 2026-11-22; read only with `PARSEC_ORBITAL_IRREPS` `once` or `split`): a projected unit whose states are a whole one-dimensional sector of a larger subgroup `K` of the full group (its projector `(1/[K:H]) sum_c mu(c) T_c` over the cosets of `K`, `Symmetry/orbital_groups.subgroup_sector_cosets`: A1 and A2 of Td are the sectors of Td itself, the 3-dimensional representations of the 192 operations of an even diamond, MgO or NaCl cube sectors of subgroups of 64) holds one partner row of each of its representations and is solved on the orbits of `K`: `[K:H]` times fewer rows (Td: 3, the cubes: 2 and 6), with a stencil and projectors of its own (prepared beside the operators of the sectors of `H`, one worker per unit) and no projector after the filters, the other partners from the average of its density over the full group; units that are no such sector (the E of Td) keep the rows of their sector (see "Partner rows"). A run with LDA+U keeps every unit on the rows of its sector under both values (its occupation matrices read the sector orbitals there; the driver decides it from the input before the units are placed, alike on every MPI rank, and parsec.out says so in a NOTE; until 2026-11-22 the partner default stopped such a run in its first occupation matrices). The partner rows come from the coset images since 2026-11-22 (no projector tables, sparse matrix or graph components): nanodiamond k10, two units of 1.3 million rows, 0.6-0.9 s instead of 1.95 s; k14, two units of 2.8 million rows, 1.55 s on two workers instead of 3.90 s. First runs on A100 (no symmetry cache, CuPy kernel cache warm unless cold is said), whole runs, `once` (clusters) or `split` (cubes), `sector` -> `partner`: nanodiamond k08 on one GPU 18.0 -> 15.9 s (three and two runs; peak 12.9 -> 10.5 GiB), k10 on one GPU 57.9 -> 51.1 s (40.3 -> 35.8 GiB), k10 on the four GPUs of one process 25.2 -> 24.3 s (three and two runs), k10 on two nodes of four GPUs 16.5 -> 16.2 s, k14 on 16 GPUs 51.0 -> 50.7 s, k08 on eight ranks of one GPU 8.0 -> 7.2 s, k06 with CHEBDAV on one GPU 16.2 -> 15.9 s (three runs each), Si795H300 (Td) on one GPU 15.7 -> 13.8 s and with CHEBDAV 23.0 -> 20.9 s; cubes on the four GPUs of one process (cold / warm cache): diamond of 1,728 atoms 19.5 -> 19.1 / 8.4 -> 7.2 s, of 4,096 atoms 39.5 -> 36.8 / 17.6 -> 16.2 s, Si of 512 atoms 18.1 -> 17.2 / 10.3 -> 9.8 s, MgO of 1,000 atoms 24.5 -> 24.4 / 14.1 -> 13.9 s, with CHEBDAV Si of 512 atoms 22.9 -> 21.8 / 11.0 -> 10.6 s and diamond of 1,728 atoms 23.1 -> 22.3 / 11.2 -> 10.8 s. The same energies within the SCF criterion (k06 at 1e-7 Ry: `sector` and `partner` within 1e-9 Ry on four ranks; k14 at 2e-4 Ry within 2.1e-5 Ry); the decomposition of `K` is that of the partner rows of before bit for bit, and the coefficients `e_s` are exact where those of before carried the rounding of the projector's tables (up to 3.5e-13; they enter only the export of the orbitals). A unit keeps the Ritz route of its sector of `H` (`rayleigh_ritz.ritz_route_rows`) | `sector`: every unit on the rows of its sector of `H`, a projected one with its projector after every filter, as before 2026-11-22 (bit for bit; until 2026-11-21 the default, when the partner rows of the projector's tables cost 1 to 2 s per unit and made k14 on 16 GPUs and k10 on two nodes slower) |
| `PARSEC_PERIODIC_SYMMETRY` | `auto`: a bulk orthorhombic cell at the Gamma point that the route of the sectors takes solves its orbitals in the sectors of its point group about the origin of the cell (with its translations, `PARSEC_PERIODIC_TRANSLATIONS`, and since 2026-11-18 also about a symmetric site off the origin, `PARSEC_PERIODIC_SYMMETRY_CENTRE`; atoms modulo the lattice, row maps that wrap; Td of diamond: 1/16 of the basis of the one sector of the identity; see "Point groups of periodic cells") | `off`: one sector of the identity, or the full grid of one GPU, bit for bit |
| `PARSEC_PERIODIC_SECTOR_FLOOR` | `auto` (default, since 2026-11-19): an unpolarized bulk cell at the Gamma point in one process whose basis (grid points x `States_Num`, or half the electrons plus 6 without it) stays below 2^24 entries, the floor of the k-point sectors and of the Gamma hybrids, keeps the full grid of one device instead of the sectors of its point group (`driver.periodic_sector_floor_setting`, `_periodic_basis_below_floor`; the sectors cost a fixed time per sector and SCF step, about 10 ms on A100 (`PARSEC_SECTOR_OVERHEAD`), that a small basis does not repay; the grid points are counted without building the grid, `gamma_hybrid_route.periodic_grid_points`); `--symmetry on`, collinear spin (whose own floor, `PARSEC_PERIODIC_SPIN_SECTOR_FLOOR`, is none by default) and the MPI sectors of the full-SCF runner keep the sectors (its `--serial-control`, one process, takes the floor as `main.py` does), and the log names the floor. Since 2026-11-19 the single-point forces and the `bfgs`/`simple` relaxations that `PARSEC_PERIODIC_SECTOR_FORCES=auto` keeps in the sectors keep them below the floor too (`prepare_single_point(sector_forces=True)` from the command line and the runner): the full grid would not average their forces and moved positions over the group, and the atoms the group fixes would drift by round-off (9.5e-5 bohr after one `simple` step of Si8); the floor applies to an SCF alone (and to bands after it). Total wall time in `parsec.out` on A100, sectors -> full grid, first run (fresh CuPy cache) / second run, one GPU; four GPUs: 3d_Si (8 atoms) 12.9 / 1.53 -> 12.8 / 0.79 s; 27.0 / 1.74 -> 13.6 / 0.89 s. Diamond 64 atoms (6.3 M entries) 11.3 / 4.24 -> 11.7 / 1.11 s; 15.0 / 4.75 -> 10.9 / 1.48 s. Above the floor the crossover is not sharp: rocksalt MgO 64 atoms (33.4 M entries, 14 vs 20 SCF steps) 24.1 / 15.5 -> 17.4 / 6.3 s; 37.0 / 15.8 -> 17.5 / 6.3 s (the full grid faster in all four); Si64 (44 M, 11 vs 19 steps) 14.7 / 6.2 -> 19.7 / 8.2 s; 29.0 / 6.5 -> 21.0 / 8.1 s; diamond 216 atoms (69 M, 7 vs 10 steps) 10.3 / 4.2 -> 13.1 / 3.6 s; 15.0 / 5.7 -> 14.5 / 5.0 s; the sectors win clearly from a few 10^8 entries: MgO 216 atoms (369 M) 21.5 / 13.2 -> 129.9 / 118.2 s; 35.0 / 14.1 -> 129.8 / 118.2 s, diamond 512 atoms (384 M) 14.6 / 8.2 -> 25.5 / 16.2 s; 19.5 / 10.3 -> 27.0 / 16.3 s (peak 0.6-0.9 GiB per GPU in the sectors against 15.6-25.4 GiB on the one device of the full grid). 2^24 is kept as a conservative value shared with the floors of the k-point sectors and of the Gamma hybrids: no cell measured below it is faster in the sectors, while cells of up to about 7 x 10^7 entries are often as fast or faster on the full grid, on four GPUs and in second runs most of all; an integer floor of 2^25-2^26 suits such cells. Re-measured on 2026-11-21 after the orbital-group plan became array operations (it took 3-5 s of the setup of the cells with 384 operations, before 6.1-8.4 s and now 1.3-3.5 s of their pre-SCF time in second runs; see "Weighted sectors of a larger subgroup"), sectors -> full grid, one A100, second run / first run: fcc Al 32 atoms (3.2 M entries) 3.68 / 15.2 -> 1.71 / 12.7 s; diamond 64 (6.3 M) 1.75 / 8.6 -> 1.04 / 10.8 s; Si 2x2x2 (12 M) 2.63 / 13.4 -> 2.93 / 17.5 s; MgO 64 (20 M) 5.00 / 16.0 -> 5.87 / 19.6 s; NaCl 64 (20 M) 4.88 / 15.7 -> 2.89 / 16.6 s; SrTiO3 40 (21 M) 6.26 / 17.2 -> 6.47 / 17.4 s; MgO 64 on the Fortran-derived routines (37 M) 6.75 / 15.8 -> 6.36 / 17.6 s (before the change 10.6 / 21.6, 10.6 / 21.3, 12.2 / 23.0 and 12.4 / 21.3 s in the sectors of MgO, NaCl, SrTiO3 and the Fortran MgO). The crossover lies near 10^7 entries and depends on the cell (the full grid ahead in second runs below it, the sectors in most first runs from 6 M), so the floor stays at 2^24. Energies agree within the SCF criterion. The tests of `acceleration/tests` run with it `off` unless a test sets it (`test_periodic_sector_floor` runs a relaxation step under `auto`). An integer sets another floor; a wrong value is refused at setup | `off`: every cell with a group takes the sectors, as before |
| `PARSEC_PERIODIC_SPIN_SECTOR_FLOOR` | `auto` (default, since 2026-11-21): no floor (`driver.PERIODIC_SPIN_FLOOR_ENTRIES` = 0), every collinear spin-polarized cell takes the sectors as before. A floor is not the default because the full grid of a small magnetic cell can need more SCF steps than `Max_Iter` allows where its sectors converge, and no size separates those cells. A100, one GPU, 21 cells (bcc Fe, Co, Ni, Cr, Mn, V; fcc Fe, Co, Ni, Mn, Cu, Al, Li; O2, N, V- and V0 in the 7-atom diamond cell; MgO8, NaCl8, Si8, TiO2) x criteria 1e-5, 1e-6, 1e-7 Ry x `Initial_Spin_Polarization` 0.3 and default, default `Max_Iter` (50), a floor of 2^21 against none: 7 of the 126 pairs converge without the floor and stop at `Max_Iter` with it (bcc Co2 at 1e-6 and 1e-7 Ry at both polarizations, Cr2 at 1e-7 at the default one, Mn2 at 1e-7 at both; Co2 at 1e-6, default polarization: 17 steps in the sectors), none the other way, and the 80 pairs where both converge agree within 1e-8 Ry. Steps at 1e-6 Ry, default polarization, sectors -> full grid: Cr2 11 -> 30, Mn2 10 -> 31, O2 19 -> 26, MgO8 11 -> 15. The first run (empty kernel cache) at 1e-6 Ry was shorter with the floor on all 42 inputs (Al4 17.1 -> 12.1 s, MgO8 14.0 -> 12.5 s, fcc Fe4 at polarization 0.3, whose guard fires, 29.6 -> 13.8 s), also where the floor then missed convergence, since such a run stops at `Max_Iter` | an integer: a single point of a bulk cell at the Gamma point with a native LDA or GGA, in one process, whose basis (grid points x states of a channel) is smaller keeps the full grid of one device (`_cell_below_sector_floor`; 2097152 is the size up to which the full grid was faster in first and second runs of the cells measured, on one GPU and on four); raise `Max_Iter` with it. `--symmetry on`, the MPI sectors of the full-SCF runner, the forces and relaxations that `PARSEC_SPIN_SECTOR_FORCES` keeps in the sectors, TS, LDA+U, a meta-GGA or libxc, and an input whose species start with polarizations of opposite sign keep the sectors under any floor. The last one is a function of the input, so that a series in the grid spacing stays on one route: bcc Mn2 (a 5.40 bohr, PBE, labels +0.5 and -0.5, 300 K) has -56.87755621 Ry in the sectors at h 0.11 and -56.87927476 at h 0.10, but -56.95948510 and -56.98222016 on the full grid, whose moment changes with h. A cell that a floor keeps on the full grid and that ends with a partly filled pair of degenerate levels gets a NOTE (`driver._say_floor_doublet`), which changes no number. `off`: no floor |
| `PARSEC_PERIODIC_HARTREE_METHOD` | `fft` (since 2026-11-16): the finite-difference Poisson equation of an orthorhombic 3-D cell on the device by dividing by the FFT of its stencil symbol (the wrapped `-nabla^2` is circulant): the exact solution of the linear system the CG iterates on, two real FFTs per solve and no matrix on the device (bulk diamond, 23.9 M points: about 9 s per SCF step before) | `cg`: the conjugate gradients of before, bit for bit (the two differ by the CG tolerance) |
| `PARSEC_PERIODIC_TRANSLATIONS` | `auto` (since 2026-11-16): a periodic cell that repeats a smaller one twice along some axes adds the translations by half of those axes to the group of its sectors (the time-reversal invariant k-points of the smaller cell are their real characters), and a cell whose only pure translations are those also the operations whose translation is a fraction of the cell on whole grid steps (non-symmorphic: Fd-3m of the diamond cube where a/4 is a whole number of steps); see "Translations by half the cell" | `half`: the half-cell translations alone; `off`: the point group about the origin alone, bit for bit |
| `PARSEC_PERIODIC_DENSITY_BINS` | `auto` (since 2026-11-16): the superposed atomic density of a bulk cell on the device from bins of the atom images (`V_ion/periodic_device.py`, `_binned_bulk_density`: each point the images of its 27 neighbouring bins, the same terms in another order, the host's sum to round-off) | `off`: every atom per point, the host's sum bit for bit |
| `PARSEC_CONFINED_SETUP` | `auto` (since 2026-11-22): the preparation of a wire or slab without scans of the whole grid (`V_ion/confined_setup.py`): the balls of the projectors from index boxes, the superposed valence and core densities summed on the device in the order of the readable loop, the image terms of the short-ranged local potential on the threads of the periodic setup. The same fields bit for bit (equal hashes, SCF energies and steps; the forces of the slab and wire cases of the consistency matrix to all digits). A100 node, first runs: 72-atom graphene 283.0 -> 13.1 s, 72-atom h-BN 535.8 -> 16.9 s, 38-atom ribbon 90.2 -> 9.4 s (section "The preparation of a wire or slab" below) | `readable`: the dense scans and the readable loop (the same numbers, slower) |
| `PARSEC_SPLINE_SETUP` | `auto` (since 2026-11-22): the `Cubic_Spline` core densities of a periodic preparation (every periodic grid, also the slab grid of the default routines since the cubic spline became the default of a species without the label; `V_ion/spline_setup.py`) with the spline of each species built once (the readable loop rebuilt the spline of the whole table for every image, about 1 ms each, and a wire or slab sums its images out to the end of the table), the terms of the images in blocks on the threads of the periodic setup, one call of the readable interpolation per block, added in the readable order: the density bit for bit (equal hashes of the fields, SCF energies and steps). A100 node, first runs, core-density setup: Ge8 slab 22.3 -> 1.0 s, Ge2 slab 10.1 -> 0.34 s, Ge64 cell 2.27 -> 0.88 s (default routines) and 0.67 s (Fortran-derived), Ge8 cell 0.65 -> 0.09 s, MgO8 (Mg core) 0.34 -> 0.09 s. A slab still sums the spline of every core-corrected atom over all its images out to the end of the table on every point, so the setup grows with the atoms: Ge32 slab 4.59 s against 0.014 s for its linear tables (its run with forces 44.9 -> 50.8 s); the spline (`x**3` is the library's `pow`) is not taken to the device nor cut short, which would change the bits | `readable`: the readable loop (the same numbers) |
| `PARSEC_PERIODIC_OBLIQUE_SPLIT` | `validated` (since 2026-11-22): the local potential of the general grid of the Fortran-derived routines (oblique cells, unequal steps) from the split of the orthorhombic grid for `Cubic_Spline` species (as before) and for linear POTRE and `fhipp` tables; a cell with a linear MARTINS_OLD or MARTINS_WANG table keeps the readable sum (their table noise past the Coulomb tail varies over the cell: up to 1.25e-6 Ry per atom in the energy; section below). Preparation of 64- to 216-atom cells on an A100 node 257-810 s -> 2.0-2.6 s. POTRE tables: the readable potential to round-off; `fhipp` tables: a constant 1e-11 to 8e-10 Ry above the readable sum (the readable sum adds the file's digits past the Coulomb tail out to a reach that grows with the cell; the orthorhombic grid leaves them out since 2026-11-07); the total energy moves by that constant times the electrons (at most 3.4e-9 Ry per atom measured), forces unchanged to all printed digits | `auto`: the default 2026-11-07 to 2026-11-22, the split for `Cubic_Spline` species and the readable sum for linear ones (bit for bit); `fast`: the split for every species; `readable`: the readable sum for every species |
| `PARSEC_SECTOR_PROJECTOR_ORBITS` | `auto` (since 2026-11-16; abelian sectors since 2026-11-18): the sectors of a point-group reduction, and since 2026-11-18 the abelian sectors of clusters and Gamma cells (`attach_abelian_atom_orbits`; with LDA+U since 2026-11-21, `PARSEC_LDAU_PROJECTOR_ORBITS`), take the projectors of one atom per orbit of their group, the columns scaled by the square root of the orbit size (the same sector operators to round-off; `Symmetry/axis_reflection.py`, `compress_projectors_for_sectors`); the forces of abelian sectors are completed over their group. D2 nanodiamond of 487 atoms on one A100: 1,842 -> 480 projector columns, SCF -9 %; a C2v diamond 5^3 cell -14 % | `weighted`: only the point-group sectors, as before 2026-11-18; `off`: the projectors of every atom, bit for bit |
| `PARSEC_SECTOR_ORBITS_EXACT` | `auto` (default, since 2026-11-21): the sectors take the projectors of one atom per orbit (`PARSEC_SECTOR_PROJECTOR_ORBITS`) only for a structure that keeps their operations within 1e-6 bohr, the distance the forces, the moves and the LDA+U columns already use (`Symmetry/axis_reflection.py`, `sector_orbits_exact`; clusters, Gamma cells and the k-point sectors, `SCF/kpoint_sectors.py`). A structure accepted within a looser `PARSEC_SYMMETRY_ATOM_TOLERANCE` keeps the projectors of every atom and says so in `symmetry_detection`, since the columns of a representative would give its whole orbit its offset. At a tolerance of 1e-6 bohr or less (the default is 1e-7) nothing is checked and nothing changes. Measured on A100, tolerance 1e-3, one atom 5e-4 bohr off, the distance of the energy from the full grid of that structure, before -> now: Ag108 1.24e-3 -> 7.6e-5 Ry, Ag13 cluster 6.2e-4 -> 7.1e-5, Ag108 with a 2x2x2 mesh 3.5e-4 -> 7.0e-5, C216 3.7e-5 -> 5.5e-6, Si216 2.5e-5 -> 1.9e-6, C64 1.1e-5 -> 1.9e-6, Si64 3.2e-7 -> 2.1e-7 (the cells of 64 and 216 atoms with `PARSEC_PERIODIC_SECTOR_FLOOR=off`); the exact structures at the same tolerance keep the orbits and print the same energies. The columns of every atom cost memory and time that grow with the cell, only in the runs this rule changes (first runs, `off` -> `auto`, the fullest of 4 A100): Ag108 523 -> 575 MiB, C216 577 -> 599, Si216 649 -> 741, C512 741 -> 855, C1000 1971 -> 2083, C1728 2457 -> 2965 (SCF 5.3 -> 9.0 s), C4096 2193 -> 3177 MiB (+45 %, SCF 10.8 -> 17.9 s), the host high-water within 5 %; up to 512 atoms the SCF time is unchanged within its noise (Ag108 4.62 -> 4.69 s, C216 2.99 -> 2.96 s, Ag108 2x2x2 19.9 -> 18.6 s). A cell close to the memory of its devices under `off` can therefore run out under `auto`; `off` then takes the cheaper orbits with their larger error. The check compares the image distances with 1e-6 bohr itself (`atom_image_table(within_tolerance=True)`, also for the LDA+U columns). Weighted sectors keep a first-order remainder after this rule, because the weights assume the group: the energy of Ag13 with an atom `d` bohr off is about `0.0125 d + 260 d^2` Ry from its full grid against `224 d^2` for the abelian sectors (7.1e-5 against 5.6e-5 at 5e-4 bohr); they are not taken here since they are 1.5 to 2.9 times slower (Ag13 2.0 -> 2.9 s, Ag108 4.7 -> 13.7 s, there the same 7.6e-5) | `off`: the orbits at the tolerance of the detection, with the search of up to twice the distance, bit for bit as before |
| `PARSEC_KPOINT_KINETIC` | `gradient` (default): `-nabla^2 - 2ik.nabla + k^2` with the gradient discretized on its own, as before | `bloch` (since 2026-11-16): the Bloch block of the finite-difference Laplacian (`Laplacian/finite_difference.py`, `build_bloch_negative_laplacian`; a scalar Hamiltonian without a meta-GGA on a bulk orthorhombic cell whose grid rows are its points, with the default routines or the Fortran-derived ones such as `Si64_k222/pbe`), the discretization of the Gamma point of the supercell; `auto`: `bloch` where it applies; `trim` (since 2026-11-21): `bloch` at the time-reversal invariant points other than Gamma (`k = G/2`) where it applies, `gradient` at the others, the discretization of the real TRIM route (`PARSEC_KPOINT_TRIM_REAL`; off by default by the accuracy rule: its energies differ from the default's by more than the SCF criterion and are not closer to the converged ones at every spacing, see "Real arithmetic at Gamma and at the time-reversal invariant points"); see "The kinetic term of a k-point" |
| `PARSEC_KPOINT_SUPERCELL` | `auto` (default since 2026-11-19): a single-point SCF on the GPU of a Gamma-centred mesh of one or two points per axis (`Monkhorst_Pack_Shift` zero, or no shift block: the time-reversal invariant k-points) of a bulk orthorhombic cell of the default routines, scalar and unpolarized, with a native LDA or GGA, runs as the Gamma point of its supercell (`SCF/kpoint_supercell.py`, `trim_supercell_route`; the Bloch kinetic term, real orbitals in the sectors of the supercell, `Net_Charges` and the states `M` times those of the cell) where the whole mesh holds at least `PARSEC_KPOINT_SUPERCELL_MINIMUM` entries, the process computes on at most `PARSEC_KPOINT_SUPERCELL_GPUS` GPUs (one) and twice the estimated difference of the two kinetic forms per atom is within `PARSEC_KPOINT_SUPERCELL_TOLERANCE`; forces, movements, band structures, `nscf`, DOS, `Restart_Run`, polarizability, `Energy_Refinement`, an MLFF start, cube or potential files, saved orbitals (`--save-wavefunctions`, `Output_All_States`; a `Save_Wfn_Bands` block selects the Fortran-derived routines, which keep the k-points), point charges, fields, a run on the CPU and the k-point ranks of `--mpi` keep the k-points (the full-SCF runner takes the supercell), and a NOTE says which route and why. `Relax_Restart` and `--restart-from` replace the atoms of the cell before the decision. The energy lines of the supercell's report (`Total Energy =` and the parts) are those of the input cell, as the k-points print them; the report closes with the energy of the input cell, per atom and the supercell's total; the `parsec.dat` and the result archive are the cell's (no orbitals: the supercell's states are no Bloch states of the cell, while the k-points write theirs since 2026-11-22, `PARSEC_KPOINT_WAVEFUNCTIONS`; the k-points, weights, levels and occupations of the k-point route since 2026-11-21: `PARSEC_KPOINT_SUPERCELL_LEVELS`), which `Restart_Run`, `nscf_kpoints` and `--bands-from` read on either route; the levels of the report and `eigen.dat` are the supercell's. One A100: silicon cube at 48 points per edge 39.1 -> 19.4 s, MgO at 56 55.3 -> 27.3 s and at 72 131.2 -> 40.4 s, 2.89 -> 0.98 GiB (see "The kinetic term of a k-point") | `on`: wherever the supercell can be solved, for any run and any number of GPUs, without the estimate and the floor (the `auto` of 2026-11-16 to 2026-11-18: the report, `parsec.dat` and the archive of the supercell, the restarts applied to the supercell); `off`: the k-points with the kinetic form of `PARSEC_KPOINT_KINETIC`, as before |
| `PARSEC_KPOINT_SUPERCELL_LEVELS` | `auto` (default, since 2026-11-21): the `parsec.dat` and the result archive of the cell after a mesh solved as its supercell (`PARSEC_KPOINT_SUPERCELL=auto`) list every level of the supercell at the k-point of its Bloch phases, the eigenvalues `+-1` of its translations by one cell (`SCF/supercell_levels.py`): from the characters of the sectors where their group holds the translations (the partners of a weighted state over the star of its k-point under the full group), else from the orbitals on the full grid, clusters of degenerate levels by the eigenvalues of `sum_a 2^a T_a` in their span (weighted sectors without the translations: the star of every solved state from its weights on the k-points, its partners over that star); the k-points and weights are those of the k-point route (its irreducible wedge), each with its lowest levels, as many as the k-point of fewest holds (a NOTE when that is fewer than the states of a k-point, which a `Fix_Number_Eigenvalues` restart then needs: its error 404 names `States_Num` of that count, or `PARSEC_KPOINT_SUPERCELL=off` or `PARSEC_KPOINT_SUPERCELL_STATES` for the run that writes the file), and their occupations; the levels carry one representation (the k-points'), and the route's NOTE says the levels are by their Bloch phases. Where the orbitals do not tell (none kept; orbitals on other MPI ranks of a group without the translations), the levels are dealt in turn with a NOTE. One A100, 216 silicon atoms at 66 points per edge with the 2 x 2 x 2 mesh (weighted sectors with the translations): the four k-points and weights of the wedge as the k-point route, the 430 lowest levels of every k-point within 6.5e-8 Ry of those of the k-point route with the Bloch kinetic term (the same total energy, -1721.32818361 Ry), the run 25.24 -> 25.10 s (no measurable cost). | `off`: the levels dealt to the k-points of the whole mesh in turn, with equal weights (their union is the mesh's, the k-point of each not), their sector labels dealt with them, no `run_states_per_kpoint` in the archive and the NOTEs of before |
| `PARSEC_KPOINT_SUPERCELL_STATES` | `cell` (default, or `0`): the supercell of a mesh solves `M` times the states of a k-point of the cell, and each k-point of the cell's `parsec.dat` and archive keeps as many levels as the k-point of fewest holds (`PARSEC_KPOINT_SUPERCELL_LEVELS`), a few below `States_Num` where the Bloch phases share the supercell's levels unevenly (silicon cubes at 24 points per cell edge: Si8 22 of 24, Si64 132 of 136, Si216 436 of 440; Si8 at 22 points per edge 20 of 24), so that a `Fix_Number_Eigenvalues` restart from that file stops with PARSEC's error 404, whose message names this switch. A100, the default route of a 2x2x2 mesh (its floor, estimate bound and GPU count lifted), silicon cubes at 24 points per cell edge, one node: Si8 (`States_Num` 24) holds 22 levels a k-point, and 24 with `n` = 2, 4 or 6; Si64 (136) 132, still 132 with 2, 135 with 4 and 136 with 6, 8 or 12; Si216 (440) 436, and 440 with 8 or 16. With `States_Num` levels a k-point the `Fix_Number_Eigenvalues` restart (on the k-point route) runs, and the energy of the supercell run is the same to the printed 1e-8 Ry for every `n`. The extra states took no measurable time (Si8 4.19 against 4.07 to 4.14 s, Si64 9.93 against 9.58 to 10.82 s, Si216 24.1 against 27.5 and 17.1 s, runs of one job one after the other, the later ones with the kernels the earlier compiled), but the `n` that suffices depends on the band structure and the grid (Si8 needs 2 at 24 points per edge and holds 4 fewer at 22), so it stays an opt-in | a non-negative integer `n` (opt-in): `M (N + n)` states in the supercell, `N` the states of a k-point of the cell; the cell's files keep at most `N` levels a k-point, so every k-point holds `States_Num` where `n` covers its deficit; the `M n` more states are empty ones solved in every iteration |
| `PARSEC_KPOINT_SUPERCELL_TOLERANCE` | `7.35e-4` Ry per atom (0.01 eV, default, since 2026-11-19): the bound of `auto` on twice the estimated `E(k-points) - E(supercell)` per atom (`kinetic_form_difference_estimate`: the momentum density of the pseudo-orbitals of the atoms against the symbols of the gradient and Bloch stencils; the SCF differences measured 0.74 to 1.8 times the estimate on silicon, diamond, aluminium, MgO and a cell of unequal steps) | any number of at least 0 (Ry per atom) |
| `PARSEC_KPOINT_SUPERCELL_MINIMUM` | `16777216` (2^24 entries, default, since 2026-11-19): the floor of `auto` on the whole mesh, k-points times grid points times states of each (the floor of the k-point sectors); below it the k-points were as fast or faster than the supercell, whose preparation takes 5 to 10 s more (one A100, 8-atom cubes: silicon at 40 points per edge 21.2 against 21.3 s, MgO at 40 20.6 against 25.3 s) | any integer of at least 0 (0: every size) |
| `PARSEC_KPOINT_SUPERCELL_GPUS` | `1` (default, since 2026-11-19): the most GPUs of one command-line process (`--gpus`, `--gpu-ids`, `PARSEC_CUPY_DEVICES`) for which `auto` takes the supercell; with more the pools of the k-points divide the k-points over the GPUs, while the largest sectors of the supercell share one. Four A100 in one process, k-points against the supercell, peak GPU memory and time: 216 carbon atoms at a/18 1.37 against 4.27 GiB, 32.2 against 29.8 s; 216 silicon atoms at 66 points per edge 2.0 against 2.7 GiB, 35.8 against 36.4 s; 64 silicon atoms 2.0 against 0.98 GiB, 24.7 against 21.9 s (on one GPU the supercell held less in every case measured). The full-SCF runner, which has no k-point route, does not read it | any integer of at least 1 |
| `PARSEC_SUPERCELL_KPOINTS` | `auto`: a Gamma-point cell that repeats a smaller orthorhombic one is solved as that cell with the Gamma-centred mesh of the repeats (`SCF/kpoint_supercell.py`; the same energies). Order of the decisions (since the merge of 2026-11-20): the supercell of `PARSEC_KPOINT_SUPERCELL` is decided first, on an input with a k-point mesh; this route is asked only where no such supercell was taken, on a Gamma-point input (it takes no input with k-points), so neither undoes the other; `PARSEC_PERIODIC_SECTOR_FLOOR` then decides on the Gamma-point cell that remains, which a supercell of `PARSEC_KPOINT_SUPERCELL` always passes (M^2 x points x states of its cell, at least M x 2^24 entries) | `off` (default) |
| `PARSEC_SUPERCELL_KPOINTS_WEDGE` | `auto` (default, since 2026-11-19; read under `PARSEC_SUPERCELL_KPOINTS=auto`): the mesh of a single point is reduced to the irreducible wedge of the operations of the cell that map its grid and the mesh onto themselves, with time reversal, the density symmetrized with them (`SCF/supercell_mesh.py`; see "Every translation of a perfect supercell"; A100, 4 GPUs, the whole process of a first run with an empty kernel cache, the kernels compiled in brackets: diamond 5^3 as one cube with the 5^3 mesh 10 k-points instead of 63, 26.6 s instead of 38.2 s (7.0 s instead of 19.0 s), 6^3 20 instead of 112, 29.0 s instead of 50.3 s (9.9 s instead of 31.7 s), the same energies to 1.2e-9 Ry per atom; diamond 21^3 286 k-points instead of 4631) | `off`: time reversal alone, as before |
| `PARSEC_SUPERCELL_KPOINTS_MESH` | `all` (default): the mesh of all the repeats on the smallest repeating cell, as before | `K1xK2xK3` (since 2026-11-19): a mesh that divides the repeats, the cell of the remaining repeats; `auto`: the symmetric mesh of most points with an entry above two whose cell holds at least 2^22 grid points times states, else the mesh of all the repeats, else the smallest larger cell that takes the grid (a smallest cell too small for the stencil), else (repeats of ones and twos, or no cell that takes the grid) the Gamma point of the supercell with a note; where the cell of `all` cannot take the grid, the Gamma point of the supercell with a note (A100, 4 GPUs, the whole process of a first run with an empty kernel cache against `all`, the kernels compiled in brackets: diamond 6^3 with 3^3 37.5 s against 29.0 s (8.0 s against 9.9 s), 9^3 with 3^3 35.8 s against 32.9 s (16.1 s against 14.0 s), 10^3 with 5^3 40.7 s against 40.1 s (11.3 s against 19.1 s), 12^3 with 6^3 45.9 s against 50.1 s (16.6 s against 27.4 s), 7^3 the same mesh: a larger cell compiles the kernels of its grid; not the default for that reason). On 16 A100, first runs: a mesh of 7^3 takes diamond 21^3 in 29.6 s, 6^3 18^3 in 27.1 s, 5^3 15^3 in 28.4 s |
| `PARSEC_KPOINT_SECTORS` | `auto` (default, since 2026-11-16): the k-points of a scalar or collinear-spin periodic SCF on a bulk orthorhombic cell (also with forces, in one process or on the ranks of the MPI k-point pools, and with hybrid functionals, since 2026-11-17) are split into the sectors of the commuting involutions of their little groups (`SCF/kpoint_sectors.py`): each k-point becomes several eigenproblems of a fraction of its rows and states, the same spectrum; a k-point whose basis holds fewer than 2^24 entries (rows times states) or whose sectors would not halve its work stays whole; see "The sectors of a k-point" | `off`: one task per k-point, bit for bit as before |
| `PARSEC_KPOINT_SECTOR_MARGIN` | `1.3` (default): the states of a sector over its share `M n_s / N` of the states of its k-point (plus six); a sector that does not reach the highest level the k-point keeps gets 1.5 times its states and the step is solved again | a number of at least 1 |
| `PARSEC_KPOINT_SECTOR_STENCIL` | `auto` (default, since 2026-11-19): the kinetic term of a sector of a k-point is a stencil gathered from the grid (`Hamiltonian/sector_bloch_stencil.py`: per row and neighbour the sector column and the code of a real factor, 5 bytes against the 20 of a complex CSR entry, and one complex weight per neighbour that carries `-2ik.grad` or the Bloch phase), with the step of the Chebyshev recurrence in its kernel, and in the gradient form (the default kinetic term) no complex matrix of the whole grid is built per k-point; where the kinetic matrices are not an axial stencil of the grid or the factors of a sector need more than 256 values, the CSR of before. Only where the operators of the sectors go to the device code (a k-point backend on a GPU): the CPU route (`--device cpu`, no GPU, the MPI pools on CPU ranks) keeps the CSR, bit for bit, because the host applies it in one scipy product, 3-4 times faster there than a loop over the slots of a table (two SCF steps of `Si64_k222/pbe` at 64^3 points on the CPU of an A100 node: 105.4 s with `off` or `auto`, 484.1 s with the tables). Four A100, first runs, the final energies identical and every SCF step within 1e-8 Ry (the last printed digit of one step of `Si64_k222/pbe`; all digits elsewhere), the same steps and operator applications: diamond 6^3 cubes, 3 x 3 x 3 mesh, 50.3 -> 38.7 s (filter 25.1 -> 14.0 worker-seconds, host peak 5.2 -> 3.5 GB, fullest GPU 12,105 -> 11,835 MiB); with the first kernel (palette and weights from global memory, the same bits) 8^3 cubes, 2 x 2 x 2 mesh, 228.7 -> 178.5 s (host peak 10.8 -> 6.8 GB) and `Si64_k222/pbe` 29.3 -> 25.1 s; see "The sectors of a k-point" | `off`: the reduced complex CSR of the kinetic term, bit for bit as before |
| `PARSEC_KPOINT_REAL_GAMMA` | `auto` (default, since 2026-11-21): the Gamma point (`k = 0`) of a k-point SCF, whose Bloch Hamiltonian is real, is solved in float64 on the device (`SCF/kpoint_real.py`) where the basis of a k-point (grid points times states) reaches the floor of the k-point sectors, `2^24` (`real_gamma_applies`; below it the complex solve of before runs, bit for bit: there the first compile of the real kernels costs more than Gamma's share of the mesh saves); the device solver of one process and the shared columns of the MPI pools (`Eigensolvers/chebff_columns.py`, any number of ranks per pool, one round buffer per name for both dtypes) on real blocks, the real kernels of the sector stencil and of the whole-grid stencil; a scalar Hamiltonian, whole or in the sectors of its little group (their bases are real), without the exchange of a hybrid, LDA+U or a meta-GGA's orbital term, not on the row-distributed solver of the spin-orbit and non-collinear drivers, not with `PARSEC_KPOINT_FIRST_SOLVE=chebdav`; the band structures (also the band paths in sectors) and the CPU routes keep their complex solves. The orbitals handed to callers are cast to complex. The start vectors are real, so the numbers are those of the complex solve to the tolerance of the eigensolver, not bit for bit (the final energies of the runs below the same to every printed digit or within 2.3e-7 Ry, criterion 2e-4 Ry). A100, first runs with the kernels of earlier runs in the CuPy cache: diamond 6^3 cubes (1,728 atoms), `run_mpi_kpoints.py` with four ranks, 3 x 3 x 3 mesh 34.7 -> 33.5 s (fullest GPU 11,835 -> 11,353 MiB), 4 x 4 x 4 mesh 86.1 -> 84.8 s (27,751 -> 26,593 MiB); diamond 4^3 cubes, 2 x 2 x 2 mesh, one pool of four ranks sharing the columns (block route, sectors off) 38.6 -> 34.4 s (fullest GPU 14,785 -> 14,889 MiB; 46.3 -> 43.1 s with empty kernel caches); 216 carbon atoms with the 2 x 2 x 2 mesh in one process on four GPUs 19.0-19.4 -> 17.8 s (1,355 -> 1,275 MiB); `Si64_k222/pbe` 23.3 -> 21.5 s; `Si64_k222/pbe0` 160.6 -> 159.8 s. First runs with empty kernel caches (CuPy and CUDA; the real route compiles 9-11 kernels more, mostly the real sector stencil for each count of slots): no gain where Gamma's share is small, within the noise of the runs (off against off differed by up to 0.3 s): the 6^3 cubes 43.7, 43.9 -> 43.7, 44.1 s (3 x 3 x 3) and 94.9 -> 95.2 s (4 x 4 x 4); the 216 carbon atoms 38.9, 38.7 -> 39.0, 39.1 s; `Si64_k222/pbe` 40.9, 40.6 -> 41.2, 40.9 s (785 -> 773 MiB); `Si64_k222/pbe0` 180.8 -> 180.9 s (14,805 -> 14,277 MiB). Below the floor nothing changes (the 2-atom cells of `examples/3d_hybrid_functionals`, `hBN_k661/pbe`, 64 silicon atoms at 85,184 points with 144 states: the same kernels, energies and times); without the floor the 2-atom cells were 4-10 % slower with empty caches (`Si_k222/pbe` 9.9 -> 10.9 s, `C_k333/pbe` 22.7 -> 23.7 s) and the 64 silicon atoms took one SCF step more in two of three runs (19.3 against 19.5 s on average); see "Real arithmetic at Gamma and at the time-reversal invariant points" | `off`: the complex solve of before, bit for bit |
| `PARSEC_KPOINT_TRIM_REAL` | `auto` (default, since 2026-11-21): where a time-reversal invariant point other than Gamma (`k = G/2`) takes the Bloch form of the kinetic term (`PARSEC_KPOINT_KINETIC=bloch`, `auto` where it applies, or `trim`), its Hamiltonian is real in the psi gauge (`psi = e^{ik.r} u`, the neighbours across a face taken with `-1`): solved whole in a k-point SCF, it is solved there in float64 from a basis of `2^22` grid points times states (`SCF/kpoint_real.py`, `PsiGauge`, `psi_gauge_for`; the twisted real stencil kernel on the pools, `BlochStencil.from_psi`; not the band paths; below `2^22` the compile of its kernels costs more than it saves: the 8-atom silicon cube, 8,000 points and 24 states, empty kernel caches, 7.9 -> 8.4-8.5 s), and in the sectors of an unpolarized loop without the exchange of a hybrid one sector of each conjugate pair is solved (`trim_partners`: `e^{-iG.r} conj` carries a sector into its partner, of the same levels and density; the partner's orbitals and forces come from the sector solved). The same energies as the u gauge to round-off and the tolerance of the eigensolver. The default kinetic form (`gradient`) is not affected. Four A100, first runs, with `PARSEC_KPOINT_KINETIC=trim` (`off` -> `auto`, the same energies to the printed digits or within 1e-6 Ry, criterion 2e-4 Ry): 64 silicon atoms, 2 x 2 x 2 mesh, one task per k-point in one process, 13.0-13.1 -> 8.3-8.5 s (1,911 -> 1,149 MiB); 216 carbon atoms, one process, 18.3-18.4 -> 14.7-15.0 s (6 of 32 sectors solved by their partners); `Si64_k222/pbe` 23.5 -> 20.3 s (16 of 64); diamond 6^3 cubes, 4 x 4 x 4 mesh, four ranks, 87.8 -> 76.0-76.2 s (16 of 120; fullest GPU 26,593 -> 23,139 MiB); with `PARSEC_KPOINT_KINETIC=bloch`, diamond 4^3 cubes, 2 x 2 x 2 mesh, sectors off, four ranks (the twisted kernel) 35.9 -> 17.7 s (14,343 -> 7,271 MiB, 9.5e-7 Ry); with empty kernel caches 64 silicon atoms at 32 and 44 points per edge 34.0, 33.8 -> 33.6, 33.6 s and 39.5, 39.4 -> 38.6, 39.3 s, 64 carbon atoms at a/18 34.3, 34.7 -> 34.0, 34.1 s | `off`: the u gauge and every sector, bit for bit as before |
| `PARSEC_KPOINT_SECTOR_CLASSES` | `auto` (default, since 2026-11-22): the sectors of a k-point that an operation of its little group beyond the commuting involutions `H_k` carries into each other are solved once (`SCF/kpoint_sectors.py`, `sector_classes`): an operation `g` with `R k = k` that normalizes `H_k` maps the sector of character `chi` onto that of `chi(g^-1 h g)`, and the antiunitary `P_g K` of an operation with `R k = -k` does the same; both commute with `H_k` where the potential is invariant under `g`, so the two sectors have one spectrum and the orbitals of the second are `P_g u` (a signed permutation of the sector rows, `ClassMember`). Only operations the density is symmetrized over are taken (the irreducible wedge of a mesh with `Symmetrize_Charge_Density`), so the sector Hamiltonians are images of each other at every step, not only at self-consistency. The sector of the lowest index of each class is solved and takes the occupations of the class; the density of the class is the mean of the images of its density (the exact sum where the copies hold equal occupations, as their equal levels do), the forces and kept orbitals of the others come from the images of its states expanded in their own sectors. The scalar and the collinear-spin k-point SCF and the band paths in sectors (not the loops of a Fock loop, not the pairs of `PARSEC_KPOINT_TRIM_REAL`, not a loop with the Hubbard term of LDA+U in its sectors, whose occupation matrices the images would not add); an operation that holds the atoms only to a looser `PARSEC_SYMMETRY_ATOM_TOLERANCE` than 1e-6 bohr gives no images (the rule of `PARSEC_SECTOR_ORBITS_EXACT`, `exact_operation_test`; the real forms take the same rule); a line `Sectors of classes` in the K-point solver detail, and the band line names the sectors not solved. This is the weighted-sector rule of the clusters (`Symmetry/orbital_groups.py`) on the real characters of `H_k`: Td about an atom solves 10 of the 32 sectors of the Gamma point of 64 silicon atoms and half of those of the k-points on its threefold axes; C4v solves three of the four sectors of its mirrors on a fourfold axis. Four A100, one process, first calculation of each structure in a fresh process (the kernel cache of the node shared), SCF to 1e-7 Ry, SCF wall `off` -> `auto` (two runs each): Si64 (48^3 points, 160 states) 3 x 3 x 3 mesh 23.1, 21.8 -> 12.3, 12.5 s (32 of 64 sectors images, energies identical to the printed digits), 4 x 4 x 4 33.9, 33.4 -> 22.6, 23.0 s (46 of 120, 1e-10 Ry; 15 SCF steps instead of 14); MgO64 (48^3, 176 states) 33.1, 33.3 -> 17.3, 17.2 s (64 of 120) and 49.0, 49.9 -> 29.9, 29.3 s (88 of 208); Al108 (48^3, 270 states, 1000 K) 24.0, 20.4 -> 11.3, 11.2 s (32 of 60, 1.6e-9 Ry) and 31.8, 32.5 -> 19.9, 20.0 s (44 of 104); C216 (54^3, 540 states) 19.7, 18.6 -> 11.6, 11.7 s (16 of 32) and 36.4, 36.1 -> 24.3, 24.7 s (23 of 60; fullest GPU 2,143 -> 1,681 MiB); the GPU peak equal or lower everywhere. MPI (216 carbon atoms, 3 x 3 x 3 mesh, SCF to 1e-6 Ry, `run_mpi_kpoints.py`): one process on one GPU, four ranks in four pools and four ranks in one pool sharing the columns, `off` and `auto`: all six -2463.2375880796 Ry in 14 steps; four pools 6.2 -> 4.3 s, one pool 13.6 -> 7.8 s, one process 9.4 s with `auto`. The CPU route as well (the classes do not depend on the device code); with forces the nonlocal terms summed over the wedge agree within 2e-12 Ry/bohr (diamond cube, CPU) and the spin channels and band paths (band levels within 1e-11 Ry of the whole k-points) keep their numbers. | `off`: every sector solved, bit for bit as before |
| `PARSEC_KPOINT_SECTOR_CHARACTERS` | `real` (default): the commuting involutions of the little group of a k-point, with real characters, as before | `complex` (opt-in, since 2026-11-22): at every k-point other than Gamma a largest abelian subgroup of the little group with complex characters (`Symmetry/complex_sectors.py`) where its sectors and their classes cost less than those of the commuting involutions (a threefold or fourfold axis without the translations of a supercell: a defect cell); the gathered stencil with a complex palette, `U^H B` of one atom per orbit, the classes of `PARSEC_KPOINT_SECTOR_CLASSES` with `P_g K` taking `chi` to `conj(chi^g)`; not the pairs of `PARSEC_KPOINT_TRIM_REAL`, no real forms. Four A100, one process, first runs in fresh processes, `real` -> `complex` (two runs each, SCF to 1e-7 Ry, energies within 3e-10 Ry, the same SCF steps): MgO64 with a vacancy, 3 x 3 x 3 mesh, 11.0, 10.5 -> 10.4, 10.5 s (eigensolver 7.6 -> 7.2 s, rows times states squared 0.119 -> 0.065, fullest GPU 1,371 -> 1,009 MiB), 4 x 4 x 4 18.8, 18.9 -> 18.4, 18.5 s (1,801 -> 1,581 MiB); Si64 with a vacancy 9.3, 9.1 -> 8.5, 8.5 s and 38.7, 38.9 -> 37.9, 37.9 s; the perfect Si64 12.9, 12.3 -> 11.9, 11.8 s and 22.2, 22.5 -> 22.0, 22.2 s; the perfect MgO64 takes no complex sectors (the same work, the same times). Diamond 3^3 with an atom moved along (1, 1, 1) (C3v, no translations), four ranks in four pools and in one pool sharing the columns, `real` and `complex`: all -2463.2311876133 Ry in 14 steps. The gains in time are small where the k-points are many and the sectors of the largest k-point set the pace; the memory of the vacancy cells falls more. Not the default for now: measured on these cells only. See "Classes of sectors and real forms inside a k-point" |
| `PARSEC_KPOINT_SECTOR_REAL` | `off` (default): the complex sectors of before | `auto` (opt-in, since 2026-11-22): a sector of a k-point other than Gamma that an antiunitary `P_g K` maps onto itself with `(P_g K)^2 = 1` (`g` an operation the density is symmetrized over with `R k = -k` that normalizes `H_k` and fixes the character of the sector: the inversion of a centrosymmetric cell, a twofold axis or a mirror normal to `k`) is solved on float64 blocks in its real form `T^H H_s T` (`SCF/kpoint_real_forms.py`; the paired stencil kernel of `Hamiltonian/sector_pair_stencil.py`, the real projector parts `[R, -I] diag(D, D) [R, -I]^T` with CSR kernels of row-major real blocks, the orbitals `U T x`), in the device code of the scalar and the collinear-spin k-point SCF (not the CPU route, the band paths, the loops of a Fock loop, Gamma or the pairs of `PARSEC_KPOINT_TRIM_REAL`). Four A100, one process, first runs in fresh processes, the classes on in both, `off` -> `auto` (two runs each, SCF to 1e-7 Ry): Al108 (48^3 points, 270 states) 3 x 3 x 3 11.8, 11.9 -> 11.2, 11.6 s (18 sectors real; fullest GPU 877 -> 713 MiB), 4 x 4 x 4 19.5, 19.7 -> 20.1, 20.4 s (50 real; 965 -> 875-893 MiB), energies within 8e-9 Ry (12 or 13 SCF steps instead of 13); MgO64 17.7, 18.5 -> 17.2, 17.5 s (36 real; 621 -> 563 MiB) and 29.8, 29.8 -> 29.0, 29.2 s (100 real; 705-713 -> 649 MiB), the same energies to the printed digits; the total forces of MgO64 with Mg moved by 0.05 bohr within 8e-9 Ry/bohr of the complex sectors. The Chebyshev filter takes as long as the complex one (Al108 3 x 3 x 3: 13.4 against 13.5 worker-seconds; its gathers read the two coordinates of a neighbour on two rows where the complex kernel reads one complex entry, the same memory transactions), while the orthonormalization (2.2 -> 1.6 s) and the Rayleigh-Ritz steps (3.2 -> 3.0 s) fall. With the cuSPARSE products of the real projectors of a first version the runs were 1.2-1.6 times slower (Si64, C216, MgO64, Al108); four ranks in four pools or in one pool sharing the columns gave the energy of one process (216 carbon atoms, -2463.2375880796 Ry in 14 steps). Not the default: no gain in time on these cells, only in memory. A pair-major order of the rows (the two coordinates of a pair side by side, read at once) is the next step. See "Classes of sectors and real forms inside a k-point" |
| `PARSEC_ORDERED_SUMS` | `auto` (default, since 2026-11-22): the sparse matrices of the device operators of the k-point, band and spin-orbit solvers (the Kleinman-Bylander, LDA+U and spin-orbit projectors, the Laplacian and `k.grad` where no stencil kernel takes them; and the overlaps of the k-point nonlocal forces, the fields of a periodic meta-GGA, the reduction of a hybrid's exchange to a sector, the boundary term of the device wire and slab Hartree solve and the spinor backend) take products that add the terms of every entry in a fixed order (`Hamiltonian/complex_csr_device.py::OrderedCSR`: one thread per entry, a fixed tree over 32 threads for rows of more than 256 entries, the kernels the sectors' `DeviceComplexCSR` had already), and the k-point workers of one process add their density parts in the order of the workers, not in the order they finish: an input run again on the same devices gives the same bits. cuSPARSE's `csr @ dense` gives other last bits from call to call (A100: each of 30 repeats of a projector product differed from the first, by up to 1e-13), which a slowly converging or chaotic SCF amplifies: four runs of the 8-atom silicon cube with spin and a band path (the spin k-point driver at Gamma) took 35, 39, 54 and 79 iterations, other runs of it ended unconverged up to 0.13 Ry above the converged energy, and bcc Cr16 AFM with LDA+U took 132-152 iterations at the same energy; the CPU route gave the same bits every time. Under `auto` four runs of each, on different GPUs, give the same bits and steps: Si8 with spin and the band path 30 steps (`off`: 59, 59, 59, 98), Cr16 133 steps (`off`: 129-149); the MPI k-point pools too (Si8 3 x 3 x 3 mesh, four ranks, within 1.4e-14 Ry of one process). The energies agree with those of `off` within the SCF criterion (3e-14 to 2e-9 Ry). A100, first runs, kernel caches of the node warm, `off` -> `auto`: Cr16 on one GPU 31.6-34.4 -> 22.2-22.5 s, on four 22.1-27.8 -> 16.4-17.0 s; on four GPUs `Si64_k222/pbe` 25.4-26.8 -> 22.1-22.4 s, the 8-atom silicon cube with a 3 x 3 x 3 mesh 6.4-7.2 -> 5.6-5.8 s, with spin and a 2 x 2 x 2 mesh 10.6-11.6 -> 7.7-8.6 s, the PBE silicon cell of `examples/3d_meta_gga/Si_k222` (Fortran-derived routines) 7.2 -> 6.0-6.6 s. With empty kernel caches (every kernel compiled in the run), four GPUs: `Si64_k222/pbe` 49.1, 54.3 -> 44.8, 45.0 s; the 3 x 3 x 3 silicon cube 31.6, 34.3 -> 30.8, 30.0 s; Si8 with spin and the band path 33.7 s (57 steps) and 36.3 s (100 steps, unconverged) -> 28.0, 28.3 s (30 steps). The bits still depend on the number of GPUs a process uses (Cr16: 133 steps and -306.35013065002 Ry on one GPU, 138 steps and -306.35013065050 Ry on four). | `off`: the CuPy (cuSPARSE) products and the order in which the workers finish, as before |
| `PARSEC_SECTOR_DEVICE_POOL` | `auto` (default, since 2026-11-16, experimental runner): every sector of a rank spreads over all its devices, one after another, instead of disjoint groups that leave a whole sector on one device, where that holds less: under `PARSEC_SECTOR_PLACEMENT=size` (the default since 2026-11-18) where it lowers the fullest GPU by more than 5 % and the first solve divides every basis over the GPUs (see that row: five sectors on four GPUs take it, three with `Eigensolver: chebdav` keep disjoint groups), under `cyclic` where the sectors do not divide the devices evenly (three sectors on four GPUs). Measured with the count rule of before (`cyclic`): diamond n = 20 on 16 A100: 176.6 s and 50.1 GiB per GPU instead of 195.5 s and 52.2 GiB; n = 16 61.8 s / 17.9 GiB instead of 64.1 / 15.2; the units of `PARSEC_ORBITAL_IRREPS` that fit whole on single devices keep disjoint groups (the units of the rank, counted alone; five Td units of k10 on four GPUs: 19.9 s pooled, 16.3 / 16.5 s disjoint), larger ones the pool (the 20 units of a 64,000-atom diamond cube ran out of memory on disjoint GPUs) | `off`: disjoint groups, as before |
| `PARSEC_SINGLE_SECTOR_NODE_LAYOUT` | `refuse` (since 2026-11-19, the MPI full-SCF runner): one sector shared by several ranks that each hold the GPUs of a node (one rank per node on several nodes: a run without a point group, a cell under `PARSEC_PERIODIC_SYMMETRY=off`) is refused with advice where nested sharing does not spread it over the GPUs of every node (since the merge of 2026-11-20: after a CHEBDAV first solve or under `PARSEC_MPI_NESTED_SHARING=off`; with `Eigensolver: chebff` and the default `auto` such a run takes the nested route instead, `experimental/mpi_scf.py::nested_sharing_applies`, diamond 6^3 on 2 nodes 42.9 s), before the preparation where no group is looked for and right after the detection otherwise (`experimental/mpi_scf.py::single_sector_layout_refusal`, also checked when the sectors are configured): each rank would hold its share on its first GPU only. Diamond 6^3 (1,728 atoms) without symmetry: 2 nodes 327.0 s, 3 of 4 GPUs per node idle, against 36.8 s on one node (four GPUs share the basis); `Eigensolver: chebff` with nested sharing, one node, or one rank per GPU (`scripts/run_multi_gpu_scf.sh --rank-per-gpu`), shares the basis over every GPU. Runs of one rank or of ranks with one GPU each, and the exchange of a hybrid (its sector lies on the first GPU of the root by design), are not affected | `allow`: the run as before, its share of the basis on GPU 0 of every node |
| `PARSEC_MPI_RESTART_RUN` | `auto` (since 2026-11-19, the MPI full-SCF runner): `Restart_Run` as `main.py` takes it: the root loads `parsec.dat` beside the input once the system is prepared and the first SCF loop (of a single point, forces, a relaxation or molecular dynamics) starts from its potential, with the restart lines of `main.py` in parsec.out; without the file the run stops before its preparation with PARSEC's error 371; `Restart_From_Wfndat` is refused, and a spin-polarized restart too (both channels on the full grid, the rule of `main.py`), except with `--serial-control`, which takes that rule (`benchmarks/mpi_full_scf.py::_restart_rules`). CH4 PBE (268,096 points) restarted beside the `parsec.dat` of a converged run on A100: 1 SCF iteration on 1 and 4 ranks, the energy of `main.py` (-16.15720133 Ry), where the runner took the 15 iterations of a first run from the atomic densities and printed the restart header all the same | `off` (or `0`, `false`, `no`): the keyword ignored as before, bit for bit |
| `PARSEC_KPOINT_POOL_ASSIGN` | `auto` (default, since 2026-11-17): tasks of the k-point pools of different sizes (the sectors of the k-points) are planned by their sizes (`SCF/kpoint_pools.py`, `plan_pools_by_size`): the largest work (rows times states squared) first, each task to the pool that finishes it first among those whose GPUs hold it and whose nodes hold what their ranks keep on the host (80 % of the memory a node has free when the pools are formed); balances within 2 % count as equal, then no subspaces on the host, then the most pools; tasks of one size keep the round-robin plan | `cyclic`: every task planned as the largest one and dealt round-robin (the plan of before) |
| `PARSEC_KPOINT_FORCE_WEDGE` | `auto` (default, since 2026-11-18): the forces of a single point and a `bfgs`/`simple` relaxation of scalar or collinear-spin orbitals (and of non-magnetic spin-orbit coupling since 2026-11-19, `PARSEC_SOC_KPOINT_WEDGE`) on a Monkhorst-Pack mesh of the default routines run its irreducible wedge, as a single point without forces and Fortran's routines do: the operations of the space group that map the grid onto itself (and its finite-difference operators, `PARSEC_SCALAR_KPOINT_WEDGE`, since 2026-11-21) reduce the mesh and symmetrize the density, the nonlocal force of the wedge is completed over them and the total averaged over them (over every operation that maps the grid and the mesh where the wedge keeps fewer, `PARSEC_KPOINT_FORCE_GROUP`, since 2026-11-21; `Forces/total.py`, `compute_periodic_forces`; `symm_forces.f90`), so the structure keeps the group (`Input/parsec_input.py`, `_default_routines_force_wedge`). Diamond 6 x 6 x 6 cubes with the 3 x 3 x 3 mesh, one relaxation step on 16 A100: 4 of 27 points (24 operations) instead of the 14 pairs, 135.6 -> 54.6 s (SCF 118.2 -> 39.3 s), 12.7 -> 5.8 GiB on the fullest GPU. One atom of 3 x 3 x 3 cubes moved by 0.05 bohr along (1, 1, 1) (6 operations, forces up to 0.081 Ry/bohr), 2 x 2 x 2 mesh, SCF to 1e-6 Ry: 4 of 8 points, the same total energy to the digits printed and forces within 1e-6 Ry/bohr (at the 2e-4 Ry of the benchmark 2.6e-5: the forces of the whole mesh break the group by 2.1e-5 there) | `off`: the whole mesh, run as its pairs `k`, `-k` (`PARSEC_KPOINT_TIME_REVERSAL`), as before |
| `PARSEC_KPOINT_FORCE_ORBITS` | `auto` (default, since 2026-11-19): the nonlocal forces of the k-points solved in sectors (`PARSEC_KPOINT_SECTORS`) are built for one atom per orbit of the little group of each k-point and turned onto the other atoms of the orbit (`SCF/kpoints.py::kpoint_nonlocal_force` with `atoms`, `SCF/kpoint_sectors.py::KPointSectors.force_orbits` and `complete_orbit_forces`; the Bloch projector gradients of the other atoms are not built); a displaced cell on the CPU agrees with every atom within 1.8e-15 Ry/bohr. One node of 4 A100: diamond 4^3 cubes (512 atoms) with the 3x3x3 mesh, the nonlocal force pass 27.7 -> 3.8 s (SCF with forces 49.3 -> 25.1 s); 5^3 cubes (1,000 atoms) with the 2x2x2 mesh 54.0 -> 10.3 s (107.3 -> 63.4 s), the same energies and printed forces | `off`: the gradients and forces of every atom, as before |
| `PARSEC_KPOINT_TIME_REVERSAL` | `auto` (default, since 2026-11-18): forces and moving atoms of scalar or collinear-spin orbitals (and of non-magnetic spin-orbit coupling since 2026-11-19, `PARSEC_SOC_KPOINT_WEDGE`: the Kramers partner of a spinor at `-k` has the same density and forces) that keep the whole Monkhorst-Pack mesh of the default routines (molecular dynamics and `manual` movements, which use no symmetry, a `Move_Flag` that an operation of the group does not respect where no operation but the identity keeps its fixed atoms or under `PARSEC_MOVE_FLAG_SPACE_GROUP=off`, `Ignore_Symmetry`, a structure without operations, `PARSEC_KPOINT_FORCE_WEDGE=off`), and the SCF runs its pairs `k`, `-k`, one point of each with twice the weight (`Grid/kpoints.py`, `kpoint_sampling` with `SinglePointInput.kpoint_time_reversal`; `u_-k = u_k^*` gives the same density and forces, as Fortran's `irrbz` uses for every run that is not non-collinear); a mesh for the DOS tetrahedra or `Output_GW` stays whole (diamond 6 x 6 x 6 cubes with the 3 x 3 x 3 mesh, one relaxation step on 16 A100: 14 of 27 points, SCF 118.2 instead of 205.4 s, within 9e-8 Ry and 6e-6 Ry/bohr) | `off`: every point of the mesh, as before |
| `PARSEC_SOC_KPOINT_WEDGE` | `auto` (default, since 2026-11-19): the Monkhorst-Pack mesh of a bulk cell with self-consistent spin-orbit coupling (`SO_from_scratch`), neither spin-polarized nor non-collinear, on the default routines is reduced as PARSEC 1.5's `irrbz` reduces every `is_so` run that is not `ncl` (`parsec.F90:477`, `irrbz.f90:96-99`), by the operations that keep the discrete Hamiltonian: a single point takes the irreducible wedge of the operations of the crystal that map the grid, the mesh and the grid's finite-difference operators onto themselves (`Grid/kpoints.py`, `stencil_preserving_operations`: signed permutations of the grid axes that keep every weighted second difference; on an orthogonal grid every operation that maps the grid onto itself, on an oblique one a subgroup, hcp 8 of 24, the primitive fcc cell 12 of 48), with time reversal, and the scalar density is averaged over them (a spinor turned with its operation and SU(2) matrix has the turned density, its Kramers partner at `-k` the same one; `SCF/kpoints_soc.py`, `_symmetrized`); where no operation but the identity keeps the stencil, time reversal alone; forces and a `bfgs`/`simple` relaxation take the same wedge (`PARSEC_KPOINT_FORCE_WEDGE`; the force of each k-point weighted by its own weight, `SpinorForceExtras`, and completed over the group), molecular dynamics, `manual` movements and the other moving runs that keep the whole mesh run its time-reversal pairs (`PARSEC_KPOINT_TIME_REVERSAL`); collinear spin and non-collinear magnetism keep the whole mesh under every value (time reversal and the operations that turn the spin axis are no symmetries of their moments, and the group found from the atoms ignores the moments) (`Input/parsec_input.py`). The wedge gave the energy of the same points without the group (time reversal alone, exact for the discrete problem) to SCF precision on every cell measured. One A100 node, first runs, Au with its relativistic pseudopotential: bcc Au2 with one atom moved along (1, 1, 1) (C3v, 0.31 bohr, 4 x 4 x 4), forces: 13 of 64 points, 77.6 -> 22.5 s, host 2.25 -> 1.61 GiB, GPU 0.89 -> 0.67 GiB, the energy within 4e-8 Ry and the forces within 6e-6 Ry/bohr of the whole mesh (its time-reversal pairs, 36 points, 47.3 s: within 4e-8 Ry and 3e-6 Ry/bohr of the whole mesh, 2e-10 Ry of the wedge); two relaxation steps 251.6 -> 62.2 s (host 3.11 -> 1.83 GiB), every energy within 1.2e-6 Ry (SCF criterion 1e-6 Ry) and the final forces within 1.6e-7 Ry/bohr; molecular dynamics, 3 x 3 x 3, 14 of 27 points, 84.8 -> 51.9 s, within 2e-9 Ry and 8e-8 Ry/bohr. hcp Au2 (0.3 bohr, 4 x 4 x 2): 14 points against 20 (time reversal alone) and 32 (whole mesh), 21.3 against 30.9 and 46.7 s, host 1.47 against 1.54 and 1.72 GiB, GPU 0.70 against 0.72 and 0.79 GiB, within 1e-10 Ry of time reversal alone (at 0.2 bohr 2e-11 Ry, 28.7 against 39.3 s in a second run); hcp Au2 with one atom moved along c, forces: 2 operations kept, 14 points, the forces within 1e-7 Ry/bohr and the energy within 1.3e-8 Ry of the time-reversal pairs, 27.0 against 32.8 s (pairs) and 48.7 s (whole mesh) in a second run. fcc Au1 (0.3 bohr, 4 x 4 x 4): 13 points, 19.9 against 32.5 s (36 points of time reversal alone) and 53.3 s (64), within 2e-9 Ry. The first run of a grid shape on a node also pays the compilation of its CuPy kernels once, which falls on whichever run comes first (about 5 s at 0.3 bohr and 17 s at 0.2 bohr for hcp). Since 2026-11-21 the points of the whole mesh (`off`) and of the wedge are the shortest of their images modulo the reciprocal lattice, a point of the zone boundary with all its shortest images at equal shares (`PARSEC_KPOINT_FOLD=shortest`; the stars of the wedge split into orbits of exact images, fcc Au1 4 x 4 x 4 14 points instead of 13), and the wedge gives the energy of that whole mesh itself: fcc Au1 (0.3 bohr, 4 x 4 x 4) within 3.6e-10 Ry (14 against 89 points, 15.3 against 75.6 s, one A100 per run and four runs on a node); its forces are averaged over every operation that maps the grid and the mesh (`PARSEC_KPOINT_FORCE_GROUP`). With `PARSEC_KPOINT_FOLD=centred` the whole mesh takes its points folded into (-1/2, 1/2] in lattice coordinates, which the Bloch operator of an oblique grid does not treat alike (fcc 1.8e-3 Ry, hcp 2.2e-3 Ry from the wedge, as before 2026-11-21) | `grid`: every operation that maps the grid and the mesh onto themselves, the group scalar orbitals took before 2026-11-21 (`PARSEC_SCALAR_KPOINT_WEDGE=grid`; the `auto` of the first version of this switch, bit for bit): fewer points on an oblique grid (hcp 8, 14.2 s; fcc 8, 13.0 s), but their wedge differs from the same points without them at the order of the discretization (hcp Au2 7.7e-3 Ry at 0.3 bohr, 1.0e-3 Ry at 0.2 bohr for the moved hcp; fcc Au 1.2e-4 Ry). `off`: the whole mesh of equal weights, as before, bit for bit |
| `PARSEC_SCALAR_KPOINT_WEDGE` | `auto` (default, since 2026-11-21): the irreducible wedge of the Monkhorst-Pack mesh of scalar or collinear-spin orbitals of a bulk cell on the default routines (a single point, and forces and a `bfgs`/`simple` relaxation under `PARSEC_KPOINT_FORCE_WEDGE`) keeps the operations of the crystal that map the grid, the mesh and the grid's finite-difference operators onto themselves (`Grid/kpoints.py`, `stencil_preserving_operations`, as `PARSEC_SOC_KPOINT_WEDGE=auto` does for spinors), with time reversal; the density is averaged over them (`Input/parsec_input.py`, `_default_routines_wedge`) and the forces over the larger group of the grid and the mesh (`PARSEC_KPOINT_FORCE_GROUP`). On an orthogonal grid that is every operation that maps the grid and the mesh, bit for bit as `grid`; on an oblique one a subgroup (hcp Au2, hexagonal Si and AB graphite 8 of 24, primitive fcc and bcc cells 12 of 48, primitive Si 6 of 24), time reversal alone where only the identity is left. With `PARSEC_KPOINT_FOLD=shortest` the wedge computes exactly the points of the whole mesh without symmetry on the same grid, in which a point of the zone boundary takes all its shortest images with equal shares of its weight (the same-grid reference that does not depend on how a tie is broken). A100, one GPU per run, first runs, order 6 and 0.3 bohr unless named, wedge minus that whole mesh: fcc Au 4x4x4, 6x6x6 and 8x8x8 6e-12, 2e-11 and 3e-13 Ry (14, 34 and 71 points against 89, 259 and 597), 6x6x6 at 0.25 bohr 6e-11 Ry and at order 12 5e-11 Ry; hcp Au2 4x4x2 and 6x6x4 5e-11 and 8e-11 Ry (14 and 42 points against 57 and 215); primitive bcc Au 6x6x6 1.4e-10 Ry (34 against 279); primitive Si 6x6x6 (0.4 bohr, order 12) 4e-11 Ry; spin-orbit fcc Au 4x4x4 3.6e-10 Ry (`PARSEC_SOC_KPOINT_WEDGE`); hcp Au2 with an atom moved along c, 19 points against 57: 9e-11 Ry, forces along c within 8e-8 Ry/bohr. The cost is the points the dropped operations no longer fold: one node with four A100, one run at a time, first runs (second runs within 1 s): hcp Au2 (0.2 bohr, order 12, 4x4x2) 14 against 8 points, 13.1 against 9.9 s; fcc Au (0.25 bohr, order 12, 6x6x6) 34 against 16, 15.8 against 10.2 s; AB graphite (0.25 bohr, 8x8x2) 44 against 20, 27.3 against 16.4 s; hexagonal Si (0.3 bohr, 6x6x3) 28 against 14, 20.2 against 13.1 s; primitive bcc Na (0.3 bohr, 8x8x8) 69 against 29, 27.0 against 14.6 s; primitive Si (0.4 bohr, order 12, 6x6x6) 34 against 16, 17.1 against 11.5 s (1.3 to 1.8 times). The memory of the orbitals and the time of the eigensolver grow with the number of k-points, here 1.75 to 2.4 times as many (on these cells of one to four atoms the host memory grew by 0.05 to 0.27 GiB and the GPU peak by 40 to 90 MiB; a large oblique cell near the memory limit needs that factor on its orbitals). Light species at order 12 gain less than the SCF criterion for that cost (from the former route: hexagonal Si 1.7e-8 Ry, primitive bcc Na 1.4e-10 Ry, primitive Si 1.8e-7 Ry); heavy ones more (fcc Au 9.1e-6 Ry at 0.25 bohr and order 12, hcp Au2 1.6e-6 Ry at 0.2 bohr). No estimate of the error of the dropped operations before the run is implemented, so the default is the route that gives the same-grid result for any species and step | `grid`: every operation that maps the grid and the mesh onto themselves, as before 2026-11-21, bit for bit: fewer points, but the wedge differs from the same-grid whole mesh at the order of the discretization; against the whole mesh of shared ties, order 6 and 0.3 bohr: fcc Au 4x4x4 -4.7e-4, 6x6x6 -3.1e-4 and 8x8x8 -9.7e-6 Ry, 6x6x6 at 0.25 bohr -1.8e-4 Ry and at order 12 -2.5e-5 Ry, hcp Au2 4x4x2 -1.2e-3 and 6x6x4 -1.5e-3 Ry, primitive bcc Au +2.8e-3 Ry, primitive Si (0.4 bohr, order 12) -1.8e-7 Ry; hcp Au2 4x4x2 at order 12 -7.2e-5 Ry (0.3 bohr) and -1.6e-6 Ry (0.2 bohr), hexagonal Si 4x4x2 at order 12 -1.9e-6, -1.8e-7 and -5.7e-9 Ry at 0.5, 0.4 and 0.3 bohr (meshes whose stars hold exact images only); the force along c of the moved hcp Au2 3.4e-5 Ry/bohr below that of the whole mesh at order 6 |
| `PARSEC_KPOINT_FOLD` | `shortest` (default, since 2026-11-21): the k-points of a Monkhorst-Pack mesh on the default routines (a bulk cell, and a slab of `Use_Fortran_Routines: .false.`) are computed at the shortest of their images modulo the reciprocal lattice, the points of the first Brillouin zone (`Grid/kpoints.py`, `shortest_images`: the point folded into (-1/2, 1/2] where it is one of the shortest, else found by an exhaustive search). A point of the zone boundary with several equally short images (the fcc X, L and W points, the hexagonal M and K points, the plane kz = 1/2 of a hexagonal mesh) runs as all of them, each with an equal share of its weight (`SinglePointInput.kpoint_tie_images`): in the explicit list of the whole mesh (forces and moving atoms without a wedge, molecular dynamics, magnetic spinors, `PARSEC_SOC_KPOINT_WEDGE=off`, slabs; `tie_shared_points`) and in the stars of a wedge, split into the orbits of exact images where the operations of the wedge do not relate the images of a point (`irreducible_monkhorst_pack` with `ties`; a wedge whose stars hold exact images only keeps every bit: fcc Au 4x4x4 14 points instead of 13, 6x6x6 34 instead of 32, 8x8x8 71 instead of 65, hcp 6x6x4 42 instead of 39, hcp 4x4x2 14 as before; the time-reversal pairs of a whole mesh: fcc 2x2x2 and 3x3x3 8 and 14 as before, 4x4x4 45 instead of 36, bcc 2x2x2 10 instead of 8, 4x4x4 47 instead of 36, hexagonal 2x2x2 11 instead of 8, 3x3x2 20 instead of 10, 4x4x2 29 instead of 20: molecular dynamics and the forces of a cell without symmetry on such meshes take that many more points). The result then does not depend on how a tie is broken, every operation of the lattice maps the set onto itself, and the time-reversal pairs are exact (`-k` itself is in the set; `time_reversal_reduced` takes it): a primitive fcc Au2 cell without symmetry (6x6x6, 0.3 bohr, order 6, forces) on its 130 pairs gave the energy of the whole mesh of 259 points within 7e-12 Ry and its forces within 5e-9 Ry/bohr (the pairs of the folded points, one per class, 3.3e-4 Ry and 1.9e-5 Ry/bohr from it), hcp Au2 without symmetry within 5e-11 Ry and 5e-8 Ry/bohr (29 against 57 points). A hybrid keeps one image per point, as its exchange needs one point per class, chosen so that the set is closed under `k -> -k` (`negation_closed_shortest`), and so do `Create_DOS` and `Output_GW` (which run on the Fortran-derived routines): PBE0 hexagonal Si 4x4x2 on its 20 pairs moved by -3.7e-6 Ry from the folded points, and its wedge (`PARSEC_HYBRID_KPOINT_WEDGE=auto`, 14 points) lay 4.4e-8 Ry from them, within `Exx_Tolerance` 1e-7 Ry. Why the shortest images: the Bloch operator of these routines, `-nabla^2 - 2ik.grad + k^2` on the periodic factor `u_k`, is not invariant under `k -> k + G`; the leading plane wave `e^{i k_s.r}` of a low band, `k_s` the shortest image, is a constant `u` at `k_s` (no finite-difference error) and the plane wave `e^{-iG.r}` at `k_s + G`. Band energies relative to the lowest level at Gamma, whole mesh without symmetry, against the same at 0.15 bohr and order 12, on the points whose two images differ (A100): fcc Au 4x4x4 at 0.3 bohr and order 6 rms 9.0e-4 Ry with the shortest images against 1.6e-3 Ry with the folded ones (largest 2.4e-3 against 3.9e-3 Ry), at order 12 1.26e-4 against 1.24e-4 Ry (largest 2.1e-4 against 2.9e-4 Ry); hcp Au2 4x4x2 8.4e-4 against 2.0e-3 Ry at order 6, 4.4e-5 against 1.1e-4 Ry at order 12. The total energy of these coarse grids lies below that of the fine grid, and the folded images raise it by error cancellation, so it is closer to the fine value with the folded points (fcc Au -1.92e-2 against -2.10e-2 Ry, hcp Au2 -4.33e-2 against -4.77e-2 Ry at order 6): the case for the shortest images rests on the levels, not on the absolute energy. The two sets are the same points of the continuum, and their difference falls with the step and the order (the whole mesh of one shortest image per point minus that of the folded ones, order 6, 0.3, 0.25 and 0.2 bohr: hcp Au2 4.3e-3, 2.5e-3 and 5.7e-4 Ry, fcc Au 1.8e-3, 1.0e-3 and 2.4e-4 Ry; order 12 at 0.2 bohr 3.0e-6 and 1.2e-6 Ry; the h-BN slab, PBE, order 12, 6x6x1: 8.6e-6, 4.9e-7 and 2.4e-8 Ry at 0.25, 0.2 and 0.15 bohr). The same points as before on a lattice of orthogonal vectors, whose folded points are the shortest ones (a tie at 1/2 keeps the folded point there): Si8 3x3x3, bcc Au2 4x4x4 with forces and an orthorhombic Si8 4x4x2 keep their k-points, weights and groups bit for bit (energies on the A100 within 3e-17 Ry, forces within 3e-15 Ry/bohr), and the 37 periodic inputs of `examples/` that the reader takes locally parse unchanged. The Fortran-derived routines keep their points under both values (PARSEC 1.5's `irrbz` keeps them in [0, 1)), except a mesh with a free level below zero on routines that were not requested (`PARSEC_KPOINT_MESH_GUARD`) | `centred`: every point folded into (-1/2, 1/2] along the reciprocal lattice vectors, one point per mesh point, as before, bit for bit |
| `PARSEC_KPOINT_LIST_FOLD` | `auto` (default, since 2026-11-21): the points of a `Kpoint_List` or `Kpoint_File` on an oblique cell (`Grid/kpoints.py`, `kpoint_list_fold_setting`; the points of an orthogonal cell follow `PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD`, and `literal` keeps them as written). PARSEC 1.5 runs the written vectors (`usrinputfile.F90`, `init_var.F90`), but the discrete Bloch operator `-L - 2ik.G + k^2` of an oblique grid has plane waves with `-L < |G|^2`, so beyond some `|k|`, in general outside the first Brillouin zone, it has levels below zero that no continuum state has (on an orthogonal grid each second difference bounds the square of its first difference, and no level is below zero at any `k`). On the default routines every point runs at its shortest images modulo the reciprocal lattice, a point of the zone boundary at all of them with equal shares of its weight; a list with no point outside the zone and none on its boundary keeps its bits. Ties are found up to the precision of the input (`written_tie_images`: `|k - G|^2 - |k|^2 <= 1e-3 |G|^2`, the distance from the bisector of `G` in units of `|G|`, so that the close planes of a long cell are not joined; the mesh points nearest a bisector on the seven cells below, meshes up to 16 per axis, are at 0.010 or more). A whole mesh written out then gives the result of its `Monkhorst_Pack_Grid` for any choice of images; what remains is the rounding of the written coordinates, which is not corrected: on 6x6x6 and 6x6x4 meshes, lists with 16 decimals within 8.3e-11 Ry, with 8, 6 and 4 decimals -2.4e-9, -2.4e-7 and -2.4e-5 Ry (fcc Au), -2.9e-9, -3.0e-7 and -3.0e-5 Ry (hcp Au2), -4.3e-9, -4.2e-7 and -4.2e-5 Ry (graphite). A hybrid keeps one image per point, those of its mesh whatever the order of the lines (`negation_closed_written`: components within 1e-9 of a half-integer taken as that half-integer, the images chosen in the order of the mesh, and of each pair `k`, `-k` the member the mesh keeps): PBE0 hexagonal Si, primitive Si, fcc Au and hcp Au2 written in [0, 1) or in (-1/2, 1/2], in the order of the mesh, reversed or shuffled, within 3e-14 Ry of the time-reversal pairs of the mesh (before: up to 4.7e-5 Ry on hcp Au2). `Create_DOS` and `Output_GW` keep one point per mesh point. The Fortran-derived routines keep the written vectors unless a written point has a free level below zero (`lowest_free_levels`: every plane wave of the grid from the symbols of its stencil, one plane at a time, at parse time and the same on every MPI rank; 0.4 s for 2.1e7 waves and 3.3 s for 1.7e8 on one core of an A100 node; equal to the FFT of the real matrices within 3e-14 Ry where it is below zero, on 13 bulk and 3 slab grids): where they were not requested (for instance a grid of at most `Expansion_Order` points along an axis) that list is then treated as on the default routines, with a note; under `Use_Fortran_Routines: .true.` it keeps its vectors with a warning that names the point, its `|k|` and `shared`. The test fires on none of the 35 periodic inputs of `examples/` with k-points (four of them with another functional, the same grid and mesh, where libxc is not available), whose 25 lists keep their k-points and weights. It can fire on a point of the first zone of a very coarse grid (a primitive bcc cell of a = 5.5 bohr with 6 points per axis at order 12, the point (1/4, 1/4, 1/4) of a 4x4x4 mesh at -1.59 Ry): no image removes that level, so the run goes on with a warning that the grid is too coarse. A100, one GPU per run, a fresh process per run with the kernel cache of the node: hcp Au2, fcc Au, primitive Si, hexagonal Si, AB graphite, primitive bcc Na and an h-BN slab, the whole mesh as a list written in (-1/2, 1/2] or in [0, 1), against `Monkhorst_Pack_Grid`: within 1.1e-9 Ry under `auto` for both writings (`literal`: up to 4.3e-3 Ry and 3.8e-2 Ry, bcc Na in [0, 1) -1.83 Ry at 0.4 bohr); forces of fcc Au2 without symmetry 1.3e-11 Ry and 3.3e-8 Ry/bohr, hcp Au2 with an atom moved along c 9e-11 Ry and 8e-8 Ry/bohr along c (in the plane the list keeps the grid's 4.2e-5 Ry/bohr, which the mesh averages away over the crystal's group), collinear spin bcc Na 7e-9 Ry, spin-orbit fcc Au 3.6e-10 Ry. Primitive bcc Na 6x6x6 written in [0, 1) on the automatic Fortran route (order 12), against -0.47199 Ry of the mesh at 0.3 bohr: at 0.6 bohr -8.44798 -> -0.47216 Ry, at 0.5 bohr -7.84742 -> -0.47204 Ry. The cost of the shared images of a written-out mesh is that of its extra points against the same list unshared: 32 -> 57 points 15.5 -> 25.3 s (hcp Au2), 64 -> 89 22.9 -> 28.9 s (fcc Au), 64 -> 89 23.1 -> 30.7 s (primitive Si), 32 -> 57 16.6 -> 28.7 s (hexagonal Si), 72 -> 129 36.9 -> 55.9 s (graphite), 216 -> 279 57.0 -> 72.1 s (bcc Na), 36 -> 43 32.4 -> 33.2 s (h-BN), 1.03 to 1.73 times; `Monkhorst_Pack_Grid` reaches the same result on its wedge | `shared`: the treatment of the default routines on every route, `Use_Fortran_Routines: .true.` included. `literal`: the written vectors on every route, as before (the same k-points; A100 energies within 4e-13 Ry of the former source on the seven cells and within 5e-14 Ry on bcc Na at 0.6 and 0.5 bohr, the run-to-run scatter of the unchanged mesh route), with the same warning where a written point has a free level below zero |
| `PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD` | `auto` (default, since 2026-11-22): a point of a `Kpoint_List` or `Kpoint_File` of an orthogonal cell that lies outside the first Brillouin zone runs at its shortest image modulo the reciprocal lattice, with its weight, on the default routines (and on every route under `PARSEC_KPOINT_LIST_FOLD=shared`; `Input/parsec_input.py`, `_folded_orthogonal_list`). An orthogonal grid has no free level below zero, but its finite-difference dispersion departs from `|k + G|^2` the more the farther the written vector lies from the zone: a Bloch state of the zone point `k_s` written at `k_s + G0` is held by the waves `q - G0`, nearer the edge of the grid. Free levels of the plane waves within 2 bohr^-1, written point against its image: fcc Au cube 7.71 bohr at 0.35 bohr and order 6, (3/4, 3/4, 3/4) 3.6e-3 against 3.1e-4 Ry; a tetragonal cell 5.6 x 8.4 bohr, (3/4, 3/4, 2/3) 7.4e-3 against 1.8e-4 Ry; a simple cube of 6 bohr at order 12 3.9e-5 against 3.3e-6 Ry. A point of the zone boundary (a component 1/2) keeps its written vector: its images give the same levels and density by time reversal on these grids (the u-gauge operator at `-k` is the conjugate of that at `k`), so no weight is shared and no point is added; a list whose points all lie in the first zone keeps every bit (the 25 lists and 10 meshes of `examples/` keep their k-points). The Fortran-derived routines keep the written vectors, as PARSEC 1.5, under `PARSEC_KPOINT_LIST_FOLD=auto`; a Monkhorst-Pack mesh of an orthogonal cell is already in the first zone on both routine sets. A100, one GPU per run, the whole mesh written as a list in [0, 1) minus the mesh, `off` -> `auto`: fcc Au2 in its tetragonal cell 4x4x3 (forces) +9.23e-2 -> +7.6e-11 Ry (forces 1.1e-6 -> 6.5e-10 Ry/bohr), the fcc Au cube 4x4x4 +5.81e-2 -> -1.0e-10 Ry, bcc Au2 in its cube 4x4x4 with forces +2.69e-2 -> -6.4e-8 Ry (the list without symmetry against the wedge, the same as the list written in (-1/2, 1/2]), Si8 3x3x3 +1.59e-5 -> +4.4e-12 Ry, a tetragonal Si8 4x4x2 +1.24e-5 -> -1.2e-10 Ry, simple cubic Na 6x6x6 +1.79e-7 -> +1.9e-11 Ry; the same points as before, so the same time (26.5 against 26.8 s, 33.7 against 33.4 s) | `off`: the written vectors of an orthogonal cell on every route, as before |
| `PARSEC_KPOINT_MESH_GUARD` | `auto` (default, since 2026-11-21): a Monkhorst-Pack mesh with a point whose free finite-difference level is below zero (`Grid/kpoints.py`, `lowest_free_levels`, see `PARSEC_KPOINT_LIST_FOLD`) on Fortran-derived routines that the input did not request (for instance a grid of at most `Expansion_Order` points along an axis) runs at the shortest representatives of its wedge and, outside a hybrid, with the images of a point of the zone boundary at equal shares (`Input/parsec_input.py`, `_guarded_mesh`), with a note; `Create_DOS` and `Output_GW` keep their points with a warning, and so does `Use_Fortran_Routines: .true.`. A mesh without such a point changes in nothing: none of the 35 periodic inputs of `examples/` with k-points. A100, one GPU per run, a fresh process per run with the kernel cache of the node, primitive bcc Na at 0.6 bohr (order 12) against the same mesh at 0.3 bohr: 4x4x4 -0.70503 -> -0.46615 Ry (0.3 bohr -0.46599), 6x6x6 -0.61897 -> -0.47216 Ry (-0.47199), 8x8x8 -0.60609 -> -0.47405 Ry (-0.47389), where (1/2, 1/2, 1/2) at `|k|` = 1.362 1/bohr has a free level of -2.41 Ry and an occupied SCF level of -2.41 Ry; collinear spin 4x4x4 -0.70684 -> -0.46894 Ry (-0.46878); PBE0 4x4x4 -0.69391 -> -0.46102 Ry (-0.46085); two Na atoms without symmetry, 4x4x4, forces at 0.7 and 0.65 bohr: 2.2e-4 and 8.1e-5 Ry from 0.3 bohr (off: 0.26 and 0.054 Ry), the largest force component 3.8e-4 and 2.5e-4 Ry/bohr from it (off: 5.4e-4 and 7.1e-4); two `bfgs` moves of that cell at 0.65 bohr end at -0.764385 Ry (0.3 bohr -0.764383, off -0.818008); spin-orbit bcc Au 4x4x4 on a grid of 10 points per axis -67.30481 -> -66.87120 Ry (0.25 bohr -66.82571; the same mesh as a list with all its images -66.86699: the 4.2e-3 Ry of the wedge came from the operations that do not keep the finite-difference operators of the grid, which `PARSEC_KPOINT_GUARD_WEDGE` leaves out since 2026-11-22); four MPI ranks and a `Restart_Run` give the energy of the single run (-0.47215632 Ry, the restart within 1.4e-10 Ry). The time is that of the points: the same 16 points 6.7 -> 7.0 s, 36 -> 47 points with forces 12.1 -> 14.5 s, the relaxation 30.6 -> 39.5 s. These numbers were taken with every operation of the space group in the wedge (`PARSEC_KPOINT_GUARD_WEDGE=crystal` since 2026-11-22; the default moves the bcc Na energies above by less than 2e-7 Ry) | `off`: the points of the mesh as before (bcc Na 6x6x6 at 0.6 bohr within 2e-15 Ry of the former source), with a warning |
| `PARSEC_KPOINT_GUARD_WEDGE` | `auto` (default, since 2026-11-22): the wedge of a mesh that `PARSEC_KPOINT_MESH_GUARD` moves (the Fortran-derived routines chosen automatically on a coarse oblique grid) takes the operations of the space group that also map the finite-difference operators of the grid onto themselves (`Grid/kpoints.py`, `stencil_preserving_operations` with `grid_stencil_lines`; `Input/parsec_input.py`, `_guarded_stencil_wedge`), with time reversal, the density averaged over them, and the forces over every operation of the space group (`SinglePointInput.force_space_group`, as `PARSEC_KPOINT_FORCE_GROUP=crystal` does on the default routines). Fortran's general grid of a primitive bcc cell has second differences along its three axes and the fourth body diagonal and first differences along the axes alone: 12 of the 48 operations keep them, and the other 36 turn an axis onto a direction of no first difference, which changes the discrete Bloch Hamiltonian at the order of the discretization, so that their wedge is not the energy of the same points without symmetry. A hybrid keeps its k-points: on its time-reversal pairs (the whole mesh) it averages no density, as on the default routines, a wedge it takes after an insulating start averages over the subgroup, and so does its wedge under `PARSEC_HYBRID_KPOINT_WEDGE=on`; primitive bcc Au with PBE0 4x4x4 at 0.5 bohr, wedge minus list -9.29e-4 Ry over every operation (`crystal`; -3.4e-4 Ry over the subgroup: the exact exchange of the pairs follows their orbitals, which a density average does not make symmetric), -2.8e-11 Ry without the average (51.2 against 44.1 s); a mesh whose operations all keep the operators changes in nothing; a subgroup of the identity alone reduces the mesh by time reversal and averages nothing. A100, one GPU per run, first runs, against the whole mesh at its shortest images with its ties shared, written as a `Kpoint_List` (the run without symmetry), wedge minus list `crystal` -> `auto`: spin-orbit bcc Au 4x4x4 on a grid of 10 points per axis -4.21e-3 -> +4.3e-11 Ry (8 -> 14 k-points, 15.0 -> 17.2 s; the list of 93 points 89 s); the same cell without spin-orbit coupling -4.13e-3 -> -7.1e-11 Ry (5.5 -> 7.3 s); primitive bcc Na at 0.6 bohr 6x6x6 -1.44e-7 -> -3.4e-11 Ry (16 -> 34 k-points, 6.6 -> 10.4 s), 4x4x4 -1.94e-7 -> -2.7e-11 Ry (4.3 -> 5.7 s), with collinear spin -1.92e-7 -> -3.1e-11 Ry (9.5 -> 14.3 s); body-centred tetragonal Na (4 of 16 operations kept) -3.7e-7 -> -1e-13 Ry at 1.85 times the time (30 -> 72 k-points), with spin -2.1e-7 -> +1e-12 Ry at 1.94 times. The cost is that of the points: 1.2 to 1.6 times on Au, 1.3 to 1.9 times on Na, where the gain is only 1.4e-7 to 3.7e-7 Ry. Two Na atoms in the primitive cell (16 operations, 4 of them kept) at 0.7 bohr: the force on the moved atom -0.10043559 Ry/bohr as the list's (`crystal` -0.10043548), the slope of the free energy of the displaced cells -0.10043716 at steps of 1e-3 bohr and -0.10045097 at 2e-3 bohr (the two differ by the truncation of this coarse grid; extrapolated -0.10043255). Four MPI ranks (the k-point pools) give the energy of one process: bcc Na 6x6x6 within 1.5e-15 Ry; spin-orbit bcc Au (the row-distributed MPI solver; before 2026-11-22 every spin-orbit or non-collinear k-point run with `--mpi` stopped there with a `TypeError` of `distributed_bloch.py`) within 3.1e-11 Ry, and within 1.8e-11 Ry under `crystal`. | `crystal`: every operation of the space group, as before (bcc Na bit for bit, bcc Au within 2.4e-11 Ry, the noise of the GPU sums of the unchanged list) |
| `PARSEC_KPOINT_FORCE_GROUP` | `crystal` (default, since 2026-11-21): where the wedge of the forces or of a `bfgs`/`simple` relaxation on the default routines keeps only the operations that keep an oblique grid's finite-difference operators (`PARSEC_SCALAR_KPOINT_WEDGE=auto`, `PARSEC_SOC_KPOINT_WEDGE=auto`), the total force is averaged over every operation of the space group that maps the grid and the mesh onto themselves (`SinglePointInput.force_space_group`; `Forces/total.py`, `compute_periodic_forces`; `Forces/periodic.py`, `periodic_force_operations`), the group of the forces before 2026-11-21, while the k-point sum and the density stay those of the wedge (`Input/parsec_input.py`, `_default_routines_force_wedge`, with a note in `parsec.out`; dropped where an operation exchanges fixed and movable atoms). A structure whose operations all turn the stencil (a hexagonal cell with C3 alone) runs the whole mesh as its time-reversal pairs, and its forces are averaged over those operations too (a CPU test: the forces of the C3 cell become their C3 average to 1e-9 Ry/bohr, the energy unchanged). The forces of the converged grid have that symmetry, and the average is the orthogonal projection onto it, so it brings the forces of all atoms closer to them (or leaves them): it removes the part that the oblique grid breaks and keeps a relaxed structure in the group. hcp Au2 with an atom moved along c (0.3 bohr, order 6, 4x4x2, A100): the in-plane forces 0 instead of 4.8e-5 Ry/bohr, the force along c that of the wedge (8e-8 Ry/bohr from the whole mesh), the same energy and time (12.6 against 12.3 s); three `bfgs` movements kept the 12 operations of the structure, its atoms on the three-fold axes, 6.5e-6 bohr from the positions of the former route along c (34.7 against 34.6 s) | `wedge`: the total averaged over the operations of the wedge only, the forces of the same points without symmetry on that grid: the in-plane forces of that cell 4.8e-5 Ry/bohr at order 6 and 0.3 bohr, 2.8e-5 and 6.6e-6 Ry/bohr at 0.25 and 0.2 bohr, 3.4e-6 and 5.5e-7 Ry/bohr at order 12 and 0.3 and 0.2 bohr; three `bfgs` movements moved its atoms off the axes by 1.2e-4 bohr, and the relaxed structure has 4 of the 12 operations |
| `PARSEC_KPOINT_LIST_FORCE_GROUP` | `auto` (default, since 2026-11-22): the forces and the moved positions of a `Kpoint_List` of scalar or collinear-spin orbitals on the default routines of a bulk cell (the forces of a single point, `bfgs`, `simple`) are averaged over the operations of the space group that map the grid and the weighted list (its classes `{k, -k}` modulo the reciprocal lattice) onto themselves (`Grid/kpoints.py`, `list_preserving_operations`; `Input/parsec_input.py`, `_default_routines_list_force_group`, with a note in `parsec.out`; `SinglePointInput.force_space_group`), as the same points given as a `Monkhorst_Pack_Grid` are (`PARSEC_KPOINT_FORCE_GROUP=crystal`); nothing where only the identity keeps the list, with `Ignore_Symmetry`, or where an operation exchanges fixed and movable atoms. The forces of the converged grid have that symmetry, and the average removes the part an oblique grid breaks. Measured on A100, list against the mesh (the whole mesh written as a list, in [0, 1) and in (-1/2, 1/2]), largest force difference: hcp Au2 with an atom moved along c (4x4x2, 0.3 bohr) 4.2e-5 -> 1.6e-10 Ry/bohr, the same with a shift of 1/2 along c 1.1e-4 -> 2.9e-10, hexagonal Si4 (3x3x2) 1.2e-5 -> 6.9e-10, fcc Au2 in a tetragonal cell (orthogonal grid, 8 operations) 1.6e-11 -> 9.2e-12; energies unchanged (the lists equal the meshes within 4e-11 Ry), times unchanged | `off`: no average, as before (bit for bit the forces of 2026-11-21) |
| `PARSEC_KPOINT_GENERALIZED_RITZ` | `auto` (default, since 2026-11-17): the Rayleigh-Ritz steps of the k-point pools on the block route (the SUBSPACE steps and the cycles after the first of a first solve) take the filtered block as it is: the overlap `S = X^H X` and `X^H H X` are formed in the same pass, `S` equilibrated (`D S D`), factored by Cholesky and audited (`max |R^-H S R^-1 - I| <= 1e-10`), and the whitened problem solved (`Eigensolvers/chebff_columns.py`, `_whitened_ritz`); a block that fails the audit falls back to the orthonormalization of before for that step (counted as `generalized_ritz_fallbacks`). The residuals of the states are summed by one reduction kernel on the device (`_residual_squares`; the host arithmetic unchanged). On 16 A100, diamond with the 2 x 2 x 2 mesh: 10 x 10 x 10 cubes 360.8 -> 300.2 s, 9 x 9 x 9 cubes 476.1 -> 369.4 s, the same total energies to the digits printed | `off`: the shifted Cholesky QR three times before every Rayleigh-Ritz step, as before |
| `PARSEC_PERIODIC_SETUP_DEVICE` | `auto` (since 2026-11-16): the local ionic potential of a bulk cell of the default routines with diagonal lattice vectors on the device where the run has one: the structure-factor series as the same blocked product and FFT, the short-ranged terms gathered per grid point from the images in the neighbouring bins (`V_ion/periodic_local_device.py`; the host sums to round-off) | `off`: the host sums, bit for bit |
| `PARSEC_PERIODIC_DEFER_LAPLACIAN` | `auto` (default, since 2026-11-18): the accelerated preparation of an orthorhombic bulk box holds a descriptor of its finite-difference matrix (`acceleration/Laplacian/deferred.py`); the sector stencils and the FFT Hartree symbol come from the grid, any other reader builds the same matrix (diamond 20^3 on 16 A100: 141.0 -> 132.7 s with the threaded grid maps below, the setup before the SCF 35.2 -> 26.9 s, host memory per rank 30.6-31.4 -> 8.1-10.7 GiB, the same energies) | `off`: the matrix built in the preparation, as before |
| `PARSEC_PERIODIC_SECTOR_FORCES` | `auto` (default, since 2026-11-18): the single-point forces (`Skip_Force: .false.`) and the `bfgs`/`simple` relaxations (with `Relax_Restart`, `Potential_Extrapolation`, `Relaxation_SCF_Upscale`) of a bulk cell at the Gamma point on the default routines keep the sectors of its point group, on the command line and in the full-SCF runner (`cli.py`, `periodic_sector_forces_refusal`): the forces are averaged over the full group G of the sectors, translations included, the ionic loop averages forces and moved positions over G, and `--symmetry on` is accepted for them; such runs keep the sectors below the size floor of `PARSEC_PERIODIC_SECTOR_FLOOR` too (since 2026-11-19, `prepare_single_point(sector_forces=True)`); molecular dynamics and manual movements (on the command line under `PARSEC_MOVING_SECTORS=auto` they keep the sectors of the group their structures keep, checked at every structure; row below `PARSEC_MOVE_FLAG_SUBGROUP`), a `Move_Flag` fixing atoms under `PARSEC_MOVE_FLAG_SUBGROUP=off` (under `auto` the subgroup that keeps the fixed atoms), point charges, `Polarizability`, nscf, `Restart_From_Wfndat`, k-points beyond Gamma and the TRIM supercell keep the full grid (the runner: one sector of the identity), and so do the Fortran-derived routines under `PARSEC_FORTRAN_SECTOR_FORCES=off` (row below); a band structure after them keeps the sectors (`PARSEC_PERIODIC_SECTOR_BANDS`). Diamond on 16 A100, one relaxation step of perfect cubes: 20^3 (64,000 atoms) 164.7 s, 22^3 (85,184) 304.1 s | `off`: every forces, relaxation or MD run of a cell on the full grid (the runner: one sector of the identity), bit for bit as before |
| `PARSEC_FORTRAN_SECTOR_FORCES` | `auto` (default, since 2026-11-19): the single-point forces and the `bfgs`/`simple` relaxations (with `Relax_Restart`, and on the command line with `Restart_Run`) of a bulk cell at the Gamma point on the Fortran-derived routines keep the sectors of its point group, as `PARSEC_PERIODIC_SECTOR_FORCES` does for the default routines, on the command line and in the full-SCF runner (which reads no `parsec.dat` and so leaves a `Restart_Run` of these routines to `main.py`, as it does their spin-polarized cells, and names the reason). Their relaxations with fixed atoms (`Move_Flag`) keep the sectors of the subgroup that maps the fixed atoms onto fixed atoms on the command line under `PARSEC_MOVING_SECTORS=auto` where the cell's basis reaches the floor of `PARSEC_PERIODIC_SECTOR_FLOOR` and those sectors have 4 operations or more, in one process (the full grid otherwise and under `off`; the full-SCF runner takes them in the sectors of that subgroup whatever their number since 2026-11-22, `PARSEC_RUNNER_FORTRAN_FIXED_SECTORS`, refused there before), and the forces of a TS cell, whose SCF may take the sectors under `PARSEC_GAMMA_TS_SECTORS`, keep the full grid (not validated in sectors); those of an LDA+U cell keep the sectors since 2026-11-21 under `PARSEC_LDAU_SECTOR_FORCES=auto`. The driver's preparation refuses a value the switch does not know before the SCF. Those routines are chosen by `Use_Fortran_Routines: .true.` or by the parser (`Input/routines.py`: Gamma forces with `Cubic_Spline` species, `Pseudopotential_Format: fhipp`, a core correction, `Restart_Run`, a `Mixing_Method` other than Anderson, another `Eigensolver`, `Double_Grid_Order` > 1, `Dynamic_Diag_Tol`), and their single points already ran in sectors. Their own force terms (`Forces/periodic_device.compute_device_periodic_forces`): the Ewald sum; the local ionic, self-consistency (`V_out - V_in`) and core-correction terms of their tables (splines or linear, the double grid) on the fields the sectors expand to the full grid, summed for one atom per orbit of the group of the sectors and turned onto the others (`PARSEC_PERIODIC_SECTOR_LOCAL_FORCES`, `PARSEC_PERIODIC_LOCAL_ORBITS`); the nonlocal term contracted in the sectors with the gradients of their projectors with lattice images, of the representative atoms (`PARSEC_SECTOR_GRADIENT_ATOMS`), completed over G; the total averaged over the group the operations of those routines (`periodic_force_operations`) and of the sectors generate (`Forces/sector_symmetry.fortran_force_operations`: the routines' own where they hold the sectors', else the sectors' or the product group), and so are the moved positions; `driver.move_atoms` checks the moved atoms against the sectors and rebuilds the fields of those routines. Molecular dynamics and manual movements keep the sectors of the group their structures keep on the command line under `PARSEC_MOVING_SECTORS=auto`, as on the default routines (the full grid under `off` and in the runner). Displaced Si8 (Td, weighted sectors; Abelian C2v), `Use_Fortran_Routines`, `Cubic_Spline`, `Restart_Run` (single point and a BFGS relaxation), Broyden mixing, `Double_Grid_Order: 2`, MgO with FHI pseudopotentials and the core correction of Mg, two BFGS moves: the forces of the full grid within 2e-6 Ry/bohr and its moves within 1e-6 bohr (A100, `acceleration/tests/test_fortran_sector_forces.py`); the full-SCF runner on 4 ranks gives the moves of `main.py` within 1e-14 bohr. Diamond cubes of `make_diamond.py` with `Skip_Force: false` (`Cubic_Spline`, so these routines), `main.py` on one node of 4 A100, first runs (each with an empty CuPy and CUDA kernel cache, both orders) and a warm rerun: 4^3 (512 atoms) 34.1 -> 23.6 s on a first run, 27.0 -> 14.4 s warm (SCF 15.1 -> 3.2 s, forces 9.1 -> 1.9 s, the setup before the SCF 2.8 -> 9.2 s; 15.6 GiB on one GPU -> 0.59 GiB on the fullest of four), at the criterion of 2e-4 Ry the energies within 1e-7 Ry and the forces, which vanish by symmetry, within 3.2e-4 Ry/bohr (SCF noise); 6^3 (1,728 atoms) out of memory on the full grid of one A100, in sectors 37.2 s on a first run and 18.9 s warm, 1.3 GiB; 10^3 (8,000 atoms) one relaxation step 88.0 s on a first run (setup 38.9 s, SCF 42.6 s, forces 4.8 s) and 69.2 s warm (setup 37.0 s, SCF 26.4 s, forces 4.3 s), 5.6 GiB; 4^3 with the origin atom moved 0.02 bohr along [111], SCF to 1e-8 Ry: the forces of the full grid within 4.9e-8 Ry/bohr. The point operations of these routines (`Forces/periodic.periodic_symmetry_operations`, also on the full grid) are matched with a tree of the atoms since then, the same operations as the loop over all pairs: 2.5 -> 0.09 s at 1,728 atoms, the forces of 8,000 atoms in sectors 46.7 -> 5.3 s. Si64 PBE0 with the FHI pseudopotential and a displaced atom (`examples/3d_hybrid_functionals`, 54^3 points, Abelian sectors of 2): forces within 4.2e-8 Ry/bohr, 42.8 -> 40.1 s on a first run and 14.7 -> 11.0 s warm; one BFGS step 57.6 -> 46.7 s on a first run and 29.3 -> 16.9 s warm (on an empty cache both routes compile kernels for about 16 s before the SCF and more during it; an earlier pairing of a full-grid first run with a warm sector run, 29.4 -> 16.8 s, overstated the gain). An atom within round-off of a grid point (half the atoms of these cubes, 4e-12 to 2e-11 bohr from one) got the slope of the spline local potential at its cone tip in its local force, on either route, up to 2026-11-20: moving it by 1e-11 bohr changed that force by up to 1.5e-5 Ry/bohr (`PARSEC_SPLINE_FORCE_CENTER`, the next row, leaves such points out since 2026-11-21); the sums of one atom per orbit turn the representative's onto its orbit (`PARSEC_PERIODIC_SECTOR_LOCAL_FORCES=all`: every atom's, averaged) | `off`: the forces, relaxations and restarts of those routines on the full grid of one GPU (the runner refuses them), bit for bit as before |
| `PARSEC_SPLINE_FORCE_CENTER` | `auto` (default, since 2026-11-22): the slopes of the local potential and of the core density in the local forces at a grid point closer to its atom than `c = 2 rs(2)`, twice the first radial sample of the table, are those of the smooth behaviour of the table, `s(c) d/c`, with `s` the slope the route itself takes at `c` (the derivative of the spline or of the linear interpolant, or the interpolated slope table of `pseudo.f90`), so that `s/d` is the constant `s(c)/c` there and joins the route's slope at `c` (`Pseudopotential/radial_derivatives.py`, `spline_force_center_setting`, `smooth_center_radius`). Every route: clusters (`Forces/local.py`, `acceleration/Forces/local_device.py`), wires and slabs (`Forces/wire_slab.py`, which no longer leave the points within `rs(2)` out of the table terms), the default routines of a cell (`Forces/periodic_local.py`, `acceleration/Forces/periodic_local.py` and its fused kernel) and the Fortran-derived ones (`Forces/periodic.py`, `acceleration/Forces/periodic_spline.py`, both kernels of `acceleration/Forces/periodic_device.py`); farther points keep the slope of their route bit for bit. So the tables sampled down to the origin (POTRE, `rs(2)` 4e-7 to 8e-5 bohr) keep their own slope beyond `2 rs(2)`, and a table whose first sample lies at or beyond 3e-3 bohr (the H and He FHI tables, 6.25e-3 bohr) keeps `tip`: steps of 1e-3 and 2e-3 bohr resolve its padded interval, and at fixed density the rule was up to 3.4e-5 Ry/bohr off the central differences of its local energy against 2.3e-5 for the spline slope of a cell (the skip of the clusters 4.3e-5 at 6e-3 bohr; CPU). Energies unchanged. Below its first sample a table has no data: the padded spline continues the first value to the origin with a slope of its own at `r = 0+` (carbon FHI 0.42 Ry/bohr, first sample 1.04e-3 bohr; carbon POTRE 3.6e-3, 1.3e-5 bohr) and relaxes to the table over the first intervals, and a step of 1e-3 bohr or more averages that interval out where it is narrower than the step (the FHI tables, 4.5e-4 to 1.25e-3 bohr) and follows `V(0) + a r^2`, whose slope `2 a d` neither the spline slope nor the skip of `forcloc.f90` gives. The radius, at fixed density (CPU): the local term of an atom 1e-11 to 3e-3 bohr from a grid point against the central differences of the local energy at steps of 1e-3 and 2e-3 bohr, C, O, N, B, Si, Mg FHI and C, Ge, Au POTRE tables, clusters and the Fortran-derived routines of a cell, 234 positions: the worst error of the two steps summed 5.69e-3 Ry/bohr against 9.59e-2 under `tip`, the error against their extrapolation to zero step `(4 F(1e-3) - F(2e-3))/3` 4.22e-3 against 9.10e-2; for carbon FHI the local term is the central difference over 2e-3 bohr within 2.7e-6 below 1e-3 bohr from the point (`tip` up to 1.9e-3). The wider `max(2 rs(2), 1.5e-3 bohr)` of the first form of the rule gave 5.50e-3 and 4.32e-3: it sits between the two steps but departs from the own slope of the tables sampled to the origin (carbon POTRE 5e-4 bohr from a grid point 1.6e-5 off the extrapolation against 3.0e-6), and the carbon POTRE cube 5e-4 bohr from a grid point was 1.57e-5 / 2.65e-6 Ry/bohr off the slope of the SCF energy (steps 1e-3 / 2e-3 bohr) with it against 5.25e-6 / 1.31e-5 with its own slope, which it has again. A100, force of an atom 1e-6, 1e-4, 5e-4, 1e-3 and 1.5e-3 bohr from a grid point against the slope of the SCF free energy at steps of 1e-3 / 2e-3 bohr, worst of x and z, `tip` -> `auto`: carbon FHI cube (Fortran-derived routines) 3.7e-4/1.8e-4 -> 1.9e-4/1.3e-6, 1.1e-3/8.9e-4 -> 2.0e-4/2.3e-5, 7.8e-4/8.6e-4 -> 2.1e-4/1.2e-4, 2.5e-4/1.4e-4 -> 3.5e-4/2.4e-4 (worse, just inside `rs(2)` = 1.04e-3 bohr), 1.2e-4/2.4e-5 -> 1.0e-4/7.2e-6; CH4 cluster with carbon FHI 8.1e-5/7.1e-7 -> 8.1e-5/2.9e-7, 1.2e-4/4.3e-5 -> 8.3e-5/6.3e-6, 1.7e-4/2.1e-4 -> 7.5e-5/3.3e-5, 3.5e-4/4.3e-4 -> 1.4e-4/6.6e-5, 9.5e-5/1.1e-5 -> 9.0e-5/5.2e-6; carbon FHI wire 1.5e-3/5.7e-6 -> 1.5e-3/6.0e-6, 1.4e-3/2.1e-5 -> 1.4e-3/1.5e-5, 3.0e-4/1.3e-4 -> 4.8e-4/5.3e-5 (worse at the step of 1e-3), 4.2e-4/2.6e-4 -> 2.8e-4/1.0e-4, 2.9e-4/4.8e-5 -> 3.0e-4/4.4e-5; B of the hBN slab (FHI) 9.5e-5/7.3e-8 -> 9.5e-5/7.2e-8, 9.5e-5/9.0e-7 -> 9.5e-5/1.2e-6, 9.3e-5/4.7e-6 -> 9.4e-5/6.0e-6, 5.7e-5/1.5e-6 -> 5.8e-5/3.9e-6, 8.0e-6/7.0e-6 -> 8.7e-6/7.7e-6 (worse by up to 2.3e-6 at the step of 2e-3; the step of 1e-3 next to the point shows the nonlocal feature below); silicon FHI cube 2.4e-5/2.3e-5 -> 8.9e-8/4.4e-8, 3.2e-5/3.1e-5 -> 3.6e-6/3.1e-6, 4.9e-6/2.6e-6 -> 1.2e-5/1.0e-5 (worse, below), the bits of `tip` from 1e-3 bohr on (`c` = 8.9e-4 bohr); oxygen of MgO (FHI) 1.7e-6/5.5e-6 -> 2.7e-6/1.5e-6, 2.1e-3/2.0e-3 -> 5.2e-5/4.6e-5, 1.5e-3/1.6e-3 -> 2.9e-4/2.4e-4, 5.5e-4/8.0e-5 -> 4.9e-4/1.9e-5, 3.9e-5/4.6e-5 -> 4.4e-5/4.1e-5; magnesium of MgO (FHI with its core) within 4e-7 either way; the POTRE cases (carbon cubes on both routine sets, CH4, the carbon POTRE wire, Ge8 with its core on both routine sets, Ag3, fcc Ag4) keep the bits of `tip` from 1e-4 bohr on, and at 1e-6 bohr their errors change by at most 1.6e-6 (carbon cube on the default routines 7.3e-6/6.9e-6 -> 8.9e-6/8.5e-6, Ge8 on the Fortran-derived ones 1.5e-6 -> 6.4e-7). What remains next to a grid point is not the local slope, and is open: (1) the gap that both steps show lies in the projector values near the origin (nonlocal): with the local potential replaced by its smooth form within `c` in the energy itself (CPU), the carbon FHI cube 5e-4 bohr from a grid point keeps 1.16e-4 / 1.20e-4 Ry/bohr (constant in the step, linear in the distance: 2.5e-4 / 2.4e-4 at 1e-3), and the step-1e-3 feature of the carbon wire with an atom on a grid point (1.5e-3 Ry/bohr in x) is unchanged by that smoothing but moves by 1.9e-4 with the projector values of `PARSEC_PROJECTOR_CENTER=off`; for the carbon POTRE table the projector values replaced by their smooth form within 3e-3 bohr took the gap 1e-4 bohr from a grid point from 7.6e-5 to 2.3e-6; the orbitals respond to such features at first order, so no force of the structure has the slope of the energy over steps that leave the point; (2) the difference between the two steps next to the point is the padded interval of the local table, which a step of 1e-3 bohr resolves (8.6e-5 at 5e-4 bohr on the carbon FHI cube, 3.8e-6 with the smoothing above); (3) the silicon FHI cube 5e-4 bohr from a grid point is worse than under `tip`: its local term at fixed density is within 2.8e-6 of the central differences under `auto` and 1.1e-5 to 1.3e-5 off under `tip`, whose error had cancelled a residual of about -1.1e-5 outside the local term; (4) the oxygen FHI table keeps 2.9e-4 / 2.4e-4 at 5e-4 bohr (not analysed; at fixed density the central differences of its local energy alone differ between the two steps by up to 1.5e-4); (5) the wider radius followed the energy of the Ge POTRE table of Ge8 1e-4 bohr from a grid point (8.2e-6 -> 9e-8, now 8.2e-6 again) and, at fixed density, of the PBE Au table of the pseudopotential library (2.7e-5 -> 1e-8 Ry/bohr against the extrapolation 1e-4 bohr from a grid point, both routes), whose `V'(r)/r` bends within 1e-4 bohr of the origin, but departed from the light POTRE tables; a radius per table is not chosen. For carbon, steps of 1e-3 and 2e-3 bohr differ by up to 2.5e-5 Ry/bohr also away from the rule, since `V'(r)/r` changes from -813 to -796 Ry/bohr^2 between 1e-3 and 2e-3 bohr. Time and memory unchanged (one more slope per species). | `tip`: the forces of 2026-11-21 on every route (bit for bit, CPU and A100): the Fortran-derived routines leave the spline slopes within `min(rs(2), 1e-6 bohr)` out and keep them farther, the default routines, clusters, wires and slabs leave the points closer than `rs(2)` out (`forcloc.f90`). The padded spline has a slope of its own at `r = 0+`, so the energy has a cone at every atom (in PARSEC 1.5's `ionpot.f90` too); tip slope and first radial sample `rs(2)` per table: carbon POTRE 3.6e-3 Ry/bohr (1.3e-5 bohr), silicon POTRE 5.5e-6 (2.2e-6), FHI Si 0.035 (4.5e-4), Mg 3.4e-3 (5.2e-4), O 0.73 (7.8e-4), C 0.42 (1.04e-3); before, a round-off point took it along the direction of its round-off. Measured: CPU (`tests/unit/test_spline_force_center.py`), carbon POTRE: an atom 1e-11 bohr off a grid point has the force of the atom on it within 1.6e-11 and the central difference of the energy within 1.5e-10 (`off`: 3.6e-5 off), an atom 5e-6 bohr off (between 1e-6 bohr and `rs(2)`) the bits of `off`; carbon FHI on the cube of `make_diamond.py` (h a/18), local term of a round-off atom against the central difference of the local energy 5e-10 (`off` 2.1e-3). A100, the CLI, SCF to 1e-8 Ry, `make_diamond.py` cubes with the origin atom moved 0.02 bohr along [111] (half the atoms 1.7e-11 to 4.4e-11 bohr from grid points), carbon POTRE: 8 atoms, the forces change by up to 3.6e-6 Ry/bohr, 64 atoms 6.8e-6, alike on the sectors and the full grid; carbon FHI (with `PARSEC_PROJECTOR_CENTER=auto`) 2.7e-4; an Si8 POTRE cube with atoms on grid points 1.2e-8; MgO with FHI tables and the core correction (no atom near a grid point) bit for bit; energies identical, run times unchanged (4-10 s a run). `fortran`: `forcloc.f90`'s rule (`if (r1 < rmin) cycle`, `rmin = rs(2)`) on the Fortran-derived routines too; `off`: the spline slopes at every positive distance on the Fortran-derived routines, as before 2026-11-21 |
| `PARSEC_CORE_FORCE_XC` | `auto` (default): the core-correction force of a 3-D cell takes the exchange-correlation potential with the average PARSEC's `potshift` removed added back (`Zero_Average_Vxc`, the default of a 3-D cell without spin; `Forces/periodic.py`, `core_force_xc_average`: one evaluation of the functional without the shift on the density of the result, on every route that reads the potential for that term: both routine sets, Gamma, k-points, sectors, the device and the MPI runner, collinear spin and spin-orbit cells; a spin-orbit result of collinear spin (an explicit `Zero_Average_Vxc`) takes the channel averages from the output densities it keeps for `parsec.dat`, since 2026-11-22, before which it kept its potential (Ge8 with collinear spin and spin-orbit coupling, atom 0.02 and 0.05 bohr off its site, A100, both routine sets: 1.4e-3 and 3.4e-3 Ry/bohr off the slope of the SCF energy -> 1.6e-5 and 6.3e-6 at step 1e-3, 2.9e-5 and 1.5e-5 at 2e-3, the accuracy of the spin-orbit term; energies unchanged); a non-collinear result keeps its potential; on the host 0.3 s for 885k points of PBE on a laptop, and only where a species has a nonzero core density: a cell without a core charge pays nothing). With the average `c` removed the term `dV sum_r V_xc d rho_c(r - R)/dR` gains `c dQ_c/dR`, where `Q_c` is the grid sum of the core density, which changes as an atom moves against the grid; PARSEC 1.5 forms the term in G-space (`forpbc.f90`), where the average drops out. Measured on A100, force against the slope of the SCF free energy (central step 1e-3 bohr), both routine sets alike: atom 6 of Ge8 (Ge POTRE with core, CA, Gamma) 0.01 to 0.1 bohr off its site 2.8e-4 to 6.1e-3 Ry/bohr (`shifted`, 5 to 7 % of the force) -> 2.4e-6 or less; with a 2x2x2 mesh 1.4e-3 and 6.2e-3 -> 2.3e-7 and 2.4e-6 (0.02 and 0.1 bohr); with collinear spin and an explicit `Zero_Average_Vxc` 1.4e-3 and 6.1e-3 -> 2.2e-7 and 2.4e-6; with self-consistent spin-orbit coupling 1.4e-3 and 6.2e-3 -> 1.6e-5 and 7.5e-6 (1.1e-5 after a Richardson step, the accuracy of the spin-orbit term); atom 0 of MgO8 (Mg FHI with `Nonlinear_Core_Correction`, PBE) 1.5e-6 to 1.3e-5 -> 1.2e-7. The Gamma sectors give the forces without them within 4e-11 and the MPI runner on 4 ranks those of one process within 5e-11. No tracked example computes forces of a 3-D cell with a core density. A meta-GGA keeps its own `dE_xc/dn_core` (not shifted), a wire or slab has no `potshift`. Energies unchanged | `shifted`: the potential as the loop kept it, bit for bit as before |
| `PARSEC_WIRE_SLAB_GAUSSIAN` | `auto` (default): the two terms of the Gaussian compensation charge in the local forces of a wire or slab (the force of the Gaussian density in the Hartree field and the slope of `2 Z erf(r/sigma)/r`) at every grid point; the terms of the radial tables take the slopes of `PARSEC_SPLINE_FORCE_CENTER` (since 2026-11-22 the smooth slopes within `2 rs(2)`, under `tip` `forcloc.f90`'s skip of points within the first radial sample `rs(2)`; `Forces/wire_slab.py`). This departs from `forcloc.f90`, which skips every term of such a point, although the Gaussian density is largest there. Measured on A100, force of an atom on a grid point against the slope of the SCF energy (central step 2e-3 bohr): hBN slab (FHI B and N) z 2.1e-3 -> 7.2e-5 Ry/bohr, x 2.0e-4 -> 7.8e-6; carbon chain wire (FHI C) x 7.2e-3 -> 7.7e-6, z 2.9e-3 -> 5.8e-8; energies unchanged. The 7.5e-5 left in z of the slab, also 2e-3 bohr off the point, was the error of the solve of `V_G` (`PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE`, below). The radial-table terms that kept the skip left 4.2e-4 / 2.6e-4 Ry/bohr (steps 1e-3 / 2e-3 bohr, worst of x and z) on the wire 1e-3 bohr off the point, just inside `rs(2)` = 1.04e-3 bohr of the C table (2.8e-4 / 1.0e-4 with the smooth slopes) | `off`: the skip of `forcloc.f90` for every term, bit for bit as before |
| `PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE` | `auto` (default, since 2026-11-22): the potential `V_G` of the Gaussian charges of a wire or slab, solved once per structure from `n_g` (`ionpot_wire.f90`/`ionpot_slab.f90`'s `hpotcg`), is solved to a relative residual of 1e-12 (absolute at most 1e-14, at most 4 times the iterations of the run; `SCF/wire_slab.py`, `gaussian_solve_setting`), a deliberate departure from PARSEC 1.5. The error of that solve enters the energy at first order (`-dV sum rho V_G`) and moves with the atoms (the start `n_g` moves with them), so the energy over steps of 1e-3 bohr carries its slope. At the Hartree tolerance of the run (relative 1e-7) `-dV rho . V_G` lies 6.8e-6 Ry (hBN slab, FHI B with core and N, h 0.22) and 1.35e-7 Ry (carbon FHI wire) from a solve to 1e-14, at 1e-12 within 2.2e-11 and 7e-13 (CPU, fixed density). With the Hartree operator of the wire or slab symmetric (9.4e-11 relative on that hBN slab; not where the density reaches the edge of the domain, `PARSEC_HARTREE_VARIATIONAL`) and the tight solve, the force of the Gaussian charges is the slope of that energy within 1.3e-7 Ry/bohr. A100, force against the slope of the SCF energy (step 2e-3 bohr): B of the hBN slab on a grid point, 5e-4 and 2e-3 bohr from it, z 7.2e-5, 7.5e-5, 7.6e-5 -> 4.7e-7, 5.8e-6, 1.0e-5 Ry/bohr; carbon wires unchanged within 1e-6 (the POTRE wire 1.4e-5 -> 7.7e-6 next to the point). Energies move toward the converged value: hBN slab +8.5e-6 Ry, carbon wires +2e-7 Ry; the conjugate gradients take 253 against 193 iterations (hBN) and 316 against 202 (wire) once per structure, run times equal (22 to 28 s a run). The five examples of `examples/3d_hybrid_functionals/hBN_k661` (`pbe`, `pbe0`, `hse06`, `pbe0_periodic`, `hse06_periodic`, all slabs) change their total energies by the error of the former solve, and their `parsec_python.out` hold the new values (A100): `pbe` (h 0.15, 6x6x1 list) -25.80787466 -> -25.80786051 Ry, `pbe0` -25.79727632 -> -25.79726206, `pbe0_periodic` -25.79727466 -> -25.79726040, `hse06` -25.79666683 -> -25.79665257, `hse06_periodic` -25.79666674 -> -25.79665249 Ry (each +1.4e-5); the setup of the local potential of `pbe` 1.55 -> 1.64 s of a 220 to 224 s run | `off`: the Hartree settings of the run, as before (bit for bit) |
| `PARSEC_WIRE_SLAB_GAUSSIAN_FORCE` | `auto` (default, since 2026-11-22): the force on the Gaussian charge of an atom of a wire or slab in the Hartree field of the electrons is `dV sum_r V_H(r) dn_g(r - R)/dR` with the analytic slope of the Gaussian, `dn_g/dR = n_g 2 (r - R)/sigma^2` (`Forces/wire_slab.py`, `wire_slab_gaussian_force_setting`): the derivative of `-dV rho . V_G` where the Hartree operator `G` of the wire or slab is symmetric (`G^T rho` in place of `V_H` otherwise, which a slab under `PARSEC_HARTREE_VARIATIONAL=on` takes), a departure from `forcloc_slab.f90`, which takes the finite-difference gradient of `V_H` (no gradient of `V_H` is formed now). At fixed density (CPU, tight solves) the analytic form is the central difference of that energy within 1.5e-7 to 7.7e-7 Ry/bohr on hydrogen slabs and wires at h 0.2 to 0.6, where the gradient form is off by 6e-7, 8.9e-6, 2.5e-4 and 2.2e-3 Ry/bohr at h 0.2, 0.3, 0.4 and 0.6 (the stencil does not resolve the Gaussian, sigma = rcore/2); the forces of a hydrogen slab at h 0.5 to 0.2 are closer to those of h 0.15 with it (4.91e-3 against 5.12e-3 Ry/bohr at h 0.4). A100, force of an atom 0.03 bohr off its grid point against the slope of the SCF energy (Richardson step over 1e-3 and 2e-3 bohr): hBN slab at h 0.22 and 0.4 5.8e-7 and 7.9e-5 -> 6.7e-9 and 6.0e-8 Ry/bohr, carbon wire at h 0.24 and 0.4 2.2e-6 and 1.9e-4 -> 1.8e-7 and 3.6e-8; energies unchanged | `gradient`: the finite-difference gradient form, as before (bit for bit); `PARSEC_WIRE_SLAB_GAUSSIAN=off` keeps it too |
| `PARSEC_PROJECTOR_CENTER` | `auto` (default): the projector **values** within the first radial sample `rs(2)` of an atom divide the harmonics by `rs(2)`, the rule of `nonloc.F90` (`rnorm = max(|d|, rrmin)` in the radial interpolation, `ylm_real` and `ylm_cplx`): `V_ion/ionic_potential.py` (`build_nonlocal_projectors`, `projector_center_setting`, `projector_center_radius`), the fast periodic builder (`V_ion/periodic_projectors.py`), the native cluster builder (the points within `rs(2)` evaluated again in `V_ion/native_ionic.py`; the C++ kernel is unchanged), the two double-grid builders and the spin-orbit projectors. A channel `l` is then `w(rs(2)) Y_l(d/rs(2))` inside its first sample: continuous, zero at the atom for `l >= 1`. The **gradients** deliberately depart from PARSEC 1.5 next to an atom: within 3e-3 bohr (`PROJECTOR_GRADIENT_FLOOR`, `projector_gradient_distance`) the radial function, its slope and the direction are taken at that distance, on the default and Fortran-derived routines of a cell, clusters, wires, slabs, the double grid, the spin-orbit and the LDA+U projectors (`Forces/hubbard_force.py`, but not the smooth LDA+U projectors of `PARSEC_LDAU_SMOOTH_PROJECTORS=on`, whose gradients are their exact derivative at every distance; not the `tabulated` slopes of `PARSEC_PERIODIC_FORCE_SLOPES`, which reproduce Fortran's forces). This is the gradient of the `c r^l` the table follows. A table whose first sample lies at or beyond 3e-3 bohr (no table of the package with projectors: the H and He FHI tables, first samples 6.25e-3 and 3.1e-3 bohr, have none; coarse test tables) keeps the gradient of `tip` (`projector_gradient_rule`): the energy's polynomial `w(rs(2)) Y(d/rs(2))` then covers every realistic step. The tables are not `c r^l` next to the origin: POTRE tables are flat out to 7e-6 to 2.7e-4 bohr (Au `l = 1`: about -6.7e-6 out to 7e-6 bohr, then -0.9045 `r`), and the first eight samples of an FHI table, up to 2.5e-3 bohr, lie 25 to 48 % above `c r` for a p channel. The value `w(rs(2)) dY/du / rs(2)` that `nonloc.F90` forms at the point is therefore the derivative only for displacements within `rs(2)` (4e-7 to 2e-3 bohr): 19 times the slope of a p projector on a POTRE table, 1.48 times on an FHI table. Full-SCF central differences on A100, tight SCF, an atom on a grid point, force against the slope (`auto`; in brackets `tip`), Ry/bohr, at steps of 1e-3 and 2e-3 bohr: C64 POTRE on both routine sets 5.5e-8 and 2.1e-7 (4.1e-3); the same with a 2x2x2 mesh and the four neighbours moved out by 0.05 bohr 1.8e-8 and 6.9e-8 (1.3e-3); C64 FHI at 2e-3 1.8e-7 (1.1e-4); `Si_disp_k222` 2.3e-9 and 9.2e-9 (7.9e-6); Ge8 1.7e-8 and 5.4e-8 (4.2e-4). Au2 cell at 1e-4 bohr, with spin-orbit coupling and its scalar part, on both routine sets: within 1.9e-7 (at 1e-3 and 2e-3 up to 2.7e-5, the truncation of the difference and the kinks of the linearly interpolated spin-orbit table of then, splined since 2026-11-22 by `PARSEC_SOC_PROJECTOR_SPLINE`; the scalar part extrapolates to 1e-7). The two routine sets agree within 1e-9 Ry/bohr, and the energies equal those of `tip` within 2e-11 Ry. Ag4 with U = 4 eV (LDA+U), an atom on a grid point, both routine sets: within 2.3e-7, 4.7e-8 and 4.6e-8 at steps of 1e-4, 1e-3 and 2e-3 bohr (`tip` 3.8e-4, from the KB part); the LDA+U part itself changes by less than the digits printed, since its d tables follow `c r^2` from the first sample on. First runs on A100 cost the same as `tip`: C512 at Gamma 18.1-18.5 s against 18.3-18.4 s (forces 2.53-2.56 s both), the Si28H36 cluster 6.8-7.1 s against 6.8-6.9 s, GPU peak and host high-water within their run-to-run noise. `tip` reproduces the forces of the first form of `auto`: bit for bit on 8 of 13 checked runs, the rest within 8e-14 Ry/bohr (2.7e-12 for spin-orbit cells, whose energies differ by the same SCF noise). A step inside the part of a table that is not `c r^l` (below 1.5e-4 bohr on carbon POTRE, 1.3e-3 on carbon FHI) sees that part: the force then differs from such a chord by up to 6e-4 Ry/bohr (POTRE) and 1.9e-4 (FHI). An Au2 cluster in a sphere of radius 8 bohr stays 3 to 5e-5 Ry/bohr from the slope of its SCF energy at every step with an atom on a grid point (9.7e-5 off the grid, forces near 10 Ry/bohr); each of its force terms is the derivative of its energy at fixed density and orbitals within 1.2e-7. The residual comes from the Hartree multipole boundary of the small domain, and a radius of 11 bohr brings it to about 6e-7: the Poisson operator with multipole boundary values is not symmetric, and the atomic tail of the boundary is rebuilt from the positions with no force term of its own. A relaxation follows the forces it gets: from the displaced 8-atom diamond cube, whose four atoms on grid points have other forces under `auto` than under `tip`, the unbounded first steps of `bfgs` take other paths under the two values (`PARSEC_BFGS_FIRST_STEP`). | `tip`: the same values, the gradients as `nonloc.F90` forms them at the point (`w(rs(2)) dY/du / rs(2)` within `rs(2)`), the forces of `auto` as first published. `off`: `min(rs(2), 1e-6 bohr)` for the scalar projectors and the gradients of the Fortran-derived routines (`1/|d| = 0` on the atom), the true direction for the double-grid and spin-orbit projectors, the 1e-3 bohr floor for the scalar and spin-orbit gradients of Andrew Grieder's routines, bit for bit as before; an atom within round-off of a grid point then got a nonlocal force of 0.1 (carbon POTRE) to 0.75 Ry/bohr (carbon FHI) from that one point |
| `PARSEC_CUBIC_SPLINE_DEFAULT` | `on` (default since 2026-11-22; `Input/parsec_input.py`: `cubic_spline_default_setting`): a species whose input gives no `Cubic_Spline` takes PARSEC's cubic spline of its tables, as with `Cubic_Spline: .true.`: the local potential (clusters, wires, slabs and the Fortran-derived routines of a cell; the default routines of a cell take the spline for it anyway), the core density, the Kleinman-Bylander projectors and, under `PARSEC_SOC_PROJECTOR_SPLINE=auto`, the spin-orbit projectors. The valence densities (`initchrg.f90`) and the radial functions of LDA+U (`upot.F90`) stay linear, as for every species. The parser decides it once per input, so every route reads the same flag (both command lines, the MPI runners, hybrids, restarts, which read the input again; the effective input of `Generate_Complete_Parsec_In` writes `Cubic_Spline` for every species, and `Use_Fortran_Routines .false.` for a cell on Andrew Grieder's routines whose species took the spline by default, so that it reproduces the run, routines included, under either value), and the rules for `Cubic_Spline` species apply to such a species (`PARSEC_PERIODIC_OBLIQUE_SPLIT` splits the potential of its `martins_old`/`martins_wang` tables, `PARSEC_SPLINE_SETUP` builds its core density), except the choice of the routines: the forces, relaxation and molecular dynamics of a bulk cell at the Gamma point, which take the Fortran-derived routines for `Cubic_Spline: .true.` where both grids coincide (`Input/routines.py`, `bulk_default_features`), keep Andrew Grieder's routines for a species that takes the spline by default, as before. His local potential is the spline in any case and the two give the same energy (Si216, C8, Si8 and a spin C7N cell within 2e-11 Ry); his are faster for large cells (Si216 with forces 10.1 against 12.0 s) and keep the full-grid ionic loops of the MPI runner (molecular dynamics, manual moves, fixed atoms), which it refuses on the Fortran-derived routines. A value in the input wins; a cluster with `Old_Interpolation_Format: .true.` (an interpolation of the linear tables) keeps the linear tables of its species without a value. parsec.out names the species the default gave the spline, in one line below the routines line (nothing under `off`). Why: with the linear tables the energy has a kink wherever a grid point crosses a radial sample, and the forces, which differentiate smooth tables (`pseudo.f90`), miss the slope of the energy at steps of 1e-3 bohr by 1e-3 to 0.45 Ry/bohr, in PARSEC 1.5 as here (`PARSEC_LINEAR_TABLE_SLOPES`, `FORCES_AND_RELAXATION.md`). A100, full SCF to 1e-10 Ry, inputs without `Cubic_Spline`: the 33 force inputs of the consistency matrix and 22 other inputs (C, Si, Ge, Ag, Au, Mg, O, N with a core-hole table, Cr, H, B; POTRE, FHI and converted tables; clusters, Gamma cells on both routine sets, 2x2x2 meshes, wires, slabs, spin, spin-orbit, LDA+U) give the energies, forces and SCF steps of the same inputs with `Cubic_Spline: .true.` bit for bit (50 of them, and the spin-orbit cell up to the run-to-run noise of its forces, 1.7e-17 Ry/bohr between two runs of one tree; the four cells with Gamma-point forces and no `Use_Fortran_Routines` keep Andrew Grieder's routines, see below: energies within 1.8e-11 Ry, forces within 1.1e-9 Ry/bohr); their forces follow the slope of `E - TS` at 1e-3 and 2e-3 bohr within 1e-5 Ry/bohr except where the spline itself does not (the CA step of `PARSEC_CA_CONTINUITY`, the near-grid-point regime of FHI tables and of the carbon POTRE table 5e-4 bohr from a grid point, 1.3e-5 at 2e-3 bohr, the LDA+U kink; every one the same in the matrix with `Cubic_Spline: .true.` before this switch; a Ge2 slab 1.05e-5 and a spin-polarized C7N cell 6e-4 at one atom, the same with `Cubic_Spline: .true.` before this switch). Spline minus linear energy per atom (with `PARSEC_PROJECTOR_TAIL=auto`): Ag +1.05e-2 to +1.19e-2 Ry, Au2 +9.5e-3, Cr2 +4.6e-3, Ge +8.4e-4 to +1.31e-3 (core correction), MgO +1.4e-4 to +2.2e-4, Si -5.4e-5 to -2.6e-4, carbon POTRE -2.3e-5 (diamond) to -3.3e-4 (C60), carbon FHI -7.0e-4, O and H -4.1e-4 to -6.0e-4, N2 with a core hole -1.2e-3, and +8.25e-3 for the spin-polarized PBE Fe table of the tests (bcc Fe2, a 5.42 bohr, h 0.22, 100 K, at the Gamma point, in the same SCF state: the cell has at least four, which both interpolations reach; -79.54679865 -> -79.53030710 Ry in the lowest). Against the limit of the same tables sampled up to 8 times more densely, the error of the spline at the file's sampling is 1.9 to 540 times smaller than that of the linear tables on every species measured (d metals: spline +7.3e-4 Ry per atom for Cr to +1.9e-3 for Ag, linear -3.3e-3 to -9.9e-3; the same in PARSEC 1.5; what remains of the spline's is the quadrature of `PARSEC_KB_QUADRATURE`); the forces move by up to 5.5e-2 Ry/bohr (a Ge atom 5e-4 bohr from a grid point, an O atom of MgO8 with FHI tables) and by 0.8 at the atom of the LDA+U case of the matrix (its LDA+U term has a kink at that position), 1e-5 to 3e-3 elsewhere. Molecular dynamics conserves the energy better: spread of `E + K` over 21 steps of 20 a.u. of C60 2.25e-3 -> 9.7e-4 Ry, 16 steps of Si28H36 7.1e-4 -> 3.3e-4 Ry. Cost, first runs on one A100 (the two values launched together, walls in s, `off` -> `on`): clusters and the default routines unchanged within 5 % where the SCF takes as many steps (C60 6.8 / 6.7, Si28H36 8.3 / 8.4, Au2 4.1 / 4.2, MgO8 3.7 / 3.9), but the SCF can need more: bcc Fe2 above 75 -> 199 steps, which depends on the path of the SCF and not on the spline (the run ends in another of its states, and with tables 8 times denser the linear run takes 395 steps and the spline 115), the AuH cluster with spin-orbit coupling in Kramers sectors 29 -> 49 (39 against 31 on the whole grid, +6.8e-4 Ry), a Ge2 slab 77 -> 92; a core correction costs the spline of its core density (Ge64 on the default routines 8.4 -> 9.1, core density 0.002 -> 0.8 s; on the Fortran-derived routines 8.2 -> 9.2; a Ge8 slab 6.6 -> 7.9, core density 0.01 -> 1.06 s); a Ge2 slab 6.5 -> 7.7 (77 -> 92 SCF steps); the Gamma-point forces of cells on Andrew Grieder's routines within 3 % (Si216 9.8 / 9.8 -> 9.9 / 10.0, C8 3.6 / 3.5 -> 3.4 / 3.4, Si8 3.4 / 3.5 -> 3.5 / 3.6, a spin C7N cell 4.3 / 4.6 -> 4.5 / 4.4). The spline of a table is built once per table and the local forces of `Cubic_Spline` species on the Fortran-derived routines run on threads, the same bits (forces of Ge64 2.09 -> 0.71 s, Ge8 slab 1.53 -> 0.31 s, Si216 with `Cubic_Spline: .true.` 0.81 -> 0.32 s). | `off`: the linear tables of PARSEC 1.5 for such a species, bit for bit as before (energies, forces and SCF steps of the 55 inputs, the spin-orbit cell's forces up to their run-to-run noise; the MPI runner on two ranks and `--mpi` of a k-point mesh), and no line in parsec.out |
| `PARSEC_PROJECTOR_TAIL` | `auto` (default since 2026-11-22; `off` before). Under `off` the projectors of a `MARTINS_NEW` table hold the tabulated `V_l - V_local` at every radius (`pseudo.f90`), and the support sphere of `nonloc.F90` ends at `rcore`, the first radial sample beyond the largest channel cutoff `rc` of the file's pseudo-wavefunction headers. About half of the distinct tables of the package that have projectors (943 of 1921, counted over every choice of the local channel), nearly all converted from UPF files (the carbon table of `examples/3d_diamond_benchmark`, the Fe table of the tests, the PBE transition metals, the core-level tables of `examples/0d_xps_core_levels`), hold a nonzero value there, above 1e-5 of the projector's maximum for 895 of them and at most 5.7e-4 (La): the interpolation of `V_l` and `V_local` onto the denser PARSEC grid overshoots at `rc`, where the UPF file holds `V_l - V_local = 0` exactly (checked for C, O, Cr and Br). The nonlocal energy then jumps whenever a grid point crosses `rcore` (3.5e-6 Ry for one point of the carbon table at fixed orbitals, `tests/unit/test_projector_tail.py`), and no force follows the slope of the energy across such a crossing. `auto` (`Pseudopotential/potre.py`: `projector_tail_setting`, `ParsecPseudopotential.potential_difference`): `V_l - V_local = 0` beyond the largest channel cutoff in the projectors, their derivative tables and the spin-orbit projectors of every route, so that every projector is zero at `rcore` and the energy is continuous there. Tables without channel cutoffs (FHI, `MARTINS_OLD`) and tables whose difference is zero there already (O, H, Ag, Ge and Na of the tests, the benzene and nanodiamond carbon) keep their bits. A100, the command line, one atom moved by 1e-3 and 2e-3 bohr, force (before the net force is removed) against the slope of `E - TS`, `off` -> `auto` at 1e-3 bohr: the diamond cell of eight atoms at h 0.37 bohr with that carbon table and `Cubic_Spline`, on the default and the Fortran-derived routines, 7.4e-5 -> 8.9e-6 Ry/bohr (Richardson value of the two steps 8.1e-5 -> 5e-7); Cr2 AFM at h 0.3 with `Cubic_Spline` 6.7e-4 -> 3.2e-6; Si8 with the tests' Si table and `Cubic_Spline` 3.6e-6 -> 1.2e-8; Br2 1.3e-6 under both (no point crosses); O2 bit for bit. Energies move by -2.5e-6 Ry (C8), -7.3e-7 (Si8), +3.6e-6 (Cr2), -1.8e-5 (Br2) and +6.7e-7 (Fe2); the supports, and so the cost, are unchanged; the MPI runner on 4 ranks gives the energy and forces of one process (C8: 1.4e-14 Ry, 7.8e-9 Ry/bohr). The default since 2026-11-22: it moves the energies of a run with such a table away from those of PARSEC 1.5 with the same table by that much (with the cubic spline, per atom: -7.0e-7 Ry for the diamond carbon table, -2.1e-7 for the tests' Si table, +1.4e-6 for Cr2, -4.2e-7 for N2 with the core-hole N table, +8e-9 for H2O; forces by up to 9.9e-5 Ry/bohr; the tables with zero tails, among them Ag, Au, Ge, the benzene carbon and every FHI table, keep their bits), and `off` restores them. With the cubic spline of `PARSEC_CUBIC_SPLINE_DEFAULT`, A100, full SCF to 1e-10 Ry, force against the slope of `E - TS` at 1e-3 and 2e-3 bohr within 1e-5 Ry/bohr (consistency matrix and the cases of that row): the diamond cell with that carbon table at Gamma on both routine sets (atom on a grid point 1.5e-7 / 5.9e-7, atom 0.05 bohr off its site 1.6e-6 / 6.5e-6) and with a 2x2x2 mesh, N2 with the core-hole N table of `examples/0d_xps_core_levels` (2.0e-6 / 8.1e-6, Richardson 6e-8), the spin-polarized Cr2 (7.7e-6 at 1e-3, Richardson 3e-7), Si8 with the tests' Si table; the spin-orbit tables of the repository have no tail. The rule is a property of the file, so every process of an MPI run decides the same. An unknown value is refused when the tables are read, before the SCF. With linear tables (`PARSEC_CUBIC_SPLINE_DEFAULT=off` or `Cubic_Spline: .false.`) the linear interpolation gives kinks of the energy that are larger than this jump (`FORCES_AND_RELAXATION.md`, Forces). | `off`: the tables as PARSEC 1.5 reads them, bit for bit (the default before 2026-11-22) |
| `PARSEC_KB_QUADRATURE` | `trapezoid` (default): the Kleinman-Bylander denominators `integral u_l^2 (V_l - V_local) dr` of the radial tables with the weights `0.5*(r(i+1) - r(i-1))` of `pseudo.f90`. On the logarithmic mesh `r(i) = a*exp(b*i) - c` of a table these weights are the exact Jacobian `b*(r(i) + c)` times `sinh(b)/b = 1 + b^2/6`: every denominator is too large by `b^2/6` (2.6e-5 for the common `b = 0.0125`), every projector too small by `b^2/12`, the nonlocal energy of an occupied channel too small in magnitude by `b^2/6` of itself. With `Cubic_Spline` this is the whole error of a table at its own sampling (what remains under `logmesh` below, at most 3e-7 Ry per atom on clusters and 2.2e-6 on cells, bounds the spline's own interpolation error): A100, command line, `PARSEC_PROJECTOR_TAIL=auto`, against the limit of the same potentials sampled 2, 4 and 8 times more densely (every sample of the file kept), per atom, +1.4e-3 Ry (Au2, fcc Au4, Au2 with spin-orbit coupling), +1.9e-3 (Ag4, fcc Ag4), +1.5e-3 (Fe2, bcc Fe2), +7.5e-4 (Cr2, bcc Cr2), -3.3e-5 (H2O), -5.1e-5 (Si8), -3.2e-5 (C8), -2.6e-6 for Ge8 (its occupied nonlocal channel, s, has a KB integral of only 0.09 Ry, against about 10 Ry for the d shell of Fe); it scales as the square of the sampling and is the same in PARSEC 1.5 (Ag4: spline energy minus that of the eight times denser table +1.89e-3 Ry per atom in both). Linear tables carry it too, below their larger interpolation error. `logmesh` (opt-in; `Pseudopotential/radial_quadrature.py`: `kb_quadrature_setting`, `log_mesh_integral`; `ParsecPseudopotential.kb_denominator`): the trapezoid rule in the mesh index with the exact Jacobian, exact to the digits of the table for these smooth integrands that vanish at both ends, in the projectors, the spin-orbit projectors and the projector slope tables of every route; with `Cubic_Spline` the energies of those runs at the file's own sampling are those of the dense-table limit within 3e-7 Ry per atom on the clusters and 2.2e-6 on the cells. A table whose radii are not the logarithmic mesh of its parameters (radius by radius, to 1e-6 relative; every table of the package but two synthetic six-point test tables passes) is refused under `logmesh`. Not the default: it moves the energy of every run with an occupied nonlocal channel away from PARSEC 1.5 with the same table. An unknown value is refused when the projectors are prepared, before the SCF; the resident set-up key changes only under `logmesh`. | `trapezoid`: the denominators of PARSEC 1.5, bit for bit |
| `PARSEC_SOC_PROJECTOR_SPLINE` | `auto` (default, since 2026-11-22): the radial functions of the spin-orbit projectors (`vionr` and `vsor` of each channel) of a species with `Cubic_Spline: .true.` (or without the label, `PARSEC_CUBIC_SPLINE_DEFAULT=on`) are PARSEC's cubic spline of their tables at `max(r, r_1)`, as `nonloc.F90` evaluates them under `uspline` and as the Kleinman-Bylander projectors of that species already were, and their gradients the derivative of that spline (`Eigensolvers/perturbative_soc.py`, `spin_orbit_radial_functions`: the projectors of both routine sets, isolated and periodic, every k-point, the double grid, and the forces of `Forces/spin_orbit.py` and `Forces/spin_orbit_fortran.py`; the perturbative correction `SO_PSP` uses the same projectors). The energy of linear tables has a kink wherever a grid point crosses a radial knot, so their forces, exact derivatives over steps of 1e-6 bohr, missed the slope of the energy over the steps of a relaxation. One A100, full SCF to 1e-10 Ry (1e-8 for the mesh), force against the central difference of `E - TS` at 1e-3 / 2e-3 bohr, Ry/bohr, `off` -> `auto`: Ge8 cell with spin-orbit coupling at Gamma (atom 0 moved by 0.05 bohr), atom 0, on both routine sets 1.58e-5 / 2.49e-5 -> 1.52e-6 / 6.05e-6, the h^2 truncation of the difference (Richardson 1.28e-5 -> 1.35e-8); atom 6 1.79e-6 / 2.86e-6 -> 1.2e-8 / 3.7e-8; atom 6 at 5e-4 bohr from a grid point (Fortran-derived routines) 6.2e-6 / 2.2e-5 -> 7.0e-7 / 8.1e-7; the Ge8 diamond cube on the Fortran-derived routines with a shifted 2x2x2 mesh at the second structure of a BFGS relaxation, atom 4 (force 0.0834) -2.62e-5 / -3.12e-5 -> 7.1e-8 / 2.9e-7; clusters on the reference solver (CPU: the GPU path has no spin-orbit cluster forces), SCF to 1e-10 Ry, both routine sets alike: Ge5 (the C4v pyramid moved off its sites, h = 0.4 bohr) atom 0 (force 0.0401) 4.1e-7 / 1.7e-6 -> -5.4e-7 / -2.7e-7, both within 6.3e-7 (Richardson); Au2 (h = 0.4 bohr) 4.9e-4 / 2.9e-4 -> 4.9e-4 / 2.5e-4, a kink of that cluster's energy between steps of 1e-3 and 2e-3 that its scalar run without spin-orbit coupling has as well (4.9e-4 at 2e-3, 1.1e-7 at 1e-3), while the spin-orbit term alone with fixed spinors on such a grid follows its energy within 8.1e-8 (linear tables 2.5e-5). Energies move by the difference of the tables: -1.4e-5 Ry for the Ge8 cell, +3.5e-5 Ry for the relaxed cube, +1.8e-5 Ry for Ge5, +1.3e-5 Ry for Au2. The projectors of a structure take 0.15 to 0.3 s more to build (Au20 and Ge35 clusters on a laptop CPU). A species with linear tables (`Cubic_Spline: .false.`, or no value under `PARSEC_CUBIC_SPLINE_DEFAULT=off`) keeps them under both values, as Fortran does; a spin-orbit species without the label (those of `examples/0d_AuH`) takes the spline by default since 2026-11-22; a value it does not know is refused when the input is read | `off`: the linear interpolation of the tables for every species, bit for bit as before |
| `PARSEC_CA_CONTINUITY` | `off` (default): the Perdew-Zunger correlation of `Correlation_Type: CA` (unpolarized, spin-polarized, and under LB94 and Casida-Salahub) with the high-density coefficients of PARSEC 1.5 (`exc.f90`, `corca_nspn` and `corlsd_ca`: `C = 0.0020`, `D = -0.0116` Hartree for the unpolarized gas, `0.0007`, `-0.0048` for the fully polarized one, the digits of Perdew and Zunger's paper). Its two forms then meet at `r_s = 1` (the density `3/(4 pi)` e/bohr^3) with a step of 6.41e-5 Ry in the correlation energy per electron and 5.55e-5 Ry in the potential, so that the total energy steps by `(3/(4 pi)) 6.41e-5 h^3` Ry (6.6e-7 to 1.9e-6 Ry for h = 0.35 to 0.5 bohr; a multiple of it where the points of a symmetry orbit cross together) whenever the density of a grid point crosses that value: between two structures of a relaxation, or two displacements of a finite-difference check. The forces are the slope of the energy between the steps. A100, the command line, `Cubic_Spline`, SCF to 1e-10 Ry, central differences of `E - TS` at 1e-3 and 2e-3 bohr against the force before the net force is removed (`FORCES_AND_RELAXATION.md`, Forces at a fixed structure): the energies of Au2, Au3, Au4, Ag4, a spin-polarized Au3 and a cell of fcc Au equal the integral of the forces within 2e-9 Ry apart from such steps, and a difference whose two points straddle one misses the force by the step over twice the displacement: Au2 (h 0.35) 3.3e-4 and 1.7e-4 Ry/bohr, Au3 (h 0.42) 5.7e-4 and 2.8e-4, Au3 (h 0.5) 4.8e-4 at 2e-3, Ag4 (h 0.45) 6.9e-4 and 3.1e-4, the spin Au3 2.8e-4 at 2e-3, the fcc Au cell 6.0e-4 at 2e-3 (two points at once, both routine sets); fcc Cu4 (h 0.341, an atom 8.7e-2 bohr from its grid point) 3.0e-4 at 1e-3, from steps of +6.07e-7 and -6.07e-7 Ry in `E_xc` at shifts between -1.0e-3 and -0.75e-3 and between +1.5e-3 and +1.75e-3 bohr (a scan at steps of 2.5e-4 bohr), which cancel in the difference at 2e-3 (3.5e-5); under `on` 8.3e-6 and 3.5e-5 on both routine sets, Richardson value 4.8e-7; with the tabulated projector tail of its table (`PARSEC_PROJECTOR_TAIL=off`, both routine sets, where the scan was made) 3.2e-4 and 2.3e-5, under `on` 1.45e-5 and 2.3e-5 | `on` (opt-in): `C` and `D` from the continuity of the correlation energy per electron and of its slope at `r_s = 1` (the values of libxc's `LDA_C_PZ_MOD`, `V_xc/ca_lda.py` `ca_high_density_coefficients`), in the NumPy functionals, the CUDA kernels (`V_xc/device_xc_kernels.cu`) and the C++ evaluator of the clusters, whose loops then gain the change of the coefficients at the points of `r_s < 1` (`V_xc/native_ca_lda.py`). Energy and potential are continuous in the density and the energy in the positions: the same checks agree to the truncation of the central difference, 1e-7 to 9e-6 Ry/bohr at 1e-3 bohr and 5e-9 to 8e-7 after Richardson extrapolation (an atom off and on a grid point, spin, both routine sets of the cell), and the MPI runner on two ranks gives the energy and forces of one process (Au3 6.7e-11 Ry and 7.1e-11 Ry/bohr, Au4 4.3e-12 and 7.3e-11). First runs on one A100, wall `off` -> `on` (both orders): C60 (Th, h 0.4) 5.68 / 5.56 -> 5.39 / 5.25 s, Au3 3.59 / 3.79 -> 3.71 / 3.89 s (33 -> 35 SCF steps), Ag4 in 14 bohr 5.05 / 5.10 -> 5.42 / 5.23 s (47 -> 56 steps), Au4 in a sphere of 20 bohr at h 0.3 (1.24 M points, no symmetry) 29.8 / 29.7 -> 28.6 / 28.6 s (71 -> 66 steps): the functional itself takes 0.2 to 12 ms more per step, the rest is the number of steps, which moves both ways. Not the default: by design it moves energies away from those of PARSEC 1.5, by -2.9e-4 to -3.5e-4 Ry per atom for these Au and Ag systems (C60 -6.0e-5 Ry per atom), and forces by up to 1.8e-5 Ry/bohr; the off value restores the former bits (clusters, a spin cluster, the cell on both routine sets). A value it does not know is refused when the input is read |
| `PARSEC_LINEAR_TABLE_SLOPES` | `tabulated` (default): the forces of a species with linear tables (`Cubic_Spline: .false.`, or no value under `PARSEC_CUBIC_SPLINE_DEFAULT=off`) take the slope tables of `pseudo.f90` (an 18th-order difference on the logarithmic grid, interpolated linearly) for its local potential, valence and core densities and projectors, as `forcloc.f90` and `forcnloc.F90` do: PARSEC 1.5's forces. The energy samples those tables by linear interpolation, so it is rough on the scale of a relaxation step and the forces are not its slope (`FORCES_AND_RELAXATION.md`, linear tables). Full SCF on A100, criterion 1e-8 Ry, `E - TS`, central differences along unit vectors at 1e-3 and 2e-3 bohr, force minus slope in Ry/bohr (in brackets with `Cubic_Spline: .true.`): Si28H36 of `examples/0d_Si28H36` along all atoms 2.2e-3 and 2.1e-3 (-2.6e-6 and -6.9e-7), one H atom -3.0e-3 and -2.1e-3 (4e-7, 6e-7); CH4 with FHI tables 5.1e-3, 5.3e-3 (-4e-7, -1e-6); GeH4 with the Ge core correction -4.1e-3, -2.0e-3 (2e-7, -2.4e-4); NH4 at h = 0.4 bohr -1.2e-2, -1.1e-2 (5.9e-5, 2.6e-4); a silicon cell of 216 atoms at Gamma (default routines) 6.4e-3, 8.2e-3 (6e-8, 1.6e-7); MgO64 and Si64 with a 3x3x3 mesh on the default routines, one atom 0.05 bohr off its site, -9.5e-4, -1.06e-3 and -1.07e-3, -9.7e-4 (5e-7, 2e-6 and -8e-9, -9.9e-6: their local potential is the spline in any case, so the projectors make the difference, and for MgO the Mg core density too, interpolated linearly on these routines: 1.4e-4 to 1.9e-4 Ry/bohr of the frozen-field mismatch of MgO8); Ag4 at h = 0.45 bohr 0.12 and 0.066, one atom 0.45 and 0.25 (-7e-4 at both in a sphere of 10 bohr, 3.4e-6 and 4.5e-6 in one of 14 bohr, which leaves the linear tables as they were: the domain). `interpolant` (opt-in): the slopes of the linear interpolants, the exact derivative of the computed energy except within 3e-3 bohr of a grid point, where `PARSEC_PROJECTOR_CENTER=auto` takes the projector gradient at that distance (the nonlocal term up to 2.6e-2 Ry/bohr off there on an Ag table, 1.9e-3 Fe, 1.4e-3 Au, 6.6e-4 Si FHI; exact under `tip`) (`Pseudopotential/radial_derivatives.linear_slope_setting`, `interpolant_slope`; `Forces/local.py`, `Forces/nonlocal_force.py` for clusters, the double grid and the default routines of a cell at Gamma and on k-point meshes, `Forces/wire_slab.py`, the core density of `Forces/periodic_local.py` and of the ball sums of `acceleration/Forces/periodic_local.py`; the device sums of the local forces of a cluster give way to the readable routine on the host, a loop over atoms and grid points: molecular dynamics of Si28H36 and C60 took 45 -> 86 s and 34 -> 72 s). Central differences at 1e-7 bohr within 2e-6 Ry/bohr (`tests/unit/test_linear_table_slopes.py`); at 1e-3 and 2e-3 bohr the same runs miss the slope by 3.6e-4 and 5.2e-4 (Si28H36), 2.0e-3 (CH4), 7.4e-4 and 2.8e-3 (GeH4), 5.4e-4 and 7.7e-4 (NH4), 3.7e-3 and 5.5e-3 (silicon cell), 0.15 and 0.35 (one Ag4 atom), and the forces stray from those of `Cubic_Spline` by up to 4.5e-3 (Si28H36), 1.6e-2 (GeH4) and 0.75 Ry/bohr (Ag4), against 9e-5, 1.6e-4 and 3.3e-3 for `tabulated`. Not the default: it changes PARSEC 1.5's forces without making them a better description of the slope over realistic steps. Relaxations on one A100 each (first runs; structures to `Force_Min`, `tabulated` / `interpolant` / `Cubic_Spline`): `examples/0d_water_geoopt` 7 / 23 (ending 0.16 bohr away) / 7, the hand-built benzene of `examples/0d_mlff_initial_structure` 14 / not converged after its 60 moves / 13, `examples/0d_SiNa4_relaxation` 4 / 4 / 4; microcanonical molecular dynamics restarted from a breathing state, spread of the conserved energy over the run: C60 (21 steps of 20 a.u.) 2.2e-3 / 1.6e-2 / 9.5e-4 Ry, Si28H36 (16 steps) 7.3e-4 / 6.4e-4 / 3.5e-4 Ry. These measurements made the cubic spline the default of a species without the label (`PARSEC_CUBIC_SPLINE_DEFAULT`, since 2026-11-22), so this switch acts on `Cubic_Spline: .false.` and on inputs run under `PARSEC_CUBIC_SPLINE_DEFAULT=off`. The Fortran-derived routines of a cell keep `PARSEC_PERIODIC_FORCE_SLOPES` (their default is the interpolant); `Cubic_Spline` species and the periodic local potential of the default routines (always the spline) are not affected. The preparation refuses an unknown value before the SCF. | `tabulated`: the forces of before, bit for bit |
| `PARSEC_PERIODIC_FORCE_SLOPES` | `interpolant` (default): the forces of the Fortran-derived routines of a cell (`Forces/periodic.py`, also the device sums of `Forces/periodic_device.py` and the k-point meshes) differentiate the linear tables of a species with `Cubic_Spline: .false.` (or without the label under `PARSEC_CUBIC_SPLINE_DEFAULT=off`) as the energy samples them, the slopes of the interpolants of `r V`, of the densities and of the projectors: the exact derivative of the computed energy (chosen so when these routines were ported, against central differences at 1e-5 bohr). `tabulated`: the projectors take `dvw` of `pseudo.f90` (Fortran's nonlocal forces to 1e-6 Ry/bohr), the rest the interpolants. `tables` (opt-in, since 2026-11-22): every slope from the tables of `pseudo.f90` (`dvion`, `drhodr`, `ddenc`, `dvw`), the forces of the clusters and of the default routines under `PARSEC_LINEAR_TABLE_SLOPES=tabulated`; the device and FFT sums of the local terms give way to the readable routine on the host for such species (`tests/unit/test_linear_table_slopes.py`: its core term is the default routines' to 1e-12). Measured on A100 (first runs, one GPU each, SCF 1e-9 Ry; force minus the slope of `E - TS` at 1e-3 and 2e-3 bohr along all atoms; BFGS to `Force_Min` 1e-3 Ry/bohr; microcanonical dynamics of 25 steps of 40 a.u. from rest, spread of the conserved energy), `interpolant` / `tables` / `Cubic_Spline`: Si8 POTRE with a 2x2x2 mesh, one atom 0.05 bohr off its site: 2.8e-4, -2.2e-4 / 1.5e-3, 1.0e-3 / 7e-8, 2.8e-7; forces 1.3e-3 / 1.2e-5 from those of `Cubic_Spline`; BFGS 41 structures without converging (unbounded first step) and 17 with `Max_Step: 0.3 bohr` / 4 and 3 / 4 and 3, final positions 1.2e-2 / 4.9e-4 bohr from `Cubic_Spline`; dynamics 1.1e-5 / 6.4e-5 / 6.6e-7 Ry. Si8 POTRE at Gamma: 1.2e-4 / 7.6e-4 / 1e-7; dynamics 1.7e-5 / 7.7e-5 / 1.8e-5 Ry. MgO8 POTRE (PBE, 2x2x2, an O atom moved): 4.5e-3, 1.9e-2 / -2.3e-2, -9.0e-3 / 1e-5, 4e-5; forces 2.7e-2 / 1.0e-3 from `Cubic_Spline`; BFGS 41 without converging / 18 / 22, final positions 0.11 / 1.6e-3 bohr from `Cubic_Spline`; dynamics 2.7e-2 / 1.4e-2 / 1.4e-2 Ry. Si8 with FHI tables at Gamma: 3.8e-3, 5.1e-3 / -2.4e-2, -2.2e-2 / 8e-5, 3e-5 (the coarser FHI grid offsets the energy of the linear tables, which `tables` does not follow); BFGS 41 without converging / stopped by its line search after 16 at 1.4e-2 Ry/bohr; dynamics 7.6e-5 / 4.6e-3 / 3.9e-5 Ry. `tabulated` mixes the two conventions and is the worst on MgO8 (5.4e-2 off the slope). The default stays `interpolant`: `tables` follows the forces of `Cubic_Spline` and relaxes POTRE cells as it does, but is farther from the slope at realistic steps on Si8, worse on FHI tables and in small-amplitude dynamics, and slower (the host sums: a single point 4.6 -> 8.4 s, 25 steps of dynamics 18.9 -> 112.7 s on Si8 at Gamma); `Cubic_Spline` is better than either everywhere. The preparation refuses an unknown value before the SCF. | `interpolant`: the forces of before, bit for bit |
| `PARSEC_MOVE_FLAG_SUBGROUP` | `auto` (since 2026-11-19): a `bfgs`/`simple` relaxation whose `Move_Flag` holds atoms that an operation of the structure exchanges with movable ones keeps the subgroup of the operations that map the fixed atoms onto fixed atoms (`Relaxation/run.py`, `symmetry_decision`, `symmetry_fixed_atoms`): its forces and moved positions are averaged over that subgroup on every backend, and the accelerated driver detects it with the fixed atoms as a species of their own (`driver._detection_atoms`; `prepare_single_point(symmetry_fixed_atoms=)`), for a cluster where the flags break its group and for a bulk cell at the Gamma point whenever it fixes atoms (first the whole group where no operation exchanges fixed and movable atoms, the rule of the k-point force wedge; on a k-point mesh, and for the density average of `Symmetrize_Charge_Density`, the space group the problem carries takes the subgroup too, `PARSEC_MOVE_FLAG_SPACE_GROUP`); the single-point forces of a cell with `Move_Flag` keep its sectors (the flags move nothing). A relaxation with fixed atoms of a spin-polarized cell is not taken (the subgroup sectors are validated for unpolarized cells): it keeps the full grid of both channels that `PARSEC_SPIN_SECTOR_FORCES` gives it, and the runner refuses it on several ranks; one on the Fortran-derived routines, whose forces in sectors (`PARSEC_FORTRAN_SECTOR_FORCES`) are averaged over the group of their own operations (which ignore `Move_Flag`, as PARSEC 1.5 does) and of the sectors, keeps the sectors of the subgroup on the command line since 2026-11-21 under `PARSEC_MOVING_SECTORS=auto` where the cell's basis reaches the floor of the sectors and they have 4 operations or more, in one process (the ionic loop keeps the operations of that group that respect the flags), and their full grid otherwise, under `off` and in the runner (`acceleration/cli.py::periodic_sector_forces_refusal`, `fortran_fixed_atoms`). The main.py and the full-SCF runner alike; the WARNING of parsec.out names the subgroup and `symmetry_detection` the labels. Diamond 4^3 (512 atoms, benchmark settings, `Use_Fortran_Routines: .false.`, one node of 4 A100, first runs): one atom of the perfect crystal fixed, Td about it (24 operations) where the cell used the full grid: SCF 15.30 -> 2.59 s, process 27.7 -> 12.6 s (5^3, 1,000 atoms: SCF 66.3 -> 5.4 s, process 79.8 -> 13.9 s, energies within 7e-6 Ry at the criterion 2e-4 Ry); the atom at the origin moved along (1,1,1) and the last atom fixed, the mirror x <-> y (Abelian, 2 sectors): SCF of the two structures 15.2 + 4.9 -> 11.6 + 3.4 s. Ge5 cluster with one outer atom fixed (C3v, the mirror in sectors) and the silicon cube with 3 or 7 fixed atoms: positions and energies of two BFGS moves within 1e-6 of the full grid, forces within 2e-6 Ry/bohr (`test_move_flag_subgroup.py`). A subgroup that is not abelian takes the weighted sectors of `PARSEC_ORBITAL_GROUP` where their plan solves less than its abelian sectors (2026-11-21, with forces, 4 A100, first runs, two `bfgs` moves at 1e-6 Ry): diamond 4^3 with the neighbours of four centres of a face-centred lattice moved out and the atom at the origin fixed (Td about it kept, the sectors of D2d with weights 1, 1, 3, 3, forces averaged over 24 operations): 14.2 s and 0.91 GiB on the fullest GPU, against 21.8 s and 1.26 GiB in the abelian sectors (`PARSEC_ORBITAL_GROUP=abelian`, 4 operations) and 45.0 s and 18.1 GiB on the full grid (`off`); forces of the first structure within 2.3e-6 Ry/bohr of the full grid, energies within 3e-10 Ry, the third structure within 9e-8 Ry and 6.7e-6 Ry/bohr (abelian: 8e-9 Ry, 1.0e-5 Ry/bohr); Si64 with the four neighbours of the origin moved out and the origin fixed (Td kept): 9.3 s weighted, 11.7 s abelian, 8.3 s full grid (a small cell, which the sector forces keep in sectors below `PARSEC_PERIODIC_SECTOR_FLOOR`), forces within 1.8e-6 Ry/bohr. A C3v subgroup (one neighbour of a breathing atom fixed) keeps its abelian sectors (a mirror, 2 operations): the weighted plan of C3v is that mirror group at the same cost, since its E levels have one partner in each of its two sectors (diamond 4^3: 18.7 s and 1.80 GiB against 46.7 s and 18.1 GiB on the full grid; `PARSEC_ORBITAL_IRREPS=once` takes the same route, 19.0 s). | `off`: such a cluster relaxation uses no symmetry and every cell with a fixed atom keeps the full grid (the runner: one sector of the identity), as before (PARSEC 1.5 ignores `Move_Flag` in its group); the space group a cell carries (k-point meshes, `Symmetrize_Charge_Density`) keeps the routes of before 2026-11-21 as well (`PARSEC_MOVE_FLAG_SPACE_GROUP` needs `auto` here) |
| `PARSEC_MOVING_SECTORS` | `auto` (since 2026-11-21, `main.py`; `SCF/moving_sectors.py`): molecular dynamics and manual movements, which PARSEC 1.5 runs without point group, keep the sectors of the group their structures keep, where a cluster's full grid times its states reaches 2^24 (`MOVING_SECTOR_WORK`) and a cell's basis the floor of `PARSEC_PERIODIC_SECTOR_FLOOR` (the full grid needs not average these structures, as it must those of a relaxation) with sectors of 4 operations or more (`MOVING_CELL_MINIMUM_ORDER`: `prepare_single_point(minimum_sector_order=)` drops a smaller group after the detection, before the sectors are built, and the cell takes the full grid with a NOTE); a cell below the floor is planned on the full grid (`--dry-run` and parsec.out give the reason). Taken: the microcanonical integrator restarted from an `mdinit.dat` whose positions, last displacement and accelerations keep a group, or started with one movable atom (its Maxwell velocity is removed with the centre of mass), and `Minimization: manual`, also with `Movement_Num: 0` (a single point without recentring, as the inputs of `examples/0d_xps_core_levels`). A fresh run of several movable atoms (random Maxwell velocities) and a Langevin run (a random force at every step) leave every group after their first structure and keep the full grid from the start, bit for bit. A cluster keeps the signed permutations about the origin of its atoms (the atoms `Move_Flag` fixes as a species of their own) that also keep the state, or every structure of `manual.dat` with the permutation of the input; where they keep less than the atoms the detection takes them as pseudo-atoms (`prepare_single_point(symmetry_constraints=)`; the orbits of the Abelian sectors of such a detection are those of the atoms alone since then, `driver`, as they were for a point group). A Gamma cell keeps the group its detection finds (parsec.out names its operations after the preparation), checked against the state of the restart and the structures of `manual.dat` after its preparation, else prepared again on the full grid with a NOTE (a path that starts from the perfect crystal loses its translations at its first move; an `mdinit.dat` holds 8 digits, so the images of a cell across its boundary match to about 1e-6 bohr and the detection may keep fewer operations: diamond 3^3 restarted from a Td breathing state, 2 of 24). At every structure the forces are averaged over the group of the sectors; after every step the positions, the orbits of their atoms and the state (the displacement and the accelerations times `Time_Step^2`) are checked against it to 1e-6 bohr and the state is averaged over it; a structure that leaves it is prepared again (in place) in the sectors of the group it keeps (a cluster, once) or on the full grid, from the potential of the structure before with a first filter, and parsec.out says so. Also under `auto`: a `bfgs`/`simple` relaxation with atoms fixed by `Move_Flag` on the Fortran-derived routines keeps the sectors of the subgroup that maps fixed atoms onto fixed atoms where the cell's basis reaches the floor of the sectors and those sectors have 4 operations or more, in one process (`periodic_sector_forces_refusal(fortran_fixed_atoms=True)`, `fortran_fixed_atoms_lifted`, `fortran_fixed_minimum_order`; the command line); a smaller cell or group keeps the full grid of those routines, bit for bit as before (the displaced Si8 cube with `Move_Flag: 5`, 5,832 points x 22 states, five BFGS moves: 4.8-5.4 s on the full grid against 6.3-6.4 s in the 2 sectors of its mirror, slower at every step; diamond 2^3 with a C3v defect, 6.3e6, 15.1-15.2 -> 14.3-14.4 s). Not taken: two spin channels under `PARSEC_MOVING_SPIN_SECTORS=off`, and antiferromagnetic labels under its `auto` (the next row), spinors, `Ignore_Symmetry`, `--symmetry off`, the other reasons PARSEC switches the symmetry off for (point charges, `Polarizability`, FDET, fold blocks), `Restart_From_Wfndat`, MPI, wires, slabs and k-points; the MPI full-SCF runner keeps all of these runs on one sector of the identity. One A100 node (4 GPUs), first runs in both orders, eleven structures of molecular dynamics restarted from a breathing state, SCF to 1e-6 Ry, the full grid -> the sectors: Ge5 (3.9e5 points x states, Td) 9.5 -> 9.8 s, benzene (5.4e6, D2h) 11.6-12.5 -> 16.4-17.1 s, naphthalene (1.6e7) 18.5-19.0 -> 22.3-22.8 s (these three keep the full grid under `auto`), Si28H36 on a grid of 0.5 bohr (1.8e7, the mirror of C3v) 19.2-19.4 -> 17.2-18.1 s, naphthalene on 0.36 bohr (2.2e7) 27.9-28.5 -> 23.6-23.8 s, Si28H36 (2.9e7) 27.1-29.0 -> 22.4-23.3 s (4.0 -> 1.4 GiB), C60 (3.5e7, Th) 19.5-19.8 -> 15.0-15.4 s (2.6 -> 0.52 GiB); energies within 1.1e-7 Ry, forces within 3e-6 Ry/bohr and the positions within 2.5e-7 bohr after ten steps of 0.05 to 0.09 bohr (Ge5 at 1e-8 Ry: 2e-10 Ry, 4e-9 Ry/bohr, 2e-10 bohr); a manual breathing path of Si28H36, 29.5 -> 25.5 s, energies within 8e-11 Ry and forces within 6e-7 Ry/bohr. Diamond 3^3 (216 atoms, `make_diamond.py`, Cubic_Spline: the Fortran-derived routines), SCF to 1e-7 Ry: MD restarted from the Td breathing state (the 2 operations its 8-digit `mdinit.dat` keeps; 2 sectors, so under `on`) 71.7 -> 49.2 s, 3.9 -> 0.78 GiB, energies within 1.9e-9 Ry, forces within 3.8e-6 Ry/bohr, positions within 2.3e-7 bohr (at the benchmark criterion of 2e-4 Ry 64.1 -> 44.1 s, the forces of the full grid then off by its SCF noise, 1e-3 Ry/bohr); with a lattice constant whose coordinates hold in those digits (a = 6.75 bohr: Td, 24 operations) 71.3-72.2 -> 42.8-44.1 s in both orders, 3.9 -> 0.70 GiB (a warm SCF step 2.0-2.3 s against 1.24-1.36 s, the forces 0.41 against 3.8 s), energies within 7.3e-9 Ry and the forces of the atoms off the grid points within 1.3e-6 Ry/bohr, but up to 2.8e-4 Ry/bohr on the atoms within 1.3e-4 bohr of a grid point, where the spline local force has its cone tip on either route (the positions then part by 5e-6 bohr in ten steps); silicon 3^3 (216 atoms, LDA, a = 10.26 bohr, Td, 24 operations, SCF to 1e-7 Ry) restarted from the same breathing state: on the default routines 35.8 -> 26.1 s, 5.2 -> 1.0 GiB, on the Fortran-derived routines 35.2-36.1 -> 29.6-30.2 s, 5.3 -> 0.85 GiB, energies within 1e-9 Ry, forces within 2.4e-7 Ry/bohr, positions within 2.5e-8 bohr; a manual path from the displaced structure (Td, 24 operations) 73.0 -> 44.4 s (62.6 -> 30.5 s at 2e-4 Ry), energies within 1.9e-10 Ry, forces within 2.5e-6 Ry/bohr but 1.2e-5 on the atoms within 3.4e-13 bohr of a grid point (the cone tip of the spline local force, on either route, `PARSEC_FORTRAN_SECTOR_FORCES`); a BFGS relaxation with the neighbour on +x+y+z fixed (C3v; the 2 sectors of its mirror, so under `on`) 79.5 -> 53.0 s, the forces of the first structure within 3.8e-7 Ry/bohr, the energies of the first three within 2e-7 Ry, after which the paths part at an atom on a grid point (the same cone tip; final energies within 4e-8 Ry); the displaced Si8 cube with `Use_Fortran_Routines` and `Move_Flag: 5`, five BFGS moves (under `on`: below the floor, 2 sectors): positions within 6e-9 bohr, energies within 1.4e-9 Ry, forces within 1.2e-8 Ry/bohr. A cell whose sectors have fewer than 4 operations keeps the full grid under `auto` (`MOVING_CELL_MINIMUM_ORDER`; one A100 node, first runs in both orders, the full grid -> the sectors): silicon 3^3 with the breathing defect, fresh MD of eleven structures with one movable atom on a bond (C3v, the 2 sectors of a mirror, the default Chebyshev-Davidson solver) 30.1-30.4 -> 31.4-31.7 s on the default routines and 27.7-27.8 -> 30.0-30.2 s on the Fortran-derived routines, 5.2 -> 11.9 GiB of device memory (the first SCF 9.6 -> 14.8 s, a later step 1.56 -> 1.16 s); five BFGS moves on the Fortran-derived routines with that neighbour fixed 29.8-30.2 -> 32.2-32.5 s, 5.3 -> 12.3 GiB; under `auto` these give the bits of `off` (MD 29.3-30.3 s against 30.0-30.3 s: the detection that finds the 2 operations costs about 0.4 s). The diamond cubes of `make_diamond.py` (Chebyshev filtering, spline forces) gain in 2 sectors all the same (one movable atom on a bond of the breathing defect 67.2-67.5 -> 46.4-46.5 s, 3.9 -> 0.78 GiB, forces within 9.3e-7 Ry/bohr; the restart and the BFGS relaxation above), which `on` takes. With 4 sectors (the movable or the fixed atom at (a, 0, 0), C2v): silicon MD 23.6-23.8 -> 16.6-17.1 s (5.2 -> 2.6 GiB) on the default routines and 24.2-24.4 -> 17.0 s (5.3 -> 2.3 GiB) on the Fortran-derived routines, forces within 2.9e-7 Ry/bohr; the Fortran BFGS 31.1-31.3 -> 24.5-25.1 s (5.3 -> 2.4 GiB), energies within 4.2e-7 Ry and forces within 6e-6 Ry/bohr after five moves whose paths part by 2.5e-6 bohr; diamond MD 60.5-60.6 -> 34.3-34.8 s (3.9 -> 0.89 GiB), energies within 1e-12 Ry, forces within 1.2e-5 Ry/bohr at 1e-7 and at 1e-9 Ry alike: either route's forces deviate from the Td of the first structure (6.5e-6 Ry/bohr on the full grid, 7.9e-6 in the sectors, which are exact in their C2v) through the atoms on grid points (the cone tip above), and their Td averages agree within 3.3e-6 (`acceleration/tests/test_moving_sectors.py`). Since 2026-11-22 the MPI full-SCF runner takes the same plan on the layouts that hold one sector of the identity (one rank, `--rank-per-gpu`, nested sharing; `benchmarks/mpi_full_scf._moving_plan`): a structure that leaves the group is prepared again on every rank (`MPISectorContext.reprepare` with `moving`, each rank applying the atoms to its own input), in the sectors of the group it keeps or as the sector of the identity whose basis the ranks share; before, every such run of the runner was that sector, which `off` keeps. A hybrid, the serial control, the supercell of a k-point mesh and the other layouts keep what they did. A100, the code defaults, first runs of each structure (no symmetry cache), `off` -> `auto`, wall seconds of one rank of four GPUs: Si28H36 molecular dynamics (restarted breathing, 3 steps, SCF to 1e-6 Ry, 2 sectors) 12.86 -> 10.70 (peak 3493 -> 1393 MiB), C60 (Th, 24 operations) 9.36 -> 7.21, silicon 3^3 (216 atoms, the default routines, a Td breathing restart, 1e-7 Ry) 20.69 -> 14.13 (4733 -> 951 MiB); four ranks of one GPU each: Si28H36 13.38 -> 9.35, silicon 3^3 21.91 -> 7.69; the trajectories of `off` within 5.7e-8 Ry, 7.5e-8 bohr and 1.1e-6 Ry/bohr (silicon: 7e-11 Ry, 3e-9 bohr, 7e-8 Ry/bohr), those of `main.py` to the bit (Si28H36, C60, silicon), one rank and four within 5e-11 Ry and 4e-11 Ry/bohr; a structure of C60 forced off the group at the second step was prepared again on every rank in the sectors of a subgroup (1.1e-6 Ry/bohr from `off`); a manual path of silicon 3^3 from the perfect crystal, whose structures do not keep its translations, was prepared again as the sector of the identity before its first SCF (the bits of `off`). Under `on` the manual path of benzene (4 of 8 operations kept, 3 structures) took 5.83 against 11.52 s and the molecular dynamics of Ge5 4.25 against 3.28 s, both below the 2^24 of `auto`. | `off`: molecular dynamics and manual movements on the full grid (on the runner one sector of the identity), and relaxations with fixed atoms on the Fortran-derived routines on their full grid, bit for bit as before; `on`: the sectors whatever the size and order (a cluster below 2^24, a cell below the floor or with fewer than 4 operations, a Fortran relaxation with fixed atoms of any size) |
| `PARSEC_MOVING_SPIN_SECTORS` | `auto` (since 2026-11-22, `SCF/moving_sectors.py`; read and checked in every molecular dynamics and manual run): the molecular dynamics and manual movements of two spin channels take the sectors of `PARSEC_MOVING_SECTORS` as an unpolarized run does (its plan, size floor of 2^24 grid points x states of a channel and per-structure checks, both channels in the sectors of the group of the labels), but for a start whose labels carry `Initial_Spin_Polarization` of both signs (an antiferromagnetic or broken-symmetry start), which keeps both channels on the full grid, as before, with a `NOTE` (`ANTIFERROMAGNETIC_REASON`): the sectors of its labels can converge to a symmetric state that two channels on the full grid leave, and neither the guard nor the stability test of the flip sees it (one A100 node, `Minimization: manual` with `Movement_Num: 0` and forces, PBE, a sphere of 14 bohr at 0.25 bohr, the 8 sectors of C4v against the full grid: Fe2 at 3.8 bohr from `+-0.8` -78.62964813 Ry converged against -78.64828958 (the full grid with the flip, converged) and -78.65719336 (unconverged), forces 0.125 Ry/bohr apart; Co2 at 3.9 bohr from `+-0.7` -105.79380344 against -105.80443025, unconverged, 10.6 mRy). `on` takes such starts too (Cr2 at 3.6 bohr from `+-0.8` agreed: eleven structures of a breathing restart on 0.25 bohr, 2.8e7, 98.2 -> 48.1 s, energies within 1.1e-7 Ry, forces 4.5e-6 Ry/bohr). The spin guard of every SCF loop leaves the sectors where it fires, as in a single point: weighted sectors go on to the abelian sectors of the group (or straight to the full grid where those would stop the same levels, `PARSEC_WEIGHTED_GUARD_SHORTCUT`), abelian sectors to the full grid; the system is prepared again in place and the later structures keep what it holds (a guard stop of a weighted Cr2 run at its third structure: abelian sectors from there on, the trajectory of `off`). A structure that leaves the group is prepared again from the potentials of both channels (`MovingSectors._warm`). One A100 node, first runs, eleven structures from a breathing restart (SCF to 1e-7 Ry, the Si28H36 example to 1e-6), `off` -> `auto` (the full grid of both channels -> the sectors), each against its own full grid: O2 triplet on 0.18 bohr in a sphere of 13 (1.9e7, D4h) 170.6 -> 60.1 s, 3.98 -> 0.85 GiB, energies within 3.4e-9 Ry, forces 6.6e-7 Ry/bohr; Si28H36 with two channels 51.8 -> 40.2 s, 4.93 -> 1.54 GiB, 3.2e-7 Ry, 4.9e-7 Ry/bohr, its manual breathing path 52.4 -> 42.4 s, 1.3e-8 Ry, 6.3e-7 Ry/bohr; Si28H36+ (a hole in the t2 shell: the guard leaves the sectors in the first structure) 53.1 -> 54.8 s, the trajectory of `off` (1.3e-10 Ry, 8e-9 Ry/bohr); the polarized single points of `examples/0d_xps_core_levels/jctc2022_atoms` (`Minimization: manual`, `Movement_Num: 0`, `occup.in`, 2.3e7, the 48 operations of Oh) the same energies to the printed digits, total 24.0 -> 19.0 s (C atom), 21.2 -> 18.4 s (C+ with a 1s hole), 22.8 -> 18.8 s (O atom), 22.2 -> 18.7 s (N+ with a 1s hole). A spin-polarized PBE cell, whose moves take the 1x1x1 k-point driver (`PARSEC_SPIN_IONIC_KPOINT`), keeps that route (V- in diamond 3^3, one movable atom: 91.0 s both). Below the floor `auto` keeps the full grid (`PARSEC_MOVING_SECTORS=on` takes the sectors: O2 on 0.3 bohr 14.6 -> 22.7 s, 5.4e-10 Ry, 1.4e-7 Ry/bohr, and manual 14.4 -> 23.9 s; O2 in a periodic cube of 12 bohr, LDA, 11.3 -> 18.6 s, 1.7e-8 Ry, 5.7e-6 Ry/bohr: on small grids the sectors cost more than they save, as for an unpolarized run). The forces of both routes agree with each other at single points to 4e-8 Ry/bohr (Cr2 at 3.6 and 3.65 bohr, O2 at 2.28 and 2.4 bohr, breathing). A run whose two channels the driver keeps on the full grid in one process (`Spin_Symmetry_Sectors: off`, `PARSEC_CUPY_DISTRIBUTED_FILTER=1`: `driver._spin_sector_refusal`) plans the full grid too, with that reason. The MPI full-SCF runner keeps both channels of these runs as one sector of the identity whose basis the ranks share, under every value (`benchmarks/mpi_full_scf._moving_plan`). | `off`: both channels on the full grid, as before |
| `PARSEC_RUNNER_FORTRAN_FIXED_SECTORS` | `auto` (default, since 2026-11-22; not under `PARSEC_MOVING_SECTORS=off`): the full-SCF runner (`benchmarks/mpi_full_scf.py`, `scripts/run_multi_gpu_scf.sh`) takes a `bfgs`/`simple` relaxation of a Gamma cell on the Fortran-derived routines whose `Move_Flag` fixes atoms in the sectors of the subgroup that maps the fixed atoms onto fixed atoms, as it takes those of the default routines, the ranks dividing the sectors whatever their number (`SCF/moving_sectors.py`, `runner_fortran_fixed_atoms`; the forces and moved positions averaged as `main.py` averages them in one process). `main.py` in one process keeps such a relaxation on the full grid below the floor of the sectors and for fewer than 4 operations (`PARSEC_MOVING_SECTORS=auto`, measured slower there); the runner has no full grid to fall back to, and `main.py --mpi` refuses every Gamma relaxation. Measured on one A100 node (first runs, `bfgs` with two moves, SCF to 1e-7 Ry): silicon 3^3 (216 atoms) with its breathing defect and the atom at (a, 0, 0) fixed (the 4 sectors of C2v): 2 ranks (one GPU each) and 4 ranks give the energies of `main.py`'s sectors in one process to the bit and its forces within 2.9e-13 Ry/bohr at every structure, its full grid within 2.2e-9 Ry and 2.1e-7 Ry/bohr; accelerated wall 10.5 s (2 ranks) and 8.4 s (4 ranks) against 18.3 s (sectors) and 22.1 s (full grid) of `main.py` on the node; the neighbour on +x+y+z fixed (C3v, the 2 sectors of a mirror) on 2 ranks: within 3.3e-8 Ry and 4.4e-7 Ry/bohr of `main.py`'s full grid, 19.0 against 19.6 s; diamond 3^3 (PBE, the atom at (a, 0, 0) fixed, 4 sectors) on 2 ranks: `main.py`'s sectors within 4.6e-13 Ry and 1.5e-12 Ry/bohr, its full grid within 1.4e-12 Ry and 1.4e-8 Ry/bohr at the first structure (after the first L-BFGS-B move, positions 3.4e-7 bohr apart: 4.6e-7 Ry and 6.2e-6 Ry/bohr, as between `main.py`'s own two routes), 9.4 s against 13.0 s (sectors) and 19.7 s (full grid) | `off`: refused by name, as before (`main.py` runs it) |
| `PARSEC_MOVE_FLAG_SPACE_GROUP` | `auto` (default, since 2026-11-21; with `PARSEC_MOVE_FLAG_SUBGROUP=auto`): a `bfgs`/`simple` relaxation of a bulk cell whose `Move_Flag` fixes atoms that an operation of the space group the problem carries exchanges with movable ones (a Monkhorst-Pack mesh, `Symmetrize_Charge_Density`) keeps the operations that map the fixed atoms onto fixed atoms, on both routine sets (`Input/parsec_input.py`, `_move_flag_space_group`): the mesh is reduced by them and time reversal, the density averaged over them, the nonlocal forces of the wedge completed and the forces averaged over them, so every structure keeps them. The group is searched again with the fixed atoms as a species of their own (`Grid/space_group.py`, `atoms_with_fixed_species`): the space group of the atoms holds one translation per rotation, and in a supercell the one that keeps a fixed atom is another (the 8-atom diamond cube with one atom fixed: 24 operations of Td about it, of which the group of the atoms holds 2). Of the operations found only those that also map the grid's finite-difference operators onto themselves are kept, for every kind of orbitals (`Grid/kpoints.py`, `stencil_preserving_operations`, with the stencil of the default routines or of Fortran's general grid): these inputs ran the whole mesh before, and on an oblique grid the other operations change the discrete Hamiltonian at the order of the discretization (hcp Au2 at 0.3 bohr with one atom fixed: D3h about it, 12 operations, whose wedge was 7.7e-3 Ry off the whole mesh; 4 are kept); an orthogonal grid keeps every operation. The default routines search with the conditions of their wedge (the grid of `Grid_Spacing`, the mesh), the Fortran-derived ones keep the grid of the whole group and take the subgroup where its translations map that grid onto itself (else the refusal of before); a weighted `Kpoint_List` (a wedge of the input's group), non-collinear magnetism and spin-polarized spin-orbit coupling (time reversal flips the moments; the spinors of non-magnetic spin-orbit coupling take the subgroup since 2026-11-22, `PARSEC_SPINOR_MOVE_FLAG_SUBGROUP` below) and hybrids on a mesh (whose mesh stays its time-reversal pairs, so the subgroup saves no point, and whose density average holds a state that exact exchange may break: O2 PBE0 below) keep the routes of before, and a subgroup of the identity alone runs the time-reversal pairs (default routines) or time reversal alone (Fortran-derived). Collinear spin takes the subgroup (its relaxations with k-points run on the Fortran-derived routines). Also a Gamma-point relaxation with `Symmetrize_Charge_Density` (Fortran-derived routines) now averages the density of the moved structures over that subgroup instead of the group of the first structure, which they no longer have; where no subgroup is kept (spinor orbitals, a subgroup whose translations the grid does not take) it averages nothing. Measured on one A100 node (4 GPUs, first runs in fresh processes, process seconds; `bfgs` with two moves). Cubic cells, SCF to 1e-6 Ry: 2 x 2 x 2 conventional cells (64 atoms) of diamond (18 points per cubic edge) and of Si (`examples/3d_Si`, 24 points per edge) with the four neighbours of the atom at the origin moved out radially (Td about it) and one of them fixed (the 6 operations of C3v along that bond kept), Gamma-centred meshes. Default routines, against the time-reversal pairs of `off`: C 2 x 2 x 2 4 instead of 8 points, 37.4 -> 21.0 s; C 3 x 3 x 3 6 instead of 14, 65.3 -> 32.4 s; Si 2 x 2 x 2 68.8 -> 38.5 s (2.59 -> 2.37 GiB on the fullest GPU). Fortran-derived routines, which refused these relaxations before, against the whole mesh as a `Kpoint_List` of equal weights: C 2 x 2 x 2 48.9 -> 28.7 s, Si 2 x 2 x 2 95.2 -> 52.8 s, Si 3 x 3 x 3 (6 of 27 points) 299.4 -> 81.2 s (3.78 -> 2.59 GiB). The first structure (forces up to 0.02 Ry/bohr): forces within 1.5e-6 Ry/bohr, energies within 1.5e-8 Ry. The second: the first L-BFGS-B move turns those differences into positions up to 3.6e-5 bohr apart, and there the energies agree within 4.4e-6 Ry and the forces, which reach 0.18 Ry/bohr, within 2.5e-5 Ry/bohr (at this SCF criterion the forces of the whole mesh themselves break the kept group by up to 2.1e-5 Ry/bohr there; those of the wedge are symmetric to round-off); the third within 1e-8 Ry and 2.8e-6 Ry/bohr. At an SCF criterion of 1e-9 Ry (diamond 2 x 2 x 2, both routine sets) the routes agree within 8e-10 Ry and 1.3e-8 Ry/bohr at every structure. Perfect diamond 3^3 (216 atoms) with the last atom fixed (Td kept), SCF to 1e-5 Ry, one structure: default routines 2 x 2 x 2 55.0 -> 48.2 s with an empty kernel cache in every process (23.0 -> 16.6 s with a warm one), 3 x 3 x 3 34.3 -> 15.1 s warm, within 1.5e-9 Ry and 2e-6 Ry/bohr; Fortran-derived 2 x 2 x 2 53.2 -> 32.7 s, within 1e-8 Ry; diamond 2^3 with its second sublattice moved along (1,1,1) and the last atom fixed (C3v kept, default routines, 3 x 3 x 3) 64.0 -> 32.9 s, within 2.2e-8 Ry and 3.4e-6 Ry/bohr. Oblique cells, SCF to 1e-7 Ry, one atom fixed: hcp Au2 (0.3 bohr, 4 x 4 x 2, 4 operations kept, 14 of 32 points) and diamond Si in its primitive cell (20 points per edge, 4 x 4 x 4, 6 of the 24 operations of Td kept, 13 of 64 points). Default routines against `off`: Au 55.6 -> 43.3 s, Si 76.8 -> 33.1 s, the first structure within 5e-11 Ry and 9e-7 Ry/bohr; Fortran-derived routines against the whole mesh written as the exact pairs k, -k of their time-reversal points (a `Kpoint_List` on the same grid): Au 85.1 -> 42.5 s, within 6e-11 Ry and 2.4e-8 Ry/bohr; Si against the routines' own run with `Ignore_Symmetry` (the same grid) 80.5 -> 35.0 s, within 6e-11 Ry and 8.5e-7 Ry/bohr (the list of pairs gives the same energy within 2e-11 Ry and forces 1.0e-5 Ry/bohr smaller, which vanish by symmetry: a list's forces are averaged over the lattice operations that keep its points, which the oblique grid does not keep). These start at forces of 1e-5 to 3e-5 Ry/bohr, so the first L-BFGS-B move follows their direction alone and the later structures of any two routes differ by up to 0.05 bohr. The whole mesh as a `Kpoint_List` folded into (-1/2, 1/2] differs from the wedge and from the time-reversal pairs alike, by 2.2e-3 Ry (Au) and 4.3e-6 Ry (Si) on either routine set: its points on the zone boundary are other Cartesian vectors than theirs (an open question of that list, not of the wedge). Spin: O2 (bond 2.20 bohr, moment 2) in a 10-bohr cube, 2 x 2 x 2, one atom fixed (C4v of D4h kept, 6 of 8 points, Fortran-derived routines), against the whole mesh: 79.5 -> 61.2 s, energies within 4e-10 Ry and forces within 3e-7 Ry/bohr at every structure. Hybrids (not taken, above): with the subgroup's density average, H2 PBE0 (3 x 3 x 1) agreed with the whole mesh within 1e-13 Ry, but O2 PBE0 (spin, 2 x 2 x 2) at 1e-9 Ry and `Exx_Tolerance` 1e-9 Ry reached at the stretched second structure a state 8.5e-3 Ry below the symmetric one without the average (the Fock loop names it unstable under exact exchange), with no point saved. Gamma, Fortran-derived routines, `Symmetrize_Charge_Density`, the 64-atom diamond and Si cells above: the average over the subgroup agrees with no average within 1.4e-7 Ry and 2.3e-6 Ry/bohr at the first and third structures (6.9e-6 Ry and 1.8e-5 Ry/bohr at the second, after the first move); the average over the whole group of before is off at the second structure by 1.0e-3 Ry and 9.5e-2 Ry/bohr (C) and 0.20 Ry and 0.38 Ry/bohr (Si) | `off`: the routes of before: the whole mesh as its time-reversal pairs on the default routines, a refusal by name on the Fortran-derived ones, the density of a Gamma-point relaxation averaged over the whole group |
| `PARSEC_SPINOR_MOVE_FLAG_SUBGROUP` | `auto` (default, since 2026-11-22; with `PARSEC_MOVE_FLAG_SPACE_GROUP=auto` and `PARSEC_MOVE_FLAG_SUBGROUP=auto`): the spinors of non-magnetic self-consistent spin-orbit coupling (`SO_from_scratch` without spin polarization or non-collinear magnetism) on a Monkhorst-Pack mesh take the subgroup of the row above, on both routine sets (`Grid/kpoints.py`, `spinor_move_flag_subgroup_setting`; `Input/parsec_input.py`): a `bfgs`/`simple` relaxation whose `Move_Flag` fixes atoms that an operation exchanges with movable ones reduces its mesh by the operations that map the fixed atoms onto fixed atoms (and the grid's finite-difference operators onto themselves) with time reversal (the SU(2) matrices turn a spinor, time reversal maps it onto its Kramers partner at `-k`; neither changes the density or the forces), averages the density over them and completes and averages the forces over them, as for scalar orbitals; the force formula is that of the spinor wedge (`PARSEC_SOC_KPOINT_WEDGE`). Measured on one A100 node (first runs in fresh processes, process seconds, SCF to 1e-7 Ry, `bfgs` with two moves): a chain of three Au atoms along (1,1,1) in a 10-bohr cube with the atom on +x+y+z fixed (C3v kept of D3d, 2x2x2: 4 points instead of the 8 time-reversal pairs), default routines 53.6 -> 28.6 s, the first structure within 9.5e-10 Ry and 1.2e-7 Ry/bohr of the time-reversal pairs and of the whole mesh, the third within 2.5e-8 Ry and 8.8e-7 Ry/bohr; Fortran-derived routines (which refused it) against the whole mesh as a `Kpoint_List` of equal weights 64.9 -> 34.5 s, every structure within 7.5e-8 Ry and 1.9e-6 Ry/bohr; H-Au-Au-H along z in an 8 x 8 x 14 bohr cell with one H fixed (C4v kept of D4h, 4x4x2: 12 points instead of 20 pairs and 32 points), default routines 109.2 (pairs) and 169.5 (whole mesh) -> 66.1 s, the first structure within 5e-10 Ry and 2.5e-7 Ry/bohr; Fortran-derived 222.7 -> 94.5 s, the first structure within 4.6e-10 Ry and 1.4e-7 Ry/bohr. At the second structure the first L-BFGS-B move has turned those differences into positions 1e-6 to 9e-6 bohr apart, and the forces there differ by up to 2.1e-5 Ry/bohr; at an SCF criterion of 1e-9 Ry (H-Au-Au-H) the routes agree within 1.0e-9 Ry and 1.9e-8 Ry/bohr at every structure on the Fortran-derived routines (268.9 -> 109.7 s) and within 8.4e-8 Ry and 3.2e-7 Ry/bohr at the first two on the default routines (227.8 -> 87.1 s), so those differences are the optimizer's amplification of the SCF precision, as for scalar orbitals (the third structure of the default routines, after an L-BFGS-B step of the movable H through the cell boundary next to an Au atom, has forces of 45 Ry/bohr in both routes and is no comparison) | `off`: the time-reversal pairs of the whole mesh on the default routines and the refusal by name on the Fortran-derived ones, as before; non-collinear magnetism, spin-polarized spin-orbit coupling and a Gamma-point relaxation of spinors with `Symmetrize_Charge_Density` keep their routes under both values |
| `PARSEC_BFGS_FIRST_STEP` | `off` (default): the first trial of `bfgs` is the one of `lnsrlb` in PARSEC 1.5, a step of norm 1 bohr over all coordinates along the forces, whatever their size. While the energy keeps falling each further trial of that line search extrapolates (on a constant force 1, 5, 21, 85 ... times the first), and only `Max_Step` bounds the whole run. On four recorded unbounded trajectories (`nbd = 0`) of the displaced 8-atom diamond cube, `lbfgsb.f` of PARSEC 1.5 compiled with gfortran and `Relaxation/domove.py` ask for the same 63 points, bit for bit. A positive length in bohr (opt-in) limits the largest displacement of an atom in that first trial: the trial is only ever shortened, the later trials and iterations follow the rules of L-BFGS-B and can still extrapolate, runs bounded by `Max_Step` and runs continued from a saved BFGS state are not affected (a `relax_restart.dat` without a BFGS block starts BFGS afresh, and the limit acts), parsec.out says when it acted, and any other value stops before the first SCF loop. Why it exists: that cube with one atom moved by 0.02 bohr along [111] (Gamma point, 18 points per edge, four atoms on grid points) leaves diamond (FHI carbon -90.10787 Ry) on the default step. With the forces of `PARSEC_PROJECTOR_CENTER=auto` its line search goes on to structures more than 1 Ry above the start, and the SCF of the seventh structure does not converge within its 150 iterations, which ends the run (FHI and POTRE carbon alike: -88.388 and -88.450 Ry, 1.6 Ry/bohr); with the gradients of `tip` the same input stops at a distorted stationary structure of the Gamma-only sampling (-90.30391 Ry, 3e-5 Ry/bohr; POTRE carbon reaches a like structure), which is not a minimum. The two values give different forces only at the start, whose atoms on grid points take the gradient of `auto` there; at the end structures their forces are the same bit for bit. Measured with FHI carbon at 18 points per edge only (the same under both values): the energy falls on both sides along one mode, six bounded BFGS runs (`Max_Step` 0.15 and 0.3 bohr) from it plus 0.05 bohr all end below it without reaching `Force_Min`, and unbounded runs from such starts under a limit of 0.05 bohr reach -90.7192 and -90.7196 Ry (a third stopped at -90.694 Ry after 40 moves), with four neighbours per atom at 2.71 to 3.27 bohr and the next at 3.47; on the default step four other displacements of that atom along [111] (0.005 to 0.2 bohr per coordinate) end at four different energies, -90.3034 to -90.3096 Ry, while two random displacements of all atoms stay near diamond (-90.1085 and -90.1082 Ry). As single points of these Gamma structures, the stationary structure lies 0.196 Ry below diamond at Gamma and 0.400 and 0.429 Ry above it on 2x2x2 and 3x3x3 meshes, and the -90.72 Ry ends lie 0.61 Ry below it at Gamma and 0.72 to 0.74 Ry above it on those meshes, so a cell this small needs a k-point mesh. A 3x3x3 relaxation from the 15-move structure of the `tip` path ends with four neighbours per atom (bonds 2.90 to 2.96 bohr) at -91.224554 Ry, 2.31 mRy above the run from diamond (-91.226862 Ry), a difference not explained. Limits of 0.05 to 0.5 bohr keep both cubes at the diamond energy, and so does PARSEC's own `Max_Step: 0.3 bohr`. It is not the default because it changes the path and is not faster everywhere. Structures until `Force_Min`, `off` -> 0.2 bohr, first runs on one A100 each: C8 at Gamma, FHI and POTRE, 7 (the unconverged SCF above) -> 18 and 7 -> 18, both at the diamond energy (under `tip` 21 -> 18 and 18 -> 15), C8 3x3x3 15 -> 17, Si64 7 -> 6, Si2 2x2x2 4 -> 4, MgO 3x3x3 8 -> 4, water 7 -> 6, SiNa4 4 -> 4, benzene 14 -> 13: MgO saves 4, the other six cases total 51 -> 50, and the two cubes reach diamond only under the limit (limits of 0.05, 0.1 and 0.5 bohr: 15, 19 and 17 structures; `Max_Step: 0.3 bohr` 12 and 13). The final energies of the other cells (C8 3x3x3, Si64, Si2, MgO) agree within 3e-7 Ry; the clusters stop at their loose `Force_Min`, and under 0.2 bohr differ from `off` by +5.2e-3 Ry (SiNa4), +8.4e-5 (benzene) and -7.8e-5 (water). | `off`: the step of PARSEC 1.5, bit for bit |
| `PARSEC_SECTOR_GRADIENT_ATOMS` | `representatives` (default, since 2026-11-18): where the sectors hold the projectors of one atom per orbit (`PARSEC_SECTOR_PROJECTOR_ORBITS`), the projector gradients of the forces are built for those atoms only (`Symmetry/axis_reflection.py`, `sector_gradient_atoms`; the gradient builders take `atom_indices`), the same forces bit for bit (diamond 6^3 at Gamma: gradient phase 5.4 -> 0.23 s; the host builder at 64,000 atoms 188 s / 5.9 GiB -> about 7 s / 0.75 GiB) | `all`: the gradients of every atom, the other columns dropped afterwards, as before |
| `PARSEC_CUPY_FORCE_UPLOAD_CAP` | `on` (default, since 2026-11-18): the nonlocal force contraction uploads the orbital columns of a sector held on the host (`PARSEC_CUPY_SECTOR_STATE_STORAGE`) at most 1 GiB at a time (`Forces/nonlocal_device.py`); the device copy of a sector's gradients is made once per device and force call | `off`: the former chunks (the last bits of host-spilled forces may change with the order of the sums) |
| `PARSEC_PERIODIC_SECTOR_LOCAL_FORCES` | `orbits` (default, since 2026-11-18): a cell in point-group sectors sums its local ionic and core-correction forces for one atom per orbit of G and turns them onto the orbit (`Forces/periodic_local.py`; the density and V_xc are G-invariant; since 2026-11-19 also the three local terms of the Fortran-derived routines, `PARSEC_FORTRAN_SECTOR_FORCES`), with an equivariance check on up to eight more atoms and `phase_seconds['local_ewald_asymmetry']` (Ry/bohr); diamond 20^3 on one A100: 503.6 -> 4.3 s, the same forces within 1.6e-11 Ry/bohr | `all`: every atom, bit for bit as before |
| `PARSEC_LOCAL_FORCE_SUMS` | `wedge` (default): the local ionic, self-consistency and core-correction forces of a cluster in symmetry sectors are summed over the wedge of the solved group and completed by its operations (`Forces/local_device.py`, `cupy_wedge_local_forces`; see `FORCES_AND_RELAXATION.md`) | `full`: the whole grid; `orbits` (opt-in, 2026-11-19): weighted sectors over one point per orbit of the full group (three times fewer points for Td in D2d, forces within 4e-14 Ry/bohr), no faster on A100 because the fields are averaged over the full group on the host first (nanodiamonds k06 and k08 on 4 A100, warm: 0.070 and 0.118 s against 0.058 and 0.090 s) |
| `PARSEC_PERIODIC_FORCE_SERIES` | `stream` (default, since 2026-11-18): the reciprocal series of the periodic local forces from half of the reciprocal box (real FFT of the density), weighted and contracted slab by slab, on the host when the device is short of memory (360^3: 0.70 instead of 3.71 GiB, 3 x faster, the same sums to ~1e-13) | `host`: the streamed sums on the host; `stack`: the former (3, M0, M1, M2) stack, bit for bit |
| `PARSEC_ORBITAL_GROUP_MAX_COST` | `0.75` (since 2026-11-18): a non-abelian orbital group takes its weighted sectors where their cost (rows x states) is at most this fraction of the abelian route's (`orbital_group_plan`): D4h, D2d, C4v and D4 molecules (plan ratio 0.75) now take merged E partners of weight 2 (D2d nanodiamond of 533 atoms: SCF -16 to -17 %, the energy of the abelian route to 1.5e-10 Ry at tight convergence; a metallic D4h Na cluster -22 % per iteration). C4h, S4 and D3d gain nothing at any value. A one-rank-per-node MPI run whose ranks cannot divide the weighted sectors keeps the abelian group, with a note | `0.5`: the former routes (only plans at half the abelian cost: Td, Oh, T, Th) |
| `PARSEC_SECTOR_OVERHEAD` | `auto` (default, since 2026-11-21): the weighted plan of an unpolarized periodic cell in one process whose basis (grid points x states) is at most 2^21 entries counts a fixed cost of 2^21 entries for every solved sector (`orbital_group_plan(sector_cost=)`, `driver._periodic_sector_cost`): a sector costs an SCF iteration of one A100 about 10 ms whatever its size (7-16 ms measured on eight cells of 2-8 atoms and three of 64), the filter work of about 2^21 grid points x states at 5 ns each. Such a cell solves 2-4 larger sectors of a smaller subgroup instead of 5-16 (Si8 5 -> 2, fcc Al4, MgO8, NaCl8 10 -> 2, tetragonal Si8 10 -> 3, orthorhombic Si8 16 -> 4, rutile TiO2 6 -> 2, bcc Fe2 16 -> 2), with the same full group G (forces and moved atoms averaged over it), the same route (the comparison with the abelian group is that of the plan of least rows x states) and the energies and forces of before within the SCF criterion (1e-8 Ry and 2e-6 Ry/bohr, the printed digits; the first BFGS structures of moved rutile agree to 1e-8 Ry). These cells keep sectors where they compute forces, relax or run `--symmetry on` (an SCF alone takes the full grid below `PARSEC_PERIODIC_SECTOR_FLOOR`). Total wall time on one A100, second run, this switch -> `off` -> the full grid: forces of Si8 1.05 / 1.26 / 0.96 s, Al4 1.09 / 1.81 / 0.75, MgO8 1.30 / 2.12 / 1.07, NaCl8 1.35 / 2.11 / 1.15, tetragonal Si8 1.20 / 1.78 / 1.06, orthorhombic Si8 1.67 / 2.44 / 1.03, TiO2 1.28 / 1.94 / 1.23, Fe2 1.22 / 2.33 / 1.13; three BFGS steps of moved TiO2 3.49 / 4.95 / 2.92 s. Spin-polarized runs (and non-collinear or spin-orbit ones) keep the plan of least rows x states under every value of the switch (since 2026-11-21): the guard of the spin sectors (`SCF/spin_sectors.py`) counts a partly filled multiplet only where its partners lie in two or more sectors or in a state of weight above one, while the weighted sectors average each channel density over G, and the plans of fewer sectors hold the partners of most multiplets of G as several states of one sector of weight 1 (bcc Fe2: the subgroup of E and the body-centring translation, weights 1 and 1; the same for the fcc, rock-salt, perovskite, CsCl, diamond, tetragonal and rutile cells measured on the CPU), where the guard cannot fire. Spin-polarized bcc Fe2 at 100 K then stayed in the symmetric state, 0.125 Ry above the full grid (-79.42178923 against -79.54679865 Ry; fcc Fe4 0.040 Ry), which the plan of before reaches through its guard. First runs (fresh CuPy cache): 12.6-14.0 s against 13.3-14.7 s and 12.3-12.9 s; in two more first runs each TiO2 13.1 / 13.3 against 14.1 / 14.4 s (a first one 14.9 against 13.9 s), orthorhombic Si8 13.7 / 13.9 against 14.4 / 14.9 s, Al4 13.7 / 13.0 against 13.4 / 13.8 s. The SCF takes more steps in fewer sectors (Si8 11 against 8, the full grid 12): the many small sectors of the plan of before solve about twice the states the SCF needs, a buffer of 6 per sector, which shortens their SCF. On larger cells that outweighs the fixed cost, so `auto` stops at 2^21: an integer applies that cost per sector to every cell, which on 64-atom cells (2x2x2 of Si, diamond, MgO, NaCl, SrTiO3, fcc Al; 4 sectors instead of 10-20) is faster in six (MgO 5.00 -> 4.16 s, SrTiO3 6.26 -> 4.70 s, fcc Al 3.68 -> 2.12 s, second runs, the floor off) and slower for Si 2x2x2 (24 instead of 10 SCF steps, 2.63 -> 3.19 s). MPI runs keep the plan of least rows x states (their ranks divide the sectors); a wrong value is refused at setup. The tests of `acceleration/tests` run with it `off` unless a test sets it | `off`: the plan of the least rows x states, as before |
| `PARSEC_PERIODIC_TRIVIAL_ORIGIN` | `auto` (since 2026-11-18): a periodic cell whose abelian group about the origin is the identity (a supercell of a P1 or low-symmetry cell) still looks for its group: its half-cell translations or other operations give point-group sectors (P1 2 x 1 x 1: 2 operations, 2 x 2 x 2: 8) | `off`: one sector of the identity or the full grid, as before |
| `PARSEC_PERIODIC_SYMMETRY_CENTRE` | `auto` (since 2026-11-18; with `PARSEC_PERIODIC_TRANSLATIONS=auto` only): a supercell whose symmetric site is not at the origin or a half-cell point takes its operations about that site (atoms and bond midpoints searched, since 2026-11-19 also the centre of a defect the translations repeat, `PARSEC_PERIODIC_DEFECT_CENTRE`; r -> R r + (I - R) x0 + h in whole grid steps; also in the k-point sectors through `CellGroup`): diamond or Si supercells moved off an atom 16 -> 192 operations (2 x 2 x 2), 4 -> 96 (3 x 3 x 3), 8 -> 192 (4 x 4 x 4); moved Si 3 x 3 x 3 on one node 22.9 -> 11.1 s (full grid 1569 s) | `off`: the operations about the origin, as before |
| `PARSEC_PERIODIC_DEFECT_CENTRE` | `auto` (since 2026-11-19; with `PARSEC_PERIODIC_TRANSLATIONS=auto` only): a translation search of the periodic detection that gives up (more than 32 candidates pass its sampled atoms: a defect off the origin of a cell of 3^3 diamond cubes or more) takes its candidates from the atoms of the rarest environment of the cell (at most 32; every operation carries one of them onto another, so the answer is that of a search of every candidate, `axis_reflection._seeded_translations`; the environment of an atom with the fewest alike within four atom tolerances, so positions off by less than a loose `PARSEC_SYMMETRY_ATOM_TOLERANCE` do not hide the defect: diamond 3^3 V0 at (a, a, a) with atoms up to 1e-5 bohr off at 1e-4 keeps Td, where classes rounded to 1e-4 bohr lost it for 7 of 10 noise seeds; 5^3 V0 so moved and so noisy, A100: SCF 48.5 s under `off` -> 5.3 s, 0.6 s for the environments of 22^3 cubes), and a cell whose pure translations repeat a defect also tries the centre of that environment (an empty site), matched in full. The defect keeps the group it has at the origin, moved with it, on every route that takes the Gamma detection or `CellGroup` (weighted and abelian Gamma sectors, spin and hybrid Gamma cells, the k-point sectors and their guard; the k-point wedge and force wedge of the reference routines found it already), and `symmetry_detection` names the site. One A100 node, `main.py`, make_diamond settings: V0 in 5^3 cubes (999 atoms) at (a, a, a) 2 -> 4 sectors (Td in D2d), SCF 46.8 -> 5.4 s (5.2-5.3 s at the origin), GPU 0 peak 14.2 -> 1.4 GiB, energy 8e-7 Ry from the origin run; V- (spin) 73.8 -> 26.3 s (22.8-24.1 s at the origin: the same D2 sectors moved with the vacancy take 15 iterations instead of 10 at both sites off the origin, through the route of the sectors with translations rather than the sign changes about the origin), 18.3 -> 4.2 GiB, energy 1.3e-5 Ry from the origin run (criterion 2e-4; the two sites off the origin within 1e-10 Ry of each other); V0 3^3 (215 atoms) 4.7 -> 2.3 s; V0 at (a/2, a/2, 0) of 4^3 12.4 -> 4.3 s; V0 3^3 with a 2x2x2 mesh 8 -> 12 k-point tasks, SCF 8.8 -> 7.7 s. Perfect crystals and searches that do not give up keep their operations bit for bit (diamond 4^3: the same parsec.out but the timings); the detection of a perfect 16^3 or 22^3 cell takes at most 0.04 s more | `off`: the searches give up as before (a vacancy at (a, a, a): the 6 operations of C3v about the origin; at (a/2, a/2, 0): one mirror) |
| `PARSEC_PERIODIC_ABELIAN_TRANSLATIONS` | `auto` (since 2026-11-18): a periodic cell whose orbitals keep an abelian group (a hybrid, `occup.in` with `PARSEC_ORBITAL_GROUP_OCCUPATIONS=off` or on several MPI ranks under `PARSEC_ORBITAL_GROUP_MPI=off`, `Fix_Number_Eigenvalues` with `PARSEC_ORBITAL_GROUP_FIXED_COUNT=off`, spin with `PARSEC_ORBITAL_GROUP_SPIN=off` or on several ranks under `PARSEC_ORBITAL_GROUP_MPI=off`, a run that left the weighted sectors, `PARSEC_ORBITAL_GROUP=abelian`, a plan that gains too little) takes the largest group of commuting involutions with translations (Si 2 x 1 x 1: 4 -> 16 operations, 2 x 2 x 2: 8 -> 32; Si 4 x 4 x 4 abelian 74.0 -> 37.1 s) | `off`: the signed permutations about the origin, as before |
| `PARSEC_SYMMETRY_ATOM_TOLERANCE` | `1e-7` bohr: the distance within which an image of an atom is that atom in the symmetry detection of clusters and cells (since 2026-11-18 a switch; positions are not symmetrized unless `PARSEC_SYMMETRIZE_POSITIONS=on`, so a larger value alone gives approximate results for inputs that are not exactly symmetric; since 2026-11-21 a structure that keeps the group only to more than 1e-6 bohr takes the projectors of every atom, `PARSEC_SECTOR_ORBITS_EXACT`; forces in sectors and the moves of a structure with a detected group (a cluster also where its orbitals take the full grid, `Spin_Symmetry_Sectors: off` or a spin `Restart_Run` of main.py) still check atoms to 1e-6 bohr: since 2026-11-19 a run with forces, a relaxation or molecular dynamics whose group holds the atoms only beyond that is refused after its preparation and before its SCF, by the checks of its forces and moves themselves, `Forces/sector_symmetry.py::sector_force_tolerance_refusal`, where they stopped it with a `ValueError` after the whole SCF (Ge5 within 4e-5 bohr of Td at 1e-4: refused 4.1 s after the start instead of after its 11 SCF iterations); a tolerance of 1e-6 or less is not checked, as every operation taken then holds the atoms within it). The report names operations that hold within 1e-3 bohr but not within the tolerance, and advises the tolerance only to runs without forces where they need more than 1e-6 bohr. Since 2026-11-19 the k-point sectors and the `kpoint` route of `PARSEC_GAMMA_HYBRID_SECTORS` take it as well (they kept 1e-7 before; `kpoint_sectors.kpoint_sector_atom_tolerance`): at most 1e-6 bohr in a loop with forces, and their guard tests only the operations beyond the density group that hold the atoms to 1e-6 bohr (as the Gamma guard); the default 1e-7 gives the same bits, and `PARSEC_KPOINT_SECTOR_TOLERANCE=fixed` keeps the 1e-7 of before there whatever this says. Both command lines read it (and the other switches of the periodic detection and the k-point sectors) at the preparation of a periodic run, so a bad value is refused before the run, also on the k-point route of the reference solver | any value up to 0.1 |
| `PARSEC_KPOINT_SECTOR_TOLERANCE` | `follow` (since 2026-11-19): the cell group of the k-point sectors and of the `kpoint` route of `PARSEC_GAMMA_HYBRID_SECTORS` is built at `PARSEC_SYMMETRY_ATOM_TOLERANCE` (at most 1e-6 bohr in a loop with forces), as the Gamma detection; unset, that is the 1e-7 of before, the same bits. With a looser tolerance the k-point sectors solve the Hamiltonian averaged over the group, as the Gamma sectors do. Diamond 3^3 V0 at (a, a, a) with its atoms up to 1e-5 bohr off, tolerance 1e-4, 2x2x2 mesh (8 irreducible k-points: the PARSEC 1.5 wedge finds no symmetry at its tolerance), `main.py` on one A100 node: SCF 115.7 s and 32 iterations under `fixed`, 16.8 s and 13 iterations under `follow`, GPU peak 6.2 -> 3.3 GiB, energies 2.3e-6 Ry apart (the exact vacancy at the origin: 7.7 s, 3.3e-6 Ry from `follow`) | `fixed`: 1e-7 bohr whatever `PARSEC_SYMMETRY_ATOM_TOLERANCE` says, as before 2026-11-19, bit for bit |
| `PARSEC_SYMMETRIZE_POSITIONS` | `off` (default): the positions of the input as given. `on` (since 2026-11-19): before anything is prepared, the positions are replaced by their average over the group of operations that map every atom within `PARSEC_SYMMETRY_ATOM_TOLERANCE` onto an atom of its species (`Forces/position_symmetry.py`; a cluster: the signed permutations about the centre of its box; a bulk cell: `r -> R r + t` modulo the lattice, the translations of a supercell included; wires and slabs are kept), so that an input rounded to a few decimals gets its group exactly: the sectors, the forces (checked to 1e-6 bohr) and a relaxation then take it. parsec.out names the order of the group and the largest move; operations found within a loose tolerance that form no group are refused before the SCF. The average of a bulk cell floats as a whole by the mean of the noise; it is then shifted as a whole (by at most the tolerance) so that every point operation whose translation lies within 4x the tolerance of one that maps the grid onto itself has that translation exactly (the operations about the origin or a site of the grid that the sectors take; parsec.out gives the shift and the number of such operations). The pure translations of a supercell are averaged from the image tables of their generators alone (memory linear in the atoms). On main.py, the reference CLI and the full-SCF runner (every rank alike). The search and the average on the CPU of an A100 node (diamond with 1e-6 bohr of noise, tolerance 1e-5; time, then peak memory of the Python heap): 6^3 (1,728 atoms, 41,472 operations) 0.6 s / 2.2 MiB, 10^3 (8,000) 1.7 s / 6.1 MiB, 20^3 (64,000) 6.9 s / 48 MiB, 22^3 (85,184, 2,044,416 operations) 8.4 s / 64 MiB, each the perfect crystal within 2e-13 bohr; the first version of 2026-11-19 kept a table of every translation (165 MiB at 4,096 atoms, growing as the square of the atoms, tens of GiB at 20^3) and left the cell shifted, so that Si 2x2x2 (64 atoms, 1e-5 bohr of noise, tolerance 1e-4) stopped in the sector forces (an operation of the sectors carried an atom onto no atom); it now runs them (192 operations), within 1e-8 Ry/bohr and 2e-11 Ry of the full grid; the displaced Si cube with that noise has the forces of the full grid within 2e-6 Ry/bohr (test); the Ge5 cluster with its outer atoms' z scaled by 1+1e-5 (Td within 4e-5 bohr, whose forces stopped with a ValueError under `PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-4` alone) gets Td and the forces of the full grid of the same positions within 2e-6 Ry/bohr. | `on` |
| `PARSEC_STANDARD_ORIENTATION` | `off` (since 2026-11-19, opt-in): a cluster is calculated as PARSEC places it. `on`: the input reader turns and moves the cluster into the frame and about the centre of its largest group of signed permutations (largest group of commuting involutions first, then the whole group), symmetrizes the atoms exactly in it, and reports coordinates, forces, the dipole, `atom.cor` and `relaxation.xyz` back in the frame of the input; the grid, its files and `relax_restart.dat` (which records the frame) stay in the calculation frame (`Input/standard_orientation.py`; see "Standard orientation of a cluster"). Nothing changes where the group does not grow. Measured on one A100 node (4 GPUs; first runs, no symmetry cache): methane rotated by 10 degrees, by an arbitrary rotation with an offset, or 0.05 bohr off the origin (h 0.25 bohr) is calculated in the weighted D2d sectors of the aligned input with its energy to the last printed digit and its forces turned to 8e-7 Ry/bohr, in 9 SCF iterations instead of 11-14; C87H76 rotated (Td, h 0.4 bohr, 237 states): SCF 7.38 -> 2.02 s (aligned input 1.84 s), energy within 6e-8 Ry and forces within 5e-6 Ry/bohr of the aligned input; SF6 rotated: Oh with the D4h sectors, energy to the last digit, forces to 5e-7 Ry/bohr (SCF 1.50 -> 1.73 s: eight sectors of a seven-atom molecule cost more than two). A BFGS relaxation of a stretched rotated methane follows the aligned one to 3e-6 bohr (4 structures, 6 without the switch); the water example rotated takes 7 structures like the aligned input (9 without), ending within 7e-6 Ry. The MPI runner gives the energies of main.py. Off by default: it moves the atoms on the grid (egg-box effect) and by the symmetrization | `off` |
| `PARSEC_STANDARD_ORIENTATION_TOLERANCE` | `1e-3` bohr (since 2026-11-19): with `PARSEC_STANDARD_ORIENTATION=on`, the distance within which an operation must carry every atom onto an atom of its species to be kept; the atoms are symmetrized over the group found and move by up to about this distance (reported in parsec.out) | any value above 0 and up to 0.01 |
| `PARSEC_PERIODIC_CA_XC` | `auto` (since 2026-11-18): CA-LDA and CA-LSD of periodic cells (bulk on the orthorhombic grid) on the device: on the orbits of the sectors at Gamma, on the full grid in the k-point loops and the periodic spin drivers (`V_xc/device_xc.py`, the host formula line by line, no fused multiply-add). Diamond 20^3 at Gamma with CA on 16 A100: 151.7 -> 131.1 s, XC 2.44 -> 0.011 s per step, the energy within 1e-10 Ry | `host`: NumPy over the full grid every step, as before (`PARSEC_CUPY_XC=off` too) |
| `PARSEC_PERIODIC_SECTOR_BANDS` | `auto` (since 2026-11-18): a Gamma bulk cell with `bandstruc` runs its SCF in the sectors of its point group; the bands take the expanded converged potential, the lowest `States_Num` levels and their Fermi level by the full-grid formula (Si8: bands within 9.2e-11 Ry of the full grid; diamond 5^3, which ran out of memory in the band stage before, completes) | `off`: the whole SCF on the full grid, as before |
| `PARSEC_SYMMETRIZED_DENSITY_SECTORS` | `auto` (since 2026-11-19): `Symmetrize_Charge_Density: .true.` keeps the symmetry sectors of an unpolarized bulk cell at the Gamma point that the route of the sectors takes (the parser still takes the Fortran-derived routines for it, as since 2026-11-08; `cli._symmetrized_cell_in_sectors`): the density of every iteration is averaged over the parser's space group on the orbits of the sectors (`SCF/density_symmetry.py`, the identity alone where PARSEC 1.5 keeps no group, as for a centred cubic cell), after a check on a sample of rows that the operations of the sectors map that group onto itself (refused by name otherwise); and the two channels of a spin-polarized cluster stay in the sectors, each averaged over the point group on the orbits (`_spin_sector_refusal`, `SCF/spin_polarized.py`). Diamond Gamma cells with `Symmetrize_Charge_Density: .true.` (benchmark settings, one node of 4 A100, first runs; PARSEC 1.5 keeps the identity for these centred cubes): 2^3 (64 atoms) 36.4 -> 13.9 s per process (SCF 30.5 -> 1.5 s), 3^3 (216 atoms) 480.4 -> 12.8 s (SCF 473.0 -> 1.6 s), energies within 1.6e-6 Ry (criterion 2e-4 Ry); CsCl Si/Ge (48 operations of the parser, the only one of these cases with a group beyond the identity) and the N atom (spin) within 2e-6 Ry of the reference; the Si cube with 24 states within 5e-9 Ry of the reference solver (9 and 14 iterations). With the 20 states it had before 2026-11-21 the reference solver and the full grid of the accelerated one did not converge it (the residual stays near 2e-4 for 200 iterations, on both sets of routines and with or without the symmetrization; the sectors, which split the six-fold level over their irreps, did): they cut the six-fold lowest conduction level, which `Fermi_Temp: 300` occupies at 1e-4 per state, after four of its states, and the eigensolver returns another part of it every iteration; 22 states or 80 K converge in 15-17 iterations, and an SCF loop that ends so says why (`Output/state_truncation.py`). | `off`: such a cell runs on the reference solver and such a spin cluster on the full grid, as before |
| `PARSEC_PERIODIC_IMAGE_ROWS` | `auto` (since 2026-11-21): the density group of a periodic cell (`Symmetrize_Charge_Density`: the irreducible wedge of a mesh on either set of routines, a weighted list, a Gamma cell PARSEC keeps a group for; `Grid/space_group.py`, `periodic_grid_symmetry`) holds no table of operations x points (int32: 48 operations of 10^7 points are 1.9 GB on the host of every rank) where every operation is an integer affine map of a row-major grid of at least 2^17 points and the averages run on two threads or more (the periodic setup's, `PARSEC_PERIODIC_SETUP_THREADS`, else `OMP_NUM_THREADS`; `AffineImageRows`). The rows are formed where they are read (the orbit table of the Gamma sectors and the invariance test of the k-point sectors at the points they need, the images of a hybrid's mesh one row at a time); the host averages of scalar and vector fields (the density; the orbital density gradient of a periodic meta-GGA) gather the field at the images through broadcast index vectors, by slabs of the grid on the kept threads of the periodic setup, adding the operations in the same order and laying out the gathers as the table's; an oblique operation (a row of its matrix reads all three lattice axes: 36 of the 48 operations of a cubic crystal in the primitive fcc cell), whose index vector would be an int64 array of every point, keeps its int32 row of the table instead, so no cell holds more than the table. A smaller grid (a table of at most 24 MiB) and a run on one thread keep the table, whose gathers are faster there. The same fields bit for bit (CPU tests on cubic, tetragonal and oblique fcc grids, scalar and vector averages and gathers, threads 1-8; equal hashes in every A100 case below; a k-point SCF on one CPU worker repeats its energy and density to the bit). The device average formed its indices from the affine maps already (`SCF/periodic_symmetry.py`), so the GPU peaks do not change. A100 node, 16 threads, `off` -> `auto`, what the group holds after its averages, one steady scalar average, one vector average (3 components) and the peak RSS of the process: the diamond cube on 134^3 points (2.4 M, 24 operations) 220.3 -> 0.1 MiB, 0.206 -> 0.048 s, 1.068 -> 0.605 s, 0.77 -> 0.58 GiB; diamond in the primitive fcc cell on 136^3 (2.5 M, 48 operations, 36 oblique) 460.6 -> 345.5 MiB, 0.460 -> 0.076 s, 2.359 -> 1.331 s, 1.02 -> 1.01 GiB (building the group 0.750 -> 0.032 s, the first average, which forms the 36 rows, 0.467 -> 0.579 s), on 104^3 (1.1 M) 206.0 -> 154.5 MiB, 0.138 -> 0.024 s, 1.059 -> 0.408 s, on 68^3 (314 k) 57.6 -> 43.2 MiB, 0.035 -> 0.019 s, 0.315 -> 0.145 s; a bcc cell on 116^3 (1.6 M, 48) 285.8 -> 7.5 MiB, 0.223 -> 0.037 s, 1.450 -> 0.745 s, 0.67 -> 0.42 GiB; an hcp cell on 108 x 108 x 176 (2.1 M, 24) 187.9 -> 1.5 MiB, 0.145 -> 0.040 s, 0.934 -> 0.485 s, 0.67 -> 0.51 GiB; on one thread `auto` keeps the table (the times of `off`), where the rows (`on`) would take 0.305 s and 1.403 s for the cube of 2.4 M points and 0.349 s for the bcc cell. Diamond n^3 cubes on the unshifted default grid of 16 points per edge (all 48 operations), one steady host average and the peak RSS: 4^3 (262,144 points, a 0.05 GiB table) 0.035 -> 0.034 s, 0.17 -> 0.13 GiB; 8^3 (2.1 M, 0.38 GiB) 0.68 -> 0.064 s, 0.67 -> 0.31 GiB; 16^3 (16.8 M, 3.0 GiB) 4.47 -> 1.31 s, 4.63 -> 1.75 GiB, building the group 0.95 -> 0.22 s; the device average 23-48 ms either way. Whole first runs on 4 A100 (`main.py`, the same SCF steps, the totals within 7e-15 Ry, the spread of GPU runs): Si in the primitive fcc cell with SCAN, the 2x2x2 mesh on 80^3 points (scalar and gradient averages every iteration) 239.2 -> 225.9 s, host peak 14.99 -> 14.51 GiB; Si with PBE on 100^3 points 91.4 -> 87.3 s, 9.75 -> 9.51 GiB; rocksalt MgO with PBE and the 3x3x3 mesh on 92^3 points 83.6 -> 80.4 s, 8.22 -> 8.06 GiB; diamond 4^3 with the 2x2x2 mesh 41.0 s and 2.86 GiB either way; GPU peaks the same. Measured before the oblique rows and the kept threads (the route of a cube is the same): the 8-atom cube on the Fortran-derived routines with the 2x2x2 mesh on 216^3 points (48 operations, a 1.93 GB table) 585.3 -> 530.6 s, host peak 26.3 -> 24.4 GiB; the 64-atom cube on 152^3 points 125.9 -> 119.6 s, 10.5 -> 9.8 GiB; diamond 8^3 cubes (4,096 atoms) with the 2x2x2 mesh in the MPI k-point pools (`run_mpi_kpoints.py`, 4 ranks, the 24 operations of the grid on 144^3 points) 178.5 -> 172.5 s, host peak per rank 6.1-6.5 -> 5.7-5.8 GiB; Gamma 8^3 and 10^3 with `Symmetrize_Charge_Density` (whose supercells PARSEC 1.5 keeps at the identity) as fast as before. Slabs take the rows under the same rule since 2026-11-22 (`confined_grid_symmetry`, `_confined_affine_maps`: their points fill the surface cell times a range of planes, numbered in the row-major order of their reversed indices, so every operation is an integer affine map of that box; the same integers): hBN on 58 x 58 x 302 points (1.0 M, 12 operations), A100 node, 16 threads, `off` -> `auto`: building the group 3.49 -> 0.047 s, 46.2 -> 0.2 MiB held, a scalar average 0.024 -> 0.012 s, a vector one 0.287 -> 0.087 s, peak RSS 0.42 -> 0.32 GiB; on 38 x 38 x 202 (289 k) building 1.00 -> 0.014 s, the averages as fast; the same hashes. Wires (a cylinder of points) and grids whose operations are no such maps keep the table. `on`: the rows at every size and thread count, and since 2026-11-22 no row of an oblique operation either: its gathers form the indices of a few planes at a time from one plane of each index component (`_ObliqueFactor`; the same integers), the least memory, slower than `auto` where oblique operations are averaged: diamond in the primitive fcc cell, 16 threads, `auto` -> `on`: 2.5 M points (36 oblique of 48) 345.5 -> 2.7 MiB held, peak RSS 1.01 -> 0.62 GiB, a steady scalar average 0.074 -> 0.223 s (the table: 0.467 s), the first 0.592 -> 0.241 s, a vector average 1.313 -> 1.575 s; 0.83 M points (18 of 24) 57.0 -> 0.7 MiB, 0.013 -> 0.042 s; cubic, bcc and hcp grids (no oblique operation) the same times as `auto`; on one thread the rows of every grid (0.304 s for the 2.4 M cube against 0.207 s with the table), the same hashes. Whole first runs on 4 A100 (Si in the primitive fcc cell, the 2x2x2 mesh, the density averaged on the host every iteration; energies within 4e-15 Ry): PBE on 100^3 points `off` 76.5 s and a host peak of 10.68 GiB, `auto` 72.3 s and 10.50 GiB, `on` 75.7 s and 10.50 GiB; SCAN on 80^3 points `auto` 210.5 s and 14.59 GiB, `on` 222.9 s and 15.00 GiB: the rows of the oblique operations do not set the host peak of such a run, so there `on` saves nothing and costs 5-6 % | `off`: the table of every grid built at the preparation, as before |
| `PARSEC_BAND_KPOINT_SECTORS` | `auto` (default, since 2026-11-19): the band k-points after a Gamma SCF in the sectors of its point group (`PARSEC_PERIODIC_SECTOR_BANDS`) are solved in the sectors of their little groups: the operations of the converged potential (the full group of weighted sectors) that fix the k-point, split by the machinery of the k-point sectors (`SCF/band_sectors.py`, `SCF/bands.py::solve_kpoint_bands` with `symmetry`), as many k-points at a time as the devices, each solver closed before the next; a k-point below the floor of the k-point sectors (2^24 entries) or whose sectors would not halve the work is solved whole. One node of 4 A100, diamond cubes with the path G-X, G-R (9 k-points), band stage and peak per GPU: 3^3 (216 atoms, `Bands_FF_MaxIter` 15) 120.8 -> 24.4 s, 5.3 -> 1.7 GiB; 4^3 (`Bands_FF_MaxIter` 10) 504.4 -> 36.0 s, 21.6 -> 2.9 GiB; 5^3 (default cycles) 604.5 -> 65.8 s, 74 -> 16 GiB; 6^3 (1,728 atoms), which ran out of memory, 81.0 s and 20 GiB. The occupied bands agree within 1.3e-10 Ry where the cycles converge them (3^3, 4^3); the empty buffer states differ within their residuals, and at the default cycles of 5^3 neither route converges the block (largest residual 0.10 Ry in the sectors, 0.31 Ry whole; converged since 2026-11-21 by `PARSEC_BAND_CONVERGENCE`) | `off`: every k-point on the whole grid, as before |
| `PARSEC_BAND_CONVERGENCE` | `auto` (default, since 2026-11-21): the CHEBFF first solve of every band and `nscf` k-point (`Eigensolver: chebff`) is a converged one: after the `Bands_FF_MaxIter` cycles the CHEBFF cycles go on until the residuals of the reported bands pass CHEBDAV's relative test with the input's `Diag_Tolerance` (times their largest `abs(e)`; 1e-4 where the input gives none or CHEBFF's -1) or `PARSEC_KPOINT_EXTRA_CYCLES` (24; a value that is not a non-negative integer is an input error with a bandstruc block or `nscf_kpoints`) more cycles were made, on the host (`Eigensolvers/chebff.run_chebff`, `converged`), on the device workers and in the sectors of the little groups (the device-resident first solve of `Eigensolver: chebdav`, `converge_first`); a k-point that passes after the fixed cycles keeps its levels bit for bit (`Eigensolvers/eigval.ConvergedFirstSolveSettings`, `SCF/bands.converged_band_settings`). PARSEC 1.5's `calc_bands` runs `Bands_FF_MaxIter` cycles without a test at the CHEBFF default `Diag_Tolerance -1`. One node of 4 A100, the path G-X, G-R (9 k-points in the sectors of their little groups), `Eigensolver: chebff` with `FF_MaxIter` 3 and `Diag_Tolerance` 1e-4, band stage and largest residual off -> auto: diamond 3^3 (216 atoms) 7.7 -> 18.6 s, 0.171 -> 2.0e-3 Ry; 4^3 15.3 -> 43.0 s, 0.156 -> 2.7e-3 Ry; 5^3 (1000 atoms) 45.9 -> 281.4 s, 0.102 -> 4.1e-3 Ry; silicon 3^3 (216 atoms) 14.4 -> 35.4 s, 0.179 -> 2.6e-3 Ry (the CHEBDAV first solve of `Eigensolver: chebdav` 35.0 s, its bands within 2.6e-5 Ry of auto's). Under off the occupied bands are off by up to 2.8e-3 Ry and the empty ones by up to 7.6e-2 Ry; auto gives those of 15 and 10 fixed cycles (3^3, 4^3) within 5e-15 Ry (occupied) and 1e-10 Ry (empty). These numbers are of the test of every state of a sector (`PARSEC_BAND_REPORTED_RESIDUALS=off`), whose margin states (`PARSEC_KPOINT_SECTOR_MARGIN`), which the bands do not report, take the cycles to the cap on 5^3. Since 2026-11-22 a k-point whose largest residual still misses the test (`Diag_Tolerance` times the largest `abs(e)` of its levels) is named in a WARNING of the band summary with the largest residual (`SCF/bands.py::unconverged_band_points`), where its solve applied that test (the CHEBFF-based first solves); before, only residuals above 1e-2 Ry were warned of, which the fixed cycles of off and the host CHEBDAV (whose own test is relative to its running Ritz values) keep. The `nscf` output prints the residuals of its k-points without this WARNING. | `off`: the fixed `Bands_FF_MaxIter` cycles, as before (and PARSEC 1.5) |
| `PARSEC_BAND_BUFFER` | `auto` (default, since 2026-11-22): the CHEBFF-based converged first solve (the device-resident one of `PARSEC_BAND_CONVERGENCE` or `Eigensolver: chebdav`, or the host CHEBFF) of a band or `nscf` k-point solved on the whole grid takes a working buffer of 60 % of its states (`SCF/bands.py::band_working_buffer`; `Subspace_Buffer_Size` where that is more), within half the memory of one solve for three complex bases (`affordable_band_buffer`, never below `Subspace_Buffer_Size`: the total memory of a device divided by the processes that share it, or the available host memory divided by the worker threads and processes of a host, read once per band stage and the same on every MPI rank, `band_stage_route`); the residual test still covers the reported states only. The host CHEBDAV keeps `Subspace_Buffer_Size`: it converges every working state, so more of them only cost time. With `Subspace_Buffer_Size` (6) alone the first level above the working space lies close to the top reported levels (C64, 160 states: the edge of the 166 states cuts a four-fold level 0.029 Ry above the top reported one, against a spectrum of hundreds of Ry), so the CHEBFF cycles gain a factor of about 1.2 a cycle and stop at the cap (`PARSEC_KPOINT_EXTRA_CYCLES`) with residuals up to 9e-3 Ry, also at the default `Diag_Tolerance`. The k-points that split into the sectors of their little groups (`PARSEC_BAND_KPOINT_SECTORS`) keep their own margin; the k-points of such a path that do not split are solved on the whole grid with this buffer (under off in the solver of the sectors, as before). A number in (0, 2] sets the fraction; on C64 (9 k-points) the band stage takes 43.8 s at 0.1, 41.1 s at 0.2 (one k-point unconverged), 37.2 s at 0.3, 34.5 s at 0.5, 28.6 s at 0.6, 29.0 s at 0.8 and 30.1 s at 1.0. One node of 4 A100, first calculation, the path G-X, G-R (9 k-points), `Eigensolver: chebff`, `Diag_Tolerance` 1e-4, band stage, largest residual and peak per GPU off -> auto: C64 (no symmetry) 46.4 -> 28.5 s, 9.1e-3 -> 1.9e-4 Ry, 2.77 -> 3.21 GiB; the same at `Diag_Tolerance` 1e-10 46.3 -> 52.3 s, 9.1e-3 -> 2.3e-10 Ry (off does not converge); spin-polarized 92.5 -> 55.1 s; after a 2x2x2 k-point SCF 45.6 -> 27.8 s, 2.58 -> 3.70 GiB; diamond 216 atoms with one atom off its site (438 states) 201.1 -> 84.9 s, 9.4e-4 -> 2.1e-4 Ry, 6.55 -> 8.36 GiB; silicon 64 atoms 35.5 -> 20.4 s, 2.6e-3 -> 8.1e-5 Ry; fcc Ag 32 atoms 23.1 -> 7.3 s, 2.4e-3 -> 6.0e-5 Ry; silicon 8 atoms 2.5 -> 2.2 s (both converged; levels 4.3e-6 Ry apart), at `Diag_Tolerance` 1e-10 4.2 -> 3.3 s, 3.2e-5 -> 1.0e-10 Ry. Under off the reported levels of those cases are off by up to 3.9e-3 Ry (C64 3.8e-3, spin 3.2e-3, mesh 3.9e-3, diamond 216 6.9e-4, Ag 5.7e-4, silicon 64 3.6e-4). Host memory within 0.1 GiB of off. The MPI band route (k-points dealt to the ranks) gives the levels of one process within 7e-15 Ry. | `off`: `Subspace_Buffer_Size`, as before and as PARSEC 1.5's `calc_bands` (`nstate + winsize`) |
| `PARSEC_BAND_REPORTED_RESIDUALS` | `auto` (default, since 2026-11-21): the residual test of the converged first solve of a band k-point in the sectors of its little group (`PARSEC_BAND_CONVERGENCE`, and `Eigensolver: chebdav` on the device) covers the levels the k-point reports, not the margin states of its sectors (`PARSEC_KPOINT_SECTOR_MARGIN`), which converge last and are never reported (`SCF/kpoint_sectors.py::SectorKPointSolver.solve_reported`). Which levels those are follows from all the sectors: each sector makes its `Bands_FF_MaxIter` cycles; then each goes on from where it stood until the levels it holds at or below the highest level of its k-point pass; with the levels of all the sectors the test is made again on every pair with `e - r <= cut` (`cut` the highest reported level, `r` the residual), and a sector that misses it goes on, within `PARSEC_KPOINT_EXTRA_CYCLES`. That condition is a heuristic for the pairs that may still belong among the reported levels: it does not prove that no level is missing from a sector. The test asks 1 % of `Diag_Tolerance` (not below 1e-10 unless `Diag_Tolerance` is): at `Diag_Tolerance` itself the levels moved by up to 1.8e-7 Ry from those of the test of every state, at 1 % by at most 1.9e-11 Ry. A k-point that stays whole (no little group to split by, below the floor of the k-point sectors) keeps its one first solve with the test at `Diag_Tolerance`, whatever k-points share its chunk, so the bands do not depend on the number of GPUs. The subspaces stay on the device unless it is short (the rule of `PARSEC_KPOINT_STATE_STORAGE=auto`); then they wait on the host between the parts if 75 % of the available host memory holds them, else the k-points take the test of every state. Their host copies cost host memory of the size of all the subspaces of the chunk: 1000 diamond atoms with `PARSEC_KPOINT_STATE_STORAGE=host` 9.6 GiB per GPU and 40.4 GiB of host memory (139.7 s), on the device 17.8 GiB per GPU and 4.1 GiB of host memory (123.7 s). On the device the peak per GPU is up to 17 % above that of the test of every state (2-17 % on the paths of 9 k-points below), since the largest sector continues while the others hold their subspaces. The largest residual printed (and `residual_norms` of `parsec_python_bands.npz`, with the key `residual_scope`) is that of the reported levels, and the line of `parsec.out` says so. The MPI k-point pools keep the test of every state. One node of 4 A100, first calculation, the path G-X, G-R (9 k-points, all split), 3 first-filter cycles, `Diag_Tolerance` 1e-4, band stage, peak per GPU and peak host memory off -> auto (largest level difference; the fixed cycles of `PARSEC_BAND_CONVERGENCE=off` in brackets): diamond 216 atoms 18.2 -> 12.5 s, 1.75 -> 1.79 GiB, 1.87 -> 1.88 GiB (1.9e-11 Ry; 7.7 s); with a vacancy 31.4 -> 16.7 s, 3.32 -> 3.88 GiB, 2.00 -> 2.23 GiB (1.4e-11; 6.9 s); 512 atoms 43.0 -> 25.7 s, 2.91 -> 3.09 GiB, 2.63 -> 2.67 GiB (1.5e-11; 15.0 s); 1000 atoms 273.4 -> 123.3 s, 16.7 -> 17.8 GiB, 4.24 -> 3.97 GiB (9.8e-12; 53.6 s), where the test of every state stops at the cap with residuals of 4e-3 Ry and this one passes at 2.4e-6 Ry; silicon 216 atoms 35.5 -> 28.4 s, 2.92 -> 3.08 GiB, 3.15 -> 3.22 GiB (5.3e-12; 14.6 s), with `Eigensolver: chebdav` 34.7 -> 28.8 s (5.3e-12); silicon 512 atoms 91.5 -> 69.7 s, 5.30 -> 5.63 GiB, 5.02 -> 5.33 GiB (4.1e-12; 34.9 s); fcc Ag 108 atoms 18.1 -> 13.1 s, 1.64 -> 1.69 GiB, 1.84 -> 1.82 GiB (1.7e-11; 8.1 s). Paths that start at points of high symmetry and run along general lines, where the k-points without a little group stay whole and share the chunks of the split ones (3 of 12 k-points split): diamond 216 atoms with a vacancy 230.7 -> 227.8 s on 4 GPUs, 682.0 -> 670.7 s on one; silicon 216 atoms with a vacancy 488.1 -> 485.9 s; fcc Ag 108 atoms with a vacancy 265.4 -> 261.8 s, 787.3 -> 773.6 s on one; with 7 of 11 split (diamond with a vacancy) 164.8 -> 159.9 s, 349.5 -> 329.3 s on one. The peak per GPU there is the same as before or 3 % above it, the whole k-points keep their levels within 7e-15 Ry, and the bands of 4 GPUs and of one agree within 7e-15 Ry. | `off`: every state of every sector is tested, as before |
| `PARSEC_MPI_ACCELERATED_BANDS` | `auto` (default, since 2026-11-22): under `--mpi` the `bandstruc` of a scalar Gamma-point cell stays on the accelerated path (`Output/post_scf.py`): every rank runs the accelerated SCF of one process, then the chunks of band k-points are dealt to the ranks in turn, each solved by the band solver of one process on the GPUs of its rank (in the sectors of its little group, `SCF/band_sectors.py`, with the test of the reported levels; whole k-points through `SCF/kpoint_parallel.map_tasks`), every rank gathers all the levels, and rank 0 writes the band files (the ranks deal chunks of the smallest of their lengths, so ranks that see different numbers of GPUs deal one partition, and a k-point not solved exactly once stops the run; `--mpi` on one rank takes the solver of one process). Before, such an input ran whole on the reference solver: its SCF on the MPI k-point pools (the full grid, another route) and the band levels to `Diag_Tolerance` on every state. One node of 4 A100, diamond 216 atoms with the path G-X, G-R (9 k-points), first runs: four ranks `off` -> `auto` 668.8 -> 9.6 s (band stage 182.6 -> 7.0 s), SCF energy 1.65e-6 Ry from one process -> the same bits, band levels up to 3.4e-5 Ry from one process with a largest residual of 9.8e-4 Ry -> the levels of one process on one GPU bit for bit (6.4e-15 Ry from those of one process on four GPUs), residual 2.2e-6 Ry; one process 16.1 s on four GPUs, 22.0 s on one. fcc Ag 108 atoms, the same path: four ranks 1463.9 -> 11.2 s (SCF 1351 s on the pools; band stage 111.7 -> 6.6 s), SCF energy 9.5e-8 Ry and levels 4.2e-4 Ry from one process with a largest residual of 2.8e-3 Ry -> those of one process bit for bit (on one GPU and on four), residual 3.1e-6 Ry. The former difference was that of the band solver: from the same potential the whole k-points of the reference route (every state tested at `Diag_Tolerance`) differ from the levels in sectors by 3.5e-5 Ry on the diamond cube, alike in one process and on four ranks (`PARSEC_BAND_KPOINT_SECTORS=off`), while the occupied levels agree within 1.3e-11 Ry. Every rank runs the whole SCF on its GPUs and the ranks of one node solve their k-points side by side; the device and host storage of the band stage follows the rules of one process (`PARSEC_KPOINT_STATE_STORAGE`). Peaks on one node, one process on four GPUs against four ranks of one GPU: diamond 1728 atoms 20.5-22.4 against 19.8-22.4 GiB per GPU, host 28.6 against 33.6 GiB (band stage 161 against 187 s); 2744 atoms 55.6-55.9 against 63.3-73.6 GiB per GPU, host 318 against 303 GiB (1105 against 1328 s, the same levels within 1.1e-14 Ry), on nodes of 512 GiB; on nodes of 256 GiB both stop for host memory; the SCF alone of 4096 atoms 1.8-2.2 against 4.7-5.1 GiB per GPU. | `off`: the reference solver under `--mpi`, as before |
| `PARSEC_KPOINT_SECTOR_GUARD` | `auto` (since 2026-11-18): spin-polarized k-point loops in sectors that rely on operations the density is not symmetrized over (a single Gamma point, a full mesh, the translations of a supercell) run the three tests of the cluster spin guard (a partly filled multiplet across sectors in the start Hamiltonian, a channel density that breaks the operations, those of the cell group the sectors assume and, since 2026-11-19, those of the atoms found from their environments, `PARSEC_SPIN_GUARD_CELL_GROUP`; fractional occupations of such a multiplet); a loop that fails runs again with one task per k-point and a NOTE in `parsec.out` (an O atom in a periodic cube: the broken-symmetry state, more than 1 mRy below the symmetric one the sectors held). The inner loops of the Fock loop of a hybrid: `PARSEC_KPOINT_SECTOR_GUARD_FOCK` (since 2026-11-19; untested before) | `off`: the sectors without the tests, as before (also those of `PARSEC_KPOINT_SECTOR_GUARD_FOCK`) |
| `PARSEC_KPOINT_SECTOR_GUARD_FOCK` | `auto` (since 2026-11-19): the inner loops of the Fock loop of a spin-polarized hybrid in the sectors of its k-points (a spin hybrid at the Gamma point through the 1x1x1 driver, k-point meshes; before, only its semilocal start was tested) run tests 2 and 3 of `PARSEC_KPOINT_SECTOR_GUARD` at their end (`SCF/kpoints.py`, `_spin_kpoint_loop`): an inner loop at the final criterion that fails one makes the Fock loop start again from its semilocal start with one task per k-point (`SCF/hybrid_periodic.py`, `run_periodic_hybrid_scf`; its carried mixer and the Ritz states of the sectors cannot be taken back; a NOTE in `parsec.out`), the run of `PARSEC_KPOINT_SECTORS=off` for that structure; a looser inner loop is measured only, and its asymmetry between the tolerance and the noise bound tightens no criterion; `parsec.out` prints what the last tested loop measured (`Test 2/3, last Fock loop`). Tests that pass change no number (H2 in a box, spin PBE0: the same bits; the spin PBE0 of Si64 at the Gamma point keeps no sectors past its start, whose levels have a partly filled multiplet (test 1), 138.3 and 139.2 s with and without on 4 A100, the same total energy). No test can see a state that breaks an operation of `H_k` itself (a hybrid polaron on sites that `H_k` exchanges): the sector orbitals are invariant under `H_k` by construction | `off`: the semilocal start tested alone, as before |
| `PARSEC_KPOINT_SECTOR_LEVELS` | `auto` (since 2026-11-19): the k-point loop with refined levels, the last inner loop at the final criterion of the Fock loop of a hybrid with the adaptive inner criterion or the outer DIIS (`SCFWarmStart.level_tolerance`: k-point hybrids, spin hybrids at the Gamma point through the 1x1x1 driver, `PARSEC_GAMMA_HYBRID_SECTORS=kpoint`), keeps the sectors of its k-points and starts from the Ritz states of the loop before; its level passes solve the sectors, whose union holds the levels of the k-point (`SCF/kpoints.py`, `_sectors_take_levels`). Before, it left the sectors and started from fresh subspaces on whole complex k-points: Si64 with the 2x2x2 mesh, PBE0 on 4 A100 (first runs), the last loop 126.8 s (a cold first solve of 87.7 s, then 14 level passes in 39.1 s) -> 4.4 s (one pass), SCF 367.4 -> 253.4 s, GPU 25.6 -> 24.3 GiB, the same total energy to the printed digits (`E_x` within 3.4e-7 Ry, the convergence of the Fock loop); the 8-atom Si cube on Gamma and X: 3e-8 Ry, whose loop on whole k-points split two partners at X by 3.4e-5 Ry that the sectors hold degenerate | `off`: that loop with one task per k-point from fresh subspaces, bit for bit as before |
| `PARSEC_KPOINT_SECTOR_REFIT` | `auto` (since 2026-11-18): after the first solve of a k-point loop each sector keeps the levels it holds among those of its k-point plus max(6, 10 %), and grows by its deficit later (diamond on 16 A100: 6^3 with a 3 x 3 x 3 mesh, later steps -15 %; 7^3 with 2 x 2 x 2, 86.6 -> 81.8 s; energies within the SCF criterion) | `off`: the first counts kept, bit for bit as before |
| `PARSEC_KPOINT_SECTOR_LDAU` | `off` (default): a k-point loop with LDA+U solves one task per k-point, as before. Opt-in since 2026-11-22: `on` takes the sectors of the k-points for the loops of unpolarized orbitals: the term `P_k D P_k^H` of a k-point joins the Hamiltonian of a sector as `(U^H P_k) D (U^H P_k)^H` (`SCF/kpoint_sectors.py`, `KPointSectors.hamiltonian`; the blocks between two sectors are dropped, the average of the term over the operations of the sectors, as the local potential takes its orbit means), and the occupation matrices of a sector come from its states through the same reduced projectors (`SectorKPointSolver._accumulate`, `SCF/kpoint_hubbard.py`, `KPointHubbard.occupation_with`); `spin` takes the loops of collinear spin as well, with the symmetry guard of the spin loops (a magnetic order held by the term on some atoms is averaged over the operations of the sectors, the reason `PARSEC_LDAU_SPIN_SECTORS` stays off). A100, 2 x 2 x 2 meshes, criterion 1e-6 Ry unless said, `on` against `off`: Ag 32 atoms (U = 1 eV) 5.0e-9 Ry, occupation matrices within 2.5e-7, 13 against 19 steps; Cu 32 (2 eV) 5.2e-8 Ry, 3.4e-7, 14 against 21 steps; SrTiO3 40 atoms (3 eV on Ti) 1.1e-10 Ry, 1.1e-5 (at the criterion 1e-8: 5.8e-12 Ry and 2.6e-8), 27 against 47 steps; Ag 32 at U = 4 eV 8.9e-9 Ry; Ag 108 (1 eV) 3.7e-8 Ry. Time on one GPU 67 against 139 s (Ag 32), 70 against 142 s (Cu 32), 113 against 189 s (SrTiO3); four GPUs of one process: Ag 108 73.5 against 426.4 s (peak 3.3 against 18.9 GiB), but Cu 32 64.5 against 45.6 s and SrTiO3 40 113.3 against 84.5 s, so not the default: the sector loop of such cells does not scale with the GPUs of one process (SrTiO3 40 in sectors without U 43.2 s on one GPU and 43.7 s on four, with U 105.9 and 115.0 s; the summed solve time of the workers grows about four times), while one task per k-point divides its four k-points over the GPUs; the complex arithmetic the term asks of the real sectors costs about 8 %. Fe 54 atoms with spin and U = 3 eV converged on neither route in 100 steps; under `spin` its guard returned it to one task per k-point. LDA+U with `--mpi` stays refused, as before | `on`, `spin` (opt-in, see the left column) |
| `PARSEC_MP_UNSHIFTED_MESH` | `fortran` (since 2026-11-18): a `Monkhorst_Pack_Grid` without a `Monkhorst_Pack_Shift` block is PARSEC 1.5's mesh on both routines: the Gamma-centred grid (i-1)/n reduced to its irreducible wedge (space group and time reversal, as `irrbz`), the density symmetrized (a hybrid: the Fock loop of an insulating single point, `PARSEC_HYBRID_KPOINT_WEDGE`, else the time-reversal pairs, as the notes of `parsec.out` say); it runs as the same input with a zero shift block, without selecting the Fortran-derived routines. On the default routines that is the wedge of scalar orbitals and (since 2026-11-19, `PARSEC_SOC_KPOINT_WEDGE`) of non-magnetic spin-orbit coupling (by the operations that also keep the grid's finite-difference operators, for scalar orbitals since 2026-11-21, `PARSEC_SCALAR_KPOINT_WEDGE`; the points the shortest of their images, `PARSEC_KPOINT_FOLD`); a hybrid single point the time-reversal pairs for its semilocal start and that wedge (of the operations that keep the stencil too) for the Fock loop of an insulating start (`PARSEC_HYBRID_KPOINT_WEDGE`), the time-reversal pairs of the mesh otherwise, molecular dynamics and `manual` movements the whole mesh as its pairs, and non-collinear magnetism and spin-polarized spin-orbit coupling the whole mesh there; the note in parsec.out says which (until 2026-11-19 it called every such mesh a wedge) (the k-point tables of PARSEC 1.5 matched for seven Si8 inputs; diamond 4 x 4 x 4: 10 points, 47.0 -> 10.5 s, the same energy). Even meshes use other points than before | `port`: the former unreduced (2i-n-1)/(2n) mesh, bit for bit |
| `PARSEC_HYBRID_KPOINT_WEDGE` | `auto` (since 2026-11-20): the Monkhorst-Pack mesh of a hybrid single point of a bulk cell (either set of routines, a shift block or none, any grid) runs its semilocal start on the time-reversal pairs, bit for bit as `off`; where that start fills every state to 0 or 1 (`Grid/kpoints.py`, `partly_filled_states`: no occupation `f` with `1e-6 <= f <= 1 - 1e-6`, `HYBRID_WEDGE_FRACTIONAL_TOLERANCE`; an insulator at `Fermi_Temp` 1 K is exactly 0/1, and so is one at the default 80 K whose gap exceeds about 0.014 Ry; at 1000 K a gap below about 0.17 Ry counts, which keeps the pairs, only slower) the Fock loop moves to the irreducible wedge of the mesh, from the representatives' orbitals, levels and Ritz states of that start (`SCF/hybrid_periodic.py`, `BlochExchange.after_start`, `hybrid_wedge_plan`: each star's representative is a point of the pairs with its label; the k-point sectors of the wedge start from the Ritz states the start left in the same k-points' sectors under `PARSEC_HYBRID_WEDGE_SECTOR_STATES=auto`, the default, afresh under `off`), with the density symmetrized over the wedge's operations from then on; a start with a partly filled state keeps the pairs for the whole run, bit for bit as `off` (`parsec.out`: a NOTE after the start names the route and why; the input reader keeps the wedge aside, `SinglePointInput.hybrid_kpoint_wedge_deferred`). The wedge: the operations of the space group that map the grid, the mesh and the grid's finite-difference operators onto themselves, on any grid (`hybrid_wedge_operations`, `stencil_preserving_matrices`: hcp keeps 8 of its 24 operations, the primitive fcc cell 12 of 48; on an orthogonal grid every operation that keeps the mesh), and time reversal; the exchange sums over the whole mesh through the images of the wedge's Bloch factors, `u_k` with its grid points permuted (`V_xc/bloch_exchange.py`, `MeshStars`; host and device engines), as Quantum ESPRESSO's `exxinit` rotates its wavefunctions; on the Fortran-derived routines the density average keeps the whole space group, and the wedge and the images take only the stencil-exact operations that keep the mesh, `mesh_stars`: the exchange of the mesh commutes with those alone. Why the occupations decide: a metal's level partly filled at the Fermi level at a set of equivalent k-points can take, in the Fock loop of the pairs, an occupation that breaks the crystal symmetry (exact exchange favours integer occupations: the same-k exchange goes as f^2/N_k), and that state lies below the symmetric one, which is all the wedge can hold (its star members are images of one representative). A100, PBE0, first runs (2026-11-20): hcp Au2 4x4x2 at 0.3 bohr: the semilocal start is symmetric (a doublet at the two L points at E_F + 0.006 Ry, f = 0.275 each), the pairs move the electron of the doublet into one member at one L point (f 0.991 against 0.006, the two L points 6.2e-2 Ry apart) and end 9.7e-4 Ry below the wedge of the 8 operations (the same at 0.2 bohr and on the Fortran-derived routines; at 0.4 bohr both split the doublet alike at both L points, 4.9e-5 Ry, the discretization); primitive fcc Au 4x4x4 at 0.35 bohr: one of the three X points takes the level (f 0.986 against 0.003), 6.9e-4 Ry; the conventional cubic fcc Au cell (4 atoms, 2x2x2, an orthogonal grid) 5.8e-3 Ry, where neither route converges its Fock loop at `Exx_Tolerance` 1e-8 (the exchange breaks the face-centring translations of the cube too and creeps on through states of that kind; `PARSEC_HYBRID_KPOINT_SECTORS=on` keeps them and converges both, next row). With the breaking removed (hcp Au2 at 4000 K) the wedge is the pairs to the discretization (3.8e-6 Ry at order 6 and 0.3 bohr, 4.4e-7 at 0.2 bohr, 3.3e-8 at order 12: the finite-difference Bloch operator and the edge of the exchange kernel's box are not invariant under k -> k + G, and an image's label `R k` is another than the folded label of the pairs where it lands on a zone boundary), on orthogonal grids too (bcc Au2 2.7e-6 Ry at order 6); exact covariance of the exchange under the stencil-exact operations of the hcp grid (1e-16, CPU). Before 2026-11-20 the gap was put down to oblique grids; it is not theirs: insulators on oblique grids take the wedge (primitive Si 4x4x4 at 0.5 bohr 1e-10 Ry, hexagonal Si 4x4x2 5.4e-8 Ry, its semilocal starts already 1.4e-7 Ry apart). A100, PBE0, first runs, one process each (2026-11-20, `off` / `auto` / `on`, prepare + SCF): the 8-atom Si cube, 2x2x2 with a zero shift block, 0.57 bohr, Fermi 1 K: 15.3 / 10.3 / 8.2 s on the default routines (8, 8 then 4, 4 points), 13.5 / 9.8 / 7.9 s on the Fortran-derived ones, the totals equal to 1e-10 Ry; primitive Si 4x4x4, 0.5 bohr, an oblique grid: 52.0 / 26.3 / 21.2 s (36, 36 then 13, 13 points), 1e-10 Ry; hexagonal Si 4x4x2: 31.1 / 22.4 / 20.3 s (20, 20 then 14, 14), 5.4e-8 Ry; the metals take the pairs under `auto`, the run of `off` (the same objects handed on; on the device the totals agree to the last bits, 1e-14 Ry, the run-to-run spread of `off` itself): hcp Au2 at 0.3 bohr 102.1 / 100.0 s (`on` 56.7 s, 9.7e-4 Ry above), primitive fcc Au 90.7 / 85.6 s, conventional fcc Au with `Exx_Max_Iter` 20 38.7 / 38.2 s (the same total to the bit). The insulator's start on the pairs costs the difference between `auto` and `on` (2 s of the cube's 7 s gain); before this, the wedge from the start: the 8-atom Si cube, 2x2x2, one A100 node, 4 against 8 points, 6.3 against 12.9 s, the totals equal to 1e-10 Ry on both routines; the cube on the Fortran-derived routines with meshes its whole group does not keep (0.57 bohr: 2x2x1 4 -> 3 points, the total 4e-9 Ry apart; 2x2x3 8 -> 6, 1.3e-9 Ry; 2x2x4 12 -> 9, 5.5e-8 Ry, an image on the other sign of a zone-boundary label; 2x2x1 at 0.855 bohr and order 8: 7.2e-7 Ry); Si64 with the 2x2x2 mesh on 4 A100: 4 of the 8 points, SCF 253.4 -> 161.4 s, the exchange 71.1 -> 48.5 s, GPU 24.3 -> 14.9 GiB, the same total; its spin PBE0 813.1 -> 444.7 s. Forces and `bfgs`/`simple` relaxations take the same route since 2026-11-21 (`PARSEC_HYBRID_KPOINT_FORCE_WEDGE`, below; before, they kept the pairs); molecular dynamics, `manual` movements, spinors, a `Kpoint_List`, slabs and a mesh that time reversal does not keep (a shift other than 0 or 1/2) keep them always, and `parsec.out`'s note of a mesh without a shift block gives the reason | `on`: the wedge always, its semilocal start too (the symmetric state, also for a metal); `off`: the time-reversal pairs of the mesh, bit for bit as before; any other value is refused (`grid`, the operations of scalar orbitals on any grid of 2026-11-19, among them those that turn an oblique grid's stencil, is gone) |
| `PARSEC_HYBRID_KPOINT_SECTORS` | `off` (default): the SCF loops of the Fock loop of a hybrid with k-points split their k-points as every k-point loop does (`PARSEC_KPOINT_SECTORS`: a basis of at least 2^24 entries whose sectors halve the work), bit for bit as before | `on` (since 2026-11-20, opt-in): the semilocal start and every inner loop split each k-point whose little group has commuting involutions into their sectors, whatever its size and the work they save (`SCF/hybrid_periodic.py`, `BlochExchange`: the key `kpoint_sectors_any_size` in the projector cache its loops share; `SCF/kpoint_sectors.py`, `plan_sectors`), so that the orbitals keep the operations of `H_k`, the half-cell translations of a cell that repeats a smaller one among them, and the exchange of a metal cannot break them; bulk orthorhombic cells only, as the k-point sectors (elsewhere nothing changes); the pair-solver line of `parsec.out` names it; with `PARSEC_KPOINT_SECTORS=off` it is refused at the preparation. Why (A100, first runs, one process, 2026-11-20): the conventional cubic cell of fcc Au (4 atoms, a = 7.7 bohr, PBE0, 2x2x2, 1000 K, 0.3 bohr) has a 4-fold level at the Fermi level at its three points (0, 1/2, 1/2) (f 0.42 after the start: four primitive Sigma points folded together, one per character of the face-centring translations); the Fock loop of either `PARSEC_HYBRID_KPOINT_WEDGE` route fills it unequally, then mixes its states so that the density loses the face-centring translations (asymmetry 1e-8 -> 5.8e-3, x2.5 per outer iteration: the four atoms become inequivalent) and creeps on through nearly degenerate states of that kind (occupations fixed, successive density-matrix steps parallel, 1e-3 to 2.5e-2 per outer iteration, the energy falling 1e-7 to 1e-5 Ry per iteration): at `Exx_Tolerance` 1e-8 not converged after 300 outer iterations (pairs -265.91021 Ry, wedge -265.90427 Ry, both still falling), nor with a fixed inner criterion at 1e-10 or an outer DIIS opened to fractional occupations (a trial build). With `on`, at `Exx_Tolerance` 1e-10: the pairs converge in 26 outer iterations (dexx 3.9e-11, -265.9095674 Ry, the single and double occupations of the 4-fold level at the three points that the pairs take without the switch, the translations kept; 231 s), the wedge in 12 (-265.8971549 Ry, 67 s); at 1e-8 the pairs stop after 10 on a stationary point symmetric across the three points (-265.8970181 Ry, the total of the sectors forced by hand to 1e-11 Ry), which a tighter tolerance leaves, the wedge after 10 (-265.8971544 Ry). Not the default: it changes the state such a metal reaches (today the broken translations, whose energy is lower), and a cell below the floor pays for small sectors (the 4-atom cube: 8 to 32 sectors per k-point, 9 to 13 s per outer iteration of the pairs against 2 s) |
| `PARSEC_HYBRID_IMAGE_PAIRS` | `on` (since 2026-11-21): on the irreducible wedge of a hybrid's mesh (`PARSEC_HYBRID_KPOINT_WEDGE`) the pair of an occupied target `(t, n)` with an image source `(i, m)` (`P_g u_r`) and the pair of `(r, m)` with the image's mirror source `(j, n)` (`P_(g^-1) u_t`, a source of the list: `MeshStars.mirrors`, `V_xc/bloch_exchange.py`, `mesh_stars`) come from one transform, as the pairs of two representatives do (`PARSEC_CUPY_BLOCH_PAIR_SYMMETRY`): the densities are `rho` and `P_(g^-1) conj(rho)` at `q` and `-R^-1 q` (`P_(g^-1) rho` at `R^-1 q` for a conjugated image), so the second potential is `P_(g^-1) conj(w)` (`P_(g^-1) w`) wherever the pair potentials keep the operation, which a random-density test of each pair of wave vectors checks once per run (`image_pair_relation_holds`, relative 1e-10; it held to 1e-15 on the orthogonal grids of both routines and on the oblique grid of the primitive fcc cell, whose images take only stencil-exact operations); host loop and device tiles (`_image_pair_roles`: the first of the two blocks in (target, source) order solves them; a fourth kernel adds the partner sides to `W` of `r` at the turned points, no buffer). Only pairs of two occupied states, and only where the mirror source is in the list: a target whose point the operation keeps (Gamma, R) has none, which keeps the gain below 2x (the 8-atom Si cube 2x2x2: 8 of 16 image blocks; 4x4x4: 176 of 480; primitive Si 4x4x4: 334 of 546). W of `on` and `off` agree to 2e-16 (CPU, random orbitals, every column and the occupied ones then the empty ones). A100, PBE0, first runs, one process (2026-11-21, `on` / `off` / `on`): Si64 2x2x2 on 4 A100: pair solves 3.44M -> 2.92M (-15 %), Poisson 34.0 -> 29.0 s, exchange 47.1 -> 43.2 / 41.8 s, SCF 163.2 -> 157.6 / 156.1 s, GPU peak 14.46 GiB both, total -504.67423353 Ry both, E_x within 1e-8 Ry; cubic MgO (8 atoms, 2x2x2, 0.18 bohr): solves -11 %, SCF 35.1 / 35.2 s against 35.2 s on one A100, 16.1 / 16.1 against 16.0 s on four (its exchange is 1 s of the run); primitive Si 4x4x4 at 0.25 bohr on one A100: solves -10 %, SCF 28.5 / 26.9 against 27.0 s, where the first `on` run compiled the whole exchange (a cold CuPy kernel cache: `off` pays the same on a cold cache). Each with a fresh kernel cache of its own (2026-11-21, `off` / `on`, then both on a warm cache): primitive Si 4x4x4 SCF 33.74 / 33.98 s, first build 1.78 / 2.04 s (the relation tests, 0.13 to 0.16 s per run, and about 0.1 s for the compile of the partner kernel's module, once per cache), warm 26.60 / 26.61 s; cubic MgO 45.79 / 46.06 s, first build 4.40 / 4.59 s, warm 35.16 / 35.17 s, GPU peak 2.64 / 2.62 GiB. So the gain is Si64's (SCF -3.5 %); the small cells are neutral warm and 0.25 s (0.6 %) slower on a cold cache. The totals and E_x of every pair the same to the last printed digit | `off`: every pair of an image source solved one way, as before, bit for bit |
| `PARSEC_HYBRID_KPOINT_FORCE_WEDGE` | `auto` (since 2026-11-21): the single-point forces and the `bfgs`/`simple` relaxations of a bulk hybrid with a Monkhorst-Pack mesh take the route `PARSEC_HYBRID_KPOINT_WEDGE` gives its single point (`auto`: the semilocal start of the first structure on the time-reversal pairs, the Fock loop on the wedge where that start fills every state to 0 or 1, so a metal keeps the pairs; `on`: the wedge throughout), on both sets of routines; the wedge's group keeps movable and fixed atoms apart (`Input/parsec_input.py`, `_default_routines_force_wedge` with the stencil-exact operations; the Fortran-derived routines check the space group as for every k-point force); the nonlocal forces of the wedge's k-points are completed over the group and the forces averaged over it, as for semilocal k-point forces (`PARSEC_KPOINT_FORCE_WEDGE`, whose `off` turns this off too on the default routines; the Fortran-derived routines do not read that switch, for semilocal forces either, so there this switch's `off` is the one that keeps the pairs; `Forces/total.py` with the system of the wedge, `SCF/hybrid_periodic.py`, `hybrid_force_system`), so a relaxed structure keeps the group; the later structures of a relaxation keep the route of the first (their Fock loops start from the orbitals of the structure before, with the stars of the wedge: `result_stars`, `BlochExchange.adopt_carried`); `parsec.out` names the route in a note (the note of a mesh without a shift block names the fixed atoms where `Move_Flag` fixes an atom that an operation exchanges with a movable one, which keeps the pairs), and a Fock loop on a wedge that `auto` took whose last inner loop has a partly filled state (a later structure of a relaxation turned metallic, or a Fock loop that closed the start's gap) gets a NOTE that it is a metal held in the symmetric state of the wedge and which switch keeps the pairs (`BlochExchange.after_loop`; the run is not changed, and the route stays that of the first structure). A100, PBE0, first runs (2026-11-21, `off` / `auto` / `PARSEC_HYBRID_KPOINT_WEDGE=on`): Si64 2x2x2 with atom 1 moved 0.05 bohr along z (a group of 4, 6 of the 8 points) on 4 A100, forces: SCF 322.1 / 296.9 / 289.9 s (344.5 s for `auto` with `PARSEC_HYBRID_WEDGE_SECTOR_STATES=off`: the first solve of the wedge's sectors from random starts, 56 s), the same total (-504.70239670 Ry) and forces to 1e-6 Ry/bohr (largest 0.018), GPU peak 33.5 / 31.7 / 33.0 GiB; its `bfgs` relaxation of 2 movements: 940.9 -> 843.1 s with the defaults (892.8 s with `PARSEC_HYBRID_WEDGE_SECTOR_STATES=off`), the same final energy (-504.70282556 Ry), positions and forces to 1e-6, GPU peak 39.4 -> 38.7 GiB; cubic MgO (8 atoms, 0.18 bohr, 2x2x2, an O atom moved 0.05 bohr along x: C4v, 6 of 8 points) on one A100: forces 47.0 / 39.6 / 33.2 s, forces to 1e-6 Ry/bohr (largest 0.011), the same total; its `bfgs` relaxation of 3 movements 122.2 -> 101.0 s, the same structure; H2 in a cube with 3x3x1 (CPU test): forces and a 3-step relaxation to 1e-5. A displaced primitive cell on an oblique grid often keeps no stencil-exact operation (primitive Si with an atom moved along x, MgO with O moved): it keeps the pairs, unchanged | `off`: forces, moving atoms and relaxations of a hybrid keep the time-reversal pairs of the whole mesh, as before, bit for bit; any other value is refused |
| `PARSEC_HYBRID_WEDGE_SECTOR_STATES` | `auto` (since 2026-11-21): where the Fock loop of an insulating start moves to the wedge (`PARSEC_HYBRID_KPOINT_WEDGE=auto`), the k-point sectors of the wedge's representatives start from the Ritz states the semilocal start left in the sectors of the same k-points (`SCF/hybrid_periodic.py`, `_restricted_sector_states`: each representative is a point of the pairs with its label, so its sector plan is the start's; the layout of the pairs restricted to them and renumbered, which the sector solver takes where it is its own, `SectorKPointSolver.import_states`, else its sectors start afresh as before); states held on the devices of the pairs' workers move with their k-points. A100, PBE0, first runs (2026-11-21, `auto` / `off`): Si64 2x2x2 on 4 A100: SCF 152.2 / 157.1 s (the first solve of the wedge 15.7 s gone), the same total, E_x 9e-7 Ry apart (inner loops of other lengths, within the SCF criterion); Si64 with a moved atom, forces (`PARSEC_HYBRID_KPOINT_FORCE_WEDGE`): 296.9 / 345.7 s, the same total and forces; cubic MgO with the k-point sectors at any size (`PARSEC_HYBRID_KPOINT_SECTORS=on`): 56.2 / 59.3 s on one A100, forces of the moved cell 23.6 / 26.3 s on four, the same totals and forces. Cells without k-point sectors (below the size floor of `PARSEC_KPOINT_SECTORS`) are not touched | `off`: the sectors of the wedge start afresh (random starts), as before, bit for bit |
| `PARSEC_SLAB_MESH_SYMMETRY` | `auto` (since 2026-11-19): a slab (`Boundary_Conditions: slab`, Fortran-derived routines, the default for slabs) with a Monkhorst-Pack mesh of more than one point (a `Monkhorst_Pack_Shift` block, or PARSEC 1.5's zero shift without it), scalar or collinear-spin orbitals and a native LDA or GGA takes the operations of its group (PARSEC's wire/slab search, `alatt` with `4 rmax` along z) that map its grid and its mesh onto themselves and whose lattice matrix is a signed permutation of the in-plane lattice directions (`Grid/space_group.py`, `slab_mesh_subgroup`; `Input/parsec_input.py`, `_slab_mesh_space_group`): the mesh is reduced to their irreducible wedge with time reversal, the density is symmetrized and the forces averaged with them, as for a wire. The finite differences along the lattice directions map onto each other under such an operation, so the Hamiltonian of every k-point follows it to round-off (every operation of a rectangular or square slab; of a hexagonal one the mirrors through the bisector of the lattice vectors, `z -> -z` and their products). The group is used only where every point of the time-reversal wedge of `off` is an exact image of its representative, not one modulo a reciprocal vector, which the finite-difference Bloch operator does not treat alike (`slab_mesh_wedge_exactness`): the levels are then those of `off` to round-off. A four-fold rotation without a diagonal mirror (S4, C4, C4h) on a mesh through the zone boundary (a component 1/2) fails this test and keeps the former route with a warning (since 2026-11-19; A100, a square S4 slab of four C atoms at 0.45 bohr and order 6 on the unshifted 4 x 4 x 1 mesh had been 2.4e-5 Ry off; shifted by (0.5, 0.5, 0) it takes the wedge, 8 -> 4 points, SCF 7.5 -> 3.3 s, dE 1.2e-11 Ry), and so does a group whose wedge has no fewer points than time reversal alone (a two-fold axis alone). Hexagonal slabs, and meshes without a component 1/2, give the energy of `off` to round-off; the mirrors of a rectangular or square slab on a mesh through the zone boundary average the density of such a point over images that differ from the point of `off` by a reciprocal vector, a difference of second order in the error of the finite differences (A100 at 0.45 bohr and order 6 on the unshifted 4 x 4 x 1 mesh: a D4h square slab of four C atoms -5.3e-9 Ry, a D2h rectangular one -1.7e-10 Ry, h-BN -5.2e-12 Ry; Si(001) below at order 12: the same energy). Charged sheets (`add_charged_sheets`) that are not symmetric about z = 0 drop the operations that reverse z (A100, an h-BN capacitor of +0.05 e at z = -4 bohr and -0.05 e at z = +4 bohr, 3 x 3 x 1 at 0.45 bohr, had been averaged over z -> -z, 1.05e-2 Ry off with the dipole erased; now 5 -> 4 points, dE -3.9e-12 Ry and the dipole of `off`). An operation whose fractional translation is no whole number of grid steps is dropped, so the grid is that of `off`. PARSEC 1.5 searches no symmetry for a slab without bands, spin-orbit coupling or an NSCF run. Hybrids, meta-GGA, libxc, LDA+U, TS, molecular dynamics, `manual` movements, a `Move_Flag` that breaks the group, `Symmetrize_Charge_Density: .false.`, the Ylm-projected DOS (`Create_DOS` with `YLM_DOS_L`: the tetrahedra would reuse the projections of a representative for its whole star without rotating the harmonics, which put the px/py LDOS of h-BN off by up to 0.07 of 0.14; now the DOS of `off` to 6e-11) and slabs on Andrew Grieder's routines keep the former route. A100, one node, `main.py`, order 12: h-BN 6 x 6 x 1 PBE at 0.15 bohr (the `hBN_k661/pbe` input with a `Monkhorst_Pack_Grid`) 20 -> 13 points, 91.3 -> 80.9 s (SCF 42.7 -> 32.8 s, the 48.6 s set-up unchanged), the same energy; graphene 9 x 9 x 1 PBE at 0.25 bohr 41 -> 25 points, 37.7 -> 25.8 s; silicene 6 x 6 x 1 LDA at 0.30 bohr with forces 20 -> 13 points, 184.2 -> 121.6 s, the same energy and forces; bare Si(001) 1 x 1 of five layers (D2d, square cell) 6 x 6 x 1 LDA at 0.40 bohr with forces 20 -> 10 points, 101.0 -> 53.7 s, the same energy, forces within 1e-6 Ry/bohr; buckled h-BN (C3v) 6 x 6 x 1 PBE at 0.25 bohr with forces 20 -> 13 points, 162.7 -> 104.8 s, the same energy and forces; spin-polarized h-BN at 0.25 bohr 20 -> 13 points, 55.2 -> 37.7 s; h-BN 4 x 4 x 1 shifted by (0.5, 0.5, 0) 8 -> 6 points, 21.2 -> 18.7 s, the same energies; the set-up and the GPU peak (0.6-0.8 GiB) unchanged, one to two SCF iterations fewer. A `Kpoint_List` (the shipped `hBN_k661` inputs) is used as given. `full`: every operation that keeps the grid and the mesh, without the exactness test, the three- and six-fold rotations of a hexagonal slab included (h-BN 7 points, 66.1 s, SCF 17.2 s; graphene 12 points, 16.7 s; silicene 7 points, 69.4 s; buckled h-BN with forces 7 points, 62.8 s; spin-polarized h-BN 7 points, 28.1 s); the Bloch term `-2 i k . nabla` differentiates along the two lattice directions only, so the Hamiltonian of a k-point follows such a rotation only to the error of those differences, as in PARSEC's wedge of a hexagonal bulk cell: the energy differs from the whole mesh by -1.8e-7 Ry (h-BN), +3.0e-6 Ry (graphene), -2e-8 Ry (silicene), -1.0e-5 Ry with forces within 5e-6 Ry/bohr (buckled h-BN at 0.25 bohr), -1.1e-5 Ry (spin-polarized h-BN at 0.25 bohr), 5.1e-4 Ry for a two-atom h-BN slab at 0.45 bohr and order 6 | `off`: no search, the mesh reduced by time reversal alone, as PARSEC 1.5 and before |
| `PARSEC_CUPY_SECTOR_GROWTH_RELEASE` | `auto` (since 2026-11-18): a sector that grows in the spectral bracket gives its basis back before its new first solve (the same numbers; memory only) | `off`: only sectors with cut margins do, as before |
| `PARSEC_CUPY_SECTOR_WARM_GROWTH` | `auto` (default, since 2026-11-21): a sector that grows in the spectral bracket with a CHEBFF first solve keeps the vectors that a later pass has refined. They wait on the host, so that the devices hold what they hold for a random start, and the first solve filters them with new random columns, on one GPU and on a basis that GPUs or MPI ranks share. The vectors of the first solve of a run are not kept. CHEBDAV (the input default) and ARPACK grow as before: CHEBDAV draws a new trial block only when fewer than six active vectors remain, and started from the first six saved columns it used up its outer restarts in metals (fcc Al of 108 atoms, growth forced at step 5: 14 -> 25 SCF steps). The host holds the saved columns of the process during each such growth (22^3 diamond, one rank per node with a basis over its four GPUs: 76.8 GiB per node, about 7 s to the host and 4.7 s back per grown sector). Where the saved columns of a process exceed its share of half of the memory the host reports as available (`PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST`, the row below), or the copy fails, the sector grows from random vectors, and the MPI ranks of a shared basis agree on that before they solve. Measured with CHEBFF: 22^3 diamond (85,184 atoms) on 16 A100: 18 -> 11 steps, run wall 1068 -> 573 s, peak 76,417 -> 71,759 MiB per GPU, 6.0e-4 -> 1.3e-4 Ry from the cell converged to 2e-6 Ry without growth; C-H cluster k10 with its margin cut to nothing: 14 -> 10 steps, 2.9e-5 -> 2.9e-6 Ry from the full margin; six start seeds each of MgO 1000 atoms, NaCl 512, TiO2 162: 18.2 -> 16.2, 15.3 -> 13.7, 41.5 -> 36.5 steps (13 of 18 runs fewer, 4 equal, 1 more); growth forced at step 5 in fcc Al 108 and 256, Ag 32, Cu 32: 24 -> 19, 26 -> 20, 20 -> 17, 24 -> 20 steps; forces within 4e-6 Ry/bohr. A run in which no sector grows, and every CHEBDAV and ARPACK run, is the same to the bit | `off`: every growth from random vectors, the former bits |
| `PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST` | `node` (default, since 2026-11-22): the saved columns of a growing sector (`PARSEC_CUPY_SECTOR_WARM_GROWTH`) of all the processes of a run on one node take at most half of the memory the host reports as available: each process gets that half divided by their number (`Eigensolvers/sector_growth.host_room`, `host_processes`; the MPI full-SCF runner counts the ranks of each node from their host names, `main.py --mpi` from its node communicator; one process gets the whole half, as before). Several ranks of one node, one GPU each (`--rank-per-gpu`), check and stage at the same time, and each saw the whole room, so that together they could take up to their number times the half. A100, four ranks of one node sharing the basis of a 16-atom silicon cell (CHEBFF, `--symmetry off`, growth forced at step 5): every rank staged its columns (18-29 MB), the same energy to the last bit and the same 19 steps as `process`; with the host reporting 100 MB available, `node` gave each of the four ranks 12.5 MB, none staged, and the growth started from random vectors on every rank (the bits and 22 steps of `PARSEC_CUPY_SECTOR_WARM_GROWTH=off`), where `process` let each take 50 MB, and every rank staged (the bits and 19 steps of the real room) | `process`: every process the whole half, as before |
| `PARSEC_KPOINT_SECTOR_WARM_GROWTH` | `off` (default): a sector of a k-point that grows during the SCF (`SCF/kpoint_sectors.py`, `SectorKPointSolver._grow`: it missed a level of its k-point) starts its first solve from random vectors, as before. Opt-in since 2026-11-22: `chebff` starts a CHEBFF first solve from the Ritz vectors of the sector, where a SUBSPACE step has refined them, and as many columns of the usual random start as it gains, the rule of `PARSEC_CUPY_SECTOR_WARM_GROWTH` (`resize_tasks` of `SCF/kpoint_parallel.py` and of the MPI pools, `SCF/kpoint_pools.py`, whose ranks lay the former columns out again over the pool, `Eigensolvers/chebff_columns.py`, `_start_columns`; the vectors wait on the host where it has room for them, else the sector starts from random vectors, and the ranks of a pool agree); `all` also the device first solve of `Eigensolver: chebdav` (CHEBFF cycles to CHEBDAV's residual test). Not the default because it gains nothing measured: the first solve of a grown sector makes all its cycles from either start. A100, one GPU per run, 2 x 2 x 2 meshes in sectors (`PARSEC_KPOINT_SUPERCELL=off`), the sector of most states of the first k-point forced to grow at the fifth solve, SCF steps `off` -> `chebff`: Al 108 atoms 13 -> 14 (20.1 -> 21.0 s), Si 216 13 -> 14 (27.2 -> 28.3 s), Cu 32 11 -> 11, NaCl 64 10 -> 10, total energies within 2.6e-8 Ry (the SCF criterion 1e-6 Ry); with `Eigensolver: chebdav` and `all` 12 -> 12, 10 -> 10, 10 -> 10, 12 -> 12 within 6e-11 Ry. Without forcing no sector of these cells (and of Ag 32) grew in their SCF, and both values give the same bits. Four MPI ranks (one pool of four ranks and two of two, Al 108 with a forced growth) give the energy of one process with four GPUs within 1.3e-15 Ry under both values | `chebff`, `all` (opt-in, see the left column) |
| `PARSEC_CUPY_CUT_SECTORS_FIRST` | `auto` (since 2026-11-18): where margins were cut for memory, a bracket round in which cut sectors fail grows those alone and tests the others again in the next round | `off`: every failing sector grows in the same round, as before |
| `PARSEC_CUPY_REGROWN_SECTOR_TRIM` | `auto` (since 2026-11-18): a sector grown from a new start keeps its states through the trim of that step (trimmed at once it lost them again and grew from random vectors in every step). With the leak of the SCF loop fixed (the weighted copy of the last orbitals held the basis of a regrowing sector), diamond 22^3 converges with the default margin (1044 s, 18 steps, the energy of the full margin within 5.5e-5 Ry; the full margin, `PARSEC_CUPY_SECTOR_MARGIN=fixed`, takes 262.8 s) | `off`: trimmed in the same step, as before |
| `PARSEC_SECTOR_PLACEMENT` | `size` (since 2026-11-18): the MPI ranks and GPUs take the symmetry sectors by their cost (rows times first state count; `experimental/mpi_scf.py`, `plan_sector_groups`, `sector_rank_plan`): one rank per node gets balanced groups, costliest first, the root the lightest; the GPUs of a rank get shares in proportion to the cost, and the device pool where it lowers the fullest GPU by more than 5 % (also for more sectors than GPUs) and the first solve divides every basis over the GPUs (`Eigensolver: chebff` and rows x states^2 of each sector at the threshold of the generalized Ritz step, `Eigensolvers/symmetry.sector_bases_divisible`; since 2026-11-19: after CHEBDAV, or for sectors below it, the pool had put every basis on the first GPU: diamond 8^3 with `Eigensolver: chebdav` on one node took 35.8 s of SCF with all ten sectors on GPU 0, 28.7 s now, as by count), otherwise one GPU per sector, longest first (diamond 15^3 on one node keeps the pool: SCF 126.6 s against 202.8 s by count); with `--rank-per-gpu` and `Eigensolver: chebff` a group of ranks may solve several sectors one after another. The same sectors, and the same counts unless `PARSEC_CUPY_SECTOR_MARGIN=auto` cuts the margins for memory, which it judges per device of the placement; so the same results up to the order of MPI sums. Diamond 22^3 with the default settings and `--rank-per-gpu`: no margin cut, 322.9 s (the former placement ran out of memory), fullest GPU 66.4 GiB; 20^3, 10^3, an odd 15^3 cell and the k14 nanodiamond as fast as before with a lighter root. One rank per node still cuts the margin of 22^3 (ten nearly equal sectors on four nodes, 1044 s; `PARSEC_CUPY_SECTOR_MARGIN=fixed` holds it in 264.2 s at 76.8 GiB on the fullest GPU, `PARSEC_MPI_NESTED_SHARING=groups` in 518.6 s at 74.9 GiB) | `cyclic`: sector s on rank s mod size, the former placement |
| `PARSEC_CUPY_SECTOR_PLACEMENT` | `size` (since 2026-11-19): one process with several GPUs (`main.py`) places its symmetry sectors as the MPI runner places those of one rank (`Eigensolvers/symmetry.process_sector_device_groups`, the runner's `_rank_device_groups` with the costs of `sector_placement_costs`): disjoint device groups in proportion to the cost, costliest first; the device pool (`PARSEC_SECTOR_DEVICE_POOL`) where it lowers the fullest GPU by more than 5 %, a CHEBFF first solve divides every basis (`sector_bases_divisible`) and the mean sector gives every GPU at least `PARSEC_CUPY_SECTOR_POOL_MIN_BYTES` (24 MiB) of basis there, the GPUs then sharing each basis in turn; the one sector of the identity (`PARSEC_CUPY_SINGLE_SECTOR=on`) shared by all GPUs (before, `distributed_filter_devices` was set only from an MPI context, so that basis stayed on GPU 0). A hybrid functional, a meta-GGA and LDA+U keep the cyclic deal (their device kernels read every basis on one GPU); `orbital_sector_placement` in `parsec.out` names the placement (and the share of a GPU behind a pool decision). The same sectors; the same state counts unless `PARSEC_CUPY_SECTOR_MARGIN=auto` cuts the margins for memory, which it judges per device of the placement (`_state_fit_bytes`; none of the runs below was cut); shared bases change the order of the sums of the Ritz step and the density (round-off). One A100 node, first runs, bulk diamond at Gamma (PBE, 10 sectors; 9^3 has 5): SCF 8^3 10.7 -> 7.9 s, 9^3 28.9 -> 14.1 s (fullest GPU 7.7 -> 4.8 GiB), 10^3 25.4 -> 14.9 s, 12^3 65.6 -> 34.2 s (13.2 -> 12.0 GiB), 15^3 out of GPU memory under `cyclic`, 245.6 s at 58.1 GiB now; 5^3 and 6^3 keep disjoint groups below the least basis of the pool, as fast as `cyclic` (6^3: SCF 3.87 s against 3.86-3.94 s); energies equal to the printed digits. The identity sector (`PARSEC_PERIODIC_SYMMETRY=off`, `PARSEC_CUPY_SINGLE_SECTOR=on`): 4^3 16.3 -> 5.5 s at 2.0 GiB per GPU instead of 9.8 GiB on GPU 0, 6^3 out of memory (66.4 GiB on GPU 0) -> 53.5 s at 10.9-12.4 GiB per GPU. Nanodiamonds k08, k10, k12 (four Td sectors on four GPUs): the same placement as `cyclic`, the same energies and times within noise; k14 does not fit one node (the run ran out of host memory in the preparation after 1,351 s, before any basis lay on a GPU; not run under `cyclic`) | `cyclic`: sector s on GPU s mod (GPUs), each basis whole on its GPU, the former placement bit for bit |
| `PARSEC_CUPY_SECTOR_POOL_MIN_BYTES` | `25165824` (24 MiB, since 2026-11-19): one process with several GPUs (`main.py`, `PARSEC_CUPY_SECTOR_PLACEMENT=size`) takes the device pool only where the mean sector gives every GPU at least this many bytes of basis there (rows x first states x 8 bytes over sectors and GPUs, `Eigensolvers/symmetry.process_pool_min_bytes`, `process_pool_share_bytes`; not judged without the filter plan of the input); below it the sectors keep the disjoint groups of `PARSEC_SECTOR_DEVICE_POOL=off`, and `orbital_sector_placement` gives the share and the reason. The pool adds the exchanges and synchronizations of a shared basis to every sector it solves in turn, which for small sectors cost more than the imbalance of disjoint groups saves, in every SCF loop (a relaxation pays them in each); the memory of such sectors matters nowhere. One A100 node, first runs, bulk diamond at Gamma (PBE), pool against disjoint groups, SCF seconds and in brackets those of the iterations after the first (one SCF loop of a relaxation), two runs each: 6^3 (10 sectors, 9.4 MiB per GPU per sector) 4.33/4.30 (2.15/2.07) against 3.84/3.89 (1.79/1.82); 5^3 (5 sectors, 12.5 MiB) 2.86/2.82 (1.15/1.22) against 2.42/2.37 (1.08/1.09); 5x6x6 (10 sectors, 25.7 MiB) 6.72/6.40 (3.65/3.49) against 7.59/7.66 (3.68/3.69); 6x6x7 (10, 49.8 MiB) 8.22/8.60 (4.24/4.48) against 10.99/11.19 (4.77/4.91); 8^3 (10, 51.3 MiB) 7.61/7.68 (3.94/3.92) against 10.80/10.57 (4.60/4.51); 7^3 (5, 91.6 MiB) 5.51/5.39 (2.37/2.33) against 8.55/8.94 (3.13/3.19); `cyclic` as disjoint within noise; energies equal in every printed digit. The floor lies between the slower 12.5 MiB and the 25.7 MiB from which the pool was no slower in any loop. Since 2026-11-21 the MPI full-SCF runner applies the same floor to the GPUs of each rank that places its sectors by size where the first solve draws its start vectors on the GPU or the distributed filter runs (`PARSEC_CUPY_DEVICE_RANDOM=1` or `PARSEC_CUPY_DISTRIBUTED_FILTER=1`, both in the launch settings of `scripts/run_multi_gpu_scf.sh`; `experimental/mpi_scf.runner_pool_floor_applies`, `_pool_floor_groups`, `MPISectorContext.pool_floor`; the mean of the rank's own sectors; `parsec.out` names the decision of every rank that took one: `Sector devices of rank r: ...`), where the pool cost more. One rank of 4 A100, first runs, disjoint groups against the pool, SCF seconds of `timing.json` (in brackets those of the iterations after the first). With the launch settings, two runs each: diamond 6^3 (9.4 MiB) 1.24/1.30 (0.66/0.66) against 3.83/3.49 (2.17/2.03); diamond 5^3 (12.5 MiB) 0.91/0.94 (0.40/0.39) against 2.30/2.14 (1.18/1.14); silicon 5^3 (1,000 atoms, CA, 20 points per cubic edge, 17.2 MiB) 1.26/1.26 (0.71/0.72) against 2.98/2.91 (1.91/1.86); fcc silver 5^3 (500 atoms, CA, 3,025 states, 20 points per cubic edge, 6.6 MiB) 1.89/1.81 (1.16/1.13) against 5.08/5.12 (3.44/3.46); rocksalt LiF 5^3 (1,000 atoms, PBE, 25 points per cubic edge, 9.3 MiB) 2.03/1.92 (1.06/1.07) against 5.09/5.11 (3.16/3.14); the same energies to the last bit (diamond, silver, LiF) and within 1e-12 Ry (silicon), the same iterations; the fullest GPU 985 against 1011 MiB (6^3), 1081 against 972 MiB (silicon), 953-997 against 973 MiB (silver), 1267 against 1325 MiB (LiF). With one of the two switches, one run each: the launch settings with `PARSEC_CUPY_DISTRIBUTED_FILTER=0` (as every spin-polarized run of the runner took them before 2026-11-22) diamond 6^3 1.21 (0.65) against 3.62 (2.12); the code defaults with `PARSEC_CUPY_DISTRIBUTED_FILTER=1` diamond 6^3 2.85 (0.73) against 3.82 (2.09), 5^3 1.86 (0.49) against 2.12 (1.16), silicon 2.40 (0.80) against 3.05 (1.86); the code defaults with `PARSEC_CUPY_DEVICE_RANDOM=1` diamond 6^3 3.28 (1.73) against 3.65 (2.12), 5^3 2.01/2.02 (1.04/1.05) against 2.02/1.97 (1.15/1.16), silicon 2.50 (1.46) against 2.89 (1.85). Under the code defaults of both switches (start vectors drawn on the host, one filter per GPU) the ranks keep the pool by the memory rule alone, as before (the same bits as before: diamond 5^3 and 6^3 checked), since the floor was not faster there: diamond 5^3 2.32/2.37 against 2.09/2.17 s (10 % slower; the eigensolve of the first iteration 1.22 against 0.87 s), 6^3 3.98/3.85 against 3.89/3.85 s, silicon 3.11/3.08 against 3.05/3.12 s, the iterations after the first 5-22 % faster (5^3 1.03/1.06 against 1.09/1.18 s, 6^3 1.79/1.74 against 2.07/2.10, silicon 1.49/1.49 against 1.86/1.90), the fullest GPU 21-53 % fuller (5^3 1131-1175 against 847 MiB, 6^3 1237 against 1021, silicon 1437-1483 against 969-973), the energies within 1.8e-12 Ry. A decision needs sectors that do not divide the GPUs of a rank evenly and bases the first solve divides: the clusters measured (Td Au35 and Au56, CA: four weighted sectors with equal first counts on four GPUs), bcc sodium 8^3 (1,024 atoms, 20 sectors) and fcc silver 3^3 (108 atoms, ten sectors below the threshold of the generalized Ritz step) took the same groups and gave the same bits with the launch settings and with `PARSEC_CUPY_SECTOR_POOL_MIN_BYTES=0`. Silicon 5^3 at 24 points per edge (29.7 MiB) keeps the pool, as before. Since 2026-11-22 the runner also asks the pool to save a quarter of the fullest GPU (`PARSEC_MPI_SECTOR_POOL_MIN_GAIN`, the row below), since with the launch settings the pool of the runner was slower than the disjoint groups of `PARSEC_SECTOR_DEVICE_POOL=off` above the floor as well (silicon 5^3 at 29.7 MiB 5.74 against 3.19 s, diamond 7^3 at 91.6 MiB 3.72 against 2.73 s, 8^3 at 51.3 MiB 6.27 against 2.85 s, 10^3 at 193.6 MiB 9.32 against 6.90 s, 12^3 at 574.8 MiB 17.71 against 15.92 s with 12.2 against 9.9 GiB on the fullest GPU), while without them diamond 8^3 took 7.47 s in the pool against 11.13 s in disjoint groups, as in one process | `0`: the pool by the memory rule alone in one process (the placement of one process as first published on 2026-11-19); on the ranks of the runner together with `PARSEC_MPI_SECTOR_POOL_MIN_GAIN=0.05` (the row below, whose gain `0` does not lift), the placement of the runner before 2026-11-21, which the runner keeps under the code defaults of `PARSEC_CUPY_DEVICE_RANDOM` and `PARSEC_CUPY_DISTRIBUTED_FILTER` |
| `PARSEC_MPI_SECTOR_POOL_MIN_GAIN` | `0.25` (since 2026-11-22): where the floor of the row above applies to the GPUs of a rank of the MPI full-SCF runner (the launch settings), the rank takes the device pool only where it also saves at least this share of the fullest GPU of its disjoint groups, `1 - pooled / fullest` (`experimental/mpi_scf.runner_pool_min_gain`, `_pool_gain`, `_pool_floor_groups`; `parsec.out` names the share in the line of every rank that took a decision): ten equal sectors on four GPUs (3 against 2.5 sectors on the fullest, 1/6) and seven, eleven or thirteen keep disjoint groups, three, five (2 against 1.25, 3/8), six or nine the pool of before. With the launch settings ten sectors were faster in disjoint groups at every size measured, with less on the fullest GPU, and five faster in the pool from about half a GiB of basis per GPU per sector. One rank of four A100, first runs, SCF seconds with the kernel cache warm, disjoint against the pool, the fullest GPU: diamond 10^3 (ten sectors, 193.6 MiB per GPU per sector) 7.14 against 9.18 s, 4.3 against 5.1 GiB; 12^3 (574.8 MiB) 15.64 against 17.50 s, 9.6 against 12.0 GiB; 14^3 (1446 MiB) 33.81 against 36.90 s, 20.6 against 27.8 GiB; 16^3 (3216 MiB) 76.50 against 78.78 s, 42.9 against 52.1 GiB; silicon 8^3 (4096 atoms, CA, ten sectors, 121.5 MiB) 11.40 against 15.77 s (16.18 against 20.80 s with an empty kernel cache), 3.4 against 4.5 GiB; five sectors: silicon 5^3 (29.7 MiB) 3.22 against 5.66 s, diamond 11^3 (1361 MiB) 20.95 against 18.08 s, 12.6 against 13.1 GiB, 13^3 (3703 MiB) 59.44 against 46.62 s, 32.1 against 30.9 GiB, diamond 9^3 (410 MiB) 7.48 against 7.85 s, diamond 12^3 on two nodes (five sectors per rank, 574.8 MiB) 11.22 against 9.78 s (the pool, as before, from 24 MiB: slower below a few hundred MiB, faster above); the same energies (diamond 10^3 and 16^3 within 1e-11 Ry). With the default against `0.05`: diamond 12^3 took the disjoint groups (a gain of 16.7 %) and 16.01 against 17.63 s (21.13 against 22.55 s with an empty kernel cache), 9.6 against 12.0 GiB, the same energy; silicon 5^3 and 7^3 (217 MiB) kept the pool (37.5 %), the same groups and energies. | `0.05` (or less): the decisions of before (the memory rule asks a gain of 5 % in any case) |
| `PARSEC_PERIODIC_SECTOR_DEVICES` | `auto` (default, since 2026-11-21): one process with several GPUs keeps every sector of a periodic cell on its first GPU where the first bases of its sectors (rows x first states x 8 bytes, both spin channels; `driver._periodic_one_device_note`) total less than 512 MiB; `orbital_sector_placement` says so. Spread over the four A100 of a node by `PARSEC_CUPY_SECTOR_PLACEMENT=size`, every periodic cell measured up to 1728 atoms ran slower than on one of them or no faster, and its first run (fresh CuPy cache) about twice as long, every GPU compiling the kernels of its sectors (first diagonalization 17-20 s against 6-7 s). Total wall time in `parsec.out`, four visible A100, this switch -> `all`, first run / second run: forces of Si8 12.7 / 1.05 -> 16.4 / 1.14 s, MgO8 13.0 / 1.35 -> 16.6 / 1.37 s; diamond 216 atoms (13.9 MiB) 8.4 / 1.95 -> 11.4 / 2.08 s, 1000 atoms (250 MiB) 14.2 / 3.25 -> 23.9 / 4.82 s, 1728 atoms (375 MiB) 16.2 / 6.37 -> 27.2 / 6.80 s; with `PARSEC_PERIODIC_SECTOR_FLOOR=off` MgO 64 16.2 / 4.80 -> 28.7 / 5.38 s, SrTiO3 40 17.3 / 6.06 -> 30.2 / 7.10 s, MgO 64 on the Fortran-derived routines 15.6 / 6.74 -> 28.8 / 7.33 s, Si 2x2x2 13.5 / 2.57 -> 26.5 / 2.78 s (the best of three second runs; their SCF 1.68-1.72 against 1.92-2.03 s, while the pre-SCF time of a second run on four visible GPUs is either about 0.7 or about 2.3 s under both settings), diamond 1000 atoms again 12.8 / 3.24 -> 23.9 / 4.87 s, spin-polarized diamond 512 atoms 15.1 / 8.55 -> 19.2 / 9.66 s and 1000 atoms 15.9 / 6.68 -> 27.1 / 6.79 s; energies and SCF steps the same in every printed digit. The first GPU then holds every sector, and its peak is higher than spread (diamond 1000 atoms 1.93 against 0.96 GiB, 1728 atoms 2.31 against 1.25 GiB, spin-polarized diamond up to 2.41 against 1.34 GiB). Diamond 2744 atoms (1.8 GiB) keeps the four GPUs, which are faster there (6.8 s against 10.3 s on one GPU). A cluster, an MPI rank, one sector of the identity (`PARSEC_CUPY_SINGLE_SECTOR=on`, the shared full grid of a spin fallback), a hybrid, a meta-GGA and LDA+U (the cyclic deal) and `PARSEC_CUPY_SECTOR_PLACEMENT=cyclic` place their sectors as before; an integer sets another limit in bytes; a wrong value is refused at setup. The tests of `acceleration/tests` run with `all` unless a test sets it | `all`: the placement by size over every GPU, as before |
| `PARSEC_CUPY_COMPILE_ONCE` | `auto` (default, since 2026-11-22): a kernel that several threads of one process ask CuPy for at the same time is compiled by one of them; the others wait for it and load the binary that CuPy wrote to its kernel cache (`backends/cupy_compile.install_compile_once`: a lock per kernel, its source, options and architecture, around `cupy.cuda.compiler._compile_with_cache_cuda`; installed where CuPy is first imported, `backends/cupy.require_cupy`, and by the MPI full-SCF runner; different kernels still compile side by side). The GPUs of one process build and solve their sectors on one thread each, and with an empty kernel cache (the first run after an installation or an upgrade of CuPy) every thread compiled every kernel at once. One A100 node, first runs with an empty `CUPY_CACHE_DIR`, `main.py` on four GPUs, total wall seconds `off` -> `auto` (one GPU in brackets): Td Au56 cluster 14.40 -> 12.05 (11.04; the first eigensolve 13.5 -> 6.9 s), Td Au35 14.31 -> 11.54 (10.56), Si28H36 15.69 -> 14.08 (13.91), diamond 7^3 22.40 -> 20.28 (22.80), diamond 5^3 with `PARSEC_PERIODIC_SECTOR_DEVICES=all` 27.06 -> 16.85 (16.19); the MPI full-SCF runner, one rank of four GPUs, the launch settings, diamond 10^3: 25.93 -> 24.42 s of program time. The same energies in every case; with the kernel cache warm the two values take the same time (three alternating runs each: Au56 5.77 against 5.70 s, diamond 7^3 10.34 against 10.26 s). The measurements of `PARSEC_PERIODIC_SECTOR_DEVICES` above were made before this switch | `off`: every thread compiles what it needs, as before |
| `PARSEC_MPI_NESTED_SHARING` | `auto` (since 2026-11-19, the MPI runner): with one rank per node (several GPUs per rank, the default launch of `scripts/run_multi_gpu_scf.sh`), several nodes and `Eigensolver: chebff`, the ranks that share the one sector of a run without a point group (molecular dynamics, fixed-atom relaxations, `PARSEC_PERIODIC_SYMMETRY=off`) spread their shares over all their GPUs (`Eigensolvers/mpi_nested_basis.py`: R ranks of D GPUs act as R x D ranks of `mpi_shared_basis`, one thread per GPU, their collectives carried out by the main thread of each rank: copies between the GPUs of a node, GPU-aware MPI between nodes, `MPICH_GPU_SUPPORT_ENABLED=1` for such launches where mpi4py links the GPU transport layer of Cray MPICH, `libmpi_gtl_cuda`, and host-staged exchanges named in `parsec.out` where it does not (since 2026-11-20: Cray MPICH stops in `MPI_Init` otherwise); `experimental/mpi_scf.nested_sharing_applies` is the one check; every GPU filters with the CUDA graphs of the stencil-major kernels where they can record the sector operator and with its own recurrence otherwise, `filter_graph.PlainBlockFilter`, so that `PARSEC_CUPY_STENCIL_MAJOR=0` runs here too since 2026-11-21, where it stopped with an error from 2026-11-20). Diamond 6^3 (1,728 atoms) without symmetry on 2 nodes: SCF 327.0 -> 42.6 s (one node 36.8 s, 4 nodes 36.1 s, `--rank-per-gpu` 26.4 s on 2 and 16.8 s on 4 nodes), fullest GPU 22.4 -> 9.9 GiB, the same energy to the last digit; forces of a relaxation step within 4e-14 Ry/bohr of one node, and a three-step relaxation from a moved atom with the printed energies, forces and positions of one node (a structure this small is not faster on more nodes: 108.3 against 76.3 s of SCF). Fewer sectors than nodes, which stopped the preparation of one rank per node before ("MPI ranks must be between one and the representation count"), are shared in groups over all GPUs as well (since 2026-11-20; Si28H36 with an atom moved by 0.1 bohr, two sectors, on 3 nodes: one group of the three, the energy of one node within 2.3e-12 Ry in the same 20 iterations; spin-polarized O2 in Cs within 6e-14 Ry in 58; the driver still prefers a group of before that divides); as many sectors as nodes or more keep one rank per node as before. `groups` (opt-in) also gives such ranks the groups of `--rank-per-gpu` placed by size (groups that balance better, larger groups with fewer sectors than nodes that the driver keeps from `auto`): diamond 22^3 on 4 nodes keeps its full margin in two groups of two nodes, 518.6 s at 74.9 GiB on the fullest GPU against 1044 s with the margin cut (the same energy as the full margin), though `PARSEC_CUPY_SECTOR_MARGIN=fixed` holds that margin on one node per sector in 264.2 s at 76.8 GiB, but the five sectors of 15^3 on 4 nodes took 150.9 s of SCF as one group against 48.8 s with a node per sector (every rank exchanges through one network interface); `parsec.out` names `fixed`, `--rank-per-gpu` and `groups` with these figures in a NOTE where a margin was cut and `groups` would form groups | `off`: one sector shared by such ranks on the first GPU of each, no groups with one rank per node (fewer sectors than nodes stop the preparation, now with a message naming what runs them), host-staged launch, as before |
| `PARSEC_PARTNER_STATE_COSETS` | `auto` (since 2026-11-18): the full-grid partners of the states of weighted sectors (`PointGroupReduction.partner_states`, the orbitals a driver API run or a run writing wavefunctions exports) are formed from the first operation of each left coset gH instead of every operation of G, the same partners bit for bit (Si 4x4x4 on one node: export 67.6 -> 6.1 s, run_scf 80.2 -> 18.8 s; that export, not the SCF, had made the weighted plan look slower than abelian groups with translations, whose SCF loop is 3x slower per iteration). Under both values the images of a weighted state are also orthogonalized against the other solved states within 1e-4 Ry of its level and the partners formed for them (`state_levels` of the result's orbitals, since 2026-11-22): two solved states of one multiplet of G (a representation a sector holds more than once, or one whose partners two sectors hold) each completed alone repeated partners before, in the exported orbitals and so in `parsec.dat` (the SCF, which weighs the states, was right). A100, Si64 Gamma cells at 200 states in the sectors of the 192-operation group: spin at 0.4275 bohr |U^T U - 1| 1.0 -> 1.7e-9 and the density of the stored orbitals 2.75e-3 -> 5.7e-11 off the file's, unpolarized at 0.45 bohr 1.0 -> 4.8e-10 and 3.2e-3 -> 7.7e-10; cases without such multiplets (Si64 spin at 0.45, unpolarized at 0.4275, SiH4 and its spin-polarized run in the D2d sectors of Td) unchanged | `off`: every operation, as before |
| `PARSEC_PERIODIC_UNEQUAL_STEPS` | `auto` (since 2026-11-18): Gamma bulk cells of the default routines whose grid has a different step on each axis (orthorhombic or tetragonal cells on the parser's grid) take the sectors of their point group (axes of equal count and side may be exchanged; per-axis sector stencils, the deferred Laplacian, device XC with per-axis weights, the batched projectors; also their k-point sectors). Tetragonal Si 3x3x3 (216 atoms) on one node: 803.0 -> 16.5 s, the energy within 1.8e-11 Ry; forces within 7e-10 Ry/bohr. A cell whose sector stencils need more than the 256 coefficients of one-byte codes (such a cell has two or three times the shell values of a cube: Si stretched along z at Expansion_Order 18 needs 306, a tetragonal 2x2x2 perovskite at 12 on 36 x 36 x 38 points 263) codes them in two bytes and keeps its group since 2026-11-19 (`PARSEC_SECTOR_PALETTE_CODES`); under its `off` it steps down to a smaller group, or the full grid (`PARSEC_SECTOR_PALETTE_LADDER`), with the count in `symmetry_detection`. `Use_Fortran_Routines` (Fortran's general grid) keeps the full grid (since 2026-11-22 it takes the sectors under `PARSEC_OBLIQUE_SECTORS=auto`, row below) | `off`: the full grid, as before |
| `PARSEC_SLAB_WIRE_SECTORS` | `off` (default): a wire or slab keeps the full grid of one device at the Gamma point, as before | `auto` (since 2026-11-22, opt-in): a wire or slab of the Fortran-derived routines at the Gamma point whose periodic vectors lie along the axes, in one process, unpolarized, with a native LDA or GGA (no hybrid, TS, LDA+U, meta-GGA, libxc or `Symmetrize_Charge_Density`), above the floor of `PARSEC_PERIODIC_SECTOR_FLOOR`, solves its orbitals in the sectors of its group: the signed permutations about the origin that map its atoms modulo the periodic lattice and its cylinder or slab of points onto themselves, with translations along its periodic axes (`PARSEC_PERIODIC_TRANSLATIONS`); sector stencils reduced from its CSR Laplacian, fields on its full grid reduced to the orbits, its own Hartree solver; forces (reference solver), MPI runs and restarts keep their routes (see "Wires, slabs and oblique cells"). A100, one process (four visible GPUs), first runs, full grid -> sectors: rectangular graphene supercells (PBE, slab of half height 8 bohr, h 0.25) 4x2 (32 atoms, 295k points x 78 states, 32 operations) SCF 9.06 -> 12.77 s, GPU peak 3.55 -> 2.31 GiB, 6x3 (72 atoms, 664k x 170, 16 operations) 35.47 -> 21.28 s, 8.21 -> 5.08 GiB; hexagonal slabs with `PARSEC_OBLIQUE_SECTORS=auto`: h-BN 6x6 (72 atoms, 803k x 170) 42.73 -> 18.61 s, 9.94 -> 6.12 GiB, graphene 6x6 31.53 -> 20.88 s, 9.60 -> 5.91 GiB; wires (H-passivated armchair graphene ribbons along x, PBE, the 8 operations of D2h): 26 atoms (439k x 57) 18.79 -> 14.86 s, 5.40 -> 4.79 GiB, 38 atoms (817k x 84) 50.90 -> 40.89 s, 9.97 -> 7.72 GiB; total energies within 4e-11 Ry, and within 2e-11 Ry for small slabs and wires taken below the floor (H2, an H4 square wire, rectangular h-BN, a carbon chain, a silicon bilayer, `PARSEC_PERIODIC_SECTOR_FLOOR=off`). Opt-in because a slab just above the floor ran slower (the 32 sectors of the 4x2 cell), and because the preparation of a wire or slab (the superposed atomic density summed over every image within the 76 bohr of an FHI table, on the host) takes 87 to 594 s of these runs on either route | `off`: the full grid, bit for bit |
| `PARSEC_OBLIQUE_SECTORS` | `off` (default): a cell whose lattice vectors are not along the Cartesian axes (and a wire or slab of that kind) keeps the full grid of one device at the Gamma point, as before | `auto` (since 2026-11-22, opt-in): such a cell (the default routines' oblique grid, the general grid of the Fortran-derived routines; a hexagonal slab with `PARSEC_SLAB_WIRE_SECTORS=auto`) at the Gamma point, in one process, unpolarized, with a native LDA or GGA, above the floor, solves its orbitals in the sectors of the operations whose action on the grid indices is a signed permutation of the axes, an isometry of the cell that keeps its finite-difference operators (`Symmetry/oblique_frame.py`: detection in the frame of the grid indices; a hexagonal cell 8 of its 24 operations, primitive fcc 12 of 48), with translations; its sector Laplacian takes the mean weight of every class of equivalent second differences where a cell typed to finite precision leaves them unequal (about 1e-11). Its forces and relaxations keep the sectors (since 2026-11-22: the operations turned Cartesian, `R = S W S^-1`, `Forces/sector_symmetry.py`); Molecular dynamics and manual moves (`PARSEC_MOVING_SECTORS`) plan its sectors as they plan those of a box, with the same Cartesian operations. The MPI full-SCF runner refuses an oblique cell before its preparation under either value, as before. The command line hands a cell on the general grid of the Fortran-derived routines (FHI tables, a core correction) to the reference solver whatever the switch, so for those cells the switch acts through `prepare_single_point` only. A100, one process (four visible GPUs), first runs, full grid -> sectors: primitive fcc Si 4x4x4 (128 atoms, default routines, 195k points x 298 states, 48 operations, sectors of 16) SCF 73.33 -> 4.19 s, GPU peak 5.30 -> 1.34 GiB; hexagonal (lonsdaleite) Si 4x4x2 (128 atoms, 213k x 298, 16 operations) 15.47 -> 5.13 s, 5.61 -> 1.98 GiB; just above the floor hexagonal Si 3x3x2 (72 atoms, 120k x 170, 4 operations), fresh processes in both orders, SCF 5.8 / 5.8 / 5.8 -> 5.4 / 5.5 / 5.2 s with a preparation longer by 0.3-1.1 s: the same total, 2.03 -> 1.33 GiB (the gain is not general near the floor); on the general grid of the Fortran-derived routines (`prepare_single_point`) hcp Mg 4x4x2 (64 atoms, PBE, 229k x 89, 32 operations) 13.86 -> 6.84 s, 2.71 -> 0.80 GiB, primitive fcc Si 4x4x4 (FHI) 51.34 -> 5.25 s, 4.46 -> 0.67 GiB, and a tetragonal Si cell of unequal steps (3x3x3 cubes stretched to c = 11.2 bohr, 216 atoms, 238k x 501, 32 operations) 81.96 -> 8.80 s, 7.77 -> 1.42 GiB, 8.4e-10 Ry (their preparation, 496-810 s, is the general grid's on either route); total energies within 7.5e-12 Ry of the full grid in the other five, within 4e-10 Ry for small cells below the floor (primitive fcc and hexagonal Si on the default routines, hcp Mg and primitive fcc Si on the general grid, hexagonal h-BN and graphene slabs; `PARSEC_PERIODIC_SECTOR_FLOOR=off`) and within 1.8e-9 Ry for tetragonal Si on the general grid of unequal steps (criterion 1e-8). Below the floor of `PARSEC_PERIODIC_SECTOR_FLOOR` an SCF keeps the full grid (the same bits); forces and relaxations keep the sectors there (`sector_forces`), about 1 s slower than the full grid on cells of 16 to 27 atoms | `off`: the full grid, bit for bit |
| `PARSEC_SECTOR_PALETTE_LADDER` | `auto` (since 2026-11-19): every periodic sector cell is checked before its stencils are built (`driver._sector_palette_check`): a bound (`sector_palette_bounds`: the whole table of stencil values x signs x orbit-size ratios and the exact sums of the rows near a symmetry element; 0.26 s of CPU for the 62 M points of a 22^3 diamond cell, where the count takes 3.4 s), the count where the bound exceeds 256. A cell beyond the palette (a cube whose orbits differ in size: an even repeat on n = 2 mod 4 points per edge, Si 2x2x2 at Grid_Spacing 0.6 needs 295 at the default order 12, where the build of the stencils stopped the run with PaletteOverflow) steps down, the same on every rank: weighted sectors -> the commuting group with translations -> the commuting signed permutations about the origin (D2 of a cubic crystal) -> the full grid (the identity sector of an MPI run), with a NOTE in parsec.out. Since the two-byte codes of 2026-11-19 (`PARSEC_SECTOR_PALETTE_CODES=auto`, the row below) a cell reaches the ladder only under `PARSEC_SECTOR_PALETTE_CODES=off` or beyond 65,536 values | `off`: the full grid at once, as cells with unequal steps did before (a cube is still checked: it stopped before) |
| `PARSEC_SECTOR_PALETTE_CODES` | `auto` (since 2026-11-19): the packed stencil of a symmetry sector whose palette has more than 256 coefficients names them by two-byte codes (`uint16`, up to 65,536 values) in place of one-byte ones, so that the cell keeps the sectors of its group instead of stepping down the palette ladder; every other stencil keeps its one-byte codes, kernels and bits (`Symmetry/sector_stencil.py`, `backends/cupy_stencil_major.py`: the slot-major, pipelined, affine-tile and FP32 kernels compiled with `unsigned short` codes; the CSR route, the operator cache, the replicas of shared sectors; a cluster sector the native kernel refuses is built by NumPy's builder). The palette check counts the sectors whose bound exceeds 256 as before and names the count in `symmetry_detection`; `orbital_sector_coefficient_codes` in the backend details says which sectors take two bytes. Measured on one A100 node (four GPUs; preparation + SCF; `off` is the ladder's group, D2 about the origin here), with the node's CuPy kernel cache, the variants one after another in one process as in the measurements of the ladder: diamond 4^3 (512 atoms) on 72 points per edge, order 12 (298 values): weighted sectors 4.0 + 10.1 s, 14 steps, against D2 9.8 + 9.1 s, 16 steps, and the full grid 0.9 + 22.5 s (again: 4.2 + 10.0 against 7.4 + 10.2 s); Si 4^3 at Grid_Spacing 0.58 (70 points, 294 values): 3.9 + 11.0 s, 15 steps, against 8.9 + 23.7 s, 25 steps (again: 3.9 + 11.1 against 7.0 + 24.4 s); Si 2x2x2 at 0.6 (295 values) 2.8 + 2.7 s against 4.6 + 3.9 s (full grid 0.2 + 2.4 s). First runs, each in a process of its own with an empty kernel cache, and the peak memory of the fullest GPU in them: diamond 4^3 9.0 + 15.4 s, 0.62 GiB, against 11.5 + 24.1 s, 1.44 GiB (full grid 5.2 + 27.4 s, 17.0 GiB); Si 4^3 8.8 + 29.7 s, 0.73 GiB, against 11.2 + 52.6 s, 8.38 GiB (full grid 5.2 + 238.3 s, 24.2 GiB); Si 2x2x2 at 0.6 7.9 + 21.6 s, 0.48 GiB, against 9.1 + 22.5 s, 0.54 GiB (full grid 5.3 + 12.4 s, 1.64 GiB). An accepted exception: on a cell of about 9,000 points the default is slower than the ladder it replaces. Tetragonal Si at order 18 (8 atoms, 20 x 20 x 22 points, 306 values) takes 0.3 + 2.2 s in ten weighted sectors against 0.6 + 1.2 s on the ladder's D2 and 0.1 + 0.6 s on the full grid (in one process as above; first runs 5.8 + 20.7 s against 5.3 + 20.0 s and the full grid 5.0 + 7.0 s). The weighted route is as slow on such cells where their stencils fit one byte, which the cost model of the sectors sends to it today: the same cell at order 12 (135 values) 0.3 + 2.4 s against D2 0.2 + 1.3 s and the full grid 0.1 + 0.6 s, the cubic 8-atom Si cell at Grid_Spacing 0.5 (20^3 points, 72 values) 1.4 + 1.8 s against 0.5 + 1.2 s and 0.1 + 0.6 s. Which route a cell this small takes belongs to that cost model, not to the width of the codes, and this switch is not tuned for it. The MPI runner, one node of four ranks with a GPU each, diamond 4^3: 5.3 + 1.5 s, 681 MiB at the fullest GPU, against 7.9 + 2.2 s, 1517 MiB. Energies against the full grid within 8e-11 Ry (the runner's the driver's bits). The kernels of two-byte codes read 6 instead of 5 bytes per row and slot (2.26 instead of 1.88 in tiles of 16 rows of which 2/3 are regular) and give the bits of one-byte codes; the recurrence takes 2 to 6 % longer on stencils of 0.9 to 2.1 million rows (pipelined, slot-major +6 %, tiles +2 %), no measurable time on sector stencils of 12,000 rows; `on` on a cell that fits one byte (diamond 6^3, 1,728 atoms, first runs): 79.2 against 79.1 s, the same bits | `off`: one-byte codes and the palette ladder, as before; `on`: two-byte codes in every sector stencil (a measurement setting, the same bits as `auto`) |
| `PARSEC_DEVICE_EWALD` | `auto` (default, since 2026-11-17): the Ewald ion-ion energy and forces of a cell of 1,024 atoms or more on the device where the run has one (`V_ion/ewald_device.py`, installed with the periodic setup): the real-space pairs from the host's images binned in cells of the cutoff, one thread per atom over the 27 cells around it, and the reciprocal sum as the host's block products in CuPy; the same terms, the sums to round-off (diamond, 64,000 atoms on 16 A100: 0.7 s instead of 16.8 s, the same printed energies) | `off`: the host sums, bit for bit |
| `PARSEC_MPI_SHARED_EIGH_HELPER` | `on`: under the block route the root of a basis that three or more ranks share hands the eigensolve of its small solve to another rank (the whitened matrix there and the eigenvectors back, solved in that rank's summed pair), and the condition helper estimates on its own summed overlap instead of a copy; the same bits (bulk diamond on 16 A100, Gamma point, 5,832 atoms: 41.9 -> 38.3 GiB at the fullest GPU, 303.4 and 302.7 s). The block route forms the whitening in the summed pair either way; with both and the pool that the planner now chooses, 8,000 atoms took 71.6 -> 68.1 GiB and 657.9 -> 651.5 s on the same nodes | `off`: the eigensolve and a copy of the overlap on the root and the helper, as before |
| `PARSEC_PERIODIC_LOCAL_ORBITS` | `auto` (since 2026-11-19): the local ionic and core-correction forces of a k-point run on the irreducible wedge of its mesh (`PARSEC_KPOINT_FORCE_WEDGE`, or a weighted list with `Symmetrize_Charge_Density`; the density averaged over the operations of the wedge) and of a Gamma cell in the sectors of an abelian group (on either set of routines: the Fortran-derived ones with their self-consistency term, `PARSEC_FORTRAN_SECTOR_FORCES`) are summed for one atom per orbit of that group and turned onto the others (`Forces/total.py::_periodic_local_terms`, `Forces/periodic_local.py::orbit_local_forces`), as weighted sectors do since 2026-11-18 (`PARSEC_PERIODIC_SECTOR_LOCAL_FORCES`). `phase_seconds` of the forces then holds `local_representatives` and `local_site_asymmetry` (Ry/bohr, the checked atoms against their turned values): round-off for exactly symmetric structures, about 1e-10 Ry/bohr with the 12-decimal coordinates of `make_diamond.py`. With `PARSEC_PERIODIC_LOCAL_BALLS`: diamond 6^3 with the 3x3x3 mesh, one relaxation step on 16 A100 49.9 -> 42.3 s (forces 13.6 -> 0.49 s); the local terms of 8,000 atoms (10^3) of a k-point wedge on one A100 60.6 -> 0.49 s | `off` (or `0`, `false`, `no`; since 2026-11-19 this switch and `PARSEC_PERIODIC_LOCAL_BALLS`, `PARSEC_PERIODIC_FORCE_SERIES` and `PARSEC_PERIODIC_SECTOR_LOCAL_FORCES` are read at the preparation of every run, so that a value they do not know stops it before its SCF, `validate_local_force_settings`): every atom, bit for bit as before |
| `PARSEC_PERIODIC_LOCAL_BALLS` | `auto` (since 2026-11-19): on a GPU the ball sums of the local forces of an orthorhombic cell are formed for many atoms per call (`Forces/periodic_local.py`, `fast_periodic_local_forces`; every route of the default routines that runs them there: the full grid, the identity sector, abelian and weighted sectors, the MPI runner, k-point runs whose plan uses a GPU): each atom gets an index box whose points outside the cell stand for its images, one elementwise kernel forms the distance, the radial slopes, the erf slope and the product with the density or XC field, and the batch is reduced axis by axis. The per-atom loop built every ball on the host, about 7.5 ms an atom. The same terms summed in another order: the forces to round-off (<= 2.5e-15 Ry/bohr measured). Diamond 20^3 (64,000 atoms) without symmetry: the local forces about 382 -> 5.1 s on one A100 (3.9 s of it the reciprocal series, `PARSEC_PERIODIC_FORCE_SERIES`); a molecular-dynamics step of 6^3 at Gamma on 16 ranks, one per GPU: 22.9 -> 9.8 s | `off` (or `0`, `false`, `no`; read at the preparation, see the row above): the per-atom loop, bit for bit as before; host runs and oblique cells always keep it |
| `PARSEC_GAMMA_HYBRID_SECTORS` | `auto` (since 2026-11-19): an unpolarized hybrid (PBE0, HSE, HF, native B3LYP) of a bulk orthorhombic cell at the Gamma point whose basis (grid points x `States_Num`) reaches 2^24 entries is solved in one process in the real sectors of its abelian group with the half-cell translations (`SCF/gamma_hybrid_route.py`, `driver._gamma_hybrid_in_sectors`; a hybrid keeps no weighted sectors), the exchange in the classes of the half-cell translations of its sectors (`PARSEC_CUPY_EXCHANGE_TRANSLATIONS`) or through one full-grid block (`PARSEC_CUPY_EXCHANGE_POOL`), the energies those of the full grid to the SCF convergence ("Hybrid functionals of periodic cells on the device"). One node of 4 A100, PBE0 SCF: Si64 31.2 -> 26.7 s (GPU 8.3 -> 3.7 GiB), diamond 216 atoms 45.4 -> 27.3 s, 512 atoms 337.4 -> 111.4 s (49.0 s with the translation classes of the exchange). Forces and relaxations take the same sectors (`PARSEC_PERIODIC_SECTOR_FORCES=auto`; only the converged orbitals reach the forces): diamond 512 atoms with eight atoms (one and its images under the half-cell translations, so that the translation classes stay) moved by 0.05 bohr along (1, 1, 1), PBE0, BFGS with 2 moves on the Fortran-derived routines, 1e-6 Ry SCF and 1e-8 Ry Fock, first runs on one node of 4 A100 (2026-11-21), sectors -> full grid: forces within 6.7e-6, 1.03e-5 and 3.7e-6 Ry/bohr at the three structures (largest force 0.12 Ry/bohr; the second and third structures differ by the SCF noise of the 1e-6 Ry criterion, which the BFGS move carries into the positions), positions within 7.5e-6 bohr, energies within 1.0e-6 Ry; total 100.7 vs 1099.1 s (SCF 40.1 + 18.6 + 14.7 vs 346.0 + 398.8 + 303.5 s), GPU 0 peak 3.6 vs 45.7 GiB, host 2.9 vs 8.4 GiB. Smaller cells (their forces and relaxations on the full grid averaged over the operations of the cell: `PARSEC_GAMMA_HYBRID_SECTOR_FORCES`, the row below), spin, slabs, oblique cells, libxc and meta-GGA hybrids keep their routes. The MPI full-SCF runner, which refused every periodic hybrid before, runs the SCF of such a cell since 2026-11-22 (`driver._mpi_periodic_hybrid`): under `auto` in the same sectors above the same floor (every sector basis on one GPU of its rank) and as the one sector of the identity on the first GPU of the root below it and under `off`, the pair tiles of the exchange on every GPU of every rank with the cuFFT pair solver of the cell (`SCF/hybrid_mpi.py`, the tiles of the sectors: the translation classes and the pooled block of one process are not built there); its forces and moves, `kpoint` and spin stay refused. A100, the code defaults, against `main.py` on four GPUs: Si8 PBE0 (the sector of the identity) 2.5e-12 Ry, HSE06 4.4e-11 Ry; Si64 PBE0 (32 sectors, Fock loop to 1e-8 Ry) the same energy to the last bit and the same 36 SCF iterations on one rank of four GPUs, on four ranks of one GPU each and (1e-13 Ry) on two nodes; its SCF took 42.4 s on one rank (19.9 s in the pair tiles), 12.2 s on four ranks (6.2 s) and 26.4 s on two nodes, against 24.4 s in one process (2.0 s in the translation classes); diamond 512 atoms PBE0 (Fock loop to 1e-6 Ry, the same energy on every layout, 1.5e-7 Ry from `main.py`): 62.9 s of SCF on four ranks of one GPU each (57.8 s in the pair tiles), 41.0 s on two nodes of one rank each and 33.5 s on two nodes of four ranks each, against 25.6 s in one process with its kernel cache warm. The pair tiles of the sectors cost more than the translation classes of one process save on its GPUs, so a cell that fits one node is faster with `main.py`; the runner holds what one node cannot. `kpoint`: a large cell as the 1x1x1 k-point mesh of the reference solver in the k-point sectors of its little group (the same energies, 2-2.5x slower than the full grid on A100: kept for comparison) | `off`: the full grid of one process, bit for bit as before |
| `PARSEC_GAMMA_HYBRID_SECTOR_FORCES` | `auto` (since 2026-11-21): an unpolarized Gamma-point hybrid below the floor of `PARSEC_GAMMA_HYBRID_SECTORS` (2^24 entries) whose single-point forces or `bfgs`/`simple` relaxation the command line counts as forces of the sectors (`PARSEC_PERIODIC_SECTOR_FORCES=auto`, `prepare_single_point(sector_forces=True)`) keeps the full grid of its SCF, and on the default (Grieder) routines its forces and the moved positions of a relaxation are averaged there over the operations of the cell (`driver._gamma_hybrid_full_grid_forces`, `Forces/compute.py::compute_periodic_forces`): those the Fortran-derived routines average over on the full grid (`Forces.periodic.periodic_force_operations`: the space group the problem carries, else the point operations about the origin that keep the lattice, the grid and the atoms); parsec.out says so ("Forces of the periodic hybrid on the full grid (below the floor of its sectors): ... averaged over the N operations of the cell"). Until 2026-11-21 nothing averaged them on the default routines although the command line counted them as forces of the sectors; the Fortran-derived routines (`Cubic_Spline` for every species, an FHI pseudopotential, a core correction, ...) averaged theirs already and run as before. An accuracy fix: the SCF is the same (energies within 1e-12 Ry), the forces are their average over a group of the structure and the grid (the exact forces are invariant under it, and PARSEC 1.5 symmetrizes the forces of a cell with its group), and a relaxation stays in that group. One node of 4 A100, PBE0, one first run each (an empty CuPy kernel cache), total wall time off -> auto -> `sectors`: `examples/3d_Si` with forces (Td with the fcc translations) 34.7 -> 34.3 -> 51.5 s, the force asymmetry (largest change of the forces under the average over the operations of the cell) 2.0e-7 Ry/bohr under off, round-off under auto and in the sectors; rocksalt NaCl 8-atom cube (Martins pseudopotentials, Na at the origin moved 0.05 bohr along z: C4v, 1e-8 Ry SCF) single point 35.8 -> 35.7 -> 50.7 s, asymmetry 1.6e-8 Ry/bohr -> round-off; its BFGS with 2 moves 38.1 -> 38.5 -> 59.6 s, where off leaves C4v after the first move (one operation left at the third structure) and auto keeps all 8; Si8 of `examples/3d_Si` with atom 0 moved by (0.05, 0.05, 0.05) bohr (C3v), BFGS with 2 moves, at 1e-6 Ry SCF 41.8 -> 40.5 -> 47.5 s (asymmetry at the second structure 6.7e-6 Ry/bohr under off, 8.7e-6 in the sectors, whose abelian group has 2 of the 6 operations, round-off under auto), at 1e-9 Ry 43.1 -> 43.7 -> 54.7 s (4.0e-6, 4.8e-6, round-off); diamond 64 atoms on the default routines (`Cubic_Spline: false`, atom 0 moved along (1, 1, 1)) single point 34.3 -> 34.9 -> 35.8 s, BFGS with 2 moves 42.8 -> 42.1 -> 42.2 s; GPU and host peaks of auto those of off (the sectors 2.1 against 1.4 GiB of host memory on the 8-atom cells). Single-point forces of auto and of the sectors agree with those of off within the asymmetry the average removes (2.0e-7, 1.6e-8, 3.9e-8 Ry/bohr) and with each other within 7.8e-8 Ry/bohr. Relaxation trajectories agree only as far as the optimizer does not amplify that noise: Si8 and diamond 64 within 2.4e-7 and 6.5e-7 bohr (auto against off) and 2.3e-6 and 8.6e-7 bohr (`sectors` against off); in NaCl, whose first L-BFGS-B step of about 0.5 bohr is normalized from forces of 9e-5 Ry/bohr, the two moves carry force differences of 1e-8 Ry/bohr into positions 2.1e-4 bohr apart (auto against off), 8.3e-4 (`sectors` against off) and 1.0e-3 (auto against `sectors`), energies within 5.5e-4 Ry. On the Fortran-derived routines auto gives the bits of off: MgO 8-atom cube (FHI pseudopotentials with a core correction) single point 34.0 / 33.8 s, diamond 64 atoms with `Cubic_Spline` BFGS with 2 moves 42.1 / 42.0 s, every energy, force and position equal. Steepest descent (`simple`) of the silicon cube with Td about the origin (`test_gamma_hybrid_sector_forces`, 18^3, 1e-9 Ry): auto and `sectors` keep the atoms the group fixes, which the full grid moves with their round-off forces (by 9.5e-5 bohr after one step: the scheme of `domove.f90` moves every nonzero component by about Min_Step). `sectors`: such a cell takes the sectors of `PARSEC_GAMMA_HYBRID_SECTORS=auto` whatever the size of its basis, averaged over the abelian group of the sectors (the first build of 2026-11-21; also measured: diamond 64 atoms with `Cubic_Spline`, BFGS with 2 moves at 1e-6 Ry, within 2.8e-6 Ry/bohr and 2.7e-6 bohr of the full grid; Si8 at 1e-9 Ry within 8.9e-6 Ry/bohr and 3.3e-7 bohr), 1.1-1.6 times slower than the full grid on the 8-atom cells above and neutral on diamond 64 atoms. A wrong value is refused at setup | `off`: the full grid below the floor, its forces and moves unaveraged on the default routines, bit for bit as before |
| `PARSEC_CUPY_EXCHANGE_POOL` | `auto` (since 2026-11-19): the device Fock loop of sector orbitals of a periodic cell at the Gamma point (`PARSEC_GAMMA_HYBRID_SECTORS=auto`; their pair densities have no parity classes, the cuFFT pair solver of the cell takes the full grid) unfolds them once into one full-grid block, forms its `W` with the tiles of the full grid and folds it back to every sector once (`Hamiltonian/exchange_device.py::_apply_pooled_exchange`). Where the sectors carry half-cell translations the coset blocks of `PARSEC_CUPY_EXCHANGE_TRANSLATIONS` come first (the row below); the block is what `off` of that switch and a cell without such translations take. Si64 PBE0 exchange 13-21 s -> 2.9 s per run. Since 2026-11-19 a build takes the block only where it fits the free memory of its device (the block, its `W`, the tile and fold work and 2 % of the device, `pooled_exchange_fits`), else the tiles of `off` (the pair solver of each Fock iteration in `parsec.out` names the builds of that step that took them, "in k of n builds", the summary those of the run; each run of the cached solver, such as the next structure, starts without the record of the one before); the build of every column frees the block of the occupied build it completes (before, it stayed on the device through the next inner loop: 2 x 8 N M bytes, 4 x at the next build; one node of 4 A100, PBE0, the same energies and SCF times: GPU 0 peak of diamond 512 atoms 19.3 -> 13.6 GiB, of Si64 3.7 -> 3.1 GiB). A wrong value is refused at setup | `off`: the tiles pair the blocks of every sector, each unfolded where a tile reads it |
| `PARSEC_CUPY_EXCHANGE_TRANSLATIONS` | `auto` (since 2026-11-19): the sector orbitals of a Gamma-point hybrid that take the pooled block (`PARSEC_CUPY_EXCHANGE_POOL=auto`) and whose sectors carry half-cell translations (a supercell: the translations of `PARSEC_PERIODIC_TRANSLATIONS`, `T` of them with the identity) solve their pair densities in the classes of those translations (`Hamiltonian/exchange_translations.py`): the orbitals are gathered once into the coset form of `T` (`N/|T|` points, one block per character on `T`); the pair density of two characters has Fourier components on `1/|T|` of the G vectors only, and each such class is solved exactly by complex cuFFTs of the reduced box (half the points along every axis `T` moves, two real densities per transform) times the kernel on its sublattice; `W` is accumulated on the coset form and folded into every sector from one point of each orbit. The energies are those of the pooled block to round-off (the same printed digits: Si64 PBE0, also at 1e-9 Ry SCF and 1e-10 Ry Fock; diamond 216 atoms PBE0 and HSE06; 512 atoms PBE0). One node of 4 A100, PBE0 SCF with a warm kernel cache: diamond 512 atoms (`|T|` = 8, 36^3 box) 111.7 -> 49.0 s, the exchange 69.7 -> 7.6 s (a build 9.9 -> 1.0 s), GPU 0 peak 13.6 -> 2.4 GiB; Si64 at the tight criteria 34.8 -> 31.0 s (a build 0.57 -> 0.15 s, GPU 0 3.6 -> 1.1 GiB). First runs with an empty CuPy kernel cache: 512 atoms 123.9 -> 64.1 s (GPU 0 13.6 -> 2.4 GiB, GPUs 1-3 1.54 -> 1.23 GiB); diamond 216 atoms (`|T|` = 4: two classes per pair on the 27^3 box) 38.1 -> 31.8 s (a build 1.39 -> 0.28 s, GPU 0 3.2 -> 1.7 GiB; GPUs 1-3 rise from 0.97-1.01 to 1.28-1.32 GiB: the class boxes of a whole tile are larger than the 32-column real-to-complex work of the pooled path on the devices that take tiles, the largest device of the job still falls), with HSE06 36.3 -> 30.4 s. Outside the diamond family (`|T|` = 2; one build of random sector orbitals, 600 states, the occupied columns and their completion, warm): a body-centred cube of 64^3 points (four classes per pair on the 32^3 box) 2.59 -> 1.09 s on 4 GPUs and 8.74 -> 3.50 s on one, a simple 2 x 1 x 1 supercell of 112 x 56 x 56 points (x halved alone) 3.30 -> 1.23 s and 11.57 -> 3.79 s, the same energies; the CuPy pool of GPU 0 grows by 5.5 -> 1.7 and 7.6 -> 1.6 GiB in the build, that of GPUs 1-3 by 0.02-0.05 -> 1.0-1.2 GiB. A cell whose sectors carry no half translation and a slab's padded box keep the pooled block; a build whose coset blocks do not fit the free memory of its device takes the pooled block where that fits, else the sector tiles (the block is `|T|` times larger, so usually the tiles; the pair solver in `parsec.out` then says that the coset blocks did not fit the device in k of n builds); `PARSEC_CUPY_EXCHANGE_POOL=off` keeps the sector tiles. A wrong value is refused at setup | `off`: the pooled full-grid block, bit for bit as before |
| `PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE` | `auto` (since 2026-11-21): the update of `W` from the class potentials of a batch of pairs in the translation classes (`exchange_translations.translation_update`) runs one thread per coset row and column of `W` (`translation_pair_update_columns`) where a pair density has at most two classes (`COLUMN_UPDATE_CLASSES`: one for a supercell of even repeats, two for the face-centring translations of a cube): each column takes the pairs that write it (`column_events`) in the order of the pairs, with the same operations in the same order as the kernel by rows, so `W` and the energies are the same to the last bit (one GPU: the same SHA-256 of `W` in every case below; several GPUs add the parts of their tiles in an order of their own under either value). The kernel by rows ran one thread per coset row through every pair of the batch, a few hundred thread blocks for the coset form of a 512-atom cell, a third of the GPU time of a build. One build of random sector orbitals (the occupied columns and their completion, warm), the update on one A100 / on four, off -> auto: diamond 512 atoms (72^3, `|T|` = 8, one class, 1030 states) 0.94 -> 0.54 s / 0.26 -> 0.14 s, the build of the occupied columns 3.09 -> 2.69 s / 0.93 -> 0.85 s; diamond 216 atoms (54^3, `|T|` = 4, two classes, 438 states) 0.21 -> 0.14 s / 0.06 -> 0.04 s, that build 0.62 -> 0.56 s / 0.26 -> 0.23 s; a simple 2 x 1 x 1 supercell (112 x 56 x 56, `|T|` = 2, one class, 600 states) 0.77 -> 0.61 s / 0.22 -> 0.17 s, that build 3.00 -> 2.85 s / 0.94 -> 0.93 s. A body-centred cube (64^3, `|T|` = 2, four classes) was slower by columns, 1.31 -> 1.75 s / 0.34 -> 0.45 s (the class potentials of a pair are read for `W_i` and again for `W_j`), and keeps the update by rows. PBE0 SCF on one node of 4 A100, first runs (an empty CuPy kernel cache), two each: diamond 512 atoms 65.1 / 65.1 -> 63.6 / 63.5 s (a Fock-loop build 1.01-1.07 -> 0.92-0.96 s, the same energy -5839.82386881 Ry, the same peaks); Si64 of `examples/3d_hybrid_functionals` (140 states, a build 0.15 s either way) 50.0 / 48.6 -> 48.3 / 49.3 s, the same energy. A wrong value is refused at setup | `off`: one thread per coset row, as before |
| `PARSEC_EXCHANGE_OCCUPATION_CUT` | `on` (default, since 2026-11-22): an orbital takes part in the exact exchange (the `Energy_Refinement` of B3LYP, the exchange energy and the Fock and ACE operators of the hybrids of clusters, cells and k-point meshes, on the host, the GPU and the MPI exchange service) only where its occupation exceeds the electron-count tolerance of the Fermi-level search, 1e-10 (`FERMI_COUNT_TOLERANCE`, `small` of PARSEC 1.5's `flevel.f90`; `V_xc/exact_exchange.py`, `exchange_occupation_threshold`). Every site compares the occupation `f` of the level itself, while the tolerance of the search bounds the count weighted by the k-points: on a mesh a loose empty level at a k-point of weight `w` can keep up to 1e-10/`w` and then still enters, so the gain does not reach every loose level of a hot periodic insulator. Since 2026-11-05 that search takes PARSEC 1.5's limits (1e-10, 36.7, 1000 steps; 1e-12, 35, 100 before) and, as 1.5's own bisection, stops at the first level whose count is within 1e-10: in the gap of a molecule the empty levels then keep occupations anywhere between 0 and about 1e-10, set by where the bisection lands, and those above 1e-12 entered the exchange, each with a Poisson solve per occupied orbital (CO of `examples/0d_xps_core_levels` 7 orbitals for 5 at a Fermi level 27 kT below the empty pair, 1.6e-12 and 1.4e-12 on it; with the limits of before 5, on the same code). The occupations, the density and every run without exact exchange are unchanged. Under `PARSEC_FOCK_DIIS_TAIL=off` a route of the Fock loop changes as well: such levels also leave the occupied blocks of its outer DIIS (`Exx_Outer_Mixing: diis`, `SCF/fock_diis.py`, `OuterDIIS`, which refuses a proposal whose natural occupations fall below 1/2 among the occupied ones: "another occupation pattern: plain step"), so a loop that took plain steps can take DIIS steps; its total then differs within `Exx_Tolerance` and ends closer to convergence (H4 chain, HF, 3300 K, the default `Exx_Tolerance` 1e-6, A100: `off` -2.9084360050540594 Ry after plain steps (123 pair solves), `on` -2.908436219190189 Ry with a DIIS step (81), 2.1e-7 Ry apart, `on` 3.5e-9 Ry from the value of a tight loop). Under its default `on` the window of the outer DIIS keeps out every level below 1e-3 whichever cut applies, so for tails up to its limit that route no longer depends on this switch (the H4 chain at 3300 K with `Exx_Tolerance` 1e-10: 9 outer iterations under either cut). A100, one GPU per run, first runs, the inputs of `examples/0d_xps_core_levels` (PBE, 1160 K, B3LYP refinement), `off` -> `on`: pair solves and refinement seconds CO 28 -> 15, 2.76 -> 1.63 s, its core-hole state 28 -> 15, 2.74 -> 1.60 s; PH3 28 -> 10, 3.85 -> 1.58 s; P4 136 -> 55, 18.13 -> 7.50 s (whole run 38.75 -> 28.24 s); P(CF3)3 903 -> 820, 228.56 -> 209.77 s; H3O+(H2O)5 325 -> 300, 104.98 -> 97.27 s; H3O+(H2O)15 2145 -> 2080, 1313.53 -> 1272.01 s; CH4 of `examples/0_CH4_CF4` 15 -> 10; with spin polarization CO 56 -> 30, 5.42 -> 3.13 s, CH3SH 72 -> 56, 12.25 -> 9.65 s; the totals equal in all printed digits, the refined totals at full precision within 1.7e-11 Ry (CO 6.6e-13, its core-hole state 1.7e-11, P4 1.3e-11, spin CO 4.9e-12, H3O+(H2O)5 8e-14 Ry), every energy, level and residual of `parsec.out` the same. PBE0 SCF at the same temperature, one A100: CO 280 -> 225 and P4 997 -> 880 pair solves (the first Fock build of P4 counts the 16 levels of its PBE start), in these runs, where the outer DIIS took no step, the totals within 2e-14 Ry and the same outer and inner loop counts, PH3 unchanged (no level between the cuts), the time of these small runs, whose pair solves are cheap, unchanged (42.1 / 42.3 s, 65.2 / 64.5 s); on the MPI full-SCF runner (4 ranks) CO 280 -> 225, P4 997 -> 880, PH3 217 -> 184 pair solves, the same printed totals, which loops carry the extra levels depending on round-off. Primitive fcc Au, PBE0, 4x4x4, where levels are partly filled: at 1000 K nothing changes; at 1500, 2500 and 4000 K, where levels of some k-points lie between the two cuts, 0.8, 6.2 and 2.7 % fewer pair solves (778,192 -> 772,332, 636,925 -> 597,430, 674,290 -> 656,263), the totals equal in all printed digits and the levels within 1e-11 Ry | `off`: the cut 1e-12 (`OCCUPATION_THRESHOLD`), as before, bit for bit |
| `PARSEC_FOCK_DIIS_TAIL` | `on` (default, since 2026-11-22): the window of the outer DIIS of a Fock loop (`Exx_Outer_Mixing: diis`, `SCF/fock_diis.py`) stores only the orbitals whose occupation exceeds 1e-3 (`FRACTIONAL_GUARD`, the bound of an integer occupation in its gate; `window_occupation_floor`); the empty levels between the cut of the exact exchange (`PARSEC_EXCHANGE_OCCUPATION_CUT`) and that bound, the thermal tail of a gap at a raised Fermi temperature, complete the ACE of the extrapolated density matrix as empty states (the host, GPU and MPI loops of clusters and cells and the Bloch loops of k-points alike), as long as none of them holds more than `1e-3 sqrt(Exx_Tolerance / Ry)` (`tail_limit`: 1e-6 at the default 1e-6 Ry, 1e-7 at 1e-8, 3.2e-8 at 1e-9, 1e-8 at 1e-10); a larger tail clears the window, and the loop runs the plain operators that the refused proposals ran before, the same bits. Before, such a level stood among the kept natural occupations below one half and every extrapolation was refused (`another occupation pattern: plain step`, on 30edaacf1 as well), so a hybrid at a finite temperature with an empty level above the cut lost the outer DIIS, and whether it did depended on `PARSEC_EXCHANGE_OCCUPATION_CUT` (the H4 chain below at 3300 K, 7e-12 on its empty pair, `Exx_Tolerance` 1e-10, the host loop: 6 refusals and 11 outer iterations with the cut 1e-12, an extrapolation and 9 with 1e-10; with `on` 9 under either cut). Why the limit: the stop test bounds dexx, which is quadratic in the change of the density matrix, so at the stop the levels are converged to about `sqrt(Exx_Tolerance)` Ry only, and at a finite temperature the energy is of first order in them through their occupations (`f ln(1/f)` per Ry of level error); after an extrapolation the empty levels stopped further from converged than after plain steps, and without the limit (a bent H4 chain, Hartree-Fock, the host loop, `Exx_Tolerance` 1e-9, tails of 2e-6 to 5e-4 at 9000 to 16000 K) the totals stopped 4e-9 to 2.3e-8 Ry from the converged ones against 2e-10 to 3e-9 Ry for the plain loop. The totals stay within the Fock criterion and usually end closer to the converged ones, not always: in 3 of 8 cases with tails just below the limit `on` ended farther than `off` (an H4 chain at 5200 K and 1e-8: +2.2e-9 -> +4.1e-9 Ry). The cost: at the default `Exx_Tolerance` a very small system often keeps its outer iterations and takes 2 to 8 % more pair solves, because the build of an extrapolated operator solves the occupied pairs again (the H4 chain below at 3700 K 120 -> 123, at 6100 K 147 -> 158); the larger cases below take fewer (CO, P4, Si, the spin runs). A100, one GPU per run, first runs, `off` -> `on`: outer iterations / pair solves, and the total minus that of a plain loop converged to 1e-11 or 1e-12 Ry. The H4 chain (Hartree-Fock, 0.5 bohr, 3700 K, 1.5e-10 on its empty pair): at the default `Exx_Tolerance` 6 / 120 -> 6 / 123, 2.2e-7 -> 3.6e-9 Ry; at 1e-10 11 / 210 -> 9 / 177, 1.5e-10 -> 6.3e-11 Ry; at 5000 K (4.7e-8) default 6 / 136 -> 6 / 142, 2.2e-7 -> 3.7e-9 Ry, at 1e-8 9 / 196 -> 6 / 142, 2.1e-9 -> 3.6e-9 Ry. Hartree-Fock of the molecules of `examples/0d_xps_core_levels` at raised temperatures, default `Exx_Tolerance`: CO at 6000 K (4.6e-8) 7 / 1,039 -> 6 / 948, 7.7e-7 -> 2.5e-7 Ry; CO at 5000 K (1.8e-9) 7 / 829 -> 6 / 728, 7.6e-7 -> 2.5e-7 Ry, at 1e-9 13 / 1,477 -> 9 / 1,066, 2.1e-9 -> 8.0e-10 Ry, 52.8 -> 50.0 s; P4 at 4500 K (4.3e-7) 9 / 5,139 -> 7 / 4,283, 1.5e-6 -> 6.1e-7 Ry, 73.8 -> 68.0 s. Spin-polarized: CO at 6000 K 7 / 2,078 -> 6 / 1,896, 7.7e-7 -> 2.5e-7 Ry, at 5000 K and 1e-9 13 / 2,954 -> 9 / 2,132, 2.0e-9 -> 6.5e-10 Ry; an H3 chain (Hartree-Fock, 3000 K, 7e-10) 7 / 174 -> 6 / 153, 4.2e-7 -> 5.7e-8 Ry, at 1e-10 11 / 262 -> 7 / 176 (its dexx ends negative, -4e-9 to -6e-9, on every route: the totals are not resolved below that). The Bloch loop: primitive Si (Hartree-Fock, 2x2x2, 0.25 bohr, 3000 K, 2.1e-7) 8 / 15,780 -> 6 / 12,754, 3.0e-7 -> 2.1e-8 Ry, 25.2 -> 21.6 s. The MPI full-SCF runner (4 ranks) took the outer iterations and pair solves of one process for H4 at 3700 K and 1e-10 (11 / 210 -> 9 / 177) and the spin H3 (11 / 262 -> 7 / 176), with its totals to 1e-8 Ry; where the start of the runner already differs from that of one process (as before this switch: CO with Hartree-Fock at 5850 K and 1e-9, the first dexx 5.756e-3 against 5.761e-3, a tail of 3.06e-8 against the limit 3.16e-8) the extrapolated steps may differ (10 / 1,564 on 4 ranks against 10 / 1,480), the totals agree (-42.47674186 Ry). The same bits: tails at or below the cut (H4 at 3300 K, CO at 4000 K), above the limit (H4 at 5000 K and 1e-10, CO at 8000 K, CO at 6000 K and 1e-9, P4 and Si at 1e-9, H3 at 6000 K), PBE0 loops that never extrapolate (CO and P4 at 1160 K and 1e-9, spin CO, Si at 3000 K), primitive fcc Au with PBE0 and 4x4x4 at 1500 and 4000 K (partly filled levels keep the gate closed); no committed example output changes (their hybrid loops run at 1 to 80 K or with `occup.in`, where no level lies between the cut and 1e-3) | `off`: the window stores every orbital the exact exchange counts, bit for bit as before |
| `PARSEC_GAMMA_SPIN_SECTORS` | `auto` (since 2026-11-19): a collinear spin-polarized bulk orthorhombic cell at the Gamma point with a native LDA or GGA (no hybrid or spin-orbit species; a meta-GGA and libxc in one process under `PARSEC_GAMMA_META_GGA_SECTORS` and `PARSEC_GAMMA_LIBXC_SECTORS`, TS under `PARSEC_GAMMA_TS_SECTORS=on`, LDA+U under the opt-in `PARSEC_LDAU_SPIN_SECTORS=auto`), a single point in one process on CuPy or in the full-SCF runner (forces and relaxations: `PARSEC_SPIN_SECTOR_FORCES`), solves both channels in the abelian sectors of the cell with the half-cell translations (`SCF/gamma_spin.py`; the weighted sectors of a non-abelian point group in one process since 2026-11-19, `PARSEC_ORBITAL_GROUP_SPIN`), with the guard of `Spin_Symmetry_Sectors` (README "Spin-polarized cells at the Gamma point"). V- in diamond 5^3 (999 atoms, S = 3/2): one process on one node SCF 40.6 -> 24.6 s, 10.4 -> 4.4 GB; 16 A100, the runner with one rank per GPU SCF 3.1 s against 10.6 s of the k-point pools of `main.py --mpi`; energies within 1e-9 Ry of the 1x1x1 driver | `off`: the reference CLI's 1x1x1 k-point driver, refused by the runner, as before |
| `PARSEC_SYMMETRY_KEYWORD_ROUTINES` | `neutral` (since 2026-11-21): `Spin_Symmetry_Sectors`, the keyword of the spin sectors of this port, chooses no routines (`Input/routines.py::SYMMETRY_KEYWORDS`): a periodic input keeps the routines, grid, atoms and local potential it has without the line. PARSEC 1.5 has no such keyword; its symmetry keyword `Ignore_Symmetry` is read by both sets of routines. Before, any written value moved a periodic input to the Fortran-derived routines as a label of neither set, with a warning, and their local potential follows `Cubic_Spline` (the linear interpolation of the table without it, where the default routines take the cubic spline: at most 7.3e-4 Ry apart in the ionic potential of a Cr2 cell, 7e-15 with `Cubic_Spline: .true.`), so that `Spin_Symmetry_Sectors: off` ended away from the sectors and from `--symmetry off`. A100, PBE, the accelerated driver: the antiferromagnetic Cr2 pair in a 14.4 bohr cube, `off` (the full grid of the default routines) 3.6e-14 Ry from the sectors and equal to `--symmetry off`, 6.1e-4 Ry before; MnO AFM-II in 64 atoms 2.0e-12 Ry (2.3e-2 before). The parser reads the switch for every input | `fortran`: the former choice, bit for bit |
| `PARSEC_SPIN_START` | `auto` (since 2026-11-21): the spin loops of the reference solver that began from one polarization fraction of the whole system start as PARSEC 1.5's `initchrg.f90`, `run_spin_scf` and this driver do where the species of the atoms start with `Initial_Spin_Polarization` values of both signs (an antiferromagnetic or broken-symmetry start of two labels; `SCF/spin_polarized.py::opposite_spin_starts`): each channel the superposition of the atomic densities with `1 +/- Initial_Spin_Polarization` of their species, scaled by the factor of the total density (the channels add up to twice it, as in PARSEC 1.5), and the first Hartree potential that of the total starting density. Where the values have one sign (one value, or a species at 0.3 beside another at the default 0.1 or at 0), whose fraction keeps the sign of the pattern, the former start, bit for bit. The loops are the collinear k-point loop (`SCF/kpoints.py::run_scf_kpoints_spin_polarized`: spin cells with k-points, the 1x1x1 run of `main.py` for a spin cell at the Gamma point that leaves the sectors, `PARSEC_GAMMA_SPIN_SECTORS=off`, `Spin_Symmetry_Sectors: off`, `Symmetrize_Charge_Density`, `occup.in`, ..., and the ionic loops of `PARSEC_SPIN_IONIC_KPOINT`), `run_scf_spin_polarized` (a spin cell without a device, a spin run with spin-orbit species) and the two self-consistent spin-orbit loops with collinear spin (`SO_from_scratch` with `Spin_Polarization`: `SCF/self_consistent_soc_spin_polarized.py`, isolated; `SCF/kpoints_soc.py::run_self_consistent_soc_kpoints_spin_polarized`, periodic). The fraction was the ionic-charge weighted mean of the species' values, zero for opposite values of two labels, which then started non-magnetic. A100, the 1x1x1 k-point loop against the accelerated sectors of the same cell: antiferromagnetic Cr2 in a 14.4 bohr cube (PBE) 2.7e-9 Ry, 44 iterations, 5.6 s (before: 300 iterations without converging, 0.679 Ry above, 33.1 s; `main.py` with `Spin_Symmetry_Sectors: off` 8.8 s, converged); MnO AFM-II in 64 atoms 4.0e-8 Ry at 1e-7, 33 iterations, 13.5 s (before: 300 iterations, 174 Ry off, 424 s); stretched H2 with two labels 4.8e-10 Ry, 27 iterations (before: the non-magnetic state, 2.1e-2 Ry above). Values of one sign, the SCAN and PBE silicon cell of the meta-GGA tests with its two atoms as labels at 0.1 and 0.3 (k-point loop, 1e-7 Ry): the same bits as `weighted` (SCAN 49, PBE 14 iterations; PARSEC 1.5's start there, `initchrg`, took 59 and 27, and SCAN ended in a magnetized state 1.4e-5 Ry lower). Opposite values there (0.1 and -0.3): PBE the same energy in 27 against 13 iterations; SCAN a locally magnetized state (int abs(m) = 0.35 electrons) 1.9e-4 Ry below the non-magnetic one that the former start keeps. The spin-orbit loops with spin (`SO_from_scratch`, CA, an Au2 pair with two labels at +0.8 and -0.8, 1e-5 Ry): the isolated pair 10 bohr apart reaches its antiferromagnetic state (+-0.95 on each atom) 11.7 mRy below the non-magnetic one of the former start, in 18 against 31 iterations (92.9 against 151.9 s); 8 bohr apart both end non-magnetic, 1.2e-5 Ry apart (26 against 14 iterations); the periodic loop of that pair in a 16 bohr cube the same energy in 20 against 12 iterations; starts of one sign (0.8 and 0.1) the same bits in both loops. | `initchrg`: PARSEC 1.5's start for every input, also with values of one sign (the same energies, other paths: V- in 2^3 diamond with CA and PBE one iteration more, 29 against 28 and 33 against 32; the SCAN silicon cell of the meta-GGA tests 72 against 33 iterations; the isolated AuH example with a spin-orbit species, `examples/0d_AuH/reduced_perturbative_soc`, follows PARSEC 1.5's iterations, -68.0284, -68.2441, -68.3184 Ry against Fortran's -67.9517, -68.2381, -68.3186, before -68.2073, -68.3029, -68.3236, and ends 1.9e-7 Ry from Fortran's energy instead of 6.0e-6), and for the channels a hybrid's Fock loop hands in the Hartree potential of the total (of their sum, twice the density, otherwise; the O2 PBE0 cell: the same energy and iterations); `weighted`: the former start for every input, bit for bit |
| `PARSEC_SPIN_SECTOR_FORCES` | `auto` (since 2026-11-19): the forces and relaxations (`bfgs`, `simple`) of a collinear spin-polarized run keep the sectors of both channels. `main.py`: a bulk cell at the Gamma point that `PARSEC_GAMMA_SPIN_SECTORS` takes goes to the accelerated driver instead of the 1x1x1 k-point driver where the forces of an unpolarized cell keep their sectors (`acceleration/cli.py::periodic_sector_forces_refusal`: every atom movable, no point charges, field or supercell of a k-point mesh) on the default routines (`SCF/gamma_spin.py`): the nonlocal force terms of both channels from their sector states, the forces and the moved positions averaged over the sector group (a relaxation stays in its group, as in PARSEC 1.5), the guard at every structure; after a fallback the forces of the full grid, averaged in a relaxation over the operations of its first structure (`Forces/compute.py::_left_spin_sectors`). The full-SCF runner takes these forces and relaxations on several ranks, and those of a spin cluster whose structure keeps its point group (`benchmarks/mpi_full_scf.py::_refuse_spin_ionic_steps`). Molecular dynamics, manual movements, fixed atoms (`Move_Flag`) and forces with `Ignore_Symmetry` of such a cell go to the accelerated driver too, on the full grid as an unpolarized cell's (both channels, no group, no guard), planned from the input (`driver.spin_full_grid_plan`, 4 times both bases): one device where that fits, one sector of the identity whose basis the GPUs of the process share where it does not (`PARSEC_SPIN_FALLBACK_SHARED`), the 1x1x1 k-point driver with a NOTE where neither holds it. The full-SCF runner (since 2026-11-21; refused on several ranks before, and by the driver on one) solves them, and those of a spin cluster without a point group (molecular dynamics, manual movements, `Ignore_Symmetry`, `--symmetry off`, a relaxation whose `Move_Flag` only the identity respects or under `PARSEC_MOVE_FLAG_SUBGROUP=off`; under the default `auto` it keeps the sectors of the subgroup that respects the flags, as before), as that sector of the identity of both channels, its basis shared by the ranks, by the GPUs of each rank (nested sharing) or by those of one rank (`benchmarks/mpi_full_scf.py::_spin_moves_without_group`, `prepare_single_point(spin_identity_sector=True)`, a NOTE in `parsec.out`); what `main.py` keeps on the k-point driver (below), a hybrid and a meta-GGA (one device per basis in that layout) it refuses. Like every spin-polarized run of the runner they also run under the `PARSEC_CUPY_DISTRIBUTED_FILTER=1` of `scripts/run_multi_gpu_scf.sh` since 2026-11-22 (before, the runner stopped 1.6 s after the start with a message that named the switch; `driver._spin_sector_refusal` keeps the full grid of one process under that value, as before): A100, the launch settings, `1` against `0`, where the filter does spread over several GPUs: the NH2 radical in Cs (spin forces, two sectors on one rank of four GPUs, each filtered by the distributed filter over two of them) the same energy and forces to the last bit, -22.033833289371316 Ry, 7e-15 Ry from `main.py` in the sectors and 2.7e-13 Ry and 1.2e-9 Ry/bohr from `--symmetry off`, and the same energy on two nodes of one rank each; OH in Cs, whose guard sends it to the shared sector of the identity (filtered over all four GPUs), the bits of `0`, of `main.py` and of `--symmetry off`, on one rank and on two nodes. Where each sector lies on one GPU (O2, V- 2^3 forces and molecular dynamics on four ranks, V- 3^3 and 4^3) the value changes no route, and the times are the same within noise. A100, CA, against `main.py` (one device): V- 2^3 molecular dynamics (3 steps, 1e-8 Ry) on one rank of 4 GPUs, 4 ranks and 2 nodes (the same bits of the final energy, forces within 4.2e-13 Ry/bohr, energies of the trajectory within 1.1e-13 Ry and positions within 7e-15 bohr of each other), energies 2.1e-11 Ry, positions 5.8e-10 bohr, forces 3.5e-7 Ry/bohr (the difference of the shared layout and one device in `main.py`, above); its manual movements on 4 ranks 1.1e-13 Ry, 1.9e-9 Ry/bohr, the same positions; its `bfgs` relaxation with `Move_Flag` (40 of 63 atoms, three structures) 4.7e-9 Ry, 7.6e-9 bohr, 3.5e-6 Ry/bohr (`main.py` itself: 3.3e-6 to 3.5e-6 between the two layouts); the O2 cluster: molecular dynamics on 4 ranks 8.7e-11 Ry, 3.2e-10 bohr, 1.3e-8 Ry/bohr, manual movements on one rank 1.4e-13 Ry, 1.7e-8 Ry/bohr, `Move_Flag` fixing one atom (`PARSEC_MOVE_FLAG_SUBGROUP=off`) on 4 ranks 1.9e-9 Ry, 5.6e-7 bohr, 1.4e-6 Ry/bohr. The Fortran-derived routines (the default of the forces of a spin-polarized PBE cell, `Input/parsec_input.py::_default_routines_periodic_refusal`, and of `Cubic_Spline` inputs; their sector forces have not been checked for two channels) and `PARSEC_PERIODIC_SECTOR_FORCES=off` keep the k-point driver. A100, CA: O2 across a face of a 9.9 bohr cube, the forces of the k-point driver to 4.8e-11 Ry/bohr (largest 4.0); V- in diamond 2^3 (63 atoms, 1e-8 Ry) 2.1e-9, its SCF and forces 19.9 against 161.5 s; its relaxation (`bfgs`, 1e-6 Ry) 3-5 s against 57-107 s per structure, the energies those of the k-point driver to 2e-7 Ry until that driver, whose forces are not averaged, leaves the group by round-off (asymmetric forces of 1.7e-3 at the fourth structure); V- 3^3 (215 atoms, benchmark settings) four structures in 52.9 s (the k-point driver had not finished them after 44 min); V- 5^3 (999 atoms, benchmark settings) SCF and forces in 35.9 s, 4.3 GiB on the busiest of four GPUs (the k-point driver had not finished after 59 min); the runner on 4 ranks against `main.py`: V- 3^3 relaxation energies 9e-13 Ry, forces 1.9e-10 Ry/bohr, positions 1.9e-12 bohr, the O2 cluster forces 2.7e-14; 2 nodes of 4 ranks, V- 4^3 (511 atoms): 9e-13 Ry, 4.1e-11 Ry/bohr; molecular dynamics of V- 2^3 (3 steps, 1e-8 Ry): the trajectory of the k-point driver to 2.7e-14 bohr and 1.1e-13 Ry, 1.3 s against 91 s per step; its `bfgs` relaxation with `Move_Flag` (40 of 63 atoms movable, 1e-8 Ry): positions 4.9e-12 bohr, forces 2.7e-10 Ry/bohr, 1.8-8.8 against 123-588 s per structure; its manual movements (3): positions equal, forces 1.1e-14, 1.3-2.4 against 89-152 s per structure; its forces with `Ignore_Symmetry`: 5.8e-15, 2.5 against 159 s; V- 5^3 molecular dynamics (one step, benchmark settings), which ran out of memory on the full grid of one A100 (77.6 GiB after 102.6 s), on the identity sector of four GPUs: 52.4 s for both structures, 8.5 GiB on the busiest GPU, 32.4 GiB in all; V- 5^3 forces at 1e-6 Ry on that layout (`--symmetry off`, 53.7 s) against the sectors (37.4 s): 1.8e-6 Ry/bohr, energies 1.3e-5 Ry apart (not the route: from 1e-5 Ry on, both routes alternate between the same two energies 1.93e-5 Ry apart from one SCF iteration to the next, and stop on either at 1e-6; the CA correlation of PARSEC 1.5's `exc_spn.f90`, whose Perdew-Zunger branches differ by 6.4e-5 Ry at r_s = 1, flips on a Td orbit of 24 grid points at diamond's maximum density, 3/(4 pi) e/bohr^3, 24 x 0.2387 x 0.0525 bohr^3 x 6.4e-5 Ry = 1.93e-5 Ry in E_xc; at 1e-8 Ry the residual stalls at 1.4-3.7e-7 for the same reason); the neutral vacancy 2^3 after its guard fired (1e-9 Ry): the forces of the k-point driver to 1.3e-9 Ry/bohr, its SCF 4.4 against 287.6 s | `off`: the k-point driver for every forces, relaxation and dynamics run of a spin cell on the command line, and the runner refuses the forces, relaxations and dynamics of every spin-polarized run on several ranks, as before |
| `PARSEC_SPIN_IONIC_KPOINT` | `auto` (since 2026-11-21): the forces, relaxations and molecular dynamics of a collinear spin-polarized bulk cell at the Gamma point that run on the Fortran-derived routines of the reference solver (the default for the forces of a spin-polarized PBE cell and of `Cubic_Spline` inputs, and `Use_Fortran_Routines: .true.`; `main.py` hands them to it, see `PARSEC_SPIN_SECTOR_FORCES`) take the 1x1x1 k-point run of the device k-point solver, as the single point of such a cell does, instead of the Gamma-point loop `run_spin_scf` with the eigensolvers of the host (`cli.py::spin_ionic_kpoint_sampling`: semilocal functionals without LDA+U, TS, `occup.in`, `Symmetrize_Charge_Density` or spin-orbit species, a device k-point backend; the forces of `Forces/periodic.py` from the Bloch orbitals of k = 0, a moved structure started from the two channel potentials of the one before; `Restart_Run` and `Energy_Refinement`, which the parser refuses for such a cell, would keep the host loop). The host loop took 5 s per SCF iteration of V- in 2^3 diamond on an A100 node (its eigensolver on the CPU), the k-point solver 0.15 s. A100, one node, first runs, the benchmark PBE carbon (`Cubic_Spline`): V- 2^3 forces at 1e-8 Ry 174.8 -> 12.3 s, the same energy to the printed 1e-8 Ry, forces 2.1e-9 Ry/bohr, GPU peak 1.29 GiB on the busiest of four (2.54 GiB in all); the antiferromagnetic Cr2 pair in a 14.4 bohr cube (PBE, 1e-7 Ry) forces 152.2 -> 9.6 s, energies 1e-8 Ry, forces 2.2e-8 Ry/bohr, 1.15 GiB (2.19 GiB); V- 2^3 `bfgs` relaxation (3 moves, 1e-7 Ry) 469.9 -> 29.3 s, energies equal for three structures and 1.65e-6 Ry apart at the fourth, forces 1.7e-6 Ry/bohr, positions 2.0e-6 bohr (a force of 2.4 Ry/bohr on the displaced atom); its molecular dynamics (3 steps, 1e-8 Ry) 526.3 -> 31.6 s, energies 9.6e-10 Ry, positions 3.7e-10 bohr, forces 5.4e-9. `parsec.dat` of the k-point run holds the Bloch orbitals of k = 0 that it keeps for the forces, as the complex wavefunctions of one k-point at Gamma (the host loop writes real ones; since 2026-11-22 they are made real and the file is that of a run without k-points, `PARSEC_GAMMA_PARSEC_DAT`), and the mixed potentials: the stretched H2 pair in a 10 bohr cube with `Output_All_States` (6 + 6 states), against `off`: densities 7e-8, potentials 4e-5 Ry, levels 6e-7 Ry, each non-degenerate orbital the one of `off` up to a phase (overlap 1 to 3e-9), 6.5 against 20.2 s. The default routines keep the host loop where they get these runs (`PARSEC_SPIN_SECTOR_FORCES=off`, `PARSEC_PERIODIC_SECTOR_FORCES=off`, no full-grid layout): their periodic backend has no spin-polarized k-point forces. The command lines read the switch before every spin-polarized run | `off`: the host Gamma-point loop, as before |
| `PARSEC_KPOINT_WAVEFUNCTIONS` | `auto` (default, since 2026-11-22): the `parsec.dat` of a k-point run of scalar or collinear spin-polarized orbitals holds the selected states (the occupied ones by default, `Output_All_States`, `Save_Wfn_Bands`) of every spin channel and k-point, as `wfnsave.F90` writes them: the Bloch factors `u_nk` of the last iteration, normalized to one, the columns of each k-point in the order of its own levels (`Output/parsec_dat.py::result_channels`). Before, every k-point run wrote none (the spin ionic loop of one k-point excepted), with the NOTE 'k-point drivers keep no orbitals'. A single point, and every structure of the forces, relaxations and molecular dynamics of the default routines (`Relaxation/backend.periodic_backend`), keep the orbitals for the file where it selects some (`RunOutputFiles.kpoint_keep`; the other fields of `keep_wavefunctions` are dropped, so the file's potentials, the energies, forces and moves are those of before); the ionic loops of the Fortran-derived routines, hybrids, the Ylm-projected DOS and `Output_GW` keep them already. The size limit of the default selection (512 MiB) counts the k-points. The host holds the kept orbitals until the file is written: every state of every k-point for the default selection and `Output_All_States`, the lowest levels of each k-point up to the highest state of a `Save_Wfn_Bands` list, which the k-point solvers copy to the host without the others (all the states of k = 0 for an input without k-points, which are made real multiplet by multiplet); the file's columns are written from them in place. The MPI-distributed k-point solvers (`--mpi`) gather no orbitals: their file holds none, with a NOTE. A mesh solved as its supercell (`PARSEC_KPOINT_SUPERCELL`) writes the cell's file without orbitals (the supercell's states are no Bloch states of the cell); `Output_All_States` keeps the k-points, and a `Save_Wfn_Bands` block selects the Fortran-derived routines, which keep them too. A100, one node, first runs: Si8 (CA, 8,000 points) with a 3-point `Kpoint_List` (weights 1/4, 1/4, 1/2) and the 4-point wedge of a 2x2x2 mesh, scalar and collinear spin: every block holds its 24 states (`Output_All_States`) or the occupied ones, orthonormal to 3e-15, and the density of the stored orbitals (`2 sum_k w_k sum_n f_nk |u_nk|^2 / h^3` with one channel, per channel with two) is the file's to 4e-17; the levels, potentials and densities of the file are within 6e-15 Ry of a run without the orbitals (1e-10 on the mesh), the spread of two runs of the commit before; `Restart_Run`, a `Fix_Number_Eigenvalues` restart and `nscf_kpoints` from these files run; on 2 ranks with `--mpi` the same energy, no orbitals and the NOTE. Si64 (85,184 points, a 2-point list, the 130 occupied states of each k-point, a 359 MB file): 18.75 -> 19.01 s and 18.50 -> 19.29 s in two pairs of runs (the commit before: 18.78 to 19.33 s); its spin-polarized cell exceeds the limit with the k-points (707 MiB) and writes none, in the time of before (19.74/19.80 s). | `off`: no orbitals for k-point runs (the spin ionic loop of one k-point writes its kept ones), the limit without the k-points and the NOTE of before |
| `PARSEC_GAMMA_PARSEC_DAT` | `auto` (default, since 2026-11-22): an input without k-points (`Kpoint_Method: none`, PARSEC's `nkpt = 0` with real wavefunctions) that runs as the 1x1x1 k-point mesh (the spin ionic loop of `PARSEC_SPIN_IONIC_KPOINT`, a spin-polarized single point on the k-point driver: a wire, slab or meta-GGA cell, `PARSEC_GAMMA_SPIN_SECTORS=off`, `--mpi`; a spin-polarized periodic hybrid at the Gamma point) writes `parsec.dat` as PARSEC writes that of a run without k-points: `stype(2) = 0`, `kpnum` 1 with k = 0 and weight 1, and real wavefunctions, the k = 0 Bloch orbitals made real multiplet by multiplet (`Output/parsec_dat.py::real_gamma_orbitals`: the real basis of the span of each run of levels within 1e-4 Ry, the levels diagonalized again in it; a NOTE where a multiplet that `States_Num` cuts leaves a part outside). Every periodic file without k-points also takes PARSEC's `kptmethod` -1 (it was 0; no reader uses it). Before, such a file was that of a k-point run (`stype(2) = 1`, complex), which a `Restart_Run` with `nscf_kpoints` of the same input refused with error 362, as PARSEC's `restart_run.F90` does. A100: the stretched H2 pair in a 10 bohr cube (spin PBE forces, the ionic route, 6 + 6 states): the real orbitals against those of the host Gamma loop (`PARSEC_SPIN_IONIC_KPOINT=off`): levels within 3.3e-9 Ry, densities 4.5e-10, every level and multiplet the same span (the smallest singular value of their overlap 1.000000000), the density of the stored orbitals the file's to 3e-16; Si8 spin forces (`Use_Fortran_Routines`, multiplets of 3 and 6 at Gamma, 24 + 24 states): levels 4e-9, densities 1.9e-9, every complete multiplet the same span (the top pair, part of a multiplet that the 24 states cut, is another pair on the commit before alike). An `nscf_kpoints` restart from the file of the ionic route and from that of a single point on the k-point driver (`PARSEC_GAMMA_SPIN_SECTORS=off`) runs, where both stopped with error 362; the host loop's file is that of before record for record but `kptmethod`. | `off`: the complex file of a k-point run for those inputs and `kptmethod` 0, as before |
| `PARSEC_SPIN_FALLBACK_SHARED` | `auto` (since 2026-11-19): the full grid that a spin-polarized run in the sectors goes back to where its guard fires is planned at preparation (`driver.SpinFallbackPlan`: 4 times the bases of both channels, states and buffer at every point, against the memory of the first GPU; 77.4 GiB were in use after the second iteration of a 21.8 GiB basis before it ran out in the third). Where that does not fit one GPU, parsec.out says so in a NOTE and the fallback puts both channels in one sector of the identity whose basis the GPUs of the process share (`distributed_state`; `prepare_single_point(spin_identity_sector=True)`, no guard), or the ranks of the full-SCF runner, which all prepare it together (`MPISectorContext.reprepare`); where no layout fits, the run stops with an error before that full grid is prepared instead of running out of device memory after it (under `off` the full grid of one GPU is tried whatever the estimate, as before). The guard's NOTE in parsec.out names the layout taken and says that the forces and any later structure come from it. A spin cell that the command line solves on the full grid from its start (molecular dynamics, manual movements, `Move_Flag`, `Ignore_Symmetry`, `PARSEC_SPIN_SECTOR_FORCES`) takes the same layouts, planned from the input, and keeps the 1x1x1 k-point driver where none fits. A hybrid's exchange and a meta-GGA's orbital term keep the layout of `off`. A100: the neutral vacancy in diamond 5^3 (999 atoms, PBE, 500 K), whose guard fires after 10 sector iterations (2 electrons in the up t2 levels), ran out of memory on one GPU; now its full grid converges on the four GPUs of one process (SCF 55.9 s, total 87.9 s, 8.6 GiB on the busiest GPU, 32.5 GiB in all); under `off` it runs out of memory on one GPU, as before (the early error of a run that fits no layout, measured on this run before `off` tried one device: 45.8 s, at its guard). The shared layout gives the levels and densities of the full grid of one device (the O2 triplet: 1e-8 Ry, 1e-7 e/bohr^3); the neutral vacancy 2^3 relaxation (three structures) after the fallback: 7e-8 Ry between the two layouts and the k-point driver in one process, 3.6e-7 Ry and 1.4e-4 Ry/bohr (SCF criterion 1e-6 Ry) between the runner (4 ranks) and `main.py`; at 1e-9 Ry its forces after the fallback 1.2e-9 Ry/bohr between the two layouts, 1.3e-9 against the k-point driver, and 1.2e-9 between the runner's shared full grid and `main.py` (a cell's shared full grid stopped the runner's preparation with 'it is prepared without symmetry' from the merge of 2026-11-20 until 2026-11-21, when the driver's early refusal of spin cells without a group was lifted for this sector: again 1.2e-9 Ry/bohr and 6.6e-12 Ry on 4 ranks); the O2+ cluster at 3000 K (the guard fires at the start) on 4 ranks against `main.py`: 1e-8 Ry, 2.9e-9 Ry/bohr. A runner of one rank per node with several GPUs each takes that layout too: with a CHEBFF first solve each rank spreads its share over the GPUs of its node (`PARSEC_MPI_NESTED_SHARING=auto`, the plan counts every GPU, merge of 2026-11-20); after CHEBDAV or under `PARSEC_MPI_NESTED_SHARING=off` each rank's share lies on the first GPU of its node (not refused as `PARSEC_SINGLE_SECTOR_NODE_LAYOUT` refuses a run without a group: the sector SCF ran first), and the NOTE of the preparation says so and advises `scripts/run_multi_gpu_scf.sh --rank-per-gpu`, which shares it over every GPU (since 2026-11-20) | `on`: the shared layout for every fallback (tests); `off`: the full grid of the first GPU of the process (of the root in the runner), as before, whatever the estimate (its NOTE says where it would not fit); a spin cell on the full grid from its start that does not fit one GPU keeps the k-point driver |
| `PARSEC_SPIN_FALLBACK_DENSITY` | `off` (default): the full grid that a spin cluster leaves its symmetry sectors for keeps no symmetry, as before. Opt-in `abelian` (since 2026-11-22): that full grid (one process; not the shared layouts of `PARSEC_SPIN_FALLBACK_SHARED`, not MPI ranks, not a cell) averages the density of each channel over the abelian group of the atoms (the operations that only change the signs of coordinates, PARSEC's `Found Abelian subgroup`) in every iteration, as PARSEC with its symmetry and the reference solver of this port keep it (`SCF/fallback_density.py`; the NOTE of the fallback names the group; after every ionic step the group of the moved atoms, which a relaxation along the axis of the pair keeps: the broken-symmetry O2 pair, three BFGS steps, every loop averaged and converged in 40, 49, 19 and 17 iterations). Atoms and linear molecules with a partly filled degenerate shell have a continuous family of states of one energy on the full grid without symmetry (the orientation of the occupied orbitals), along which the loop drifts without converging; PARSEC without symmetry (`Ignore_Symmetry`, 1.5's default for clusters) shows the same plateau (broken-symmetry O2: a residual of 2e-3 Ry for about 100 iterations), PARSEC with its abelian group converges (O2 25, the Fe atom 28 iterations). A100, one GPU, criterion 1e-6 Ry (O2) and 1e-5 Ry (Fe): broken-symmetry O2 (`Chebdav_Degree: 30`, whose start Hamiltonian stops the sectors) 150 iterations without converging (residual 3e-4) -> converged in 40, -63.70691983 Ry, the energy the full grid reaches with the generalized Broyden mixer in 143; the Fe atom (PBE, 100 K) 150 iterations with the energy still falling (-39.40393 Ry, residual 7e-4) -> converged in 34, -39.39776499 Ry (the run 23.9 -> 10.7 s; from `Initial_Spin_Polarization` 0.1 the same state in 34); the Mn atom, whose half-filled shell the sectors also leave, the same energy and iterations (21). The state of the abelian group lies above a state that breaks it too (the Fe atom), which is why this is opt-in. | `off`: the full grid without symmetry, as before |
| `PARSEC_SPIN_FLIP_SYMMETRY` | `off` (since 2026-11-19, `auto` on request): `auto` solves the up channel of a collinear spin-polarized single point or relaxation alone (every structure checked; molecular dynamics steps alike, not measured) and takes the down channel as its image when the start is antiferromagnetic or broken-symmetry by two labels of one species with opposite `Initial_Spin_Polarization` (species identical otherwise, the pseudopotential file by its contents; a label at 0 is its own partner) and one operation of the atoms and the grid exchanges the partner labels (a signed permutation about the origin of a cluster; `r -> R r + t` of a bulk orthorhombic cell, `t` whole grid steps, a pure translation first) and maps the up start density onto the down one to 1e-6 electrons (`SCF/spin_flip.py`): the down levels, occupations, density (one gather of the up orbit values), orbitals (formed where they are read), sector labels and nonlocal forces are those of the up channel carried by the operation; in the sectors of the run (clusters, cells at the Gamma point of `PARSEC_GAMMA_SPIN_SECTORS`, the full-SCF runner; also the weighted sectors of `PARSEC_ORBITAL_GROUP_SPIN`, a dimer along z in C4v, whose down channel takes the state weights and partners of the up channel, merge of 2026-11-20; and the forces of a spin cell in the sectors of `PARSEC_SPIN_SECTOR_FORCES`, Na2 in a cube: those of both channels solved to 1e-6 Ry/bohr) or on the full grid of the accelerated driver (the fallback of the spin guard included); the 1x1x1 and k-point drivers of the reference CLI keep both channels. Refused (both channels; a `NOTE` in `parsec.out` at every `Output_Level` and the detail `spin_flip_symmetry` say why): hybrids, meta-GGAs, LDA+U, `Symmetrize_Charge_Density`, spin-orbit or non-collinear runs, an odd electron count, `occup.in` with unequal channels, no partner labels, no operation, or only operations that do not map the local ionic potential onto itself within 1e-10 of max(1, max \|V_ion\|) (point charges of `Add_Point_Charges` or another external term: an operation that keeps them is taken if there is one). The spin loop checks every loop again and solves both channels, with a `NOTE` before its iterations, where the atoms lost the operation (a relaxation step), the ionic potential of the loop breaks it (the six field loops of `Polarizability`; the zero-field loop keeps the route) or the down potential of a warm start is not the image of its up potential within 1e-8 Ry (`Restart_Run` and `Restart_From_Wfndat` of a state that breaks the flip; warm starts of the structures of a flip relaxation keep it, 1e-13 Ry). The route keeps the state in the magnetic group, which PARSEC 1.5 does not impose. `auto` leaves it where it can tell: where the up channel of a loop ends with a fractional Fermi-Dirac occupation (`min(f, 1 - f)` from 0.05, `SCF/spin_polarized.SPIN_FLIP_OCCUPATION_TOLERANCE`, the test 3 of the spin sectors for the pair of channels the operation exchanges; `occup.in` is not tested), and where a test of the spin sectors fails in a flip loop (the equal levels of the two channels are those of the operation there: test 1 reads them as a partly filled multiplet of the start Hamiltonian that `off` does not see). A `NOTE` says why, the rest of the run solves both channels (later loops, ionic steps and the systems that the spin guard prepares again; the detail `spin_flip_symmetry` says so) and the loop runs again from its start in the sectors it started in, after an ionic move with fresh subspaces: the route of `off` (since 2026-11-21). A100, first runs, PBE, 80 K, against `off`: O2 at 2.28 bohr from `+0.5 / -0.5` in the weighted sectors leaves the route at the start Hamiltonian and gives the triplet of `off` bit for bit (-63.73333158 Ry, 24 iterations, SCF 9.4 -> 9.8 s; where only the occupation test left the route, the guard took the run on to the abelian sectors and to the full grid, which took the flip again: 245 iterations, unconverged, 2.6e-2 Ry higher); in the full-SCF runner on 4 ranks the same input runs the loops of `off` after its flip loop (27 + 25 + 150 iterations against 25 + 150, the same energy; the runner's `off` ends unconverged on the full grid there); H2 at 20000 K (up occupations 0.75 and 0.22) gives the energy and forces of `off` (total 14.1 -> 15.4 s; 4 ranks of the runner the same energy). `off` by default: a flip loop can also settle with integer occupations in another state than two solved channels reach, which no test sees: Ni2 at 4.06 bohr from `+0.6 / -0.6`, -138.78451344 against -138.84084192 Ry (0.77 eV higher, absolute moment 2.12 against 2.31, no `NOTE`; the same at 300 K; `off` steady to 1e-8 Ry but not converged to 1e-6 in 150 iterations). Where it agrees it is faster (one process on 4 A100, first runs, SCF (total) seconds, the same energies to the printed 1e-8 Ry): H2 9.4 -> 8.6 (13.3 -> 12.2), Cr2 10.7 -> 9.0 (14.7 -> 12.9), Cr2 with forces 10.7 -> 9.3 (forces within 1e-6 Ry/bohr), Cr4 14.4 -> 11.9 (18.5 -> 16.1), Cr2 in the cube at Gamma 8.5 -> 7.2 (14.4 -> 13.0), MnO 64 atoms 14.7 -> 9.8 (21.0 -> 16.0), Fe2 at 80 K 10.9 -> 9.6, Co2 11.9 -> 9.5; Cr4 on one A100 9.9 -> 8.1; the runner with 4 ranks Cr4 7.7 -> 5.1 s of process wall. A100, PBE, 1e-7 Ry, against `off` on the same route: H2 at 4 bohr (sectors) SCF 2.6 -> 1.5 s, dE 1e-13; Cr2 at 3.6 bohr sectors 4.4 -> 2.5 s, full grid 4.0 -> 2.6 s, dE 4e-12, forces 7e-7; Cr4 square sectors 6.1 -> 3.8 s, full grid 4.9 -> 4.2 s, dE 3e-11, forces 1.4e-7; Cr2 in a 14.4 bohr cube at Gamma sectors 5.0 -> 2.7 s, full grid (the Fortran-derived routines that `Spin_Symmetry_Sectors: off` selected until 2026-11-21, `PARSEC_SYMMETRY_KEYWORD_ROUTINES`) 2.7 -> 1.4 s, dE 6e-14; MnO AFM-II 64 atoms at Gamma (the operation a translation by a) sectors 15.0 -> 8.4 s, full grid 8.8 -> 5.1 s and 3.2 -> 2.5 GB on the device, dE 5e-13 and 8e-12; BFGS 4 steps Cr2 sectors 18.2 -> 10.2 s, Cr4 full grid 18.5 -> 14.4 s, energies within 1e-8 and 3e-6 (the criterion), forces 7e-7; the runner with 4 ranks Cr4 eigensolve 1.05 -> 0.63 s, dE 3e-11; H2 with a point charge off the axis, with `Polarizability` and as a `Restart_Run` of a ferromagnetic state: the energies, dipoles and `polar.dat` of `off` to the printed digits (before these checks the down channel was the image there: 0.25 Ry too low, the dipoles zero, a different spin state); the detection 0.003-0.033 s of the preparation on these cases, and with the label k-d trees built once and the candidates filtered by 32 probe atoms at most 0.28 s for 4,096 atoms of random labels without an operation (20.6 s for 1,000 before the filter; A100 node CPU) | `off`: both channels solved, as before (an unknown value is refused at preparation) |
| `PARSEC_SPIN_FLIP_STABILITY` | `off` (since 2026-11-22, `auto` on request; read and checked in every preparation, used where `PARSEC_SPIN_FLIP_SYMMETRY=auto` takes the route): `auto` tests the converged state of every flip loop against perturbations that break the operation (`SCF/spin_polarized._flip_stability`): the down channel, which the route does not solve otherwise, is solved at the image of the converged up potential (its levels are those of the up channel) and at five probes `g u + dd` (largest change 5e-3 Ry), the up channel's response following from the operation, and the Ritz values of five Arnoldi steps on the linear map `dd - g du -> dD - g dU` of the SCF give its largest eigenvalue; from 1 the state is unstable (a direction of negative curvature of the energy: the second variation is `<d rho, ((-chi_0)^-1 + K) d rho>`), and the loop runs again from its start with both channels (a `NOTE`, the route of `off`); below 1 a `NOTE` gives the eigenvalue and the result is that of the untested route bit for bit (the up channel and its orbitals are not touched). A100, first runs, PBE, against `off` and the untested route (SCF s off / untested / tested, the largest eigenvalue): Ni2 21.3 (unconverged, -138.84084192 Ry) / 9.6 / 11.0 (-138.78451344 both), 0.612; Cr4 14.2 / 11.9 / 14.6, 0.357; Fe2 11.0 / 9.6 / 10.9, 0.482; Co2 11.7 / 9.5 / 10.9, 0.441; Cr2 10.7 / 8.9 / 10.3, 0.951; MnO AFM-II 64 atoms 12.7 / 7.5 / 11.0, 0.309; Ni2 at 300 K 20.6 (unconverged, -138.84084170) / 9.6 / 11.0 (-138.78431967 both), 0.859; the other energies as `off`; O2 broken symmetry leaves the route at the start Hamiltonian before (the triplet of `off` bit for bit). `off` by default: no state tested unstable, the test costs 1.3-3.6 s here (its first solve of the down channel about half), about what the route saves, and it does not see the Ni2 miss of the route (a stable state, 0.612: `off` reaches its own state along its own path; a ferromagnetic state of moment 4 at -138.85078776 Ry lies below both), so it does not make `PARSEC_SPIN_FLIP_SYMMETRY=auto` safe to be the default. CPU (scipy, stretched H2): 0.139 at probes of 5e-3 and 1e-3 Ry alike (the linear regime). | `off`: the state is not tested, as before |
| `PARSEC_GAMMA_TS_SECTORS` | `auto` (the default since 2026-11-21; `off` from 2026-11-19, when `auto` and `on` were opt-in because their reading of `Ignore_Symmetry` below departs from the keyword of PARSEC 1.5: since 2026-11-21 that keyword of a TS input is read as PARSEC 1.5's requirement of the correction, error 159, which every TS input must meet, not as a request for the full grid; validated on A100 with empty kernel caches, the switch `off` against the default: Si216 72^3 x 502 on 4 A100 SCF 38.2 -> 26.3 s, total 42.0 -> 30.0 s, fullest GPU 8.1 -> 0.9 GiB, host 4.7 -> 1.6 GiB; Ag108 fcc 64^3 x 778 (2.0e8) on 4 A100 SCF 61.1 -> 45.4 s, total 64.2 -> 49.3 s, and on one A100 61.2 -> 41.7 s, 63.8 -> 45.0 s, 8.6 -> 0.9 GiB; the same energies to the printed 1e-8 Ry; Si64 48^3 x 153, below the floor, the full grid with the same iterations and energy). `auto`: an unpolarized bulk orthorhombic cell at the Gamma point with the Tkatchenko-Scheffler correction (`Van_Der_Waals`), in one process on CuPy, takes the sectors of its point group (the weighted sectors of plain PBE; `SCF/gamma_features.py`) where its basis (grid points x states) reaches 2^27 entries; the correction is formed after the loop from the density on the full grid, as before. PARSEC 1.5 demands `Ignore_Symmetry` of every TS input (error 159); under `auto` or `on` the command line reads it as that requirement where the sectors take the cell and prints a NOTE (`--symmetry off`, `PARSEC_PERIODIC_SYMMETRY=off` or a backend other than CuPy keep the full grid), a departure from the meaning of the keyword in PARSEC 1.5. First runs, Si216 PBE + TS (72^3 x 502): 4 A100 SCF 30.9 -> 18.9 s, total 34.0 -> 25.5 s; one A100 SCF 31.8 -> 18.0 s, total 33.9 -> 23.8 s; the same energy to 1e-8 Ry. A spin cell keeps the 1x1x1 k-point driver under `auto`: in the spin sectors Si216 spin PBE + TS was slower (4 A100, SCF 27.3 -> 39.8 s, process wall 46.9 -> 50.5 s; again 46.1 -> 51.3 s); `on` takes it (spin O2 and Si8: the energies of the full grid and of the 1x1x1 driver). TS inputs run on the Fortran-derived routines, whose forces and relaxations keep the full grid (`PARSEC_PERIODIC_SECTOR_FORCES`). `on`: every size, spin cells too. A hybrid, the MPI runner and the supercell of a k-point mesh keep refusing it | `off` (the default before 2026-11-21): the full grid of one process (a spin cell: the 1x1x1 k-point driver), `Ignore_Symmetry` as in PARSEC 1.5, as before |
| `PARSEC_GAMMA_LDAU_SECTORS` | `auto` (since 2026-11-21; `off` from 2026-11-19 to 21, when every sector carried the columns of every atom): an unpolarized bulk orthorhombic cell at the Gamma point with LDA+U (`ldaplusu_u`, `J = 0`), in one process, from 2^26 grid points x states on (`gamma_features.MINIMUM_ENTRIES`; no count of devices since 2026-11-21, four before): the sectors of its point group with its translations, the weighted sectors of a subgroup (`PARSEC_LDAU_WEIGHTED_SECTORS`) or the abelian ones, the KB and the fifteen LDA+U columns of one atom per orbit (`PARSEC_LDAU_PROJECTOR_ORBITS`; `Hamiltonian/hubbard_device.py`), the occupation matrices summed over the sector bases. A100, first runs, the defaults (weighted sectors, one atom per orbit), SCF (total) s, full grid -> sectors on one / four GPUs: fcc Ag108 48^3 x 778 (8.6e7 entries) 11.6 (14.1) -> 6.3 (9.4) / 5.6 (10.1); Ag256 64^3 x 1836 (4.8e8) 61.9 (65.2) -> 9.9 (18.5) / 11.6 (21.5), GPU peak 19.9 -> 0.6 GiB; SrTiO3 (U on Ti) 135 atoms 60^3 x 380 (8.2e7) 8.8 (11.6) -> 4.5 (7.0) / 5.3 (9.0); 320 atoms 84^3 x 891 (5.3e8) 48.2 (52.3) -> 8.7 (18.1) / 9.9 (20.7), 21.3 -> 0.7 GiB; the energies to 1e-8 Ry, the occupation matrices within 1.5e-7 (2.2e-9 for Ag). Below the floor the setup of the sectors costs more than the loop saves: Ag32 46^3 x 234 (2.3e7) total 5.8 -> 11.6 / 13.6, SrTiO3 40 atoms 54^3 x 118 (1.9e7) 6.7 -> 13.3 / 15.3. With every atom's columns (the route of 2026-11-19, `PARSEC_LDAU_PROJECTOR_ORBITS=off`, abelian sectors) Ag256 was 106.5 -> 153.3 s on one A100 and 106.6 -> 95.8 s on four. A spin cell takes the spin sectors only under the opt-in `PARSEC_LDAU_SPIN_SECTORS` (default `off`: the 1x1x1 k-point driver); the forces and relaxations of the cell keep the sectors under `PARSEC_LDAU_SECTOR_FORCES` (default `auto`). `on`: every size. Hybrids, the MPI runner and the supercell of a k-point mesh keep refusing it | `off`: the full grid of one process, as before |
| `PARSEC_LDAU_PROJECTOR_ORBITS` | `auto` (since 2026-11-21): the sectors of an LDA+U run (clusters and Gamma cells) whose reduction knows the orbits of its atoms carry the KB and the fifteen LDA+U columns of one atom per orbit, scaled by the square root of the orbit size (`compress_projectors_for_sectors`, as without LDA+U), with the coefficients of the representatives (`hubbard_device.HubbardOrbits`, `DeviceHubbard.coefficient_layouts`); the occupation matrices are formed for the representatives and turned onto the other atoms with the d harmonics of an operation of the sectors, `N_b = D(R) N_a D(R)^T` (`d_harmonic_rotation`; the density matrix of states of real one-dimensional characters commutes with every operation, so the potential matrices follow alike and every atom of an orbit gives a sector the same block). The forces read every atom, as before. A100, abelian sectors, first runs, SCF (total) s with every atom's columns -> one atom per orbit, on one / two / four GPUs: Ag256 fcc 102.7 (153.3) -> 28.2 (75.1) / 60.6 (112.7) -> 34.0 (80.8) / 43.6 (95.8) -> 36.2 (84.2), GPU peak on one 3.4 -> 0.9 GiB; Ag108 20.8 (31.7) -> 11.7 (21.6) / 15.3 (26.4) -> 13.6 (23.9) / 16.0 (28.0) -> 13.9 (25.0); an Oh Ag43 cluster 7.6 (9.6) -> 7.6 (9.5) / 9.9 (13.1) -> 9.3 (12.2) on four; SrTiO3 135 atoms 8.4 (16.1) -> 8.5 (16.1) on four; the energies to 1e-8 Ry and the occupation matrices to 1e-12 of each other (the full grid within 2e-9, SrTiO3 1.2e-7). The images of the atoms are matched within `PARSEC_SYMMETRY_ATOM_TOLERANCE` (1e-4 bohr at least; a fixed 1e-4 stopped every LDA+U run of a structure accepted within a looser tolerance at its preparation until the evening of 2026-11-21), and a structure that keeps the group only to more than 1e-6 bohr (accepted within such a tolerance) keeps the columns of every atom (`hubbard_device.ldau_orbits_exact`, a NOTE in the detection line): the columns of one representative give its whole orbit the offset of that atom. fcc Ag108 with one atom 5e-4 bohr off, `PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-3`, A100: the energy of the full grid of that structure to 1.3e-3 Ry with the orbits (without LDA+U the KB orbits alike: 1.2e-3 Ry) and to 7.7e-5 Ry with every atom (the abelian route of before alike), total 9.3 against 10.7 s; Ag13 6.1e-4 against 7.1e-5 Ry | `off`: the columns of every atom in every sector (`driver._without_atom_orbits`), bit for bit as before in the abelian sectors; alone it keeps the weighted sectors of `PARSEC_LDAU_WEIGHTED_SECTORS=auto` (now with every atom's columns): the former bits of an LDA+U run take `PARSEC_LDAU_WEIGHTED_SECTORS=off` as well, for a cell `PARSEC_GAMMA_LDAU_SECTORS=off` (its default before 2026-11-21) and for a spin run `PARSEC_LDAU_SPIN_SECTORS=off` |
| `PARSEC_LDAU_WEIGHTED_SECTORS` | `auto` (since 2026-11-21): an LDA+U run (a cluster or a Gamma cell) takes the weighted one-dimensional sectors of a subgroup of its point group where the plan of the group does (`driver._orbital_group_refusal`, `PARSEC_ORBITAL_GROUP`): the occupation matrices of the weighted states are averaged over the full group, `N_b = (1/|G|) sum_g D(R_g) N_(g^-1 b) D(R_g)^T`, which gives those of every partner (the density of the weighted states is averaged so too), the potential matrices are then covariant and the term commutes with the group; the LDA+U forces are averaged alike (`hubbard_device.HubbardGroupAverage`). A100, first runs, SCF (total) s, abelian sectors -> weighted (both with one atom per orbit) on one / four GPUs: Ag256 28.2 (75.1) -> 9.9 (18.5) / 36.2 (84.2) -> 11.6 (21.5) (384 operations; the abelian group has 64); SrTiO3 320 atoms 25.8 (31.4) -> 8.7 (18.1) / 31.7 (38.3) -> 9.9 (20.7); SrTiO3 135 7.0 (9.6) -> 4.5 (7.0) / 8.4 (12.3) -> 5.3 (9.0); the Ag43 cluster (Oh, the default cluster route of before: abelian sectors with every atom) 7.5 (9.2) -> 3.7 (5.7) / 9.8 (12.9) -> 4.3 (7.6), Ag13 2.5 (3.7) -> 1.9 (3.2) / 3.1 (5.7) -> 2.3 (5.0); spin MnO AFM-I 216 atoms (`PARSEC_LDAU_SPIN_SECTORS=auto`) 40.1 (43.3) -> 35.7 (38.8) on four; the energies to 1e-8 Ry, the occupation matrices of the full grid within 1.5e-7 (Ag 2.2e-9) | `off`: the abelian group for LDA+U, as before |
| `PARSEC_LDAU_SECTOR_FORCES` | `auto` (since 2026-11-21): the forces and the `bfgs`/`simple` relaxations of a bulk cell at the Gamma point with LDA+U (an input of the Fortran-derived routines), whose SCF takes the sectors, keep them where `PARSEC_FORTRAN_SECTOR_FORCES=auto` takes the forces of those routines (`cli.periodic_sector_forces_refusal`): the KB term of one atom per orbit completed over the group, the LDA+U term of every atom from the sector bases with the gradients of its projectors (`DeviceHubbard.forces`, averaged over the full group in weighted sectors), the total and the moves averaged over the group. A100, first runs, against the full grid: fcc Ag108 with one atom moved 0.12 bohr (the weighted sectors of a group of 8) the forces to the printed 1e-6 Ry/bohr (max 0.645), total 13.9 -> 8.3 s; two `bfgs` steps of it the same energies and positions within 2e-9, 42.7 -> 24.3 s; Ag32 two `bfgs` steps (under `PARSEC_GAMMA_LDAU_SECTORS=on`, below the floor) the same, 6.8 -> 15.6 s; SrTiO3 40 atoms with a moved Ti (abelian, `on`) the forces of the full grid, 3.1 -> 4.5 s | `off`: the forces of these cells on the full grid, as before |
| `PARSEC_LDAU_SPIN_SECTORS` | `off` (the default again since the evening of 2026-11-21, after a day at `auto`): a spin-polarized LDA+U run keeps the routes of before, a bulk cell at the Gamma point on the 1x1x1 k-point driver of the reference CLI, a cluster on the full grid of the driver. Opt-in: `auto` solves both channels of a bulk cell at the Gamma point that the LDA+U switch takes (`PARSEC_GAMMA_LDAU_SECTORS`: from 2^26 grid points x states on) in the spin sectors (`SCF/gamma_spin.py`, the weighted sectors with `PARSEC_LDAU_WEIGHTED_SECTORS`), each channel writing its own potential matrices into the shared sector operators before its solve, with the guard of `SCF/spin_sectors.py` and its fallbacks (the abelian sectors, then the full grid of the driver); `on` also a cluster (`driver._spin_sector_refusal`). **Opt-in because the sectors can end, without a note, in another magnetic state than the k-point driver.** A100, at most 200 iterations per loop: bcc Cr 16 atoms, AFM, U = 2 eV, on 100^3 points (7.0e7 entries, above the floor), seeds +/-0.02: `auto` exits converged in one loop at the non-magnetic -303.52817316 Ry (integrated `|m|` = 0), where the k-point driver and the full grid stay magnetic near -306.549 Ry (`|m|` 69.4, neither converged); the same cell on 36^3 points with the sectors forced (`PARSEC_GAMMA_LDAU_SECTORS=on`), seeds +/-0.02, 0.05 and 0.08: the non-magnetic -303.63816245 Ry against the magnetic -306.75421598 (0.02, 0.05) and -306.38211140 (0.08) of the k-point driver. Over 30 inputs (MnO 8, 64 and 216 atoms in several AFM patterns, seeds and charges; bcc Cr 16, 54 and 128 atoms at seeds 0.02 to 0.3; fcc Ag 32 atoms) the sectors gave the final state of the k-point driver on 15 (within 1e-8 Ry); another state, both converged, on 9: the four non-magnetic exits above, and five runs that the fractional-occupation check sends to the full grid, whose state differs from that of the k-point driver (Cr 16 atoms at seeds +/-0.1 to 0.3: -306.38211140 against -306.35013065; MnO AFM-I 8 atoms with `Net_Charges` +1 and +0.5: -242.47139075 against -242.46453788 and -242.17961181 against -242.17804755); the k-point driver did not converge on 3 and the sectors on 3. None of the checks of the guard (the symmetry of each channel density, fractional occupations) fired on the four non-magnetic exits; a test of a vanishing `|m|` would also stop real non-magnetic metals (fcc Ag 32 atoms ends at `|m|` = 0 on every route). The sectors did better where the k-point driver does not converge: MnO AFM-I 64 atoms at seeds +/-0.05 (sectors -1936.21258684 Ry, the k-point driver not converged at -1935.72584565) and 216 atoms (sectors -6534.36212533, the k-point driver and the full grid diverge). They are faster above the floor: MnO AFM-I 216 atoms with `Net_Charges` +1 (78^3 x 885) 29.5 s against 336 s on the k-point driver (process wall, one node of four A100, fresh processes without a symmetry cache, the CuPy kernel cache warm from earlier runs on the node; the same energy, -6535.01714629 Ry); bcc Cr 128 atoms 306 to 470 s (the checks send it to the full grid) against 480 to 592 s. Earlier measurements: MnO AFM-I 216 atoms (76^3 x 815) the energy of the full grid of the driver to 1e-8 Ry, total 64.7 -> 38.8 s on four A100, GPU peak 23.4 -> 1.1 GiB; magnetic Mn2 dimers with U = 5 eV give the energy of the host LDA+U term (`--device cpu`) under `on`; fcc Ag108 spin pays the loops the guard discards (88.8 s on the k-point driver -> 90.7 s), and the spin sectors of a cluster were slower (Ag13 5.4 -> 13.6 s). The k-point driver itself: it did not converge for antiferromagnetic oxides while it started from one polarization fraction of the whole cell, zero for labels at `+p` and `-p` (MnO AFM-II 64 atoms, U = 5 eV: -1581.76 against -1936.26 Ry after 200 iterations; a non-magnetic start of this cell does not converge on the full grid of the driver either); with the start of `initchrg.f90` (`PARSEC_SPIN_START=auto`) it gives the -1936.25763058 Ry of that full grid in 32 iterations, and with LDA+U it folds correctly (the 2x2x2 mesh of the 8-atom MnO AFM-I cube with `PARSEC_KPOINT_KINETIC=bloch`, wedge and full mesh, gives 1/8 of the Gamma energy of the 64-atom cell to 1e-8 Ry, -242.03174572 Ry, on one grid of h = 0.35 bohr) | `auto`: the cells above in the spin sectors (the default during 2026-11-21); `on`: clusters too |
| `PARSEC_LDAU_SPIN_FULL_GRID` | `off` (the default): a spin-polarized bulk cell at the Gamma point with LDA+U below the floor of `PARSEC_GAMMA_LDAU_SECTORS` (2^26 grid points x states) runs on the 1x1x1 k-point driver of the reference CLI. Opt-in, for a single point (`J = 0`, a native LDA or GGA) whose full grid one device holds: `auto` solves both channels on the full grid of that device of the accelerated driver where the species start with `Initial_Spin_Polarization` values of both signs (both drivers then start as `initchrg.f90`, `PARSEC_SPIN_START`); `on` also takes the inputs of one sign, which the k-point driver starts from one polarization fraction of the cell (`gamma_features.ldau_spin_full_grid`). It takes such cells under `PARSEC_LDAU_SPIN_SECTORS=off` (the default) as under `auto`; forces, moves, `--symmetry on`, `--mpi` and `PARSEC_GAMMA_LDAU_SECTORS` other than `auto` keep the k-point driver, and cells above the floor keep their route (the k-point driver under the default `PARSEC_LDAU_SPIN_SECTORS=off`). **The route is faster but can converge to another magnetic state**: the two drivers follow different paths from the first iteration on, also from the same start, and nothing in the input tells which cells are affected. A100, at most 200 iterations, k-point driver -> `auto`: bcc Cr 16 atoms, AFM, U = 2 eV, seeds +/-0.1, 0.2 and 0.3: -306.35013065 Ry at moment 0 -> -306.38211140 at 5.41; seeds +/-0.05: -306.75421598 -> not converged, -306.47898289 at 17.94; MnO AFM-I 8 atoms with `Net_Charges` +1: -242.46453788 at 0 -> -242.47139075 at 1.00, +0.5: -242.17804755 at 0 -> -242.17961181 at 0.48, and +1 with one atom moved 0.03 bohr: -242.47572890 at -1.00 -> -242.47569077 at 1.00; MnO 64 atoms with seeds +/-0.05: AFM-I not converged -> -1936.21258684, AFM-II -1936.25414832 -> not converged. The energy was the same to 1e-8 Ry on the other converged inputs measured: MnO AFM-I 8 atoms (also PBE, U = 2 eV, 3000 K, one atom moved, `Net_Charges` -1); MnO 64 atoms in AFM-I, AFM-II, striped and random patterns (also seeds +/-0.1, unequal seeds, and `Net_Charges` +1 on AFM-I and AFM-II); CrN AFM-II 64 atoms; bcc Cr 2 and 54 atoms and 16 atoms at seeds +/-0.5; fcc Ag 32 atoms. FeO, CoO, NiO, bcc Fe and fcc Ni inputs did not converge within 200 iterations on either driver and show nothing either way. One node of four A100, empty CuPy and CUDA kernel caches, process wall of two runs each, k-point driver -> `auto`: MnO AFM-I 8 atoms 17.05/16.68 -> 9.20/9.42 s; MnO AFM-II 64 atoms 47.66/47.78 -> 19.42/19.28 s; bcc Cr 2 atoms 17.79/17.73 -> 9.32/9.22 s. The route also changes the output: `parsec.dat` holds the wavefunctions (MnO 64 atoms 8.5 -> 427 MB; since 2026-11-22 the k-point driver's file holds them too, made real, `PARSEC_KPOINT_WAVEFUNCTIONS` and `PARSEC_GAMMA_PARSEC_DAT`), the result archive grows (3.1 -> 24.1 MB) and its keys change (`density_up_e_per_bohr3`, `eigenvalues_up_ry`, `occupations_up` and their `down` twins, `magnetic_moment`, `k_weights` and `scf_potential_*` are replaced by `spin_*` and `net_magnetic_moment`) | `off`: the 1x1x1 k-point driver, as before |
| `PARSEC_LDAU_SMOOTH_PROJECTORS` | `off` (the default): the LDA+U projectors of `upot.F90`, `R_d` interpolated linearly on the logarithmic grid and cut at `r_c`, the radius that holds 95 % of the norm of `u_d`, where `R_d` is still about 10 % of its peak (`Hamiltonian/hubbard.py` follows those loops, `tests/unit/test_hubbard.py`). The LDA+U energy is then not a smooth function of the positions, in PARSEC 1.5 as here: it jumps when a grid point crosses the sphere of `r_c` around an atom, and its derivative jumps when the distance of a support point crosses a sample of the table. The forces are the derivative between those events, so they differ from the slope of the energy over finite steps. A100, fcc Ag4 with U = 4 eV, atom 0 at 8.7e-2 bohr from a grid point, fresh SCF runs at shifts along x: the support of the atom loses a point between +1.82e-3 and +1.83e-3 bohr (802 -> 801 entries), the LDA+U energy drops by 4.45e-4 Ry and the free energy rises by 1.96e-4 Ry, at the same 28 iterations and a continuous force; between such events the chords over 1e-3 bohr differ from the mean force by 1e-4 to 4e-4 Ry/bohr (the samples of the table). Opt-in `on` (`1`, `true`, `yes`; or a width `f` in (0, 1], `on` = 0.2; `1` is the boolean spelling of `on`, the full width `f` = 1 is written `1.0`, and any number equal to 0 is `off`; parsec.out names the switch and `f` after the LDA+U radii, which are then those of the support): `R_d` is the cubic spline of the table (zero slope at the origin, natural at its end) times the taper `1 - t^3 (10 - 15 t + 6 t^2)`, `t = (r - r_c)/(f r_c) + 1/2`, from 1 at `(1 - f/2) r_c` to 0 at `(1 + f/2) r_c` (the support radius, `hubbard_support_radius`) and antisymmetric about `r_c` (`S(r_c + x) + S(r_c - x) = 1`), so the overlaps of the projector with smooth functions, and so the occupations, are those of the cut to second order in `f`, while its norm, quadratic in the taper, changes at first order (Ag: 0.9501 for the cut, 0.9446 at `f` = 0.2, about -0.026 `f` from `f` = 0.4 down to 0.05; `ldau_smooth_projectors_fraction`); the gradients of `Forces/hubbard_force.py` are their exact derivative (no 3e-3 bohr floor). Full SCF to 1e-10 Ry, force against the slope of `E - TS` at steps of 1e-3 / 2e-3 bohr (Richardson value), Ry/bohr, `off` -> `on` (the command lines run every LDA+U input on the Fortran-derived routines, `Use_Fortran_Routines: .false.` included, with the same bits): Ag4 atom 0 3.96e-4 / 5.01e-2 -> 7.7e-6 / 3.0e-5 (3.8e-7, the `h^2` term of the difference; the curvature of this cell is the same without U), atom 1 on a grid point 4.7e-8 / 4.6e-8 -> 5.0e-8 / 2.9e-8; the same cell with atom 1 at 5e-4 bohr from its point: atom 1 5.07e-5 / 2.23e-5 -> 2.6e-7 / 1.0e-9, atom 0 3.96e-4 / 5.01e-2 -> 7.7e-6 / 3.0e-5 (3.7e-7); `Relaxation.periodic_backend` (the readable backend of the default routines, whose forces are `Forces.compute_periodic_forces`; reached from the API only, here on the system the Fortran-derived routines prepare for the first cell: the preparation of the default routines, which builds no LDA+U projectors, refuses LDA+U), atom 0 3.96e-4 / 5.01e-2 -> 7.65e-6 / 2.95e-5, atom 1 3.5e-8 / 5.2e-8 -> 5.9e-8 / 5.5e-8; an Ag3 cluster, atom 1 at 0.16 bohr from a point 8.1e-2 / 3.9e-2 -> 4.7e-6 / 3.1e-5 (4.1e-6, as the same cluster without U), atom 0 on its point 1.1e-7 / 1.3e-7 -> 7.9e-8 / 5.2e-8; MnO AFM-I (8 atoms, spin, U = 5 eV on Mn, the Fortran-derived routines of the reference solver) Mn 0 at 8.7e-2 bohr 4.3e-2 / 4.6e-2 -> 6.9e-6 / 2.9e-5 (4.6e-7), Mn 1 on its point 3.0e-8 / 5.3e-8 -> 2.6e-10 / 1.4e-9; fcc Cu4 (U = 4 eV, h = a/20) atom 1 at 5e-4 bohr from its point 5.8e-4 / 1.5e-3 -> 6.3e-6 / 1.9e-5 (2.2e-6), on its point 1.8e-7 / 2.9e-7 -> 1.9e-8 / 7.1e-9, atom 0 at 8.7e-2 bohr 1.7e-4 / 4.9e-4 -> 8.4e-6 / 3.5e-5 (4.1e-7, the `h^2` term); with the tabulated projector tail of this Cu table, which is not zero at `rcore` (`PARSEC_PROJECTOR_TAIL=off`), atom 0 keeps 1.45e-5 / 2.34e-5 under `on` (1.5e-4 / 4.8e-4 under `off`), the same to three digits for `f` = 0.4 and 1.0, a residual of its Kleinman-Bylander projectors. The energies are those of another projector: `E - TS` (LDA+U energy) Ag4 -291.81351531 (4.82192074) -> -291.81060045 (4.81492544) Ry, MnO -242.01876193 (3.10080046) -> -242.01180133 (3.08474278), the Ag3 cluster -218.54445638 (3.63083116) -> -218.53467474 (3.60861450); forces by up to 2.3e-3 Ry/bohr (Ag4), 3.2e-3 (Ag3), 1.6e-3 (MnO) and 1.2e-3 (Ag108); the same iterations. The sectors keep the term exact: fcc Ag108 (weighted sectors) against its full grid within 8.9e-11 Ry and 1.8e-9 Ry/bohr, the matrix case `s_ag4u_c3v_gamma` 0 Ry and 5.6e-11 Ry/bohr. The support grows by `(1 + f/2)^3` (799 -> 1045 entries for Ag); fresh processes without a symmetry cache on one A100, fcc Ag108 with forces: total 9.03 / 8.71 -> 9.34 / 9.35 s (SCF 6.09 / 5.79 -> 6.01 / 6.03 s, forces 1.52 / 1.51 -> 1.66 / 1.65 s). Off by default under the rule: the energies and forces move away from those of PARSEC 1.5, not towards an exact reference. The MPI full-SCF runner and `--mpi` refuse LDA+U, so the switch does not reach them | `off`: the projectors of `upot.F90`, bit for bit as before |
| `PARSEC_GAMMA_META_GGA_SECTORS` | `auto` (since 2026-11-19): a bulk orthorhombic cell at the Gamma point with a meta-GGA (native SCAN; a libxc `mgga_` functional also needs `PARSEC_GAMMA_LIBXC_SECTORS`), unpolarized or collinear spin, in one process, from 2^27 grid points x states on: the abelian sectors with the translations, `tau` and `-div(vtau grad)` as derivatives between the sectors (`Hamiltonian/meta_gga_device.py`), the PBE start of the loop on the orbit fields (`AcceleratedPreparedSinglePointSystem.evaluate_meta_gga_start_xc`). First runs, Si216 SCAN (72^3 x 502, 1.9e8 entries), 4 A100: SCF 254.3 -> 19.3 s, total 257.7 -> 28.8 s; at 70^3 (1.7e8) on one A100: SCF 250.7 -> 16.7 s, total 253.7 -> 21.7 s, GPU peak 14.3 -> 1.3 GiB; spin Si216 SCAN against the 1x1x1 k-point driver, 4 A100: SCF 516.4 -> 53.5 s, process wall 521.1 -> 65.4 s; below the floor Si64 (1.7e7, 4 A100) total 16.6 -> 25.8 s, but Si216 at 64^3 (1.3e8, one A100) 176.1 -> 29.2 s; energies and forces those of the full grid (Si8 moved atom, forces to 1e-6 Ry/bohr), spin O2 the energy of the full grid. `on`: every size. Hybrids, the MPI runner and the supercell of a mesh keep refusing it | `off`: the full grid of one process (a spin cell: the 1x1x1 k-point driver), as before |
| `PARSEC_GAMMA_LIBXC_SECTORS` | `auto` (since 2026-11-19): a bulk orthorhombic cell at the Gamma point with a libxc LDA or GGA (`LibXC_Functional`), unpolarized or collinear spin, in one process, from 2^28 grid points x states on: the abelian sectors with the translations, the device evaluators of `V_xc/device_libxc.py` on the orbit fields. First runs, 4 A100 unless noted: Si512 PBE of libxc (96^3 x 1183, 1.0e9 entries): SCF 96.1 -> 30.6 s, total 99.9 -> 41.2 s, GPU peak 41.8 GiB on one device -> 1.3 GiB on the largest of four (4.4 GiB summed; nvidia-smi), host 17.8 -> 1.7 GiB; Si216 at 84^3 x 502 (3.0e8): SCF 26.1 -> 14.1 s, total 29.1 -> 22.2 s, on one A100 total 29.0 -> 18.7 s, spin against the 1x1x1 k-point driver process wall 54.5 -> 39.9 s; below the floor Si216 at 72^3 (1.9e8) SCF 16.2 -> 12.8 s but total 19.0 -> 20.9 s (17.8 -> 20.6 s again; the setup of the sectors), Si64 2.2 -> 7.9 s, while Si216 at 80^3 (2.6e8) on one A100 was faster (total 25.3 -> 15.7 s); energies and forces those of the full grid, spin O2 that of the full grid and of the 1x1x1 driver. `on`: every size. Hybrids, the MPI runner and the supercell of a mesh keep refusing it | `off`: the full grid of one process (a spin cell: the 1x1x1 k-point driver), as before |
| `PARSEC_SPIN_GUARD_CELL_GROUP` | `auto` (since 2026-11-19): the density test (test 2) of the spin guard of a bulk cell tests generators of the operations `r -> R r + t` of the atoms, about a defect off the origin as well. The Gamma sectors (`SCF/spin_sectors.py`, `atom_group_images`): those of the sector group, of the atoms found from their environments (the atoms of the rarest environment, at most 32, carried onto each other: a defect off the origin of any cell), or, where no environment is rare (a crystal of many cells), of the cell group the detection of the sectors built (about its symmetric site with the half-cell translations; `kpoint_sectors.recent_cell_group_operations`, no second search), and the signed permutations about the origin; those beyond the sector group to 1e-6 bohr at most whatever `PARSEC_SYMMETRY_ATOM_TOLERANCE` (the sectors average the Hamiltonian over their group alone, so an operation that holds the atoms only to a looser tolerance measures the structure: 0.028 electrons for a diamond vacancy 2^3 with its atoms 3e-4 bohr off at 1e-3); rows of the periodic grid, 32-bit; the group followed up to 32,768 operations, an operation about the origin that would take it beyond tested on its own (a perfect crystal about a bond centre keeps all 12 of D3d, its inversion included). The k-point sectors of a spin loop (`PARSEC_KPOINT_SECTOR_GUARD`, `kpoint_sectors.SectorGuardGroup`: the 1x1x1 driver of forces, relaxations, MD, `main.py --mpi` and `PARSEC_GAMMA_SPIN_SECTORS=off`, and k-point meshes): their cell group and the operations found from the environments that the potential is taken to be invariant under (all at a single Gamma point; on a mesh those that map it onto itself, D2d of Td for a k-point along z). Built once per structure in 0.02-0.13 s on the CPU of a node for 63-1,728 atoms, as before. A diamond vacancy at (a/2, a/2, 0), whose Gamma sectors are D2 about it: Td (24 operations) where the test of before had the swap of x and y; the neutral vacancy 2^3 at 10 K with test 1 deferred (as the unconverged start levels of a large cell defer it) keeps a broken t2^2 state with integer occupations under `off` and fires test 2 under `auto` with 0.6991 electrons, as at the origin; V- at and off the origin passes with the same bits as `off` (A100). Under `PARSEC_PERIODIC_DEFECT_CENTRE=off` (before 2026-11-19; `auto` gives the cell group Td about it) the cell group of 3^3 cubes keeps the swap alone about that vacancy (its search of the translations gives up after 32 candidates), and so do its k-point sectors: the neutral vacancy on the 1x1x1 driver (215 atoms, 18 points per cube, 10 K, test 1 deferred) fires test 2 under `auto` at the end of its first loop with 1.448 electrons (the twofold axis about the vacancy; 73.7 s on one A100 node), at the origin with 0.561 electrons (13.6 s); under `off` test 2 measured 0 (the swap, which the sectors hold) and the state left the sectors through test 3 alone (0.110 electrons in a level of the t2 multiplet, 376 s with the loop that followed) | `off`: the signed permutations about the origin alone on the Gamma sectors and the cell group alone on the k-point sectors, as before; clusters are not affected |
| `PARSEC_SPIN_GUARD_DOUBLETS` | `note` (default, since 2026-11-21): a weighted plan can hold an E representation twice in a sector of weight 1 (the two E of the 192 operations of an fcc or rock-salt cell, the E of the point group T in one such sector, the two E of Th in two), so both partners of such a doublet are states of one sector label, which tests 1 and 3 of the spin guard do not count (`SCF/spin_sectors.py`, `PointGroupReduction.sector_repeated_irreps`). A run that ends with such a doublet fractional keeps its result, bit for bit that of `off`, and `parsec.out` says so in a NOTE. The state can be a saddle: fcc Cr4 with `Net_Charges: 3` at 100 K (one up electron in the doublet) ends at -79.70931261 Ry in the weighted and abelian sectors and on the full grid from the same symmetric start, while a full-grid solution reached through a displacement of one atom by 1e-3 bohr and carried back to the exact atoms converges at -79.73171383 Ry, 22.4 mRy lower, with integer occupations and 1.41 electrons of asymmetry (Mn4 with `Net_Charges: 7`: 29 mRy, not converged in 500 iterations). Leaving the sectors does not find that state by itself, so the default does not move the run. Since 2026-11-22 the abelian sectors are read the same way at the end of their SCF (`spin_sectors.fractional_sector_doublets`): levels of one sector label within 1e-4 Ry of each other with a fractional occupation (the e_g of Oh, both partners in A_g of D2h), which test 3 does not count either; `note` keeps the result with a NOTE (no number changes). There the full grid can leave the symmetric state: bcc Mn2 (a 5.40 bohr, PBE, labels Mn_a and Mn_b at +0.5 and -0.5, 300 K) is stopped in the weighted sectors at the start Hamiltonian and ends in the abelian sectors with a half-filled e_g pair in each channel (cubic, moments +-3.25 per site) at -56.87755621 Ry (h 0.11 bohr), while the full grid from the same start splits the pair at -56.95948510 Ry (D2h, integer occupations); bcc Cr2, Fe2, V2 and rock-salt MnO8 and NiO8 with antiparallel labels end in the same state on both routes (A100). The check runs where the spin sectors of the accelerated driver run (clusters, Gamma cells, the full-SCF runner); the k-point sector routes (the 1x1x1 driver of spin forces, relaxations and dynamics, `--mpi`, k-point meshes) do not look for such levels: Mn2 with `PARSEC_GAMMA_SPIN_SECTORS=off` ends in the same symmetric state (-56.87620932 Ry at h 0.22) without a NOTE, and which state a run without symmetry reaches depends on the driver | `stop`: the guard counts such a doublet (degenerate states of the sector within 1e-4 Ry) and the run goes straight to the full grid; of six runs of charged fcc Cr4, Mn4 and Fe4, five keep the energy within 2e-8 Ry at 1.3-2.2 times the time (warm kernel caches), and one (Mn4) ends unconverged at `Max_Iter`, 6e-6 Ry lower; a pair of the abelian sectors sends the run to the full grid as well (bcc Mn2 above: the full-grid energy). `off`: not looked for, as before |
| `PARSEC_SOC_SECTORS` | `auto` (the default since 2026-11-21): `kramers` where the whole-grid spinor basis, `2 N_grid` rows times `States_Num` plus the buffer, reaches `soc_sectors.MINIMUM_ENTRIES` (2^25 = 33,554,432) complex entries, or where `--symmetry on` asks for a reduction, and the whole grid below it with a `NOTE`; `fourfold` from `soc_sectors.FOURFOLD_MINIMUM_ENTRIES` (2^26) on under the opt-in `PARSEC_SOC_AUTO_FOURFOLD=auto` (`soc_sectors.resolve_soc_sectors`, decided from the grid and the input; `orbital_symmetry` names the route of `auto`). One A100, first runs with an empty kernel cache, process wall seconds `off` -> `kramers`: below the floor AuH 10.5 -> 13.2, Au2 19.0 -> 20.5, Au4 19.8 -> 22.0 (whole grid under `auto`); above it Au6 27.1 -> 25.5, Au13 28.6 -> 22.8, Au20 74.6 -> 29.1 (15.3 -> 4.8 GiB), energies within the SCF tolerance (README "Self-consistent spin-orbit coupling of clusters"). `off` (the default from 2026-11-19 to 2026-11-21): the self-consistent spin-orbit loop of clusters (`SO_from_scratch` with `SO_PSP` on CuPy, `SCF/self_consistent_soc.py`) filters `States_Num` plus the buffer complex spinors of `2 N_grid` rows, as PARSEC 1.5 drops to the identity with spin-orbit coupling. `kramers`: the `+i` sector of a two-fold rotation or mirror of the double group, one state of every Kramers pair on `N_grid` rows, real symmetric where a perpendicular operation makes it so (`SCF/soc_sectors.py`, section "Self-consistent spin-orbit coupling of clusters"); `fourfold` (since 2026-11-22): that sector split by the eigenvalues of C4z or S4z where the cluster has one (two real sectors; `kramers` otherwise) at every size, Au20 29.7 -> 21.5 s and 10.5 -> 7.1 GiB at h = 0.3 bohr, Au38 18.7 -> 14.6 s, Ge35 28.6 -> 14.5 s, Ag38 18.9 -> 15.8 s, slower on Au4, Au2, Ge4 and Ge8 and on the Au4 squeezed to D2d at 1.10e8 entries (section above); `on`: that sector split by inversion parity as well, slower than `kramers` on every centrosymmetric cluster measured but Au38 (Au atom 2.02 -> 3.86 s, slower than `off` at 2.25 s; Au6 3.6 -> 4.5 s; Au13 3.4 -> 3.6 s; Au38 18.6 -> 13.4 s); a cluster without such an operation, `--symmetry off` and `Ignore_Symmetry` (unless a mode on the command line overrides it) keep the whole grid with a `NOTE`; `--symmetry on` is accepted with `kramers`, `on` or `fourfold` and refused before the SCF with `off` or for a cluster without such an operation; `parsec.out` names the sector under `orbital_symmetry`; a value it does not know is refused before the SCF. One A100, `kramers` against `off`: Au20 (Td) SCF 57.2 -> 11.3 s, peak 15.3 -> 4.8 GiB (1,270,720 spinor rows: 160.7 -> 29.6 s, 35.5 -> 10.5 GiB), Au38 (Oh) 111.5 -> 18.8 s, Au2 (D2h) 10.9 -> 4.3 s, energies within the SCF tolerance and the same to 1e-8 Ry at `Convergence_Criterion` 1e-6. The first real-arithmetic run on a machine compiles its kernels once (an empty kernel cache: AuH 6.8 -> 9.8 s, Au2 14.7 -> 15.9 s), hence the floor of `auto` | `off`: the whole grid, as before (the default before 2026-11-21) |
| `PARSEC_SOC_AUTO_FOURFOLD` | `off` (the default): `PARSEC_SOC_SECTORS=auto` takes `kramers` at every size above the floor of the sectors. | `auto` (opt-in, since 2026-11-22): `PARSEC_SOC_SECTORS=auto` (and `--symmetry on` under it) takes `fourfold` where the whole-grid spinor basis reaches `soc_sectors.FOURFOLD_MINIMUM_ENTRIES` (2^26 = 67,108,864 complex entries; `kramers` where the cluster has no four-fold operation about z or its sector is complex; `soc_sectors.resolve_soc_sectors`, `soc_auto_fourfold_setting`), mainly a memory saver: the GPU peak 18 to 31 % lower on the clusters measured. Its speed depends on the cluster. One A100, fresh processes, each route with its own empty CuPy kernel cache, process wall seconds `kramers` -> `fourfold` (a warm second run in brackets). Faster: Ag13 at h = 0.30 bohr (8.90e7 entries) 27.3 -> 26.2 (11.8 -> 10.5), Ge29 (Td fragment of diamond, 8.86e7) 28.0 -> 25.3 (15.0 -> 11.9), Au20 (1.30e8) 27.6 -> 24.9 (15.0 -> 12.1), Ge35 (1.38e8) 41.7 -> 27.6 (28.6 -> 14.5), Ag38 (2.06e8) 28.8 -> 25.8 (18.9 -> 15.8), Au6 (Oh, 7.27e7) 28.5 -> 27.4 (12.7 -> 11.7, with 17 SCF iterations against 22: slower per iteration). Slower: the Au4 tetrahedron squeezed to D2d (h = 0.26 bohr) in a sphere of 13.5 bohr (7.27e7) 30.5 -> 30.8 (14.9 -> 15.5) and of 15.5 bohr (1.10e8) 36.6 -> 38.9 (21.0 -> 23.4), and every small cluster below the floor (README "Self-consistent spin-orbit coupling of clusters"); per SCF iteration `fourfold` costs more on clusters with few states (Au2 at 7.2e7 entries 29 %, Au6 12 %), and part of its gains come from SCF loops that took fewer iterations. At `Convergence_Criterion` 1e-7 the energies of `fourfold` equal those of the whole grid within 1e-8 Ry (Ge17, Ge29, Ge35, Au13, Ag13, Au6, Au4 D2d); a value it does not know is refused before the SCF |
| `PARSEC_POLARIZABILITY_SYMMETRY` | `auto` (since 2026-11-19): the seven field loops of `Polarizability` of a cluster, which PARSEC solves without point group, keep sectors where the points of the full grid times the states reach 1.2e7 (2^24 before a re-measurement on 2026-11-19; `on`: whatever the size; `SCF/symmetry_off_inputs.py`): the zero field in the sectors of the group `G` of the atoms, the loops along an axis in those of its stabilizer (the system prepared again with a pseudo-atom on the axis, the first loop from the zero-field potential with a first filter, the loops of one axis warm as in PARSEC), and a loop whose field an operation of `G` carries from a loop solved later not solved (energy of that loop, density its image; `-z`, whose forces and state the run reports, always solved). Td/Oh: zero field and `-z`; D2h: zero field, `-x`, `-y`, `-z`. One A100 node (main.py, four GPUs), first runs, accelerated wall time and peak per GPU: C35H36 (Td, 5.2e7) 14.3 -> 9.2 s; C87H76 (1.6e8) 33.6 -> 12.6 s, 11.6 -> 2.0 GiB; C185H124 (5.3e8) 96.1 -> 20.2 s, 36.6 -> 3.4 GiB; at `Convergence_Criterion` 1e-6 Ry (every run of this row) alpha from the energies equal to the printed digits (54.1704, 107.0820, 218.7514 A^3), from the dipoles within 6e-5 relative (`off` / `auto`: 52.9389 / 52.9360, 110.2704 / 110.2743, 219.4953 / 219.4930 A^3); at the default 2e-4 Ry both routes carry 0.5-1 % of SCF noise in the dipole alpha (C35H36: `off` 52.71, `auto` 52.43, 52.95 at 1e-8 Ry). Below 1.2e7 the warm loops of one full grid are faster (`on` against `off`, first runs: water, 2.1e6, 9.2 against 6.5 s; benzene, 5.4e6, 9.6 against 8.6 s; naphthalene, 1.6e7, 12.6 against 13.6 s, both D2h molecules the means of two runs; they cross near 1.1e7), and `auto` keeps them. Two spin channels: the guard of the spin sectors tests the operations that also keep the field axis, and a loop that leaves the sectors solves on a full grid that carries its field (the loops after it stay there); O2 triplet (A100 test): the seven energies within 2e-7 Ry of `off` | `off`: the seven loops on the full grid, as before |
| `PARSEC_POINT_CHARGE_SYMMETRY` | `auto` (since 2026-11-19): `Add_Point_Charges` of a cluster keeps the sectors of the operations that carry every charge onto a charge of its value (the charges matched as atoms of their own label; none where no operation does). C185H124 with four charges on the threefold axes (Td kept): 50.3 -> 9.5 s, SCF 43.3 -> 4.6 s, 32.5 -> 1.4 GiB per GPU; C35H36 6.6 -> 5.7 s; energies equal to the printed digits. The guard of a spin-polarized run in the sectors tests the same group (O2 triplet with one charge on its axis, C4v: in the sectors, the energy of `off` within 2e-7 Ry, A100 test). A relaxation whose `Move_Flag` fixes an atom that an operation of the atoms exchanges with a movable one, `--symmetry off`, spinors and spin channels on the full grid keep the full grid of `off` (since 2026-11-19: such a relaxation had stopped after its first forces; CH4 with two charges on z and one fixed H, A100 test: the energies and forces of `off`). The MPI runner still refuses point charges | `off`: no point group (PARSEC finds the group of the atoms alone; the port used none), as before |
| `PARSEC_FOLDBLOCK_SYMMETRY` | `auto` (since 2026-11-19, main.py and the MPI runner): `Foldblock_Side_Length 2`, which switches the symmetry off in PARSEC 1.5 and so takes the grid without shift, keeps that grid and the operations that map its domain of whole blocks onto itself (the permutations of the axes the atoms allow: a mirror x <-> y of a Td cluster, two sectors). C185H124: 51.3 -> 26.0 s, 33.6 -> 9.6 GiB per GPU; C35H36 7.0 s either way; energies equal to the printed digits. Self-consistent spin-orbit coupling (the spinor driver takes no sectors; it had stopped with a TypeError before 2026-11-19), a relaxation whose `Move_Flag` the group of the atoms does not respect, `--symmetry off` and spin channels on the full grid keep the full grid of `off` (A100 tests: Au with `SO_from_scratch`, CH4 with one fixed H, the energies of `off`). `parsec.out` keeps PARSEC 1.5's warning and leaves out the port's warning on `Ignore_Symmetry` only where the group is kept | `off`: no point group, as before (with the same warnings) |
| `PARSEC_SPIN_RESTART_SECTORS` | `auto` (since 2026-11-19): a spin-polarized `Restart_Run` of a cluster whose points of the full grid times the states reach 2^25 (`SPIN_RESTART_SECTOR_WORK`, the estimate of the polarizability's rule; `on`: whatever the size) keeps the sectors where both channel densities of `parsec.dat` keep the group the run was prepared in (the atoms on the grid, with the point charges where `PARSEC_POINT_CHARGE_SYMMETRY=auto` keeps their group; test 2 of the spin guard, 0.01 electrons; a NOTE gives the measure and the switch value), and is prepared again before its first iteration as `off` prepares it where they break it (A100 test: the energy of `off` within 1e-10 Ry and the same iteration count). One A100 node, first runs, the restart from the `parsec.dat` of a converged run, `off` -> `on`: C87H76 spin-polarized (1.6e8) 21.4 -> 11.9 s, SCF 16.7 -> 6.7 s, 12.8 -> 1.5 GiB per GPU; C35H36 (5.2e7) 7.3 -> 6.5 s, 5.7 -> 1.0 GiB at the fullest GPU. A smaller restart is faster on the full grid (O2, the fixture of 4,000 points, 1.8 -> 2.9 s; benzene, 5.4e6, 4.4 -> 5.5 s; naphthalene, 1.6e7, 4.9 -> 5.5 s, means of two: two iterations, whose first filter costs more in the sectors; the crossing near 2.8e7), so `auto` keeps it there since 2026-11-19 (the run of `off`) | `off`: both channels on the full grid (`Spin_Symmetry_Sectors: off` for the restart), as before |
| `PARSEC_WFNDAT_RESTART_SECTORS` | `off` (default): `Restart_From_Wfndat` seeds the full grid with the stored orbitals, as before. `auto` (opt-in since 2026-11-19): a cluster whose stored density keeps the group of the atoms (the same test) solves in the sectors from the stored potential, with a first filter in place of the stored orbitals (full-grid states of one representation, not projected onto the sectors); a density that breaks the group is prepared again on the full grid seeded with them. A spin file needs `PARSEC_SPIN_RESTART_SECTORS=auto` too. Saves memory, not time: C87H76 (CA, h 0.20 A) from the wfn.dat of a converged full-grid run, one A100 node per run, first runs, each measured twice on different nodes: `auto` 26.4 and 25.5 s, 8 iterations, at most 1.0 GiB per GPU, the energy of the file; `off` 22.3 and 28.1 s (its one SUBSPACE filter 0.3 and 5.8 s), 1 iteration, 5.8 GiB on its first GPU; both spend 22-24 s before the first iteration (reading the 1.2 GiB file and preparing); H2 test within 1e-6 Ry | `off` |
| `PARSEC_SPINOR_FILTER_COLUMNS` | `32` in the k-point pools, `64` on one device: columns per chunk of the filter recurrence, of `H X` and of the residuals; the random start basis is drawn 64 columns at a time whatever the chunk (bulk diamond on 16 A100, same nodes, energies the same to the digits printed: 1,728 atoms 33.1 -> 26.9 GiB, 216.6 -> 197.4 s; 2,744 atoms 63.5 -> 54.8 GiB, 622.6 -> 606.6 s; 1,000 atoms with a 4x4x4 mesh 162.1 -> 138.3 s) | `64`: the chunk of the pools before (their start basis was drawn by chunks, so another chunk started from another basis) |
| `PARSEC_CUPY_STREAMING_RITZ` | `auto`: a symmetry sector whose basis lies on one device keeps one `N x m` array; `H X` is projected a slab of columns at a time and the basis is rotated in place | `0`: two arrays |
| `PARSEC_CUPY_ROTATE_COLUMN_COPY` | `1`: the tiles of that rotation are written back along columns | `0` |
| `PARSEC_CUPY_DISTRIBUTED_STATE` | `auto`: a sector that was given several devices (MPI sector runs) shares its basis among them if its blocks fit | `0`: the whole basis on the sector owner |
| `PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS` | `1`: a device of a shared basis holds one block of its share in the Ritz step; its rows are laid into the block of its columns and `H X` is formed a slab of columns at a time | `2`: the rows as a second block; `3`: `H X` as a third block as well |
| `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` | `equal`: in the step with two blocks the slabs of `H X` cut a column range into equal parts, up to 6 GiB each where the device has the room (one block always cuts equal slabs, of at most 2 GiB on two devices and on four and of at most 4 GiB on three) | `full`: with two blocks, slabs of the full budget, at most 4 GiB, and a narrower last one |
| `PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT` | `interleaved`: a device of a shared basis holds its share of the lower and of the upper half of the columns, which later passes filter at different degrees | `contiguous`: one range of neighbouring columns |
| `PARSEC_CUPY_EXCHANGE_CONCURRENT`, `PARSEC_CUPY_EXCHANGE_COLUMN_COPIES` | `1`: a device of a shared basis receives from all peers at once, and copies inside a device walk down columns. On the way to their rows four devices take a chunk that all of them send from one partner per turn instead (`PARSEC_CUPY_EXCHANGE_PAIRS`, in the last table of switches below) | `0` |
| `PARSEC_CUPY_RITZ_PIPELINE` | `p2`: in the one-block Ritz step of a shared basis the exchanges of a round run on a second stream of each device, beside the apply of `H` and beside the slabs of the overlap (below, with the recurrence kernel of the filter) | `off` |
| `PARSEC_CUPY_EXCHANGE_STRIDED` | `1`: the pieces between the devices of a shared basis go as pitched copies, with no pack kernel and no exchange buffer, which the sharing rule then does not count (below) | `0` |
| `PARSEC_HARTREE_DEVICE` | `auto`: see `Hartree/GPU_HARTREE.md` | `off` |
| `PARSEC_HARTREE_CG_DEVICES` | `auto`: the GPU CG of an operator of 2,097,152 rows or more takes every device of the process, the values of one device bit for bit (`Hartree/GPU_HARTREE.md`, "The CG on several devices") | `1`: the CG on the Hartree device alone; a number takes that many |

The overlap condition number of the small problem follows its solve: the
device solve reads the symmetric spectrum and the host solve keeps the SVD,
unless `PARSEC_CUPY_RITZ_CONDITION` names `symmetric` or `svd` for both.
That spectrum is only compared with its limit and takes about as long as the
factorization and the eigensolve after it (0.09 of 0.20 s at 2,978 states on
an A100, timed alone). On the owner of a shared basis the other devices of
the sector wait meanwhile, so the first of them that does not hold the
Hartree objects of the process copies the overlap and estimates the number
while the owner factorizes and solves; the number is judged before anything
of the solve is, so the audits, their order and their errors are unchanged
and the coefficients are the same bit for bit. `condition_device` and
`condition_seconds` of a shared sector in `timing.json` name the device of
the last estimate and the seconds that helpers spent beside the `dense`
stage. A sector on one device has no helper. The helper takes memory for it:
its two copies of the overlap fit what its Gram arrays have just freed, the
work array of the spectrum does not. With CUDA 12.9 on a laptop that array is
4.05 times the overlap (274 MiB at 2,978 states), and the first estimate of a
thread takes 64 MiB more for its cuSOLVER handle; the code counts 348 MiB for
23,768 electrons and 501 MiB for 29,576 (four `m x m` arrays, 128 doubles
per state, 2 MiB, a byte per entry for the test that the overlap is finite,
and the handle), which holds what cuSOLVER asked at every size from 100 to
16,000 states there. The array stays in the free list of its stream for the
next step. That memory is why the device that holds the Hartree objects of
the process, taken to be its fullest, was passed over at first, and it is
by default (`auto`): the root's sector whose only other device holds them,
the second of its two on 8 GPUs, keeps the estimate on its owner, and its
dense stage took 3.06 against 1.85 to 1.86 s and 4.88 against 2.87 to
2.89 s of the three other sectors for 23,768 and 29,576 electrons. On A100
nodes that device was not the fullest: it
sampled 50,567 and 66,491 MiB where the owner of the first sector, which
holds the work arrays of the small solve, sampled 50,679 and 67,083.
`PARSEC_CUPY_RITZ_CONDITION_HELPER=any` therefore has such a sector ask that
device where the rule that
shares a basis counts the room: the blocks of the basis and the bytes of the
estimate together within the share of the device that the rule allows
(`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION`). The rule itself shares the
same bases under every value, and where the room is not counted, as for a
basis that is shared by name beyond its share, the estimate stays on the
owner. With
`any` named beside the default on A100 nodes, in runs that also cut the
overlap into other slabs (first runs, 8 GPUs, one allocation), the dense
stage of that sector took 1.88 instead of 3.06 s and 3.02 instead of 4.88 s,
what the others took; its helper sampled 304 and 448 MiB more, and the
fullest device of the run 192 MiB more for 23,768 electrons (50,871 MiB:
the helper became it) and none for 29,576. The four sectors of such a run
are solved side by side, so the run gains what its slowest sector loses to
the next: by the stage seconds of those runs about 0.5 and 0.2 s of 166.9
and 242.9 s of SCF, and at most the 1.2 and 1.9 s of the stage. `any` was
the default for a while on those stage seconds, and complete calculations
decided against it (2026-10-16): first runs on 8 A100 GPUs, FP64, with the
launch profile of the measured series, two of each value in alternation and
the same total energies in every run, took 163.40 s of SCF on average with
`any` (164.03 and 162.76 s) against 163.51 s with `auto` (163.43 and
163.59 s) for 23,768 electrons, 0.07 % less, and 236.16 s (236.82 and
235.50 s) against 235.92 s (235.99 and 235.85 s) for 29,576, 0.10 % more;
the fullest device peaked at 47.87 against 47.67 GiB and at 63.52 GiB with
either. `any` would have stayed the default had it taken at least 0.4 % less
SCF time in either case, so the default is `auto` again. Sectors on
four devices (16 GPUs) and the other sectors on two always had a helper and
are the same under `auto` and `any`; `off` asks no device. `any` has counted
the room since it became the default: before, it asked that device whatever
the room, which no value does now, so a run that names
it for a basis shared by name beyond its share keeps that estimate on the
owner, with the same bits. An error of
the helper other than a failed estimate, out of memory among them, ends the
run; nothing falls back to the owner. The solve with a helper first ran where
one device stood for both, with the same bits. Sectors on several devices
have run with it on 8 and 16 A100 GPUs since (`condition_device` of their
records names the device of the estimate), and `any` with the room counted
in the complete calculations above. The
blocks of a shared basis fit if they take at most
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION` of a device, 0.85 by default.
Only a CHEBFF first solve creates a shared basis. It is filtered in FP64 and
takes the generalized solve of the filtered basis (described under
saved-SUBSPACE below) in every pass. That includes every cycle of the first
solve, where one device orthonormalizes the basis unless
`PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ=1`. `auto` therefore shares only a basis
for which `PARSEC_CUPY_GENERALIZED_RITZ` and its work threshold select that
solve; `PARSEC_CUPY_DISTRIBUTED_STATE=1` shares regardless of them. A sector
whose blocks do not fit stays on its owner with one array.
A device of a shared basis holds one block of its share of the basis in the
Ritz step (`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=1`, the default): its row
block lies in the memory of its columns. The
columns leave their block in rounds, a slab of every device per round, the
last columns first; the slabs are equal parts of every block, of at most the
bytes of the streaming budget (`PARSEC_CUPY_STREAMING_RITZ_BYTES`,
by default an eighth of the sector basis but at least 1 and at most 4 GiB;
the limit of 2 GiB of a sector on one device, above, is not read here)
whatever `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` says, and with four or more
devices per sector, where that budget is not set, of at most an even share
of 4 GiB as well (`PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES`): 1 GiB with
four. With two devices per sector a slab takes at most 1 GiB too where the
budget is not set (`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`). Where
those limits leave fewer than 32 columns a slab is widened up to that, but not
beyond the 2 GiB that both limits were before
(`PARSEC_CUPY_DISTRIBUTED_STATE_MIN_SLAB_COLUMNS`; the last table of switches
below, with the runs). The workspace and
the rotation tile have the size of the slabs that were cut. A slab is moved
into the workspace and
sent to the row layout of all devices, and the rows arrive in front of those
of the rounds before at the end of the block, in the memory that the slabs
so far have left. `H` times the slab follows the same way and is projected on the rows
that the block holds by then, one product per round as wide as the slabs of
all devices together. That width is a multiple of 64 columns wherever the
blocks allow it (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, above): in a whole round
every device gives the same slab, the widest that makes the round a multiple
(32 or 64 columns each of two devices; 16, 32, 48 or 64 each of four) and is
no wider than the equal slabs were, as many such rounds as the narrowest
block has slabs; what the blocks hold beyond those rounds goes first, cut
into equal slabs as before, where the product is shortest. Equal slabs of
all devices keep a block as full as it was, so the rows still lie in the
block of their columns. A sector whose blocks or equal slabs are narrower
than such a slab is cut as before, and `slab_cut` says which: `multiple` or
`equal`. The summed Gram matrices are put back into the order
of the basis for the small solve, the rows are rotated in place and return
in the rounds reversed. The basis crosses the links as often as with two
blocks (its columns, `H` times them and the rows back: 2.25 times the basis
with four devices) and is copied twice more inside each device. After a
failed stability audit the rounds are reversed without the rotation, which
puts the filtered columns back bit for bit in the memory the step holds
already, and the fallback runs as before. The blocks of a trial basis are
given about one column of room for the step; a block without it, such as the
fallback leaves, gets a row block from the pool. The runner's `timing.json`
counts those per shared sector (`separate_row_blocks` under
`distributed_state`, next to `tall_blocks`, the blocks of the last step, and
`slab_cut` and `slab_columns`, the cut of its slabs and the columns of the
widest, and `projection_columns`, the columns of the widest right-hand side
of a product of its projection): where the count is not 0, a device held two
tall blocks in those passes, more than the fit rule counted. Beside the
`seconds` of the stages of a shared sector, `overlap_seconds` is the part of
`gram` that the overlap took; what is left of `gram` is the projection of
the rounds. The fit rule counts
one block, the workspace and the chunk: 26.0 GiB instead of 51.0 for the
84 GiB sector basis of 23,768 electrons on 16 GPUs and 31.1 instead of 61.2
for the 52 GiB of 19,392 electrons on 8 (with a chunk of 1 GiB; where no
size is named a sector on four devices counts 0.125 GiB more for a buffer
that may follow a slab, see `PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` below), so
that 16 GPUs share a sector basis
of up to 249 GiB instead of 117 and 8 GPUs one of up to 125 GiB, the 84 GiB
one among them (47.0 GiB), which
two blocks cannot place there (93.0). A sector on one device (four sectors on
1, 2 or 4 GPUs) shares nothing and never reads the setting; a basis that only
the rule of two blocks refused is now shared instead of staying on its owner.
First runs on A100 nodes (no symmetry cache, strict parity passed), one block
against two: 23,768 electrons on 16 GPUs peaked at 33.3 GiB per device in
135.5 s, against 55.3 GiB and 137.1 s with two blocks and slabs of the full
budget and 57.9 GiB and 133.9 s with two blocks and equal slabs, all three in
one allocation; 19,392 electrons on 8 GPUs at 37.8 instead of 65.1 GiB in
140.9 against 140.2 s; 19,392 electrons on 16 GPUs at 23.8 GiB in 88.5 s and
14,680 on 8 at 29.0 GiB in 101.6 s, where two blocks had taken 38.3 GiB and
86.4 s and 47.7 GiB and 102.2 s in another allocation; and 23,768 electrons
ran on 8 GPUs, at 54.9 GiB in 223.9 s. No pass of these runs took a row block
from the pool. Over the ten SCF steps the Ritz stages of a sector took 0.5 s
(19,392 electrons on 8 GPUs) and 2 s (23,768 on 16) longer than with two
blocks and equal slabs, and the total energies differ from those of two
blocks by 3e-8 to 5e-8 Ry. That is why one block is the default. Those runs
used the step as it was first written. As it is now, named `1` beside the
other defaults of this version in one allocation, it ran 23,768 electrons on
16 GPUs at 33.2 GiB in 114.1 and 114.4 s, where two blocks took 57.9 GiB and
113.1 s with equal slabs and 55.2 GiB and 116.6 s with full ones; 10,456
electrons on 8 GPUs at 16.5 GiB in 45.9 s against 26.4 GiB and 46.4 s; 3,480
and 14,680 electrons on 16 GPUs at 2.5 GiB in 8.1 s and 18.8 GiB in 53.9 s;
and 19,392 and 23,768 on 8 at 37.7 GiB in 119.0 s and 54.8 GiB in 195.7 s.
None of the 48 shared sectors of the runs with one block took a row block
from the pool, and the energies again differ from those of two blocks by 3e-8
to 5e-8 Ry. All of these runs named the block count. The default itself, the
variable unset, has run since: in one series of first runs, 3,480 to
29,576 electrons on 8 GPUs and 3,480 to 39,368 on 16,
no rank named the block count or a slab budget, and each of the 68
shared sectors recorded one tall block and no row block from the pool.
The runs that named it all cut slabs of at most 4 GiB, with which the rule
counted 30.0 GiB for the 84 GiB basis on 16 GPUs and shared up to 233 GiB. A
round projects
the slabs of all devices of a sector in one product, so four devices reach
the same width with slabs half as large, and the workspace is two slabs. With
the budget set to 2 GiB, first runs on 16 GPUs peaked at 30.1 instead of 33.2
GiB per device for 23,768 electrons (rounds of about 271 instead of 496
columns) and at 37.9 instead of 41.3 GiB for 29,576 electrons (247 instead of
463). The time moved in both directions. The eigensolver of the first run
took 94.2 s where six runs with 4 GiB took 93.1 to 93.2 s, and its SCF 108.2
instead of 107.1 to 107.4 s: about 1% more, in the Gram stage (16.2 against
15.6 s per sector) and on the way back to the columns (4.0 against 3.7 s).
That of the second took 137.9 instead of 140.2 s, and its SCF 152.7 instead
of 155.0 to 155.5 s: about 1.5% less, the Gram stage 26.6 against 29.3 s. The
program totals, 115.1 against 114.7 s and 161.9 against 164.3 s, also hold
the setup, which varies by 0.4 s between such runs. The total energies differ
from those of the 4 GiB runs by 6e-8 and 2e-8 Ry; the first passed strict
parity, the second was compared with no reference. That is the default of a
sector on four devices now. The default itself, the budget unset, has run
since, 3,480 to 39,368 electrons on 16 GPUs in that series (above). It also
changed the cut of 14,680 and 19,392 electrons on 16 GPUs (rounds of at most
371 and 349 columns instead of 614 and 609, a workspace of 3.6 and 3.8
instead of 6.0 and 6.6 GiB) and of 39,368 electrons (184 instead of 354
columns, 3.9 instead of 7.5 GiB, for the basis that ran: 5,660,200 rows and
4,928 columns). Of these three, 19,392 electrons ran with it beside slabs of
4 GiB in one allocation, the budget the only difference: SCF 65.8 against
66.0 s and 21,515 against 24,371 MiB on the fullest device, before the
Hartree boundary was brought in. The other two ran with it and not beside
slabs of 4 GiB alone: which way the cut by itself moves their time is not
known. Sectors of 10,456 electrons and fewer are cut by the budget as
before. Those are the rounds of `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1`: the
default cuts them further, at whole multiples of 64 columns (below), into
rounds of 320, 320 and 128 columns for the three.
A sector on two devices (8 GPUs) was left at 4 GiB at first: with 2 GiB its
rounds are about 135 columns wide for 23,768 electrons, which had not run,
and the products of the two-block step were slower at 90 to 141 columns (Gram
stage 17.5 s) than at 184 to 189 (14.5 s). First runs on 8 GPUs with the
budget set to 2 GiB, beside runs with 4 GiB in one allocation, peaked at 49.7
instead of 53.5 GiB per device for 23,768 electrons (22 rounds of at most 136
columns instead of 11 of 271) and at 65.5 instead of 69.2 GiB for 29,576
electrons (29 rounds of 128 instead of 15 of 248). The time moved in both
directions here too: the first took 181.8 instead of 179.7 s, its Gram stage
35.5 to 36.0 instead of 33.2 to 33.4 s per sector, and the second 256.2
instead of 258.7 s, 52.1 to 52.7 instead of 55.1 to 55.3 s. A budget of 3 GiB
gave 51.5 and 67.2 GiB in 181.4 and 258.3 s. All six runs converged in ten
steps, and the total energies of 2 and 4 GiB differ by 4e-8 and 3e-8 Ry. So
slabs of at most 2 GiB are the default of a sector on two devices as well
(`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`, 2147483648 unless it is
set): the fit rule counts 47.0 instead of 51.0 GiB for the 84 GiB basis and
62.0 instead of 66.0 for the 114 GiB one of 29,576 electrons, and two devices
share a basis of up to 125 GiB instead of 117. The default itself, the budget
unset, has run on 8 GPUs since, for these two and for 14,680 and 19,392
electrons, whose cut it changes as well (rounds of at most 205 and 174
columns instead of 369 and 348, a workspace of 4.0 and 3.7 instead of 7.2
and 7.5 GiB): the last table of switches, below, has the peaks.
The limit is the same for a basis of any size. Up to 16 GiB a
slab is the eighth of the basis that it was; from there to 32 GiB it took
that eighth and now takes 2 GiB, in five to nine rounds instead of four or
five, where no cluster has run. The sectors of 10,456 electrons on 8 GPUs
(18.8 GiB) keep their five rounds of 132 columns, and those of smaller
clusters their cut: their results stay what they were bit for bit. A limit
that began at 32 GiB, where a slab took the full 4 GiB, was tried first and
dropped: a basis just below held a workspace of nearly 8 GiB and one just
above 4, so the few columns that a trim takes from a sector after its first
solve could double the workspace that the fit rule, which is asked for the
first solve, had counted for it. With one limit a slab never narrows as
columns are added. A sector on three devices keeps slabs of 4 GiB at every
size: none has run.
`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=4294967296` gives a sector on
two devices (8 GPUs) the slabs it had, whatever the size of its basis and in
every step of a run: the limit is then that of the budget itself.
`PARSEC_CUPY_STREAMING_RITZ_BYTES=4294967296` does so for a sector on four or
more devices (16 GPUs), and it is the way back for those runs only: a sector
on three devices or on one has its former slabs without either, and with that
budget a basis below 32 GiB gets slabs of 4 GiB where it took an eighth of
itself, on two or three devices and in the one-array route of a sector on
one. For a sector on two devices it gives back the slabs of the steps in
which the basis holds 32 GiB or more, as in the runs above, and no others: not
those of a smaller basis, and not those of a sector once a trim has taken it
below 32 GiB.
Those two settings give back the columns that a slab may have. The rounds are
cut from them at whole multiples of 64 columns since
(`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`): the rounds that a run had take
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside either, and its bits
`PARSEC_CUPY_RITZ_GRAM_SLABS=8` as well, for the slabs of its overlap. With the
slabs of 2 GiB the rounds of the shared sectors of the measured series
become, on 8 GPUs, nine of 192 columns after one of 114 for nine of 205
(14,680 electrons), eighteen of 128 after one of 127 for fourteen of 174
(19,392), 23 of 128 after one of 34 for 22 of 136 (23,768) and 28 of 128
after one of 120 for 29 of 126 to 128 (29,576); on 16 GPUs five of 320 after
one of 242 for five of 371 (14,680), seven of 320 after one of 191 for seven
of 349 (19,392), eleven of 256 after one of 162 for eleven of 273 (23,768),
nineteen of 192 after one of 56 for fifteen of 248 (29,576) and 38 of 128
after one of 64 for 27 of 184 (39,368). The workspace of the slabs that are
cut falls with them, by 0.3 to 1.2 GiB per device on 16 GPUs and by 0.2 to
0.9 on 8, where 29,576 electrons keep theirs (derived from the cut). All of
these cuts but that of 19,392 electrons on 16 GPUs have run on A100 nodes
since, in first runs against `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` (the last
table of switches, below): their records read `multiple`, with slabs of 96, 66,
64 and 64 columns on 8 GPUs and of 80, 64, 48 and 32 on 16, and the fullest
device held 0.2 to 1.2 GiB less, the same for 29,576 electrons on 8 GPUs.
More rounds take time in the stages around the product. In the runs that
halved the slabs (above) a round more took 1.2 and 1.3 ms per pass on two
devices and 6.1 to 8.3 ms on four, `apply`, `to_rows` and `to_columns`
together. Held against that, the rounds of 29,576 and 39,368 electrons on 16
GPUs, 20 for 15 and 39 for 27, were to keep 0.1 to 0.2 and 0.6 to 0.9 s per
sector of the 0.6 and 1.9 s that the rounded widths give them (derived),
beside 1.4 and 1.6 s from the overlap. Measured in those first runs, one
pair each: the overlap of the two sectors took 2.1 and 2.4 s less and the
products of their rounds 0.8 and 3.2 s less, while `apply`, `to_rows`,
`to_columns` and `rotate` together took 0.5 and 5.0 s more, 3.3 s of the
latter on the way back to the columns. So the rounds of 29,576 electrons
kept 0.3 s per sector and those of 39,368 lost 1.9 s, less than their
overlap gained (SCF 275.3 against 275.8 s). On 8 GPUs the rounds of 23,768
electrons kept 6.6 s per sector and those of 14,680 and 19,392 1.2 s each.
Those runs received every chunk from all its senders at once. With the turns
of four devices (`PARSEC_CUPY_EXCHANGE_PAIRS`, below) the whole rounds have
run on 16 GPUs since, with all these switches together, beside a run with
every switch of the last table at its former value in the same allocation: for 39,368
electrons the four stages together took 81.4 against 84.3 s per sector, the
products of the rounds 26.4 against 30.0 s and the overlap 31.4 against 33.8
s. Equal slabs in turns took 79.2 s in those stages in another allocation,
2.2 s less than the whole rounds: 0.9 s on the way to the rows, 0.5 s each on
the way back and in the rotation, 0.2 s in `apply`. So with the turns the
rounds of 39,368 electrons kept about 1.3 s per sector. The way back took
10.6 s there, where the run without turns had 13.3 s and seven runs with
equal slabs 10.0 to 10.1 s. For 23,768 electrons the stages took 24.6 s
(24.9 with equal slabs in turns) and the products of the rounds 6.9 against
8.5 s. And the
slab of a whole round can be little more than half of the widest equal one:
a sector of 3,714 columns on two devices, ten more than that of 29,576
electrons, is cut into 58 rounds of 64 columns after one of 2 where it had 30
of 122 to 124, for 3 to 5% of its projection (derived) and half the
workspace. That cut has not been measured, and no product narrower than 124
columns was timed alone. `overlap_seconds` separates the parts: in a run
against `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` the overlap shows what its slabs
gave, what is left of `gram` what the rounds gave, and `apply`, `to_rows`,
`to_columns` and `rotate`, whose tile follows the slab, what they cost.
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2` selects the former step, in which a
device holds two blocks of its share of the basis: its columns and its rows
of all columns. The results of the two steps differ by round-off. `H X` is
never stored as a whole: each device applies `H` to a slab of its columns, the
slabs go to the row layout through the exchange buffers and are projected
there, and the rows are rotated in place. Next to the two blocks a device
holds a workspace of two slabs and one exchange chunk. A slab takes at most
the bytes of the streaming budget
(`PARSEC_CUPY_STREAMING_RITZ_BYTES`, by default an eighth of the sector basis
but at least 1 and at most 4 GiB) and at most half of a block. Every column
range of a device is cut into as few slabs of that width as it takes, all of
them equal, and the workspace has the size of the slabs that were cut. Where
the budget is not set and the blocks fit a device with a workspace of two
slabs of 6 GiB, by the rule above, that is the upper limit instead of 4 GiB.
Every such step asks that rule, also of a basis that
`PARSEC_CUPY_DISTRIBUTED_STATE=1` shares without it:
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION` then still decides the width
of the slabs, and a mistyped value stops the run where the trial basis is
sized. The rule counts the memory of the device and knows no limit of the
memory pool: a run under `CUPY_GPU_MEMORY_LIMIT` names the former cut or
sets the budget.
The former cut, `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS=full`, takes slabs of
the full budget and leaves what remains of a range as a last one: the ranges
of about 372 columns of 23,768 electrons on 16 GPUs became 141, 141 and 90,
and a run with the budget set to slabs of 189 columns, which cut them into
two, took 133.1 instead of 135.8 s for 2.7 GiB more. The equal cut gives
those ranges two slabs of 184 to 189 columns (workspace 10.6 instead of 8.0
GiB) and the ranges of about 304 columns of 19,392 electrons on 16 GPUs two
of about 152 instead of 186 and 118 (6.5 instead of 8.0 GiB). The results of
the two cuts differ by round-off: the projection is summed over other slabs,
and the rows are rotated in tiles of another height, a tile being as tall as
a slab of the workspace lets it be. In first runs of one allocation the equal
cut took 133.9 instead of 137.1 s for 23,768 electrons on 16 GPUs, at 57.9
instead of 55.3 GiB, and beside the other defaults of this version 113.1
instead of 116.6 s at 57.9 instead of 55.2 GiB; apart from 19,392 electrons
on 8 GPUs (140.2 s, 65.1 GiB, no run of the former cut beside it) its effect
on the smaller sectors has not been measured.
`PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` acts on this step only: named beside
the default of one block it changes nothing, and `slab_cut` of the run reads
`equal` or `multiple`. A comparison of the two cuts names
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2` on both sides.
In this step every slab is the right-hand side of a product of its own, so a
slab of 64 columns and more is cut down to a whole multiple here too, the
full slab of the budget or the widest of the equal ones, and a range ends
with a narrower one: ranges of 372 columns in two slabs of 186 become 128,
128 and 116 (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` for the cuts as they were).
That has not run on a GPU.
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=3` selects
the step before that, which holds `H` times the columns as a third block; the
results of the two differ by round-off. For 23,768 electrons on 16 GPUs two
blocks peaked at 55.3 GiB instead of 68.3 in 140.7 against 138.9 s, and
19,392 electrons on 8 GPUs took 143.6 s with a shared basis that three
blocks cannot hold there (193.2 s on the sector owners). The fit rule
counts what the setting selects. It keeps no room for the fallback after a
failed stability audit, which gathers the whole basis on the sector owner
next to the owner's columns, `1 + 1/devices` times the basis. A basis can
therefore be shared although that fallback could not run, with two blocks
on two devices as well: the 52.2 GiB per sector of that run on 8 GPUs fit,
and the fallback would need 78.3 GiB of a device of 79.25. One block shares
bases that no device could gather at all.
The columns of a device of a shared basis are two ranges: its share of the
lower and of the upper half of the columns. Later passes filter the lower
half at a lower degree than the upper half. With one range of neighbouring
columns per device, the former layout, half of the devices then filter at
the higher degree only and the others wait for them; with a share of each
half all of them do the same work in every pass, and no column moves and no
block grows. For 19,392 electrons on 16 GPUs the later passes filtered a
sector in 16.4 instead of 19.5 s and the program took 86.3 instead of 89.3
s; 23,768 electrons took 135.8 instead of 140.0 s, and 19,392 electrons on
8 GPUs 138.2 instead of 144.0 s, each at the same peak memory.
`PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=contiguous` selects the former
layout. The trial basis is created in either layout from the same random
stream, and the rows of the Ritz step hold the columns in their own order,
so the results of the two differ by round-off only. A sector with fewer
filter blocks in a half than it has devices keeps contiguous ranges. So does
`PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=balanced`, which moves columns between
the devices from pass to pass instead and takes headroom in every block for
it; `interleaved` named together with it is an error.
On the way to the rows all devices of a sector begin to receive a chunk at
the same moment, each from every other one: twelve copies at once with four
devices per sector (16 GPUs). On the four A100 of one node, pieces of 0.49
GiB issued so arrived at 58.9 GiB/s per device (17.3, 25.1 and 25.1 ms per
chunk in three repeats), and in three turns of two pairs, both directions of
a pair at once, at 83.8 GiB/s (17.6 ms each time). The stage seconds of
first runs say the same of the program from 14,680 electrons on: 50 to 53
GiB/s per device on that way with four devices per sector, and 80 to 82
with two (8 GPUs), which are one pair. The four devices of a sector
therefore take a chunk that all of them send in those three turns by
default. In a turn a device receives from one partner, which receives from
it, on the stream it computes on, and all four are done with a turn before
the next begins: two more waits for all devices per chunk, which earlier
runs put at about 0.3 ms each. `PARSEC_CUPY_EXCHANGE_PAIRS=0` issues the
twelve together as before. The bytes that arrive are the same and so are
the results, bit for bit. Once a
device has sent all its columns, the chunks that the others still send are
received as before, from all their senders at once. They have not been
timed in turns: with one sender a turn would be a single copy where three
started together, and a receiver alone took three sources side by side at
three times the rate of one (262.5 against 87.5 GiB/s). The step with one
tall block, the default, gives all devices equally many slabs, and every
chunk of the bases of 3,480 to 39,368 electrons on 16 GPUs has four senders:
with the equal slabs that the turns were timed with, and with the whole
rounds of `PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, in which every device gives the
same slab (derived from the column ranges of those bases). The whole rounds
are more and their pieces smaller, 8 to 154 chunk steps per pass where the
equal cut had 6 to 108. For three of the bases the slab of a whole round is
just wider than a chunk of 1 GiB (`PARSEC_CUPY_EXCHANGE_CHUNK_BYTES`), which
holds 31 columns of 39,368 electrons, 43 of 29,576 and 92 of 10,456 where
the slabs have 32, 48 and 96: every such slab travelled as a chunk and a
second one of 1, 5 and 4 columns, which are 76 of the 154, 38 of the 78 and
6 of the 14 chunk steps of a pass, each with its three turns (derived),
until the exchange buffer followed such a slab (below). The
turns have run with the whole rounds on 16 GPUs for 3,480, 23,768 and 39,368
electrons, with all these switches together: the way to the rows of a sector took 0.36,
6.68 and 17.38 s, where equal slabs took 0.39, 6.78 and 16.50 s in turns
(another allocation) and 0.41, 9.05 and 21.62 s received at once (the same
one). So the turns keep their gain with the whole rounds, less about 0.9 s
per sector for 39,368 electrons, whose way has 46 chunk steps more per pass.
A chunk of 1,500,000,000 bytes holds a slab of every whole round of these
bases: 78, 40 and 8 chunk steps per pass and none of fewer than 18 columns,
for 0.4 GiB more in the exchange buffer of every device, which the rule that
shares a basis counts (0.724 instead of 0.719 of a device for 39,368
electrons). That chunk has run with the whole rounds on 16 GPUs beside the
default, in one allocation (23,768, 29,576 and 39,368 electrons,
totals equal to the last bit): the way to the rows of a sector took 6.36,
8.73 and 16.04 s against 6.64, 9.32 and 17.38 s, the way back 4.28, 5.84 and
10.65 s against 3.91, 5.56 and 10.56 s, the SCF 93.2, 132.3 and 266.1 s
against 92.9, 132.9 and 267.5 s, and the fullest GPU held 408 MiB more in
each. That is at most 0.5 % of the SCF for 0.4 GiB of every device, so the
default chunk stays at 1 GiB. With the equal slabs of
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` the larger chunk would be the one just
short of a slab (46 columns against 44 for 39,368 electrons).
Since then the exchange buffer follows a slab that is little wider than a
chunk, where `PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` is not set. A slab, or in the
steps that send them a block, of which up to an eighth more than 1 GiB lies
outside the rows of a device goes to the rows in one chunk, through a buffer
of exactly its size: never more than 1,207,959,552 bytes, 128 MiB above the
chunk. Slabs that a chunk holds and wider ones travel as before. So does the
way back to the columns, which is cut by 1 GiB whatever the slab, in the
first part of the same buffer: that way has no step that all devices take
together, so a second chunk costs it no wait, and its stage seconds were
higher with one chunk per slab in every pair of runs (4.28, 5.84 and 10.65 s
against 3.91, 5.56 and 10.56 s above). What the larger buffer saves is the
fixed cost of a chunk step of four devices, a pack and three turns with a
wait for all devices behind each. On the A100 of four nodes, with the size
named so that the 32 columns of 39,368 electrons were one chunk
(`1090000000`) against the default in one allocation, the way to the rows of
a sector took 16.04 instead of 17.41 s over the 1,014 slabs of 13 passes,
1.35 ms per chunk step that was not taken, and the way back 10.65 instead of
10.57 s. In those 1.35 ms the turns move 0.11 GiB per device at the 83.8
GiB/s measured for them, which is where the eighth comes from: a second chunk
below about that size spends more of its step waiting than copying, and above
it the step is mostly copies, which no buffer saves. Both figures are single
measurements, the seconds of one pair of runs and the rate of pieces of 0.49
GiB on one node, so the eighth is a round number near 0.11 GiB and no fitted
limit. The measured slab nearest to it is that of 29,576 electrons, 0.108
GiB wider than a chunk, whose way to the rows took 8.73 instead of 9.32 s
with the chunk of 1,500,000,000 bytes above. The bytes that such a
chunk would hold are what the larger buffer costs every device for the whole
run. A slab of the full budget of four devices, 2 GiB, has 1.5 GiB outside a
device and keeps its two chunks. A Ritz step sizes the buffer for the slabs
that at least half of its conversions to the rows send. The whole rounds give
every device the same slab, and what the blocks hold beyond them goes first
in a round or two of other slabs, for which a larger buffer would cost its
bytes for next to nothing: at the default radius the sector of 39,368
electrons sends 25 rounds of 48 columns (1.44 GiB outside a device) and one
of 36 (1.08 GiB), where 83 MiB more would save two of the 104 chunk steps of
a pass. Its buffer stays 1 GiB in every pass and that round travels in two
chunks as before. The half is a choice and not a measured limit: a round that
is followed saves its two chunk steps of a pass (2.7 ms on four A100)
whatever its share of the rounds, and the half only says that most
conversions of a step use the bytes that every device holds for them. At
worst a step of two rounds, one of them within the eighth, takes up to 128
MiB on every device for those two chunk steps. Of the
bases above, the slabs of 10,456,
29,576 and 39,368 electrons are within an eighth (33, 110 and 12 MiB more in
the buffer of every device) and travel whole, in 8, 40 and 78 chunk steps
per pass where they took 14, 78 and 154; those of 14,680, 19,392 and 23,768
electrons take 0.16, 0.28 and 0.35 GiB more than a chunk and travel as
before. A sector on two devices sends slabs of at most 2 GiB, 1 GiB of them
outside a device, which a chunk holds but for a few bytes. The rule that
shares a basis counts the most that the buffer can take for the slabs of its
budget: up to 128 MiB more than before on four devices, from 10.7 GiB of
basis, where a slab of the budget begins to exceed a chunk outside a device,
and all 128 MiB from 12 GiB, each a column of the slab later at most (0.7203
instead of 0.7188 of a device for 39,368 electrons; four devices of 79.25
GiB still share up to 249 GiB), and nothing more on two with the slabs of
their default. (The pitched exchange, the default since 2026-10-22, takes no
buffer and the rule counts none; these are the counts of
`PARSEC_CUPY_EXCHANGE_STRIDED=0`.) A sector on two devices
that names its former slabs of 4 GiB
(`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=4294967296`) counts the 128
MiB too, since such a slab has 2 GiB outside a device and a narrower one can
be within the eighth: its former count takes the chunk named beside it. The
count of the state storage follows that of the rule. A buffer that a later pass needs larger, as
a trim can bring about by moving the cut of the slabs, is taken after the
free blocks of its stream have gone back to the driver, as for an outgrown
workspace (`MULTI_GPU.md`). `exchange_buffer_bytes` of a shared sector in
`timing.json` is the largest buffer a device took.
`PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=1073741824` gives the chunks, the buffer
and the counts of before, and any other size the chunks of that size both
ways. The results are the same bit for bit however the slabs are cut: the
bytes that arrive, and the order in which a row block receives them, do not
depend on it. The default itself, one chunk to the rows and chunks of 1 GiB
back, has not run on A100 nodes. With the default chunk set to a size that
the slabs of a laptop case exceed by an eighth or less, it ran complete
calculations on one GPU named as the devices of every sector, where every
copy stays inside the device, with the bits of the code before it: 176
electrons with the GPU named twice (3,600,000 bytes, slabs of 9 columns) and
four times (2,400,000 bytes, slabs of 4 columns, in the turns of four), and
864 electrons with it named twice (28,000,000 bytes, slabs of 26 and 27
columns) and four times (22,000,000 bytes, slabs of 14). Those slabs went to
the rows in one chunk and back in two, and `exchange_buffer_bytes` of their
sectors was the slab. With a size that their slabs do not follow, the 176 electrons kept the
chunk as their buffer in every pass, with the same bits: 2,700,000 bytes on
two named devices, where one round in three of a sector is within the
eighth, and 1,760,000 on four, where the widest slab of a round is beyond
it.
A block that is a whole number of such slabs gives nothing to the round that
goes first, whose chunks then have fewer senders and go together. The slabs
of the step with two blocks end with
their column range: there 23 % of the bytes of 10,456 electrons on 16 GPUs
travel in chunks of one to three senders, as many for 14,680 where its slabs
are cut wider, 6 to 9 % for 3,480 to 7,120 electrons, and none or next to
none from 19,392 on. `to_rows_copies` of a shared sector in `timing.json`
names the order of its last way to the rows: `pairs` (the chunks that all
four send in turns), `together` or, under
`PARSEC_CUPY_EXCHANGE_CONCURRENT=0`, `queued`, which takes no turns. A
sector on two devices is unchanged, and so is one on three or on more than
four, which have not been timed. The way back is unchanged too: there every
device fetches chunk after chunk at its own pace, with no wait
for the others, and the stage seconds give its four devices 75 to 81 GiB/s
each once the cost of a round is taken out (61 if none is); turns would add
three waits per chunk to it, about what they could save. At 72 to 84 GiB/s
on the way to the rows, less the waits, the program would take 1.7 to 2.5 s
less of 106.9 for 23,768 electrons on 16 GPUs and 4.1 to 6.1 s less of 290.8
for 39,368. That was derived; in first runs since, the SCF took 2.0 s less
of 99.6 for 23,768 electrons and 5.6 s less of 279.6 for 39,368 (the last
table of switches, below). For the small bases the sign was
open, and the turns are taken there too. Their stage is not limited by the
copies: its seconds give 22 to 31 GiB/s per device for 3,480 electrons on 16
GPUs (pieces of 0.05 GiB, six chunks per pass, 0.38 to 0.52 s per sector of
a program of 8.2 s), 36 to 40 for 5,264, 39 to 44 for 7,120 and 46 to 49 for
10,456, and on 8 GPUs, where a sector is one pair, 51 to 55, 61 to 65, 59 to
66 and 75 to 79 against 80 to 82 from 14,680 on. At the two rates measured
for pieces of 0.49 GiB the turns can save 3,480 electrons 0.06 s per run.
Their 156 more waits cost 0.02 to 0.05 s at 0.12 to 0.35 ms each, and 0.17 s
at the 1.1 ms per wait that the stage seconds leave room for: between 0.04
s less and 0.11 s more per run, 1.4 % of the program. The same arithmetic
gives 5,264 electrons between 0.11 s less and 0.02 s more, 7,120 electrons
0.01 to 0.18 s less and 10,456 electrons 0.16 to 0.46 s less of 28.3. No
piece size but 0.49 GiB has been timed in turns. The turns have run on the
host stand-in of the exchange and, with the one GPU of a laptop named four
times as the devices of a sector, on device memory: the Ritz steps with
one, two and three blocks gave the same bits with them as without. Every
copy stays inside the device there. Sectors on four devices have run with
the turns since, 3,480 to 39,368 electrons on 16 GPUs, default against `0` in
one allocation: the same bits, and the times in the last table of switches.
`PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION`, 0 by default, returns to two
arrays on a device where they take no more than that fraction of its memory
(0.8 was the rule before one array became the default). The one-array route
measured the same time as two arrays with 24.2 instead of 40.7 GiB for 10,456
electrons on four GPUs. For 19,392 electrons on 16 GPUs the device solve cut
the small solve from 15.5 to 1.9 s, and the exchanges of the shared basis,
chunked and with both switches on, took 7.1 instead of 17.3 s. The
`--serial-control` mode of
`benchmarks/mpi_full_scf.py` keeps the former routes wherever its launcher
names none, so that it stays an independent reference.

Set-up and filter routes that became defaults after first runs (no symmetry
cache) on the same nodes, each with the setting that selects the former one:

| Environment variable | Default | Former route | Described in |
|---|---|---|---|
| `PARSEC_CUPY_MIXED_FILTER` | `off`: every filter in FP64 | `auto`: FP32 later filter for large sectors | above |
| `PARSEC_CUPY_IMPLICIT_TILE` | `auto`: affine tiles of 16 rows for a sector whose filter work repays their packing (columns, counted applications and share of regular tiles against the devices of the process; above), unless `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` names a row limit for the sectors that one device filters | `0`: the slot-major stencil; `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` at `650000` with `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` at `1100000`: the row limits before the rule; the first at 1,100,000, 1,555,634 or 2,200,000 for one, two or four sectors on every device (4, 2 or 1 GPUs): the limit of 1,100,000 rows before those | above, `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_CUPY_IMPLICIT_PACK_WORKERS` | `4`: threads that pack the chunks of a tiled stencil | `1` | `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` | unset: under `PARSEC_CUPY_IMPLICIT_TILE=auto` a sector whose basis or filter several devices share (8 and 16 GPUs) follows the same rule in filter work; the owner packs the tiles and the others read its arrays. A value is the row count from which such a sector gets them | `1100000`: the row count before the rule; `PARSEC_CUPY_IMPLICIT_TILE=0`: the slot-major stencil on every device | above, `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_IONIC_GPU_COUNT` | `auto`: the GPU ionic sums run on every device of an MPI rank that solves symmetry sectors, on one device otherwise | `1` | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_CUPY_FILTER_GRAPH_REUSE` | `1`: one recorded filter graph per block width and degree, kept from plan to plan | `0`: one per block of a plan | `MULTI_GPU.md` |
| `PARSEC_SECTOR_STENCIL` | `direct`: sector stencils from the grid | `csr`: reduced from the full-grid matrix | above |
| `PARSEC_OVERLAP_IONIC_SETUP` | `1`: the GPU ionic fields on a thread beside symmetry and sector set-up | `0`: in line; fields in line are summed on every grid point and are the former bits, where the thread sums a symmetric structure on its wedge unless `PARSEC_IONIC_SUM_GRID=full` (below) | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_OVERLAP_CUDA_CONTEXTS` | `1`: `benchmarks/mpi_full_scf.py` creates the CUDA contexts of a rank on a thread beside its preparation | `0`: before it | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_SYMMETRY_FAST_MAPS` | `1`: symmetry detection and phases without passes over the grid coordinates | `0` | `GPU_SETUP_OPTIMIZATION.md` |

What was compared on A100 nodes, in first runs with these routes together
(strict parity passed, the filter in FP64 on both sides): all of them at
their defaults against the former route of each gave the same total energy
to the last bit, and the same differences of eigenvalues, density,
potentials and SCF history from the serial reference, for 3,480, 7,120,
10,456, 14,680 and 19,392 electrons on 4 GPUs, where no sector basis is
shared. So did each of these switched back alone, for 10,456 electrons on 4
GPUs and 23,768 on 16: `PARSEC_CUPY_FILTER_GRAPH_REUSE=0`,
`PARSEC_SECTOR_STENCIL=csr` (under `PARSEC_NATIVE_SECTOR_ASSEMBLY=1`),
`PARSEC_IONIC_GPU_COUNT=1`, and the three switches of the preparation
together; `PARSEC_CUPY_IMPLICIT_TILE=0` for 10,456 and 19,392 electrons on
4. With a shared basis the result follows its Ritz step (round-off between
the block counts, see above): 23,768 electrons on 16 GPUs with two blocks
and full slabs named beside all these defaults gave the energy of the
former routes to the last bit. Not run apart: the three switches of the
preparation from each other, and `PARSEC_CUPY_IMPLICIT_PACK_WORKERS` from
the tiles; the tests compare what they build. The FP32 filter
(`PARSEC_CUPY_MIXED_FILTER=auto`) is another precision, not round-off, and
was not run in these comparisons. `PARSEC_SECTOR_STENCIL=csr` without
`PARSEC_NATIVE_SECTOR_ASSEMBLY=1` differs by round-off, which was measured
on small test cases only (above).
The serial control of the runner keeps the former route of each except the
FP32 filter. A symmetry cache is outside that: an entry that another run
wrote gives the control the maps and stencils of that run, and its log says
so; a control that names no cache, the default, stays independent. The
backend report says
what ran: `orbital_sector_later_filter_precision`,
`orbital_sector_finite_difference_storage` with
`orbital_sector_tile_pack_workers`, `ionic_gpu_count_requested` with
`ionic_gpu_devices`, `orbital_sector_filter_graphs`,
`orbital_operator_stencil_builder`, `ionic_setup` and `symmetry_fast_maps`;
the runner records `cuda_context_initialization_overlapped` for every rank,
and `cuda_context_driver_devices` with `cuda_context_driver_seconds` for the
contexts that its thread created through the driver library. That creation,
the rule of the state storage and the download of a spill came after these
runs: they are in the last table of switches, below. The measured
runs named `device` and `pool`, the route that rule now takes by itself where
the sectors fit; `orbital_sector_state_storage` and `orbital_memory_allocator`
say what a run had.
`orbital_sector_filter_graphs` is the switch while a calculation is
prepared. In a result it is what the sector filters of the reporting
process recorded: `none recorded` unless `PARSEC_CUPY_FILTER_GRAPHS` or
`PARSEC_CUPY_DISTRIBUTED_FILTER` is set or a sector basis is shared.
`scripts/run_multi_gpu_scf.sh` and every measured run set the first, which
is off when unset. The text report adds a line where its detailed setup
named another route, and the runner counts the captures of every rank
(`graph_captures`).
`PARSEC_CUPY_FILTER_GRAPH_REUSE` and `PARSEC_SECTOR_STENCIL` are read when the
preparation starts, so a value they do not know stops the run there.

A further set of switches, in one table: each with its default, the
setting that selects the former route, what it does to the results and the
entry that says which one ran. The first eight keep the results, bit for bit
or at round-off. Each of them has run on A100 nodes in the code that
introduced it, switched on and off in one allocation, and all eight together
in the code as it was before the Hartree boundary was brought in:
first runs of 3,480 to 29,576 electrons on 1 to 16 GPUs, every item at its
default against every item on its former route, and each of the 59 runs that
have a reference passed strict parity. The two sides gave the same total
energy to the last bit on 1, 2, 4 and 8 GPUs and, up to 10,456 electrons, on
16; from 14,680 electrons on 16 GPUs the slabs of 2 GiB change the slab count
of the Ritz step and the totals differ by 1.3e-8 to 6.0e-8 Ry. (A sector on
two devices has cut slabs of at most 2 GiB since: on 8 GPUs the two sides
now differ at round-off as well, from 14,680 electrons.) For 23,768
electrons on 16 GPUs the defaults took 105.5 s and 30,421 MiB on the fullest
device where the former routes took 118.2 s and 34,013 MiB; for 29,576
electrons 147.3 s and 38,453 MiB against 162.7 s and 42,257 MiB. The switches
of the Hartree boundary change results by design where its plan engages. They
had not run on an A100 when these runs were made, nor had the code with
them; both have run there since (`result.hartree_boundary` in the
`timing.json` of a run says what it had).

| Environment variable | Default | Former route | Results | Says what ran | Described in |
|---|---|---|---|---|---|
| `PARSEC_CUPY_STREAMING_RITZ_BYTES` | unset: slabs of at most 2 GiB in the one-block Ritz step of a sector on four or more devices (16 GPUs) | `4294967296`, for 16 GPUs only; the rounds a run had take `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it | round-off | `slab_columns` of a shared sector in `timing.json` | above |
| `PARSEC_CUPY_RITZ_CONDITION_HELPER` | `auto`, as when these runs were made (`any` was the default for a while, in the table with `PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` below): another device of a shared sector estimates the condition number of the overlap beside the small solve of its owner, unless it holds the Hartree objects | `off` | same bits | `condition_device` and `condition_seconds` of a shared sector in `timing.json` | its row in the first table above |
| `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` | unset since the rule in filter work (the table of two defaults below); a value is the row count, over the square root of the sectors of the device, from which a sector alone on its device packs affine tiles: `650000` when these runs were made | `1100000`, `1555634` or `2200000` on 4, 2 or 1 GPUs | same bits | `orbital_sector_finite_difference_storage` | the row of `PARSEC_CUPY_IMPLICIT_TILE` above |
| `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` | unset since the rule in filter work (the table of two defaults below); a value is the row count from which a sector on several devices (8 and 16 GPUs) packs them: `1100000` when these runs were made | `PARSEC_CUPY_IMPLICIT_TILE=0` | same bits | `stencil_storage` of a shared sector in `timing.json` | its row above |
| `PARSEC_CUPY_SECTOR_POOL_RELEASE` | `1`: a finished sector returns the unused pool blocks of its stream where a device solves several sectors in turn (2 GPUs) | `0` | same bits | `sector_pool_release` of a rank in `timing.json` | `MULTI_GPU.md` |
| `PARSEC_CUPY_DENSITY_COLLECTION` | `changed`: the young generations are collected before the pool release after a density while the bytes in use stay | `full` | same bits | `density_pool_release` of a rank in `timing.json` | `MULTI_GPU.md` |
| `PARSEC_SYMMETRY_SCF_BUFFERS` | `1`: the wedge sums and the Anderson step of the root in arrays it keeps | `0` | same bits | `scf_scalar_field_buffers` | below |
| `PARSEC_FAST_GRID` | `1`: the cluster grid slab by slab | `0` | same bits | `grid_builder` | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_HARTREE_BOUNDARY` | `auto`: the Hartree boundary values follow the plan made from the geometry (estimate of the omitted potential, order raised from `Solver_Lpole` until the estimate meets the tolerance, atomic tail; see `Hartree/GPU_HARTREE.md`). This changes results of clusters from about 70 atoms up | `legacy`: PARSEC's multipole expansion at `Solver_Lpole`, bitwise the former boundary | changed by design where the plan engages; `legacy` gives the former bits | `hartree_boundary`, `hartree_multipole_order`, `hartree_boundary_tolerance`, `hartree_boundary_estimate`, `hartree_atomic_tail`, `hartree_boundary_plan_seconds`; `result.hartree_boundary` in `timing.json` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_TOLERANCE`, `PARSEC_HARTREE_ATOMIC_TAIL` | unset: the input keywords `Hartree_Boundary_Tolerance` (`1e-3 Ry`) and `Hartree_Atomic_Tail` (`auto`) | `off` and `off` | changed by design | `hartree_boundary_tolerance`; `hartree_atomic_tail` with `hartree_atomic_tail_max` and `hartree_atomic_tail_seconds` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_LPOLE` | unset: the input keyword `Solver_Lpole` (0 to 60, default 9; orders above 9 need native extension 0.6.0 or newer) |  | changed by design | `hartree_multipole_order` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_VARIATIONAL` | `off` (default since 2026-11-22): the Hartree potential of the Poisson solve, as PARSEC; where density reaches the sphere the forces of a cluster are not the exact slope of its energy (Au2 in a sphere of 8 bohr, `Solver_Lpole: 4` raised to 6 with the tail: 1.3 to 4.6e-5 Ry/bohr at a step of 1e-3 bohr, 1.6e-4 with an atom off the grid; H2O at the radius of the default rule 1e-9), nor those of a slab whose density reaches the edge of its domain, a gap that grows as the atoms approach the edge (Ge2 slab, h 0.45, half-height 6 bohr: atoms at z = +1.2/-1.1, density 1.1e-4 at the edge, F_z 1.05e-5 Ry/bohr from the slope at both steps; atoms at z = +2.1/-2.0, F_z 5.5e-5 / 5.8e-5 for atoms 0 / 1 and x 6.4e-6; the first geometry at a half-height of 8 bohr 5e-8) | `on` (opt-in): a cluster with a multipole boundary (a sphere without `Full_Hartree`) takes as its Hartree potential the derivative of its Hartree energy, `V' = (V_H + 8 pi u - M^T A_IB^T u)/2 + (u.t)/(2 sum rho)` with `u = A^-1 rho` solved by the CG of its route (host or GPU, full grid or wedge; the solid-harmonic sums and the sums of the tail on the device of a GPU CG), and its forces add the derivative of the atomic tail. A slab (`Boundary_Conditions: slab`; its forces run on the Fortran-derived routines) whose forces are computed takes, under `on` and under `auto` (there without an estimate; since 2026-11-22), `V' = (V_H + 8 pi u - M^T A_IB^T u)/2` with `M` the sheet sums of its boundary values (`Hartree/slab_fortran.py`; no tail), and its force of the Gaussian charges takes `G^T rho = 2 V' - V_H` in place of `V_H` (`ConfinedVariationalHartree`, `Forces/wire_slab.py`); a wire keeps the Poisson solution. A100, force against the slope of the SCF energy at 1e-3 / 2e-3 bohr: Ge2 slab at a half-height of 6 bohr F_z of atom 0 1.05e-5 / 1.02e-5 -> 7.2e-8 / 3.9e-7 Ry/bohr, of atom 1 9.1e-6 / 8.7e-6 -> 7.2e-8 / 4.0e-7, atom 0 on a grid point 1.02e-5 -> 1.1e-7, atoms at z = +2.1/-2.0 F_z 5.5e-5 / 5.8e-5 -> 4.8e-7 / 2.0e-6 (Richardson 1.8e-8 / 6e-9) and x 6.4e-6 -> 8.1e-8, Si2 slab 1.03e-5 -> 9.9e-8, its atom 1 8.9e-6 -> 1.8e-7; at a half-height of 8 bohr (atoms at z = +1.2/-1.1) the forces move by 1.5e-8 to 9e-8. The Si2 numbers take the default `PARSEC_PROJECTOR_TAIL=auto`: the Si table of the tests holds 3.8e-6 Ry beyond its channel cutoff, and with that tail (`off`) the slab gave 1.11e-5 -> 8.8e-7 (on a grid point 9.8e-6 -> 4.4e-8) while F_z of its atom 1 kept about 6.5e-6 Ry/bohr after Richardson extrapolation under either value at a half-height of 8 bohr (1.5e-7 under `on` and 6.3e-8 under `off` with the tail removed); one more CG per Hartree solve on the host (Ge2 slab 94 solves, 0.76 s on the CPU; A100 walls 9.7 -> 10.5 s, Si2 6.1 -> 7.0 s). `auto` (opt-in): the same where the forces are computed (a relaxation, molecular dynamics, manual movements without `Skip_Force: true`, `Calculate_Force`; `SinglePointInput.forces_wanted`) and the first solve of a structure estimates a force change of 3e-7 Ry/bohr or more (decided again where an atom moved outward by more than one grid spacing; never with `Hartree_Boundary_Tolerance: off`). The forces become the exact slope of the energy of the multipole boundary (Au2: 1.1e-7 to 9.9e-7 Ry/bohr at 1e-3 bohr, 4e-8 after Richardson extrapolation; the breathing mode of Ge5 in 9 bohr 6.6e-5 -> 1.6e-8), which is what a finite-difference check and the stopping test of a relaxation compare with, but not closer to the forces of the boundary summed directly: the Poisson forces lie closer to those on every cluster compared (Au2 in 8 bohr 3.1e-5 against 1.6e-4 Ry/bohr, Ge2H2 4.9e-6 against 2.0e-5), hence off by default. Cost on A100, first runs, `off` -> `auto`: C795H300 forces on four GPUs 14.2 -> 15.2 s, C185H124 9.7 -> 11.0 s and three structures of its relaxation 14.0 -> 17.6 s, Au2 3.4 -> 3.7 s; a run that `auto` leaves with the Poisson solution pays 0.4 to 1.0 s per decision. Results keep `V_H` (the archive, cubes, `pot.dat`). A restart from a run that kept the Poisson solution takes a few more SCF iterations under `auto` or `on`. A value it does not know is refused before the SCF | changed by design in the runs that take it: the energy functional is the same and the energy moves at second order (Au2 -8.3e-7 Ry, Ge2 slab -1.3e-8 Ry), the forces become its slope (Au2 up to 1.6e-4 Ry/bohr, Ge2 slab 1.05e-5) | `hartree_variational` (the route and where the sums run), `hartree_variational_solves` (estimates, decisions, solves and CG steps of `u`, seconds); `forces_hartree_boundary` in the archive and a table in `parsec.out`; a slab: the line `Hartree potential of the SCF: (V_H + G^T rho)/2` under the force tables of `parsec.out` (solves and CG steps of `u`, seconds) | `Hartree/variational.py`, `FORCES_AND_RELAXATION.md` ("Forces at a fixed structure") |
| `PARSEC_HARTREE_BOUNDARY_KERNEL` | `auto`: the GPU boundary of an engaged plan is evaluated once per unique exterior point, on the wedge of an axis-reflection group or, without one, on the full grid (round-off change against the full-grid kernels; see `Hartree/GPU_HARTREE.md`). On the laptop GPU the raised order costs less than the former boundary on the D2 wedge and, without such a group, up to about twice as much at order 34 | `full`: the full-grid kernels (the serial control of the MPI runner takes them) | round-off | `hartree_boundary_kernel`, `hartree_boundary_device_bytes` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_LINEAR_BACKEND` | unset or `auto` (since 2026-11-07): the conjugate gradients of the cluster Hartree solve, and the pair solves of `Energy_Refinement`, on the GPU of the orbitals wherever CuPy orbitals have the native Hartree components; the CG of the selected backend elsewhere (CPU runs, `--backend scipy` or `cupy`, periodic cells). The host CG, on the same wedge, also where the GPU CG cannot be made (out-of-memory, CUDA runtime or driver errors) and, beside a full-grid eigensolver, where the GPU Hartree arrays would take more than 1/20 of the free device memory; with symmetry sectors the arrays are counted by the memory rules of the sectors instead (`Hartree/GPU_HARTREE.md`). `cupy`: the GPU CG by name (refused where it does not apply) | `native`: the C++/OpenMP CG on the host | within the CG tolerance: totals of 109 example inputs within 1.3e-11 Ry, forces within 1.5e-11 Ry/bohr (1e-9 for CA CF4), C459H204 within 1e-9 Ry; the boundary of a full grid follows the CG onto the GPU (`PARSEC_HARTREE_BOUNDARY_BACKEND`) | `hartree_linear_backend`, `hartree_backend`, `hartree_cg_device`, `hartree_cg_device_bytes` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_BACKEND` | unset: the native builders; where they have no orbit table (no reduction, or the table of the raised order above 512 MiB) the GPU kernels if the orbital backend is CuPy and the plan is engaged or, since 2026-11-07, the Hartree CG runs on the GPU (`PARSEC_HARTREE_LINEAR_BACKEND`); the wedge kernels on the GPU for the wedge of the resident chain (`PARSEC_CUPY_RESIDENT_HARTREE`). `cupy`: the GPU boundary. `auto`: see `GPU_SFC_OPTIMIZATION.md` | `native`: the native builders whatever they cost | round-off where the GPU kernels take the boundary | `hartree_boundary_kernel`, `hartree_backend` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_CUPY_RESIDENT_HARTREE` | unset or `auto` (since 2026-11-07): where the GPU CG solves the symmetry wedge of the SCF fields of a sector run, the boundary of that wedge on the same GPU (the wedge kernels on an axis-reflection group, whether the plan is engaged or not) and every solve in the device-resident chain (`Hartree/cupy_resident.py`); the route before where the GPU boundary or the chain cannot be built or the GPU CG gives way, and from a solve whose device allocation fails on; also in MPI runs (the root rank). `1`: the chain by name (refused where it does not apply) | `0`: the host interface and the orbit table of the wedge on the host, bit for bit the code before (also `PARSEC_HARTREE_BOUNDARY_BACKEND=native`) | round-off: against `0`, on A100, the totals of the 85 example inputs that take the chain within 2.6e-11 Ry, exact exchange within 1.4e-10 Ry, occupied levels within 2.3e-10 Ry, forces within 1.4e-11 Ry/bohr, relaxation paths within 1.1e-12, the same SCF steps; the 30 that do not take it unchanged; 4 to 35 % less program time on one GPU, peaks within 0.1 GiB | `hartree_resident_chain`, `hartree_full_grid_transfer_policy`, `hartree_boundary_kernel`, `hartree_resident_device_bytes` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_CUPY_BLOCH_PAIR_SYMMETRY` | `on` (since 2026-11-07, new with the device exchange of periodic hybrids with k-points): each pair of two occupied states solved once for both (`Hamiltonian/bloch_exchange_device.py`) | `off`: every pair one way, the host loop's order | round-off: device and host engines within 1.3e-15 Ry (H2 box and slab, PBE0, HSE06, B3LYP spin) with either value; 20 to 46 % fewer pair solves | `pair_solves` of the Fock-loop lines of `parsec.out` | `Hybrid functionals of periodic cells on the device` below |
| `PARSEC_HARTREE_ATOMIC_TAIL_VALUES` | `auto`: the values of the atomic tail at the exterior points come from an FP64 device kernel where the orbital backend is CuPy and from host threads elsewhere | `host`: host threads on every backend (the serial control of the MPI runner takes them); there is no former route, the tail is new | round-off | `hartree_atomic_tail_values` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_CHECK` | `0`: off. `n` compares, after the SCF, the boundary values of the final density at a sample of `n` unique exterior points (the innermost, those of the largest atomic tail, those nearest to an atom, random ones) with its direct Coulomb sum on the Hartree device and reports `hartree_boundary_check_*`; the maximum is that of the sample, and `n` at or above the number of points checks all (GPU boundary only; no effect on the results; its seconds are kept out of the SCF wall time) |  | none | `hartree_boundary_check_*` | `Hartree/GPU_HARTREE.md` |

Two defaults came after every table of switches here, the last one below
included, and stand before it because neither has run on A100 nodes as it
is: their last column says what did, the stencils that the rule of the tiles
now chooses, by row limits lowered by hand, and the block after the SCF
without its thread. With the former values of both rows a calculation is
that of the code before them bit for bit. So it was on a laptop GPU, every
array of the archive and `parsec.out` apart from its times: 32 and 176
electrons through the command line and through the MPI runner, as a run
and as a serial control, and 176 electrons with every sector basis shared,
that one GPU named two and four times. The defaults gave the same there:
the rule chooses no tiles for sectors of so few states, and the thread
found no estimate to make. For 864 electrons with six first cycles of
degree 40 the rule gives every sector its tiles on that one device, and
the calculation was that of the slot-major stencil of the code before, bit
for bit.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 nodes |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` with `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` | both unset: `PARSEC_CUPY_IMPLICIT_TILE=auto` packs affine tiles for a sector whose filter work repays the packing, whether one device filters it or several share it (the rule above; `GPU_COMPACT_OPTIMIZATION.md`) | `650000` and `1100000`: the row limits of a sector on one device and of a shared one | same bits | `orbital_sector_finite_difference_storage`; `stencil_storage` of a shared sector in `timing.json` | not named: it names `PARSEC_CUPY_IMPLICIT_TILE=0` | limits of 500,000 and 600,000 rows against 650,000 and 1,100,000 in one allocation, first runs, two of each: the filter of a sector took 2.47 against 2.66 s for 3,480 electrons on 8 GPUs and 1.31 against 1.40 on 16, 5.32 against 5.83 and 2.71 against 2.97 for 5,264, 7.65 against 8.38 and 3.92 against 4.29 for 7,120 in the sphere of the default rule; on 4 GPUs the filters of 3,480 electrons in that sphere took 17.7 against 18.9 thread-seconds. The set-up of the sector operators of a rank grew by 0.17 s where it packed four stencils, by 0.09 to 0.15 s for two of 1.0 to 1.1 million rows, by 0.04 s for one of them and by 0.03 s at most at 677,542 rows. The wall time before the SCF grew by more than that timer shows: by 0.11 s on 4 GPUs and by 0.18, 0.25 and 0.26 s on 8, where the filter of a sector saved 0.19, 0.51 and 0.73 s. The time before the SCF and the SCF together fell by 0.17 s on 4 GPUs and by 0.1, 0.2 and 0.5 s on 8: next to nothing for 3,480 electrons on 8 GPUs, a gain for the others. On 16 GPUs the program times cannot be read: the first step of five of six slot-major runs took 0.8 to 10 s longer than in the earlier series and that of no run with tiles, for a reason that was not found; the tiles are judged there by the filter seconds and the bits. The fullest device peaked 52 to 98 MiB lower on 8 and 16 GPUs. The same total energies to the last bit. The rule gives these sectors their tiles and keeps those of the larger ones; it has not run there itself |
| `PARSEC_DOMAIN_REPORT_THREAD` | on: on the CuPy backend a host thread makes, while the SCF runs, the estimates of the omitted potential that the radius search of the block "Density at the sphere" can ask for, and the block reads them (below) | `0`: the search makes them after the SCF | same bits: every line of the block and every entry of `result.domain.after_scf` that is no time | `after_scf.radius_seconds` beside `estimates`; `post_scf_finalization_reporting_seconds`; `domain_report_thread` of the root in `timing.json` (the estimates the thread made, its seconds, a failure; null without a thread) | not named: a report, no route of the solver | not run with the thread. Without it the search took 1.21 s of the 1.62 s after the SCF of 39,368 electrons on 16 GPUs (five estimates) and 0.17 and 0.20 s for 23,768 and 29,576 (one each) |

The switches below were the last added before those two, the three rows from
`PARSEC_CUPY_RITZ_GRAM_SLABS` on after the ten before them: what follows is
about those ten, and the three have a paragraph behind the table. Each of
the ten ran on A100 nodes against its
own former value in one allocation (first runs, FP64; each item in the code
that introduced it, the rows of `PARSEC_CUPY_RITZ_GRAM_MULTIPLE` and
`PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE` together),
which is the last column. With every one at its former value, and an input
that gives its `Boundary_Sphere_Radius`, a calculation is that of the code
before these switches bit for bit (with the three rows after them at their
former values as well, since those came). So it was on a laptop GPU through the MPI
runner, every array of the archive and `parsec.out` apart from its times:
for 32 and 176 electrons, as a run and as a serial control, with the states
on the device and spilled (for a serial control apart from the line that
lists its routes, which names the settings of the column "Serial control"
below beside the 19 it had); for 864 electrons with a budget of 70 columns
per slab, which the multiple cuts to 64 (the defaults then give another
total energy, by 9e-11 Ry); and for 176 and 864 electrons with every sector
basis shared, that one GPU named two and four times as the devices of a
sector, where every copy stays inside the device. On A100 nodes each switch
went back by itself, and all nine together in one code (first runs,
the defaults beside them in the same allocation): 3,480
electrons on 1 and 16 GPUs, 14,680 on 4, 23,768 and 29,576 on 8 and 23,768
and 39,368 on 16 gave, to the last bit, the total energies of the series of
first runs made before these switches. The
defaults gave, to the last bit, the totals of the runs of those two rows
(3,480 on 1 GPU, 14,680 on 4 and 8, 19,392, 23,768 and 29,576 on 8, 23,768
and 39,368 on 16) and on 8 and 16 GPUs their peaks, and took 267.9 against
278.8 s of SCF for 39,368 electrons on 16 GPUs, 93.3 against 100.0 s for
23,768, 241.6 against 251.1 and 166.5 against 173.7 s for 29,576 and 23,768
on 8, and 117.6 against 123.5 s for 14,680 on 4.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 nodes, default against former |
|---|---|---|---|---|---|---|
| the input line `Boundary_Sphere_Radius` | absent or `auto`: the default rule of the parser gives the sphere its radius and, where the input has no such line, its `Hartree_Boundary_Tolerance`, from `Domain_Energy_Tolerance` (`1e-3 Ry`) | the line with a length; the two lines that `parsec.out` prints give the calculation of the rule again | an input with a radius: same bits; one without was refused | `result.domain` and `per_rank[].domain_rule_seconds` in `timing.json`, the `domain_*` details | no entry: the sphere is the input's | 3,480 to 39,368 electrons on 1, 4 and 16 GPUs got the radius, tolerance and order derived for them and, to the last bit, the energies of the runs that had named these by hand; ten systems without hydrogen at the surface got 4.6 to 8.2 ang; the rule took 0.03 to 1.27 s (15.6 s once, the first run on its node) |
| `PARSEC_DOMAIN_REPORT` | on: an input with a radius also gets the estimates of its sphere at set-up and the block "Density at the sphere" after the SCF | `0` | same bits | `result.domain` (null with `0` for an input with a radius) and its `after_scf.seconds` | not named: a report, no route of the solver | inputs with a radius kept the bits of the runs made before the rule (3,480 electrons on 4 GPUs with the report on and off, 23,768 and 39,368 on 16 with it on); the block took 0.04, 0.30 and 1.40 s with 0, 1 and 5 estimates of the omitted potential |
| `PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES` | `2147483648`: slabs of at most 2 GiB in the one-block Ritz step of a sector on two devices (8 GPUs), where `PARSEC_CUPY_STREAMING_RITZ_BYTES` is not set | `4294967296`: the slabs a run had; its rounds and bits take `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it, and the count of the sharing rule of then `PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=1073741824` (128 MiB more without it, last table below) | round-off where the rounds of a sector change: bases above 20 GiB and some of 16 to 20 GiB, on 8 GPUs from 14,680 electrons; same bits elsewhere | `slab_columns` of a shared sector in `timing.json` | not named: it shares no basis | the fullest device peaked at 25,373, 33,643, 50,859 and 67,031 MiB for 14,680, 19,392, 23,768 and 29,576 electrons on 8 GPUs, 3.2 to 3.8 GiB below the former slabs; SCF 175.2 against 172.9 s for 23,768 electrons; the former value gave the former bits and peak (54,795 MiB), in the code that introduced the item, where no round was cut at a multiple |
| `PARSEC_CUPY_DENSITY_RELEASE` | `threads`: after a density the pool of every device that a process empties is emptied by the thread of that device, and the pinned pool meanwhile; an MPI rank waits for them behind the collective calls of its density command. Every such pool is empty before the process goes on, as before. The devices of a shared basis are left out since (last row): on 8 and 16 GPUs nothing is emptied by default | `serial`: one device after another in the thread that built the density | same bits | `release`, `release_seconds` and `device_release_seconds` under `density_pool_release` of a rank | `serial` | density stage of a run 2.70 against 2.94 s (29,576 electrons, 8 GPUs), 2.12 against 2.37 (14,680, 4), 1.78 against 2.11 (23,768, 16), 1.43 against 1.78 (14,680, 16) and 0.58 against 0.62 (5,264, 8); the device threads of the root together took 1.2 to 3.7 times the seconds of its `serial` release, so the devices slow each other down and less is gained than the longest device in place of the sum; the four pairs on 8 and 16 GPUs emptied the pools of shared sectors, as the code did then |
| `PARSEC_CUPY_EXCHANGE_PAIRS` | `1`: the four devices of a shared basis (16 GPUs) copy a chunk that all of them send to their rows in three turns of two pairs | `0`: all twelve copies at once | same bits | `to_rows_copies` of a shared sector | not named: it shares no basis | the way to the rows of a sector took 6.83 against 8.88 s for 23,768 electrons and 16.56 against 21.72 s for 39,368, the SCF 97.6 against 99.6 and 274.0 against 279.6 s; 1.93 to 2.00 against 2.41 to 2.49 s for 10,456, and 0.52 to 0.54 against 0.48 to 0.54 s for 3,480; all with the rounds of equal slabs, before those of whole multiples |
| `PARSEC_CUDA_CONTEXT_CREATION` | `driver`: the context thread of `benchmarks/mpi_full_scf.py` creates the contexts through the CUDA driver library, outside Python's interpreter lock | `runtime`: inside CuPy's first calls on each device, which hold the lock | same bits | `cuda_context_driver_devices` and `cuda_context_driver_seconds` of a rank | `runtime` | preparation of the root for 3,480 electrons 1.62 to 1.77 against 2.57 to 2.70 s on 16 GPUs, 1.76 to 1.79 against 2.39 to 2.51 on 8, 2.83 to 3.03 against 3.70 to 4.44 on 4 and 2.65 to 2.74 against 3.22 to 3.26 on 1; 5.48 against 6.33 s for 23,768 on 16; every device of every rank listed, no capture repeated |
| `PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION` | unset: `auto` of `PARSEC_CUPY_SECTOR_STATE_STORAGE` and of `PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR` keeps the sector states on their devices and the pool unless the count of a device exceeds `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION` of it (0.978 then, 0.974 since its own row below; `1` for all of it) | `0.5`: the host spill and the direct allocator from half of a device's memory in sector vectors | same bits: it chooses the storage and the allocator | `orbital_sector_state_storage`, `orbital_memory_allocator` and `sector_state_fit_bytes` of a rank | `0.5`: its Ritz step takes two arrays per sector, and the count is that of one | 19,392 electrons with neither value named had the bits and the peak of the run with both named, on 4 GPUs in 183.2 against 183.7 s where the former rule had stopped, on 8 and 16 likewise; 14,680 on 2 GPUs took 242.5 s and 80,157 MiB; 10,456 on one GPU ended by the spill in 770.6 s with 24,465 MiB, and stopped for memory with the share at `1` |
| `PARSEC_CUPY_SPILL_DOWNLOAD_ORDER` | `A`: a sector basis that is spilled to the host is downloaded in the order it has, without a second array on its device | `C`: in C order, by a C-ordered copy on the device and a copy on the host, the former download only with `PARSEC_CUPY_SPILL_BUFFERS=fresh` beside it since: the host buffers of a spill, the default of that switch, take a Fortran-ordered basis by one copy whatever this says, and read it only for vectors of another layout | same bits | the variable under `environment` of a rank | `C` | 7,120 electrons spilled on one GPU: 254.1 s and a peak of 11,085 MiB against 458.9 s and 18,175 MiB, the same total energy |
| `PARSEC_CUPY_RITZ_GRAM_MULTIPLE` | `64`: the slabs of the two Gram matrices of the Ritz step, and the rounds of a shared basis, are cut at whole multiples of 64 columns, on every route (1 to 16 GPUs); a slab only narrows | `1` | round-off wherever a slab or a round changes: the overlap of every sector from 512 columns, the projection of one array from 64 columns of budget, and the rounds of a shared basis from 32 columns per slab on two devices and 16 on four, which is every shared sector of the measured series | `slab_cut` (`multiple`), `slab_columns` and `projection_columns` of a shared sector in `timing.json`; the stage `subspace_ritz_projection_seconds`, and `gram` with the `overlap_seconds` in it | `1`: its projection is formed in slabs; its overlap is DSYRK's (`PARSEC_CUPY_RITZ_SYRK=on`) and has none | SCF of a run 166.9 against 173.6 s for 23,768 electrons on 8 GPUs, 242.1 against 246.2 for 29,576, 98.6 against 101.3 for 19,392 and 69.5 against 70.5 for 14,680; on 16 GPUs 95.4 against 96.9 (23,768), 136.0 against 138.1 (29,576), 275.3 against 275.8 (39,368), 41.9 against 42.0 (14,680) and 22.7 against 23.0 (10,456); 118.4 against 121.6 for 14,680 on 4, 112.4 against 114.9 for 10,456 on 2 and 60.1 against 60.9 on 4; the fullest device of a shared sector 0.2 to 1.2 GiB lower (62,709 against 63,913 MiB for 39,368 electrons) and the same for 29,576 on 8 GPUs, whose slabs stay (67,083 against 67,079); total energies within 6e-8 Ry, 1.2e-7 for 39,368. Per sector the overlap took 0.1 to 4.2 s less and the products of the rounds 0.05 to 6.4 s less; the stages around the rounds took 5.0 s more for 39,368 electrons on 16 GPUs, whose rounds lost 1.9 s. Three totals were those of the first run on a node (64.3 against 56.5 s for 5,264 electrons and 95.7 against 86.9 for 7,120 on one GPU, 180.1 against 176.5 for 19,392 on 4) while their projection stage fell (3.35 against 5.17 s, 7.03 against 7.36, 112.4 against 127.4 thread-seconds) |
| `PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE` | `0`: after a density nothing is collected or returned for a sector whose basis several devices share (8 and 16 GPUs): its next step takes the same blocks again; its group returns a slab workspace that a pass has outgrown before it takes the larger one. The pinned pool then keeps its blocks too: 0.3 to 1.1 GiB of host memory more on the root rank, measured with the release off. A sector on one device empties its pool as before | `1`: the pools of the devices of a shared basis are emptied after every density | same bits | `shared`, `shared_kept` and `calls` under `density_pool_release` of a rank in `timing.json` | not named: it shares no basis | density stage of a run 0.88 against 1.56 s (14,680 electrons, 8 GPUs), 1.85 against 2.64 (23,768, 8), 1.68 against 2.70 (29,576, 8), 0.80 against 1.23 (14,680, 16), 1.01 against 1.67 (23,768, 16) and 0.99 against 1.89 (29,576, 16); SCF 0.6 to 2.1 s less; the same total energy to the last bit; the fullest device 2 MiB higher in four pairs and 44 and 48 MiB higher for 23,768 and 29,576 electrons on 8 GPUs (50,677 and 67,083 MiB); `calls` 0 and `shared_kept` as many as the densities on every rank; the root rank 0.7 to 1.1 GiB higher at its high-water mark of host memory, the other ranks the same to 0.01 GiB |
| `PARSEC_CUPY_RITZ_GRAM_SLABS` | unset: the overlap of a sector basis that several devices share (8 and 16 GPUs with four sectors, also 2 and 4 with fewer sectors than devices) in slabs of 128 columns (`PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS`); eight slabs, as before, for the overlap of a sector on one device and for a projection that is formed from `H X` as an array | `8`: eight slabs on every route | round-off for a shared sector below 1,017 columns and from 1,529; same bits between them, and wherever no basis is shared (1, 2 and 4 GPUs with four sectors) | `overlap_columns` of a shared sector in `timing.json`, beside its `overlap_seconds` | not named: it shares no basis, and its projection keeps the count | the count named so that the slabs are 128 columns wide, against the eight slabs in one allocation: overlap of a sector 26.79 against 31.46 s for 39,368 electrons on 16 GPUs (27.81 s in slabs of 256), 11.04 against 12.18 for 29,576 and 6.57 against 7.14 for 23,768; on 8 GPUs 22.53 against 25.83 for 29,576 and 13.59 against 14.97 for 23,768; total energies 2e-8 to 1.6e-7 Ry apart. The rule cuts those slabs; it has not run there itself, nor where it widens the slabs (below 1,017 columns) or narrows them from 192 and 256 |
| `PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES` | unset: the buffer of a sector whose basis no group of devices shares (1, 2 and 4 GPUs with four sectors), an eighth of its basis and at least 1 GiB, takes at most 2 GiB, or 64 columns of all rows where 2 GiB hold fewer (more than 4,194,304 rows), up to 4 GiB, a guard derived from the slabs of 64 columns and not measured, which no case of the measured series reaches; a budget that is set (`PARSEC_CUPY_STREAMING_RITZ_BYTES`) is the buffer as before, and no step of a shared basis reads the limit | `4294967296`: at most 4 GiB | round-off for a basis above 16 GiB, whose projection is cut into other slabs or whose rows are rotated in lower tiles; same bits up to 16 GiB | the variable under `environment` of a rank; `sector_state_fit_bytes` holds the buffer where `auto` asked | `4294967296` | the budget set to 2 GiB against the default of 4 in one allocation: 14,680 electrons on 2 GPUs 78,109 against 80,157 MiB on the fullest device in 231.4 against 231.6 s, on 4 GPUs 41,049 against 43,141 MiB in 123.9 against 124.5 s; 19,392 electrons on 4 GPUs 58,155 against 60,251 MiB in 180.7 against 180.5 s; total energies 5e-9 to 5e-8 Ry apart. The limit gives those sectors that budget; it has not run there itself |
| `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION` | `0.974`: the share of a device that the vectors of its sectors and one buffer may take before `auto` spills the states (`MULTI_GPU.md`): the reciprocal of 1.027, the least ratio of sampled peak to count with the buffer of 2 GiB | `0.978`, beside the former limit of the buffer | same bits: it chooses the storage and the allocator | `orbital_sector_state_storage`, `orbital_memory_allocator` and `sector_state_fit_bytes` of a rank | not named: it decides by the rule before (`PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=0.5`) | 10,456 electrons on one GPU count 0.976 of the device with the buffer of 2 GiB. With the storage left to the code, `0.978` kept their states on the device and the first solve stopped for memory (an allocation of 2,147,475,456 bytes with 81,874,197,504 allocated); `0.974` spilled them and the run completed (first runs). Every other layout of the measured series keeps the storage it had |
The first row is described in the reference `README.md` ("Sphere radius left
to the code") and below, the slabs, the turns and the multiple above, the
two releases, the rule of the state storage and the download in
`MULTI_GPU.md`, the creation of the contexts in `GPU_SETUP_OPTIMIZATION.md`.

Some rows meet. The rule of the state storage counts a shared basis by the
blocks of the rule that shares it, so a sector on two devices is counted
with its two slabs of 2 GiB: 62.0 instead of 66.0 GiB for 29,576 electrons
on 8 GPUs, and the former count with the former limit named. Neither
decision changes by it: both counts lie below the share of a device that the
sharing rule allows. The multiple does not move that count, nor the count of
a sector on one device: both hold the slabs of their budget, which the
multiple only narrows. The buffer of one array is then less than 8 bytes per
column below its count, and the workspace of a shared sector up to 1.2 GiB
below what the equal slabs took. So the same sectors stay on their devices
at every multiple.
The multiple cuts the rounds from the slabs that the limit of two devices
and the budget of four allow. The former value of either gives back the
slabs a run had; the rounds come back only with
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it, and the bits with
`PARSEC_CUPY_RITZ_GRAM_SLABS=8` beside both (its row, the first of the last
three).
The turns of four devices and the slab limit of two never act on one sector.
The turns are taken with the slabs of 2 GiB that a sector on four devices
cuts and, since, with the whole rounds of the multiple: every chunk of the
measured bases keeps its four senders (derived), in 8 to 154 chunk steps per
pass where the equal slabs made 6 to 108. Where the slab of a whole round is
just wider than a chunk of 1 GiB, for 10,456, 29,576 and 39,368 electrons,
every other chunk step carried 4, 5 or 1 columns in its three turns, as in
the runs named here; such a slab is one chunk since (above).
Each of the two ran on A100 nodes without the other, and both together in
one code for 3,480, 23,768 and 39,368 electrons: the way to the rows
of 39,368 electrons took 17.4 s per sector, where equal slabs in turns took
16.5 s and equal slabs received at once 21.6 s.
The two releases: `PARSEC_CUPY_DENSITY_RELEASE` says how the pools of
several devices are emptied, `PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE`
whether those of a shared basis are. On 8 and 16 GPUs every sector is shared, so by default no pool is
emptied there, no device thread is given a release and a rank has nothing to
wait for behind its density command (`release` `threads`, `shared` `keep`,
`calls` 0). The threads act on 2 and 4 GPUs, and on 8 and 16 with the last
row at `1`. A shared basis is never spilled to the host: a rank whose
sectors are all shared keeps its pools under either storage of the states,
and sector states that a rank keeps on the host beside a shared basis are
collected for as they are alone, and like those leave the pinned pool since
(`PARSEC_CUPY_SPILL_PINNED_POOL`, below). The block
"Density at the sphere" after the SCF reads the density of the result on the
host and takes nothing from a device, whatever the pools hold then.

Which sectors get tiles is a matter of their filter work and of the devices
of their process (the table of two defaults above): one process drives the 1, 2 or 4
GPUs of a node, and on 8 and 16 GPUs every rank drives the four of its own,
so the rule weighs a sector alike on 4, 8 and 16. Which row limit a sector
follows where the limits are named is a matter of its devices. On 1, 2 and 4
GPUs every sector is alone on a device that filters four, two or one of
them: 325,000, 459,620 and 650,000 rows at `650000`. On 8 and 16 GPUs the
devices of a node share a sector: the row count of
`PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`, whatever the first limit says.
`0` and a tile size are taken as they are on every route, so the stencils a
launch had before either limit was lowered
are those of `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` at 2,200,000, 1,555,634 or
1,100,000 on 1, 2 or 4 GPUs and of `PARSEC_CUPY_IMPLICIT_TILE=0` on 8 and 16.
The serial control keeps the former route of
`PARSEC_CUPY_SECTOR_POOL_RELEASE`, `PARSEC_CUPY_DENSITY_COLLECTION`,
`PARSEC_CUPY_DENSITY_RELEASE`, `PARSEC_CUPY_RITZ_GRAM_MULTIPLE`,
`PARSEC_SYMMETRY_SCF_BUFFERS` and `PARSEC_FAST_GRID` and the slot-major
stencil at every size; it shares no basis, so it reads neither the slab
budget of that step nor the slab limit of two devices nor the turns of four
nor the helper nor the row count of a group nor the release of a shared
basis nor the chunk of the exchange, and names none of them. Its contexts are CuPy's also where a launcher
asks it for the overlap (`PARSEC_CUDA_CONTEXT_CREATION=runtime`). Where its
launcher leaves the storage of the sector states or their allocator to the
solver it decides by the former rule
(`PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=0.5`) and downloads a spilled basis
by the former call (`PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C`), into a new array
and back through the pinned pool, which it empties after every density
(`PARSEC_CUPY_SPILL_BUFFERS=fresh`, `PARSEC_CUPY_SPILL_PINNED_POOL=release`):
its Ritz step
takes two arrays per sector, which the count of the rule that came after
does not hold; with both values named, as in every measured control, none
of these is read. The report of the sphere (`PARSEC_DOMAIN_REPORT`) is the
launcher's. Of the Hartree boundary it names the kernels and the evaluator
of the tail, `PARSEC_HARTREE_BOUNDARY_KERNEL=full` and
`PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host`, and nothing else: the order, the
tolerance and the tail are the launcher's and the plan made from the
geometry, which the control must share with the run it is compared with.
`result.hartree_boundary` in the `timing.json` of both says what each had.
At a raised order the full-grid kernels hold more on the first device of a
control than PARSEC's boundary did. Where a control does not fit beside them
(expected for 14,680 electrons on four devices), its launcher names
`PARSEC_HARTREE_BOUNDARY_BACKEND=native`: the boundary on the host, see
`Hartree/GPU_HARTREE.md`.

The three rows from `PARSEC_CUPY_RITZ_GRAM_SLABS` on came after the ten
before them. Each changes a default: the slabs of the overlap of a shared
basis, the buffer of a sector on one device and the share of a device that
its states may take. With the three
at their former values a calculation is that of the code before them bit for
bit, and the former values of the ten rows before them give the calculations
that they name only with these beside them. The slabs of the overlap and the
limit of the buffer are described above, where the Gram
matrices are, and the share of a device in `MULTI_GPU.md`. The serial
control names the limit at its former value, for
a launcher that asks it for one array. It shares no basis and decides the
storage of its states by the rule before, so the rows of the slabs and of
the share do not reach it. The limit and the share meet in the count of a
device: the count holds the buffer that the step takes, 2 GiB less for a
sector of 32 GiB and more, and the share was chosen anew for that count. The
former count and the former decisions take both former values. `0.978`
alone, with the buffer of 2 GiB, keeps the states of 10,456 electrons on one
device, and a first run on an A100 with it stopped for memory in its first
solve, where the default spilled them and completed (`MULTI_GPU.md`): the
share stays 0.974.
With the three former values named, complete calculations on a laptop GPU
had every array of the archive and every line of `parsec.out` apart from its
times as the code before these rows gave them: the hybrid driver and the MPI
runner with one array per sector for 32 and 176 electrons, a serial control
of the same two (apart from the line that lists its routes, which names the
limit beside the settings it had), and 176 and 864 electrons with
every sector basis shared, that one GPU named two and four times as the
devices of a sector. With the defaults the runs
without a shared basis kept every bit as well, also 176 electrons with a
budget of 6 columns per slab under a multiple of 4, with those bytes named
as the limit instead of the budget.
With every basis shared the total energy of 176 electrons moved by 9e-13 and
1.1e-11 Ry on four and two devices (sectors of 36 and 37 columns: one slab
where the count cut eight of 5), and that of 864 electrons by 2.9e-11 Ry on
either, in the same nine steps, the eigenvalues by 2e-13 Ry at most (156 and
157 columns: a slab of 128 and one of 28 or 29 for eight of 20).

Two more defaults followed, and the second has gone back since: complete
calculations on A100 nodes showed no gain from it (below the table), so its
former value is the default again. Each rests on first runs on A100 nodes in which
the existing switch was named beside the default in one allocation, which is
the last column. What became the default differs from what was named there,
as the rows say, and had not run on A100 nodes itself when the rows were
written; the second has since, in those complete calculations. With both at
their former value a calculation is that of the code before them bit for
bit, and the defaults of then gave the same bits too. So it was on a laptop GPU, every array
of the archive and `parsec.out` apart from its times: through the reference
command line for 32 electrons and the accelerated one for 32 and 176,
through the MPI runner as a run and as a serial control for the same two,
and for 176 and 864 electrons with every sector basis shared, that one GPU
named two and four times as the devices of a sector, where every copy stays
inside the device. No laptop case has a slab of 1 GiB, so shared cases also
ran with the default chunk set to a size that their slabs exceed by an
eighth or less (176 and 864 electrons, each on two and on four named
devices; the sizes are given with the turns of four devices above): the
slabs within an eighth of it that most rounds of their sector send went to
the rows in one chunk and the others in two, with the same bits, as with
that size named. And shared cases ran with the one GPU named as the device
of the Hartree objects, so that no sector had another device (176 electrons
on two and on four named devices, 864 on two): the default of then, `any`,
asked it for the estimate, `auto` did not, and the bits were the same. The
former value of the second is the default again and need not be named.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 nodes, the switch named against the default of then |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` | unset: chunks of 1 GiB, and on the way to the rows one chunk for a slab (or a block) of a shared basis of which up to an eighth more than 1 GiB lies outside a device, in a buffer of exactly its size (at most 1,207,959,552 bytes); the way back keeps the chunks of 1 GiB | `1073741824`: chunks and a buffer of 1 GiB whatever travels | same bits | `exchange_buffer_bytes` of a shared sector in `timing.json` | not named: it shares no basis | `1090000000` named, which makes the slab of 32 columns of 39,368 electrons on 16 GPUs one chunk both ways: the way to the rows of a sector 16.04 against 17.41 s, the way back 10.65 against 10.57 s. The default takes that chunk on the way to the rows only, in a buffer of 1,086,758,400 bytes |
| `PARSEC_CUPY_RITZ_CONDITION_HELPER` | `auto` again, the former value (next column). `any`, the default for a while and a value that can still be named: a sector on two devices whose second one holds the Hartree objects (one sector of the root rank on 8 GPUs) has that device estimate the condition number of its overlap beside the small solve, where the rule that shares a basis counts the room for the estimate (it no longer asks that device whatever the room) | `auto`: that device never helps and such a sector estimates on its owner first; the default again, so it need not be named | same bits | `condition_device` and `condition_seconds` of a shared sector in `timing.json` | not named: it shares no basis | `any` named on 8 GPUs, in runs that also named another count of overlap slabs: the dense stage of that sector 1.88 against 3.06 s for 23,768 electrons and 3.02 against 4.88 s for 29,576; its helper 304 and 448 MiB more, the fullest device of the run 192 MiB more (50,871 MiB) and none; by the stage seconds about 0.5 and 0.2 s of 166.9 and 242.9 s of SCF, since the sectors are solved side by side. Complete calculations of `any` against `auto` decided the default (below the table) |

The first row is described with the turns of four devices above, the second
with the device solve above. The second is a default no longer: complete
first calculations on 8 A100 GPUs, two of each value in alternation with the
same total energies, took 163.40 s of SCF on average with `any` against
163.51 s with `auto` for 23,768 electrons and 236.16 against 235.92 s for
29,576 (2026-10-16), where `any` would have stayed the default for at least
0.4 % less in either case. The two do not meet: the buffer follows a slab
that takes more than a chunk outside a device, which a sector on two devices
sends for a few bytes at most, and the helper of the second row is that of a
sector on two devices. Neither changes which bases are shared but for the up
to 128 MiB that the first adds to the count of a sector on four devices, and
of one on two that names slabs of 4 GiB.

One switch came after these runs. It was measured on A100 nodes on
2026-10-16, first while every width took the six-column kernel six columns
per launch (the last column and the first table below), then with one launch
per block (the second table). The launches of now, described below the
tables, give blocks of two to six columns the six-column kernel of the first
measurement unchanged; with them it is still to be timed there:

| Switch | Default | Former value | Results | Says what ran | Serial control | A100, default against former, six columns per launch |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS` | `6`: where the projector coefficients `diag(signs) B.T X` take the tree reduction (projector rows of at least 256 entries, `PARSEC_CUPY_PROJECTOR_REDUCTION`), a block of two to six columns is one launch of a CUDA block of 128 threads per projector, a wider block one launch of a CUDA block per projector and group of six columns, and a single column takes the former kernel; on every route that forms them: each recurrence step of a filter, in recorded graphs and outside them, `H X` of the Ritz step, every device of a shared sector, the full-grid operator, and the FP32 kernels of the opt-in later filter | `1`: a block per projector and column, every width in one launch | same bits | `orbital_sector_projector_reduction`, and `gpu_projector_reduction` without symmetry sectors: `parallel6` against `parallel` (`serial`, the kernel of shorter rows, under either value), `parallel6` also for the launches of now, whose single columns take the kernel of `parallel`. They name the kernel where the stencil kernel fuses the scatter (the default); with `PARSEC_CUPY_FUSED_PROJECTORS=0` cuSPARSE forms the coefficients and the entries say `none` and `cuSPARSE`. The variable under `environment` of a rank | `1` | complete first calculations, the same totals: SCF 25.26 against 25.70 s for 3,480 electrons on 1 GPU, 168.65 against 172.53 s for 19,392 on 4 GPUs, 257.9 against 261.8 s for 39,368 on 16 GPUs. The kernel alone in the tables below |

The kernel alone, from part A of a probe of the nonlocal step on one A100
with the real factor of sector 0 of each case (2026-10-16, six columns per
launch): the time of one projection with the former kernel divided by that
with the six-column kernel, so that above 1 the six-column kernel was faster.
The coefficients were the same to the last bit at every width.

| Columns | 3,480 electrons: 677,542 rows, 795 projectors | 19,392: 2,873,400 rows, 4,601 | 39,368: 5,660,200 rows, 9,449 |
|---|---|---|---|
| 1 | 0.81 | 0.62 (0.0196 against 0.0314 ms) | 0.67 |
| 2 | 0.81 | 1.07 | 0.99 |
| 3 | 0.89 | 1.23 | 1.27 |
| 4 | 1.01 | 1.42 | 1.48 |
| 5 | 0.92 | 1.58 | 1.66 |
| 6 | 0.94 | 1.71 (0.0894 against 0.0521 ms) | 1.80 (0.177 against 0.098 ms) |
| 126 | 0.42 (0.280 against 0.670 ms) | 1.48 (1.62 against 1.10 ms) | 1.63 |

With 3,480 electrons every call of up to six columns took the time of its
launch from the host, about 0.018 to 0.022 ms with either kernel: those
ratios compare what the host pays for a launch, which a filter graph does not
pay. At 126 columns the six-column kernel took 21 launches against one; their
enqueue alone took 0.67 ms with 3,480 electrons and 0.63 ms with 19,392.

The former kernel launches a block of 128 threads for every projector and
column: a thread sums every 128th entry of the projector row for its column,
and seven barriers reduce the 128 partial sums in a tree. With six columns
per block a thread keeps six partial sums. It reads the index and the value
of an entry once instead of six times, and the block passes its barriers
once. Every column has the partial sums it had, in the same order, and the
same tree, so the coefficients are the same to the last bit.

From the table came one launch per block: a single column took the former
kernel, in the launch that `1` makes, and every wider block one launch of the
six-column kernel on a grid of a block per projector and group of six
columns, whose `blockIdx.y` named the group. Measured the same way on one
A100 (2026-10-16), a single column then took the time of the former kernel,
0.0198 against 0.0198 ms with 19,392 electrons and 0.0482 against 0.0482 ms
with 39,368, and 126 columns in one launch were 3.9, 1.23 and 1.12 times as
fast as in 21; complete calculations gave the same totals to the last bit.
But that kernel took longer than the six-column kernel of the first table
from three to six columns, 6 % longer at six, measured side by side in one
allocation with 39,368 electrons, and blocks of six columns are the main work
of a filter:

| Columns | Case | One launch on a grid of groups | Six-column kernel, six columns per launch |
|---|---|---|---|
| 126 | 3,480 electrons | 0.157 ms | 0.616 ms (21 launches) |
| 126 | 19,392 | 0.957 ms | 1.182 ms |
| 126 | 39,368 | 1.969 ms | 2.196 ms |
| 2 | 39,368 | 0.0672 ms | 0.0693 ms |
| 3 | 39,368 | 0.0829 ms | 0.0753 ms |
| 4 | 39,368 | 0.0891 ms | 0.0831 ms |
| 5 | 39,368 | 0.0964 ms | 0.0906 ms |
| 6 | 39,368 | 0.1044 ms | 0.0984 ms (the former kernel 0.1770 ms) |

Hence the launches of now. A single column takes the former kernel. Two to
six columns take one launch of the six-column kernel of the first table,
whose text is unchanged, a block per projector; their time with these
launches is still to be measured on an A100, and as the kernel and its
launch are those of the first table it should be that time. A block of more
than six columns, as `H X` of the Ritz step and the graphs of the distributed
filter have them, takes one launch of the group kernel,
`sparse_projector_projection6_groups`, on a grid of a block per projector and
group of six columns, the narrower rest in the last group; it is the
six-column kernel with the columns of its group, and every column keeps its
partial sums and its tree. Only a block of more than 393,210 columns (65,535
groups, the largest second dimension of a CUDA grid), which no calculation
has, takes a launch for every 65,535 groups. Rows below 256 entries keep the
canonical-order kernel, one thread per projector and column, which the switch
does not reach.

An operator reads the switch when it is built and keeps its kernels. A filter
graph is recorded for one block width and replays the kernel of that width,
the other devices of a shared sector are given the columns per block of the
owner, and a graph workspace whose operator has been given a projection of
another reduction since records its graphs again: a graph replays the kernels
it was recorded with. A value that is neither `6` nor `1` stops the
construction of the operator.

The complete calculations above saved 0.44, 3.88 and 3.9 s of SCF. The share
of the projection in their filter step has not been measured: the reference
run of 19,392 electrons on 4 GPUs reports affine tiles of 16 rows
(`implicit_affine_tile_16`), while the only step timing with the projection,
0.090 ms of 1.150 ms for six columns with 19,392 electrons, was made with the
slot-major stencil. The other part of the nonlocal term in a step, the scatter
`B C` inside the fused kernel, already serves six columns per thread and is
unchanged.

Three switches of the preparation of a rank came after those tables. They
act where the GPU ionic fields run on their thread (the root of a run, with
`PARSEC_OVERLAP_IONIC_SETUP`) and, the second, on every rank that solves
symmetry sectors. When they came, only the launch profile
(`PARSEC_IONIC_BACKEND=cupy`) put the GPU ionic fields on that thread. Since
2026-11-19 every cluster run with a usable GPU does (`PARSEC_IONIC_BACKEND=auto`,
the default: CuPy finds a device, the backend is `auto` or `cupy` and the run
is not a CPU run), and its energies differ from the in-line host sums by
round-off. Only `PARSEC_IONIC_BACKEND=native` (and a CPU run, a periodic cell or
`--backend numpy`) keeps the in-line host sums, their bits and the order of
stages of before. With the three former values named
a calculation is that of the code before them bit for bit, and so it was on
a laptop GPU, every array of the archive and `parsec.out` apart from its
times: through the accelerated command line under the launch variables for
CF4 (32 electrons), benzene (30) and C35H36 (176, with the radius of its
input and with the one of the default rule), and through the MPI runner
with one rank for CF4 as a run, as a serial control and with the states of
the sectors on the host, and for C35H36 as a run and with every sector
basis shared (that one GPU named twice). With the defaults of that time the
serial control and a command line without the launch variables stayed the same
bits (since the `auto` default of 2026-11-19 a command line without the launch
variables takes the GPU sums too, and keeps those bits only with
`PARSEC_IONIC_BACKEND=native`);
the other runs differ by round-off, and by the sums on the wedge alone: the
thread of the ion-ion energy by itself and the thread of the symmetry maps
by itself gave the bits of the former values (C35H36 through the command
line), while the wedge sums by themselves gave the differences of all
three. The local ionic potential moved by at most 7.1e-15 Ry (CF4,
benzene) and 5.7e-14 Ry (C35H36), and the total energy by 8.5e-13 Ry for
CF4 on every route, 2.3e-13 Ry for benzene and 9.1e-13 to 1.5e-11 Ry for
C35H36 (the largest with its basis shared). The eigenvalues moved by at
most 7e-14 Ry for CF4 and benzene and 9.3e-12 Ry for C35H36, 1.7e-10 Ry
with the radius of the default rule, and the density by at most 9e-13
electrons per cubic bohr, with the same number of SCF steps in every pair
(11 to 14); states that are degenerate by symmetry can change places in the
list of eigenvalues. The stages the switches act on were timed on A100
nodes one by one (last column). Complete runs with the switches on and off
have run there for 39,368 electrons on 16 GPUs (first runs, the launch
profile of the series): the root prepared in 6.48 against 10.32 s and the
other ranks in 5.00 against 5.92 s, the program took 264.9 against 268.9 s
and the SCF 257.9 against 258.2 s, and the total energies were 3.2e-10 Ry
apart. With these three and the other switches that came after the series
at their former values, eight of its configurations gave its total
energies to the last bit there (`MULTI_GPU.md`). On the root the three act
on one chain, the ionic thread and the join that waits for it, so the gains
of their stages do not add up there (`GPU_SETUP_OPTIMIZATION.md`).

| Switch | Default | Former value | Results | Says what ran | Serial control | Measured |
|---|---|---|---|---|---|---|
| `PARSEC_OVERLAP_ION_ION_SETUP` | `1`: the ion-ion energy of a root whose ionic fields are on their thread is summed on a second thread (`parsec-ion-ion`) that starts with it, once the projectors are built, and is handed to the completion of the reference system where its stage stands | `0`: the last stage of the ionic thread | same bits: the same function of the same atoms | `ion_ion_setup` and `ion_ion_setup_seconds`; the stage `ion_ion_energy` of `preparation_stages` | `0`; its fields are in line in any case | host of an A100 node, 16 OpenMP threads, 39,368 electrons (11,021 atoms): the sum takes 1.08 s; beside the symmetry maps both took 1.26 s, the time of the maps alone; beside the loop of the projectors 1.90 s against 1.34 and 1.08 s alone, which is why the thread starts after them. In first runs on 16 GPUs the join of the ionic thread had waited 1.08, 0.63 and 0.44 s for 39,368, 29,576 and 23,768 electrons, never longer than this stage |
| `PARSEC_OVERLAP_SYMMETRY_SETUP` | `1`: on a rank whose reference preparation builds no ionic field and no full-grid matrix (a sector worker; a root with the ionic fields on their thread), symmetry detection and the representation phases run on a thread (`parsec-symmetry-setup`) that starts when the grid exists, beside the projectors | `0`: after the reference preparation, on the thread that prepares | same bits: the same functions of the same grid and atoms | `symmetry_setup` (`overlapped with the projectors` or `inline`), `symmetry_setup_seconds`, `symmetry_setup_wait_seconds`; the stages `symmetry_reduction` and `symmetry_decomposition` with their thread | `0` | host of an A100 node, 16 OpenMP threads, three runs per case: projectors and maps 1.34 and 1.26 s alone and 1.41 s side by side for 39,368 electrons (1.19 s less than one after the other), 0.37 and 0.42 s alone and 0.43 s together for 10,456, 0.12 and 0.18 s alone and 0.19 s together for 3,480; with the plan of the Hartree boundary as a third stage 1.18 to 1.34 times the longest, so the plan stays in line |
| `PARSEC_IONIC_SUM_GRID` | `auto`: GPU ionic fields on their thread are summed at the representative point of every symmetry orbit, and the orbit takes that value, where every operation of the detected group maps the atoms and the grid points onto themselves in every bit; every grid point otherwise (atoms that are images within the tolerance of the detector only, as coordinates from a structural relaxation usually are and as a structure can be once the recentering of the input has moved it; no operation beside the identity; fields in line) | `full`: every grid point | round-off: the value at a representative point is the full-grid one in every bit, and the other points of its orbit used to add the same terms in another order of the atoms; `full` gives the former bits | `ionic_sum_grid`: `symmetry wedge (n of N points)`, or `full grid` with the reason; `ionic_sum_grid_choice_seconds`: what the thread took before its first sum to choose (its wait for the symmetry reduction and the test of the images), which is no part of `ionic_setup_seconds`; the stages `ionic_symmetry_wait` and `ionic_sum_grid_test` | `full`; its fields are in line in any case | one A100 node, 39,368 electrons (22,640,800 points, 5,660,200 orbits): both sums 1.38 instead of 4.35 s on four devices and 2.49 instead of 10.37 s on one; the same bits at the representative rows; against the orbit average that the SCF loop kept, at most 4.5e-12 Ry in the potential (1.7e-15 of its largest value) and 1.4e-16 in the density |
| `PARSEC_IONIC_BACKEND` | `auto` (since 2026-11-19): the local ionic potential and the atomic densities of a cluster are summed on the GPU (`V_ion/cupy_ionic.py`, on their thread beside the sector set-up, on the symmetry orbits as below) where CuPy finds a device, the backend asked for is `auto` or `cupy` and the run is not a CPU run; on the host otherwise | `native`: the C++/OpenMP sums of every grid point on the host, in line (the default before; `cupy` asks for the GPU by name, as the launch settings did) | round-off of the sums: energies 2e-9 (k06), 8e-8 (k10), 1e-8 Ry (k12) at the criterion of those inputs | `ionic_setup_implementation`; the stages `local_ionic_potential` and `initial_density` | `native` | one A100 node, code defaults otherwise, first runs: through the Python API with 16 OpenMP threads, the preparation of k06 3.8 instead of 16.0 s, k10 6.3 instead of 196.5 s, k12 8.7 instead of 434.0 s (the host sums of every point took 11.6, 189 and 420 s); through `main.py` with its default thread count (28 OpenMP threads: 32 CPUs allowed less 4), the time before the SCF of k06 4.6 instead of 15.4 s, k10 8.8 instead of 216.0 s, k12 10.2 instead of 473.6 s (host sums 10.7, 207 and 462 s, GPU sums 0.9, 1.1 and 1.1 s), the whole run 17.9 instead of 29.6 s, 141.8 instead of 351.1 s and 316.6 instead of 787.9 s, energies within 7e-8 Ry; the SCF the same. A node whose process may use more CPUs runs the host sums on more threads, but 28 threads were no faster than 16 here |
| `PARSEC_IONIC_SUM_GROUP` | `full` (since 2026-11-19): those orbits are the orbits of the full point group where weighted sectors find it (`Symmetry/field_orbits.py`; Td: N/24 points instead of the N/4 of D2), published once the group is detected; where the atoms are images under it only within the detection tolerance (last-bit differences between the axes), the orbits of the sectors' group and then of the abelian group detected first are tried before every grid point, and a moved structure tries those of the full group and then of the sectors' group (`_ionic_sum_candidates`) | `first`: the reduction detected first (the abelian group), published at once, as before | round-off of one sum, as `auto` of `PARSEC_IONIC_SUM_GRID` | `ionic_sum_grid`: `symmetry wedge (n of N points)`, followed by `; a group of g operations, the larger one failed: ...` where a smaller group took the sums | `first` | one A100 node: k12 644,274 points summed instead of 3,786,832 (1.15 against 2.08 s of the thread), k06 and k10 the same time; the preparation within 0.08 s either way, the thread being off the critical path (it waits up to 1.4 s more for the group) |
| `PARSEC_IONIC_MOVED_SUMS` | `orbits` (since 2026-11-19): the GPU sums of a moved structure (relaxation, MD) run on the orbits of the full group of the first structure, or of its sectors' group, where the images of the moved atoms lie within 1e-10 bohr of their atoms (positions averaged over the group are images to their last bits) and those of the grid points are exact | `full`: every grid point of every moved structure, as before | a relative 1e-11 at most in the moved fields, which the orbit average of the SCF loop takes too | `moved_ionic_sum_grid` of the backend: `symmetry wedge (n of N points)` or `full grid` with the reason | `full` | one A100 node, first runs, one move of the atoms after the forces: the sums of the moved k10 (10,456 electrons) 1.29 instead of 2.46 s (444,382 of 10,419,384 points), of k06 0.31 against 0.29 s; the energies of the moved structures within 4e-9 (k06) and 3e-8 Ry (k10) of the full-grid sums, the forces within 1e-12 |

They are described in `GPU_SETUP_OPTIMIZATION.md`, with the record that
shows the chain of stages of a rank (`preparation_stages`, below under
"Timing and profiling"). The serial control of the runner names the three
former values, so the line of its routes in `parsec.out` has three settings
more than it had.

Three switches of the host staging came after that table. They act
where sector states are kept on the host
(`PARSEC_CUPY_SECTOR_STATE_STORAGE=host`, or `auto` where a device that
solves several sectors in turn cannot hold them together, as on 1 and 2
GPUs); which sectors are spilled is not theirs to decide. The first also
selects how the device builder of a density uploads any orbitals that it is
handed on the host: on the shipped routes only the columns of a spilled
sector are, so it changes nothing elsewhere, and a caller of the API that
hands the builder host orbitals gets the direct copy too, with the same
bits. The last column is one array of 18.84 GiB, the basis of a sector of
10,456 electrons, between an A100 and its host; a complete run follows the
table.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 node, one array of 18.84 GiB |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_SPILL_BUFFERS` | `persistent`: a spilled sector keeps one host buffer of the size of its basis; a download writes into it and an upload reads it where it lies (`ndarray.get(out=)`, `ndarray.set`), before a solve and for the density, with no staging block and no second array on the device | `fresh`: a new pageable array for every download (`cupy.asnumpy`) and a block of CuPy's pinned pool for every upload (`cupy.asarray`); `PARSEC_CUPY_SPILL_DOWNLOAD_ORDER` belongs to this staging | same bits | `spill_staging` of a rank in `timing.json`; the detail `orbital_sector_spill_staging` of a run whose sectors took buffers, without the bound, which `spill_staging` holds; `host_uploads` under `density_pool_release` | `fresh` | 0.758 s up and 0.770 s down through a page-locked buffer (24.8 and 24.5 GiB/s) and 1.40 and 1.52 s through a pageable one that is kept, against 1.75 s up with the staging block in the pool (17.99 s with the pool empty) and 2.62 s down into a new array |
| `PARSEC_CUPY_SPILL_PINNED_BYTES` | unset: the buffers of a process are page-locked while their sum stays within half of the memory that the host has available when the storage is decided, and within `RLIMIT_MEMLOCK` where the system sets one; the buffers beyond that, and one that the driver refuses to lock, are pageable | a number of bytes, taken as it is; `0` locks nothing. There is no former value: `fresh` has no buffers | same bits | `bound_bytes`, `bound_from`, `page_locked_bytes`, `pageable_bytes`, `lock_seconds` and `lock_failures` under `spill_staging` | not named: it has no buffers | page-locking one buffer took 9.6 to 9.7 s, four of them one after another 38.4 s, and their return 13.9 s, which a run spends when its process ends, outside every record of `timing.json` |
| `PARSEC_CUPY_SPILL_PINNED_POOL` | `keep`: a density that was handed a sector whose states are on the host leaves CuPy's pinned pool as it is; with `fresh` the staging block then serves every upload of the run | `release`: the pinned pool is emptied after every density | same bits | `spill_pinned_pool` and `pinned_kept` under `density_pool_release` of a rank | `release` | emptying the pool with the staging block of 32 GiB in it took 5.82 to 5.89 s, and page-locking that block again 15.9 to 16.3 s |

`MULTI_GPU.md` describes them. With `fresh` and `release` named a spilled
calculation takes the former calls in their order and is that of the code
before these switches bit for bit, also where
`PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C` asks for the download as it was before
that. So it was on a laptop GPU for 32 and 176 electrons through the hybrid
driver and the MPI runner, every array of the archive, the backend details
and `parsec.out` apart from its times; with the buffers, page-locked and
pageable, the arrays and that file were the same and the details had one
entry more. The serial control names both where its launcher names neither,
so the line that lists its routes has these two settings more.

A complete first run on one A100 node (FP64) has been made since: 10,456
electrons in the input with 5 ang of vacuum, the storage left to the code,
which spills the four sectors, with the defaults and with `fresh` and
`release` named in one allocation. The SCF took 396.9 against 746.4 s and
the reported program 401.2 against 751.3 s, with the same total energy to
the last bit. The four buffers page-locked 80.93 GB in 40.0 s, inside the
offloads, which took 73.9 against 116.3 s; the restores took 46.6 against
220.4 s. The host high-water mark was 78.2 against 109.9 GiB, and the GPU
peak 23.61 against 23.53 GiB. The page-locked buffers go back when the
process ends, after the reported program and outside every record: by the
log of the batch job the whole process, start and teardown included, took
434 against 770 s, 32.8 against 18.7 s beyond the reported program: the
return of the buffers costs about 14 s that no record holds. Two devices,
pageable buffers in a complete run and
each switch by itself have not run on A100 nodes.

The slabs of the one-block Ritz step of a shared basis (8 and 16 GPUs) take
1 GiB since, where they took 2 GiB, on two and on four devices. Where 1 GiB
holds fewer than 32 columns, a slab is widened up to that again, but never
beyond the width that 2 GiB gave. The floor is a budget an eighth above 32
columns, 36: `block_rounds` cuts a block into equal slabs and a whole round
falls to the next multiple of 64 columns below, so a budget of exactly 32
columns gave the blocks of 29,576 electrons (924 to 930 columns) slabs of 16
and 58 rounds, and that SCF took 2.3 % longer than with the slabs of 2 GiB;
36 keeps slabs of 32 columns for every block of 288 columns and more. The
floor does not depend on the columns, so a slab never widens when a trim
takes columns from a sector after the first solve that the sharing rule
counted. With
`PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES=8589934592` and
`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=2147483648` together a
calculation is that of the code before them bit for bit. The count of the
sharing rule follows the slabs, so it falls with them.

| Switch | Default | Former value | Results | Says what ran |
|---|---|---|---|---|
| `PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES` | `4294967296`: the bytes, in all rows, of a round of the one-block Ritz step of a sector on four or more devices where `PARSEC_CUPY_STREAMING_RITZ_BYTES` is not set; slabs of 1 GiB with four devices | `8589934592` (slabs of 2 GiB) | round-off where the rounds change: 1e-9 to 8e-8 Ry on A100 nodes | `slab_columns` of a shared sector in `timing.json` |
| `PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES` | `1073741824` now: slabs of at most 1 GiB on two devices | `2147483648` (the default before); `4294967296` stays the slabs of series F9 | as above | as above |
| `PARSEC_CUPY_DISTRIBUTED_STATE_MIN_SLAB_COLUMNS` | `32`: the fewest columns of such a slab where its limit leaves fewer, up to the width of the former limit | `1` (no floor) | as above | as above |

First runs on A100 nodes with the launch profile of the series, the defaults
against the two former values in one allocation, in the order old, new, new,
old (one of each for 39,368 electrons); the peak of the fullest device and
the SCF seconds are the means of each side:

| Electrons | GPUs | Peak, former / default (GiB) | SCF, former / default (s) |
|---|---|---|---|
| 10,456 | 16 | 10.02 / 9.07 | 22.6 / 23.5 |
| 14,680 | 16 | 14.93 / 13.06 | 41.0 / 40.7 |
| 19,392 | 16 | 19.61 / 17.72 | 55.9 / 56.3 |
| 23,768 | 16 | 28.48 / 26.67 | 92.6 / 91.8 |
| 29,576 | 16 | 36.19 / 35.21 | 129.3 / 131.1 |
| 39,368 | 16 | 61.23 / 61.23 | 257.7 / 257.9 |
| 10,456 | 8 | 15.73 / 13.89 | 36.3 / 35.8 |
| 14,680 | 8 | 24.65 / 22.16 | 68.0 / 68.8 |
| 19,392 | 8 | 32.18 / 30.81 | 96.3 / 97.3 |
| 23,768 | 8 | 49.68 / 47.87 | 161.6 / 162.9 |
| 29,576 | 8 | 65.50 / 63.50 | 233.5 / 235.7 |

The SCF moved by 1.3 % less to 1.1 % more, of the size of the spread of
repeated runs (10,456 electrons on 16 GPUs: 22.2 and 23.1 against 22.2 and
24.9 s), except 29,576 electrons on 16 GPUs, 1.4 % more in both pairs for
0.97 GiB less. 39,368 electrons on 16 GPUs keep their rounds (32 columns
either way), their peak and their bits. All runs converged in the steps of
the former side.

The sphere radius has no environment switch. An input that leaves
`Boundary_Sphere_Radius` out gets the radius and the Hartree tolerance of
the default rule in the parser (reference `README.md`, "Sphere radius left
to the code"); the line `Boundary_Sphere_Radius` restores the former
calculation, and the report prints it with the tolerance, so that the two
lines give the same grid, plan and result bit for bit. The rows of the table
above that begin with `PARSEC_HARTREE_` act in the driver after the parsing:
they change the boundary of the run and never the sphere, and the set-up
block then says that `Domain_Energy_Tolerance` covers the wall only.

The report of the sphere has one: `PARSEC_DOMAIN_REPORT`, on by default and
read by the reporter of both drivers. `0` (or `off`) gives an input with a
radius the `parsec.out`, the backend details and the work after the SCF it
had before the rule: no estimate at set-up, no block "Density at the
sphere", no `domain_*` detail, `result.domain` null. The results are the
same bits with and without it. A radius the rule chose keeps its lines and
its record and loses the block after the SCF. What a run had is in three
places:

| Where | What |
|---|---|
| `parsec.out`, grid data | the radius, the outermost atom and vacuum, the wall estimate before the SCF, what set the radius, the two input lines (radius of the rule only) |
| `parsec.out`, real-space setup | the energy the boundary values of the plan in use are estimated to leave; for a radius of the input also the wall estimate of its sphere |
| `parsec.out`, after the SCF | "Density at the sphere": the estimate from the converged density and the radius and tolerance that would meet the wall share, with what set that radius where the density did not |
| backend details, result archive | `domain_radius`, `domain_vacuum`, `domain_energy_tolerance`, `domain_wall_estimate`, `domain_boundary_energy`, `domain_wall_after_scf`, `domain_radius_for_wall_share` |
| `timing.json` of the MPI runner | `result.domain` with the same, the shell sums of the fit and, under `after_scf`, `estimates`, `shell_seconds`, `radius_seconds` and `seconds`; `per_rank[].domain_rule_seconds`; `per_rank[].domain_report_thread` of the root, what became of the thread of the radius search |

The MPI runner compares, before any rank prepares, the radius and the
tolerance every rank resolved, bit for bit, and stops where they differ. The
seconds of the rule are inside `reported_program_seconds`: 0.03 s for 3,480
electrons and 0.22 s for 39,368, where one estimate of the omitted potential
is needed (workstation host; pseudopotential loading included). For an input
with a radius the parser does what it did; the wall estimate of its sphere
(a few milliseconds), the shell sums after the SCF (0.156 s at 22.6 million
grid points on the same host) and the search for the radius under them are
made by the root while it reports. That search asks the largest multipole
order for the tolerance of the rule and makes no estimate of the omitted
potential for 3,480 electrons, none or one (0.12 s) for 23,768, and for
39,368 one (0.17 s) or four to seven where the wall would leave less than
4 ang of vacuum: 0.33 to 1.2 s for the block after the SCF of C9449H1572 on
that host, where the whole reporting after its SCF took 0.17 s on 16 A100
before. An input with a `Hartree_Boundary_Tolerance` line gets
no estimate. A failure of a report is printed in its place and does not
reach the result.

The estimates of that search depend on the geometry and a candidate radius
alone. On the CuPy backend a host thread therefore makes them while the SCF
runs (`PARSEC_DOMAIN_REPORT_THREAD`, on by default; `0` makes them after the
SCF, as before). The search starts where the fit of the final density puts
the wall, at most 2.5 ang from the sphere of the run and not below the
minimum vacuum: the thread makes the search of the rule from every such
start, the nearest first, each with the tolerance the rule gives it, and
keeps what it estimates: 25 estimates for C9449H1572, 8 for 14,680
electrons and 11 for 23,768 (4.3, 0.7 and 1.4 s on the host of a laptop),
and none where the series bound of the rule settles every start, as for
3,480 electrons. After the SCF the same search is made and reads the
estimates it asks for: the lines of the block and the record are the same,
`estimates` included, and `radius_seconds` is the time of a search that had
them (2.5 ms each for C9449H1572 on that host, where the 1,271 estimates of
432 searches from as many fits had taken 228 s). The search makes for itself
an estimate that was not prepared, and makes again one whose last bits could
decide. An estimate is a difference of two potentials of all the charges at
the sphere (1,130 to 1,290 Ry for C9449H1572, where the tolerance is 2e-4
Ry), and the order in which a BLAS sums the one product in it moves it by
spacings of those potentials: by 4 to 9 of them between that product and
the same sums in other orders, for 3,480 to 39,368 electrons on the host of
a laptop (2e-12 Ry for the largest). A prepared estimate is therefore read
only where it lies further from the tolerance than 64 spacings of the
largest potential the charges can have on the sphere, 1.2e-10 Ry for that
system. The thread works with NumPy on host arrays of its own and calls
nothing on a
device. The reporter ends and joins it before it reports, and both drivers
end it on every other way out of a calculation. What became of it stays
with the reporter (`radius_thread_record`), and the MPI runner writes it
beside the times of the root, `per_rank[].domain_report_thread` in
`timing.json`: the estimates the thread made, its wall seconds from start
to end, and the text of what it raised or of what kept it from starting. A
thread that fails leaves the search to make its estimates after the SCF, as
before, and that entry is where it shows; the record of the sphere is the
same either way. A calculation on a host
backend gets no thread: its SCF works on the cores the thread would take.
The thread has not run on A100 nodes.

The recurrence kernel of the filter has a switch since 2026-10-21,
`PARSEC_CUPY_STENCIL_KERNEL`. `pipelined`, the default, gives every stencil
operator, beside the former kernels, a `stencil_major_chebyshev6` whose slot
loop is a software pipeline of its loads (`backends/stencil_pipeline.py`);
`former` restores the kernels of before, exactly. For the
launches of production, six columns of unit row stride on a stencil of the
slot count it was generated for, the slots run as straight-line code: the
neighbour record of slot s + 2 and the coefficient and six values of slot
s + 1 are loaded before the six FMAs of slot s, so that a warp has two slots
of loads in flight where the former loop, which waits for each record behind
a branch, had at most one. `__launch_bounds__(256, 4)` holds the kernel at
64 registers, the four blocks of 256 threads per multiprocessor that the 56
registers of the former kernel give, with no local memory on an A100. Every
column still sums its slots in order with the FMA of the former loop, a
missing neighbour adds nothing, and the epilogue (scatter, potential,
recurrence) is the former text after the same join, so every output word is
that of the former kernel, NaN payloads included on an A100 (other GPUs at
the end of this section). `stencil_major_spmm6` (`H X`) stays the former kernel
under both values: given the same fast path, ptxas commuted the operands of
one of its additions, which changed the NaN that came out where two NaNs of
different payloads met. Every other launch of the recurrence (fewer than six
columns, another row stride) goes to the former recurrence kernel as well:
inside the generated one it took the former loop at the registers of the
fast path, 2 to 6 % slower. An operator reads the switch when it is built and
compiles all its kernels then; a filter graph records the kernel each of its
launches took, and the other devices of a shared sector build the kernels of
their owner. A value other than `former` or `pipelined` stops the
construction of the operator. A generated kernel that cannot be compiled
gives way to the former recurrence kernel with a `RuntimeWarning`: the
operator keeps its layout and the former kernels, and so the former bits,
and the record of the run says why. With an empty CuPy cache, the first
operator on a device took 0.68 instead of 0.35 s to build in tiles of 16
rows and 0.54 instead of 0.20 s slot-major, most of it compiling; from the
cache it took 0.11 to 0.14 s.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100, 2026-10-20 |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_STENCIL_KERNEL` | `pipelined`, also when unset or empty: the recurrence kernel above for the launches of six columns of unit row stride, the former kernels for every other launch and for `H X` | `former`: the stencil kernels as they were | same bits, NaN payloads included on an A100 (on another GPU the payload where two NaNs meet may differ, below) | `orbital_sector_stencil_kernel` among the backend details (`gpu_stencil_kernel` of a full-grid operator), an unset variable included: `former`, `pipelined: fast path for 25 slots`, or `pipelined: former kernels (...)` with the reason where the generated kernel could not be built; in `parsec.out`, the short report too, in `result.backend` of `timing.json` with `per_rank[].orbital_sector_stencil_kernel` of the MPI runner, and in the archive. `stencil_kernel_attributes` of the stencil of an operator gives the registers of its kernels (64 against 56 for the recurrence) | `former` | a probe of the filter on sector 0 in tiles of 16 rows, prepared as a run prepares it (CUDA events around graphs of launches), either value on its own GPU of one node: one recurrence step of six columns took 0.617 against 0.925 ms with 19,392 electrons (2,873,400 rows) and 1.289 against 1.831 ms with 39,368 (5,660,200 rows), 0.667 and 0.704 of the time, the first step 0.670 and 0.689; a block of six columns through the filter graphs 10.35 against 14.95 ms and 21.24 against 29.50 ms (0.693, 0.720), the same per block in a plan of 48 columns; one, two and five columns within 0.3 % of the former kernel. Every word was the former one on those sectors and on that of 3,480 electrons, in tiles of 16 rows and, for 3,480 and 19,392 electrons, slot-major: every launch path of the recurrence kernel, the operator's methods, `H X`, the graphs and the filter without them, with finite inputs and with NaNs (`benchmarks/stencil_pipelined_check.py`) |

The kernel alone was measured the same day with former and generated
kernels alternating in one process (medians of 11 groups of 40 launches):
0.664 and 0.699 of the time with 19,392 and 39,368 electrons, 0.639 with
3,480, and 0.690 with 19,392 electrons slot-major. Complete calculations of
3,480 electrons on 1 and 4 GPUs, and without symmetry on 4, gave every array
of the archive apart from timings as the former kernels
(`backend_detail_values` differed only in their `*_seconds` entries). Those
runs were made in a form that had the fast path in `stencil_major_spmm6` as
well and sent the narrow launches into the generated kernel. The source of
now is, to the byte, the one that was then compared with the former kernels
with the fast path in `stencil_major_chebyshev6` alone: 0 differing words in
79,272 launch pairs, every value a NaN among the inputs
(`tests/test_stencil_pipeline.py` holds its digests). In this form, complete
calculations of the same 3,480 electrons at `Expansion_Order` 6, 12 and 20
on one A100, and at 12 without symmetry on four, gave archives identical to
those of the former kernels apart from timings, and took 21.3 against 26.0 s,
27.7 against 36.4 s, 40.3 against 52.8 s and 46.7 against 54.8 s
(`Total accelerated Python wall time`; single runs, on A100 nodes whose CPUs
other jobs shared). Every slot count from 1 to 61 in the
slot-major layout and in tiles of 16 to 256 rows, 366 cases with 186,864
pairs of launches and method calls on finite inputs, NaN payloads and signed
zeros, gave no differing word either. The archives of runs made since name
the kernel among `backend_detail_values` as well, an entry that differs
between the two values by design. The former kernel was latency-bound on
these sectors (0.71 eligible of 7.04 active warps per scheduler, 71 % of the
stalls waiting on loads, 30 % of the bandwidth of the device); the generated
one keeps the same warps resident and has more loads in flight.
`tests/test_stencil_pipeline.py` compares the two kernels on synthetic
stencils of 2 to 61 slots, in the slot-major layout and in tiles of 16 to 256
rows, on every launch path, with the NaN payloads of two NaNs that meet, NaN
coefficients and signed zeros; it is the check to run again after a change
of CUDA, CuPy or driver, since the equality rests on how ptxas compiles both.

On a GeForce RTX 5070 Laptop GPU (compute capability 12.0, CUDA 12.9,
2026-10-21) the same tests found every number and every NaN of the former
kernel in its place, in the sweep of slot counts and tiles as well; where two
NaNs of different payloads met in one operation, the result sometimes carried
the other payload, which IEEE 754 leaves open, and ptxas kept local memory
under the launch bounds: 16 bytes for 25 slots in tiles (none slot-major), up
to 56 for other slot counts. The tests demand the payloads and no
local memory on compute capability 8.0 alone and, elsewhere, the NaNs in
their places and every other word. One recurrence step of six columns took
there the time of the former kernel, 0.460 ms either way on sector 0 of 864
electrons (274,784 rows in tiles of 16; two runs each),
and 0.96 and 1.04 of the former time at 12 and 24 columns (one run each). At
25 FMAs per row and column that step makes 180 Gflop/s, about half of the
360 to 410 Gflop/s of FP64 FMAs the device does at most, where an A100 makes
1.4 of its 9.7 Tflop/s: FP64 units that slow leave little waiting on loads to
hide. No other architecture has run the kernel.

The rounds of the Ritz step of a sector basis that several devices share
have two switches since 2026-10-21, `PARSEC_CUPY_RITZ_PIPELINE` and
`PARSEC_CUPY_EXCHANGE_STRIDED` (`Eigensolvers/distributed_state.py`), whose
defaults since 2026-10-22 are `p2` and `1`; `off` and `0` restore the former
step exactly. In a round of the step with one tall block every device
applies `H` to its slab, moves the slab into its workspace, sends it and
then `H` times it to the row layout, and projects. In the former step each
of these waits for all devices and the overlap is formed after the rounds,
so that either the multiprocessors of a device or its links wait. `p1`
sends the slab on a second stream of each device, its exchange stream,
while `H` is applied on the compute stream to the copy that has already left
the block. `p2` also forms each slab of the overlap (128 columns) as soon as
the round that holds its first column has arrived, beside the sending of `H`
times the slab of that round: the rows arrive from the end of the row block,
and a slab is multiplied by the columns from its own first one on. The
exchange stream waits for the leave and for the apply of its device by
events, the steps of a conversion wait for all devices as before, and the
pack of a conversion is queued before the kernel that runs beside it, so
that only copies between devices run beside a stencil or a product. Nothing
is allocated under the exchange stream: CuPy's pool keeps a free list per
stream. `p1` and `p2` give the bits of `off`. The stencil and projector
kernels read the copy with the strides of the slab, and every slab of the
overlap is the product of before, with the same operands, on the same stream
and cuBLAS handle; the exchange stream calls no cuBLAS. Where the overlap has
no slabs (`PARSEC_CUPY_RITZ_SYRK` on or off, or a basis that is one slab,
128 columns or fewer by default) `p2` forms it after the rounds as `p1`
does. The steps with two and three tall blocks do not read the switch.

The pitched exchange (`PARSEC_CUPY_EXCHANGE_STRIDED=1`) moves every piece
between two devices with one `cudaMemcpy2DAsync`: a line per column, with
the distance between the columns of each array as its pitch, so the rows of
a column block go as they lie. No pack or unpack kernel runs, no chunk is
cut and no exchange buffer is taken (1,086,758,400 bytes per device for
39,368 electrons on 16 GPUs); four devices keep their turns, and every byte
arrives where it did. It acts in every conversion of a shared basis, with
any number of tall blocks. The rule that shares a basis counts no buffer for
it either, up to 1.125 GiB less per device (about 1 GiB for the large
sectors): in the measured series that changes no decision (the count closest
to the share of 0.85 went from 0.760 to 0.748 of a device, 29,576 electrons on
two devices, and 39,368 electrons on two, 1.363 and 1.350 of a device, share
no basis either way), and the counts of the state
storage of shared sectors (`sector_state_fit_bytes`) are that much lower.
`0` restores the packed exchange and its count.

Under `p1` and `p2` the host no longer waits for the apply, the leave and
the products, so the stage seconds of a shared sector come from device
events around every stage of every round, summed per device with the most
of a device taken (`stage_clock`: `events`; `host` under `off`). A stage that
ran beside another counts all of its time, so the stages can add up to more
than the rounds took. `rounds_seconds` holds the host seconds from the first
round of every step to the end of its overlap under every value: the number
to compare. The whole round loop is then one
`subspace_ritz_projection_seconds` of the backend statistics. The stages that
follow the rounds, `dense`, `rotate`, `to_columns` and, under the Ritz
hand-over below, `handover` and `correction`, keep host clocks under every
value.

Probes on the four A100 of one node, 2026-10-21, at the shapes of the
step: the 39 products of
the overlap of a pass of 39,368 electrons on 16 GPUs took 2.03 s alone and
the exchange of `H` times the slabs 0.57 s; side by side, pack first, 70 %
of the exchange hid behind the products, 0.40 s per pass, the products 2.5 %
slower and with the same bits, and 79 % with pitched copies. Two devices of
29,576 electrons on 8 GPUs saved 0.28 s per pass the same way. Probes of
2026-10-10 had measured 87 % of the apply hidden behind the exchange of the slab with the
pack first (0.35 s per pass at that shape), and an exchange 3.8 times as
long where the pack was queued beside a product. A pitched copy over one
link ran at 86.3 to 86.9 GiB/s against 87.0 to 87.5 for a contiguous copy of
the same bytes; the three turns of four devices to their rows took 0.89 to
0.90 of the time of the packed turns with their pack (13.27 against 14.80 ms
for slabs of 32 columns of 5,660,200 rows), and the way back 0.92 to 0.93 of
the pulls into the buffer with the unpack.

Complete first calculations on dedicated A100 nodes, 2026-10-22, launch
profile of the measured series, in the order ABBA, the former `off` with the
packed exchange against `p2` with pitched copies: the same total energies
and SCF steps; the SCF 4.4 % shorter for 23,768 electrons and 3.6 % for
29,576 on 8 GPUs, 6.4 % for 23,768 and 5.4 % for 39,368 on 16 GPUs (231.1
against 218.8 s), and the fullest devices 1.0 to 1.2 GiB lower. Those made
the two values the defaults.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 nodes, default against former, 2026-10-22 |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_RITZ_PIPELINE` | `p2`, also when unset or empty: the slab of a round sent on the exchange stream beside the apply of `H`, and the slabs of the overlap round by round beside the sending of `H` times the slab. `p1` sends the slab beside the apply alone | `off`: the rounds of the step with one tall block one stage after another, as before | same bits | `orbital_sector_ritz_pipeline` among the backend details (the switch at preparation, after a solve what the shared sectors of the process ran), in `result.backend` of `timing.json` with `per_rank[].orbital_sector_ritz_pipeline` of the MPI runner, and `pipeline`, `stage_clock` and `rounds_seconds` of every shared sector in `per_rank[].distributed_state` | not named: it shares no basis | with the row below: SCF -4.4 % (23,768 electrons) and -3.6 % (29,576) on 8 GPUs, -6.4 % (23,768) and -5.4 % (39,368, 231.1 against 218.8 s) on 16; peaks 1.0 to 1.2 GiB lower; same totals and steps |
| `PARSEC_CUPY_EXCHANGE_STRIDED` | `1`: one pitched copy per piece, no kernel, no chunk, no buffer, and none counted by the rule that shares a basis | `0`: pack kernels, an exchange buffer per device and contiguous copies between devices, unpack kernels on the way back, and the buffer counted | same bits | `orbital_sector_exchange_copies` among the backend details and in `timing.json` as above, and `exchange_copies` with `exchange_buffer_bytes` (0 where nothing was packed) of every shared sector | not named: it shares no basis | with the row above |

Before the defaults changed, complete first calculations of 3,480 electrons
on two A100 nodes (8 GPUs, two sectors of two devices per rank,
`benchmarks/mpi_full_scf.py` with `PARSEC_CUPY_DISTRIBUTED_FILTER=1` and the
launch profile of the measured series, 2026-10-21), one with each of `off`,
`p1` and `p2`, each with and without pitched copies, gave the same total
energy (-9437.819950954872 Ry) after the same 10 SCF steps, and every array
of the archive the same bit for bit apart from timings and the backend
details, which differ in the entries of the two switches alone. The records
said what ran: `pipeline`, `stage_clock` `events` under `p1` and `p2`, and
`exchange_copies` `strided` with an exchange buffer of 0 bytes where
601,657,296 were taken otherwise. On four nodes (16 GPUs, one sector of four
devices per rank, whose copies to the rows take the turns of two pairs)
`off`, `p2` and `p2` with pitched copies gave the same bits among them in
the same way (-9437.81995095464 Ry after 10 steps; 16 GPUs sum other pieces
than 8). Those runs shared their nodes with other requests and were not
timed.

A value other than those of the table stops the run where the trial basis is
sized, as do mistyped values of the other switches of the shared basis.
`tests/test_ritz_pipeline.py` takes the step, a first solve with two later
passes, a failed audit and the steps with two and three tall blocks with
every value on one GPU named twice and four times (distinct GPUs where there
are two or four) and compares them with `off` and the packed exchange to the
last bit, the defaults among them; it also drops either wait of the exchange
stream while the compute stream is held busy, and the comparison sees it.
The stand-in of `tests/test_distributed_exchange.py`, whose streams run their
work only when something waits for it, takes the same step on two to four
devices; its tests of the packed exchange and of the rounds one after
another name `0` and `off`.

The Rayleigh--Ritz step of the sectors has a switch since 2026-10-21,
`PARSEC_CUPY_RITZ_HANDOVER`. `rotation`, the default (also unset or empty),
hands the next filter the Ritz vectors `X C` of every pass, as before and bit
for bit. `triangular` hands a later pass of a sector the orthonormal basis
`Q = X L^-T` instead, `L` the Cholesky factor of the overlap `X.T X`, and
keeps the eigenvectors `Z` of the whitened matrix with the saved state: the
Ritz vectors are `Q Z`, and only the few of them that the density needs are
ever formed. Any other value stops the run when the sector solver is built
(`Eigensolvers/handover.py`).

The small solve is that of the rotation up to the eigensolve, statement for
statement, so the Ritz values are those of the rotation bit for bit; it forms
`U = L^-T` instead of `C` and audits it as `C` was audited
(`max |U.T S U - I| <= 5e-10`). `U` is upper triangular, so column `j` of `Q`
combines the columns `0..j` of `X` and the product costs about half of the
rotation: per output slab of `PARSEC_CUPY_RITZ_HANDOVER_COLUMNS` columns (128)
one DGEMM over the leading columns only, 0.51 to 0.53 of the flops at 2,431
to 4,928 columns. `PARSEC_CUPY_RITZ_HANDOVER_PRODUCT=columns`, the default,
forms a slab over all rows in the buffer of the Ritz step and copies it into
its columns, the slabs from the last to the first; where that buffer cannot
hold the widest slab in all rows, and with `tiles`, a tile of rows of all
columns goes through the rotation tile as in the rotation. On one A100 with
the tiles of production (2026-10-21, the row blocks of a shared basis of
39,368 to 19,392 electrons, 1,415,050 x 4,928 to 718,350 x 2,431, and the
one array of 19,392 electrons on four GPUs, 2,873,400 x 2,431) the two cuts
took 0.531 to 0.557 and 0.554 to 0.616 of the time of the rotation; DTRMM per
tile, 0.527 to 0.556, has no CuPy binding and cannot take the dense block of
a shared basis (below).

Leading columns of `Q` span the leading columns of `X`, so the lower half of
the degree split of the next filter (`Chebyshev_Degree_Delta`) spans the
filtered lower half, and the two halves stay apart: mixing them cost an SCF
step (10 instead of 9) in an emulation on small clusters. On one array the order of the factorization
is that of the columns. On a basis that several devices share it is the row
order of the row blocks, except between the first upper-half and the last
lower-half row position, the band where the rounds of the devices bring the
two halves together: there the lower-half columns go first. `U` is then upper
triangular on the row blocks but for one dense diagonal block, the band,
which lies inside one output slab, and every device multiplies its row block
in output slabs of whole rounds in the slab workspace of the step, which the
rounds no longer read. Both routes hand on the same two spans, so the result
does not depend on the number of devices but by round-off.

The first CHEBFF solve hands on the filtered columns as they are in its first
cycle (the values of the whitened matrix only) where the condition number of
their overlap is at most `PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION` (1e4; `0`
never), `X L^-T` otherwise and in the cycles before the last, and rotates in
the last: the first solve ends with Ritz vectors. On one array this needs
`PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ=1`, which the launch profile sets;
cycles on the QR route rotate. The span of the first filter block, which a
CHEBFF cycle filters with a polynomial of its own, stays the leading one; a
shared basis whose first block is not at the first row positions rotates in
that cycle.

On a shared basis the hand-over takes the rounds as `PARSEC_CUPY_RITZ_PIPELINE`
and `PARSEC_CUPY_EXCHANGE_STRIDED` run them (above): only the order in which
the summed pair is factorized and what follows the rounds change. Its product
runs after the rounds, once every compute and exchange stream of the step
has done what they queued, in the slab workspace that the rounds and the
early overlap slabs of `p2` no longer read; it takes no exchange buffer, which
the pitched exchange leaves at none, and its corrections between the passes
run on the compute streams of the device threads, never on their exchange
streams. A hand-over gives the same bits under every value of the two
switches, as the rotation does.

The density of a hand-over basis is that of all its `m` columns less the Ritz
vectors whose occupation is not exactly one, each weighted `1 - f`:
`rho = (2/h^3) [ sum_j |q_j|^2 - sum_S (1 - f_n) |y_n|^2 ]`, `y_n = Q z_n`,
with the columns that are not selected counted as empty. Since `Z Z.T = I` it
is the density of the Ritz vectors to round-off. The set is 13 to 15 vectors
per sector of 39,368 electrons at 500 K and about ten of 3,480. `y` is formed
a piece of rows at a time, at most 128 MiB of it (`handover.CORRECTION_BYTES`),
and handed to the density builder piece by piece: on the device of a sector
one product of its rows, on a shared basis one partial product of those rows
per device, summed on the owner (the `correction` stage of the group). A
shared basis takes these pieces on the streams of its device threads, whose
pools hold the workspace of the Ritz step between passes, so the correction
needs no memory beside the pass. The first form, all rows of `y` at once
summed in a buffer of the calling thread, raised the fullest GPU of a shared
sector by up to 2.84 GiB (39,368 electrons on 16 GPUs, 2026-10-22): every step
that corrected more vectors than those before left another such buffer in the
pool. Where more than `max(64, m // 8)` vectors need a correction
(`PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE`, 8), a basis of one array is
rotated in place into its Ritz vectors, and the later passes of that sector
rotate. The nonlocal forces are linear in the same density matrix and use the
same identity, a few columns of `y` at a time, so relaxations and MD need no
rotation either; two force evaluations of one result give the same bits. The
eigenvectors `Z` of a shared basis stay in the order of its factorization,
with the row of `Z` of every slot beside them (no permuted copy), and a pass
drops those of the hand-over before when it starts, since its filter
overwrites the basis that they describe. The trim of a sector after a later pass is carried
out by its next pass, which filters all `m` columns, rotates and keeps the
trimmed Ritz pairs: that one pass filters the few columns of the trim more
than the rotation does. Where Ritz vectors themselves are read, the export of
the orbitals (`--save-wavefunctions`, `Output_All_States`, the archive member
`wavefunctions`, the default of the Python API) and `Energy_Refinement`,
`materialize_wavefunctions` forms them once, in place, in the memory that
the orbitals of the result view; a sector spilled to the host is rotated a
tile of rows at a time through its device. A shared basis cannot be exported
in any case and is refused there.

The full grid and spin, CHEBDAV first solves, a sector basis that MPI ranks
share (`--rank-per-gpu`, MD across nodes), two or three tall blocks
(`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS`), the two-array Ritz step
(`PARSEC_CUPY_STREAMING_RITZ=0`), CHEBFF cycles on the QR route and QR
fallbacks keep the rotation, and the record says why. That record is
`orbital_sector_ritz_handover` among the backend details: `rotation`, or the
steps of the first solves and of the later passes of all sectors by what they
handed over (`values`, `factor`, `triangular`, `rotation` with the reason),
the deferred trims, the products, the Ritz vectors formed for an export and
the corrections of the densities of the process. It is in `parsec.out` (the
short report too), in `result.backend` of `timing.json`, and with the MPI
runner per rank as `per_rank[].orbital_sector_ritz_handover` beside the
record behind it, `per_rank[].ritz_handover`. The triangular product of every
sector (of a shared one with its way back to the columns, as for the
rotation) is the stage `subspace_ritz_handover_seconds` (a line
`Ritz hand-over` in the text profile where it ran);
`subspace_ritz_rotation_seconds` counts full rotations only, so two runs of
the two values compare the sum of both. A shared sector has the group stages
`handover` (the product with its factor) and `correction`, and `handover_counts`, `handover_band`, `handover_slab_columns`
and `handover_product` in `per_rank[].distributed_state`.

The path of an SCF moves beyond round-off under `triangular`: from the second
pass on the lower degree of the filter meets the span of the filtered lower
half and not that of the lowest Ritz vectors, and the first solve hands on
other bases. On a workstation (RTX 5070 Laptop, 2026-10-21, one GPU, the
launch profile) benzene took 12 SCF steps either way with total energies
1.6e-7 Ry apart, Si28H36 12 either way and 8.7e-9 Ry, and the nanodiamond
C35H36 of `examples/0d_nanodiamonds` with the solver lines of the benchmark
inputs 10 steps against 11 and 1.3e-7 Ry: the first solve alone gave the 10
steps (its residual of step 10, 1.99e-4, fell below the criterion of 2e-4),
the later passes alone 11 steps and 2.8e-9 Ry; C87H76 took 9 steps either way
and 3.2e-8 Ry, with two trims carried out by the pass after them. A
calculation that ends near its criterion can therefore take a step or two
more or less.

| Switch | Default | Other value | Results | Says what ran | Serial control | A100, 2026-10-21 and 2026-10-22 |
|---|---|---|---|---|---|---|
| `PARSEC_CUPY_RITZ_HANDOVER` | `rotation`, also unset or empty: every Ritz step hands over its Ritz vectors | `triangular`: later passes of a sector hand over `X L^-T` and keep `Z`; the first CHEBFF solve hands on its filtered columns, then `X L^-T`, and rotates in its last cycle | `rotation`: the bits of before. `triangular`: for a fixed basis the Ritz values (bit for bit in the first pass from a saved state), the density and the forces of the rotation to round-off; over passes another path: totals up to 2.1e-5 Ry apart in the cases run, within a step at the default criterion and two steps at 2e-6 Ry (39,368 electrons on 16 GPUs) | `orbital_sector_ritz_handover` among the backend details, an unset variable included; `per_rank[].orbital_sector_ritz_handover` and `per_rank[].ritz_handover` of the MPI runner; the stage `subspace_ritz_handover_seconds`, the group stages `handover` and `correction` and `handover_counts` of a shared sector | not named: `rotation` | dedicated nodes (2026-10-22, launch profile, first calculations, alternating, with the pipelined rounds and pitched copies of the defaults), SCF of `rotation` against `triangular`: 219.0 against 198.7 s for 39,368 electrons on 16 GPUs, 197.9 against 180.4 s for 29,576 and 134.2 against 123.8 s for 23,768 electrons on 8, 75.1 against 70.4 s for 23,768 on 16 (6.3 to 9.3 % less); with one array per sector 2.6 % less for 3,480 electrons on 1 GPU and 3.5 to 9.5 % for 10,456 to 19,392 on 4 (14,680 electrons took 10 steps against 11); the peaks within 0.1 GiB of `rotation`. Complete calculations with the launch profile, each value once (correctness runs on shared nodes): 3,480 electrons 10 and 10 SCF steps on 1, 8 and 16 GPUs, total energies 7.50e-7 Ry apart on each, those of `triangular` within 2.1e-9 Ry of one another over the three device counts with the same residuals to four digits; at a criterion of 2e-6 Ry on 1 GPU 16 and 16 steps and 8.2e-10 Ry; 10,456 electrons on 4 GPUs 11 and 11 steps and 1.7e-6 Ry, where the last step of either moved the total by 3.9e-5 Ry. `rotation` gave the totals of the runs before the switch to the last digit on 1, 8 and 16 GPUs. Up to 11 Ritz vectors were corrected per sector density, none fell back; every first solve took `values` in its first cycle. The stage seconds of the Ritz products of 10,456 electrons on 4 GPUs, 1.5 s of last-cycle rotations and 11.2 s of hand-over against 21.6 s of rotations, are of nodes that other jobs shared. Memory (A100 nodes shared with other jobs, 2026-10-22, the fullest GPU against the rotation): with the correction in pieces of rows +8 MiB for 23,768 electrons on 8 and 16 GPUs, 29,576 on 8 and 39,368 on 16 (the first form +876, +876, +1,210 and +2,906 MiB), +40 MiB for 19,392 electrons on 4 GPUs, +156 for 10,456 on 4 and +32 for 7,120 on one. The total energies of 14,680 electrons on 4 GPUs (dedicated nodes: `rotation` 11 and `triangular` 10 steps, 1.9e-5 Ry apart) come from the later passes: with them alone 10 steps and the same total to 1e-10 Ry, with the first solve alone 11 steps and 8.4e-8 Ry, with `FIRST_CONDITION=0` the totals of the default; at a criterion of 2e-6 Ry both values took 17 steps and agreed to 1.1e-8 Ry. The difference at 2e-4 Ry is where the SCF stopped (its last step moved the total by 4.4e-5 Ry), not a bias; so at 39,368 electrons, 2.1e-5 Ry apart in 10 steps either way, where the last step moved the total by 2.6e-4 Ry and the later passes alone gave the same difference. There at 2e-6 Ry the totals agreed to 3.5e-7 Ry, about what their last steps moved them, but `triangular` took 21 steps against 19 |
| `PARSEC_CUPY_RITZ_HANDOVER_COLUMNS` | `128`: columns of an output slab of the triangular product | a positive integer | round-off | `handover_slab_columns` of a shared sector | not read | not varied |
| `PARSEC_CUPY_RITZ_HANDOVER_PRODUCT` | `columns`: a slab over all rows in the buffer of the step, `tiles` where it does not fit | `tiles`: row tiles through the rotation tile | round-off | `product` of every step in the records, `handover_product` of a shared sector | not read | the product alone (a probe on one A100): 0.531 to 0.557 and 0.554 to 0.616 of the rotation |
| `PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION` | `1e4`: the first CHEBFF cycle hands on its filtered columns up to this condition number of their overlap | `0`: never; the first cycle hands on `X L^-T` | another path | the mode and the condition estimate of every cycle in `per_rank[].ritz_handover` | not read | not varied |
| `PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE` | `8`: a density corrects at most `max(64, m // 8)` Ritz vectors before its sector rotates | a positive integer | round-off, another path where it turns a sector to the rotation | `fallbacks` of the density record | not read | no fallback in the runs |

CuPy supports either translated first-SCF path: fixed-cycle CHEBFF or
locking/restart CHEBDAV. Both retain the complete buffered Ritz space on the
device and use the translated saved-SUBSPACE path on later SCF iterations.
The selected first solver is never replaced by CHEBFF, CHEBDAV, SciPy, native
code, ARPACK, or the Fortran executable. Large basis blocks, Hamiltonian
images, residuals, filters, orthogonalization, and Ritz rotations use float64
operations. By default projected symmetric eigensystems of order at most 64
use host LAPACK, which measured faster than repeated small CUDA solver
launches; larger projected problems remain in CuPy. Set
`PARSEC_CUPY_HOST_EIGH_MAX=0` to keep every projected solve on the GPU. Other
deliberate host work is limited to deterministic PARSEC-compatible random-
vector generation, scalar control/diagnostics, and the at-most 8-by-8 Lanczos
tridiagonal solve.

CHEBDAV reuses the float64 Ritz eigenvalues already returned by that host
LAPACK solve for scalar convergence and filter-window decisions. This removes
many one-value CUDA stream synchronizations without changing the projected
matrix, eigenvectors, locking policy, or tolerances. Set
`PARSEC_CUPY_REUSE_HOST_RITZ_VALUES=0` for the explicit device-scalar control
path.

For CHEBDAV operators with at least 100,000 real-space rows, the default
orthogonalizes each appended source-sized block with FP64 block CGS2 against
the existing basis followed by two device MGS passes within the normally
six-column block. This preserves the Davidson span and all residual-locking,
restart, filter, and cleanup rules while replacing synchronized
column-by-column projections against the growing basis with cuBLAS level-3
work. Every block receives a rank/orthogonality audit; an unsafe block falls
back first to Householder QR and then to the replacement-capable literal
PARSEC MGS routine. Smaller symmetry sectors retain literal PARSEC MGS because
GPU setup is not profitable there. Set `PARSEC_CUPY_CHEBDAV_BLOCK_ORTH=off`
for a source-arithmetic comparison, `on` to force it, or change the automatic
row crossover with `PARSEC_CUPY_CHEBDAV_BLOCK_ORTH_MIN_ROWS`.
The literal fallback sweeps the whole Davidson workspace (`wanted + 12`
columns), and the columns beyond the block that no step has used yet are
zero and take one random replacement each. Only the replacements of the
block count towards the limit of 100: before, an early fallback in a wide
workspace stopped the solve ("orthogonalization exceeded the replacement
limit"; the first solve of a 216-atom Si cell at Gamma, 476 working states,
whose 108th block fell back with 325 unused columns). The sweep, its draws and
the bases are those of before, so a solve that passed keeps its numbers.
`PARSEC_CUPY_CHEBDAV_FALLBACK_TAIL=skip` sweeps the block alone: the unused
columns stay zero and draw nothing from CHEBDAV's stream, which may change the
later draws of a solve that falls back (the Si216 cell: the same energy and
levels to the printed digits, 63 instead of 70 s).
The Gram--Schmidt loops copy the growing prefix of a row-major basis into one
reused buffer, where CuPy took one new copy per column and product; that cut
the device peak of such a fallback from 24.5 to 7.2 GiB with the same bits.
The projection against the growing prefix uses the complete contiguous
C-order Davidson workspace for the coefficient GEMM. For the normal blocks
of at most six vectors, a row-major CUDA kernel then applies only the active
prefix and subtracts the update in one pass; larger/user-forced blocks retain
the full-workspace cuBLAS update. This is algebraically the same prefix
projection, but avoids repeated noncontiguous-prefix packing, zeroing inactive
coefficients, and a second full-workspace GEMM. Set
`PARSEC_CUPY_CHEBDAV_FUSED_PREFIX_UPDATE=0` for the former full-workspace
update or
`PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_CGS=0` for the direct-prefix control.
The incremental Ritz projection likewise multiplies the complete contiguous
Davidson workspace by the six new Hamiltonian images and consumes only the
active row interval, avoiding an implicit noncontiguous active-basis copy.
Set `PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_RITZ=0` for that control path.

The exact 48-bit DLARNV random sequence is generated with 2,048 skip-ahead
lanes on NumPy. This preserves every value and the final seed bit-for-bit but
avoids the scalar Python loop that previously dominated initial-basis setup.

For a proved commuting Cartesian-reflection action, the GPU path constructs
every real character expansion `U_Gamma`. For orbit `O_w` with stabilizer
`S_w`, representation `Gamma` contains that orbit exactly when
`chi_Gamma(s)=1` for every `s` in `S_w`; otherwise that orbit is zero in the
sector. On an admitted orbit,
`U_Gamma[i,w]=chi_Gamma(g_i)/sqrt(|O_w|)`. This includes the free-action case
and permits exact representation-dependent dimensions. Static terms are
projected as
`A_Gamma = U_Gamma.T @ A @ U_Gamma` and
`B_Gamma = U_Gamma.T @ B`. Each wedge Hamiltonian keeps an independent
CHEBFF/CHEBDAV-to-SUBSPACE state across SCF iterations. The initial allocation
matches PARSEC's integer policy,
`floor(N_states/N_rep) + Subspace_Buffer_Size`; sectors are grown only when
their last Ritz value does not bracket the globally requested cutoff. The
sector spectra are then stably sorted. During SCF, the selected vectors remain
on the normalized wedge: admitted phases are `+1` or `-1`, and rejected
stabilizer orbits are zero, so squared orbitals give the same scalar density
on every orbit image. The code evaluates that density once on the wedge and expands
only the length-grid scalar field. Signed full-grid orbitals are materialized
once for the final public result. PARSEC-compatible one-based representation
labels are printed and archived. The full-grid GPU Hamiltonian is not
allocated in this path. The kinetic stencil of a sector is built from the
grid (see the cache section above) and all reduced KB factors are constructed
from one canonical sparse gather. Where no cache entry is read, the default,
independent representation
assemblies run serially for at most two representations and on at most four
CPU workers for larger decompositions; the native stencil kernel is itself an
OpenMP loop and takes the sectors in turn. This cold-time policy was selected by
fresh-cache complete-run A/B measurements; set
`PARSEC_SYMMETRY_OPERATOR_WORKERS=N` to override it. Signed-permutation grid
maps use exact integer axis/sign/offset
arithmetic after validating the affine lattice offset once, instead of
rounding a full float64 coordinate transform for every operation. Identical
CUDA stencil kernels are compiled once and shared across sectors. A nonblocking-stream
scheduler is implemented for independent sectors, but measurements showed
bandwidth/compute contention on one GPU; the single-device default is
serialized.
Set `PARSEC_CUPY_SECTOR_SCHEDULER=streams` to profile overlap and optionally
limit it with `PARSEC_CUPY_SECTOR_STREAMS=N`. An MPI rank that solves several
sectors of a group of ranks sharing their bases (the runner's `--rank-per-gpu`
launch with `Eigensolver: chebff`, or one rank per node under
`PARSEC_MPI_NESTED_SHARING`, each sector then over all its GPUs) solves them one
after another under either value: their collectives share the communicator of
the group.
Independent one-vector representation Lanczos bounds can also be profiled
with `PARSEC_CUPY_COLLECTIVE_LANCZOS=1`. It is off by default because it was
numerically identical but slower on the measured GPU; large filters remain
serialized regardless.

On a multi-GPU host, `PARSEC_CUPY_DEVICES=auto` (the default) gives the
representations to the visible devices and solves independent sectors
concurrently. Since 2026-11-19 one process places them as the MPI runner
places the sectors of one rank (`PARSEC_CUPY_SECTOR_PLACEMENT=size`,
`Eigensolvers/symmetry.process_sector_device_groups`): by their cost (rows
times first states), the costliest sectors with the most devices; where the
sectors do not divide the devices and a CHEBFF first solve divides every basis
(rows x states^2 at the threshold of the generalized Ritz step), every sector
over all the devices, solved one after another (the device pool,
`PARSEC_SECTOR_DEVICE_POOL`), unless the mean sector would give each device
less than `PARSEC_CUPY_SECTOR_POOL_MIN_BYTES` (24 MiB) of basis there, where
the shared bases cost more time than the imbalance saves; and the one sector
of the identity
(`PARSEC_CUPY_SINGLE_SECTOR=on`) shared by all of them. A hybrid functional,
a meta-GGA and LDA+U keep the cyclic deal (their exchange, kinetic-energy
density and occupations read every basis on one device), and so does
`PARSEC_CUPY_SECTOR_PLACEMENT=cyclic`: sector s on device s mod (devices), as
before. Since 2026-11-21 a periodic cell whose first sector bases total less
than 512 MiB keeps every sector on the first GPU of the process
(`PARSEC_PERIODIC_SECTOR_DEVICES`): spread over four A100 every cell measured
from 8 to 1728 atoms was slower, its first run about twice as long (each
GPU compiles the kernels of its sectors). The backend detail `orbital_sector_placement` names the placement.
Use `current` to restrict execution to the current
device, or a comma-separated list such as `0,1`. These are logical indices
inside the scheduler's allocation: never overwrite `CUDA_VISIBLE_DEVICES`.
With the default sequential scheduler, each GPU runs one sector at a time;
different GPUs run concurrently, including when sector sizes are unequal.
Static operators and saved subspaces stay on their assigned GPUs. Each sector
forms its density on that same GPU, including non-contiguous orbital views,
then downloads only a grid-length density vector (a shared basis: each of its
GPUs for the columns it holds, summed in device order; an export of the
orbitals copies those columns, `DistributedBasis.column_array`). Global eigenvalue ordering,
occupation assignment, and the FP64 scalar SCF remain shared and unchanged.

#### Weighted sectors of a larger subgroup

PARSEC solves the orbitals in the one-dimensional representations of an
abelian subgroup of the point group (D2 for the nanodiamonds, whose full group
is Td with 24 operations). Its two- and three-dimensional representations
then repeat in several sectors: the three partners of a T level lie in B1, B2
and B3 of D2, the two of an E level both in A. `Symmetry/orbital_groups.py`
chooses instead, from the full group `G` of signed permutations that maps the
atoms and the grid onto themselves (its characters by Burnside's method, its
subgroups by enumeration), the subgroup `H` whose real one-dimensional
characters cover every representation of `G` at the least cost (rows times
states; an unpolarized periodic cell of a few atoms adds a fixed cost per solved
sector, `PARSEC_SECTOR_OVERHEAD`, and solves fewer, larger sectors of the same `G`).
Since 2026-11-21 its group theory (the multiplication table, the classes, the
one-dimensional characters of every subgroup and their plans) is array
operations that give the plan of the former loops entry for entry
(`tests/test_orbital_group_arrays.py`): the plan of the 384 operations of a
rock-salt, fcc or perovskite 2x2x2 cell took 3-5 s of its preparation (8.8 s
under a profiler on A100), now about 0.3 s. Only those sectors are solved; a state of sector `lambda` counts
`w_lambda = d_Gamma / n1(Gamma)` times for every `G`-representation `Gamma` it
holds (`n1` the number of one-dimensional components of `Gamma` restricted to
`H`), and a subgroup is admitted only where that weight is the same for every
`Gamma` of the sector. Sectors exchanged by the normalizer of `H` are solved
once. For Td it is D2d (S4 along a coordinate axis): sectors A1, B1, A2, B2 of
about `N/8` rows and `M/8` states each, weights 1, 1, 3, 3 (A2 and B2 hold one
partner of every T1 and T2 level, the two others lie in the E of D2d, the
threefold image of these), a quarter of the basis of D2; for Oh it is D4h.
The sectors are built from orbits of `H` with the same expansion `U_Gamma` as
above (`Symmetry.PointGroupReduction`). The eigensolver counts states by
weight, the Fermi level takes the weights (`N = 2 sum w f`), the density of
the solved sectors is averaged over the right cosets of `G` (which adds the
partners that no sector holds), the forces are averaged over `G` as before
and their nonlocal term before it is reported, and the SCF record lists every
partner of a weighted state, as do the eigenvalues, occupations and orbitals of
the result. Since 2026-11-22 a run of Fermi occupations of one channel counts
`States_Num` states: the level that the window ends in shares the states left
by weight (`PARSEC_WEIGHTED_STATE_COUNT`); the lists then hold `States_Num`
states, such a level the whole states of its shares, each with the occupation
of the shares under it. Under `off`, with `occup.in` and in the
spin channels a run can list more levels than `States_Num`, every partner of
its last multiplet (H2 on the z axis in D4h: 17 levels for 16 states,
`examples/h2_canonical_nodg`, before 2026-11-22; Fortran PARSEC lists
`States_Num`); the two
channels of a spin run may then list different numbers of levels, and the
result archive pads the shorter lists of `spin_eigenvalues_ry`,
`spin_occupations` and `spin_representations` (NaN levels, zeros) and names
the counts in `spin_state_counts` (since 2026-11-21; before, such an archive
could not be written: V- in diamond 3^3, nanodiamond k04). A calculation that reads every partner of every
state keeps the abelian group: spin-orbit coupling, hybrid functionals,
meta-GGAs and LDA+U. Since 2026-11-19 four inputs that kept it before take
the weighted sectors (`Symmetry/weighted_inputs.py`, a switch each):
`Fix_Number_Eigenvalues` (the sectors keep the levels they hold among the
lowest `States_Num` counted by weight); a libxc LDA or GGA (the device
evaluator of libxc reads the orbit fields with the operation table of the
code's GGA stencil); `occup.in`, whose n-th entry goes to the n-th level
counted with its partners, so that the `w` partners of a weighted state take
`w` equal consecutive entries, as the degenerate partners of a level that
several solved states hold must (the e of Td in two sectors of D2d; unequal
ones send the run from the atoms to the abelian sectors, with a `NOTE`); and
collinear spin in the sectors of a
cluster or a Gamma cell, with the weights in the common Fermi level, the
occupations, the density of each channel and its forces. The density of a
channel is averaged over `G`, so the potentials keep every operation of `G`
and the density test of the spin guard sees nothing outside `H`; a partly
filled multiplet shows in the levels instead, and the guard counts a state of
weight `w` as `w` partner levels in distinct sectors (a Td triplet held by one
state of D2d with one electron is a partly filled multiplet of three levels,
as the same triplet in three sectors of D2 is). A spin run the guard stops
starts again in the abelian sectors, whose own guard may send it to the full
grid; since 2026-11-21 a cluster, or a cell whose operations take no
translation (a defect at the origin of a symmorphic cell), goes straight to
the full grid where the abelian sectors would stop the same levels (for a stop
at the start exactly; for a stop at the end as predicted from the weighted
occupations, the full-grid result either way): the reported multiplet holds a
state whose representations of `G` all have partners in two or more abelian
sectors (the t1 and t2 of Td in D2; not the e, whose partners lie in A of D2,
where they mix freely: `PARSEC_WEIGHTED_GUARD_SHORTCUT`). Since 2026-11-21
`occup.in` and spin take the weighted sectors on several MPI ranks too
(`PARSEC_ORBITAL_GROUP_MPI`; before, they kept the abelian group there): a
run that leaves them is prepared again in the abelian sectors on every rank,
which the root tells to do so (the `reprepare` command with the reason).
`PARSEC_ORBITAL_GROUP=abelian` keeps it everywhere (bit for bit as before
2026-11-18 with `PARSEC_SECTOR_PROJECTOR_ORBITS=weighted`, and for a periodic
cell with `PARSEC_PERIODIC_ABELIAN_TRANSLATIONS=off`); the symmetry
caches keep no entry of the weighted sectors. GGA fluxes reach the points of
an orbit through the permutation and signs of the operation that carries the
representative there (`V_xc/device_xc_kernels.cu`).

Measured on A100 (first runs): ten Td molecules (CH4, SiH4, CF4, CCl4, P4,
neopentane, adamantane; PBE and CA; 0.2 A, SCF to 1e-7) give the total energy
of the run without symmetry within 3e-12 Ry and its distinct levels within
3e-7 Ry; with positions stretched by 3 % the forces agree to the six digits
printed. The nanodiamond k06 (1 GPU, SCF to 1e-7): 4.4e-9 Ry from the run
without symmetry (D2: 9.3e-10), 14.3 s and 7.25 GiB against 31.6 s and 11.2
GiB with D2 and 145.2 s and 39.5 GiB without symmetry. On 16 A100 (one rank per
node, the benchmark inputs, same nodes): k12 73.9 -> 29.5 s and 25.5 -> 10.6
GiB at the fullest GPU, k13 111.1 -> 37.3 s and 33.9 -> 12.6 GiB, k14 207.5 ->
63.6 s and 59.7 -> 20.5 GiB (its eigensolves 180.4 -> 38.9 s); k15
(C11487H1804, 13,291 atoms, 47,752 electrons), which D2 does not fit on 16 A100
(about 79 GiB of basis per GPU), 94.9 s and 28.7 GiB in 10 steps; the energies
within the scatter of that SCF criterion (2e-4 Ry: up to 1.3e-4 Ry between the
two routes). The sectors of D2d do not hold even shares of the lowest levels
(the A1 and T2 sectors of k14 about 4 % more): each starts 10 % above its share,
where an even start failed the spectral bracket and solved two of them again
(`PARSEC_WEIGHTED_FIRST_SHARE=orbitals`, and its default `auto` in one
process from a full-grid basis of 2^27 entries on since 2026-11-21, starts
each 3 % above the share of the valence orbitals of the atoms it holds
instead; on one node the nanodiamonds k06 to k10 then take the device pool
of the four GPUs for every sector and ran their SCF 1.5 to 2.4 times faster
on first runs).

#### Units of the irreducible representations (`PARSEC_ORBITAL_IRREPS`, default off)

A level of a representation `Gamma` of `G` lies in every solved sector whose
character its restriction to `H` contains: the E levels of Td in both A1 and
B1 of D2d, a sixth of the states solved twice. `PARSEC_ORBITAL_IRREPS=once`
solves every real irreducible representation of `G` in one sector only
(`Symmetry/orbital_groups.irrep_units`: the sector of the least multiplicity
`m`, not the totally symmetric one where another holds it), where its states
count `d / m` times; a sector that keeps only some of the representations it
holds becomes one unit per weight, on the rows of that sector, and its basis is
projected onto the kept representations after every Chebyshev filter (CHEBFF
first solve and SUBSPACE passes, one device, a group of devices or of MPI
ranks; `Eigensolvers/irrep_projection.py`: `P = sum_c b_c T_c` over the right
cosets of `H`, `b_c = sum_Gamma (d_Gamma / |G|) sum_h chi_Gamma(h c) lambda(h)`,
applied to the sector coefficients by one gather kernel per coset). Td: five
units A1, A2, E, T1, T2 of `M/24, M/24, M/12, M/8, M/8` states with weights
1, 1, 2, 3, 3 instead of four sectors of `M/8`: 1.2 times fewer rows times
states and 1.5 times less Rayleigh-Ritz work; the largest unit is the T
sector of before, so the peak memory of a sector and the time of a run whose
GPUs each hold one sector do not fall (the peak can rise a little: a unit that
grows in its first solve, more units in the device pool). The first counts follow the expected
share of every unit (`sector_shares`, with the margin widened as
`sqrt(largest share / share)`), the Fermi level, the density (averaged over
the cosets as before) and the forces take the weights, and the record lists
every partner as before. `split` also gives every representation of one
weight its own unit (the 3-dimensional ones of the 192 operations of an even
diamond cube; `once` there keeps the 10 sectors of before, which already hold
each representation once). The units of a rank take disjoint GPUs (not the
device pool, `PARSEC_SECTOR_DEVICE_POOL`) where they fit whole on single
devices (`Eigensolvers/symmetry.units_fit_whole`: the mean of the bases of the
units that the rank solves over its GPUs plus twice the largest, at the first
counts, within `PARSEC_CUPY_SECTOR_MARGIN_BUDGET` of the smallest GPU; the
units of the rank from the placement by cost, `planned_owned_units`), larger
units follow the rule of the sectors, and one process without MPI places them
by cost instead of in turn (`Eigensolvers/symmetry.unit_device_ids`).
CHEBDAV first solves (the input default, since 2026-11-22) project every
block they filter, the random start and the random candidates included, and
check each block they append after its orthonormalization as below (projected
and orthonormalized against the columns before it again where its share
outside the range exceeds 1e-7 or a column was drawn at random;
`Eigensolvers/chebdav.py`); ARPACK and the dense first solve, which filter
nothing, are refused before the SCF; `off` (the default) keeps the
sectors of before bit for bit, as do `once` and `split` where the units are
the sectors.

Rounding takes a unit's states slightly off the range of its projector
between two filters: an orthonormalization divides by what remains of nearly
dependent columns, the generalized Rayleigh-Ritz step whitens the filtered
block itself, and PARSEC's selective Gram-Schmidt draws random columns of the
whole sector for dependent ones. A coarse grid with a high first-filter degree
(Ge5, Td, at 0.7-0.8 bohr with degree 90) left 2e-7 to 9e-6 of a combination
of the states outside the range after a solve, which stopped these runs
(review of 2026-11-20), and up to 9e-3 after an orthonormalization (0.8 bohr,
60 states, in a unit of a few states); the
spacings 0.5-0.6 and degrees up to 60 leave 1e-15. Repaired as below, these
runs give the total energy, the occupied levels and the forces of `off` to
9e-12 Ry, 4e-9 Ry and 2e-10 Ry/bohr on A100, as the units do at degree 30,
where nothing is repaired (the Hartree term of the units and of `off` differs
by up to 5e-8 Ry on that grid at either degree). The share is measured with
one combination of the states (one product over the basis and the projection
of one vector) after every orthonormalization of a unit on one device or of
the gathered basis of a group of devices, and above 1e-7, or where a column
was drawn, the basis is projected and orthonormalized again
(`irrep_projection.orthonormal_in_unit`); ranks that share a basis project
their columns after the shifted Cholesky QR of the fallback, which draws
nothing. After a solve on one device, the saved states with a share above
1e-7 are projected back in place and orthonormalized symmetrically (Loewdin),
which moves them by the square of the share; beyond 1e-3, far above rounding
and below the `1/sqrt(n)` that one random column among `n` states shows, the
run stops (`check_unit_basis`). A leak of a basis that devices or ranks share
is removed by the projection after the next filter and changes the energy of
that step by the square of its share. Runs whose shares stay within 1e-7 (all
the measurements below) are unchanged bit for bit.

Measured on A100 (first runs, the benchmark inputs, SCF criterion 2e-4 Ry;
eigensolve seconds of the SCF, `once` against `off`): k08 on one GPU 13.3 ->
12.7 s (-5 %) at a peak 4.7 % higher, 12.1 -> 12.7 GiB (the A1 unit, 27 % above
its share, grew once in the first solve); k10 on one GPU 58.8 -> 49.3 s (-16 %;
first solve 23.3 -> 19.4 s, the later steps 1.19 times faster: filters 21.6 ->
19.4 s, Rayleigh-Ritz 13.5 -> 10.0 s) and 42.8 -> 39.6 GiB; k10 on four GPUs
of one rank 16.4 -> 16.3 and 16.5 s in two runs (neutral) and 12.7 -> 12.1 /
12.0 GiB (19.9 s and 17.2 GiB while the five units took the device pool); k10
on two ranks of four GPUs 9.44 / 9.47 -> 9.81 / 9.80 s in two runs each (+4 %,
the first solve 3.5 -> 3.8 s; the T units on one rank as the T sectors before,
A1, A2 and E on the other) at 8.1 GiB both; k14
on 16 GPUs (four ranks of four) 36.1 -> 36.1 s at 18.4 GiB both,
the T1 and T2 units on two ranks as their sectors before; k18 (21,951 atoms)
on 16 GPUs 207.1 -> 207.1 s (program 253.0 -> 253.6 s) at 70.0 GiB on the
fullest GPU both (the ranks of the A1, A2 and E units 49.1 instead of 69.3
GiB); k08 on eight ranks of one GPU each (sector groups of ranks) 3.5 ->
4.5 s, the five units in groups of 2, 3 and 3 ranks against four sectors of
two. `split` on
the diamond cube of 64,000 atoms (Gamma, 192 operations) on 16 A100: SCF
208.7 -> 170.0 and 175.9 s in two runs with the same energies (-19 and -16 %;
first step 144.6 -> 120.0 / 125.7 s, the later ones 9.5 -> 7.1 s), the peak
1.4 % higher (46.0 -> 46.6 GiB on the fullest GPU), the 20 units in the device
pool of their ranks; `once` there is `off` bit for bit. The total energies agree
within 1e-5 Ry for the clusters and 3.4e-5 Ry (5e-10 Ry per atom) for the
cube, the final occupied levels within 4e-5 and 2.7e-4 Ry (the scatter of the
SCF criterion); for a distorted Ge5 cluster (Td, with forces) and the Si8
cell (Td, and 96 operations with the fcc translations) the GPU tests give the
energies, levels and forces of `off` to 2e-8 Ry, 1e-7 Ry and 2e-6 Ry/bohr, and
the CPU tests the full-grid spectrum from the units of CH4 (Td), rock salt (Oh)
and diamond cells (Fd-3m) to 1e-10.
The gain needs devices that hold several sectors (one GPU for a cluster, the
ranks of a 192-operation cube); with one sector per GPU or per rank it is
nil. With `PARSEC_ORBITAL_IRREP_ROWS=sector` the units hold the rows of their
sectors, so the peak memory does not fall; the partner rows below (the default
since 2026-11-22) give the units that are sectors of a larger subgroup the rows
of that subgroup. A unit whose representations have no function on the grid (the range
of its projector is empty: the 192 operations of a rock-salt cube on 8 to 10
points per edge) is refused before the sectors are placed, under both row
settings, with a message that names the grid (`irrep_units_refusal`; the trace
of every projector is counted from the orbits that its cosets map onto
themselves, `ReflectionRepresentationDecomposition.projector_trace`). The Fermi
level printed for an insulator (mid-gap) moves by a few mRy, since the
lowest empty level of the A1 unit converges further than in the A1 sector of
before (k08 -0.19341 -> -0.19552 Ry with the highest occupied level the same
to 2e-6 Ry, k18 -0.16674 -> -0.17162 Ry). It stays off by default, measured
again on 2026-11-22 with the partner rows of the coset images, the default of
`PARSEC_ORBITAL_IRREP_ROWS`, and with CHEBDAV (whole first runs on A100, no
symmetry cache, the CuPy kernel cache warm unless cold is said, `off` ->
`once`). The units are faster where one GPU holds the sectors of a large
cluster: nanodiamond k06 with CHEBDAV 17.5 -> 15.9 s (three runs each; 27.4
-> 25.9 s with a cold kernel cache; live device memory 6.0 -> 4.7 GB), k08
18.4 -> 15.9 s (23.7 -> 22.3 s cold; 12.3 -> 10.5 GiB), k10 66.3 -> 51.1 s
(43.6 -> 35.8 GiB), Si795H300 with CHEBDAV 28.7 -> 20.9 s; converged to 1e-6
Ry, k08 25.3 -> 21.3 s and Si795H300 20.2 -> 17.8 s, in 17 and 23 SCF steps
on both routes, the energies within 1.5e-8 Ry. They are slower on the small
clusters of one GPU (with CHEBDAV k02, 71 atoms, 2.97 -> 3.35 s; k03, 163
atoms, 3.45 -> 4.12 s; k05, 663 atoms, with CHEBFF of two cycles of degree
15, 9.19 -> 10.74 s in 29 and 42 SCF steps; its first ten steps swing by
thousands of Ry on every route, `off` recovers after the tenth and the units on
the rows of their sectors diverge and stop after 47 steps, the spectral bracket
of the sectors lost, as on the code of 2026-11-21; k04, 309 atoms, was faster,
5.48 -> 5.27 s),
where the SCF criterion of 2e-4 Ry stops a route a few steps earlier
(Si795H300 with CHEBFF 13.37 -> 13.84 s in 12 and 16 steps), on the cells of
one GPU (Si of 216 atoms 3.58 -> 3.68 s; under `once` the Si cells, bcc metals
and the Oh cells take units, rock salt, fcc and zinc blende none), and on
several GPUs or ranks, where every device holds one sector or a share of one
(k10 on the four GPUs of one process 24.4 -> 24.3 s, on two nodes 15.5 ->
16.2 s, k14 on 16 GPUs 50.6 -> 50.7 s, k08 on eight ranks of one GPU 6.6 ->
7.2 s, k06 with CHEBDAV at 1e-7 Ry on four ranks of one GPU 10.0 -> 11.1 s,
the energies of one rank, four ranks and two nodes within 8e-9 Ry). `split`
is slower than `off` on the cubes of one node (the switch table; also its
later SCF steps, diamond of 8,000 atoms 0.75 -> 1.1 s). Until 2026-11-22
CHEBDAV was refused. On the cubes
of MgO and Si of 216 to 1000 atoms (one GPU, SCF to 1e-6 Ry) the 20 units of
`split` are launch-bound and 1.1 to 1.5 times slower than the 10 sectors, up
to 2 times with the SCF steps that a unit adds where it grows mid-SCF. Those
figures are of a growth from random vectors (now
`PARSEC_CUPY_SECTOR_WARM_GROWTH=off`), whose occupied levels start unconverged
(MgO 512 and 1000 atoms: 16 -> 25-30 and 19 -> 22-24 SCF steps, the energy
within 1e-8 Ry; the sectors of `off` did not grow in these runs). A unit that
grows now keeps its refined vectors (the units take CHEBFF; MgO 1000 atoms:
24 -> 16 steps).

**Partner rows** (`PARSEC_ORBITAL_IRREP_ROWS=partner`, the default since
2026-11-22; `sector` keeps the rows of before). The
projector of a unit, `sum_c b_c T_c` over the right cosets of `H`, is that of
the one-dimensional sector `mu` of a larger subgroup `K = union_{c in S} H c`
exactly where its coefficients are `mu(c) / |S|` on the cosets `S` of a
subgroup and zero elsewhere: two orthogonal projectors with the same range are
equal (`Symmetry/orbital_groups.subgroup_sector_cosets`, checked with the group
table: the cosets close into a group and the signs are a character of it).
Such a unit holds one partner row of each of its representations, and the
other rows are its images under the cosets of `K`. A1 and A2 of Td (`b = (1,
+-1, 1) / 3` over the three cosets of D2d) are the sectors of Td itself; E
(`(2, 1, -1) / 3`) is none, since Td has no real one-dimensional character
of order 12 whose induced representation is E, and keeps the rows of its
sector with the projector. For the 192 operations of an even diamond cube
(and of the MgO and NaCl cubes) `split` gives 12 units of weight 3 that are
sectors of subgroups of 64, 4 of weight 1 sectors of the whole group and 2
E-like units that are none: 1.55 times fewer rows x states and 1.36 times less
Rayleigh-Ritz work than the units on the rows of their sectors, the filters
included (Td: 1.15 and 1.06, the A units being the smallest).

A unit on partner rows (`ReflectionRepresentationDecomposition.partner_rows`)
holds the isometry `E` from the sector of `K` to the coefficients of its
sector of `H`, `P = E E^T`: an orbit of `K` is the union of the orbits of `H`
that the cosets of `K` carry onto each other, named by the smallest of its
coset images (`partner_orbit_classes`, one minimum over `[K:H]` rows of the
coset images, closure checked), and the normalized function of `mu` on it has
the coefficient `e_s = mu(c) lambda(h) sqrt(|O_s| / |O_K|)` on its orbit `s`
of `H`, where a coset `c` of `K` carries `r_s` into the first orbit, `c r_s =
h r_0` (the sign of the coset's coefficient times the phase of that row in
the sector of `H`), and zero on an orbit of `K` whose stabilizer `mu` does not
admit (tested at the first orbit with the cosets that map it onto itself).
The rows `c r_w` are those the detection mapped
(`PointGroupReduction.coset_rows`, kept where units have projectors), so
neither `E` nor the projector's tables map a representative again; the rows
found must be the projector's trace, and the CPU tests check `E E^T = P`
entry by entry (Td, rock salt, diamond cells) and on random vectors (clusters
of Oh, Th, T and D4h). Until 2026-11-21 `E` was read off the projector's
tables through a sparse matrix and its connected components (about 1 s per
unit of 1.3 million rows on A100, 2 s per unit of 2.8 million); now k10 takes
0.6-0.9 s for its two units of 1.3 million rows and k14 1.55 s for its two of
2.8 million (one worker per unit, beside the sectors of `H`). The rows, signs
and the decomposition of `K` are those of before bit for bit (CPU check on Td,
Oh, D4h and Th clusters and rock salt); `|e_s|` is now exact where the tables
left up to 3.5e-13 of rounding, which reaches only the export of the orbitals.
Its own decomposition (`partner_decomposition`: the orbits of
`K` as an orbit map of their own, the phases of the sector of `H` times the
signs of `e`) gives it a stencil and projectors through the builders of every
sector (`Symmetry/operator_cache.build_partner_operators`; the projectors of
one atom per orbit of `H` serve the sector of `K`, whose orbits hold whole
orbits of `H`), and the projector gradients of its forces. The eigensolver
solves it on these rows, with no projector after the filters and so no
corrections of its range; its local potential and its density use, per row, the
orbit of `H` of the first coefficient of its orbit of `K`, which takes the
density of the whole orbit of `K`: every density of these sectors is averaged
over the full group (by the points of each orbit), which spreads it. The
export of its orbitals expands them (`CuPySymmetryOrbitals.sector_expansions`,
`partner_expansion`); exact exchange and meta-GGA keep the abelian group and
refuse it, and LDA+U, which takes the weighted sectors
(`PARSEC_LDAU_WEIGHTED_SECTORS`), keeps every unit on the rows of its sector
(`driver._prepare_single_point`, `keep_sector_rows`, before the units are
placed; a NOTE in parsec.out). The rows of the units are counted from the trace of their
projectors, from the orbits that a coset maps onto themselves, so that every
MPI rank places the units by the same costs without the tables of the units it
does not solve; each rank builds the partner rows of its own units only
(grid-sized maps of `K`), on a thread beside the operators of its sectors of
`H` (whose totally symmetric one, where it is a unit on partner rows, gives the
Poisson stencil alone, without its projectors). `sector` keeps the rows of
before bit for bit. CHEBDAV solves a unit on partner rows as any sector (no
projector), and its E unit with the projections above.

Measured on A100 (first runs, `once`, eigensolves of the SCF, `sector` ->
`partner`): k08 on one GPU 12.8 -> 10.7 s (-16 %; off 13.4 s) and 12.6 -> 10.3
GiB (off 12.0), the A1 and A2 units on 205,222 and 193,098 instead of 603,422
and 591,298 rows (their operators 0.2 s); k10 on one GPU 49.4 -> 43.1 s (-13 %;
off 58.2 s; first solve 19.4 -> 16.8 s, later filters 19.4 -> 16.5 s, Ritz
10.0 -> 9.2 s) at 39.4 -> 35.0 GiB (off 42.6); k10 on four GPUs of one rank
16.5 -> 16.2 s (off 16.1 s) and on two ranks of four 9.8 -> 9.5 s (off 9.4 s),
neutral as the sectors; k08 on eight ranks of one GPU 4.5 -> 3.7 s (off 3.1 s)
at 6.2 -> 4.9 GiB (off 6.8). The total energies agree within 6e-6 Ry (k10)
and 4e-7 Ry (k08) at the SCF criterion of 2e-4 Ry, and those of the same
partner run on one and four GPUs bit for bit. `split` on the diamond cube of 64,000 atoms on 16 A100 (four ranks of four
GPUs, the runner's defaults): 163.8 -> 115.8 s (-29 %; off 200.2 s), the first
step 121.7 -> 81.4 s and the later ones 7.1 -> 5.0 s, at 46.6 -> 32.1 GiB on the
fullest GPU (off 46.0), 16 of its 20 units on partner rows; the total energy
8e-4 Ry (1e-8 Ry per atom) from that of off at the loose criterion of the input
(2e-4 Ry; 8 SCF steps against 7, the energy terms within the scatter of that
criterion), while the 512-atom cube converged to 1e-7 Ry gives the energy of
off and of the sector rows to the 1e-8 Ry printed. A unit keeps the Ritz route
of its sector of `H` (`rayleigh_ritz.ritz_route_rows`): the work threshold of
the generalized Ritz step, measured on sectors, sent the units of halved rows of
a 1728-atom diamond cube to the QR route at twice the time per call. On cubes
whose units are launch-bound (9 to 80 states, first runs, SCF): MgO of 1000
atoms on one GPU 17.3 -> 15.9 s (24 and 22 steps; off 11.2 s) at 2.3 -> 1.7
GiB, diamond of 1728 atoms 11.1 -> 11.1 s on one GPU and 21.5 -> 22.0 s on four
GPUs (8 and 7 steps; the later steps 0.38 -> 0.37 s; the first solve of the
units on partner rows 0.4 to 1.1 s slower, in cold runs), Si and MgO of 216
atoms 5.2 -> 5.1 s and 6.3 -> 6.0 s, the energies within 1e-8 Ry.
Over whole runs, the preparation included, the partner rows of the tables
(2026-11-21) were not faster everywhere: k10 on two ranks of four GPUs 16.6 ->
16.9 s, k14 on 16 GPUs 50.7 -> 52.7 s (the rank of the A1 and A2 units spent
1-2 s per unit on them), the diamond cube of 1,728 atoms on the four GPUs of
one process 30.1 -> 30.4 s cold, so `sector` was the default. With the rows
from the coset images (2026-11-22) every case measured is faster on partner
rows (whole first runs, `sector` -> `partner`, the figures of the switch
table): k10 on two nodes 16.5 -> 16.2 s, k14 on 16 GPUs 51.0 -> 50.7 s (1.0 s
waited for the partner rows on the rank of the A1 and A2 units, 1.8 s with
one worker), the 1,728-atom cube 19.5 -> 19.1 s cold and 8.4 -> 7.2 s warm,
k08 and k10 on one GPU 18.0 -> 15.9 s and 57.9 -> 51.1 s, and `partner`
became the default.
The GPU tests give the energies, levels and forces of the sector rows and of
`off` for Ge5 (Td, one GPU and the GPUs of a node), on a coarse grid with a
high first-filter degree and for the Si8 cube with its point group and with 96
operations, and
the occupied subspace of the exported orbitals to 1e-6; the CPU tests
`E^T E = I`, `E E^T = P`, the expansion, the density and the spectrum of the
sector of `K` against the projected unit (CH4, rock salt, diamond cells).

The Hartree problem of these sectors has the group of the orbitals, whose
operations permute the axes, and the GPU kernels of the boundary take only
axis reflections. The boundary is therefore built on the wedge of the diagonal
members of the group (`Symmetry.diagonal_subgroup_reduction`; D2 for D2d, the
wedge of the abelian route): the density per orbit of the group is gathered
onto that wedge, the multipoles and the right-hand side come from the wedge
kernels, and the right-hand side of the group is `sqrt(m) b` at its own
representative rows (`Hartree.cupy_boundary.CuPySubgroupMultipoleBoundaryBuilder`).
Before, these sectors took the kernels under the identity on the full grid
with the full-grid maps of the resident chain, about 100 bytes per grid
point on the device of the Hartree objects: at C19395H2556 (21,951 atoms,
80,136 electrons, sphere 44.9 Angstrom) on 16 A100 that device ran out of memory
in the first Ritz step. On the wedge its boundary arrays take 0.44 GiB, and the
run took 279.4 s, 18.8 s of it in the Hartree solves, at 70.0 GiB on the
fullest GPU (with `PARSEC_HARTREE_BOUNDARY_BACKEND=native` it had taken 709.1 s,
449.8 s in the Hartree solves, which repeated the full-grid recurrences there).
The right-hand side equals the projected one of the full grid to round-off
(test); `PARSEC_HARTREE_DIAGONAL_BOUNDARY=off` keeps the full-grid kernels.
Since 2026-11-19 the boundary values are evaluated only at the exterior
points of the stencils of the rows of the Hartree wedge itself, and its
right-hand side is assembled there (the moments are still summed on the
wedge of the diagonal subgroup): for the orbits of Td a sixth of the exterior
points of D2 (`PARSEC_HARTREE_BOUNDARY_ROWS=wedge`; `diagonal` evaluates them
for every row of the diagonal wedge and gathers the right-hand side from it,
bit for bit as before).

The density of these sectors is averaged over `G` every step, so it and every
potential built from it are invariant under all of `G`. Since 2026-11-19 the
scalar fields keep one value per orbit of `G` instead of `H`
(`PARSEC_SECTOR_FIELD_GROUP=full`, `Symmetry/field_orbits.py`): a `G`-orbit is
the union of the `H`-orbits that the right cosets carry the representative
into, `[G:H]` times fewer values (Td over D2d: about 3; the 192 operations of a
diamond supercell over 32: about 4.4, the orbits of the axes and planes being
smaller). The density of the sectors goes from the orbits of `H` to those of
`G` by the weighted average over each `G`-orbit, which is its average over
`G`; the mixer, the residual norms, the energy integrals, the XC evaluators on
orbits and, for a cluster, the Hartree conjugate gradients and their boundary
run on the orbits of `G`, whose finite-difference operator is built from the
stencil of the grid (`field_negative_laplacian`, the C++ kernel of the sector
stencils where its byte palette holds the coefficients, CSR otherwise); the
sector eigensolver and the MPI ranks receive the local potential per orbit of
`H` (every orbit of `H` takes the value of its orbit of `G`). `subgroup` keeps
the orbits of `H`; with `PARSEC_SECTOR_INPUT_GROUP_AVERAGE=off` as well (below)
they are the fields of before bit for bit. The device XC evaluators turn the
gradients on those orbits by a byte per grid row, an index into the table of
the distinct rotations of the group (at most 48 signed permutations, the
translations turning no vector), so a group of any order fits: a doubled
cubic cell, Oh with its 8 half-cell translations, has 384 operations.

The weights of the sectors assume a Hamiltonian invariant under `G`. A
structure accepted within a loose `PARSEC_SYMMETRY_ATOM_TOLERANCE` is not: the
fields that enter the SCF from the full grid (the local ionic potential, the
start density of SAD, ML, `Initial_Density` or `Restart_Run`, the core
density, the fields of a full-grid evaluator) are therefore averaged over the
orbits of `H` and then over the cosets of `G`
(`PARSEC_SECTOR_INPUT_GROUP_AVERAGE=auto`; with `PARSEC_SECTOR_FIELD_GROUP=full`
the orbit average of `G` is that average already), so that the sectors solve
the local Hamiltonian averaged over `G` and the energy differs from that of
the structure at second order in its distance from the group. The projectors
are those of every atom where the atoms keep `H` only to more than 1e-6 bohr
(`PARSEC_SECTOR_ORBITS_EXACT`), of one atom per orbit of `H` otherwise (no
average over `G`), and forces and moves still need the atoms on the group to
1e-6 bohr. For atoms that are images in every bit the averages change the fields
by round-off only; `off` keeps the average over `H` alone, as before.

The margin of the first counts is weighed against the devices
(`PARSEC_CUPY_SECTOR_MARGIN=auto`): every process finds, for the sectors it
solves, the largest share of the margin whose counted states
(`_state_fit_bytes`, after one growth of 2 %) take at most
`PARSEC_CUPY_SECTOR_MARGIN_BUDGET` (0.82) of each device, and the root takes the
least share of the ranks of a sector. Where the full margin fits, nothing
changes. A sector whose margin was cut and which then misses states grows by
its estimated deficit, at most 2 % a step, in a new first solve: its basis
leaves the devices first (to the host, where a CHEBFF first solve starts from
its refined vectors, `PARSEC_CUPY_SECTOR_WARM_GROWTH`; given back otherwise)
with the cached blocks of the memory pool, where the growth of before held the
basis beside half as many columns more. The deficit comes from
the sector's density of states between the values at a half and at four fifths
of its count, which a first solve has converged, carried to the cutoff at which
the weighted counts of all sectors reach the requested states. With the margin
cut to nothing on purpose (`PARSEC_CUPY_SECTOR_MARGIN_BUDGET=0.000001`): k06 on
one A100 to 1e-7 Ry gave the energy of the run with the margin within 2.8e-9 Ry
(4.64 against 4.79 GiB), k10 on 16 A100 within 3.2e-5 Ry at the benchmark
criterion of 2e-4 Ry (14 steps against 10; 22.1 against 14.0 s, 5.35 against
5.47 GiB; these are growths from random vectors, and from the refined vectors
k10 takes the 10 steps of the full margin, within 2.9e-6 Ry of its energy). On 16 A100 of 80 GB the cluster cut at 1.03 times the radius of
C19395H2556 (C21003H2700, 86,712 electrons, the same domain, 8 % more basis) got
through the first solve with its margin cut, but its A1 sector, which
holds about 3.5 % more than an even share of the lowest levels, could not grow
within the devices (the rule that shares a basis refused the second step, and with
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=0.88` a Ritz step ran out of memory):
C19395H2556 stays the largest of the series there.

The larger group of these sectors is detected, and its representations
decomposed, on the thread of the symmetry maps beside the reference
preparation (`PARSEC_OVERLAP_SYMMETRY_SETUP`), where the group of the abelian
route was decomposed before and the larger one then in line: the same
reduction and decomposition, 1.3 s less set-up for C13667H2020 on 16 A100 and
2.0 s for C19395H2556. The backend summary of the output names
the weighted sectors beside the symmetry block of the reference, which lists
PARSEC's Abelian subgroup. Since 2026-11-19 its `symmetry_detection` line is
in the short report (`Output_Level: 1`) for every route: the weighted sectors
of a cluster or of a cell with translations, an abelian group with or without
translations, the identity sector, or why the orbitals keep the full grid, so
that a group that was lost or reduced (a defect off the origin, the palette
ladder) shows; before, the short report named the weighted sectors of a point
group without translations only.

Requested final host wavefunctions are expanded in bounded column blocks on
the CPU, without gathering all orbitals into one GPU's memory. The final host
array still needs `8 * grid_points * states` bytes. Ordinary CLI runs do not
request this large array unless wavefunction output is enabled.

Through `main.py` this is **symmetry-sector parallelism inside one process**,
not spatial-domain or general orbital MPI decomposition: every sector lies on
one device, so the available exact real one-dimensional representations limit
useful GPU concurrency (four for the nanodiamond cases). An asymmetric
calculation without this decomposition remains a single-GPU solve. Extra
devices do not automatically accelerate it. Native preparation and Hartree
work use the CPU/OpenMP backend under `--backend auto`; do not launch
multiple copies of the Python driver with `mpirun`. More devices than
sectors, and more than one node, are the MPI runner's
(`benchmarks/mpi_full_scf.py`, one MPI rank per node): a rank divides its
devices among the sectors it owns, and the devices of a sector share its
basis where it fits (`PARSEC_CUPY_DISTRIBUTED_STATE`, above). The 8- and
16-GPU figures of this document were run that way.

For a Slurm example, validation commands, memory rules and timing interpretation,
see [the multi-GPU guide](MULTI_GPU.md).

For the centered finite-difference operator, production CuPy transposes each
short canonical CSR row into a stencil-major layout
``neighbor[slot, grid_row]``. Adjacent CUDA threads therefore read contiguous
int32 neighbor rows and uint8 coefficient codes (uint16 in the stencil of a
symmetry sector of more than 256 coefficients, `PARSEC_SECTOR_PALETTE_CODES`);
the palette retains the exact float64 coefficient bits. Each thread still visits its row in canonical
CSR order. A second kernel fuses kinetic, the current local diagonal,
optional nonlocal image, and the normalized Chebyshev recurrence into one
grid pass for blocks of up to six orbitals; by default its slot loop is a
software pipeline of its loads, with the same bits
(`PARSEC_CUPY_STENCIL_KERNEL=former` keeps the loop of before; the end of the
tables of switches above). For KB projectors a canonical-
order CUDA kernel computes the small `diag(signs) @ B.T @ X` contraction
without the launch overhead of many tiny cuSPARSE calls, while the large
`B @ coefficients` scatter is fused into that same row kernel. This avoids a
full-grid nonlocal temporary without ever forming a dense nonlocal matrix.
Already-resident float64 inputs retain their existing row/column strides;
the raw kernels consume those strides directly instead of forcing thousands
of temporary Fortran-order copies. Resident workers also give NVRTC a private
writable `.parsec_cache/cupy-temp` workspace, preventing a restricted system
temporary directory from silently disabling the custom projector kernel.
The default `PARSEC_CUPY_PROJECTOR_REDUCTION=auto` retains canonical serial
summation for short projector rows and selects a deterministic 128-thread
tree reduction only for rows with at least 256 entries. `serial` and
`parallel` force either policy for profiling. A CUDA block of the tree
reduction serves one projector and up to six vector columns: a block of two
to six columns is one launch of the six-column kernel, a wider block one
launch of its group kernel, a CUDA block per projector and group of
six columns, and a single column takes the kernel of a block per projector
and column (`PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS=6`, the default; `1`, the
former value, takes that kernel at every width, see the switch below the last
table of switches).
Set `PARSEC_CUPY_CUSTOM_PROJECTOR_DOT=0` to restore cuSPARSE and
`PARSEC_CUPY_FUSED_PROJECTORS=0` for an unfused comparison. The earlier compact CSR-order
kernel and generic CuPy CSR remain automatic fallbacks. Set
`PARSEC_CUPY_STENCIL_MAJOR=0` before Python starts to exercise the compact
CSR-order fallback explicitly. The other GPUs of a sector then hold replicas
in that layout, built from the owner's CSR (`distributed_filter.replica_csr`),
and they, the ranks of a shared basis and the GPUs of a nested group filter
block by block without CUDA graphs (`filter_graph.PlainBlockFilter`), with
the arithmetic of the filter of one device; before 2026-11-21 these layouts
stopped with an `AttributeError` (`neighbors`, `chebyshev_recurrence`).
On A100 (the MPI runner with the launch settings, Si28H36, 20 iterations
everywhere): with an atom moved by 0.1 bohr (two sectors) on one rank of
four GPUs, -263.5110009208224 Ry against -263.51100092085244 with the
filter of one device per sector, and on four ranks of one GPU the printed
energies of the stencil-major default; without symmetry (one sector) on one
rank, four ranks and two nodes of nested sharing within 6.3e-11 Ry of the
stencil-major default (-263.51152283930924 Ry: 6.3e-11, 2.2e-11 and 1.5e-11). The CSR kernels are slower
(5.4 against 2.9 s of SCF on one rank) and their projector products need
not repeat their last bit from run to run.

Orbitals also remain on the device through density construction. In symmetry
mode the fused row kernel works on the wedge and downloads a wedge-length
density. Density, local ionic/Hartree/XC potentials, Anderson history,
residual norms, and energy quadrature then remain one physical value per orbit
through the entire SCF loop; expansion occurs only at the final public result.
Without orbital symmetry the code downloads the ordinary full-grid density. The final requested
wavefunctions are expanded/downloaded once. Later-SUBSPACE Ritz residuals are disabled in
the production SCF adapter because they are reporting diagnostics and do not
control filtering, occupations, density, or SCF convergence. Direct modular
eigensolver calls retain residual construction by default.

Saved-SUBSPACE basis treatment uses a size-adaptive default. Small complete
bases retain the audited modified Gram--Schmidt path described below. When
the estimated complete-basis work `N * states^2` reaches 100,000,000, the
default solves Rayleigh--Ritz directly in the non-orthogonal filtered basis,

`(X.T H X) C = (X.T X) C diag(epsilon)`.

A Cholesky factor of the small overlap matrix whitens this generalized
eigenproblem. The code audits its condition number, factorization, and final
coefficient orthogonality; any unsafe basis falls back to stable blocked
Householder QR for that and subsequent SCF iterations. This constructs the
same Ritz subspace without a tall QR in the well-conditioned regime. A
symmetry sector keeps one column-major array and projects `H X` a slab of
columns at a time (`PARSEC_CUPY_STREAMING_RITZ` above); elsewhere a
persistent column-major `H X` workspace avoids repeated large CuPy allocations
and strided real-space-kernel reads. Set `PARSEC_CUPY_GENERALIZED_RITZ=off` to
disable it, or `on` to request an audited attempt for every complete basis.
Only a sector basis that `PARSEC_CUPY_DISTRIBUTED_STATE=1` forces onto
several devices takes this solve whatever the setting says.
`PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD` and
`PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX` control the automatic work and
stability limits.

This leaves the small representation sectors used by the symmetry benchmarks
on their measured-fast MGS route. Override that orthogonalization route with
`PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION=mgs`, `qr`, or `cholqr2`; the automatic
threshold can be changed with `PARSEC_CUPY_SUBSPACE_QR_WORK_THRESHOLD`.

In the complete-basis MGS route, filtered bases execute the common PARSEC
first-projection branch with device scalars and transfer all norm decisions
once; any failed 0.1 test restores the untouched input and reruns the literal
two-pass/replacement routine. Small CHEBDAV appended blocks use the literal
path directly. That literal path
queues the unchanged input and first-projection norms in their original order
but downloads each pair together, removing one synchronization per tested
column without changing the 0.1/0.68 decisions. Set
`PARSEC_CUPY_SPECULATIVE_MGS=off` for a literal trace or `all` to speculate
on appended blocks too. Exact QR and two-pass Cholesky-QR implementations remain
available explicitly for architecture-specific profiling.

CHEBFF does not form unused Ritz residuals. Optional cross-block Chebyshev
batching can reduce launch count, but the measured GPU was bandwidth limited
and ran faster with PARSEC's ordinary block traversal. It is therefore opt-in
with `PARSEC_CUPY_BATCH_FILTERS=1`, not the default.

#### Standard orientation of a cluster

The group of a cluster is found among the signed permutations of the axes about
the origin, after PARSEC's recentring of the bounding box (only beyond 0.1
bohr, and not in a restart). A molecule whose symmetry axes are not along the
coordinate axes, as coordinates from a database or another code often are,
keeps a smaller group or none: methane rotated by 10 degrees about z keeps four
operations (two of them commuting) instead of 24, methane 0.05 bohr off the
origin four, so that it loses the weighted sectors of D2d. `PARSEC_STANDARD_ORIENTATION=on`
(`Input/standard_orientation.py`, opt-in) lets the input reader look for the
frame and the centre of the largest group first. The centre is the centroid of
the atoms. The candidate axes are the two-fold axes and mirror normals of the
cluster, found among the directions of the atoms, the sums, differences and
cross products of pairs of atoms of its smallest shells (atoms of one species at
one distance) and the principal axes of inertia. Every frame whose coordinate
axes are one such axis, or two orthogonal ones, is scored by the group it gives:
first the order of its largest group of commuting involutions (the abelian
sectors), then the order of the whole group (the weighted sectors). Of the best
frames the one nearest the input wins, so that an input that is already aligned
and centred is not touched, bit for bit. A threefold axis is never put along a
body diagonal: C3v there keeps two commuting operations, as a mirror normal
along an axis does, and the code has no weighted plan for it. In a relaxation
with `Move_Flag` a frame counts as the relaxation decides
(`Relaxation/run.py symmetry_decision`): its group is that of the species; in a
frame in which an operation of that group exchanges a fixed atom with a moving
one, a relaxation by an optimizer keeps the subgroup of the operations that map
the fixed atoms onto fixed atoms (`PARSEC_MOVE_FLAG_SUBGROUP=auto`; the score
follows it since 2026-11-20: methane with two fixed H atoms takes the frame of
Td and its C2v, four commuting operations, where the mirror frame of before kept two; with one fixed H the C3v of that frame, six operations, two of them
commuting, its threefold axis along a body diagonal), and under `off` or for
other moves the frame counts as no group at all, since the relaxation would
then use none.

The frame is refined against the symmetrized structure (a Kabsch fit) and the
atoms are symmetrized exactly over the group: one atom of each orbit takes the
orbit average, averaged over its stabilizer with exactly rounded sums, and the
others are its images under signed permutations, so that the group maps the
atoms onto each other bit for bit. The bounding box is then centred at the
origin, as PARSEC centres a cluster. An operation is accepted within
`PARSEC_STANDARD_ORIENTATION_TOLERANCE` (1e-3 bohr), and the atoms move by up
to that distance; parsec.out reports the largest move (2e-13 bohr for
coordinates given to 12 decimals, 1e-5 bohr for coordinates in angstrom to five
decimals). Nothing changes unless the group grows; the frame keeps the operations
that map a shifted grid onto itself.

The calculation, its grid and the files of the grid (cube files, `parsec.dat`,
`wfn.dat`, the `.npz` archive) are in the new frame (a `Restart_Run` with the
switch on and the same input finds it again); so is `relax_restart.dat`, the
state of the optimizer, which records the frame in a block after all that
PARSEC reads. A restart from it reports in the frame of the run that wrote it
(also where this run does not recentre, as a `Restart_Run`) and needs the switch
on; a `relax_restart.dat` written without the switch is continued in the frame
of the input. The coordinates, forces (every table, the torque taken again
about the origin of the input) and the dipole (`Q mu + q d`) of parsec.out,
`atom.cor`, `relaxation.xyz` and the forces and final positions of the MPI
runner's JSON report are in the frame of the input as PARSEC places it, so
that they compare directly with a run without the switch;
parsec.out prints the map `x_calc = Q^T (x - d)` beside the symmetry block. A
structure given by `--restart-from` as an XYZ file is moved into the new
frame. The switch does nothing (with a note) for periodic cells, with
`Ignore_Symmetry`, where the run uses no point group (manual movements,
molecular dynamics, `Polarizability`, point charges, `FoldBlock`), for a domain
other than a sphere, with `Recenter_Atoms_In_Aperiodic_Directions: .false.`,
spin-orbit coupling or non-collinear magnetism, FDET (`Potential_Field`,
`Write_Potential_Field`, the AFM scan), `Initial_Structure: mlff` and an
initial density from a file (`Initial_Density: file`, `ML_Density_File`), which
was made for the atoms in the frame of the input. Energies differ from those of
the input orientation by the egg-box effect of the grid.

Measured on one A100 node (4 GPUs, `main.py`, first runs without a symmetry
cache, CA-LDA unless noted):

| Input | without the switch | `PARSEC_STANDARD_ORIENTATION=on` | aligned input |
|---|---|---|---|
| methane rotated by 10 degrees about z (h 0.25 bohr) | 4 operations (2 commuting), 13 SCF iterations, energy -1.8e-4 Ry, forces off by 1.7e-3 Ry/bohr | Td, D2d sectors, 9 iterations, energy and forces of the aligned input (8e-7 Ry/bohr) | 9 iterations |
| methane, arbitrary rotation and offset | 1 operation, 14 iterations, +2.4e-4 Ry, 2.3e-3 Ry/bohr, a dipole of 1e-3 D | as above | |
| methane 0.05 bohr off the origin | C2v (4), 11 iterations, +1.1e-4 Ry, 1.2e-3 Ry/bohr | as above | |
| C87H76 rotated (Td, h 0.4 bohr, 237 states) | no group, SCF 7.38 s, +2.1e-2 Ry, 4.1e-2 Ry/bohr | D2d sectors, SCF 2.02 s, within 6e-8 Ry and 5e-6 Ry/bohr | SCF 1.84 s |
| SF6 rotated (PBE, h 0.22 bohr) | 2 sectors, SCF 1.50 s, +7.1e-3 Ry, 3.3e-2 Ry/bohr | Oh, D4h sectors, SCF 1.73 s, the aligned energy, forces within 5e-7 Ry/bohr | SCF 1.85 s |
| BFGS of methane stretched by 8 %, rotated | 6 structures, final energy -2.6e-4 Ry off | 4 structures, the aligned trajectory within 3e-6 bohr | 4 structures |
| `examples/0d_water_geoopt` rotated | 9 structures, final energy -2.0e-3 Ry off | 7 structures, final energy within 7e-6 Ry, trajectory within 4e-4 bohr (SCF criterion 2e-4 Ry; its sectors come in another order) | 7 structures |

The differences without the switch are the egg-box effect of these grids (the
atoms sit elsewhere between the grid points), not an error of either run. An
aligned input gives the same bits with the switch on (methane, water). The MPI
runner (one node, two ranks) gives the energies of `main.py` with and without
the switch, and with it the forces and final positions of its JSON report
agree with parsec.out and `atom.cor` to their printed digits. A BFGS
relaxation of methane with two hydrogens fixed by `Move_Flag`, given in a
frame where it has Cs, keeps that frame and Cs with the switch on (the
energy of the run without it after 4 structures): the frame of its C2v gives
D2d for the species, half of whose operations exchange fixed and moving atoms. The search itself takes 0.06-0.3 s for diamond balls of 885 to
5,107 atoms on a laptop CPU.

#### Point groups of periodic cells

A bulk cell at the Gamma point that the route of the sectors takes
(`driver.periodic_sector_refusal`: orthorhombic, a native LDA or GGA,
unpolarized or, since 2026-11-19, collinear spin (`PARSEC_GAMMA_SPIN_SECTORS`)
and unpolarized hybrids of 2^24 basis entries or more
(`PARSEC_GAMMA_HYBRID_SECTORS`), both in abelian sectors; in one process
since 2026-11-19 also meta-GGAs and libxc functionals from 2^27 or 2^28
basis entries on, LDA+U from 2^26 on since 2026-11-21, and the TS correction
under its opt-in switch, "TS, LDA+U, meta-GGA and libxc in the Gamma sectors"
below; since 2026-11-22 wires, slabs and oblique cells under their opt-in switches, "Wires, slabs and oblique
cells" below; the CuPy backend)
looks for its point group about the origin of the cell
(`PARSEC_PERIODIC_SYMMETRY=auto`, since 2026-11-16): the Cartesian signed
permutations that map the atoms, species by species and modulo the lattice
(nearest images, a periodic k-d tree), and the periodic grid onto themselves.
Since 2026-11-18 it also takes the operations about a symmetric site off the
origin (`PARSEC_PERIODIC_SYMMETRY_CENTRE`), looks for a group where the one
about the origin is the identity alone (`PARSEC_PERIODIC_TRIVIAL_ORIGIN`),
adds translations (below) and takes cells whose grid has a different step on
each axis (`PARSEC_PERIODIC_UNEQUAL_STEPS`). Since 2026-11-19 a defect off the
origin of a large cell keeps its group (`PARSEC_PERIODIC_DEFECT_CENTRE`): the
search of the translation `t` of an operation `R` tries `x_j - R x_0` over the
atoms of a species and gave up after 32 candidates that pass its sampled atoms,
so a vacancy at `(a, a, a)` of 3 x 3 x 3 diamond cubes or more kept the six
operations of C3v about the origin (at `(a/2, a/2, 0)` one mirror). Such a search
now takes `x_0` among the atoms of the rarest environment of the cell (the
sorted distances to their 12 nearest atoms of every species; a vacancy's four
neighbours; an atom with the fewest others whose environments agree with its
own within four atom tolerances, so noise in the positions below the
tolerance does not split the crystal into classes of one atom) and `x_j`
among the at most 32 atoms alike: every operation carries
`x_0` onto one of them, so the search is that of every candidate, and Td about
the vacancy is found wherever the vacancy is. A cell whose pure translations
repeat a defect (three vacancies a third of the cell apart) takes the operations
about the centre of that environment, an empty site, where they are more.
The `symmetry_detection` line of the report names the site. The k-d tree
queries of the detection run on the calling thread below 16,384 points (since
2026-11-19; scipy started 128 threads for each of thousands of small queries on
a GPU node): on one A100 node the cell group of diamond 6^3 / 22^3 took
1.78 / 2.50 s before and 0.37 / 1.36 s after, a check of moved atoms
(`check_sector_atoms`) of 6^3 0.40 -> 0.06 s, with the same answers. The grid points of a cell are `(i + shift) h` from its corner, and its row maps
wrap around the cell, so no image falls outside it; an operation that carries
one axis onto another needs their point counts and sides to agree. The eight
points of half lattice vectors give the same maps as the origin, so a
structure whose symmetric site is at the corner or at the centre of the cell
keeps its group (diamond with an atom at the origin: Td, with the 18 points per
cubic edge of the benchmark grid; rocksalt about Na: Oh). The orbitals then take
the sectors of that group, with the weighted sectors of a larger subgroup
where they solve less (above): Td in D2d solves 1/16 of the grid times the
states of the one sector of the identity, Oh in D4h 1/32. Their kinetic
operators are the sector stencils with rows that wrap around the cell, built
from the grid (since 2026-11-18 the accelerated preparation holds only a
descriptor of the finite-difference matrix of the cell, which the sectors never
build, `PARSEC_PERIODIC_DEFER_LAPLACIAN`), and
their nonlocal operators are reduced from the projectors with their lattice
images, of one atom per orbit (below);
the density, the potentials and the mixer keep one value per orbit; the
Hartree potential is solved on the full grid of the cell
(`Hartree/periodic_device.py`, expanded and reduced again); the device XC
evaluators take the orbits with wrapped stencils (`V_xc/device_xc.py`); the
ionic sums keep the whole cell. A cell without a group, another backend, or
`PARSEC_PERIODIC_SYMMETRY=off` keeps the route of before (one sector of the
identity, or the full grid of one GPU), bit for bit. Since 2026-11-19 an
unpolarized SCF of such a cell in one process whose basis stays below 2^24
grid points x states keeps the full grid of one device
(`PARSEC_PERIODIC_SECTOR_FLOOR`; a collinear spin-polarized one only under an
integer `PARSEC_PERIODIC_SPIN_SECTOR_FLOOR`). Since 2026-11-18 the
single-point forces and the `bfgs`/`simple` relaxations of such a cell keep
these sectors as well, on the command line and in the full-SCF runner
(`PARSEC_PERIODIC_SECTOR_FORCES=auto`, `cli.periodic_sector_forces_refusal`):
the forces are averaged over the full group of the sectors, and the ionic loop
averages forces and moved positions over it; these runs keep the sectors below
the size floor too (`PARSEC_PERIODIC_SECTOR_FLOOR`). Since 2026-11-19 so do the
forces, relaxations and restarts of the Fortran-derived routines
(`Use_Fortran_Routines`, also where the parser chose them: `Cubic_Spline`
forces, FHI pseudopotentials, a core correction, `Restart_Run`, ...;
`PARSEC_FORTRAN_SECTOR_FORCES=auto`), with the force terms of those routines on
the expanded fields, their nonlocal term in the sectors (the full-SCF runner,
which reads no `parsec.dat`, leaves their `Restart_Run` and their
spin-polarized cells to `main.py`). A relaxation whose `Move_Flag`
fixes atoms keeps the sectors of the subgroup that maps the fixed atoms onto
fixed atoms, which the driver detects with them as a species of their own
(`PARSEC_MOVE_FLAG_SUBGROUP=auto`, since 2026-11-19), on the default routines
and for an unpolarized cell: a spin-polarized cell keeps the full grid of both
channels, and the Fortran-derived routines keep the sectors of that subgroup on
`main.py` under `PARSEC_MOVING_SECTORS=auto` (since 2026-11-21, above the floor
of the sectors and with 4 operations or more; their full grid otherwise, under
`off` and in the runner). Molecular dynamics and manual movements keep the
sectors of the group their structures keep on `main.py` under
`PARSEC_MOVING_SECTORS=auto` (since 2026-11-21, checked at every structure;
a cell above the floor of the sectors and with 4 operations or more; the full
grid otherwise, under `off` and in the runner); a `Move_Flag` that fixes atoms under
`PARSEC_MOVE_FLAG_SUBGROUP=off`, point
charges, `Polarizability`, `nscf` runs, `Restart_From_Wfndat` and k-points
beyond Gamma keep the full grid (`main.py`; the runner: one sector of the
identity), as does every forces or moving run of a cell with
`PARSEC_PERIODIC_SECTOR_FORCES=off`, and of the Fortran-derived routines with
`PARSEC_FORTRAN_SECTOR_FORCES=off`. A band structure (`bandstruc`) keeps the
sectors of its SCF and takes the expanded converged potential
(`PARSEC_PERIODIC_SECTOR_BANDS=auto`, since 2026-11-18; `off` puts its SCF on
the full grid, as before; the MPI runner leaves bands to `main.py`); since
2026-11-19 its k-points are solved in the sectors of their little groups
under the group of that potential (`PARSEC_BAND_KPOINT_SECTORS`). Under
`PARSEC_FORTRAN_SECTOR_FORCES=off` the driver refuses to move the atoms of a
cell of the Fortran-derived routines prepared in these sectors and names the
switch. The parser keeps no space group for a cell on
the default routines (`use_symmetry`: its forces are not symmetrized); both
runners let a Gamma-point SCF look for its point group unless `Ignore_Symmetry`
or `--symmetry off` says otherwise.

Measured on A100 (bulk diamond of `examples/3d_diamond_benchmark`, first runs):
SCF to 1e-7 Ry, the sectors of Td give the total energy of the one sector of
the identity within 4e-14 (8 atoms), 5e-13 (64) and 9e-13 Ry (216), one rank
per GPU or one process on the four GPUs of a node. With the settings of the
benchmark (2e-4 Ry) on 16 A100 (four nodes, one process each), with the
Hartree solve by the stencil symbol and the local ionic potential on the GPU
(`PARSEC_PERIODIC_HARTREE_METHOD`, `PARSEC_PERIODIC_SETUP_DEVICE` in the table
of switches): 1,728 atoms 5.7 s and 0.98 GiB at the fullest GPU, 4,096 atoms
12.4 s and 2.54 GiB (published 136.7 s and 21.5 GiB, the one sector of the
identity), 8,000 atoms 36.2 s and 6.48 GiB (published 651.5 s and 68.1 GiB,
the largest cell of the one sector of the identity), and up to 32,768 atoms
(131,072 electrons, 23.9 M grid points, 65,542 states): 424.8 s, 9 SCF steps,
66.3 GiB at the fullest GPU, the largest cube on 16 A100
(`examples/3d_diamond_benchmark/README.md`). Its SCF is mostly eigensolves
(355 s of 362 s), its setup (63 s) mostly the projectors (23 s; since built for
all atoms at once, `V_ion/periodic_projectors.py`, 1.5 s with the same arrays)
and the initial density (11 s); with the translations by half the cell (below)
the same cell takes 187.8 s. Before these two device paths the same cell took
690.7 s and 74.0 GiB: a setup of 240 s, 153 s of it the local ionic potential
on the host of the root, and 84 s of CG Hartree solves.

#### Translations by half the cell

A cell that repeats a smaller one twice along some of its axes is invariant
under the translations by half of those axes, and so is its grid where those
axes have an even number of points: exact symmetries of the discretized
Hamiltonian at the Gamma point (`PARSEC_PERIODIC_TRANSLATIONS=auto`, since
2026-11-16; `Symmetry/axis_reflection.py`, `_half_cell_translations`). They form
a group of order two per element; its real characters are the time-reversal
invariant k-points of the smaller cell, whose Bloch functions are real (periodic
or antiperiodic along each axis), so a Gamma calculation of the cell becomes
the calculation of those k-points of the smaller cell with the discretization
of the cell itself: the energies are those of the point group alone, not those
of a k-point mesh (whose kinetic term discretizes `-nabla^2 - 2ik.nabla + k^2`
on the periodic part). With the signed permutations about the origin they make
the operations `r -> R r + t` of the cell; the `t` that go with an `R` form a
coset of the translations, so one is searched among the eight half-cell steps
(a k-d tree of a few atoms refuses the rest first). The orbitals take the
weighted sectors of this group as of a point group (above); its multiplication
table counts the translations modulo the grid, and the search for the solved
subgroup runs over the subgroups that contain the pure translations (a normal
subgroup of real characters, in the chosen subgroup of every cell measured).

Every supercell of a face-centred structure has them: the face-centring
translations `(0, a/2, a/2)` are half-cell translations of its conventional
cube and of every cube of an odd number of cubes; a cube of an even number has
all seven. Diamond with an atom at the origin then has 96 (one cube, or an odd
number) or 192 operations instead of the 24 of Td, and its orbitals use 5 or 10
weighted sectors of a subgroup of 16 or 32 (`1/32` of the grid each, about
`1/102` of the rows times states of the one sector of the identity, against
`1/16` with Td alone). The translations change only the row maps: the density,
the potentials and the mixer keep one value per orbit of the larger group
(since 2026-11-19, `PARSEC_SECTOR_FIELD_GROUP=full`; before, and with
`subgroup`, one per orbit of the solved subgroup of 16 or 32 operations,
`N/16` or `N/32` values instead of `N/96` or `N/192`, the density averaged over
all operations of the larger group: Si 2x2x2 on 36^3, 330 values against
1,458), the Hartree potential is solved on the full grid, the device XC
evaluators turn vectors by `R` alone, and the partner states of a weighted sector
are its images under the operations with their translations.
`PARSEC_PERIODIC_TRANSLATIONS=off` keeps the point group about the origin, bit
for bit; a cell that does not repeat has the zero translation alone and the
group of before.

Measured on A100 (bulk diamond of `examples/3d_diamond_benchmark`, first runs):
SCF to 1e-7 Ry, the translations give the total energy of the point group
alone within 1e-14 (8 atoms, 96 operations), 2e-13 (64), 2e-12 (216) and
2e-12 Ry (512 atoms, 192 operations), four ranks on one node. With the settings
of the benchmark on 16 A100 (four nodes, one process each): 8,000 atoms 19.8 s
and 2.38 GiB at the fullest GPU (36.2 s and 6.48 GiB with Td alone), 32,768
atoms 187.8 s and 16.1 GiB (424.8 s and 66.3 GiB), and up to 64,000 atoms
(256,000 electrons, 46.7 M grid points, 128,006 states): 668.1 s, 7 SCF steps,
53.8 GiB, the largest cube on 16 A100. The ten sectors of 32,768 atoms hold
2,054 states each; its SCF took 142.9 s (the first eigensolve 71 s, a later
step 12 s, against 133 s and 28 s), its setup 44.7 s.

Operations whose translation is a fraction of the cell (non-symmorphic, the
other half of Oh for diamond: `R` outside Td with `t = a/4 (1, 1, 1)`) join
the group where they close with the half-cell translations, which happens
where those are all the pure translations of the crystal modulo the cell (the
conventional cube of diamond, not a supercell of several cubes, where the
products of such operations give the face-centring vectors of the crystal,
translations of complex characters) and `t` is a whole number of grid steps
(16 or 20 points per edge of the diamond cube, not the 18 of the benchmark):
the translations are searched from the atoms (`x_j - R x_0` on whole grid
steps, a k-d tree of a few atoms first, at most 32 full matchings), and a set
that does not close keeps the half-cell translations alone. The diamond cube
then has Fd-3m modulo the cube, 192 operations, 10 sectors of a subgroup of 32
(1.65 times fewer orbits than its 96 operations). `PARSEC_PERIODIC_TRANSLATIONS=half`
keeps the half-cell translations alone.

The sectors of such a group take the projectors of one atom per orbit
(`PARSEC_SECTOR_PROJECTOR_ORBITS=auto`): the projectors of an atom carried
onto another by an operation of the group are its own moved by the operation,
their harmonics turned within each channel of one sign, so every atom of an
orbit gives a sector the same `U^T B_a D B_a^T U`, and the representative's
columns scaled by the square root of the orbit size give the sum. The sector
operators are the same to round-off (1e-13 of the largest), with up to the
group order fewer columns: the cost of the projectors no longer grows with the
number of sectors. Since 2026-11-18 the projector gradients of the forces are
built for the same representatives (`PARSEC_SECTOR_GRADIENT_ATOMS=representatives`,
the same forces bit for bit; before, the forces kept the projectors of every
atom), and the abelian sectors of clusters and Gamma cells take one atom per
orbit as well, their forces completed over the group
(`PARSEC_SECTOR_PROJECTOR_ORBITS`; with LDA+U since 2026-11-21,
`PARSEC_LDAU_PROJECTOR_ORBITS`). On 16 A100,
with the binned atomic density of the first step as well
(`PARSEC_PERIODIC_DENSITY_BINS`), the 32,768-atom diamond took 64.3 s (187.8 s
with the projectors of every atom: its SCF 30.2 s instead of 142.9 s) and the
64,000-atom one 195.5 s (668.1 s), the same energies to 2e-9 Ry
(`examples/3d_diamond_benchmark/README.md`); a Td nanodiamond cluster of 15,687
atoms (D2d sectors) 106.0 s instead of 111.2 s.

#### Every translation of a perfect supercell

A cell that repeats a smaller one `m_1 x m_2 x m_3` times has `m_1 m_2 m_3`
pure translations. The Gamma route keeps those by half the cell, whose
characters are real (above): along an axis of odd repeats none, and of four
repeats only the half. The others have complex characters, and Bloch's theorem
is the way to them: the Gamma point of the supercell is the cell of
`m_a / K_a` repeats with the Gamma-centred mesh `K_1 x K_2 x K_3`, for every
divisor `K_a` of `m_a`, each k-point a complex Hermitian problem of `1/K` of the
rows and states (`K = K_1 K_2 K_3`), the mesh reduced by the operations of
the cell. With the Bloch block of the finite-difference Laplacian as the
kinetic term of the k-points ("The kinetic term of a k-point", below) the two
are one discretization: the same eigenvalues, the total energy of the cell
`1/K` of the supercell's. `PARSEC_SUPERCELL_KPOINTS=auto` takes this route
(`SCF/kpoint_supercell.py`, `SCF/supercell_mesh.py`; the reference CLI runs
the k-points, on the GPUs of the process or the MPI k-point pools of
`main.py --mpi`; the full-SCF runner refuses it by name). It stays `off` by
default: only a perfect crystal repeats, a defect, a displacement or a
disorder breaks the translations, and the default routes are not fitted to
the benchmark family. Since 2026-11-19 two switches shape it:

- `PARSEC_SUPERCELL_KPOINTS_WEDGE=auto` (default) reduces the mesh of a single
  point to the irreducible wedge of the operations of the cell that map its
  grid and the mesh onto themselves, with time reversal, and symmetrizes the
  density with them, as the reader does for an input of that cell and mesh
  (`_default_routines_wedge`): the diamond cube with an odd `n x n x n` mesh
  takes `(n+1)(n+3)(n+5)/48` k-points instead of the `(n^3+1)/2` of time
  reversal alone (10 instead of 63 for `n = 5`, 286 instead of 4631 for
  `n = 21`). Forces, relaxations, molecular dynamics and `Ignore_Symmetry`
  keep time reversal alone; `off` restores it everywhere, as before.
- `PARSEC_SUPERCELL_KPOINTS_MESH` chooses the mesh: `all` (default) that of
  all the repeats on the smallest cell, as before; `K1xK2xK3` a mesh whose
  entries divide the repeats (an error otherwise); `auto` the mesh of most
  points among those with the same entry along axes of the same side and
  repeats (the cell keeps the shape, so the point group, of the smallest one)
  and an entry above two (a mesh of ones and twos is the time-reversal
  invariant k-points, which the Gamma route solves as the translations by half
  the cell in real arithmetic) whose cell holds at least 2^22 grid points
  times states (below that a k-point costs a device about the same whatever
  its size), else the mesh of all the repeats, else (repeats of ones and twos
  alone) the Gamma point of the supercell. It stays opt-in: in first runs it
  lost at 6^3 and 9^3 and tied at 10^3 (table below), since a larger cell
  compiles the kernels of a grid of its own; with those compiled it wins
  everywhere but 9^3.

The measurements are of first runs: every run starts with an empty CuPy
kernel cache (`CUPY_CACHE_DIR` a new folder, no warm-up run), as after a new
installation or a change of the code, and then runs the same input again with
the kernels it compiled (in brackets). On one node (4 A100, a new process per
run, the whole process from the input of the supercell;
`examples/3d_diamond_benchmark/make_diamond.py`, the default settings of that
example, 7 or 8 SCF iterations; 2026-11-20), the diamond supercell of `n^3`
cubes at the Gamma point of today's route against
`PARSEC_SUPERCELL_KPOINTS=auto` (the wedge of the cell by default; the peak of
the fullest GPU as `nvidia-smi` samples it, with the CUDA context):

| n | atoms | Gamma | mesh of all repeats (`all`) | other meshes | `auto` picks |
|---|---|---|---|---|---|
| 5 | 1,000 | 31.5 s (10.6 s), 1.3 GB | 26.6 s (7.0 s), 0.6 GB, 10 k; time reversal alone 38.2 s (19.0 s), 63 k | | `all` |
| 6 | 1,728 | 37.0 s (12.9 s), 1.3 GB | 29.0 s (9.9 s), 0.6 GB, 20 k; time reversal alone 50.3 s (31.7 s), 112 k | 3^3: 37.5 s (8.0 s), 1.3 GB, 4 k | 3^3 |
| 7 | 2,744 | 38.4 s (19.1 s), 2.6 GB | 29.1 s (9.6 s), 0.6 GB, 20 k | | `all` |
| 9 | 5,832 | 60.5 s (45.2 s), 8.5 GB | 32.9 s (14.0 s), 0.6 GB, 35 k | 3^3: 35.8 s (16.1 s), 1.3 GB | 3^3 |
| 10 | 8,000 | 58.6 s (39.3 s), 5.7 GB | 40.1 s (19.1 s), 0.7 GB, 56 k | 5^3: 40.7 s (11.3 s), 1.5 GB, 10 k | 5^3 |
| 12 | 13,824 | 102.8 s (85.6 s), 13.9 GB | 50.1 s (27.4 s), 0.7 GB, 84 k | 6^3: 45.9 s (16.6 s), 1.7 GB, 20 k | 6^3 |

A first run of the route takes 19 to 30 s longer than the second on that node
(the Gamma route 15 to 24 s), mostly the compilation of its kernels, which is
most of its time at these sizes. On 16 A100
(4 nodes; first runs, every rank with an empty kernel cache; the wall time of
the whole case on rank 0, 2026-11-20), the Gamma point of today's route in the
full-SCF runner (one rank per node on its four GPUs, the launch of
`scripts/run_multi_gpu_scf.sh --nodes 4` and of the published table of that
example) against the cube of 3 x 3 x 3 cubes with the mesh of the remaining
repeats in `run_mpi_kpoints.py` of that example (one rank per GPU, k-point
pools):

| n | atoms | Gamma | cell of 27 cubes with a mesh |
|---|---|---|---|
| 15 | 27,000 | 92.6 s (81.0 s in the runner's report), 21.9 GB | 5^3, 10 k: 28.4 s, 1.0 GB |
| 18 | 46,656 | 107.3 s (90.2 s), 28.1 GB | 6^3, 20 k: 27.1 s, 1.4 GB |
| 21 | 74,088 | out of memory (one GPU in the sector solve, after 159 s) | 7^3, 20 k: 29.6 s, 1.4 GB |

The energies per atom agree to 1.5e-8 Ry on one node (the Gamma route stops an
iteration earlier at the default tolerance; 6e-10 Ry at 7^3, where both take
eight) and to 4.6e-9 Ry on 16 A100, those of the meshes among themselves to
3e-9 Ry, the wedge and time reversal alone to 1.2e-9 Ry.

The cell of a repeat takes the steps of the supercell; where its side over
that spacing rounds to a smaller count under PARSEC's rule of the points (the
largest even count whose step is not below the spacing,
`Grid.pbc.fortran_points_per_axis`: the diamond cube of 18 points within 3, 5
or 6 cubes got 16), a spacing a hair below the step gives it the same grid. A
spacing whose grid the builder refuses is passed over the same way (the cube
of 14 points at `Expansion_Order` 12, which first rounds to 12); a cell too
small for the stencil (12 points at order 12) leaves the supercell at the
Gamma point with a note, or `PARSEC_SUPERCELL_KPOINTS_MESH=auto` takes the
smallest larger cell that holds it, and an explicit mesh of such a cell is an
input error. The cell takes `1/K` of `Net_Charges` (with the uniform
background the same density and energy per cell; until 2026-11-19 every cell
took the whole charge, a crystal `K` times as charged) and of the states,
without a count where the input gives none. The repeats come from the counts of
the supercell's grid, which is no longer built for them. Before 2026-11-19 the
route did not apply to a cell whose count rounded off and the supercell ran at
the Gamma point without a note; the full-SCF runner ran every input at the
Gamma point under this switch, without a note, and now refuses a repeating
cell by name.

#### Wires, slabs and oblique cells

Until 2026-11-22 a wire or slab (`Boundary_Conditions: wire` or `slab`) and a cell whose lattice vectors are not
along the Cartesian axes (hexagonal, primitive cubic) kept the full grid of one device at the Gamma point. Two
opt-in switches give them the sectors of their groups, in one process, unpolarized, with a native LDA or GGA and
none of TS, LDA+U, a hybrid, a meta-GGA, libxc or (wire or slab) `Symmetrize_Charge_Density`, above the size floor
of `PARSEC_PERIODIC_SECTOR_FLOOR`; everything else keeps its route, and so do MPI runs and restarts. The forces of a
wire or slab are the reference solver's; since 2026-11-22 the forces, relaxations, molecular dynamics and manual
moves of an oblique cell keep its sectors (below). The command line hands a cell on the general grid of the Fortran-derived routines (an FHI table or a
core correction makes the parser choose them) to the reference solver whatever the switch: the sectors of such a cell
are those of the driver's interface (`prepare_single_point`), and its forces, which the accelerated path does not
compute on that grid, stay with the reference solver.

`PARSEC_SLAB_WIRE_SECTORS=auto` takes a wire or slab of the Fortran-derived routines whose periodic vectors lie along
the axes (`a1 = (a, 0, 0)`, a slab's `a2 = (0, b, 0)`). Its grid (`Grid/confined.py`) holds the points of the index
box inside the cylinder or slab; along a periodic axis the steps of `adjust_step`, along a confined one the input
spacing. The detection (`Symmetry/axis_reflection.py`) matches the atoms modulo the periodic lattice (a periodic k-d
tree whose period along a confined axis is far beyond the domain, so no image wraps there), maps the rows with a wrap
along the periodic axes only (an image outside the cylinder or slab refuses the operation), carries an axis only
onto one of its own kind and step (the mirror `z -> -z` of a slab, the exchange of the confined axes of a wire), and
adds translations along the periodic axes (the half-cell and fractional ones of `PARSEC_PERIODIC_TRANSLATIONS`, about
symmetric sites in the plane; a translation along a confined axis is never a whole number of steps of that axis,
`_cell_steps`). The sector stencils are reduced from the CSR of the grid's Laplacian (the direct builders take the
stencil of a bulk box); every field stays on the full grid of the wire or slab, reduced to the orbits, and the Hartree
potential is that of its own solver (the boundary values of `hartset_wire`/`hartset_slab`).

`PARSEC_OBLIQUE_SECTORS=auto` takes a bulk cell on the default routines' oblique grid or the general grid of the
Fortran-derived routines, and with the switch above a wire or slab of oblique periodic vectors (a hexagonal slab).
Its points are `r = S (i + s)`, `S` the steps along the three grid axes; an operation maps the grid onto itself and
commutes with its operators where `W = S^-1 R S` is a signed permutation of the axes (the centred first differences
of the gradient, made Cartesian by `S^-T`, then turn with `R`) that carries every second difference of the
Laplacian onto one of the same weight: the operations the stencil-exact k-point wedge keeps (a hexagonal cell eight
of its 24, the primitive fcc cell 12 of 48; `Symmetry/oblique_frame.py`). The detection runs unchanged in the frame of
the grid indices (`IndexFrame`: the atoms at `S^-1 r`, a grid of the same rows whose cell is the box of the counts, the
atom tolerance divided by the smallest singular value of `S`, only those `W` admissible); its orbits and characters are
those of the grid, and the operations of the reduction are the `W` (`index_frame` on the reduction, whose Cartesian
operations are `S W S^-1`). A cell typed to finite precision (a hexagonal `a2 = (-a/2, a sqrt(3)/2)` to ten decimals)
has the weights of equivalent second differences unequal by about 1e-11, more than the 5e-13 the reduction of an
operator to a sector accepts: its sectors take the Laplacian with the mean weight of every class of lines
(`sector_negative_laplacian`, a change of about 1e-12 of the operator).

PARSEC 1.5 keeps no rotation or mirror in the Abelian group of a cell whose lattice vectors are not orthogonal
(`symmetries.f90:194-203`, `norot = 1`: the identity and the inversion only, `symgen.f90:1225`). The port takes the
operations above because each is tested to map the grid and its finite-difference operators onto themselves exactly
(the stencil test), so the sector Hamiltonian is the full-grid one restricted, not an approximation of it. The frame
is built only for what the route takes (`driver._symmetry_index_frame` asks `_oblique_sector_refusal` for every grid
of the frame, a diagonal lattice on the general grid of the Fortran-derived routines included): an MPI run, spin, a
hybrid, TS, LDA+U, a meta-GGA or libxc keep the full grid as under `off` (the MPI full-SCF runner refuses an
oblique cell before its preparation under either value). The command line hands a bulk cell on the general grid of the
Fortran-derived routines (an FHI table or a core correction makes the parser choose them) to the reference solver
whatever the switch (`cli._reference_only_features`): there the sectors are those of the driver's interface,
`prepare_single_point`, alone. Both switches are read by the command line before that hand-over, so a value they do
not know is refused there too.

The forces and moves of such sectors (`Forces/sector_symmetry.py`) turn the operations into atom operations of the
cell: `R = S W S^-1` and, for a translation of `t` grid steps, `tau = -S W^T t` (`R (r_a - tau) = r_b` modulo the
lattice); the atom images are matched in the frame, as is the check of moved atoms. The nonlocal term of the sectors,
the local terms of one atom per orbit and the averages of a relaxation then take them as they take the signed
permutations of a box. On A100 the forces of the sectors equal those of the full grid within 2.5e-10 Ry/bohr
(primitive fcc and hexagonal Si, graphite, bcc Na, fcc Ag on the default routines, groups with and without
translations, steps 0.5 and 0.3 bohr; the command line within 2.4e-8 at the criterion 1e-8), and three BFGS steps
of hexagonal Si end within 8.6e-8 bohr and 7.1e-11 Ry of the full grid.
With `Cubic_Spline: .true.` for every species both routes equal the central difference of the free energy at
steps of 1e-3 and 2e-3 bohr within 4.2e-7 Ry/bohr (primitive fcc and hexagonal Si, bcc Na with an atom on a grid
point, the cubic Si8 cell), graphite 2x1 within 1.86e-6 and 7.14e-6 (the truncation of the central difference,
which grows as the square of the step: extrapolated to a zero step about 1e-7), and within 1.5e-5 where half of
fcc Si 2x2x2 is moved at 300 K; the command line then gives the forces of the full grid within 1.5e-8 Ry/bohr
and three BFGS steps of hexagonal Si within 9e-8 bohr. Without it (the linear tables a species without the label
had before `PARSEC_CUBIC_SPLINE_DEFAULT`) the slope differs from the forces by 1e-3 to
6e-2 Ry/bohr at atoms moved off their sites, on either route and on cubic cells alike; diamond (C, 0.25 bohr)
keeps 4e-5 to 1.7e-4 at its moved atoms with it, on either route and with no sectors at all, not decreasing with
the step (measured with the tabulated projector tail, `PARSEC_PROJECTOR_TAIL=off` today).
Molecular dynamics and manual moves (`SCF/moving_sectors.py`) take the same operations: the group of the atoms
(`group_atom_operations`) and the check of every moved structure (`check_sector_atoms`, distances in bohr against
1e-6 bohr). Three steps of molecular dynamics with one movable atom, hexagonal Si 2x2x2 (32 atoms) and primitive
fcc Si 2x2x2 (16 atoms), in the 2 sectors of their groups (`PARSEC_MOVING_SECTORS=on`; under `auto` a group of
fewer than 4 operations keeps the full grid, bit for bit), end with the positions and forces of the full grid to
the printed digits (1e-9); manual breathing paths of fcc Ag and bcc Na 3x3x3 (27 atoms) and graphite 3x3x2 (72
atoms) in the 4 sectors of their groups (`auto`, `PARSEC_PERIODIC_SECTOR_FLOOR=off`) give the energies and forces
of the full grid to the printed digits until a structure leaves the group, after which the run is prepared again
on the full grid with a NOTE (a path from a perfect crystal leaves it at its first move, which costs that second
preparation, 3-9 s on these cells).

#### The kinetic term of a k-point

A supercell of `m` cells along an axis has the finite-difference Laplacian of
the cell in each copy; its eigenfunctions of the translation by the cell
satisfy `psi(r + L) = e^{ik.L} psi(r)`, so on the cell they solve the cell's
stencil with every wrapped neighbour taken times `e^{ik.L}`, and the periodic
factor `u = e^{-ik.r} psi` solves the stencil whose weight of the neighbour
`delta` away is multiplied by `e^{ik.delta}` (`build_bloch_negative_laplacian`
of `Laplacian/finite_difference.py`, the u gauge used by the k-point
Hamiltonian; the psi gauge is real at a time-reversal invariant k). That is
the exact Bloch block of the supercell's matrix: the spectra of a mesh of `m`
points along each axis are together the spectrum of the supercell, and the SCF
of a silicon cube with the 2 x 2 x 2 mesh is the Gamma-point SCF of the cube of
eight cubes within 6e-13 Ry. The form before (`-nabla^2 - 2ik.nabla + k^2`
with the gradient discretized on its own, `PARSEC_KPOINT_KINETIC=gradient`) is
another discretization of the same operator, 1.2 mRy per atom from it at 0.86
bohr; both converge to the same energy, and neither is closer at every
spacing. On silicon with 2 x 2 x 2 and 3 x 3 x 3 meshes the total energy of the
Bloch form is as close or closer to the converged one at every spacing
measured (12, 16, 20, 24 and 28 points per cubic edge; at 16 points 6e-5
against 1.2e-4 and 5.6e-5 against 2.5e-4 Ry per atom); on the gold cell of the
band-structure tests at 0.6 bohr its lowest band dips 1.2 mRy below Gamma at
k = 1/4, where the gradient form rises 5.3 mRy (3.7 and 4.4 mRy at 0.45 bohr,
10.4 and 10.9 at 0.35). The gradient form therefore stays the default;
`PARSEC_KPOINT_KINETIC=bloch` takes the Bloch form (a scalar Hamiltonian
without a meta-GGA on a bulk orthorhombic cell of the default routines), `auto`
where it applies, `trim` (since 2026-11-21) at the time-reversal invariant points
other than Gamma only, where the Hamiltonian is then real (see "Real arithmetic
at Gamma and at the time-reversal invariant points"). With `PARSEC_KPOINT_KINETIC=bloch` the reduced (irreducible
wedge) and the whole list of an even mesh agree exactly: the wedge solves a
member of a star at `R k_rep`, the whole list at its image `k + G` folded into
the zone, and the Bloch form at `k + G` is the form at `k` in another gauge;
the gradient form is not, and the two lists then agree only to its
discretization (3e-8 Ry and 4e-7 Ry/bohr measured with a moved atom), which
says nothing about the accuracy of either. The band structure takes the kinetic term of the SCF. On the
device the sparse route applies the complex matrix, and the stencil kernel of
the k-point pools takes its weights `Re(v)` and `-Im(v)` of
`v = c e^{ik.delta}` (`Hamiltonian/bloch_stencil.py`, `BlochStencil.from_bloch`);
the row-distributed k-point solver of the forces refuses it.

With the Bloch form the two calculations are one, and either can stand for
the other (`SCF/kpoint_supercell.py`): with `PARSEC_KPOINT_SUPERCELL` a
Gamma-centred mesh of one or two points per axis (`Monkhorst_Pack_Shift` zero,
or since 2026-11-18 no shift block: the time-reversal invariant k-points) runs
as the Gamma point of its supercell, whose real orbitals take the translations
by half the cell with the point group (the little groups of the k-points); with
`PARSEC_SUPERCELL_KPOINTS=auto` (`off` by default) a Gamma-point cell that
repeats a smaller orthorhombic cell runs as that cell with the Gamma-centred
mesh of the repeats and the Bloch form (an explicit
`PARSEC_KPOINT_KINETIC=gradient` refuses it). The cell of the second route
takes `1/M` of the supercell's states and six more per k-point, which keep the
filters of the small eigenproblems away from the occupied levels (unoccupied:
the energies stay those of the supercell). Diamond of 21 x 21 x 21 cubes
(74,088 atoms) runs this way as one cube with the 21 x 21 x 21 mesh (4631
k-points of time reversal alone) in 1449 s on 16 A100 with 4.4 GB per GPU, at
-11.4039349 Ry per atom (the code of 2026-11-16); since 2026-11-19 the wedge of
the cube takes 286 of them (65.3 s in a launch whose kernels an earlier case
had compiled), and the cube of 3 x 3 x 3 cubes with the 7 x 7 x 7 mesh
(`PARSEC_SUPERCELL_KPOINTS_MESH=7x7x7` or `auto`, 20 k-points) takes 29.6 s in
a first run with an empty kernel cache ("Every translation of a perfect
supercell", above). The Gamma point of that supercell does not fit 16 A100
(an odd number of cubes along each axis gives no half-cell translations: five
sectors, about 175 GB on the fullest GPU by the estimate; measured on
2026-11-20, one GPU ran out of memory in the sector solve).

Since 2026-11-19 the first route is the default for single points
(`PARSEC_KPOINT_SUPERCELL=auto`, `trim_supercell_route`), where it gives
the energy of the k-points within 0.01 eV per atom. The supercell solves the
Bloch form, the k-points the gradient form of PARSEC 1.5
(`PARSEC_KPOINT_KINETIC=gradient`): along an axis where half the mesh sits at
`k = pi/L`, a plane wave `e^{ipx}` of an orbital has the kinetic energy
`s(p)` in the supercell (`s`: the symbol of the stencil of `Expansion_Order`)
and `s(p - k) + 2k d(p - k) + k^2` at the k-point (`d`: that of the
first-derivative stencil), a difference that falls as `k^2 h^{2M} p^{2M}`
for order `2M` and is zero at Gamma. `kinetic_form_difference_estimate`
takes its first-order effect on the energy, `sum f <u|T_gradient - T_Bloch|u>`,
with the momentum density of the occupied states from the pseudo-orbitals of
the atoms (their occupations, the radial Hankel transform; about a second
for a species, before the SCF). The default takes the supercell where twice
the estimate is within `PARSEC_KPOINT_SUPERCELL_TOLERANCE` (0.01 eV per atom)
and the whole mesh holds at least `PARSEC_KPOINT_SUPERCELL_MINIMUM` entries
(k-points times grid points times states, 2^24), for a single point on the
GPU in a process of one GPU (`PARSEC_KPOINT_SUPERCELL_GPUS`: on four A100
in one process the supercell held 1.35 to 3.1 times the k-points' peak memory
for 216-atom cells, for 0 to 8 % in time; not with `--mpi`, whose ranks
divide the k-points; the full-SCF runner divides the sectors of the
supercell); everything else keeps the k-points
with a NOTE that names the reason
(the Fortran-derived routines, which every FHIPP species, `Restart_Run` and
the symmetric cells whose space group sets the grid select; spin, spin-orbit,
hybrids, meta-GGAs, `occup.in`; what the Gamma route refuses: libxc, LDA+U,
the TS correction; forces and the other runs that read the atoms or the
k-points of the cell; a small cell; a large estimate). `on` takes the
supercell for any run (its forces and moves are those of the copies, its
`parsec.dat` the supercell's), `off` never. The supercell carries `M` times
the states and the `Net_Charges` of the cell (the charge is that of one
cell, as for the k-points); `Relax_Restart` and `--restart-from` give the
atoms of the cell before the decision, and the supercell is made of them.
Under `auto` the report of the supercell prints the energies of the input
cell in its energy lines (`Total Energy =`, `Energy/atom` and the parts, as
the k-points print them, so that a series of runs reads the same quantity on
either route), its own levels and atoms in the header and the iterations,
a NOTE that gives the estimate, and closes with the total energy of the input
cell, per atom and the supercell's total; `on` keeps the supercell's
energies in those lines (`M` times the cell's). `auto` writes the
`parsec.dat` and the result archive of the cell: the potential and density
of its first cell, its atoms, grid and energies, the k-points and weights
of the k-point route with the levels of each (every level of the supercell
at the k-point of its Bloch phases, `PARSEC_KPOINT_SUPERCELL_LEVELS`; before
2026-11-21 the levels were dealt to the k-points of the mesh in turn, whose
union is the mesh's; a k-point holds at least the fewest levels any k-point
of the supercell's holds, which can be a few below `States_Num`, with a
NOTE), no orbitals (the k-points write theirs since 2026-11-22); a `Restart_Run` of the input reads `parsec.dat` (silicon cube at 16
points: the restart on the Fortran-derived routines, which `Restart_Run`
selects, reached their energy from scratch in 9 iterations instead of 17),
and `--bands-from` of the input the archive. A cell of unequal steps (orthorhombic, the
default routines) takes the route where one grid spacing gives the supercell
the cell's steps (steps within about `1/n` of each other); cells whose steps
differ more keep the k-points.

Measured on one A100 (2 x 2 x 2 mesh, no shift block, SCF to 1e-7 Ry, the
supercell equal to the k-points with `PARSEC_KPOINT_KINETIC=bloch` within
1e-8 Ry per atom in every case; "converged": the finest grid of the system:
silicon at 48 points per edge, diamond at 36, aluminium at 36 and 0.2 bohr,
MgO at 72), in meV per atom:

| system | order | h (bohr) | points per edge | E(k-points) - E(supercell) | estimate | k-points - converged | supercell - converged | 2 x estimate within 10 meV |
|---|---|---|---|---|---|---|---|---|
| Si 8-atom cube, LDA (`examples/3d_Si`) | 8 | 0.641 | 16 | 2.358 | 2.032 | +1.89 | -0.47 | yes |
| | 8 | 0.513 | 20 | 0.489 | 0.427 | +2.86 | +2.38 | yes |
| | 8 | 0.466 | 22 | 0.245 | 0.215 | -0.56 | -0.81 | yes |
| | 12 | 0.641 | 16 | 0.439 | 0.371 | +3.14 | +2.70 | yes |
| | 12 | 0.513 | 20 | 0.057 | 0.050 | +3.20 | +3.14 | yes |
| | 12 | 0.466 (the example's 0.45) | 22 | 0.023 | 0.021 | -0.38 | -0.40 | yes |
| | 12 | 0.366 | 28 | 0.002 | 0.002 | -0.14 | -0.14 | yes |
| diamond 8-atom cube, PBE (`examples/3d_diamond_benchmark`) | 8 | 0.562 | 12 | 185.4 | 103.3 | -717.9 | -903.3 | no |
| | 8 | 0.481 | 14 | 66.15 | 45.26 | -115.0 | -181.2 | no |
| | 8 | 0.421 | 16 | 28.32 | 20.83 | +98.32 | +70.00 | no |
| | 8 | 0.374 (the benchmark's a/18) | 18 | 14.08 | 10.07 | -34.01 | -48.10 | no |
| | 8 | 0.281 | 24 | 2.042 | 1.481 | +0.35 | -1.70 | yes |
| | 8 | 0.187 | 36 | 0.108 | 0.078 | +0.11 | 0.00 | yes |
| Al fcc 4-atom cube, PBE, 1000 K | 12 | 0.547 | 14 | 0.423 | 0.486 | +14.10 | +13.67 | yes |
| | 12 | 0.478 | 16 | 0.140 | 0.180 | -15.44 | -15.58 | yes |
| | 12 | 0.383 | 20 | 0.023 | 0.030 | -5.76 | -5.78 | yes |
| | 12 | 0.319 | 24 | 0.005 | 0.006 | -4.61 | -4.62 | yes |
| MgO 8-atom cube, PBE | 12 | 0.398 | 20 | 21.85 | 21.29 | +964.5 | +942.7 | no |
| | 12 | 0.332 | 24 | 5.310 | 5.032 | -18.64 | -23.95 | no |
| | 12 | 0.284 | 28 | 1.386 | 1.293 | -106.6 | -108.0 | yes |
| | 12 | 0.249 | 32 | 0.388 | 0.366 | -7.79 | -8.17 | yes |
| | 12 | 0.199 | 40 | 0.041 | 0.039 | +0.17 | +0.13 | yes |
| | 12 | 0.142 (`examples/3d_hybrid_functionals/MgO_k333`'s 0.14) | 56 | 0.001 | 0.001 | -0.88 | -0.88 | yes |
| Al, c/a = 1.07 (unequal steps), PBE, 1000 K | 12 | 0.450 | 16 x 16 x 18 | 0.122 | 0.155 | -2.38 | -2.51 | yes |
| | 12 | 0.360 | 20 x 20 x 22 | 0.021 | 0.027 | -5.67 | -5.69 | yes |
| | 12 | 0.300 | 24 x 24 x 26 | 0.004 | 0.006 | -3.68 | -3.68 | yes |

The pseudopotentials: `examples/3d_Si/Si_POTRE.DAT`,
`examples/3d_diamond_benchmark/C_POTRE.DAT` and the PBE files of `NCPPs`
for Al, Mg and O. Larger cells take a smaller difference (`k^2`): 64 silicon
atoms at 0.466 bohr 0.005 meV (estimate 0.004), 216 carbon atoms at a/18
1.32 meV (0.96). The supercell was lower in every case measured; against the
converged energy neither form is closer at every spacing (the sign of the
rest of the grid error decides). Where the default takes the supercell the
difference is at most 2.4 meV per atom (silicon, order 8, 16 points), a
quarter of the bound, and mostly below the grid error of either route from
the converged energy (not for silicon at order 8 and 16 points and diamond
at 24 points, where the k-points happen to lie within 1.9 and 0.35 meV of
it). No correction removes it exactly: the measured difference is 0.74 to
1.8 times the estimate (aluminium and diamond at the extremes), so the
estimate is reported in the NOTE, not added to the energy; the k-points with
the Bloch form (`PARSEC_KPOINT_KINETIC=bloch`) give the supercell's energies
exactly. On the CPU the supercell is 35 to 42 times slower than the
reference k-points (64-core node, silicon cube at 16 and 22 points: 5.7
against 239.1 s and 11.8 against 422.2 s), which `auto` therefore keeps there.

#### The sectors of a k-point

The Bloch Hamiltonian of a k-point acts on the periodic factor `u` (the u
gauge). An operation `r -> R r + t` of the cell (a signed permutation of the
orthorhombic cell with a translation of whole grid steps: the point group about
the origin or, since 2026-11-18, about a symmetric site off it
(`PARSEC_PERIODIC_SYMMETRY_CENTRE`, through `CellGroup`), the half-cell
translations of a cell that repeats a smaller one, and the operations with a
fraction of the cell as translation where they close into a group, as at the
Gamma point) permutes the grid points and commutes with `H`; it
moves the phase `e^{ik.r}` to `e^{i(Rk).(r - t)}`, so where `R k = k` it commutes
with `H_k` as well. `R k = k` exactly, not modulo a reciprocal lattice vector:
where `R k = k + G` the operation carries `H_k` into
`H_{k+G} = e^{-iG.r} H_k e^{iG.r}`, so what commutes with `H_k` is
`e^{iG.r} P_g`, a complex operator without the real `+-1` characters of the
sector bases. (The Bloch form of `PARSEC_KPOINT_KINETIC=bloch` obeys that
identity exactly, since `e^{iG.r}` is periodic on the grid; the default
gradient form only to its discretization, about 1e-8 Ry: see "The kinetic
term of a k-point" above.) On the grid each term commutes:
the stencils of a uniform periodic grid are invariant under its translations and
turn under `R` as a scalar and a vector (`k.grad` is invariant where `R^T k = k`),
and the Bloch projectors of an atom go over into those of its image times one
constant phase, which drops out of `B D B^H`. These operations form the little
group `G_k`; its largest subgroup `H_k` of commuting involutions (the half-cell
translations, the twofold axes and the mirrors that fix `k`) has the real
characters `+-1`, so the sector bases of the Gamma point (real) reduce the complex
Hermitian `H_k` the way they reduce a real symmetric operator, from the
representative rows (`ReflectionRepresentationDecomposition.reduce_hermitian_operators`),
and the projectors of one atom per orbit of `H_k` serve every sector. The spectrum
of `H_k` is the union of the sector spectra (silicon cube, seven k-points, both
kinetic forms: the lowest 30 levels to 1e-13 Ry): each k-point becomes `|H_k|`
eigenproblems of about `1/|H_k|` of its rows and states, and the work of its
eigensolver (rows times states squared) falls by about `|H_k|^2`.

The potential must be invariant under the operations: an operation is taken where
the density is symmetrized over a group that holds it (`Symmetrize_Charge_Density`,
the wedge of a reduced mesh), where the k-point set maps onto itself under its
`R` (with time reversal, modulo the reciprocal lattice, at equal weights), or for
a translation of the cell (every `rho_k` is invariant under it), and the group
those generate; the sector Hamiltonian takes the potential averaged over each
orbit (its invariant part). A sector of `n_s` of the `N` rows solves
`ceil(c M n_s / N) + 6` states (`M` the states of the k-point, `c` =
`PARSEC_KPOINT_SECTOR_MARGIN`, 1.3) and the k-point keeps the lowest `M` levels
of the union; after each solve every sector with fewer states than rows must
reach the `M`-th level of the union, else it gets 1.5 times its states and the
step is solved again. Since 2026-11-18 (`PARSEC_KPOINT_SECTOR_REFIT=auto`) each
sector keeps, after the first solve of the loop, the levels it holds among the
lowest `M` of its k-point plus max(6, 10 %), and a sector so cut that later
misses the `M`-th level grows by its estimated deficit, the rule of the Gamma
sectors (`off`: the first counts kept, as before). The density of a sector, `|U v|^2`, is expanded on the
host and summed by the job of the driver. The scalar and the collinear-spin
k-point SCF take the split (not with a meta-GGA, LDA+U or fixed occupations; the
loop with refined levels, the last inner loop of the Fock loop of a hybrid, since
2026-11-19, `PARSEC_KPOINT_SECTOR_LEVELS`); the two channels of a k-point share its sectors (both spin
densities are symmetrized over the same group, the static operators are built
once). The forces (in one process, and since 2026-11-17 on the ranks of the MPI
k-point pools, each adding the nonlocal terms of its tasks, summed over the
ranks; `../FORCES_AND_RELAXATION.md`) and the kept orbitals come
from the orbitals of the sectors expanded to the whole grid, `U v`, in the order
of the levels of the k-point. The exchange of a hybrid functional,
`-a_x xi xi^H` of a k-point, joins the Hamiltonian of a sector as
`-a_x (U^H xi)(U^H xi)^H` (it commutes with the group where the occupied orbitals
of every k-point are symmetry adapted, as those of the sectors are), and its
energy `sum f <v|U^H V U|v>` is taken on the sector; the states of the sectors
go from one loop to the next (the Fock loop, the next structure) with their
layout and numbers of states, and a solver of other sectors starts afresh. LDA+U
of unpolarized orbitals (`PARSEC_KPOINT_SECTOR_LDAU=on`, opt-in since 2026-11-22) joins the
Hamiltonian of a sector as `(U^H P_k) D (U^H P_k)^H` of the Bloch projectors
`P_k` of the k-point, and the occupation matrices of a sector come from its
states through the same reduced projectors. A
sector that misses a level gets more states and is solved again alone: the
solvers change the states of some tasks (`resize_tasks`) and keep the Ritz
states of the others, a new solver is made only where the pools could not hold
the grown task (`PARSEC_KPOINT_SECTOR_WARM_GROWTH`, opt-in: the first solve of a
grown sector from its refined Ritz vectors). On A100 (diamond, first runs): spin-polarized 6 x 6 x 6 cubes with
the 3 x 3 x 3 mesh on 16 GPUs 89.4 s instead of 392.2 s (8.2 instead of 41.8 GiB per
GPU), of which 37 s were the spin-polarized PBE on the host of every rank; with it
on the GPU (`V_xc/periodic_xc.py`, since the same day) 50.9 s (0.5 s of XC, 8.9 GiB),
the same total energy to the digits printed; PBE0 3 x 3 x 3 cubes (216 atoms) with the 1 x 1 x 2 mesh in one process
on four GPUs 53.5 s instead of 69.4 s (the inner SCF steps of the Fock loop 0.7
instead of 2.1 s; the plans of the sectors and their static operators are kept in
the projector cache for every loop of the structure, and `U^H xi` is formed on
the GPU of `xi`); one relaxation step of 3 x 3 x 3 cubes with the 2 x 2 x 2 mesh in
one process 83.1 s instead of 97.9 s.
The tasks of the k-point pools are then the sectors, the largest first, and the
pools are planned by the size of each task (`PARSEC_KPOINT_POOL_ASSIGN=auto`):
each sector goes to the pool that would finish it first among those whose GPUs
hold it and whose nodes hold what their ranks keep on the host, so that a few
large sectors get pools of their own. Planned as if every sector were the
largest one and dealt round-robin (`cyclic`, the rule of before), the eight
65 GB sectors of diamond 9 x 9 x 9 with the 2 x 2 x 2 mesh went to one-rank
pools beside smaller ones, their bases moved to host memory, 260 GB on a node
of 256.

On the device the reduced projectors of a sector are complex matrices of no
stencil form; the CuPy product of a complex CSR matrix and a row-major block
(cuSPARSE, with layout conversions) ran about forty times below the stencil
kernel per row and column, so a kernel of their own forms one entry of the
result per thread along the columns of a row (`Hamiltonian/complex_csr_device.py`;
rows of many entries, the adjoint of the projectors, split over 32 lanes summed
in a fixed tree). Only the operators of the sectors take it; every other operator
keeps the products of before. The kinetic term of a sector (with `|k|^2` and the
potential in the same pass) took that kernel too, reduced from a complex matrix
of the whole grid that the host built for every k-point, until 2026-11-19. Since
then (`PARSEC_KPOINT_SECTOR_STENCIL=auto`, `Hamiltonian/sector_bloch_stencil.py`)
it is a stencil gathered from the grid (in the gradient form, the default; in the
Bloch form the driver still builds the complex Bloch Laplacian of the whole grid
per k-point, and the weights are read from it): row `r` of a sector (the representative
point `p_r` of its orbit `O_r`) adds, for every neighbour `q = p_r +- s e_a` of
the axial stencil, the complex weight of that neighbour (`c_{a,s} -+ 2i k_a g_{a,s}`
of the gradient form, `c_{a,s} e^{+-i k_a s h_a}` of the Bloch form, the same at
every point) times the real factor `chi(q) sqrt(|O_r| / |O_q|)` of the sector
basis times the entry of the sector column of the orbit of `q`: the products of
`reduce_hermitian_operators`, added in another order. A sector keeps per row and
neighbour the column (int32) and a code of its real factor (uint8, a palette of
at most 256 values), 5 bytes against the 20 of a CSR entry, and the kernel takes
a whole step of the Chebyshev recurrence of the filters, so the four elementwise
passes that follow the CSR product are not made. Where the kinetic matrices are
not an axial stencil of the grid (checked row by row once per structure) or the
factors of a sector need more than 256 values, the sector keeps the CSR. So
does every sector on the CPU route (no k-point backend on a device: `--device
cpu`, a node without a GPU, the MPI pools on CPU ranks; the same answer on every
rank, from the environment): the host applies the CSR in one scipy product,
3-4 times faster than a loop over the slots of the table (one Hamiltonian
application of 64 columns on the CPU of an A100 node, `k = (pi / L, 0, 0)`:
diamond 4^3 cubes, a sector of 23,328 rows, 42 against 139 ms; 6^3 cubes,
78,732 rows, 135 against 539 ms; two SCF steps of `Si64_k222/pbe` at 0.3 bohr,
64^3 points, its 4 k-points in 64 sector tasks on 16 worker threads of that CPU,
first runs: 105.4 s with `off`, 105.6 s with `auto`, 484.1 s with the tables
applied by a loop over their slots, the same energies). The
kernel reads the palette and the weights from shared memory and unrolls the
loop over the neighbours; on its own its product is as fast as the CSR product
(0.234 against 0.239 ms; it keeps the neighbours that the CSR merges where two
of them fall in one orbit), and one step of the recurrence takes 0.255 ms
(diamond 6^3 cubes, `k = (pi / L, 0, 0)`, a sector of 78,732 rows and 24
neighbours, 32 columns on one A100; the kinetic term 9.4 instead of 39.6 MB on
the device); the filter is 1.5-1.6 times faster with the first kernel (palette
and weights from global memory, the same bits). Whole first runs on four
A100 (Gamma-centred meshes, `run_mpi_kpoints.py` with one rank per GPU unless
noted; the final energies identical and every SCF step within 1e-8 Ry: all
printed digits the same but the last one of step 5 of `Si64_k222/pbe`, -504.37492389
against -504.37492388 Ry; the same steps and operator applications): diamond 6^3
cubes with the 3 x 3 x 3 mesh
38.7 s instead of 50.3 s (filter 14.0 instead of 25.1 worker-seconds; host peak
3.5 instead of 5.2 GB, the fullest GPU 11,835 instead of 12,105 MiB, rank 0's
CuPy pool 10.7 instead of 12.2 GB); the following with
the first kernel: the same in one
process (`run_case.py --gpus 4`) 69.2 s instead of 81.7 s, diamond 8^3 cubes with
the 2 x 2 x 2 mesh 178.5 s instead of 228.7 s (filter 86.6 instead of 134.6
worker-seconds, host peak 6.8 instead of 10.8 GB, the fullest GPU 51,781 instead
of 52,073 MiB), `Si64_k222/pbe` of
`examples/3d_hybrid_functionals` 25.1 s instead of 29.3 s (peak 0.76 instead of
0.93 GiB per GPU). `off` keeps the CSR, bit for bit. On 16 A100 (diamond, Gamma-centred meshes reduced
by Td and time reversal, the density symmetrized, the pools planned by the
sizes of the sectors): 6 x 6 x 6 cubes with the 3 x 3 x 3 mesh 18.3 s instead of
195.7 s (6.0 instead of 26.5 GiB per GPU), 7 x 7 x 7 cubes with the 2 x 2 x 2 mesh
87.7 s instead of 586.1 s (33.6 instead of 54.1 GiB), total energies within 8e-6 Ry
of the runs without sectors (the convergence criterion of these inputs is
2e-4 Ry); with the 2 x 2 x 2 mesh 8 x 8 x 8 cubes (4,096 atoms) take 85.9 s,
9 x 9 x 9 cubes (5,832 atoms) 477.1 s (58.1 GiB per GPU) and 10 x 10 x 10 cubes
(8,000 atoms) 374.1 s (49.5 GiB; `examples/3d_diamond_benchmark/README.md`).

Since 2026-11-21 a k-point run of the command line writes the line
`kpoint_symmetry_detection` to `parsec.out` and stdout, at every `Output_Level`, as
the Gamma route prints `symmetry_detection`: the cell group the sectors took (its
operations `r -> R r + t`, how many of them carry a translation) and the atom
tolerance it was built at (`PARSEC_SYMMETRY_ATOM_TOLERANCE`, at most 1e-6 bohr with
forces, `PARSEC_KPOINT_SECTOR_TOLERANCE`), with how many k-points split and the
largest group of their sectors; or why no group was searched (a basis below the
floor of 2^24 entries, `PARSEC_KPOINT_SECTORS=off`, a loop that takes no sectors:
the spinors of a spin-orbit or non-collinear run, LDA+U, a meta-GGA, ...; a grid that
is no orthorhombic box), or that the cell keeps the identity alone
(`SCF/kpoint_sectors.py`, `plan_sectors(job, notes)`). The cell group is searched
when the first k-point solver of a loop is made, so the line is written then,
before the first solve of that loop (after the set-up block, before the first SCF
iteration; `SCF/kpoint_solver.py`, `start_kpoint_setup_notes`, which the command
line starts with its reporter): a run that stops in its SCF has it. Each text is
written once per run; a later loop whose group differs (a moved structure of a
relaxation) writes its own. The K-point solver detail block of the timings after a
completed loop of the scalar and spin drivers opens with the same line
(`RunTimings.notes`). It changes nothing else.

#### Classes of sectors and real forms inside a k-point

The sectors of a k-point use the commuting involutions `H_k` of its little group
`G_k`, whose characters are the real `+-1`. The rest of `G_k` (the threefold and
fourfold axes, the mirrors that `H_k` leaves out) and the operations with `R k = -k`
still act on the sectors (`SCF/kpoint_sectors.py`, `sector_classes`): an operation
`g` of `G_k` that normalizes `H_k` carries an orbital of the sector of character
`chi` into one of the sector of `chi^g(h) = chi(g^-1 h g)` (`P_h P_g u = P_g
P_{g^-1 h g} u`), and commutes with `H_k` where the potential is invariant under it;
the antiunitary `P_g K` of an operation with `R k = -k` commutes with `H_k` as well
(`K H_k K = H_{-k}` for the real potential, stencils and projectors of the u gauge)
and maps the sectors in the same way. Two sectors of one class have one spectrum,
and in their coefficients `P_g` is a signed permutation of the rows: row `w` (the
representative point `r_w`) goes to the row of the orbit of `g r_w` with the phase
of that point in the target sector (`ClassMember`). With
`PARSEC_KPOINT_SECTOR_CLASSES=auto` one sector of each class is solved; the others
take its levels, its occupations are summed onto it, the density of the class is
the mean of the images of its density (equal to the sum of the densities of the
class where the copies hold equal occupations, which their equal levels give), and
their forces and kept orbitals come from the images of its states expanded in their
own sectors. Only operations the density is symmetrized over are taken, so the sector
Hamiltonians are images of each other at every SCF step (the orbit means of the
potential included), not only at self-consistency. This is the rule of the weighted
sectors of a cluster (`Symmetry/orbital_groups.py`) on the real characters of
`H_k`. Abelian subgroups with complex characters (a threefold axis, the
quarter-cell translations of a 2 x 2 x 2 supercell) were counted against it on the
irreducible points of the 3 x 3 x 3 and 4 x 4 x 4 meshes of Si64, MgO64 and Al32
(C64 gives the numbers of Si64): rows times states squared summed over the sectors
solved, as a fraction of a
whole k-point and weighted over the mesh, today -> classes -> the largest abelian
subgroup with its classes: Si64 0.0220 -> 0.0144 -> 0.0161 (3 x 3 x 3) and 0.0238 ->
0.0181 -> 0.0188 (4 x 4 x 4); MgO64 0.0128 -> 0.0075 -> 0.0094 and 0.0153 -> 0.0109 ->
0.0118; Al32 0.0234 -> 0.0141 -> 0.0171 and 0.0268 -> 0.0193 -> 0.0207: the classes
of the real sectors take more, because the translations of these supercells do not
commute with the threefold axes. Complex characters gain in cells without such
translations: on a threefold axis of C3v two classes of a third of the rows in
place of the two halves of `C_s` (MgO64 with a vacancy at an atom, Oh about it,
3 x 3 x 3: today 0.141, classes 0.136, the better of the two groups per k-point
0.077). `PARSEC_KPOINT_SECTOR_CHARACTERS=complex` (opt-in;
`Symmetry/complex_sectors.py`) takes, at every k-point other than Gamma, a largest
abelian subgroup of the little group as well (greedy from every cyclic subgroup;
its characters from the joint eigenvectors of its regular representation, kept as
exponents modulo the exponent of the group, the phase of a row as an int8
exponent), and keeps whichever of it and the commuting involutions gives the less
work with its classes: the sector basis `chi(g_i)/sqrt(|O_w|)` on the orbits whose
stabilizer the character leaves alone, `U^H H U` from the representative rows (the
gathered stencil with a complex palette and its kernel, `U^H B` of one atom per
orbit, whose phases of modulus one leave the compression of the projectors valid),
and the classes with `P_g K` taking `chi` to `conj(chi^g)`. The real forms below
are not taken by such sectors. Four A100, one process, first runs in fresh
processes, `real` -> `complex` (two runs each, SCF to 1e-7 Ry, energies within 3e-10
Ry, the same SCF steps): MgO64 with a vacancy, 3 x 3 x 3 mesh, 11.0, 10.5 -> 10.4,
10.5 s (eigensolver 7.6 -> 7.2 s, rows times states squared 0.119 -> 0.065, fullest
GPU 1,371 -> 1,009 MiB), 4 x 4 x 4 18.8, 18.9 -> 18.4, 18.5 s (1,801 -> 1,581 MiB);
Si64 with a vacancy 9.3, 9.1 -> 8.5, 8.5 s and 38.7, 38.9 -> 37.9, 37.9 s; the
perfect Si64 12.9, 12.3 -> 11.9, 11.8 s and 22.2, 22.5 -> 22.0, 22.2 s; the perfect
MgO64 takes no complex sectors (the same work, the same times). Diamond 3^3 with an
atom moved along (1, 1, 1) (C3v, no translations), four ranks in four pools and in
one pool sharing the columns, `real` and `complex`: all -2463.2311876133 Ry in 14
steps. The gains in time are small where the k-points are many and the sectors of
the largest k-point set the pace; the memory of the vacancy cells falls more. Not
the default for now: measured on these cells only.

A sector that some `P_g K` maps onto itself with `(P_g K)^2 = 1` has a real
structure (`SCF/kpoint_real_forms.py`): with `S` the signed permutation of its rows,
the vectors `(e_w + s_w e_pi(w))/sqrt(2)`, `i (e_w - s_w e_pi(w))/sqrt(2)` of a pair
and `e_w` or `i e_w` of a fixed row are orthonormal and kept by `P_g K`, so `T^H H_s
T` is real symmetric and the sector is solved on float64 blocks under
`PARSEC_KPOINT_SECTOR_REAL=auto`: its kinetic term is the gathered stencil of the
sector applied to the real coordinates (`Hamiltonian/sector_pair_stencil.py`: one
complex row per pair, each neighbour rebuilt from its pair, the outputs `sqrt(2) Re
y_w` and `sqrt(2) Im y_w`, the step of the recurrence in the kernel), its projectors
`T^H B = R + iI` as the real `[R, -I] diag(D, D) [R, -I]^T`, its potential the orbit
means; the orbitals handed out are `U T x`. Every sector of a centrosymmetric cell
has such a form at every k-point (the inversion is central); in diamond about an
atom (Td) the k-points on the axes and the face diagonals have one, those on the
threefold axes none. Four A100, one process, first runs in fresh processes, the
classes on in both, `off` -> `auto` (two runs each, SCF to 1e-7 Ry): Al108 (48^3
points, 270 states) 3 x 3 x 3 11.8, 11.9 -> 11.2, 11.6 s (18 sectors real; fullest
GPU 877 -> 713 MiB), 4 x 4 x 4 19.5, 19.7 -> 20.1, 20.4 s (50 real; 965 -> 875-893
MiB), energies within 8e-9 Ry (12 or 13 SCF steps instead of 13); MgO64 17.7, 18.5
-> 17.2, 17.5 s (36 real; 621 -> 563 MiB) and 29.8, 29.8 -> 29.0, 29.2 s (100 real;
705-713 -> 649 MiB), the same energies to the printed digits; the total forces of
MgO64 with Mg moved by 0.05 bohr within 8e-9 Ry/bohr of the complex sectors. The
Chebyshev filter takes as long as the complex one (Al108 3 x 3 x 3: 13.4 against
13.5 worker-seconds; its gathers read the two coordinates of a neighbour on two rows
where the complex kernel reads one complex entry, the same memory transactions),
while the orthonormalization (2.2 -> 1.6 s) and the Rayleigh-Ritz steps (3.2 -> 3.0
s) fall. With the cuSPARSE products of the real projectors of a first version the
runs were 1.2-1.6 times slower (Si64, C216, MgO64, Al108); four ranks in four pools
or in one pool sharing the columns gave the energy of one process (216 carbon atoms,
-2463.2375880796 Ry in 14 steps). Not the default: no gain in time on these cells,
only in memory. A pair-major order of the rows (the two coordinates of a pair side
by side, read at once) is the next step.

#### Real arithmetic at Gamma and at the time-reversal invariant points

At `k = 0` the Bloch Hamiltonian of the u gauge is the real symmetric
Hamiltonian of the Gamma point: no gradient term, no `|k|^2`, projectors
without phases. Since 2026-11-21 (`PARSEC_KPOINT_REAL_GAMMA=auto`,
`SCF/kpoint_real.py`) the device converter gives that point a real operator
(`DeviceBlochOperator.real_arithmetic`), whole or in the sectors of its little
group (their bases are real, so `U^T H_0 U` is real too), and the eigensolvers
of the k-points work on float64 blocks: the device solver of one process
(`chebff_spinor.solve_spinor_eigval`, which the real sectors of the spin-orbit
solver had made real-capable) and the shared columns of the MPI pools
(`chebff_columns.solve_column_eigval`, its exchanges, rounds, block route, Gram
matrices and residual kernel by the dtype of the basis, the round buffers one
allocation per name for both dtypes; on threads as ranks the real columns give
the eigenvalues of the real one-device solve to 1e-14 on 1, 2 and 3 ranks).
The kinetic term takes the real kernels of the sector stencil (the real part of
the complex kernel's result, bit for bit) and of the whole-grid stencil, the
projectors their real products. The basis holds half the bytes, the filter
reads half, the dense products need a quarter of the flops. The start vectors
are real, so the numbers are those of the complex solve to the tolerance of the
eigensolver; the orbitals handed to callers (kept orbitals, the exchange of a
hybrid after its semilocal start) are cast to complex, and a complex subspace
is not handed to a real solve. Not on the row-distributed solver of the
spin-orbit and non-collinear drivers, the band structures (the band paths in
sectors included: their one first solve leaves the buffer states unconverged,
and their levels would follow the start vectors), the CPU routes or with
`PARSEC_KPOINT_FIRST_SOLVE=chebdav`.

Gamma is one of the k-points of a mesh, so the gain is its share, and nothing
where the other points set the time (one point per GPU); the real route also
compiles kernels of its own on a first run with an empty kernel cache (9 to 11
in the runs below: the real sector stencil for each count of slots of the
sectors of Gamma and a few CuPy kernels of float64). On a small cell the
compile costs more than the share saves: with empty kernel caches (a CuPy and a
CUDA cache of their own for each run) `Si_k222/pbe` took 9.95, 9.79 -> 10.87,
10.92 s on one A100 and `C_k333/pbe` 22.69, 22.70 -> 23.78, 23.58 s, and 64
silicon atoms at 44 points per edge (85,184 points, 144 states; one task per
k-point on four GPUs, the kernels in the cache) took 13, 13, 13 SCF steps in
19.40, 19.21, 19.31 s with `off` against 14, 13, 14 steps in 19.67, 19.29,
19.63 s. In one process the peak memory of a Gamma point solved whole rose a
little as well (`Si_k222/pbe0` 1.619 -> 1.639 GiB, `hBN_k661/pbe` 1.926 -> 1.969
GiB, 58.5 -> 59.3 s). So `auto` takes the real route only where the basis of a
k-point (grid points times states) reaches `2^24` (`real_gamma_applies`, the
number of the floor of the k-point sectors as a constant of its own; every rank
takes the same answer); below it the complex solve runs as before, bit for bit
(the 2-atom cells of `examples/3d_hybrid_functionals`, `hBN_k661/pbe` and the
64 silicon atoms above: the same kernels, energies and times). Above it, first
runs on A100 ("cache": the kernels of earlier runs in the CuPy cache of the
node; "empty": a CuPy and a CUDA cache of their own for each run; the energies
the same to every printed digit unless noted):

| system | mesh, route | off, cache | auto, cache | off, empty | auto, empty |
|---|---|---|---|---|---|
| diamond 6^3 cubes (1,728 atoms), a/18 | 3 x 3 x 3, four ranks (`run_mpi_kpoints.py`) | 34.7 s, fullest GPU 11,835 MiB | 33.5 s, 11,353 MiB (2.3e-7 Ry, criterion 2e-4) | 43.7, 43.9 s | 43.7, 44.1 s |
| | 4 x 4 x 4, four ranks | 86.1 s, 27,751 MiB | 84.8 s, 26,593 MiB (1.4e-7 Ry) | 94.9 s | 95.2 s |
| diamond 4^3 cubes (512 atoms), a/18 | 2 x 2 x 2, one pool of four ranks sharing the columns (block route, sectors off) | 38.6 s, fullest GPU 14,785 MiB | 34.4 s, 14,889 MiB (5e-8 Ry) | 46.3 s | 43.1 s |
| diamond 3^3 cubes (216 atoms), a/18 | 2 x 2 x 2, one process on four GPUs, 32 sector tasks | 19.0, 19.4 s, 1,355 MiB | 18.4, 17.8 s, 1,275 MiB | 38.9, 38.7 s | 39.0, 39.1 s |
| `Si64_k222/pbe` (Fortran-derived routines) | 2 x 2 x 2, four GPUs, 64 sector tasks | 23.3 s | 21.5 s | 40.9, 40.6 s, 785 MiB | 41.2, 40.9 s, 773 MiB |
| `Si64_k222/pbe0` | the same, the semilocal start of its Fock loop | 160.6 s | 159.8 s | 180.8 s, 14,805 MiB | 180.9 s, 14,277 MiB |

With the kernels in the cache the real Gamma point saves its share, 1.5 to 11 %;
with empty caches the compile takes it back where that share is small, and the
times agree within the noise of the runs (`off` against `off` differed by up to
0.3 s), while the pool of four ranks sharing its columns keeps 7 %. GPU memory is
lower where Gamma, or each of its sectors, is a task of its own (4 % on the
pools of one rank, 6 % in one process); in a pool of several ranks sharing its
columns the fullest GPU holds about the same (+0.7 %; the memory pool of rank 0
14.2 -> 15.0 GB, cached blocks of both dtypes; 15.6 GB, fullest GPU 15,363 MiB,
before the round buffers became one allocation per name).

A time-reversal invariant point other than Gamma (`k = G/2`) is real in the psi
gauge `psi = e^{ik.r} u` where its kinetic term is the Bloch form
(`PARSEC_KPOINT_KINETIC=bloch`, or `trim`: that form at these points only, the
gradient form elsewhere): the twisted Laplacian of the cell (the neighbours
across a face along an axis where `k = pi/L` taken with `-1`), the real Bloch
projectors `sum_R e^{ik.R} beta(r - tau - R)` and the potential, `e^{ik.r}` times
the u-gauge operator times `e^{-ik.r}` to 2e-15. The gradient form of PARSEC
1.5 is not real there (it would need an operation that maps `k` onto `-k`
exactly on the grid, inversion or a twofold axis across `k`). With
`PARSEC_KPOINT_TRIM_REAL=auto` (the default since 2026-11-21; it acts only where
an opt-in form takes the Bloch form) a point solved whole runs in that gauge in
float64 (`PsiGauge`; the pools take a twisted real stencil kernel,
`BlochStencil.from_psi`; the density `|psi|^2 = |u|^2` needs no conversion, the
forces and orbitals take `u`), and a point in sectors solves one sector of each
conjugate pair: the antiunitary `A u = e^{-iG.r} conj(u)` (complex conjugation in
the psi gauge) carries the sector of a character `chi` of `H_k` into that of
`chi lambda`, `lambda(h) = e^{-iG.t}` (`-1` on the half-cell translations along
the axes where `k = pi/L`), whose reduced Hamiltonian is `Phi conj(H_chi) Phi^*`
(silicon cube at X and at (1, 1, 0) pi/L: one spectrum to 1e-11). The partner
takes the levels of the sector solved, its occupations go onto that sector's
columns (the densities agree pointwise), its orbitals are `A u` and its
nonlocal forces those of `u` (`SectorKPointSolver`, an unpolarized loop
without the exchange of a hybrid; the silicon cube with one sublattice moved:
the forces of `off` to 1e-8 Ry/bohr, whole and in sectors). The energies are
those of the u gauge to round-off and the tolerance of the eigensolver. Four
A100, first runs, with `PARSEC_KPOINT_KINETIC=trim`: silicon 2^3 cubes at 44
points per edge (one task per k-point, the psi gauge) 13.0, 13.1 -> 8.5, 8.3 s,
1,911 -> 1,149 MiB; 216 carbon atoms (32 sector tasks, 6 solved by their
partners) 18.3, 18.4 -> 14.7, 15.0 s; `Si64_k222/pbe` (64 sector tasks, 16
solved by their partners) 23.5 -> 20.3 s; diamond 6^3 cubes with the 4 x 4 x 4
mesh on four ranks (16 of 120 sectors) 87.8 -> 76.0, 76.2 s, fullest GPU 26,593
-> 23,139 MiB (84.8 s with the gradient form; the driver's complex Bloch
Laplacians of the whole grid take 1.9 GB more host memory); with
`PARSEC_KPOINT_KINETIC=bloch`, diamond 4^3 cubes, sectors off, four ranks (the
twisted kernel) 35.9 -> 17.7 s, 14,343 -> 7,271 MiB (9.5e-7 Ry). The psi gauge
compiles kernels of its own too (8 more): with empty kernel caches the 8-atom
silicon cube at 20 points per edge (8,000 points, 24 states, one GPU) took
7.90, 7.94 -> 8.48, 8.41 s, while 64 silicon atoms at 32 and 44 points per
edge (144 states) took 34.0, 33.8 -> 33.6, 33.6 s and 39.5, 39.4 -> 38.6,
39.3 s and 64 carbon atoms at a/18 (46,656 points, 144 states) 34.3, 34.7 ->
34.0, 34.1 s on four GPUs; so a point solved whole takes the psi gauge from a
basis of `2^22` grid points times states (`psi_gauge_for`), the u gauge below
it. The conjugate pairs need no kernel of their own (they come with sectors,
above the floor of the sectors).

The Bloch form at these points is another discretization: `trim` is not the
default. Its energy lies below the gradient form's at every spacing measured,
by an amount that falls as `h^{2M}` and with the size of the cell (`k^2`): with
the 4 x 4 x 4 mesh (SCF to 1e-7 Ry; "converged": the finest grid, where both
forms agree to 0.014 meV per atom; meV per atom):

| system | points per edge | trim - gradient | gradient - converged | trim - converged |
|---|---|---|---|---|
| Si 8-atom cube, LDA, order 12 | 16 | -0.055 | +3.078 | +3.023 |
| | 20 | -0.007 | +3.235 | +3.228 |
| | 24 | -0.001 | -0.231 | -0.232 |
| | 28 | -0.0003 | -0.134 | -0.134 |
| diamond 8-atom cube, PBE, order 8 | 12 | -23.24 | -771.9 | -795.2 |
| | 14 | -8.285 | -131.8 | -140.1 |
| | 16 | -3.546 | +88.42 | +84.88 |
| | 18 (a/18) | -1.764 | -37.31 | -39.08 |
| | 24 | -0.256 | -0.145 | -0.401 |
| Al fcc 4-atom cube, PBE, order 12 | 14 | -0.052 | -2.743 | -2.795 |
| | 16 | -0.017 | -9.732 | -9.749 |
| | 20 | -0.003 | -5.289 | -5.292 |
| MgO 8-atom cube, PBE, order 12 | 24 | -0.665 | -19.05 | -19.71 |
| | 28 | -0.173 | -106.0 | -106.2 |
| | 32 | -0.049 | -7.018 | -7.067 |
| | 40 | -0.005 | +1.055 | +1.050 |

As for the 2 x 2 x 2 meshes ("The kinetic term of a k-point", where `trim` is
`bloch`), it is closer to the converged energy only where the rest of the
grid error is positive, and the difference exceeds the SCF criterion of the
benchmark inputs (diamond 6^3 cubes with the 4 x 4 x 4 mesh at a/18: 5.2e-3 Ry,
criterion 2e-4 Ry; `Si64_k222/pbe` 1.0e-5 Ry): by the rule of the defaults it
stays opt-in, and with it its real arithmetic.

#### ARPACK, Dynamic_Diag_Tol, Fix_Number_Eigenvalues and the TEST mode

These PARSEC controls (the reports of the
Fortran-parity round) run in the accelerated SCF loop, which is the shared
one of `SCF/single_point.py` and `SCF/spin_polarized.py`; an input that
gives none of them runs as before, bit for bit.

- `Dynamic_Diag_Tol`: the loop hands the tolerance of each iteration (0.1 in
  the first of every SCF loop, then `max(min(tol, SRE/10), Diag_Tolerance)`)
  to the eigensolver settings of that iteration; the CuPy CHEBDAV first
  solve of every sector, and of a sector that the spectral bracket grows,
  uses it, CHEBFF and SUBSPACE use none (as in PARSEC). `Restart_Run`
  switches it off in the parser.
- `Fix_Number_Eigenvalues` (with `Restart_Run`): no `Subspace_Buffer_Size`
  buffer; on the full grid the solver keeps `States_Num` vectors (PARSEC's
  `nn = nec`). In the symmetry sectors, since 2026-11-21
  (`PARSEC_FIXED_STATES_BUFFER=auto`), the run solves `States_Num` states with
  the working buffer and the sector counts of a run without the keyword, and
  a NOTE says so; the restart file must hold `States_Num` levels. Under `off`
  (before) the sectors are bracketed by the first solve of the run, then each
  keeps the number of the lowest `States_Num` levels it holds (at least one)
  and no later solve grows or trims it
  (`CuPySymmetrySCFEigensolver._fix_sector_states`; in the weighted sectors of
  a subgroup, since 2026-11-19 by default, the levels are counted with their
  weights: `PARSEC_ORBITAL_GROUP_FIXED_COUNT`). A restart reads no stored
  orbitals (`restart_run.F90`), so that first solve starts anew, and
  its counts, frozen without a buffer, lost occupied levels where few states
  are empty.
  PARSEC takes those counts per representation from the restart file; the
  sectors of this code need not be the representations of that file.
  `orbital_sector_final_state_counts` among the backend details shows them.
- `Eigensolver: none`: PARSEC's TEST mode stops after the accelerated
  preparation (`reporter.setup` and the execution resources) with
  " TEST MODE COMPLETE" and no archive.
- `Eigensolver: arpack` on the CuPy eigensolvers: `Eigensolvers/arpack.py`,
  a device thick-restart Lanczos in ARPACK's role, one per sector (or on the
  full grid), with the parameters of `arpk.f90z` (`ncv = 2 nev + 5`,
  `which = 'SA'`, `180 nev` restarts, `Diag_Tolerance`), ARPACK's DGKS
  reorthogonalization, convergence test and restart rule (`dsaitr`,
  `dsconv`, `dsaup2`), the `dlarnv` start vector of every call from the
  seed the previous call left, eigval.F90's five calls and its error 402.
  Without `Chebyshev_Filtering` it solves every SCF iteration from a new
  start vector, with it once before the device SUBSPACE passes. The
  iteration report lists the converged pairs and products per
  representation, as `eigval.F90` does. A sector is solved on its device;
  several devices serve several sectors but never share an ARPACK basis.
  ARPACK with spin-orbit coupling, and every input whose backend does not
  resolve to CuPy, still runs on the reference solver.

DIAGLA stays on the reference solver: PARSEC itself refuses it
(`usrinputfile.F90` ierr 126, "to be removed"), it works with blocks of one
vector and a host decision after every product, its preconditioner
(`fdo5scl`) is defined on the full-grid neighbour map, and its cost per
product is that of the device ARPACK above, which is already the slower
solver on the device.

Validation (A100, 1 and 4 GPUs, against `reference_main.py` on the same
input): ARPACK on the four sectors of H2O, total energy within 1.3e-13 Ry and
eigenvalues within 3.1e-11 Ry; with `Chebyshev_Filtering` and a criterion of
1e-8 Ry 4.3e-14 Ry and 4.5e-9 Ry; spin-polarized H2O (full grid) 1.3e-13 Ry
and 3.3e-11 Ry. CH4: the occupied levels within 1.5e-12 Ry; the reference
ARPACK on the full grid misses the third copy of the empty t2 level, which
the sectors split (0.1006 Ry against 0.1529 Ry). `Dynamic_Diag_Tol` with
CHEBDAV on sectors, criterion 1e-8 Ry: 1.0e-13 Ry and 1.9e-9 Ry (18 against
24 steps); spin 1.4e-14 Ry and 1.1e-11 Ry. `Fix_Number_Eigenvalues` after a
restart: 1.4e-13 Ry, occupied levels within 5e-9 Ry (the empty top levels,
which have no buffer, within 3e-4 Ry). On the full grid (`--symmetry off`)
every one of these runs gives the SRE history of the reference to the
printed digits.

First calculations of C185H124 (309 atoms, 1,099,136 points, 605 states,
four sectors of 274,784 rows) on A100 nodes shared with other requests (the
seconds are indicative; peaks are those of the process, from nvidia-smi):

| input | 1 GPU: program s, SCF steps, peak MiB | 4 GPUs: program s, peak MiB of the fullest |
|---|---|---|
| CHEBDAV (the input) | 12.2, 11, 4,440 | 11.7, 3,334 |
| `Dynamic_Diag_Tol` | 22.2, 23, 26,380 | 20.7, 23,948 |
| CHEBFF | 14.6, 20, 4,324 | 15.7, 1,958 |
| ARPACK + `Chebyshev_Filtering` | 37.4, 11, 3,344 | 21.5, 2,302 |
| ARPACK in every step | 299, 9, 6,104 | 148, 3,634 |
| `Fix_Number_Eigenvalues` (restart from the converged file) | 11.9, 1, 6,798 | 10.2, 5,294 |
| TEST mode | 2.4, 0, 574 | 3.1, 460 |

With the launch profile of the measured series on four GPUs the same
inputs took 7.7 s (CHEBDAV), 13.7 s (`Dynamic_Diag_Tol`), 17.6 s (ARPACK with
`Chebyshev_Filtering`), 8.7 s (`Fix_Number_Eigenvalues`) and 1.5 s (TEST
mode), with the same total energies to the printed digits and peaks within
0.3 GiB of the plain runs (`Dynamic_Diag_Tol` 1.0 GiB more on its fullest
GPU).
ARPACK takes about 4,300 products per sector and SCF step (28 per pair),
1.8 ms each with its reorthogonalization: 30 s a step on one GPU against
0.16 to 4.7 s of CHEBDAV/SUBSPACE. The reference path is far slower:
C35H36 (421,000 points, 130 states, full grid on the host) took 5,046 s of
SCF with the reference ARPACK against 39 s on one A100 (the same 9 steps and
total energy to the printed 1e-8 Ry), and 597 s with the reference CHEBDAV
against 2.9 s. The dynamic tolerance makes the first CHEBDAV solves loose:
their spectral bracket then grows sectors, in the first step (two sectors,
157 to 236 states) and again in the fourth (two, 126 to 189), and the
CHEBDAV restart of a sector after SUBSPACE passes raises the pool from 3.8
to 24.5 GiB while its arrays take 0.4 GiB. The code before these features,
driven through the same API, gives the same counts and the same peak: the
growth path is the existing one, which the default inputs did not reach. A `Fix_Number_Eigenvalues` first solve under
`PARSEC_FIXED_STATES_BUFFER=off` starts without a buffer, so sectors that hold more than their share grow once before
the counts are fixed (two of four on this cluster); under the default `auto` it is a first solve with the buffer, as
without the keyword.

### Exchange-correlation on the device

`V_xc/device_xc.py` evaluates on a CUDA device the functionals that the
readable package otherwise evaluates on the host: PW92 (`pl`), X-alpha,
Wigner, Hedin-Lundqvist, LB94 and Casida-Salahub (`lb`, `cs`), PBE and PW91,
and the spin forms of PW92, LB94, Casida-Salahub, PBE and PW91 that the loop
of two spin channels evaluates (`evaluate_spin_xc`). CA-LDA of a cluster keeps
its own path, the native C++ evaluator; CA-LDA and CA-LSD of a periodic cell run
here since 2026-11-18 (`PARSEC_PERIODIC_CA_XC=auto`, on the orbits of the
sectors at Gamma and on the full grid of the k-point loops and the periodic spin
drivers; `host` keeps the host formula).

- The discretization is that of `parsec_python.V_xc`: the density gradient
  of `pbe.py` (`density_gradient`: the centered stencil of `Expansion_Order`,
  zero outside the cluster domain, wrapped around an orthorhombic cell),
  summed in the order of `scipy.ndimage.correlate1d`, and the GGA potential
  `df/dn + D.T (2 df/dsigma D n)`, the derivative of the discrete energy.
  The pointwise formulas of `device_xc_kernels.cu` translate the NumPy ones
  operation by operation and are compiled with `--fmad=false`; the results
  differ from the host's only in the last bits of CUDA's elementary
  functions (`cbrt`, `pow`, `log1p`, `expm1`, `exp`, `asinh`) and in the
  order of the energy sum: fields to 1e-15 relative, energies of complete
  calculations to 1e-13 Ry for small molecules.
- With representation sectors the functional is evaluated once per orbit,
  from the orbit values at the stencil neighbours of each representative.
  The GGA flux `2 f_sigma D n` is a vector field: at another point of an
  orbit it is that of the representative with the signs of the reflection
  that carries one onto the other (a byte per point of the box). No
  full-grid field is formed; the host path expands the density to the full
  grid and averages the three result fields back.
- The evaluator runs on the device of the Hartree objects
  (`PARSEC_HARTREE_DEVICE`), the current device where there is none. It
  keeps the value map of the box (int32), the reflection signs (uint8), the
  representative positions and the orbit multiplicities there (35 MB for
  the 677,542 orbits of C795H300) and takes about eleven fields of the
  values during an evaluation. The SCF loop hands it the host density of
  each step and receives the potential, energy density and energy.
- The memory rules of the symmetry sectors count those bytes on that device
  (since 2026-10-30): the driver names `planned_device_bytes` of the
  evaluator (what it keeps, the core density and the most fields of one
  evaluation: 4 to 11 fields unpolarized, 5 to 20 with spin; from the
  shapes alone, within 1 % and 60 KB of what CuPy's pool holds after an
  evaluation) to the sector eigensolver (`resident_device_bytes`), whose
  rule of the state storage (0.974 of a device) and rule that shares a basis
  (0.85, and the room of a condition helper) add them on the device that
  holds them. Nothing is named for CA (its evaluator runs on the host) or
  with `PARSEC_CUPY_XC=off`: those runs count as before. PBE of the
  nanodiamond series (D2 orbits, A100 of 79.25 GiB): 90 MiB for 3,480
  electrons, 255 MiB for 10,456, 502 MiB for 23,768, 750 MiB for 39,368;
  the count of the fullest device of a sector shared by four devices of
  39,368 electrons goes from 0.6944 to 0.7037 of a device, of 10,456
  electrons on one device from 0.9763 to 0.9795 (spilled either way), of
  29,576 on two from 0.7475 to 0.7543: no decision of the measured series
  changes.
- Each functional compiles only the kernels it launches (NVRTC, once; CuPy's
  kernel cache keeps them).
- Grids: cluster domains and orthorhombic periodic cells. The
  non-orthogonal grid and wires and slabs keep the host functionals (a
  Gamma-point wire or slab on the accelerated driver evaluates its functional
  on the host: a few milliseconds per step, "Periodic features on the
  device" below).

| `PARSEC_CUPY_XC` | functionals on the device (CuPy backend) |
| --- | --- |
| `auto` (unset or empty) | the functionals added for Fortran parity: PW92, X-alpha, Wigner, Hedin-Lundqvist, PW91, LB94, Casida-Salahub, and PBE (since 2026-10-30; it stayed on the host before), unpolarized and spin-polarized, in the accelerated driver (clusters and the Gamma point of an orthorhombic cell); the spin PBE of the reference periodic drivers stays on the host (Periodic features on the device, below) |
| `on` | the same in the accelerated driver; also the spin PBE of the reference periodic drivers |
| `off` | none: every functional on the host, as before the device evaluators (PBE: as before 2026-10-30, bit for bit) |

`xc_implementation` among the backend details names the path, for example
`CuPy float64 discrete-variational PBE on the symmetry orbits` or
`vectorized NumPy discrete-variational PBE`.

PBE of C795H300 (k06 of `examples/0d_nanodiamonds/gpu_benchmark` with the
PBE files of `NCPPs/PBE`, C local `d`; D2 sectors, 677,542 orbits; 11 SCF
steps to 2e-4 Ry; code defaults otherwise; A100 of 80 GB, requests that
shared the CPUs of a node with others):

| | 1 GPU host | 1 GPU device | 4 GPUs host | 4 GPUs device |
| --- | --- | --- | --- | --- |
| one evaluation of the final density | 1.13 s | 3.9 ms | 1.12 s | 3.1 ms |
| XC of the SCF (initial + 11 steps) | 14.7 s | 0.28 s | 14.2 s | 0.19 s |
| SCF wall time (`parsec.out`) | 65.6 s | 50.5 s | 51.1 s | 37.9 s |
| `run_scf` with the download of the final orbitals | 101.7 s | 88.1 s | 86.7 s | 73.2 s |
| device memory peak (fullest device) | 22,562 MiB | 22,632 MiB | 8,718 MiB | 8,818 MiB |

A second pair, without the evaluations of the first row: XC 15.3 and
0.18 s, SCF wall time 65.8 and 49.7 s on one GPU, 14.2 and 0.16 s, 51.1 and
39.0 s on four, the same peaks; on four GPUs the evaluator sat on the first
device (that of the Hartree objects), the other three peaked alike.

The two paths gave the same SCF history (11 steps, the same residuals to
four digits), total energies 5.8e-10 Ry apart (-9421.2233762 Ry; the
Hartree and electron-ion terms of 7e5 and 1.5e6 Ry moved by 3e-9 and
1.6e-9 Ry), levels at most 1.8e-12 Ry and forces at most 6.5e-13 Ry/bohr
apart. On these numbers the device became the default for PBE on
2026-10-30 (it stayed on the host before, because existing calculations
ran it there); `PARSEC_CUPY_XC=off` gives the former host path bit for bit.
The change against the host path, laptop (RTX 5070), accelerated CLI, the
same iterations in every case: water 2.8e-14 Ry (17 steps), spin-polarized
water 0 Ry in the total and 5e-15 Ry in the levels, C5H12 at 0.30 A 6.3e-13
Ry, C35H36 at 0.20 A (420,504 points, D2 sectors, 13 steps) 1.3e-11 Ry of
-441.90 Ry, levels 8.6e-13 Ry. The XC of the C35H36 SCF took 0.021 instead
of 2.8 s and the SCF 4.6 instead of 7.8 s (two runs each, one at a time);
water spends 0.01 s instead of 0.04 s (on a first run of a node the PBE
kernels compile once, about 1 s on the laptop, 0.5 s on an A100 node). The
reference CLI, CA and the other functionals are unchanged bit for bit.

#### libxc on the device

The functionals of libxc (`LibXC_Functional`, PARSEC 1.5's `Use_LibXC`; see
"Functionals of libxc" in the package README) are evaluated on the GPU when
the library that `PARSEC_LIBXC_LIBRARY` names is a CUDA build of libxc 7.1 or
later (`bash scripts/build_libxc.sh --cuda --cuda-arch 80`; on Perlmutter
`<prefix>/libxc/7.1.2-cuda-sm80/lib/libxc.so`).
`V_xc/device_libxc.py`:

- libxc's own kernels evaluate the functional on the device pointers of
  CuPy arrays: each functional is initialized once per process and device
  with `xc_func_init_flags(..., XC_FLAGS_ON_DEVICE)` (libxc 7.1; the same
  library initializes the host functionals of the reference path with
  `XC_FLAGS_ON_HOST`), called through ctypes with the raw pointers. libxc
  launches on the legacy default stream: the current CuPy stream is
  synchronized before the call and the device after it.
- Around it the stencils of the device functionals above: the density
  gradient and the transpose of the discrete potential (`xc_gradient`,
  `xc_adjoint` of `device_xc_kernels.cu`), on the full grid or once per
  symmetry orbit with the reflection signs of the flux, and packing kernels
  (`device_libxc_kernels.cu`: `sigma`, libxc's interleaved spin layout, the
  fluxes `2 v_aa D n_a + v_ab D n_b`) compiled with `--fmad=false` in the
  order of the NumPy code. Host and device agree to round-off (libxc's
  elementary functions on two processors).
- Unpolarized, spin (also the spin channels in the symmetry sectors) and
  the semilocal part of a libxc hybrid in the device Fock loop.
  `planned_device_bytes`: 18 fields of the values unpolarized, 36 with spin
  (37 with a core density), counted by the memory rules of the sectors.
- A meta-GGA is evaluated pointwise by the same evaluator on CuPy arrays
  (`LibxcSemilocal`, `supports_device`); its grid terms belong to the
  meta-GGA Hamiltonian.
- A host-only libxc in a GPU run: the host evaluates libxc (expanded to the
  full grid in the sectors, as for any host functional); `xc_implementation`
  names the path (`libxc CUDA float64 ... on the symmetry orbits` on the
  device). `PARSEC_CUPY_XC=off` keeps libxc on the host as every functional.
- `python -m parsec_python.acceleration.benchmarks.libxc_timing parsec.in`
  times one evaluation of libxc's and of the code's device PBE on the grid
  and orbits of an input.

#### Meta-GGA functionals on the device

A meta-GGA (native `scan`, or a libxc `mgga_` functional; see "Meta-GGA
functionals" in the package README) adds an orbital term to the device
Hamiltonians and forms `tau` (and, in the orbital form of the density
gradient, `g`) where the orbitals are. `Hamiltonian/meta_gga_device.py`:

- **The term.** `H_tau = sum_c G_c^T diag(vtau) G_c` of the centered
  first-derivative stencil `G_c` of `Expansion_Order` (Rydberg, the exact
  derivative of the discrete energy), and in the orbital form
  `2 sum_c [W_c G_c + G_c^T W_c]` of `W = de/dg`. The CuPy operator adds it
  in `apply`, `apply_into` and the Chebyshev recurrence
  (`bind_kinetic_term`). The derivatives are CSR matrices on the device,
  products in one CUDA kernel of a CSR matrix with a strided block
  (`--fmad=false`), transposes stored as CSR (deterministic sums).
- **Symmetry sectors.** The groups of the sectors are commuting involutions;
  their 3x3 operations have common eigenvectors `q_a` (the Cartesian axes for
  reflections), and the derivative along `q_a` maps sector `Gamma` into
  sector `Gamma chi_a`. The term of a sector is
  `sum_a B_a^T diag(vtau) B_a` with `B_a = U_{Gamma chi_a}^T D'_a U_Gamma`,
  assembled once from representative rows and checked on a random vector;
  `tau` of the sector orbitals needs no full-grid field. `W` turns with the
  operations; its component along `q_a` has the character `chi_a` and acts
  on the orbits shared by a sector and its target. The fields of the
  orbital form leave and enter the SCF as Cartesian components at the orbit
  representatives (`OrbitVectorFields`; any commuting signed-permutation
  group).
- **Functional.** `V_xc/meta_gga_xc.py` evaluates the functional pointwise on
  the device (`supports_device`: the native SCAN, libxc's CUDA build) on the
  full grid or once per orbit with the stencils of the device GGAs; groups
  with operations that exchange axes, functionals of the Laplacian and
  host-only libxc builds take the host functional on expanded fields.
- **Filter bound.** Every bound term reports Gershgorin's bound of its
  eigenvalues (`spectral_margin`). Where `vtau` has narrow spikes (low
  density, `vtau ~ n^(-1/3)` for SCAN) the term has localized eigenvalues
  above the rest of the spectrum that the short Lanczos run from a random
  vector does not see, and the filter amplified them (H2O and O2 SCAN on
  A100 stopped at "upper_bound must be greater than lower_bound").
  `Eigensolvers/spectral_bounds.with_orbital_term` makes the filter's upper
  bound `max(lanczos, bare + m)`, `bare` the last Lanczos bound of the same
  operator without the term (`PARSEC_META_GGA_FILTER_BOUND=gershgorin`, the
  default). It is safe but wide: the top of `H_0` (the Laplacian's
  checkerboard mode, which the derivative stencil does not see) and that of
  the term add up, and the wider interval filters less per pass (H2O SCAN,
  orbital form, A100: 37 and 45 SCF iterations at h = 0.20 and 0.16 bohr
  against 25 and 34 with the Lanczos bound alone; k04: 20 either way).
  `targeted` runs a second short Lanczos recurrence from the term's
  Gershgorin row sums with random signs, which overlap the localized modes,
  and takes the larger of the two Lanczos bounds; `lanczos` keeps the
  Lanczos bound alone (comparisons only). A last guard in the subspace pass
  lowers the filter's lower end to the largest saved Ritz value below the
  upper end where they would cross.
- **Memory.** The matrices hold `2 x 3 x Expansion_Order` nonzeros per grid
  point (12 bytes each), about 0.86 KiB per point at order 12, counted by
  the memory rules of the sectors; the orbital form adds `2 W` per
  derivative on the rows of a sector and its target and the int32 maps
  between them (at most 120 bytes per point). Its first-order part runs in
  the same two CSR passes per derivative (one kernel each way); on a laptop
  GPU (H2O, h = 0.2 bohr, a sector of 95,584 rows, 16 vectors) the term takes
  2.2 ms with it and 1.96 ms without it.
- **Routes.** The term runs in one process on the GPUs of one node (one
  device per sector basis) and on the CPU backends (the host term).
  Refused: `--mpi` (with it a sector basis shared by several devices),
  `PARSEC_CUPY_MIXED_FILTER=on` (the FP32 filter) and
  `PARSEC_CUPY_RITZ_HANDOVER=triangular`. Captured filter graphs and a filter
  spread over devices give way to the ordinary filter by themselves.
- **Cost.** k04 (C185H124, h = 0.16 bohr, 4 A100, 4 sectors of 274,784 rows,
  density form): one application of the operator to 16 vectors 1.69 ms with
  the term against 0.90 ms without (the term alone 0.68 ms), `tau` 0.074 s
  per SCF iteration, 173 MB per device.
- `python -m parsec_python.acceleration.benchmarks.meta_gga_check` runs a
  molecule of the meta-GGA references or an input (`--term-timing`,
  `--trace`, `--bound-trace`) and prints the timings, the filter intervals and
  the meta-GGA statistics.

### Hartree acceleration

All accelerated backends use an associated-Legendre recurrence for the same
normalized complex multipole moments and boundary potential as the reference
SciPy-special-function implementation. It keeps only a few grid-length work
arrays instead of storing a dense density-to-boundary map. In the native
spherical path, a reusable C++ object caches angular coordinates and groups
missing stencil entries by interior row. Each SCF call then forms all moments
and `8*pi*rho - A_IB*V_B` in one compiled float64/OpenMP operation. Box
calculations retain the exact Python direct discrete-Coulomb boundary.

In the default hybrid path, this reusable Hartree boundary geometry is built
on a CPU worker while the independent reduced orbital operators are loaded or
constructed on the GPU. The main thread joins the worker before the first
Hartree solve, so no SCF work races and the resulting arrays are identical to
inline construction. Set `PARSEC_OVERLAP_HARTREE_SETUP=0` for an inline A/B
control. The output records the worker time, join wait, and estimated hidden
setup time.

After the identical boundary-corrected right-hand side is formed, SciPy uses
the reference-equivalent host CG recurrence, native uses a cached
C++17/OpenMP CSR CG solver, and CuPy reuses the very same device CSR allocation
as the Kohn-Sham Hamiltonian. The native solver retains canonical row
summation order but losslessly compacts the repeated finite-difference matrix
to int32 column rows and one-byte codes into a float64 coefficient palette.
Fixed-size parallel dot-product blocks are merged deterministically, and the
residual norm is reused for `beta` exactly as PARSEC's bundled SPARSKIT CG
does. Warm starts, tolerances, matrix-vector budgets, breakdown rules, and
final true-residual diagnostics are preserved.
The native iteration fuses `A @ p` with the canonical-block
`p dot (A @ p)` reduction, eliminating a second traversal while retaining the
original CSR row order and deterministic 4,096-row reduction topology.
In the default hybrid this native CG solved the Hartree problem until
2026-11-07, while the Kohn--Sham Hamiltonian and eigensolver stayed on the
GPU. Since then the same recurrence runs on the GPU of the orbitals
(`Hartree/cupy_prepared.py`: the slot-major stencil, fixed-order block
reductions, CUDA graphs of eight steps), and
`PARSEC_HARTREE_LINEAR_BACKEND=native` selects the host CG described here;
`Hartree/GPU_HARTREE.md` has the measurements behind the change.
After two completed SCF solves, native CG forms a clipped two-step
chronological prediction from the two preceding right-hand sides and Hartree
solutions. CG still solves the unchanged linear system to the unchanged
tolerance and recomputes the final true residual. Set
`PARSEC_HARTREE_CHRONOLOGICAL_GUESS=0` to use only the immediately preceding
Hartree potential.

Before constructing native CG, `auto` tests all 48 Cartesian signed-
permutation operations against the labeled atoms and every active lattice
point, then selects the largest exact commuting involution subgroup. If a
nontrivial group is proved, the totally symmetric Poisson system is projected as
`A_w = U.T @ A @ U`, `b_w = U.T @ b`, solved on the wedge, and expanded by
`U`. Native extension 0.5 (0.6.0 stores its angular arrays for the order
in use, up to 60, with unchanged arithmetic; 0.6.1 hands the prefactors of
orders up to 9 to the GPU boundary in the layout of 0.5 again, the only one a
Python tree from before 0.6.0 reads, so a build of it can sit in such a
checkout) additionally precomputes orbit-summed multipole
coefficients (and stores them where a symmetry cache is named), constructs the boundary-corrected
normalized `b_w` directly, and fuses the CG matrix-vector/dot traversal, so
repeated SCF steps never form the full-grid Hartree RHS. Orbit normalization
also handles points lying on reflection planes.
`PARSEC_HARTREE_SYMMETRY=0` disables this optimization for a controlled
Hartree-only comparison. Prefer `--symmetry off` when both Hartree and the
orbital eigensolver must use the full grid; a failed proof in `auto` always
falls back safely.

When orbital sectors are active, the density and local potentials are exact
totally symmetric scalar fields. Residual norms, Anderson history/Gram
matrices, and density-potential energy integrals therefore retain one physical
value per orbit and use multiplicity-weighted quadrature. No repeated scalar
field expansion is needed by the Hamiltonian, Hartree, XC, mixer, or energy
path. The formulas, convergence criterion, and
energy terms are unchanged, including for unequal orbit sizes.

`PARSEC_SYMMETRY_SCF_BUFFERS` (default `1`) removes allocations from that
host work of the root: the weighted sums read the orbit multiplicities as
float64, an energy forms the weighted density once for its four integrals,
and the Anderson step works on blocks of 16,384 orbits in arrays the mixer
keeps, twelve wedge fields beside a history of four. `0` allocates an array
for every integral and every term of the step, as before. With the float64
multiplicities the root so keeps thirteen wedge fields through the SCF that
the former step allocated and returned in every mix: 0.37 GiB for the
3,786,832 orbits of the 23,768-electron cluster, 0.07 GiB for 3,480
electrons. Its resident memory between steps is higher by that, and its
high-water mark by up to that wherever the SCF sets the mark outside a mix;
the peak inside a mix is the same. `scf_scalar_field_buffers` in the
backend report says `on` or `off`, as the preparation found the switch. Each
floating-point operation keeps its operands and their order, and the dense
products of the step receive arrays of the former shape and layout, so the
norms, energies and mixed potentials are the same to the last bit wherever
the BLAS products do not depend on the address of an operand (true of the
NumPy tested). With the orbit map of the 23,768-electron cluster (3,786,832
orbits) and the mixing controls of its input, ten steps of norms, energy and
mixing took 1.19 instead of 2.98 s on a workstation, the Anderson step with
a full history 0.099 instead of 0.297 s, every mixed potential, residual and
energy term equal bit for bit; so were a complete SCF of 424 electrons on
its GPU. Those seconds are a workstation's. On A100 nodes the switch ran on
and off in one allocation together with `PARSEC_FAST_GRID` and
`PARSEC_CUPY_DENSITY_COLLECTION` (3,480 electrons on 4, 8 and 16 GPUs, 10,456
and 19,392 on 4, 23,768 on 16), with the same total energies to the last bit.
The serial control of the runner keeps
`0`, and `full` for `PARSEC_CUPY_DENSITY_COLLECTION` and `serial` for
`PARSEC_CUPY_DENSITY_RELEASE` (`MULTI_GPU.md`).

### Periodic features on the device

Wires and slabs (`Boundary_Conditions: wire`/`slab`), the reciprocal-space
Hartree potential (`Calc_Hartree_G`), Fortran's G-space local potential
(`Periodic_Local_Potential: g_space`), the zero-average XC potential
(`Zero_Average_Vxc`) and collinear spin of a periodic system take these
device paths:

- A Gamma-point single point of a wire, a slab, or a 3-D cell with
  `Calc_Hartree_G`, `g_space` or `Zero_Average_Vxc` runs on the accelerated
  driver (`SCF/periodic_routes.py`): the reference periodic preparation, the
  CuPy eigensolver on the sparse operator, and the Hartree solve of the
  prepared system. k-points, spin, forces, relaxation and molecular dynamics
  of these inputs, and a restart of a wire or slab, stay with the reference
  CLI (`_reference_only_features`), whose k-point eigenproblems run on the
  device workers.
- `Hartree/periodic_device.py` solves on the device, for every driver, the
  wire and slab Hartree problem (the cylindrical moments of `hartset_wire`,
  the plane moments of `hartset_slab` and the boundary series as products of
  the readable tables, `8 pi n - A_IB V_B`, and the CG recurrence of
  `Hartree.poisson`), the spectral solve of `Calc_Hartree_G` (CuPy FFT) and,
  since 2026-10-30 by default, the finite-difference CG of an orthorhombic
  3-D cell, through the hooks of `Hartree.confined` and `Hartree.pbc`; it
  runs where the k-point plan puts eigenproblems on a CUDA device (and so
  also in the accelerated driver's Gamma-point path of a cell, which solves
  with the readable periodic CG).
- `V_xc/device_xc.py::ZeroAverageXC` removes the average of a device XC
  potential of a 3-D cell with `Zero_Average_Vxc`, as the readable
  `evaluate_xc` does (`potshift`); on the orbit fields of the symmetry
  sectors it weighs every orbit by its size.
- A bulk orthorhombic cell at the Gamma point (unpolarized or collinear
  spin, native LDA or GGA) solves its orbitals in the sectors of its group
  (about the origin of the cell, about a symmetric site off it, with the
  half-cell translations: `PARSEC_PERIODIC_SYMMETRY`, "Point groups of
  periodic cells" below), or else, unpolarized, runs on several GPUs and nodes
  as one sector of the identity whose orbitals the MPI ranks share:
  `scripts/run_multi_gpu_scf.sh --rank-per-gpu`, or one rank per node with
  `Eigensolver: chebff` (`PARSEC_MPI_NESTED_SHARING`; without it one rank per
  node on several nodes is refused, `PARSEC_SINGLE_SECTOR_NODE_LAYOUT`; `MULTI_GPU.md`, "Periodic
  cells at the Gamma point on several nodes"; `driver.periodic_sector_refusal`
  names what it does not take). In one process `PARSEC_CUPY_SINGLE_SECTOR=on`
  takes that sector, whose basis the GPUs of the process share after a CHEBFF
  first solve (`PARSEC_CUPY_SECTOR_PLACEMENT=size`); unset, a cell without a
  point group keeps the full grid of one GPU.
- `V_xc/periodic_xc.py` evaluates the spin functionals of a periodic cell in
  the reference spin drivers on the device, by `PARSEC_CUPY_XC`: the
  functionals added for Fortran parity by default; PBE of
  `evaluate_xc_spin_polarized` (the k-point and Gamma spin drivers) by default
  since 2026-11-17, as the formula of `pbe_spin_polarized` (the variables
  `s_ab = D n_a . D n_b`) with forward derivatives in place of its complex
  step (`V_xc/dual.py`; the energy density the host's bit for bit on the host,
  the partials to 4e-14 relative), the `"ab"` form of `DeviceSpinXCEvaluator`;
  PBE of `evaluate_spin_xc` only with `on` (`HOST_UNLESS_ASKED`: the kernel of
  `pbe_spin`); CA on the host.
- A Gamma-only spin run of a wire or slab takes the 1x1x1 k-point driver on
  the device (`cli.py::_gamma_spin_on_kpoint_driver`), as a 3-D cell does
  outside the route below. Andrew Grieder's slab (the default routines, a
  3-D grid of a cell with `dimensions = 2`) takes the route below since
  2026-11-19 as its unpolarized twin takes the accelerated driver: on the full
  grid of one device (its detection refuses a grid that is not periodic along
  three axes); three H atoms in such a slab, CA, gave the energy of the
  k-point driver to 1e-8 Ry (A100).

#### Spin-polarized cells at the Gamma point

Since 2026-11-19 (`PARSEC_GAMMA_SPIN_SECTORS=auto`, `SCF/gamma_spin.py`) a
collinear spin-polarized bulk cell runs on the accelerated Gamma route of the
sectors when it is orthorhombic, sampled at the Gamma point only, with a
native LDA or GGA (no hybrid or spin-orbit species; in one process
also a meta-GGA or libxc under `PARSEC_GAMMA_META_GGA_SECTORS` and
`PARSEC_GAMMA_LIBXC_SECTORS`, TS under `PARSEC_GAMMA_TS_SECTORS=on` and, since
2026-11-21, LDA+U under the opt-in `PARSEC_LDAU_SPIN_SECTORS=auto`, below), as
a single point in one process on CuPy or in the full-SCF runner. An integer
`PARSEC_PERIODIC_SPIN_SECTOR_FLOOR` (no floor by default) keeps a smaller
single point of such a cell with a native LDA or GGA on the full grid of one
device.
Both channels are solved in the abelian sectors of the cell: the commuting
signed-permutation involutions with the half-cell translations (a perfect
diamond supercell 32 operations, 28 of them with a translation; a vacancy at
the origin D2, 4 sectors; O2 along z D2h, 8). Since 2026-11-19 a cell whose
point group gives a weighted plan takes those sectors instead in one process
(`PARSEC_ORBITAL_GROUP_SPIN`; the vacancy in Td: the four sectors of D2d,
weights 1, 1, 3, 3), since 2026-11-21 on several MPI ranks too
(`PARSEC_ORBITAL_GROUP_MPI`), and a run the guard stops there starts again in
the abelian sectors. The orbitals are real; the
channel densities, potentials and mixer history keep one value per orbit;
the Hartree potential of the total density is solved by the periodic FFT on
the device and the spin functional (CA, PBE) runs on the device on the
orbits. With `PARSEC_SPIN_FLIP_SYMMETRY=auto` an antiferromagnetic cell whose
partner labels an operation `r -> R r + t` exchanges (MnO AFM-II: the
translation by `a`) solves the sectors of the up channel alone and takes the
down channel as their image. The guard of `Spin_Symmetry_Sectors` applies unchanged (a partly
filled multiplet of the start Hamiltonian; a channel density that breaks the
operations of the atoms, for a cell the operations `r -> R r + t` that map
the atoms onto themselves modulo the lattice: those of the sector group,
those found from the environments of the atoms (a defect off the origin of
any cell: Td about it), those of the cell group about its symmetric site with
the half-cell translations, and the signed permutations about the origin,
all beyond the sector group to 1e-6 bohr at most whatever
`PARSEC_SYMMETRY_ATOM_TOLERANCE`, `PARSEC_SPIN_GUARD_CELL_GROUP`; a fractional
occupation of a multiplet at the end): when it fires, parsec.out prints its
NOTE and the run starts again on the full grid in the same driver: of one
device, or, where both channels would not fit there, one sector of the
identity whose basis the GPUs of the process (the ranks of the runner)
share (`PARSEC_SPIN_FALLBACK_SHARED`; the preparation says so in a NOTE, and
a run for which no layout fits stops with an error before that full grid,
except under `off`, which tries one device as before). A fractional E doublet
whose two partners are states of one weighted sector (fcc and rock-salt cells, the
point groups T and Th)
does not stop the run: the result is kept with a NOTE, because the abelian
sectors and the full grid keep the same symmetric state from a symmetric start
(`PARSEC_SPIN_GUARD_DOUBLETS`). The same NOTE marks fractional degenerate
levels that the abelian sectors hold in one sector (the e_g of
Oh in A_g of D2h); there the full grid can split them and end lower
(bcc Mn2 with antiparallel labels at 300 K, h 0.11 bohr: 82 mRy), and
`stop` of that switch sends such a run to the full grid; the k-point
sector routes do not look for such levels.
Partly filled degenerate defect levels, such as the neutral vacancy, end
there: in diamond 5^3 (999 atoms) its full grid ran out of memory on one A100
and now converges on the four GPUs of one process in 55.9 s (8.6 GiB on the
busiest GPU). The 1x1x1 k-point driver that the routes below keep tests the
operations of its k-point sectors and those found from the environments of
the atoms (`PARSEC_KPOINT_SECTOR_GUARD`): test 2 of either driver takes Td
about a diamond vacancy at or off the origin.
An open discrepancy of the routes (a user decision, not changed): where two
labels are one species and the cell has several SCF solutions, the sectors
and the run without symmetry can end in different ones. The non-magnetic Cr2
pair of two labels (Cr_a, Cr_b, the same file, both
`Initial_Spin_Polarization: 0`) in a 14.4 bohr cube (PBE) ends at -37.4886 Ry
in the sectors and at -37.4402 Ry on the full grid (`--symmetry off`). The
group of the sectors is C4v of the labeled atoms, as PARSEC 1.5 takes the
group of its atom types: z -> -z exchanges the two labels, so neither the
sectors nor test 2 of the guard hold it, and the sector state breaks that
mirror (the two halves of the cell within 6e-4 electrons of each other). The
full grid keeps the mirror-symmetric atomic start in a symmetric solution,
48 mRy higher and metastable (it returns there after a small perturbation,
and reaches it from the atomic start also unpolarized, with
`Mixing_Param: 0.1`, `chebdav`, 36 states or a tighter `Diag_Tolerance`).
Neither default route reaches the lowest state found: the full grid started
from the sector potentials goes on to -37.48875 Ry, 1.1e-4 Ry below the
sectors, breaking C4v as well (A100, 2026-11-21). The parser says so in a
note wherever two labels are one species (the same pseudopotential bytes and
settings) and hide operations of the atoms (`Input/parsec_input.py::_identical_label_note`;
one label for those atoms keeps the whole group). A guard that tests the
operations of the merged species would send such a run to the full grid,
which starts again from the atomic start, so that both routes would end in
the higher, symmetric state (expected, not measured); it is not built.
Single-point forces and relaxations by an optimizer (`bfgs`, `simple`)
take the sectors too (`PARSEC_SPIN_SECTOR_FORCES=auto`, since 2026-11-19)
where those of an unpolarized cell do, on the default routines: the
nonlocal force terms of both channels from their sector states, the forces
and moved positions averaged over the sector group, the guard at every
structure (V- in diamond 2^3, CA: forces of the k-point driver to 2e-9
Ry/bohr, a relaxation step 3-5 s against 57-107 s).
Molecular dynamics, manual movements, fixed atoms (`Move_Flag`) and forces
with `Ignore_Symmetry` run on the full grid of both channels in the same
driver (where the sectors of `PARSEC_MOVING_SECTORS` do not take them, `PARSEC_MOVING_SPIN_SECTORS`): of one device where its estimate (4 x
both bases) fits, else one sector of the identity whose basis the GPUs of
the process share (`PARSEC_SPIN_FALLBACK_SHARED`), else on the 1x1x1 k-point
driver with a NOTE (V- 2^3: the trajectory of the k-point driver to 1e-13,
1.3 against 91 s per step, the `Move_Flag` and manual relaxations to 5e-12
bohr; V- 5^3, 999 atoms: a molecular dynamics step ran out of memory on one
A100, the shared layout of four takes 52.4 s and 8.5 GiB on the busiest).
`PARSEC_GAMMA_SPIN_SECTORS=off`, `Spin_Symmetry_Sectors: off`,
`Symmetrize_Charge_Density`, `occup.in`, `Restart_Run`, the forces of the
Fortran-derived routines (the default for the forces of a spin-polarized PBE
cell and of `Cubic_Spline` inputs), `PARSEC_SPIN_SECTOR_FORCES=off`, band
structures,
`Polarizability`, `Energy_Refinement`, `main.py --mpi` and other backends
keep the 1x1x1 k-point driver. Their forces, relaxations and dynamics run in
the ionic loop of the reference solver: on the Fortran-derived routines the
1x1x1 k-point run of the device k-point solver too since 2026-11-21
(`PARSEC_SPIN_IONIC_KPOINT`); before, and on the default routines still (under
`PARSEC_SPIN_SECTOR_FORCES=off` or `PARSEC_PERIODIC_SECTOR_FORCES=off`, or
where no full-grid layout fits), the Gamma-point loop `run_spin_scf` with the
eigensolvers of the host, which the "k-point driver" times of the forces,
relaxations and dynamics quoted here and in the switch table measured (5 s per
SCF iteration of V- 2^3 against 0.15 s on the k-point solver). The full-SCF runner refuses spin cells only
under `off` or `Spin_Symmetry_Sectors: off`; on several ranks the spin
forces and relaxations that leave the sectors, molecular dynamics, and
every spin forces run under `PARSEC_SPIN_SECTOR_FORCES=off`; and, since
2026-11-19 with the reason of a spin cell (`driver.spin_cell_group_refusal`), a cell that keeps
no abelian group of more than one operation, whose channels the shared
sector of the identity does not take: before any preparation where no group
is looked for (`PARSEC_PERIODIC_SYMMETRY=off`, no symmetry), right after the
detection otherwise (before, it was refused only at the end of the
detection with a message that named `PARSEC_CUPY_SINGLE_SECTOR`). The one
sector of the identity that a guard fallback shares among the ranks
(`PARSEC_SPIN_FALLBACK_SHARED`) is not such a cell: it is prepared after the
sectors, and neither this refusal nor that of one sector shared by ranks of
several GPUs each (`PARSEC_SINGLE_SECTOR_NODE_LAYOUT`) stops it; with one rank
per node of several GPUs each rank spreads its share over the GPUs of its
node after a CHEBFF first solve (`PARSEC_MPI_NESTED_SHARING`), and otherwise
holds it on the first GPU of its node, which the NOTE of the preparation
says, advising `--rank-per-gpu` (`driver.SpinFallbackPlan.first_gpu_shares`). Measured with the benchmark diamond inputs (`Spin_Polarization`, PBE,
500 K, 2e-4 Ry), V- in diamond 5^3 (999 atoms, S = 3/2): 16 A100, the runner
with one rank per GPU SCF 3.1 s, total 7.9 s, 1.8 GiB per GPU, against SCF
10.6 s, total 12.7 s of the k-point pools of `main.py --mpi`; one node in one
process SCF 24.6 against 40.6 s, peak 4.4 against 10.4 GB. The energies agree
with the 1x1x1 driver to about 1e-9 Ry at a 1e-9 criterion, the moments to
1e-14.

| switch | values |
| --- | --- |
| `PARSEC_PERIODIC_HARTREE` | `auto` (default) and `device`: wire, slab and `Calc_Hartree_G` solves and the finite-difference CG of an orthorhombic 3-D cell on the device (the CG of a cell by default since 2026-10-30, `device` only before; it changes periodic results by the CG tolerance); `host`: every solve on the host (for a plain 3-D cell the former default, bit for bit) |
| `PARSEC_CUPY_XC` | `auto` (default): the spin PBE of the k-point and Gamma spin drivers of a periodic cell (`evaluate_xc_spin_polarized`) on the device since 2026-11-17, the formula of `pbe_spin_polarized` with forward derivatives (the host's numbers to round-off); `on` also puts the PBE of `evaluate_spin_xc` there (the kernel of `pbe_spin`); `off`: the host for every functional (the former path of the spin drivers' PBE, bit for bit) |
| `PARSEC_PERIODIC_HARTREE_BATCH` | CG iterations per batch without a host synchronization (default 8) |
| `PARSEC_WIRE_SLAB_GAMMA_SPIN` | `gamma` keeps the Gamma-only spin run of a wire or slab on the Gamma driver |

The device CG (`FusedConjugateGradient`, `Hartree/periodic_device_cg.cu`)
runs the recurrence of `Hartree.poisson` in five kernels per iteration (the
CSR product in SciPy's row order fused with `p.Ap`, dot products reduced in a
fixed order) and in batches of 8 iterations without a host synchronization;
the host then applies the stopping test, the breakdown rule and the budget
iteration by iteration and updates the solution with exactly the accepted
iterations, so iteration counts and solutions are those of the sequential
recurrence. The results equal the host solves to round-off where the CG is
converged tightly (1e-13 Ry); at the default relative tolerance 1e-7 of
`hpotcg` the two iterate paths stop at different points within it, up to
8e-8 Ry in the total energy of a wire (polyacetylene) and 1e-8 for a slab.

The CG of an orthorhombic 3-D cell runs there by default since 2026-10-30
(`PARSEC_PERIODIC_HARTREE=host` gives the host CG bit for bit). Laptop
(RTX 5070), against the host CG: the Si8 cube at h 0.35 (24,389 points) with
the criterion at 1e-10 moves by -8e-12 Ry (accelerated Gamma path) and
-1.2e-11 Ry (reference CLI), levels by at most 1.1e-10 Ry; at the input's own
criterion (1e-7) the SCF stops at another step (38 against 25 steps, 1.2e-9
Ry). The Hartree subtotal of its SCF fell from 2.12 to 0.19 s (0.056 to 0.008
s per step) and the SCF from 3.5 to 1.2 s; `examples/3d_Si` (12,167 points)
0.28 to 0.06 s of Hartree; the 8 k-points of the Si8 cube on the device
workers 2.07 to 0.29 s of Hartree in 57 s of SCF. Wires and slabs are
unchanged (their solves were on the device before).

The preparation of an orthorhombic cell (`V_ion/periodic_setup.py`, installed
with the package, both CLIs) sums the radial functions of the atom images out
to where their terms stop being nonzero or above round-off
(`PARSEC_PERIODIC_SETUP_REACH=physical`, the default since 2026-10-30);
`table`, the default before, sums to the end of the radial tables (about 118
bohr), bit for bit the readable definitions. Laptop: `examples/3d_Si` prepares
in 0.20 instead of 5.2 s, the Si8 cube at h 0.35 in 0.30 instead of 10.3 s;
total energies move by 1e-12 to 1.2e-11 Ry. The general (oblique) grid takes
the split below; wires and slabs take the sums of `V_ion/confined_setup.py`
(below), which give the readable numbers bit for bit.

The short-ranged terms of the local potential stop at the radius where the
table of `r V` is the Coulomb tail `-2 Z`: to round-off for a POTRE table, to
the digits of the file for the FHIPP, MARTINS_OLD and MARTINS_WANG tables
(`V_ion.ionic_potential.coulomb_tail_radius`, since 2026-11-07; before, those
tables never settled to round-off and the sums ran to the end of the table,
75 bohr). On the orthorhombic grid of the Fortran-derived routines, which every
`fhipp` input takes, that was both slow and, with `Cubic_Spline`, wrong: the
error of the spline of `V` near the end of the table entered every image
(V_ion 0.05 Ry low, an 8-atom silicon cube 1.62 Ry low in total energy).
Setup on one A100 node (PBE silicon `fhipp` with `Cubic_Spline`, first runs),
local potential / whole preparation, before -> after: Si8 h 0.25 (64,000
points) 132.3 / 132.8 s -> 0.09 / 0.52 s; Si64 h 0.30 (314,432 points)
645.3 / 647.4 s -> 0.62 / 2.79 s; Si216 h 0.30 (1,061,208 points) 2.16 / 10.0
s now. Without `Cubic_Spline` the potential moves by 9e-10 Ry (Si64: 8.0 ->
0.11 s). POTRE tables keep their radius, bit for bit.
The local forces of `Cubic_Spline` species on that grid
(`Forces/periodic_spline.py`, installed with the preparation) are the
derivatives of this potential: the short-ranged terms over the same balls on
the host, the long-ranged one from one FFT of the density, and the
self-consistency and core-correction terms in the CUDA kernel of
`Forces/periodic_device.py` (host balls without a device). Before, the
readable routine summed every grid point of every image within
`erfcinv(1e-26)/eta`, 43 to 86 bohr; it differs from these forces by the
spline error between the two reaches (2.6e-7 Ry/bohr in
`tests/test_periodic_coulomb_tail.py`). Both give the grid points within
`2 rs(2)` of an atom (twice its first radial sample) the slopes of the smooth
behaviour of the table (`PARSEC_SPLINE_FORCE_CENTER=auto`, since 2026-11-22; under `tip`,
the default of 2026-11-21, they leave the points within round-off of an atom,
`min(rs(2), 1e-6 bohr)`, out of the spline slopes: an atom on a grid point
had taken the slope of the spline at its tip; `fortran` leaves out every point
within the first radial sample, as `forcloc.f90` does). Local forces on one A100 node, readable
-> now: Si8 h 0.25 (displaced atom) 34.1 -> 3.5 s; Si64 h 0.30 891.9 -> 5.2 s.
The general (oblique) grid of the Fortran-derived routines takes the same
split since 2026-11-07 where a species uses `Cubic_Spline`, and since
2026-11-22 for the linear POTRE and `fhipp` tables too
(`V_ion/periodic_setup_fortran.py`, `_oblique_local_ionic_potential`;
`PARSEC_PERIODIC_OBLIQUE_SPLIT`: `validated`, the default since 2026-11-22;
`auto`, the default before, the split for `Cubic_Spline` species only; `fast`
for every species; `readable` for the readable direct sum, the route before
2026-11-07). The
readable sum reaches the erfc cutoff of `eta = sqrt(pi)/V^(1/3)` plus the
diameter of the cell, which runs into the spline error near the end of the
table as the cell grows: an fcc silicon cell of 54 atoms at h 0.30 had its
local potential 0.049 Ry low (about -10 Ry in the total energy), 16 atoms
1.5e-5 Ry. Primitive cells move by less than 1e-4 Ry: A100, PBE `fhipp`,
E(split) - E(readable) = -2.7e-5 Ry (Si, h 0.25, Gamma and 2x2x2 and 3x3x3
meshes), -4.2e-5 (C, h 0.14), -3.6e-5 (MgO, h 0.14). Against Quantum
ESPRESSO with the same tables (the converged cutoff and the KB-energy shift of
`refs/pbc`), five of the six cases come closer: Si +1.9e-5, +2.6e-5, +4.5e-6
-> -8.7e-6, -1.5e-6, -2.3e-5; C +2.0e-4, +1.3e-4 -> +1.6e-4, +8.9e-5; MgO
+5.3e-4 -> +5.0e-4. The POTRE files of the same potentials move by 2e-6 (their
denser radial grid). Setup of the local potential: 2.6-7 s -> 0.04-0.15 s for
these cells, 89.9 -> 0.28 s for Si16 and 817 -> 1.0 s for Si54. Their local
forces are the derivatives of this potential (`Forces/periodic_spline.py`
over the lattice balls of `V_ion/periodic_setup.py`). The atomic densities
of the general grid and the density terms of these forces (30-bohr balls for
silicon: 13 s for Si16 and 44 s for Si54 on an A100 node) run in two lattice
kernels on the device where one is usable
(`V_ion/periodic_device.py::device_lattice_atomic_density`,
`Forces/periodic_device.py::device_lattice_density_forces`), equal to the host
sums to round-off (3e-15 relative); laptop RTX 5070, fcc Si16 h 0.30:
density 48 -> 0.11 s, local forces 95 -> 1.7 s. A `Cubic_Spline` core
density keeps the host sum.
Cells whose species all interpolate linearly kept the readable sum up to
2026-11-22. It evaluates every image within its Ewald reach at every grid point,
which was nearly the whole preparation of a general grid: on an A100 node,
first runs, hcp Mg 64 atoms (`fhipp`, h 0.35) 256.9 s, fcc Si 128 atoms (POTRE,
h 0.5) 496.1 s, tetragonal Si 216 atoms (`fhipp`, unequal steps) 810.0 s; with
the split 2.0, 2.6 and 2.2 s, the SCF unchanged (25, 21 and 25 steps). For a
POTRE table the two potentials agree to round-off (largest difference 1e-13 Ry;
total energies within 1e-12 Ry at an SCF criterion of 1e-11). For a `fhipp`
table they differ by a constant: the digits of `r V + 2 Z` past the Coulomb tail
radius (`V_ion.ionic_potential.coulomb_tail_radius`), which the readable sum
adds out to a reach that grows with the cell and the split, like the orthorhombic
grid of these routines since 2026-11-07, leaves out. Si `fhipp` 2.6e-10 Ry (8
atoms), 8.4e-10 (64 and 216 atoms); Mg `fhipp` 1.2e-11 (2 atoms) to 5.3e-11 (36
atoms); the part that is not constant is below 1e-11 Ry. The total energy moves
by that constant times the electrons (tight SCF: Si8 +8.3e-9 Ry, Mg16 +1.1e-9;
Si216 +7.3e-7 Ry, 3.4e-9 Ry per atom), and forces, densities and level
differences do not move: the forces of five general cells against the slope of
the SCF energy (steps 1e-3 and 2e-3 bohr) are those of the readable route to
all printed digits. With `r V` set to `-2 Z` exactly past the Coulomb tail
radius the split and the readable sum agree to 5e-14 Ry for every format
measured (`fhipp`, MARTINS_OLD, MARTINS_WANG): the difference is that noise,
summed out to the readable reach, and its size depends on the format. For the
MARTINS_OLD and MARTINS_WANG tables of the tests it is not a constant, so the
default (`validated`) keeps their readable sum, as before, and only `fast`
takes the split for them: `Na_POTRW` (bcc, h 0.6) varies by 2.3e-7 to 6.3e-7 Ry
over the cell, its mean 7e-8 Ry (1 atom) to -1.25e-6 Ry (64 atoms, -8.0e-5 Ry
of total energy); a tight SCF of 27 atoms moves by -1.39e-5 Ry, the forces by
at most 1.1e-8 Ry/bohr; `Si_POTR` (fcc, h 0.5) varies by 1.6e-8 to 2.5e-8 Ry,
its total energy moves by +4.0e-7 Ry for 16 atoms and -1.2e-6 Ry for 54, the
forces by 1e-9. `PARSEC_PERIODIC_OBLIQUE_SPLIT=auto` restores the readable sum
of every linear species bit for bit.

The preparation of a wire or slab (the grid of `Grid/confined.py`) summed
every image of every atom over the whole grid: the superposed valence and core
densities out to the end of the radial table (75 to 78 bohr for the `fhipp`
tables, so the ball of nearly every image holds every point), the projectors
from a scan of the whole grid for each image of a ball of a few bohr, and the
short-ranged local potential (`ionpot_wire`/`ionpot_slab`) over every image of
the Gaussian compensation. `V_ion/confined_setup.py` (`PARSEC_CONFINED_SETUP=auto`,
the default since 2026-11-22) gives the same numbers bit for bit: the balls
from index boxes along the lattice directions (rows in the order of the scan;
a box of half the grid or more is the scan itself, as for the 75-bohr density
balls of a run without a device, where slicing and sorting the box made a
small slab 9 to 29 % slower to prepare than the readable route; with the scan
a run without a device prepares 9 to 12 % faster than it, a 18-atom hexagonal
graphene 33.1 -> 19.9 s), the density sums on the device with one thread per point adding the terms of
the images in the order of the readable loop with the host's operations (the
offset from the centre the host forms, `(dx^2 + dy^2) + dz^2`, the correctly
rounded root, `np.interp`'s value, no fused multiply-adds), and the image terms
of the local potential on the threads of the periodic setup
(`PARSEC_PERIODIC_SETUP_THREADS`), added by the calling thread in the readable
order. The workers are as many as fit 1 GiB of host memory with their own
temporaries (13 grid arrays each, an upper bound) and the two finished terms
held ahead, and never fewer than one, the memory of the readable loop: eight up
to 1.2 M grid points, five at 1.8 M: 531 MiB at the peak there against 181 MiB
for the readable loop (2077 MiB with eight workers before the bound), 4.5
against 20.8 s. `readable` restores the scans and the loop (a `Cubic_Spline` core
density and a run without a device keep the readable sum anyway). Without a
switch, since the numbers are the same: the Bessel tables of the wire Hartree
boundary are evaluated once per distinct distance from the axis (every line of
points along the axis shares one), the row lookup of the stencil builders works
on columns, and the gradient matrices are built when first read (a Gamma single
point never reads them). A100 node, first runs (no symmetry cache), preparation
before -> after with the same SCF energy and steps to all printed digits and the
same GPU peak: graphene 32 atoms (295k points) 87.2 -> 4.9 s, 72 atoms (664k)
283.0 -> 13.1 s; hexagonal graphene 72 atoms 385.2 -> 16.4 s; h-BN 72 atoms
(hexagonal, `fhipp` B with its core correction) 535.8 -> 16.9 s; armchair
ribbons (wires, C and H) 26 atoms 44.5 -> 4.9 s, 38 atoms 90.2 -> 9.4 s; equal
hashes of the ionic potential, both densities and the projectors in eight
smaller slabs and wires (rectangular and oblique, POTRE and `fhipp`).

A core density with `Cubic_Spline` takes neither the device sums nor the
index-box blocks above: its spline's `x**3` is the library's `pow`, which a
device does not reproduce. The readable loop
(`V_ion.ionic_potential.superpose_atomic_density`) also rebuilt the spline of
the whole table (a Python recurrence over its 400 to 1000 samples) for every
image of every atom, and a wire or slab sums its images out to the end of the
table (118 bohr for a POTRE file). `V_ion/spline_setup.py`
(`PARSEC_SPLINE_SETUP=auto`, the default since 2026-11-22) runs the same loop
with the spline of each species built once (the same knots and second
derivatives), the images of one atom in blocks on the threads of the periodic
setup (the grid points of each image from the installed support finder; a wire
or slab block from one dense scan of its images, the rows and distances of
`dense_support`), one call of the readable interpolation for the values of a
block (elementwise functions of the distances) and the additions in the order
of the readable loop: the density bit for bit. The images and supports are those
of the readable loop (`V_ion.ionic_potential.atomic_density_images`, which both
use). A100 node, first runs, core-density setup before -> after with equal
hashes of the ionic potential, both densities and the projectors and the same
SCF energy and steps: Ge8 slab (6,656 points) 22.3 -> 1.0 s (preparation 24.2
-> 2.8 s), Ge2 slab 10.1 -> 0.34 s (12.0 -> 2.0 s), Ge64 cell 2.27 -> 0.88 s on
the default routines and 2.27 -> 0.67 s on the Fortran-derived ones, Ge8 cell
(Fortran-derived) 0.65 -> 0.09 s, rocksalt MgO8 (POTRE Mg with its core
correction) 0.34 -> 0.09 s. A Si `fhipp` cell without a core correction and a
GeH4 cluster (the cluster builders of `V_ion/native_ionic.py`) do not take it
and are unchanged.

SCF seconds on one A100 (runs that shared the CPUs of a node; code
defaults, warm CuPy cache), before -> after:

| input | SCF before | SCF now |
| --- | --- | --- |
| graphene slab (2 C), Gamma, 20,736 points | 7.31 (host eigensolver and Hartree) | 0.93 |
| H2 slab, Gamma, 36,864 points | 13.61 | 0.95 |
| polyacetylene wire, Gamma, 23,744 points | 7.82 | 0.82 |
| polyacetylene wire, 4 k-points | 7.35 (Hartree 3.49) | 3.64 (Hartree 0.17) |
| Si2 wire, spin, Gamma | 4.94 (Gamma spin driver) | 1.81 (1x1x1 k-point driver) |
| Si8 cube, Gamma, `Calc_Hartree_G` | 14.19 | 1.55 |
| Si8 cube, Gamma, plain, `PARSEC_PERIODIC_HARTREE=device` (the default since 2026-10-30) | 3.03 (host CG) | 1.52 |

A first run with an empty CuPy kernel cache compiles the kernels of the
accelerated eigensolver, the device workers and these solves (20-45 s on a
shared A100 node); the device Hartree's own share is 6 kernels (one module for
the CG).

#### TS, LDA+U, meta-GGA and libxc in the Gamma sectors

Since 2026-11-19 (`SCF/gamma_features.py`) four inputs that the route of the
Gamma sectors refused can take it in one process, each under its own switch
(`auto`: from 2^27 grid points x states on for TS and a meta-GGA, from 2^28
for libxc, from 2^26 for LDA+U since 2026-11-21, with the conditions below;
`on`: every size; `off`: the full grid as before). The defaults are `auto` for
a meta-GGA, libxc and (since 2026-11-21) LDA+U and TS:

- `PARSEC_GAMMA_TS_SECTORS` (default `auto` since 2026-11-21, `off` before):
  the Tkatchenko-Scheffler
  correction. It is formed after the loop from the density on the full grid
  and adds no potential, so the loop is that of the cell without it (the
  weighted sectors of PBE). `auto` takes an unpolarized cell only: a spin cell
  keeps the 1x1x1 k-point driver, which was faster than its spin sectors;
  `on` takes spin cells too. Every TS input carries `Ignore_Symmetry` (PARSEC
  1.5 error 159: its Hirshfeld sums read the density it would otherwise keep
  on the irreducible wedge); under `auto` or `on` the command line reads that
  keyword of a TS cell as the requirement where the sectors of a CuPy run
  with the point group of the cell take it, and prints a NOTE; `--symmetry
  off`, `PARSEC_PERIODIC_SYMMETRY=off` or a backend other than CuPy keep the
  full grid. In PARSEC 1.5 the keyword switches symmetry off, but every TS
  input must carry it (error 159) and the correction reads the full-grid
  density here whatever the route, so the energies are those of the full
  grid: since 2026-11-21 this reading is the default (`off` restores the
  former meaning of the keyword).
- `PARSEC_GAMMA_LDAU_SECTORS` (default `auto` since 2026-11-21, `off` from
  2026-11-19): LDA+U (`J = 0`) of an unpolarized cell. The fifteen columns of
  every LDA+U atom are reduced to every sector with the KB factor and the
  occupation matrices are summed over the sector bases. Since 2026-11-21 the
  sectors carry the columns of one atom per orbit
  (`PARSEC_LDAU_PROJECTOR_ORBITS`) and LDA+U takes the weighted sectors of a
  subgroup of the point group (`PARSEC_LDAU_WEIGHTED_SECTORS`, the occupation
  matrices averaged over the full group); before, every sector carried the
  columns of every atom in the abelian sectors with the translations, which
  four devices shared and one carried alone (`auto` then needed four CUDA
  devices; `gamma_features.MINIMUM_DEVICES` is empty now). Its forces and
  relaxations keep the sectors (`PARSEC_LDAU_SECTOR_FORCES`); a spin cell
  takes the spin sectors only under the opt-in `PARSEC_LDAU_SPIN_SECTORS=auto`
  (the default `off` keeps the 1x1x1 k-point driver).
- `PARSEC_GAMMA_META_GGA_SECTORS` (default `auto`): native SCAN or a libxc
  meta-GGA (which also needs the libxc switch), unpolarized or collinear
  spin: `tau` and the term `-div(vtau grad)` as derivatives between the
  sectors (`Hamiltonian/meta_gga_device.py`), the PBE start of the loop on the
  orbit fields. Abelian sectors with the translations.
- `PARSEC_GAMMA_LIBXC_SECTORS` (default `auto`): a libxc LDA or GGA,
  unpolarized or collinear spin, on the device evaluators of the orbit fields.
  Abelian sectors with the translations.

One node, one process, first runs, SCF (total) seconds, full grid (a spin
cell: the 1x1x1 k-point driver, total = wall time of the process) ->
sectors, the same energies to 1e-8 Ry. GPU peak from nvidia-smi: the largest
device, with the sum over the four devices in brackets where the run spreads
over them (the full grid uses one device). `auto` is the route under that
value (TS and LDA+U are `auto` by default since 2026-11-21; the LDA+U rows of 2026-11-19 carried the columns of every atom in
abelian sectors, those of 2026-11-21 below the table one atom per orbit):

| cell | grid x states (entries) | input | A100 | `auto` | SCF [s] | total [s] | GPU peak [GiB] |
|---|---|---|---|---|---|---|---|
| Si64 | 48^3 x 153 (1.7e7) | libxc PBE | 4 | full grid | 2.2 -> 7.9 | 4.4 -> 15.9 | 1.5 -> 0.6 (2.1) |
| Si64 | 48^3 x 153 (1.7e7) | SCAN | 4 | full grid | 14.2 -> 17.2 | 16.6 -> 25.8 | 2.1 -> 0.5 (2.1) |
| Si64 | 48^3 x 153 (1.7e7) | PBE + TS | 4 | full grid | 5.2 -> 5.1 | 7.1 -> 11.4 | |
| Ag108 | 48^3 x 778 (8.6e7) | LDA+U | 4 | full grid | 13.4 -> 17.9 | 23.8 -> 33.7 | |
| Si216 | 64^3 x 502 (1.3e8) | SCAN | 1 | full grid | 173.6 -> 23.2 | 176.1 -> 29.2 | 11.1 -> 1.1 |
| Si216 | 70^3 x 502 (1.7e8) | SCAN | 1 | sectors | 250.7 -> 16.7 | 253.7 -> 21.7 | 14.3 -> 1.3 |
| Si216 | 72^3 x 502 (1.9e8) | SCAN | 4 | sectors | 254.3 -> 19.3 | 257.7 -> 28.8 | |
| Si216 | 72^3 x 502 (1.9e8) | spin SCAN | 4 | sectors | 516.4 -> 53.5 | 521.1 -> 65.4 | 19.2 (38.1) -> 1.3 (4.5) |
| Si216 | 72^3 x 502 (1.9e8) | libxc PBE | 4 | full grid | 16.2 -> 12.8 | 19.0 -> 20.9 (17.8 -> 20.6 again) | 8.2 -> 0.9 (3.0) |
| Si216 | 72^3 x 502 (1.9e8) | PBE + TS | 4 | sectors | 30.9 -> 18.9 | 34.0 -> 25.5 | |
| Si216 | 72^3 x 502 (1.9e8) | PBE + TS | 1 | sectors | 31.8 -> 18.0 | 33.9 -> 23.8 | 8.1 -> 1.0 |
| Si216, again (2026-11-21, the default `auto`) | 72^3 x 502 (1.9e8) | PBE + TS | 4 | sectors | 38.2 -> 26.3 | 42.0 -> 30.0 | 8.1 -> 0.9 (2.5) |
| Ag108 fcc (2026-11-21) | 64^3 x 778 (2.0e8) | PBE + TS | 4 | sectors | 61.1 -> 45.4 | 64.2 -> 49.3 | 8.6 -> 0.9 (2.3) |
| Ag108 fcc (2026-11-21) | 64^3 x 778 (2.0e8) | PBE + TS | 1 | sectors | 61.2 -> 41.7 | 63.8 -> 45.0 | 8.6 -> 0.9 |
| Si216 | 72^3 x 502 (1.9e8) | spin PBE + TS | 4 | k-point driver | 27.3 -> 39.8 (27.1 -> 40.2 again) | 46.9 -> 50.5 (46.1 -> 51.3) | 1.5 (4.3) -> 1.2 (3.3) |
| Si216 | 80^3 x 502 (2.6e8) | libxc PBE | 1 | full grid | 23.1 -> 11.2 | 25.3 -> 15.7 | 11.0 -> 1.2 |
| Ag108 | 70^3 x 778 (2.7e8) | LDA+U | 4 | full grid | 41.6 -> 23.5 | 67.8 -> 55.2 | 11.2 -> 1.3 (5.2) |
| Si216 | 84^3 x 502 (3.0e8) | libxc PBE | 4 | sectors | 26.1 -> 14.1 | 29.1 -> 22.2 | 12.7 -> 1.1 (3.5) |
| Si216 | 84^3 x 502 (3.0e8) | libxc PBE | 1 | sectors | 26.3 -> 11.8 | 29.0 -> 18.7 | 12.7 -> 1.3 |
| Si216 | 84^3 x 502 (3.0e8) | spin libxc PBE | 4 | sectors | 48.8 -> 27.6 | 54.5 -> 39.9 | 1.1 (4.3) -> 1.2 (3.7) |
| Ag256 | 64^3 x 1836 (4.8e8) | LDA+U | 4 | sectors | 70.3 -> 46.9 | 115.6 -> 103.3 | 19.9 -> 1.4 (5.2) |
| Ag256 | 64^3 x 1836 (4.8e8) | LDA+U | 1 | full grid | 69.9 -> 115.5 | 114.4 -> 170.9 | 19.9 -> 3.4 |
| Si512 | 96^3 x 1183 (1.0e9) | libxc PBE | 4 | sectors | 96.1 -> 30.6 | 99.9 -> 41.2 | 41.8 -> 1.3 (4.4) |

The grid of a cell takes the nearest size of small prime factors: the
spacing of 72 or 66 points along the Ag108 or Si216 cell gives 70^3 or 64^3.
TS and the meta-GGAs, which save more per iteration (the weighted sectors of
TS; the host functional of SCAN on the orbits), take the sectors from 2^27
entries on; LDA+U and libxc, whose full grid has cheap device terms, from
2^28 (`gamma_features.MINIMUM_ENTRIES`). Every measured cell that `auto`
takes is faster in the sectors, on four A100 and on one. The floors are
conservative: below them the small cells (Si64, Ag108 at 48^3) and libxc
Si216 at 72^3 on four A100 were slower in total time, but SCAN at 64^3 and
libxc at 80^3 on one A100 and LDA+U Ag108 at 70^3 on four were faster.

LDA+U since 2026-11-21 (the defaults: weighted sectors, one atom per orbit),
A100, first runs, SCF (total) seconds, full grid -> sectors on one, two and
four GPUs, the energies to 1e-8 Ry and the occupation matrices of the full
grid within 1.5e-7:

| cell | grid x states (entries) | 1 A100 | 2 A100 | 4 A100 | GPU peak [GiB] |
|---|---|---|---|---|---|
| fcc Ag32 | 46^3 x 234 (2.3e7) | 3.4 (5.8) -> 3.5 (11.6) | 4.0 (12.2) | 4.3 (13.6) | |
| SrTiO3 40 | 54^3 x 118 (1.9e7) | 4.3 (6.7) -> 5.1 (13.3) | 5.8 (14.4) | 5.9 (15.3) | |
| SrTiO3 135 | 60^3 x 380 (8.2e7) | 8.8 (11.6) -> 4.5 (7.0) | | 5.3 (9.0) | 4.0 -> 0.6 |
| fcc Ag108 | 48^3 x 778 (8.6e7) | 11.6 (14.1) -> 6.3 (9.4) | | 5.6 (10.1) | 4.0 -> 0.5 |
| fcc Ag256 | 64^3 x 1836 (4.8e8) | 61.9 (65.2) -> 9.9 (18.5) | | 11.6 (21.5) | 19.9 -> 0.6 |
| SrTiO3 320 | 84^3 x 891 (5.3e8) | 48.2 (52.3) -> 8.7 (18.1) | | 9.9 (20.7) | 21.3 -> 0.7 |

The floor of LDA+U, 2^26 entries, lies between the cells below it, whose
sector setup (about 6 s on the first run) outweighs the loop, and those above
it, which are faster on every count of devices measured. The 2-GPU column of
the larger cells and the cells between the two sizes are in the switch
table (`PARSEC_GAMMA_LDAU_SECTORS`).
One atom moved (an order-4 group), four A100, total seconds: libxc Si216 at
84^3 29.0 -> 15.7, PBE + TS Si216 34.1 -> 24.8, LDA+U Ag256 113.2 -> 72.2.
Forces of a meta-GGA and of libxc in the sectors are those of the full grid
(Si8 with a moved atom, to the printed 1e-6 Ry/bohr; two bfgs steps too);
TS and LDA+U inputs run on the Fortran-derived routines; the forces and
relaxations of TS keep the full grid (`PARSEC_PERIODIC_SECTOR_FORCES`), those
of LDA+U the sectors since 2026-11-21 (`PARSEC_LDAU_SECTOR_FORCES`). A spin cell
with a meta-GGA or libxc (and TS under `on`, LDA+U under the opt-in
`PARSEC_LDAU_SPIN_SECTORS=auto`) goes from the 1x1x1 k-point driver to the
spin sectors (O2 and Si8: the energies of both full grids; MnO AFM-II and
AFM-I with LDA+U: the energy of the full grid of the driver). A hybrid with
any of them, the MPI full-SCF runner and the supercell of a k-point mesh
(`PARSEC_KPOINT_SUPERCELL`) keep their routes.

### Hybrid functionals of periodic cells on the device

The Fock loop of a periodic cell (`../README.md`, the periodic part of
"Self-consistent hybrid functionals"; host definitions in
`V_xc/bloch_exchange.py` and `SCF/hybrid_periodic.py`) takes two device
routes, both with the exchange kernel and FFT box of the host loop, so that
their numbers are the host's to round-off:

- **Gamma point, without spin** (bulk or slab, real orbitals): the
  accelerated driver runs the device Fock loop of isolated systems
  (`SCF/hybrid.py`, `Hamiltonian/exchange_device.py`: tiles of orbital blocks
  on every GPU of the process, each pair of two occupied states solved once,
  `-a_x xi xi^T` in the CuPy operator) with
  `Hamiltonian/bloch_exchange_device.py::DevicePeriodicPairSolvers` in place
  of the Kronecker solver: a batch of real pair densities goes through a
  real-to-complex cuFFT of the FFT box (a slab's box zero-padded along z and
  the potentials cropped), is multiplied by the half spectrum of `K(G)` and
  transformed back.
  Since 2026-11-19 a bulk orthorhombic cell whose basis (grid points x
  `States_Num`) reaches 2^24 entries is solved in the real one-dimensional
  sectors of its abelian group with the half-cell translations (the Gamma
  sector route, `PARSEC_GAMMA_HYBRID_SECTORS=auto`, `SCF/gamma_hybrid_route.py`;
  a hybrid keeps no weighted sectors). The exchange gathers the sector
  orbitals once into the coset form of the half-cell translations their
  sectors carry (`PARSEC_CUPY_EXCHANGE_TRANSLATIONS=auto`,
  `Hamiltonian/exchange_translations.py`): the pair density of two sectors
  has one character on those translations, so its Fourier components fill a
  sublattice of `1/|T|` of the G vectors, and the cuFFT pair solver solves
  each such class on the reduced box of the class (complex transforms of two
  real densities at once, the kernel on the sublattice), exactly. A cell
  without such translations unfolds the orbitals into one full-grid block
  (`PARSEC_CUPY_EXCHANGE_POOL=auto`) whose pair densities are solved on the
  full grid; a build whose coset blocks do not fit the free memory of its
  device takes that block where it fits, else the tiles of the sectors (the
  block is `|T|` times larger, so usually the tiles; the pair solver in
  `parsec.out` names such builds). `W` is folded back to every sector, whose ACE block
  `-a_x xi_G xi_G^T` joins its Hamiltonian. The energies are those of the
  full grid to the SCF convergence. One node of 4 A100, PBE0 SCF: Si64
  31.2 -> 26.7 s (GPU 8.3 -> 3.7 GiB), diamond 216 atoms 45.4 -> 27.3 s,
  512 atoms 337.4 -> 111.4 s (GPU 37.1 -> 19.3 GiB, host 8.3 -> 1.8 GiB;
  the exchange is then 69.7 s of it; GPU 13.6 GiB since the build of every
  column frees the pooled block of the occupied build it completes, and the
  block is taken only where it fits the free memory of its device, else the
  tiles of the sectors); in the translation classes 49.0 s, the exchange
  7.6 s, GPU 0 2.4 GiB. Smaller cells, `off`, slabs, oblique
  cells and libxc or meta-GGA hybrids keep the full grid of one process; since
  2026-11-21 the forces and the `bfgs`/`simple` relaxation of a smaller cell
  on the default routines are averaged there over the operations of the cell,
  as the Fortran-derived routines average theirs on the full grid
  (`PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto`; `sectors` takes the sectors
  instead, slower on such cells). The update of `W` in the translation classes runs one thread per
  coset row and column where a pair density has at most two classes
  (`PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE=auto`, the same bits; a build of 512
  atoms 0.93 -> 0.85 s on 4 A100). The forces and relaxations of the cells
  above the floor keep the sectors (diamond 512 atoms, two BFGS moves: 100.7 s
  against 1099.1 s on the full grid, forces within 6.7e-6, 1.03e-5 and 3.7e-6
  Ry/bohr at the three structures at a 1e-6 Ry SCF criterion). The MPI runner
  refuses periodic hybrids. `PARSEC_GAMMA_HYBRID_SECTORS=kpoint`
  hands a large cell to the reference solver as the 1x1x1 k-point mesh in
  the k-point sectors of its little group (complex Bloch factors, the same
  energies); it measured 2-2.5x slower than the full grid (then the level passes
  of the Fock loop left the k-point sectors, which they keep since
  `PARSEC_KPOINT_SECTOR_LEVELS=auto`, and the sector tasks are small) and is
  kept for comparison.
- **k-points, and spin** (also spin at the Gamma point, the 1x1x1 Bloch loop
  of the reference solver): the reference CLI runs the Fock loop with the
  k-point drivers on the device k-point pool (`SCF/kpoint_parallel.py`, one
  worker per GPU, each k-point's Ritz subspace resident on its device), and
  `SCF/hybrid_periodic.py::DeviceBlochExchange` (installed into
  `parsec_python.SCF.hybrid_periodic` with the package) is the engine of the
  loop where that pool runs on CUDA devices. The orbitals stay where the
  workers keep them (`PooledKPointSolver.device_vectors`; the Ritz states
  stay on their devices between the loops). `W_t = V_x U_t`
  (`apply_device_bloch_exchange`): every device takes one copy of the occupied
  columns of every k-point of the mesh, tiles of up to 32 target columns of a
  k-point go to the GPUs (most expensive first, the GPU of the k-point on a
  tie), and for every source `k'` of the mesh (`-k` as `conj(u_k)`) a CUDA
  kernel writes a chunk of pair densities `conj(u_mk') u_nt / dV` straight
  into the FFT box (1 GiB of them, at most 5 % of the device's memory and at
  least `Exx_Batch_Size`: fewer, larger transforms leave less Python per
  transform to the threads of the devices, which run it in turn), cuFFT
  transforms them in place, they are multiplied by `K(k_t - k' + G)` (built
  on the device per `q` in one launch, `Hartree/exchange_kernel_device.py`,
  kept per device up to 5 % of its memory) and transformed back, and a second
  kernel adds `-(f_m/N) u_mk' w` to `W`. Each device works on a thread of its
  own kept for the run (cuFFT plans are cached per thread).
  The pairs of two occupied states are solved once: the potential of the
  partner pair is `conj(w)` for a source of the mesh point itself and the
  same `w` for the partner `-k` of a representative (when the target has a
  partner too), and a third kernel adds it to the other state's `W`
  (`PARSEC_CUPY_BLOCH_PAIR_SYMMETRY=off`: every pair one way, the order of
  the host loop). On the irreducible wedge of a single point
  (`PARSEC_HYBRID_KPOINT_WEDGE`, since 2026-11-19) the sources of the
  points outside the wedge are the images of its representatives: for each
  tile the occupied rows of the representative are gathered into the grid
  order of the operation (`P_g u`, one `take` of the rows on the device, the
  index kept per device and operation), conjugated by the kernel where time
  reversal is taken too; a pair of an occupied target with an image `(r, g)` and the
  pair of the image's mirror `(t, g^-1)` (a source of the list, `MeshStars.mirrors`) come from
  one transform, the mirror's potential `P_(g^-1) conj(w)` (`P_(g^-1) w` for a conjugated
  image), added to `W` of `r` at the turned points by a fourth kernel, where a test of the
  pair potentials shows that they keep the operation (`PARSEC_HYBRID_IMAGE_PAIRS`, since
  2026-11-21; the other image pairs are solved one way). A build of fewer than 1e9 pair-density points runs on the
  GPU that holds the most k-points (several GPUs only add launch and thread
  overhead there). ACE per k-point on the GPU of its orbitals
  (`compress_device_bloch_exchange`: `M = -U^H W` by one ZGEMM, cuSOLVER
  Cholesky, ZTRSM into `xi`), and `KPointExchangeHamiltonian` becomes
  `-a_x xi (xi^H X)` in the device Bloch operator (`Hamiltonian/device_bloch.py`),
  so the CHEBFF and SUBSPACE filters, the Lanczos bounds, Rayleigh-Ritz and
  the residuals of the device eigensolver hold it. `E_x`, `E_x(P, P0)` and the
  overlaps of the outer DIIS are formed where the orbitals are; the result of
  the loop holds host copies of the orbitals (forces, output). A build whose
  copies, tile work and `W` do not fit a device is refused with the numbers.

The device k.grad of a Bloch Hamiltonian is cached by its gradient operators
and k-point (since 2026-11-07): the k-point drivers build a new host
Hamiltonian for every task and SCF step, and the former cache by that object
kept one host and one device copy of k.grad per task and step (17.6 GB of
host memory for Si at h 0.20 with 36 k-points on master).

Measured on A100 nodes (2026-11-07, first runs of `main.py`, PBE0 with
`Exx_Tolerance` 1e-8 Ry, the inputs of the periodic references of the round:
Si fcc a = 10.26 bohr at h 0.25 (28^3 points, 8 states), the cubic Si8 cell
(41^3, 20 states), h-BN at h 0.18 (26x26x168, the FFT box padded to 336
planes, 8 states) and a 64-atom Si cell at h 0.30 (68^3, 140 states); seconds
of the program, of its SCF, of the k-point or Gamma eigensolver and of the
exchange builds (pair solves on the GPUs, ACE), peak device memory by
`benchmarks/periodic_hybrid_run.py`):

| input | k-points solved | GPUs | program s | SCF s | eigensolver s | exchange s | pair solves | peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Si 2x2x2 | 8 | 1 | 22.6 | 12.8 | 11.4 | 0.67 | 6,736 | 0.8 |
| Si 3x3x3 | 14 | 1 / 4 | 31.1 / 32.9 | 22.6 / 22.1 | 19.5 / 18.7 | 2.2 / 2.4 | 40,102 | 1.0 / 0.8 |
| Si 4x4x4 | 36 | 1 / 4 | 79.4 / 76.4 | 62.7 / 63.3 | 54.5 / 53.1 | 6.8 / 8.5 | 306,432 | 2.3 / 1.5 |
| Si8 2x2x2 | 8 | 1 / 4 | 75.0 / 40.7 | 67.0 / 29.8 | 57.4 / 23.8 | 3.0 / 2.8 | 82,368 / 90,624 | 3.1 / 2.4 |
| h-BN 6x6x1 | 20 | 1 / 4 | 221.5 / 142.8 | 182.6 / 105.2 | 169.3 / 92.8 | 6.3 / 5.6 | 95,664 | 4.0 / 2.5 |
| Si64 Gamma | 1 (real) | 1 / 4 | 162.7 / 156.4 | 158.1 / 151.5 | 133.9 / 134.8 | 13.3 / 5.9 | 111,360 | 9.0 / 9.2 |
| Si64 2x2x2 | 8 | 4 | 423.8 | 415.4 | 351.4 | 43.2 | 2,707,968 | 15.3 |

The exchange is 4 to 13 % of these SCF loops. The eigensolver of the k-point
pool takes most of the rest, and on cells of a few ten thousand points (Si
at 28^3) four GPUs do not shorten it: each k-point's solve is a series of
small launches from Python, and the threads of the GPUs take turns at them.
The eigensolver of the Gamma point uses one GPU (its four-GPU exchange builds
are faster: 13.3 against 5.9 s). The FFT box is the grid of the cell, so its
edges set the cost of the transforms: one batch of 32 complex transforms
takes 1.28 ms at 64^3, 2.46 ms at 68^3 (4 x 17), 1.89 ms at 72^3, and 0.60 ms
at 41^3 against 0.34 ms at 40^3 (A100, `benchmarks/exchange_fft_sizes.py`).

## Selection, fallback, and provenance

An explicitly requested `scipy`, `native`, or `cupy` backend is a strict,
clean comparison mode. If its runtime is unavailable or the selected
physical/eigensolver path is unsupported, the run reports an actionable error
instead of silently composing it with another backend or choosing another
solver.

`auto` is the hybrid and fallback-enabled mode. It probes capabilities,
selects valid implementations independently for finite-difference
construction, Hamiltonian/eigensolver execution, and Hartree, and records why
any preferred component could not be used. The dry run, text report, and
archive provenance identify at least:

- requested and selected backend;
- finite-difference builder and Hartree backend;
- float dtype and CPU/GPU device;
- implementation description and build/runtime details;
- symmetry mode, detected group order, wedge size, orbital-sector policy,
  and any representation fallback;
- every fallback reason;
- sparse Laplacian size and nonlocal projector count where available.

This makes a successful fallback visible and keeps performance results
auditable.

## Timing and profiling

The reference source timings remain available and are carried into accelerated
results. Static preparation reports pseudopotential loading, grid creation,
finite-difference construction, local and nonlocal ionic setup, valence/core
density setup, ion-ion energy, and total preparation wall time. SCF reports
initial Hartree and XC, Hamiltonian binding, diagonalization,
occupation/density construction, iterative Hartree and XC, mixing/energy, and
total SCF wall time.

Accelerated backends add coarse execution statistics:

- initialization and optional warmup;
- local-potential update count and time;
- complete Hamiltonian application count, orbital-vector count, total time,
  and average time;
- host-to-device, synchronized device, and device-to-host time when relevant;
- accelerated Hartree call, boundary/RHS, linear-solve, and transfer totals;
- backend selection and fallback provenance.

The accelerated preparation also records when each of its stages ran:
`preparation_stages` of a prepared system is a list of stage, thread,
start and end, in seconds from the entry of `prepare_single_point` and in
the order of the starts. Its stages run on several threads beside each
other and some lie inside others, so it is a timeline and not a sum; a
stage whose name ends in `_wait` is the wait of its thread at the join of
another. The MPI runner writes the list for every rank
(`per_rank[].preparation_stages` in `timing.json`), where the stage that
ends last shows what a rank waited for before its first SCF step; the
backend report and the result archive do not hold it.
`GPU_SETUP_OPTIMIZATION.md` names the stages.

`--profile-operator` requests one representative synchronized breakdown of
finite-difference, local-potential, and nonlocal-projector application. It is
an opt-in diagnostic. Production eigensolvers do not place timers or device
synchronizations around every Hamiltonian term inside Chebyshev recurrences,
because that would materially distort the workload being measured.

## Package layout

```text
parsec_python/acceleration/
├── backends/       SciPy, optional native, and optional CuPy execution layers
├── benchmarks/     The MPI full-SCF runner (mpi_full_scf.py), captures and timing tools
├── experimental/   The MPI sector context of that runner (mpi_scf.py) and replay prototypes
├── Grid/           The reference cluster grid, built slab by slab
├── Laplacian/      Exact-key lazy full-grid finite-difference descriptor
├── Hamiltonian/    Backend-bound matrix-free Hamiltonian API
├── Eigensolvers/   CuPy full-grid and representation-sector eigensolvers
├── Hartree/        Fast multipoles plus SciPy/native/CuPy Poisson solvers, the device wire/slab/Calc_Hartree_G solves
├── Symmetry/       Reflection orbits, characters, sector stencils, projected operators
├── Occupations/    Fused device-resident orbital-density construction
├── V_ion/          Native radial local/density/KB setup wrappers
├── V_xc/           Cached native CA/PZ-LDA evaluator, CuPy evaluators of the other functionals (also periodic spin)
├── SCF/            Reference SCF composition with backend substitution
├── Output/         Reference report plus backend provenance/statistics
├── native/         Optional CMake/pybind11 C++17/OpenMP extension
├── tests/          Backend parity, selection, fallback, and timing tests
├── models.py       Backend identity, provenance, statistics, result wrapper
├── cli.py          Accelerated command-line orchestration
├── resident.py     Authenticated local warmed-worker runtime
└── driver.py       Optimized preparation and workflow orchestration
```

See [ACCELERATION_AUDIT.md](ACCELERATION_AUDIT.md) for the PARSEC source
mapping, supported-scope alignment status, and the C++/CuPy/NumPy decision for
every single-point stage.

Acceleration modules may depend on readable `parsec_python` components. Core
scientific modules do not import acceleration internals; only the public API
and canonical launcher select the optimized workflow. This keeps the physical
implementation independently inspectable and testable.

## Validation

From the repository root, make `src` importable and run the accelerated tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s src\parsec_python\acceleration\tests -p "test_*.py" -v
```

The consistency matrix (`benchmarks/consistency_matrix.py`) checks physical
invariants across the routes of the command line on GPUs, one line per case
with `PASS`, `FAIL` or `ERROR` and the numbers: the force on an atom on a grid
point, 5e-4 bohr from one or 0.05 bohr off a site against the slope of the SCF
free energy `E - TS` at steps of 1e-3 and 2e-3 bohr (1e-5 Ry/bohr; C POTRE and
FHI, Si, Ge and Mg with core corrections, Ag with its d projector, O, H, B, N;
both routine sets; clusters, Gamma cells, a 2x2x2 mesh, a slab, a wire; spin,
spin-orbit, LDA+U), MPI runs of 2 and 4 ranks against one process (1e-8 Ry,
1e-7 Ry/bohr), the default route against the same input without symmetry
(a mesh also against the whole mesh written as a list; 1e-7 Ry, 1e-5 Ry/bohr),
the off values of the switches published after 3b5d134c0 against that
commit's results pinned in `tests/data/consistency_pins.json` (1e-9 Ry,
1e-8 Ry/bohr) and the same input twice in fresh processes (1e-10 Ry, the same
SCF steps). Further groups compare three-step BFGS relaxations and molecular
dynamics with and without symmetry and in fresh processes (`traj`: energies
1e-7 Ry, positions 1e-6 bohr), band paths with and without symmetry (`band`:
levels 1e-7 Ry), guarded meshes against the whole mesh as a list, the real
Gamma and TRIM solves and the k-point sectors against their off values
(`kpoint`), and the spin-orbit sectors that `PARSEC_SOC_SECTORS` forces
against the whole grid (`soc`). `--list` names the cases, `--group` and
`--cases` select them, and the MPI cases run one per `srun` launch on one or
more nodes (`--group mpi --cases NAME`). The
inputs are written by the script from tables of the repository, so later
trees run it unchanged; a tree that publishes a switch adds its off value to
`ALL_OFF`.

Performance claims should always state the input, grid, eigenstate/filter
settings, selected backend, dtype, device, thread count, and whether operator
profiling was enabled. Compare energies, eigenvalues, residual histories, and
densities against the unchanged `parsec_python` result before comparing
wall time. State also whether a symmetry cache was named: the workstation
figures below that speak of a cache hit, a warm cache or a cache load are of
runs with the cache that was then the default and now needs
`--symmetry-cache DIRECTORY`.

As one machine-specific implementation check, the canonical H2 grid
(`N=179,944`) with a 16-vector block and ten warmed applications measured
about 0.0363 s/application for SciPy and 0.00835 s/application for the
C++/OpenMP backend (about 4.35x faster, OpenMP maximum 32 threads). This is a
kernel microbenchmark, not a promise for total SCF speed: Hartree work,
projected dense algebra, problem size, memory bandwidth, and thread settings
also matter.

On the benzene grid (`N=268,096`), the initial spherical boundary/RHS stage
measured 11.92 s with the reference spherical-harmonic calls and 1.41 s with
the recurrence (8.46x). The complete initial Hartree solve measured 13.02 s
for the reference, 2.85 s for accelerated SciPy, 2.08 s for native with 24
OpenMP threads, and 4.52 s on the available laptop GPU. GPU CG is limited by
the scalar reductions required by every iteration on this case; the GPU path
is retained because its eigensolver can dominate larger CHEBFF workloads.
On that same benzene domain, construction of the 9,286,528-nonzero
finite-difference CSR matrix measured 0.620 s in Python/SciPy and 0.125 s in
the native builder (4.94x).

On the 523,984-point naphthalene benchmark, the cached native boundary/RHS
took 0.063 s versus 2.639 s for the NumPy recurrence path (41.9x), with
`max |delta b_eff| = 2.27e-13`. The compact native CG took 0.710 s for the
initial 276-iteration solve. Across the complete 11-solve SCF run, Hartree
fell from 55.77 s to 9.76 s: 0.56 s for all boundary/RHS work and 9.20 s for
all linear solves. Total accelerated Python wall time fell from 88.94 s to
31.66 s, while the final energy remained `-123.37042729 Ry` (the recorded
28-rank Fortran result is `-123.37042748 Ry`). These are workstation-specific
measurements, but they explain why the fastest default keeps the GPU
Hamiltonian/eigensolver and selects native Hartree on this machine.

With the stencil-major layout, fused Chebyshev recurrence, and diagnostic-only
later-SUBSPACE residual work removed, the same naphthalene input retained the
complete printed SCF energy sequence and final `-123.37042729 Ry` energy.
Diagonalization fell from 9.66 s to 7.10 s, SCF from 19.68 s to 16.11 s, and
the one-process wall time from 24.41 s to 21.16 s on the RTX 5070 Laptop GPU.
The recorded 28-rank Fortran wall time remains 19.63 s.

The recorded PARSEC job obtains that time using an eight-operation `D2h`
wedge (65,498 points rather than 523,984) and four concurrent seven-rank MPI
representation groups. After applying the same totally symmetric reduction
to Hartree, the Python Hartree subtotal fell from 7.36 s to 0.96 s and total
process wall time to 13.68 s. The bit-exact skip-ahead DLARNV generator then
reduced the latest run to 6.52 s diagonalization, 8.12 s SCF, and 11.43 s
best observed complete wall time (13.13 s in a repeated cold-start run), with
unchanged final `-123.37042729 Ry`.

With all eight `D2h` reflection representations active on the GPU, the same
input converged in ten SCF iterations to `-123.37042737 Ry`; the recorded
28-rank Fortran value is `-123.37042748 Ry`. The final representation label of
every one of the 30 printed states matched PARSEC. On the measured RTX 5070
Laptop GPU, repeated runs gave 4.06--4.21 s diagonalization, 5.68--5.86 s
SCF, and 12.42--12.55 s cold complete-process time. PARSEC-style global
sorting trimmed the final active sector counts to `9 8 8 9 7 9 9 9`. The
subsequent structural pass removed the unused full-grid CUDA allocation,
batched host projection, added exact-key operator caching, and shared the
invariant CUDA kernels. The current GPU-ready cache also shares sector
neighbor topology: its naphthalene entry is about 29 MB rather than 211 MB,
and a measured cache load fell from about 0.131 s to 0.035 s. Cache-hit runs
retained the same energy and representation labels. Keeping selected orbitals
on the wedge reduced the ten repeated occupation/density stages from about
0.025 s to 0.012--0.013 s and cut that repeated orbital workspace by eight;
the overall wall change is small because diagonalization dominates. CUDA-
stream schedules with 2 or 8 workers were measured and were slower, so they
remain an explicit profiling option rather than the default. Multi-GPU sector
distribution is implemented automatically when more than one device is
visible, but this one-GPU workstation cannot supply a scaling measurement.

The next general optimization pass retained the same ten-iteration energy
trajectory and final `-123.37042737 Ry`. On the same naphthalene run, fusing
the KB scatter, reducing SCF scalar algebra, and constructing the native
Hartree RHS directly on symmetry orbits reduced diagonalization from 4.326 s
to 3.868 s, Hartree from 1.031 s to 0.611 s, mixing/energy from 0.363 s to
0.173 s, and SCF wall time from 5.932 s to 4.825 s. A warm geometry/phase/
operator-cache process completed in 6.69 s versus 10.68 s for the prior v3
profile. These are measured workstation timings, not portable guarantees.

The latest exact pass added the custom canonical-order CUDA `B.T` projector,
persistent native Hartree geometry, fused native CG matrix-vector/dot work,
compact end-to-end scalar fields, cached CUDA discovery, small projected host
LAPACK, broader exact scalar symmetry, and multi-GPU sector assignment. In a
paired naphthalene comparison, replacing only cuSPARSE `B.T` with the custom
kernel reduced diagonalization from 3.842 s to 3.622 s and SCF from 4.585 s
to 4.403 s. The final default validation took 3.615 s diagonalization,
0.627 s Hartree, 0.063 s mixing/energy, and 4.380 s SCF. Every printed
ten-iteration energy was unchanged and the final value remained
`-123.37042737 Ry`. Complete-process wall time was 8.29 s in that cold Python
process; CUDA initialization makes this figure more variable than the SCF
subtotal.

The subsequent synchronization/structure audit removed cuSPARSE construction
from the production representation path, retained raw KB CSR factors, kept
each short Lanczos scalar recurrence on its CUDA stream, and batched its
single tiny host transfer. It also added audited complete-subspace MGS,
stable shared local-potential buffers, adaptive projector reductions, a
2,048-lane bit-exact DLARNV tile, and a lightweight exact atom-matching path
that avoids importing SciPy optimize. The final cache-hit naphthalene run
again reproduced all ten printed energies and `-123.37042737 Ry`, with
3.502 s diagonalization, 0.601 s Hartree, 4.235 s SCF, and 8.39 s internal
wall time (9.08 s measured complete process). The independent H2 full-
nonlocal case also converged with the new defaults. These timings vary with
CUDA initialization and system load; the unchanged SCF trajectory is the
acceptance criterion.

The 8.39 s one-shot total is **not** an end-to-end improvement over the saved
v12 value of 8.29 s, despite the 0.145 s reduction in its SCF subtotal. Phase
instrumentation showed that the difference is in pre-SCF CUDA/process setup,
not the DFT kernels. Repeated current-code warm-cache runs measured 7.06,
7.17, and 7.29 s internally (7.74, 7.82, and 7.93 s for the complete Python
process). Consequently, compare medians from interleaved runs with identical
cache, console, and power-state conditions; do not rank two implementations
from one `Total accelerated Python wall time` line. New reports expose both
`Pre-SCF setup/reporting wall time` and `Post-SCF finalization/reporting` to
make that distinction visible.

The startup-recovery pass overlaps CUDA driver/device discovery with the
independent CPU reference construction. The final default naphthalene runs
used automatic symmetry and the canonical-order custom CUDA projector. Their
internal totals were 7.99 s for the first cold-driver process, followed by
6.03 s and 6.39 s, giving a three-run median of **6.39 s**; complete-process
times were 8.62, 6.65, and 7.01 s. Every one of the 300 printed eigenvalue
rows, all ten total energies, the final `-123.37042737 Ry`, and sector counts
`9 8 8 9 7 9 9 9` exactly matched `parsec_v10_custom_valid.out`. Thus the
validated default recovered and surpassed the historical 6.54 s internal
target without changing the DFT or eigensolver trajectory. Set
`PARSEC_OVERLAP_CUDA_INITIALIZATION=0` to restore sequential initialization
for profiling; it is not the measured-fast default.

The subsequent fast-start pass removed full-grid Laplacian construction from
exact reduced-operator cache hits, replaced repeated full-buffer hashing with
validated upstream keys, reused the cached totally symmetric operator for
native Hartree CG, and memory-mapped the 55 MiB native multipole table. On the
same naphthalene case, operator hashing fell from 0.119 s to 0.0001 s and
Hartree cache restoration from 0.539 s to 0.050--0.078 s. A fresh process was
still dominated by a machine-variable 2.65 s CUDA context initialization.
Through the resident worker, backend resolution took 0.00001 s, pre-SCF setup
fell to 0.464 s, and the second complete internal run took **4.53 s** (4.71 s
client round trip), versus 6.69 s for the stored v4 cache-hit run. All ten SCF
energies, the final `-123.37042737 Ry`, and sector counts remained unchanged.

The next resident pass added exact in-process static-system/operator reuse,
an exact finite-difference-NNZ cache, workload-sized OpenMP teams for repeated
Hartree/CA-LDA vector loops, and one-thread OpenBLAS for the tiny projected
host eigensystems. Against `parsec_resident_optimized.out` at 5.00 s, three
warm naphthalene runs measured **4.21, 4.23, and 4.19 s**. The complete ten-
step energy trajectory was bit-identical at printed precision; all 300 printed
eigenvalues differed by at most `2.0e-10 Ry`, and the final energy and sector
counts remained `-123.37042737 Ry` and `9 8 8 9 7 9 9 9`.

The fixed-point symmetry and CHEBDAV/Hartree audit then used Si28H36 as the
large-sector acceptance case. Exact stabilizer selection produced sector
dimensions `182718 178378` from 361,096 full-grid points. Against the prior
25.17 s full-orbital-grid validation, the retained default measured 22.68 s
total, 19.164 s SCF, 16.731 s diagonalization, and 1.937 s Hartree. It
converged in 12 SCF steps to `-263.51147823 Ry`; the full-grid result was
`-263.51147864 Ry`, a `4.1e-7 Ry` difference, and the largest final printed
eigenvalue change was `1.11e-4 Ry` at the input's `1e-4` diagonal tolerance.
The retained changes are exact stabilizer-aware sectors, contiguous-workspace
CHEBDAV projection, and chronological Hartree initialization. Fortran-order
Davidson storage, an alternate in-place GEMM, single-GPU sector streams, and
PyAMG were measured slower or unstable and are not production defaults.

The cold-first-calculation pass retained three further arithmetic-preserving
changes. For naphthalene, exact integer symmetry maps reduced geometry and
representation construction from about `0.69 + 0.34 s` to `0.35 + 0.18 s`;
four-way cache-miss operator assembly reduced the operator build from
`1.88--1.91 s` to `1.59 s`. A six-run fresh-cache A/B comparison measured
median complete times of `7.66 s` with one worker and `7.31 s` with four.
The contiguous full-workspace CHEBDAV Ritz
projection reduced its fresh-cache median from `8.35 s` to `7.46 s`, with the
projected stage falling from about `1.06 s` to `0.106 s`. For Si28H36, that
Ritz change reduced projected work from about `3.00 s` to `2.39 s`, while the
fused active-prefix update reduced the initial eigensolver from `13.63 s` to
`13.16 s` and diagonalization from `16.11 s` to `15.63 s`. Parallel two-sector
assembly reduced that component, but its six-run complete-time median was
`22.97 s` versus `22.93 s` serial, so the adaptive production policy retains
one worker for at most two representations. Every retained Si run kept
the 12-step trajectory and final `-263.51147823 Ry`; every retained
naphthalene run kept all ten printed energies and `-123.37042737 Ry`.
Standalone totals still vary by roughly one to two seconds with Windows CUDA
driver startup, so retain/reject decisions used interleaved stage timings and
fresh molecule-specific caches. Audited one-prefix CGS, CholeskyQR2, a
column-major Davidson basis, a 12-vector Davidson block, and deferred Hartree
cache publication were slower and were removed.

# Acceleration implementation

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
as PARSEC does. Fixed-point orbits are admitted by their exact stabilizer
characters, so representation dimensions may differ. All operations,
dimensions, device assignments, and fallback decisions are reported.

The canonical launcher accepts the same supported `parsec.in` and adjacent
`*_POTRE.DAT` files on every backend:

```powershell
python src\parsec_python\main.py INPUT
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
  --backend {auto,scipy,native,cupy} \
  --symmetry {auto,on,off} \
  [--symmetry-cache DIRECTORY | --no-symmetry-cache] \
  [--profile-operator]
```

The symmetry mode defaults to `auto`: detect exact supported operations,
apply every reduction supported by the selected backend, and safely retain
the full grid if no nontrivial group or orbital representation is usable.
`on` requires nontrivial usable symmetry and reports an error instead of
falling back; `off` skips detection and forces the full-grid comparison path.
A PARSEC input with `Ignore_Symmetry=true` selects `off` unless an explicit
command-line mode overrides it.

Symmetry geometry, native multipole/boundary geometry, character phases, and
representation operators are
content-addressed under
`.parsec_cache/symmetry` beside the input by default. The SHA-256 key covers
the complete labeled atomic geometry, exact active grid, detector tolerances,
canonical finite-difference/projector buffers, projector signs,
grid-to-wedge map, and character phases. A grid, geometry, pseudopotential,
or symmetry change therefore builds a new entry. Loaded orbit metadata is
validated before use; a missing, old, or damaged entry is rebuilt by the
exact detector. Use `--symmetry-cache DIR`
to relocate it or `--no-symmetry-cache` for a controlled cold setup.
The cache stores GPU-ready stencil-major coefficient codes and palettes rather
than duplicate reduced CSR matrices. When representation sectors have the
same neighbor topology, one host archive array and one CUDA int32 neighbor
allocation are shared across all sectors; only representation-dependent
codes, palettes, and KB factors remain separate. Native multipole geometry is
exported to its own exact-key archive, so later Python processes reconstruct
the C++ boundary builder without repeating its angular and missing-neighbor
setup.

On an exact representation-cache hit, the default symmetry/GPU path keeps
the full-grid finite-difference matrix lazy. Its SHA-256 provenance key
covers every discrete grid and stencil input, while the cached reduced bundle
contains the operators actually consumed by the representation eigensolvers.
A cache miss, symmetry fallback, explicit full-grid run, or modular request
for the full operator materializes the same validated C++ CSR. The report
records whether this happened. The largest native Hartree coefficient table
is stored as a memory-mapped NPY sidecar, eliminating one full archive copy
while retaining the same native-owned complex128 buffers.

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
| `cupy` | Predominantly-float64 device-resident Hamiltonian, CHEBFF/CHEBDAV/SUBSPACE state, size-gated FP32 later filtering, and shared-CSR Poisson CG | CuPy build matching the installed CUDA runtime and a usable GPU | GPU eigensolver and Hartree linear algebra |
| `auto` | Accuracy-preserving hybrid when possible: native finite-difference construction and Hartree, plus a device-resident CuPy Hamiltonian and eigensolver | Native extension and CuPy for the complete hybrid; missing components fall back with recorded provenance | Fastest default execution while retaining a reproducible component-by-component fallback trail |

`auto` selects components independently. When both optional runtimes are
available, finite-difference construction and Hartree use the native C++
kernels, while Hamiltonian applications and CHEBFF/CHEBDAV/SUBSPACE use
CuPy. If only one accelerator is available, `auto` uses the compatible
combination and records each choice. Explicit `scipy`, `native`, and `cupy`
remain clean end-to-end comparison modes rather than hybrid aliases.

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

### CuPy

The CuPy path uses float64 for physical operators, initial eigensolver
acceptance, Rayleigh--Ritz, densities, potentials, convergence, and energies. The finite-difference operator,
projector factors and cached transpose, local potential, Chebyshev blocks, and
saved eigensolver state remain on the device across Hamiltonian applications
and SCF iterations. Synchronization is restricted to coarse transfer and
solver timing boundaries.

On large sectors (at least 100,000 real-space rows by default), only the
later-SCF Chebyshev recurrence uses FP32 stencil coefficients, projector
factors, vectors, and local-potential shadows. Its output is converted back to
FP64 before the generalized Ritz projection and every SCF/energy operation.
Initial CHEBDAV/CHEBFF remains entirely FP64. This boundary reproduced the
full Si28H36 energy trajectory while reducing its repeated filtering cost;
small symmetry sectors stay FP64 because the conversion overhead is not
profitable. Set `PARSEC_CUPY_MIXED_FILTER=off` for a strict FP64 comparison,
`on` to force the experimental path, or change the automatic threshold with
`PARSEC_CUPY_MIXED_FILTER_MIN_ROWS`. Auto also requires
`N_grid * N_working_states^2 >= 100000000`; override that crossover with
`PARSEC_CUPY_MIXED_FILTER_MIN_WORK`. Shared symmetry-sector potentials update
both the FP64 field and every enabled FP32 shadow before a solve.

Large generalized Rayleigh--Ritz overlap matrices use FP64 cuBLAS DSYRK to
form only the lower triangle of `X.T @ X`; the former FP64 GEMM remains the
automatic fallback. Set `PARSEC_CUPY_RITZ_SYRK=off` to force GEMM. Optional
asynchronous stage timing is enabled with `PARSEC_CUPY_STAGE_TIMING=1`; it
reports first-CHEBDAV and later-SUBSPACE spectral-bound, filtering,
orthogonalization, projection, residual/locking, cleanup, Ritz-Hamiltonian,
overlap, and rotation subtotals without synchronizing every Hamiltonian
application.

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
allocated in this path. All reduced kinetic and KB factors are constructed
from one canonical sparse gather. On a cache miss, independent representation
assemblies run serially for at most two representations and on at most four
CPU workers for larger decompositions. This cold-time policy was selected by
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
limit it with `PARSEC_CUPY_SECTOR_STREAMS=N`.
Independent one-vector representation Lanczos bounds can also be profiled
with `PARSEC_CUPY_COLLECTIVE_LANCZOS=1`. It is off by default because it was
numerically identical but slower on the measured GPU; large filters remain
serialized regardless.

On a multi-GPU host, `PARSEC_CUPY_DEVICES=auto` (the default) assigns
representations round-robin to all visible devices and solves independent
sectors concurrently. Use `current` to restrict execution to the current
device, or a comma-separated list such as `0,2`. Static operators and saved
subspaces stay on their assigned GPUs; only globally selected wedge orbitals
are gathered to the primary device, using peer access when available and an
exact host-staged fallback otherwise.

For the centered finite-difference operator, production CuPy transposes each
short canonical CSR row into a stencil-major layout
``neighbor[slot, grid_row]``. Adjacent CUDA threads therefore read contiguous
int32 neighbor rows and uint8 coefficient codes; the palette retains the
exact float64 coefficient bits. Each thread still visits its row in canonical
CSR order. A second kernel fuses kinetic, the current local diagonal,
optional nonlocal image, and the normalized Chebyshev recurrence into one
grid pass for blocks of up to six orbitals. For KB projectors a canonical-
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
`parallel` force either policy for profiling.
Set `PARSEC_CUPY_CUSTOM_PROJECTOR_DOT=0` to restore cuSPARSE and
`PARSEC_CUPY_FUSED_PROJECTORS=0` for an unfused comparison. The earlier compact CSR-order
kernel and generic CuPy CSR remain automatic fallbacks. Set
`PARSEC_CUPY_STENCIL_MAJOR=0` before Python starts to exercise the compact
CSR-order fallback explicitly.

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
persistent column-major `H X` workspace avoids repeated large CuPy allocations
and strided real-space-kernel reads. Set `PARSEC_CUPY_GENERALIZED_RITZ=off` to
disable it, or `on` to request an audited attempt for every complete basis.
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
In the default hybrid, the native CG implementation is selected for Hartree
while the Kohn--Sham Hamiltonian and eigensolver remain on the GPU; this avoids
the reduction-heavy GPU CG path without moving eigensolver state off-device.
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
`U`. Native extension 0.5 additionally precomputes and persistently caches
orbit-summed multipole coefficients, constructs the boundary-corrected
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

`--profile-operator` requests one representative synchronized breakdown of
finite-difference, local-potential, and nonlocal-projector application. It is
an opt-in diagnostic. Production eigensolvers do not place timers or device
synchronizations around every Hamiltonian term inside Chebyshev recurrences,
because that would materially distort the workload being measured.

## Package layout

```text
parsec_python/acceleration/
├── backends/       SciPy, optional native, and optional CuPy execution layers
├── Laplacian/      Exact-key lazy full-grid finite-difference descriptor
├── Hamiltonian/    Backend-bound matrix-free Hamiltonian API
├── Eigensolvers/   CuPy full-grid and representation-sector eigensolvers
├── Hartree/        Fast multipoles plus SciPy/native/CuPy Poisson solvers
├── Symmetry/       Reflection orbits, characters, and projected operators
├── Occupations/    Fused device-resident orbital-density construction
├── V_ion/          Native radial local/density/KB setup wrappers
├── V_xc/           Cached native CA/PZ-LDA evaluator
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

Performance claims should always state the input, grid, eigenstate/filter
settings, selected backend, dtype, device, thread count, and whether operator
profiling was enabled. Compare energies, eigenvalues, residual histories, and
densities against the unchanged `parsec_python` result before comparing
wall time.

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

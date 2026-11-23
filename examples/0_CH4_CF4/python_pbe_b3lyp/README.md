# Python PP-PBE(B3LYP) reproduction of the ARES CH4/CF4 C 1s benchmark

This directory extends [`../python_pbe`](../python_pbe/README.md) by the
refinement step of the paper (Xu, Prendergast, Qian, *J. Chem. Theory
Comput.* **18**, 5471 (2022)): after the ordinary PBE SCF, the
exchange-correlation energy of the converged PBE orbitals and density is
evaluated once with B3LYP and replaces the PBE one in the total energy. It
is not a B3LYP SCF. In ARES the step is `Lrefine=T`; here it is the input
line

```text
Energy_Refinement: b3lyp
```

or the command-line option `--energy-refinement b3lyp`. The native Python
solver does everything; ARES and PARSEC are not called.

The full sets of molecules of the paper and of its successor for the third
row are in [`../../0d_xps_core_levels`](../../0d_xps_core_levels/README.md).

## Directory correspondence

```text
python_pbe_b3lyp/
  CH4/{IS,FS_1s}/parsec.in
  CF4/{IS,FS_1s}/parsec.in
  pseudopotentials/            the four POTRE files of ../python_pbe
  analyze.py                   tables against ../0_ARES_CH4_CF4/test_CH4_CF4/b3lyp
                               (its numbers are also kept inside the script)
```

The four inputs are those of `../python_pbe` (same geometry, grid,
pseudopotentials, atomic reference corrections and solver controls) plus
the `Energy_Refinement` line. `IS` is the neutral molecule with the ordinary
carbon potential, `FS_1s` the fully screened final state with the carbon
potential generated with one 1s electron removed and `Net_Charges: 1 e`.
The refinement needs no pseudopotential of its own: the PBE potentials and
their atomic all-electron-minus-pseudo reference energies are reused, as in
ARES.

Each case folder holds the logs of up to two runs. A run writes the first
name; the copy kept with the example has the second, as `parsec_python.out`
in `../python_pbe`, and `analyze.py` reads it where the first is absent:

| log of a run | kept copy | run |
|---|---|---|
| `parsec_gpu_hartree.out` | `parsec_gpu_hartree_python.out` | the command below with `PARSEC_HARTREE_LINEAR_BACKEND=cupy` (Poisson solves on the GPU, the default since 2026-11-07); all four cases |
| `parsec.out` | `parsec_python.out` | the command below with `PARSEC_HARTREE_LINEAR_BACKEND=native` (Poisson solves on the CPU, the default before 2026-11-07); all four cases |

The kept logs are of 23 November 2026, when the pseudopotential files were
resampled again (`../python_pbe/README.md`), one NVIDIA A100 per run ("Time
and memory" below). Until 20 November 2026 CF4 FS_1s had no CPU-solver log: its run of
11 October 2026 on a laptop was stopped during its refinement, when other
jobs on the machine had slowed its CPU Poisson solves to more than a minute
each.

## Running

From one of the four case directories:

```powershell
..\..\..\..\..\.venv312\Scripts\python.exe `
  ..\..\..\..\..\src\parsec_python\main.py parsec.in `
  --pp-dir ..\..\pseudopotentials `
  --backend auto --symmetry auto --log parsec.out --no-archive
```

On a GPU the Poisson solves of the SCF and of the refinement run there, the
default since 2026-11-07, which is 22 times faster per pair for CF4 than
the 28 CPU threads of the runs below (`--log parsec_gpu_hartree.out` for
that log); set `$env:PARSEC_HARTREE_LINEAR_BACKEND = "native"` before the
command to solve them on the CPU, as for `parsec_python.out`.
After the four jobs:

```powershell
.venv312\Scripts\python.exe examples\0_CH4_CF4\python_pbe_b3lyp\analyze.py --log parsec_gpu_hartree.out --parts
.venv312\Scripts\python.exe examples\0_CH4_CF4\python_pbe_b3lyp\analyze.py
```

`parsec.out` ends with a block "Energy refinement" that lists the five
parts of the B3LYP energy, both exchange-correlation energies, the PBE and
the refined total, and the two reference-corrected all-electron totals that
the binding energies are built from:

```text
PBE all-electron total (as above)           :         -80.96104086 [Ry]
B3LYP-refined all-electron total            :         -81.00287497 [Ry]
```

`analyze.py` reads the `[Ry]` lines and converts with the CODATA Rydberg
energy, 13.605693123 eV. The `[eV]` lines of `parsec.out` use PARSEC's
display constant 13.6058, which is 7.9e-6 too large: 2.3 meV on a binding
energy of 297 eV. The table of `../python_pbe/README.md` was made from
those lines, so its binding energies are 2.3 to 2.4 meV above the ones
here; the calculations are the same to the last printed digit.

## The B3LYP energy that matches ARES

`ares.log` prints the five parts and their weights, and its `*B3LYP-XC` is
their weighted sum to the last printed digit in all four logs:

```text
E_xc(B3LYP) = 0.20 E_x(HF) + 0.08 E_x(Slater) + 0.72 E_x(B88)
            + 0.19 E_c(VWN-RPA) + 0.81 E_c(LYP)
```

- `B88-X` is the complete Becke 88 exchange, Slater part included (it is
  9 % larger in magnitude than `Slater-X`), so this is the usual
  `0.2 HF + 0.8 Slater + 0.72 (B88 - Slater)`.
- The local correlation is the VWN fit to the **RPA** energy of the uniform
  gas (`VWN1RPA-C`; functional III of Gaussian, `LDA_C_VWN_RPA` and the
  default `B3LYP` of libxc), not the fit to the Monte Carlo data (VWN5). The
  two differ by a quarter: for CH4 IS 15.2 eV against 11.2 eV, where ARES
  prints 15.17 eV.
- `HF-X` is the exact exchange of the occupied pseudo-valence orbitals,
  `-sum_ij (ij|ij)` over doubly occupied orbitals; no core-valence exchange.
- `*ref.Total-energy` and `*ref.AE energy` are the PBE values plus
  `B3LYP-XC - XC`, to 1e-7 eV in all four logs: the refinement replaces the
  exchange-correlation energy and nothing else, and the atomic reference
  correction stays the PBE one.
- None of the four pseudopotentials carries a nonlinear core correction
  (`core_correction="F"` in the UPF files), so the benchmark does not say
  how ARES treats one. The solver then gives the four density functionals
  the valence plus core density of the SCF and the exact exchange the
  valence orbitals, and says in `parsec.out` that this is unchecked.
- ARES ran spin-polarized (`LSPIN=T`) and ended with equal spin densities;
  the solver is spin-unpolarized.

The four density functionals of the solver agree with libxc 7.0.0 to
machine precision at every tested density and gradient (unit tests). They
are integrated with the gradient of the self-consistent PBE: the centered
finite-difference first derivative of the grid order, with the density
extended by zero outside the sphere. The exact exchange solves one Poisson
equation per pair of occupied orbitals with the Hartree solver of the SCF
on the full grid, with the multipole boundary values (`Solver_Lpole: 9`) of
the pair density and conjugate gradients started from zero: 10 solves for
CH4, 136 for CF4.

## Results

All runs converged. The energies below are those of the four runs with the
GPU Poisson solver; the four runs with the CPU solver give the same numbers
to the eight printed decimals in Rydberg. Energies in eV:

| case | quantity | Python | ARES | Python - ARES |
|---|---|---:|---:|---:|
| CH4 IS | PBE XC | -86.292505 | -85.779604 | -0.5129 |
| CH4 IS | B3LYP XC | -86.861687 | -86.339786 | -0.5219 |
| CH4 IS | refined total | -220.374881 | -220.322634 | -0.0522 |
| CH4 IS | refined all-electron total | -1102.100259 | -1102.047921 | -0.0523 |
| CH4 FS_1s | PBE XC | -98.040415 | -97.676314 | -0.3641 |
| CH4 FS_1s | B3LYP XC | -98.504940 | -98.151206 | -0.3537 |
| CH4 FS_1s | refined total | -322.216283 | -322.190689 | -0.0256 |
| CH4 FS_1s | refined all-electron total | -804.840695 | -804.815051 | -0.0256 |
| CF4 IS | PBE XC | -552.746340 | -549.927090 | -2.8193 |
| CF4 IS | B3LYP XC | -555.381336 | -552.472080 | -2.9093 |
| CF4 IS | refined total | -2784.098753 | -2783.874322 | -0.2244 |
| CF4 IS | refined all-electron total | -11912.200991 | -11911.975615 | -0.2254 |
| CF4 FS_1s | PBE XC | -558.349877 | -555.693413 | -2.6565 |
| CF4 FS_1s | B3LYP XC | -560.262726 | -557.528220 | -2.7345 |
| CF4 FS_1s | refined total | -2874.764376 | -2874.549938 | -0.2144 |
| CF4 FS_1s | refined all-electron total | -11603.765647 | -11603.550306 | -0.2153 |

The binding energies, `BE = E_AE(FS_1s) - E_AE(IS)`:

| quantity | Python | ARES | Python - ARES | paper | experiment |
|---|---:|---:|---:|---:|---:|
| PP-PBE BE(CH4) | 297.1549 | 297.1476 | 0.0073 | 297.146 | 290.80 |
| PP-PBE BE(CF4) | 307.7132 | 307.7151 | -0.0019 | 307.715 | 301.85 |
| PP-PBE CF4-CH4 shift | 10.5583 | 10.5675 | -0.0093 | 10.569 | 11.05 |
| PP-PBE(B3LYP) BE(CH4) | 297.2596 | 297.2329 | 0.0267 | 297.232 | 290.80 |
| PP-PBE(B3LYP) BE(CF4) | 308.4353 | 308.4253 | 0.0100 | 308.425 | 301.85 |
| PP-PBE(B3LYP) CF4-CH4 shift | 11.1758 | 11.1924 | -0.0167 | 11.193 | 11.05 |
| B3LYP correction of BE(CH4) | 0.1047 | 0.0853 | 0.0194 | | |
| B3LYP correction of BE(CF4) | 0.7221 | 0.7102 | 0.0120 | | |

"ARES" is the supplied `b3lyp` tree, "paper" the production runs behind the
paper as `../0_ARES_CH4_CF4/test_CH4_CF4/README.md` tabulates them.

The PBE binding energies agree with ARES within 7.3 meV, as in `../python_pbe`.
The refined ones agree within 27 meV and the CF4-CH4 shift within 17 meV;
against experiment the shift is +0.13 eV here and +0.14 eV in ARES, where
PBE gives -0.49 eV and -0.48 eV. With the linearly resampled pseudopotential
files used before 23 November 2026 (`../python_pbe/README.md`; the same code
and defaults, measured on 23 November 2026) the refined totals are 0.019 to
0.118 eV lower and the refined binding energies 297.2566 and 308.4289 eV
(shift 11.1724 eV).

## Time and memory

First calculations (no symmetry cache, no restart) in double precision with
the code of 23 November 2026 on one NVIDIA A100 (80 GB) per run, each run
with 32 CPUs of a Perlmutter GPU node (28 threads) and four runs side by side
on the node, after one throwaway warm-up run on each GPU; the two CPU-solver
runs of CF4 shared a node, the other two GPUs idle. "Refinement" is `Time for energy
refinement` of the log, "host memory" the high-water mark of the process at
the end of the run.

With `PARSEC_HARTREE_LINEAR_BACKEND=cupy` (conjugate gradients in double
precision on the GPU, for the Hartree potential of the SCF and for the pair
densities; the default since 2026-11-07):

| case | grid points | SCF steps | SCF [s] | occupied orbitals | pair solves | refinement [s] | s per pair | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CH4 IS | 539,152 | 28 | 4.0 | 4 | 10 | 0.8 | 0.061 | 7.6 | 2.21 |
| CH4 FS_1s | 539,152 | 21 | 3.2 | 4 | 10 | 0.8 | 0.060 | 6.7 | 2.22 |
| CF4 IS | 3,103,688 | 40 | 33.3 | 16 | 136 | 59.6 | 0.422 | 102.2 | 9.27 |
| CF4 FS_1s | 3,103,688 | 37 | 31.0 | 16 | 136 | 59.6 | 0.422 | 99.7 | 9.25 |

The default settings before 2026-11-07, now
`PARSEC_HARTREE_LINEAR_BACKEND=native` (the same conjugate gradients in
C++/OpenMP on 28 threads):

| case | grid points | SCF steps | SCF [s] | occupied orbitals | pair solves | refinement [s] | s per pair | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CH4 IS | 539,152 | 28 | 18.7 | 4 | 10 | 4.7 | 0.443 | 25.8 | 2.12 |
| CH4 FS_1s | 539,152 | 21 | 13.3 | 4 | 10 | 4.3 | 0.403 | 19.4 | 2.10 |
| CF4 IS | 3,103,688 | 40 | 458.2 | 16 | 136 | 1309.2 | 9.607 | 1773.8 | 8.73 |
| CF4 FS_1s | 3,103,688 | 37 | 406.8 | 16 | 136 | 1284.8 | 9.429 | 1698.3 | 8.71 |

Where the Fermi search stopped by bisection (`PARSEC_FERMI_GAP_ROOT=off`),
the first empty level of CH4 IS kept an occupation between 1e-12 and 1e-10
at the Fermi temperature of the input (1.4e-12). The exact exchange leaves out
orbitals whose occupation is within the electron-count tolerance of the
Fermi search, 1e-10 (`PARSEC_EXCHANGE_OCCUPATION_CUT`, default on since
22 November 2026), so CH4 IS counts four occupied orbitals and 10 pairs;
with `off` (the former threshold 1e-12) it counted five and 15, with the
same energies to the printed digits. Since 23 November 2026 the Fermi level
of the gap is the root of the electron count (`PARSEC_FERMI_GAP_ROOT`,
default on), 45 kT below the empty level, which then holds nothing: four
orbitals and 10 pairs under either threshold. On the GPU a pair solve on the CF4 grid takes 0.42 s, on
the CPU 9.4 to 9.6 s. The logs of 11 October 2026 came from a laptop (32 logical
cores, RTX 5070 Laptop GPU with 8 GB, 32 GB of memory): CF4 IS took 680 s
with the GPU solver (1.5 s per pair) and a CPU solve 8.5 to 8.8 s there,
with the same iteration counts and the same energies.

The exact exchange is the refinement: the four density functionals take
0.1 to 0.2 s (CH4) and 0.9 s (CF4). Its cost is the number of occupied pairs
times one Poisson solve from a zero start, about 565 conjugate-gradient
iterations on the 3.1 million points of CF4. ARES needed 22 s (CH4) and
1850 s (CF4) for the same step with 64 processes (total minus SCF time of
its logs).

What the refinement adds to the host memory is the orbitals: for CF4 the 65
states come to the host once (1.5 GiB) and the 16 occupied ones are copied
(0.4 GiB). The logs print the high-water mark of the process before that
download and after the refinement: it rose by 0.65 to 0.88 GiB for CF4 (GPU
solver 8.59 to 9.27 GiB, CPU solver 7.85 to 8.73 GiB for CF4 IS) and by
0.05 GiB at most for CH4. On the laptop GPU memory peaked at 2.2 to 2.3 GiB
(CH4) and at 7.7 of the 8.0 GiB of the card (CF4, mostly the memory pool of
the eigensolver); the GPU pair solver holds 0.5 GiB of it for CF4.

## Limits

- Isolated systems only; spin-unpolarized. With fractional occupations the
  exact exchange is that of the spin-restricted ensemble density matrix,
  `-sum_ij f_i f_j (ij|ij)`, and `parsec.out` says so; it is not a
  spin-polarized open-shell energy.
- The cost grows with the square of the number of occupied orbitals; no
  symmetry, screening or density fitting is used for the pair densities.
- The refinement is not variational: it carries the error of the density
  to first order. A PBE total converged to 1e-6 Ry can leave a larger error
  in it; here a five times looser SCF criterion moved it by 1e-8 Ry (CH4
  FS_1s).
- The multi-node runner (`acceleration/benchmarks/mpi_full_scf.py`) refuses
  the keyword; `main.py` on one node, with or without GPU, evaluates it.

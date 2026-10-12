# Pseudopotential generation

`pp_generation` generates norm-conserving pseudopotentials, tests their
Kleinman–Bylander form for ghost states, and writes them in the formats of
PARSEC, Quantum ESPRESSO / ARES (UPF), FHI98PP and abinit. It can generate a
potential for the ground state of an element or for the atom with an electron
removed from a core shell (a *core-hole* potential, used for core-level binding
energies).

Everything runs in Python with numpy only. The generator is a port of the
Fortran program FHI98PP (M. Fuchs and M. Scheffler, Comput. Phys. Commun.
**119**, 67 (1999)); the port reproduces the Fortran program's results (see
[Validation](#validation)). No Fortran compiler and no other program is needed.

## Quick start

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path

# graphical front end: pick the element in a periodic table, press Generate
python -m pp_generation gui

# silicon, PBE, ground state: writes PARSEC, UPF and FHI files to pp-output\Si
python -m pp_generation Si -o pp-output\Si

# silicon with one electron removed from the 2p shell, LDA
python -m pp_generation Si --xc ca --core-hole 2p -o pp-output\Si-2p

# rebuild the two libraries described below (all elements, PBE and LDA)
python -m pp_generation library --jobs 16
```

On Linux or macOS use `export PYTHONPATH=$PWD/src` and forward slashes. A
ground-state potential takes about one second, a core-hole potential two to
four seconds.

`python -m pp_generation --help` lists every option. The ones used most:

| option | meaning |
|---|---|
| `--xc pbe` \| `ca` \| `pw92` | functional, see [Functionals](#functionals) |
| `--core-hole 1s` | core shell that loses an electron; `--hole-charge 0.5` for half a hole |
| `--format parsec` (repeatable) | `parsec`, `upf`, `fhi`, `abinit`; default: the first three |
| `--scheme tm` \| `hamann` | Troullier–Martins (default) or Hamann construction |
| `--lmax 2`, `--local-channel 1` | fix the highest angular momentum or the local channel |
| `--radius s=2.5` (repeatable) | cutoff radius of one channel in bohr |
| `--core-radius 1.3` | radius of the nonlinear core correction; `0` switches it off |
| `--non-relativistic` | non-relativistic atom instead of scalar-relativistic |
| `--input-file my.ini` | an FHI98PP input written by hand, used as it is |

The same from Python:

```python
from pathlib import Path
from pp_generation import CoreHole, GenerationRequest, generate

result = generate(GenerationRequest(element="C", output_dir=Path("pp-output/C-1s"),
                                    xc="pbe", core_hole=CoreHole("1s")))
print(result.selected_local_channel, result.ae_energy_hartree, result.artifacts["parsec_species"])
```

## What is written

For `python -m pp_generation C --core-hole 1s -o out`:

| file | content |
|---|---|
| `C-1s_POTRE.DAT` | PARSEC "Martins new" format, read by the Python solver (`C_POTRE.DAT` for a ground state) |
| `c_1s_hole_1_pbe_tm.UPF` | UPF v2, semilocal; `total_psenergy` and `total_aeenergy` are set in the header |
| `C-1s_FHIPP.DAT` | the FHI98PP table (`.cpi`) on the generator's own mesh, read by Fortran PARSEC |
| `*.fhi` | the same table with an abinit header (`--format abinit`) |
| `*.ini`, `*.dat` | the FHI98PP input that was run and the generator's log |
| `*.test`, `*.local-<l>.test` | full test report for the selected local channel; ghost analysis for every channel tried |
| `*.report.json` | everything in machine-readable form, with SHA-256 of each file |

Three numbers from the summary are needed to use a potential in the solver, and
the libraries store them ready to paste as `PARSEC/parsec_atom_type.in`:

```text
Atom_Type: C-1s
Element_Symbol: C
Pseudopotential_Format: martins_new
Local_Component: d
Atomic_Energy_Correction: -35.465193 Ry
```

`Local_Component` is the channel that passed the ghost test.
`Atomic_Energy_Correction` is the all-electron minus the pseudo total energy of
the generating atom; with it the solver reports a reference-corrected
all-electron total, which is what binding energies are built from (potentials
with different ionic charge do not share an energy zero).

## Functionals

| `--xc` | functional | FHI98PP `iexc` | UPF label | PARSEC `Correlation_Type` |
|---|---|---|---|---|
| `pbe` | Perdew–Burke–Ernzerhof GGA | 6 | `PBE` | `pbe` |
| `ca` | LDA, Ceperley–Alder in the Perdew–Zunger 1981 parametrization | 3 | `PZ` | `ca` |
| `pw92` | LDA, Perdew–Wang 1992 parametrization | 8 | `PW` | not available in the Python solver |

`ca` is the functional behind `Correlation_Type ca` of the solver. (Until
October 2026 this package generated "ca" potentials with `iexc = 8`, which is
`pw92`.)

**PBE(B3LYP)** of the binding-energy papers needs no potential of its own. It
is a PBE calculation with PBE potentials followed by one evaluation of the
B3LYP exchange-correlation energy on the PBE orbitals; the atomic energy
correction stays the PBE one. In the solver it is the input line
`Energy_Refinement: b3lyp`; see `examples/0_CH4_CF4/python_pbe_b3lyp`. (The
refinement step is in this branch, `one-shot-double-hybrid`, not yet in the
default branch.)

## Core-hole potentials

A core-hole potential is generated from the atom with the occupation of one
core shell reduced, here by one electron. Its ionic charge is one higher than
that of the ground-state potential, and the generating atom is a +1 ion.

*Matched to the ground state.* A binding energy is the difference of two
calculations, one with the ground-state potential and one with the core-hole
potential on the excited atom. So that the two differ only by the hole, a
core-hole potential is generated with the `lmax`, the cutoff radii and the
local channel of the ground-state potential of the same element and
functional. This is the default of the command line, the GUI and the library
builder (`--no-ground-state-match` switches it off). If the hole is not
ghost-free in that form it gets its own choice and the result says so.

*Reference energies.* Some element templates generate an unoccupied channel at
a fixed energy a few eV above the valence levels. With a core hole all valence
levels move down by 5 to 10 eV, so these energies are shifted by the same
amount as the highest valence level (`--no-reference-shift` switches it off).
Without the shift several elements (Y, Zr, ...) cannot be generated at all.

*Using one.* Give the excited atom its own `Atom_Type` (`C-1s`) with
`Element_Symbol: C`. With the photoelectron removed (full core hole) the system
has one electron less: add `1 e` to `Net_Charges`. With `Net_Charges` unchanged
the extra ionic charge is screened by one more valence electron (neutralized
core hole). A 2p hole is spherically averaged; it stands for the average of the
2p1/2 and 2p3/2 holes.

*What is not in it.* The atoms are spin-unpolarized and scalar-relativistic.
An atom with a core hole is strongly spin-polarized, which shifts absolute
binding energies by a constant of several eV per element and shell (Xu,
Prendergast, Qian, J. Chem. Theory Comput. **18**, 5471 (2022)); binding-energy
*shifts* between chemical environments are not affected.

## The libraries

Two libraries are built with `python -m pp_generation library`:

```text
NCPPs/<XC>/<Element>/                         ground state
CoreHole_NCPPs/<XC>/<Element>/<Element>-<shell>/    one electron removed from <shell>
    PARSEC/   <label>_POTRE.DAT, parsec_atom_type.in
    UPF/      *.UPF
    FHI/      <label>_FHIPP.DAT
    diagnostics/   input, log, test reports, report.json
NCPPs/manifest.csv, CoreHole_NCPPs/manifest.csv
```

`<XC>` is `PBE` or `CA`. Each has its own README with the numbers of the
current build; each `manifest.csv` has one row per potential with status,
`lmax`, local channel, smallest ghost margin, ionic charge, both total
energies, the atomic energy correction, cutoff radii, core radius, the shift of
the reference energies, the plane-wave cutoff at which the kinetic energy is
converged to 1 meV, and whether a core hole is matched to its ground state.

The ground-state library and the core-hole potentials of H–Kr are in this
branch. The core-hole potentials of the heavier elements are in the branch
`pseudopotential` (they would add about 320 MB here); both manifests list
every potential. `python -m pp_generation library --jobs 16` rebuilds
everything in about ten minutes;
`--elements Si P S`, `--xc pbe`, `--no-core-holes` and `--format parsec`
restrict it.

### How the choices are made

1. *Ground state.* The element template, the highest `lmax` not above the
   template's and not below an occupied valence state that has a ghost-free
   local channel, then the channel with the largest worst-case margin. A margin
   below 0.002 Ha counts only if nothing else passes.
2. *Core holes.* The same input records as the ground state plus the hole and
   the shift of the reference energies; `lmax` and local channel of the ground
   state.
3. *A member cannot be constructed.* The Troullier–Martins equations have no
   solution at isolated mesh points. Then one channel's radius is moved by one
   or two mesh points (2.5 % each) for the whole family, so radii stay equal.
4. *A hole is not ghost-free in the ground-state form.* It takes its own best
   form and is marked `unmatched`. A hole without any ghost-free form is marked
   `failed`.

### Element templates

`elements/TM` and `elements/Ham` hold one FHI98PP input per element
(configuration, core/valence split, `lmax`, core correction, sometimes radii
and reference energies). `elements/sources.csv` gives the origin of each:

- Z = 1–56 and 72–80: the element database distributed with FHI98PP, unchanged.
- Z = 57–71 and 81–103: written for this package. Where abinit's table of FHI
  pseudopotentials exists for the element, the radii were chosen so that the
  template reproduces that table (to 1e-10 for the occupied channels); the
  others follow their neighbours. Eu, Tb, Dy, Ho use 4f^(n-1) 5d^1 6s^2 because
  the atom with 4f^n 6s^2 does not converge.

`elements/reviewed.csv` lists the departures from a template that are applied
by default, with the reason for each (H: purely local; K, Y: s radius moved by
one mesh point).

Lanthanide and actinide potentials with the f shell in the valence are hard and
have not been tested in molecules or solids. Treat them as starting points.

## Validation

The port was compared with the Fortran program on its own outputs.

| what | cases | result |
|---|---|---|
| all-electron atom | 256 runs, all template elements, PBE and LDA, ground state and 1s hole | log identical in 253; in 3 the iteration count differs by one |
| potential construction | 592 runs over all 103 elements, ground state and core holes | no case where only one program succeeds; logs identical in 557, iteration count only in 8; tables agree to 2e-9 |
| ghost analysis | 4332 verdicts on those runs | all equal; margins equal to the printed precision |
| format conversion | 36 elements of the earlier Fortran-made library | UPF and FHI files identical, PARSEC files to 1e-12 |
| the two libraries | all 1907 potentials, regenerated with the Fortran program from the library's own input | all reproduced; tables agree to 4e-9, energies to the Fortran log's five decimals, ghost verdicts of the selected channel all equal |

The first four rows are checks on cases that were not used while the
translation was written; during that work about 7000 generation runs and
14000 test reports were compared, with the same outcome. The remaining differences come from the last
bit of `exp`, `log` and `pow` in different math libraries: the atom's
convergence test sits at 1e-12, so the iteration count can change by one. The
Fortran program shows the same sensitivity to its own math library. The last
row can be repeated after a rebuild with `tools/crosscheck_fortran.py`, which
needs a built FHI98PP.

Things the port does differently on purpose:

- For Z > 80 the Fortran routine with the default radii writes outside its
  table and the program can crash. The port treats these elements as "no
  tabulated radius"; their templates give all radii explicitly.
- The Fortran test program crashes in its kinetic-energy test when a valence
  state of the pseudo-atom has no bound solution (17 % of the reference runs
  for Z up to 80) and then leaves an empty report. The port reports the ghost analysis in these
  cases and omits the kinetic-energy table.
- With the Hedin–Lundqvist functional (`iexc = 2`) the Fortran program prints
  an uninitialised number as correlation energy; the port prints the energy
  without it.

What a passed ghost test does not tell: transferability to other chemical
environments, convergence with grid spacing, and whether a semicore shell
should be in the valence. Check a potential on a small system you know before
relying on it.

## Other backends

`--backend fhi98pp` and `--backend atom` drive the external programs FHI98PP
and ATOM instead of the built-in port (`--fhi-root`, `--atom-executable`).
They exist for cross-checks and need those programs installed.

## Tests

```powershell
python -m unittest pp_generation.tests.test_backend_pyfhi98pp `
  pp_generation.tests.test_pyfhi98pp_atom pp_generation.tests.test_pyfhi98pp_psgen `
  pp_generation.tests.test_pyfhi98pp_pswatch pp_generation.tests.test_converters `
  pp_generation.tests.test_gui_logic pp_generation.tests.test_pp_generation
```

The reference files under `tests/data` were written by the Fortran program, so
the tests run without it.

## Credit

The generator in `pyfhi98pp` is a translation of FHI98PP, written by Martin
Fuchs at the Fritz-Haber-Institut der Max-Planck-Gesellschaft, and the element
templates for Z = 1–56 and 72–80 are FHI98PP's element database; the method,
the algorithms and the numerical choices are theirs. Cite
M. Fuchs and M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999) for
potentials made with this package, together with D. R. Hamann, Phys. Rev. B
**40**, 2980 (1989) or N. Troullier and J. L. Martins, Phys. Rev. B **43**,
1993 (1991) for the construction, and X. Gonze, R. Stumpf, M. Scheffler, Phys.
Rev. B **44**, 8503 (1991) for the ghost-state analysis. The full list, what
was translated and where the translation departs from the Fortran programs are
in [`pyfhi98pp/README.md`](pyfhi98pp/README.md).

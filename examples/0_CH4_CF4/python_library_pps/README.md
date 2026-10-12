# C 1s binding energies of CH4 and CF4 with the library pseudopotentials

The benchmark of `../python_pbe`, repeated with the pseudopotentials of
`src/pp_generation/NCPPs` and `src/pp_generation/CoreHole_NCPPs` instead of the
potentials of the paper, for PBE and for LDA (CA), and compared with
all-electron calculations. Geometries, grids and solver controls are those of
`../python_pbe`; only the pseudopotentials (and for CA the functional) differ.

## Result

Binding energy = reference-corrected all-electron total of the final state
(carbon potential with a 1s hole, one electron removed) minus that of the
initial state. In eV:

| | CH4 | CF4 | shift |
|---|---:|---:|---:|
| **PBE** | | | |
| library potentials | 297.098 | 307.340 | 10.242 |
| all-electron, spin-unpolarized hole | 297.019 | 307.290 | 10.272 |
| paper's potentials (`../python_pbe`) | 297.152 | 307.706 | 10.554 |
| ARES with the paper's potentials | 297.148 | 307.715 | 10.567 |
| all-electron, spin-polarized hole | 289.999 | 300.207 | 10.208 |
| **LDA (CA)** | | | |
| library potentials | 293.902 | 304.449 | 10.547 |
| all-electron, spin-unpolarized hole | 293.832 | 304.382 | 10.550 |
| all-electron, spin-polarized hole | 287.074 | 297.547 | 10.473 |
| experiment | 290.80 | 301.85 | 11.05 |

- The pseudopotential calculation works with spin-unpolarized atoms and
  molecules, so its all-electron counterpart is the calculation with occupation
  1 in the C 1s orbital of a spin-restricted molecule. The library potentials
  agree with it to 0.05–0.08 eV in the binding energies and to 0.03 eV (PBE)
  and 0.003 eV (CA) in the shift.
- Spin polarization of the core hole lowers the binding energies by 7.02 eV
  (CH4) and 7.08 eV (CF4) with PBE: the offset the papers describe, constant to
  0.06 eV between these two molecules.
- With the paper's potentials the shift is 0.28 eV larger than the all-electron
  PBE shift, and CF4 is 0.42 eV above the all-electron binding energy.

## Where the paper's potentials differ

The paper's potentials differ from the library ones in two ways: slightly
smaller cutoff radii, and `lmax` 1 with the p channel local instead of `lmax` 2
with the d channel local (`../paper_pp_inputs`). The four combinations, PBE,
CH4 / CF4 / shift in eV:

| cutoff radii | `lmax` 2, d local | `lmax` 1, p local |
|---|---|---|
| library (C 1.50; F 1.35; H 1.30) | 297.098 / 307.340 / 10.242 | 297.205 / 307.754 / 10.549 |
| paper (C s 1.47, p 1.50; F 1.28; H 1.13) | 297.045 / 307.332 / 10.287 | 297.152 / 307.706 / 10.554 |
| all-electron | 297.019 / 307.290 / 10.272 | |

The form decides: with `lmax` 1 and p local the binding energy is 0.11 eV
higher for CH4 and 0.37–0.41 eV higher for CF4, for either set of radii. The
radii change it by 0.01 to 0.05 eV. With p local, every angular momentum above
s feels the p potential of carbon and fluorine; with d local, p is treated by
its own projector.

What this does and does not show: it is one core level in two molecules, and
the all-electron reference is the one computed here, not the Q-Chem values of
the paper. For these two molecules, the PP-PBE shift of 10.57 eV owes about
0.3 eV of its closeness to experiment (11.05 eV) to the p-local form and not to
the functional. Whether that holds for the other molecules of the paper would
need the same comparison for them.

## Reproducing

```powershell
python examples\0_CH4_CF4\python_library_pps\setup.py        # writes the eight cases
```

then in each of `pbe\{CH4,CF4}\{IS,FS_1s}` and `ca\...`:

```powershell
..\..\..\..\..\..\.venv312\Scripts\python.exe `
  ..\..\..\..\..\..\src\parsec_python\main.py parsec.in `
  --pp-dir ..\..\pseudopotentials `
  --backend auto --symmetry auto --log parsec.out --no-archive
```

and finally `python examples\0_CH4_CF4\python_library_pps\analyze.py`, which
prints the first table. The kept logs are `parsec_python.out`. `setup.py` copies
the potentials from the two libraries and takes `Local_Component` and
`Atomic_Energy_Correction` of each species from the `parsec_atom_type.in` file
beside the potential; the libraries have to be built first
(`python -m pp_generation library --elements H C F`).

On the laptop these were first runs without cache, double precision, with the
Poisson solves on the GPU (`PARSEC_HARTREE_LINEAR_BACKEND=cupy`): 20–33 s per
CH4 case, 336–402 s per CF4 case (3.1 million grid points).

The two mixed combinations of the second table were generated with
`python -m pp_generation` (`--radius`, `--lmax`, `--local-channel`) and run the
same way; their inputs are not kept here.

## All-electron reference

`all_electron_reference.py` (PySCF 2.14): spin-free X2C, aug-cc-pwCVQZ on
carbon and aug-cc-pVQZ on hydrogen and fluorine, grid level 5, the geometries
of the `IS` inputs for both states. The unpolarized final state keeps
occupation 1 in the C 1s orbital, followed by overlap with the ground-state
orbital; the polarized one is a spin-unrestricted calculation with the maximum
overlap method. All calculations converged. With the triple-zeta sets the PBE
values are 297.046 / 307.315 (unpolarized) and 290.026 / 300.229 (polarized):
the binding energies change by 0.03 eV and the shift by 0.003 eV from triple to
quadruple zeta. For CA the functional is Slater exchange with Perdew–Zunger
correlation (`LDA_X,LDA_C_PZ`), the functional of `Correlation_Type: ca`.

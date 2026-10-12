# Core-hole norm-conserving pseudopotentials

One pseudopotential for every core shell of every element, generated from the
atom with one electron removed from that shell, for PBE and for LDA (CA). They
describe the core-excited atom in a calculation of a core-level binding energy
(Xu, Prendergast, Qian, J. Chem. Theory Comput. **18**, 5471 (2022); Liu et
al., J. Chem. Theory Comput. **20**, 6134 (2024)). Built with
`python -m pp_generation library`; this README describes the build of
12 October 2026.

| folder | functional | potentials | elements | elements with every core shell |
|---|---|---|---|---|
| `PBE/` | Perdew–Burke–Ernzerhof | 864 | 100 | 96 |
| `CA/` | LDA, Ceperley–Alder / Perdew–Zunger | 840 | 99 | 93 |

**Where the files are.** This branch, `pseudopotential`, holds all of them.
The default branch holds only H–Kr (128 per functional), to stay small;
`manifest.csv` lists all of them in either branch.

The core shells of an element are the ones its template keeps in the core:
1s for Li–Ne; 1s, 2s, 2p for Na–Ar; up to 15 shells (1s … 6p) for the
actinides. `manifest.csv` lists every shell with its status.

## Layout

```text
<XC>/<Element>/<Element>-<shell>/          e.g. PBE/Si/Si-2p/
    PARSEC/<Element>-<shell>_POTRE.DAT     read by the Python solver
    PARSEC/parsec_atom_type.in             the Atom_Type lines for parsec.in
    UPF/<prefix>.UPF
    FHI/<Element>-<shell>_FHIPP.DAT
    diagnostics/
manifest.csv
```

## Using one

Each potential belongs with the ground-state potential of the same element
and functional in `../NCPPs`: it has the same `lmax`, the same cutoff radii
and the same local channel, so the two differ only by the hole
(`matched_to_ground_state` in the manifest). Use the pair together.

1. Ground state: an ordinary calculation with the potentials of `../NCPPs`.
2. Core-excited state: give the excited atom its own species, for example

   ```text
   Atom_Type: Si-2p
   Element_Symbol: Si
   Pseudopotential_Format: martins_new
   Local_Component: d
   Atomic_Energy_Correction: -559.865198 Ry
   ```

   (the lines are in `parsec_atom_type.in`; the values differ per potential)
   and copy `Si-2p_POTRE.DAT` into the pseudopotential folder. The ionic
   charge of this potential is one higher than that of `Si`. With the
   photoelectron removed, add `1 e` to `Net_Charges`; with `Net_Charges`
   unchanged the hole is neutralized by one more valence electron.
3. Binding energy = difference of the two *reference-corrected all-electron
   totals* the solver prints. The pseudo total energies cannot be subtracted:
   the two potentials have different ionic charges.

A hole in a p, d or f shell is spherically averaged: `Si-2p` stands for the
average of the 2p1/2 and 2p3/2 holes. The atoms are spin-unpolarized and
scalar-relativistic, which leaves a constant offset of several eV per element
and shell in absolute binding energies; shifts between chemical environments
of the same element and shell are not affected by it.

## What was chosen

The rules are in `../README.md`. In this build:

- 1702 potentials are matched to their ground state. Two are not, because the
  ground-state form is not ghost-free for the hole, and have their own `lmax`
  and local channel (status `unmatched`): La 1s and Th 6p, both CA.
- The reference energies of unoccupied channels were shifted with the valence
  levels for 763 (PBE) and 737 (CA) potentials in 77 and 75 elements, by
  between −5.3 and −23.6 eV (`reference_shift_eV`).
- Not provided (status `failed`): 27 shells with PBE and 40 with CA, all in
  f-block elements. No ghost-free form: Ce (1s–4d; with CA also 5s), Es
  (1s, 2s, 2p; with CA all but 6p), No (1s–5d; with CA all), Fm 1s (CA).
  Cannot be constructed at any radius tried: Th 6s and 6p (PBE).
- No potentials at all for Lr, and none for Ho with CA, because there is no
  ground-state potential (see `../NCPPs/README.md`).
- Smallest ghost-free margins: Cr 1s 0.0019 Ha (PBE) and 0.0009 Ha (CA),
  V 1s 0.0030 and 0.0020 Ha; all others are 0.0044 Ha or more. A margin this
  small means the potential is close to having a ghost state; test Cr 1s and
  V 1s on a small system before use.

## Checks

- Every potential was regenerated with the Fortran FHI98PP from the input in
  its `diagnostics` folder: 1704 of 1704 reproduced; tables agree to 4e-9,
  total energies to the five decimals the Fortran log prints, and the ghost
  verdicts of the selected local channel agree in all cases. (For Z > 80 a
  copy of the Fortran program with one array bound corrected was used.)
- For every potential marked matched, the cutoff radii, `lmax` and local
  channel in the manifest equal those of the ground-state row.
- One molecular test, C 1s in CH4 and CF4 (`examples/0_CH4_CF4/python_library_pps`
  in the branch `one-shot-double-hybrid`): with these potentials and their
  ground-state partners the binding energies are within 0.05–0.08 eV of
  all-electron calculations with a spin-unpolarized core hole, and the
  CF4–CH4 shift within 0.03 eV (PBE) and 0.003 eV (CA).

What has not been checked: any other element, shell or molecule. The radii are
the generator's defaults, not the radii of the published calculations of the
group (`examples/0_CH4_CF4/paper_pp_inputs` in the branch
`one-shot-double-hybrid`), and deep core levels of heavy
elements are outside what a scalar-relativistic, spin-averaged atom describes
well.

Rebuild the library with `python -m pp_generation library`, or generate one
potential with `python -m pp_generation Si --core-hole 2p`.

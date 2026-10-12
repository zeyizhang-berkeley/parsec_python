# FHI98PP inputs of the published C, H and F pseudopotentials

The ARES benchmark in `../0_ARES_CH4_CF4` uses the pseudopotentials of Xu,
Prendergast, Qian, J. Chem. Theory Comput. **18**, 5471 (2022). They were made
with FHI98PP, but the UPF files do not record the inputs. The four files here
are those inputs, recovered from the UPF files, and they regenerate the
published potentials with the Python generator of this repository.

| file | potential | channels | cutoff radii (bohr) | local |
|---|---|---|---|---|
| `H.fhi98pp.in` | H | s | 1.13 | s |
| `C.fhi98pp.in` | C | s, p | s 1.47, p 1.50 | p |
| `C-1s.fhi98pp.in` | C with one 1s electron removed | s, p | s 1.47, p 1.50 | p |
| `F.fhi98pp.in` | F | s, p | 1.28, 1.28 | p |

All are PBE, Troullier–Martins, scalar-relativistic, without core correction.
The carbon potential and its core-hole partner have the same radii.

## Regenerating

From the repository root (`PYTHONPATH` set to `src`):

```powershell
python -m pp_generation C --input-file examples\0_CH4_CF4\paper_pp_inputs\C.fhi98pp.in `
  --local-channel 1 --prefix C --format upf --format parsec -o pp-output\paper\C
python -m pp_generation C --input-file examples\0_CH4_CF4\paper_pp_inputs\C-1s.fhi98pp.in `
  --local-channel 1 --prefix C-1s --format upf --format parsec -o pp-output\paper\C-1s
python -m pp_generation H --input-file examples\0_CH4_CF4\paper_pp_inputs\H.fhi98pp.in `
  --local-channel 0 --prefix H --format upf --format parsec -o pp-output\paper\H
python -m pp_generation F --input-file examples\0_CH4_CF4\paper_pp_inputs\F.fhi98pp.in `
  --local-channel 1 --prefix F --format upf --format parsec -o pp-output\paper\F
```

With `--input-file` the file is used as it is; the core hole of `C-1s` is the
occupation 1.00 of the 1s state in the file.

## Result

Compared with the UPF files of the benchmark
(`../0_ARES_CH4_CF4/test_CH4_CF4/pbe/*/*/*.UPF`):

| potential | all-electron / pseudo-atom energy (Ha) | header of the published UPF (Ry) | largest relative deviation of a numerical block |
|---|---|---|---|
| H | −0.458933 / −0.458596 | −0.91786 / −0.91720 | 3e-13 |
| C | −37.764780 / −5.363285 | −75.52956 / −10.72658 | 9e-12 |
| C-1s | −26.579090 / −8.844287 | −53.15818 / −17.68858 | 2e-11 |
| F | −99.744308 / −23.981793 | −199.48862 / −47.96358 | 3e-11 |

Every numerical block of the UPF (mesh, local potential, semilocal potentials,
projectors, D matrix, pseudo wave functions, atomic density) agrees. The
headers differ only in notation (`core_correction="false"` for `"F"`) and in
the energies, which the published files give as twice the five-decimal Hartree
value of the FHI98PP log.

## How the inputs were recovered

The radial mesh of a UPF converted from FHI98PP is the generator's mesh, and
the pseudo wave function of an occupied channel depends only on that channel's
cutoff radius. Generating the channel at every mesh point between 0.5 and 3.2
bohr and comparing with `PP_CHI` identifies the radius: for the carbon s
channel the largest difference is 3e-14 at the right mesh point and 5e-3 at
each of its neighbours. A radius
in the table is any two-decimal value that selects that mesh point (C s:
mesh point 1.4620, C p: 1.4982, H: 1.1298, F: 1.2748).

These radii are smaller than the FHI98PP defaults used for the library
potentials in `src/pp_generation/NCPPs` (H 1.30, C 1.50 for all channels,
F 1.35), and the published potentials have `lmax` 1 with the p channel local,
where the library has `lmax` 2 with the d channel local.
`../python_library_pps` shows what that changes in the C 1s binding energies.

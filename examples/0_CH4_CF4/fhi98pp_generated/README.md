# fhi98pp interface smoke test

These files are an independent generation test, not the pseudopotentials used
for the matched ARES reproduction under `../python_pbe`.  That calculation
uses ARES's exact supplied UPFs so PP-generation choices do not enter the code
comparison.

Both references were generated through `python -m pp_generation` using the
local FHI98PP carbon Troullier-Martins template, PBE, automatic local-channel
screening, the QE FHI-to-UPF converter, and the repository UPF-to-POTRE
converter.

| reference | ionic charge | reference electrons | selected local l | ghost-free | AE-PS correction (Ry) |
|---|---:|---:|---:|---:|---:|
| ordinary C | 4 | 4 | 2 (d) | yes | -64.80126 |
| C-1s full hole | 5 | 4 | 2 (d) | yes | -35.46520 |

The core-hole run automatically emits `C-1s_POTRE.DAT` and
`C-1s_FHIPP.DAT` aliases.  `*_generated.dat` is the log of FHI98PP.

## How the files were made

The `*_FHIPP.DAT` files are the tables of FHI98PP (M. Fuchs and
M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999)) for the inputs of the
same two potentials in the libraries,
`src/pp_generation/NCPPs/PBE/C/diagnostics/c_pbe_tm.ini` and
`src/pp_generation/CoreHole_NCPPs/PBE/C/C-1s/diagnostics/c_1s_hole_1_pbe_tm.ini`
(default radii, `lmax` 2): the Fortran program run on them writes these files
byte for byte.

The `*_POTRE.DAT` files are the cubic resampling (`src/tools/upf_to_parsec.py`)
of the library UPF files of those potentials, whose numerical blocks agree
with these FHI98PP tables to 1e-11:

```powershell
python src\tools\upf_to_parsec.py `
  src\pp_generation\NCPPs\PBE\C\UPF\c_pbe_tm.UPF `
  examples\0_CH4_CF4\fhi98pp_generated\C\C_POTRE.DAT
python src\tools\upf_to_parsec.py --allow-ionized-reference `
  src\pp_generation\CoreHole_NCPPs\PBE\C\C-1s\UPF\c_1s_hole_1_pbe_tm.UPF `
  examples\0_CH4_CF4\fhi98pp_generated\C-1s\C-1s_POTRE.DAT
```

They are the library POTRE files byte for byte, and `C-1s/C_POTRE.DAT` is a
copy of `C-1s/C-1s_POTRE.DAT`. Until 23 November 2026 they were the linear
resampling of the converter's first version (`--interpolation linear`), whose
values between the points of the UPF mesh rippled: the s and p eigenvalues of
the pseudo atom, with the screening potential of FHI98PP, were 0.16 to
0.25 mRy off those of the generator; the cubic files are within 1.9e-7 Ry.

These template defaults are only a functional interface/ghost-screening
test.  Production pseudopotentials still require cutoff convergence,
logarithmic-derivative and chemical-transferability validation.  In
particular, the automatically selected d-local channel differs from the
p-local supplied ARES benchmark PP and must not be substituted into only one
side of that comparison.

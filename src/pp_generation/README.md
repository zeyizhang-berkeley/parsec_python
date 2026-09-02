# Pseudopotential generation

`pp_generation` is a general command-line and Python interface for generating
and ghost-checking norm-conserving pseudopotentials. It is independent of the
DFT solver and delegates atomic generation to an installed FHI98PP or ATOM
toolchain.

The interface supports ordinary reference configurations and explicit
core-hole configurations such as `1s` or `2p`. It can request FHI, PARSEC,
UPF, PSP8, SIESTA, or CPW2000-family output when the selected backend and
installed converters support that format.

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pp_generation --help
```

Examples:

```powershell
# Ordinary PBE Troullier--Martins Ni potential
python -m pp_generation Ni --backend fhi98pp --xc pbe --scheme tm `
  --format fhi --format parsec --format upf -o generated\Ni

# Remove one electron from the Ni 2p shell
python -m pp_generation Ni --backend fhi98pp --xc pbe --scheme tm `
  --core-hole 2p --hole-charge 1.0 `
  --format fhi --format parsec --format upf -o generated\Ni-2p
```

Use `--input-file` for a backend-native expert input when the generic atomic
configuration is insufficient. `--local-channel` forces a particular
Kleinman--Bylander local channel; otherwise the package can scan available
choices and reject ghosted candidates. `--allow-ghosts` is intended only for
diagnostics.

Generated files are not scientifically accepted merely because the generator
finishes. Inspect the report and validate logarithmic derivatives,
eigenvalues, cutoff convergence, transferability, ghost states, XC
consistency, core-hole charge, and relativistic requirements before using a
potential in production. In particular, scalar FHI98PP output does not become
spin--orbit-resolved through file conversion.

`NCPPs/manifest.csv` records the local reviewed library when that generated
collection is present in a checkout. Generated bulk data and diagnostic work
files are local artifacts unless deliberately published separately.

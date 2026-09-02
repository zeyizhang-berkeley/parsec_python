# Periods 1–4 norm-conserving pseudopotentials

This collection contains FHI98PP norm-conserving pseudopotentials for H
through Kr in FHI, UPF v2, and PARSEC Martins-new formats. The baseline
settings are PBE exchange-correlation and the Troullier–Martins construction,
using the distributed FHI98PP element templates. Every available
Kleinman–Bylander local channel was checked using FHI98PP `pswatch`; the
published local channel is recorded in `manifest.csv` and in each
`diagnostics/*.report.json`.

## Results

- 36/36 final element directories have an accepted ghost-free representation.
- 32 elements use the distributed default template with only XC changed to
  the package default, PBE.
- H, K, Ga, and Ge failed the strict first default pass. Their complete
  original diagnostics are retained under `default_rejected/`.
- For those four elements, the reviewed input omits only artificial,
  unoccupied high-angular-momentum channels: H uses `lmax=0`; K, Ga, and Ge
  use `lmax=1`. Occupations, PBE, Troullier–Martins construction, NLCC choice,
  and remaining default channel parameters are unchanged. The reviewed
  inputs are stored under `reviewed_inputs/`.

The default H failure was indeterminate in its unused p/d channels. K had
indeterminate d channels and a ghosted d-local representation. Ga and Ge had
indeterminate d projectors and a ghosted p projector for d-local. None of
these rejected default products is published as `Element_FHIPP.DAT` in the
final element directories.

## Layout and which file PARSEC uses

Each element directory contains four subdirectories:

- `FHI/Element_FHIPP.DAT`: lossless FHI98PP source/archive table;
- `UPF/*_pbe_tm.UPF`: semilocal norm-conserving UPF v2 interchange file;
- `PARSEC/Element_POTRE.DAT`: Martins-new file read directly by the Python
  PARSEC implementation;
- `diagnostics/`: exact input, JSON report, all local-channel ghost reports,
  raw converter output, radial data, wavefunctions, and logarithmic
  derivatives.

The `.DAT` suffix alone does not identify a format. The Python solver supports
only the Martins-new `Element_POTRE.DAT` file. Point its `--pp-dir` option at a
directory containing the required POTRE files, and set `Local_Component` for
each species to the value in `manifest.csv`. `Correlation_Type` is `pb`.

The UPF-to-POTRE step resamples the FHI/UPF exponential radial grid onto
PARSEC's shifted-log grid. It preserves UPF nonlinear core correction as
`4*pi*r**2*rho_core`. H and He are all-local potentials with no KB projector.

`manifest.csv` is the collection-level index. A blank minimum margin for H or
He means there is no nonlocal projector; the local representation is
vacuously free of Kleinman–Bylander projector ghosts.

## Scope of validation

The spectral ghost check is necessary but not sufficient for production use.
Before treating this collection as a production-quality library, perform
logarithmic-derivative inspection, excited/ionized atomic transferability,
real-space grid convergence, and representative molecule/solid benchmarks.
In particular, transition-metal semicore and NLCC choices require application-
specific validation.

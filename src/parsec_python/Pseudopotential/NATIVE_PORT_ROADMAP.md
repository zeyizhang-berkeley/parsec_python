# Roadmap to a native Python pseudopotential generator

The current Python layer deliberately treats FHI98PP and ATOM as numerical
oracles. Calling those executables is a useful first release, but it is not a
native Python port. Replacing them safely should proceed by numerical kernel,
with reference arrays retained at every boundary.

## Stable public boundary

`GenerationRequest` separates:

```text
family -> construction scheme -> numerical backend -> output format
```

Today only family `ncpp` exists; `tm` and `hamann` are schemes; `fhi98pp` and
`atom` are backends. ONCV belongs under NCPP as another scheme/backend. PAW
and ultrasoft require new family-specific data models (augmentation charges,
multiple projectors, overlap operators), not extra flags on an NCPP object.

## Port phases

1. **Canonical radial data model**
   - logarithmic/quasi-logarithmic grids, units, orbital configuration,
     semilocal channels, reference functions, core/valence densities;
   - immutable provenance and serializers independent of solvers.
2. **All-electron radial atom**
   - Poisson/Hartree, CA-PZ LDA and PBE, radial Schrödinger shooting,
     scalar-relativistic equations, occupation handling, SCF mixing;
   - compare energies, eigenvalues, densities, nodes, and norms with both codes.
3. **NCPP construction**
   - Troullier--Martins polynomial solve and inversion first;
   - Hamann next, then ONCV as the modern multi-projector extension;
   - NLCC/pseudocore construction as a separately tested component.
4. **Separable representation and atomic validation**
   - KB projectors and normalization signs;
   - bound spectra, Gonze classification, energy-dependent logarithmic
     derivatives, Fourier/kinetic convergence;
   - scan/rank every local channel but retain the full evidence.
5. **Serialization**
   - direct FHIPP and Martins-new readers/writers first;
   - UPF v2, PSP8, PSF, and POTKB with explicit unit/grid contracts;
   - round-trip tests and downstream QE/ABINIT/PARSEC reader smoke tests.
6. **Transferability database**
   - neutral/excited/ionized atomic tests;
   - SSSP/PseudoDojo-style solid-state and cutoff grading;
   - separate ground, FCH, XCH, and fractional-hole records.

## Acceptance criteria for removing a legacy backend

Do not replace a reference kernel merely because final plots look similar.
For every checked configuration require agreed tolerances for:

- total/eigen energies and integrated electron counts;
- radial potential, wavefunction, density, and norm arrays on a common grid;
- cutoff matching derivatives and projector normalization;
- ghost classifications and logarithmic-derivative poles;
- all serialized physical arrays after unit conversion/resampling;
- downstream PARSEC energies at converged real-space spacing.

The checked Si 2p-hole and P 1s-hole fixtures are the first mandatory native
port targets. Ground-state Si must be run through both FHI98PP and ATOM as an
independent triangulation case; agreement with only one historical code is not
enough to distinguish a shared assumption from a porting error.

## Licensing boundary

`parsec_python` carries GPLv3. The ATOM tree states GPLv2, while the bundled
FHI98PP README refers to license terms supplied by its distribution source but
does not include an equally clear license file in this checkout. Calling
separate executables and comparing numerical output is different from copying
or translating their implementation. Before a line-by-line native port,
confirm the exact ATOM “v2 or later” status and obtain/verify FHI98PP's source
license. A clean-room implementation from published equations and black-box
regression data is the safer route when provenance is uncertain. This is a
project risk to resolve, not a scientific detail to hide.

# pyfhi98pp: FHI98PP in Python

This folder is a Python translation of **FHI98PP**, the pseudopotential
package of the Fritz-Haber-Institut der Max-Planck-Gesellschaft, Berlin,
written by Martin Fuchs (release of 13 October 1998). It is not an
independent implementation: the methods, the algorithms, the numerical
constants and the structure of the programs are those of FHI98PP, translated
routine by routine from Fortran. The element templates in
[`../elements`](../elements) for Z = 1–56 and 72–80 are the element database
distributed with FHI98PP (see [`../elements/sources.csv`](../elements/sources.csv)).

## Citing

FHI98PP asks that results obtained with its pseudopotentials give credit to
the following work, and the same holds for potentials made with this
translation:

1. M. Fuchs and M. Scheffler, *Ab initio pseudopotentials for electronic
   structure calculations of poly-atomic systems using density-functional
   theory*, Comput. Phys. Commun. **119**, 67 (1999).
2. D. R. Hamann, *Generalized norm-conserving pseudopotentials*, Phys. Rev. B
   **40**, 2980 (1989).
3. N. Troullier and J. L. Martins, *Efficient pseudopotentials for plane-wave
   calculations*, Phys. Rev. B **43**, 1993 (1991).
4. X. Gonze, R. Stumpf and M. Scheffler, *Analysis of separable potentials*,
   Phys. Rev. B **44**, 8503 (1991); X. Gonze, P. Käckell and M. Scheffler,
   Phys. Rev. B **41**, 12264 (1990).
5. L. Kleinman and D. M. Bylander, Phys. Rev. Lett. **48**, 1425 (1982).

The README of FHI98PP states: "Use, copying, and redistribution of this
software and its documentation is subject to license provisions as provided
by the source from which you obtained this package."

## What is translated

| FHI98PP | here |
|---|---|
| `fhipp.x` (`fhipp.f`, `ncpp.f`, `hamann.f`, `tromar*.f`, `dnlcc7*.f`, `defrtm.f`) and the script `psgen` | `psgen.py`, `ncpp.py`, `hamann.py`, `tromar.py`, `dnlcc.py`, `defrtm.py` |
| `pslp.x` (`pslp.f`, `psatom.f`, `ppcheck.f`, `derlkb.f`, `klbyii*.f`, `kinkon.f`, ...) and the script `pswatch` | `pswatch.py`, `psatom.py`, `ppcheck.py`, `klbyii.py`, `kinkon.py` |
| shared routines (`sratom_n.f`, `dftseq.f`, `vestat.f`, `vexcor.f` and the exchange-correlation files, `logmesh.f`, ...) | `atom.py`, `radial.py`, `xc.py`, `mesh.py`, `labels.py`, `constants.py` |
| LAPACK routines the programs call | `lapack.py` (`dgesv`), `dspevx.py` |

`fortran_io.py`, `fortran_list.py` and `libm.py` reproduce Fortran formatted
output and isolate the math-library functions, so that the files written here
can be compared with those of the Fortran programs. The module docstrings name
the Fortran file of every routine.

Entry points: `psgen.psgen(...)` generates a potential, `pswatch.pswatch(...)`
writes the test report, `pswatch.ghost_scan(...)` tests every local channel
for ghost states. Most users will not call these directly but go through
`python -m pp_generation` (see [`../README.md`](../README.md)).

## Differences from the Fortran programs

The translation follows the Fortran statement by statement and reproduces its
results (see "Validation" in [`../README.md`](../README.md)). It departs from
it in these places:

- `defrtm` returns "no tabulated radius" for Z > 80, where the Fortran routine
  writes outside its table.
- Where `pslp.x` crashes in its kinetic-energy test (a valence state without
  bound solution), the translation raises an exception after the ghost
  analysis is complete.
- With the Hedin–Lundqvist functional (`iexc = 2`) the Fortran program prints
  an uninitialised number as correlation energy; here it is left out of the
  energy.
- More than 20 states, or empty fields in the input header, stop with an
  error where the Fortran program would run on.
- The self-interaction branch (disabled in the Fortran source) and the xmgr
  graphics files are not translated.

"""Array dimensions, mesh and accuracy settings of FHI98PP.

Port of ``parameter.h`` and ``default.h``.  The values are part of the
numerical definition of the program (mesh, convergence thresholds, the
Hartree to eV factor used for printing) and must not be "improved".
"""

from __future__ import annotations

# --- parameter.h ---------------------------------------------------------
MX: int = 4000        # mx: radial mesh index (maximum number of mesh points)
MS: int = 20          # ms: states index (maximum number of states)
IE: int = 6           # ie: error output unit (stdout)
IOFHIPP: int = 23     # iofhipp: regular fhipp output unit
IOPSLP: int = 6       # iopslp: regular pslp output unit

# --- default.h -----------------------------------------------------------
RMIN: float = 0.00625   # rmin: nuclear charge times minimum radius on grid
RMAX: float = 80.0      # rmax: maximum radius of grid
AINC: float = 1.0247    # ainc: mesh increment r(i+1)/r(i)
RY2: float = 27.2116    # ry2: Hartree in eV as used by the program
EPSAE: float = 1.0e-12  # relative accuracy of all-electron eigenvalues
EPSPP: float = 1.0e-12  # relative accuracy of pseudo eigenvalues
ITMX: int = 100         # maximum number of self-consistency iterations

# --- fhipp.f -------------------------------------------------------------
SREV: str = "rev 1998"  # srev (character*10, printed with a10)
TSIC: bool = False      # tsic: self-interaction correction (compiled out)
# igrmap(iexc): 1 if the functional needs density gradients
IGRMAP: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# --- dftseq.f ------------------------------------------------------------
DFTSEQ_ITMAX: int = 200           # itmax: eigenvalue iterations
FSS_I: float = 137.0359896        # fss_i: inverse fine structure constant

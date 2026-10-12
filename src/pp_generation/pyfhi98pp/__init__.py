"""Python translation of the FHI98PP pseudopotential generator.

This package is derived from FHI98PP, release of 13 October 1998, written by
Martin Fuchs (Fritz-Haber-Institut der Max-Planck-Gesellschaft, Berlin).  It
translates the Fortran programs ``fhipp.x`` and ``pslp.x`` and their driver
scripts ``psgen`` and ``pswatch``; the methods, the algorithms and the
numerical constants are those of the original.  Please cite

    M. Fuchs and M. Scheffler, Comput. Phys. Commun. 119, 67 (1999)

for pseudopotentials made with it, together with the references given in
``README.md`` beside this file.

Runtime dependency: numpy only.  The numerics follow the Fortran statement by
statement; the module docstrings give the mapping to the source files.

The all-electron atomic core shared by ``fhipp.x`` and ``pslp.x``:

=================  ====================================================
``constants``      ``parameter.h``, ``default.h``
``fortran_io``     Fortran edit descriptors and list-directed input
``libm``           ``exp``/``log``/``pow`` of the platform math library
``labels``         ``labelmap.f``
``mesh``           ``logmesh.f``
``radial``         ``dftseq.f``, ``atoaux.f``, ``vestat.f``, ``moment.f``,
                   ``anderson.f``, ``darraux.f``
``xc``             ``vexcor.f``, ``ldaxc.f``, ``relxc.f``, ``ggarad.f``,
                   ``gga91_sr.f``, ``bpgcxc.f``, ``corlyp.f``, ``pbe.f``
``atom``           ``atomini.f``, ``sratom_n.f``
``psgen``          ``fhipp.f`` (all-electron stage), ``out38`` of
                   ``outwfct.f`` and the csh driver ``bin/Tools/psgen``
=================  ====================================================

The pseudopotential construction (the rest of ``fhipp.x``):

=================  ====================================================
``psgen``          ``fhipp.f`` (pseudopotential branch) and the files
                   assembled by the driver script (``.dat``, ``.cpi``)
``ncpp``           ``ncpp.f``: construction and unscreening
``tromar``         ``tromarLAPACK.f``: Troullier-Martins scheme
``hamann``         ``hamann.f``: Hamann scheme
``dnlcc``          ``dnlcc7LAPACK.f``: model core density
``defrtm``         ``defrtm.f``: default Troullier-Martins radii
``outwfct``        ``outcore``, ``outpot``, ``out39`` of ``outwfct.f``
``lapack``         ``dgesv`` with the arithmetic of the reference build
``fortran_list``   list-directed output (``write(unit,*)``)
=================  ====================================================

The test program ``pslp.x`` with its driver script ``pswatch``:

=================  ====================================================
``pswatch``        ``pslp.f`` and the csh driver ``bin/Tools/pswatch``;
                   entry points ``pswatch.pswatch`` (complete report),
                   ``ghost_check`` / ``ghost_scan`` (ghost test only)
``psatom``         ``psatom.f``: self-consistent pseudo-atom
``ppcheck``        ``ppcheck.f``, ``derlkb.f``, ``rcovalent.f``: ghost-state
                   analysis and logarithmic derivatives
``klbyii``         ``klbyiiLAPACK.f``, ``laguerre.f``, ``gaussq.f``,
                   ``gauss.h``, ``basis.h``: spectrum in a Laguerre basis
``dspevx``         the LAPACK eigenvalue routine called there
``kinkon``         ``kinkon.f``, ``dfbt_srt.f``, ``bessel.f``: kinetic-energy
                   convergence in momentum space
=================  ====================================================
"""

from __future__ import annotations

from . import atom, constants, fortran_io, labels, libm, mesh, psgen, radial, xc
from .atom import SratomResult, atomini, sratom, tfapot
from .errors import FortranStop
from .labels import labelmap, spp
from .mesh import logmesh
from .psgen import (
    FhiInput, FhippState, OptionalLine, PsgenResult, all_electron,
    apply_psgen_options, fhipp, format_fort19, format_fort37, out38,
    parse_ini, read_fort20,
)
from .radial import (
    DftseqResult, anderson, dextv, dftseq, dmelm, dspv, fmom, gltfmv, vestat,
)
from .xc import XcResult, vexcor

# Pseudopotential construction.  The routines ``ncpp.ncpp``,
# ``tromar.tromar``, ``hamann.hamann`` and ``defrtm.defrtm`` carry the names
# of their modules and are not re-exported either.
from . import defrtm, dnlcc, fortran_list, hamann, lapack, ncpp, outwfct, tromar  # noqa: E402
from .dnlcc import DnlccResult, dnlcc7  # noqa: E402
from .hamann import HamannResult  # noqa: E402
from .lapack import dgesv  # noqa: E402
from .ncpp import ExcitedState, NcppChannel, NcppResult  # noqa: E402
from .psgen import CpiData, format_cpi, format_dat, pseudopotentials  # noqa: E402
from .tromar import TromarResult  # noqa: E402

# The driver function ``psgen.psgen`` and the tag routine ``labels.labels``
# carry the names of their modules and are therefore not re-exported here.
__all__ = [
    "DftseqResult", "FhiInput", "FhippState", "FortranStop", "OptionalLine",
    "PsgenResult", "SratomResult", "XcResult", "all_electron", "anderson",
    "apply_psgen_options", "atom", "atomini", "constants", "dextv", "dftseq",
    "dmelm", "dspv", "fhipp", "fmom", "format_fort19", "format_fort37",
    "fortran_io", "gltfmv", "labelmap", "labels", "libm", "logmesh", "mesh",
    "out38", "parse_ini", "psgen", "radial", "read_fort20", "spp", "sratom",
    "tfapot", "vestat", "vexcor", "xc",
    # pseudopotential construction
    "CpiData", "DnlccResult", "ExcitedState", "HamannResult", "NcppChannel",
    "NcppResult", "TromarResult", "defrtm", "dgesv", "dnlcc", "dnlcc7",
    "format_cpi", "format_dat", "fortran_list", "hamann", "lapack", "ncpp",
    "outwfct", "pseudopotentials", "tromar",
]

# The test program pslp.x.  The driver ``pswatch.pswatch`` and the
# routines ``psatom.psatom``, ``ppcheck.ppcheck``, ``klbyii.klbyii`` and
# ``kinkon.kinkon`` carry the names of their modules and are not re-exported.
from . import dspevx, kinkon, klbyii, ppcheck, psatom, pswatch  # noqa: E402
from .kinkon import FortranCrash, KinkonResult  # noqa: E402
from .ppcheck import KbChannel, LogDerivative  # noqa: E402
from .pswatch import (  # noqa: E402
    GhostResult, PseudoAtom, PslpControl, PswatchResult, ghost_check, ghost_scan,
    pslp,
)

__all__ += [
    "FortranCrash", "GhostResult", "KbChannel", "KinkonResult", "LogDerivative",
    "PseudoAtom", "PslpControl", "PswatchResult", "dspevx", "ghost_check",
    "ghost_scan", "kinkon", "klbyii", "ppcheck", "psatom", "pslp", "pswatch",
]

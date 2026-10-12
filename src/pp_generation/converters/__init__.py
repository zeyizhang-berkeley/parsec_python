"""In-process pseudopotential format converters (standard library only).

The chain replaces the external programs of the FHI98PP workflow::

    FHI98PP table (.cpi)  --upf-->  UPF v2 (semilocal)  --potre-->  POTRE.DAT

``cpi``
    :class:`CpiTable` with :func:`read_cpi`/:func:`write_cpi` for the
    ``.cpi`` file (``Element_FHIPP.DAT``) and :func:`write_fhi` for its abinit
    ``.fhi`` form; :func:`read_cpi` reads both.
``upf``
    :func:`cpi_to_upf` and :func:`write_upf`, a port of ``fhi2upf_qe.x``.
``potre``
    :func:`upf_to_potre`, the implementation of ``src/tools/upf_to_parsec.py``.

:func:`convert_outputs` runs the whole chain for one generated potential::

    from pp_generation.converters import convert_outputs

    written = convert_outputs(
        "si_pbe_tm.cpi", "out", element="Si", xc="pbe", local_channel=2,
        valence=[(3, 0, 2.0), (3, 1, 2.0)], prefix="si_pbe_tm",
        ae_energy_hartree=-289.2, pseudo_energy_hartree=-3.76,
    )
    # {"fhi": out/Si_FHIPP.DAT, "upf": out/si_pbe_tm.UPF, "parsec": out/Si_POTRE.DAT}
"""

from .cpi import (
    AbinitHeader,
    CoreDensity,
    CpiChannel,
    CpiTable,
    format_cpi,
    format_fhi,
    parse_cpi,
    read_cpi,
    write_cpi,
    write_fhi,
)
from .outputs import XC_LABELS, XcLabels, convert_outputs, valence_channels, xc_labels
from .potre import ConversionError, infer_parsec_xc, upf_to_potre
from .upf import (
    DEFAULT_GENERATED,
    FHI2UPF_GENERATED,
    SemilocalUpf,
    cpi_to_upf,
    format_upf,
    write_upf,
)

__all__ = [
    "AbinitHeader", "ConversionError", "CoreDensity", "CpiChannel", "CpiTable",
    "DEFAULT_GENERATED", "FHI2UPF_GENERATED", "SemilocalUpf", "XC_LABELS",
    "XcLabels", "convert_outputs", "cpi_to_upf", "format_cpi", "format_fhi",
    "format_upf", "infer_parsec_xc", "parse_cpi", "read_cpi", "upf_to_potre",
    "valence_channels", "write_cpi", "write_fhi", "write_upf", "xc_labels",
]

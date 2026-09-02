from __future__ import annotations

import re
from pathlib import Path

from .errors import BackendError
from .parsers import energies


HEADER_RE = re.compile(r"<PP_HEADER\b.*?/>", re.DOTALL)


def _set_attribute(header: str, name: str, value: float) -> str:
    rendered = f'{name}="{value:.10f}"'
    pattern = re.compile(rf"\b{re.escape(name)}=\"[^\"]*\"")
    if pattern.search(header):
        return pattern.sub(rendered, header, count=1)
    return header[:-2].rstrip() + "\n" + rendered + "/>"


def annotate_upf_energies(dat_path: Path, raw_upf: Path, output_upf: Path) -> None:
    """Record FHI energies in UPF's Rydberg convention."""

    ae_hartree, pseudo_hartree = energies(dat_path)
    text = raw_upf.read_text(encoding="utf-8")
    match = HEADER_RE.search(text)
    if match is None:
        raise BackendError(f"{raw_upf} has no self-closing PP_HEADER")
    header = _set_attribute(match.group(0), "total_psenergy", 2.0 * pseudo_hartree)
    # This is useful provenance used by the existing workflow, but is an
    # extension rather than a standard UPF v2 attribute.
    header = _set_attribute(header, "total_aeenergy", 2.0 * ae_hartree)
    output_upf.write_text(text[: match.start()] + header + text[match.end() :], encoding="utf-8")

"""Element input templates shipped with the package.

``elements/TM`` and ``elements/Ham`` hold the FHI98PP element database
(Troullier--Martins and Hamann defaults) under portable file names
(``14-Si_tm.ini`` instead of ``14-Si:tm.ini``), plus templates added by this
package for elements the FHI98PP distribution does not cover.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import ConfigurationError
from .fhi_input import L_LABELS, ParsedInput, parse
from .models import ConstructionScheme

ELEMENTS_DIR = Path(__file__).resolve().parent / "elements"

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()

_NAME_RE = re.compile(r"(\d+)-([A-Z][a-z]?)(?:-([0-9a-z]+))?_(tm|h)\.ini")


def atomic_number(element: str) -> int:
    try:
        return SYMBOLS.index(element) + 1
    except ValueError:
        raise ConfigurationError(f"no atomic number known for element {element!r}") from None


def _directory(scheme: ConstructionScheme) -> tuple[Path, str]:
    if scheme is ConstructionScheme.TROULLIER_MARTINS:
        return ELEMENTS_DIR / "TM", "tm"
    return ELEMENTS_DIR / "Ham", "h"


def template_path(
    element: str, scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS,
    variant: str | None = None,
) -> Path:
    """Path of the built-in template; ``variant`` selects e.g. ``3d`` for Ga."""

    directory, suffix = _directory(scheme)
    z = atomic_number(element)
    middle = f"-{variant}" if variant else ""
    path = directory / f"{z:02d}-{element}{middle}_{suffix}.ini"
    if not path.is_file():
        raise ConfigurationError(
            f"no built-in {scheme.value} template for {element}{middle}; provide --input-file"
        )
    return path


def template_text(
    element: str, scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS,
    variant: str | None = None,
) -> str:
    return template_path(element, scheme, variant).read_text(encoding="utf-8")


def available_elements(
    scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS,
) -> list[str]:
    """Elements with a default (variant-free) built-in template, by atomic number."""

    directory, suffix = _directory(scheme)
    found = set()
    for path in directory.glob(f"*_{suffix}.ini"):
        match = _NAME_RE.fullmatch(path.name)
        if match and match.group(3) is None:
            found.add(match.group(2))
    return [symbol for symbol in SYMBOLS if symbol in found]


def core_shells(parsed: ParsedInput) -> list[str]:
    """Labels of the core shells of a parsed input, e.g. ``['1s', '2s', '2p']``."""

    return [f"{n}{L_LABELS[ell]}" for n, ell, _ in parsed.orbitals[: parsed.ncore]]


def element_core_shells(
    element: str, scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS,
) -> list[str]:
    return core_shells(parse(template_text(element, scheme)))

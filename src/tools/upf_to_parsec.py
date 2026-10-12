#!/usr/bin/env python3
"""Convert a semilocal norm-conserving UPF v2 file to PARSEC POTRE.DAT.

Command-line front end of ``pp_generation.converters.potre``, which holds the
implementation and documents the conversion.  Usage is unchanged::

    python src/tools/upf_to_parsec.py [--xc-code CODE] [--allow-ionized-reference]
        [--force] [--grid-refinement {1,2}] input.UPF output_POTRE.DAT

The implementation is loaded by file path, so this script needs only the
standard library and runs under a bare ``python3`` without the rest of the
``pp_generation`` package being importable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_NAME = "pp_generation.converters.potre"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "pp_generation" / "converters" / "potre.py"
)


def _load_implementation():
    """Return the converter module without importing its parent packages."""

    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the converter from {_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_MODULE_NAME]
            raise
    return module


_implementation = _load_implementation()
# Keep ``import upf_to_parsec`` working as it did when the code lived here.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not name.startswith("_")
    }
)
main = _implementation.main


if __name__ == "__main__":
    raise SystemExit(main())

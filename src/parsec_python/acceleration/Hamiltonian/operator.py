"""Public matrix-free Hamiltonian type for accelerated backends."""

from ..backends.base import BoundHamiltonian


class AcceleratedKohnShamHamiltonian(BoundHamiltonian):
    """Named façade for a local potential bound to a cached backend."""


__all__ = ["AcceleratedKohnShamHamiltonian"]

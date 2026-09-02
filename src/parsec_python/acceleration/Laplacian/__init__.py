"""Lazy, exactly keyed finite-difference construction for accelerated runs."""

from .deferred import (
    DeferredNativeNegativeLaplacian,
    materialize_negative_laplacian,
)

__all__ = [
    "DeferredNativeNegativeLaplacian",
    "materialize_negative_laplacian",
]

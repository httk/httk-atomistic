"""
A view presenting any sites backend as a Sites object (the class representation).
"""

import fractions
from functools import cached_property
from typing import Any, Self

from httk.core import FracVector, unwrap

from httk.atomistic.models._vector_guards import to_fracvector, to_precision
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view_base import SitesViewBase


class SitesView(SitesViewBase, Sites):
    """
    A view presenting an underlying sites backend as a ``Sites`` object.

    This view is a genuine ``Sites``, so it can be passed anywhere a Sites is accepted.
    Its state is built lazily on first access from the backend.
    """

    _backend: SitesBackend

    def __new__(cls, obj: SitesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: SitesLike, **hints: Any) -> None:
        pass

    # Validate then assign: failed fills leave no partial presentation state, and fills must not
    # read shadowed attributes or they recurse.
    def _fill_reduced_coords(self) -> None:
        coords = to_fracvector(self._backend.reduced_coords)
        if coords.dim != () and not (len(coords.dim) == 2 and coords.dim[1] == 3):
            raise ValueError("Sites reduced_coords must be an Nx3 vector-like")
        object.__setattr__(self, "_reduced_coords", coords)

    def _fill_precision(self) -> None:
        object.__setattr__(self, "_precision", to_precision(self._backend.precision))

    @cached_property
    def _reduced_coords(self) -> FracVector:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_reduced_coords()
        return self.__dict__["_reduced_coords"]

    @cached_property
    def _precision(self) -> fractions.Fraction | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_precision()
        return self.__dict__["_precision"]

    def unwrap(self) -> Any:
        return unwrap(self._backend)

    def unview(self) -> Sites:
        # The folded design makes a genuine Sites backend exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is Sites:
            return backend
        return Sites(self._reduced_coords, self._precision)

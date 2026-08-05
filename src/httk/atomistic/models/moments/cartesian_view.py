"""Cartesian class view for site moments."""

import fractions
from functools import cached_property
from typing import Any, Self

from httk.core import SurdVector

from httk.atomistic.models._vector_guards import to_precision, to_surdvector
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.like import SiteMomentsLike
from httk.atomistic.models.moments.view_base import SiteMomentsViewBase


class CartesianSiteMomentsView(SiteMomentsViewBase, CartesianSiteMoments):
    """A lazy Cartesian site-moments presentation of any site-moments backend."""

    _backend: SiteMomentsBackend

    def __new__(cls, obj: SiteMomentsLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: SiteMomentsLike, **hints: Any) -> None:
        pass

    def _fill_cartesian_moments(self) -> None:
        moments = to_surdvector(self._backend.cartesian_moments)
        if len(moments.dim) != 2 or moments.dim[1] != 3:
            raise ValueError("CartesianSiteMoments moments must be an Nx3 vector-like")
        object.__setattr__(self, "_cartesian_moments", moments)

    def _fill_precision(self) -> None:
        object.__setattr__(self, "_precision", to_precision(self._backend.precision))

    @cached_property
    def _cartesian_moments(self) -> SurdVector:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_cartesian_moments()
        return self.__dict__["_cartesian_moments"]

    @cached_property
    def _precision(self) -> fractions.Fraction | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_precision()
        return self.__dict__["_precision"]

    def unview(self) -> CartesianSiteMoments:
        # A genuine CartesianSiteMoments backend is exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is CartesianSiteMoments:
            return backend
        return CartesianSiteMoments(self._cartesian_moments, self._precision)

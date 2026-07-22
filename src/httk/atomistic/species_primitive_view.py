"""
A view presenting any species backend as an OPTIMADE species dict.
"""

from typing import Any, Self

from httk.core import unwrap

from .species_backend import SpeciesBackend
from .species_like import SpeciesLike
from .species_view import SpeciesView


class SpeciesPrimitiveView(SpeciesView, dict):
    """
    A view presenting an underlying species backend as an OPTIMADE species dict.

    This view is a genuine ``dict`` carrying the OPTIMADE ``species`` fields (optional
    fields that are ``None`` are omitted; list-valued fields are plain lists). Unlike the
    immutable-subclass views, a dict is mutable, so this view is a detached copy: mutating
    it does not affect the underlying backend.
    """

    _backend: SpeciesBackend

    def __new__(cls, obj: SpeciesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        payload: dict[str, Any] = {
            "name": backend.name,
            "chemical_symbols": list(backend.chemical_symbols),
            "concentration": list(backend.concentration),
        }
        if backend.mass is not None:
            payload["mass"] = list(backend.mass)
        if backend.original_name is not None:
            payload["original_name"] = backend.original_name
        if backend.attached is not None:
            payload["attached"] = list(backend.attached)
        if backend.nattached is not None:
            payload["nattached"] = list(backend.nattached)
        instance = super().__new__(cls)
        # dict is mutable, so its contents are initialized here in __new__ (keeping __init__ a no-op),
        # so that rewrapping an existing view via cls(view) does not re-initialize it.
        dict.__init__(instance)
        instance.update(payload)
        instance._backend = backend
        return instance

    def __init__(self, obj: SpeciesLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)

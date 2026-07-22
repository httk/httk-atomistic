"""
A view presenting any structure backend as a Structure (the Simple representation).
"""

from typing import Any, Self

from httk.core import unwrap

from .structure import Structure
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_view import StructureView


class StructureSimpleView(StructureView, Structure):
    """
    A view presenting an underlying structure backend as a ``Structure``.

    This view is a genuine ``Structure``, so it can be passed anywhere a Structure
    is accepted. Its quartet is built eagerly from the backend on construction.
    """

    _backend: StructureBackend

    def __new__(cls, obj: StructureLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # Structure is mutable, so its state is initialized here in __new__ (keeping __init__ a no-op),
        # so that rewrapping an existing view via cls(view) does not re-initialize it.
        Structure.__init__(instance, backend.cell, backend.sites, backend.species, backend.species_at_sites)
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)

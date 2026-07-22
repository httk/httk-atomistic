"""
A view presenting any species backend as a Species (the class representation).
"""

from typing import Any, Self

from httk.core import unwrap

from .species import Species
from .species_backend import SpeciesBackend
from .species_like import SpeciesLike
from .species_view import SpeciesView


class SpeciesClassView(SpeciesView, Species):
    """
    A view presenting an underlying species backend as a ``Species``.

    This view is a genuine frozen ``Species``, so it can be passed anywhere a Species is
    accepted. Its fields are built eagerly from the backend on construction, with full
    ``Species`` validation applied at that point.
    """

    _backend: SpeciesBackend

    def __new__(cls, obj: SpeciesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # Species is a frozen dataclass whose generated __init__ assigns via object.__setattr__,
        # so its state is initialized here in __new__ (keeping __init__ a no-op); this also means
        # rewrapping an existing view via cls(view) does not re-initialize it.
        Species.__init__(
            instance,
            name=backend.name,
            chemical_symbols=backend.chemical_symbols,
            concentration=backend.concentration,
            mass=backend.mass,
            original_name=backend.original_name,
            attached=backend.attached,
            nattached=backend.nattached,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: SpeciesLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)

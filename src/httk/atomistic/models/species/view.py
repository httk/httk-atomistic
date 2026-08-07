"""
A view presenting any species backend as a Species (the class representation).
"""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.species.backend import SpeciesBackend
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view_base import SpeciesViewBase


class SpeciesView(SpeciesViewBase, Species):
    r"""
    A view presenting an underlying species backend as a ``Species``.

    This view is a genuine frozen ``Species``, so it can be passed anywhere a Species is
    accepted. Its fields are built eagerly from the backend on construction, with full
    ``Species`` validation applied at that point.

    :param obj: The species-like object to present.
    :param \**hints: Backend-selection hints.
    """

    _backend: SpeciesBackend

    def __new__(cls, obj: SpeciesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = Species.create(obj) if isinstance(obj, (bool, str, int)) else cls._prepare_backend(obj, hints)
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
            concentration_precision=backend.concentration_precision,
            charges=backend.charges,
            spins=backend.spins,
            labels=backend.labels,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: SpeciesLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Species:
        """Return this presentation as standalone species.

        :return: The exact species representation.
        """
        # The folded design makes a genuine Species backend exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is Species:
            return backend
        return Species(
            name=self.name,
            chemical_symbols=self.chemical_symbols,
            concentration=self.concentration,
            mass=self.mass,
            original_name=self.original_name,
            attached=self.attached,
            nattached=self.nattached,
            concentration_precision=self.concentration_precision,
            charges=self.charges,
            spins=self.spins,
            labels=self.labels,
        )

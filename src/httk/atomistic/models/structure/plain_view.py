"""
A view presenting any structure backend as an spglib-like (lattice, positions, numbers) triple.
"""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic._atomic_projection import require_bare_atomic_projection
from httk.atomistic.elements import atomic_number
from httk.atomistic.models._vector_guards import to_float_tuples
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.view import StructureView


class PlainStructureView(StructureView, tuple):
    """
    A view presenting an underlying structure backend as a primitive triple.

    This view is a genuine ``(lattice, positions, numbers)`` tuple, built eagerly
    and immutable. Because the primitive representation carries only bare atomic
    numbers, every site's species must be a single, unattached chemical element
    (see ``Species.is_single_element``); otherwise a TypeError is raised.

    :param obj: The structure backend or source to present.
    :param \\*\\*hints: Backend-selection hints passed to structure coercion.
    """

    _backend: StructureBackend

    def __new__(cls, obj: StructureLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        require_bare_atomic_projection(backend, "a primitive structure")
        species_by_name = {species.name: species for species in backend.species}
        numbers: list[int] = []
        for name in backend.species_at_sites:
            species = species_by_name[name]
            if not species.is_single_element:
                raise TypeError(
                    "This structure cannot be represented as a primitive structure "
                    f"(species {name!r} is not a single, unattached chemical element)"
                )
            numbers.append(atomic_number(species.chemical_symbols[0]))
        payload = (
            to_float_tuples(backend.cell.basis),
            to_float_tuples(backend.sites.reduced_coords),
            tuple(numbers),
        )
        instance = super().__new__(cls, payload)
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        super().__init__()

    def unwrap(self) -> Any:
        """Return the raw value wrapped by the backend.

        :return: The original source value.
        """
        return unwrap(self._backend)

    def unview(self) -> tuple:
        """Return this presentation as an ordinary primitive triple.

        :return: The lattice, positions, and atomic numbers.
        """
        # The view IS its presentation (lattice, positions, numbers) triple; shed to a plain tuple.
        return tuple(self)

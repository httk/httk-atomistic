"""ASE ``Atoms`` view for the :mod:`httk.atomistic` structure family.

ASE is an optional dependency. This module intentionally imports it unconditionally,
so the package-level exports can guard the import while documentation tools can see
the public :class:`ASEAtomsView` definition.
"""

from typing import TYPE_CHECKING, Any, Self

import ase
from httk.core import unwrap

from httk.atomistic._atomic_projection import require_bare_atomic_projection
from httk.atomistic.elements import atomic_number
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView

if TYPE_CHECKING:
    from httk.atomistic.models.structure.like import StructureLike


class ASEAtomsView(StructureView, ase.Atoms):
    """Present a :data:`~httk.atomistic.models.structure.like.StructureLike` as ASE ``Atoms``.

    The conversion carries the structure quartet: cell, reduced positions, atomic
    numbers, and periodicity. ASE must be installed to construct this view. Mixed or
    attached species cannot be represented by ASE atomic numbers and raise
    :class:`TypeError`.
    """

    _backend: StructureBackend

    def __new__(cls, obj: "StructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        require_bare_atomic_projection(backend, "ASE Atoms")
        instance = super().__new__(cls)
        species_by_name = {species.name: species for species in backend.species}
        numbers: list[int] = []
        for name in backend.species_at_sites:
            species = species_by_name[name]
            if not species.is_single_element:
                raise TypeError(
                    "This structure cannot be represented as ASE Atoms "
                    f"(species {name!r} is not a single, unattached chemical element)"
                )
            numbers.append(atomic_number(species.chemical_symbols[0]))
        ase.Atoms.__init__(
            instance,
            cell=backend.cell.basis.to_floats(),
            scaled_positions=backend.sites.reduced_coords.to_floats(),
            numbers=numbers,
            pbc=backend.cell.periodicity,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: "StructureLike", **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        """Return the raw object represented by the underlying structure backend."""
        return unwrap(self._backend)


__all__ = ["ASEAtomsView"]

"""Backend adapting an ordinary structure to an anonymous unit cell."""

from collections import Counter
from functools import cached_property
from typing import Any

from httk.core import unwrap

from httk.atomistic.models.prototype.anonymize import canonical_dummy_assignment, dummy_species, require_anonymizable
from httk.atomistic.models.prototype.anonymous import AnonymousStructure
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structure.view import StructureView


class AnonymizedStructure(AnonymousStructureBackend):
    r"""Project an ordinary structure lazily to anonymous species.

    :param obj: The ordinary structure to anonymize.
    :param \*\*hints: Backend-selection hints.
    """

    _structure: StructureBackend
    kind = "structure"

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "structure") != "structure":
            return None
        if isinstance(obj, AnonymousStructureBackend):
            return None
        if isinstance(obj, StructureView):
            backend = obj._backend
        elif isinstance(obj, StructureBackend):
            backend = obj
        else:
            try:
                backend = StructureBackend.create(obj)
            except TypeError as exc:
                # Only the backend factory's own no-match error means this probe should fall through.
                if str(exc) == f"Cannot represent {type(obj)} as StructureBackend":
                    return None
                raise
        require_anonymizable(UnitcellStructureView(backend))
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        if isinstance(obj, StructureView):
            self._structure = obj._backend
        elif isinstance(obj, StructureBackend):
            self._structure = obj
        else:
            self._structure = StructureBackend.create(obj)

    @cached_property
    def _derived(self) -> AnonymousStructure:
        view = UnitcellStructureView(self._structure)
        species_by_name = {species.name: species for species in view.species}
        counts: Counter[str] = Counter()
        element_by_name: dict[str, str] = {}
        for name in view.species_at_sites:
            species = species_by_name[name]
            element = species.chemical_symbols[0]
            counts[element] += 1
            element_by_name[name] = element
        assignment = canonical_dummy_assignment(tuple(counts.items()))
        mapped_species = tuple(dummy_species(label) for label in assignment.values())
        mapped_sites = tuple(assignment[element_by_name[name]] for name in view.species_at_sites)
        return AnonymousStructure(view.cell, view.sites, mapped_species, mapped_sites)

    @property
    def cell(self):
        """Return the source cell."""
        return self._derived.cell

    @property
    def sites(self):
        """Return the source reduced sites."""
        return self._derived.sites

    @property
    def species(self):
        """Return the canonical dummy species."""
        return self._derived.species

    @property
    def species_at_sites(self):
        """Return dummy species names in site order."""
        return self._derived.species_at_sites

    @property
    def coordinate_precision(self):
        """Return the source coordinate precision."""
        return self._derived.coordinate_precision

    @property
    def basis_precision(self):
        """Return the source basis precision."""
        return self._derived.basis_precision

    def unwrap(self) -> Any:
        """Return the original ordinary structure."""
        return unwrap(self._structure)

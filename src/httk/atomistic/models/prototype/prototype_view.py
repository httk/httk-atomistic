"""Eager prototype-recognition view."""

from collections.abc import Callable
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.prototype.anonymize import canonical_dummy_assignment, dummy_species
from httk.atomistic.models.prototype.anonymized import AnonymizedStructure
from httk.atomistic.models.prototype.anonymous import AnonymousStructure
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import WyckoffSite
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.symmetry.standardization import conventional_cell


def _relabel_sites(
    sites: tuple[WyckoffSite, ...],
    multiplicities: tuple[int, ...],
    key_for_species: Callable[[str], str],
) -> tuple[tuple[WyckoffSite, ...], tuple[Species, ...]]:
    amounts: dict[str, int] = {}
    for site, multiplicity in zip(sites, multiplicities):
        key = key_for_species(site.species)
        amounts[key] = amounts.get(key, 0) + multiplicity
    assignment = canonical_dummy_assignment(tuple((key, value) for key, value in amounts.items()))
    mapped_sites = tuple(
        WyckoffSite(site.wyckoff, site.free_params, assignment[key_for_species(site.species)]) for site in sites
    )
    mapped_species = tuple(dummy_species(label) for label in assignment.values())
    return mapped_sites, mapped_species


class PrototypeView(AnonymousStructureViewBase, Prototype):
    """An eager standard-setting prototype view recognized from a structure."""

    _backend: AnonymousStructureBackend

    def __new__(
        cls,
        obj: Any,
        *,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if isinstance(obj, cls):
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("PrototypeView rewrapping does not accept recognition arguments")
            return obj
        forbidden = {name for name in ("setting", "standard", "transform") if name in hints}
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(
                f"PrototypeView does not accept {names}=; use PrototypeView(ASUStructureView(source, {names}=...))"
            )
        backend = cls._prepare_backend(obj, hints)
        if isinstance(backend, Prototype):
            if tolerance is not None or limit_denominator is not None:
                raise ValueError("PrototypeView tolerance and limit_denominator cannot be used with a Prototype")
            instance = super().__new__(cls)
            Prototype.__init__(
                instance,
                backend.cell,
                backend.spacegroup,
                backend.wyckoff_sites,
                backend.species,
                backend.coordinate_precision,
            )
            instance._backend = backend
            return instance

        anonymous_source = isinstance(backend, AnonymousStructure)
        source: StructureBackend
        if anonymous_source:
            source = UnitcellStructure(
                backend.cell,
                backend.sites,
                backend.species,
                backend.species_at_sites,
            )
            key_for_species = lambda name: name
        elif isinstance(backend, AnonymizedStructure):
            source = backend._structure
            real_species = {species.name: species.chemical_symbols[0] for species in source.species}
            key_for_species = lambda name: real_species[name]
        else:
            raise TypeError(f"Cannot recognize {type(backend).__name__} as a prototype source")

        result = conventional_cell(source, tolerance=tolerance, limit_denominator=limit_denominator)
        mapped_sites, mapped_species = _relabel_sites(
            result.asu.wyckoff_sites,
            result.asu.multiplicities(),
            key_for_species,
        )
        instance = super().__new__(cls)
        Prototype.__init__(
            instance,
            result.asu.cell,
            result.asu.spacegroup,
            mapped_sites,
            mapped_species,
            result.asu.coordinate_precision,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)

    def unview(self) -> Prototype:
        if type(self._backend) is Prototype:
            return self._backend
        return Prototype(
            self.cell,
            self.spacegroup,
            self.wyckoff_sites,
            self.species,
            self.coordinate_precision,
        )

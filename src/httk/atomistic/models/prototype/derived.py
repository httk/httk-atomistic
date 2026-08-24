"""Lazy adapters from assigned structures to anonymous prototypes."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.notation import canonical_label_map
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate


def _prototype_to_structure(prototype):
    """Return the exact dummy-species structure represented by a prototype representative."""
    structure = prototype._domain
    if isinstance(structure, ASUStructure):
        return structure
    return ASUStructure(
        structure.cell,
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
    )


def _anonymous_template_from_structure(structure: FundamentalDomainStructure) -> FundamentalDomainTemplate:
    by_name: dict[str, list[str]] = {}
    for site in structure.wyckoff_sites:
        by_name.setdefault(site.species, []).append(site.wyckoff)
    labels = canonical_label_map({name: tuple(sorted(letters)) for name, letters in by_name.items()})
    species = tuple(Species(label, ("X",), (1,), labels=(label,)) for label in labels.values())
    sites = tuple(
        WyckoffSite(site.wyckoff, site.free_params, labels[site.species], representative=site.representative)
        for site in structure.wyckoff_sites
    )
    return FundamentalDomainTemplate(
        structure.cell, structure.spacegroup, sites, species, structure.coordinate_precision
    )


class DerivedPrototype(PrototypeBackend):
    """Erase assigned Protostructure occupations to anonymous labels lazily.

    A base Protostructure erases to a base Prototype. An explicit refinement carried by the
    source is preserved: the discriminator verbatim, and the representative by the exact
    conversion of its :class:`~httk.atomistic.FundamentalDomainStructure` to the anonymous
    :class:`~httk.atomistic.FundamentalDomainTemplate` that a Prototype representative uses.
    """

    kind = "prototype"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        if hints and hints.get("kind", "prototype") != "prototype":
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return cls(obj, **hints)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        if not isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            raise TypeError("DerivedPrototype requires a Protostructure source")
        self._source = obj

    @cached_property
    def _derived(self) -> Prototype:
        source = self._source
        value = source._backend if isinstance(source, ProtostructureViewBase) else source
        by_name: dict[str, list[str]] = {}
        for occupation in value.occupations:
            by_name.setdefault(occupation.species.name, []).append(occupation.wyckoff)
        relabel = canonical_label_map({name: tuple(sorted(letters)) for name, letters in by_name.items()})
        occupations = [PrototypeOccupation(o.wyckoff, relabel[o.species.name]) for o in value.occupations]
        representative = value.representative
        anon_rep = None if representative is None else _anonymous_template_from_structure(representative)
        return Prototype(value.spacegroup, occupations, representative=anon_rep, discriminator=value.discriminator)

    def resolve(self) -> Prototype:
        return self._derived

    @property
    def spacegroup(self):
        return self._derived.spacegroup

    @property
    def occupations(self):
        return self._derived.occupations

    @property
    def representative(self):
        return self._derived.representative

    @property
    def discriminator(self):
        return self._derived.discriminator

    def unwrap(self) -> Any:
        return unwrap(self._source)

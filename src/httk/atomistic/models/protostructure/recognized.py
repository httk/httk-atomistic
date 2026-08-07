"""The structure-to-protostructure recognition adapter."""

from functools import cached_property
from typing import Any

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.symmetry.recognition import recognize_asu


class RecognizedProtostructure(ProtostructureBackend):
    """A lazy protostructure projection retaining an ordinary structure source."""

    kind = "structure"
    _structure: StructureBackend

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if isinstance(obj, (AnonymousStructureBackend, AnonymousStructureViewBase)):
            return None
        if isinstance(obj, (ChemicalFormulaBackend, ChemicalFormulaViewBase)):
            return None
        if isinstance(obj, StructureView):
            backend = obj._backend
        elif isinstance(obj, StructureBackend):
            backend = obj
        else:
            try:
                backend = StructureBackend.create(obj)
            except TypeError:
                return None
        for species in backend.species:
            if "X" in species.chemical_symbols or "X" in (species.attached or ()):
                raise ValueError(
                    f"Protostructure cannot represent structure species {species.name!r} with unknown symbol 'X'"
                )
        if getattr(backend, "assemblies", None) is not None:
            raise ValueError("Protostructure cannot represent assemblies")
        if getattr(backend, "chemical_composition", None) is not None:
            raise ValueError("Protostructure cannot represent chemical_composition")
        if isinstance(backend, FundamentalDomainStructure):
            has_site_moments = any(site.moment is not None for site in backend.wyckoff_sites)
        else:
            has_site_moments = getattr(backend, "site_moments", None) is not None
        if has_site_moments:
            raise ValueError("Protostructure cannot represent site_moments")
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        if isinstance(obj, StructureView):
            self._structure = obj._backend
        elif isinstance(obj, StructureBackend):
            self._structure = obj
        else:
            self._structure = StructureBackend.create(obj)

    @cached_property
    def _derived(self) -> Protostructure:
        asu = getattr(self._structure, "asu", None)
        if not isinstance(asu, FundamentalDomainStructure):
            asu = recognize_asu(self._structure)
        species_by_name = {species.name: species for species in asu.species}
        occupations = tuple((site.wyckoff, species_by_name[site.species]) for site in asu.wyckoff_sites)
        return Protostructure(asu.spacegroup, occupations)

    @property
    def spacegroup(self):
        return self._derived.spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        return self._derived.occupations

    def unwrap(self) -> Any:
        return unwrap(self._structure)

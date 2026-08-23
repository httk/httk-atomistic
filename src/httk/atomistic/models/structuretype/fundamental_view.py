"""Lazy prototype-recognition view."""

from collections.abc import Callable
from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structuretype.anonymize import canonical_dummy_assignment, dummy_species
from httk.atomistic.models.structuretype.anonymized import AnonymizedStructure
from httk.atomistic.models.structuretype.backend import StructuretypeBackend
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.models.structuretype.structuretype import Structuretype
from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase
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


class FundamentalDomainTemplateView(StructuretypeViewBase, FundamentalDomainTemplate):
    r"""Recognize a lazy standard-setting prototype view from a structure.

    Recognition accepts optional ``tolerance`` and ``limit_denominator`` values through
    the recognition hints.

    :param obj: The structuretype-like or structure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: StructuretypeBackend
    _resolved_prototype: FundamentalDomainTemplate | None
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_cell", "_spacegroup", "_wyckoff_sites", "_species", "_coordinate_precision"})

    def __new__(
        cls,
        obj: Any = MISSING,
        *,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if obj is MISSING:  # pickle/copy rebuild an empty instance; __setstate__ restores it
            return super().__new__(cls)
        if isinstance(obj, cls):
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("FundamentalDomainTemplateView rewrapping does not accept recognition arguments")
            return obj
        forbidden = {name for name in ("setting", "standard", "transform") if name in hints}
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(
                f"FundamentalDomainTemplateView does not accept {names}=; use FundamentalDomainTemplateView(ASUStructureView(source, {names}=...))"
            )
        backend = cls._prepare_backend(obj, hints)
        if isinstance(backend, AnonymizedStructure) and (tolerance is not None or limit_denominator is not None):
            from httk.atomistic.models.structure.asu_view import ASUStructureView

            source = backend._structure
            if isinstance(source, (FundamentalDomainStructure, ASUStructureView)) or isinstance(
                getattr(source, "_view", None), ASUStructureView
            ):
                raise ValueError(
                    "FundamentalDomainTemplateView tolerance and limit_denominator cannot be used with an existing ASU"
                )
        if isinstance(backend, FundamentalDomainTemplate):
            if tolerance is not None or limit_denominator is not None:
                raise ValueError(
                    "FundamentalDomainTemplateView tolerance and limit_denominator cannot be used with a FundamentalDomainTemplate"
                )
            instance = super().__new__(cls)
            instance._backend = backend
            instance._resolved_prototype = None
            instance._tolerance = tolerance
            instance._limit_denominator = limit_denominator
            return instance
        instance = super().__new__(cls)
        instance._backend = backend
        instance._resolved_prototype = None
        instance._tolerance = tolerance
        instance._limit_denominator = limit_denominator
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_prototype")()
        return object.__getattribute__(self, name)

    def _effective_prototype(self) -> FundamentalDomainTemplate:
        cached = object.__getattribute__(self, "_resolved_prototype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if isinstance(backend, FundamentalDomainTemplate):
            resolved = backend
        else:
            source: Any
            anonymous_source = isinstance(backend, Structuretype)
            if anonymous_source:
                source = UnitcellStructure(
                    backend.cell,
                    backend.sites,
                    backend.species,
                    backend.species_at_sites,
                )
                key_for_species = lambda name: name
            elif isinstance(backend, AnonymizedStructure):
                source = backend._effective_structure
                backend.resolve()
                real_species = {species.name: species.chemical_symbols[0] for species in source.species}
                key_for_species = lambda name: real_species[name]
            else:
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a prototype source")

            result = conventional_cell(
                source,
                tolerance=object.__getattribute__(self, "_tolerance"),
                limit_denominator=object.__getattribute__(self, "_limit_denominator"),
            )
            mapped_sites, mapped_species = _relabel_sites(
                result.asu.wyckoff_sites,
                result.asu.multiplicities(),
                key_for_species,
            )
            resolved = FundamentalDomainTemplate(
                result.asu.cell,
                result.asu.spacegroup,
                mapped_sites,
                mapped_species,
                result.asu.coordinate_precision,
            )
        state = dict(resolved.__dict__)
        state["_resolved_prototype"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> FundamentalDomainTemplate:
        """Return the recognized prototype as a standalone value.

        :return: The prototype value.
        """
        return self._effective_prototype()

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "backend": self._backend,
            "tolerance": self._tolerance,
            "limit_denominator": self._limit_denominator,
        }
        if self._resolved_prototype is not None:
            state["resolved"] = self._resolved_prototype
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._tolerance = state["tolerance"]
        self._limit_denominator = state["limit_denominator"]
        self._resolved_prototype = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_prototype"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)

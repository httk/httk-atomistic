"""
A view presenting any structure backend as a UnitcellStructure (the Unitcell representation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap
from httk.core.optimade import IncompleteOptimadeResourceError

from httk.atomistic.composition import Assembly
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import _METADATA_UNSET, _resolve_view_metadata, _semantic_value
from httk.atomistic.models.structure.unitcell import (
    UnitcellStructure,
    _check_site_moments,
    _check_sites_length,
    _check_species_at_sites,
    _check_species_names,
    _norm_cell,
    _norm_site_moments,
    _norm_sites,
    _norm_species,
    _norm_species_at_sites,
)
from httk.atomistic.models.structure.view import StructureView


class UnitcellStructureView(StructureView, UnitcellStructure):
    """
    A view presenting an underlying structure backend as a ``UnitcellStructure``.

    This view is a genuine ``UnitcellStructure``, so it can be passed anywhere a UnitcellStructure
    is accepted. Each component is normalized lazily on first access. For an
    ASU-backed view, accessing ``cell`` or ``species`` never triggers expansion.
    """

    _backend: StructureBackend
    _deferred_immutable_id: str | None | object
    _deferred_last_modified: Any
    _effective_backend_cache: StructureBackend | None

    def __new__(
        cls,
        obj: StructureLike,
        *,
        immutable_id: str | None | object = _METADATA_UNSET,
        last_modified: Any = _METADATA_UNSET,
        **hints: Any,
    ) -> Self:
        if isinstance(obj, cls):
            if immutable_id is _METADATA_UNSET and last_modified is _METADATA_UNSET:
                return obj
            backend = obj._backend
            resolver = getattr(backend, "resolve", None)
            if resolver is None:
                resolved_immutable_id, resolved_last_modified = _resolve_view_metadata(
                    obj,
                    immutable_id=immutable_id,
                    last_modified=last_modified,
                )
                if (resolved_immutable_id, resolved_last_modified) == (obj.immutable_id, obj.last_modified):
                    return obj
                cls._validate_span(backend)
            else:
                resolved_immutable_id, resolved_last_modified = None, None
                immutable_id = (
                    getattr(obj, "_deferred_immutable_id", _METADATA_UNSET)
                    if immutable_id is _METADATA_UNSET
                    else immutable_id
                )
                last_modified = (
                    getattr(obj, "_deferred_last_modified", _METADATA_UNSET)
                    if last_modified is _METADATA_UNSET
                    else last_modified
                )
        else:
            backend = cls._prepare_backend(obj, hints)
            resolver = getattr(backend, "resolve", None)
            if resolver is not None:
                resolved_immutable_id, resolved_last_modified = None, None
            else:
                resolved_immutable_id, resolved_last_modified = _resolve_view_metadata(
                    obj,
                    immutable_id=immutable_id,
                    last_modified=last_modified,
                )
                cls._validate_span(backend)
        instance = super().__new__(cls)
        instance._backend = backend
        instance._immutable_id = resolved_immutable_id
        instance._last_modified = resolved_last_modified
        instance._effective_backend_cache = None
        if resolver is not None:
            instance._deferred_immutable_id = immutable_id
            instance._deferred_last_modified = last_modified
        return instance

    @staticmethod
    def _validate_span(backend: StructureBackend) -> None:
        span = getattr(backend, "site_coordinate_span", None)
        if span in {
            "fundamental_domain",
            "asymmetric_unit",
            "molecular_fundamental_domain",
            "molecular_asymmetric_unit",
            "molecular_entities",
            "other",
        } and not isinstance(backend, FundamentalDomainStructure):
            raise IncompleteOptimadeResourceError(
                f"site_coordinate_span={span!r} cannot be projected as a native unit-cell UnitcellStructure view"
            )

    def _effective_backend(self) -> StructureBackend:
        cached = self._effective_backend_cache
        if cached is not None:
            return cached
        resolver = getattr(self._backend, "resolve", None)
        if resolver is None:
            return self._backend
        backend = resolver()
        immutable_id, last_modified = _resolve_view_metadata(
            backend,
            immutable_id=self._deferred_immutable_id,
            last_modified=self._deferred_last_modified,
        )
        self._validate_span(backend)
        object.__setattr__(self, "_immutable_id", immutable_id)
        object.__setattr__(self, "_last_modified", last_modified)
        self._effective_backend_cache = backend
        return backend

    def _metadata(self, name: str, default: Any = None) -> Any:
        return _semantic_value(self._effective_backend(), name, default)

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        pass

    def _fill_cell(self) -> None:
        object.__setattr__(self, "_cell", _norm_cell(self._effective_backend().cell))

    def _fill_species(self) -> None:
        species = _norm_species(self._effective_backend().species)
        _check_species_names(species)
        object.__setattr__(self, "_species", species)

    def _fill_species_at_sites(self) -> None:
        # Exception to the no-shadowed-read rule: this cheap dependency is acyclic because
        # species never reads species_at_sites.
        species_at_sites = _norm_species_at_sites(self._effective_backend().species_at_sites)
        _check_species_at_sites(species_at_sites, self._species)
        object.__setattr__(self, "_species_at_sites", species_at_sites)

    def _fill_sites(self) -> None:
        sites = _norm_sites(self._effective_backend().sites)
        _check_sites_length(sites, self._species_at_sites)
        object.__setattr__(self, "_sites", sites)

    def _fill_site_moments(self) -> None:
        value = _norm_site_moments(self._effective_backend().site_moments)
        _check_site_moments(value, self.sites, self.cell)
        object.__setattr__(self, "_site_moments", value)

    @cached_property
    def _cell(self) -> Cell:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_cell()
        return self.__dict__["_cell"]

    @cached_property
    def _sites(self) -> Sites:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_sites()
        return self.__dict__["_sites"]

    @cached_property
    def _species(self) -> tuple[Species, ...]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_species()
        return self.__dict__["_species"]

    @cached_property
    def _species_at_sites(self) -> tuple[str, ...]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_species_at_sites()
        return self.__dict__["_species_at_sites"]

    @cached_property
    def _site_moments(self) -> SiteMomentsBackend | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_site_moments()
        return self.__dict__["_site_moments"]

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        return self._site_moments

    def unwrap(self) -> Any:
        return unwrap(self._backend)

    @property
    def immutable_id(self) -> str | None:
        self._effective_backend()
        return _semantic_value(self, "immutable_id", private_name="_immutable_id")

    @property
    def last_modified(self) -> Any:
        self._effective_backend()
        return _semantic_value(self, "last_modified", private_name="_last_modified")

    @property
    def molecular(self) -> bool:
        return bool(self._metadata("molecular", False))

    @property
    def site_coordinate_span(self) -> str:
        self._effective_backend()
        return "molecular_unit_cell" if self.molecular else "unit_cell"

    @property
    def symmetry(self) -> Any:
        return self._metadata("symmetry")

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        backend = self._effective_backend()
        if "_assemblies" in self.__dict__:
            return _semantic_value(self, "assemblies", private_name="_assemblies")
        if isinstance(backend, FundamentalDomainStructure) and "_assemblies" not in self.__dict__:
            return backend._expanded_assemblies()
        return self._metadata("assemblies")

    @property
    def chemical_composition(self) -> Any:
        return self._metadata("chemical_composition")

    @property
    def chemical_formula_descriptive(self) -> str | None:
        return self._metadata("chemical_formula_descriptive")

    @property
    def chemical_formula_hill(self) -> str | None:
        return self._metadata("chemical_formula_hill")

    @property
    def optimization_type(self) -> str | None:
        return self._metadata("optimization_type")

    @property
    def site_coordinate_span_description(self) -> str | None:
        return self._metadata("site_coordinate_span_description")

    @property
    def space_group_it_number(self) -> int | None:
        return self._space_group_metadata("space_group_it_number")

    @property
    def space_group_symbol_hall(self) -> str | None:
        return self._space_group_metadata("space_group_symbol_hall")

    @property
    def space_group_symbol_hermann_mauguin(self) -> str | None:
        return self._space_group_metadata("space_group_symbol_hermann_mauguin")

    @property
    def space_group_symbol_hermann_mauguin_extended(self) -> str | None:
        return self._space_group_metadata("space_group_symbol_hermann_mauguin_extended")

    @property
    def space_group_symmetry_operations_xyz(self) -> tuple[str, ...] | None:
        value = self._space_group_metadata("space_group_symmetry_operations_xyz")
        return value if value is not None else (("x,y,z",) if self.nperiodic_dimensions else None)

    @property
    def wyckoff_positions(self) -> tuple[str, ...] | None:
        return self._space_group_metadata("wyckoff_positions")

    def _space_group_metadata(self, name: str) -> Any:
        if getattr(self._backend, "resolve", None) is None:
            symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        else:
            backend = self._effective_backend()
            value = getattr(backend, name, None)
            if value is not None:
                return value
            symmetry = getattr(backend, "symmetry", None)
        return None if symmetry is None else getattr(symmetry, name, None)

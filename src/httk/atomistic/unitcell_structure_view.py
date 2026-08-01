"""
A view presenting any structure backend as a Structure (the Unitcell representation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import IncompleteOptimadeResourceError, unwrap

from .cell import Cell
from .sites import Sites
from .species import Species
from .structure import (
    Structure,
    _check_sites_length,
    _check_species_at_sites,
    _check_species_names,
    _norm_cell,
    _norm_sites,
    _norm_species,
    _norm_species_at_sites,
)
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_view import StructureView


class UnitcellStructureView(StructureView, Structure):
    """
    A view presenting an underlying structure backend as a ``Structure``.

    This view is a genuine ``Structure``, so it can be passed anywhere a Structure
    is accepted. Each component is normalized lazily on first access. For an
    ASU-backed view, accessing ``cell`` or ``species`` never triggers expansion.
    """

    _backend: StructureBackend

    def __new__(cls, obj: StructureLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        span = getattr(backend, "site_coordinate_span", None)
        if span in {
            "fundamental_domain",
            "asymmetric_unit",
            "molecular_fundamental_domain",
            "molecular_asymmetric_unit",
            "molecular_entities",
            "other",
        }:
            raise IncompleteOptimadeResourceError(
                f"site_coordinate_span={span!r} cannot be projected as a native unit-cell Structure view"
            )
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        pass

    def _fill_cell(self) -> None:
        self.__dict__["_cell"] = _norm_cell(self._backend.cell)

    def _fill_species(self) -> None:
        species = _norm_species(self._backend.species)
        _check_species_names(species)
        self.__dict__["_species"] = species

    def _fill_species_at_sites(self) -> None:
        # Exception to the no-shadowed-read rule: this cheap dependency is acyclic because
        # species never reads species_at_sites.
        species_at_sites = _norm_species_at_sites(self._backend.species_at_sites)
        _check_species_at_sites(species_at_sites, self._species)
        self.__dict__["_species_at_sites"] = species_at_sites

    def _fill_sites(self) -> None:
        sites = _norm_sites(self._backend.sites)
        _check_sites_length(sites, self._species_at_sites)
        self.__dict__["_sites"] = sites

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

    def unwrap(self) -> Any:
        return unwrap(self._backend)

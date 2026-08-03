"""UnitcellStructure backend for the three exact native storage records."""

from functools import cached_property
from typing import Any, cast

from httk.atomistic.composition import Assembly
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view import SitesView
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import StructureSymmetry
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    UnitcellStructureRecord,
    _assembly_from_record,
    _chemical_composition_from_record,
    _domain_structure_from_record,
    _symmetry_from_record,
)


class RecordStructure(StructureBackend):
    """Expose a concrete record through the existing structure view family."""

    _record: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)):
            return None
        return super().__new__(cls)

    def __init__(
        self,
        obj: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord,
        **hints: Any,
    ) -> None:
        self._record = obj

    @cached_property
    def _native(self) -> Any:
        record = self._record
        assert isinstance(record, (FundamentalDomainStructureRecord, ASUStructureRecord))
        return _domain_structure_from_record(record)

    @cached_property
    def _expanded(self) -> Any:
        from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

        return UnitcellStructureView(self._native)

    @property
    def _is_unitcell(self) -> bool:
        return isinstance(self._record, UnitcellStructureRecord)

    @cached_property
    def cell(self) -> Cell:  # pyright: ignore[reportIncompatibleMethodOverride]
        return CellView(cast(Any, self._record.cell)) if self._is_unitcell else self._native.cell

    @cached_property
    def sites(self) -> Sites:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            SitesView(cast(Any, self._record.sites))
            if isinstance(self._record, UnitcellStructureRecord)
            else self._expanded.sites
        )

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            tuple(SpeciesView(cast(Any, value)) for value in self._record.species)
            if self._is_unitcell
            else self._native.species
        )

    @cached_property
    def species_at_sites(self) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            self._record.species_at_sites
            if isinstance(self._record, UnitcellStructureRecord)
            else self._expanded.species_at_sites
        )

    @property
    def molecular(self) -> bool:
        return self._record.molecular

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        if self._is_unitcell:
            return (
                None
                if self._record.assemblies is None
                else tuple(_assembly_from_record(value) for value in self._record.assemblies)
            )
        return self._expanded.assemblies

    @property
    def symmetry(self) -> StructureSymmetry | None:
        if isinstance(self._record, UnitcellStructureRecord):
            return None if self._record.symmetry is None else _symmetry_from_record(self._record.symmetry)
        native = self._native
        assert isinstance(native, FundamentalDomainStructure)
        positions = native.wyckoff_positions
        expanded_positions = (
            None
            if positions is None
            else tuple(letter for letter, count in zip(positions, native.multiplicities()) for _ in range(count))
        )
        return StructureSymmetry(
            native.space_group_it_number,
            native.space_group_symbol_hall,
            native.space_group_symbol_hermann_mauguin,
            native.space_group_symbol_hermann_mauguin_extended,
            native.space_group_symmetry_operations_xyz,
            expanded_positions,
        )

    @property
    def chemical_composition(self) -> Any:
        return (
            None
            if self._record.chemical_composition is None
            else _chemical_composition_from_record(self._record.chemical_composition)
        )

    @property
    def chemical_formula_descriptive(self) -> str | None:
        return self._record.chemical_formula_descriptive

    @property
    def chemical_formula_hill(self) -> str | None:
        return self._record.chemical_formula_hill

    @property
    def optimization_type(self) -> str | None:
        return self._record.optimization_type

    @property
    def immutable_id(self) -> str | None:
        return self._record.immutable_id

    @property
    def last_modified(self) -> Any:
        return self._record.last_modified

    @property
    def asu(self) -> ASUStructure | FundamentalDomainStructure | None:
        if isinstance(self._record, (ASUStructureRecord, FundamentalDomainStructureRecord)):
            return self._native
        return None

    def unwrap(self) -> UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord:
        """Return the exact fetched record rather than a reconstructed structure."""
        return self._record

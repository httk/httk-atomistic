"""Structure backend for the three exact native storage records."""

from functools import cached_property
from typing import Any

from .asu_structure import ASUStructure, FundamentalDomainStructure
from .cell import Cell
from .composition import Assembly
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend
from .structure_record import ASUStructureRecord, FundamentalDomainStructureRecord, UnitcellStructureRecord
from .structure_semantics import StructureSymmetry


class StructureRecordBackend(StructureBackend):
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
        return self._record.to_structure()

    @cached_property
    def _expanded(self) -> Any:
        from .unitcell_structure_view import UnitcellStructureView

        return UnitcellStructureView(self._native)

    @property
    def _is_unitcell(self) -> bool:
        return isinstance(self._record, UnitcellStructureRecord)

    @cached_property
    def cell(self) -> Cell:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._record.cell.to_cell() if self._is_unitcell else self._native.cell

    @cached_property
    def sites(self) -> Sites:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._record.sites.to_sites() if self._is_unitcell else self._expanded.sites

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            tuple(value.to_species() for value in self._record.species) if self._is_unitcell else self._native.species
        )

    @cached_property
    def species_at_sites(self) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._record.species_at_sites if self._is_unitcell else self._expanded.species_at_sites

    @property
    def molecular(self) -> bool:
        return self._record.molecular

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        if self._is_unitcell:
            return (
                None
                if self._record.assemblies is None
                else tuple(value.to_assembly() for value in self._record.assemblies)
            )
        return self._expanded.assemblies

    @property
    def symmetry(self) -> StructureSymmetry | None:
        if self._is_unitcell:
            return None if self._record.symmetry is None else self._record.symmetry.to_symmetry()
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
        return None if self._record.chemical_composition is None else self._record.chemical_composition.to_composition()

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

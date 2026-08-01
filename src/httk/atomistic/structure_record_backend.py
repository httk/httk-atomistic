"""Structure backend for :class:`~httk.atomistic.StructureRecord` snapshots."""

from functools import cached_property
from typing import Any

from .cell import Cell
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend
from .structure_record import StructureRecord, _basis_vector


class StructureRecordBackend(StructureBackend):
    """Backend exposing a ``StructureRecord`` through the structure quartet."""

    _record: StructureRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, StructureRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: StructureRecord, **hints: Any) -> None:
        self._record = obj

    @cached_property
    def _expanded(self) -> Any:
        from .unitcell_structure_view import UnitcellStructureView

        return UnitcellStructureView(self._record.to_structure())

    @cached_property
    def cell(self) -> Cell:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._record.representation != "unit_cell":
            return self._expanded.cell
        return Cell(
            _basis_vector(self._record.basis),
            precision=self._record.basis_precision,
            periodicity=self._record.periodicity,
        )

    @cached_property
    def sites(self) -> Sites:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._record.representation != "unit_cell":
            return self._expanded.sites
        return Sites(self._record.reduced_coords, precision=self._record.coordinate_precision)

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._record.representation != "unit_cell":
            return self._expanded.species
        return tuple(value.to_species() for value in self._record.species)

    @cached_property
    def species_at_sites(self) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._record.representation != "unit_cell":
            return self._expanded.species_at_sites
        return self._record.species_at_sites

    @property
    def molecular(self) -> bool:
        return self._record.molecular

    @property
    def assemblies(self) -> Any:
        if self._record.representation != "unit_cell":
            return self._expanded.assemblies
        return (
            None if self._record.assemblies is None else tuple(value.to_assembly() for value in self._record.assemblies)
        )

    @property
    def symmetry(self) -> Any:
        return None if self._record.symmetry is None else self._record.symmetry.to_symmetry()

    @property
    def chemical_composition(self) -> Any:
        value = self._record.chemical_composition
        return None if value is None else value.to_composition()

    @property
    def chemical_formula_descriptive(self) -> str | None:
        return self._record.chemical_formula_descriptive

    @property
    def chemical_formula_hill(self) -> str | None:
        return self._record.chemical_formula_hill

    @property
    def optimization_type(self) -> str | None:
        return self._record.optimization_type

    def unwrap(self) -> StructureRecord:
        """Return the stored snapshot rather than the reconstructed structure."""
        return self._record

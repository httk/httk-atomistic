"""UnitcellStructure backend for the three exact native storage records."""

from functools import cached_property
from typing import Any, Self, cast

from httk.atomistic.composition import Assembly
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.formula.composition_view import CompositionView
from httk.atomistic.models.formula.record import RecordComposition
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
    _moment_from_record,
    _symmetry_from_record,
)


class RecordStructure(StructureBackend):
    """Expose a storage record through the existing structure view family.

    Unit-cell records expose their stored components directly; fundamental-domain and
    asymmetric-unit records expand through the native domain structure when a unit-cell
    view is requested.

    :param obj: The unit-cell, fundamental-domain, or asymmetric-unit record.
    :param \\*\\*hints: Backend-selection hints.
    """

    _record: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a structure record.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)):
            return None
        return cls(obj, **hints)

    def __init__(
        self,
        obj: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord,
        **hints: Any,
    ) -> None:
        self._record = obj

    @cached_property
    def composition(self) -> CompositionView:
        """Expose the record's authoritative normalized composition."""
        return CompositionView(RecordComposition(self._record.normalized_composition))

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
        """Expose the record-backed cell.

        :return: The cell, directly from a unit-cell record or from the expanded domain.
        """
        # These wrap sites hold record components by construction, so the kind
        # hint selects the record backend without probing the raw-input ones.
        return CellView(cast(Any, self._record.cell), kind="record") if self._is_unitcell else self._native.cell

    @cached_property
    def sites(self) -> Sites:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Expose the record-backed sites.

        :return: The sites, directly from a unit-cell record or from the expanded domain.
        """
        return (
            SitesView(cast(Any, self._record.sites), kind="record")
            if isinstance(self._record, UnitcellStructureRecord)
            else self._expanded.sites
        )

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Expose the record-backed distinct species.

        :return: The species referenced by the structure.
        """
        return (
            tuple(SpeciesView(cast(Any, value), kind="record") for value in self._record.species)
            if self._is_unitcell
            else self._native.species
        )

    @cached_property
    def species_at_sites(self) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Expose the species occupying each record-backed site.

        :return: Site species names in site order.
        """
        return (
            self._record.species_at_sites
            if isinstance(self._record, UnitcellStructureRecord)
            else self._expanded.species_at_sites
        )

    @cached_property
    def site_moments(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Expose optional record-backed site moments.

        :return: Site moments, or ``None`` when they are unstated.
        """
        if isinstance(self._record, UnitcellStructureRecord):
            return _moment_from_record(
                self._record.site_moments_kind,
                self._record.site_moments,
                self._record.site_moments_precision,
                self.cell,
            )
        return self._expanded.site_moments

    @property
    def charge(self) -> Any:
        """Expose the record's explicitly assigned charge.

        :return: The assigned charge, or ``None`` when it is unstated.
        """
        return getattr(self._record, "charge", None)

    @property
    def molecular(self) -> bool:
        """Expose whether the record describes a molecular unit cell.

        :return: Whether molecular semantics are enabled.
        """
        return self._record.molecular

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        """Expose the record's site assemblies.

        :return: Assemblies, or ``None`` when they are unstated.
        """
        if self._is_unitcell:
            return (
                None
                if self._record.assemblies is None
                else tuple(_assembly_from_record(value) for value in self._record.assemblies)
            )
        return self._expanded.assemblies

    @property
    def symmetry(self) -> StructureSymmetry | None:
        """Expose the record's symmetry metadata.

        :return: Symmetry metadata, or ``None`` when it is absent.
        """
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
        """Expose the record's chemical composition.

        :return: Chemical composition, or ``None`` when it is absent.
        """
        return (
            None
            if self._record.chemical_composition is None
            else _chemical_composition_from_record(self._record.chemical_composition)
        )

    @property
    def chemical_formula_descriptive(self) -> str | None:
        """Expose the record's descriptive chemical formula.

        :return: The descriptive formula, or ``None`` when it is absent.
        """
        return self._record.chemical_formula_descriptive

    @property
    def chemical_formula_hill(self) -> str | None:
        """Expose the record's Hill chemical formula.

        :return: The Hill formula, or ``None`` when it is absent.
        """
        return self._record.chemical_formula_hill

    @property
    def optimization_type(self) -> str | None:
        """Expose the record's optimization provenance.

        :return: The optimization type, or ``None`` when it is absent.
        """
        return self._record.optimization_type

    @property
    def immutable_id(self) -> str | None:
        """Expose the record's immutable source identifier.

        :return: The identifier, or ``None`` when it is absent.
        """
        return self._record.immutable_id

    @property
    def last_modified(self) -> Any:
        """Expose the record's modification timestamp.

        :return: The timestamp, or ``None`` when it is absent.
        """
        return self._record.last_modified

    @property
    def asu(self) -> ASUStructure | FundamentalDomainStructure | None:
        """Expose the native domain structure when the record stores one.

        :return: The native asymmetric or fundamental domain, or ``None`` for unit-cell records.
        """
        if isinstance(self._record, (ASUStructureRecord, FundamentalDomainStructureRecord)):
            return self._native
        return None

    def unwrap(self) -> UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord:
        """Return the exact fetched record rather than a reconstructed structure.

        :return: The original storage record.
        """
        return self._record

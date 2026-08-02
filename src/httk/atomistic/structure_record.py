"""Frozen storage records for complete atomistic structures."""

import builtins
import datetime
import fractions
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, TypedDict, cast

import httk.core.storage_markers
from httk.core import (
    FracVector,
    IdentitySkip,
    StorageInfo,
    SurdScalar,
    SurdVector,
    content_id,
)

from ._composition_values import as_fraction
from ._vector_guards import to_precision
from .asu_structure import ASUSite, ASUStructure, FundamentalDomainStructure
from .cell import Cell
from .composition import Assembly, ChemicalComposition, validate_assemblies
from .setting_transform import SettingTransform
from .sites import Sites
from .species import Species
from .structure import Structure
from .structure_semantics import StructureSymmetry

__all__ = [
    "ASUStructureRecord",
    "AssemblyGroupRecord",
    "AssemblyRecord",
    "CellRecord",
    "ChemicalCompositionRecord",
    "CompositionAmountRecord",
    "DomainSiteRecord",
    "FundamentalDomainStructureRecord",
    "SettingTransformRecord",
    "SitesRecord",
    "SpeciesRecord",
    "SymmetryRecord",
    "UnitcellStructureRecord",
]


def _extract_surd_scalar(vector: SurdVector, index: tuple[int, int]) -> SurdScalar:
    """Extract one exact scalar through SurdVector's public coefficient API."""
    components = {radicand: vector.coefficient(radicand)[index].to_fraction() for radicand in vector.radicands}
    return cast(SurdScalar, SurdVector.from_radicand_map(components))


def _basis_vector(basis: tuple[SurdScalar, ...]) -> SurdVector:
    """Reconstruct the 3x3 basis from the record's row-major scalar values."""
    radicands = sorted({radicand for value in basis for radicand in value.radicands})
    components = {
        radicand: [
            [basis[3 * row + column].coefficient(radicand).to_fraction() for column in range(3)] for row in range(3)
        ]
        for radicand in radicands
    }
    return SurdVector.from_radicand_map(components)


@dataclass(frozen=True)
class SpeciesRecord:
    """The storable frozen snapshot of an atomistic :class:`~httk.atomistic.Species`.

    The presence flags preserve ``None`` for optional tuple fields because a relational child
    table cannot distinguish a missing optional value from an empty tuple when reconstructing.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_species_record")
    __httk_canonical_source__: ClassVar[type[Species]] = Species

    name: str
    chemical_symbols: tuple[str, ...]
    concentration: tuple[fractions.Fraction, ...]
    mass: tuple[float, ...] | None = None
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None
    mass_present: bool = False
    attached_present: bool = False
    concentration_precision: tuple[fractions.Fraction, ...] = ()
    concentration_precision_present: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "chemical_symbols", tuple(self.chemical_symbols))
        concentration = tuple(
            as_fraction(value, field="SpeciesRecord concentration")[0] for value in self.concentration
        )
        object.__setattr__(self, "concentration", concentration)
        if self.mass is not None:
            object.__setattr__(self, "mass", tuple(self.mass))
        if self.attached is not None:
            object.__setattr__(self, "attached", tuple(self.attached))
        if self.nattached is not None:
            object.__setattr__(self, "nattached", tuple(self.nattached))
        raw_precision = tuple(self.concentration_precision)
        if not isinstance(self.concentration_precision_present, bool):
            raise TypeError("SpeciesRecord concentration_precision_present must be a bool")
        if self.concentration_precision_present:
            if len(raw_precision) != len(concentration):
                raise ValueError("SpeciesRecord concentration_precision must match concentration")
            converted_precision: list[fractions.Fraction] = []
            for value in raw_precision:
                central, _ = as_fraction(value, field="SpeciesRecord concentration precision")
                if central < 0:
                    raise ValueError("SpeciesRecord concentration precision cannot be negative")
                converted_precision.append(central)
            object.__setattr__(self, "concentration_precision", tuple(converted_precision))
        elif raw_precision:
            raise ValueError("SpeciesRecord concentration_precision requires concentration_precision_present=True")

        if not isinstance(self.mass_present, bool) or not isinstance(self.attached_present, bool):
            raise TypeError("SpeciesRecord presence flags must be bool values")
        if self.mass_present and self.mass is None:
            raise ValueError("SpeciesRecord mass_present=True requires a mass tuple")
        if not self.mass_present and self.mass:
            raise ValueError("SpeciesRecord mass_present=False cannot have non-empty mass")
        if (self.attached is None) != (self.nattached is None):
            raise ValueError("SpeciesRecord attached and nattached must be provided together")
        if self.attached_present and self.attached is None:
            raise ValueError("SpeciesRecord attached_present=True requires attached and nattached tuples")
        if not self.attached_present and (self.attached or self.nattached):
            raise ValueError("SpeciesRecord attached_present=False cannot have non-empty attachments")

        for value_name, values in (
            ("concentration", self.concentration),
            ("mass", self.mass or ()),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"SpeciesRecord {value_name} values must be finite floats")

        mass = self.mass if self.mass_present else None
        attached = self.attached if self.attached_present else None
        nattached = self.nattached if self.attached_present else None
        try:
            Species(
                name=self.name,
                chemical_symbols=self.chemical_symbols,
                concentration=self.concentration,
                mass=mass,
                original_name=self.original_name,
                attached=attached,
                nattached=nattached,
                concentration_precision=(
                    None
                    if not self.concentration_precision_present
                    else tuple(None if value == 0 else value for value in self.concentration_precision)
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("SpeciesRecord fields do not describe a valid Species") from error
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "attached", attached)
        object.__setattr__(self, "nattached", nattached)

    @classmethod
    def __httk_project__(cls, species: Species) -> Mapping[str, object]:
        precision = species.concentration_precision or ()
        present = not all(value is None for value in precision)
        return {
            "name": species.name,
            "chemical_symbols": species.chemical_symbols,
            "concentration": species.concentration,
            "mass": species.mass,
            "original_name": species.original_name,
            "attached": species.attached,
            "nattached": species.nattached,
            "mass_present": species.mass is not None,
            "attached_present": species.attached is not None,
            "concentration_precision": ()
            if not present
            else tuple(value or fractions.Fraction() for value in precision),
            "concentration_precision_present": present,
        }

    @classmethod
    def from_species(cls, species: Species) -> "SpeciesRecord":
        """Create a storage record containing the exact species metadata."""
        return cls(
            name=species.name,
            chemical_symbols=species.chemical_symbols,
            concentration=species.concentration,
            mass=species.mass,
            original_name=species.original_name,
            attached=species.attached,
            nattached=species.nattached,
            mass_present=species.mass is not None,
            attached_present=species.attached is not None,
            concentration_precision=(
                ()
                if all(value is None for value in species.concentration_precision or ())
                else tuple(value or fractions.Fraction() for value in species.concentration_precision or ())
            ),
            concentration_precision_present=not all(value is None for value in species.concentration_precision or ()),
        )

    def to_species(self) -> Species:
        """Reconstruct the domain :class:`~httk.atomistic.Species` from this snapshot."""
        return Species(
            name=self.name,
            chemical_symbols=self.chemical_symbols,
            concentration=self.concentration,
            mass=self.mass if self.mass_present else None,
            original_name=self.original_name,
            attached=self.attached if self.attached_present else None,
            nattached=self.nattached if self.attached_present else None,
            concentration_precision=(
                None
                if not self.concentration_precision_present
                else tuple(None if value == 0 else value for value in self.concentration_precision)
            ),
        )

    def to_stored_entry_value(self) -> dict[str, object]:
        """Return exactly the standard OPTIMADE species dictionary shape."""
        value: dict[str, object] = {
            "name": self.name,
            "chemical_symbols": self.chemical_symbols,
            "concentration": self.concentration,
        }
        if self.mass_present:
            value["mass"] = self.mass
        if self.original_name is not None:
            value["original_name"] = self.original_name
        if self.attached_present:
            value["attached"] = self.attached
            value["nattached"] = self.nattached
        return value


@dataclass(frozen=True)
class AssemblyGroupRecord:
    """One exact site-index group in a stored assembly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_assembly_group_record")
    __httk_canonical_source__: ClassVar[type[tuple]] = tuple

    sites: tuple[int, ...]

    @classmethod
    def __httk_project__(cls, group: tuple[Any, ...]) -> Mapping[str, object]:
        return {"sites": group}

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        if not sites or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in sites):
            raise ValueError("AssemblyGroupRecord sites must be non-empty non-negative integer indices")
        if len(set(sites)) != len(sites):
            raise ValueError("AssemblyGroupRecord cannot repeat a site index")
        object.__setattr__(self, "sites", sites)


@dataclass(frozen=True)
class AssemblyRecord:
    """Exact durable form of :class:`~httk.atomistic.Assembly`, including per-value precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_assembly_record")
    __httk_canonical_source__: ClassVar[type[Assembly]] = Assembly

    groups: tuple[AssemblyGroupRecord, ...]
    group_probabilities: tuple[fractions.Fraction, ...]
    group_probabilities_precision: tuple[fractions.Fraction, ...] = ()
    group_probabilities_precision_present: bool = False

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not all(isinstance(group, AssemblyGroupRecord) for group in groups):
            raise TypeError("AssemblyRecord groups must contain AssemblyGroupRecord values")
        probabilities = tuple(
            as_fraction(value, field="AssemblyRecord group probability")[0] for value in self.group_probabilities
        )
        precision = tuple(
            as_fraction(value, field="AssemblyRecord group probability precision")[0]
            for value in self.group_probabilities_precision
        )
        if not groups or len(groups) != len(probabilities):
            raise ValueError("AssemblyRecord groups and probabilities must have matching non-empty lengths")
        if not isinstance(self.group_probabilities_precision_present, bool):
            raise TypeError("AssemblyRecord group_probabilities_precision_present must be a bool")
        if self.group_probabilities_precision_present:
            if len(precision) != len(probabilities) or any(value < 0 for value in precision):
                raise ValueError("AssemblyRecord precision must match probabilities and be non-negative")
        elif precision:
            raise ValueError("AssemblyRecord precision values require the presence flag")
        assembly = Assembly(
            tuple(group.sites for group in groups),
            probabilities,
            None
            if not self.group_probabilities_precision_present
            else tuple(None if value == 0 else value for value in precision),
        )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "group_probabilities", assembly.group_probabilities)
        object.__setattr__(self, "group_probabilities_precision", precision)

    @classmethod
    def __httk_project__(cls, assembly: Assembly) -> Mapping[str, object]:
        precision = assembly.group_probabilities_precision or ()
        present = not all(value is None for value in precision)
        return {
            "groups": assembly.sites_in_groups,
            "group_probabilities": assembly.group_probabilities,
            "group_probabilities_precision": ()
            if not present
            else tuple(value or fractions.Fraction() for value in precision),
            "group_probabilities_precision_present": present,
        }

    @classmethod
    def from_assembly(cls, assembly: Assembly) -> "AssemblyRecord":
        precision = assembly.group_probabilities_precision or ()
        present = not all(value is None for value in precision)
        return cls(
            tuple(AssemblyGroupRecord(group) for group in assembly.sites_in_groups),
            assembly.group_probabilities,
            () if not present else tuple(value or fractions.Fraction() for value in precision),
            present,
        )

    def to_assembly(self) -> Assembly:
        return Assembly(
            tuple(group.sites for group in self.groups),
            self.group_probabilities,
            None
            if not self.group_probabilities_precision_present
            else tuple(None if value == 0 else value for value in self.group_probabilities_precision),
        )

    def to_stored_entry_value(self) -> dict[str, object]:
        """Return exactly the standard OPTIMADE assembly dictionary shape."""
        return {
            "sites_in_groups": tuple(group.sites for group in self.groups),
            "group_probabilities": self.group_probabilities,
        }


@dataclass(frozen=True)
class DomainSiteRecord:
    """Exact Wyckoff/free-parameter site with its retained representative."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_domain_site_record")
    __httk_canonical_source__: ClassVar[type[ASUSite]] = ASUSite

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] = ()
    representative_present: bool = False

    @classmethod
    def __httk_project__(cls, site: ASUSite) -> Mapping[str, object]:
        return {
            "wyckoff": site.wyckoff,
            "free_parameters": tuple(site.free_params.to_fractions()),
            "species": site.species,
            "representative": () if site.representative is None else tuple(site.representative.to_fractions()),
            "representative_present": site.representative is not None,
        }

    def __post_init__(self) -> None:
        free = tuple(as_fraction(value, field="DomainSiteRecord free parameter")[0] for value in self.free_parameters)
        representative = tuple(
            as_fraction(value, field="DomainSiteRecord representative")[0] for value in self.representative
        )
        if not isinstance(self.wyckoff, str) or len(self.wyckoff) != 1:
            raise ValueError("DomainSiteRecord wyckoff must be a single letter")
        if not isinstance(self.species, str) or not self.species:
            raise ValueError("DomainSiteRecord species must be non-empty")
        if not isinstance(self.representative_present, bool):
            raise TypeError("DomainSiteRecord representative_present must be a bool")
        if self.representative_present and len(representative) != 3:
            raise ValueError("DomainSiteRecord representative must have exactly three values")
        if not self.representative_present and representative:
            raise ValueError("DomainSiteRecord representative values require the presence flag")
        object.__setattr__(self, "free_parameters", free)
        object.__setattr__(self, "representative", representative)


@dataclass(frozen=True)
class SettingTransformRecord:
    """Exact durable standard-to-own setting transform."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_setting_transform_record")
    __httk_canonical_source__: ClassVar[type[SettingTransform]] = SettingTransform

    matrix: Annotated[FracVector, httk.core.storage_markers.Shape(3, 3)]
    vector: tuple[fractions.Fraction, ...]
    hall_entry: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        matrix = FracVector.create(self.matrix)
        vector = tuple(as_fraction(value, field="SettingTransformRecord vector")[0] for value in self.vector)
        transform = SettingTransform(matrix, vector, hall_entry=self.hall_entry)
        object.__setattr__(self, "matrix", transform.matrix)
        object.__setattr__(self, "vector", tuple(transform.vector.to_fractions()))

    @classmethod
    def __httk_project__(cls, transform: SettingTransform) -> Mapping[str, object]:
        return {
            "matrix": transform.matrix,
            "vector": tuple(transform.vector.to_fractions()),
            "hall_entry": transform.hall_entry,
        }

    @classmethod
    def from_transform(cls, transform: SettingTransform) -> "SettingTransformRecord":
        return cls(transform.matrix, tuple(transform.vector.to_fractions()), transform.hall_entry)

    def to_transform(self) -> SettingTransform:
        return SettingTransform(self.matrix, self.vector, hall_entry=self.hall_entry)


@dataclass(frozen=True)
class SymmetryRecord:
    """Typed optional symmetry metadata of an ordinary unit-cell structure."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_symmetry_v3")
    __httk_canonical_source__: ClassVar[type[StructureSymmetry]] = StructureSymmetry

    space_group_it_number: int | None = None
    space_group_symbol_hall: str | None = None
    space_group_symbol_hermann_mauguin: str | None = None
    space_group_symbol_hermann_mauguin_extended: str | None = None
    space_group_symmetry_operations_xyz: tuple[str, ...] | None = None
    wyckoff_positions: tuple[str, ...] | None = None
    operations_present: bool = False
    wyckoff_positions_present: bool = False

    def __post_init__(self) -> None:
        operations = self.space_group_symmetry_operations_xyz
        positions = self.wyckoff_positions
        if not isinstance(self.operations_present, bool) or not isinstance(self.wyckoff_positions_present, bool):
            raise TypeError("SymmetryRecord presence flags must be bool values")
        if self.operations_present and operations is None:
            operations = ()
        if self.wyckoff_positions_present and positions is None:
            positions = ()
        if not self.operations_present and operations:
            raise ValueError("SymmetryRecord operations require their presence flag")
        if not self.wyckoff_positions_present and positions:
            raise ValueError("SymmetryRecord Wyckoff positions require their presence flag")
        value = StructureSymmetry(
            self.space_group_it_number,
            self.space_group_symbol_hall,
            self.space_group_symbol_hermann_mauguin,
            self.space_group_symbol_hermann_mauguin_extended,
            None if not self.operations_present else tuple(operations or ()),
            None if not self.wyckoff_positions_present else tuple(positions or ()),
        )
        object.__setattr__(self, "space_group_symmetry_operations_xyz", value.space_group_symmetry_operations_xyz)
        object.__setattr__(self, "wyckoff_positions", value.wyckoff_positions)

    @classmethod
    def __httk_project__(cls, symmetry: StructureSymmetry) -> Mapping[str, object]:
        return {
            "space_group_it_number": symmetry.space_group_it_number,
            "space_group_symbol_hall": symmetry.space_group_symbol_hall,
            "space_group_symbol_hermann_mauguin": symmetry.space_group_symbol_hermann_mauguin,
            "space_group_symbol_hermann_mauguin_extended": symmetry.space_group_symbol_hermann_mauguin_extended,
            "space_group_symmetry_operations_xyz": symmetry.space_group_symmetry_operations_xyz,
            "wyckoff_positions": symmetry.wyckoff_positions,
            "operations_present": symmetry.space_group_symmetry_operations_xyz is not None,
            "wyckoff_positions_present": symmetry.wyckoff_positions is not None,
        }

    @classmethod
    def from_symmetry(cls, symmetry: StructureSymmetry) -> "SymmetryRecord":
        return cls(
            symmetry.space_group_it_number,
            symmetry.space_group_symbol_hall,
            symmetry.space_group_symbol_hermann_mauguin,
            symmetry.space_group_symbol_hermann_mauguin_extended,
            symmetry.space_group_symmetry_operations_xyz,
            symmetry.wyckoff_positions,
            symmetry.space_group_symmetry_operations_xyz is not None,
            symmetry.wyckoff_positions is not None,
        )

    def to_symmetry(self) -> StructureSymmetry:
        return StructureSymmetry(
            self.space_group_it_number,
            self.space_group_symbol_hall,
            self.space_group_symbol_hermann_mauguin,
            self.space_group_symbol_hermann_mauguin_extended,
            self.space_group_symmetry_operations_xyz if self.operations_present else None,
            self.wyckoff_positions if self.wyckoff_positions_present else None,
        )


@dataclass(frozen=True)
class CompositionAmountRecord:
    """One exact declared element amount and its optional precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_composition_amount_record")
    __httk_canonical_source__: ClassVar[type[tuple]] = tuple

    element: str
    amount: fractions.Fraction
    precision: fractions.Fraction | None = None

    @classmethod
    def __httk_project__(cls, amount: tuple[Any, ...]) -> Mapping[str, object]:
        if len(amount) != 3:
            raise ValueError("composition amount projection requires element, amount, and precision")
        return {"element": amount[0], "amount": amount[1], "precision": amount[2]}


@dataclass(frozen=True)
class ChemicalCompositionRecord:
    """Durable authoritative or implicit composition declaration."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_chemical_composition_record")
    __httk_canonical_source__: ClassVar[type[ChemicalComposition]] = ChemicalComposition

    amounts: tuple[CompositionAmountRecord, ...]
    mode: str

    @classmethod
    def __httk_project__(cls, composition: ChemicalComposition) -> Mapping[str, object]:
        precision = dict(composition.amounts_precision)
        return {
            "amounts": tuple((element, amount, precision[element]) for element, amount in composition.amounts),
            "mode": composition.mode,
        }

    def __post_init__(self) -> None:
        value = self.to_composition()
        object.__setattr__(
            self,
            "amounts",
            tuple(
                CompositionAmountRecord(element, amount, dict(value.amounts_precision)[element])
                for element, amount in value.amounts
            ),
        )

    @classmethod
    def from_composition(cls, composition: ChemicalComposition) -> "ChemicalCompositionRecord":
        precision = dict(composition.amounts_precision)
        return cls(
            tuple(
                CompositionAmountRecord(element, amount, precision[element]) for element, amount in composition.amounts
            ),
            composition.mode,
        )

    def to_composition(self) -> ChemicalComposition:
        return ChemicalComposition(
            {value.element: value.amount for value in self.amounts},
            cast(Any, self.mode),
            {value.element: value.precision for value in self.amounts if value.precision is not None},
        )


# These concrete records retain each representation's native fields; recursive
# storage projection follows their annotations.


@dataclass(frozen=True)
class CellRecord:
    """Exact durable cell basis, precision, and periodicity."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_cell_v1",
        identity_name="httk.atomistic.CellRecord",
    )
    __httk_canonical_source__: ClassVar[type[Cell]] = Cell

    basis: tuple[SurdScalar, ...]
    precision: fractions.Fraction | None
    periodicity: tuple[bool, ...]

    def __post_init__(self) -> None:
        basis = tuple(self.basis)
        periodicity = tuple(self.periodicity)
        if len(basis) != 9 or not all(isinstance(value, SurdScalar) for value in basis):
            raise ValueError("CellRecord basis must contain exactly nine SurdScalar values")
        cell = Cell(_basis_vector(basis), precision=self.precision, periodicity=periodicity)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "precision", cell.precision)
        object.__setattr__(self, "periodicity", cell.periodicity)

    @classmethod
    def __httk_project__(cls, cell: Cell) -> Mapping[str, object]:
        return {
            "basis": tuple(_extract_surd_scalar(cell.basis, (row, column)) for row in range(3) for column in range(3)),
            "precision": cell.precision,
            "periodicity": cell.periodicity,
        }

    def to_cell(self) -> Cell:
        return Cell(_basis_vector(self.basis), precision=self.precision, periodicity=self.periodicity)


@dataclass(frozen=True)
class SitesRecord:
    """Exact durable reduced coordinates and their stated precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_sites_v1",
        identity_name="httk.atomistic.SitesRecord",
    )
    __httk_canonical_source__: ClassVar[type[Sites]] = Sites

    reduced_coords: Annotated[FracVector, httk.core.storage_markers.Shape(0, 3)]
    precision: fractions.Fraction | None

    def __post_init__(self) -> None:
        sites = Sites(self.reduced_coords, precision=self.precision)
        object.__setattr__(self, "reduced_coords", sites.reduced_coords)
        object.__setattr__(self, "precision", sites.precision)

    @classmethod
    def __httk_project__(cls, sites: Sites) -> Mapping[str, object]:
        return {"reduced_coords": sites.reduced_coords, "precision": sites.precision}

    def to_sites(self) -> Sites:
        return Sites(self.reduced_coords, precision=self.precision)


def _project_common(structure: Any) -> dict[str, object]:
    return {
        "cell": structure.cell,
        "species": structure.species,
        "molecular": structure.molecular,
        "assemblies": structure.assemblies,
        "assemblies_present": structure.assemblies is not None,
        "chemical_composition": structure.chemical_composition,
        "chemical_formula_descriptive": structure.chemical_formula_descriptive,
        "chemical_formula_hill": structure.chemical_formula_hill,
        "optimization_type": structure.optimization_type,
        "immutable_id": structure.immutable_id,
        "last_modified": structure.last_modified,
    }


def _normalize_common(record: Any, *, nsites: int) -> None:
    if not isinstance(record.cell, CellRecord):
        raise TypeError(f"{type(record).__name__} cell must be a CellRecord")
    species = tuple(record.species)
    if not all(isinstance(value, SpeciesRecord) for value in species):
        raise TypeError(f"{type(record).__name__} species must contain SpeciesRecord values")
    names = tuple(value.name for value in species)
    if len(names) != len(set(names)):
        raise ValueError(f"{type(record).__name__} species names must be unique")
    if not isinstance(record.molecular, bool):
        raise TypeError(f"{type(record).__name__} molecular must be a bool")
    if not isinstance(record.assemblies_present, bool):
        raise TypeError(f"{type(record).__name__} assemblies_present must be a bool")
    if record.assemblies_present:
        assemblies = tuple(record.assemblies or ())
    else:
        if record.assemblies not in (None, ()):
            raise ValueError(f"{type(record).__name__} assembly values require assemblies_present=True")
        assemblies = None
    if assemblies is not None:
        if not all(isinstance(value, AssemblyRecord) for value in assemblies):
            raise TypeError(f"{type(record).__name__} assemblies must contain AssemblyRecord values")
        validate_assemblies((value.to_assembly() for value in assemblies), nsites)
    if record.chemical_composition is not None and not isinstance(
        record.chemical_composition, ChemicalCompositionRecord
    ):
        raise TypeError(f"{type(record).__name__} chemical_composition must be a ChemicalCompositionRecord or None")
    for name in ("chemical_formula_descriptive", "chemical_formula_hill", "optimization_type"):
        value = getattr(record, name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{type(record).__name__} {name} must be a string or None")
    if record.immutable_id is not None and not isinstance(record.immutable_id, str):
        raise TypeError(f"{type(record).__name__} immutable_id must be a string or None")
    if record.last_modified is not None:
        if not isinstance(record.last_modified, datetime.datetime):
            raise TypeError(f"{type(record).__name__} last_modified must be a datetime or None")
        if record.last_modified.tzinfo is None or record.last_modified.utcoffset() is None:
            raise ValueError(f"{type(record).__name__} last_modified must include a timezone")
    object.__setattr__(record, "species", species)
    object.__setattr__(record, "assemblies", assemblies)


class _CommonConstructorValues(TypedDict):
    molecular: bool
    assemblies: tuple[Any, ...] | None
    chemical_composition: ChemicalComposition | None
    chemical_formula_descriptive: str | None
    chemical_formula_hill: str | None
    optimization_type: str | None
    immutable_id: str | None
    last_modified: datetime.datetime | None


def _common_constructor_values(record: Any) -> _CommonConstructorValues:
    return {
        "molecular": record.molecular,
        "assemblies": None if record.assemblies is None else tuple(value.to_assembly() for value in record.assemblies),
        "chemical_composition": None
        if record.chemical_composition is None
        else record.chemical_composition.to_composition(),
        "chemical_formula_descriptive": record.chemical_formula_descriptive,
        "chemical_formula_hill": record.chemical_formula_hill,
        "optimization_type": record.optimization_type,
        "immutable_id": record.immutable_id,
        "last_modified": record.last_modified,
    }


@dataclass(frozen=True)
class UnitcellStructureRecord:
    """Native durable backing for an explicit unit-cell structure."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_unitcell_structure_v1",
        identity_name="httk.atomistic.UnitcellStructureRecord",
        indexes=(("immutable_id",), ("last_modified",), ("optimization_type",)),
    )
    __httk_canonical_source__: ClassVar[type[Structure]] = Structure

    cell: CellRecord
    sites: SitesRecord
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    molecular: bool = False
    assemblies: tuple[AssemblyRecord, ...] | None = None
    assemblies_present: bool = False
    symmetry: SymmetryRecord | None = None
    chemical_composition: ChemicalCompositionRecord | None = None
    chemical_formula_descriptive: str | None = None
    chemical_formula_hill: str | None = None
    optimization_type: str | None = None
    immutable_id: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @property
    def type(self) -> str:
        return "structures"

    @property
    def id(self) -> str:
        return content_id(self)

    def __post_init__(self) -> None:
        if not isinstance(self.sites, SitesRecord):
            raise TypeError("UnitcellStructureRecord sites must be a SitesRecord")
        species_at_sites = tuple(self.species_at_sites)
        if not all(isinstance(value, str) for value in species_at_sites):
            raise TypeError("UnitcellStructureRecord species_at_sites must contain strings")
        _normalize_common(self, nsites=len(self.sites.to_sites()))
        known = {value.name for value in self.species}
        if len(species_at_sites) != len(self.sites.to_sites()):
            raise ValueError("UnitcellStructureRecord species_at_sites must match sites")
        if unknown := set(species_at_sites) - known:
            raise ValueError(f"UnitcellStructureRecord references unknown species: {sorted(unknown)!r}")
        if self.symmetry is not None and not isinstance(self.symmetry, SymmetryRecord):
            raise TypeError("UnitcellStructureRecord symmetry must be a SymmetryRecord or None")
        object.__setattr__(self, "species_at_sites", species_at_sites)
        self.to_structure()

    @classmethod
    def __httk_project__(cls, structure: Structure) -> Mapping[str, object]:
        values = _project_common(structure)
        values.update(
            {
                "sites": structure.sites,
                "species_at_sites": structure.species_at_sites,
                "symmetry": structure.symmetry,
            }
        )
        return values

    def to_structure(self) -> Structure:
        return Structure(
            self.cell.to_cell(),
            self.sites.to_sites(),
            tuple(value.to_species() for value in self.species),
            self.species_at_sites,
            symmetry=None if self.symmetry is None else self.symmetry.to_symmetry(),
            **_common_constructor_values(self),
        )


@dataclass(frozen=True)
class FundamentalDomainStructureRecord:
    """Native durable backing for a symmetry fundamental domain."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_fundamental_domain_structure_v1",
        identity_name="httk.atomistic.FundamentalDomainStructureRecord",
        indexes=(
            ("spacegroup_it_number",),
            ("immutable_id",),
            ("last_modified",),
            ("optimization_type",),
        ),
    )
    __httk_canonical_source__: ClassVar[type[FundamentalDomainStructure]] = FundamentalDomainStructure

    cell: CellRecord
    domain_sites: tuple[DomainSiteRecord, ...]
    species: tuple[SpeciesRecord, ...]
    spacegroup_it_number: int
    setting_transform: SettingTransformRecord
    coordinate_precision: fractions.Fraction | None
    molecular: bool = False
    assemblies: tuple[AssemblyRecord, ...] | None = None
    assemblies_present: bool = False
    chemical_composition: ChemicalCompositionRecord | None = None
    chemical_formula_descriptive: str | None = None
    chemical_formula_hill: str | None = None
    optimization_type: str | None = None
    immutable_id: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @property
    def type(self) -> str:
        return "structures"

    @property
    def id(self) -> str:
        return content_id(self)

    def __post_init__(self) -> None:
        domain_sites = tuple(self.domain_sites)
        if not all(isinstance(value, DomainSiteRecord) for value in domain_sites):
            raise TypeError("FundamentalDomainStructureRecord domain_sites must contain DomainSiteRecord values")
        if not isinstance(self.spacegroup_it_number, int) or isinstance(self.spacegroup_it_number, bool):
            raise TypeError("FundamentalDomainStructureRecord spacegroup_it_number must be an integer")
        if not 1 <= self.spacegroup_it_number <= 230:
            raise ValueError("FundamentalDomainStructureRecord spacegroup_it_number must be in [1, 230]")
        if not isinstance(self.setting_transform, SettingTransformRecord):
            raise TypeError("FundamentalDomainStructureRecord setting_transform must be a SettingTransformRecord")
        coordinate_precision = to_precision(self.coordinate_precision)
        object.__setattr__(self, "domain_sites", domain_sites)
        object.__setattr__(self, "coordinate_precision", coordinate_precision)
        _normalize_common(self, nsites=len(domain_sites))
        known = {value.name for value in self.species}
        if unknown := {value.species for value in domain_sites} - known:
            raise ValueError(f"FundamentalDomainStructureRecord references unknown species: {sorted(unknown)!r}")
        self.to_structure()

    @classmethod
    def __httk_project__(cls, structure: FundamentalDomainStructure) -> Mapping[str, object]:
        values = _project_common(structure)
        values.update(
            {
                "domain_sites": structure.domain_sites,
                "spacegroup_it_number": structure.spacegroup.it_number,
                "setting_transform": structure.transform,
                "coordinate_precision": structure.coordinate_precision,
            }
        )
        return values

    def _to_structure_type(
        self, structure_type: builtins.type[FundamentalDomainStructure]
    ) -> FundamentalDomainStructure:
        from .spacegroup import Spacegroup

        return structure_type(
            self.cell.to_cell(),
            Spacegroup.standard(self.spacegroup_it_number),
            tuple(
                ASUSite(
                    value.wyckoff,
                    FracVector.create(value.free_parameters),
                    value.species,
                    FracVector.create(value.representative) if value.representative_present else None,
                )
                for value in self.domain_sites
            ),
            tuple(value.to_species() for value in self.species),
            self.setting_transform.to_transform(),
            self.coordinate_precision,
            **_common_constructor_values(self),
        )

    def to_structure(self) -> FundamentalDomainStructure:
        return self._to_structure_type(FundamentalDomainStructure)


@dataclass(frozen=True)
class ASUStructureRecord(FundamentalDomainStructureRecord):
    """Native durable backing for an asserted asymmetric unit."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_asu_structure_v1",
        identity_name="httk.atomistic.ASUStructureRecord",
        indexes=(
            ("spacegroup_it_number",),
            ("immutable_id",),
            ("last_modified",),
            ("optimization_type",),
        ),
    )
    __httk_canonical_source__: ClassVar[type[ASUStructure]] = ASUStructure

    @property
    def type(self) -> str:
        return "structures"

    @property
    def id(self) -> str:
        return content_id(self)

    def to_structure(self) -> ASUStructure:
        return cast(ASUStructure, self._to_structure_type(ASUStructure))

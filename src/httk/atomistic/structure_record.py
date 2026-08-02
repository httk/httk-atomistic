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
from .composition import Assembly, ChemicalComposition, CompositionResult, validate_assemblies
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
    "NormalizedCompositionAmountRecord",
    "NormalizedCompositionRecord",
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
    """The storable frozen snapshot of an atomistic :class:`~httk.atomistic.Species`."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_species_record")
    __httk_canonical_source__: ClassVar[type[Species]] = Species

    name: str
    chemical_symbols: tuple[str, ...]
    concentration: tuple[fractions.Fraction, ...]
    mass: tuple[float, ...] | None = None
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None
    concentration_precision: tuple[fractions.Fraction, ...] | None = None

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
        raw_precision = None if self.concentration_precision is None else tuple(self.concentration_precision)
        if raw_precision is not None:
            if len(raw_precision) != len(concentration):
                raise ValueError("SpeciesRecord concentration_precision must match concentration")
            converted_precision: list[fractions.Fraction] = []
            for value in raw_precision:
                central, _ = as_fraction(value, field="SpeciesRecord concentration precision")
                if central < 0:
                    raise ValueError("SpeciesRecord concentration precision cannot be negative")
                converted_precision.append(central)
            object.__setattr__(self, "concentration_precision", tuple(converted_precision))

        if (self.attached is None) != (self.nattached is None):
            raise ValueError("SpeciesRecord attached and nattached must be provided together")

        for value_name, values in (
            ("concentration", self.concentration),
            ("mass", self.mass or ()),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"SpeciesRecord {value_name} values must be finite floats")

        try:
            Species(
                name=self.name,
                chemical_symbols=self.chemical_symbols,
                concentration=self.concentration,
                mass=self.mass,
                original_name=self.original_name,
                attached=self.attached,
                nattached=self.nattached,
                concentration_precision=(
                    None
                    if self.concentration_precision is None
                    else tuple(None if value == 0 else value for value in self.concentration_precision)
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("SpeciesRecord fields do not describe a valid Species") from error

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
            "concentration_precision": (
                None if not present else tuple(value or fractions.Fraction() for value in precision)
            ),
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
            concentration_precision=(
                None
                if all(value is None for value in species.concentration_precision or ())
                else tuple(value or fractions.Fraction() for value in species.concentration_precision or ())
            ),
        )

    def to_species(self) -> Species:
        """Reconstruct the domain :class:`~httk.atomistic.Species` from this snapshot."""
        return Species(
            name=self.name,
            chemical_symbols=self.chemical_symbols,
            concentration=self.concentration,
            mass=self.mass,
            original_name=self.original_name,
            attached=self.attached,
            nattached=self.nattached,
            concentration_precision=(
                None
                if self.concentration_precision is None
                else tuple(None if value == 0 else value for value in self.concentration_precision)
            ),
        )


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
    group_probabilities_precision: tuple[fractions.Fraction, ...] | None = None

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not all(type(group) is AssemblyGroupRecord for group in groups):
            raise TypeError("AssemblyRecord groups must contain AssemblyGroupRecord values")
        probabilities = tuple(
            as_fraction(value, field="AssemblyRecord group probability")[0] for value in self.group_probabilities
        )
        precision = (
            None
            if self.group_probabilities_precision is None
            else tuple(
                as_fraction(value, field="AssemblyRecord group probability precision")[0]
                for value in self.group_probabilities_precision
            )
        )
        if not groups or len(groups) != len(probabilities):
            raise ValueError("AssemblyRecord groups and probabilities must have matching non-empty lengths")
        if precision is not None and (len(precision) != len(probabilities) or any(value < 0 for value in precision)):
            raise ValueError("AssemblyRecord precision must match probabilities and be non-negative")
        assembly = Assembly(
            tuple(group.sites for group in groups),
            probabilities,
            None if precision is None else tuple(None if value == 0 else value for value in precision),
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
            "group_probabilities_precision": (
                None if not present else tuple(value or fractions.Fraction() for value in precision)
            ),
        }

    @classmethod
    def from_assembly(cls, assembly: Assembly) -> "AssemblyRecord":
        precision = assembly.group_probabilities_precision or ()
        present = not all(value is None for value in precision)
        return cls(
            tuple(AssemblyGroupRecord(group) for group in assembly.sites_in_groups),
            assembly.group_probabilities,
            None if not present else tuple(value or fractions.Fraction() for value in precision),
        )

    def to_assembly(self) -> Assembly:
        return Assembly(
            tuple(group.sites for group in self.groups),
            self.group_probabilities,
            None
            if self.group_probabilities_precision is None
            else tuple(None if value == 0 else value for value in self.group_probabilities_precision),
        )


@dataclass(frozen=True)
class DomainSiteRecord:
    """Exact Wyckoff/free-parameter site with its retained representative."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_domain_site_record")
    __httk_canonical_source__: ClassVar[type[ASUSite]] = ASUSite

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] | None = None

    @classmethod
    def __httk_project__(cls, site: ASUSite) -> Mapping[str, object]:
        return {
            "wyckoff": site.wyckoff,
            "free_parameters": tuple(site.free_params.to_fractions()),
            "species": site.species,
            "representative": None if site.representative is None else tuple(site.representative.to_fractions()),
        }

    def __post_init__(self) -> None:
        free = tuple(as_fraction(value, field="DomainSiteRecord free parameter")[0] for value in self.free_parameters)
        representative = (
            None
            if self.representative is None
            else tuple(as_fraction(value, field="DomainSiteRecord representative")[0] for value in self.representative)
        )
        if not isinstance(self.wyckoff, str) or len(self.wyckoff) != 1:
            raise ValueError("DomainSiteRecord wyckoff must be a single letter")
        if not isinstance(self.species, str) or not self.species:
            raise ValueError("DomainSiteRecord species must be non-empty")
        if representative is not None and len(representative) != 3:
            raise ValueError("DomainSiteRecord representative must have exactly three values")
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

    def __post_init__(self) -> None:
        value = StructureSymmetry(
            self.space_group_it_number,
            self.space_group_symbol_hall,
            self.space_group_symbol_hermann_mauguin,
            self.space_group_symbol_hermann_mauguin_extended,
            None
            if self.space_group_symmetry_operations_xyz is None
            else tuple(self.space_group_symmetry_operations_xyz),
            None if self.wyckoff_positions is None else tuple(self.wyckoff_positions),
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
        )

    def to_symmetry(self) -> StructureSymmetry:
        return StructureSymmetry(
            self.space_group_it_number,
            self.space_group_symbol_hall,
            self.space_group_symbol_hermann_mauguin,
            self.space_group_symbol_hermann_mauguin_extended,
            self.space_group_symmetry_operations_xyz,
            self.wyckoff_positions,
        )


@dataclass(frozen=True)
class CompositionAmountRecord:
    """One exact declared element amount and its optional precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_composition_amount_record")
    __httk_canonical_source__: ClassVar[type[tuple]] = tuple

    element: str
    amount: fractions.Fraction
    precision: fractions.Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, str) or not self.element:
            raise TypeError("CompositionAmountRecord element must be a non-empty string")
        amount, _ = as_fraction(self.amount, field="CompositionAmountRecord amount")
        if amount <= 0:
            raise ValueError("CompositionAmountRecord amount must be positive")
        precision = None
        if self.precision is not None:
            precision, _ = as_fraction(self.precision, field="CompositionAmountRecord precision")
            if precision <= 0:
                raise ValueError("CompositionAmountRecord precision must be positive or None")
        ChemicalComposition({self.element: amount})
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "precision", precision)

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


@dataclass(frozen=True)
class NormalizedCompositionRecord:
    """Authoritative exact elemental composition projected from one structure.

    This relation is semantic normalized data, not a rendered-formula cache. It
    retains each exact central element amount together with its source precision,
    and makes the same complete-composition facts available to every durable
    structure backing for response construction and exact filtering.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_normalized_composition_record",
        identity_name="httk.atomistic.NormalizedCompositionRecord",
    )
    __httk_canonical_source__: ClassVar[type[CompositionResult]] = CompositionResult

    amounts: tuple["NormalizedCompositionAmountRecord", ...]
    complete: bool

    @classmethod
    def __httk_project__(cls, result: CompositionResult) -> Mapping[str, object]:
        precision = dict(result.uncertainties)
        total = sum((amount for _, amount in result.amounts), fractions.Fraction())
        return {
            "amounts": tuple(
                (element, amount / total, amount, precision[element]) for element, amount in result.amounts
            ),
            "complete": result.complete,
        }

    def __post_init__(self) -> None:
        amounts = tuple(self.amounts)
        if not all(isinstance(value, NormalizedCompositionAmountRecord) for value in amounts):
            raise TypeError("NormalizedCompositionRecord amounts must contain NormalizedCompositionAmountRecord values")
        if not isinstance(self.complete, bool):
            raise TypeError("NormalizedCompositionRecord complete must be a bool")
        elements = tuple(value.element for value in amounts)
        if elements != tuple(sorted(elements)) or len(elements) != len(set(elements)):
            raise ValueError("NormalizedCompositionRecord amounts must be uniquely ordered by element")
        if amounts:
            total = sum((value.amount for value in amounts), fractions.Fraction())
            if sum((value.ratio for value in amounts), fractions.Fraction()) != 1:
                raise ValueError("NormalizedCompositionRecord ratios must sum exactly to one")
            if any(value.ratio != value.amount / total for value in amounts):
                raise ValueError("NormalizedCompositionRecord ratios must exactly normalize their amounts")
        object.__setattr__(self, "amounts", amounts)

    @classmethod
    def from_result(cls, result: CompositionResult) -> "NormalizedCompositionRecord":
        precision = dict(result.uncertainties)
        total = sum((amount for _, amount in result.amounts), fractions.Fraction())
        return cls(
            tuple(
                NormalizedCompositionAmountRecord(element, amount / total, amount, precision[element])
                for element, amount in result.amounts
            ),
            result.complete,
        )

    def to_result(self) -> CompositionResult:
        amounts = tuple((value.element, value.amount) for value in self.amounts)
        uncertainties = tuple((value.element, value.precision) for value in self.amounts)
        exact = all(value.precision is None for value in self.amounts)
        return CompositionResult(
            amounts,
            uncertainties,
            self.complete,
            exact,
            True,
            "exact" if exact else "within_precision",
        )


@dataclass(frozen=True)
class NormalizedCompositionAmountRecord:
    """One exact normalized central ratio with the source amount and precision retained."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_normalized_composition_amount_record",
        identity_name="httk.atomistic.NormalizedCompositionAmountRecord",
    )
    __httk_canonical_source__: ClassVar[type[tuple]] = tuple

    element: str
    ratio: fractions.Fraction
    amount: fractions.Fraction
    precision: fractions.Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, str) or not self.element:
            raise TypeError("NormalizedCompositionAmountRecord element must be a non-empty string")
        ratio, _ = as_fraction(self.ratio, field="NormalizedCompositionAmountRecord ratio")
        amount, _ = as_fraction(self.amount, field="NormalizedCompositionAmountRecord amount")
        if not 0 < ratio <= 1:
            raise ValueError("NormalizedCompositionAmountRecord ratio must be in (0, 1]")
        if amount <= 0:
            raise ValueError("NormalizedCompositionAmountRecord amount must be positive")
        precision = None
        if self.precision is not None:
            precision, _ = as_fraction(self.precision, field="NormalizedCompositionAmountRecord precision")
            if precision <= 0:
                raise ValueError("NormalizedCompositionAmountRecord precision must be positive or None")
        ChemicalComposition({self.element: amount})
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "precision", precision)

    @classmethod
    def __httk_project__(cls, amount: tuple[Any, ...]) -> Mapping[str, object]:
        if len(amount) != 4:
            raise ValueError("normalized composition amount projection requires element, ratio, amount, and precision")
        return {"element": amount[0], "ratio": amount[1], "amount": amount[2], "precision": amount[3]}


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
        "normalized_composition": structure.composition,
        "molecular": structure.molecular,
        "assemblies": structure.assemblies,
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
    if not isinstance(record.normalized_composition, NormalizedCompositionRecord):
        raise TypeError(f"{type(record).__name__} normalized_composition must be a NormalizedCompositionRecord")
    names = tuple(value.name for value in species)
    if len(names) != len(set(names)):
        raise ValueError(f"{type(record).__name__} species names must be unique")
    if not isinstance(record.molecular, bool):
        raise TypeError(f"{type(record).__name__} molecular must be a bool")
    assemblies = None if record.assemblies is None else tuple(record.assemblies)
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


def _validate_normalized_composition(record: Any) -> None:
    """Reject a root record whose central relation contradicts its native fields.

    The structure's exact composition is authoritative because it deduplicates
    overlap across distinct stored domain sites. A per-orbit multiplicity
    shortcut would overcount a valid duplicate orbit.
    """
    expected = NormalizedCompositionRecord.from_result(record.to_structure().composition)
    if record.normalized_composition != expected:
        raise ValueError(
            f"{type(record).__name__} normalized_composition contradicts the composition reconstructed from native fields"
        )


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
    normalized_composition: NormalizedCompositionRecord
    molecular: bool = False
    assemblies: tuple[AssemblyRecord, ...] | None = None
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
        _validate_normalized_composition(self)

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
    normalized_composition: NormalizedCompositionRecord
    molecular: bool = False
    assemblies: tuple[AssemblyRecord, ...] | None = None
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
        _validate_normalized_composition(self)

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
                    None if value.representative is None else FracVector.create(value.representative),
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


from .stored_structure_properties import attach_structure_property_projections

attach_structure_property_projections(
    UnitcellStructureRecord,
    FundamentalDomainStructureRecord,
    ASUStructureRecord,
)

"""Frozen storage records for complete atomistic structures."""

import datetime
import fractions
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, TypedDict, cast

import httk.core.storage.markers
from httk.core import (
    FracVector,
    SurdScalar,
    SurdVector,
)
from httk.core.storage import IdentitySkip, StorageInfo, content_id

from ._composition_values import as_fraction
from ._vector_guards import to_periodicity, to_precision
from .asu_structure import ASUStructure, FundamentalDomainStructure, WyckoffSite
from .cell import Cell
from .composition import Assembly, ChemicalComposition, CompositionResult, validate_assemblies
from .setting_transform import SettingTransform
from .sites import Sites
from .species import Species
from .structure_semantics import StructureSymmetry
from .unitcell_structure import UnitcellStructure

__all__ = [
    "ASUStructureRecord",
    "AssemblyGroupRecord",
    "AssemblyRecord",
    "CellRecord",
    "ChemicalCompositionRecord",
    "CompositionAmountRecord",
    "FundamentalDomainStructureRecord",
    "NormalizedCompositionAmountRecord",
    "NormalizedCompositionRecord",
    "SettingTransformRecord",
    "SitesRecord",
    "SpeciesRecord",
    "SymmetryRecord",
    "UnitcellStructureRecord",
    "WyckoffSiteRecord",
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
    """The storable frozen snapshot of an atomistic :class:`~httk.atomistic.Species`; hand-built records are shape-checked and semantically validated at storage or explicitly."""

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

    @classmethod
    def __httk_validate__(cls, record: "SpeciesRecord") -> None:
        _validate_species_record(record)

    def __post_init__(self) -> None:
        symbols = tuple(self.chemical_symbols)
        if not isinstance(self.name, str):
            raise TypeError("SpeciesRecord name must be a string")
        if not symbols or not all(isinstance(value, str) for value in symbols):
            raise TypeError("SpeciesRecord chemical_symbols must contain non-empty strings")
        object.__setattr__(self, "chemical_symbols", symbols)
        concentration = tuple(
            as_fraction(value, field="SpeciesRecord concentration")[0] for value in self.concentration
        )
        if len(concentration) != len(symbols):
            raise ValueError("SpeciesRecord concentration must match chemical_symbols")
        if any(value < 0 or value > 1 for value in concentration):
            raise ValueError("SpeciesRecord concentration values must be in [0, 1]")
        object.__setattr__(self, "concentration", concentration)
        if self.mass is not None:
            mass = tuple(self.mass)
            if len(mass) != len(symbols):
                raise ValueError("SpeciesRecord mass must match chemical_symbols")
            object.__setattr__(self, "mass", mass)
        if self.original_name is not None and not isinstance(self.original_name, str):
            raise TypeError("SpeciesRecord original_name must be a string or None")
        if self.attached is not None:
            attached = tuple(self.attached)
            if not all(isinstance(value, str) for value in attached):
                raise TypeError("SpeciesRecord attached must contain strings")
            object.__setattr__(self, "attached", attached)
        if self.nattached is not None:
            nattached = tuple(self.nattached)
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in nattached):
                raise ValueError("SpeciesRecord nattached must contain non-negative integers")
            object.__setattr__(self, "nattached", nattached)
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
        if self.attached is not None and len(self.attached) != len(self.nattached or ()):
            raise ValueError("SpeciesRecord attached and nattached must have matching lengths")

        for value_name, values in (
            ("concentration", self.concentration),
            ("mass", self.mass or ()),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"SpeciesRecord {value_name} values must be finite floats")

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

    @property
    def sites_in_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(group.sites for group in self.groups)

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


@dataclass(frozen=True)
class WyckoffSiteRecord:
    """Exact Wyckoff/free-parameter site with its retained representative.

    The owning record's ``domain_sites`` field is storage-visible and deliberately unchanged.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_domain_site_record")
    __httk_canonical_source__: ClassVar[type[WyckoffSite]] = WyckoffSite

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] | None = None

    @classmethod
    def __httk_project__(cls, site: WyckoffSite) -> Mapping[str, object]:
        return {
            "wyckoff": site.wyckoff,
            "free_parameters": tuple(site.free_params.to_fractions()),
            "species": site.species,
            "representative": None if site.representative is None else tuple(site.representative.to_fractions()),
        }

    def __post_init__(self) -> None:
        free = tuple(as_fraction(value, field="WyckoffSiteRecord free parameter")[0] for value in self.free_parameters)
        representative = (
            None
            if self.representative is None
            else tuple(as_fraction(value, field="WyckoffSiteRecord representative")[0] for value in self.representative)
        )
        if not isinstance(self.wyckoff, str) or len(self.wyckoff) != 1:
            raise ValueError("WyckoffSiteRecord wyckoff must be a single letter")
        if not isinstance(self.species, str) or not self.species:
            raise ValueError("WyckoffSiteRecord species must be non-empty")
        if representative is not None and len(representative) != 3:
            raise ValueError("WyckoffSiteRecord representative must have exactly three values")
        object.__setattr__(self, "free_parameters", free)
        object.__setattr__(self, "representative", representative)


@dataclass(frozen=True)
class SettingTransformRecord:
    """Exact durable standard-to-own setting transform; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_setting_transform_record")
    __httk_canonical_source__: ClassVar[type[SettingTransform]] = SettingTransform

    matrix: Annotated[FracVector, httk.core.storage.markers.Shape(3, 3)]
    vector: tuple[fractions.Fraction, ...]
    hall_entry: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "SettingTransformRecord") -> None:
        _validate_setting_transform_record(record)

    def __post_init__(self) -> None:
        matrix = FracVector.create(self.matrix)
        vector = tuple(as_fraction(value, field="SettingTransformRecord vector")[0] for value in self.vector)
        if matrix.dim != (3, 3):
            raise ValueError(f"AffineOperation matrix must be 3x3, got dim {matrix.dim}")
        if len(vector) != 3:
            raise ValueError(f"AffineOperation vector must have 3 elements, got dim {(len(vector),)}")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "vector", vector)
        if self.hall_entry is not None and not isinstance(self.hall_entry, str):
            raise TypeError("SettingTransformRecord hall_entry must be a string or None")

    @classmethod
    def __httk_project__(cls, transform: SettingTransform) -> Mapping[str, object]:
        return {
            "matrix": transform.matrix,
            "vector": tuple(transform.vector.to_fractions()),
            "hall_entry": transform.hall_entry,
        }


@dataclass(frozen=True)
class SymmetryRecord:
    """Typed optional symmetry metadata of an ordinary unit-cell structure; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_symmetry_v3")
    __httk_canonical_source__: ClassVar[type[StructureSymmetry]] = StructureSymmetry

    space_group_it_number: int | None = None
    space_group_symbol_hall: str | None = None
    space_group_symbol_hermann_mauguin: str | None = None
    space_group_symbol_hermann_mauguin_extended: str | None = None
    space_group_symmetry_operations_xyz: tuple[str, ...] | None = None
    wyckoff_positions: tuple[str, ...] | None = None

    @classmethod
    def __httk_validate__(cls, record: "SymmetryRecord") -> None:
        _validate_symmetry_record(record)

    def __post_init__(self) -> None:
        if self.space_group_it_number is not None and (
            not isinstance(self.space_group_it_number, int)
            or isinstance(self.space_group_it_number, bool)
            or not 1 <= self.space_group_it_number <= 230
        ):
            raise ValueError("space_group_it_number must be an integer in [1, 230]")
        for name in (
            "space_group_symbol_hall",
            "space_group_symbol_hermann_mauguin",
            "space_group_symbol_hermann_mauguin_extended",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{name} must be a non-empty string or None")
        object.__setattr__(
            self,
            "space_group_symmetry_operations_xyz",
            None
            if self.space_group_symmetry_operations_xyz is None
            else tuple(self.space_group_symmetry_operations_xyz),
        )
        object.__setattr__(
            self,
            "wyckoff_positions",
            None if self.wyckoff_positions is None else tuple(self.wyckoff_positions),
        )
        for name in ("space_group_symmetry_operations_xyz", "wyckoff_positions"):
            values = getattr(self, name)
            if values is not None and not all(isinstance(value, str) for value in values):
                raise TypeError(f"{name} must contain strings or be None")

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
        value = _chemical_composition_from_record(self)
        object.__setattr__(
            self,
            "amounts",
            tuple(
                CompositionAmountRecord(element, amount, dict(value.amounts_precision)[element])
                for element, amount in value.amounts
            ),
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
    """Exact durable cell basis, precision, and periodicity; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_cell_v1",
        identity_name="httk.atomistic.CellRecord",
    )
    __httk_canonical_source__: ClassVar[type[Cell]] = Cell

    basis: tuple[SurdScalar, ...]
    precision: fractions.Fraction | None
    periodicity: tuple[bool, ...]

    @classmethod
    def __httk_validate__(cls, record: "CellRecord") -> None:
        _validate_cell_record(record)

    def __post_init__(self) -> None:
        basis = tuple(self.basis)
        periodicity = tuple(self.periodicity)
        if len(basis) != 9 or not all(isinstance(value, SurdScalar) for value in basis):
            raise ValueError("CellRecord basis must contain exactly nine SurdScalar values")
        precision = to_precision(self.precision)
        periodicity = to_periodicity(periodicity)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "periodicity", periodicity)

    @classmethod
    def __httk_project__(cls, cell: Cell) -> Mapping[str, object]:
        return {
            "basis": tuple(_extract_surd_scalar(cell.basis, (row, column)) for row in range(3) for column in range(3)),
            "precision": cell.precision,
            "periodicity": cell.periodicity,
        }


@dataclass(frozen=True)
class SitesRecord:
    """Exact durable reduced coordinates and their stated precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_sites_v1",
        identity_name="httk.atomistic.SitesRecord",
    )
    __httk_canonical_source__: ClassVar[type[Sites]] = Sites

    reduced_coords: Annotated[FracVector, httk.core.storage.markers.Shape(0, 3)]
    precision: fractions.Fraction | None

    def __post_init__(self) -> None:
        sites = Sites(self.reduced_coords, precision=self.precision)
        object.__setattr__(self, "reduced_coords", sites.reduced_coords)
        object.__setattr__(self, "precision", sites.precision)

    @classmethod
    def __httk_project__(cls, sites: Sites) -> Mapping[str, object]:
        return {"reduced_coords": sites.reduced_coords, "precision": sites.precision}


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
        validate_assemblies((_assembly_from_record(value) for value in assemblies), nsites)
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
    expected = _normalized_composition_record_from_result(_structure_from_record(record).composition)
    if record.normalized_composition != expected:
        raise ValueError(
            f"{type(record).__name__} normalized_composition contradicts the composition reconstructed from native fields"
        )


def _validate_species_record(record: SpeciesRecord) -> None:
    try:
        _species_from_record(record)
    except (TypeError, ValueError) as error:
        raise ValueError("SpeciesRecord fields do not describe a valid Species") from error


def _validate_cell_record(record: CellRecord) -> None:
    Cell(_basis_vector(record.basis), precision=record.precision, periodicity=record.periodicity)


def _validate_symmetry_record(record: SymmetryRecord) -> None:
    _symmetry_from_record(record)


def _validate_setting_transform_record(record: SettingTransformRecord) -> None:
    _setting_transform_from_record(record)


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
        "assemblies": None
        if record.assemblies is None
        else tuple(_assembly_from_record(value) for value in record.assemblies),
        "chemical_composition": None
        if record.chemical_composition is None
        else _chemical_composition_from_record(record.chemical_composition),
        "chemical_formula_descriptive": record.chemical_formula_descriptive,
        "chemical_formula_hill": record.chemical_formula_hill,
        "optimization_type": record.optimization_type,
        "immutable_id": record.immutable_id,
        "last_modified": record.last_modified,
    }


@dataclass(frozen=True)
class UnitcellStructureRecord:
    """Native durable backing for an explicit unit-cell structure; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_unitcell_structure_v1",
        identity_name="httk.atomistic.UnitcellStructureRecord",
        indexes=(("immutable_id",), ("last_modified",), ("optimization_type",)),
    )
    __httk_canonical_source__: ClassVar[type[UnitcellStructure]] = UnitcellStructure

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

    @classmethod
    def __httk_validate__(cls, record: "UnitcellStructureRecord") -> None:
        validate_structure_record(record)

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
        _normalize_common(self, nsites=len(self.sites.reduced_coords))
        known = {value.name for value in self.species}
        if len(species_at_sites) != len(self.sites.reduced_coords):
            raise ValueError("UnitcellStructureRecord species_at_sites must match sites")
        if unknown := set(species_at_sites) - known:
            raise ValueError(f"UnitcellStructureRecord references unknown species: {sorted(unknown)!r}")
        if self.symmetry is not None and not isinstance(self.symmetry, SymmetryRecord):
            raise TypeError("UnitcellStructureRecord symmetry must be a SymmetryRecord or None")
        object.__setattr__(self, "species_at_sites", species_at_sites)

    @classmethod
    def __httk_project__(cls, structure: UnitcellStructure) -> Mapping[str, object]:
        values = _project_common(structure)
        values.update(
            {
                "sites": structure.sites,
                "species_at_sites": structure.species_at_sites,
                "symmetry": structure.symmetry,
            }
        )
        return values


@dataclass(frozen=True)
class FundamentalDomainStructureRecord:
    """Native durable backing for a symmetry fundamental domain; hand-built records are shape-checked and semantically validated at storage or explicitly."""

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
    domain_sites: tuple[WyckoffSiteRecord, ...]
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

    @classmethod
    def __httk_validate__(cls, record: "FundamentalDomainStructureRecord") -> None:
        validate_structure_record(record)

    @property
    def type(self) -> str:
        return "structures"

    @property
    def id(self) -> str:
        return content_id(self)

    def __post_init__(self) -> None:
        domain_sites = tuple(self.domain_sites)
        if not all(isinstance(value, WyckoffSiteRecord) for value in domain_sites):
            raise TypeError("FundamentalDomainStructureRecord domain_sites must contain WyckoffSiteRecord values")
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


@dataclass(frozen=True)
class ASUStructureRecord(FundamentalDomainStructureRecord):
    """Native durable backing for an asserted asymmetric unit; hand-built records are shape-checked and semantically validated at storage or explicitly."""

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

    @classmethod
    def __httk_validate__(cls, record: "FundamentalDomainStructureRecord") -> None:
        validate_structure_record(record)

    @property
    def type(self) -> str:
        return "structures"

    @property
    def id(self) -> str:
        return content_id(self)


def _concentration_precision_from_record(
    precision: tuple[fractions.Fraction, ...] | None,
) -> tuple[fractions.Fraction | None, ...] | None:
    return None if precision is None else tuple(None if value == 0 else value for value in precision)


def _species_from_record(record: SpeciesRecord) -> Species:
    return Species(
        name=record.name,
        chemical_symbols=record.chemical_symbols,
        concentration=record.concentration,
        mass=record.mass,
        original_name=record.original_name,
        attached=record.attached,
        nattached=record.nattached,
        concentration_precision=_concentration_precision_from_record(record.concentration_precision),
    )


def _assembly_from_record(record: AssemblyRecord) -> Assembly:
    return Assembly(
        tuple(group.sites for group in record.groups),
        record.group_probabilities,
        None
        if record.group_probabilities_precision is None
        else tuple(None if value == 0 else value for value in record.group_probabilities_precision),
    )


def _setting_transform_from_record(record: SettingTransformRecord) -> SettingTransform:
    return SettingTransform(record.matrix, record.vector, hall_entry=record.hall_entry)


def _symmetry_from_record(record: SymmetryRecord) -> StructureSymmetry:
    return StructureSymmetry(
        record.space_group_it_number,
        record.space_group_symbol_hall,
        record.space_group_symbol_hermann_mauguin,
        record.space_group_symbol_hermann_mauguin_extended,
        record.space_group_symmetry_operations_xyz,
        record.wyckoff_positions,
    )


def _chemical_composition_from_record(record: ChemicalCompositionRecord) -> ChemicalComposition:
    return ChemicalComposition(
        {value.element: value.amount for value in record.amounts},
        cast(Any, record.mode),
        {value.element: value.precision for value in record.amounts if value.precision is not None},
    )


def _composition_result_from_record(record: NormalizedCompositionRecord) -> CompositionResult:
    amounts = tuple((value.element, value.amount) for value in record.amounts)
    uncertainties = tuple((value.element, value.precision) for value in record.amounts)
    exact = all(value.precision is None for value in record.amounts)
    return CompositionResult(
        amounts, uncertainties, record.complete, exact, True, "exact" if exact else "within_precision"
    )


def _normalized_composition_record_from_result(result: CompositionResult) -> NormalizedCompositionRecord:
    values = NormalizedCompositionRecord.__httk_project__(result)
    amounts = tuple(NormalizedCompositionAmountRecord(*item) for item in values["amounts"])  # type: ignore[arg-type]
    return NormalizedCompositionRecord(amounts, values["complete"])  # type: ignore[arg-type]


def _cell_from_record(record: CellRecord) -> Cell:
    return Cell(_basis_vector(record.basis), precision=record.precision, periodicity=record.periodicity)


def _sites_from_record(record: SitesRecord) -> Sites:
    return Sites(record.reduced_coords, precision=record.precision)


def _domain_structure_from_record(
    record: FundamentalDomainStructureRecord | ASUStructureRecord,
) -> FundamentalDomainStructure:
    from .spacegroup import Spacegroup

    structure_type = ASUStructure if isinstance(record, ASUStructureRecord) else FundamentalDomainStructure
    return structure_type(
        _cell_from_record(record.cell),
        Spacegroup.standard(record.spacegroup_it_number),
        tuple(
            WyckoffSite(
                value.wyckoff,
                FracVector.create(value.free_parameters),
                value.species,
                None if value.representative is None else FracVector.create(value.representative),
            )
            for value in record.domain_sites
        ),
        tuple(_species_from_record(value) for value in record.species),
        _setting_transform_from_record(record.setting_transform),
        record.coordinate_precision,
        **_common_constructor_values(record),
    )


def _structure_from_record(
    record: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord,
) -> Any:
    if not isinstance(record, UnitcellStructureRecord):
        return _domain_structure_from_record(record)
    return UnitcellStructure(
        _cell_from_record(record.cell),
        _sites_from_record(record.sites),
        tuple(_species_from_record(value) for value in record.species),
        record.species_at_sites,
        symmetry=None if record.symmetry is None else _symmetry_from_record(record.symmetry),
        **_common_constructor_values(record),
    )


def validate_structure_record(
    record: UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord,
) -> None:
    """Validate a hand-built root record by rebuilding its native structure semantics."""
    if type(record) not in (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord):
        raise TypeError("validate_structure_record expects an exact root structure record")
    _validate_normalized_composition(record)


from .stored_structure_properties import attach_structure_property_projections

attach_structure_property_projections(
    UnitcellStructureRecord,
    FundamentalDomainStructureRecord,
    ASUStructureRecord,
)

"""Frozen storage records for complete atomistic structures."""

import datetime
import fractions
import itertools
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any, ClassVar, TypedDict, cast

import httk.core.storage.markers
from httk.core import (
    FracVector,
    SurdScalar,
    SurdVector,
)
from httk.core.storage import IdentitySkip, StorageInfo, content_id

from httk.atomistic._composition_values import as_fraction
from httk.atomistic.composition import Assembly, ChemicalComposition, CompositionResult, validate_assemblies
from httk.atomistic.models._vector_guards import to_periodicity, to_precision
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structure.semantics import StructureSymmetry
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.api import TrajectoryAPI
from httk.atomistic.symmetry.setting_transform import SettingTransform

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
    "ObservableSummaryRecord",
    "SettingTransformRecord",
    "SitesRecord",
    "SpeciesConstituentRecord",
    "SpeciesRecord",
    "SymmetryRecord",
    "TrajectoryRecord",
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


_SITE_MOMENT_KINDS = frozenset(("cartesian", "crystalaxis", "collinear"))


def _moment_components(moment: Any) -> tuple[SurdScalar, ...]:
    if moment.kind == "collinear":
        return tuple(SurdVector.create(value)._as_scalar() for value in moment.collinear_moments.to_fractions())
    values = moment.cartesian_moments if moment.kind == "cartesian" else moment.crystalaxis_moments
    return tuple(values._element(index) for index in itertools.product(*[range(size) for size in values.dim]))


def _moment_from_record(
    kind: str | None,
    components: tuple[SurdScalar, ...] | None,
    precision: fractions.Fraction | None,
    cell: Cell | None = None,
) -> Any:
    if kind is None:
        return None
    assert components is not None
    if kind == "collinear":
        if any(not value.is_rational for value in components):
            raise ValueError("collinear moment components must be rational")
        return CollinearSiteMoments(tuple(value.to_fractions_approx() for value in components), precision=precision)
    rows = [list(components[offset : offset + 3]) for offset in range(0, len(components), 3)]
    vector = SurdVector._from_scalar_grid(rows, (len(rows), 3))
    if kind == "cartesian":
        return CartesianSiteMoments(vector, precision=precision)
    assert cell is not None
    return CrystalAxisSiteMoments(vector, cell, precision=precision)


def _validate_moment_fields(record: Any, record_name: str, *, nsites: int | None = None) -> None:
    kind = record.site_moments_kind if hasattr(record, "site_moments_kind") else record.moment_kind
    components = record.site_moments if hasattr(record, "site_moments") else record.moment
    precision_name = "site_moments_precision" if hasattr(record, "site_moments_precision") else "moment_precision"
    precision = to_precision(getattr(record, precision_name))
    if kind is None:
        if components is not None or precision is not None:
            raise ValueError(f"{record_name} moment components and precision require a moment kind")
        return
    if kind not in _SITE_MOMENT_KINDS:
        raise ValueError(f"{record_name} moment kind must be one of {sorted(_SITE_MOMENT_KINDS)!r}")
    if components is None:
        raise ValueError(f"{record_name} moment kind requires components")
    components = tuple(
        value if isinstance(value, SurdScalar) else SurdVector.create(value)._as_scalar() for value in components
    )
    expected = (
        (nsites if kind == "collinear" else 3 * nsites) if nsites is not None else (1 if kind == "collinear" else 3)
    )
    if len(components) != expected:
        raise ValueError(f"{record_name} moment components have the wrong shape for {kind!r}")
    object.__setattr__(record, "site_moments" if hasattr(record, "site_moments") else "moment", components)
    object.__setattr__(record, precision_name, precision)


@dataclass(frozen=True)
class SpeciesConstituentRecord:
    """One aligned, optionally decorated constituent of a stored species."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v1_species_constituent_record",
        identity_name="atomistic_v1_species_constituent_record",
    )

    chemical_symbol: str
    concentration: fractions.Fraction
    mass: float | None = None
    charge: fractions.Fraction | None = None
    spin: fractions.Fraction | None = None
    label: str | None = None
    concentration_precision: fractions.Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chemical_symbol, str):
            raise TypeError("SpeciesConstituentRecord chemical_symbol must be a string")
        if not isinstance(self.concentration, fractions.Fraction):
            raise TypeError("SpeciesConstituentRecord concentration must be a Fraction")
        if self.mass is not None and (not isinstance(self.mass, float) or isinstance(self.mass, bool)):
            raise TypeError("SpeciesConstituentRecord mass must be a float or None")
        if self.mass is not None and not math.isfinite(self.mass):
            raise ValueError("SpeciesConstituentRecord mass must be finite")
        for name in ("charge", "spin", "concentration_precision"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, fractions.Fraction):
                raise TypeError(f"SpeciesConstituentRecord {name} must be a Fraction or None")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("SpeciesConstituentRecord label must be a string or None")


@dataclass(frozen=True)
class SpeciesRecord:
    """The storable frozen snapshot of an atomistic :class:`~httk.atomistic.Species`; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v4_species_record",
        identity_name="atomistic_v4_species_record",
    )
    __httk_canonical_source__: ClassVar[type[Species]] = Species

    name: str
    constituents: tuple[SpeciesConstituentRecord, ...]
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None

    @classmethod
    def __httk_validate__(cls, record: "SpeciesRecord") -> None:
        _validate_species_record(record)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("SpeciesRecord name must be a string")
        constituents = tuple(self.constituents)
        if not constituents or not all(type(value) is SpeciesConstituentRecord for value in constituents):
            raise TypeError("SpeciesRecord constituents must contain SpeciesConstituentRecord values")
        object.__setattr__(self, "constituents", constituents)
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
        if (self.attached is None) != (self.nattached is None):
            raise ValueError("SpeciesRecord attached and nattached must be provided together")
        if self.attached is not None and len(self.attached) != len(self.nattached or ()):
            raise ValueError("SpeciesRecord attached and nattached must have matching lengths")

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        return tuple(value.chemical_symbol for value in self.constituents)

    @property
    def concentration(self) -> tuple[fractions.Fraction, ...]:
        return tuple(value.concentration for value in self.constituents)

    @property
    def mass(self) -> tuple[float, ...] | None:
        values = tuple(value.mass for value in self.constituents)
        return None if all(value is None for value in values) else cast(tuple[float, ...], values)

    @property
    def concentration_precision(self) -> tuple[fractions.Fraction | None, ...] | None:
        values = tuple(value.concentration_precision for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def charges(self) -> tuple[fractions.Fraction | None, ...] | None:
        values = tuple(value.charge for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def spins(self) -> tuple[fractions.Fraction | None, ...] | None:
        values = tuple(value.spin for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def labels(self) -> tuple[str | None, ...] | None:
        values = tuple(value.label for value in self.constituents)
        return None if all(value is None for value in values) else values

    @classmethod
    def __httk_project__(cls, species: Species) -> Mapping[str, object]:
        return {
            "name": species.name,
            "constituents": tuple(
                SpeciesConstituentRecord(
                    chemical_symbol=symbol,
                    concentration=concentration,
                    mass=None if species.mass is None else species.mass[index],
                    charge=None if species.charges is None else species.charges[index],
                    spin=None if species.spins is None else species.spins[index],
                    label=None if species.labels is None else species.labels[index],
                    concentration_precision=(
                        None if species.concentration_precision is None else species.concentration_precision[index]
                    ),
                )
                for index, (symbol, concentration) in enumerate(zip(species.chemical_symbols, species.concentration))
            ),
            "original_name": species.original_name,
            "attached": species.attached,
            "nattached": species.nattached,
        }


@dataclass(frozen=True)
class AssemblyGroupRecord:
    """One exact site-index group in a stored assembly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_assembly_group_record",
        identity_name="atomistic_v3_assembly_group_record",
    )
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

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_assembly_record",
        identity_name="atomistic_v3_assembly_record",
    )
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

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_domain_site_record",
        identity_name="atomistic_v3_domain_site_record",
    )
    __httk_canonical_source__: ClassVar[type[WyckoffSite]] = WyckoffSite

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] | None = None
    moment_kind: str | None = None
    moment: tuple[SurdScalar, ...] | None = None
    moment_precision: fractions.Fraction | None = None

    @classmethod
    def __httk_project__(cls, site: WyckoffSite) -> Mapping[str, object]:
        return {
            "wyckoff": site.wyckoff,
            "free_parameters": tuple(site.free_params.to_fractions()),
            "species": site.species,
            "representative": None if site.representative is None else tuple(site.representative.to_fractions()),
            "moment_kind": None if site.moment is None else cast(Any, site.moment).kind,
            "moment": None if site.moment is None else _moment_components(site.moment),
            "moment_precision": None if site.moment is None else site.moment.precision,
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
        _validate_moment_fields(self, "WyckoffSiteRecord")
        object.__setattr__(self, "free_parameters", free)
        object.__setattr__(self, "representative", representative)


@dataclass(frozen=True)
class SettingTransformRecord:
    """Exact durable standard-to-own setting transform; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_setting_transform_record",
        identity_name="atomistic_v3_setting_transform_record",
    )
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

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_symmetry_v3",
        identity_name="atomistic_symmetry_v3",
    )
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

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_composition_amount_record",
        identity_name="atomistic_v3_composition_amount_record",
    )
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

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_v3_chemical_composition_record",
        identity_name="atomistic_v3_chemical_composition_record",
    )
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
        identity_name="atomistic_v3_normalized_composition_record",
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
        identity_name="atomistic_v3_normalized_composition_amount_record",
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
        identity_name="atomistic_cell_v1",
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
        identity_name="atomistic_sites_v1",
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
        "charge": structure.charge,
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
    charge: fractions.Fraction | None
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
        "charge": record.charge,
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
        storage_name="atomistic_unitcell_structure_v2",
        identity_name="atomistic_unitcell_structure_v2",
        indexes=(("immutable_id",), ("last_modified",), ("optimization_type",)),
    )
    __httk_canonical_source__: ClassVar[type[UnitcellStructure]] = UnitcellStructure

    cell: CellRecord
    sites: SitesRecord
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    normalized_composition: NormalizedCompositionRecord
    charge: fractions.Fraction | None = None
    site_moments_kind: str | None = None
    site_moments: tuple[SurdScalar, ...] | None = None
    site_moments_precision: fractions.Fraction | None = None
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
        _validate_moment_fields(self, "UnitcellStructureRecord", nsites=len(species_at_sites))
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
                "site_moments_kind": None if structure.site_moments is None else cast(Any, structure.site_moments).kind,
                "site_moments": None if structure.site_moments is None else _moment_components(structure.site_moments),
                "site_moments_precision": None if structure.site_moments is None else structure.site_moments.precision,
            }
        )
        return values


@dataclass(frozen=True)
class FundamentalDomainStructureRecord:
    """Native durable backing for a symmetry fundamental domain; hand-built records are shape-checked and semantically validated at storage or explicitly."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_fundamental_domain_structure_v2",
        identity_name="atomistic_fundamental_domain_structure_v2",
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
    charge: fractions.Fraction | None = None
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
        storage_name="atomistic_asu_structure_v2",
        identity_name="atomistic_asu_structure_v2",
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
    precision: tuple[fractions.Fraction | None, ...] | None,
) -> tuple[fractions.Fraction | None, ...] | None:
    return precision


def _species_from_record(record: SpeciesRecord) -> Species:
    constituents = record.constituents
    mass_values = tuple(value.mass for value in constituents)
    charge_values = tuple(value.charge for value in constituents)
    spin_values = tuple(value.spin for value in constituents)
    label_values = tuple(value.label for value in constituents)
    precision_values = tuple(value.concentration_precision for value in constituents)
    return Species(
        name=record.name,
        chemical_symbols=tuple(value.chemical_symbol for value in constituents),
        concentration=tuple(value.concentration for value in constituents),
        mass=None if all(value is None for value in mass_values) else cast(tuple[float, ...], mass_values),
        original_name=record.original_name,
        attached=record.attached,
        nattached=record.nattached,
        concentration_precision=None if all(value is None for value in precision_values) else precision_values,
        charges=None if all(value is None for value in charge_values) else charge_values,
        spins=None if all(value is None for value in spin_values) else spin_values,
        labels=None if all(value is None for value in label_values) else label_values,
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
    values = cast(dict[str, Any], NormalizedCompositionRecord.__httk_project__(result))
    amounts = tuple(NormalizedCompositionAmountRecord(*item) for item in values["amounts"])
    return NormalizedCompositionRecord(amounts, values["complete"])


def _cell_from_record(record: CellRecord) -> Cell:
    return Cell(_basis_vector(record.basis), precision=record.precision, periodicity=record.periodicity)


def _sites_from_record(record: SitesRecord) -> Sites:
    return Sites(record.reduced_coords, precision=record.precision)


def _domain_structure_from_record(
    record: FundamentalDomainStructureRecord | ASUStructureRecord,
) -> FundamentalDomainStructure:
    from httk.atomistic.symmetry.spacegroup import Spacegroup

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
                moment=_moment_from_record(
                    value.moment_kind,
                    value.moment,
                    value.moment_precision,
                    _cell_from_record(record.cell),
                ),
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
        site_moments=_moment_from_record(
            record.site_moments_kind,
            record.site_moments,
            record.site_moments_precision,
            _cell_from_record(record.cell),
        ),
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


def _assembly_record_from_assembly(assembly: Assembly) -> AssemblyRecord:
    projected = cast(dict[str, Any], AssemblyRecord.__httk_project__(assembly))
    groups = tuple(AssemblyGroupRecord(sites=sites) for sites in cast(tuple[tuple[int, ...], ...], projected["groups"]))
    return AssemblyRecord(
        groups=groups,
        group_probabilities=cast(tuple[fractions.Fraction, ...], projected["group_probabilities"]),
        group_probabilities_precision=cast(
            tuple[fractions.Fraction, ...] | None, projected["group_probabilities_precision"]
        ),
    )


def _unitcell_record_from_structure(structure: UnitcellStructure) -> UnitcellStructureRecord:
    values: dict[str, Any] = dict(UnitcellStructureRecord.__httk_project__(structure))
    values["cell"] = CellRecord(**cast(dict[str, Any], CellRecord.__httk_project__(structure.cell)))
    values["sites"] = SitesRecord(**cast(dict[str, Any], SitesRecord.__httk_project__(structure.sites)))
    values["species"] = tuple(
        SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(species))) for species in structure.species
    )
    values["normalized_composition"] = _normalized_composition_record_from_result(structure.composition)
    values["assemblies"] = (
        None
        if structure.assemblies is None
        else tuple(_assembly_record_from_assembly(assembly) for assembly in structure.assemblies)
    )
    values["symmetry"] = (
        None
        if structure.symmetry is None
        else SymmetryRecord(**cast(dict[str, Any], SymmetryRecord.__httk_project__(structure.symmetry)))
    )
    values["chemical_composition"] = (
        None
        if structure.chemical_composition is None
        else ChemicalCompositionRecord(
            **cast(dict[str, Any], ChemicalCompositionRecord.__httk_project__(structure.chemical_composition))
        )
    )
    return UnitcellStructureRecord(**values)


def _observable_summary(name: str, values: tuple[Any, ...]) -> "ObservableSummaryRecord":
    try:
        if any(isinstance(value, bool) or not isinstance(value, numbers.Number | Decimal) for value in values):
            raise ValueError
        converted = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in converted):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        converted = ()
    if not converted:
        return ObservableSummaryRecord(name)
    return ObservableSummaryRecord(name, converted[0], converted[-1], min(converted), max(converted))


@dataclass(frozen=True)
class ObservableSummaryRecord:
    """Bounded numeric summary for one trajectory observable."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_trajectory_observable_summary_v1",
        identity_name="atomistic_trajectory_observable_summary_v1",
    )

    name: str
    first: float | None = None
    last: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ObservableSummaryRecord name must be a non-empty string")
        for field_name in ("first", "last", "minimum", "maximum"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, float) or not math.isfinite(value)):
                raise TypeError(f"ObservableSummaryRecord {field_name} must be a finite float or None")


def _validate_trajectory_record(record: "TrajectoryRecord") -> None:
    for structure in record.reference_frame_structures:
        validate_structure_record(structure)
        if tuple(value.name for value in structure.species) != tuple(value.name for value in record.species):
            raise ValueError("TrajectoryRecord reference structure species disagree with trajectory species")
        if structure.species_at_sites != record.species_at_sites:
            raise ValueError("TrajectoryRecord reference structure composition disagrees with trajectory")


@dataclass(frozen=True)
class TrajectoryRecord:
    """Bounded trajectory identity and reference-frame summary; frame data is never stored."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_trajectory_v1",
        identity_name="atomistic_trajectory_v1",
        indexes=(("immutable_id",), ("last_modified",), ("nframes",)),
    )
    __httk_canonical_source__: ClassVar[type[TrajectoryAPI]] = cast(type[TrajectoryAPI], TrajectoryAPI)

    nframes: int
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    reference_frame_indexes: tuple[int, ...]
    reference_frame_structures: tuple[UnitcellStructureRecord, ...]
    observable_summaries: tuple[ObservableSummaryRecord, ...]
    source_locator: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "TrajectoryRecord") -> None:
        _validate_trajectory_record(record)

    def __post_init__(self) -> None:
        if not isinstance(self.nframes, int) or isinstance(self.nframes, bool) or self.nframes < 1:
            raise ValueError("TrajectoryRecord nframes must be a positive integer")
        species = tuple(self.species)
        if not species or not all(type(value) is SpeciesRecord for value in species):
            raise TypeError("TrajectoryRecord species must contain SpeciesRecord values")
        if len({value.name for value in species}) != len(species):
            raise ValueError("TrajectoryRecord species names must be unique")
        species_at_sites = tuple(self.species_at_sites)
        if not all(isinstance(value, str) for value in species_at_sites):
            raise TypeError("TrajectoryRecord species_at_sites must contain strings")
        if set(species_at_sites) - {value.name for value in species}:
            raise ValueError("TrajectoryRecord species_at_sites references an unknown species")
        indexes = tuple(self.reference_frame_indexes)
        structures = tuple(self.reference_frame_structures)
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in indexes):
            raise TypeError("TrajectoryRecord reference_frame_indexes must contain integers")
        if any(not 0 <= value < self.nframes for value in indexes):
            raise ValueError("TrajectoryRecord reference_frame_indexes contains an out-of-bounds index")
        if indexes != tuple(sorted(set(indexes))):
            raise ValueError("TrajectoryRecord reference_frame_indexes must be sorted and deduplicated")
        if len(indexes) != len(structures) or not all(type(value) is UnitcellStructureRecord for value in structures):
            raise TypeError("TrajectoryRecord reference frames must match UnitcellStructureRecord values")
        summaries = tuple(self.observable_summaries)
        if not all(type(value) is ObservableSummaryRecord for value in summaries):
            raise TypeError("TrajectoryRecord observable_summaries must contain ObservableSummaryRecord values")
        if len({value.name for value in summaries}) != len(summaries):
            raise ValueError("TrajectoryRecord observable summary names must be unique")
        if self.source_locator is not None and not isinstance(self.source_locator, str):
            raise TypeError("TrajectoryRecord source_locator must be a string or None")
        if self.immutable_id is not None and not isinstance(self.immutable_id, str):
            raise TypeError("TrajectoryRecord immutable_id must be a string or None")
        if self.last_modified is not None:
            if not isinstance(self.last_modified, datetime.datetime):
                raise TypeError("TrajectoryRecord last_modified must be a datetime or None")
            if self.last_modified.tzinfo is None or self.last_modified.utcoffset() is None:
                raise ValueError("TrajectoryRecord last_modified must include a timezone")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "species_at_sites", species_at_sites)
        object.__setattr__(self, "reference_frame_indexes", indexes)
        object.__setattr__(self, "reference_frame_structures", structures)
        object.__setattr__(self, "observable_summaries", summaries)

    @property
    def type(self) -> str:
        return "trajectories"

    @property
    def id(self) -> str:
        return content_id(self)

    @classmethod
    def __httk_project__(cls, trajectory: TrajectoryAPI) -> Mapping[str, object]:
        declared = trajectory.reference_frames
        indexes = tuple(declared) if declared is not None else tuple(dict.fromkeys((0, trajectory.nframes - 1)))
        return {
            "nframes": trajectory.nframes,
            "species": tuple(
                SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(species)))
                for species in trajectory.species
            ),
            "species_at_sites": trajectory.species_at_sites,
            "reference_frame_indexes": indexes,
            "reference_frame_structures": tuple(
                _unitcell_record_from_structure(trajectory.frame(index)) for index in indexes
            ),
            "observable_summaries": tuple(
                _observable_summary(name, trajectory.observable(name)) for name in sorted(trajectory.observable_names)
            ),
            "source_locator": getattr(trajectory, "source_locator", None),
            "immutable_id": getattr(trajectory, "immutable_id", None),
            "last_modified": getattr(trajectory, "last_modified", None),
        }


from httk.atomistic.storage.stored_properties import attach_structure_property_projections

attach_structure_property_projections(
    UnitcellStructureRecord,
    FundamentalDomainStructureRecord,
    ASUStructureRecord,
)

"""Frozen storage records for complete atomistic structures."""

import datetime
import fractions
import itertools
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypedDict, cast

import httk.core.storage.markers
from httk.core import (
    FracVector,
    ReferenceRecord,
    SurdScalar,
    SurdVector,
)
from httk.core.exactmath import any_to_fraction
from httk.core.files import FileRecord
from httk.core.storage import (
    IdentitySkip,
    IdentitySkipWhenNone,
    Indexed,
    StorageInfo,
    Unique,
    WeakLink,
    float_matrix,
    stored_property,
)

from httk.atomistic._composition_values import as_fraction
from httk.atomistic.composition import Assembly, ChemicalComposition, validate_assemblies
from httk.atomistic.models._vector_guards import to_periodicity, to_precision
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.numeric import NumericCell, finite_float, float_precision
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.numeric import NumericSiteMoments
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.notation import _undetermined
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.classes import _species_class
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, SiteBond, WyckoffSite
from httk.atomistic.models.structure.numeric import NumericUnitcellStructure
from httk.atomistic.models.structure.semantics import StructureSymmetry, _stated_symmetry
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.models.trajectory.api import TrajectoryAPI
from httk.atomistic.symmetry._canonical_form import _LABEL_FORM_VERSION, _NET_FORM_VERSION, _REFINED_FORM_VERSION
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.setting_transform import SettingTransform

if TYPE_CHECKING:  # a runtime import is circular (records load during package import)
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
    from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__ = [
    "ASUStructureRecord",
    "AssemblyGroupRecord",
    "AssemblyRecord",
    "BareProtostructureRecord",
    "BarePrototypeRecord",
    "CellRecord",
    "ChemicalCompositionRecord",
    "CompositionAmountRecord",
    "FundamentalDomainStructureRecord",
    "FundamentalDomainTemplateRecord",
    "NormalizedCompositionAmountRecord",
    "NormalizedCompositionRecord",
    "NumericCellRecord",
    "NumericCompositionAmountRecord",
    "NumericNormalizedCompositionRecord",
    "NumericSitesRecord",
    "NumericSpeciesConstituentRecord",
    "NumericSpeciesRecord",
    "NumericUnitcellStructureRecord",
    "ObservableSummaryRecord",
    "ProtonetBondRecord",
    "ProtonetstructureRecord",
    "ProtonettypeRecord",
    "ProtostructureRecord",
    "PrototypeRecord",
    "SettingTransformRecord",
    "SiteBondRecord",
    "SitesRecord",
    "SpeciesConstituentRecord",
    "SpeciesRecord",
    "SymmetryRecord",
    "TrajectoryRecord",
    "UnitcellStructureRecord",
    "WyckoffOccupationRecord",
    "WyckoffSiteRecord",
    "numeric_structure_from_record",
]


def _effective_record_type(value: Any) -> type:
    """Return a value's storable record type, unwrapping a lazy store-row subclass.

    A record fetched from a store may be a lazy row proxy whose ``type()`` is a
    generated subclass; the record author's exact-type checks compare against the
    base record. ``__httk_row_base__`` (set on the row subclass by the store)
    names that base; a plain record has no such attribute and reports its own
    type.

    :param value: The value whose effective record type is wanted.
    :return: The base record type for a proxy, otherwise ``type(value)``.
    """
    return cast(type, getattr(type(value), "__httk_row_base__", type(value)))


def _stored_floats(record: Any, field: str) -> list[list[float]] | None:
    """Return a lazy storage row's already-fetched float companion rows for ``field``, or ``None``.

    :param record: The storage record, possibly a lazy row proxy.
    :param field: The child field name.
    :return: The stored float rows, or ``None`` for a materialized record or a field without them.
    """
    stored = getattr(record, "_httk_stored_floats", None)
    return None if stored is None else stored(field)


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
        return tuple(SurdVector(value)._as_scalar() for value in moment.collinear_moments.to_fractions())
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
        value if isinstance(value, SurdScalar) else SurdVector(value)._as_scalar() for value in components
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
    """Represent one aligned, optionally decorated species constituent.

    :param chemical_symbol: The constituent's chemical symbol.
    :param concentration: The constituent occupancy.
    :param mass: The constituent mass, if stated.
    :param charge: The constituent charge, if stated.
    :param spin: The constituent spin, if stated.
    :param label: The constituent label, if stated.
    :param concentration_precision: The occupancy precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_species_constituent_record",
        identity_name="atomistic_species_constituent_record",
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
    """Represent a frozen storable snapshot of an atomistic species.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary or explicitly through the validation hook.

    :param name: The species name.
    :param constituents: The aligned constituent records.
    :param original_name: The source species name, if stated.
    :param attached: The attached constituent symbols, if stated.
    :param nattached: The counts corresponding to ``attached``, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_species_record",
        identity_name="atomistic_species_record",
    )
    __httk_canonical_source__: ClassVar = Species

    name: str
    constituents: tuple[SpeciesConstituentRecord, ...]
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None

    @classmethod
    def __httk_validate__(cls, record: "SpeciesRecord") -> None:
        """Validate the semantic species record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        _validate_species_record(record)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("SpeciesRecord name must be a string")
        constituents = tuple(self.constituents)
        if not constituents or not all(
            _effective_record_type(value) is SpeciesConstituentRecord for value in constituents
        ):
            raise TypeError("SpeciesRecord constituents must contain SpeciesConstituentRecord values")
        object.__setattr__(self, "constituents", constituents)
        if len({value.mass is None for value in constituents}) > 1:
            raise ValueError(f"SpeciesRecord {self.name!r} must state a mass for all constituents or none")
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
        """Expose the constituent chemical symbols.

        :return: The chemical symbols in constituent order.
        """
        return tuple(value.chemical_symbol for value in self.constituents)

    @property
    def concentration(self) -> tuple[fractions.Fraction, ...]:
        """Expose the constituent occupancies.

        :return: The concentrations in constituent order.
        """
        return tuple(value.concentration for value in self.constituents)

    @property
    def mass(self) -> tuple[float, ...] | None:
        """Expose the constituent masses.

        :return: The masses in constituent order, or ``None`` when unstated.
        """
        values = tuple(value.mass for value in self.constituents)
        return None if all(value is None for value in values) else cast(tuple[float, ...], values)

    @property
    def concentration_precision(self) -> tuple[fractions.Fraction | None, ...] | None:
        """Expose the constituent occupancy precision.

        :return: The precisions in constituent order, or ``None`` when unstated.
        """
        values = tuple(value.concentration_precision for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def charges(self) -> tuple[fractions.Fraction | None, ...] | None:
        """Expose the constituent charges.

        :return: The charges in constituent order, or ``None`` when unstated.
        """
        values = tuple(value.charge for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def spins(self) -> tuple[fractions.Fraction | None, ...] | None:
        """Expose the constituent spins.

        :return: The spins in constituent order, or ``None`` when unstated.
        """
        values = tuple(value.spin for value in self.constituents)
        return None if all(value is None for value in values) else values

    @property
    def labels(self) -> tuple[str | None, ...] | None:
        """Expose the constituent labels.

        :return: The labels in constituent order, or ``None`` when unstated.
        """
        values = tuple(value.label for value in self.constituents)
        return None if all(value is None for value in values) else values

    @classmethod
    def __httk_project__(cls, species: Species) -> Mapping[str, object]:
        """Project a species into its durable record fields.

        :param species: The species to project.
        :return: The projected record fields.
        """
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
    """Represent one exact site-index group in a stored assembly.

    :param sites: The distinct non-negative site indexes in the group.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_assembly_group_record",
        identity_name="atomistic_assembly_group_record",
    )
    __httk_canonical_source__: ClassVar = tuple

    sites: tuple[int, ...]

    @classmethod
    def __httk_project__(cls, group: tuple[Any, ...]) -> Mapping[str, object]:
        """Project one site-index group into durable fields.

        :param group: The site indexes to project.
        :return: The projected group fields.
        """
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
    """Represent the exact durable form of an assembly.

    :param groups: The site-index groups.
    :param group_probabilities: The probability of each group.
    :param group_probabilities_precision: The probability precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_assembly_record",
        identity_name="atomistic_assembly_record",
    )
    __httk_canonical_source__: ClassVar = Assembly

    groups: tuple[AssemblyGroupRecord, ...]
    group_probabilities: tuple[fractions.Fraction, ...]
    group_probabilities_precision: tuple[fractions.Fraction, ...] | None = None

    @property
    def sites_in_groups(self) -> tuple[tuple[int, ...], ...]:
        """Expose the site indexes grouped by assembly value.

        :return: The site indexes in group order.
        """
        return tuple(group.sites for group in self.groups)

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not all(_effective_record_type(group) is AssemblyGroupRecord for group in groups):
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
        """Project an assembly into durable fields.

        :param assembly: The assembly to project.
        :return: The projected assembly fields.
        """
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
    """Represent an exact Wyckoff site with its retained representative.

    The owning record's ``domain_sites`` field is storage-visible and deliberately unchanged.

    :param wyckoff: The Wyckoff letter.
    :param free_parameters: The exact free-parameter values.
    :param species: The owning species name.
    :param representative: The retained representative coordinate, if present.
    :param moment_kind: The site-moment kind, if present.
    :param moment: The flattened exact site-moment components, if present.
    :param moment_precision: The site-moment precision, if present.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_domain_site_record",
        identity_name="atomistic_domain_site_record",
    )
    __httk_canonical_source__: ClassVar = WyckoffSite

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] | None = None
    moment_kind: str | None = None
    moment: tuple[SurdScalar, ...] | None = None
    moment_precision: fractions.Fraction | None = None

    @classmethod
    def __httk_project__(cls, site: WyckoffSite) -> Mapping[str, object]:
        """Project a Wyckoff site into durable fields.

        :param site: The Wyckoff site to project.
        :return: The projected site fields.
        """
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
    """Represent an exact stored-setting-to-own transform.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary or explicitly through the validation hook.

    :param matrix: The stored-setting-to-own fractional coordinate matrix.
    :param vector: The stored-setting-to-own fractional origin shift.
    :param hall_entry: The normalized Hall entry, if known.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_setting_transform_record",
        identity_name="atomistic_setting_transform_record",
        dedup="by_value",
    )
    __httk_canonical_source__: ClassVar = SettingTransform

    matrix: Annotated[FracVector, httk.core.storage.markers.Shape(3, 3)]
    vector: tuple[fractions.Fraction, ...]
    hall_entry: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "SettingTransformRecord") -> None:
        """Validate the semantic setting transform.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        _validate_setting_transform_record(record)

    def __post_init__(self) -> None:
        matrix = FracVector(self.matrix)
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
        """Project a setting transform into durable fields.

        :param transform: The setting transform to project.
        :return: The projected transform fields.
        """
        return {
            "matrix": transform.matrix,
            "vector": tuple(transform.vector.to_fractions()),
            "hall_entry": transform.hall_entry,
        }


@dataclass(frozen=True)
class SiteBondRecord:
    """Represent one bond orbit of a fundamental domain's bond network.

    The durable form of a :class:`~httk.atomistic.SiteBond`: the reference atom of domain site
    ``site`` is bonded to the affine map ``x -> rotation x + translation`` (column convention, in
    the stored setting) applied to the reference atom of domain site ``partner``.

    :param site: The 0-based index of the first domain site.
    :param partner: The 0-based index of the second domain site.
    :param rotation: The exact rotation part of the group element.
    :param translation: The exact translation part of the group element, lattice translation included.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_site_bond_record",
        identity_name="atomistic_site_bond_record",
    )
    __httk_canonical_source__: ClassVar = SiteBond

    site: int
    partner: int
    rotation: Annotated[FracVector, httk.core.storage.markers.Shape(3, 3)]
    translation: tuple[fractions.Fraction, ...]

    def __post_init__(self) -> None:
        for name in ("site", "partner"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"SiteBondRecord {name} must be a non-negative integer")
        rotation = FracVector(self.rotation)
        translation = tuple(as_fraction(value, field="SiteBondRecord translation")[0] for value in self.translation)
        if rotation.dim != (3, 3):
            raise ValueError(f"SiteBondRecord rotation must be 3x3, got dim {rotation.dim}")
        if len(translation) != 3:
            raise ValueError(f"SiteBondRecord translation must have 3 elements, got {len(translation)}")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def __httk_project__(cls, bond: SiteBond) -> Mapping[str, object]:
        """Project a site bond into durable fields.

        :param bond: The site bond to project.
        :return: The projected bond fields.
        """
        return {
            "site": bond.site,
            "partner": bond.partner,
            "rotation": bond.operation.matrix,
            "translation": tuple(bond.operation.vector.to_fractions()),
        }


@dataclass(frozen=True)
class SymmetryRecord:
    """Represent optional symmetry metadata for a unit-cell structure.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary or explicitly through the validation hook.

    :param space_group_it_number: The International Tables space-group number, if known.
    :param space_group_symbol_hall: The Hall symbol, if known.
    :param space_group_symbol_hermann_mauguin: The Hermann-Mauguin symbol, if known.
    :param space_group_symbol_hermann_mauguin_extended: The extended Hermann-Mauguin symbol, if known.
    :param space_group_symmetry_operations_xyz: The symmetry operations in xyz form, if known.
    :param wyckoff_positions: The Wyckoff position symbols, if known.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_symmetry",
        identity_name="atomistic_symmetry",
    )
    __httk_canonical_source__: ClassVar = StructureSymmetry

    space_group_it_number: int | None = None
    space_group_symbol_hall: str | None = None
    space_group_symbol_hermann_mauguin: str | None = None
    space_group_symbol_hermann_mauguin_extended: str | None = None
    space_group_symmetry_operations_xyz: tuple[str, ...] | None = None
    wyckoff_positions: tuple[str, ...] | None = None

    @classmethod
    def __httk_validate__(cls, record: "SymmetryRecord") -> None:
        """Validate the semantic symmetry record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
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
        """Project symmetry metadata into durable fields.

        :param symmetry: The symmetry metadata to project.
        :return: The projected symmetry fields.
        """
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
    """Represent one exact declared element amount.

    :param element: The element symbol.
    :param amount: The exact element amount.
    :param precision: The amount precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_composition_amount_record",
        identity_name="atomistic_composition_amount_record",
    )
    __httk_canonical_source__: ClassVar = tuple

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
        """Project one composition amount into durable fields.

        :param amount: The element, amount, and precision values to project.
        :return: The projected amount fields.
        :raises ValueError: If the projected tuple does not contain three values.
        """
        if len(amount) != 3:
            raise ValueError("composition amount projection requires element, amount, and precision")
        return {"element": amount[0], "amount": amount[1], "precision": amount[2]}


@dataclass(frozen=True)
class ChemicalCompositionRecord:
    """Represent a durable authoritative or implicit composition declaration.

    :param amounts: The exact element amounts.
    :param mode: The composition declaration mode.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_chemical_composition_record",
        identity_name="atomistic_chemical_composition_record",
    )
    __httk_canonical_source__: ClassVar = ChemicalComposition

    amounts: tuple[CompositionAmountRecord, ...]
    mode: str

    @classmethod
    def __httk_project__(cls, composition: ChemicalComposition) -> Mapping[str, object]:
        """Project a chemical composition into durable fields.

        :param composition: The chemical composition to project.
        :return: The projected composition fields.
        """
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
    """Represent the authoritative exact elemental composition of a structure.

    This relation is semantic normalized data, not a rendered-formula cache. It
    retains each exact central element amount together with its source precision,
    and makes the same complete-composition facts available to every durable
    structure backing for response construction and exact filtering.

    :param amounts: The normalized element amounts.
    :param complete: Whether the composition accounts for all structure content.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_normalized_composition_record",
        identity_name="atomistic_normalized_composition_record",
    )
    __httk_canonical_source__: ClassVar = Composition

    amounts: tuple["NormalizedCompositionAmountRecord", ...]
    complete: bool

    @classmethod
    def __httk_project__(cls, result: Composition) -> Mapping[str, object]:
        """Project a composition into normalized durable fields.

        :param result: The composition to project.
        :return: The projected normalized-composition fields.
        """
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
    """Represent one exact normalized composition ratio.

    :param element: The element symbol.
    :param ratio: The exact normalized element ratio.
    :param amount: The exact source element amount.
    :param precision: The source amount precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_normalized_composition_amount_record",
        identity_name="atomistic_normalized_composition_amount_record",
    )
    __httk_canonical_source__: ClassVar = tuple

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
        """Project one normalized amount into durable fields.

        :param amount: The element, ratio, amount, and precision values to project.
        :return: The projected normalized amount fields.
        :raises ValueError: If the projected tuple does not contain four values.
        """
        if len(amount) != 4:
            raise ValueError("normalized composition amount projection requires element, ratio, amount, and precision")
        return {"element": amount[0], "ratio": amount[1], "amount": amount[2], "precision": amount[3]}


# These concrete records retain each representation's native fields; recursive
# storage projection follows their annotations.


@dataclass(frozen=True)
class CellRecord:
    """Represent an exact durable cell basis, precision, and periodicity.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary.

    :param basis: The row-major exact cell basis values.
    :param precision: The absolute basis precision, if stated.
    :param periodicity: The flags identifying periodic basis rows.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_cell",
        identity_name="atomistic_cell",
    )
    __httk_canonical_source__: ClassVar = Cell

    basis: tuple[SurdScalar, ...]
    precision: fractions.Fraction | None
    periodicity: tuple[bool, ...]

    @classmethod
    def __httk_validate__(cls, record: "CellRecord") -> None:
        """Validate the semantic cell record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
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
        """Project a cell into durable fields.

        :param cell: The cell to project.
        :return: The projected cell fields.
        """
        return {
            "basis": tuple(_extract_surd_scalar(cell.basis, (row, column)) for row in range(3) for column in range(3)),
            "precision": cell.precision,
            "periodicity": cell.periodicity,
        }


@dataclass(frozen=True)
class SitesRecord:
    """Represent exact durable reduced coordinates and their stated precision.

    :param reduced_coords: The exact reduced coordinates, one site per row.
    :param precision: The fractional coordinate precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_sites",
        identity_name="atomistic_sites",
    )
    __httk_canonical_source__: ClassVar = Sites

    reduced_coords: Annotated[FracVector, httk.core.storage.markers.Shape(0, 3)]
    precision: fractions.Fraction | None

    def __post_init__(self) -> None:
        sites = Sites(self.reduced_coords, precision=self.precision)
        object.__setattr__(self, "reduced_coords", sites.reduced_coords)
        object.__setattr__(self, "precision", sites.precision)

    @classmethod
    def __httk_project__(cls, sites: Sites) -> Mapping[str, object]:
        """Project site coordinates into durable fields.

        :param sites: The sites to project.
        :return: The projected sites fields.
        """
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
        "id": None,
        "immutable_id": None,
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
    """Represent the native durable backing for an explicit unit-cell structure.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary or explicitly through the validation hook. The record's content identity
    is independent of its storage layout.

    :param cell: The durable cell record.
    :param sites: The durable site-coordinate record.
    :param species: The distinct durable species records.
    :param species_at_sites: The species name occupying each site.
    :param normalized_composition: The authoritative normalized composition.
    :param charge: The explicitly assigned cell charge, if stated.
    :param site_moments_kind: The site-moment kind, if stated.
    :param site_moments: The flattened exact site-moment components, if stated.
    :param site_moments_precision: The site-moment precision, if stated.
    :param molecular: Whether the structure describes molecular entities.
    :param assemblies: The site assemblies, if stated.
    :param symmetry: The symmetry metadata, if stated.
    :param chemical_composition: The chemical composition declaration, if stated.
    :param chemical_formula_descriptive: The descriptive formula, if stated.
    :param chemical_formula_hill: The Hill formula, if stated.
    :param optimization_type: The optimization provenance, if stated.
    :param source_file: The verbatim file this structure was read from, when the entry records an input file exactly
        as given; outside content identity while ``None``.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_unitcell_structure",
        identity_name="atomistic_unitcell_structure",
        indexes=(("last_modified",), ("optimization_type",)),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = UnitcellStructure

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
    source_file: Annotated[FileRecord | None, IdentitySkipWhenNone()] = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "UnitcellStructureRecord") -> None:
        """Validate the semantic unit-cell record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        validate_structure_record(record)

    @classmethod
    def from_structure(cls, structure: "UnitcellStructure | UnitcellStructureView") -> "UnitcellStructureRecord":
        """Build the complete record of a unit-cell structure, with ``source_file`` unset.

        Use :func:`dataclasses.replace` on the result to set the record-only ``source_file``.

        :param structure: The unit-cell structure or view to record.
        :return: The record, with every nested child record built.
        """
        return _unitcell_record_from_structure(cast(UnitcellStructure, structure))

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"

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
        if self.source_file is not None and not isinstance(self.source_file, FileRecord):
            raise TypeError("UnitcellStructureRecord source_file must be a FileRecord or None")
        object.__setattr__(self, "species_at_sites", species_at_sites)

    @classmethod
    def __httk_project__(cls, structure: UnitcellStructure) -> Mapping[str, object]:
        """Project a unit-cell structure into durable fields.

        :param structure: The unit-cell structure to project.
        :return: The projected structure fields.
        """
        values = _project_common(structure)
        values.update(
            {
                "sites": structure.sites,
                "species_at_sites": structure.species_at_sites,
                "symmetry": _stated_symmetry(structure.symmetry),
                "site_moments_kind": None if structure.site_moments is None else cast(Any, structure.site_moments).kind,
                "site_moments": None if structure.site_moments is None else _moment_components(structure.site_moments),
                "site_moments_precision": None if structure.site_moments is None else structure.site_moments.precision,
                "source_file": None,
            }
        )
        return values


@dataclass(frozen=True)
class FundamentalDomainStructureRecord:
    """Represent the native durable backing for a symmetry fundamental domain.

    Hand-built records are shape-checked on construction and semantically validated at the
    storage boundary or explicitly through the validation hook. The record's content identity
    is independent of its storage layout.

    :param cell: The durable cell record.
    :param domain_sites: The symmetry-distinct durable site records.
    :param species: The distinct durable species records.
    :param spacegroup_it_number: The International Tables space-group number.
    :param spacegroup_hall_entry: The setting that names the stored Wyckoff data.
    :param setting_transform: The stored-setting-to-own transform.
    :param coordinate_precision: The reduced-coordinate precision, if stated.
    :param normalized_composition: The authoritative normalized composition.
    :param charge: The explicitly assigned cell charge, if stated.
    :param molecular: Whether the structure describes molecular entities.
    :param assemblies: The site assemblies, if stated.
    :param chemical_composition: The chemical composition declaration, if stated.
    :param chemical_formula_descriptive: The descriptive formula, if stated.
    :param chemical_formula_hill: The Hill formula, if stated.
    :param optimization_type: The optimization provenance, if stated.
    :param bonds: The bond network, one least orbit member per bond orbit, or ``None`` when no
        network is known; ``None`` leaves the content identity unchanged.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_fundamental_domain_structure",
        identity_name="atomistic_fundamental_domain_structure",
        indexes=(
            ("spacegroup_it_number",),
            ("spacegroup_hall_entry",),
            ("last_modified",),
            ("optimization_type",),
        ),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = FundamentalDomainStructure

    cell: CellRecord
    domain_sites: tuple[WyckoffSiteRecord, ...]
    species: tuple[SpeciesRecord, ...]
    spacegroup_it_number: int
    spacegroup_hall_entry: str
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
    bonds: Annotated[tuple[SiteBondRecord, ...] | None, IdentitySkipWhenNone()] = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "FundamentalDomainStructureRecord") -> None:
        """Validate the semantic fundamental-domain record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        validate_structure_record(record)

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"

    def __post_init__(self) -> None:
        domain_sites = tuple(self.domain_sites)
        if not all(isinstance(value, WyckoffSiteRecord) for value in domain_sites):
            raise TypeError("FundamentalDomainStructureRecord domain_sites must contain WyckoffSiteRecord values")
        if not isinstance(self.spacegroup_it_number, int) or isinstance(self.spacegroup_it_number, bool):
            raise TypeError("FundamentalDomainStructureRecord spacegroup_it_number must be an integer")
        if not 1 <= self.spacegroup_it_number <= 230:
            raise ValueError("FundamentalDomainStructureRecord spacegroup_it_number must be in [1, 230]")
        if not isinstance(self.spacegroup_hall_entry, str):
            raise TypeError("FundamentalDomainStructureRecord spacegroup_hall_entry must be a string")
        if not isinstance(self.setting_transform, SettingTransformRecord):
            raise TypeError("FundamentalDomainStructureRecord setting_transform must be a SettingTransformRecord")
        if self.bonds is not None:
            bonds = tuple(self.bonds)
            if not all(_effective_record_type(value) is SiteBondRecord for value in bonds):
                raise TypeError("FundamentalDomainStructureRecord bonds must contain SiteBondRecord values")
            object.__setattr__(self, "bonds", bonds)
        coordinate_precision = to_precision(self.coordinate_precision)
        object.__setattr__(self, "domain_sites", domain_sites)
        object.__setattr__(self, "coordinate_precision", coordinate_precision)
        _normalize_common(self, nsites=len(domain_sites))
        known = {value.name for value in self.species}
        if unknown := {value.species for value in domain_sites} - known:
            raise ValueError(f"FundamentalDomainStructureRecord references unknown species: {sorted(unknown)!r}")

    @classmethod
    def __httk_project__(cls, structure: FundamentalDomainStructure) -> Mapping[str, object]:
        """Project a fundamental-domain structure into durable fields.

        :param structure: The fundamental-domain structure to project.
        :return: The projected structure fields.
        """
        values = _project_common(structure)
        values.update(
            {
                "domain_sites": structure.domain_sites,
                "spacegroup_it_number": structure.spacegroup.it_number,
                "spacegroup_hall_entry": structure.spacegroup.hall_entry,
                "setting_transform": structure.transform,
                "coordinate_precision": structure.coordinate_precision,
                "bonds": structure.bonds,
            }
        )
        return values


@dataclass(frozen=True)
class ASUStructureRecord(FundamentalDomainStructureRecord):
    """Represent the native durable backing for an asserted asymmetric unit.

    Inherit the fundamental-domain constructor fields and validation contract while retaining
    the asymmetric-unit record identity.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_asu_structure",
        identity_name="atomistic_asu_structure",
        indexes=(
            ("spacegroup_it_number",),
            ("spacegroup_hall_entry",),
            ("last_modified",),
            ("optimization_type",),
        ),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = ASUStructure

    @classmethod
    def __httk_validate__(cls, record: "FundamentalDomainStructureRecord") -> None:
        """Validate the semantic asymmetric-unit record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        validate_structure_record(record)

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"


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


def _composition_from_record(record: NormalizedCompositionRecord) -> Composition:
    amounts = tuple((value.element, value.amount) for value in record.amounts)
    uncertainties = tuple((value.element, value.precision) for value in record.amounts)
    exact = all(value.precision is None for value in record.amounts)
    return Composition(amounts, uncertainties, record.complete, exact, True, "exact" if exact else "within_precision")


def _normalized_composition_record_from_result(result: Composition) -> NormalizedCompositionRecord:
    values = cast(dict[str, Any], NormalizedCompositionRecord.__httk_project__(result))
    amounts = tuple(NormalizedCompositionAmountRecord(*item) for item in values["amounts"])
    return NormalizedCompositionRecord(amounts, values["complete"])


def _cell_from_record(record: CellRecord) -> Cell:
    return Cell(_basis_vector(record.basis), precision=record.precision, periodicity=record.periodicity)


def _sites_from_record(record: SitesRecord) -> Sites:
    return Sites(record.reduced_coords, precision=record.precision)


def _site_bonds_from_record(records: Iterable[SiteBondRecord]) -> tuple[SiteBond, ...]:
    return tuple(
        SiteBond(value.site, value.partner, AffineOperation(value.rotation, value.translation)) for value in records
    )


def _domain_structure_from_record(
    record: FundamentalDomainStructureRecord | ASUStructureRecord,
) -> FundamentalDomainStructure:
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    structure_type = ASUStructure if isinstance(record, ASUStructureRecord) else FundamentalDomainStructure
    spacegroup = Spacegroup.from_hall_entry(record.spacegroup_hall_entry)
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    return structure_type(
        _cell_from_record(record.cell),
        spacegroup,
        tuple(
            WyckoffSite(
                value.wyckoff,
                FracVector(value.free_parameters),
                value.species,
                None if value.representative is None else FracVector(value.representative),
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
        bonds=None if record.bonds is None else _site_bonds_from_record(record.bonds),
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
    """Validate a hand-built root record against native structure semantics.

    :param record: The exact root structure record to validate.
    :return: ``None`` after successful validation.
    :raises TypeError: If ``record`` is not an exact supported root record.
    :raises ValueError: If the record's normalized composition contradicts its native fields.
    """
    if _effective_record_type(record) not in (
        UnitcellStructureRecord,
        FundamentalDomainStructureRecord,
        ASUStructureRecord,
    ):
        raise TypeError("validate_structure_record expects an exact root structure record")
    _validate_normalized_composition(record)
    # the structure validates the bonds; a stored network must already be in its normal form
    if (
        not isinstance(record, UnitcellStructureRecord)
        and record.bonds is not None
        and _domain_structure_from_record(record).bonds != _site_bonds_from_record(record.bonds)
    ):
        raise ValueError("FundamentalDomainStructureRecord bonds are not the normalized bond network")


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
    symmetry = _stated_symmetry(structure.symmetry)
    values["symmetry"] = (
        None if symmetry is None else SymmetryRecord(**cast(dict[str, Any], SymmetryRecord.__httk_project__(symmetry)))
    )
    composition = (
        None
        if structure.chemical_composition is None
        else cast(dict[str, Any], ChemicalCompositionRecord.__httk_project__(structure.chemical_composition))
    )
    values["chemical_composition"] = (
        None
        if composition is None
        else ChemicalCompositionRecord(
            amounts=tuple(CompositionAmountRecord(*item) for item in composition["amounts"]),
            mode=composition["mode"],
        )
    )
    return UnitcellStructureRecord(**values)


# Numeric (all-float) records: the storage form of NumericUnitcellStructure. Every number is a
# plain float column; nothing carries exact text.


def _optional_float(value: Any, what: str) -> float | None:
    return None if value is None else finite_float(value, what)


@dataclass(frozen=True)
class NumericCellRecord:
    """Represent a float cell basis, precision and periodicity.

    :param basis: The 3x3 basis, one lattice vector per row.
    :param precision: The absolute basis precision, if stated.
    :param periodicity: The flags identifying periodic basis rows.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_cell",
        identity_name="atomistic_numeric_cell",
    )
    __httk_canonical_source__: ClassVar = NumericCell

    basis: Annotated[tuple[tuple[float, ...], ...], httk.core.storage.markers.Shape(3, 3)]
    precision: float | None
    periodicity: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", float_matrix(self.basis, 3, 3))
        object.__setattr__(self, "precision", float_precision(self.precision))
        object.__setattr__(self, "periodicity", to_periodicity(self.periodicity))

    @classmethod
    def __httk_project__(cls, cell: NumericCell) -> Mapping[str, object]:
        """Project a numeric cell into durable fields.

        :param cell: The cell to project.
        :return: The projected cell fields.
        """
        return {
            "basis": float_matrix(cast(Any, cell.basis), 3, 3),
            "precision": cell.precision,
            "periodicity": cell.periodicity,
        }


@dataclass(frozen=True)
class NumericSitesRecord:
    """Represent float reduced coordinates and their stated precision.

    :param reduced_coords: The reduced coordinates, one site per row.
    :param precision: The fractional coordinate precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_sites",
        identity_name="atomistic_numeric_sites",
    )
    __httk_canonical_source__: ClassVar = NumericSites

    reduced_coords: Annotated[tuple[tuple[float, ...], ...], httk.core.storage.markers.Shape(0, 3)]
    precision: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reduced_coords", float_matrix(self.reduced_coords, 3))
        object.__setattr__(self, "precision", float_precision(self.precision))

    @classmethod
    def __httk_project__(cls, sites: NumericSites) -> Mapping[str, object]:
        """Project numeric sites into durable fields.

        :param sites: The sites to project.
        :return: The projected sites fields.
        """
        return {"reduced_coords": float_matrix(cast(Any, sites.reduced_coords), 3), "precision": sites.precision}


@dataclass(frozen=True)
class NumericSpeciesConstituentRecord:
    """Represent one species constituent with float numbers.

    :param chemical_symbol: The constituent's chemical symbol.
    :param concentration: The constituent occupancy.
    :param mass: The constituent mass, if stated.
    :param charge: The constituent charge, if stated.
    :param spin: The constituent spin, if stated.
    :param concentration_precision: The occupancy precision, if stated.
    :param label: The constituent label, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_species_constituent",
        identity_name="atomistic_numeric_species_constituent",
    )

    chemical_symbol: str
    concentration: float
    mass: float | None = None
    charge: float | None = None
    spin: float | None = None
    concentration_precision: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chemical_symbol, str):
            raise TypeError("NumericSpeciesConstituentRecord chemical_symbol must be a string")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("NumericSpeciesConstituentRecord label must be a string or None")
        object.__setattr__(self, "concentration", finite_float(self.concentration, "concentration"))
        for name in ("mass", "charge", "spin", "concentration_precision"):
            object.__setattr__(self, name, _optional_float(getattr(self, name), name))


@dataclass(frozen=True)
class NumericSpeciesRecord:
    """Represent a species whose numbers are stored as floats.

    :param name: The species name.
    :param constituents: The aligned constituent records.
    :param original_name: The source species name, if stated.
    :param attached: The attached constituent symbols, if stated.
    :param nattached: The counts corresponding to ``attached``, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_species",
        identity_name="atomistic_numeric_species",
    )
    __httk_canonical_source__: ClassVar = Species

    name: str
    constituents: tuple[NumericSpeciesConstituentRecord, ...]
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        constituents = tuple(self.constituents)
        if not constituents or not all(isinstance(value, NumericSpeciesConstituentRecord) for value in constituents):
            raise TypeError("NumericSpeciesRecord constituents must contain NumericSpeciesConstituentRecord values")
        object.__setattr__(self, "constituents", constituents)
        for name in ("attached", "nattached"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else tuple(value))

    @classmethod
    def __httk_project__(cls, species: Species) -> Mapping[str, object]:
        """Project a species into float fields.

        :param species: The species to project.
        :return: The projected species fields.
        """

        def at(values: tuple[Any, ...] | None, index: int) -> Any:
            return None if values is None else values[index]

        return {
            "name": species.name,
            "constituents": tuple(
                NumericSpeciesConstituentRecord(
                    chemical_symbol=symbol,
                    concentration=float(concentration),
                    mass=at(species.mass, index),
                    charge=at(species.charges, index),
                    spin=at(species.spins, index),
                    concentration_precision=at(species.concentration_precision, index),
                    label=at(species.labels, index),
                )
                for index, (symbol, concentration) in enumerate(zip(species.chemical_symbols, species.concentration))
            ),
            "original_name": species.original_name,
            "attached": species.attached,
            "nattached": species.nattached,
        }


@dataclass(frozen=True)
class NumericCompositionAmountRecord:
    """Represent one normalized composition ratio as floats.

    :param element: The element symbol.
    :param ratio: The normalized element ratio.
    :param amount: The element amount.
    :param precision: The amount precision, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_composition_amount",
        identity_name="atomistic_numeric_composition_amount",
    )

    element: str
    ratio: float
    amount: float
    precision: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, str) or not self.element:
            raise TypeError("NumericCompositionAmountRecord element must be a non-empty string")
        object.__setattr__(self, "ratio", finite_float(self.ratio, "ratio"))
        object.__setattr__(self, "amount", finite_float(self.amount, "amount"))
        object.__setattr__(self, "precision", float_precision(self.precision))


@dataclass(frozen=True)
class NumericNormalizedCompositionRecord:
    """Represent the elemental composition of a numeric structure as floats.

    Each ratio and amount is ``float()`` of the exact value computed from the structure's
    float-quantised species, so the ratios need not sum to exactly one.

    :param amounts: The element amounts, ordered by element.
    :param complete: Whether the composition accounts for all structure content.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_normalized_composition",
        identity_name="atomistic_numeric_normalized_composition",
    )
    __httk_canonical_source__: ClassVar = Composition

    amounts: tuple[NumericCompositionAmountRecord, ...]
    complete: bool

    def __post_init__(self) -> None:
        amounts = tuple(self.amounts)
        if not all(isinstance(value, NumericCompositionAmountRecord) for value in amounts):
            raise TypeError(
                "NumericNormalizedCompositionRecord amounts must contain NumericCompositionAmountRecord values"
            )
        if not isinstance(self.complete, bool):
            raise TypeError("NumericNormalizedCompositionRecord complete must be a bool")
        object.__setattr__(self, "amounts", amounts)

    @classmethod
    def __httk_project__(cls, result: Composition) -> Mapping[str, object]:
        """Project an exact composition into float fields.

        :param result: The composition to project.
        :return: The projected composition fields.
        """
        exact = cast(dict[str, Any], NormalizedCompositionRecord.__httk_project__(result))
        return {
            "amounts": tuple(
                NumericCompositionAmountRecord(
                    element, float(ratio), float(amount), _optional_float(precision, "precision")
                )
                for element, ratio, amount, precision in exact["amounts"]
            ),
            "complete": exact["complete"],
        }


def _numeric_composition_record(structure: NumericUnitcellStructure) -> NumericNormalizedCompositionRecord:
    projected = NumericNormalizedCompositionRecord.__httk_project__(structure._exact.composition)
    return NumericNormalizedCompositionRecord(**cast(dict[str, Any], projected))


@dataclass(frozen=True)
class NumericUnitcellStructureRecord:
    """Represent the all-float durable backing for a numeric unit-cell structure.

    The storage form of :class:`~httk.atomistic.NumericUnitcellStructure`: every number is a
    plain float, with no exact companion text. Hand-built records are shape-checked on
    construction and semantically validated at the storage boundary.

    :param cell: The float cell record.
    :param sites: The float site-coordinate record.
    :param species: The distinct species records.
    :param species_at_sites: The species name occupying each site.
    :param normalized_composition: The float normalized composition.
    :param charge: The explicitly assigned cell charge, if stated.
    :param molecular: Whether the structure describes molecular entities.
    :param chemical_formula_descriptive: The descriptive formula, if stated.
    :param chemical_formula_hill: The Hill formula, if stated.
    :param optimization_type: The optimization provenance, if stated.
    :param collinear_site_moments: The signed collinear moment of each site, if the structure
        states collinear moments.
    :param cartesian_site_moments: The Cartesian moment of each site, one row per site, if the
        structure states Cartesian moments.
    :param site_moments_precision: How precisely the stated moments were given, if known.
    :param source_file: The verbatim file this structure was read from, when the entry records an input file exactly
        as given; outside content identity while ``None``.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    :raises ValueError: If both moment fields are set, the moments do not match the sites, a
        moment precision is given without moments, or a moment is non-finite.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_numeric_unitcell_structure",
        identity_name="atomistic_numeric_unitcell_structure",
        indexes=(("last_modified",), ("optimization_type",)),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = NumericUnitcellStructure

    cell: NumericCellRecord
    sites: NumericSitesRecord
    species: tuple[NumericSpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    normalized_composition: NumericNormalizedCompositionRecord
    charge: float | None = None
    molecular: bool = False
    chemical_formula_descriptive: str | None = None
    chemical_formula_hill: str | None = None
    optimization_type: str | None = None
    collinear_site_moments: tuple[float, ...] | None = None
    cartesian_site_moments: Annotated[tuple[tuple[float, ...], ...], httk.core.storage.markers.Shape(0, 3)] | None = (
        None
    )
    site_moments_precision: float | None = None
    source_file: Annotated[FileRecord | None, IdentitySkipWhenNone()] = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "NumericUnitcellStructureRecord") -> None:
        """Validate the record by rebuilding its value, and check its stored composition.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        :raises ValueError: If the record does not describe a valid structure or its
            composition contradicts its species.
        """
        value = numeric_structure_from_record(record)
        if record.normalized_composition != _numeric_composition_record(value):
            raise ValueError("NumericUnitcellStructureRecord normalized_composition contradicts its species")

    @classmethod
    def from_structure(cls, structure: NumericUnitcellStructure) -> "NumericUnitcellStructureRecord":
        """Build the complete record of a numeric structure, with ``source_file`` unset.

        Use :func:`dataclasses.replace` on the result to set the record-only ``source_file``.

        :param structure: The numeric structure to record.
        :return: The record, with every nested child record built.
        """
        return cls(**cast(dict[str, Any], cls.__httk_project__(structure)))

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"

    def __post_init__(self) -> None:
        name = type(self).__name__
        for field_name, record_type in (
            ("cell", NumericCellRecord),
            ("sites", NumericSitesRecord),
            ("normalized_composition", NumericNormalizedCompositionRecord),
        ):
            if not isinstance(getattr(self, field_name), record_type):
                raise TypeError(f"{name} {field_name} must be a {record_type.__name__}")
        species = tuple(self.species)
        if not all(isinstance(value, NumericSpeciesRecord) for value in species):
            raise TypeError(f"{name} species must contain NumericSpeciesRecord values")
        species_at_sites = tuple(self.species_at_sites)
        if not all(isinstance(value, str) for value in species_at_sites):
            raise TypeError(f"{name} species_at_sites must contain strings")
        if not isinstance(self.molecular, bool):
            raise TypeError(f"{name} molecular must be a bool")
        if self.source_file is not None and not isinstance(self.source_file, FileRecord):
            raise TypeError(f"{name} source_file must be a FileRecord or None")
        collinear = self.collinear_site_moments
        if collinear is not None:
            collinear = tuple(finite_float(value, "collinear_site_moments") for value in collinear)
        cartesian = self.cartesian_site_moments
        if cartesian is not None:
            cartesian = float_matrix(cartesian, 3)
        if collinear is not None and cartesian is not None:
            raise ValueError(f"{name} sets both collinear_site_moments and cartesian_site_moments; at most one")
        moments = collinear if collinear is not None else cartesian
        if moments is None and self.site_moments_precision is not None:
            raise ValueError(f"{name} site_moments_precision requires site moments")
        if moments is not None and len(moments) != len(species_at_sites):
            raise ValueError(f"{name} site moments must have one entry per site")
        if moments is not None and not moments:  # mirrors NumericUnitcellStructure
            raise ValueError(f"{name} site moments of a structure without sites must be None")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "species_at_sites", species_at_sites)
        object.__setattr__(self, "charge", _optional_float(self.charge, "charge"))
        object.__setattr__(self, "collinear_site_moments", collinear)
        object.__setattr__(self, "cartesian_site_moments", cartesian)
        object.__setattr__(self, "site_moments_precision", float_precision(self.site_moments_precision))

    @classmethod
    def __httk_project__(cls, structure: NumericUnitcellStructure) -> Mapping[str, object]:
        """Project a numeric structure into durable fields, child records included.

        :param structure: The numeric structure to project.
        :return: The projected structure fields.
        """
        moments = structure.site_moments
        kind, values = (None, None) if moments is None else (moments.kind, moments._moments)
        return {
            "cell": NumericCellRecord(**cast(dict[str, Any], NumericCellRecord.__httk_project__(structure.cell))),
            "sites": NumericSitesRecord(**cast(dict[str, Any], NumericSitesRecord.__httk_project__(structure.sites))),
            "species": tuple(
                NumericSpeciesRecord(**cast(dict[str, Any], NumericSpeciesRecord.__httk_project__(species)))
                for species in structure.species
            ),
            "species_at_sites": structure.species_at_sites,
            "normalized_composition": _numeric_composition_record(structure),
            "charge": structure.charge,
            "molecular": structure.molecular,
            "chemical_formula_descriptive": structure.chemical_formula_descriptive,
            "chemical_formula_hill": structure.chemical_formula_hill,
            "optimization_type": structure.optimization_type,
            "collinear_site_moments": values if kind == "collinear" else None,
            "cartesian_site_moments": values if kind == "cartesian" else None,
            "site_moments_precision": None if moments is None else moments.precision,
            "source_file": None,
            "id": None,
            "immutable_id": None,
            "last_modified": structure.last_modified,
        }


def _numeric_species_from_record(record: NumericSpeciesRecord) -> Species:
    def column(name: str) -> tuple[Any, ...] | None:
        values = tuple(getattr(value, name) for value in record.constituents)
        return None if all(value is None for value in values) else values

    def exact(values: tuple[float | None, ...] | None) -> tuple[fractions.Fraction | None, ...] | None:
        return None if values is None else tuple(None if value is None else any_to_fraction(value) for value in values)

    return Species(
        record.name,
        tuple(value.chemical_symbol for value in record.constituents),
        tuple(any_to_fraction(value.concentration) for value in record.constituents),
        mass=column("mass"),
        original_name=record.original_name,
        attached=record.attached,
        nattached=record.nattached,
        concentration_precision=exact(column("concentration_precision")),
        charges=exact(column("charge")),
        spins=exact(column("spin")),
        labels=column("label"),
    )


def numeric_structure_from_record(record: NumericUnitcellStructureRecord) -> NumericUnitcellStructure:
    """Rebuild the numeric structure a record stores.

    The rebuild is exact: ``numeric_structure_from_record(NumericUnitcellStructureRecord.from_structure(value))``
    equals ``value`` and has its content identity.

    :param record: The numeric structure record, possibly fetched from a store.
    :return: The numeric structure.
    :raises ValueError: If the record does not describe a valid numeric structure.
    """
    moments = (
        record.collinear_site_moments if record.collinear_site_moments is not None else record.cartesian_site_moments
    )
    return NumericUnitcellStructure(
        NumericCell(record.cell.basis, precision=record.cell.precision, periodicity=record.cell.periodicity),
        NumericSites(record.sites.reduced_coords, precision=record.sites.precision),
        tuple(_numeric_species_from_record(value) for value in record.species),
        record.species_at_sites,
        site_moments=None if moments is None else NumericSiteMoments(moments, precision=record.site_moments_precision),
        charge=record.charge,
        molecular=record.molecular,
        chemical_formula_descriptive=record.chemical_formula_descriptive,
        chemical_formula_hill=record.chemical_formula_hill,
        optimization_type=record.optimization_type,
        immutable_id=record.immutable_id,
        last_modified=record.last_modified,
    )


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
    """Represent a bounded numeric summary for one trajectory observable.

    :param name: The observable name.
    :param first: The first finite value, if available.
    :param last: The last finite value, if available.
    :param minimum: The minimum finite value, if available.
    :param maximum: The maximum finite value, if available.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_trajectory_observable_summary",
        identity_name="atomistic_trajectory_observable_summary",
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
    """Represent bounded trajectory identity and reference-frame summary.

    Frame data is never stored in this record. Hand-built records are shape-checked on
    construction and semantically validated at the storage boundary or explicitly through
    the validation hook.

    :param nframes: The total number of trajectory frames.
    :param species: The distinct durable species records.
    :param species_at_sites: The species name occupying each site.
    :param reference_frame_indexes: The sorted indexes of retained reference frames.
    :param reference_frame_structures: The retained reference-frame records.
    :param observable_summaries: The summaries of trajectory observables.
    :param source_locator: The source locator, if stated.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_trajectory",
        identity_name="atomistic_trajectory",
        indexes=(("last_modified",), ("nframes",)),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = cast(type[TrajectoryAPI], TrajectoryAPI)

    nframes: int
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    reference_frame_indexes: tuple[int, ...]
    reference_frame_structures: tuple[UnitcellStructureRecord, ...]
    observable_summaries: tuple[ObservableSummaryRecord, ...]
    source_locator: Annotated[str | None, IdentitySkip()] = field(default=None, compare=False)
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    @classmethod
    def __httk_validate__(cls, record: "TrajectoryRecord") -> None:
        """Validate the semantic trajectory record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        _validate_trajectory_record(record)

    def __post_init__(self) -> None:
        if not isinstance(self.nframes, int) or isinstance(self.nframes, bool) or self.nframes < 1:
            raise ValueError("TrajectoryRecord nframes must be a positive integer")
        species = tuple(self.species)
        if not species or not all(_effective_record_type(value) is SpeciesRecord for value in species):
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
        if len(indexes) != len(structures) or not all(
            _effective_record_type(value) is UnitcellStructureRecord for value in structures
        ):
            raise TypeError("TrajectoryRecord reference frames must match UnitcellStructureRecord values")
        summaries = tuple(self.observable_summaries)
        if not all(_effective_record_type(value) is ObservableSummaryRecord for value in summaries):
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
        """Expose the OPTIMADE entry type.

        :return: ``trajectories``.
        """
        return "trajectories"

    @classmethod
    def __httk_project__(cls, trajectory: TrajectoryAPI) -> Mapping[str, object]:
        """Project a trajectory into its bounded durable summary.

        Reference frames default to the first and last frame when the source does not provide
        an explicit reference list.

        :param trajectory: The trajectory to project.
        :return: The projected trajectory fields.
        """
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
            "id": None,
            "immutable_id": None,
            "last_modified": getattr(trajectory, "last_modified", None),
        }


@dataclass(frozen=True)
class WyckoffOccupationRecord:
    """Represent one occupied standard-setting Wyckoff orbit and its real species class.

    The occupation carries the real (possibly disordered) species class as a ``SpeciesRecord``; it is
    the durable analogue of :class:`~httk.atomistic.models.protostructure.occupation.WyckoffOccupation`.

    :param wyckoff: The Wyckoff letter in the standard setting.
    :param species: The durable real species occupying the orbit.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_wyckoff_occupation",
        identity_name="atomistic_wyckoff_occupation",
    )
    __httk_canonical_source__: ClassVar = WyckoffOccupation

    wyckoff: str
    species: SpeciesRecord

    @classmethod
    def __httk_project__(cls, occupation: WyckoffOccupation) -> Mapping[str, object]:
        """Project one Wyckoff occupation into durable fields.

        :param occupation: The occupation to project.
        :return: The projected occupation fields.
        """
        return {"wyckoff": occupation.wyckoff, "species": occupation.species}

    def __post_init__(self) -> None:
        if not isinstance(self.wyckoff, str) or len(self.wyckoff) != 1:
            raise ValueError("WyckoffOccupationRecord wyckoff must be a single letter")
        if _effective_record_type(self.species) is not SpeciesRecord:
            raise TypeError("WyckoffOccupationRecord species must be a SpeciesRecord")


@dataclass(frozen=True)
class BareProtostructureRecord:
    """Store a Wyckoff-only protostructure without geometrical refinement.

    Labels are query conveniences, not unique identities.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry.
    :param occupations: The canonical occupied Wyckoff positions.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The public lineage id, if assigned.
    :param immutable_id: The revision id, if assigned.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_bare_protostructure",
        identity_name="atomistic_bare_protostructure",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = BareProtostructure

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[WyckoffOccupationRecord, ...]
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _LABEL_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the deterministic Wyckoff-only label for querying."""
        if self.undetermined is not None:
            return self.undetermined
        return _protostructure_record_label(self)

    def __post_init__(self) -> None:
        _validate_protostructure_fields(type(self).__name__, self)

    @classmethod
    def __httk_validate__(cls, record: "BareProtostructureRecord") -> None:
        """Validate semantic consistency and canonical occupation order.

        :param record: The record to validate.
        """
        _require_species_classes("BareProtostructureRecord", "occupation", (o.species for o in record.occupations))
        canonical = _bare_protostructure_record_from_value(_bare_protostructure_from_record(record))
        _require_canonical("BareProtostructureRecord", "occupations", record.occupations, canonical.occupations)

    @classmethod
    def __httk_project__(cls, value: BareProtostructure) -> Mapping[str, object]:
        """Project a bare value into durable fields.

        :param value: The bare value to project.
        :return: The projected record fields.
        """
        if (fields := _undetermined_fields(value)) is not None:
            return {**fields, "canonical_form": _LABEL_FORM_VERSION, "id": None, "immutable_id": None}
        return {
            "spacegroup_it_number": value.spacegroup.it_number,
            "spacegroup_hall_entry": value.spacegroup.hall_entry,
            "occupations": value.occupations,
            "undetermined": None,
            "canonical_form": _LABEL_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


@dataclass(frozen=True)
class ProtostructureRecord:
    """Represent the durable backing for an assigned-species classification key.

    The record carries exactly the value identity of
    :class:`~httk.atomistic.models.protostructure.protostructure.Protostructure`: its
    standard-setting space group and its occupied Wyckoff positions with real species,
    in canonical order, plus an exact representative and/or discriminator.
    The record's content identity is independent of its storage layout, and two equal
    protostructures produce the same content identity.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry that names the stored Wyckoff data.
    :param occupations: The occupied Wyckoff positions and their real species.
    :param representative: The optional durable exact class anchor.
    :param discriminator: The optional external class discriminator.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_protostructure",
        identity_name="atomistic_protostructure",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = Protostructure

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[WyckoffOccupationRecord, ...]
    representative: FundamentalDomainStructureRecord | None = None
    discriminator: str | None = None
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _REFINED_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the httk protostructure label as a deterministic query column.

        The label is the httk protostructure label
        (``"AB_cF8_225_a_b:Na-Cl"`` for rocksalt): the prototype label of the erased
        template followed by ``:`` and the species class names. It is a convenience and
        query column only; it is not the record's identity (the content id is). The class
        names determine the stored species (a class name is derived from the species'
        constituents, concentrations and attached atoms alone), but the label omits the
        representative and the discriminator, so it is NOT unique across distinct refined
        protostructures and a ``GROUP BY label`` may under-count them — count and deduplicate
        by row (content id), never by label.

        :return: The httk protostructure label.
        """
        if self.undetermined is not None:
            return self.undetermined
        return _protostructure_record_label(self)

    def __post_init__(self) -> None:
        _validate_protostructure_fields(type(self).__name__, self)
        if self.representative is not None and not isinstance(self.representative, FundamentalDomainStructureRecord):
            raise TypeError("ProtostructureRecord representative must be a FundamentalDomainStructureRecord or None")
        _validate_discriminator(type(self).__name__, self.discriminator)
        if self.representative is None and self.discriminator is None and self.undetermined is None:
            raise ValueError(f"{type(self).__name__} requires a representative or discriminator")

    @classmethod
    def __httk_validate__(cls, record: "ProtostructureRecord") -> None:
        """Validate the semantic protostructure record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        _require_species_classes("ProtostructureRecord", "occupation", (o.species for o in record.occupations))
        if record.representative is not None:
            _require_species_classes("ProtostructureRecord", "representative", record.representative.species)
        canonical = _protostructure_record_from_value(_protostructure_from_record(record))
        _require_canonical("ProtostructureRecord", "occupations", record.occupations, canonical.occupations)
        _require_canonical("ProtostructureRecord", "representative", record.representative, canonical.representative)

    @classmethod
    def __httk_project__(cls, protostructure: Protostructure) -> Mapping[str, object]:
        """Project a protostructure into durable fields.

        :param protostructure: The protostructure to project.
        :return: The projected protostructure fields.
        """
        if (fields := _undetermined_fields(protostructure)) is not None:
            return {
                **fields,
                "representative": None,
                "discriminator": None,
                "canonical_form": _REFINED_FORM_VERSION,
                "id": None,
                "immutable_id": None,
            }
        return {
            "spacegroup_it_number": protostructure.spacegroup.it_number,
            "spacegroup_hall_entry": protostructure.spacegroup.hall_entry,
            "occupations": protostructure.occupations,
            "representative": protostructure.representative,
            "discriminator": protostructure.discriminator,
            "undetermined": None,
            "canonical_form": _REFINED_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


@dataclass(frozen=True)
class FundamentalDomainTemplateRecord:
    """Represent the durable backing for a standard-setting dummy-species fundamental domain.

    The record carries the geometric per-structure fundamental-domain template: its
    standard-setting space group, its cell (surd-capable), the symmetry-distinct Wyckoff
    sites with their exact free parameters, and the distinct dummy species. Distinct templates
    with different free parameters are distinct values, so no content deduplication is
    expected; the content identity remains deterministic.

    :param cell: The durable standard-setting cell record.
    :param wyckoff_sites: The symmetry-distinct durable Wyckoff site records.
    :param species: The distinct durable dummy species records.
    :param spacegroup_it_number: The International Tables space-group number.
    :param spacegroup_hall_entry: The standard-setting Hall entry that names the stored Wyckoff data.
    :param coordinate_precision: The reduced-coordinate precision, if stated.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_fundamental_domain_template",
        identity_name="atomistic_fundamental_domain_template",
        indexes=(("spacegroup_it_number",), ("spacegroup_hall_entry",)),
    )
    __httk_canonical_source__: ClassVar = FundamentalDomainTemplate

    cell: CellRecord
    wyckoff_sites: tuple[WyckoffSiteRecord, ...]
    species: tuple[SpeciesRecord, ...]
    spacegroup_it_number: int
    spacegroup_hall_entry: str
    coordinate_precision: fractions.Fraction | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cell, CellRecord):
            raise TypeError("FundamentalDomainTemplateRecord cell must be a CellRecord")
        wyckoff_sites = tuple(self.wyckoff_sites)
        if not all(_effective_record_type(value) is WyckoffSiteRecord for value in wyckoff_sites):
            raise TypeError("FundamentalDomainTemplateRecord wyckoff_sites must contain WyckoffSiteRecord values")
        species = tuple(self.species)
        if not all(_effective_record_type(value) is SpeciesRecord for value in species):
            raise TypeError("FundamentalDomainTemplateRecord species must contain SpeciesRecord values")
        names = tuple(value.name for value in species)
        if len(names) != len(set(names)):
            raise ValueError("FundamentalDomainTemplateRecord species names must be unique")
        if not isinstance(self.spacegroup_it_number, int) or isinstance(self.spacegroup_it_number, bool):
            raise TypeError("FundamentalDomainTemplateRecord spacegroup_it_number must be an integer")
        if not 1 <= self.spacegroup_it_number <= 230:
            raise ValueError("FundamentalDomainTemplateRecord spacegroup_it_number must be in [1, 230]")
        if not isinstance(self.spacegroup_hall_entry, str) or not self.spacegroup_hall_entry:
            raise TypeError("FundamentalDomainTemplateRecord spacegroup_hall_entry must be a non-empty string")
        if unknown := {value.species for value in wyckoff_sites} - set(names):
            raise ValueError(f"FundamentalDomainTemplateRecord references unknown species: {sorted(unknown)!r}")
        object.__setattr__(self, "wyckoff_sites", wyckoff_sites)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "coordinate_precision", to_precision(self.coordinate_precision))

    @classmethod
    def __httk_validate__(cls, record: "FundamentalDomainTemplateRecord") -> None:
        """Validate the semantic fundamental-domain-template record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        canonical = _fundamental_domain_template_record_from_value(_fundamental_domain_template_from_record(record))
        _require_canonical(
            "FundamentalDomainTemplateRecord", "wyckoff_sites", record.wyckoff_sites, canonical.wyckoff_sites
        )

    @classmethod
    def __httk_project__(cls, template: FundamentalDomainTemplate) -> Mapping[str, object]:
        """Project a fundamental-domain template into durable fields.

        :param template: The fundamental-domain template to project.
        :return: The projected template fields.
        """
        return {
            "cell": template.cell,
            "wyckoff_sites": template.wyckoff_sites,
            "species": template.species,
            "spacegroup_it_number": template.spacegroup.it_number,
            "spacegroup_hall_entry": template.spacegroup.hall_entry,
            "coordinate_precision": template.coordinate_precision,
            "id": None,
            "immutable_id": None,
        }


@dataclass(frozen=True)
class BarePrototypeRecord:
    """Store a Wyckoff-only prototype without geometrical refinement.

    Labels are query conveniences, not unique identities.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry.
    :param occupations: The canonical occupied Wyckoff positions.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The public lineage id, if assigned.
    :param immutable_id: The revision id, if assigned.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_bare_prototype",
        identity_name="atomistic_bare_prototype",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = BarePrototype

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[PrototypeOccupation, ...]
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _LABEL_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the deterministic Wyckoff-only label for querying."""
        if self.undetermined is not None:
            return self.undetermined
        return _prototype_label_from_fields(cast(str, self.spacegroup_hall_entry), self.occupations)

    def __post_init__(self) -> None:
        _validate_prototype_fields(type(self).__name__, self)

    @classmethod
    def __httk_validate__(cls, record: "BarePrototypeRecord") -> None:
        """Validate semantic consistency and canonical occupation order.

        :param record: The record to validate.
        """
        canonical = _bare_prototype_record_from_value(_bare_prototype_from_record(record))
        _require_canonical("BarePrototypeRecord", "occupations", record.occupations, canonical.occupations)

    @classmethod
    def __httk_project__(cls, value: BarePrototype) -> Mapping[str, object]:
        """Project a bare value into durable fields.

        :param value: The bare value to project.
        :return: The projected record fields.
        """
        if (fields := _undetermined_fields(value)) is not None:
            return {**fields, "canonical_form": _LABEL_FORM_VERSION, "id": None, "immutable_id": None}
        return {
            "spacegroup_it_number": value.spacegroup.it_number,
            "spacegroup_hall_entry": value.spacegroup.hall_entry,
            "occupations": value.occupations,
            "undetermined": None,
            "canonical_form": _LABEL_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


@dataclass(frozen=True)
class PrototypeRecord:
    """Represent the durable backing for an anonymous prototype.

    The record carries anonymous class-partitioned Wyckoff occupations, plus an optional
    exact fundamental-domain-template representative and/or discriminator. At least
    one refinement is required. The discriminator is species-independent and is not part of the label.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry that names the stored Wyckoff data.
    :param occupations: The occupied Wyckoff positions and anonymous class labels.
    :param representative: The durable class representative, if one is held.
    :param discriminator: The externally assigned class discriminator, if one is held.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_prototype",
        identity_name="atomistic_prototype",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = Prototype

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[PrototypeOccupation, ...]
    representative: FundamentalDomainTemplateRecord | None = None
    discriminator: str | None = None
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _REFINED_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the httk prototype label as a deterministic query column.

        The discriminator names the geometrical class and is not part of the label, so
        prototypes that share occupations but differ in class collide on this column; it
        is a convenience and query column only, not the record's identity (the content id is).

        :return: The httk prototype label.
        """
        if self.undetermined is not None:
            return self.undetermined
        return _prototype_label_from_fields(cast(str, self.spacegroup_hall_entry), self.occupations)

    def __post_init__(self) -> None:
        _validate_prototype_fields(type(self).__name__, self)
        if self.representative is not None and _effective_record_type(self.representative) is not (
            FundamentalDomainTemplateRecord
        ):
            raise TypeError("PrototypeRecord representative must be a FundamentalDomainTemplateRecord or None")
        _validate_discriminator(type(self).__name__, self.discriminator)
        if self.representative is None and self.discriminator is None and self.undetermined is None:
            raise ValueError(f"{type(self).__name__} requires a representative or discriminator")

    @classmethod
    def __httk_validate__(cls, record: "PrototypeRecord") -> None:
        """Validate the semantic prototype record.

        :param record: The record to validate.
        :return: ``None`` after successful validation.
        """
        canonical = _prototype_record_from_value(_prototype_from_record(record))
        _require_canonical("PrototypeRecord", "occupations", record.occupations, canonical.occupations)

    @classmethod
    def __httk_project__(cls, prototype: Prototype) -> Mapping[str, object]:
        """Project a prototype into durable fields.

        :param prototype: The prototype to project.
        :return: The projected prototype fields.
        """
        if (fields := _undetermined_fields(prototype)) is not None:
            return {
                **fields,
                "representative": None,
                "discriminator": None,
                "canonical_form": _REFINED_FORM_VERSION,
                "id": None,
                "immutable_id": None,
            }
        template = prototype
        return {
            "spacegroup_it_number": template.spacegroup.it_number,
            "spacegroup_hall_entry": template.spacegroup.hall_entry,
            "occupations": tuple(template.occupations),
            "representative": prototype.representative,
            "discriminator": prototype.discriminator,
            "undetermined": None,
            "canonical_form": _REFINED_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


def _validate_undetermined_fields(record_name: str, record: Any) -> bool:
    """Validate the fields of an undetermined record and return whether the record is undetermined.

    An undetermined record stores its label, the canonical-form version and, when the label
    determines it, the space group; occupations, net, representative and discriminator are empty.
    """
    if record.undetermined is None:
        return False
    if not isinstance(record.undetermined, str) or not record.undetermined:
        raise TypeError(f"{record_name} undetermined must be a non-empty string or None")
    if (record.spacegroup_it_number is None) != (record.spacegroup_hall_entry is None):
        raise ValueError(f"{record_name} stores both or neither of spacegroup_it_number and spacegroup_hall_entry")
    if record.spacegroup_it_number is not None:
        _validate_spacegroup_fields(record_name, record)
    if not isinstance(record.canonical_form, int) or isinstance(record.canonical_form, bool):
        raise TypeError(f"{record_name} canonical_form must be an integer")
    if (
        tuple(record.occupations)
        or tuple(getattr(record, "net", ()))
        or getattr(record, "representative", None) is not None
        or getattr(record, "discriminator", None) is not None
    ):
        raise ValueError(f"{record_name} of an undetermined value stores only its label and space group")
    object.__setattr__(record, "occupations", ())
    return True


def _validate_spacegroup_fields(record_name: str, record: Any) -> None:
    """Validate the IT number and Hall entry fields of a Wyckoff record."""
    if not isinstance(record.spacegroup_it_number, int) or isinstance(record.spacegroup_it_number, bool):
        raise TypeError(f"{record_name} spacegroup_it_number must be an integer")
    if not 1 <= record.spacegroup_it_number <= 230:
        raise ValueError(f"{record_name} spacegroup_it_number must be in [1, 230]")
    if not isinstance(record.spacegroup_hall_entry, str) or not record.spacegroup_hall_entry:
        raise TypeError(f"{record_name} spacegroup_hall_entry must be a non-empty string")


def _validate_protostructure_fields(record_name: str, record: Any) -> None:
    """Validate assigned Wyckoff storage fields shared by bare and refined records."""
    if _validate_undetermined_fields(record_name, record):
        return
    _validate_spacegroup_fields(record_name, record)
    if not isinstance(record.canonical_form, int) or isinstance(record.canonical_form, bool):
        raise TypeError(f"{record_name} canonical_form must be an integer")
    occupations = tuple(record.occupations)
    if not occupations or not all(_effective_record_type(value) is WyckoffOccupationRecord for value in occupations):
        raise TypeError(f"{record_name} occupations must contain WyckoffOccupationRecord values")
    object.__setattr__(record, "occupations", occupations)


def _validate_prototype_fields(record_name: str, record: Any) -> None:
    """Validate the space group and anonymous occupations of a prototype record."""
    if _validate_undetermined_fields(record_name, record):
        return
    _validate_spacegroup_fields(record_name, record)
    if not isinstance(record.canonical_form, int) or isinstance(record.canonical_form, bool):
        raise TypeError(f"{record_name} canonical_form must be an integer")
    occupations = tuple(record.occupations)
    if not occupations or not all(isinstance(value, PrototypeOccupation) for value in occupations):
        raise ValueError(f"{record_name} occupations must be a non-empty tuple of PrototypeOccupation values")
    object.__setattr__(record, "occupations", occupations)


def _validate_discriminator(record_name: str, discriminator: str | None) -> None:
    """Validate an optional externally assigned geometrical-class discriminator."""
    if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
        raise ValueError(f"{record_name} discriminator must be a non-empty string when given")


def _require_species_classes(record_name: str, field: str, records: Iterable[SpeciesRecord]) -> None:
    """Reject stored species that are not species classes (rows from before the classes).

    :param record_name: The record class name, for the error message.
    :param field: The field holding the species, for the error message.
    :param records: The stored species.
    :raises ValueError: If a stored species is not its own species class.
    """
    for record in records:
        species = _species_from_record(record)
        if _species_class(species) != species:
            raise ValueError(
                f"{record_name} {field} species {species.name!r} is not a reduced species class; "
                f"store a {record_name} built from its value, or rebuild the database"
            )


def _require_canonical(record_name: str, field: str, stored: Any, canonical: Any) -> None:
    """Reject stored fields that reconstruct to a value whose canonical form differs.

    The model constructor silently re-canonicalizes order, so a non-canonical stored record
    would reconstruct to an equal value yet hash to a different content id, breaking dedup.

    :param record_name: The record class name, for the error message.
    :param field: The non-canonical field name, for the error message.
    :param stored: The field value as stored.
    :param canonical: The field value the value's canonical projection would store.
    :raises ValueError: If ``stored`` differs from ``canonical``.
    """
    if stored != canonical:
        raise ValueError(
            f"{record_name} {field} are not in canonical order; store a {record_name} built from its value"
        )


def _require_canonical_form(record: Any, family: str, version: int) -> None:
    """Reject a canonical-label record built under another version of its family's canonical form.

    :param record: The durable record.
    :param family: The record's version family (``"label"``, ``"refined"`` or ``"net"``), for the message.
    :param version: The family's current canonical-form version.
    :raises ValueError: If ``record.canonical_form`` is not ``version``.
    """
    if record.canonical_form != version:
        raise ValueError(
            f"{type(record).__name__} was built with canonical {family} form {record.canonical_form}, but this "
            f"httk-atomistic uses {version}: the canonical {family} form changed; rebuild the database"
        )


def _undetermined_fields(value: Any) -> dict[str, Any] | None:
    """Return the record fields of an undetermined value: its label and its space group when known.

    :param value: A label value.
    :return: The fields, or ``None`` for a determined value.
    """
    parts = _undetermined(value)
    if parts is None:
        return None
    spacegroup = parts.spacegroup
    return {
        "spacegroup_it_number": None if spacegroup is None else spacegroup.it_number,
        "spacegroup_hall_entry": None if spacegroup is None else spacegroup.hall_entry,
        "occupations": (),
        "undetermined": value._undetermined,
    }


def _undetermined_from_record(record: Any, family: Any, version_family: str, version: int) -> Any:
    """Rebuild an undetermined value from its record, parsing the label strictly.

    :param record: The durable record, whose ``undetermined`` is set.
    :param family: The value class.
    :param version_family: The record's version family, for the message.
    :param version: The family's current canonical-form version.
    :return: The undetermined value.
    :raises ValueError: If the label is not an undetermined label of the family, the record was
        built under another canonical form, or the stored space group is not the label's.
    """
    _require_canonical_form(record, version_family, version)
    value = family.undetermined(record.undetermined)
    expected = cast(dict[str, Any], _undetermined_fields(value))
    if (record.spacegroup_it_number, record.spacegroup_hall_entry) != (
        expected["spacegroup_it_number"],
        expected["spacegroup_hall_entry"],
    ):
        raise ValueError(f"{type(record).__name__} space group contradicts its undetermined label")
    return value


def _protostructure_record_label(record: BareProtostructureRecord | ProtostructureRecord) -> str:
    """Render the httk protostructure label for a durable protostructure record.

    :param record: The durable protostructure record.
    :return: The httk protostructure label.
    """
    from httk.atomistic.models.prototype.notation import render_protostructure_label
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    return render_protostructure_label(
        spacegroup, [(occupation.wyckoff, occupation.species.name) for occupation in record.occupations]
    )


def _protostructure_from_record(record: ProtostructureRecord) -> Protostructure:
    """Reconstruct a protostructure value from its durable record.

    :param record: The durable protostructure record.
    :return: The reconstructed protostructure value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, Protostructure, "refined", _REFINED_FORM_VERSION)
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    _require_canonical_form(record, "refined", _REFINED_FORM_VERSION)
    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    representative = None if record.representative is None else _domain_structure_from_record(record.representative)
    return Protostructure(
        spacegroup,
        tuple(WyckoffOccupation(value.wyckoff, _species_from_record(value.species)) for value in record.occupations),
        representative=representative,
        discriminator=record.discriminator,
    )


def _fundamental_domain_template_from_record(record: FundamentalDomainTemplateRecord) -> FundamentalDomainTemplate:
    """Reconstruct a fundamental-domain template value from its durable record.

    :param record: The durable fundamental-domain-template record.
    :return: The reconstructed fundamental-domain template value.
    """
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    spacegroup = Spacegroup.from_hall_entry(record.spacegroup_hall_entry)
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    return FundamentalDomainTemplate(
        _cell_from_record(record.cell),
        spacegroup,
        tuple(
            WyckoffSite(value.wyckoff, FracVector(value.free_parameters), value.species)
            for value in record.wyckoff_sites
        ),
        tuple(_species_from_record(value) for value in record.species),
        record.coordinate_precision,
    )


def _protostructure_record_from_value(value: Protostructure) -> ProtostructureRecord:
    """Build a durable protostructure record from a protostructure value.

    :param value: The protostructure value to store.
    :return: The durable protostructure record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return ProtostructureRecord(**fields)
    return ProtostructureRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(
            WyckoffOccupationRecord(
                wyckoff=occupation.wyckoff,
                species=SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(occupation.species))),
            )
            for occupation in value.occupations
        ),
        representative=None
        if value.representative is None
        else _domain_structure_record_from_value(value.representative),
        discriminator=value.discriminator,
    )


def _fundamental_domain_template_record_from_value(
    value: FundamentalDomainTemplate,
) -> FundamentalDomainTemplateRecord:
    """Build a durable fundamental-domain-template record from a template value.

    :param value: The fundamental-domain template value to store.
    :return: The durable fundamental-domain-template record.
    """
    return FundamentalDomainTemplateRecord(
        cell=CellRecord(**cast(dict[str, Any], CellRecord.__httk_project__(value.cell))),
        wyckoff_sites=tuple(
            WyckoffSiteRecord(**cast(dict[str, Any], WyckoffSiteRecord.__httk_project__(site)))
            for site in value.wyckoff_sites
        ),
        species=tuple(
            SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(species))) for species in value.species
        ),
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        coordinate_precision=value.coordinate_precision,
    )


def _domain_structure_record_from_value(
    structure: FundamentalDomainStructure,
) -> FundamentalDomainStructureRecord:
    """Build a durable fundamental-domain (or asymmetric-unit) structure record from a value.

    The exact record class follows the value's storage opt-in, so an
    :class:`~httk.atomistic.ASUStructure` durably keeps its asymmetric-unit identity.

    :param structure: The fundamental-domain structure value to store.
    :return: The durable fundamental-domain (or asymmetric-unit) structure record.
    """
    from httk.core.storage import resolve_storage_record

    record_type = cast(type[FundamentalDomainStructureRecord], resolve_storage_record(structure))
    values: dict[str, Any] = dict(record_type.__httk_project__(structure))
    values["cell"] = CellRecord(**cast(dict[str, Any], CellRecord.__httk_project__(structure.cell)))
    values["domain_sites"] = tuple(
        WyckoffSiteRecord(**cast(dict[str, Any], WyckoffSiteRecord.__httk_project__(site)))
        for site in structure.domain_sites
    )
    values["species"] = tuple(
        SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(species))) for species in structure.species
    )
    values["setting_transform"] = SettingTransformRecord(
        **cast(dict[str, Any], SettingTransformRecord.__httk_project__(structure.transform))
    )
    values["normalized_composition"] = _normalized_composition_record_from_result(structure.composition)
    values["assemblies"] = (
        None
        if structure.assemblies is None
        else tuple(_assembly_record_from_assembly(assembly) for assembly in structure.assemblies)
    )
    values["chemical_composition"] = (
        None
        if structure.chemical_composition is None
        else ChemicalCompositionRecord(
            **cast(dict[str, Any], ChemicalCompositionRecord.__httk_project__(structure.chemical_composition))
        )
    )
    values["bonds"] = (
        None
        if structure.bonds is None
        else tuple(
            SiteBondRecord(**cast(dict[str, Any], SiteBondRecord.__httk_project__(bond))) for bond in structure.bonds
        )
    )
    return record_type(**values)


def _prototype_label_from_fields(spacegroup_hall_entry: str, occupations: tuple[PrototypeOccupation, ...]) -> str:
    """Render the httk prototype label from stored anonymous occupations.

    :param spacegroup_hall_entry: The standard-setting Hall entry.
    :param occupations: The occupied Wyckoff positions and anonymous labels.
    :return: The prototype label text.
    """
    from httk.atomistic.models.prototype.notation import render_prototype_label
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    spacegroup = Spacegroup.from_hall_entry(spacegroup_hall_entry)
    return render_prototype_label(spacegroup, [(occupation.wyckoff, occupation.label) for occupation in occupations])


def _prototype_from_record(record: PrototypeRecord) -> Prototype:
    """Reconstruct a prototype value from its durable record.

    The model constructor re-checks that any representative agrees with the stored
    anonymous occupations.

    :param record: The durable prototype record.
    :return: The reconstructed prototype value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, Prototype, "refined", _REFINED_FORM_VERSION)
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    _require_canonical_form(record, "refined", _REFINED_FORM_VERSION)
    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    representative = (
        None if record.representative is None else _fundamental_domain_template_from_record(record.representative)
    )
    return Prototype(
        spacegroup,
        tuple(record.occupations),
        representative=representative,
        discriminator=record.discriminator,
    )


def _prototype_record_from_value(value: Prototype) -> PrototypeRecord:
    """Build a durable prototype record from a prototype value.

    :param value: The prototype value to store.
    :return: The durable prototype record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return PrototypeRecord(**fields)
    representative = (
        None if value.representative is None else _fundamental_domain_template_record_from_value(value.representative)
    )
    return PrototypeRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(value.occupations),
        representative=representative,
        discriminator=value.discriminator,
    )


def _bare_protostructure_from_record(record: BareProtostructureRecord) -> BareProtostructure:
    """Reconstruct a protostructure value from its durable record.

    :param record: The durable protostructure record.
    :return: The reconstructed protostructure value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, BareProtostructure, "label", _LABEL_FORM_VERSION)
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    _require_canonical_form(record, "label", _LABEL_FORM_VERSION)
    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    return BareProtostructure(
        spacegroup,
        tuple(WyckoffOccupation(value.wyckoff, _species_from_record(value.species)) for value in record.occupations),
    )


def _bare_protostructure_record_from_value(value: BareProtostructure) -> BareProtostructureRecord:
    """Build a durable protostructure record from a protostructure value.

    :param value: The protostructure value to store.
    :return: The durable protostructure record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return BareProtostructureRecord(**fields)
    return BareProtostructureRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(
            WyckoffOccupationRecord(
                wyckoff=occupation.wyckoff,
                species=SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(occupation.species))),
            )
            for occupation in value.occupations
        ),
    )


def _bare_prototype_from_record(record: BarePrototypeRecord) -> BarePrototype:
    """Reconstruct a prototype value from its durable record.

    :param record: The durable prototype record.
    :return: The reconstructed prototype value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, BarePrototype, "label", _LABEL_FORM_VERSION)
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    _require_canonical_form(record, "label", _LABEL_FORM_VERSION)
    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    return BarePrototype(
        spacegroup,
        tuple(record.occupations),
    )


def _bare_prototype_record_from_value(value: BarePrototype) -> BarePrototypeRecord:
    """Build a durable prototype record from a prototype value.

    :param value: The prototype value to store.
    :return: The durable prototype record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return BarePrototypeRecord(**fields)
    return BarePrototypeRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(value.occupations),
    )


@dataclass(frozen=True)
class ProtonetBondRecord:
    """Represent one bonding-net row of a protonettype or protonetstructure.

    The durable analogue of a net row ``(i, j, k, n1, n2, n3)``.

    :param site: The 1-based site ``i`` of the reference atom.
    :param target: The 1-based site ``j`` of the bonded atom.
    :param operation: The 1-based general-position operation ``k`` (identity = 1).
    :param translation_a: The lattice translation ``n1``.
    :param translation_b: The lattice translation ``n2``.
    :param translation_c: The lattice translation ``n3``.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_protonet_bond",
        identity_name="atomistic_protonet_bond",
    )

    site: int
    target: int
    operation: int
    translation_a: int
    translation_b: int
    translation_c: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.row):
            raise TypeError("ProtonetBondRecord fields must be integers")

    @property
    def row(self) -> tuple[int, int, int, int, int, int]:
        """Return the net row ``(i, j, k, n1, n2, n3)``."""
        return (self.site, self.target, self.operation, self.translation_a, self.translation_b, self.translation_c)


@dataclass(frozen=True)
class ProtonettypeRecord:
    """Store an anonymous protonettype: a bare prototype with a bonding net.

    Occupations keep the value's site order, which the net's site numbers refer to.
    Labels are query conveniences, not unique identities.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry.
    :param occupations: The anonymous occupied Wyckoff positions in site order.
    :param net: The bonding-net rows in label order.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The public lineage id, if assigned.
    :param immutable_id: The revision id, if assigned.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_protonettype",
        identity_name="atomistic_protonettype",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = Protonettype

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[PrototypeOccupation, ...]
    net: tuple[ProtonetBondRecord, ...]
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _NET_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the protonettype label for querying."""
        if self.undetermined is not None:
            return self.undetermined
        return _protonet_record_label(self, [(o.label, o.wyckoff) for o in self.occupations], named=False)

    def __post_init__(self) -> None:
        _validate_prototype_fields(type(self).__name__, self)
        _validate_net_field(type(self).__name__, self)

    @classmethod
    def __httk_validate__(cls, record: "ProtonettypeRecord") -> None:
        """Validate that the record stores the canonical net of its bare prototype.

        :param record: The record to validate.
        """
        value = _protonettype_from_record(record)
        if not value.determined:
            return
        canonical = _protonettype_record_from_value(Protonettype(value.spacegroup, value.occupations, value.net))
        _require_canonical("ProtonettypeRecord", "occupations", record.occupations, canonical.occupations)
        _require_canonical("ProtonettypeRecord", "net rows", _net_rows(record), _net_rows(canonical))

    @classmethod
    def __httk_project__(cls, value: Protonettype) -> Mapping[str, object]:
        """Project a protonettype into durable fields.

        :param value: The protonettype to project.
        :return: The projected record fields.
        """
        if (fields := _undetermined_fields(value)) is not None:
            return {**fields, "net": (), "canonical_form": _NET_FORM_VERSION, "id": None, "immutable_id": None}
        return {
            "spacegroup_it_number": value.spacegroup.it_number,
            "spacegroup_hall_entry": value.spacegroup.hall_entry,
            "occupations": value.occupations,
            "net": _bond_records(value.net),
            "undetermined": None,
            "canonical_form": _NET_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


@dataclass(frozen=True)
class ProtonetstructureRecord:
    """Store an assigned-species protonetstructure: a bare protostructure with a bonding net.

    Occupations keep the value's site order, which the net's site numbers refer to.
    Labels are query conveniences, not unique identities.

    :param spacegroup_it_number: The International Tables space-group number, or ``None``.
    :param spacegroup_hall_entry: The standard-setting Hall entry.
    :param occupations: The occupied Wyckoff positions in site order.
    :param net: The bonding-net rows in label order.
    :param undetermined: The label of an undetermined value (``X`` where a part was not
        determined within the canonicalisation budget), which is then all the record stores
        besides the space group, when known; ``None`` for a determined value.
    :param canonical_form: The canonical-label rule version the record was built with.
    :param id: The public lineage id, if assigned.
    :param immutable_id: The revision id, if assigned.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_protonetstructure",
        identity_name="atomistic_protonetstructure",
        indexes=(("spacegroup_it_number",), ("label",)),
    )
    __httk_canonical_source__: ClassVar = Protonetstructure

    spacegroup_it_number: int | None
    spacegroup_hall_entry: str | None
    occupations: tuple[WyckoffOccupationRecord, ...]
    net: tuple[ProtonetBondRecord, ...]
    undetermined: Annotated[str | None, IdentitySkipWhenNone()] = None
    canonical_form: int = _NET_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @stored_property
    def label(self) -> str:
        """Expose the protonetstructure label for querying."""
        if self.undetermined is not None:
            return self.undetermined
        return _protonet_record_label(self, [(o.species.name, o.wyckoff) for o in self.occupations], named=True)

    def __post_init__(self) -> None:
        _validate_protostructure_fields(type(self).__name__, self)
        _validate_net_field(type(self).__name__, self)

    @classmethod
    def __httk_validate__(cls, record: "ProtonetstructureRecord") -> None:
        """Validate that the record stores the canonical net of its bare protostructure.

        :param record: The record to validate.
        """
        _require_species_classes("ProtonetstructureRecord", "occupation", (o.species for o in record.occupations))
        value = _protonetstructure_from_record(record)
        if not value.determined:
            return
        canonical = _protonetstructure_record_from_value(
            Protonetstructure(value.spacegroup, value.occupations, value.net)
        )
        _require_canonical("ProtonetstructureRecord", "occupations", record.occupations, canonical.occupations)
        _require_canonical("ProtonetstructureRecord", "net rows", _net_rows(record), _net_rows(canonical))

    @classmethod
    def __httk_project__(cls, value: Protonetstructure) -> Mapping[str, object]:
        """Project a protonetstructure into durable fields.

        :param value: The protonetstructure to project.
        :return: The projected record fields.
        """
        if (fields := _undetermined_fields(value)) is not None:
            return {**fields, "net": (), "canonical_form": _NET_FORM_VERSION, "id": None, "immutable_id": None}
        return {
            "spacegroup_it_number": value.spacegroup.it_number,
            "spacegroup_hall_entry": value.spacegroup.hall_entry,
            "occupations": value.occupations,
            "net": _bond_records(value.net),
            "undetermined": None,
            "canonical_form": _NET_FORM_VERSION,
            "id": None,
            "immutable_id": None,
        }


def _validate_net_field(record_name: str, record: Any) -> None:
    """Require a tuple of bond records in a protonet record's ``net`` field."""
    net = tuple(record.net)
    if not all(_effective_record_type(value) is ProtonetBondRecord for value in net):
        raise TypeError(f"{record_name} net must contain ProtonetBondRecord values")
    object.__setattr__(record, "net", net)


def _net_rows(record: ProtonettypeRecord | ProtonetstructureRecord) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return a protonet record's net as plain rows, in stored order."""
    return tuple(bond.row for bond in record.net)


def _bond_records(net: tuple[tuple[int, int, int, int, int, int], ...]) -> tuple[ProtonetBondRecord, ...]:
    """Return durable bond records for value net rows."""
    return tuple(ProtonetBondRecord(*row) for row in net)


def _protonet_record_label(
    record: ProtonettypeRecord | ProtonetstructureRecord, sites: list[tuple[str, str]], *, named: bool
) -> str:
    """Render the protonet label from stored fields, keeping the stored class order.

    :param record: The durable protonet record.
    :param sites: The ``(class key, Wyckoff letter)`` of each occupation, in site order.
    :param named: Whether to close the label with the ``:`` species suffix.
    :return: The protonet label text.
    """
    from httk.atomistic.models.prototype.notation import render_protonet
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    runs = [
        (key, tuple(letter for _, letter in run)) for key, run in itertools.groupby(sites, key=lambda pair: pair[0])
    ]
    return render_protonet(
        Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry)),
        [letters for _, letters in runs],
        _net_rows(record),
        [key for key, _ in runs] if named else None,
    )


def _protonet_record_spacegroup(record: ProtonettypeRecord | ProtonetstructureRecord) -> "Spacegroup":
    """Return a protonet record's space group, checking the Hall entry against the IT number."""
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    spacegroup = Spacegroup.from_hall_entry(cast(str, record.spacegroup_hall_entry))
    if spacegroup.it_number != record.spacegroup_it_number:
        raise ValueError("stored space-group Hall entry contradicts its International Tables number")
    return spacegroup


def _protonettype_from_record(record: ProtonettypeRecord) -> Protonettype:
    """Reconstruct a protonettype value from its durable record.

    :param record: The durable protonettype record.
    :return: The reconstructed protonettype value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, Protonettype, "net", _NET_FORM_VERSION)
    _require_canonical_form(record, "net", _NET_FORM_VERSION)
    return Protonettype._of(_protonet_record_spacegroup(record), tuple(record.occupations), _net_rows(record))


def _protonettype_record_from_value(value: Protonettype) -> ProtonettypeRecord:
    """Build a durable protonettype record from a protonettype value.

    :param value: The protonettype value to store.
    :return: The durable protonettype record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return ProtonettypeRecord(**fields, net=())
    return ProtonettypeRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(value.occupations),
        net=_bond_records(value.net),
    )


def _protonetstructure_from_record(record: ProtonetstructureRecord) -> Protonetstructure:
    """Reconstruct a protonetstructure value from its durable record.

    :param record: The durable protonetstructure record.
    :return: The reconstructed protonetstructure value.
    """
    if record.undetermined is not None:
        return _undetermined_from_record(record, Protonetstructure, "net", _NET_FORM_VERSION)
    _require_canonical_form(record, "net", _NET_FORM_VERSION)
    return Protonetstructure._of(
        _protonet_record_spacegroup(record),
        tuple(WyckoffOccupation(value.wyckoff, _species_from_record(value.species)) for value in record.occupations),
        _net_rows(record),
    )


def _protonetstructure_record_from_value(value: Protonetstructure) -> ProtonetstructureRecord:
    """Build a durable protonetstructure record from a protonetstructure value.

    :param value: The protonetstructure value to store.
    :return: The durable protonetstructure record.
    """
    if (fields := _undetermined_fields(value)) is not None:
        return ProtonetstructureRecord(**fields, net=())
    return ProtonetstructureRecord(
        spacegroup_it_number=value.spacegroup.it_number,
        spacegroup_hall_entry=value.spacegroup.hall_entry,
        occupations=tuple(
            WyckoffOccupationRecord(
                wyckoff=occupation.wyckoff,
                species=SpeciesRecord(**cast(dict[str, Any], SpeciesRecord.__httk_project__(occupation.species))),
            )
            for occupation in value.occupations
        ),
        net=_bond_records(value.net),
    )


from httk.atomistic.storage.stored_properties import attach_structure_property_projections

attach_structure_property_projections(
    UnitcellStructureRecord,
    FundamentalDomainStructureRecord,
    ASUStructureRecord,
    NumericUnitcellStructureRecord,
)

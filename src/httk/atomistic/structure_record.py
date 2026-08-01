"""Frozen storage records for complete atomistic structures."""

import datetime
import fractions
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, cast

import httk.core.storage_markers
from httk.core import FracVector, StorageInfo, StoredEntryProjection, SurdScalar, SurdVector, Unique

from ._composition_values import as_fraction
from ._vector_guards import to_precision
from .composition import Assembly, ChemicalComposition, validate_assemblies
from .setting_transform import SettingTransform
from .species import Species
from .structure_semantics import StructureSymmetry

if TYPE_CHECKING:
    from .asu_structure import FundamentalDomainStructure
    from .structure import Structure
    from .structure_like import StructureLike

type StructureRepresentation = Literal["unit_cell", "fundamental_domain", "asymmetric_unit"]


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
class ExactVectorRowsRecord:
    """A durable flattened exact Cartesian matrix with an N×3 served shape."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_exact_vector_rows")

    values: tuple[SurdScalar, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if len(values) % 3 or not all(isinstance(value, SurdScalar) for value in values):
            raise ValueError("ExactVectorRowsRecord values must contain complete SurdScalar three-vectors")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_vector(cls, vector: SurdVector) -> "ExactVectorRowsRecord":
        if len(vector.dim) != 2 or vector.dim[1] != 3:
            raise ValueError("ExactVectorRowsRecord source must be an N×3 SurdVector")
        return cls(
            tuple(_extract_surd_scalar(vector, (row, column)) for row in range(vector.dim[0]) for column in range(3))
        )

    def to_vector(self) -> SurdVector:
        if not self.values:
            return SurdVector.create(())
        rows = len(self.values) // 3
        # A matrix made entirely from exact zeros has no scalar radicands from
        # which ``from_radicand_map`` can infer its shape.  Retain an explicit
        # rational component so zero Cartesian rows remain N x 3.
        radicands = sorted({radicand for value in self.values for radicand in value.radicands}) or [1]
        components = {
            radicand: [
                [self.values[3 * row + column].coefficient(radicand).to_fraction() for column in range(3)]
                for row in range(rows)
            ]
            for radicand in radicands
        }
        return SurdVector.from_radicand_map(components)

    def to_stored_entry_value(self) -> list[list[float]]:
        if not self.values:
            return []
        return self.to_vector().to_floats()


@dataclass(frozen=True)
class StringTupleRecord:
    """A short-named durable wrapper for a served list of strings."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_strings_v3")

    values: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if not all(isinstance(value, str) for value in values):
            raise TypeError("StringTupleRecord values must be strings")
        object.__setattr__(self, "values", values)

    def to_stored_entry_value(self) -> list[str]:
        return list(self.values)


@dataclass(frozen=True)
class SpeciesRecord:
    """The storable frozen snapshot of an atomistic :class:`~httk.atomistic.Species`.

    The presence flags preserve ``None`` for optional tuple fields because a relational child
    table cannot distinguish a missing optional value from an empty tuple when reconstructing.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_species_record")

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

        for field, values in (
            ("concentration", self.concentration),
            ("mass", self.mass or ()),
        ):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"SpeciesRecord {field} values must be finite floats")

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

    sites: tuple[int, ...]

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        if not sites or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in sites):
            raise ValueError("AssemblyGroupRecord sites must be non-empty non-negative integer indices")
        if len(set(sites)) != len(sites):
            raise ValueError("AssemblyGroupRecord cannot repeat a site index")
        object.__setattr__(self, "sites", sites)


@dataclass(frozen=True)
class AssemblyRecord:
    """Exact durable form of :class:`Assembly`, including per-value precision."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_assembly_record")

    groups: tuple[AssemblyGroupRecord, ...]
    group_probabilities: tuple[fractions.Fraction, ...]
    group_probabilities_precision: tuple[fractions.Fraction, ...] = ()
    group_probabilities_precision_present: bool = False

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
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

    wyckoff: str
    free_parameters: tuple[fractions.Fraction, ...]
    species: str
    representative: tuple[fractions.Fraction, ...] = ()
    representative_present: bool = False

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

    matrix: Annotated[FracVector, httk.core.storage_markers.Shape(3, 3)]
    vector: tuple[fractions.Fraction, ...]
    hall_entry: str | None = None

    def __post_init__(self) -> None:
        matrix = FracVector.create(self.matrix)
        vector = tuple(as_fraction(value, field="SettingTransformRecord vector")[0] for value in self.vector)
        transform = SettingTransform(matrix, vector, hall_entry=self.hall_entry)
        object.__setattr__(self, "matrix", transform.matrix)
        object.__setattr__(self, "vector", tuple(transform.vector.to_fractions()))

    @classmethod
    def from_transform(cls, transform: SettingTransform) -> "SettingTransformRecord":
        return cls(transform.matrix, tuple(transform.vector.to_fractions()), transform.hall_entry)

    def to_transform(self) -> SettingTransform:
        return SettingTransform(self.matrix, self.vector, hall_entry=self.hall_entry)


@dataclass(frozen=True)
class SymmetryRecord:
    """Typed optional symmetry metadata of an ordinary unit-cell structure."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_symmetry_v3")

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

    element: str
    amount: fractions.Fraction
    precision: fractions.Fraction | None = None


@dataclass(frozen=True)
class ChemicalCompositionRecord:
    """Durable authoritative or implicit composition declaration."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_v3_chemical_composition_record")

    amounts: tuple[CompositionAmountRecord, ...]
    mode: str

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
class StructureRecord:
    """A storable snapshot of a complete :class:`~httk.atomistic.Structure`.

    This record is a snapshot, not a live proxy: changing a source ``Structure`` does not
    update it, and changing a record does not update a previously reconstructed structure.
    Rationals, surd bases, precisions, and periodicity remain exact. Species float fields
    round-trip at IEEE-double fidelity, with the SQL layer's documented caveat that ``-0.0``
    may return as ``+0.0``.

    Construction validates the component and cross-component invariants directly without
    building and discarding a domain ``Structure``.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="atomistic_structure_v3")

    basis: tuple[SurdScalar, ...]
    reduced_coords: Annotated[FracVector, httk.core.storage_markers.Shape(0, 3)]
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    periodicity: tuple[bool, ...]
    basis_precision: fractions.Fraction | None
    coordinate_precision: fractions.Fraction | None
    representation: str = "unit_cell"
    molecular: bool = False
    domain_sites: tuple[DomainSiteRecord, ...] = ()
    spacegroup_it_number: int | None = None
    setting_transform: SettingTransformRecord | None = None
    assemblies: tuple[AssemblyRecord, ...] | None = None
    assemblies_present: bool = False
    symmetry: SymmetryRecord | None = None
    chemical_composition: ChemicalCompositionRecord | None = None
    chemical_formula_descriptive: str | None = None
    chemical_formula_hill: str | None = None
    optimization_type: str | None = None

    def __post_init__(self) -> None:
        basis = tuple(self.basis)
        if len(basis) != 9 or not all(isinstance(value, SurdScalar) for value in basis):
            raise ValueError("StructureRecord basis must contain exactly 9 SurdScalar values")
        basis_vector = _basis_vector(basis)
        if basis_vector.dim != (3, 3):
            raise ValueError("StructureRecord basis must be a 3x3 vector-like")
        if not isinstance(self.reduced_coords, FracVector):
            raise TypeError("StructureRecord reduced_coords must be a FracVector")
        if self.reduced_coords.dim != () and not (
            len(self.reduced_coords.dim) == 2 and self.reduced_coords.dim[1] == 3
        ):
            raise ValueError("Sites reduced_coords must be an Nx3 vector-like")
        species = tuple(self.species)
        if not all(isinstance(value, SpeciesRecord) for value in species):
            raise TypeError("StructureRecord species must contain SpeciesRecord values")
        species_at_sites = tuple(self.species_at_sites)
        if not all(isinstance(value, str) for value in species_at_sites):
            raise TypeError("StructureRecord species_at_sites must contain string values")
        periodicity = tuple(self.periodicity)
        if len(periodicity) != 3 or not all(isinstance(value, bool) for value in periodicity):
            raise ValueError("StructureRecord periodicity must contain exactly 3 bool values")
        basis_precision = to_precision(self.basis_precision)
        coordinate_precision = to_precision(self.coordinate_precision)
        from .cell import Cell
        from .sites import Sites

        Cell(basis_vector, precision=basis_precision, periodicity=periodicity)
        sites = Sites(self.reduced_coords, precision=coordinate_precision)
        if len(species_at_sites) != len(sites):
            raise ValueError("Structure species_at_sites must have the same length as sites")
        names = [value.name for value in species]
        if len(names) != len(set(names)):
            raise ValueError("Structure species names must be unique")
        known = set(names)
        for name in species_at_sites:
            if name not in known:
                raise ValueError(f"Structure species_at_sites references unknown species name: {name!r}")
        if self.representation not in {"unit_cell", "fundamental_domain", "asymmetric_unit"}:
            raise ValueError("StructureRecord representation must be unit_cell, fundamental_domain, or asymmetric_unit")
        if not isinstance(self.molecular, bool):
            raise TypeError("StructureRecord molecular must be a bool")
        domain_sites = tuple(self.domain_sites)
        if not all(isinstance(value, DomainSiteRecord) for value in domain_sites):
            raise TypeError("StructureRecord domain_sites must contain DomainSiteRecord values")
        if self.representation == "unit_cell":
            if domain_sites or self.spacegroup_it_number is not None or self.setting_transform is not None:
                raise ValueError("A unit-cell StructureRecord cannot contain fundamental-domain fields")
        else:
            if (
                len(domain_sites) != len(species_at_sites)
                or tuple(value.species for value in domain_sites) != species_at_sites
            ):
                raise ValueError("A domain StructureRecord must align domain_sites with species_at_sites")
            if self.spacegroup_it_number is None or not 1 <= self.spacegroup_it_number <= 230:
                raise ValueError("A domain StructureRecord requires a space-group IT number in [1, 230]")
            if self.setting_transform is None:
                raise ValueError("A domain StructureRecord requires an exact setting transform")
            if self.symmetry is not None:
                raise ValueError(
                    "A domain StructureRecord derives symmetry and cannot contain unit-cell symmetry metadata"
                )
            from .spacegroup import Spacegroup

            group = Spacegroup.standard(self.spacegroup_it_number)
            transform = self.setting_transform.to_transform()
            expected_coordinates: list[list[fractions.Fraction]] = []
            for site in domain_sites:
                if site.representative_present:
                    expected_coordinates.append(list(FracVector.create(site.representative).normalize().to_fractions()))
                else:
                    standard = group.wyckoff_position(site.wyckoff).coordinates(site.free_parameters)[0]
                    own = transform.to_setting(standard) + transform.lattice_cosets()[0]
                    expected_coordinates.append(list(own.normalize().to_fractions()))
            expected_reduced = (
                FracVector.create(expected_coordinates) if expected_coordinates else FracVector.create(())
            )
            if self.reduced_coords != expected_reduced:
                raise ValueError(
                    "A domain StructureRecord reduced_coords disagrees with retained or derived domain representatives"
                )
        if not isinstance(self.assemblies_present, bool):
            raise TypeError("StructureRecord assemblies_present must be a bool")
        assemblies = None if not self.assemblies_present else tuple(self.assemblies or ())
        if not self.assemblies_present and self.assemblies not in (None, ()):
            raise ValueError("StructureRecord assembly values require assemblies_present=True")
        if assemblies is not None:
            if not all(isinstance(value, AssemblyRecord) for value in assemblies):
                raise TypeError("StructureRecord assemblies must contain AssemblyRecord values")
            validate_assemblies((value.to_assembly() for value in assemblies), len(species_at_sites))
        if self.symmetry is not None and not isinstance(self.symmetry, SymmetryRecord):
            raise TypeError("StructureRecord symmetry must be a SymmetryRecord or None")
        if self.chemical_composition is not None and not isinstance(
            self.chemical_composition, ChemicalCompositionRecord
        ):
            raise TypeError("StructureRecord chemical_composition must be a ChemicalCompositionRecord or None")
        for name in ("chemical_formula_descriptive", "chemical_formula_hill", "optimization_type"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"StructureRecord {name} must be a string or None")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "reduced_coords", FracVector.use(self.reduced_coords))
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "species_at_sites", species_at_sites)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "basis_precision", basis_precision)
        object.__setattr__(self, "coordinate_precision", coordinate_precision)
        object.__setattr__(self, "domain_sites", domain_sites)
        object.__setattr__(self, "assemblies", assemblies)

    @classmethod
    def from_structure(cls, structure: "StructureLike") -> "StructureRecord":
        """Take a canonical snapshot of any :data:`~httk.atomistic.StructureLike` input."""
        from httk.core import unwrap

        from .asu_structure import ASUStructure, FundamentalDomainStructure
        from .unitcell_structure_view import UnitcellStructureView

        if isinstance(structure, StructureRecord):
            return structure
        unwrapped = unwrap(structure)
        canonical: Structure | FundamentalDomainStructure
        if isinstance(unwrapped, FundamentalDomainStructure):
            canonical = unwrapped
        else:
            canonical = UnitcellStructureView(structure)
        basis = canonical.cell.basis
        basis_values = tuple(_extract_surd_scalar(basis, (row, column)) for row in range(3) for column in range(3))
        representation: StructureRepresentation = "unit_cell"
        domain_sites: tuple[DomainSiteRecord, ...] = ()
        spacegroup_it_number = None
        setting_transform = None
        symmetry = None
        if isinstance(canonical, FundamentalDomainStructure):
            representation = "asymmetric_unit" if isinstance(canonical, ASUStructure) else "fundamental_domain"
            domain_sites = tuple(
                DomainSiteRecord(
                    site.wyckoff,
                    tuple(site.free_params.to_fractions()),
                    site.species,
                    () if site.representative is None else tuple(site.representative.to_fractions()),
                    site.representative is not None,
                )
                for site in canonical.domain_sites
            )
            spacegroup_it_number = canonical.spacegroup.it_number
            setting_transform = SettingTransformRecord.from_transform(canonical.transform)
        else:
            symmetry_value = canonical.symmetry
            symmetry = None if symmetry_value is None else SymmetryRecord.from_symmetry(symmetry_value)
        return cls(
            basis=basis_values,
            reduced_coords=canonical.sites.reduced_coords,
            species=tuple(SpeciesRecord.from_species(value) for value in canonical.species),
            species_at_sites=canonical.species_at_sites,
            periodicity=canonical.periodicity,
            basis_precision=canonical.cell.precision,
            coordinate_precision=canonical.coordinate_precision,
            representation=representation,
            molecular=canonical.molecular,
            domain_sites=domain_sites,
            spacegroup_it_number=spacegroup_it_number,
            setting_transform=setting_transform,
            assemblies=(
                None
                if canonical.assemblies is None
                else tuple(AssemblyRecord.from_assembly(value) for value in canonical.assemblies)
            ),
            assemblies_present=canonical.assemblies is not None,
            symmetry=symmetry,
            chemical_composition=(
                None
                if canonical.chemical_composition is None
                else ChemicalCompositionRecord.from_composition(canonical.chemical_composition)
            ),
            chemical_formula_descriptive=canonical.chemical_formula_descriptive,
            chemical_formula_hill=canonical.chemical_formula_hill,
            optimization_type=canonical.optimization_type,
        )

    def to_structure(self) -> "Structure | FundamentalDomainStructure":
        """Rebuild the exact native representation selected by :attr:`representation`."""
        from .asu_structure import ASUSite, ASUStructure, FundamentalDomainStructure
        from .cell import Cell
        from .sites import Sites
        from .spacegroup import Spacegroup
        from .structure import Structure

        basis = _basis_vector(self.basis)
        cell = Cell(basis, precision=self.basis_precision, periodicity=self.periodicity)
        species = tuple(value.to_species() for value in self.species)
        assemblies = None if self.assemblies is None else tuple(value.to_assembly() for value in self.assemblies)
        composition = None if self.chemical_composition is None else self.chemical_composition.to_composition()
        if self.representation != "unit_cell":
            assert self.spacegroup_it_number is not None and self.setting_transform is not None
            sites = tuple(
                ASUSite(
                    value.wyckoff,
                    FracVector.create(value.free_parameters),
                    value.species,
                    FracVector.create(value.representative) if value.representative_present else None,
                )
                for value in self.domain_sites
            )
            cls = ASUStructure if self.representation == "asymmetric_unit" else FundamentalDomainStructure
            return cls(
                cell,
                Spacegroup.standard(self.spacegroup_it_number),
                sites,
                species,
                self.setting_transform.to_transform(),
                self.coordinate_precision,
                molecular=self.molecular,
                assemblies=assemblies,
                chemical_composition=composition,
                chemical_formula_descriptive=self.chemical_formula_descriptive,
                chemical_formula_hill=self.chemical_formula_hill,
                optimization_type=self.optimization_type,
            )
        return Structure(
            cell=cell,
            sites=Sites(self.reduced_coords, precision=self.coordinate_precision),
            species=species,
            species_at_sites=self.species_at_sites,
            molecular=self.molecular,
            assemblies=assemblies,
            symmetry=None if self.symmetry is None else self.symmetry.to_symmetry(),
            chemical_composition=composition,
            chemical_formula_descriptive=self.chemical_formula_descriptive,
            chemical_formula_hill=self.chemical_formula_hill,
            optimization_type=self.optimization_type,
        )


_STRUCTURE_PROPERTY_FIELDS = {
    name: name
    for name in (
        "id",
        "immutable_id",
        "last_modified",
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "chemical_formula_hill",
        "chemical_formula_anonymous",
        "dimension_types",
        "nperiodic_dimensions",
        "lattice_vectors",
        "space_group_symmetry_operations_xyz",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_it_number",
        "cartesian_site_positions",
        "fractional_site_positions",
        "site_coordinate_span",
        "site_coordinate_span_description",
        "nsites",
        "species_at_sites",
        "species",
        "assemblies",
        "wyckoff_positions",
        "structure_features",
        "optimization_type",
    )
}
_STRUCTURE_PROPERTY_FIELDS["space_group_symmetry_operations_xyz"] = "symmetry_operations"


@dataclass(frozen=True)
class StructureEntryRecord:
    """Durable OPTIMADE entry metadata plus an exact tagged structure snapshot.

    The denormalized fields are an intentional standard query/serving projection.
    ``structure`` retains the authoritative exact native snapshot; identity and
    timestamps live only here and therefore never affect :class:`StructureRecord`
    equality.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_structure_entry_v3",
        indexes=(
            ("id",),
            ("immutable_id",),
            ("last_modified",),
            ("chemical_formula_reduced",),
            ("chemical_formula_hill",),
            ("nsites",),
            ("site_coordinate_span",),
            ("space_group_it_number",),
            ("optimization_type",),
        ),
        dedup="none",
    )
    __httk_entry_projection__: ClassVar[StoredEntryProjection] = StoredEntryProjection(
        entry_type="structures",
        definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
        property_fields=_STRUCTURE_PROPERTY_FIELDS,
        filterable=frozenset(
            {
                "id",
                "immutable_id",
                "last_modified",
                "elements",
                "nelements",
                "elements_ratios",
                "chemical_formula_descriptive",
                "chemical_formula_reduced",
                "chemical_formula_hill",
                "chemical_formula_anonymous",
                "dimension_types",
                "nperiodic_dimensions",
                "space_group_symbol_hall",
                "space_group_symbol_hermann_mauguin",
                "space_group_symbol_hermann_mauguin_extended",
                "space_group_it_number",
                "site_coordinate_span",
                "nsites",
                "structure_features",
                "optimization_type",
            }
        ),
        obsolete_storage_names=("structure_record",),
    )

    id: Annotated[str, Unique()]
    structure: StructureRecord
    immutable_id: str | None
    last_modified: datetime.datetime | None
    elements: tuple[str, ...]
    nelements: int
    elements_ratios: tuple[fractions.Fraction, ...]
    chemical_formula_descriptive: str | None
    chemical_formula_reduced: str | None
    chemical_formula_hill: str | None
    chemical_formula_anonymous: str | None
    dimension_types: tuple[int, ...]
    nperiodic_dimensions: int
    lattice_vectors: ExactVectorRowsRecord
    symmetry_operations: StringTupleRecord | None
    space_group_symbol_hall: str | None
    space_group_symbol_hermann_mauguin: str | None
    space_group_symbol_hermann_mauguin_extended: str | None
    space_group_it_number: int | None
    cartesian_site_positions: ExactVectorRowsRecord
    fractional_site_positions: Annotated[FracVector, httk.core.storage_markers.Shape(0, 3)]
    site_coordinate_span: str
    site_coordinate_span_description: str | None
    nsites: int
    species_at_sites: tuple[str, ...]
    species: tuple[SpeciesRecord, ...]
    assemblies: tuple[AssemblyRecord, ...] | None
    assemblies_present: bool
    wyckoff_positions: tuple[str, ...] | None
    wyckoff_positions_present: bool
    structure_features: tuple[str, ...]
    optimization_type: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("StructureEntryRecord id must be a non-empty string")
        if not isinstance(self.structure, StructureRecord):
            raise TypeError("StructureEntryRecord structure must be a StructureRecord")
        if self.immutable_id is not None and not isinstance(self.immutable_id, str):
            raise TypeError("StructureEntryRecord immutable_id must be a string or None")
        if self.last_modified is not None and (
            not isinstance(self.last_modified, datetime.datetime)
            or self.last_modified.tzinfo is None
            or self.last_modified.utcoffset() is None
        ):
            raise ValueError("StructureEntryRecord last_modified must be a timezone-aware datetime or None")
        elements = tuple(self.elements)
        if not all(isinstance(value, str) for value in elements):
            raise TypeError("StructureEntryRecord elements must contain strings")
        ratios = tuple(
            as_fraction(value, field="StructureEntryRecord elements ratio")[0] for value in self.elements_ratios
        )
        dimension_types = tuple(self.dimension_types)
        if len(dimension_types) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1) for value in dimension_types
        ):
            raise ValueError("StructureEntryRecord dimension_types must contain exactly three 0/1 integers")
        for name in ("nelements", "nperiodic_dimensions", "nsites"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"StructureEntryRecord {name} must be a non-negative integer")
        if sum(dimension_types) != self.nperiodic_dimensions:
            raise ValueError("StructureEntryRecord dimension_types must agree with nperiodic_dimensions")
        for name in (
            "chemical_formula_descriptive",
            "chemical_formula_reduced",
            "chemical_formula_hill",
            "chemical_formula_anonymous",
            "space_group_symbol_hall",
            "space_group_symbol_hermann_mauguin",
            "space_group_symbol_hermann_mauguin_extended",
            "site_coordinate_span_description",
            "optimization_type",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"StructureEntryRecord {name} must be a string or None")
        if not isinstance(self.site_coordinate_span, str):
            raise TypeError("StructureEntryRecord site_coordinate_span must be a string")
        if self.space_group_it_number is not None and (
            not isinstance(self.space_group_it_number, int) or isinstance(self.space_group_it_number, bool)
        ):
            raise TypeError("StructureEntryRecord space_group_it_number must be an integer or None")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "elements_ratios", ratios)
        object.__setattr__(self, "dimension_types", dimension_types)
        if not isinstance(self.lattice_vectors, ExactVectorRowsRecord) or len(self.lattice_vectors.values) != 9:
            raise ValueError("StructureEntryRecord lattice_vectors must be an exact 3x3 record")
        if not isinstance(self.cartesian_site_positions, ExactVectorRowsRecord):
            raise TypeError("StructureEntryRecord cartesian_site_positions must be an exact rows record")
        object.__setattr__(self, "fractional_site_positions", FracVector.use(self.fractional_site_positions))
        species_at_sites = tuple(self.species_at_sites)
        species = tuple(self.species)
        if not all(isinstance(value, str) for value in species_at_sites):
            raise TypeError("StructureEntryRecord species_at_sites must contain strings")
        if not all(isinstance(value, SpeciesRecord) for value in species):
            raise TypeError("StructureEntryRecord species must contain SpeciesRecord values")
        object.__setattr__(self, "species_at_sites", species_at_sites)
        object.__setattr__(self, "species", species)
        if self.symmetry_operations is not None and not isinstance(self.symmetry_operations, StringTupleRecord):
            raise TypeError("StructureEntryRecord symmetry_operations must be a StringTupleRecord or None")
        if not isinstance(self.assemblies_present, bool) or not isinstance(self.wyckoff_positions_present, bool):
            raise TypeError("StructureEntryRecord presence flags must be bool values")
        if not self.assemblies_present and self.assemblies not in (None, ()):
            raise ValueError("StructureEntryRecord assembly values require assemblies_present=True")
        if not self.wyckoff_positions_present and self.wyckoff_positions not in (None, ()):
            raise ValueError("StructureEntryRecord Wyckoff values require wyckoff_positions_present=True")
        assemblies = tuple(self.assemblies or ()) if self.assemblies_present else None
        positions = tuple(self.wyckoff_positions or ()) if self.wyckoff_positions_present else None
        if assemblies is not None and not all(isinstance(value, AssemblyRecord) for value in assemblies):
            raise TypeError("StructureEntryRecord assemblies must contain AssemblyRecord values")
        if positions is not None and not all(isinstance(value, str) for value in positions):
            raise TypeError("StructureEntryRecord wyckoff_positions must contain strings")
        object.__setattr__(self, "assemblies", assemblies)
        object.__setattr__(self, "wyckoff_positions", positions)
        features = tuple(self.structure_features)
        if not all(isinstance(value, str) for value in features):
            raise TypeError("StructureEntryRecord structure_features must contain strings")
        object.__setattr__(self, "structure_features", features)
        if len(self.cartesian_site_positions.values) != 3 * self.nsites:
            raise ValueError("StructureEntryRecord flattened lattice/Cartesian vectors have inconsistent lengths")
        if self.fractional_site_positions.dim not in ((), (0,)) and self.fractional_site_positions.dim != (
            self.nsites,
            3,
        ):
            raise ValueError("StructureEntryRecord fractional positions must agree with nsites")
        if len(self.species_at_sites) != self.nsites:
            raise ValueError("StructureEntryRecord species_at_sites must agree with nsites")
        native = cast(Any, self.structure.to_structure())
        expected_assemblies = (
            None
            if native.assemblies is None
            else tuple(AssemblyRecord.from_assembly(value) for value in native.assemblies)
        )
        expected_species = tuple(SpeciesRecord.from_species(value) for value in native.species)
        expected: dict[str, object] = {
            "elements": tuple(native.elements),
            "nelements": native.nelements,
            "elements_ratios": tuple(native.elements_ratios),
            "chemical_formula_descriptive": native.chemical_formula_descriptive,
            "chemical_formula_reduced": native.chemical_formula_reduced,
            "chemical_formula_hill": native.chemical_formula_hill,
            "chemical_formula_anonymous": native.chemical_formula_anonymous,
            "dimension_types": tuple(native.dimension_types),
            "nperiodic_dimensions": native.nperiodic_dimensions,
            "lattice_vectors": ExactVectorRowsRecord.from_vector(native.cell.basis),
            "space_group_symmetry_operations_xyz": native.space_group_symmetry_operations_xyz,
            "space_group_symbol_hall": native.space_group_symbol_hall,
            "space_group_symbol_hermann_mauguin": native.space_group_symbol_hermann_mauguin,
            "space_group_symbol_hermann_mauguin_extended": native.space_group_symbol_hermann_mauguin_extended,
            "space_group_it_number": native.space_group_it_number,
            "cartesian_site_positions": (
                ExactVectorRowsRecord(())
                if native.nsites == 0
                else ExactVectorRowsRecord.from_vector(native.cartesian_sites())
            ),
            "fractional_site_positions": native.sites.reduced_coords,
            "site_coordinate_span": native.site_coordinate_span,
            "site_coordinate_span_description": native.site_coordinate_span_description,
            "nsites": native.nsites,
            "species_at_sites": tuple(native.species_at_sites),
            "species": expected_species,
            "assemblies": expected_assemblies,
            "wyckoff_positions": native.wyckoff_positions,
            "structure_features": tuple(native.structure_features),
            "optimization_type": native.optimization_type,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError(
                "StructureEntryRecord denormalized fields disagree with its exact structure snapshot: "
                + ", ".join(mismatches)
            )

    @property
    def space_group_symmetry_operations_xyz(self) -> tuple[str, ...] | None:
        """The standard property value represented by the short storage wrapper."""
        return None if self.symmetry_operations is None else self.symmetry_operations.values

    @classmethod
    def from_structure(
        cls,
        structure: "StructureLike",
        *,
        id: str,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
    ) -> "StructureEntryRecord":
        snapshot = StructureRecord.from_structure(structure)
        native = snapshot.to_structure()
        basis = native.cell.basis
        cartesian = native.cartesian_sites()
        nsites = native.nsites
        return cls(
            id=id,
            structure=snapshot,
            immutable_id=immutable_id,
            last_modified=last_modified,
            elements=tuple(native.elements),
            nelements=native.nelements,
            elements_ratios=tuple(native.elements_ratios),
            chemical_formula_descriptive=native.chemical_formula_descriptive,
            chemical_formula_reduced=native.chemical_formula_reduced,
            chemical_formula_hill=native.chemical_formula_hill,
            chemical_formula_anonymous=native.chemical_formula_anonymous,
            dimension_types=tuple(native.dimension_types),
            nperiodic_dimensions=native.nperiodic_dimensions,
            lattice_vectors=ExactVectorRowsRecord.from_vector(basis),
            symmetry_operations=(
                None
                if native.space_group_symmetry_operations_xyz is None
                else StringTupleRecord(tuple(native.space_group_symmetry_operations_xyz))
            ),
            space_group_symbol_hall=native.space_group_symbol_hall,
            space_group_symbol_hermann_mauguin=native.space_group_symbol_hermann_mauguin,
            space_group_symbol_hermann_mauguin_extended=native.space_group_symbol_hermann_mauguin_extended,
            space_group_it_number=native.space_group_it_number,
            cartesian_site_positions=(
                ExactVectorRowsRecord(()) if nsites == 0 else ExactVectorRowsRecord.from_vector(cartesian)
            ),
            fractional_site_positions=native.sites.reduced_coords,
            site_coordinate_span=native.site_coordinate_span,
            site_coordinate_span_description=native.site_coordinate_span_description,
            nsites=nsites,
            species_at_sites=tuple(native.species_at_sites),
            species=tuple(SpeciesRecord.from_species(value) for value in native.species),
            assemblies=(
                None
                if native.assemblies is None
                else tuple(AssemblyRecord.from_assembly(value) for value in native.assemblies)
            ),
            assemblies_present=native.assemblies is not None,
            wyckoff_positions=native.wyckoff_positions,
            wyckoff_positions_present=native.wyckoff_positions is not None,
            structure_features=tuple(native.structure_features),
            optimization_type=native.optimization_type,
        )

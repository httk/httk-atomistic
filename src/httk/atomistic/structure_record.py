"""Frozen storage records for complete atomistic structures."""

import fractions
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, cast

import httk.core.storage_markers
from httk.core import FracVector, SurdScalar, SurdVector

from ._vector_guards import to_precision
from .species import Species

if TYPE_CHECKING:
    from .structure import Structure
    from .structure_like import StructureLike


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

    name: str
    chemical_symbols: tuple[str, ...]
    concentration: tuple[float, ...]
    mass: tuple[float, ...] | None = None
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None
    mass_present: bool = False
    attached_present: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "chemical_symbols", tuple(self.chemical_symbols))
        object.__setattr__(self, "concentration", tuple(self.concentration))
        if self.mass is not None:
            object.__setattr__(self, "mass", tuple(self.mass))
        if self.attached is not None:
            object.__setattr__(self, "attached", tuple(self.attached))
        if self.nattached is not None:
            object.__setattr__(self, "nattached", tuple(self.nattached))

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

    basis: tuple[SurdScalar, ...]
    reduced_coords: Annotated[FracVector, httk.core.storage_markers.Shape(0, 3)]
    species: tuple[SpeciesRecord, ...]
    species_at_sites: tuple[str, ...]
    periodicity: tuple[bool, ...]
    basis_precision: fractions.Fraction | None
    coordinate_precision: fractions.Fraction | None

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
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "reduced_coords", FracVector.use(self.reduced_coords))
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "species_at_sites", species_at_sites)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "basis_precision", basis_precision)
        object.__setattr__(self, "coordinate_precision", coordinate_precision)

    @classmethod
    def from_structure(cls, structure: "StructureLike") -> "StructureRecord":
        """Take a canonical snapshot of any :data:`~httk.atomistic.StructureLike` input."""
        from .unitcell_structure_view import UnitcellStructureView

        canonical = UnitcellStructureView(structure)
        basis = canonical.cell.basis
        basis_values = tuple(_extract_surd_scalar(basis, (row, column)) for row in range(3) for column in range(3))
        return cls(
            basis=basis_values,
            reduced_coords=canonical.sites.reduced_coords,
            species=tuple(SpeciesRecord.from_species(value) for value in canonical.species),
            species_at_sites=canonical.species_at_sites,
            periodicity=canonical.periodicity,
            basis_precision=canonical.basis_precision,
            coordinate_precision=canonical.coordinate_precision,
        )

    def to_structure(self) -> "Structure":
        """Rebuild the exact :class:`~httk.atomistic.Structure` represented by this snapshot."""
        from .cell import Cell
        from .sites import Sites
        from .structure import Structure

        basis = _basis_vector(self.basis)
        return Structure(
            cell=Cell(basis, precision=self.basis_precision, periodicity=self.periodicity),
            sites=Sites(self.reduced_coords, precision=self.coordinate_precision),
            species=tuple(value.to_species() for value in self.species),
            species_at_sites=self.species_at_sites,
        )

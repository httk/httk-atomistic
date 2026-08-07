"""A structure represented by a CIF site's own complete symmetry operations."""

import datetime
import fractions
from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar

from httk.core import FracVector, SurdVector

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.crystalaxis_view import CrystalAxisSiteMomentsView
from httk.atomistic.models.moments.like import SiteMomentsLike
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import StructureSemanticsMixin, initialize_semantics
from httk.atomistic.models.structure.unitcell import (
    _check_site_moments,
    _check_sites_length,
    _check_species_at_sites,
    _check_species_names,
    _norm_cell,
    _norm_site_moments,
    _norm_sites,
    _norm_species,
    _norm_species_at_sites,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.xyz import operation_from_xyz, operation_from_xyzt

__all__ = ["SymopsStructure"]


def _normalize_symop(value: str | AffineOperation | tuple[AffineOperation, int]) -> tuple[AffineOperation, int]:
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) == 4:
            return operation_from_xyzt(value)
        if len(parts) == 3:
            return operation_from_xyz(value), 1
        raise ValueError(f"symmetry operation must have three or four comma-separated parts: {value!r}")
    if isinstance(value, AffineOperation):
        return value, 1
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], AffineOperation):
        return value
    raise TypeError("symops must contain strings, AffineOperation objects, or (AffineOperation, time) pairs")


def _moment_rows(value: SiteMomentsBackend, cell: Cell) -> tuple[tuple[Any, ...], ...] | tuple[Any, ...]:
    if isinstance(value, CollinearSiteMoments):
        return value.collinear_moments.to_fractions()
    if isinstance(value, CrystalAxisSiteMoments):
        crystalaxis = value.crystalaxis_moments
    elif isinstance(value, CartesianSiteMoments):
        crystalaxis = CrystalAxisSiteMomentsView(value, cell=cell).crystalaxis_moments
    else:
        raise TypeError(f"unsupported SiteMomentsBackend kind: {getattr(value, 'kind', None)!r}")
    lengths = cell.lengths
    return tuple(
        tuple((crystalaxis._element((row, column)) / lengths[column])._as_scalar() for column in range(3))
        for row in range(len(value))
    )


def _transform_lattice_moment(row: tuple[Any, ...], operation: AffineOperation, time_reversal: int) -> tuple[Any, ...]:
    matrix = operation.matrix.to_fractions()
    factor = time_reversal * int(operation.determinant())
    return tuple(
        factor * sum((matrix[out][column] * row[column] for column in range(3)), start=SurdVector.zero()._as_scalar())
        for out in range(3)
    )


class SymopsStructure(StructureBackend, StructureSemanticsMixin):
    """Represent a magCIF cell, listed sites, and its complete symmetry-operation list.

    The operations are taken as declared. They are not checked for group closure: magCIF
    lists complete coset representatives, and an incomplete list consequently under-expands.
    Expansion is exact and deduplicates only by normalized fractional coordinates, species,
    and exact transformed moments.

    :param cell: The cell geometry.
    :param sites: The listed site coordinates.
    :param species: The distinct species definitions.
    :param species_at_sites: The species name occupying each listed site.
    :param symops: The declared spatial or magnetic symmetry operations.
    :param site_moments: Optional moments aligned with the listed sites.
    :param bns_number: Optional Belov–Neronova–Smirnova number.
    :param bns_label: Optional Belov–Neronova–Smirnova label.
    :param chemical_composition: Optional chemical composition metadata.
    :param chemical_formula_descriptive: Optional descriptive chemical formula.
    :param chemical_formula_hill: Optional Hill chemical formula.
    :param optimization_type: Optional optimization provenance.
    :param immutable_id: Optional immutable source identifier.
    :param last_modified: Optional source modification timestamp.
    :param charge: An explicitly assigned charge for the cell content.
    :raises TypeError: If an input component or magnetic label has the wrong kind.
    :raises ValueError: If component lengths, operations, or semantic values are invalid.
    """

    kind: ClassVar[str] = "symops"

    def __init__(
        self,
        cell: CellLike,
        sites: SitesLike,
        species: Sequence[SpeciesLike],
        species_at_sites: Sequence[str],
        symops: Sequence[str | AffineOperation | tuple[AffineOperation, int]],
        *,
        site_moments: SiteMomentsLike | None = None,
        bns_number: str | None = None,
        bns_label: str | None = None,
        chemical_composition: Any = None,
        chemical_formula_descriptive: str | None = None,
        chemical_formula_hill: str | None = None,
        optimization_type: str | None = None,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
        charge: fractions.Fraction | int | str | None = None,
    ) -> None:
        norm_cell = _norm_cell(cell)
        norm_sites = _norm_sites(sites)
        norm_species = _norm_species(species)
        norm_species_at_sites = _norm_species_at_sites(species_at_sites)
        _check_sites_length(norm_sites, norm_species_at_sites)
        _check_species_names(norm_species)
        _check_species_at_sites(norm_species_at_sites, norm_species)
        norm_site_moments = _norm_site_moments(site_moments)
        _check_site_moments(norm_site_moments, norm_sites, norm_cell)

        normalized = tuple(_normalize_symop(value) for value in symops)
        if not normalized:
            raise ValueError("SymopsStructure requires at least one symmetry operation")
        for operation, time_reversal in normalized:
            if time_reversal not in (-1, 1):
                raise ValueError("symmetry-operation time reversal must be +1 or -1")
            if operation.matrix.dim != (3, 3) or operation.determinant() not in (-1, 1):
                raise ValueError("symmetry-operation rotation must be 3x3 with determinant +1 or -1")

        if bns_number is not None and not isinstance(bns_number, str):
            raise TypeError("bns_number must be a string or None")
        if bns_label is not None and not isinstance(bns_label, str):
            raise TypeError("bns_label must be a string or None")
        self._cell = norm_cell
        self._listed_sites = norm_sites
        self._species = norm_species
        self._listed_species_at_sites = norm_species_at_sites
        self._listed_site_moments = norm_site_moments
        self._charge = None if charge is None else fractions.Fraction(charge)
        self._symops = normalized
        self._bns_number = bns_number
        self._bns_label = bns_label
        initialize_semantics(
            self,
            nsites=len(norm_sites),
            molecular=False,
            assemblies=None,
            symmetry=None,
            chemical_composition=chemical_composition,
            chemical_formula_descriptive=chemical_formula_descriptive,
            chemical_formula_hill=chemical_formula_hill,
            optimization_type=optimization_type,
            immutable_id=immutable_id,
            last_modified=last_modified,
        )

    @property
    def cell(self) -> Cell:
        """Expose the cell geometry.

        :return: The exact cell.
        """
        return self._cell

    @property
    def listed_sites(self) -> Sites:
        """Expose the sites before symmetry expansion.

        :return: The listed site coordinates.
        """
        return self._listed_sites

    @property
    def listed_species_at_sites(self) -> tuple[str, ...]:
        """Expose species names before symmetry expansion.

        :return: Listed site species names in input order.
        """
        return self._listed_species_at_sites

    @property
    def listed_site_moments(self) -> SiteMomentsBackend | None:
        """Expose moments before symmetry expansion.

        :return: Listed site moments, or ``None`` when unstated.
        """
        return self._listed_site_moments

    @property
    def symops(self) -> tuple[tuple[AffineOperation, int], ...]:
        """Expose normalized spatial and time-reversal operations.

        :return: Operation and time-reversal pairs in declaration order.
        """
        return self._symops

    @property
    def bns_number(self) -> str | None:
        """Expose the Belov–Neronova–Smirnova number.

        :return: The BNS number, or ``None`` when unstated.
        """
        return self._bns_number

    @property
    def bns_label(self) -> str | None:
        """Expose the Belov–Neronova–Smirnova label.

        :return: The BNS label, or ``None`` when unstated.
        """
        return self._bns_label

    @property
    def species(self) -> tuple[Species, ...]:
        """Expose the distinct species.

        :return: Species referenced by the listed sites.
        """
        return self._species

    @property
    def coordinate_precision(self) -> Any:
        """Expose the listed-coordinate precision.

        :return: The fractional coordinate precision.
        """
        return self._listed_sites.precision

    @property
    def basis_precision(self) -> Any:
        """Expose the cell-basis precision.

        :return: The basis precision.
        """
        return self._cell.precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Expose the cell periodicity flags.

        :return: Periodicity flags for the three cell directions.
        """
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """Expose the number of periodic directions.

        :return: The number of periodic directions.
        """
        return self._cell.nperiodic_dimensions

    @property
    def molecular(self) -> bool:
        """Expose whether the structure is molecular.

        :return: Always ``False`` for this backend.
        """
        return False

    @property
    def site_coordinate_span(self) -> str:
        """Expose the coordinate span of the listed sites.

        :return: ``"unit_cell"``.
        """
        return "unit_cell"

    @property
    def charge(self) -> fractions.Fraction | None:
        """Expose the explicitly assigned charge.

        :return: The charge, or ``None`` when unstated.
        """
        return self._charge

    def cartesian_sites(self) -> SurdVector:
        """Compute exact Cartesian positions for the expanded sites.

        :return: Cartesian positions in the exact surd representation.
        """
        return SurdVector.create(self.sites.reduced_coords) * self._cell.basis

    @cached_property
    def _expansion(self) -> tuple[Sites, tuple[str, ...], SiteMomentsBackend | None]:
        source_moments = self._listed_site_moments
        source_rows = None if source_moments is None else _moment_rows(source_moments, self._cell)
        generated: list[tuple[tuple[Any, ...], str, tuple[Any, ...] | Any]] = []
        seen: dict[tuple[Any, ...], tuple[str, tuple[Any, ...] | Any]] = {}

        for site_index, (site, species) in enumerate(zip(self._listed_sites, self._listed_species_at_sites)):
            block: list[tuple[tuple[Any, ...], str, tuple[Any, ...] | Any]] = []
            for operation_index, (operation, time_reversal) in enumerate(self._symops):
                point = operation.apply_wrapped(site)
                position_key = tuple(point.to_fractions())
                if source_rows is None:
                    moment = None
                elif isinstance(source_moments, CollinearSiteMoments):
                    moment = time_reversal * int(operation.determinant()) * source_rows[site_index]
                else:
                    moment = _transform_lattice_moment(source_rows[site_index], operation, time_reversal)
                previous = seen.get(position_key)
                if previous is not None:
                    if previous != (species, moment):
                        raise ValueError(
                            "internally inconsistent structure: operation "
                            f"{operation_index} maps site {site_index} onto an already-generated site "
                            "with a different species/moment"
                        )
                    continue
                seen[position_key] = (species, moment)
                block.append((position_key, species, moment))
            generated.extend(sorted(block, key=lambda item: item[0]))

        coordinates = FracVector.create([list(item[0]) for item in generated])
        sites = Sites(coordinates, self._listed_sites.precision)
        species_at_sites = tuple(item[1] for item in generated)
        moments: SiteMomentsBackend | None
        if source_moments is None:
            moments = None
        elif isinstance(source_moments, CollinearSiteMoments):
            moments = CollinearSiteMoments([item[2] for item in generated], precision=source_moments.precision)
        else:
            crystalaxis_rows = [
                [(item[2][column] * self._cell.lengths[column])._as_scalar() for column in range(3)]
                for item in generated
            ]
            crystalaxis = CrystalAxisSiteMoments(
                SurdVector._from_scalar_grid(crystalaxis_rows, (len(generated), 3)),
                self._cell,
                precision=source_moments.precision,
            )
            moments = (
                crystalaxis
                if isinstance(source_moments, CrystalAxisSiteMoments)
                else CartesianSiteMomentsView(crystalaxis)
            )
        return sites, species_at_sites, moments

    @property
    def sites(self) -> Sites:
        """Expose the symmetry-expanded sites.

        :return: Expanded site coordinates.
        """
        return self._expansion[0]

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose species names for the expanded sites.

        :return: Expanded site species names in site order.
        """
        return self._expansion[1]

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Expose moments transformed onto the expanded sites.

        :return: Expanded site moments, or ``None`` when moments were unstated.
        """
        return self._expansion[2]

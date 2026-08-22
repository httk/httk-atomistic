"""A structure represented by a CIF site's own complete symmetry operations."""

import datetime
import fractions
import logging
import re
from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar

from httk.core import FracVector, SurdScalar, SurdVector

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

logger = logging.getLogger(__name__)

type _MomentClaims = tuple[tuple[fractions.Fraction | None, ...], ...]

_MOMENT_FORM_TERM = re.compile(r"^(?P<sign>[+-]?)(?P<coefficient>\d+(?:/\d+)?)?(?P<variable>m[xyz])$", re.IGNORECASE)


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


def _normalize_moment_claims(
    values: Sequence[Sequence[Any | None]] | None,
    count: int,
    *,
    name: str,
) -> _MomentClaims | None:
    if values is None:
        return None
    if len(values) != count:
        raise ValueError(f"SymopsStructure {name} must have the same length as sites")
    rows: list[tuple[fractions.Fraction | None, ...]] = []
    for row in values:
        if len(row) != 3:
            raise ValueError(f"SymopsStructure {name} rows must have three components")
        normalized: list[fractions.Fraction | None] = []
        for value in row:
            if value is None:
                normalized.append(None)
                continue
            claim = fractions.Fraction(value)
            if claim < 0:
                raise ValueError(f"SymopsStructure {name} entries must be non-negative or None")
            normalized.append(claim if claim else None)
        rows.append(tuple(normalized))
    return tuple(rows)


def _normalize_moment_symmforms(values: Sequence[str | None] | None, count: int) -> tuple[str | None, ...] | None:
    if values is None:
        return None
    if len(values) != count:
        raise ValueError("SymopsStructure moment_symmforms must have the same length as sites")
    return tuple(None if value is None else str(value) for value in values)


def _rational_null_space(
    rows: Sequence[tuple[fractions.Fraction, ...]], width: int
) -> tuple[tuple[fractions.Fraction, ...], ...]:
    """Return an exact row basis for the null space of a small rational matrix."""
    matrix = [list(row) for row in rows if any(row)]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        divisor = matrix[pivot_row][column]
        matrix[pivot_row] = [value / divisor for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [left - factor * right for left, right in zip(matrix[row], matrix[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    free_columns = tuple(column for column in range(width) if column not in pivot_columns)
    basis: list[tuple[fractions.Fraction, ...]] = []
    for free_column in free_columns:
        vector = [fractions.Fraction(0)] * width
        vector[free_column] = fractions.Fraction(1)
        for row, column in enumerate(pivot_columns):
            vector[column] = -matrix[row][free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def _solve_surd_system(matrix: list[list[SurdScalar]], target: list[SurdScalar]) -> tuple[SurdScalar, ...]:
    """Solve one positive-definite system over the exact surd field."""
    size = len(target)
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if not augmented[row][column].is_zero()), None)
        if pivot is None:
            raise ValueError("magnetic stabilizer projection produced a singular normal matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [(value / divisor)._as_scalar() for value in augmented[column]]
        for row in range(size):
            if row == column or augmented[row][column].is_zero():
                continue
            factor = augmented[row][column]
            augmented[row] = [
                (left - factor * right)._as_scalar() for left, right in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[index][-1] for index in range(size))


def _claim_widths(
    resolutions: tuple[fractions.Fraction | None, ...],
    esds: tuple[fractions.Fraction | None, ...],
) -> tuple[fractions.Fraction | None, ...]:
    """Combine half-last-digit intervals and explicit standard uncertainties componentwise."""
    widths: list[fractions.Fraction | None] = []
    for resolution, esd in zip(resolutions, esds):
        candidates = tuple(
            value for value in (None if resolution is None else resolution / 2, esd) if value is not None
        )
        widths.append(max(candidates) if candidates else None)
    known = tuple(value for value in widths if value is not None)
    if not known:
        return tuple(widths)
    fallback = max(known)
    return tuple(fallback if value is None else value for value in widths)


def _absolute(value: SurdScalar) -> SurdScalar:
    return (-value)._as_scalar() if value.sign() < 0 else value


def _moment_symmform_basis(value: str | None) -> tuple[tuple[fractions.Fraction, ...], ...] | None:
    """Parse the linear mCIF symmetry-form subset; return ``None`` for descriptive junk."""
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 3:
        return None
    columns: dict[str, list[fractions.Fraction]] = {}
    for component, expression in enumerate(parts):
        try:
            if fractions.Fraction(expression) == 0:
                continue
        except (ValueError, ZeroDivisionError):
            pass
        match = _MOMENT_FORM_TERM.fullmatch(expression)
        if match is None:
            return None
        coefficient = fractions.Fraction(match.group("coefficient") or 1)
        if match.group("sign") == "-":
            coefficient = -coefficient
        variable = match.group("variable").lower()
        columns.setdefault(variable, [fractions.Fraction(0)] * 3)[component] = coefficient
    return tuple(tuple(columns[variable]) for variable in sorted(columns))


def _weighted_projection(
    mapping: SurdVector,
    source: tuple[SurdScalar, ...],
    widths: tuple[fractions.Fraction | None, ...],
) -> tuple[SurdScalar, ...]:
    """Project one moment onto an invariant basis using its componentwise source claims."""
    parameter_count = mapping.dim[0]
    if parameter_count == 0:
        return (SurdVector.zero()._as_scalar(),) * 3
    weights = tuple(fractions.Fraction(1) if width is None else 1 / (width * width) for width in widths)
    zero = SurdVector.zero()._as_scalar()
    normal = [
        [
            sum(
                (
                    mapping._element((left, component)) * weights[component] * mapping._element((right, component))
                    for component in range(3)
                ),
                start=zero,
            )._as_scalar()
            for right in range(parameter_count)
        ]
        for left in range(parameter_count)
    ]
    target = [
        sum(
            (
                source[component] * weights[component] * mapping._element((parameter, component))
                for component in range(3)
            ),
            start=zero,
        )._as_scalar()
        for parameter in range(parameter_count)
    ]
    parameters = _solve_surd_system(normal, target)
    return tuple(
        sum(
            (parameters[parameter] * mapping._element((parameter, component)) for parameter in range(parameter_count)),
            start=zero,
        )._as_scalar()
        for component in range(3)
    )


def _matches_moment_symmform(moment: tuple[SurdScalar, ...], value: str | None) -> bool:
    basis = _moment_symmform_basis(value)
    if basis is None:
        return True
    if not basis:
        return all(component.is_zero() for component in moment)
    projection = _weighted_projection(
        SurdVector(basis),
        moment,
        (fractions.Fraction(1),) * 3,
    )
    return projection == moment


class SymopsStructure(StructureSemanticsMixin, StructureBackend):
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
    :param moment_component_resolutions: Optional decimal steps for each listed moment
        component, aligned in the moment backend's native frame.
    :param moment_component_esds: Optional standard uncertainties for each listed moment
        component, aligned in the moment backend's native frame.
    :param moment_symmforms: Optional source magnetic symmetry-form declarations aligned
        with the listed sites.
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
        moment_component_resolutions: Sequence[Sequence[Any | None]] | None = None,
        moment_component_esds: Sequence[Sequence[Any | None]] | None = None,
        moment_symmforms: Sequence[str | None] | None = None,
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
        component_resolutions = _normalize_moment_claims(
            moment_component_resolutions,
            len(norm_sites),
            name="moment_component_resolutions",
        )
        component_esds = _normalize_moment_claims(
            moment_component_esds,
            len(norm_sites),
            name="moment_component_esds",
        )
        symmforms = _normalize_moment_symmforms(moment_symmforms, len(norm_sites))
        if norm_site_moments is None and any(
            value is not None for value in (component_resolutions, component_esds, symmforms)
        ):
            raise ValueError("SymopsStructure moment metadata requires site_moments")

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
        self._moment_component_resolutions = component_resolutions
        self._moment_component_esds = component_esds
        self._moment_symmforms = symmforms
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
    def moment_component_resolutions(
        self,
    ) -> tuple[tuple[fractions.Fraction | None, ...], ...] | None:
        """Expose source decimal steps for each listed moment component, in its native frame."""
        return self._moment_component_resolutions

    @property
    def moment_component_esds(
        self,
    ) -> tuple[tuple[fractions.Fraction | None, ...], ...] | None:
        """Expose source ESDs for each listed moment component, in its native frame."""
        return self._moment_component_esds

    @property
    def moment_symmforms(self) -> tuple[str | None, ...] | None:
        """Expose source `_atom_site_moment.symmform` declarations aligned with listed sites."""
        return self._moment_symmforms

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
        return SurdVector(self.sites.reduced_coords) * self._cell.basis

    def _reconciled_moment_rows(self) -> tuple[tuple[Any, ...], ...] | tuple[Any, ...] | None:
        source_moments = self._listed_site_moments
        if source_moments is None:
            return None
        source_rows = _moment_rows(source_moments, self._cell)
        if isinstance(source_moments, CollinearSiteMoments):
            return source_rows
        if self._moment_component_resolutions is None and self._moment_component_esds is None:
            return source_rows

        resolutions = self._moment_component_resolutions or ((None, None, None),) * len(self._listed_sites)
        esds = self._moment_component_esds or ((None, None, None),) * len(self._listed_sites)
        if isinstance(source_moments, CrystalAxisSiteMoments):
            source_frame = source_moments.crystalaxis_moments
        elif isinstance(source_moments, CartesianSiteMoments):
            source_frame = source_moments.cartesian_moments
        else:
            raise TypeError(f"unsupported SiteMomentsBackend kind: {getattr(source_moments, 'kind', None)!r}")

        reconciled: list[tuple[Any, ...]] = []
        identity = ((fractions.Fraction(1), 0, 0), (0, fractions.Fraction(1), 0), (0, 0, fractions.Fraction(1)))
        for site_index, site in enumerate(self._listed_sites):
            source_row = source_rows[site_index]
            frame_row = tuple(source_frame._element((site_index, component)) for component in range(3))
            symmform = None if self._moment_symmforms is None else self._moment_symmforms[site_index]
            own_position = site.normalize()
            stabilizer = tuple(
                (operation, time_reversal)
                for operation, time_reversal in self._symops
                if operation.apply_wrapped(site) == own_position
            )
            if all(
                _transform_lattice_moment(source_row, operation, time_reversal) == source_row
                for operation, time_reversal in stabilizer
            ):
                if not _matches_moment_symmform(frame_row, symmform):
                    logger.warning(
                        "site %d moment symmetry form %r does not describe the stabilizer-compatible "
                        "moment; retained it as source metadata",
                        site_index,
                        symmform,
                        extra={"context": "mcif"},
                    )
                reconciled.append(source_row)
                continue

            constraints: list[tuple[fractions.Fraction, ...]] = []
            for operation, time_reversal in stabilizer:
                matrix = operation.matrix.to_fractions()
                factor = time_reversal * int(operation.determinant())
                constraints.extend(
                    tuple(factor * matrix[row][column] - identity[row][column] for column in range(3))
                    for row in range(3)
                )
            invariant_basis = _rational_null_space(constraints, 3)
            widths = _claim_widths(resolutions[site_index], esds[site_index])
            corrected_frame: tuple[SurdScalar, ...]
            if not invariant_basis:
                corrected_frame = (SurdVector.zero()._as_scalar(),) * 3
            else:
                basis = SurdVector(invariant_basis)
                if isinstance(source_moments, CrystalAxisSiteMoments):
                    mapping = SurdVector(
                        [
                            [
                                basis._element((parameter, component)) * self._cell.lengths[component]
                                for component in range(3)
                            ]
                            for parameter in range(len(invariant_basis))
                        ]
                    )
                else:
                    mapping = basis * self._cell.basis
                corrected_frame = _weighted_projection(mapping, frame_row, widths)
            for component, (original, corrected, width) in enumerate(zip(frame_row, corrected_frame, widths)):
                difference = _absolute((corrected - original)._as_scalar())
                if difference.is_zero():
                    continue
                if width is None or difference > width:
                    claim = "unknown" if width is None else str(width)
                    raise ValueError(
                        "internally inconsistent structure: site "
                        f"{site_index} moment component {component} differs from its symmetry-invariant "
                        f"projection by {difference} beyond source resolution/ESD {claim}; "
                        "different species/moment would occupy one generated site"
                    )

            if not _matches_moment_symmform(corrected_frame, symmform):
                logger.warning(
                    "site %d moment symmetry form %r does not describe the stabilizer-compatible "
                    "moment; retained it as source metadata",
                    site_index,
                    symmform,
                    extra={"context": "mcif"},
                )

            if isinstance(source_moments, CrystalAxisSiteMoments):
                corrected_row = tuple(
                    (corrected_frame[component] / self._cell.lengths[component])._as_scalar() for component in range(3)
                )
            else:
                converted = SurdVector._from_scalar_grid([corrected_frame], (1, 3)) * self._cell.basis.inv()
                corrected_row = tuple(converted._element((0, component)) for component in range(3))
            if any(
                _transform_lattice_moment(corrected_row, operation, time_reversal) != corrected_row
                for operation, time_reversal in stabilizer
            ):
                raise RuntimeError("magnetic stabilizer projection did not produce an invariant moment")
            reconciled.append(corrected_row)
        return tuple(reconciled)

    @cached_property
    def _expansion(self) -> tuple[Sites, tuple[str, ...], SiteMomentsBackend | None]:
        source_moments = self._listed_site_moments
        source_rows = self._reconciled_moment_rows()
        generated: list[tuple[tuple[Any, ...], str, tuple[Any, ...] | Any]] = []
        seen: dict[tuple[Any, ...], list[tuple[str, tuple[Any, ...] | Any, int]]] = {}

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
                previous = seen.get(position_key, [])
                if any(
                    previous_species == species and previous_moment == moment
                    for previous_species, previous_moment, _ in previous
                ):
                    continue
                if previous and any(previous_site == site_index for _, _, previous_site in previous):
                    raise ValueError(
                        "internally inconsistent structure: operation "
                        f"{operation_index} maps site {site_index} onto an already-generated site "
                        "with a different species/moment"
                    )
                seen.setdefault(position_key, []).append((species, moment, site_index))
                block.append((position_key, species, moment))
            generated.extend(sorted(block, key=lambda item: item[0]))

        coordinates = FracVector([list(item[0]) for item in generated])
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

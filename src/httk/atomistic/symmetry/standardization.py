"""Express an asymmetric-unit structure in its IT standard-setting cell.

The operation is exact after any optional recognition step. A setting-local ASU is mapped
to the standard Wyckoff table only here, because this operation explicitly requests the
standard conventional cell. An untabulated ASU instead uses its stored exact transform.
"""

import fractions
from dataclasses import dataclass

from httk.core import unwrap

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import initialize_semantics
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.recognition import _cartesian_distance_squared, recognize_asu, structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

from ._standardization_common import (
    CRYSTAL_AXIS_MOMENT_REFUSAL,
    MAGNETIC_SUPERCELL_REFUSAL,
    _as_existing_asu,
    _exact_site_bijection,
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _moments_close,
    _remap_assemblies,
    _reorder_site_moments,
    _scaled_composition,
    _scaled_precision,
    _semantic_value,
)

__all__ = ["ConventionalCellResult", "conventional_cell"]


@dataclass(frozen=True, slots=True)
class ConventionalCellResult:
    """Store a structure in its space group's IT standard-setting conventional cell.

    ``asu`` is the new standard-setting ASU that was expanded to make ``structure``.
    ``transform`` is the standard-to-own transform from the ASU that was supplied to, or
    recognized from, the operation; its orientation is :math:`f_own = f_std M^T + v`, so
    this operation undoes it for the cell basis. ``multiplier`` is the exact ratio of
    conventional-cell site count to input-cell site count. For the 527 vendored settings
    it is at least one; an untabulated, caller-supplied supercell transform may still
    produce a ratio below one.

    :param structure: The resulting full conventional-cell structure.
    :param asu: The resulting asymmetric-unit structure in the standard setting.
    :param spacegroup: The space group represented by the result.
    :param transform: The standard-to-own transform used for the input structure.
    :param multiplier: The exact ratio of result site count to input site count.
    """

    structure: UnitcellStructure
    asu: ASUStructure
    spacegroup: Spacegroup
    transform: SettingTransform
    multiplier: fractions.Fraction


def _moment_correspondence(
    original: UnitcellStructureView,
    result: UnitcellStructureView,
    transform: SettingTransform,
    tolerance: float,
    moments: SiteMomentsBackend,
) -> tuple[int, ...]:
    """Match each idealized output site to the input site whose moment it carries.

    Every output site (centring copies included, being input-lattice translates) is mapped
    back into the input setting through the known exact transform, wrapped, and matched to an
    input site of the same species within the Cartesian recognition tolerance. Input
    coordinates may be noisy floats and the output is idealized, so this is a tolerance match
    under a fixed transform, not a free registration.

    When nuclear recognition finds a smaller cell (a magnetic supercell input), several input
    sites fold onto one output site. The output match alone would silently drop the surplus
    moments, so every input site left unmatched is folded into the output cell and required to
    agree with the site it lands on; disagreement means a genuine magnetic supercell.

    :param original: The input structure carrying the per-site moments.
    :param result: The idealized standard-setting output structure.
    :param transform: The standard-to-input setting transform.
    :param tolerance: The Cartesian matching distance in the input cell's units.
    :param moments: The per-site input moments, used for the fold-agreement check.
    :return: For each output site, the index of the input site it corresponds to.
    :raises ValueError: If an output site has no unique input match, an unmatched input site
        does not fold onto exactly one output site, or a folded input moment disagrees.
    """
    input_sites = list(zip(original.sites.reduced_coords, original.species_at_sites, strict=True))
    cell = original.cell
    limit = tolerance * tolerance
    order: list[int] = []
    for out_index, (out_coord, out_species) in enumerate(
        zip(result.sites.reduced_coords, result.species_at_sites, strict=True)
    ):
        own = transform.to_setting(out_coord).normalize()
        matches = [
            index
            for index, (coord, species) in enumerate(input_sites)
            if species == out_species and _cartesian_distance_squared(own - coord, cell) <= limit
        ]
        if len(matches) != 1:
            raise ValueError(
                f"conventional_cell could not carry site moments: output site {out_index} matched "
                f"{len(matches)} input sites within the recognition tolerance; keep the original setting"
            )
        order.append(matches[0])

    matched = set(order)
    if len(matched) < len(input_sites):
        output_sites = list(zip(result.sites.reduced_coords, result.species_at_sites, strict=True))
        result_cell = result.cell
        for index, (coord, species) in enumerate(input_sites):
            if index in matched:
                continue
            standard = transform.to_standard(coord).normalize()
            folded = [
                out_index
                for out_index, (out_coord, out_species) in enumerate(output_sites)
                if species == out_species and _cartesian_distance_squared(standard - out_coord, result_cell) <= limit
            ]
            if len(folded) != 1:
                raise ValueError(
                    f"conventional_cell could not carry site moments: input site {index} folded onto "
                    f"{len(folded)} output sites within the recognition tolerance; keep the original setting"
                )
            if not _moments_close(moments, order[folded[0]], index):
                raise ValueError(MAGNETIC_SUPERCELL_REFUSAL)
    return tuple(order)


def conventional_cell(
    structure: StructureLike | ASUStructure,
    *,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
) -> ConventionalCellResult:
    """Return ``structure`` in its space group's IT standard-setting conventional cell.

    An existing :class:`~httk.atomistic.ASUStructure` (including an
    :class:`~httk.atomistic.ASUStructureView`, an ASU backend, or a full-cell view backed
    by one) is used exactly as stored. Supplying ``tolerance`` or ``limit_denominator`` for
    that path raises :class:`ValueError`, because those arguments belong to recognition.
    Any other :class:`~httk.atomistic.StructureLike` is first passed to
    :func:`~httk.atomistic.recognize_asu`. That tolerant step may snap measured coordinates
    onto symmetry positions and chooses the transform recorded in the result; it does not
    preserve an unstated input transform or promise a :func:`~httk.atomistic.same_crystal`
    match to noisy input coordinates. The optional tolerance is a Cartesian matching
    distance and the optional denominator limit idealises free parameters.

    The returned ``transform`` is the existing ASU's transform, or the transform chosen by
    recognition for a plain input; the returned ``asu`` has an identity transform.
    Construction and expansion are exact, including the rhombohedral case where the
    standard hexagonal cell contains three primitive cells. Basis precision is multiplied by
    ``M.T()`` and coordinate precision by the maximum absolute column sum of
    ``inv(M.T())``; unknown precision remains unknown. Requires a fully 3D-periodic
    structure.

    :param structure: The structure or asymmetric-unit structure to standardize.
    :param tolerance: The Cartesian recognition tolerance, or ``None`` to derive it.
    :param limit_denominator: The maximum denominator for idealised free parameters, or
        ``None`` to retain their exact stated values.
    :return: The standardized structure and transform metadata.
    :raises ImportError: If recognition is needed and the optional spglib dependency is
        unavailable.
    :raises ValueError: If recognition arguments are supplied for an existing ASU, the
        structure is not fully periodic, or unsupported site moments are present.
    """
    original = UnitcellStructureView(structure)
    asu = _as_existing_asu(structure)
    had_existing_asu = asu is not None
    source: object = original if asu is not None else unwrap(original)
    source_molecular = bool(_semantic_value(source, original, "molecular", False))
    source_assemblies = _semantic_value(source, original, "assemblies")
    if source_assemblies is not None:
        source_assemblies = tuple(source_assemblies)
    source_composition = _semantic_value(source, original, "chemical_composition")
    source_descriptive = _semantic_value(source, original, "chemical_formula_descriptive")
    source_hill = _semantic_value(source, original, "chemical_formula_hill")
    source_optimization = _semantic_value(source, original, "optimization_type")
    carried_moments: SiteMomentsBackend | None = None
    if asu is not None:
        if tolerance is not None or limit_denominator is not None:
            raise ValueError("conventional_cell() tolerance and limit_denominator cannot be used with an existing ASU")
    else:
        recognition_source: StructureLike = original
        if original.site_moments is not None:
            if isinstance(original.site_moments, CrystalAxisSiteMoments):
                raise ValueError(CRYSTAL_AXIS_MOMENT_REFUSAL)
            # Moments are per-site data, never symmetry input. Recognize the nuclear structure so
            # an altermagnet's opposite moments on one orbit cannot make the moment-aware ASU
            # collapse refuse; the moments are re-attached below by site correspondence.
            carried_moments = original.site_moments
            recognition_source = UnitcellStructure(
                original.cell,
                original.sites,
                original.species,
                original.species_at_sites,
                molecular=source_molecular,
                charge=original.charge,
            )
        asu = recognize_asu(
            recognition_source,
            tolerance=tolerance,
            limit_denominator=limit_denominator,
        )

    assert asu is not None
    # A Cartesian moment is a lab-frame vector and a collinear moment a frame-free scalar, so both
    # ride a setting change unchanged (it recombines the basis and shifts the origin, without
    # rotating the Cartesian frame). The existing per-orbit ASU machinery carries them through
    # expansion. A crystal-axis moment is stated against the old cell and cannot pass a cell change.
    if any(isinstance(site.moment, CrystalAxisSiteMoments) for site in asu.wyckoff_sites):
        raise ValueError(CRYSTAL_AXIS_MOMENT_REFUSAL)
    transform = asu.transform_from_standard
    standard, standard_sites = asu._standard_wyckoff_sites()
    basis_matrix = transform.matrix.T()
    coordinate_matrix = basis_matrix.inv()
    new_cell_precision = _scaled_precision(
        asu.cell.precision,
        _matrix_row_sum_factor(basis_matrix),
    )
    new_coordinate_precision = _scaled_precision(
        asu.coordinate_precision,
        _matrix_column_sum_factor(coordinate_matrix),
    )
    new_cell = Cell(
        transform.basis_to_standard(asu.cell.basis),
        precision=new_cell_precision,
        periodicity=asu.cell.periodicity,
    )

    def _standard_asu(charge: fractions.Fraction | None = None) -> ASUStructure:
        return ASUStructure(
            new_cell,
            standard,
            standard_sites,
            asu.species,
            transform=SettingTransform.identity(),
            coordinate_precision=new_coordinate_precision,
            molecular=source_molecular,
            # An existing reduced representation indexes assemblies against its domain.
            # Plain input assemblies index the full unit cell and are remapped below.
            assemblies=asu.assemblies if had_existing_asu else None,
            charge=charge,
        )

    standard_asu = _standard_asu()
    result_structure: UnitcellStructure = UnitcellStructureView(standard_asu)
    original_count = len(original.sites)
    if original_count == 0:
        raise ValueError("conventional_cell() cannot determine a site-count multiplier for an empty structure")
    multiplier = fractions.Fraction(len(result_structure.sites), original_count)

    if original.charge is not None:
        # The multiplier is only known after the first expansion, so rebuild with scaled charge.
        standard_asu = _standard_asu(original.charge * multiplier)
        result_structure = UnitcellStructureView(standard_asu)

    # All tabulated transforms have determinant 1 or 3. Keep this invariant explicit while
    # allowing a caller-supplied untabulated transform to describe a larger own cell.
    if asu.setting() is not None:
        assert multiplier >= 1

    if had_existing_asu:
        result_assemblies = result_structure.assemblies
    else:
        mapping = _exact_site_bijection(original, UnitcellStructureView(result_structure), transform)
        result_assemblies = _remap_assemblies(source_assemblies, mapping)
    scaled_composition = _scaled_composition(source_composition, multiplier) if source_composition is not None else None
    # The returned ASU is part of the public transform result, so retain every annotation
    # it can express directly. Full-cell assemblies from a plain input remain on the
    # expanded structure because their indices do not name domain sites.
    initialize_semantics(
        standard_asu,
        nsites=len(standard_asu.wyckoff_sites),
        molecular=source_molecular,
        assemblies=asu.assemblies if had_existing_asu else None,
        symmetry=None,
        chemical_composition=scaled_composition,
        chemical_formula_descriptive=source_descriptive,
        chemical_formula_hill=source_hill,
        optimization_type=source_optimization,
    )
    if carried_moments is None:
        initialize_semantics(
            result_structure,
            nsites=len(result_structure.sites),
            molecular=source_molecular,
            assemblies=result_assemblies,
            symmetry=None,
            chemical_composition=scaled_composition,
            chemical_formula_descriptive=source_descriptive,
            chemical_formula_hill=source_hill,
            optimization_type=source_optimization,
        )
    else:
        # Re-attach the carried moments to the idealized output by site correspondence. The
        # nuclear ASU stays moment-free; the moments live only on the expanded structure.
        effective_tolerance = tolerance if tolerance is not None else structure_tolerance(original)
        order = _moment_correspondence(
            original, UnitcellStructureView(result_structure), transform, effective_tolerance, carried_moments
        )
        result_structure = UnitcellStructure(
            new_cell,
            result_structure.sites,
            result_structure.species,
            result_structure.species_at_sites,
            site_moments=_reorder_site_moments(carried_moments, order),
            molecular=source_molecular,
            assemblies=result_assemblies,
            chemical_composition=scaled_composition,
            chemical_formula_descriptive=source_descriptive,
            chemical_formula_hill=source_hill,
            optimization_type=source_optimization,
            charge=None if original.charge is None else original.charge * multiplier,
        )

    return ConventionalCellResult(
        result_structure,
        standard_asu,
        standard,
        transform,
        multiplier,
    )

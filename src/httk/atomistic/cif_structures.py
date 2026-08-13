"""Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

:func:`asu_structure_from_cif` consumes the plain, string-preserving mapping produced by
``httk.io.cif`` (format tag ``"cif"``) and turns it into an exact ASU representation. The
conversion only understands the neutral mapping shape, keeping the parser and domain model
decoupled; the private reader bridge below adds precision metadata needed by this adapter.

A CIF is the natural source for an ASU: it lists one site per orbit and states the symmetry
operations that generate the rest. That means no symmetry *search* is needed, and spglib is
not involved. The setting is identified by comparing the file's operations against the
tabulated ones exactly, so a file written in a non-standard setting is recognized as such
rather than silently reinterpreted.
"""

import fractions
import math
import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from httk.core import FracVector, decimal_precision

from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map
from httk.atomistic.symmetry.symop_key import symop_key_v1
from httk.atomistic.symmetry.xyz import operation_from_xyz

from . import data as symmetry_data
from ._composition_values import as_fraction

__all__ = ["asu_structure_from_cif", "asu_structures_from_cif", "cif_setting"]

_ALLOW_LARGE_CIF_UNCERTAINTY = "_httk_atomistic_allow_large_cif_uncertainty"

_TYPE_SYMBOL = re.compile(r"^(?P<symbol>[A-Z][a-z]?)(?:(?P<magnitude>\d+)?(?P<sign>[+-])|(?P<neutral>0))?$")

CIF_POSITIONAL_UNCERTAINTY_WARNING = fractions.Fraction(1, 10)
CIF_POSITIONAL_UNCERTAINTY_ERROR = fractions.Fraction(1)


def asu_structures_from_cif(
    payload: Mapping[str, Any], *, autocorrect: bool = False, **options: Any
) -> list[ASUStructure]:
    r"""Return every structure in a loaded CIF payload, one per structural data block.

    Accepts either a whole loaded payload (with ``blocks``) or a single block.

    Reading a CIF is tolerant — a file may hold blocks that are not structures at all —
    but *asking it for structures* is not. If the file yielded none, the reasons the
    reader recorded are raised here rather than returning an empty list, so a file that
    could not be interpreted does not read as a file that contained nothing.

    :param payload: The loaded whole-CIF payload or one loaded CIF block.
    :param autocorrect: Apply documented CIF input repairs, also enabled by a stamped payload.
    :param \*\*options: Options forwarded to :func:`asu_structure_from_cif`.
    :return: One asymmetric-unit structure for each structural data block.
    :raises ValueError: If the payload has no interpretable structural data or a block is invalid.
    """
    options.setdefault("allow_large_cif_uncertainty", bool(payload.get(_ALLOW_LARGE_CIF_UNCERTAINTY, False)))
    options.setdefault("autocorrect", autocorrect or bool(payload.get("autocorrect", False)))
    blocks = payload.get("blocks")
    if blocks is None:
        return [asu_structure_from_cif(payload, **options)]

    if not blocks:
        unparsed = payload.get("unparsed") or []
        if unparsed:
            detail = "; ".join(f"block {item['block']!r}: {item['reason']}" for item in unparsed)
            raise ValueError(f"this CIF holds no structure that could be interpreted ({detail})")
        raise ValueError("this CIF holds no structural data blocks (none of them have atom sites)")

    return [asu_structure_from_cif(block, **options) for block in blocks]


def asu_structure_from_cif(
    data: Mapping[str, Any],
    *,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
    trust_declared_symmetry: bool = True,
    allow_large_cif_uncertainty: bool = False,
    autocorrect: bool = False,
) -> ASUStructure:
    """Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

    ``data`` is one block of the mapping returned by ``httk.io.cif`` (its ``format`` must be
    ``"cif"``).

    The cell is built exactly from the file's ``a, b, c, alpha, beta, gamma`` rather than
    from the pre-multiplied floating-point basis, so a cubic cell keeps exact right angles
    and a hexagonal one keeps its ``sqrt(3)`` instead of inheriting rounding noise.

    Coordinates are embedded as the rational the file literally wrote — ``0.3333`` is
    ``3333/10000``, not the binary value of ``float("0.3333")`` — and are then snapped onto
    the Wyckoff position they lie within ``tolerance`` of. That snapping is the only
    tolerant step; see :mod:`~httk.atomistic.symmetry.recognition` for the full contract.

    ``tolerance`` left unspecified is derived from the precision the file's own digits
    imply, so a coarsely written file is matched loosely and a carefully written one
    tightly, without anybody choosing a constant.

    Site occupancies become the composition of the corresponding
    :class:`~httk.atomistic.Species`, so a half-occupied site survives into the structure
    instead of being dropped.

    ``trust_declared_symmetry=False`` ignores the file's declared Hall symbol or space-group
    number and identifies the setting from its symmetry operations alone; see
    :func:`cif_setting` for when that is the right thing to do.

    :param data: One loaded CIF data block.
    :param tolerance: The Cartesian matching tolerance, or ``None`` to derive it from the CIF.
    :param limit_denominator: The maximum denominator for snapped free parameters, if supplied.
    :param trust_declared_symmetry: Whether to validate the declared symmetry before matching operations.
    :param allow_large_cif_uncertainty: Whether to allow positional uncertainty at or above one angstrom.
    :param autocorrect: Apply documented CIF input repairs with warnings.
    :return: The exact asymmetric-unit structure.
    :raises ValueError: If the block format, symmetry, coordinates, occupancies, or Wyckoff matches are invalid.
    """
    fmt = data.get("format")
    if fmt != "cif":
        raise ValueError(f"asu_structure_from_cif expected a 'cif' mapping, got format={fmt!r}.")

    try:
        setting = cif_setting(data, trust_declared_symmetry=trust_declared_symmetry)
    except ValueError as error:
        declared = _declared_symmetry(data)
        if (
            not autocorrect
            or not trust_declared_symmetry
            or declared is None
            or "names no known space-group setting" not in str(error)
        ):
            raise
        setting = cif_setting(data, trust_declared_symmetry=False)
        _cif_warning(
            f"CIF block {_block_name(data)!r}: ignored declared symmetry {declared} and identified "
            f"setting {setting.setting!r} from its symmetry operations"
        )
    standard = setting.standard_setting()
    transform = setting.transform_from_standard
    cell = _cell_from_cif(data)

    derived_tolerance = tolerance is None
    if derived_tolerance:
        # Derived from the digits the file itself wrote, rather than a constant. The sites
        # are the asymmetric unit rather than a full cell, which is all this needs: the
        # tolerance depends on the precision and the cell, not on how many atoms there are.
        tolerance = _tolerance_from_cif(data, cell)
    assert tolerance is not None

    coordinates = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    occupancies = data.get("occupancies")
    occupancies_exact = data.get("occupancies_exact")
    occupancy_precisions = data.get("occupancy_precisions")
    declared_wyckoff = data.get("_httk_atomistic_wyckoff_labels")
    declared_multiplicities = data.get("_httk_atomistic_symmetry_multiplicities")

    species_by_name: dict[str, Species] = {}
    wyckoff_sites: list[WyckoffSite] = []
    warning_uncertainties: list[Any] = []
    for index, coordinate in enumerate(coordinates):
        if occupancies_exact is not None and occupancies_exact[index] is not None:
            occupancy = occupancies_exact[index]
        elif occupancies is None:
            occupancy = 1
        elif occupancies[index] is None:
            raise ValueError(f"CIF occupancy is missing for site {labels[index]!r}")
        else:
            occupancy = occupancies[index]
        occupancy_precision = None if occupancy_precisions is None else occupancy_precisions[index]
        raw_symbol = symbols[index]
        symbol, charge = _parse_type_symbol(raw_symbol)
        name = _species_name(raw_symbol, labels[index], occupancy)
        if name not in species_by_name:
            species_by_name[name] = Species(
                name=name,
                chemical_symbols=(symbol,),
                concentration=(occupancy,),
                original_name=None if labels[index] == symbols[index] else labels[index],
                concentration_precision=(occupancy_precision,) if occupancy_precisions is not None else None,
                charges=(charge,) if charge is not None else None,
            )

        standard_point = transform.to_standard(coordinate).normalize()
        uncertainty = _site_uncertainty(data, index, cell) if derived_tolerance else None
        orbit_screen: list[tuple[tuple[float, float, float], Any, FracVector]] = []
        declared_position, declaration, declaration_error = _declared_wyckoff_position(
            declared_wyckoff, declared_multiplicities, index, setting, standard
        )
        ignored_declaration: tuple[str, str] | None = None
        exact_match = None if declaration is not None else _exact_wyckoff_match(standard, standard_point)
        if declaration is not None and declared_position is not None:
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                tolerance,
                uncertainty=uncertainty,
                exact_match=None,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                positions=(declared_position,),
                orbit_screen=orbit_screen,
            )
            if match is None:
                declaration_error = (
                    f"does not lie on its declared Wyckoff position {declared_position.letter!r}: "
                    f"measured distance {_nearest_wyckoff_distance(declared_position, standard_point, coordinate, cell, transform):.6g} "
                    f"exceeds tolerance {tolerance:.6g}"
                )
            elif _has_rounded_orbit_overlap(
                standard,
                transform,
                match,
                cell,
                tolerance,
                orbit_screen,
                include_coincident=True,
                expected_distinct=_setting_local_multiplicity(standard, setting, declared_position),
            ):
                actual = _snap(
                    standard,
                    standard_point,
                    coordinate,
                    cell,
                    transform,
                    tolerance,
                    allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                    most_specific=True,
                )
                assert actual is not None
                declaration_error = (
                    f"declares Wyckoff position {declared_position.letter!r}, but its coordinate lies on the "
                    f"more-specific Wyckoff position {actual[0]!r}"
                )
        else:
            match = None
        if declaration_error is not None:
            if not autocorrect:
                raise ValueError(
                    f"CIF site {labels[index]!r} has invalid declaration {declaration!r}: {declaration_error}. "
                    "Remedy: load(..., autocorrect=True) ignores the declaration and searches the coordinates."
                )
            assert declaration is not None  # an error always describes a present declaration
            ignored_declaration = declaration, declaration_error
            declaration = None
            exact_match = _exact_wyckoff_match(standard, standard_point)
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                tolerance,
                uncertainty=uncertainty,
                exact_match=exact_match,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                orbit_screen=orbit_screen,
            )
        elif declaration is None:
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                tolerance,
                uncertainty=uncertainty,
                exact_match=exact_match,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                orbit_screen=orbit_screen,
            )
        if (
            autocorrect
            and declaration is None
            and match is not None
            and _has_rounded_orbit_overlap(standard, transform, match, cell, tolerance, orbit_screen)
        ):
            corrected_match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                tolerance,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                most_specific=True,
            )
            if corrected_match != match:
                assert corrected_match is not None
                # Undeclared, fully occupied near-special sites are ordinarily rounded
                # measurements. Partial occupancy instead denotes a deliberate split site.
                if as_fraction(occupancy, field="CIF occupancy")[0] == 1:
                    match = corrected_match
                    _cif_warning(
                        f"CIF block {_block_name(data)!r}, site {labels[index]!r}: snapped its rounded coordinate "
                        f"to the more-specific Wyckoff position {match[0]!r}"
                    )
        if match is None:
            if ignored_declaration is not None:
                rejected, reason = ignored_declaration
                _cif_warning(
                    f"CIF block {_block_name(data)!r}, site {labels[index]!r}: ignored declared Wyckoff data "
                    f"{rejected!r} ({reason}) and selected no Wyckoff position from the coordinates"
                )
            raise ValueError(
                f"CIF site {labels[index]!r} at {tuple(coordinate.to_fractions())} does not lie on any "
                f"Wyckoff position of {setting.setting} within {tolerance}; the file's coordinates and its "
                f"symmetry operations disagree"
            )
        letter, parameters = match
        if ignored_declaration is not None:
            rejected, reason = ignored_declaration
            _cif_warning(
                f"CIF block {_block_name(data)!r}, site {labels[index]!r}: ignored declared Wyckoff data "
                f"{rejected!r} ({reason}) and selected Wyckoff position {letter!r} from the coordinates"
            )
        if limit_denominator is not None and parameters.dim not in ((), (0,)):
            parameters = FracVector([value.limit_denominator(limit_denominator) for value in parameters.to_fractions()])
        wyckoff_sites.append(WyckoffSite(letter, parameters, name, coordinate.normalize()))
        if exact_match is None and uncertainty is not None and uncertainty[0] >= CIF_POSITIONAL_UNCERTAINTY_WARNING**2:
            warning_uncertainties.append(uncertainty[0])

    if warning_uncertainties:
        maximum = max(warning_uncertainties)
        _cif_warning(
            f"CIF block has {len(warning_uncertainties)} site(s) with projected positional uncertainty; "
            f"maximum is {math.sqrt(maximum.to_float()):.6g} Å"
        )

    canonical_sites = _deduplicate_wyckoff_sites(
        standard,
        transform,
        wyckoff_sites,
        species_by_name,
        labels,
        block_name=_block_name(data),
        autocorrect=autocorrect,
    )
    used_species = {site.species for site in canonical_sites}
    return ASUStructure(
        cell,
        standard,
        canonical_sites,
        [species for name, species in species_by_name.items() if name in used_species],
        transform,
        data.get("coordinate_precision"),
    )


def _deduplicate_wyckoff_sites(
    spacegroup: Spacegroup,
    transform: Any,
    sites: list[WyckoffSite],
    species_by_name: Mapping[str, Species],
    labels: Sequence[str],
    *,
    block_name: str,
    autocorrect: bool,
) -> list[WyckoffSite]:
    """Remove redundant identical-species CIF orbits before building the ASU."""
    seen: dict[tuple[fractions.Fraction, ...], tuple[str, WyckoffSite]] = {}
    cosets = transform.lattice_cosets()
    canonical: list[WyckoffSite] = []
    for index, site in enumerate(sites):
        position = spacegroup.wyckoff_position(site.wyckoff)
        keys = {
            tuple((transform.to_setting(point) + coset).normalize().to_fractions())
            for point in position.coordinates(site.free_params)
            for coset in cosets
        }
        overlaps: list[tuple[tuple[fractions.Fraction, ...], tuple[str, WyckoffSite]]] = []
        for key in keys:
            previous = seen.get(key)
            if previous is None:
                continue
            overlaps.append((key, previous))
        conflicts = [
            (key, previous)
            for key, previous in overlaps
            if species_by_name[previous[0]] != species_by_name[site.species]
        ]
        if conflicts:
            key, previous = conflicts[0]
            if autocorrect and len(overlaps) == len(keys):
                _cif_warning(
                    f"CIF block {block_name!r}, site {labels[index]!r}: dropped co-located disorder site; "
                    "the ASU model cannot represent co-located different-species sites and occupancy "
                    "information is lost"
                )
                continue
            raise ValueError(
                f"{site!r} coincides with {previous[1]!r} at {key} but has a different species. "
                "Remedy: load(..., autocorrect=True) keeps the first co-located site and drops the later one."
            )
        if overlaps:
            if len(overlaps) == len(keys):
                continue
            raise ValueError(f"{site!r} partially overlaps an earlier orbit; the CIF is not a valid ASU")
        canonical.append(site)
        seen.update({key: (site.species, site) for key in keys})
    return canonical


def _has_rounded_orbit_overlap(
    spacegroup: Spacegroup,
    transform: Any,
    match: tuple[str, FracVector],
    cell: Cell,
    tolerance: float,
    orbit_screen: Sequence[tuple[tuple[float, float, float], Any, FracVector]],
    *,
    include_coincident: bool = False,
    expected_distinct: int | None = None,
) -> bool:
    """Whether a matched orbit has distinct images within the CIF-derived tolerance.

    Float Cartesian buckets screen possible pairs at twice the tolerance. Exact distance
    arithmetic confirms every screened pair, so floats only prune work. With
    ``include_coincident=True``, exact duplicate images also count; an authoritative
    declaration names a Wyckoff *stratum*, so its coordinate may not collapse into a
    proper sub-stratum.
    """
    from itertools import product

    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    letter, parameters = match
    position = spacegroup.wyckoff_position(letter)
    if position.multiplicity == 1 and len(transform.lattice_cosets()) == 1:
        return False
    if len(orbit_screen) < 2:
        return False
    if include_coincident:
        assert expected_distinct is not None
        points = {
            tuple((transform.to_setting(branch.coordinate(parameters)) + coset).normalize().to_fractions())
            for _, branch, coset in orbit_screen
        }
        if len(points) < expected_distinct:
            return True

    basis = cell.basis.to_floats()
    inverse = cell.basis.inv().to_floats()
    offsets = tuple(product((-1, 0, 1), repeat=3))
    screen = tolerance * 2 + 1e-9
    coordinates = [item[0] for item in orbit_screen]
    bins = tuple(
        max(1, int(1 / (screen * math.sqrt(sum(inverse[row][column] ** 2 for row in range(3)))))) for column in range(3)
    )

    def bucket(coordinate: tuple[float, float, float]) -> tuple[int, int, int]:
        first, second, third = (min(count - 1, int(value * count)) for value, count in zip(coordinate, bins))
        return (first, second, third)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, coordinate in enumerate(coordinates):
        buckets.setdefault(bucket(coordinate), []).append(index)

    for index, coordinate in enumerate(coordinates):
        for offset in offsets:
            shifted = tuple((value + delta) % count for value, delta, count in zip(bucket(coordinate), offset, bins))
            nearby = (shifted[0], shifted[1], shifted[2])
            for other in buckets.get(nearby, []):
                if other <= index:
                    continue
                difference = tuple(
                    (first - second + 0.5) % 1 - 0.5 for first, second in zip(coordinate, coordinates[other])
                )
                cartesian = tuple(
                    sum(value * basis[row][column] for row, value in enumerate(difference)) for column in range(3)
                )
                if sum(value * value for value in cartesian) > screen * screen:
                    continue
                first = (
                    transform.to_setting(orbit_screen[index][1].coordinate(parameters)) + orbit_screen[index][2]
                ).normalize()
                second = (
                    transform.to_setting(orbit_screen[other][1].coordinate(parameters)) + orbit_screen[other][2]
                ).normalize()
                if first != second and _cartesian_distance_squared(first - second, cell) <= tolerance * tolerance:
                    return True
    return False


def _declared_wyckoff_position(
    wyckoff_labels: Any,
    multiplicities: Any,
    index: int,
    setting: Spacegroup,
    standard: Spacegroup,
) -> tuple[Any | None, str | None, str | None]:
    """Resolve one CIF declaration into the corresponding standard-setting position.

    CIF Wyckoff labels and multiplicities describe the file's own setting, including its
    centring convention. They are therefore resolved in ``setting`` and only then mapped
    to ``standard``. A malformed or ambiguous declaration is an integrity error, not weak
    evidence: callers choose whether to reject it or explicitly fall back to a search. A
    declaration names a Wyckoff *stratum*, not merely a containing affine map: callers also
    reject coordinates whose declared orbit collapses into a proper, more-specific stratum.

    :param wyckoff_labels: The block's per-site Wyckoff label column, or ``None``.
    :param multiplicities: The block's per-site multiplicity column, or ``None``.
    :param index: The site's position in the block's site lists.
    :param setting: The identified setting the declarations are expressed in.
    :param standard: The standard setting the resolved position is mapped into.
    :return: ``(standard_position, declaration, error)``; no declaration is all ``None``.
    """
    label = _site_declaration(wyckoff_labels, index)
    letter = None if label is None else label.lstrip("0123456789").lower()
    multiplicity = _site_declaration(multiplicities, index)
    if label is None and multiplicity is None:
        return None, None, None
    declaration = ", ".join(
        item
        for item in (
            None if label is None else f"Wyckoff label {label!r}",
            None if multiplicity is None else f"multiplicity {multiplicity!r}",
        )
        if item is not None
    )
    positions = _setting_wyckoff_declarations(standard, setting)
    by_letter = None
    if letter is not None:
        by_letter = next(
            ((position, local_multiplicity) for local, position, local_multiplicity in positions if local == letter),
            None,
        )
        if by_letter is None:
            return None, declaration, f"unknown setting-local Wyckoff letter {label!r}"
    try:
        value = None if multiplicity is None else int(multiplicity)
    except ValueError:
        return None, declaration, f"invalid setting-local multiplicity {multiplicity!r}"
    by_multiplicity = None
    if value is not None:
        matching = [position for _, position, multiplicity in positions if multiplicity == value]
        if by_letter is not None:
            if by_letter[1] != value:
                return None, declaration, "the declared letter and multiplicity identify different positions"
            by_multiplicity = by_letter[0]
        elif len(matching) != 1:
            return None, declaration, f"ambiguous or unknown setting-local multiplicity {multiplicity!r}"
        else:
            by_multiplicity = matching[0]
    if by_letter is not None and by_multiplicity is not None and by_letter[0] != by_multiplicity:
        return None, declaration, "the declared letter and multiplicity identify different positions"
    return (None if by_letter is None else by_letter[0]) or by_multiplicity, declaration, None


def _site_declaration(values: Any, index: int) -> str | None:
    """One optional raw CIF atom-site declaration, if present."""
    if not isinstance(values, list) or index >= len(values):
        return None
    value = str(values[index]).strip()
    return None if value in {"", ".", "?"} else value


@cache
def _setting_wyckoff_declarations(standard: Spacegroup, setting: Spacegroup) -> tuple[tuple[str, Any, int], ...]:
    """Cache setting-local declarations translated to standard Wyckoff positions."""
    letters = wyckoff_letter_map(standard, setting)
    local_multiplicities = {position.letter: position.multiplicity for position in setting.wyckoff}
    return tuple(
        (local, standard.wyckoff_position(letter), local_multiplicities[local]) for letter, local in letters.items()
    )


def _setting_local_multiplicity(standard: Spacegroup, setting: Spacegroup, position: Any) -> int:
    """Return the setting-local multiplicity for a standard position from a CIF declaration."""
    return next(
        multiplicity
        for _, candidate, multiplicity in _setting_wyckoff_declarations(standard, setting)
        if candidate == position
    )


def _tolerance_from_cif(data: Mapping[str, Any], cell: Cell) -> float:
    """A matching tolerance from the precision this CIF block states.

    Built directly rather than via :func:`~httk.atomistic.structure_tolerance`, which needs
    an assembled structure; here the sites are still an asymmetric unit and the cell has
    only just been made. The arithmetic is the same one: the coordinate precision is a
    fraction of a cell edge, so it becomes a length against the longest one, floored by the
    cell's own precision, and doubled because two independently rounded values can differ by
    twice their precision.
    """
    from httk.atomistic.symmetry.recognition import _SAFETY_FACTOR, DEFAULT_TOLERANCE

    coordinate_precision = data.get("coordinate_precision")
    basis_precision = data.get("basis_precision")
    if coordinate_precision is None and basis_precision is None:
        return DEFAULT_TOLERANCE

    longest = max(length.to_float() for length in cell.lengths)
    cartesian = 0.0 if coordinate_precision is None else float(coordinate_precision) * longest
    if basis_precision is not None:
        cartesian = max(cartesian, float(basis_precision))
    return cartesian * _SAFETY_FACTOR


def _site_uncertainty(data: Mapping[str, Any], index: int, cell: Cell) -> tuple[Any, str] | None:
    """Return one site's exact Cartesian uncertainty and the token causing it."""
    precisions = data.get("position_precisions")
    tokens = data.get("positions_exact")
    raw_tokens = data.get("position_tokens", tokens)
    if tokens is None:
        return None
    if precisions is None:
        stated_precision = data.get("coordinate_precision")
        if stated_precision is not None:
            precisions = tuple(tuple(stated_precision for _ in row) for row in tokens)
        else:
            precisions = tuple(tuple(decimal_precision(token) for token in row) for row in tokens)
    stated = [
        (precision, component, str(token))
        for component, (precision, token) in enumerate(zip(precisions[index], tokens[index]))
        if precision is not None and token is not None
    ]
    if not stated:
        return None
    _, component, token = max(stated, key=lambda item: item[0])
    if raw_tokens is not None:
        token = str(raw_tokens[index][component])
    from itertools import product

    from httk.atomistic.symmetry.recognition import _SAFETY_FACTOR

    corners = []
    for signs in product((-1, 1), repeat=3):
        fractional = FracVector([sign * (precision or 0) for sign, precision in zip(signs, precisions[index])])
        vector = cell.basis.T() * fractional
        corners.append((vector.lengthsqr(), vector))
    # Keep the exact squared norm: nested radicals can be impractical for triclinic cells,
    # while threshold comparisons remain exact and the square root is only for diagnostics.
    squared = max(corners, key=lambda item: item[0])[0]
    return squared * _SAFETY_FACTOR**2, token


def _cif_warning(message: str) -> None:
    """Send one CIF warning through httk-core's report channel."""
    import logging

    logging.getLogger(__name__).warning(message, extra={"context": "cif"})


def _block_name(data: Mapping[str, Any]) -> str:
    """The CIF data-block name retained by the autocorrect reader bridge."""
    return str(data.get("_httk_atomistic_block_name", "<unnamed>"))


def _declared_symmetry(data: Mapping[str, Any]) -> str | None:
    """The Hall symbol or International Tables number written by the CIF."""
    hall = data.get("space_group_name_hall")
    if hall:
        return f"Hall symbol {str(hall).strip()!r}"
    number = data.get("space_group_nbr")
    if number is not None:
        return f"International Tables number {str(number).strip()!r}"
    return None


def _exact_wyckoff_match(standard: Spacegroup, standard_point: FracVector) -> tuple[str, FracVector] | None:
    """Return an exact special-position match after a floating-point zero-distance screen.

    The screen can only nominate candidates: ``parameters_of`` remains the exact decision.
    Every coordinate is reduced into one unit cell and the affine maps have tiny integer
    coefficients, so ``1e-13`` is deliberately far above double-rounding noise while far
    below any CIF precision that can affect a tolerant match.
    """
    point = tuple(standard_point.to_floats())
    for position in standard.wyckoff:
        if position.free_count == 3:
            continue
        for branch in position.branches:
            candidate = branch.coordinate_float(branch.nearest_parameters_float(point))
            if any(abs((first - second + 0.5) % 1.0 - 0.5) > 1e-13 for first, second in zip(point, candidate)):
                continue
            parameters = branch.parameters_of(standard_point)
            if parameters is not None:
                return position.letter, parameters
    return None


def cif_setting(data: Mapping[str, Any], *, trust_declared_symmetry: bool = True) -> Spacegroup:
    """The space-group setting a CIF block is written in.

    The setting is identified from the file's symmetry **operations**, by exact set
    comparison against the tabulated settings. That is what makes a file written in a
    non-standard setting come out as itself rather than being silently reinterpreted.

    What the file *declares* — a Hall symbol, or an International Tables number — is treated
    as a claim to be checked, not a hint to be taken or dropped. A declaration that names no
    known setting, or that names one whose operations are not the file's, is a genuine
    inconsistency in the file and raises rather than being worked around: the two halves of
    the file disagree, and quietly believing one of them is how a wrong structure gets built.

    Pass ``trust_declared_symmetry=False`` to ignore the declaration entirely and identify
    the setting from the operations alone. That is the escape hatch for a file whose symbols
    are known to be wrong but whose operations are good.

    The Hermann-Mauguin symbol is deliberately **not** checked. It has too many legitimate
    spellings, and OPTIMADE itself notes that it does not unambiguously communicate the axis,
    cell, or origin choice, so treating a mismatch there as an error would reject good files.

    Raises :class:`ValueError` when the block states no operations, when a declaration is
    inconsistent with them, or when the operations match no tabulated setting at all. In the
    last case the transform to the standard setting genuinely cannot be *derived* — infinitely
    many are equally valid and they describe different crystals — so such a file has to be
    built with an explicit :class:`~httk.atomistic.SettingTransform`.

    :param data: The loaded CIF data block.
    :param trust_declared_symmetry: Whether to check the declared Hall symbol or IT number.
    :return: The tabulated space-group setting matching the block's operations.
    :raises ValueError: If operations are absent, inconsistent with the declaration, or unknown.
    """
    operations = data.get("symops_xyz")
    if not operations:
        raise ValueError("this CIF block states no symmetry operations, so its setting cannot be determined")
    target = frozenset(operation_from_xyz(operation).wrapped() for operation in operations)

    candidates: Sequence[Mapping[str, Any]] | None = None
    declared = None
    if trust_declared_symmetry:
        candidates, declared = _declared_settings(data)
    try:
        candidate = Spacegroup(symmetry_data.spacegroup_setting_by_symop_key(symop_key_v1(target)))
    except KeyError:
        candidate = None
    if candidate is not None:
        candidate_operations = frozenset(operation.wrapped() for operation in candidate.symmetry_operations)
        if candidate_operations != target:
            raise RuntimeError(
                f"symop-key index internal inconsistency: CIF operations key maps to setting {candidate.setting!r}, "
                "but that setting's exact operations differ from the CIF operations"
            )
        if candidates is None or any(record["hall_entry"] == candidate.hall_entry for record in candidates):
            return candidate

    if declared is not None:
        raise ValueError(
            f"this CIF declares {declared}, but its {len(target)} symmetry operations are not that "
            f"setting's. The file contradicts itself. If the operations are the trustworthy half, "
            f"pass trust_declared_symmetry=False to identify the setting from them alone."
        )
    raise ValueError(
        f"the {len(target)} symmetry operations in this CIF match no tabulated space-group setting. "
        f"If this is a genuinely non-standard setting, build the structure with an explicit "
        f"SettingTransform instead; a transform cannot be derived, since infinitely many are "
        f"equally valid and they describe different crystals."
    )


def _declared_settings(data: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    """The settings the file's own declaration allows, and how it was described.

    ``(None, None)`` when the file declares nothing, in which case every tabulated setting is
    a candidate. A declaration that cannot name a real setting raises here rather than being
    ignored, because silently ignoring it is indistinguishable from the file being right.
    """
    hall = data.get("space_group_name_hall")
    if hall:
        written = str(hall).strip()
        try:
            record = symmetry_data.spacegroup_setting(hall_entry=_normalized_hall(written))
        except KeyError:
            raise ValueError(
                f"this CIF declares the Hall symbol {written!r}, which names no known space-group "
                f"setting. Pass trust_declared_symmetry=False to ignore the declaration and identify "
                f"the setting from the symmetry operations alone. Remedy: load(..., autocorrect=True) "
                f"ignores an unrecognized declared symmetry and identifies the setting from the operations."
            ) from None
        return [record], f"the Hall symbol {written!r}"

    number = data.get("space_group_nbr")
    if number is not None:
        written = str(number).strip()
        try:
            it_number = int(written)
        except ValueError:
            raise ValueError(
                f"this CIF declares the International Tables number {written!r}, which is not a "
                f"number. Pass trust_declared_symmetry=False to ignore the declaration."
            ) from None
        if not 1 <= it_number <= 230:
            raise ValueError(
                f"this CIF declares the International Tables number {it_number}, which is outside the "
                f"range 1-230. Pass trust_declared_symmetry=False to ignore the declaration."
            )
        narrowed = [record for record in symmetry_data.spacegroup_settings() if record["it_number"] == it_number]
        return narrowed, f"International Tables number {it_number}"

    return None, None


def _normalized_hall(symbol: str) -> str:
    """A Hall symbol in the spelling the tables index it under.

    A CIF writes Hall symbols conventionally — ``-C 2yc`` — while the tables key them as
    ``-c_2yc``. Lower-casing and turning spaces into underscores reproduces the tabulated
    spelling for all 527 settings, which ``tests/test_symmetry_data.py`` checks. Without this
    step every correctly declared Hall symbol looks unknown.
    """
    return symbol.lower().replace(" ", "_")


def _cell_from_cif(data: Mapping[str, Any]) -> Cell:
    """The cell, built exactly from the lattice parameters where the file gives them."""
    precision = data.get("basis_precision")
    exact = data.get("cell_parameters_exact")
    if exact is not None and all(value is not None for value in exact):
        # The text the file wrote, so 5.6402 becomes 56402/10000 rather than the binary
        # value of float("5.6402").
        return Cell(CellParams([fractions.Fraction(value) for value in exact]).basis, 1, precision)
    raise ValueError("CIF payload has no complete exact cell-parameter channel")


def _exact_positions(data: Mapping[str, Any]) -> list[FracVector]:
    """Site coordinates as the exact rationals the file wrote.

    Prefers the preserved decimal text over the parsed floats: ``Fraction("0.3333")`` is
    ``3333/10000``, whereas ``Fraction(float("0.3333"))`` is a binary approximation that
    states a precision the file never claimed.
    """
    exact = data.get("positions_exact")
    if exact is not None:
        return [FracVector([fractions.Fraction(value) for value in row]) for row in exact]
    raise ValueError("CIF payload has no exact fractional-coordinate channel")


def _species_name(symbol: str, label: str, occupancy: Any) -> str:
    """A species name: the element where that is unambiguous, else the file's site label.

    A fully occupied site is named for its element, which keeps ordinary structures
    readable. A partially occupied one is named for its CIF label instead, since two sites
    of the same element can carry different occupancies and would otherwise collide.
    """
    return symbol if as_fraction(occupancy, field="CIF occupancy")[0] == 1 else label


def _parse_type_symbol(symbol: str) -> tuple[str, fractions.Fraction | None]:
    """Split a CIF type symbol into an element symbol and an optional charge.

    :param symbol: The raw ``_atom_site_type_symbol`` value.
    :return: The element symbol and its explicit charge, if the value is decorated.
    """
    match = _TYPE_SYMBOL.fullmatch(symbol)
    if match is None:
        return symbol, None
    element = match.group("symbol")
    if element not in SYMBOLS:
        return symbol, None
    magnitude = match.group("magnitude")
    sign = match.group("sign")
    if match.group("neutral") is not None:
        return element, fractions.Fraction(0)
    if sign is None:
        return element, None
    charge = fractions.Fraction(1 if magnitude is None else int(magnitude))
    return element, charge if sign == "+" else -charge


def _read_cif_for_atomistic(
    source: Any, *, allow_large_cif_uncertainty: bool = False, autocorrect: bool = False
) -> Mapping[str, Any]:
    """Read CIF through httk-io and carry the atomistic override to its adapter."""
    from httk.io.cif import read_cif
    from httk.io.cif.cif_parser import cifblock_to_asu

    if autocorrect:
        try:
            raw_blocks, header = read_cif(source, allow_cif2=False, autocorrect=True)
        except TypeError as error:
            if "autocorrect" not in str(error):
                raise
            raise ValueError(
                "CIF autocorrect requires httk-io with CIF autocorrect support, which is not yet released."
            ) from error
    else:
        raw_blocks, header = read_cif(source, allow_cif2=False)
    blocks = []
    unparsed = []
    for name, raw_block in raw_blocks:
        if "atom_site_label" not in raw_block:
            continue
        try:
            block = cifblock_to_asu(raw_block)
        except Exception as error:
            unparsed.append({"block": name, "reason": f"{type(error).__name__}: {error}"})
        else:
            blocks.append(
                {
                    **block,
                    "position_precisions": _position_precisions(raw_block),
                    "position_tokens": _position_tokens(raw_block),
                    "_httk_atomistic_wyckoff_labels": raw_block.get("atom_site_wyckoff_label"),
                    "_httk_atomistic_symmetry_multiplicities": raw_block.get("atom_site_symmetry_multiplicity"),
                    **({"_httk_atomistic_block_name": name} if autocorrect else {}),
                }
            )
    payload: dict[str, Any] = {"format": "cif", "blocks": blocks, "unparsed": unparsed, "header": header}
    if autocorrect:
        payload["autocorrect"] = True
    if not allow_large_cif_uncertainty:
        return payload
    return {**payload, _ALLOW_LARGE_CIF_UNCERTAINTY: True}


def _position_precisions(block: Mapping[str, Any]) -> list[tuple[fractions.Fraction | None, ...]]:
    """Preserve per-component CIF digit/ESD precision for the compatibility reader bridge."""
    from httk.core import combined_precision
    from httk.io.cif.cif_parser import cif_exact_token, parse_cif_float

    columns = [block[f"atom_site_fract_{axis}"] for axis in "xyz"]
    companions = [block.get(f"httk_atom_site_fract_{axis}_exact") for axis in "xyz"]
    has_companion = any(value is not None for value in companions)
    result = []
    for index, values in enumerate(zip(*columns)):
        row: list[fractions.Fraction | None] = []
        for axis, value in enumerate(values):
            companion = companions[axis]
            companion_value = companion[index] if isinstance(companion, list) and index < len(companion) else None
            if (companion_value is not None and cif_exact_token(companion_value) is not None) or (
                has_companion and cif_exact_token(value) in {"0", "1"}
            ):
                row.append(None)
            else:
                meta = parse_cif_float(value, meta=True)[1]
                row.append(combined_precision((meta["precision"], meta["esd"])))
        result.append(tuple(row))
    return result


def _position_tokens(block: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Preserve raw CIF coordinate tokens for precise uncertainty diagnostics."""
    return list(zip(*(block[f"atom_site_fract_{axis}"] for axis in "xyz")))


def _snap(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Cell,
    transform: Any,
    tolerance: float,
    *,
    uncertainty: tuple[Any, str] | None = None,
    exact_match: tuple[str, FracVector] | None = None,
    allow_large_cif_uncertainty: bool = False,
    most_specific: bool = False,
    positions: Sequence[Any] | None = None,
    orbit_screen: list[tuple[tuple[float, float, float], Any, FracVector]] | None = None,
) -> tuple[str, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance``, and its free parameters.

    Floating point screens branch candidates at twice the tolerance, but an exact distance
    check still accepts every result. ``positions`` limits the search to authoritative CIF
    declarations; otherwise every standard position is tried in its established order.
    """
    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    def finish(letter: str, parameters: FracVector) -> tuple[str, FracVector]:
        if orbit_screen is not None:
            orbit_screen.clear()
            position = standard.wyckoff_position(letter)
            values = tuple(parameters.to_floats())
            matrix = transform.matrix.to_floats()
            vector = transform.vector.to_floats()
            for branch in position.branches:
                candidate = branch.coordinate_float(values)
                own = tuple(
                    sum(matrix[row][column] * candidate[column] for column in range(3)) + vector[row]
                    for row in range(3)
                )
                for coset in transform.lattice_cosets():
                    orbit_screen.append(
                        (tuple((value + shift) % 1.0 for value, shift in zip(own, coset.to_floats())), branch, coset)
                    )
        return letter, parameters

    if exact_match is not None and positions is None and not most_specific:
        return finish(*exact_match)

    if uncertainty is not None:
        projected, token = uncertainty
        if projected >= CIF_POSITIONAL_UNCERTAINTY_ERROR**2 and not allow_large_cif_uncertainty:
            raise ValueError(
                f"CIF coordinate token {token!r} implies a projected positional uncertainty of "
                f"{math.sqrt(projected.to_float()):.6g} Å; pass allow_large_cif_uncertainty=True to override"
            )
    limit = tolerance * tolerance
    screen = (tolerance * 2 + 1e-9) ** 2
    basis = cell.basis.to_floats()
    point = tuple(own_point.to_floats())
    standard_float = tuple(standard_point.to_floats())
    matrix = transform.matrix.to_floats()
    vector = transform.vector.to_floats()
    candidates = standard.wyckoff if positions is None else positions

    def screened(branch: Any) -> bool:
        parameters = branch.nearest_parameters_float(standard_float)
        candidate = branch.coordinate_float(parameters)
        own_candidate = tuple(
            sum(matrix[row][column] * candidate[column] for column in range(3)) + vector[row] for row in range(3)
        )
        difference = tuple((first - second + 0.5) % 1 - 0.5 for first, second in zip(point, own_candidate))
        cartesian = tuple(sum(difference[row] * basis[row][column] for row in range(3)) for column in range(3))
        return sum(value * value for value in cartesian) <= screen

    matches: list[tuple[int, str, FracVector]] = []
    for position in candidates:
        for branch in position.branches:
            if not screened(branch):
                continue
            parameters = branch.nearest_parameters(standard_point)
            candidate = transform.to_setting(branch.coordinate(parameters))
            if _cartesian_distance_squared(own_point - candidate, cell) <= limit:
                if not most_specific:
                    return finish(position.letter, parameters)
                matches.append((position.free_count, position.letter, parameters))
                break
    if not matches:
        return None
    _, letter, parameters = min(matches, key=lambda match: match[0])
    return finish(letter, parameters)


def _nearest_wyckoff_distance(
    position: Any,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Cell,
    transform: Any,
) -> float:
    """Return the exact nearest projected distance for a declaration diagnostic."""
    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    return (
        min(
            _cartesian_distance_squared(
                own_point - transform.to_setting(branch.coordinate(branch.nearest_parameters(standard_point))), cell
            )
            for branch in position.branches
        )
        ** 0.5
    )

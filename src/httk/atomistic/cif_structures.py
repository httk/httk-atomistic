"""Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

:func:`asu_structure_from_cif` consumes the plain, string-preserving mapping produced by
``httk.atomistic.io.cif`` (format tag ``"cif"``) and turns it into an exact ASU representation. The
conversion only understands the neutral mapping shape, keeping the parser and domain model
decoupled; the private reader bridge below adds precision metadata needed by this adapter.

A CIF is the natural source for an ASU: it lists one site per orbit and states the symmetry
operations that generate the rest. That means no symmetry *search* is needed, and spglib is
not involved. The setting is identified by comparing the file's operations against the
tabulated ones exactly, so a file written in a non-standard setting is recognized as such
rather than silently reinterpreted.
"""

import fractions
import itertools
import logging
import math
import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, NamedTuple

from httk.core import FracVector, decimal_precision

from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite, _ValidatedASUProof
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map
from httk.atomistic.symmetry.symop_key import symop_key_v1
from httk.atomistic.symmetry.xyz import operation_from_xyz

from . import data as symmetry_data
from ._composition_values import as_fraction, normalization

__all__ = ["asu_structure_from_cif", "asu_structures_from_cif", "cif_setting"]

_ALLOW_LARGE_CIF_UNCERTAINTY = "_httk_atomistic_allow_large_cif_uncertainty"

_CIF_CORE_TYPE_SYMBOLS = frozenset(
    [
        "H",
        "D",
        "H1-",
        "He",
        "Li",
        "Li1+",
        "Be",
        "Be2+",
        "B",
        "C",
        "N",
        "O",
        "O1-",
        "F",
        "F1-",
        "Ne",
        "Na",
        "Na1+",
        "Mg",
        "Mg2+",
        "Al",
        "Al3+",
        "Si",
        "Si4+",
        "P",
        "S",
        "Cl",
        "Cl1-",
        "Ar",
        "K",
        "K1+",
        "Ca",
        "Ca2+",
        "Sc",
        "Sc3+",
        "Ti",
        "Ti2+",
        "Ti3+",
        "Ti4+",
        "V",
        "V2+",
        "V3+",
        "V5+",
        "Cr",
        "Cr2+",
        "Cr3+",
        "Mn",
        "Mn2+",
        "Mn3+",
        "Mn4+",
        "Fe",
        "Fe2+",
        "Fe3+",
        "Co",
        "Co2+",
        "Co3+",
        "Ni",
        "Ni2+",
        "Ni3+",
        "Cu",
        "Cu1+",
        "Cu2+",
        "Zn",
        "Zn2+",
        "Ga",
        "Ga3+",
        "Ge",
        "Ge4+",
        "As",
        "Se",
        "Br",
        "Br1-",
        "Kr",
        "Rb",
        "Rb1+",
        "Sr",
        "Sr2+",
        "Y",
        "Y3+",
        "Zr",
        "Zr4+",
        "Nb",
        "Nb3+",
        "Nb5+",
        "Mo",
        "Mo3+",
        "Mo5+",
        "Mo6+",
        "Tc",
        "Ru",
        "Ru3+",
        "Ru4+",
        "Rh",
        "Rh3+",
        "Rh4+",
        "Pd",
        "Pd2+",
        "Pd4+",
        "Ag",
        "Ag1+",
        "Ag2+",
        "Cd",
        "Cd2+",
        "In",
        "In3+",
        "Sn",
        "Sn2+",
        "Sn4+",
        "Sb",
        "Sb3+",
        "Sb5+",
        "Te",
        "I",
        "I1-",
        "Xe",
        "Cs",
        "Cs1+",
        "Ba",
        "Ba2+",
        "La",
        "La3+",
        "Ce",
        "Ce3+",
        "Ce4+",
        "Pr",
        "Pr3+",
        "Pr4+",
        "Nd",
        "Nd3+",
        "Pm",
        "Sm",
        "Sm3+",
        "Eu",
        "Eu2+",
        "Eu3+",
        "Gd",
        "Gd3+",
        "Tb",
        "Tb3+",
        "Dy",
        "Dy3+",
        "Ho",
        "Ho3+",
        "Er",
        "Er3+",
        "Tm",
        "Tm3+",
        "Yb",
        "Yb2+",
        "Yb3+",
        "Lu",
        "Lu3+",
        "Hf",
        "Hf4+",
        "Ta",
        "Ta5+",
        "W",
        "W6+",
        "Re",
        "Os",
        "Os4+",
        "Ir",
        "Ir3+",
        "Ir4+",
        "Pt",
        "Pt2+",
        "Pt4+",
        "Au",
        "Au1+",
        "Au3+",
        "Hg",
        "Hg1+",
        "Hg2+",
        "Tl",
        "TL1+",
        "Tl1+",
        "Tl3+",
        "Pb",
        "Pb2+",
        "Pb4+",
        "Bi",
        "Bi3+",
        "Bi5+",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ra2+",
        "Ac",
        "Ac3+",
        "Th",
        "Th4+",
        "Pa",
        "U",
        "U3+",
        "U4+",
        "U6+",
        "Np",
        "Np3+",
        "Np4+",
        "Np6+",
        "Pu",
        "Pu3+",
        "Pu4+",
        "Pu6+",
        "Am",
        "Cm",
        "Bk",
        "Cf",
    ]
)

_TYPE_SYMBOL_SUFFIX_CHARGE = re.compile(r"^(?P<label>.+?)(?P<magnitude>\d+)?(?P<sign>[+-])$")
_TYPE_SYMBOL_PREFIX_CHARGE = re.compile(r"^(?P<label>.+?)(?P<sign>[+-])(?P<magnitude>\d+)$")


class _DecodedCIFType(NamedTuple):
    """Semantic interpretation of one CIF atom-type symbol."""

    chemical_symbol: str
    charge: fractions.Fraction | None
    species_label: str | None
    mass: float | None
    recognized: bool


CIF_POSITIONAL_UNCERTAINTY_WARNING = fractions.Fraction(1, 10)
CIF_POSITIONAL_UNCERTAINTY_ERROR = fractions.Fraction(1)

_FLOAT_SCREEN_MAGNITUDE_LIMIT = 2**40
_FLOAT_SCREEN_ULPS = 4096


def _float_screen_slack(values: Sequence[float]) -> float | None:
    """A conservative Cartesian float-screen error bound, or ``None`` when unsafe."""
    if any(not math.isfinite(value) or abs(value) > _FLOAT_SCREEN_MAGNITUDE_LIMIT for value in values):
        return None
    magnitude = max(1.0, *(abs(value) for value in values))
    # A screen coordinate needs fewer than 20 correctly-rounded operations.  This gives it
    # 4096 ULPs at both input scales, including the later matrix product, so it exceeds the
    # double-rounding bound by two orders of magnitude while keeping ordinary screens sharp.
    return _FLOAT_SCREEN_ULPS * magnitude * math.ulp(magnitude)


type _SpecialPositionRule = tuple[tuple[tuple[int, int, int], float], ...]
type _GeneralPositionScreen = tuple[tuple[tuple[tuple[int, int, int], float, float], ...], ...]


@cache
def _setting_special_position_rules(setting: Spacegroup) -> tuple[_SpecialPositionRule, ...]:
    """Return each special branch's affine relations in this setting's coordinates."""
    result: set[_SpecialPositionRule] = set()
    for position in setting.wyckoff:
        if position.free_count == 3:
            continue
        for branch in position.branches:
            rows = branch._unimodular.to_fractions()
            vector = branch.operation.vector.to_fractions()
            rule = []
            for row in range(len(branch.free), 3):
                coefficients = (int(rows[row][0]), int(rows[row][1]), int(rows[row][2]))
                constant = (
                    sum(
                        (fractions.Fraction(coefficient) * value for coefficient, value in zip(coefficients, vector)),
                        start=fractions.Fraction(),
                    )
                    % 1
                )
                if next(value for value in coefficients if value) < 0:
                    coefficients = (-coefficients[0], -coefficients[1], -coefficients[2])
                    constant = (-constant) % 1
                rule.append((coefficients, float(constant)))
            result.add(tuple(rule))
    return tuple(sorted(result))


def _general_position_screen(setting: Spacegroup, cell: Cell, tolerance: float) -> _GeneralPositionScreen | None:
    """Build setting-local affine-relation bounds that can certify a general position."""
    try:
        inverse = cell.basis.inv()
        # If a Cartesian displacement has length at most t, each reduced component is
        # bounded by t times the L1 norm of the corresponding inverse-basis column.
        # Doubling that exact coefficient after float conversion keeps this a rejection
        # screen rather than a numerical decision boundary.
        inverse_float = inverse.to_floats()
        component_bounds = tuple(2 * sum(abs(inverse_float[row][column]) for row in range(3)) for column in range(3))
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return None
    scale = abs(tolerance) + 1e-9
    rules = _setting_special_position_rules(setting)
    values = [scale, *component_bounds, *(constant for rule in rules for _, constant in rule)]
    slack = _float_screen_slack(values)
    if slack is None:
        return None
    return tuple(
        tuple(
            (
                coefficients,
                constant,
                scale
                * sum(abs(coefficient) * component_bounds[index] for index, coefficient in enumerate(coefficients))
                + slack,
            )
            for coefficients, constant in rule
        )
        for rule in rules
    )


def _definitely_general(own_point: FracVector, screen: _GeneralPositionScreen) -> bool:
    """Return whether every special branch violates at least one required relation."""
    try:
        point = tuple(own_point.to_floats())
    except OverflowError:
        return False
    if _float_screen_slack([*point, *(constant for rule in screen for _, constant, _ in rule)]) is None:
        return False
    return all(
        any(
            abs(
                (sum(coefficient * value for coefficient, value in zip(coefficients, point)) - constant + 0.5) % 1 - 0.5
            )
            > limit
            for coefficients, constant, limit in rule
        )
        for rule in screen
    )


def asu_structures_from_cif(payload: Mapping[str, Any], *, repair: bool = False, **options: Any) -> list[ASUStructure]:
    r"""Return every structure in a loaded CIF payload, one per structural data block.

    Accepts either a whole loaded payload (with ``blocks``) or a single block.

    Reading a CIF is tolerant — a file may hold blocks that are not structures at all —
    but *asking it for structures* is not. If the file yielded none, the reasons the
    reader recorded are raised here rather than returning an empty list, so a file that
    could not be interpreted does not read as a file that contained nothing.

    :param payload: The loaded whole-CIF payload or one loaded CIF block.
    :param repair: Apply documented CIF input repairs, also enabled by a stamped payload.
    :param \*\*options: Options forwarded to :func:`asu_structure_from_cif`.
    :return: One asymmetric-unit structure for each structural data block.
    :raises ValueError: If the payload has no interpretable structural data or a block is invalid.
    """
    options.setdefault("allow_large_cif_uncertainty", bool(payload.get(_ALLOW_LARGE_CIF_UNCERTAINTY, False)))
    options.setdefault("repair", repair or bool(payload.get("repair", False)))
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
    repair: bool = False,
) -> ASUStructure:
    """Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

    ``data`` is one block of the mapping returned by ``httk.atomistic.io.cif`` (its ``format`` must be
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
    :param repair: Apply documented CIF input repairs with warnings.
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
            not repair
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
    standard = setting
    transform = SettingTransform.identity()
    cell = _cell_from_cif(data)

    derived_tolerance = tolerance is None
    if derived_tolerance:
        # Derived from the digits the file itself wrote, rather than a constant. The sites
        # are the asymmetric unit rather than a full cell, which is all this needs: the
        # tolerance depends on the precision and the cell, not on how many atoms there are.
        tolerance = _tolerance_from_cif(data, cell)
    assert tolerance is not None
    from httk.atomistic.symmetry.recognition import _SAFETY_FACTOR

    general_screens: dict[float, _GeneralPositionScreen | None] = {
        tolerance: _general_position_screen(setting, cell, tolerance)
    }
    uncertainty_metric = None
    if derived_tolerance:
        metric = cell.metric()
        uncertainty_metric = metric.coefficient(1) if metric.is_rational else metric

    coordinates = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    occupancies = data.get("occupancies")
    occupancies_exact = data.get("occupancies_exact")
    occupancy_precisions = data.get("occupancy_precisions")
    masses = data.get("masses")
    declared_wyckoff = data.get("_httk_atomistic_wyckoff_labels")
    declared_multiplicities = data.get("_httk_atomistic_symmetry_multiplicities")
    declared_site_symmetry_orders = data.get("_httk_atomistic_site_symmetry_orders")

    species_by_name: dict[str, Species] = {}
    wyckoff_sites: list[WyckoffSite] = []
    warning_uncertainties: list[Any] = []
    warned_type_symbols: set[str] = set()
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
        stated_mass = None if masses is None else masses[index]
        decoded = _decode_type_symbol(raw_symbol, stated_mass)
        if not decoded.recognized and raw_symbol not in warned_type_symbols:
            _cif_warning(
                f"unrecognized CIF atom-type symbol {raw_symbol!r}; represented as chemical symbol 'X' "
                f"with species label {decoded.species_label!r}"
            )
            warned_type_symbols.add(raw_symbol)
        name = _species_name(raw_symbol, labels[index], occupancy)
        if name not in species_by_name:
            species_by_name[name] = Species(
                name=name,
                chemical_symbols=(decoded.chemical_symbol,),
                concentration=(occupancy,),
                mass=(decoded.mass,) if decoded.mass is not None else None,
                original_name=None if labels[index] == symbols[index] else labels[index],
                concentration_precision=(occupancy_precision,) if occupancy_precisions is not None else None,
                charges=(decoded.charge,) if decoded.charge is not None else None,
                labels=(decoded.species_label,) if decoded.species_label is not None else None,
            )

        standard_point = coordinate.normalize()
        uncertainty = _site_uncertainty(data, index, uncertainty_metric) if derived_tolerance else None
        position_bounds = data.get("position_snap_bounds")
        coordinate_bounds = None if not derived_tolerance or position_bounds is None else position_bounds[index]
        site_tolerance = tolerance
        if uncertainty is not None and data.get("position_precisions") is not None:
            site_tolerance = math.sqrt(uncertainty[0].to_float())
            basis_precision = data.get("basis_precision")
            if basis_precision is not None:
                site_tolerance = max(site_tolerance, float(basis_precision) * _SAFETY_FACTOR)
        if site_tolerance not in general_screens:
            general_screens[site_tolerance] = _general_position_screen(setting, cell, site_tolerance)
        general_screen = general_screens[site_tolerance]
        declared_position, declaration, declaration_error, declared_positions = _declared_wyckoff_position(
            declared_wyckoff, declared_multiplicities, declared_site_symmetry_orders, index, setting, standard
        )
        orbit_screen: list[tuple[tuple[float, float, float], Any, FracVector]] | None = (
            [] if declaration is not None or repair else None
        )
        ignored_declaration: tuple[str, str] | None = None
        if declaration is not None and declared_position is not None:
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                site_tolerance,
                uncertainty=uncertainty,
                coordinate_bounds=coordinate_bounds,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                positions=(declared_position,),
                orbit_screen=orbit_screen,
            )
            if match is None:
                declaration_error = (
                    f"does not lie on its declared Wyckoff position {declared_position.letter!r}: "
                    f"measured distance {_nearest_wyckoff_distance(declared_position, standard_point, coordinate, cell, transform):.6g} "
                    f"exceeds tolerance {site_tolerance:.6g}"
                )
            elif match is not None:
                assert orbit_screen is not None
                if _has_rounded_orbit_overlap(
                    standard,
                    transform,
                    match,
                    cell,
                    site_tolerance,
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
                        site_tolerance,
                        coordinate_bounds=coordinate_bounds,
                        allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                        most_specific=True,
                    )
                    assert actual is not None
                    declaration_error = (
                        f"declares Wyckoff position {declared_position.letter!r}, but its coordinate lies on the "
                        f"more-specific Wyckoff position {actual[0]!r}"
                    )
        elif declaration is not None:
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                site_tolerance,
                uncertainty=uncertainty,
                coordinate_bounds=coordinate_bounds,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                positions=declared_positions,
                orbit_screen=orbit_screen,
            )
            if match is None:
                declaration_error = f"does not lie on any position allowed by declared {declaration}"
        else:
            match = None
        if declaration_error is not None:
            if not repair:
                raise ValueError(
                    f"CIF site {labels[index]!r} has invalid declaration {declaration!r}: {declaration_error}. "
                    "Remedy: load(..., repair=True) ignores the declaration and searches the coordinates."
                )
            assert declaration is not None  # an error always describes a present declaration
            ignored_declaration = declaration, declaration_error
            declaration = None
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                site_tolerance,
                uncertainty=uncertainty,
                coordinate_bounds=coordinate_bounds,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                orbit_screen=orbit_screen,
                general_screen=general_screen,
            )
        elif declaration is None:
            match = _snap(
                standard,
                standard_point,
                coordinate,
                cell,
                transform,
                site_tolerance,
                uncertainty=uncertainty,
                coordinate_bounds=coordinate_bounds,
                allow_large_cif_uncertainty=allow_large_cif_uncertainty,
                orbit_screen=orbit_screen,
                general_screen=general_screen,
            )
        if repair and declaration is None and match is not None:
            assert orbit_screen is not None
            if _has_rounded_orbit_overlap(standard, transform, match, cell, site_tolerance, orbit_screen):
                corrected_match = _snap(
                    standard,
                    standard_point,
                    coordinate,
                    cell,
                    transform,
                    site_tolerance,
                    coordinate_bounds=coordinate_bounds,
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
                f"Wyckoff position of {setting.setting} within {site_tolerance}; the file's coordinates and its "
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
        matched_position = standard.wyckoff_position(match[0])
        exact_special = (
            declaration is None
            and matched_position.free_count != 3
            and matched_position.parameters_of(standard_point) is not None
        )
        if not exact_special and uncertainty is not None and uncertainty[0] >= CIF_POSITIONAL_UNCERTAINTY_WARNING**2:
            warning_uncertainties.append(uncertainty[0])

    if warning_uncertainties:
        maximum = max(warning_uncertainties)
        _cif_warning(
            f"CIF block has {len(warning_uncertainties)} site(s) with projected positional uncertainty; "
            f"maximum is {math.sqrt(maximum.to_float()):.6g} Å"
        )

    proof, canonical_species = _deduplicate_wyckoff_sites(
        standard,
        transform,
        wyckoff_sites,
        species_by_name,
        labels,
        block_name=_block_name(data),
        coordinate_precision=data.get("coordinate_precision"),
    )
    return ASUStructure._from_validated_proof(
        cell,
        standard,
        proof,
        canonical_species,
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
    coordinate_precision: Any,
) -> tuple[_ValidatedASUProof, tuple[Species, ...]]:
    """Collapse repeated CIF orbits and combine co-located disorder losslessly."""
    cosets = transform.lattice_cosets()
    identity = transform.is_identity()
    groups: list[tuple[frozenset[tuple[fractions.Fraction, ...]], list[int]]] = []
    group_at_coordinate: dict[tuple[fractions.Fraction, ...], int] = {}
    for index, site in enumerate(sites):
        position = spacegroup.wyckoff_position(site.wyckoff)
        keys = frozenset(
            {tuple(point.normalize().to_fractions()) for point in position.coordinates(site.free_params)}
            if identity
            else {
                tuple((transform.to_setting(point) + coset).normalize().to_fractions())
                for point in position.coordinates(site.free_params)
                for coset in cosets
            }
        )
        overlapping_groups = {group_at_coordinate[key] for key in keys if key in group_at_coordinate}
        if not overlapping_groups:
            group_index = len(groups)
            groups.append((keys, [index]))
            group_at_coordinate.update((key, group_index) for key in keys)
            continue
        if len(overlapping_groups) != 1:
            raise ValueError(f"{site!r} partially overlaps earlier orbits; the CIF is not a valid ASU")
        group_index = next(iter(overlapping_groups))
        previous_keys, members = groups[group_index]
        if keys != previous_keys:
            raise ValueError(f"{site!r} partially overlaps an earlier orbit; the CIF is not a valid ASU")
        members.append(index)

    canonical: list[WyckoffSite] = []
    canonical_species: dict[str, Species] = {}
    coordinates: list[tuple[fractions.Fraction, ...]] = []
    species_at_sites: list[str] = []
    counts: list[int] = []
    for keys, members in groups:
        source_sites = [sites[index] for index in members]
        source_species = [(species_by_name[site.species], labels[index]) for site, index in zip(source_sites, members)]
        species = _combine_cif_species(source_species, block_name=block_name)
        previous = canonical_species.get(species.name)
        if previous is not None and previous != species:
            source_labels = ", ".join(repr(labels[index]) for index in members)
            raise ValueError(
                f"CIF block {block_name!r} uses site labels {source_labels} to form species name "
                f"{species.name!r}, but that name already describes a different species"
            )
        canonical_species.setdefault(species.name, species)
        representative = source_sites[0]
        canonical.append(
            WyckoffSite(
                representative.wyckoff,
                representative.free_params,
                species.name,
                representative.representative,
                representative.moment,
            )
        )
        ordered = sorted(keys)
        coordinates.extend(ordered)
        species_at_sites.extend((species.name,) * len(ordered))
        counts.append(len(ordered))
    reduced = FracVector([list(point) for point in coordinates]) if coordinates else FracVector(())
    return (
        _ValidatedASUProof._issue_from_cif_deduplication(
            spacegroup,
            transform,
            canonical,
            (reduced, tuple(species_at_sites), tuple(counts)),
            coordinate_precision,
        ),
        tuple(canonical_species.values()),
    )


def _combine_cif_species(sources: Sequence[tuple[Species, str]], *, block_name: str) -> Species:
    """Combine the distinct CIF rows for one orbit and make vacancies explicit."""
    distinct: list[tuple[Species, str]] = []
    for source in sources:
        if any(source[0] == previous[0] for previous in distinct):
            continue
        distinct.append(source)

    symbols: list[str]
    concentrations: list[fractions.Fraction]
    precisions: list[fractions.Fraction | None]
    charges: list[fractions.Fraction | None] | None
    labels: list[str | None] | None
    masses: list[float] | None
    if len(distinct) == 1:
        species, _ = distinct[0]
        symbols = list(species.chemical_symbols)
        concentrations = list(species.concentration)
        precisions = list(species.concentration_precision or (None,) * len(symbols))
        charges = None if species.charges is None else list(species.charges)
        labels = None if species.labels is None else list(species.labels)
        masses = None if species.mass is None else list(species.mass)
        name = species.name
        original_name = species.original_name
    else:
        constituents: list[
            tuple[
                str,
                fractions.Fraction,
                fractions.Fraction | None,
                fractions.Fraction | None,
                float | None,
                str,
                str,
            ]
        ] = []
        for species, source_label in distinct:
            if len(species.chemical_symbols) != 1:
                raise ValueError("internal CIF disorder aggregation expected one constituent per source row")
            source_precisions = species.concentration_precision
            constituent_species_labels = species.labels
            constituents.append(
                (
                    species.chemical_symbols[0],
                    species.concentration[0],
                    None if source_precisions is None else source_precisions[0],
                    None if species.charges is None else species.charges[0],
                    None if species.mass is None else species.mass[0],
                    source_label
                    if constituent_species_labels is None
                    else constituent_species_labels[0] or source_label,
                    source_label,
                )
            )
        constituents.sort(
            key=lambda item: (
                item[0],
                item[3] is None,
                "" if item[3] is None else str(item[3]),
                item[5],
                item[1],
            )
        )
        symbols = [item[0] for item in constituents]
        concentrations = [item[1] for item in constituents]
        precisions = [item[2] for item in constituents]
        charges = [item[3] for item in constituents]
        mass_values = [item[4] for item in constituents]
        has_nonvacancy_mass = any(
            symbol != "vacancy" and mass is not None for symbol, mass in zip(symbols, mass_values)
        )
        if has_nonvacancy_mass:
            if any(symbol != "vacancy" and mass is None for symbol, mass in zip(symbols, mass_values)):
                raise ValueError(
                    "CIF disorder orbit gives masses for only some constituents; the Species mass list "
                    "cannot represent that partial declaration exactly"
                )
            masses = []
            for symbol, mass in zip(symbols, mass_values):
                if symbol == "vacancy":
                    masses.append(0.0)
                else:
                    assert mass is not None
                    masses.append(mass)
        else:
            masses = None
        labels = [item[5] for item in constituents]
        name = "/".join(item[6] for item in constituents)
        original_name = None

    normalized, _, total, width = normalization(tuple(concentrations), tuple(precisions))
    if total > 1 and not normalized:
        source_label_text = ", ".join(repr(label) for _, label in distinct)
        raise ValueError(
            f"CIF block {block_name!r}, co-located sites {source_label_text}: occupancies sum to {total}, "
            "outside their stated precision around one"
        )
    if total < 1:
        symbols.append("vacancy")
        concentrations.append(1 - total)
        precisions.append(width)
        if charges is not None:
            charges.append(None)
        if labels is not None:
            labels.append(None)
        if masses is not None:
            masses.append(0.0)

    return Species(
        name=name,
        chemical_symbols=symbols,
        concentration=concentrations,
        mass=masses,
        original_name=original_name,
        concentration_precision=precisions,
        charges=charges,
        labels=labels,
    )


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

    Float Cartesian buckets screen possible pairs at the tolerance plus a scale-aware
    double-rounding margin; exact distance arithmetic confirms every screened pair. A
    non-finite, huge, or near-degenerate float conversion skips the screen and checks every
    pair exactly. With ``include_coincident=True``, exact duplicate images also count; an
    authoritative declaration names a Wyckoff *stratum*, so its coordinate may not collapse
    into a proper sub-stratum.
    """
    from itertools import product

    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    letter, parameters = match
    position = spacegroup.wyckoff_position(letter)
    identity = transform.is_identity()
    if position.multiplicity == 1 and len(transform.lattice_cosets()) == 1:
        return False
    if len(orbit_screen) < 2:
        return False
    if include_coincident:
        assert expected_distinct is not None
        points = {
            tuple(
                (
                    branch.coordinate(parameters)
                    if identity
                    else transform.to_setting(branch.coordinate(parameters)) + coset
                )
                .normalize()
                .to_fractions()
            )
            for _, branch, coset in orbit_screen
        }
        if len(points) < expected_distinct:
            return True

    basis = cell.basis.to_floats()
    inverse = cell.basis.inv().to_floats()
    offsets = tuple((first, second, third) for first, second, third in product((-1, 0, 1), repeat=3))
    coordinates = [item[0] for item in orbit_screen]
    slack = _float_screen_slack(
        [
            tolerance,
            *(value for row in basis for value in row),
            *(value for coordinate in coordinates for value in coordinate),
        ]
    )
    inverse_slack = _float_screen_slack([value for row in inverse for value in row])
    screen = None if slack is None or inverse_slack is None else abs(tolerance) + slack * 4
    bins = (1, 1, 1)
    if screen is not None:
        assert inverse_slack is not None
        try:
            calculated_bins = tuple(
                max(
                    1,
                    int(1 / (screen * (math.sqrt(sum(inverse[row][column] ** 2 for row in range(3))) + inverse_slack)))
                    - 1,
                )
                for column in range(3)
            )
            bins = (calculated_bins[0], calculated_bins[1], calculated_bins[2])
        except (OverflowError, ValueError, ZeroDivisionError):
            screen = None

    def bucket(coordinate: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            min(bins[0] - 1, int(coordinate[0] * bins[0])),
            min(bins[1] - 1, int(coordinate[1] * bins[1])),
            min(bins[2] - 1, int(coordinate[2] * bins[2])),
        )

    def neighbouring_bucket(
        coordinate: tuple[float, float, float], offset: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        current = bucket(coordinate)
        return (
            (current[0] + offset[0]) % bins[0],
            (current[1] + offset[1]) % bins[1],
            (current[2] + offset[2]) % bins[2],
        )

    if screen is None:
        pairs = ((index, other) for index in range(len(orbit_screen)) for other in range(index + 1, len(orbit_screen)))
    else:
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, coordinate in enumerate(coordinates):
            buckets.setdefault(bucket(coordinate), []).append(index)
        pairs = (
            (index, other)
            for index, coordinate in enumerate(coordinates)
            for offset in offsets
            for other in buckets.get(neighbouring_bucket(coordinate, offset), [])
            if other > index
        )

    for index, other in pairs:
        if screen is not None:
            difference = tuple(
                (first - second + 0.5) % 1 - 0.5 for first, second in zip(coordinates[index], coordinates[other])
            )
            cartesian = tuple(
                sum(value * basis[row][column] for row, value in enumerate(difference)) for column in range(3)
            )
            if sum(value * value for value in cartesian) > screen * screen:
                continue
        first = orbit_screen[index][1].coordinate(parameters)
        second = orbit_screen[other][1].coordinate(parameters)
        if not identity:
            first = transform.to_setting(first) + orbit_screen[index][2]
            second = transform.to_setting(second) + orbit_screen[other][2]
        first = first.normalize()
        second = second.normalize()
        if first != second and _cartesian_distance_squared(first - second, cell) <= tolerance * tolerance:
            return True
    return False


def _declared_wyckoff_position(
    wyckoff_labels: Any,
    multiplicities: Any,
    site_symmetry_orders: Any,
    index: int,
    setting: Spacegroup,
    standard: Spacegroup,
) -> tuple[Any | None, str | None, str | None, Sequence[Any] | None]:
    """Resolve one CIF declaration into the corresponding standard-setting position.

    CIF Wyckoff labels and multiplicities describe the file's own setting, including its
    centring convention. Labels are therefore resolved in ``setting`` and only then mapped
    to ``standard``. A multiplicity without a label filters the candidate strata but does not
    identify one; coordinate matching chooses among those candidates. A malformed or
    inconsistent declaration is an integrity error, not weak evidence: callers choose
    whether to reject it or explicitly fall back to a search. Callers also reject coordinates
    whose declared orbit collapses into a proper, more-specific stratum.

    :param wyckoff_labels: The block's per-site Wyckoff label column, or ``None``.
    :param multiplicities: The block's per-site multiplicity column, or ``None``.
    :param site_symmetry_orders: The block's per-site site-symmetry order column, or ``None``.
    :param index: The site's position in the block's site lists.
    :param setting: The identified setting the declarations are expressed in.
    :param standard: The standard setting the resolved position is mapped into.
    :return: ``(standard_position, declaration, error, candidate_positions)``.
    """
    label = _site_declaration(wyckoff_labels, index)
    letter = None if label is None else label.lstrip("0123456789").lower()
    multiplicity = _site_declaration(multiplicities, index)
    site_symmetry_order = _site_declaration(site_symmetry_orders, index)
    if label is None and multiplicity is None and site_symmetry_order is None:
        return None, None, None, None
    declaration = ", ".join(
        item
        for item in (
            None if label is None else f"Wyckoff label {label!r}",
            None if multiplicity is None else f"multiplicity {multiplicity!r}",
            None if site_symmetry_order is None else f"site-symmetry order {site_symmetry_order!r}",
        )
        if item is not None
    )
    positions = _setting_wyckoff_declarations(standard, setting)
    general_multiplicity = max(local_multiplicity for _, _, local_multiplicity in positions)
    declared_site_symmetry_order = None
    if site_symmetry_order is not None:
        try:
            declared_site_symmetry_order = int(site_symmetry_order)
        except ValueError:
            return None, declaration, f"invalid setting-local site-symmetry order {site_symmetry_order!r}", None

    if label is None and multiplicity is None:
        assert declared_site_symmetry_order is not None
        if declared_site_symmetry_order <= 0 or general_multiplicity % declared_site_symmetry_order != 0:
            return None, declaration, f"invalid setting-local site-symmetry order {site_symmetry_order!r}", None
        value = general_multiplicity // declared_site_symmetry_order
        matching = tuple(position for _, position, local_multiplicity in positions if local_multiplicity == value)
        if not matching:
            return None, declaration, f"unknown setting-local site-symmetry order {site_symmetry_order!r}", None
        candidates = tuple(
            sorted(matching, key=lambda position: (position.free_count, position.multiplicity, position.letter))
        )
        return None, declaration, None, candidates
    if label is None:
        assert multiplicity is not None
        try:
            multiplicity_value = int(multiplicity)
        except ValueError:
            return None, declaration, f"invalid setting-local multiplicity {multiplicity!r}", None
        if (
            declared_site_symmetry_order is not None
            and declared_site_symmetry_order * multiplicity_value != general_multiplicity
        ):
            return (
                None,
                declaration,
                "the declared multiplicity and site-symmetry order identify different positions",
                None,
            )
        matching = tuple(
            position for _, position, local_multiplicity in positions if local_multiplicity == multiplicity_value
        )
        if not matching:
            return None, declaration, f"unknown setting-local multiplicity {multiplicity!r}", None
        candidates = tuple(
            sorted(matching, key=lambda position: (position.free_count, position.multiplicity, position.letter))
        )
        return None, declaration, None, candidates
    by_letter = None
    if letter is not None:
        by_letter = next(
            ((position, local_multiplicity) for local, position, local_multiplicity in positions if local == letter),
            None,
        )
        if by_letter is None:
            return None, declaration, f"unknown setting-local Wyckoff letter {label!r}", None
        if (
            declared_site_symmetry_order is not None
            and general_multiplicity // by_letter[1] != declared_site_symmetry_order
        ):
            return (
                None,
                declaration,
                "the declared letter and site-symmetry order identify different positions",
                None,
            )
    try:
        declared_multiplicity = None if multiplicity is None else int(multiplicity)
    except ValueError:
        return None, declaration, f"invalid setting-local multiplicity {multiplicity!r}", None
    by_multiplicity = None
    if declared_multiplicity is not None:
        if (
            declared_site_symmetry_order is not None
            and declared_site_symmetry_order * declared_multiplicity != general_multiplicity
        ):
            return (
                None,
                declaration,
                "the declared multiplicity and site-symmetry order identify different positions",
                None,
            )
        matching_positions = [
            position for _, position, multiplicity in positions if multiplicity == declared_multiplicity
        ]
        if by_letter is not None:
            if by_letter[1] != declared_multiplicity:
                return None, declaration, "the declared letter and multiplicity identify different positions", None
            by_multiplicity = by_letter[0]
        elif len(matching_positions) != 1:
            return None, declaration, f"ambiguous or unknown setting-local multiplicity {multiplicity!r}", None
        else:
            by_multiplicity = matching_positions[0]
    if by_letter is not None and by_multiplicity is not None and by_letter[0] != by_multiplicity:
        return None, declaration, "the declared letter and multiplicity identify different positions", None
    return (None if by_letter is None else by_letter[0]) or by_multiplicity, declaration, None, None


def _site_declaration(values: Any, index: int) -> str | None:
    """One optional raw CIF atom-site declaration, if present."""
    if not isinstance(values, list) or index >= len(values):
        return None
    value = str(values[index]).strip()
    return None if value in {"", ".", "?"} else value


@cache
def _setting_wyckoff_declarations(standard: Spacegroup, setting: Spacegroup) -> tuple[tuple[str, Any, int], ...]:
    """Cache setting-local declarations translated to standard Wyckoff positions."""
    if standard == setting:
        return tuple((position.letter, position, position.multiplicity) for position in setting.wyckoff)
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


def _site_uncertainty(data: Mapping[str, Any], index: int, metric: Any) -> tuple[Any, str] | None:
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
    from httk.atomistic.symmetry.recognition import _SAFETY_FACTOR

    corners = []
    # Opposite corners have the same squared norm.  The cell metric is per-file invariant,
    # so evaluating four representatives avoids rebuilding Cartesian vectors for every site.
    for signs in ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
        fractional = FracVector([sign * (precision or 0) for sign, precision in zip(signs, precisions[index])])
        corners.append((metric * fractional).dot(fractional))
    # Keep the exact squared norm: nested radicals can be impractical for triclinic cells,
    # while threshold comparisons remain exact and the square root is only for diagnostics.
    squared = max(corners)
    return squared * _SAFETY_FACTOR**2, token


def _cif_warning(message: str) -> None:
    """Send one CIF warning through httk-core's report channel."""
    import logging

    logging.getLogger(__name__).warning(message, extra={"context": "cif"})


def _block_name(data: Mapping[str, Any]) -> str:
    """The CIF data-block name retained by the repair reader bridge."""
    return str(data.get("_httk_atomistic_block_name", "<unnamed>"))


def _declared_symmetry(data: Mapping[str, Any]) -> str | None:
    """The Hall symbol or International Tables number enforced for the CIF."""
    hall = data.get("space_group_name_hall")
    if hall:
        return f"Hall symbol {str(hall).strip()!r}"
    number = data.get("space_group_nbr")
    if number is not None:
        return f"International Tables number {str(number).strip()!r}"
    return None


def cif_setting(data: Mapping[str, Any], *, trust_declared_symmetry: bool = True) -> Spacegroup:
    """The space-group setting a CIF block is written in.

    The setting is identified from the file's symmetry **operations**, by exact set
    comparison against the tabulated settings. That is what makes a file written in a
    non-standard setting come out as itself rather than being silently reinterpreted.

    What the file *declares* — a Hall symbol, an International Tables number, or a recognized
    H-M symbol — is treated as a claim to be checked, not a hint to be taken or dropped. A
    declaration that names no known setting, or that names one whose operations are not the
    file's, is a genuine inconsistency in the file and raises rather than being worked around:
    the two halves of the file disagree, and quietly believing one of them is how a wrong
    structure gets built. An unrecognized H-M spelling is the exception and remains ignored.

    Pass ``trust_declared_symmetry=False`` to ignore the declaration entirely and identify
    the setting from the operations alone. That is the escape hatch for a file whose symbols
    are known to be wrong but whose operations are good.

    A Hermann-Mauguin symbol is consulted, when neither a Hall symbol nor an International
    Tables number is declared, only if its normalized spelling is recognized. A recognized
    symbol narrows the candidate IT number; the operations still identify the exact setting
    and a contradiction fails like a contradicting IT-number declaration. Unrecognized H-M
    spellings are ignored for compatibility with the previous operations-only behavior.

    Raises :class:`ValueError` when the block states no operations, when a declaration is
    inconsistent with them, or when the operations match no tabulated setting at all. In the
    last case the transform to the standard setting genuinely cannot be *derived* — infinitely
    many are equally valid and they describe different crystals — so such a file has to be
    built with an explicit :class:`~httk.atomistic.SettingTransform`.

    :param data: The loaded CIF data block.
    :param trust_declared_symmetry: Whether to check the declared Hall, IT, or H-M symbol.
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
    a candidate. Hall and IT declarations that cannot name a real setting raise here; an
    unrecognized H-M spelling is ignored for operations-only compatibility.
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
                f"the setting from the symmetry operations alone. Remedy: load(..., repair=True) "
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

    hm = data.get("space_group_name_hm")
    if hm:
        written = str(hm).strip()
        hm_number = _hm_it_numbers().get(_normalized_hm(written))
        if hm_number is not None:
            narrowed = [record for record in symmetry_data.spacegroup_settings() if record["it_number"] == hm_number]
            return narrowed, f"Hermann-Mauguin symbol {written!r}"

    return None, None


def _normalized_hall(symbol: str) -> str:
    """A Hall symbol in the spelling the tables index it under.

    A CIF writes Hall symbols conventionally — ``-C 2yc`` — while the tables key them as
    ``-c_2yc``. Lower-casing and turning spaces into underscores reproduces the tabulated
    spelling for all 527 settings, which ``tests/test_symmetry_data.py`` checks. Without this
    step every correctly declared Hall symbol looks unknown.
    """
    return symbol.lower().replace(" ", "_")


_HM_IT_NUMBERS: dict[str, int] | None = None


def _normalized_hm(symbol: str) -> str:
    """Normalize compact and spaced Hermann-Mauguin spellings to one lookup key."""
    return "".join(symbol.casefold().replace("_", "").split())


def _hm_it_numbers() -> dict[str, int]:
    """Return the unambiguous normalized H-M-to-IT lookup, built on first use."""
    global _HM_IT_NUMBERS
    if _HM_IT_NUMBERS is None:
        by_symbol: dict[str, set[int]] = {}
        for record in symmetry_data.spacegroup_settings():
            by_symbol.setdefault(_normalized_hm(record["hm_entry"]), set()).add(record["it_number"])
        _HM_IT_NUMBERS = {symbol: next(iter(numbers)) for symbol, numbers in by_symbol.items() if len(numbers) == 1}
    return _HM_IT_NUMBERS


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
    """Return the chemical symbol and charge represented by a CIF type symbol.

    :param symbol: The raw ``_atom_site_type_symbol`` value.
    :return: The mapped chemical symbol and its explicit charge, if stated.
    """
    decoded = _decode_type_symbol(symbol, None)
    return decoded.chemical_symbol, decoded.charge


def _type_symbol_parts(symbol: str) -> tuple[str, fractions.Fraction | None]:
    """Remove a conventional charge suffix while retaining arbitrary CIF labels."""
    suffix = _TYPE_SYMBOL_SUFFIX_CHARGE.fullmatch(symbol)
    if suffix is not None:
        magnitude = fractions.Fraction(int(suffix.group("magnitude") or 1))
        return suffix.group("label"), magnitude if suffix.group("sign") == "+" else -magnitude
    prefix = _TYPE_SYMBOL_PREFIX_CHARGE.fullmatch(symbol)
    if prefix is not None:
        magnitude = fractions.Fraction(int(prefix.group("magnitude")))
        return prefix.group("label"), magnitude if prefix.group("sign") == "+" else -magnitude
    if len(symbol) > 1 and symbol.endswith("0"):
        return symbol[:-1], fractions.Fraction(0)
    return symbol, None


def _decode_type_symbol(symbol: str, stated_mass: float | None) -> _DecodedCIFType:
    """Interpret CIF core symbols, isotopes, pseudo-sites, and arbitrary labels."""
    raw = symbol.strip()
    label, charge = _type_symbol_parts(raw)
    if label == "TL":
        label = "Tl"

    if label == "D":
        return _DecodedCIFType("H", charge, "D", 2.008 if stated_mass is None else stated_mass, True)
    if label == "T":
        return _DecodedCIFType("H", charge, "T", 3.0160 if stated_mass is None else stated_mass, True)
    if label == "X":
        return _DecodedCIFType("X", charge, None, stated_mass, True)
    if label in {"Vac", "Va", "vacancy"}:
        return _DecodedCIFType("vacancy", charge, None, 0.0, True)

    neutral_core_symbol = charge == 0 and label in _CIF_CORE_TYPE_SYMBOLS
    if (raw in _CIF_CORE_TYPE_SYMBOLS or neutral_core_symbol) and label in SYMBOLS:
        return _DecodedCIFType(label, charge, None, stated_mass, True)
    return _DecodedCIFType("X", charge, label, stated_mass, False)


def _read_cif_for_atomistic(
    source: Any, *, allow_large_cif_uncertainty: bool = False, repair: bool = False
) -> Mapping[str, Any]:
    """Read CIF and carry the atomistic override to its adapter."""
    from httk.atomistic.io.cif import read_cif
    from httk.atomistic.io.cif.cif_parser import cifblock_to_asu

    if repair:
        raw_blocks, header = read_cif(source, allow_cif2=False, repair=True, structural_only=True)
    else:
        raw_blocks, header = read_cif(source, allow_cif2=False, structural_only=True)
    blocks = []
    unparsed = []
    for name, raw_block in raw_blocks:
        if (
            "atom_site_symmetry_multiplicity" in raw_block
            and "atom_site_site_symmetry_multiplicity" not in raw_block
            and "atom_site_site_symmetry_order" not in raw_block
        ):
            logging.getLogger(__name__).info(
                f"CIF block {name!r}: deprecated data name _atom_site_symmetry_multiplicity was ignored; "
                "it is deprecated by the CIF core dictionary, and legacy values are ambiguous between "
                "IT multiplicities and site-symmetry orders",
                extra={"context": "cif"},
            )
        if "atom_site_label" not in raw_block:
            continue
        try:
            block = cifblock_to_asu(raw_block)
        except Exception as error:
            unparsed.append({"block": name, "reason": f"{type(error).__name__}: {error}"})
        else:
            position_precisions, position_snap_bounds = _position_precision_metadata(raw_block)
            blocks.append(
                {
                    **block,
                    "position_precisions": position_precisions,
                    "position_snap_bounds": position_snap_bounds,
                    "position_tokens": _position_tokens(raw_block),
                    "_httk_atomistic_wyckoff_labels": raw_block.get("atom_site_wyckoff_label"),
                    "_httk_atomistic_symmetry_multiplicities": raw_block.get("atom_site_site_symmetry_multiplicity"),
                    "_httk_atomistic_site_symmetry_orders": raw_block.get("atom_site_site_symmetry_order"),
                    **({"_httk_atomistic_block_name": name} if repair else {}),
                }
            )
    payload: dict[str, Any] = {"format": "cif", "blocks": blocks, "unparsed": unparsed, "header": header}
    if repair:
        payload["repair"] = True
    if not allow_large_cif_uncertainty:
        return payload
    return {**payload, _ALLOW_LARGE_CIF_UNCERTAINTY: True}


def _position_precision_metadata(
    block: Mapping[str, Any],
) -> tuple[
    list[tuple[fractions.Fraction | None, ...]],
    list[tuple[fractions.Fraction | None, ...]],
]:
    """Preserve per-component uncertainty and snapping bounds for the reader bridge."""
    from httk.core import combined_precision

    from httk.atomistic.io.cif.cif_parser import cif_exact_token, parse_cif_float

    columns = [block[f"atom_site_fract_{axis}"] for axis in "xyz"]
    companions = [block.get(f"httk_atom_site_fract_{axis}_exact") for axis in "xyz"]
    has_companion = any(value is not None for value in companions)
    precisions = []
    snap_bounds = []
    for index, values in enumerate(zip(*columns)):
        row: list[fractions.Fraction | None] = []
        bounds_row: list[fractions.Fraction | None] = []
        for axis, value in enumerate(values):
            companion = companions[axis]
            companion_value = companion[index] if isinstance(companion, list) and index < len(companion) else None
            if (companion_value is not None and cif_exact_token(companion_value) is not None) or (
                has_companion and cif_exact_token(value) in {"0", "1"}
            ):
                row.append(None)
                bounds_row.append(fractions.Fraction())
            else:
                meta = parse_cif_float(value, meta=True)[1]
                row.append(combined_precision((meta["precision"], meta["esd"])))
                digit_bound = None if meta["precision"] is None else meta["precision"] / 2
                bound = combined_precision((digit_bound, meta["esd"]))
                bounds_row.append(None if bound is None or bound >= fractions.Fraction(1, 2) else bound)
        precisions.append(tuple(row))
        snap_bounds.append(tuple(bounds_row))
    return precisions, snap_bounds


def _position_tokens(block: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Preserve raw CIF coordinate tokens for precise uncertainty diagnostics."""
    return list(zip(*(block[f"atom_site_fract_{axis}"] for axis in "xyz")))


def _parameters_inside_coordinate_bounds(
    branch: Any,
    own_point: FracVector,
    transform: SettingTransform,
    bounds: Sequence[fractions.Fraction | None],
) -> FracVector | None:
    """Return branch parameters whose coordinate lies inside the periodic component box."""
    rank = len(branch.free)
    if rank not in (1, 2):
        return None

    zero = [fractions.Fraction()] * rank
    base = transform.to_setting(branch.coordinate(zero)).to_fractions()
    columns = []
    for parameter in range(rank):
        basis_parameters = zero.copy()
        basis_parameters[parameter] = fractions.Fraction(1)
        point = transform.to_setting(branch.coordinate(basis_parameters)).to_fractions()
        columns.append(tuple(value - origin for value, origin in zip(point, base)))
    coefficients = tuple(tuple(column[row] for column in columns) for row in range(3))
    own = own_point.to_fractions()

    constrained = []
    for row, bound in enumerate(bounds):
        if bound is None:
            continue
        coefficient = coefficients[row]
        offset = base[row] - own[row]
        minimum = offset + sum((min(value, 0) for value in coefficient), start=fractions.Fraction())
        maximum = offset + sum((max(value, 0) for value in coefficient), start=fractions.Fraction())
        integers = range(math.ceil(minimum - bound), math.floor(maximum + bound) + 1)
        constrained.append((coefficient, offset, bound, integers))

    for lattice_shifts in itertools.product(*(item[3] for item in constrained)):
        inequalities: list[tuple[tuple[fractions.Fraction, ...], fractions.Fraction]] = []
        for (coefficient, offset, bound, _), shift in zip(constrained, lattice_shifts):
            inequalities.append((coefficient, fractions.Fraction(shift) - offset + bound))
            inequalities.append((tuple(-value for value in coefficient), offset - shift + bound))
        for parameter in range(rank):
            axis = tuple(fractions.Fraction(index == parameter) for index in range(rank))
            inequalities.append((axis, fractions.Fraction(1)))
            inequalities.append((tuple(-value for value in axis), fractions.Fraction()))

        for active in itertools.combinations(inequalities, rank):
            solution: tuple[fractions.Fraction, ...]
            if rank == 1:
                pivot = active[0][0][0]
                if pivot == 0:
                    continue
                solution = (active[0][1] / pivot,)
            else:
                (a, b), (c, d) = (item[0] for item in active)
                first, second = (item[1] for item in active)
                determinant = a * d - b * c
                if determinant == 0:
                    continue
                solution = ((first * d - b * second) / determinant, (a * second - first * c) / determinant)
            if all(
                sum((value * parameter for value, parameter in zip(row, solution)), start=fractions.Fraction()) <= limit
                for row, limit in inequalities
            ):
                return FracVector(solution).normalize()
    return None


def _snap(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Cell,
    transform: Any,
    tolerance: float,
    *,
    uncertainty: tuple[Any, str] | None = None,
    coordinate_bounds: Sequence[fractions.Fraction | None] | None = None,
    allow_large_cif_uncertainty: bool = False,
    most_specific: bool = False,
    positions: Sequence[Any] | None = None,
    orbit_screen: list[tuple[tuple[float, float, float], Any, FracVector]] | None = None,
    general_screen: _GeneralPositionScreen | None = None,
) -> tuple[str, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance``, and its free parameters.

    Floating point screens branch candidates at twice the tolerance, but exact distance and
    per-component coordinate bounds still decide every result. ``positions`` limits the search
    to authoritative CIF declarations; otherwise every standard position is tried in its established order.
    """
    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    identity = transform.is_identity()

    def inside_coordinate_bounds(candidate: FracVector) -> bool:
        if coordinate_bounds is None:
            return True
        return all(
            bound is None or abs((own - snapped + fractions.Fraction(1, 2)) % 1 - fractions.Fraction(1, 2)) <= bound
            for own, snapped, bound in zip(own_point.to_fractions(), candidate.to_fractions(), coordinate_bounds)
        )

    def accepted_parameters(branch: Any, parameters: FracVector, *, general: bool) -> FracVector | None:
        candidate = branch.coordinate(parameters)
        if not identity:
            candidate = transform.to_setting(candidate)
        if not inside_coordinate_bounds(candidate) and coordinate_bounds is not None:
            bounded_parameters = _parameters_inside_coordinate_bounds(branch, own_point, transform, coordinate_bounds)
            if bounded_parameters is None:
                return None
            parameters = bounded_parameters
            candidate = branch.coordinate(bounded_parameters)
            if not identity:
                candidate = transform.to_setting(candidate)
        if _cartesian_distance_squared(own_point - candidate, cell) > limit:
            return None
        if general or candidate.normalize() != own_point.normalize():
            reject_large_uncertainty()
        return parameters

    def finish(letter: str, parameters: FracVector) -> tuple[str, FracVector]:
        if orbit_screen is not None:
            orbit_screen.clear()
            position = standard.wyckoff_position(letter)
            values = tuple(parameters.to_floats())
            matrix = None if identity else transform.matrix.to_floats()
            vector = None if identity else transform.vector.to_floats()
            cosets = tuple((coset, tuple(coset.to_floats())) for coset in transform.lattice_cosets())
            common = [
                *values,
                *(() if matrix is None else (value for row in matrix for value in row)),
                *(() if vector is None else vector),
            ]
            for branch in position.branches:
                candidate = branch.coordinate_float(values)
                if identity:
                    own = candidate
                else:
                    assert matrix is not None and vector is not None
                    own = tuple(
                        sum(matrix[row][column] * candidate[column] for column in range(3)) + vector[row]
                        for row in range(3)
                    )
                for coset, shift in cosets:
                    # Do not let a lossy large-coordinate modulo prune an exact orbit pair.
                    safe = _float_screen_slack([*common, *candidate, *own, *shift]) is not None
                    orbit_screen.append(
                        (
                            tuple((value + delta) % 1.0 for value, delta in zip(own, shift))
                            if safe
                            else (math.nan, math.nan, math.nan),
                            branch,
                            coset,
                        )
                    )
        return letter, parameters

    exact_first = coordinate_bounds is None and positions is None and not most_specific

    def reject_large_uncertainty() -> None:
        if uncertainty is not None:
            projected, token = uncertainty
            if projected >= CIF_POSITIONAL_UNCERTAINTY_ERROR**2 and not allow_large_cif_uncertainty:
                raise ValueError(
                    f"CIF coordinate token {token!r} implies a projected positional uncertainty of "
                    f"{math.sqrt(projected.to_float()):.6g} Å; pass allow_large_cif_uncertainty=True to override"
                )

    if not exact_first and coordinate_bounds is None:
        reject_large_uncertainty()
    elif general_screen is not None and _definitely_general(own_point, general_screen):
        reject_large_uncertainty()
        general = standard.wyckoff[-1]
        parameters = general.representative.parameters_of(standard_point)
        assert general.free_count == 3 and parameters is not None
        return finish(general.letter, parameters)

    limit = tolerance * tolerance
    standard_float: tuple[float, ...] | None = None
    float_geometry: tuple[Any, tuple[float, ...], Any | None, tuple[float, ...] | None] | None = None
    screen_values: list[float] = []
    try:
        basis = cell.basis.to_floats()
        point = tuple(own_point.to_floats())
        standard_float = tuple(standard_point.to_floats())
        matrix = None if identity else transform.matrix.to_floats()
        vector = None if identity else transform.vector.to_floats()
    except OverflowError:
        screen = None
    else:
        screen_values = [
            *point,
            *standard_float,
            *(value for row in basis for value in row),
            *(() if matrix is None else (value for row in matrix for value in row)),
            *(() if vector is None else vector),
        ]
        float_geometry = basis, point, matrix, vector
        screen = None if _float_screen_slack(screen_values) is None else (tolerance * 2 + 1e-9) ** 2
    candidates = sorted(
        standard.wyckoff if positions is None else positions, key=lambda item: (item.multiplicity, item.letter)
    )
    deferred: list[tuple[Any, Any]] = []
    matches: list[tuple[int, str, FracVector]] = []
    for position in candidates:
        if exact_first and position.free_count == 3:
            reject_large_uncertainty()
            for deferred_position, deferred_branch in deferred:
                parameters = deferred_branch.nearest_parameters(standard_point)
                parameters = accepted_parameters(deferred_branch, parameters, general=False)
                if parameters is not None:
                    return finish(deferred_position.letter, parameters)
            exact_first = False
        for branch in position.branches:
            projection = (
                None
                if standard_float is None or (not exact_first and screen is None)
                else branch.nearest_parameters_float(standard_float)
            )
            projected = None if projection is None else branch.coordinate_float(projection)
            exact_candidate = projected is None
            if standard_float is not None and projected is not None:
                exact_candidate = _float_screen_slack([*standard_float, *projected]) is None or all(
                    abs((first - second + 0.5) % 1.0 - 0.5) <= 1e-13 for first, second in zip(standard_float, projected)
                )
            if exact_first and position.free_count != 3 and exact_candidate:
                parameters = branch.parameters_of(standard_point)
                if parameters is not None:
                    return finish(position.letter, parameters)
            if screen is not None and projected is not None:
                assert float_geometry is not None
                basis, point, matrix, vector = float_geometry
                if identity:
                    own_candidate = projected
                else:
                    assert matrix is not None and vector is not None
                    own_candidate = tuple(
                        sum(matrix[row][column] * projected[column] for column in range(3)) + vector[row]
                        for row in range(3)
                    )
                if _float_screen_slack([*screen_values, *projected, *own_candidate]) is not None:
                    difference = tuple((first - second + 0.5) % 1 - 0.5 for first, second in zip(point, own_candidate))
                    cartesian = tuple(
                        sum(difference[row] * basis[row][column] for row in range(3)) for column in range(3)
                    )
                    if sum(value * value for value in cartesian) > screen:
                        continue
            if exact_first:
                deferred.append((position, branch))
                continue
            parameters = branch.nearest_parameters(standard_point)
            parameters = accepted_parameters(branch, parameters, general=position.free_count == 3)
            if parameters is not None:
                if not most_specific:
                    return finish(position.letter, parameters)
                matches.append((position.multiplicity, position.letter, parameters))
                break
    if exact_first:
        reject_large_uncertainty()
        for position, branch in deferred:
            parameters = branch.nearest_parameters(standard_point)
            parameters = accepted_parameters(branch, parameters, general=False)
            if parameters is not None:
                return finish(position.letter, parameters)
    if not matches:
        return None
    _, letter, parameters = min(matches)
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

    identity = transform.is_identity()
    return (
        min(
            _cartesian_distance_squared(
                own_point
                - (
                    branch.coordinate(branch.nearest_parameters(standard_point))
                    if identity
                    else transform.to_setting(branch.coordinate(branch.nearest_parameters(standard_point)))
                ),
                cell,
            )
            for branch in position.branches
        )
        ** 0.5
    )

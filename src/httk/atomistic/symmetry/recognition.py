"""Recognizing the asymmetric unit of a full structure.

This is the direction where tolerance lives, and the only one. A measured or file-read
structure gives coordinates like ``0.3333`` that lie *near* a symmetric arrangement rather
than on one, so recognizing it means deciding which Wyckoff position each site is meant to
occupy. Once decided, the position supplies exact values for its fixed components and
everything downstream — expansion, comparison, round-tripping — is exact again.

The contract, stated once so the asymmetry is not a surprise:

* Expansion (:meth:`~httk.atomistic.FundamentalDomainStructure.expand_sites`) is **lossless**.
* Recognition is **lossy at the tolerance level**: it snaps a measured structure
  onto an idealised symmetric one.
* Therefore ``expand -> recognize -> expand`` is idempotent, while
  ``recognize -> expand`` is not the identity on the input coordinates and must not be
  advertised as such.

A file's decimals are embedded as the exact rational they literally say — ``0.3333`` is
``3333/10000``, never silently ``1/3``. Free parameters keep that value; only the fixed
components of the chosen position are replaced. Idealising the free parameters too is
available on request through ``limit_denominator``, but it is not the default, because
inventing ``1/3`` from a file that said ``0.3333`` is a claim about the data that only the
caller can make.
"""

import fractions
import math
from collections.abc import Sequence
from typing import Any

from httk.core import FracVector, SurdVector, register_citation

from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.symmetry._periodic_wrap import wrap_periodic_half
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.wyckoff import WyckoffBranch, WyckoffPosition

__all__ = ["DEFAULT_TOLERANCE", "recognize_asu", "structure_tolerance"]

#: Fallback matching tolerance, as a Cartesian distance in the cell's own length units
#: (angstrom for ordinary crystallographic data), used only when the structure does not say
#: how precisely it was stated. Comfortably larger than the rounding in a CIF written to
#: four decimals, and far smaller than any real interatomic distance.
DEFAULT_TOLERANCE = 1e-3

#: How much room a tolerance is given beyond the stated precision. Two coordinates that
#: should be equal are each rounded independently, so they can differ by twice the
#: precision; a tolerance below that would reject data that is in fact consistent.
_SAFETY_FACTOR = 2

#: A derived tolerance is only checked against the interatomic distances when it is large
#: enough relative to the cell for merging to be a real risk. That check is quadratic in the
#: number of sites, and for ordinary well-written data the tolerance is thousands of times
#: smaller than any bond, so paying for it every time would be waste.
_CAP_TRIGGER = 0.05


def structure_tolerance(structure: StructureLike, *, fallback: float = DEFAULT_TOLERANCE) -> float:
    """Derive a matching tolerance from how precisely the structure was stated.

    This is the point of recording precision at all: instead of a constant somebody
    guessed, the tolerance follows the data. Coordinates written to four decimals in a 5 A
    cell are good to about ``5e-4``, and the tolerance comes out near ``1e-3``; the same
    coordinates in a 30 A cell justify a tolerance six times larger, and coordinates written
    to two decimals justify one a hundred times larger.

    Returns ``fallback`` when the structure does not state a precision — a structure built
    by hand, or read from a format that does not write its numbers to a definite number of
    digits. A caller that needs to know whether that happened can compare the result against
    the structure's own ``cartesian_precision()``.

    The value is capped so that it can never reach half the smallest distance between two
    sites, which is what would let genuinely distinct atoms be merged. Minimum separation
    bounds a tolerance from above; treating it as precision could make a structure with
    accidentally close atoms look far more precisely stated than it is.

    :param structure: The structure whose stated precision determines the tolerance.
    :param fallback: The tolerance to use when the structure has no stated precision.
    :return: The Cartesian matching tolerance in the structure's cell units.
    """
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    view = UnitcellStructureView(structure)
    precision = view.cartesian_precision()
    if precision is None:
        return fallback

    tolerance = float(precision) * _SAFETY_FACTOR
    shortest_edge = min(length.to_float() for length in view.cell.lengths)
    if tolerance <= _CAP_TRIGGER * shortest_edge:
        return tolerance

    cap = _half_minimum_separation(view)
    # Leave a small relative margin as well as stepping below the float boundary. A
    # one-ULP step alone can be lost when the later Cartesian squared-distance path
    # recomputes the same separation through matrix arithmetic.
    strict_cap = None if cap is None else math.nextafter(cap * (1 - 1e-12), 0.0)
    return tolerance if strict_cap is None or tolerance < strict_cap else strict_cap


def _half_minimum_separation(view: Any) -> float | None:
    """Half the shortest distance between two distinct sites, or ``None`` if there is one site.

    Computed in floating point over the nearest-image difference of each pair: this bounds a
    tolerance, so a fast approximate answer is the right kind of answer. The nearest-image
    reduction is per component, which is exact for a cell with orthogonal axes and can
    overestimate for a strongly oblique one — erring towards a looser cap rather than a
    tighter one.

    Only the periodic directions are reduced. Along a non-periodic one there is no other
    image to be nearer, and folding it would report two well-separated atoms as close
    neighbours — tightening the cap, and so the derived tolerance, below what the data
    justifies.
    """
    coords = view.sites.reduced_coords
    count = len(coords)
    if count < 2:
        return None

    periodicity = view.cell.periodicity
    basis = view.cell.basis.to_floats()
    shortest: float | None = None
    for first in range(count):
        for second in range(first + 1, count):
            difference = wrap_periodic_half(coords[first] - coords[second], periodicity).to_floats()
            cartesian = [sum(difference[axis] * basis[axis][component] for axis in range(3)) for component in range(3)]
            distance = math.sqrt(sum(value * value for value in cartesian))
            if distance > 0 and (shortest is None or distance < shortest):
                shortest = distance
    return None if shortest is None else shortest / 2


def recognize_asu(
    structure: StructureLike,
    *,
    setting: Spacegroup | None = None,
    standard: Spacegroup | None = None,
    transform: SettingTransform | None = None,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
    _retain_found_transform: bool = False,
) -> ASUStructure:
    """Build an :class:`~httk.atomistic.ASUStructure` from a full structure.

    The space group can be supplied three ways, in decreasing order of preference:

    * ``setting`` — the structure's own tabulated setting, as when a CIF names its group.
      Nothing is searched for and spglib is not involved.
    * ``standard`` together with ``transform`` — for a structure in a setting that appears
      in no table. Also spglib-free.
    * neither — the symmetry is found with spglib, which must be installed
      (``pip install httk-atomistic[default]``).

    ``tolerance`` is a Cartesian distance, measured in the real cell, so it means the same
    thing along a short axis and a long one; a fractional tolerance would not. Left
    unspecified it is **derived from how precisely the structure was stated** — see
    :func:`structure_tolerance` — falling back to :data:`DEFAULT_TOLERANCE` for a structure
    that does not say. Pass a value to override that.

    Raises :class:`ValueError` if a site cannot be placed on any Wyckoff position within
    the tolerance, or if the sites do not group into complete orbits — both of which mean
    the structure does not actually have the symmetry it was said to have.

    :param structure: The full structure to recognize.
    :param setting: The structure's own tabulated setting, if known.
    :param standard: The IT standard setting for an untabulated own setting.
    :param transform: The stored standard-to-own transform for an untabulated setting.
    :param tolerance: The Cartesian matching tolerance in the real cell, or ``None`` to
        derive it from the structure's stated precision.
    :param limit_denominator: The largest denominator allowed when idealising free
        parameters, or ``None`` to retain their exact stated values.
    :param _retain_found_transform: Internal canonicalization hook retaining spglib's
        recognized-to-standard transform instead of folding it into the returned basis.
    :return: The recognized asymmetric-unit structure.
    :raises ImportError: If symmetry must be searched and the optional spglib dependency is
        unavailable.
    :raises TypeError: If the supplied setting arguments are incomplete or mutually
        exclusive.
    :raises ValueError: If the structure is not fully periodic, the standard setting is
        invalid, or the sites cannot be placed into complete Wyckoff orbits within the
        tolerance.
    """
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    view = UnitcellStructureView(structure)
    require_full_periodicity(view.cell, "recognize_asu")
    if tolerance is None:
        tolerance = structure_tolerance(view)

    wyckoff_hints: tuple[str, ...] | None = None
    equivalent_hints: tuple[int, ...] | None = None
    if setting is not None:
        if standard is not None or transform is not None:
            raise TypeError("recognize_asu() takes either 'setting' or 'standard'/'transform', not both")
        standard = setting
        transform = SettingTransform.identity()
    elif standard is not None or transform is not None:
        if standard is None or transform is None:
            raise TypeError("recognize_asu() needs both 'standard' and 'transform' when either is given")
        if not standard.is_standard_setting:
            raise ValueError(f"'standard' must be an IT standard setting, got {standard.setting}")
    else:
        standard, found, wyckoff_hints, equivalent_hints = _find_symmetry(view, tolerance)
        # Public recognition keeps its established preference for an already-standard input frame.
        # ``canonical_asu`` privately reverses the attempts: retaining spglib's standardizing frame
        # prevents arbitrary index-one shears from reaching expensive exact metric arithmetic.  Both
        # candidates still pass the same exact-table reconstruction, and the other remains a fallback.
        if abs(found.determinant()) == 1:
            attempts = (
                (found, SettingTransform.identity())
                if _retain_found_transform
                else (SettingTransform.identity(), found)
            )
            first_error: ValueError | None = None
            for attempt in attempts:
                try:
                    return _recognize(
                        view,
                        standard,
                        attempt,
                        tolerance,
                        limit_denominator,
                        wyckoff_hints=wyckoff_hints,
                        equivalent_hints=equivalent_hints,
                    )
                except ValueError as error:
                    if first_error is None:
                        first_error = error
            assert first_error is not None
            raise first_error
        transform = found

    return _recognize(
        view,
        standard,
        transform,
        tolerance,
        limit_denominator,
        wyckoff_hints=wyckoff_hints,
        equivalent_hints=equivalent_hints,
    )


def _recognize(
    view: Any,
    standard: Spacegroup,
    transform: SettingTransform,
    tolerance: float,
    limit_denominator: int | None,
    *,
    wyckoff_hints: tuple[str, ...] | None = None,
    equivalent_hints: tuple[int, ...] | None = None,
) -> ASUStructure:
    """Recognize through exact tables, using validated spglib partitions when available."""
    if wyckoff_hints is not None and equivalent_hints is not None:
        try:
            return _recognize_impl(
                view,
                standard,
                transform,
                tolerance,
                limit_denominator,
                wyckoff_hints=wyckoff_hints,
                equivalent_hints=equivalent_hints,
            )
        except ValueError:
            # Hints accelerate the common spglib path but never change recognition semantics.  If
            # their setting/gauge disagrees with the exact tables, repeat the established exhaustive
            # placement and grouping and let that result (or its error) decide.
            pass
    return _recognize_impl(view, standard, transform, tolerance, limit_denominator)


def _recognize_impl(
    view: Any,
    standard: Spacegroup,
    transform: SettingTransform,
    tolerance: float,
    limit_denominator: int | None,
    *,
    wyckoff_hints: tuple[str, ...] | None = None,
    equivalent_hints: tuple[int, ...] | None = None,
) -> ASUStructure:
    """Place every site on a Wyckoff position, then group the sites into orbits."""
    cell = view.cell
    own_coords = view.sites.reduced_coords.normalize()
    species_at_sites = tuple(view.species_at_sites)
    count = len(species_at_sites)

    placed: list[tuple[WyckoffPosition, FracVector, FracVector]] = []
    representatives: dict[int, tuple[WyckoffPosition, FracVector]] = {}
    for index in range(count):
        own_point = own_coords[index]
        equivalent = None if equivalent_hints is None else equivalent_hints[index]
        if equivalent is not None and equivalent in representatives:
            position, parameters = representatives[equivalent]
            # The representative fixed the exact orbit and its free parameters.  Keep this member's
            # measured coordinate for the later tolerance/bijection validation instead of repeating
            # the full Wyckoff-branch scan for every atom in the same spglib equivalence class.
            placed.append((position, parameters, own_point))
            continue
        standard_point = transform.to_standard(own_point).normalize()
        hint = None if wyckoff_hints is None else wyckoff_hints[index]
        match = _place_on_position(
            standard,
            standard_point,
            own_point,
            cell,
            transform,
            tolerance,
            wyckoff_hint=hint,
        )
        if match is None:
            raise ValueError(
                f"site {index} at {tuple(own_point.to_fractions())} does not lie on any Wyckoff position of "
                f"{standard.setting} within a tolerance of {tolerance}; the structure does not have "
                f"the symmetry it was given"
            )
        position, branch, parameters, coset = match
        if limit_denominator is not None and position.free_count:
            parameters = FracVector([value.limit_denominator(limit_denominator) for value in parameters.to_fractions()])
        snapped_own = (transform.to_setting(branch.coordinate(parameters)) + coset).normalize()
        placed.append((position, parameters, snapped_own))
        if equivalent is not None:
            representatives[equivalent] = (position, parameters)

    return _group_into_orbits(
        view,
        standard,
        transform,
        placed,
        species_at_sites,
        cell,
        tolerance,
        equivalent_hints=equivalent_hints,
    )


def _place_on_position(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Any,
    transform: SettingTransform,
    tolerance: float,
    *,
    wyckoff_hint: str | None = None,
) -> tuple[WyckoffPosition, WyckoffBranch, FracVector, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance`` of a site.

    Returns the position, the orbit branch the site sits on, that branch's free
    parameters, and the matched setting-lattice coset. Positions are tried most specific first, so a site a hair away from a
    special position is recognized as being on it rather than as a general-position site
    that happens to sit there. The general position matches everything with zero
    displacement, so the walk always terminates for a coordinate inside the cell.

    The parameters are the matched *branch's*, not the representative branch's, and are
    not translated between the two. They do not need to be: evaluating the whole position
    at those parameters regenerates the orbit that contains this site either way, and
    grouping is done by orbit membership rather than by comparing parameters.
    """
    tolerance_squared = tolerance * tolerance
    cosets = transform.lattice_cosets()
    positions = standard.wyckoff
    if wyckoff_hint is not None:
        try:
            positions = (standard.wyckoff_position(wyckoff_hint),)
        except KeyError:
            pass
    for position in positions:
        for branch in position.branches:
            parameters = branch.nearest_parameters(standard_point)
            candidate = transform.to_setting(branch.coordinate(parameters))
            for coset in cosets:
                if _cartesian_distance_squared(own_point - candidate - coset, cell) <= tolerance_squared:
                    return position, branch, parameters, coset
    if wyckoff_hint is not None and positions is not standard.wyckoff:
        # Spglib's letter is an acceleration hint, never an authority.  A setting bridge or a
        # tolerance-boundary decision can make it unsuitable for httk's exact table; retry the full
        # table so recognition retains its existing acceptance semantics.
        return _place_on_position(
            standard,
            standard_point,
            own_point,
            cell,
            transform,
            tolerance,
        )
    return None


def _cartesian_distance_squared(difference: FracVector, cell: Any) -> float:
    """The squared length of a fractional difference, in the cell's real geometry.

    Reduced to the shortest lattice representative first, so a site at ``0.999`` and one at
    ``0.001`` count as neighbours rather than as a whole cell apart.
    """
    shortest = SurdVector(difference.normalize_half()) * cell.basis
    return float(shortest.lengthsqr().to_float())


def _orbit_has_bijective_members(
    member_indices: Sequence[int],
    placed: Sequence[tuple[WyckoffPosition, FracVector, FracVector]],
    orbit: Sequence[FracVector],
    cell: Any,
    tolerance: float,
) -> bool:
    """Whether the candidate input sites occupy every generated orbit point once."""
    limit = tolerance * tolerance
    basis = cell.basis.to_floats()
    orbit_floats = [[float(value) for value in point.to_fractions()] for point in orbit]
    member_floats = {member: [float(value) for value in placed[member][2].to_fractions()] for member in member_indices}
    candidates = {
        member: tuple(
            index
            for index, point in enumerate(orbit_floats)
            if _fractional_distance_squared(member_floats[member], point, basis) <= limit
        )
        for member in member_indices
    }
    if any(not choices for choices in candidates.values()):
        return False
    matched_orbit: dict[int, int] = {}

    def augment(member: int, visited: set[int]) -> bool:
        for orbit_index in candidates[member]:
            if orbit_index in visited:
                continue
            visited.add(orbit_index)
            previous = matched_orbit.get(orbit_index)
            if previous is None or augment(previous, visited):
                matched_orbit[orbit_index] = member
                return True
        return False

    return all(augment(member, set()) for member in member_indices)


def _fractional_distance_squared(
    first: Sequence[float],
    second: Sequence[float],
    basis: Sequence[Sequence[float]],
) -> float:
    """Minimum-image Cartesian squared distance for two fractional float coordinates."""
    wrapped = [first[index] - second[index] - round(first[index] - second[index]) for index in range(3)]
    cartesian = [sum(wrapped[index] * basis[index][axis] for index in range(3)) for axis in range(3)]
    return sum(component * component for component in cartesian)


def _group_into_orbits(
    view: Any,
    standard: Spacegroup,
    transform: SettingTransform,
    placed: Sequence[tuple[WyckoffPosition, FracVector, FracVector]],
    species_at_sites: tuple[str, ...],
    cell: Any,
    tolerance: float,
    *,
    equivalent_hints: tuple[int, ...] | None = None,
) -> ASUStructure:
    """Collapse symmetry-equivalent sites, keeping one representative each.

    Grouping is by *orbit membership*, not by comparing recovered parameters: the orbit a
    site belongs to is generated from one member and the remaining sites are matched
    against it. That also verifies as it goes — an orbit that is not fully occupied, or is
    occupied by more than one species, means the claimed symmetry is wrong, and says so
    instead of quietly producing a structure with missing atoms.

    Membership is tested within the tolerance rather than exactly. It has to be: in a
    measured structure every member of an orbit carries its own rounding, so each recovers
    a slightly different free parameter and generates a slightly different orbit. Exact
    membership would then place each atom in an orbit of its own and report the structure
    as having no symmetry at all. The representative's parameters are the ones kept, so the
    idealised structure adopts one member's value for the whole orbit.
    """
    cosets = transform.lattice_cosets()
    consumed = [False] * len(placed)
    wyckoff_sites: list[WyckoffSite] = []
    site_moments = view.site_moments
    if isinstance(site_moments, CrystalAxisSiteMoments) and site_moments.cell != cell:
        raise ValueError(
            "recognize_asu cannot re-express crystal-axis site moments in the target cell; "
            "convert them to Cartesian first"
        )

    for index, (position, parameters, _snapped) in enumerate(placed):
        if consumed[index]:
            continue
        species = species_at_sites[index]

        # Deduplicated exactly, as expansion does: a transform onto a smaller cell — the
        # rhombohedral settings, where det M == 3 — maps several standard-setting orbit
        # points onto one. Leaving the repeats in would make a complete orbit look
        # three-quarters empty and reject a perfectly symmetric structure.
        orbit: list[FracVector] = []
        seen: set[tuple[fractions.Fraction, ...]] = set()
        for point in position.coordinates(parameters):
            for coset in cosets:
                candidate = (transform.to_setting(point) + coset).normalize()
                key = tuple(candidate.to_fractions())
                if key not in seen:
                    seen.add(key)
                    orbit.append(candidate)

        if equivalent_hints is None:
            members = [
                other
                for other in range(len(placed))
                if not consumed[other] and _lies_in_orbit(placed[other][2], orbit, cell, tolerance)
            ]
        else:
            equivalent = equivalent_hints[index]
            members = [
                other for other in range(len(placed)) if not consumed[other] and equivalent_hints[other] == equivalent
            ]
        occupants = {species_at_sites[other] for other in members}
        if len(occupants) != 1:
            raise ValueError(
                f"the orbit of Wyckoff position {position.multiplicity}{position.letter} is occupied by "
                f"more than one species ({', '.join(sorted(occupants))}); a single orbit cannot be"
            )
        if len(members) != len(orbit):
            raise ValueError(
                f"Wyckoff position {position.multiplicity}{position.letter} generates {len(orbit)} sites "
                f"but only {len(members)} of them are present; the structure is not symmetric under "
                f"{standard.setting} as claimed"
            )
        if not _orbit_has_bijective_members(members, placed, orbit, cell, tolerance):
            raise ValueError(
                f"Wyckoff position {position.multiplicity}{position.letter} generates {len(orbit)} sites, "
                "but the input does not occupy each generated position exactly once"
            )

        for other in members:
            consumed[other] = True
        moment = None if site_moments is None else _orbit_moment(site_moments, members, species, position, cell)
        wyckoff_sites.append(WyckoffSite(position.letter, parameters, species, moment=moment))

    # The recognized ASU inherits the precision of the structure it came from: nothing
    # about recognition sharpens the data, and dropping it here would mean the value had
    # to be guessed again by everything downstream.
    return ASUStructure(
        cell, standard, wyckoff_sites, view.species, transform, view.sites.precision, charge=view.charge
    )


def _orbit_moment(
    site_moments: SiteMomentsBackend,
    members: Sequence[int],
    species: str,
    position: WyckoffPosition,
    cell: Any,
) -> SiteMomentsBackend:
    """Collapse one orbit's exact, uniform structure-level moment into one row."""
    first = members[0]
    values: Any
    row: Any
    uniform: bool
    moment: SiteMomentsBackend
    if isinstance(site_moments, CollinearSiteMoments):
        values = site_moments.collinear_moments
        row = values._element((first,))
        uniform = all(values._element((index,)) == row for index in members[1:])
        moment = CollinearSiteMoments([row], precision=site_moments.precision)
    elif isinstance(site_moments, CartesianSiteMoments):
        values = site_moments.cartesian_moments
        row = [values._element((first, column)) for column in range(3)]
        uniform = all([values._element((index, column)) for column in range(3)] == row for index in members[1:])
        moment = CartesianSiteMoments(SurdVector._from_scalar_grid([row], (1, 3)), precision=site_moments.precision)
    elif isinstance(site_moments, CrystalAxisSiteMoments):
        values = site_moments.crystalaxis_moments
        row = [values._element((first, column)) for column in range(3)]
        uniform = all([values._element((index, column)) for column in range(3)] == row for index in members[1:])
        moment = CrystalAxisSiteMoments(
            SurdVector._from_scalar_grid([row], (1, 3)), cell, precision=site_moments.precision
        )
    else:
        raise TypeError(f"unsupported SiteMomentsBackend kind: {getattr(site_moments, 'kind', None)!r}")
    if not uniform:
        raise ValueError(
            f"orbit of {species} at {position.multiplicity}{position.letter} carries non-uniform site moments; "
            "a fundamental domain cannot represent this magnetic structure — keep the unit cell"
        )
    return moment


def _lies_in_orbit(point: FracVector, orbit: Sequence[FracVector], cell: Any, tolerance: float) -> bool:
    """Whether a site coincides with any point of an orbit, to within the tolerance."""
    limit = tolerance * tolerance
    return any(_cartesian_distance_squared(point - other, cell) <= limit for other in orbit)


def _find_symmetry(
    view: Any,
    tolerance: float,
) -> tuple[Spacegroup, SettingTransform, tuple[str, ...], tuple[int, ...]]:
    """Find the space group of a structure that carries no symmetry information, via spglib.

    This is the only place spglib is used, and the only path that needs it. Anything that
    arrives with its symmetry already stated — a CIF, or a structure built from an ASU —
    goes through the exact tabulated route instead.

    Two conversions matter here and are easy to get wrong:

    * spglib reports its transformation as floats. The matrix is a small crystallographic
      rational, but the origin shift can be fixed by an input atom and consequently carry the
      full precision of the measured coordinates. They are recovered with separate denominator
      bounds so an arbitrary inversion centre is not rounded to a nearby crystallographic fraction.
    * spglib standardizes to *its* default setting, which differs from the International
      Tables standard setting for the 24 space groups with two origin choices. The two are
      bridged explicitly through the tabulated spglib-default setting rather than assumed
      to coincide, which would misplace a structure by a fraction of a cell while still
      passing a symmetry check.
    """
    try:
        import spglib
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "recognizing the symmetry of a structure that does not carry any requires spglib; "
            "install it with `pip install httk-atomistic[default]`, or supply the space group "
            "explicitly via recognize_asu(..., setting=...)"
        ) from error
    register_citation(
        applies_to="Symmetry recognition uses spglib",
        references={
            "authors": (
                {"name": "Atsushi Togo"},
                {"name": "Kohei Shinohara"},
                {"name": "Isao Tanaka"},
            ),
            "title": "Spglib: a software library for crystal symmetry search",
            "journal": "Science and Technology of Advanced Materials: Methods",
            "volume": "4",
            "pages": "2384822",
            "year": "2024",
            "doi": "10.1080/27660400.2024.2384822",
            "bib_type": "article",
        },
    )

    names = sorted(set(view.species_at_sites))
    cell = (
        view.cell.basis.to_floats(),
        view.sites.reduced_coords.to_floats(),
        [names.index(name) + 1 for name in view.species_at_sites],
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=tolerance)
    if dataset is None:
        raise ValueError(f"spglib could not determine the symmetry of this structure at symprec={tolerance}")

    standard = Spacegroup.standard(int(dataset.number))

    # spglib's convention: x_spglib_std = P x_own + p.
    own_to_spglib = _exact_operation(dataset.transformation_matrix, dataset.origin_shift)
    # Ours: x_spglib_default = M x_standard + v, for the setting spglib treats as default.
    from httk.atomistic import data as symmetry_data

    spglib_default = symmetry_data.spglib_default_spacegroup_setting(standard.it_number)
    standard_to_spglib = SettingTransform.from_hall_entry(spglib_default["hall_entry"]).operation

    standard_to_own = own_to_spglib.inverse() * standard_to_spglib
    return (
        standard,
        SettingTransform(standard_to_own.matrix, standard_to_own.vector),
        tuple(str(letter) for letter in dataset.wyckoffs),
        tuple(int(equivalent) for equivalent in dataset.equivalent_atoms),
    )


def _exact_operation(matrix: Any, vector: Any) -> Any:
    """An exact affine operation from spglib's floating-point transformation."""
    from httk.atomistic.symmetry.affine_operation import AffineOperation

    def exact_matrix(value: Any) -> fractions.Fraction:
        return fractions.Fraction(float(value)).limit_denominator(_SPGLIB_MATRIX_MAX_DENOMINATOR)

    def exact_origin(value: Any) -> fractions.Fraction:
        floating = float(value)
        crystallographic = fractions.Fraction(floating).limit_denominator(_SPGLIB_MATRIX_MAX_DENOMINATOR)
        if abs(float(crystallographic) - floating) <= _SPGLIB_SMALL_RATIONAL_TOLERANCE:
            return crystallographic
        # An arbitrary origin is data-derived, not a crystallographic constant. Retain twelve
        # decimal places -- well beyond spglib's useful positional tolerance -- on one common
        # decimal grid instead of accepting a best-approximant denominator that then multiplies
        # every measured free parameter into an enormous, relatively-prime fraction.
        return fractions.Fraction(f"{floating:.12f}")

    return AffineOperation(
        [[exact_matrix(entry) for entry in row] for row in matrix],
        [exact_origin(entry) for entry in vector],
    )


#: Change-of-basis entries are crystallographic halves, thirds, quarters, sixths, and eighths.
_SPGLIB_MATRIX_MAX_DENOMINATOR = 48

#: Preserve true crystallographic origin fractions when spglib's float is merely round-off away.
_SPGLIB_SMALL_RATIONAL_TOLERANCE = 1e-12

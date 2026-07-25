"""Recognizing the asymmetric unit of a full structure.

This is the direction where tolerance lives, and the only one. A measured or file-read
structure gives coordinates like ``0.3333`` that lie *near* a symmetric arrangement rather
than on one, so recognizing it means deciding which Wyckoff position each site is meant to
occupy. Once decided, the position supplies exact values for its fixed components and
everything downstream — expansion, comparison, round-tripping — is exact again.

The contract, stated once so the asymmetry is not a surprise:

* Expansion (:meth:`~httk.atomistic.ASUStructure.expand_sites`) is **lossless**.
* Recognition is **lossy at the tolerance level, by design**: it snaps a measured structure
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
from typing import Any, Sequence

from httk.core import FracVector, SurdVector

from ._periodic_wrap import wrap_periodic_half
from ._periodicity_guard import require_full_periodicity
from .asu_structure import ASUSite, ASUStructure
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup
from .structure_like import StructureLike
from .wyckoff import WyckoffBranch, WyckoffPosition

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
    """A matching tolerance derived from how precisely the structure was stated.

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
    sites, which is what would let genuinely distinct atoms be merged. That is the honest
    role for the minimum separation: bounding a tolerance from above, not — as the code this
    replaces did — being folded into the precision itself, where two accidentally-close
    atoms would make a structure look far more precisely stated than it is.
    """
    from .structure_simple_view import StructureSimpleView

    view = StructureSimpleView(structure)
    precision = view.cartesian_precision()
    if precision is None:
        return fallback

    tolerance = float(precision) * _SAFETY_FACTOR
    shortest_edge = min(length.to_float() for length in view.cell.lengths)
    if tolerance <= _CAP_TRIGGER * shortest_edge:
        return tolerance

    cap = _half_minimum_separation(view)
    return tolerance if cap is None or tolerance < cap else cap


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
    """
    from .structure_simple_view import StructureSimpleView

    view = StructureSimpleView(structure)
    require_full_periodicity(view.cell, "recognize_asu")
    if tolerance is None:
        tolerance = structure_tolerance(view)

    if setting is not None:
        if standard is not None or transform is not None:
            raise TypeError("recognize_asu() takes either 'setting' or 'standard'/'transform', not both")
        standard = setting.standard_setting()
        transform = setting.transform_from_standard
    elif standard is not None or transform is not None:
        if standard is None or transform is None:
            raise TypeError("recognize_asu() needs both 'standard' and 'transform' when either is given")
        if not standard.is_standard_setting:
            raise ValueError(f"'standard' must be an IT standard setting, got {standard.setting}")
    else:
        standard, found = _find_symmetry(view, tolerance)
        # spglib fixes the group but not the frame: its transformation is one of many that
        # describe the structure correctly, differing by an element of the group's affine
        # normalizer. All of them are faithful — the round trip holds either way — but they
        # name different Wyckoff letters, so prefer the standard setting when the structure
        # is already in it, which is by far the common case and gives the tidier answer.
        try:
            return _recognize(view, standard, SettingTransform.identity(), tolerance, limit_denominator)
        except ValueError:
            transform = found

    return _recognize(view, standard, transform, tolerance, limit_denominator)


def _recognize(
    view: Any,
    standard: Spacegroup,
    transform: SettingTransform,
    tolerance: float,
    limit_denominator: int | None,
) -> ASUStructure:
    """Place every site on a Wyckoff position, then group the sites into orbits."""
    cell = view.cell
    own_coords = view.sites.reduced_coords.normalize()
    species_at_sites = tuple(view.species_at_sites)
    count = len(species_at_sites)

    placed: list[tuple[WyckoffPosition, FracVector, FracVector]] = []
    for index in range(count):
        own_point = own_coords[index]
        standard_point = transform.to_standard(own_point).normalize()
        match = _place_on_position(standard, standard_point, own_point, cell, transform, tolerance)
        if match is None:
            raise ValueError(
                f"site {index} at {tuple(own_point.to_fractions())} does not lie on any Wyckoff position of "
                f"{standard.setting} within a tolerance of {tolerance}; the structure does not have "
                f"the symmetry it was given"
            )
        position, branch, parameters = match
        if limit_denominator is not None and position.free_count:
            parameters = FracVector.create(
                [value.limit_denominator(limit_denominator) for value in parameters.to_fractions()]
            )
        snapped_own = transform.to_setting(branch.coordinate(parameters)).normalize()
        placed.append((position, parameters, snapped_own))

    return _group_into_orbits(view, standard, transform, placed, species_at_sites, cell, tolerance)


def _place_on_position(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Any,
    transform: SettingTransform,
    tolerance: float,
) -> tuple[WyckoffPosition, WyckoffBranch, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance`` of a site.

    Returns the position, the orbit branch the site sits on, and that branch's free
    parameters. Positions are tried most specific first, so a site a hair away from a
    special position is recognized as being on it rather than as a general-position site
    that happens to sit there. The general position matches everything with zero
    displacement, so the walk always terminates for a coordinate inside the cell.

    The parameters are the matched *branch's*, not the representative branch's, and are
    not translated between the two. They do not need to be: evaluating the whole position
    at those parameters regenerates the orbit that contains this site either way, and
    grouping is done by orbit membership rather than by comparing parameters.
    """
    tolerance_squared = tolerance * tolerance
    for position in standard.wyckoff:
        for branch in position.branches:
            parameters = branch.nearest_parameters(standard_point)
            candidate = transform.to_setting(branch.coordinate(parameters))
            if _cartesian_distance_squared(own_point - candidate, cell) <= tolerance_squared:
                return position, branch, parameters
    return None


def _cartesian_distance_squared(difference: FracVector, cell: Any) -> float:
    """The squared length of a fractional difference, in the cell's real geometry.

    Reduced to the shortest lattice representative first, so a site at ``0.999`` and one at
    ``0.001`` count as neighbours rather than as a whole cell apart.
    """
    shortest = SurdVector.create(difference.normalize_half()) * cell.basis
    return float(shortest.lengthsqr().to_float())


def _group_into_orbits(
    view: Any,
    standard: Spacegroup,
    transform: SettingTransform,
    placed: Sequence[tuple[WyckoffPosition, FracVector, FracVector]],
    species_at_sites: tuple[str, ...],
    cell: Any,
    tolerance: float,
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
    asu_sites: list[ASUSite] = []

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

        members = [
            other
            for other in range(len(placed))
            if not consumed[other] and _lies_in_orbit(placed[other][2], orbit, cell, tolerance)
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

        for other in members:
            consumed[other] = True
        asu_sites.append(ASUSite(position.letter, parameters, species))

    # The recognized ASU inherits the precision of the structure it came from: nothing
    # about recognition sharpens the data, and dropping it here would mean the value had
    # to be guessed again by everything downstream.
    return ASUStructure(cell, standard, asu_sites, view.species, transform, view.sites.precision)


def _lies_in_orbit(point: FracVector, orbit: Sequence[FracVector], cell: Any, tolerance: float) -> bool:
    """Whether a site coincides with any point of an orbit, to within the tolerance."""
    limit = tolerance * tolerance
    return any(_cartesian_distance_squared(point - other, cell) <= limit for other in orbit)


def _find_symmetry(view: Any, tolerance: float) -> tuple[Spacegroup, SettingTransform]:
    """Find the space group of a structure that carries no symmetry information, via spglib.

    This is the only place spglib is used, and the only path that needs it. Anything that
    arrives with its symmetry already stated — a CIF, or a structure built from an ASU —
    goes through the exact tabulated route instead.

    Two conversions matter here and are easy to get wrong:

    * spglib reports its transformation as floats. They are small rationals in practice, so
      they are recovered exactly with a bounded denominator; the fixed coordinates in the
      vendored tables all have denominators dividing 12.
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
    from . import data as symmetry_data

    spglib_default = symmetry_data.spglib_default_spacegroup_setting(standard.it_number)
    standard_to_spglib = SettingTransform.for_hall_entry(spglib_default["hall_entry"]).operation

    standard_to_own = own_to_spglib.inverse() * standard_to_spglib
    return standard, SettingTransform(standard_to_own.matrix, standard_to_own.vector)


def _exact_operation(matrix: Any, vector: Any) -> Any:
    """An exact affine operation from spglib's floating-point transformation."""
    from .affine_operation import AffineOperation

    def exact(value: Any) -> fractions.Fraction:
        return fractions.Fraction(float(value)).limit_denominator(_SPGLIB_MAX_DENOMINATOR)

    return AffineOperation(
        [[exact(entry) for entry in row] for row in matrix],
        [exact(entry) for entry in vector],
    )


#: Denominator bound when recovering spglib's floating-point transformation exactly.
#: Change-of-basis entries and origin shifts between settings are built from halves,
#: thirds, quarters, sixths and eighths; 48 leaves generous headroom without inventing
#: precision that is not there.
_SPGLIB_MAX_DENOMINATOR = 48

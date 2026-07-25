"""Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

:func:`asu_structure_from_cif` consumes the plain, string-preserving mapping produced by
``httk.io.cif`` (format tag ``"cif"``) and turns it into an exact ASU representation. It
imports nothing from *httk-io* — it only understands the neutral mapping shape — keeping
the parsing capability and the domain model decoupled, exactly as
:func:`~httk.atomistic.structure_from_poscar` already does for POSCAR files.

A CIF is the natural source for an ASU: it lists one site per orbit and states the symmetry
operations that generate the rest. That means no symmetry *search* is needed, and spglib is
not involved. The setting is identified by comparing the file's operations against the
tabulated ones exactly, so a file written in a non-standard setting is recognized as such
rather than silently reinterpreted.
"""

import fractions
from collections.abc import Mapping
from typing import Any

from httk.core import FracVector

from . import data as symmetry_data
from .affine_operation import AffineOperation
from .asu_recognition import DEFAULT_TOLERANCE
from .asu_structure import ASUSite, ASUStructure
from .cell import Cell
from .cell_params import CellParams
from .spacegroup import Spacegroup
from .species import Species

__all__ = ["asu_structure_from_cif", "asu_structures_from_cif", "cif_setting"]


def asu_structures_from_cif(payload: Mapping[str, Any], **options: Any) -> list[ASUStructure]:
    """Every structure in a loaded CIF payload, one per data block that describes one.

    Accepts either a whole loaded payload (with ``blocks``) or a single block.

    Reading a CIF is tolerant — a file may hold blocks that are not structures at all —
    but *asking it for structures* is not. If the file yielded none, the reasons the
    reader recorded are raised here rather than returning an empty list, so a file that
    could not be interpreted does not read as a file that contained nothing.
    """
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
    tolerance: float = DEFAULT_TOLERANCE,
    limit_denominator: int | None = None,
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
    tolerant step; see :mod:`~httk.atomistic.asu_recognition` for the full contract.

    Site occupancies become the composition of the corresponding
    :class:`~httk.atomistic.Species`, so a half-occupied site survives into the structure
    instead of being dropped.
    """
    fmt = data.get("format")
    if fmt != "cif":
        raise ValueError(f"asu_structure_from_cif expected a 'cif' mapping, got format={fmt!r}.")

    setting = cif_setting(data)
    standard = setting.standard_setting()
    transform = setting.transform_from_standard
    cell = _cell_from_cif(data)

    coordinates = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    occupancies = data.get("occupancies")

    species_by_name: dict[str, Species] = {}
    asu_sites: list[ASUSite] = []
    for index, coordinate in enumerate(coordinates):
        occupancy = 1.0 if occupancies is None or occupancies[index] is None else float(occupancies[index])
        name = _species_name(symbols[index], labels[index], occupancy)
        if name not in species_by_name:
            species_by_name[name] = Species(
                name=name,
                chemical_symbols=(symbols[index],),
                concentration=(occupancy,),
                original_name=labels[index],
            )

        standard_point = transform.to_standard(coordinate).normalize()
        match = _snap(standard, standard_point, coordinate, cell, transform, tolerance)
        if match is None:
            raise ValueError(
                f"CIF site {labels[index]!r} at {tuple(coordinate.to_fractions())} does not lie on any "
                f"Wyckoff position of {setting.setting} within {tolerance}; the file's coordinates and its "
                f"symmetry operations disagree"
            )
        letter, parameters = match
        if limit_denominator is not None and parameters.dim not in ((), (0,)):
            parameters = FracVector.create(
                [value.limit_denominator(limit_denominator) for value in parameters.to_fractions()]
            )
        asu_sites.append(ASUSite(letter, parameters, name))

    return ASUStructure(cell, standard, asu_sites, list(species_by_name.values()), transform)


def cif_setting(data: Mapping[str, Any]) -> Spacegroup:
    """The space-group setting a CIF block is written in.

    Identified from the file's symmetry operations by exact set comparison against the
    tabulated settings, which is what makes a non-standard setting come out as itself. The
    declared Hall symbol, Hermann-Mauguin symbol, or IT number only narrows the search;
    where a symbol and the operations disagree, the operations win, because they are what
    the file's coordinates were actually generated with.

    Raises :class:`ValueError` for a setting that matches no tabulated one. That is a real
    limitation and is deliberate: the transform to the standard setting cannot be *derived*
    without choosing arbitrarily among an infinite family of equally valid ones, so such a
    file must be handled by supplying the transform explicitly.
    """
    operations = data.get("symops")
    if not operations:
        raise ValueError("this CIF block states no symmetry operations, so its setting cannot be determined")
    target = frozenset(AffineOperation(rotation, translation).wrapped() for rotation, translation in operations)

    for record in _candidate_settings(data):
        candidate = Spacegroup(record)
        if frozenset(operation.wrapped() for operation in candidate.symmetry_operations) == target:
            return candidate

    declared = data.get("space_group_name_hm") or data.get("space_group_name_hall") or data.get("space_group_nbr")
    raise ValueError(
        f"the {len(target)} symmetry operations in this CIF match no tabulated space-group setting "
        f"(the file declares {declared!r}). If this is a genuinely non-standard setting, build the "
        f"structure with an explicit SettingTransform instead; a transform cannot be derived, since "
        f"infinitely many are equally valid and they describe different crystals."
    )


def _candidate_settings(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Settings worth comparing against, narrowed by whatever the file declares."""
    hall = data.get("space_group_name_hall")
    if hall:
        try:
            return [symmetry_data.spacegroup_setting(hall_entry=str(hall).strip())]
        except KeyError:
            pass

    number = data.get("space_group_nbr")
    if number is not None:
        try:
            it_number = int(str(number).strip())
        except ValueError:
            it_number = 0
        if 1 <= it_number <= 230:
            narrowed = [record for record in symmetry_data.spacegroup_settings() if record["it_number"] == it_number]
            if narrowed:
                return narrowed

    return list(symmetry_data.spacegroup_settings())


def _cell_from_cif(data: Mapping[str, Any]) -> Cell:
    """The cell, built exactly from the lattice parameters where the file gives them."""
    parameters = data.get("cell_parameters")
    if parameters is not None:
        return Cell(CellParams([fractions.Fraction(str(value)) for value in parameters]).basis)
    return Cell(data["basis"])


def _exact_positions(data: Mapping[str, Any]) -> list[FracVector]:
    """Site coordinates as the exact rationals the file wrote.

    Prefers the preserved decimal text over the parsed floats: ``Fraction("0.3333")`` is
    ``3333/10000``, whereas ``Fraction(float("0.3333"))`` is a binary approximation that
    states a precision the file never claimed.
    """
    exact = data.get("positions_exact")
    if exact is not None:
        return [FracVector.create([fractions.Fraction(value) for value in row]) for row in exact]
    return [FracVector.create(list(row)) for row in data["positions"]]


def _species_name(symbol: str, label: str, occupancy: float) -> str:
    """A species name: the element where that is unambiguous, else the file's site label.

    A fully occupied site is named for its element, which keeps ordinary structures
    readable. A partially occupied one is named for its CIF label instead, since two sites
    of the same element can carry different occupancies and would otherwise collide.
    """
    return symbol if occupancy == 1.0 else label


def _snap(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Cell,
    transform: Any,
    tolerance: float,
) -> tuple[str, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance``, and its free parameters."""
    from .asu_recognition import _cartesian_distance_squared

    limit = tolerance * tolerance
    for position in standard.wyckoff:
        for branch in position.branches:
            parameters = branch.nearest_parameters(standard_point)
            candidate = transform.to_setting(branch.coordinate(parameters))
            if _cartesian_distance_squared(own_point - candidate, cell) <= limit:
                return position.letter, parameters
    return None

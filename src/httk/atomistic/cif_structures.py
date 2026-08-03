"""Build an exact :class:`~httk.atomistic.ASUStructure` from a neutral CIF mapping.

:func:`asu_structure_from_cif` consumes the plain, string-preserving mapping produced by
``httk.io.cif`` (format tag ``"cif"``) and turns it into an exact ASU representation. It
imports nothing from *httk-io* — it only understands the neutral mapping shape — keeping
the parsing capability and the domain model decoupled, exactly as
the private POSCAR adapter already does for POSCAR files.

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

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup

from . import data as symmetry_data
from ._composition_values import as_fraction

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
    tolerance: float | None = None,
    limit_denominator: int | None = None,
    trust_declared_symmetry: bool = True,
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
    """
    fmt = data.get("format")
    if fmt != "cif":
        raise ValueError(f"asu_structure_from_cif expected a 'cif' mapping, got format={fmt!r}.")

    setting = cif_setting(data, trust_declared_symmetry=trust_declared_symmetry)
    standard = setting.standard_setting()
    transform = setting.transform_from_standard
    cell = _cell_from_cif(data)

    if tolerance is None:
        # Derived from the digits the file itself wrote, rather than a constant. The sites
        # are the asymmetric unit rather than a full cell, which is all this needs: the
        # tolerance depends on the precision and the cell, not on how many atoms there are.
        tolerance = _tolerance_from_cif(data, cell)

    coordinates = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    occupancies = data.get("occupancies")
    occupancies_exact = data.get("occupancies_exact")
    occupancy_precisions = data.get("occupancy_precisions")

    species_by_name: dict[str, Species] = {}
    wyckoff_sites: list[WyckoffSite] = []
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
        name = _species_name(symbols[index], labels[index], occupancy)
        if name not in species_by_name:
            species_by_name[name] = Species(
                name=name,
                chemical_symbols=(symbols[index],),
                concentration=(occupancy,),
                original_name=labels[index],
                concentration_precision=(occupancy_precision,) if occupancy_precisions is not None else None,
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
        wyckoff_sites.append(WyckoffSite(letter, parameters, name, coordinate.normalize()))

    return ASUStructure(
        cell,
        standard,
        wyckoff_sites,
        list(species_by_name.values()),
        transform,
        data.get("coordinate_precision"),
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
    """
    operations = data.get("symops")
    if not operations:
        raise ValueError("this CIF block states no symmetry operations, so its setting cannot be determined")
    target = frozenset(AffineOperation(rotation, translation).wrapped() for rotation, translation in operations)

    candidates: list[dict[str, Any]] | None = None
    declared = None
    if trust_declared_symmetry:
        candidates, declared = _declared_settings(data)
    if candidates is None:
        candidates = symmetry_data.spacegroup_settings()

    for record in candidates:
        candidate = Spacegroup(record)
        if frozenset(operation.wrapped() for operation in candidate.symmetry_operations) == target:
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


def _declared_settings(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
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
                f"the setting from the symmetry operations alone."
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
    parameters = data.get("cell_parameters")
    if parameters is not None:
        return Cell(CellParams([fractions.Fraction(str(value)) for value in parameters]).basis, 1, precision)
    return Cell(data["basis"], 1, precision)


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


def _species_name(symbol: str, label: str, occupancy: Any) -> str:
    """A species name: the element where that is unambiguous, else the file's site label.

    A fully occupied site is named for its element, which keeps ordinary structures
    readable. A partially occupied one is named for its CIF label instead, since two sites
    of the same element can carry different occupancies and would otherwise collide.
    """
    return symbol if as_fraction(occupancy, field="CIF occupancy")[0] == 1 else label


def _snap(
    standard: Spacegroup,
    standard_point: FracVector,
    own_point: FracVector,
    cell: Cell,
    transform: Any,
    tolerance: float,
) -> tuple[str, FracVector] | None:
    """The most specific Wyckoff position within ``tolerance``, and its free parameters."""
    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    limit = tolerance * tolerance
    for position in standard.wyckoff:
        for branch in position.branches:
            parameters = branch.nearest_parameters(standard_point)
            candidate = transform.to_setting(branch.coordinate(parameters))
            if _cartesian_distance_squared(own_point - candidate, cell) <= limit:
                return position.letter, parameters
    return None

"""Private neutral geometry payloads shared by the simulation-code integrations.

The Quantum ESPRESSO, ABINIT, CP2K and LAMMPS-data readers parse their text into one
neutral *geometry payload*; :func:`_structure_from_geometry` turns it into an exact
:class:`~httk.atomistic.UnitcellStructure` and :func:`_geometry_from_structure` produces
it for their writers. The payload is a mapping with these keys:

``format``
    The producing format's tag (a string).
``cell``
    Three rows (the a, b and c lattice vectors) of three entries. An entry is a decimal
    string (Fortran ``d``/``D`` exponents allowed) or an exact number: an ``int``, a
    :class:`~fractions.Fraction` or an *httk* exact scalar such as a ``SurdScalar``.
``cell_unit``
    ``"angstrom"`` or ``"bohr"``.
``bohr_angstrom``
    The producing code's own Bohr length in Å as a decimal string; required when any
    unit is ``"bohr"``. Conversion through it is exact.
``positions``
    One row of three entries per site, with the same entry types as ``cell``.
``positions_kind``
    ``"fractional"`` or ``"cartesian"``.
``positions_unit``
    ``"angstrom"`` or ``"bohr"``; required when ``positions_kind`` is ``"cartesian"``.
``species``
    A list of ``{"name": str, "element": str, "mass": str | None}``; ``element`` is an
    element symbol and ``mass`` a decimal string in atomic mass units.
``species_at_sites``
    One declared species name per site.
``periodicity``
    Three booleans, one per cell row; all ``True`` when absent.
``precision_override``
    Optional Cartesian precision in Å with the meaning and validation of POSCAR's.
``raw``
    Optional original text (a string), written verbatim by the writers.

Any other key is format-specific and ignored here. Without ``precision_override``, the
cell and site precisions are derived from the digits of the decimal-string entries exactly
as for POSCAR (in fractional positions ``0.0``, ``0.5`` and ``1.0`` make no claim); exact
entries make no precision claim. A trajectory payload is ``{"format": tag, "frames":
[geometry payloads without "format"], ...}``.
"""

import fractions
import io
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from httk.core import SurdScalar, SurdVector, combined_precision

from httk.atomistic._atomic_projection import require_bare_atomic_projection
from httk.atomistic._loading import _cartesian_precision_factor, _cell_with_precision_override
from httk.atomistic._writing import _poscar_token
from httk.atomistic.elements import SYMBOLS, atomic_number
from httk.atomistic.integrations.vasp.io.poscar_reader import _coordinate_precision as _token_precision
from httk.atomistic.integrations.vasp.io.poscar_writer import _write_poscar_payload
from httk.atomistic.io import _check_load_precision
from httk.atomistic.models._vector_guards import to_surdscalar
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.trajectory.trajectory import Trajectory

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")
_LABEL_SUFFIX = re.compile(r"\d*(?:[_-].*)?")
_ELEMENTS = frozenset(SYMBOLS)
_UNITS = ("angstrom", "bohr")


def _number(token: str) -> fractions.Fraction:
    """Parse a decimal token exactly, accepting Fortran ``d``/``D`` exponents.

    :param token: A decimal literal such as ``"1.5"``, ``"-2e3"`` or ``"1.0d-3"``.
    :return: The exact value.
    :raises ValueError: If ``token`` is not a decimal literal.
    """
    if not isinstance(token, str) or not _NUMBER.fullmatch(token.strip()):
        raise ValueError(f"expected a decimal number, got {token!r}")
    return fractions.Fraction(_normalized(token))


def _normalized(token: str) -> str:
    """A decimal token with a Fortran exponent letter replaced by ``e``."""
    return token.strip().replace("d", "e").replace("D", "e")


def _render(value: Any) -> str:
    """Render an exact number as a geometry token by the POSCAR rule.

    Finite decimals stay exact; other values are rounded to 16 significant digits.

    :param value: An ``int``, :class:`~fractions.Fraction` or *httk* exact scalar.
    :return: The decimal token.
    """
    return _poscar_token(fractions.Fraction(value) if isinstance(value, int) else value)


def _element_from_label(label: str) -> str | None:
    """Return the element a site or species label names, or ``None``.

    The label starts with an element symbol in any letter case (a valid two-letter symbol
    is preferred), optionally followed by digits and then optionally by a ``_`` or ``-``
    suffix: ``"Fe1"``, ``"FE"``, ``"O_2"`` and ``"Si-a"`` all qualify. Callers add any
    code-specific limits such as a maximum length.

    :param label: The label to interpret.
    :return: The canonical element symbol, or ``None`` when the label names none.
    """
    for size in (2, 1):
        head = label[:size]
        symbol = head.capitalize()
        if len(head) == size and symbol in _ELEMENTS and _LABEL_SUFFIX.fullmatch(label[size:]):
            return symbol
    return None


def _write_text(
    destination: str | os.PathLike[str] | io.TextIOBase,
    payload: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    render: Callable[..., str],
) -> None:
    """Write a payload's ``raw`` text verbatim, or else ``render(payload, **kwargs)``.

    :param destination: A filesystem path (written as UTF-8) or an open text stream.
    :param payload: The geometry payload.
    :param kwargs: Writer options, passed to ``render``.
    :param render: Renders the payload as file text.
    :raises ValueError: If writer options are given for a payload carrying ``raw`` text.
    """
    raw = payload.get("raw")
    if isinstance(raw, str):
        if kwargs:
            raise ValueError(
                f"writer options {sorted(kwargs)} cannot apply to a payload carrying its original 'raw' text; "
                "remove 'raw' to render the geometry instead"
            )
        text = raw
    else:
        text = render(payload, **kwargs)
    # POSCAR's raw channel is the generic verbatim text sink (path or stream, newlines kept).
    _write_poscar_payload(destination, {"raw": text})


def _rows(value: Any, field: str, count: int | None = None) -> Sequence[Sequence[Any]]:
    """Validate a sequence of rows of three entries."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or (count is not None and len(value) != count)
        or any(isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3 for row in value)
    ):
        size = "rows" if count is None else f"{count} rows"
        raise ValueError(f"geometry payload {field} must be {size} of 3 entries, got {value!r}")
    return value


def _exact(entry: Any, field: str) -> SurdScalar:
    """One exact payload entry."""
    if isinstance(entry, str):
        return to_surdscalar(_number(entry))
    if isinstance(entry, (bool, float)):
        raise ValueError(f"geometry payload {field} entries must be decimal strings or exact numbers, got {entry!r}")
    try:
        return to_surdscalar(entry)
    except (TypeError, ValueError) as error:
        raise ValueError(f"geometry payload {field} entry {entry!r} is not an exact number: {error}") from None


def _matrix(rows: Sequence[Sequence[Any]], field: str) -> SurdVector:
    """An exact matrix from rows of decimal strings and exact numbers."""
    values = [[_exact(entry, field) for entry in row] for row in rows]
    radicands = sorted({radicand for row in values for value in row for radicand in value.radicands} | {1})
    return SurdVector.from_radicand_map(
        {
            radicand: [[value.coefficient(radicand).to_fraction() for value in row] for row in values]
            for radicand in radicands
        }
    )


def _tokens(rows: Sequence[Sequence[Any]]) -> list[list[str]]:
    """The decimal-string entries of some rows, with Fortran exponents normalized."""
    return [[_normalized(entry) for entry in row if isinstance(entry, str)] for row in rows]


def _unit_scale(data: Mapping[str, Any], key: str) -> fractions.Fraction:
    """The exact factor converting a unit field's lengths into Å."""
    unit = data.get(key)
    if unit not in _UNITS:
        raise ValueError(f"geometry payload {key} must be one of {_UNITS}, got {unit!r}")
    if unit == "angstrom":
        return fractions.Fraction(1)
    bohr = data.get("bohr_angstrom")
    if not isinstance(bohr, str):
        raise ValueError(f"geometry payload with {key}='bohr' needs bohr_angstrom as a decimal string, got {bohr!r}")
    value = _number(bohr)
    if value <= 0:
        raise ValueError(f"geometry payload bohr_angstrom must be positive, got {bohr!r}")
    return value


def _species(data: Mapping[str, Any]) -> list[Species]:
    """The declared species of a geometry payload."""
    declared = data.get("species")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence) or not declared:
        raise ValueError(f"geometry payload species must be a non-empty list, got {declared!r}")
    species: list[Species] = []
    for item in declared:
        if not isinstance(item, Mapping):
            raise ValueError(f"geometry payload species entries must be mappings, got {item!r}")
        name, element, mass = item.get("name"), item.get("element"), item.get("mass")
        if not isinstance(name, str) or not name:
            raise ValueError(f"geometry payload species name must be a non-empty string, got {name!r}")
        if not isinstance(element, str):
            raise ValueError(f"geometry payload species {name!r} element must be a string, got {element!r}")
        atomic_number(element)
        if mass is not None and not isinstance(mass, str):
            raise ValueError(f"geometry payload species {name!r} mass must be a decimal string or None, got {mass!r}")
        species.append(
            Species(
                name=name,
                chemical_symbols=(element,),
                concentration=(1,),
                mass=None if mass is None else (float(_number(mass)),),
            )
        )
    names = [item.name for item in species]
    if len(set(names)) != len(names):
        raise ValueError(f"geometry payload species names must be unique, got {names}")
    return species


def _unitcell(data: Mapping[str, Any]) -> UnitcellStructure:
    """Build one structure from a geometry payload, ignoring its ``format``."""
    cell_rows = _rows(data.get("cell"), "cell", 3)
    positions = _rows(data.get("positions"), "positions")
    if not positions:
        raise ValueError("geometry payload positions must hold at least one site")
    kind = data.get("positions_kind")
    if kind not in ("fractional", "cartesian"):
        raise ValueError(f"geometry payload positions_kind must be 'fractional' or 'cartesian', got {kind!r}")
    cartesian = kind == "cartesian"
    periodicity = data.get("periodicity")
    if periodicity is None:
        periodicity = (True, True, True)
    if (
        isinstance(periodicity, (str, bytes))
        or not isinstance(periodicity, Sequence)
        or len(periodicity) != 3
        or any(not isinstance(flag, bool) for flag in periodicity)
    ):
        raise ValueError(f"geometry payload periodicity must be three booleans, got {periodicity!r}")
    override = data.get("precision_override")
    _check_load_precision(override, "geometry payload")

    species = _species(data)
    species_at_sites = data.get("species_at_sites")
    if (
        isinstance(species_at_sites, (str, bytes))
        or not isinstance(species_at_sites, Sequence)
        or len(species_at_sites) != len(positions)
    ):
        raise ValueError(f"geometry payload species_at_sites must list one species name per site ({len(positions)})")
    names = {item.name for item in species}
    undeclared = sorted({str(name) for name in species_at_sites if name not in names})
    if undeclared:
        raise ValueError(f"geometry payload species_at_sites names undeclared species {undeclared}")

    raw_basis = _matrix(cell_rows, "cell")
    if raw_basis.det().sign() == 0:
        raise ValueError("geometry payload cell is degenerate (zero determinant)")
    cell_scale = _unit_scale(data, "cell_unit")
    coordinates = _matrix(positions, "positions")
    # Cartesian rows in their own unit, re-expressed in the cell's unit, then reduced.
    position_factor = _unit_scale(data, "positions_unit") / cell_scale if cartesian else fractions.Fraction(1)
    reduced = coordinates * raw_basis.inv() * position_factor if cartesian else coordinates

    coordinate_precision: fractions.Fraction | None
    if override is not None:
        cell, coordinate_precision = _cell_with_precision_override(raw_basis, cell_scale, override)
    else:
        cell_precision = combined_precision(token for row in _tokens(cell_rows) for token in row)
        cell = Cell(raw_basis, cell_scale, None if cell_precision is None else cell_precision * cell_scale)
        coordinate_precision = _token_precision(_tokens(positions), cartesian=cartesian)
        if coordinate_precision is not None and cartesian:
            coordinate_precision *= position_factor * _cartesian_precision_factor(raw_basis)
    cell = Cell(cell, periodicity=tuple(periodicity))
    return UnitcellStructure(cell, Sites(reduced, coordinate_precision), species, list(species_at_sites))


def _structure_from_geometry(data: Mapping[str, Any]) -> UnitcellStructure:
    """Build an exact :class:`~httk.atomistic.UnitcellStructure` from a neutral geometry payload.

    :param data: A geometry payload as described in the module docstring.
    :return: The exact structure.
    :raises ValueError: If the payload is malformed.
    """
    if not isinstance(data, Mapping) or not isinstance(data.get("format"), str):
        raise ValueError("geometry payload must be a mapping with a string 'format' tag")
    return _unitcell(data)


def _trajectory_from_frames(data: Mapping[str, Any]) -> Trajectory:
    """Build the native :class:`~httk.atomistic.Trajectory` of a geometry frames payload.

    Every frame is built by :func:`_structure_from_geometry`'s rules, so each keeps its own
    precision.

    :param data: ``{"format": tag, "frames": [geometry payloads without "format"], ...}``.
    :return: The trajectory, one frame per payload frame.
    :raises ValueError: If the payload or any frame is malformed.
    """
    if not isinstance(data, Mapping) or not isinstance(data.get("format"), str):
        raise ValueError("geometry frames payload must be a mapping with a string 'format' tag")
    frames = data.get("frames")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence) or not frames:
        raise ValueError("geometry frames payload must hold a non-empty list of frames")
    structures: list[UnitcellStructure] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"geometry frame {index} must be a mapping, got {type(frame).__name__}")
        try:
            structures.append(_unitcell(frame))
        except ValueError as error:
            raise ValueError(f"geometry frame {index}: {error}") from error
    return Trajectory(structures)


def _geometry_from_structure(obj: Any, format_tag: str) -> dict[str, Any]:
    """Serialize a structure to a neutral geometry payload (Å cell, fractional positions).

    :param obj: Anything :class:`~httk.atomistic.UnitcellStructureView` accepts.
    :param format_tag: The payload's ``format`` tag.
    :return: The geometry payload, without ``raw``.
    :raises ValueError: If the structure is empty or has disordered or partially occupied species.
    """
    structure = UnitcellStructureView(obj)
    require_bare_atomic_projection(structure, format_tag)
    species_at_sites = list(structure.species_at_sites)
    if not species_at_sites:
        raise ValueError(f"{format_tag} cannot represent an empty structure")
    used = set(species_at_sites)
    species: list[dict[str, Any]] = []
    for item in structure.species:
        if item.name not in used:
            continue
        if not item.is_single_element:
            raise ValueError(
                f"{format_tag} cannot represent disorder/partial occupancy for species {item.name!r}; "
                "each occupied site must have one fully occupied chemical symbol"
            )
        mass = None if item.mass is None else repr(item.mass[0])
        species.append({"name": item.name, "element": item.chemical_symbols[0], "mass": mass})
    basis = structure.cell.basis
    return {
        "format": format_tag,
        "cell": [[_render(basis._element((row, column))) for column in range(3)] for row in range(3)],
        "cell_unit": "angstrom",
        "positions": [[_render(value) for value in row.to_fractions()] for row in structure.sites.reduced_coords],
        "positions_kind": "fractional",
        "species": species,
        "species_at_sites": species_at_sites,
        "periodicity": list(structure.cell.periodicity),
    }

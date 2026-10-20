"""The lossless ``httk-structure`` JSON file format for one exact unit-cell structure.

A ``.httkstruct`` file holds the complete exact state of a
:class:`~httk.atomistic.UnitcellStructure`, so ``load(save(s)) == s`` and the reloaded
structure has the same content id. It is *httk*-specific: use it to keep a structure
exactly (for example as the recorded input of a calculation), and CIF or POSCAR to
exchange it with other software. Reading never evaluates or imports anything; the reader
is a strict validating parser.

The document is one JSON object with exactly these members (all required; ``null`` marks
an unstated value)::

    format, version (1), cell, sites, species, species_at_sites, site_moments, charge,
    molecular, assemblies, symmetry, chemical_composition, chemical_formula_descriptive,
    chemical_formula_hill, optimization_type, immutable_id, last_modified

* ``cell``: ``unscaled_basis`` (3x3 surds), ``scale`` (surd), ``precision`` (rational),
  ``periodicity`` (three booleans).
* ``sites``: ``reduced_coords`` (Nx3 rationals), ``precision`` (rational).
* ``species``: objects with ``name``, ``chemical_symbols``, ``concentration``,
  ``concentration_precision``, ``mass`` (JSON floats), ``original_name``, ``attached``,
  ``nattached``, ``charges``, ``spins`` and ``labels``.
* ``site_moments``: ``kind`` (``collinear``, ``cartesian`` or ``crystalaxis``),
  ``moments`` (N surds for collinear, otherwise Nx3) and ``precision``.
* ``assemblies``: objects with ``sites_in_groups``, ``group_probabilities`` and
  ``group_probabilities_precision``.
* ``symmetry``: the six :class:`~httk.atomistic.models.structure.semantics.StructureSymmetry`
  fields; ``chemical_composition``: ``mode`` and ``amounts`` (objects with ``element``,
  ``amount``, ``precision``); ``last_modified``: ISO 8601 text with a UTC offset.

Exact numbers are strings in one canonical spelling, so the same structure state gives
identical bytes. A *rational* is ``-?N`` or ``-?N/D`` in lowest terms with ``D > 1``
(``"0"``, ``"-3"``, ``"173/97"``). A *surd* is a rational, or a sum of at most four terms
ordered by increasing radicand, each ``R`` (the rational part, first) or ``R*sqrt(K)`` with
a squarefree ``1 < K <= 1000``, joined by their signs (``"1/2*sqrt(3)"``,
``"1-3/2*sqrt(3)"``).

The reader is a trust boundary, so its limits keep every accepted document cheap to build:
a number string has at most 1024 characters (256 in ``cell``), another string at most 1024,
a symmetry operation at most 64 and there are at most 192 of them, and a cell whose scale
or basis determinant cancels to within ``2**-128`` of its own terms (an exact sign decision
then takes time growing with the cancellation depth) is refused. The writer refuses the same.
"""

import datetime
import io
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import IO, Any, cast

from httk.core import FracVector, SurdScalar, SurdVector, TextstreamFileView

from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.io import _check_as_written
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import StructureSymmetry, _stated_symmetry
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structure.view import StructureView

FORMAT = "httk-structure"
VERSION = 1

_MAX_NUMBER_LENGTH = 1024
_MAX_CELL_NUMBER_LENGTH = 256
_MAX_STRING_LENGTH = 1024
# Surd arithmetic factors radicand products by trial division (Cell computes a determinant),
# so these two bounds keep the worst cell cheap: realistic cells carry sqrt(2), sqrt(3), sqrt(6).
_MAX_RADICAND = 1000
_MAX_SURD_TERMS = 4
# The largest space-group operation count (modulo lattice translations) in the vendored settings.
_MAX_OPERATIONS = 192
_MAX_OPERATION_LENGTH = 64
_SEPARATION_BITS = 128
# ponytail: whole-document cap, read into memory at once; stream-parse if structures this large appear.
_MAX_DOCUMENT_CHARACTERS = 64 * 1024 * 1024

_RATIONAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")
_SURD_TERM = re.compile(r"([+-]?)((?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?)(?:\*sqrt\(([1-9][0-9]{0,3})\))?")

_TOP_KEYS = frozenset(
    {
        "format",
        "version",
        "cell",
        "sites",
        "species",
        "species_at_sites",
        "site_moments",
        "charge",
        "molecular",
        "assemblies",
        "symmetry",
        "chemical_composition",
        "chemical_formula_descriptive",
        "chemical_formula_hill",
        "optimization_type",
        "immutable_id",
        "last_modified",
    }
)
_SPECIES_KEYS = frozenset(
    {
        "name",
        "chemical_symbols",
        "concentration",
        "concentration_precision",
        "mass",
        "original_name",
        "attached",
        "nattached",
        "charges",
        "spins",
        "labels",
    }
)
_SYMMETRY_KEYS = (
    "space_group_it_number",
    "space_group_symbol_hall",
    "space_group_symbol_hermann_mauguin",
    "space_group_symbol_hermann_mauguin_extended",
    "space_group_symmetry_operations_xyz",
    "wyckoff_positions",
)
_MOMENT_KINDS = frozenset({"collinear", "cartesian", "crystalaxis"})


# --------------------------------------------------------------------------- numbers


def _rational_text(value: Fraction, field: str) -> str:
    text = str(value)
    if len(text) > _MAX_NUMBER_LENGTH:
        raise ValueError(f"httk-structure {field} has more than {_MAX_NUMBER_LENGTH} characters")
    return text


def _surd_text(value: SurdScalar, field: str, limit: int = _MAX_NUMBER_LENGTH) -> str:
    if len(value.radicands) > _MAX_SURD_TERMS:
        raise ValueError(f"httk-structure {field} has more than {_MAX_SURD_TERMS} surd terms")
    text = ""
    for radicand in value.radicands:
        if radicand > _MAX_RADICAND:
            raise ValueError(f"httk-structure {field} has a radicand above {_MAX_RADICAND}")
        coefficient = value.coefficient(radicand).to_fraction()
        sign = "-" if coefficient < 0 else ("+" if text else "")
        term = str(abs(coefficient)) if radicand == 1 else f"{abs(coefficient)}*sqrt({radicand})"
        text += sign + term
    text = text or "0"
    if len(text) > limit:
        raise ValueError(f"httk-structure {field} has more than {limit} characters")
    return text


def _number_string(value: object, field: str, limit: int = _MAX_NUMBER_LENGTH) -> str:
    if not isinstance(value, str):
        raise ValueError(f"httk-structure {field} must be an exact number string, got {type(value).__name__}")
    if len(value) > limit:
        raise ValueError(f"httk-structure {field} has more than {limit} characters")
    return value


def _rational(value: object, field: str) -> Fraction:
    text = _number_string(value, field)
    if _RATIONAL.fullmatch(text) is None:
        raise ValueError(f"httk-structure {field} is not an exact rational: {text!r}")
    result = Fraction(text)
    if str(result) != text:
        raise ValueError(f"httk-structure {field} is not in canonical lowest terms: {text!r}")
    return result


def _surd(value: object, field: str, limit: int = _MAX_NUMBER_LENGTH) -> SurdScalar:
    text = _number_string(value, field, limit)
    terms: dict[int, Fraction] = {}
    position = 0
    while position < len(text):
        match = _SURD_TERM.match(text, position)
        if match is None or (position > 0 and not match.group(1)):
            raise ValueError(f"httk-structure {field} is not an exact surd: {text!r}")
        if len(terms) == _MAX_SURD_TERMS:
            raise ValueError(f"httk-structure {field} has more than {_MAX_SURD_TERMS} surd terms")
        radicand = 1 if match.group(3) is None else int(match.group(3))
        if radicand > _MAX_RADICAND or radicand in terms:
            raise ValueError(f"httk-structure {field} is not a canonical exact surd: {text!r}")
        terms[radicand] = Fraction(match.group(2)) * (-1 if match.group(1) == "-" else 1)
        position = match.end()
    result = cast(SurdScalar, SurdVector.from_radicand_map(terms))
    # Canonical spelling only: one text per value (sorted squarefree radicands, nonzero
    # lowest-terms coefficients), so a re-rendering must reproduce the input exactly.
    if _surd_text(result, field, limit) != text:
        raise ValueError(f"httk-structure {field} is not a canonical exact surd: {text!r}")
    return result


# --------------------------------------------------------------------------- shape checks


def _object(value: object, keys: frozenset[str] | tuple[str, ...], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"httk-structure {field} must be a JSON object")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        unknown = sorted(set(value) - set(keys))
        raise ValueError(f"httk-structure {field} has missing members {missing} or unknown members {unknown}")
    return cast(dict[str, Any], value)


def _list(value: object, field: str, length: int | None = None, limit: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"httk-structure {field} must be a JSON array")
    if length is not None and len(value) != length:
        raise ValueError(f"httk-structure {field} must have {length} entries")
    if limit is not None and len(value) > limit:
        raise ValueError(f"httk-structure {field} has more than {limit} entries")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"httk-structure {field} must be a string")
    if len(value) > _MAX_STRING_LENGTH:
        raise ValueError(f"httk-structure {field} has more than {_MAX_STRING_LENGTH} characters")
    return value


def _symbol(value: object, field: str) -> str:
    if not _string(value, field):
        raise ValueError(f"httk-structure {field} must be a non-empty string")
    return cast(str, value)


def _operation(value: object, field: str) -> str:
    if not _symbol(value, field) or len(cast(str, value)) > _MAX_OPERATION_LENGTH:
        raise ValueError(
            f"httk-structure {field} must be a symmetry operation of at most {_MAX_OPERATION_LENGTH} characters"
        )
    return cast(str, value)


def _cell_surd(value: object, field: str) -> SurdScalar:
    return _surd(value, field, _MAX_CELL_NUMBER_LENGTH)


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"httk-structure {field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"httk-structure {field} must be a boolean")
    return value


def _float(value: object, field: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"httk-structure {field} must be a finite JSON floating-point number")
    return value


def _optional[T](parse: Callable[[object, str], T], value: object, field: str) -> T | None:
    return None if value is None else parse(value, field)


def _items[T](parse: Callable[[object, str], T], value: object, field: str, length: int | None = None) -> tuple[T, ...]:
    return tuple(parse(item, f"{field}[{index}]") for index, item in enumerate(_list(value, field, length)))


def _optional_items[T](
    parse: Callable[[object, str], T], value: object, field: str, limit: int | None = None
) -> tuple[T, ...] | None:
    if value is None:
        return None
    _list(value, field, limit=limit)  # reject an oversized array before parsing any entry
    return _items(parse, value, field)


def _items_or_null[T](parse: Callable[[object, str], T]) -> Callable[[object, str], T | None]:
    return lambda value, field: _optional(parse, value, field)


# --------------------------------------------------------------------------- decoding


def _decode_arguments(document: object) -> dict[str, Any]:
    """Validate a document and decode it into exact constructor arguments.

    This checks every member's presence, type, shape and number spelling, but constructs no
    domain objects: semantic consistency (which can be costly, e.g. space-group closure) is
    left to the constructors called by :func:`_structure_from_document`.
    """
    top = _object(document, _TOP_KEYS, "document")
    if top["format"] != FORMAT:
        raise ValueError(f"httk-structure document format must be {FORMAT!r}")
    if not isinstance(top["version"], int) or isinstance(top["version"], bool) or top["version"] != VERSION:
        raise ValueError(f"unsupported httk-structure version; this reader supports version {VERSION}")

    cell = _object(top["cell"], ("unscaled_basis", "scale", "precision", "periodicity"), "cell")
    sites = _object(top["sites"], ("reduced_coords", "precision"), "sites")
    rows = _items(lambda row, field: _items(_rational, row, field, 3), sites["reduced_coords"], "sites.reduced_coords")
    arguments: dict[str, Any] = {
        "unscaled_basis": _items(
            lambda row, field: _items(_cell_surd, row, field, 3), cell["unscaled_basis"], "cell.unscaled_basis", 3
        ),
        "scale": _cell_surd(cell["scale"], "cell.scale"),
        "cell_precision": _optional(_rational, cell["precision"], "cell.precision"),
        "periodicity": _items(_boolean, cell["periodicity"], "cell.periodicity", 3),
        "reduced_coords": rows,
        "sites_precision": _optional(_rational, sites["precision"], "sites.precision"),
        "species": _items(_decode_species, top["species"], "species"),
        "species_at_sites": _items(_string, top["species_at_sites"], "species_at_sites", len(rows)),
        "site_moments": _optional(_decode_moments, top["site_moments"], "site_moments"),
        "charge": _optional(_rational, top["charge"], "charge"),
        "molecular": _boolean(top["molecular"], "molecular"),
        "assemblies": _optional_items(_decode_assembly, top["assemblies"], "assemblies"),
        "symmetry": _optional(_decode_symmetry, top["symmetry"], "symmetry"),
        "chemical_composition": _optional(_decode_composition, top["chemical_composition"], "chemical_composition"),
        "last_modified": _optional(_decode_timestamp, top["last_modified"], "last_modified"),
    }
    for name in ("chemical_formula_descriptive", "chemical_formula_hill", "optimization_type", "immutable_id"):
        arguments[name] = _optional(_string, top[name], name)
    return arguments


def _decode_species(value: object, field: str) -> dict[str, Any]:
    species = _object(value, _SPECIES_KEYS, field)
    return {
        "name": _string(species["name"], f"{field}.name"),
        "chemical_symbols": _items(_string, species["chemical_symbols"], f"{field}.chemical_symbols"),
        "concentration": _items(_rational, species["concentration"], f"{field}.concentration"),
        "concentration_precision": _optional_items(
            _items_or_null(_rational), species["concentration_precision"], f"{field}.concentration_precision"
        ),
        "mass": _optional_items(_float, species["mass"], f"{field}.mass"),
        "original_name": _optional(_string, species["original_name"], f"{field}.original_name"),
        "attached": _optional_items(_string, species["attached"], f"{field}.attached"),
        "nattached": _optional_items(_integer, species["nattached"], f"{field}.nattached"),
        "charges": _optional_items(_items_or_null(_rational), species["charges"], f"{field}.charges"),
        "spins": _optional_items(_items_or_null(_rational), species["spins"], f"{field}.spins"),
        "labels": _optional_items(_items_or_null(_string), species["labels"], f"{field}.labels"),
    }


def _decode_moments(value: object, field: str) -> tuple[str, tuple[SurdScalar, ...], Fraction | None]:
    moments = _object(value, ("kind", "moments", "precision"), field)
    kind = _string(moments["kind"], f"{field}.kind")
    if kind not in _MOMENT_KINDS:
        raise ValueError(f"httk-structure {field}.kind must be one of {sorted(_MOMENT_KINDS)}")
    if kind == "collinear":
        components = _items(_surd, moments["moments"], f"{field}.moments")
    else:
        rows = _items(lambda row, item: _items(_surd, row, item, 3), moments["moments"], f"{field}.moments")
        components = tuple(component for row in rows for component in row)
    return kind, components, _optional(_rational, moments["precision"], f"{field}.precision")


def _decode_assembly(value: object, field: str) -> tuple[Any, ...]:
    assembly = _object(value, ("sites_in_groups", "group_probabilities", "group_probabilities_precision"), field)
    return (
        _items(
            lambda group, item: _items(_integer, group, item), assembly["sites_in_groups"], f"{field}.sites_in_groups"
        ),
        _items(_rational, assembly["group_probabilities"], f"{field}.group_probabilities"),
        _optional_items(
            _items_or_null(_rational),
            assembly["group_probabilities_precision"],
            f"{field}.group_probabilities_precision",
        ),
    )


def _decode_symmetry(value: object, field: str) -> tuple[Any, ...]:
    symmetry = _object(value, _SYMMETRY_KEYS, field)
    return (
        _optional(_integer, symmetry["space_group_it_number"], f"{field}.space_group_it_number"),
        _optional(_symbol, symmetry["space_group_symbol_hall"], f"{field}.space_group_symbol_hall"),
        _optional(
            _symbol, symmetry["space_group_symbol_hermann_mauguin"], f"{field}.space_group_symbol_hermann_mauguin"
        ),
        _optional(
            _symbol,
            symmetry["space_group_symbol_hermann_mauguin_extended"],
            f"{field}.space_group_symbol_hermann_mauguin_extended",
        ),
        _optional_items(
            _operation,
            symmetry["space_group_symmetry_operations_xyz"],
            f"{field}.space_group_symmetry_operations_xyz",
            _MAX_OPERATIONS,
        ),
        _optional_items(_symbol, symmetry["wyckoff_positions"], f"{field}.wyckoff_positions"),
    )


def _decode_composition(value: object, field: str) -> tuple[Any, ...]:
    composition = _object(value, ("mode", "amounts"), field)
    amounts = [
        _object(item, ("element", "amount", "precision"), f"{field}.amounts[{index}]")
        for index, item in enumerate(_list(composition["amounts"], f"{field}.amounts"))
    ]
    elements = [_string(item["element"], f"{field}.amounts.element") for item in amounts]
    if len(set(elements)) != len(elements):
        raise ValueError(f"httk-structure {field}.amounts repeats an element")
    precisions = {
        element: _rational(item["precision"], f"{field}.amounts.precision")
        for element, item in zip(elements, amounts)
        if item["precision"] is not None
    }
    mode = _string(composition["mode"], f"{field}.mode")
    if mode not in {"implicit", "full"}:
        raise ValueError(f"httk-structure {field}.mode must be 'implicit' or 'full'")
    return (
        {element: _rational(item["amount"], f"{field}.amounts.amount") for element, item in zip(elements, amounts)},
        mode,
        precisions,
    )


def _decode_timestamp(value: object, field: str) -> datetime.datetime:
    text = _string(value, field)
    try:
        result = datetime.datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"httk-structure {field} is not an ISO 8601 timestamp: {text!r}") from error
    if result.utcoffset() is None or result.isoformat() != text:
        raise ValueError(f"httk-structure {field} must be a canonical ISO 8601 timestamp with a UTC offset")
    return result


def _check_separated_from_zero(value: SurdScalar, field: str) -> None:
    """Refuse a value whose terms cancel to within ``2**-128`` of the largest term.

    ``SurdScalar.sign`` refines one decimal digit per step until the value's interval
    excludes zero, so its cost grows with the cancellation depth (a Pell pair ``p - q*sqrt(2)``
    of 500 digits takes seconds). Each term is approximated here as an integer scaled so the
    largest term has about ``2**192``; each approximation is off by less than one unit, so a
    sum of at least ``2**64`` proves the value is no closer to zero than ``2**-128`` of its
    largest term, which bounds the exact sign decision to a few dozen steps.
    """
    terms = [(value.coefficient(radicand).to_fraction(), radicand) for radicand in value.radicands]
    if len(terms) < 2:  # zero, or one term: the sign is immediate
        return
    exponent = max(
        abs(q.numerator).bit_length() - q.denominator.bit_length() + radicand.bit_length() // 2 for q, radicand in terms
    )
    shift = 2 * (_SEPARATION_BITS + 64 - exponent)
    total = 0
    for q, radicand in terms:
        numerator, denominator = q.numerator * q.numerator * radicand, q.denominator * q.denominator
        if shift >= 0:
            numerator <<= shift
        else:
            denominator <<= -shift
        approximation = math.isqrt(numerator // denominator)
        total += approximation if q > 0 else -approximation
    if abs(total) < 2**64:
        raise ValueError(f"httk-structure {field} cancels too closely to zero to be decided exactly")


def _unscaled_basis(arguments: Mapping[str, Any]) -> SurdVector:
    """Build the unscaled basis, refusing a scale or determinant that nearly cancels."""
    from httk.atomistic.storage.records import _basis_vector

    basis = _basis_vector(tuple(value for row in arguments["unscaled_basis"] for value in row))
    _check_separated_from_zero(arguments["scale"], "cell.scale")
    _check_separated_from_zero(cast(SurdScalar, basis.det()), "cell.unscaled_basis determinant")
    return basis


def _structure_from_document(document: object) -> UnitcellStructure:
    """Build the exact structure a document describes.

    :param document: The decoded JSON document.
    :return: The described unit-cell structure.
    :raises ValueError: If the document is malformed or describes an inconsistent structure.
    """
    arguments = _decode_arguments(document)
    try:
        return _build_structure(arguments)
    except (TypeError, ArithmeticError) as error:
        raise ValueError(f"httk-structure document describes an invalid structure: {error}") from error


def _build_structure(arguments: Mapping[str, Any]) -> UnitcellStructure:
    from httk.atomistic.storage.records import _moment_from_record

    cell = Cell(
        _unscaled_basis(arguments),
        scale=arguments["scale"],
        precision=arguments["cell_precision"],
        periodicity=arguments["periodicity"],
    )
    moments, assemblies = arguments["site_moments"], arguments["assemblies"]
    symmetry, composition = arguments["symmetry"], arguments["chemical_composition"]
    return UnitcellStructure(
        cell,
        Sites(arguments["reduced_coords"], precision=arguments["sites_precision"]),
        tuple(Species(**species) for species in arguments["species"]),
        arguments["species_at_sites"],
        site_moments=None if moments is None else _moment_from_record(moments[0], moments[1], moments[2], cell),
        molecular=arguments["molecular"],
        assemblies=None if assemblies is None else tuple(Assembly(*assembly) for assembly in assemblies),
        symmetry=None if symmetry is None else StructureSymmetry(*symmetry),
        chemical_composition=None if composition is None else ChemicalComposition(*composition),
        chemical_formula_descriptive=arguments["chemical_formula_descriptive"],
        chemical_formula_hill=arguments["chemical_formula_hill"],
        optimization_type=arguments["optimization_type"],
        immutable_id=arguments["immutable_id"],
        last_modified=arguments["last_modified"],
        charge=arguments["charge"],
    )


# --------------------------------------------------------------------------- encoding


def _json_value(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _optional_rationals(values: Any, field: str) -> list[str | None] | None:
    if values is None:
        return None
    return [None if value is None else _rational_text(value, field) for value in values]


def _structure_document(obj: Any) -> dict[str, Any]:
    """Serialize an httk structure to its exact ``httk-structure`` document.

    :param obj: A unit-cell structure, or any httk structure that ``UnitcellStructureView`` presents.
    :return: The JSON-compatible document.
    :raises ValueError: If ``obj`` is not an httk structure, or holds a value outside the format's bounds.
    """
    from httk.atomistic.storage.records import _extract_surd_scalar, _moment_components

    if isinstance(obj, UnitcellStructure):
        structure = obj
    elif isinstance(obj, StructureBackend | StructureView):
        structure = UnitcellStructureView(obj)
    else:
        raise ValueError(
            f"httk-structure saves httk structures only, not {type(obj).__name__}; convert other values "
            "explicitly with UnitcellStructureView(value) first"
        )
    cell = structure.cell
    basis, scale, coords = cell.unscaled_basis, cell.scale, structure.sites.reduced_coords
    if not isinstance(basis, SurdVector) or not isinstance(scale, SurdScalar) or not isinstance(coords, FracVector):
        raise ValueError("httk-structure can only save a structure whose cell and sites are exact")
    moments = structure.site_moments
    moment_values = None
    if moments is not None:
        components = [_surd_text(value, "site moment") for value in _moment_components(moments)]
        kind = cast(Any, moments).kind
        moment_values = {
            "kind": kind,
            "moments": components
            if kind == "collinear"
            else [components[offset : offset + 3] for offset in range(0, len(components), 3)],
            "precision": None if moments.precision is None else _rational_text(moments.precision, "moment precision"),
        }
    symmetry = _stated_symmetry(structure.symmetry)
    composition = structure.chemical_composition
    last_modified = structure.last_modified
    return {
        "format": FORMAT,
        "version": VERSION,
        "cell": {
            "unscaled_basis": [
                [_surd_text(_extract_surd_scalar(basis, (row, column)), "cell basis") for column in range(3)]
                for row in range(3)
            ],
            "scale": _surd_text(scale, "cell scale"),
            "precision": None if cell.precision is None else _rational_text(cell.precision, "cell precision"),
            "periodicity": list(cell.periodicity),
        },
        "sites": {
            "reduced_coords": [[_rational_text(value, "coordinate") for value in row] for row in coords.to_fractions()],
            "precision": None
            if structure.sites.precision is None
            else _rational_text(structure.sites.precision, "site precision"),
        },
        "species": [
            {
                "name": species.name,
                "chemical_symbols": list(species.chemical_symbols),
                "concentration": [_rational_text(value, "concentration") for value in species.concentration],
                "concentration_precision": _optional_rationals(
                    species.concentration_precision, "concentration precision"
                ),
                "mass": None if species.mass is None else list(species.mass),
                "original_name": species.original_name,
                "attached": None if species.attached is None else list(species.attached),
                "nattached": None if species.nattached is None else list(species.nattached),
                "charges": _optional_rationals(species.charges, "species charge"),
                "spins": _optional_rationals(species.spins, "species spin"),
                "labels": None if species.labels is None else list(species.labels),
            }
            for species in structure.species
        ],
        "species_at_sites": list(structure.species_at_sites),
        "site_moments": moment_values,
        "charge": None if structure.charge is None else _rational_text(structure.charge, "charge"),
        "molecular": structure.molecular,
        "assemblies": None
        if structure.assemblies is None
        else [
            {
                "sites_in_groups": [list(group) for group in assembly.sites_in_groups],
                "group_probabilities": [
                    _rational_text(value, "group probability") for value in assembly.group_probabilities
                ],
                "group_probabilities_precision": _optional_rationals(
                    assembly.group_probabilities_precision, "group probability precision"
                ),
            }
            for assembly in structure.assemblies
        ],
        "symmetry": None
        if symmetry is None
        else {name: _json_value(getattr(symmetry, name)) for name in _SYMMETRY_KEYS},
        "chemical_composition": None
        if composition is None
        else {
            "mode": composition.mode,
            "amounts": [
                {
                    "element": element,
                    "amount": _rational_text(amount, "composition amount"),
                    "precision": None if precision is None else _rational_text(precision, "composition precision"),
                }
                for (element, amount), (_element, precision) in zip(composition.amounts, composition.amounts_precision)
            ],
        },
        "chemical_formula_descriptive": structure.chemical_formula_descriptive,
        "chemical_formula_hill": structure.chemical_formula_hill,
        "optimization_type": structure.optimization_type,
        "immutable_id": structure.immutable_id,
        "last_modified": None if last_modified is None else last_modified.isoformat(),
    }


# --------------------------------------------------------------------------- file layer


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("httk-structure document repeats a JSON object member")
    return result


def _reject_constant(name: str) -> None:
    raise ValueError(f"httk-structure document contains the non-JSON constant {name}")


_OVERSIZED = f"httk-structure document exceeds {_MAX_DOCUMENT_CHARACTERS} characters"


def _read_text(source: str | os.PathLike[str] | IO[str] | IO[bytes]) -> str:
    """Read at most the document cap, counted in characters after strict UTF-8 decoding."""
    limit = _MAX_DOCUMENT_CHARACTERS + 1
    if isinstance(source, str | os.PathLike):
        with TextstreamFileView(os.fspath(source)) as stream:
            text = stream.read(limit)
    else:
        content = source.read(4 * limit)
        if isinstance(content, bytes):
            # A UTF-8 character takes at most four bytes, so 4 * limit bytes exceed the cap.
            if len(content) == 4 * limit:
                raise ValueError(_OVERSIZED)
            content = content.decode("utf-8")
        text = content
    if len(text) > _MAX_DOCUMENT_CHARACTERS:
        raise ValueError(_OVERSIZED)
    return text


def read_httk_structure(
    source: str | os.PathLike[str] | IO[str] | IO[bytes], *, as_written: bool = False
) -> dict[str, Any]:
    """Read and validate an ``httk-structure`` file into its JSON document.

    The document is checked completely (members, types, shapes and exact-number spellings)
    before it is returned; ``httk.core.load`` then builds the
    :class:`~httk.atomistic.UnitcellStructure`, whose constructors check the remaining
    semantic consistency.

    :param source: A filename, optionally compressed, or an open text or binary (UTF-8) stream.
    :param as_written: Already exact; accepted for a uniform load interface.
    :return: The validated document, a mapping whose ``format`` is ``"httk-structure"``.
    :raises ValueError: If the text is not a valid ``httk-structure`` version-1 document.
    """
    _check_as_written(as_written, "read_httk_structure")
    text = _read_text(source)
    try:
        document = json.loads(text, object_pairs_hook=_reject_duplicate_members, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"httk-structure document is not valid JSON: {error}") from error
    except RecursionError as error:
        raise ValueError("httk-structure document is nested too deeply") from error
    _decode_arguments(document)
    return cast(dict[str, Any], document)


def _write_httk_structure(destination: str | os.PathLike[str] | io.TextIOBase, document: Mapping[str, Any]) -> None:
    """Write a validated ``httk-structure`` document as deterministic JSON text."""
    _unscaled_basis(_decode_arguments(dict(document)))
    text = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
    if isinstance(destination, str | os.PathLike):
        with open(destination, "w", encoding="ascii", newline="\n") as handle:
            handle.write(text)
    else:
        destination.write(text)


__all__ = ["FORMAT", "VERSION", "read_httk_structure"]
